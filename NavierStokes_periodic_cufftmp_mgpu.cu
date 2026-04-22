// NavierStokes_periodic_cufftmp_mgpu.cu
// Multi-GPU Navier-Stokes solver using cuFFTMp (single node, multiple GPUs via MPI)
//
// =============================================================================
//  FIXES APPLIED (vs. previous version that hit CUFFT_INVALID_VALUE at line 330):
// =============================================================================
//  1. REMOVED cufftXtSetDistribution.
//     The correct signature is
//        cufftXtSetDistribution(plan, rank,
//           lower_input, upper_input, lower_output, upper_output,
//           strides_input, strides_output)
//     i.e. strides come LAST. The old code interleaved strides between boxes,
//     producing a NULL lower_output pointer -> CUFFT_INVALID_VALUE. Additionally,
//     the upper bounds were inclusive, but the API requires them exclusive
//     (lower[i] <= p[i] < upper[i]). Rather than fix both bugs, we drop the
//     custom distribution entirely and use cuFFTMp's built-in slab decomposition.
//
//  2. BUFFERS ARE NOW NVSHMEM-ALLOCATED via cufftXtMalloc (was: cudaMalloc).
//     cuFFTMp requires buffers passed to cufftExec* / cufftXtExecDescriptor* to
//     come from the NVSHMEM symmetric heap. Regular cudaMalloc pointers fail at
//     runtime. cufftXtMalloc internally uses nvshmem_malloc for us.
//
//  3. TRANSFORMS ARE IN-PLACE (was: out-of-place with separate V_r / V_c).
//     cuFFTMp only supports in-place transforms for its built-in slab format.
//     Each buffer is a single descriptor that is reinterpreted as real (padded,
//     X-slab) or complex (Y-slab) depending on subFormat.
//
//  4. SPECTRAL KERNELS USE Y-SLAB LAYOUT (was: X-slab).
//     After D2Z with built-in slab decomposition, cuFFTMp distributes the
//     complex output in CUFFT_XT_FORMAT_INPLACE_SHUFFLED, which is a Y-slab:
//       rank r owns all global x, y in [r*ny_local, (r+1)*ny_local), all kz.
//     Local linear index i on rank r decodes as:
//        gx      = i / (ny_local * NZC)        // full range 0..NX-1
//        local_y = (i / NZC) % ny_local
//        gy      = r * ny_local + local_y      // partial range
//        kz      = i % NZC
//
//  Run: mpirun -np <nprocs> ./navier_stokes_cufftmp_mgpu
//   nprocs must divide both NX and NY (for equal-sized X-slab and Y-slab).
//
// Data layouts (per rank, single in-place descriptor reinterpreted):
//  - Real   (subFormat=2, X-slab, padded last dim):
//        V[lx][j][k]      at lx * NY * 2*NZC + j * 2*NZC + k
//  - Complex(subFormat=3, Y-slab):
//        V[gx][local_y][kz] at gx * ny_local * NZC + local_y * NZC + kz
// =============================================================================

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftMp.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

using namespace std;

typedef double             GReal;
typedef cufftDoubleComplex GCplx;

// ============================================================
// Grid parameters
// ============================================================
constexpr int    NX = 128, NY = 128, NZ = 128;
constexpr int    NZC      = NZ/2 + 1;          // 65
constexpr double LX = 2.0*M_PI, LY = 2.0*M_PI, LZ = 2.0*M_PI;
constexpr double DX = LX/NX, DY = LY/NY, DZ = LZ/NZ;
constexpr int    NT_TOTAL = 20000;
constexpr int    NT_RUN   = 10;
constexpr double TAU      = 1.0 / NT_TOTAL;
constexpr int    BLOCK    = 256;

// ============================================================
// Error macros
// ============================================================
#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[rank?] CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUFFT_CHECK(e) do { cufftResult _e=(e); if(_e!=CUFFT_SUCCESS){ \
    fprintf(stderr,"[rank?] cuFFT error %s:%d: code %d\n",__FILE__,__LINE__,(int)_e); \
    MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

// ============================================================
// subFormat helpers (cuFFTMp uses same cudaLibXtDesc_t as cuFFTXt)
// ============================================================
static inline void force_d2z_format(cudaLibXtDesc* d) { d->subFormat = CUFFT_XT_FORMAT_INPLACE;          } // = 2
static inline void force_z2d_format(cudaLibXtDesc* d) { d->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED; } // = 3
// 1 MPI rank owns 1 GPU in cuFFTMp → data[0] is the local pointer.
static inline void* gpu_ptr(cudaLibXtDesc* d) { return d->descriptor->data[0]; }

// ============================================================
// Analytical solution (Taylor-Green variant)
// ============================================================
__host__ __device__ double func_V1(double x,double y,double z,double t){return (t*t+1.)*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_V2(double x,double y,double z,double t){return (t*t+1.)*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_V3(double x,double y,double z,double t){return -(t*t+1.)*exp(sin(3.*x+3.*y))*cos(3.*x+3.*y)*sin(6.*z);}
__host__ __device__ double func_dV1_dt(double x,double y,double z,double t){return 2.*t*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_dV2_dt(double x,double y,double z,double t){return 2.*t*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_dV3_dt(double x,double y,double z,double t){return -2.*t*exp(sin(3.*x+3.*y))*cos(3.*x+3.*y)*sin(6.*z);}
__host__ __device__ double func_laplace_V1(double x,double y,double z,double t){double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);return (t*t+1.)*exp(s)*(18.*(c*c-s)-36.)*cos(6.*z);}
__host__ __device__ double func_laplace_V2(double x,double y,double z,double t){return func_laplace_V1(x,y,z,t);}
__host__ __device__ double func_laplace_V3(double x,double y,double z,double t){double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);return (t*t+1.)*exp(s)*c*((-18.)*((c*c-s)-(2.*s+1.))+36.)*sin(6.*z);}
__host__ __device__ double func_rot1(double x,double y,double z,double t){double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);return -(t*t+1.)*exp(s)*(3.*(c*c-s)-6.)*sin(6.*z);}
__host__ __device__ double func_rot2(double x,double y,double z,double t){double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);return -(t*t+1.)*exp(s)*(6.-3.*(c*c-s))*sin(6.*z);}
__host__ __device__ double func_rot3(double,double,double,double){return 0.;}
__host__ __device__ double func_vcr1(double x,double y,double z,double t){return func_V2(x,y,z,t)*func_rot3(x,y,z,t)-func_V3(x,y,z,t)*func_rot2(x,y,z,t);}
__host__ __device__ double func_vcr2(double x,double y,double z,double t){return func_V3(x,y,z,t)*func_rot1(x,y,z,t)-func_V1(x,y,z,t)*func_rot3(x,y,z,t);}
__host__ __device__ double func_vcr3(double x,double y,double z,double t){return func_V1(x,y,z,t)*func_rot2(x,y,z,t)-func_V2(x,y,z,t)*func_rot1(x,y,z,t);}
__host__ __device__ double func_grad_p1(double x,double y,double z,double t){return -(t*t+1.)*sin(x)*cos(y)*cos(z);}
__host__ __device__ double func_grad_p2(double x,double y,double z,double t){return -(t*t+1.)*cos(x)*sin(y)*cos(z);}
__host__ __device__ double func_grad_p3(double x,double y,double z,double t){return -(t*t+1.)*cos(x)*cos(y)*sin(z);}
__host__ __device__ double func_f1(double x,double y,double z,double t){return func_dV1_dt(x,y,z,t)-func_laplace_V1(x,y,z,t)-func_vcr1(x,y,z,t)+func_grad_p1(x,y,z,t);}
__host__ __device__ double func_f2(double x,double y,double z,double t){return func_dV2_dt(x,y,z,t)-func_laplace_V2(x,y,z,t)-func_vcr2(x,y,z,t)+func_grad_p2(x,y,z,t);}
__host__ __device__ double func_f3(double x,double y,double z,double t){return func_dV3_dt(x,y,z,t)-func_laplace_V3(x,y,z,t)-func_vcr3(x,y,z,t)+func_grad_p3(x,y,z,t);}

// ============================================================
// REAL-SPACE KERNELS (X-slab, subFormat=2, padded stride 2*NZC in last dim)
//   gi = x_offset + lx,  lx = 0..nx_local-1
// ============================================================

__global__ void kernel_fill_velocity(double* V1, double* V2, double* V3,
                                      int nx_local, int x_offset, double t) {
    long long nr_local = (long long)nx_local * NY * NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k  = (int)(idx % NZ);
    int j  = (int)((idx / NZ) % NY);
    int lx = (int)(idx / ((long long)NY * NZ));
    int gi = x_offset + lx;
    double x = gi * DX, y = j * DY, z = k * DZ;
    long long pidx = (long long)lx * NY * 2*NZC + j * 2*NZC + k;
    V1[pidx] = func_V1(x, y, z, t);
    V2[pidx] = func_V2(x, y, z, t);
    V3[pidx] = func_V3(x, y, z, t);
}

__global__ void kernel_fill_forcing(double* W1, double* W2, double* W3,
                                     int nx_local, int x_offset, double t) {
    long long nr_local = (long long)nx_local * NY * NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k  = (int)(idx % NZ);
    int j  = (int)((idx / NZ) % NY);
    int lx = (int)(idx / ((long long)NY * NZ));
    int gi = x_offset + lx;
    double x = gi * DX, y = j * DY, z = k * DZ;
    long long pidx = (long long)lx * NY * 2*NZC + j * 2*NZC + k;
    W1[pidx] = func_f1(x, y, z, t);
    W2[pidx] = func_f2(x, y, z, t);
    W3[pidx] = func_f3(x, y, z, t);
}

__global__ void kernel_cross_product(const double* V1, const double* V2, const double* V3,
                                      double* rot1, double* rot2, double* rot3,
                                      int nx_local) {
    long long nr_local = (long long)nx_local * NY * NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k  = (int)(idx % NZ);
    int j  = (int)((idx / NZ) % NY);
    int lx = (int)(idx / ((long long)NY * NZ));
    long long pidx = (long long)lx * NY * 2*NZC + j * 2*NZC + k;
    double v1 = V1[pidx], v2 = V2[pidx], v3 = V3[pidx];
    double w1 = rot1[pidx], w2 = rot2[pidx], w3 = rot3[pidx];
    rot1[pidx] = v2*w3 - v3*w2;
    rot2[pidx] = v3*w1 - v1*w3;
    rot3[pidx] = v1*w2 - v2*w1;
}

__global__ void kernel_error_sq(const double* V1, const double* V2, const double* V3,
                                  double* scratch, int nx_local, int x_offset, double t) {
    long long nr_local = (long long)nx_local * NY * NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k  = (int)(idx % NZ);
    int j  = (int)((idx / NZ) % NY);
    int lx = (int)(idx / ((long long)NY * NZ));
    int gi = x_offset + lx;
    double x = gi * DX, y = j * DY, z = k * DZ;
    long long pidx = (long long)lx * NY * 2*NZC + j * 2*NZC + k;
    double d1 = V1[pidx] - func_V1(x, y, z, t);
    double d2 = V2[pidx] - func_V2(x, y, z, t);
    double d3 = V3[pidx] - func_V3(x, y, z, t);
    scratch[idx] = d1*d1 + d2*d2 + d3*d3;
}

// ============================================================
// SPECTRAL-SPACE KERNELS (Y-slab, subFormat=3)
//   gy = y_offset + local_y, local_y = 0..ny_local-1
//   gx = full range 0..NX-1 (each rank has all of X in spectral)
// ============================================================

__global__ void kernel_scale_cplx(GCplx* A, long long nc_local, double scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    A[idx].x *= scale;
    A[idx].y *= scale;
}

__global__ void kernel_compute_rot(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                    GCplx* rot1, GCplx* rot2, GCplx* rot3,
                                    int ny_local, int y_offset) {
    long long nc_local = (long long)NX * ny_local * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;

    int kz      = (int)(idx % NZC);
    int local_y = (int)((idx / NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * NZC));
    int gy      = y_offset + local_y;

    double kx = (gx <= NX/2) ? (double)gx : (double)(gx - NX);
    double ky = (gy <= NY/2) ? (double)gy : (double)(gy - NY);
    double kzd = (double)kz;

    rot1[idx].x = -(ky * V3[idx].y - kzd * V2[idx].y);
    rot1[idx].y =   ky * V3[idx].x - kzd * V2[idx].x;
    rot2[idx].x = -(kzd * V1[idx].y - kx * V3[idx].y);
    rot2[idx].y =   kzd * V1[idx].x - kx * V3[idx].x;
    rot3[idx].x = -(kx * V2[idx].y - ky * V1[idx].y);
    rot3[idx].y =   kx * V2[idx].x - ky * V1[idx].x;
}

__global__ void kernel_compute_viscous(const GCplx* V, GCplx* visc,
                                        int ny_local, int y_offset) {
    long long nc_local = (long long)NX * ny_local * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;

    int kz      = (int)(idx % NZC);
    int local_y = (int)((idx / NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * NZC));
    int gy      = y_offset + local_y;

    double kx = (gx <= NX/2) ? (double)gx : (double)(gx - NX);
    double ky = (gy <= NY/2) ? (double)gy : (double)(gy - NY);
    double kzd = (double)kz;
    double k2 = kx*kx + ky*ky + kzd*kzd;

    visc[idx].x = -k2 * V[idx].x;
    visc[idx].y = -k2 * V[idx].y;
}

__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
                                      int ny_local, int y_offset) {
    long long nc_local = (long long)NX * ny_local * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;

    int kz      = (int)(idx % NZC);
    int local_y = (int)((idx / NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * NZC));
    int gy      = y_offset + local_y;

    double kx = (gx <= NX/2) ? (double)gx : (double)(gx - NX);
    double ky = (gy <= NY/2) ? (double)gy : (double)(gy - NY);
    double kzd = (double)kz;
    double k2 = kx*kx + ky*ky + kzd*kzd;
    if (k2 < 1e-10) return;

    double div_r = -(kx*V1[idx].y + ky*V2[idx].y + kzd*V3[idx].y);
    double div_i =   kx*V1[idx].x + ky*V2[idx].x + kzd*V3[idx].x;
    double phi_r = div_r / (-k2);
    double phi_i = div_i / (-k2);
    V1[idx].x -= -kx  * phi_i;  V1[idx].y -=  kx  * phi_r;
    V2[idx].x -= -ky  * phi_i;  V2[idx].y -=  ky  * phi_r;
    V3[idx].x -= -kzd * phi_i;  V3[idx].y -=  kzd * phi_r;
}

__global__ void kernel_div_abs(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                 double* scratch, int ny_local, int y_offset) {
    long long nc_local = (long long)NX * ny_local * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;

    int kz      = (int)(idx % NZC);
    int local_y = (int)((idx / NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * NZC));
    int gy      = y_offset + local_y;

    double kx = (gx <= NX/2) ? (double)gx : (double)(gx - NX);
    double ky = (gy <= NY/2) ? (double)gy : (double)(gy - NY);
    double kzd = (double)kz;

    double dr = -(kx*V1[idx].y + ky*V2[idx].y + kzd*V3[idx].y);
    double di =   kx*V1[idx].x + ky*V2[idx].x + kzd*V3[idx].x;
    scratch[idx] = sqrt(dr*dr + di*di);
}

// ============================================================
// Layout-agnostic element-wise kernels
// ============================================================

__global__ void kernel_add_to_rhs(GCplx* rhs1, GCplx* rhs2, GCplx* rhs3,
                                   const GCplx* visc1, const GCplx* visc2, const GCplx* visc3,
                                   const GCplx* f1, const GCplx* f2, const GCplx* f3,
                                   long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    rhs1[idx].x += visc1[idx].x + f1[idx].x; rhs1[idx].y += visc1[idx].y + f1[idx].y;
    rhs2[idx].x += visc2[idx].x + f2[idx].x; rhs2[idx].y += visc2[idx].y + f2[idx].y;
    rhs3[idx].x += visc3[idx].x + f3[idx].x; rhs3[idx].y += visc3[idx].y + f3[idx].y;
}

__global__ void kernel_rk4_axpy(GCplx* V1, GCplx* V2, GCplx* V3,
                                  const GCplx* o1, const GCplx* o2, const GCplx* o3,
                                  const GCplx* k1, const GCplx* k2, const GCplx* k3,
                                  double alpha, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    V1[idx].x = o1[idx].x + alpha * k1[idx].x; V1[idx].y = o1[idx].y + alpha * k1[idx].y;
    V2[idx].x = o2[idx].x + alpha * k2[idx].x; V2[idx].y = o2[idx].y + alpha * k2[idx].y;
    V3[idx].x = o3[idx].x + alpha * k3[idx].x; V3[idx].y = o3[idx].y + alpha * k3[idx].y;
}

__global__ void kernel_rk4_update(GCplx* V1, GCplx* V2, GCplx* V3,
                                   const GCplx* o1, const GCplx* o2, const GCplx* o3,
                                   const GCplx* k1v1, const GCplx* k2v1,
                                   const GCplx* k3v1, const GCplx* k4v1,
                                   const GCplx* k1v2, const GCplx* k2v2,
                                   const GCplx* k3v2, const GCplx* k4v2,
                                   const GCplx* k1v3, const GCplx* k2v3,
                                   const GCplx* k3v3, const GCplx* k4v3,
                                   double dtd6, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    V1[idx].x = o1[idx].x + dtd6*(k1v1[idx].x + 2.*k2v1[idx].x + 2.*k3v1[idx].x + k4v1[idx].x);
    V1[idx].y = o1[idx].y + dtd6*(k1v1[idx].y + 2.*k2v1[idx].y + 2.*k3v1[idx].y + k4v1[idx].y);
    V2[idx].x = o2[idx].x + dtd6*(k1v2[idx].x + 2.*k2v2[idx].x + 2.*k3v2[idx].x + k4v2[idx].x);
    V2[idx].y = o2[idx].y + dtd6*(k1v2[idx].y + 2.*k2v2[idx].y + 2.*k3v2[idx].y + k4v2[idx].y);
    V3[idx].x = o3[idx].x + dtd6*(k1v3[idx].x + 2.*k2v3[idx].x + 2.*k3v3[idx].x + k4v3[idx].x);
    V3[idx].y = o3[idx].y + dtd6*(k1v3[idx].y + 2.*k2v3[idx].y + 2.*k3v3[idx].y + k4v3[idx].y);
}

__global__ void kernel_copy_cplx(const GCplx* src, GCplx* dst, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    dst[idx] = src[idx];
}

// ============================================================
// Per-rank state
// ============================================================
struct State {
    int rank, nprocs, gpu;
    int nx_local, ny_local;
    long long nc_local, nr_local;
    int x_offset, y_offset;

    // cuFFTMp in-place descriptors (allocated via cufftXtMalloc, uses NVSHMEM)
    cudaLibXtDesc *V1_buf, *V2_buf, *V3_buf;
    cudaLibXtDesc *rot1_buf, *rot2_buf, *rot3_buf;
    cudaLibXtDesc *work1_buf, *work2_buf, *work3_buf;

    // Aux per-rank spectral-only arrays (regular cudaMalloc — not passed to cufft exec)
    GCplx *visc1_c, *visc2_c, *visc3_c;
    GCplx *f1_c,    *f2_c,    *f3_c;
    GCplx *rhs1_c,  *rhs2_c,  *rhs3_c;
    GCplx *k1v1, *k2v1, *k3v1, *k4v1;
    GCplx *k1v2, *k2v2, *k3v2, *k4v2;
    GCplx *k1v3, *k2v3, *k3v3, *k4v3;
    GCplx *V1_orig, *V2_orig, *V3_orig;
    double *scratch;
};

static void alloc_state(State& s, cufftHandle plan_r2c) {
    // In-place cuFFTMp buffers (NVSHMEM-backed)
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V1_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V2_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V3_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot1_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot2_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot3_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work1_buf, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work2_buf, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work3_buf, CUFFT_XT_FORMAT_INPLACE));

    // Aux complex arrays (cudaMalloc is fine here — no cufft exec touches them)
    auto C = [&](GCplx** p){ CUDA_CHECK(cudaMalloc(p, s.nc_local*sizeof(GCplx))); };
    C(&s.visc1_c); C(&s.visc2_c); C(&s.visc3_c);
    C(&s.f1_c);    C(&s.f2_c);    C(&s.f3_c);
    C(&s.rhs1_c);  C(&s.rhs2_c);  C(&s.rhs3_c);
    C(&s.k1v1); C(&s.k2v1); C(&s.k3v1); C(&s.k4v1);
    C(&s.k1v2); C(&s.k2v2); C(&s.k3v2); C(&s.k4v2);
    C(&s.k1v3); C(&s.k2v3); C(&s.k3v3); C(&s.k4v3);
    C(&s.V1_orig); C(&s.V2_orig); C(&s.V3_orig);

    long long sc_size = std::max(s.nc_local, s.nr_local);
    CUDA_CHECK(cudaMalloc(&s.scratch, sc_size * sizeof(double)));
}

static void free_state(State& s) {
    CUFFT_CHECK(cufftXtFree(s.V1_buf));    CUFFT_CHECK(cufftXtFree(s.V2_buf));    CUFFT_CHECK(cufftXtFree(s.V3_buf));
    CUFFT_CHECK(cufftXtFree(s.rot1_buf));  CUFFT_CHECK(cufftXtFree(s.rot2_buf));  CUFFT_CHECK(cufftXtFree(s.rot3_buf));
    CUFFT_CHECK(cufftXtFree(s.work1_buf)); CUFFT_CHECK(cufftXtFree(s.work2_buf)); CUFFT_CHECK(cufftXtFree(s.work3_buf));
    cudaFree(s.visc1_c); cudaFree(s.visc2_c); cudaFree(s.visc3_c);
    cudaFree(s.f1_c);    cudaFree(s.f2_c);    cudaFree(s.f3_c);
    cudaFree(s.rhs1_c);  cudaFree(s.rhs2_c);  cudaFree(s.rhs3_c);
    cudaFree(s.k1v1); cudaFree(s.k2v1); cudaFree(s.k3v1); cudaFree(s.k4v1);
    cudaFree(s.k1v2); cudaFree(s.k2v2); cudaFree(s.k3v2); cudaFree(s.k4v2);
    cudaFree(s.k1v3); cudaFree(s.k2v3); cudaFree(s.k3v3); cudaFree(s.k4v3);
    cudaFree(s.V1_orig); cudaFree(s.V2_orig); cudaFree(s.V3_orig);
    cudaFree(s.scratch);
}

// ============================================================
// Physics — same structure as the cuFFTXt version, minus the OMP loop over GPUs
// (because in cuFFTMp, 1 rank = 1 GPU, so no inner loop needed).
// ============================================================

// Nonlinear term: V × rot(V) → rhs_c
static void compute_nonlinear(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s) {
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    const long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // 1. Spectral vorticity written directly into rot_buf (Y-slab complex)
    kernel_compute_rot<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        (GCplx*)gpu_ptr(s.rot1_buf), (GCplx*)gpu_ptr(s.rot2_buf), (GCplx*)gpu_ptr(s.rot3_buf),
        s.ny_local, s.y_offset);
    force_z2d_format(s.rot1_buf); force_z2d_format(s.rot2_buf); force_z2d_format(s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. IFFT velocity: Y-slab complex → X-slab real
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V3_buf, s.V3_buf));
    // 3. IFFT vorticity: Y-slab complex → X-slab real
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.rot1_buf, s.rot1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.rot2_buf, s.rot2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.rot3_buf, s.rot3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Real-space cross product V × rot → rot_buf (X-slab padded)
    kernel_cross_product<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        (double*)gpu_ptr(s.rot1_buf), (double*)gpu_ptr(s.rot2_buf), (double*)gpu_ptr(s.rot3_buf),
        s.nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. FFT cross product → Y-slab spectral (subFormat becomes 3)
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.rot1_buf, s.rot1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.rot2_buf, s.rot2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.rot3_buf, s.rot3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copy and scale → rhs_c
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot1_buf), s.rhs1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot2_buf), s.rhs2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot3_buf), s.rhs3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs3_c, nc, inv_N);
}

// Full RHS = nl + visc + f, projected to divergence-free
static void compute_rhs(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    const long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // Viscous (BEFORE nonlinear which consumes V_buf via Z2D)
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.visc1_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.visc2_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.visc3_c, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Nonlinear → rhs_c (destroys V_buf via Z2D inside)
    compute_nonlinear(plan_r2c, plan_c2r, s);

    // Forcing
    force_d2z_format(s.work1_buf); force_d2z_format(s.work2_buf); force_d2z_format(s.work3_buf);
    kernel_fill_forcing<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.work1_buf), (double*)gpu_ptr(s.work2_buf), (double*)gpu_ptr(s.work3_buf),
        s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.work1_buf, s.work1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.work2_buf, s.work2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.work3_buf, s.work3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work1_buf), s.f1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work2_buf), s.f2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work3_buf), s.f3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f3_c, nc, inv_N);

    // rhs += visc + f
    kernel_add_to_rhs<<<gc, BLOCK>>>(
        s.rhs1_c, s.rhs2_c, s.rhs3_c,
        s.visc1_c, s.visc2_c, s.visc3_c,
        s.f1_c, s.f2_c, s.f3_c, nc);

    // Project rhs to divergence-free space
    kernel_make_div_free<<<gc, BLOCK>>>(
        s.rhs1_c, s.rhs2_c, s.rhs3_c, s.ny_local, s.y_offset);
}

static void save_v_orig(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.V1_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.V2_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.V3_orig, s.nc_local);
}

static void restore_v_buf(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V1_orig, (GCplx*)gpu_ptr(s.V1_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V2_orig, (GCplx*)gpu_ptr(s.V2_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V3_orig, (GCplx*)gpu_ptr(s.V3_buf), s.nc_local);
    force_z2d_format(s.V1_buf); force_z2d_format(s.V2_buf); force_z2d_format(s.V3_buf);
}

static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    const long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);

    save_v_orig(s);

    compute_rhs(plan_r2c, plan_c2r, s, t);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k1v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k1v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k1v3, nc);

    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k1v1, s.k1v2, s.k1v3, 0.5*TAU, nc);

    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k2v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k2v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k2v3, nc);

    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k2v1, s.k2v2, s.k2v3, 0.5*TAU, nc);

    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k3v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k3v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k3v3, nc);

    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k3v1, s.k3v2, s.k3v3, TAU, nc);

    compute_rhs(plan_r2c, plan_c2r, s, t + TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k4v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k4v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k4v3, nc);

    restore_v_buf(s);
    kernel_rk4_update<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k1v1, s.k2v1, s.k3v1, s.k4v1,
        s.k1v2, s.k2v2, s.k3v2, s.k4v2,
        s.k1v3, s.k2v3, s.k3v3, s.k4v3,
        TAU/6.0, nc);
}

// Diagnostics with MPI reductions
static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
                                 State& s, double t, double& L2_err, double& max_div) {
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    const long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // max|div V| (spectral Y-slab, before IFFT)
    kernel_div_abs<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.scratch, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_ptr<double> sp(s.scratch);
    double local_max = thrust::reduce(thrust::device, sp, sp + nc, 0.0, thrust::maximum<double>());
    MPI_Allreduce(&local_max, &max_div, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // IFFT V_buf → X-slab real
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V3_buf, s.V3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Squared error
    kernel_error_sq<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        s.scratch, s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    double local_sq = thrust::reduce(thrust::device, sp, sp + nr, 0.0);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    L2_err = sqrt(global_sq * DX * DY * DZ);

    // Re-FFT V_buf → Y-slab spectral
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V3_buf, s.V3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Renormalize (D2Z is not normalized)
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), nc, inv_N);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    State s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.nprocs);

    // Each rank binds to one GPU (single-node assumption)
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    s.gpu = s.rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(s.gpu));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, s.gpu));

    // Decomposition — require clean divisibility (built-in slab assumption)
    if (NX % s.nprocs != 0 || NY % s.nprocs != 0) {
        if (s.rank == 0) {
            fprintf(stderr, "Error: nprocs=%d must divide both NX=%d and NY=%d\n",
                    s.nprocs, NX, NY);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.nx_local  = NX / s.nprocs;
    s.ny_local  = NY / s.nprocs;
    s.x_offset  = s.rank * s.nx_local;
    s.y_offset  = s.rank * s.ny_local;
    s.nc_local  = (long long)NX * s.ny_local * NZC;
    s.nr_local  = (long long)s.nx_local * NY * NZ;

    if (s.rank == 0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes — cuFFTMp Built-in Slab (fixed)\n";
        cout << "============================================================\n";
        cout << "MPI ranks: " << s.nprocs << "  GPUs/node: " << num_gpus << "\n";
        cout << "Grid: " << NX << " x " << NY << " x " << NZ << "\n";
        cout << "  Real  (X-slab): nx_local=" << s.nx_local << " per rank\n";
        cout << "  Spec. (Y-slab): ny_local=" << s.ny_local << " per rank\n";
        cout << "Steps: " << NT_RUN << " / " << NT_TOTAL << "  dt=" << TAU << "\n";
        cout << "============================================================\n";
    }
    printf("  Rank %d → GPU %d (%s)  X=[%d,%d)  Y=[%d,%d)\n",
           s.rank, s.gpu, prop.name,
           s.x_offset, s.x_offset + s.nx_local,
           s.y_offset, s.y_offset + s.ny_local);
    MPI_Barrier(MPI_COMM_WORLD);
    if (s.rank == 0) cout << "============================================================\n";

    // ---- cuFFTMp plans (built-in slab decomposition, NO cufftXtSetDistribution) ----
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    MPI_Comm world = MPI_COMM_WORLD;
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &world));

    size_t ws_r2c = 0, ws_c2r = 0;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    // ---- Allocate GPU memory ----
    alloc_state(s, plan_r2c);

    // ---- Initial condition ----
    if (s.rank == 0) cout << "Setting initial conditions...\n";
    force_d2z_format(s.V1_buf); force_d2z_format(s.V2_buf); force_d2z_format(s.V3_buf);
    {
        int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);
        kernel_fill_velocity<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
            s.nx_local, s.x_offset, 0.0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V3_buf, s.V3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    const double inv_N = 1.0/(double)(NX*NY*NZ);
    {
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.nc_local, inv_N);

        // Project to divergence-free (Y-slab kernel)
        kernel_make_div_free<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
            s.ny_local, s.y_offset);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initial diagnostics
    {
        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, s, 0.0, L2_err, max_div);
        if (s.rank == 0) {
            cout << "\nInitial (t=0):\n";
            cout << "  L2 error:   " << scientific << L2_err << "\n";
            cout << "  max|div V|: " << max_div << "\n\n";
        }
    }

    // ---- Time integration ----
    if (s.rank == 0) {
        cout << "============================================================\n";
        cout << "  Time Integration (RK4)\n";
        cout << "------------------------------------------------------------\n";
        cout << setw(6) << "Step" << setw(12) << "Wall(s)"
             << setw(16) << "L2 Error" << setw(16) << "max|div V|" << "\n";
        cout << "------------------------------------------------------------\n";
    }

    double t_wall = 0.0;
    for (int it = 0; it <= NT_RUN; ++it) {
        double t_cur = it * TAU;
        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, s, t_cur, L2_err, max_div);

        if (s.rank == 0) {
            cout << setw(6) << it
                 << setw(12) << fixed << setprecision(4) << t_wall
                 << setw(16) << scientific << setprecision(4) << L2_err
                 << setw(16) << max_div << "\n";
        }

        if (it < NT_RUN) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            rk4_step(plan_r2c, plan_c2r, s, t_cur);
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Barrier(MPI_COMM_WORLD);
            double dt_step = MPI_Wtime() - t0, dt_max = 0.0;
            MPI_Allreduce(&dt_step, &dt_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            t_wall += dt_max;
        }
    }

    if (s.rank == 0) {
        cout << "============================================================\n";
        cout << "  Total wall: " << fixed << setprecision(4) << t_wall << " s\n";
        cout << "  Avg/step:   " << t_wall / NT_RUN << " s\n";
        cout << "============================================================\n";
    }

    // Cleanup
    free_state(s);
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
