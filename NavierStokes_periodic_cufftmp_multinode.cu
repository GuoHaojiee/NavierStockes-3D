// NavierStokes_periodic_cufftmp_multinode.cu
// =============================================================================
// Pseudo-spectral Navier-Stokes solver (periodic domain) with cuFFTMp.
// Distributed across multiple GPUs/nodes via MPI + cuFFTMp.
//
// Discretisation:
//   - Spatial: Fourier spectral (D2Z / Z2D via cuFFTMp)
//   - Temporal: explicit RK4
//
// Buffer layout:
//   ALL FFT buffers allocated as CUFFT_XT_FORMAT_INPLACE (X-slab, larger).
//   subFormat is set explicitly via force_d2z_format / force_z2d_format,
//   embedded in the FFT_FORWARD / FFT_INVERSE macros and at kernel write sites,
//   so cuFFTMp always receives the correct layout descriptor before each call.
// =============================================================================

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvshmem.h>
#include <cufftMp.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "common_mode.h"

using namespace std;
typedef cufftDoubleComplex GCplx;

// Run mode / phase timing (see common_mode.h).
//   g_with_force = true  (verify): analytic forcing f every RHS (original behaviour)
//   g_with_force = false (bench) : f = 0; forcing fill and its 3 FFTs are skipped;
//                                  work*_buf are not allocated.
static bool g_with_force = true;
static nsmode::GpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;

static int    NX, NY, NZ, NZC, NT_RUN;
static double LX, LY, LZ, DX, DY, DZ, TAU;
__device__ __constant__ int    d_NX, d_NY, d_NZ, d_NZC;
__device__ __constant__ double d_DX, d_DY, d_DZ;
constexpr int BLOCK = 256;

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    fflush(stderr); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUFFT_CHECK(e) do { cufftResult _e=(e); if(_e!=CUFFT_SUCCESS){ \
    fprintf(stderr,"cuFFT error %s:%d: code %d\n",__FILE__,__LINE__,(int)_e); \
    fflush(stderr); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

static inline void* gpu_ptr(cudaLibXtDesc* d) { return d->descriptor->data[0]; }
static inline size_t gpu_size_bytes(cudaLibXtDesc* d) { return d->descriptor->size[0]; }

// =============================================================================
// subFormat helpers.
//   CUFFT_XT_FORMAT_INPLACE          = 2  -> real X-slab   (D2Z input)
//   CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 3  -> complex Y-slab (Z2D input)
//
// Called inside FFT_FORWARD / FFT_INVERSE macros and at kernel write sites
// to keep the descriptor consistent with the data actually present in the buffer.
// =============================================================================
static inline void force_d2z_format(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE;          // real X-slab
}
static inline void force_z2d_format(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED; // complex Y-slab
}

// FFT execution macros.
//   FFT_FORWARD: INPLACE (X-slab real)      -> INPLACE_SHUFFLED (Y-slab complex)
//   FFT_INVERSE: INPLACE_SHUFFLED (Y-slab)  -> INPLACE (X-slab real)
#define FFT_FORWARD(plan, buf) do {                                           \
    force_d2z_format(buf);                                                    \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_FORWARD));     \
} while(0)
#define FFT_INVERSE(plan, buf) do {                                           \
    force_z2d_format(buf);                                                    \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_INVERSE));     \
} while(0)

// =============================================================================
// Manufactured-solution functions
// =============================================================================
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

// =============================================================================
// Real-space kernels (X-slab, padded stride 2*NZC)
// =============================================================================
__global__ void kernel_fill_velocity(double* V1, double* V2, double* V3,
                                      int nx_local, int x_offset, double t) {
    long long nr_local = (long long)nx_local * d_NY * d_NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    int gi = x_offset + lx;
    double x = gi * d_DX, y = j * d_DY, z = k * d_DZ;
    long long pidx = (long long)lx * d_NY * 2*d_NZC + (long long)j * 2*d_NZC + k;
    V1[pidx] = func_V1(x, y, z, t);
    V2[pidx] = func_V2(x, y, z, t);
    V3[pidx] = func_V3(x, y, z, t);
}

__global__ void kernel_fill_forcing(double* W1, double* W2, double* W3,
                                     int nx_local, int x_offset, double t) {
    long long nr_local = (long long)nx_local * d_NY * d_NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    int gi = x_offset + lx;
    double x = gi * d_DX, y = j * d_DY, z = k * d_DZ;
    long long pidx = (long long)lx * d_NY * 2*d_NZC + (long long)j * 2*d_NZC + k;
    W1[pidx] = func_f1(x, y, z, t);
    W2[pidx] = func_f2(x, y, z, t);
    W3[pidx] = func_f3(x, y, z, t);
}

__global__ void kernel_cross_product(const double* V1, const double* V2, const double* V3,
                                      double* rot1, double* rot2, double* rot3,
                                      int nx_local) {
    long long nr_local = (long long)nx_local * d_NY * d_NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    long long pidx = (long long)lx * d_NY * 2*d_NZC + (long long)j * 2*d_NZC + k;
    double v1 = V1[pidx], v2 = V2[pidx], v3 = V3[pidx];
    double w1 = rot1[pidx], w2 = rot2[pidx], w3 = rot3[pidx];
    rot1[pidx] = v2*w3 - v3*w2;
    rot2[pidx] = v3*w1 - v1*w3;
    rot3[pidx] = v1*w2 - v2*w1;
}

__global__ void kernel_error_sq(const double* V1, const double* V2, const double* V3,
                                  double* scratch, int nx_local, int x_offset, double t) {
    long long nr_local = (long long)nx_local * d_NY * d_NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    int gi = x_offset + lx;
    double x = gi * d_DX, y = j * d_DY, z = k * d_DZ;
    long long pidx = (long long)lx * d_NY * 2*d_NZC + (long long)j * 2*d_NZC + k;
    double d1 = V1[pidx] - func_V1(x, y, z, t);
    double d2 = V2[pidx] - func_V2(x, y, z, t);
    double d3 = V3[pidx] - func_V3(x, y, z, t);
    scratch[idx] = d1*d1 + d2*d2 + d3*d3;
}

// |v|^2 per point (bench-mode kinetic-energy diagnostic; X-slab real, padded stride)
__global__ void kernel_energy_sq(const double* V1, const double* V2, const double* V3,
                                  double* scratch, int nx_local) {
    long long nr_local = (long long)nx_local * d_NY * d_NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    long long pidx = (long long)lx * d_NY * 2*d_NZC + (long long)j * 2*d_NZC + k;
    double v1 = V1[pidx], v2 = V2[pidx], v3 = V3[pidx];
    scratch[idx] = v1*v1 + v2*v2 + v3*v3;
}

// =============================================================================
// Spectral-space kernels (Y-slab complex, packed stride NZC)
// =============================================================================
__global__ void kernel_scale_cplx(GCplx* A, long long nc_local, double scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    A[idx].x *= scale; A[idx].y *= scale;
}

// Kernel A -- spectral, start of every RHS evaluation (merges curl and viscous term).
// Reads V; writes rot = i k x V and rhs = -k^2 V.
// Layout: [gx][local_y][kz]  (Y-slab, INPLACE_SHUFFLED)
__global__ void kernel_rot_and_viscous(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                        GCplx* rot1, GCplx* rot2, GCplx* rot3,
                                        GCplx* rhs1, GCplx* rhs2, GCplx* rhs3,
                                        int ny_local, int y_offset) {
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    int kz      = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * d_NZC));
    int gy = y_offset + local_y;
    double kx = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double k2 = kx*kx + ky*ky + kzd*kzd;
    GCplx v1 = V1[idx], v2 = V2[idx], v3 = V3[idx];
    rot1[idx].x = -(ky * v3.y - kzd * v2.y);
    rot1[idx].y =   ky * v3.x - kzd * v2.x;
    rot2[idx].x = -(kzd * v1.y - kx * v3.y);
    rot2[idx].y =   kzd * v1.x - kx * v3.x;
    rot3[idx].x = -(kx * v2.y - ky * v1.y);
    rot3[idx].y =   kx * v2.x - ky * v1.x;
    rhs1[idx].x = -k2 * v1.x;  rhs1[idx].y = -k2 * v1.y;
    rhs2[idx].x = -k2 * v2.x;  rhs2[idx].y = -k2 * v2.y;
    rhs3[idx].x = -k2 * v3.x;  rhs3[idx].y = -k2 * v3.y;
}

// What kernel_rk4_stage does with the accumulator acc.
enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

// Kernel B -- spectral, end of every RK4 stage (classic RK4, accumulated form).
//   k = P[ inv_N*rot + rhs (+ inv_N*work, verify only) ]    held in registers only
//   ACC_INIT: acc  = base + a_acc*k
//   ACC_ADD : acc += a_acc*k
//   ACC_KEEP: acc unchanged
//   V = base + a_v*k
// Stage 1: base = V_orig, ACC_INIT, a_acc = tau/6, a_v = tau/2
// Stage 2: base = V_orig, ACC_ADD,  a_acc = tau/3, a_v = tau/2
// Stage 3: base = V_orig, ACC_ADD,  a_acc = tau/3, a_v = tau
// Stage 4: base = acc,    ACC_KEEP,                a_v = tau/6
// The k^2 = 0 mode is left unprojected. Sum order nl + (visc + f) as before.
// Layout: [gx][local_y][kz]  (Y-slab, INPLACE_SHUFFLED)
__global__ void kernel_rk4_stage(GCplx* V1, GCplx* V2, GCplx* V3,
                                  const GCplx* base1, const GCplx* base2, const GCplx* base3,
                                  GCplx* acc1, GCplx* acc2, GCplx* acc3,
                                  const GCplx* rot1, const GCplx* rot2, const GCplx* rot3,
                                  const GCplx* rhs1, const GCplx* rhs2, const GCplx* rhs3,
                                  const GCplx* work1, const GCplx* work2, const GCplx* work3,
                                  double inv_N, double a_acc, double a_v, int acc_mode,
                                  int ny_local, int y_offset) {
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    int kz      = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * d_NZC));
    int gy = y_offset + local_y;
    double kx = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double k2 = kx*kx + ky*ky + kzd*kzd;

    double s1x = rhs1[idx].x, s1y = rhs1[idx].y;
    double s2x = rhs2[idx].x, s2y = rhs2[idx].y;
    double s3x = rhs3[idx].x, s3y = rhs3[idx].y;
    if (work1) {
        s1x += work1[idx].x * inv_N;  s1y += work1[idx].y * inv_N;
        s2x += work2[idx].x * inv_N;  s2y += work2[idx].y * inv_N;
        s3x += work3[idx].x * inv_N;  s3y += work3[idx].y * inv_N;
    }
    double k1x = rot1[idx].x * inv_N + s1x, k1y = rot1[idx].y * inv_N + s1y;
    double k2x = rot2[idx].x * inv_N + s2x, k2y = rot2[idx].y * inv_N + s2y;
    double k3x = rot3[idx].x * inv_N + s3x, k3y = rot3[idx].y * inv_N + s3y;

    if (k2 >= 1e-10) {
        double div_r = -(kx*k1y + ky*k2y + kzd*k3y);
        double div_i =   kx*k1x + ky*k2x + kzd*k3x;
        double phi_r = div_r / (-k2);
        double phi_i = div_i / (-k2);
        k1x -= -kx  * phi_i;  k1y -=  kx  * phi_r;
        k2x -= -ky  * phi_i;  k2y -=  ky  * phi_r;
        k3x -= -kzd * phi_i;  k3y -=  kzd * phi_r;
    }

    GCplx b1 = base1[idx], b2 = base2[idx], b3 = base3[idx];
    if (acc_mode == ACC_INIT) {
        acc1[idx].x = b1.x + a_acc*k1x;  acc1[idx].y = b1.y + a_acc*k1y;
        acc2[idx].x = b2.x + a_acc*k2x;  acc2[idx].y = b2.y + a_acc*k2y;
        acc3[idx].x = b3.x + a_acc*k3x;  acc3[idx].y = b3.y + a_acc*k3y;
    } else if (acc_mode == ACC_ADD) {
        acc1[idx].x += a_acc*k1x;  acc1[idx].y += a_acc*k1y;
        acc2[idx].x += a_acc*k2x;  acc2[idx].y += a_acc*k2y;
        acc3[idx].x += a_acc*k3x;  acc3[idx].y += a_acc*k3y;
    }
    V1[idx].x = b1.x + a_v*k1x;  V1[idx].y = b1.y + a_v*k1y;
    V2[idx].x = b2.x + a_v*k2x;  V2[idx].y = b2.y + a_v*k2y;
    V3[idx].x = b3.x + a_v*k3x;  V3[idx].y = b3.y + a_v*k3y;
}

__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
                                      int ny_local, int y_offset) {
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    // Layout: [gx][local_y][kz]
    int kz      = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * d_NZC));
    int gy = y_offset + local_y;
    double kx = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
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
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    // Layout: [gx][local_y][kz]
    int kz      = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * d_NZC));
    int gy = y_offset + local_y;
    double kx = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double dr = -(kx*V1[idx].y + ky*V2[idx].y + kzd*V3[idx].y);
    double di =   kx*V1[idx].x + ky*V2[idx].x + kzd*V3[idx].x;
    scratch[idx] = sqrt(dr*dr + di*di);
}

__global__ void kernel_copy_cplx(const GCplx* src, GCplx* dst, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    dst[idx] = src[idx];
}

// =============================================================================
// Per-rank state
// =============================================================================
struct State {
    int rank, nprocs, gpu;
    int local_rank, local_size;
    int nx_local, ny_local;
    long long nc_local, nr_local;
    int x_offset, y_offset;

    // FFT-managed buffers -- ALL allocated as INPLACE (larger of the two layouts).
    // INPLACE = X-slab real padded; SHUFFLED = Y-slab complex.
    // Allocating rot/work buffers as SHUFFLED would under-size them: FFT_INVERSE
    // writes INPLACE-sized output, overflowing into adjacent allocations.
    cudaLibXtDesc *V1_buf, *V2_buf, *V3_buf;
    cudaLibXtDesc *rot1_buf, *rot2_buf, *rot3_buf;
    cudaLibXtDesc *work1_buf = nullptr, *work2_buf = nullptr, *work3_buf = nullptr;  // verify only

    // Plain cudaMalloc'd complex arrays
    GCplx *rhs1_c, *rhs2_c, *rhs3_c;        // holds -k^2 V between kernel A and kernel B
    GCplx *V1_orig, *V2_orig, *V3_orig;     // V^n
    GCplx *acc1_c, *acc2_c, *acc3_c;        // RK4 accumulator
    double *scratch;                        // diagnostics only: shares rhs1_c's memory
};

// cufftXtMalloc + immediate memset to zero.
static void alloc_and_zero(cufftHandle plan, cudaLibXtDesc** desc, cufftXtSubFormat fmt) {
    CUFFT_CHECK(cufftXtMalloc(plan, desc, fmt));
    CUDA_CHECK(cudaMemset((*desc)->descriptor->data[0], 0,
                          (*desc)->descriptor->size[0]));
}

static void alloc_state(State& s, cufftHandle plan_r2c) {
    alloc_and_zero(plan_r2c, &s.V1_buf,    CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.V2_buf,    CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.V3_buf,    CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.rot1_buf,  CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.rot2_buf,  CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.rot3_buf,  CUFFT_XT_FORMAT_INPLACE);
    // Forcing-only buffers (used solely by the forcing block of compute_rhs).
    if (g_with_force) {
        alloc_and_zero(plan_r2c, &s.work1_buf, CUFFT_XT_FORMAT_INPLACE);
        alloc_and_zero(plan_r2c, &s.work2_buf, CUFFT_XT_FORMAT_INPLACE);
        alloc_and_zero(plan_r2c, &s.work3_buf, CUFFT_XT_FORMAT_INPLACE);
    }

    auto C = [&](GCplx** p){
        CUDA_CHECK(cudaMalloc(p, s.nc_local*sizeof(GCplx)));
        CUDA_CHECK(cudaMemset(*p, 0, s.nc_local*sizeof(GCplx)));
    };
    C(&s.rhs1_c);  C(&s.rhs2_c);  C(&s.rhs3_c);
    C(&s.V1_orig); C(&s.V2_orig); C(&s.V3_orig);
    C(&s.acc1_c);  C(&s.acc2_c);  C(&s.acc3_c);

    // Diagnostics run only after the timed loop, when rhs is no longer needed,
    // so they reuse rhs1's memory as scratch (one double per mode or grid point).
    const long long sc_size = std::max(s.nc_local, s.nr_local);
    if ((long long)(s.nc_local * sizeof(GCplx)) < sc_size * (long long)sizeof(double)) {
        fprintf(stderr, "Rank %d: rhs1 (%lld complex) too small for scratch (%lld double)\n",
                s.rank, s.nc_local, sc_size);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.scratch = (double*)s.rhs1_c;
}

static void free_state(State& s) {
    CUFFT_CHECK(cufftXtFree(s.V1_buf));    CUFFT_CHECK(cufftXtFree(s.V2_buf));    CUFFT_CHECK(cufftXtFree(s.V3_buf));
    CUFFT_CHECK(cufftXtFree(s.rot1_buf));  CUFFT_CHECK(cufftXtFree(s.rot2_buf));  CUFFT_CHECK(cufftXtFree(s.rot3_buf));
    if (s.work1_buf) {
        CUFFT_CHECK(cufftXtFree(s.work1_buf)); CUFFT_CHECK(cufftXtFree(s.work2_buf)); CUFFT_CHECK(cufftXtFree(s.work3_buf));
    }
    cudaFree(s.rhs1_c);  cudaFree(s.rhs2_c);  cudaFree(s.rhs3_c);
    cudaFree(s.V1_orig); cudaFree(s.V2_orig); cudaFree(s.V3_orig);
    cudaFree(s.acc1_c);  cudaFree(s.acc2_c);  cudaFree(s.acc3_c);
}

// =============================================================================
// Warm up all FFT buffers with one zero-data FORWARD+INVERSE cycle each.
// Forces cuFFTMp to initialize per-buffer internal state (transpose plans,
// NVSHMEM symmetric heap, etc.) before any simulation FFT call that relies
// on a manually set subFormat.  Without this, virgin rot/work buffers produce
// incorrect results on the first FFT_INVERSE with a hacked subFormat.
// =============================================================================
static void warmup_buffers(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s) {
    auto warm = [&](cudaLibXtDesc* buf) {
        CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_size_bytes(buf)));
        FFT_FORWARD(plan_r2c, buf);
        FFT_INVERSE(plan_c2r, buf);
        CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_size_bytes(buf)));
    };
    warm(s.V1_buf);    warm(s.V2_buf);    warm(s.V3_buf);
    warm(s.rot1_buf);  warm(s.rot2_buf);  warm(s.rot3_buf);
    if (s.work1_buf) { warm(s.work1_buf); warm(s.work2_buf); warm(s.work3_buf); }  // not allocated in bench
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// Physics: classic RK4 with nonlinear term V x (curl V), viscous term, forcing, projection
// =============================================================================
//
// One RK4 stage: evaluates the RHS at V_buf and advances V_buf (see kernel_rk4_stage).
// ENTRY: V_buf data is Y-slab complex (spectral), holding the stage value.
// EXIT:  V_buf data is Y-slab complex (spectral), holding the next stage value
//        (after stage 4: V^{n+1}).
static void rk4_stage(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t, int stage) {
    const double inv_N = 1.0/((double)NX*NY*NZ);
    const long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // rot_buf receives Y-slab complex data from kernel A.
    // Mark subFormat = SHUFFLED so the subsequent FFT_INVERSE interprets
    // the layout correctly.
    force_z2d_format(s.rot1_buf);
    force_z2d_format(s.rot2_buf);
    force_z2d_format(s.rot3_buf);

    kernel_rot_and_viscous<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        (GCplx*)gpu_ptr(s.rot1_buf), (GCplx*)gpu_ptr(s.rot2_buf), (GCplx*)gpu_ptr(s.rot3_buf),
        s.rhs1_c, s.rhs2_c, s.rhs3_c,
        s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    g_timer.mark(PH_SPEC);

    // Nonlinear term. After this: V_buf is X-slab real, rot_buf is Y-slab complex.
    FFT_INVERSE(plan_c2r, s.V1_buf);
    FFT_INVERSE(plan_c2r, s.V2_buf);
    FFT_INVERSE(plan_c2r, s.V3_buf);
    FFT_INVERSE(plan_c2r, s.rot1_buf);
    FFT_INVERSE(plan_c2r, s.rot2_buf);
    FFT_INVERSE(plan_c2r, s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    g_timer.mark(PH_FFT);

    kernel_cross_product<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        (double*)gpu_ptr(s.rot1_buf), (double*)gpu_ptr(s.rot2_buf), (double*)gpu_ptr(s.rot3_buf),
        s.nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());
    g_timer.mark(PH_PHYS);

    FFT_FORWARD(plan_r2c, s.rot1_buf);
    FFT_FORWARD(plan_r2c, s.rot2_buf);
    FFT_FORWARD(plan_r2c, s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    g_timer.mark(PH_FFT);

    if (g_with_force) {
        // Forcing: write real X-slab data directly into work_buf.
        force_d2z_format(s.work1_buf);
        force_d2z_format(s.work2_buf);
        force_d2z_format(s.work3_buf);
        kernel_fill_forcing<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.work1_buf), (double*)gpu_ptr(s.work2_buf), (double*)gpu_ptr(s.work3_buf),
            s.nx_local, s.x_offset, t);
        CUDA_CHECK(cudaDeviceSynchronize());

        FFT_FORWARD(plan_r2c, s.work1_buf);
        FFT_FORWARD(plan_r2c, s.work2_buf);
        FFT_FORWARD(plan_r2c, s.work3_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        g_timer.mark(PH_FORCE);
    }

    // Classic RK4 coefficients, in accumulated form (see kernel_rk4_stage).
    const GCplx *b1 = s.V1_orig, *b2 = s.V2_orig, *b3 = s.V3_orig;
    double a_acc = 0.0, a_v = 0.0;
    int acc_mode = ACC_ADD;
    switch (stage) {
        case 1:  acc_mode = ACC_INIT; a_acc = TAU/6.0; a_v = 0.5*TAU; break;
        case 2:                       a_acc = TAU/3.0; a_v = 0.5*TAU; break;
        case 3:                       a_acc = TAU/3.0; a_v = TAU;     break;
        default: acc_mode = ACC_KEEP;                  a_v = TAU/6.0;
                 b1 = s.acc1_c; b2 = s.acc2_c; b3 = s.acc3_c;         break;
    }

    // V_buf held X-slab real data; kernel B overwrites it with Y-slab complex data.
    force_z2d_format(s.V1_buf);
    force_z2d_format(s.V2_buf);
    force_z2d_format(s.V3_buf);
    kernel_rk4_stage<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        b1, b2, b3,
        s.acc1_c, s.acc2_c, s.acc3_c,
        (const GCplx*)gpu_ptr(s.rot1_buf), (const GCplx*)gpu_ptr(s.rot2_buf), (const GCplx*)gpu_ptr(s.rot3_buf),
        s.rhs1_c, s.rhs2_c, s.rhs3_c,
        g_with_force ? (const GCplx*)gpu_ptr(s.work1_buf) : nullptr,
        g_with_force ? (const GCplx*)gpu_ptr(s.work2_buf) : nullptr,
        g_with_force ? (const GCplx*)gpu_ptr(s.work3_buf) : nullptr,
        inv_N, a_acc, a_v, acc_mode, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    g_timer.mark(PH_SPEC);
}

static void save_v_orig(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.V1_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.V2_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.V3_orig, s.nc_local);
}

static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    save_v_orig(s);
    CUDA_CHECK(cudaDeviceSynchronize());
    g_timer.mark(PH_SPEC);

    rk4_stage(plan_r2c, plan_c2r, s, t,           1);
    rk4_stage(plan_r2c, plan_c2r, s, t + 0.5*TAU, 2);
    rk4_stage(plan_r2c, plan_c2r, s, t + 0.5*TAU, 3);
    rk4_stage(plan_r2c, plan_c2r, s, t + TAU,     4);
}

// verify: L2_or_energy = L2 error vs analytic V(t)
// bench : L2_or_energy = kinetic energy 0.5*sum|v|^2 (no analytic solution for f = 0)
static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
                                 State& s, double t, bool verify,
                                 double& L2_or_energy, double& max_div) {
    const double inv_N = 1.0/((double)NX*NY*NZ);
    const long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // V data is Y-slab complex (spectral) on entry.
    kernel_div_abs<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.scratch, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_ptr<double> sp(s.scratch);
    double local_max = thrust::reduce(thrust::device, sp, sp + nc, 0.0, thrust::maximum<double>());
    MPI_Allreduce(&local_max, &max_div, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    FFT_INVERSE(plan_c2r, s.V1_buf);
    FFT_INVERSE(plan_c2r, s.V2_buf);
    FFT_INVERSE(plan_c2r, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (verify)
        kernel_error_sq<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
            s.scratch, s.nx_local, s.x_offset, t);
    else
        kernel_energy_sq<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
            s.scratch, s.nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());
    double local_sq = thrust::reduce(thrust::device, sp, sp + nr, 0.0);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    L2_or_energy = verify ? sqrt(global_sq * DX * DY * DZ) : 0.5 * global_sq;

    // Restore V back to spectral with proper normalization.
    FFT_FORWARD(plan_r2c, s.V1_buf);
    FFT_FORWARD(plan_r2c, s.V2_buf);
    FFT_FORWARD(plan_r2c, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), nc, inv_N);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    State s;
    MPI_Comm_rank(MPI_COMM_WORLD, &s.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s.nprocs);

    if (argc < 6) {
        if (s.rank == 0)
            fprintf(stderr, "Usage: mpirun -np <NP> %s %s\n", argv[0], nsmode::usage_tail(false));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    nsmode::RunArgs rargs;
    {
        std::string perr;
        if (!nsmode::parse_tail_args(argc, argv, 6, /*allow_threads=*/false, rargs, perr)) {
            if (s.rank == 0)
                fprintf(stderr, "Error: %s\nUsage: mpirun -np <NP> %s %s\n",
                        perr.c_str(), argv[0], nsmode::usage_tail(false));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const nsmode::RunMode mode = rargs.mode;
    g_with_force = (mode == nsmode::MODE_VERIFY);
    NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]);
    TAU = atof(argv[4]); NT_RUN = atoi(argv[5]);
    NZC = NZ/2 + 1;
    LX = LY = LZ = 2.0*M_PI; DX = LX/NX; DY = LY/NY; DZ = LZ/NZ;

    // =========================================================================
    // CRITICAL: bind to the correct GPU BEFORE any cudaMemcpyToSymbol.
    // cudaMemcpyToSymbol writes to the *current* device's constant memory.
    // Without this, all ranks default to GPU 0, so ranks 1..P-1 have
    // d_NX=d_NY=d_NZC=0 on their own GPUs -- every kernel returns immediately.
    // =========================================================================
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, s.rank, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &s.local_rank);
    MPI_Comm_size(node_comm, &s.local_size);
    MPI_Comm_free(&node_comm);

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (s.local_rank >= num_gpus) {
        fprintf(stderr, "Rank %d: local_rank %d >= num_gpus %d\n", s.rank, s.local_rank, num_gpus);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.gpu = s.local_rank;
    CUDA_CHECK(cudaSetDevice(s.gpu));

    cudaMemcpyToSymbol(d_NX,  &NX,  sizeof(int));
    cudaMemcpyToSymbol(d_NY,  &NY,  sizeof(int));
    cudaMemcpyToSymbol(d_NZ,  &NZ,  sizeof(int));
    cudaMemcpyToSymbol(d_NZC, &NZC, sizeof(int));
    cudaMemcpyToSymbol(d_DX,  &DX,  sizeof(double));
    cudaMemcpyToSymbol(d_DY,  &DY,  sizeof(double));
    cudaMemcpyToSymbol(d_DZ,  &DZ,  sizeof(double));
    if (s.rank == 0)
        printf("Grid: %d x %d x %d, dt=%.2e, steps=%d, mode=%s%s\n", NX, NY, NZ, TAU, NT_RUN,
               nsmode::mode_name(mode),
               mode == nsmode::MODE_BENCH ? " (f = 0, 36 FFT/step)" : " (analytic f, 48 FFT/step)");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, s.gpu));

    if (NX % s.nprocs != 0 || NY % s.nprocs != 0) {
        if (s.rank == 0)
            fprintf(stderr, "Error: nprocs=%d must divide both NX=%d and NY=%d\n", s.nprocs, NX, NY);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.nx_local  = NX / s.nprocs;
    s.ny_local  = NY / s.nprocs;
    s.x_offset  = s.rank * s.nx_local;
    s.y_offset  = s.rank * s.ny_local;
    s.nc_local  = (long long)NX * s.ny_local * NZC;
    s.nr_local  = (long long)s.nx_local * NY * NZ;

    printf("  Rank %d (node-local %d) -> GPU %d (%s)  X=[%d,%d)  Y=[%d,%d)\n",
           s.rank, s.local_rank, s.gpu, prop.name,
           s.x_offset, s.x_offset + s.nx_local, s.y_offset, s.y_offset + s.ny_local);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (s.rank == 0) cout << "============================================================\n" << flush;

    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    MPI_Comm world = MPI_COMM_WORLD;
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &world));

    // SetSubformatDefault MUST be called BEFORE MakePlan3d.
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_r2c,
        CUFFT_XT_FORMAT_INPLACE,
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_c2r,
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
        CUFFT_XT_FORMAT_INPLACE));

    // The two plans never execute at the same time, so they can share one cuFFT
    // work area instead of auto-allocating one each. Sequence as in NVIDIA's
    // cuFFTMp sample r2c_c2r_shared_scratch. Set NS_SHARE_WORK_AREA=0 to disable.
    int share_ws = 1;
    if (s.rank == 0) {
        const char* env = getenv("NS_SHARE_WORK_AREA");
        if (env && strcmp(env, "0") == 0) share_ws = 0;
    }
    MPI_Bcast(&share_ws, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (share_ws) {
        CUFFT_CHECK(cufftSetAutoAllocation(plan_r2c, 0));
        CUFFT_CHECK(cufftSetAutoAllocation(plan_c2r, 0));
    }

    size_t ws_r2c = 0, ws_c2r = 0;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    // nvshmem_malloc needs the same size on every rank: max over plans and ranks.
    long long ws_shared = (long long)std::max(ws_r2c, ws_c2r);
    MPI_Allreduce(MPI_IN_PLACE, &ws_shared, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    void* shared_work_area = nullptr;
    if (share_ws && ws_shared > 0) {
        shared_work_area = nvshmem_malloc((size_t)ws_shared);
        if (!shared_work_area) {
            fprintf(stderr, "Rank %d: nvshmem_malloc(%lld bytes) failed for the shared work area\n",
                    s.rank, ws_shared);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUFFT_CHECK(cufftSetWorkArea(plan_r2c, shared_work_area));
        CUFFT_CHECK(cufftSetWorkArea(plan_c2r, shared_work_area));
    }

    alloc_state(s, plan_r2c);

    // Warm up all FFT buffers before setting initial conditions.
    warmup_buffers(plan_r2c, plan_c2r, s);

    if (s.rank == 0) cout << "Setting initial conditions...\n" << flush;
    {
        int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);
        kernel_fill_velocity<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
            s.nx_local, s.x_offset, 0.0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    FFT_FORWARD(plan_r2c, s.V1_buf);
    FFT_FORWARD(plan_r2c, s.V2_buf);
    FFT_FORWARD(plan_r2c, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double inv_N = 1.0/((double)NX*NY*NZ);
    {
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.nc_local, inv_N);
        // Project the initial condition to divergence-free space (as all other versions do).
        kernel_make_div_free<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
            s.ny_local, s.y_offset);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    g_timer.init();
    double t_wall = 0.0;
    for (int it = 0; it < NT_RUN; ++it) {
        double t_cur = it * TAU;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        g_timer.start();
        rk4_step(plan_r2c, plan_c2r, s, t_cur);
        CUDA_CHECK(cudaDeviceSynchronize());
        g_timer.collect();
        MPI_Barrier(MPI_COMM_WORLD);
        double dt_step = MPI_Wtime() - t0, dt_max = 0.0;
        MPI_Allreduce(&dt_step, &dt_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_wall += dt_max;
    }
    double phase_max[nsmode::PH_COUNT];
    nsmode::reduce_phases_max(g_timer.acc, phase_max);
    g_timer.destroy();

    // Diagnostics (not timed).
    {
        double t_final = NT_RUN * TAU;
        double L2_or_E, max_div;
        const bool verify = (mode == nsmode::MODE_VERIFY);
        compute_diagnostics(plan_r2c, plan_c2r, s, t_final, verify, L2_or_E, max_div);
        if (s.rank == 0) {
            if (verify)
                cout << "  L2 error   (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(4) << L2_or_E << "\n";
            else
                cout << "  Kinetic energy 0.5*sum|v|^2 (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(10) << L2_or_E << "\n";
            cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << max_div << "\n";
        }
    }

    if (s.rank == 0) {
        cout << "============================================================\n";
        cout << "  Total steps:     " << NT_RUN << "\n";
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall << " s\n";
        cout << "  Avg per step:    " << t_wall / NT_RUN << " s\n" << flush;
        nsmode::print_phase_summary(phase_max, t_wall, NT_RUN, mode);
        cout << "============================================================\n" << flush;
    }

    free_state(s);
    // Must precede cufftDestroy, which may finalize NVSHMEM.
    if (shared_work_area) nvshmem_free(shared_work_area);
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
