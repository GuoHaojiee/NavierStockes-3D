/**
 * NavierStokes_periodic_cufftmp_mgpu.cu
 * Navier-Stokes solver — cuFFTMp built-in slab, single-node multi-GPU
 *
 * Architecture overview
 * ─────────────────────
 * • NO cufftXtSetDistribution — uses cuFFTMp internal slab decomposition only.
 * • Buffers allocated with cufftXtMalloc (NVSHMEM symmetric heap).
 *   Regular cudaMalloc pointers are invalid as FFT in/out; they crash inside NVSHMEM.
 * • In-place transforms: one cudaLibXtDesc* buffer doubles as real (X-slab, padded
 *   stride 2*NZC) and spectral (Y-slab SHUFFLED).
 * • After D2Z the buffer layout is CUFFT_XT_FORMAT_INPLACE_SHUFFLED:
 *     kz      = idx % NZC
 *     local_y = (idx / NZC) % ny_local
 *     gx      = idx / (ny_local * NZC)
 *     gy      = rank * ny_local + local_y
 * • Before Z2D the descriptor's subFormat must be INPLACE_SHUFFLED.
 *   force_z2d_format() / force_d2z_format() handle that.
 * • Every user kernel launch is followed by cudaDeviceSynchronize() because
 *   cuFFTMp uses internal streams and will not wait for stream-0 work.
 *
 * CRITICAL launch requirement
 * ───────────────────────────
 *   export NVSHMEM_BOOTSTRAP=MPI
 *   export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_mpi.so
 * Without these two lines cufftMakePlan3d returns CUFFT_INTERNAL_ERROR (5).
 *
 * Compile (inside enroot container):
 *   nvcc -O3 -std=c++17 -arch=sm_80 -ccbin /usr/bin/mpicxx \
 *        -Xcompiler -fopenmp \
 *        -I$CUFFTMP_HOME/include/cufftmp -I$CUFFTMP_HOME/include \
 *        -I$NVSHMEM_HOME/include \
 *        NavierStokes_periodic_cufftmp_mgpu.cu \
 *        -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib \
 *        -L/usr/local/cuda-12.2/lib64 \
 *        -lcufftMp -lnvshmem_host $NVSHMEM_HOME/lib/libnvshmem.a \
 *        -lcufft -lcudart -lmpi -lm \
 *        -Xlinker -rpath,$CUFFTMP_HOME/lib \
 *        -Xlinker -rpath,$NVSHMEM_HOME/lib \
 *        -o navier_stokes_cufftmp_mgpu_docker
 *
 * Run (4 GPUs on 1 node):
 *   mpirun -np 4 ./navier_stokes_cufftmp_mgpu_docker
 */

#include <iostream>
#include <cmath>
#include <iomanip>
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
// Grid / simulation parameters — host-side (set at runtime from argv)
// ============================================================
static int    NX, NY, NZ, NZC, NT_TOTAL, NT_RUN;
static double LX, LY, LZ, DX, DY, DZ, TAU;

__device__ __constant__ int    d_NX, d_NY, d_NZ, d_NZC;
__device__ __constant__ double d_DX, d_DY, d_DZ;

constexpr int BLOCK = 256;

// ============================================================
// Error macros
// ============================================================
#define CUDA_CHECK(e) do {                                              \
    cudaError_t _e = (e);                                               \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        MPI_Abort(MPI_COMM_WORLD, 1); }                                 \
} while (0)

#define CUFFT_CHECK(e) do {                                             \
    cufftResult _e = (e);                                               \
    if (_e != CUFFT_SUCCESS) {                                          \
        fprintf(stderr, "cuFFT error %s:%d: code %d\n",                \
                __FILE__, __LINE__, (int)_e);                           \
        MPI_Abort(MPI_COMM_WORLD, 1); }                                 \
} while (0)

// ============================================================
// cuFFTMp helper — subFormat manipulation
// ============================================================
// After D2Z the descriptor is automatically INPLACE_SHUFFLED (Y-slab).
// Before Z2D we must set INPLACE_SHUFFLED; before D2Z we must set INPLACE.
static inline void force_d2z_format(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE;
}
static inline void force_z2d_format(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
}
static inline void* gpu_ptr(cudaLibXtDesc* d) {
    return d->descriptor->data[0];
}

// ============================================================
// Analytical solution
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
// Real-space kernels  (X-slab, padded stride 2*NZC)
// ============================================================
__global__ void kernel_fill_velocity(double* V1, double* V2, double* V3,
                                      int nx_local, int x_offset, double t) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nx_local * d_NY * d_NZ) return;
    int k  = (int)(idx % d_NZ);
    int j  = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    double x = (x_offset + lx) * d_DX, y = j * d_DY, z = k * d_DZ;
    long long p = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
    V1[p] = func_V1(x,y,z,t); V2[p] = func_V2(x,y,z,t); V3[p] = func_V3(x,y,z,t);
}

__global__ void kernel_fill_forcing(double* W1, double* W2, double* W3,
                                     int nx_local, int x_offset, double t) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nx_local * d_NY * d_NZ) return;
    int k  = (int)(idx % d_NZ);
    int j  = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    double x = (x_offset + lx) * d_DX, y = j * d_DY, z = k * d_DZ;
    long long p = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
    W1[p] = func_f1(x,y,z,t); W2[p] = func_f2(x,y,z,t); W3[p] = func_f3(x,y,z,t);
}

__global__ void kernel_cross_product(const double* V1, const double* V2, const double* V3,
                                      double* rot1, double* rot2, double* rot3,
                                      int nx_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nx_local * d_NY * d_NZ) return;
    int k  = (int)(idx % d_NZ);
    int j  = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    long long p = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
    double v1=V1[p], v2=V2[p], v3=V3[p], w1=rot1[p], w2=rot2[p], w3=rot3[p];
    rot1[p] = v2*w3 - v3*w2;
    rot2[p] = v3*w1 - v1*w3;
    rot3[p] = v1*w2 - v2*w1;
}

__global__ void kernel_error_sq(const double* V1, const double* V2, const double* V3,
                                  double* scratch, int nx_local, int x_offset, double t) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nx_local * d_NY * d_NZ) return;
    int k  = (int)(idx % d_NZ);
    int j  = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    double x = (x_offset + lx) * d_DX, y = j * d_DY, z = k * d_DZ;
    long long p = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
    double d1=V1[p]-func_V1(x,y,z,t), d2=V2[p]-func_V2(x,y,z,t), d3=V3[p]-func_V3(x,y,z,t);
    scratch[idx] = d1*d1 + d2*d2 + d3*d3;
}

// ============================================================
// Spectral-space kernels  (Y-slab SHUFFLED: gx × local_y × kz)
// ============================================================
__global__ void kernel_scale_cplx(GCplx* A, long long n, double sc) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    A[idx].x *= sc; A[idx].y *= sc;
}

__global__ void kernel_compute_rot(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                    GCplx* rot1, GCplx* rot2, GCplx* rot3,
                                    int ny_local, int y_offset) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    if (idx >= nc_local) return;
    int kz  = (int)(idx % d_NZC);
    int ly  = (int)((idx / d_NZC) % ny_local);
    int gx  = (int)(idx / ((long long)ny_local * d_NZC));
    int gy  = y_offset + ly;
    double kx  = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky  = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    rot1[idx].x = -(ky*V3[idx].y - kzd*V2[idx].y); rot1[idx].y = ky*V3[idx].x - kzd*V2[idx].x;
    rot2[idx].x = -(kzd*V1[idx].y - kx*V3[idx].y); rot2[idx].y = kzd*V1[idx].x - kx*V3[idx].x;
    rot3[idx].x = -(kx*V2[idx].y - ky*V1[idx].y);  rot3[idx].y = kx*V2[idx].x - ky*V1[idx].x;
}

__global__ void kernel_compute_viscous(const GCplx* V, GCplx* visc,
                                        int ny_local, int y_offset) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    if (idx >= nc_local) return;
    int kz  = (int)(idx % d_NZC);
    int ly  = (int)((idx / d_NZC) % ny_local);
    int gx  = (int)(idx / ((long long)ny_local * d_NZC));
    int gy  = y_offset + ly;
    double kx  = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky  = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double k2  = kx*kx + ky*ky + kzd*kzd;
    visc[idx].x = -k2 * V[idx].x;
    visc[idx].y = -k2 * V[idx].y;
}

__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
                                      int ny_local, int y_offset) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    if (idx >= nc_local) return;
    int kz  = (int)(idx % d_NZC);
    int ly  = (int)((idx / d_NZC) % ny_local);
    int gx  = (int)(idx / ((long long)ny_local * d_NZC));
    int gy  = y_offset + ly;
    double kx  = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky  = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double k2  = kx*kx + ky*ky + kzd*kzd;
    if (k2 < 1e-10) return;
    double dr = -(kx*V1[idx].y + ky*V2[idx].y + kzd*V3[idx].y);
    double di  =   kx*V1[idx].x + ky*V2[idx].x + kzd*V3[idx].x;
    double pr  = dr / (-k2), pi = di / (-k2);
    V1[idx].x -= -kx*pi;  V1[idx].y -= kx*pr;
    V2[idx].x -= -ky*pi;  V2[idx].y -= ky*pr;
    V3[idx].x -= -kzd*pi; V3[idx].y -= kzd*pr;
}

__global__ void kernel_div_abs(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                 double* scratch, int ny_local, int y_offset) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    if (idx >= nc_local) return;
    int kz  = (int)(idx % d_NZC);
    int ly  = (int)((idx / d_NZC) % ny_local);
    int gx  = (int)(idx / ((long long)ny_local * d_NZC));
    int gy  = y_offset + ly;
    double kx  = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky  = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double dr  = -(kx*V1[idx].y + ky*V2[idx].y + kzd*V3[idx].y);
    double di  =   kx*V1[idx].x + ky*V2[idx].x + kzd*V3[idx].x;
    scratch[idx] = sqrt(dr*dr + di*di);
}

// ============================================================
// Layout-agnostic element-wise kernels
// ============================================================
__global__ void kernel_copy_cplx(const GCplx* src, GCplx* dst, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx];
}

__global__ void kernel_add_to_rhs(GCplx* r1, GCplx* r2, GCplx* r3,
                                   const GCplx* visc1, const GCplx* visc2, const GCplx* visc3,
                                   const GCplx* f1, const GCplx* f2, const GCplx* f3,
                                   long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    r1[idx].x += visc1[idx].x + f1[idx].x; r1[idx].y += visc1[idx].y + f1[idx].y;
    r2[idx].x += visc2[idx].x + f2[idx].x; r2[idx].y += visc2[idx].y + f2[idx].y;
    r3[idx].x += visc3[idx].x + f3[idx].x; r3[idx].y += visc3[idx].y + f3[idx].y;
}

__global__ void kernel_rk4_axpy(GCplx* V1, GCplx* V2, GCplx* V3,
                                  const GCplx* o1, const GCplx* o2, const GCplx* o3,
                                  const GCplx* k1, const GCplx* k2, const GCplx* k3,
                                  double alpha, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    V1[idx].x = o1[idx].x + alpha*k1[idx].x; V1[idx].y = o1[idx].y + alpha*k1[idx].y;
    V2[idx].x = o2[idx].x + alpha*k2[idx].x; V2[idx].y = o2[idx].y + alpha*k2[idx].y;
    V3[idx].x = o3[idx].x + alpha*k3[idx].x; V3[idx].y = o3[idx].y + alpha*k3[idx].y;
}

__global__ void kernel_rk4_update(GCplx* V1, GCplx* V2, GCplx* V3,
                                   const GCplx* o1, const GCplx* o2, const GCplx* o3,
                                   const GCplx* k1v1, const GCplx* k2v1,
                                   const GCplx* k3v1, const GCplx* k4v1,
                                   const GCplx* k1v2, const GCplx* k2v2,
                                   const GCplx* k3v2, const GCplx* k4v2,
                                   const GCplx* k1v3, const GCplx* k2v3,
                                   const GCplx* k3v3, const GCplx* k4v3,
                                   double dtd6, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    V1[idx].x = o1[idx].x + dtd6*(k1v1[idx].x + 2.*k2v1[idx].x + 2.*k3v1[idx].x + k4v1[idx].x);
    V1[idx].y = o1[idx].y + dtd6*(k1v1[idx].y + 2.*k2v1[idx].y + 2.*k3v1[idx].y + k4v1[idx].y);
    V2[idx].x = o2[idx].x + dtd6*(k1v2[idx].x + 2.*k2v2[idx].x + 2.*k3v2[idx].x + k4v2[idx].x);
    V2[idx].y = o2[idx].y + dtd6*(k1v2[idx].y + 2.*k2v2[idx].y + 2.*k3v2[idx].y + k4v2[idx].y);
    V3[idx].x = o3[idx].x + dtd6*(k1v3[idx].x + 2.*k2v3[idx].x + 2.*k3v3[idx].x + k4v3[idx].x);
    V3[idx].y = o3[idx].y + dtd6*(k1v3[idx].y + 2.*k2v3[idx].y + 2.*k3v3[idx].y + k4v3[idx].y);
}

// ============================================================
// Per-rank state
// ============================================================
struct State {
    int rank, nprocs, gpu;
    int nx_local, ny_local;
    long long nc_local, nr_local;
    int x_offset, y_offset;

    // NVSHMEM-backed buffers (cufftXtMalloc)
    cudaLibXtDesc *V1_buf, *V2_buf, *V3_buf;
    cudaLibXtDesc *rot1_buf, *rot2_buf, *rot3_buf;
    cudaLibXtDesc *work1_buf, *work2_buf, *work3_buf;

    // Regular device buffers for intermediate spectral data
    GCplx *visc1_c, *visc2_c, *visc3_c;
    GCplx *f1_c, *f2_c, *f3_c;
    GCplx *rhs1_c, *rhs2_c, *rhs3_c;
    GCplx *k1v1, *k2v1, *k3v1, *k4v1;
    GCplx *k1v2, *k2v2, *k3v2, *k4v2;
    GCplx *k1v3, *k2v3, *k3v3, *k4v3;
    GCplx *V1_orig, *V2_orig, *V3_orig;
    double *scratch;
};

static void alloc_state(State& s, cufftHandle plan_r2c) {
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V1_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V2_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V3_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot1_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot2_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot3_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work1_buf, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work2_buf, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work3_buf, CUFFT_XT_FORMAT_INPLACE));

    auto C = [&](GCplx** p) { CUDA_CHECK(cudaMalloc(p, s.nc_local * sizeof(GCplx))); };
    C(&s.visc1_c); C(&s.visc2_c); C(&s.visc3_c);
    C(&s.f1_c);    C(&s.f2_c);    C(&s.f3_c);
    C(&s.rhs1_c);  C(&s.rhs2_c);  C(&s.rhs3_c);
    C(&s.k1v1); C(&s.k2v1); C(&s.k3v1); C(&s.k4v1);
    C(&s.k1v2); C(&s.k2v2); C(&s.k3v2); C(&s.k4v2);
    C(&s.k1v3); C(&s.k2v3); C(&s.k3v3); C(&s.k4v3);
    C(&s.V1_orig); C(&s.V2_orig); C(&s.V3_orig);
    CUDA_CHECK(cudaMalloc(&s.scratch, std::max(s.nc_local, s.nr_local) * sizeof(double)));
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
// Physics helpers
// ============================================================
static void compute_nonlinear(cufftHandle pr, cufftHandle pc, State& s) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // rot = curl(V)  in spectral Y-slab
    kernel_compute_rot<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        (GCplx*)gpu_ptr(s.rot1_buf), (GCplx*)gpu_ptr(s.rot2_buf), (GCplx*)gpu_ptr(s.rot3_buf),
        s.ny_local, s.y_offset);
    force_z2d_format(s.rot1_buf); force_z2d_format(s.rot2_buf); force_z2d_format(s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // V → real space; rot → real space
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.V1_buf,   s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.V2_buf,   s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.V3_buf,   s.V3_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.rot1_buf, s.rot1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.rot2_buf, s.rot2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.rot3_buf, s.rot3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // nl = V × rot  (in-place in rot buffers)
    kernel_cross_product<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        (double*)gpu_ptr(s.rot1_buf), (double*)gpu_ptr(s.rot2_buf), (double*)gpu_ptr(s.rot3_buf),
        s.nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());

    // nl → spectral
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.rot1_buf, s.rot1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.rot2_buf, s.rot2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.rot3_buf, s.rot3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy & scale nl → rhs buffers
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot1_buf), s.rhs1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot2_buf), s.rhs2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot3_buf), s.rhs3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs3_c, nc, inv_N);
}

static void compute_rhs(cufftHandle pr, cufftHandle pc, State& s, double t) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // viscous term: -k² V̂
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.visc1_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.visc2_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.visc3_c, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    // nonlinear term → rhs{1,2,3}_c
    compute_nonlinear(pr, pc, s);

    // forcing term: fill real, forward FFT, scale, add to rhs
    force_d2z_format(s.work1_buf); force_d2z_format(s.work2_buf); force_d2z_format(s.work3_buf);
    kernel_fill_forcing<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.work1_buf), (double*)gpu_ptr(s.work2_buf), (double*)gpu_ptr(s.work3_buf),
        s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.work1_buf, s.work1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.work2_buf, s.work2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.work3_buf, s.work3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work1_buf), s.f1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work2_buf), s.f2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work3_buf), s.f3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f3_c, nc, inv_N);

    // rhs = nl + visc + f
    kernel_add_to_rhs<<<gc, BLOCK>>>(
        s.rhs1_c, s.rhs2_c, s.rhs3_c,
        s.visc1_c, s.visc2_c, s.visc3_c,
        s.f1_c, s.f2_c, s.f3_c, nc);

    // project onto divergence-free subspace
    kernel_make_div_free<<<gc, BLOCK>>>(s.rhs1_c, s.rhs2_c, s.rhs3_c, s.ny_local, s.y_offset);
}

static void save_v_orig(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.V1_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.V2_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.V3_orig, s.nc_local);
}

static void restore_v_buf(State& s) {
    // Copy back and re-mark as SHUFFLED so Z2D / kernel reads are correct
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V1_orig, (GCplx*)gpu_ptr(s.V1_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V2_orig, (GCplx*)gpu_ptr(s.V2_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V3_orig, (GCplx*)gpu_ptr(s.V3_buf), s.nc_local);
    force_z2d_format(s.V1_buf); force_z2d_format(s.V2_buf); force_z2d_format(s.V3_buf);
}

static void rk4_step(cufftHandle pr, cufftHandle pc, State& s, double t) {
    const long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);

    save_v_orig(s);

    // k1
    compute_rhs(pr, pc, s, t);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k1v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k1v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k1v3, nc);

    // k2
    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k1v1, s.k1v2, s.k1v3, 0.5*TAU, nc);
    compute_rhs(pr, pc, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k2v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k2v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k2v3, nc);

    // k3
    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k2v1, s.k2v2, s.k2v3, 0.5*TAU, nc);
    compute_rhs(pr, pc, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k3v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k3v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k3v3, nc);

    // k4
    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k3v1, s.k3v2, s.k3v3, TAU, nc);
    compute_rhs(pr, pc, s, t + TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k4v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k4v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k4v3, nc);

    // final update
    restore_v_buf(s);
    kernel_rk4_update<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k1v1, s.k2v1, s.k3v1, s.k4v1,
        s.k1v2, s.k2v2, s.k3v2, s.k4v2,
        s.k1v3, s.k2v3, s.k3v3, s.k4v3,
        TAU / 6.0, nc);
}

static void compute_diagnostics(cufftHandle pr, cufftHandle pc,
                                  State& s, double t,
                                  double& L2_err, double& max_div) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    // div check in spectral space
    kernel_div_abs<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.scratch, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_ptr<double> sp(s.scratch);
    double lmax = thrust::reduce(thrust::device, sp, sp + nc, 0.0, thrust::maximum<double>());
    MPI_Allreduce(&lmax, &max_div, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // inverse FFT → real space
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(pc, s.V3_buf, s.V3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    // L2 error
    kernel_error_sq<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        s.scratch, s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    double lsq = thrust::reduce(thrust::device, sp, sp + nr, 0.0);
    double gsq = 0.0;
    MPI_Allreduce(&lsq, &gsq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    L2_err = sqrt(gsq * DX * DY * DZ);

    // forward FFT to restore spectral state
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(pr, s.V3_buf, s.V3_buf));
    CUDA_CHECK(cudaDeviceSynchronize());
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

    if (argc < 6) {
        if (s.rank == 0) fprintf(stderr, "Usage: %s NX NY NZ dt NSTEPS\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    NX=atoi(argv[1]); NY=atoi(argv[2]); NZ=atoi(argv[3]);
    TAU=atof(argv[4]); NT_RUN=atoi(argv[5]);
    NT_TOTAL=NT_RUN; NZC=NZ/2+1;
    LX=LY=LZ=2.0*M_PI; DX=LX/NX; DY=LY/NY; DZ=LZ/NZ;
    cudaMemcpyToSymbol(d_NX, &NX, sizeof(int));
    cudaMemcpyToSymbol(d_NY, &NY, sizeof(int));
    cudaMemcpyToSymbol(d_NZ, &NZ, sizeof(int));
    cudaMemcpyToSymbol(d_NZC,&NZC,sizeof(int));
    cudaMemcpyToSymbol(d_DX, &DX, sizeof(double));
    cudaMemcpyToSymbol(d_DY, &DY, sizeof(double));
    cudaMemcpyToSymbol(d_DZ, &DZ, sizeof(double));
    if (s.rank == 0) printf("Grid: %d x %d x %d, dt=%.2e, steps=%d\n",NX,NY,NZ,TAU,NT_RUN);

    // GPU assignment: simple round-robin (single node)
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    s.gpu = s.rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(s.gpu));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, s.gpu));

    if (NX % s.nprocs != 0 || NY % s.nprocs != 0) {
        if (s.rank == 0)
            fprintf(stderr, "Error: nprocs=%d must divide NX=%d and NY=%d\n",
                    s.nprocs, NX, NY);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    s.nx_local = NX / s.nprocs;
    s.ny_local = NY / s.nprocs;
    s.x_offset = s.rank * s.nx_local;
    s.y_offset = s.rank * s.ny_local;
    s.nc_local = (long long)NX * s.ny_local * NZC;
    s.nr_local = (long long)s.nx_local * NY * NZ;

    if (s.rank == 0) {
        cout << "============================================================\n"
             << "  Navier-Stokes — cuFFTMp, Single-Node Multi-GPU\n"
             << "============================================================\n"
             << "MPI ranks: " << s.nprocs << "  GPUs/node: " << num_gpus << "\n"
             << "Grid: " << NX << "x" << NY << "x" << NZ << "\n"
             << "  Real  (X-slab): nx_local=" << s.nx_local << " per rank\n"
             << "  Spec. (Y-slab): ny_local=" << s.ny_local << " per rank\n"
             << "Steps: " << NT_RUN << "/" << NT_TOTAL << "  dt=" << TAU << "\n"
             << "============================================================\n";
    }
    printf("  Rank %d → GPU %d (%s)  X=[%d,%d)  Y=[%d,%d)\n",
           s.rank, s.gpu, prop.name,
           s.x_offset, s.x_offset + s.nx_local,
           s.y_offset, s.y_offset + s.ny_local);
    MPI_Barrier(MPI_COMM_WORLD);
    if (s.rank == 0) cout << "============================================================\n";

    // Build cuFFTMp plans (no cufftXtSetDistribution — built-in slab)
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    MPI_Comm world = MPI_COMM_WORLD;
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &world));
    size_t ws_r2c = 0, ws_c2r = 0;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    alloc_state(s, plan_r2c);

    // Initial conditions
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
    {
        const double inv_N = 1.0 / (double)(NX * NY * NZ);
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.nc_local, inv_N);
        kernel_make_div_free<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
            s.ny_local, s.y_offset);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_wall = 0.0;
    for (int it = 0; it < NT_RUN; ++it) {
        double tc = it * TAU;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        rk4_step(plan_r2c, plan_c2r, s, tc);
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        double dt = MPI_Wtime() - t0, dtmax = 0.0;
        MPI_Allreduce(&dt, &dtmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_wall += dtmax;
    }
    {
        double t_final = NT_RUN * TAU;
        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, s, t_final, L2_err, max_div);
        if (s.rank == 0)
            cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << L2_err << "\n";
    }

    if (s.rank == 0)
        cout << "============================================================\n"
             << "  Total wall: " << fixed << setprecision(4) << t_wall << " s\n"
             << "  Avg/step:   " << t_wall / NT_RUN << " s\n"
             << "============================================================\n";

    free_state(s);
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
