// NavierStokes_periodic_cufftmp_multinode.cu
// =============================================================================
// ROOT CAUSE (found via NaN bisection, 2025):
//   rot_buf and work_buf were allocated as CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
//   which only allocates Y-slab complex size (NX * ny_local * NZC * 2 doubles).
//   However, FFT_INVERSE writes X-slab real output (nx_local * NY * 2*NZC doubles),
//   which is 2x larger for a square grid.  This caused out-of-bounds writes into
//   adjacent allocations, producing NaN in cross_product (confirmed on 1/4 ranks,
//   then propagated to 4/4 via NVSHMEM all-to-all in the next FFT_FORWARD).
//
// FIX:
//   1. ALL FFT buffers (V_buf, rot_buf, work_buf) allocated as INPLACE (larger).
//   2. force_d2z_format / force_z2d_format embedded into FFT_FORWARD / FFT_INVERSE
//      macros, so cuFFTMp always receives the correct subFormat before each call.
//   3. kernel_compute_rot writes Y-slab complex to rot_buf: explicit
//      force_z2d_format call before the kernel sets subFormat correctly.
//   4. kernel_fill_forcing writes X-slab real to work_buf: explicit
//      force_d2z_format call before the kernel.
//   5. restore_v_buf: no dummy FFT needed; force_z2d_format alone suffices
//      (confirmed by subFormat hack test: L2 < 1e-10).
//   6. warmup_buffers: each FFT buffer is driven through one FORWARD+INVERSE
//      cycle on zero data after alloc_state, before IC.  This forces cuFFTMp
//      to initialize per-buffer transpose metadata (NVSHMEM symmetric heap,
//      peer mappings) so that subsequent hacked-subFormat FFT calls on
//      rot/work buffers work correctly rather than producing silent garbage.
//
// Diagnosis tool (NAN_DIAG=1):
//   Bisects NaN to first occurrence across compute_nonlinear / compute_rhs /
//   rk4_step Stage 1 using thrust::reduce + MPI_Allreduce per checkpoint.
//   Set NAN_DIAG=0 after root cause is resolved to avoid per-step overhead.
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
typedef cufftDoubleComplex GCplx;

// Set to 1 to run a forward+inverse FFT round-trip diagnostic at startup.
// If this prints L2 error > 1e-10 the FFT layer itself is broken (most
// likely the kernel indexing does not match cuFFTMp's INPLACE_SHUFFLED
// Y-slab layout). Aborts before simulation.
#define RUN_FFT_TEST 1

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
// subFormat helpers -- mirror cuFFTXt's manual subFormat hack.
// Call IMMEDIATELY before the FFT exec, or immediately after a kernel writes
// data of the matching layout directly into an FFT buffer.
//
//   CUFFT_XT_FORMAT_INPLACE           = 2  -> real X-slab        (D2Z input)
//   CUFFT_XT_FORMAT_INPLACE_SHUFFLED  = 3  -> complex Y-slab     (Z2D input)
// =============================================================================
static inline void force_d2z_format(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE;          // = 2, real X-slab
}
static inline void force_z2d_format(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED; // = 3, complex Y-slab
}

// FFT execution macros: pure cufftXtExecDescriptor calls.
// cuFFTMp drives its own internal state machine:
//   FFT_FORWARD: INPLACE (X-slab real)     -> INPLACE_SHUFFLED (Y-slab complex)
//   FFT_INVERSE: INPLACE_SHUFFLED (Y-slab) -> INPLACE (X-slab real)
// We rely on cuFFTMp's state, not on desc->subFormat.
// force_d2z_format / force_z2d_format are retained ONLY for the diagnostic
// test in the RUN_FFT_TEST block; they are NOT called in the simulation path.
#define FFT_FORWARD(plan, buf) do {                                           \
    force_d2z_format(buf);                                                    \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_FORWARD));     \
} while(0)
#define FFT_INVERSE(plan, buf) do {                                           \
    force_z2d_format(buf);                                                    \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_INVERSE));     \
} while(0)

// =============================================================================
// NaN bisection diagnostic
// Set NAN_DIAG=1 and run with NT_RUN=1 to find first occurrence of NaN/Inf.
// Set NAN_DIAG=0 after root cause is found.
// =============================================================================
#define NAN_DIAG 0

#if NAN_DIAG
// Returns true if buffer contains NaN/Inf on ANY rank.
// n_doubles = total double-precision elements to scan (complex => 2*nc; real => nr or n_padded).
static bool has_nan(const void* buf, long long n_doubles, const char* label, int rank, int nprocs) {
    thrust::device_ptr<const double> dp((const double*)buf);
    double local_sum = thrust::reduce(thrust::device, dp, dp + n_doubles, 0.0);
    int local_bad = (std::isnan(local_sum) || std::isinf(local_sum)) ? 1 : 0;
    int global_bad = 0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_bad > 0 && rank == 0)
        printf("  *** NaN/Inf in: %-50s (bad on %d/%d ranks, local sum=%g)\n",
               label, global_bad, nprocs, local_sum);
    fflush(stdout);
    return global_bad > 0;
}
#define CHECK_CPLX(buf, n, label)  has_nan((const void*)(buf), 2LL*(long long)(n), label, s.rank, s.nprocs)
#define CHECK_REAL(buf, n, label)  has_nan((const void*)(buf),   (long long)(n),   label, s.rank, s.nprocs)
#else
#define CHECK_CPLX(...) ((void)0)
#define CHECK_REAL(...) ((void)0)
#endif

// =============================================================================
// Manufactured-solution functions (unchanged)
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
    long long pidx = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
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
    long long pidx = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
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
    long long pidx = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
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
    long long pidx = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
    double d1 = V1[pidx] - func_V1(x, y, z, t);
    double d2 = V2[pidx] - func_V2(x, y, z, t);
    double d3 = V3[pidx] - func_V3(x, y, z, t);
    scratch[idx] = d1*d1 + d2*d2 + d3*d3;
}

// FFT smoke test: write a smooth real X-slab pattern + zero padding bytes.
__global__ void kernel_fill_test_real(double* V, int nx_local, int x_offset) {
    long long n_padded = (long long)nx_local * d_NY * (2*d_NZC);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_padded) return;
    int k_padded = (int)(idx % (2*d_NZC));
    int j        = (int)((idx / (2*d_NZC)) % d_NY);
    int lx       = (int)(idx / ((long long)d_NY * (2*d_NZC)));
    int gi = x_offset + lx;
    if (k_padded >= d_NZ) { V[idx] = 0.0; return; } // zero R2C padding
    double x = gi * d_DX, y = j * d_DY, z = k_padded * d_DZ;
    V[idx] = sin(x) * cos(2.0*y) * sin(3.0*z) + 0.5;
}

__global__ void kernel_test_error(const double* V, double* scratch,
                                   int nx_local, int x_offset) {
    long long nr_local = (long long)nx_local * d_NY * d_NZ;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_local) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int lx = (int)(idx / ((long long)d_NY * d_NZ));
    int gi = x_offset + lx;
    double x = gi * d_DX, y = j * d_DY, z = k * d_DZ;
    long long pidx = (long long)lx * d_NY * 2*d_NZC + j * 2*d_NZC + k;
    double ref = sin(x) * cos(2.0*y) * sin(3.0*z) + 0.5;
    double d   = V[pidx] - ref;
    scratch[idx] = d*d;
}

// =============================================================================
// Spectral-space kernels (Y-slab complex, packed stride NZC)
// =============================================================================
__global__ void kernel_scale_cplx(GCplx* A, long long nc_local, double scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    A[idx].x *= scale; A[idx].y *= scale;
}

__global__ void kernel_scale_real(double* A, long long n, double scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    A[idx] *= scale;
}

__global__ void kernel_compute_rot(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                    GCplx* rot1, GCplx* rot2, GCplx* rot3,
                                    int ny_local, int y_offset) {
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    // Layout-A (confirmed by probe): [gx][local_y][kz]
    // idx = gx * ny_local * NZC + local_y * NZC + kz
    int kz      = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * d_NZC));
    int gy = y_offset + local_y;
    double kx = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
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
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    // Layout-A (confirmed by probe): [gx][local_y][kz]
    int kz      = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx      = (int)(idx / ((long long)ny_local * d_NZC));
    int gy = y_offset + local_y;
    double kx = (gx <= d_NX/2) ? (double)gx : (double)(gx - d_NX);
    double ky = (gy <= d_NY/2) ? (double)gy : (double)(gy - d_NY);
    double kzd = (double)kz;
    double k2 = kx*kx + ky*ky + kzd*kzd;
    visc[idx].x = -k2 * V[idx].x;
    visc[idx].y = -k2 * V[idx].y;
}

__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
                                      int ny_local, int y_offset) {
    long long nc_local = (long long)d_NX * ny_local * d_NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    // Layout-A (confirmed by probe): [gx][local_y][kz]
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
    // Layout-A (confirmed by probe): [gx][local_y][kz]
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
                                   const GCplx* k1v1, const GCplx* k2v1, const GCplx* k3v1, const GCplx* k4v1,
                                   const GCplx* k1v2, const GCplx* k2v2, const GCplx* k3v2, const GCplx* k4v2,
                                   const GCplx* k1v3, const GCplx* k2v3, const GCplx* k3v3, const GCplx* k4v3,
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

// =============================================================================
// Per-rank state
// =============================================================================
struct State {
    int rank, nprocs, gpu;
    int local_rank, local_size;
    int nx_local, ny_local;
    long long nc_local, nr_local;
    int x_offset, y_offset;

    // FFT-managed buffers -- ALL allocated as INPLACE (largest layout).
    // subFormat is set explicitly via force_d2z_format / force_z2d_format
    // (embedded in FFT_FORWARD / FFT_INVERSE macros, and at kernel write sites).
    cudaLibXtDesc *V1_buf, *V2_buf, *V3_buf;
    cudaLibXtDesc *rot1_buf, *rot2_buf, *rot3_buf;
    cudaLibXtDesc *work1_buf, *work2_buf, *work3_buf;

    // Plain cudaMalloc'd complex scratch
    GCplx *visc1_c, *visc2_c, *visc3_c;
    GCplx *f1_c, *f2_c, *f3_c;
    GCplx *rhs1_c, *rhs2_c, *rhs3_c;
    GCplx *k1v1, *k2v1, *k3v1, *k4v1;
    GCplx *k1v2, *k2v2, *k3v2, *k4v2;
    GCplx *k1v3, *k2v3, *k3v3, *k4v3;
    GCplx *V1_orig, *V2_orig, *V3_orig;
    double *scratch;
};

// Helper: cufftXtMalloc + immediate memset to zero, to guarantee no NaN bit
// patterns leak from uninitialized NVSHMEM heap memory into FFT operations.
static void alloc_and_zero(cufftHandle plan, cudaLibXtDesc** desc, cufftXtSubFormat fmt) {
    CUFFT_CHECK(cufftXtMalloc(plan, desc, fmt));
    CUDA_CHECK(cudaMemset((*desc)->descriptor->data[0], 0,
                          (*desc)->descriptor->size[0]));
}

static void alloc_state(State& s, cufftHandle plan_r2c) {
    // ALL buffers allocated as INPLACE -- this is the LARGER of the two layouts
    // (INPLACE = X-slab real padded; SHUFFLED = Y-slab complex).
    // For NX=NY=NZ=128, 4 ranks: INPLACE=1,064,960 doubles vs SHUFFLED=532,480.
    // Allocating rot_buf/work_buf as SHUFFLED would under-size them, causing
    // FFT_INVERSE output (INPLACE-sized) to overflow into adjacent memory -> NaN.
    // We use force_d2z_format / force_z2d_format (embedded in FFT_FORWARD /
    // FFT_INVERSE macros) to tell cuFFTMp the layout before every FFT call.
    alloc_and_zero(plan_r2c, &s.V1_buf,    CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.V2_buf,    CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.V3_buf,    CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.rot1_buf,  CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.rot2_buf,  CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.rot3_buf,  CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.work1_buf, CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.work2_buf, CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.work3_buf, CUFFT_XT_FORMAT_INPLACE);

    auto C = [&](GCplx** p){
        CUDA_CHECK(cudaMalloc(p, s.nc_local*sizeof(GCplx)));
        CUDA_CHECK(cudaMemset(*p, 0, s.nc_local*sizeof(GCplx)));
    };
    C(&s.visc1_c); C(&s.visc2_c); C(&s.visc3_c);
    C(&s.f1_c);    C(&s.f2_c);    C(&s.f3_c);
    C(&s.rhs1_c);  C(&s.rhs2_c);  C(&s.rhs3_c);
    C(&s.k1v1); C(&s.k2v1); C(&s.k3v1); C(&s.k4v1);
    C(&s.k1v2); C(&s.k2v2); C(&s.k3v2); C(&s.k4v2);
    C(&s.k1v3); C(&s.k2v3); C(&s.k3v3); C(&s.k4v3);
    C(&s.V1_orig); C(&s.V2_orig); C(&s.V3_orig);

    long long sc_size = std::max(s.nc_local, s.nr_local);
    CUDA_CHECK(cudaMalloc(&s.scratch, sc_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(s.scratch,  0, sc_size * sizeof(double)));
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

// =============================================================================
// Warm-up: drive each FFT buffer through one full FORWARD+INVERSE cycle on
// zero data.  This forces cuFFTMp to lazy-initialize its per-buffer internal
// state (transpose plans, NVSHMEM symmetric heap allocations, etc.) under the
// natural state flow.
//
// Without this, rot1/2/3_buf and work1/2/3_buf are VIRGIN (alloc'd as INPLACE,
// never FFT'd) when compute_nonlinear's first FFT_INVERSE runs with a hacked
// subFormat.  cuFFTMp's transpose metadata is set up for the alloc format
// (INPLACE), so a hacked SHUFFLED inverse on a virgin buffer produces garbage
// silently.  One warm-up round-trip forces the metadata to be established
// consistently before any simulation FFT call.
// =============================================================================
static void warmup_buffers(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s) {
    auto warm = [&](cudaLibXtDesc* buf) {
        CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_size_bytes(buf)));
        FFT_FORWARD(plan_r2c, buf);   // INPLACE (zeros) -> SHUFFLED (zeros)
        FFT_INVERSE(plan_c2r, buf);   // SHUFFLED (zeros) -> INPLACE (zeros)
        CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_size_bytes(buf)));
    };
    warm(s.V1_buf);    warm(s.V2_buf);    warm(s.V3_buf);
    warm(s.rot1_buf);  warm(s.rot2_buf);  warm(s.rot3_buf);
    warm(s.work1_buf); warm(s.work2_buf); warm(s.work3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (s.rank == 0) printf("FFT buffers warmed up.\n");
}

// =============================================================================
// FFT round-trip smoke test
// =============================================================================
static double fft_roundtrip_test(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s) {
    const long long n_padded = (long long)s.nx_local * NY * (2*NZC);
    const long long nr_local = s.nr_local;
    const double inv_N = 1.0/(double)(NX*NY*NZ);

    int gp = (int)((n_padded + BLOCK - 1) / BLOCK);
    int gr = (int)((nr_local + BLOCK - 1) / BLOCK);

    // V1_buf was just alloc_and_zero'd above.
    // Fill real X-slab pattern, then mark subFormat as INPLACE (matches data).
    kernel_fill_test_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.nx_local, s.x_offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    FFT_FORWARD(plan_r2c, s.V1_buf);   // X-slab real -> Y-slab complex
    CUDA_CHECK(cudaDeviceSynchronize());

    FFT_INVERSE(plan_c2r, s.V1_buf);   // Y-slab complex -> X-slab real (un-normalized)
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), n_padded, inv_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_test_error<<<gr, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.scratch, s.nx_local, s.x_offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<double> sp(s.scratch);
    double local_sq = thrust::reduce(thrust::device, sp, sp + nr_local, 0.0);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double l2 = sqrt(global_sq * DX * DY * DZ);

    // Re-zero V1_buf so simulation starts from a clean state.
    CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));
    return l2;
}

// =============================================================================
// Physics: nonlinear convective term V x (curl V), forcing, viscous, div-free
// =============================================================================
//
// ENTRY: V_buf data is Y-slab complex (spectral). subFormat may be stale.
// EXIT:  V_buf data is X-slab real (after the Z2D below); rot_buf data is
//        Y-slab complex (after the final FFT_FORWARD).
static void compute_nonlinear(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s) {
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    const long long nc = s.nc_local, nr = s.nr_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((nr + BLOCK - 1) / BLOCK);

    CHECK_CPLX(gpu_ptr(s.V1_buf), nc, "nonlin ENTRY V1_buf (spec)");
    CHECK_CPLX(gpu_ptr(s.V2_buf), nc, "nonlin ENTRY V2_buf (spec)");
    CHECK_CPLX(gpu_ptr(s.V3_buf), nc, "nonlin ENTRY V3_buf (spec)");

    // rot_buf will receive Y-slab complex data from kernel_compute_rot.
    // Mark subFormat = SHUFFLED so the subsequent FFT_INVERSE (inside the macro)
    // interprets the layout correctly. FFT_INVERSE macro also calls force_z2d_format
    // again for safety, but we do it here explicitly before the kernel write.
    force_z2d_format(s.rot1_buf);
    force_z2d_format(s.rot2_buf);
    force_z2d_format(s.rot3_buf);

    kernel_compute_rot<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        (GCplx*)gpu_ptr(s.rot1_buf), (GCplx*)gpu_ptr(s.rot2_buf), (GCplx*)gpu_ptr(s.rot3_buf),
        s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(gpu_ptr(s.rot1_buf), nc, "after compute_rot rot1 (spec)");
    CHECK_CPLX(gpu_ptr(s.rot2_buf), nc, "after compute_rot rot2 (spec)");
    CHECK_CPLX(gpu_ptr(s.rot3_buf), nc, "after compute_rot rot3 (spec)");

    // FFT_INVERSE macro calls force_z2d_format internally before each exec.
    FFT_INVERSE(plan_c2r, s.V1_buf);
    FFT_INVERSE(plan_c2r, s.V2_buf);
    FFT_INVERSE(plan_c2r, s.V3_buf);
    FFT_INVERSE(plan_c2r, s.rot1_buf);
    FFT_INVERSE(plan_c2r, s.rot2_buf);
    FFT_INVERSE(plan_c2r, s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_REAL(gpu_ptr(s.V1_buf), nr, "after Z2D V1 (real)");
    CHECK_REAL(gpu_ptr(s.V2_buf), nr, "after Z2D V2 (real)");
    CHECK_REAL(gpu_ptr(s.V3_buf), nr, "after Z2D V3 (real)");
    CHECK_REAL(gpu_ptr(s.rot1_buf), nr, "after Z2D rot1 (real)");
    CHECK_REAL(gpu_ptr(s.rot2_buf), nr, "after Z2D rot2 (real)");
    CHECK_REAL(gpu_ptr(s.rot3_buf), nr, "after Z2D rot3 (real)");

    kernel_cross_product<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        (double*)gpu_ptr(s.rot1_buf), (double*)gpu_ptr(s.rot2_buf), (double*)gpu_ptr(s.rot3_buf),
        s.nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_REAL(gpu_ptr(s.rot1_buf), nr, "after cross_product rot1 (real)");
    CHECK_REAL(gpu_ptr(s.rot2_buf), nr, "after cross_product rot2 (real)");
    CHECK_REAL(gpu_ptr(s.rot3_buf), nr, "after cross_product rot3 (real)");

    // FFT_FORWARD macro calls force_d2z_format internally before each exec.
    FFT_FORWARD(plan_r2c, s.rot1_buf);
    FFT_FORWARD(plan_r2c, s.rot2_buf);
    FFT_FORWARD(plan_r2c, s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(gpu_ptr(s.rot1_buf), nc, "after D2Z rot1 (spec)");
    CHECK_CPLX(gpu_ptr(s.rot2_buf), nc, "after D2Z rot2 (spec)");
    CHECK_CPLX(gpu_ptr(s.rot3_buf), nc, "after D2Z rot3 (spec)");

    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot1_buf), s.rhs1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot2_buf), s.rhs2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot3_buf), s.rhs3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs3_c, nc, inv_N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(s.rhs1_c, nc, "nonlin EXIT rhs1_c (V\xd7rot only)");
    CHECK_CPLX(s.rhs2_c, nc, "nonlin EXIT rhs2_c (V\xd7rot only)");
    CHECK_CPLX(s.rhs3_c, nc, "nonlin EXIT rhs3_c (V\xd7rot only)");
}

// ENTRY: V_buf data is Y-slab complex (spectral, from initial conditions or
//        from the previous restore_v_buf).
// EXIT:  V_buf data is X-slab real (left over from compute_nonlinear).
//        rhs_*_c contain projected, scaled, summed RHS.
static void compute_rhs(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    const long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);

    CHECK_CPLX(gpu_ptr(s.V1_buf), nc, "rhs ENTRY V1_buf (spec)");

    // V_buf is in SHUFFLED state (spectral) on entry.
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.visc1_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.visc2_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.visc3_c, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(s.visc1_c, nc, "after compute_viscous visc1_c");

    // Nonlinear term. After this: V_buf is X-slab real (INPLACE), rot_buf is Y-slab complex.
    compute_nonlinear(plan_r2c, plan_c2r, s);

    // Forcing: write real X-slab data directly into work_buf.
    // Mark subFormat = INPLACE before fill, then FFT_FORWARD (macro also forces d2z).
    // No dummy FFT needed -- we just assert the layout via force_d2z_format.
    force_d2z_format(s.work1_buf);
    force_d2z_format(s.work2_buf);
    force_d2z_format(s.work3_buf);
    kernel_fill_forcing<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.work1_buf), (double*)gpu_ptr(s.work2_buf), (double*)gpu_ptr(s.work3_buf),
        s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_REAL(gpu_ptr(s.work1_buf), s.nr_local, "after fill_forcing work1 (real)");

    // FFT_FORWARD macro calls force_d2z_format internally before each exec.
    FFT_FORWARD(plan_r2c, s.work1_buf);
    FFT_FORWARD(plan_r2c, s.work2_buf);
    FFT_FORWARD(plan_r2c, s.work3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(gpu_ptr(s.work1_buf), nc, "after D2Z work1 (spec)");

    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work1_buf), s.f1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work2_buf), s.f2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.work3_buf), s.f3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.f3_c, nc, inv_N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(s.f1_c, nc, "after scale f1_c");

    kernel_add_to_rhs<<<gc, BLOCK>>>(
        s.rhs1_c, s.rhs2_c, s.rhs3_c,
        s.visc1_c, s.visc2_c, s.visc3_c,
        s.f1_c, s.f2_c, s.f3_c, nc);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(s.rhs1_c, nc, "after add_to_rhs rhs1_c");

    kernel_make_div_free<<<gc, BLOCK>>>(
        s.rhs1_c, s.rhs2_c, s.rhs3_c, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(s.rhs1_c, nc, "rhs EXIT rhs1_c (final)");
}

static void save_v_orig(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.V1_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.V2_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.V3_orig, s.nc_local);
}

// V_buf data is X-slab real (INPLACE) after compute_nonlinear.
// Overwrite with V_orig (Y-slab complex, spectral) and mark subFormat = SHUFFLED
// so the next FFT_INVERSE call correctly interprets the layout.
// No dummy FFT needed -- force_z2d_format is sufficient (confirmed by hack test).
static void restore_v_buf(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    force_z2d_format(s.V1_buf);
    force_z2d_format(s.V2_buf);
    force_z2d_format(s.V3_buf);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V1_orig, (GCplx*)gpu_ptr(s.V1_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V2_orig, (GCplx*)gpu_ptr(s.V2_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V3_orig, (GCplx*)gpu_ptr(s.V3_buf), s.nc_local);
}

static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    const long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);

    save_v_orig(s);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(s.V1_orig, nc, "after save_v_orig V1_orig");
    CHECK_CPLX(s.V2_orig, nc, "after save_v_orig V2_orig");
    CHECK_CPLX(s.V3_orig, nc, "after save_v_orig V3_orig");

    // --- STAGE 1 ---
    compute_rhs(plan_r2c, plan_c2r, s, t);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k1v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k1v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k1v3, nc);

    restore_v_buf(s);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(gpu_ptr(s.V1_buf), nc, "STAGE1 after restore_v_buf V1_buf");

    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k1v1, s.k1v2, s.k1v3, 0.5*TAU, nc);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CPLX(gpu_ptr(s.V1_buf), nc, "STAGE1 after rk4_axpy V1_buf=Vo+0.5*tau*k1");

    // --- STAGE 2 --- (no NAN_DIAG checks here -- if Stage 1 was clean, this part is structurally identical)
    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k2v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k2v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k2v3, nc);
    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k2v1, s.k2v2, s.k2v3, 0.5*TAU, nc);

    // --- STAGE 3 ---
    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k3v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k3v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k3v3, nc);
    restore_v_buf(s);
    kernel_rk4_axpy<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        s.V1_orig, s.V2_orig, s.V3_orig,
        s.k3v1, s.k3v2, s.k3v3, TAU, nc);

    // --- STAGE 4 ---
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
    // kernel_rk4_update wrote Y-slab complex data into V_buf.
    // subFormat is INPLACE_SHUFFLED (set by restore_v_buf above). Consistent.
}

static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
                                 State& s, double t, double& L2_err, double& max_div) {
    const double inv_N = 1.0/(double)(NX*NY*NZ);
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

    kernel_error_sq<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        s.scratch, s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    double local_sq = thrust::reduce(thrust::device, sp, sp + nr, 0.0);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    L2_err = sqrt(global_sq * DX * DY * DZ);

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
// LAYOUT PROBE
// Inject V1 = cos(KX0 * x) into the real buffer, forward-FFT, then scan the
// complex Y-slab buffer to find which linear indices hold the non-zero modes.
// This directly reveals the actual INPLACE_SHUFFLED storage order at runtime.
// =============================================================================
__global__ void kernel_probe_fill_cosX(double* V, int nx_local, int x_offset, int KX0) {
    long long n_padded = (long long)nx_local * d_NY * (2 * d_NZC);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_padded) return;
    int k_padded = (int)(idx % (2 * d_NZC));
    int j        = (int)((idx / (2 * d_NZC)) % d_NY);
    int lx       = (int)(idx / ((long long)d_NY * (2 * d_NZC)));
    int gi = x_offset + lx;
    (void)j;
    if (k_padded >= d_NZ) { V[idx] = 0.0; return; }
    V[idx] = cos(2.0 * M_PI * (double)KX0 * (double)gi / (double)d_NX);
}

static void probe_layout(cufftHandle plan_r2c, State& s, int KX0, const char* tag) {
    const long long n_padded = (long long)s.nx_local * NY * (2 * NZC);
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    int gp = (int)((n_padded   + BLOCK - 1) / BLOCK);
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);

    CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));
    kernel_probe_fill_cosX<<<gp, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), s.nx_local, s.x_offset, KX0);
    CUDA_CHECK(cudaDeviceSynchronize());

    FFT_FORWARD(plan_r2c, s.V1_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_scale_cplx<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), s.nc_local, inv_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GCplx> host(s.nc_local);
    CUDA_CHECK(cudaMemcpy(host.data(), gpu_ptr(s.V1_buf),
                          s.nc_local * sizeof(GCplx), cudaMemcpyDeviceToHost));

    if (s.rank == 0) {
        printf("\n=== %s: V = cos(%d * x) ===\n", tag, KX0);
        printf("Expected non-zero modes: (kx,ky,kz)=(%d,0,0) and (%d,0,0), amplitude~0.5\n",
               KX0, NX - KX0);
        printf("NX=%d NY=%d NZC=%d ny_local(rank0)=%d\n", NX, NY, NZC, s.ny_local);
        // Predicted idx for both candidate layouts (ky=0 => local_y=0 on the rank holding y_offset=0)
        // Layout A: buf[gx][local_y][kz]
        long long predA_pos = (long long)KX0        * s.ny_local * NZC;
        long long predA_neg = (long long)(NX - KX0) * s.ny_local * NZC;
        // Layout B: buf[local_y][gx][kz]
        long long predB_pos = (long long)KX0        * NZC;
        long long predB_neg = (long long)(NX - KX0) * NZC;
        printf("Layout-A predicted idx (rank with ky=0): %lld and %lld\n", predA_pos, predA_neg);
        printf("Layout-B predicted idx (rank with ky=0): %lld and %lld\n", predB_pos, predB_neg);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < s.nprocs; ++r) {
        if (r == s.rank) {
            printf("--- rank %d  y_offset=%d  ny_local=%d ---\n",
                   s.rank, s.y_offset, s.ny_local);
            int found = 0;
            for (long long idx = 0; idx < s.nc_local; ++idx) {
                double mag = sqrt(host[idx].x * host[idx].x +
                                  host[idx].y * host[idx].y);
                if (mag > 0.05) {
                    // Layout A: [gx][local_y][kz]
                    long long A_gx = idx / ((long long)s.ny_local * NZC);
                    long long A_ly = (idx / NZC) % s.ny_local;
                    long long A_kz = idx % NZC;
                    // Layout B: [local_y][gx][kz]
                    long long B_ly = idx / ((long long)NX * NZC);
                    long long B_gx = (idx / NZC) % NX;
                    long long B_kz = idx % NZC;
                    printf("  idx=%lld  V=(%+.6f, %+.6f)  |V|=%.6f\n"
                           "    Layout-A: gx=%lld  local_y=%lld  kz=%lld  -> gy=%lld\n"
                           "    Layout-B: local_y=%lld  gx=%lld  kz=%lld  -> gy=%lld\n",
                           idx,
                           host[idx].x, host[idx].y, mag,
                           A_gx, A_ly, A_kz, (long long)s.y_offset + A_ly,
                           B_ly, B_gx, B_kz, (long long)s.y_offset + B_ly);
                    ++found;
                    if (found >= 8) { printf("  ... (truncated)\n"); break; }
                }
            }
            if (found == 0)
                printf("  (no entries with |V| > 0.05 on this rank)\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));
    CUDA_CHECK(cudaDeviceSynchronize());
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
            fprintf(stderr, "Usage: mpirun -np <NP> %s NX NY NZ dt NSTEPS\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]);
    TAU = atof(argv[4]); NT_RUN = atoi(argv[5]);
    NZC = NZ/2 + 1;
    LX = LY = LZ = 2.0*M_PI; DX = LX/NX; DY = LY/NY; DZ = LZ/NZ;

    // =========================================================================
    // ★ CRITICAL FIX: bind to correct GPU BEFORE any cudaMemcpyToSymbol.
    // cudaMemcpyToSymbol writes to the *current* device's constant memory.
    // Without this, all ranks default to GPU 0 when writing d_NX/d_NY/d_NZC,
    // so ranks 1..P-1 have d_NX=d_NY=d_NZC=0 on their own GPUs.  Every
    // kernel then computes nc_local = 0 and returns immediately -- silent
    // no-ops.  Only rank 0's kernels run, cuFFTMp transposes still execute,
    // spectral data is completely wrong, giving L2 ~ 1e27.
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
    CUDA_CHECK(cudaSetDevice(s.gpu));          // ★ must be before cudaMemcpyToSymbol

    // Now current device = this rank's actual GPU; constants land in the right place.
    cudaMemcpyToSymbol(d_NX,  &NX,  sizeof(int));
    cudaMemcpyToSymbol(d_NY,  &NY,  sizeof(int));
    cudaMemcpyToSymbol(d_NZ,  &NZ,  sizeof(int));
    cudaMemcpyToSymbol(d_NZC, &NZC, sizeof(int));
    cudaMemcpyToSymbol(d_DX,  &DX,  sizeof(double));
    cudaMemcpyToSymbol(d_DY,  &DY,  sizeof(double));
    cudaMemcpyToSymbol(d_DZ,  &DZ,  sizeof(double));
    if (s.rank == 0)
        printf("Grid: %d x %d x %d, dt=%.2e, steps=%d\n", NX, NY, NZ, TAU, NT_RUN);

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

    // ★ SetSubformatDefault MUST be called BEFORE MakePlan3d.
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_r2c,
        CUFFT_XT_FORMAT_INPLACE,
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_c2r,
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
        CUFFT_XT_FORMAT_INPLACE));

    size_t ws_r2c = 0, ws_c2r = 0;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    alloc_state(s, plan_r2c);

#if RUN_FFT_TEST
    {
        double fft_l2 = fft_roundtrip_test(plan_r2c, plan_c2r, s);
        if (s.rank == 0) {
            cout << "FFT round-trip L2 error: " << scientific << setprecision(4)
                 << fft_l2 << "  (should be < 1e-10)\n" << flush;
        }

        // =====================================================================
        // subFormat HACK diagnostic test
        // =====================================================================
        // Goal: verify empirically whether manually setting desc->subFormat
        // (force_z2d_format) has any effect on cuFFTMp's transpose logic.
        //
        // Protocol:
        //   1. Forward-FFT sin(x)cos(2y)sin(3z) into V1_buf -> spectral data.
        //   2. Save spectral data to a temporary buffer.
        //   3. FFT_INVERSE V1_buf -> back to INPLACE real (known good state).
        //   4. Zero V1_buf memory.
        //   5. Write the saved spectral data DIRECTLY into V1_buf while its
        //      cuFFTMp state is INPLACE (from step 3).
        //   6. Call force_z2d_format (THE HACK) to claim it is INPLACE_SHUFFLED.
        //   7. Run FFT_INVERSE (raw cufftXtExecDescriptor, bypassing our macro).
        //   8. Measure L2 error vs original function.
        //
        // Expected result if hack WORKS:  L2 < 1e-10
        // Expected result if hack FAILS:  L2 >> 0 or NaN   <- confirms root cause
        // =====================================================================
        {
            const long long n_padded = (long long)s.nx_local * NY * (2*NZC);
            const double inv_N = 1.0 / (double)(NX*NY*NZ);
            int gp = (int)((n_padded        + BLOCK - 1) / BLOCK);
            int gr = (int)((s.nr_local      + BLOCK - 1) / BLOCK);
            int gc = (int)((s.nc_local      + BLOCK - 1) / BLOCK);

            // Step 1: fill real pattern and forward-FFT.
            kernel_fill_test_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf),
                                                  s.nx_local, s.x_offset);
            CUDA_CHECK(cudaDeviceSynchronize());
            FFT_FORWARD(plan_r2c, s.V1_buf);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 2: save spectral data.
            GCplx* spec_save = nullptr;
            CUDA_CHECK(cudaMalloc(&spec_save, s.nc_local * sizeof(GCplx)));
            CUDA_CHECK(cudaMemcpy(spec_save, gpu_ptr(s.V1_buf),
                                  s.nc_local * sizeof(GCplx),
                                  cudaMemcpyDeviceToDevice));

            // Step 3: inverse-FFT back to INPLACE real (natural cuFFTMp state).
            FFT_INVERSE(plan_c2r, s.V1_buf);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 4: zero V1_buf so nothing leaks through.
            CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));

            // Step 5: write spectral data directly into V1_buf.
            // V1_buf internal cuFFTMp state is INPLACE (from step 3).
            kernel_copy_cplx<<<gc, BLOCK>>>(spec_save, (GCplx*)gpu_ptr(s.V1_buf),
                                             s.nc_local);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 6: THE HACK -- claim V1_buf is INPLACE_SHUFFLED.
            force_z2d_format(s.V1_buf);

            // Step 7: inverse-FFT.  Use raw cufftXtExecDescriptor (NOT the
            // FFT_INVERSE macro) so we test the raw hack without any masking.
            CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, s.V1_buf, s.V1_buf,
                                               CUFFT_INVERSE));
            CUDA_CHECK(cudaDeviceSynchronize());
            kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), n_padded, inv_N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 8: measure L2 error.
            kernel_test_error<<<gr, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.scratch,
                                              s.nx_local, s.x_offset);
            CUDA_CHECK(cudaDeviceSynchronize());
            thrust::device_ptr<double> sp(s.scratch);
            double local_sq  = thrust::reduce(thrust::device, sp, sp + s.nr_local, 0.0);
            double global_sq = 0.0;
            MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double l2_hack = sqrt(global_sq * DX * DY * DZ);

            if (s.rank == 0) {
                cout << "subFormat hack test L2 error: " << scientific << setprecision(4)
                     << l2_hack
                     << "  (< 1e-10 = hack works; >> 0 or NaN = hack FAILS -> root cause confirmed)\n"
                     << flush;
            }

            // Cleanup: free temp buffer, zero V1_buf.
            cudaFree(spec_save);
            CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));
            // Reset V1_buf cuFFTMp state to INPLACE_SHUFFLED so the diagnostic
            // shows whether the natural-state-flow approach avoids the issue.
            // (Force-write an FFT_FORWARD on the zeroed buffer to drive state.)
            FFT_FORWARD(plan_r2c, s.V1_buf);
            FFT_INVERSE(plan_c2r, s.V1_buf);  // back to INPLACE
            CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // =====================================================================

        if (fft_l2 > 1e-6) {
            if (s.rank == 0)
                cerr << "FFT round-trip FAILED -- the kernel indexing assumption for "
                        "INPLACE_SHUFFLED is wrong. Aborting before simulation.\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
#endif

    // Warm up all FFT buffers: one zero-data FORWARD+INVERSE cycle per buffer.
    // Forces cuFFTMp to initialize per-buffer transpose metadata (NVSHMEM heap,
    // peer mappings, etc.) before any simulation FFT call that relies on a
    // hacked subFormat.  Without this, virgin rot/work buffers produce silent
    // garbage on the first hacked FFT_INVERSE.
    warmup_buffers(plan_r2c, plan_c2r, s);

    if (s.rank == 0) cout << "Setting initial conditions...\n" << flush;
    {
        int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);
        // V_buf was alloc_and_zero'd as INPLACE.  Fill real X-slab values
        // and FFT_FORWARD -- naturally consistent with INPLACE state.
        // No force_* needed.
        kernel_fill_velocity<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
            s.nx_local, s.x_offset, 0.0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    FFT_FORWARD(plan_r2c, s.V1_buf);
    FFT_FORWARD(plan_r2c, s.V2_buf);
    FFT_FORWARD(plan_r2c, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double inv_N = 1.0/(double)(NX*NY*NZ);
    {
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.nc_local, inv_N);
        // kernel_make_div_free<<<gc, BLOCK>>>(
        //     (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        //     s.ny_local, s.y_offset);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // V_buf is now Y-slab complex (spectral).
    // cuFFTMp internal state is INPLACE_SHUFFLED after FFT_FORWARD above.
    // No force_* needed.

    double t_wall = 0.0;
    for (int it = 0; it < NT_RUN; ++it) {
        double t_cur = it * TAU;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        rk4_step(plan_r2c, plan_c2r, s, t_cur);
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        double dt_step = MPI_Wtime() - t0, dt_max = 0.0;
        MPI_Allreduce(&dt_step, &dt_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        t_wall += dt_max;
    }
    {
        double t_final = NT_RUN * TAU;
        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, s, t_final, L2_err, max_div);
        if (s.rank == 0) {
            cout << "  L2 error   (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << L2_err << "\n";
            cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << max_div << "\n";
        }
    }

    if (s.rank == 0) {
        cout << "============================================================\n";
        cout << "  Total steps:     " << NT_RUN << "\n";
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall << " s\n";
        cout << "  Avg per step:    " << t_wall / NT_RUN << " s\n";
        cout << "============================================================\n" << flush;
    }

    free_state(s);
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
