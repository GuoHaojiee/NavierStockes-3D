// NavierStokes_periodic_cufftxt.cu
// Multi-GPU Navier-Stokes solver using cuFFTXt (single-node, multiple GPUs)
//
// Physics: periodic boundary conditions, pseudo-spectral method, RK4 time integration,
//          divergence-free projection. Identical equations to the single-GPU cufft version.
//
// Key cuFFTXt constraints addressed:
//  1. In-place only: cufftXtExecDescriptor* requires input==output descriptor (same pointer).
//  2. subFormat hack: writing complex data directly into a buffer requires manually setting
//     desc->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED (=3) before Z2D, because cuFFTXt
//     only accepts Z2D when the descriptor is in "shuffled" format (= output of a D2Z).
//  3. Synchronization: cuFFTXt multi-GPU uses internal CUDA streams per GPU (NOT stream 0).
//     User kernels (stream 0) and cuFFT ops race unless explicit cudaDeviceSynchronize() is
//     called. See SYNC [A-I] comments for all 9 required sync points.
//
// *** INDEX MAPPING ASSUMPTION (MUST VERIFY ON CLUSTER) ***
//  After D2Z, spectral data is assumed to be distributed as X-slabs: GPU g owns global X
//  indices [g*nx_local, (g+1)*nx_local). This is the most natural layout for a D2Z slab
//  decomposition, but it is NOT officially documented by NVIDIA for cuFFTXt multi-GPU.
//  If the actual "INPLACE_SHUFFLED" layout differs (e.g., slab along Y, or permuted modes),
//  all spectral kernels (compute_rot, viscous, make_div_free, div_abs) will compute wrong
//  wavenumbers and produce incorrect results. Verify by printing gpu_ptr(V_buf, g) values
//  against expected modes at a known input (e.g., a single sine wave).
//
// Memory layout (in-place padded, per GPU):
//  - As double* (real): stride 2*NZC in last dimension; real[lx][j][k] at lx*NY*2*NZC+j*2*NZC+k
//  - As GCplx* (spectral): stride NZC; cplx[lx][j][kc] at lx*NY*NZC + j*NZC + kc
//  Both accesses address the same physical memory (sizeof(GCplx) == 2*sizeof(double))

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <omp.h>

using namespace std;

typedef cufftDoubleReal    GReal;
typedef cufftDoubleComplex GCplx;

// ============================================================
// Grid parameters
// ============================================================
constexpr int    NX = 128, NY = 128, NZ = 128;
constexpr int    NZC      = NZ / 2 + 1;          // 65
constexpr long long NR    = (long long)NX * NY * NZ;
constexpr long long NC_TOT = (long long)NX * NY * NZC;
constexpr double LX = 2.0 * M_PI, LY = 2.0 * M_PI, LZ = 2.0 * M_PI;
constexpr double DX = LX / NX, DY = LY / NY, DZ = LZ / NZ;
constexpr int    NT_TOTAL = 20000;
constexpr int    NT_RUN   = 10;
constexpr double TAU      = 1.0 / NT_TOTAL;      // dt = 5e-5

constexpr int    MAX_GPUS = 16;
constexpr int    BLOCK    = 256;

// ============================================================
// Error checking
// ============================================================
#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); \
    } \
} while (0)

#define CUFFT_CHECK(e) do { \
    cufftResult _e = (e); \
    if (_e != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: code %d\n", __FILE__, __LINE__, (int)_e); exit(1); \
    } \
} while (0)

// ============================================================
// subFormat hack helpers
//
// cudaLibXtDesc_t (from cudalibxt.h):
//   int version; cudaXtDesc *descriptor; libFormat library; int subFormat; void *libDescriptor;
//
// cudaXtDesc_t (from cudalibxt.h):
//   int version; int nGPUs; int GPUs[64]; void *data[64]; size_t size[64]; void *cudaXtState;
//
// CUFFT_XT_FORMAT_INPLACE           = 2  -> linear layout; valid input for D2Z (R2C)
// CUFFT_XT_FORMAT_INPLACE_SHUFFLED  = 3  -> shuffled layout; valid input for Z2D (C2R)
//
// Use force_d2z_format() before writing real data manually + executing D2Z.
// Use force_z2d_format() before writing complex data manually + executing Z2D.
// ============================================================
static inline void force_d2z_format(cudaLibXtDesc* desc) {
    desc->subFormat = CUFFT_XT_FORMAT_INPLACE;          // = 2
}
static inline void force_z2d_format(cudaLibXtDesc* desc) {
    desc->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED; // = 3
}

// Accessor helper
static inline void* gpu_ptr(cudaLibXtDesc* d, int g) { return d->descriptor->data[g]; }

// ============================================================
// Analytical solution (Taylor-Green variant)
// ============================================================
__host__ __device__ double func_V1(double x, double y, double z, double t) {
    return (t*t + 1.0) * exp(sin(3.0*x + 3.0*y)) * cos(6.0*z);
}
__host__ __device__ double func_V2(double x, double y, double z, double t) {
    return (t*t + 1.0) * exp(sin(3.0*x + 3.0*y)) * cos(6.0*z);
}
__host__ __device__ double func_V3(double x, double y, double z, double t) {
    return -(t*t + 1.0) * exp(sin(3.0*x + 3.0*y)) * cos(3.0*x + 3.0*y) * sin(6.0*z);
}
__host__ __device__ double func_dV1_dt(double x, double y, double z, double t) {
    return 2.0*t * exp(sin(3.0*x + 3.0*y)) * cos(6.0*z);
}
__host__ __device__ double func_dV2_dt(double x, double y, double z, double t) {
    return 2.0*t * exp(sin(3.0*x + 3.0*y)) * cos(6.0*z);
}
__host__ __device__ double func_dV3_dt(double x, double y, double z, double t) {
    return -2.0*t * exp(sin(3.0*x + 3.0*y)) * cos(3.0*x + 3.0*y) * sin(6.0*z);
}
__host__ __device__ double func_laplace_V1(double x, double y, double z, double t) {
    double s = sin(3.0*x + 3.0*y), c = cos(3.0*x + 3.0*y);
    double coef = (t*t + 1.0) * exp(s);
    return 2.0 * 9.0 * coef * (c*c - s) * cos(6.0*z) - 36.0 * coef * cos(6.0*z);
}
__host__ __device__ double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x, y, z, t);
}
__host__ __device__ double func_laplace_V3(double x, double y, double z, double t) {
    double s = sin(3.0*x + 3.0*y), c = cos(3.0*x + 3.0*y);
    double coef = (t*t + 1.0) * exp(s) * c;
    return 2.0 * (-9.0) * coef * ((c*c - s) - (2.0*s + 1.0)) * sin(6.0*z)
           + 36.0 * coef * sin(6.0*z);
}
__host__ __device__ double func_rot1(double x, double y, double z, double t) {
    double s = sin(3.0*x + 3.0*y), c = cos(3.0*x + 3.0*y);
    double dv3_dy = -(t*t + 1.0) * 3.0 * exp(s) * (c*c - s) * sin(6.0*z);
    double dv2_dz = -(t*t + 1.0) * 6.0 * exp(s) * sin(6.0*z);
    return dv3_dy - dv2_dz;
}
__host__ __device__ double func_rot2(double x, double y, double z, double t) {
    double s = sin(3.0*x + 3.0*y), c = cos(3.0*x + 3.0*y);
    double dv1_dz = -(t*t + 1.0) * 6.0 * exp(s) * sin(6.0*z);
    double dv3_dx = -(t*t + 1.0) * 3.0 * exp(s) * (c*c - s) * sin(6.0*z);
    return dv1_dz - dv3_dx;
}
__host__ __device__ double func_rot3(double, double, double, double) { return 0.0; }
__host__ __device__ double func_vcr1(double x, double y, double z, double t) {
    return func_V2(x,y,z,t)*func_rot3(x,y,z,t) - func_V3(x,y,z,t)*func_rot2(x,y,z,t);
}
__host__ __device__ double func_vcr2(double x, double y, double z, double t) {
    return func_V3(x,y,z,t)*func_rot1(x,y,z,t) - func_V1(x,y,z,t)*func_rot3(x,y,z,t);
}
__host__ __device__ double func_vcr3(double x, double y, double z, double t) {
    return func_V1(x,y,z,t)*func_rot2(x,y,z,t) - func_V2(x,y,z,t)*func_rot1(x,y,z,t);
}
__host__ __device__ double func_grad_p1(double x, double y, double z, double t) {
    return -(t*t + 1.0) * sin(x) * cos(y) * cos(z);
}
__host__ __device__ double func_grad_p2(double x, double y, double z, double t) {
    return -(t*t + 1.0) * cos(x) * sin(y) * cos(z);
}
__host__ __device__ double func_grad_p3(double x, double y, double z, double t) {
    return -(t*t + 1.0) * cos(x) * cos(y) * sin(z);
}
__host__ __device__ double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x,y,z,t) - func_laplace_V1(x,y,z,t) - func_vcr1(x,y,z,t) + func_grad_p1(x,y,z,t);
}
__host__ __device__ double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t) - func_laplace_V2(x,y,z,t) - func_vcr2(x,y,z,t) + func_grad_p2(x,y,z,t);
}
__host__ __device__ double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t) - func_laplace_V3(x,y,z,t) - func_vcr3(x,y,z,t) + func_grad_p3(x,y,z,t);
}

// ============================================================
// CUDA Kernels — all accept (nx_local, x_offset) for multi-GPU
//
// Real-space buffers use in-place padded layout: stride 2*NZC in last dim.
// Spectral-space buffers use complex layout: stride NZC in last dim.
// ============================================================

// Fill in-place padded buffer with analytical velocity (real, double* with stride 2*NZC)
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

// Fill in-place padded buffer with forcing (real, double* with stride 2*NZC)
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

// Scale complex buffer: A *= scale  (spectral layout, GCplx* with stride NZC)
__global__ void kernel_scale_cplx(GCplx* A, long long nc_local, double scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    A[idx].x *= scale;
    A[idx].y *= scale;
}

// Spectral vorticity: rot_c = ik × V_c  (both in spectral layout)
// Global kx computed from local lx + x_offset.
__global__ void kernel_compute_rot(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                    GCplx* rot1, GCplx* rot2, GCplx* rot3,
                                    int nx_local, int x_offset) {
    long long nc_local = (long long)nx_local * NY * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    int kz = (int)(idx % NZC);
    int j  = (int)((idx / NZC) % NY);
    int lx = (int)(idx / ((long long)NY * NZC));
    int gi = x_offset + lx;
    double kx = (gi <= NX/2) ? (double)gi : (double)(gi - NX);
    double ky = (j  <= NY/2) ? (double)j  : (double)(j  - NY);
    double kzd = (double)kz;

    rot1[idx].x = -(ky * V3[idx].y - kzd * V2[idx].y);
    rot1[idx].y =   ky * V3[idx].x - kzd * V2[idx].x;
    rot2[idx].x = -(kzd * V1[idx].y - kx * V3[idx].y);
    rot2[idx].y =   kzd * V1[idx].x - kx * V3[idx].x;
    rot3[idx].x = -(kx * V2[idx].y - ky * V1[idx].y);
    rot3[idx].y =   kx * V2[idx].x - ky * V1[idx].x;
}

// Viscous term: visc_c = -k² V_c  (spectral layout)
__global__ void kernel_compute_viscous(const GCplx* V, GCplx* visc,
                                        int nx_local, int x_offset) {
    long long nc_local = (long long)nx_local * NY * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    int kz = (int)(idx % NZC);
    int j  = (int)((idx / NZC) % NY);
    int lx = (int)(idx / ((long long)NY * NZC));
    int gi = x_offset + lx;
    double kx = (gi <= NX/2) ? (double)gi : (double)(gi - NX);
    double ky = (j  <= NY/2) ? (double)j  : (double)(j  - NY);
    double kzd = (double)kz;
    double k2 = kx*kx + ky*ky + kzd*kzd;
    visc[idx].x = -k2 * V[idx].x;
    visc[idx].y = -k2 * V[idx].y;
}

// Projection: make spectral rhs divergence-free (in-place, spectral layout)
__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
                                      int nx_local, int x_offset) {
    long long nc_local = (long long)nx_local * NY * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    int kz = (int)(idx % NZC);
    int j  = (int)((idx / NZC) % NY);
    int lx = (int)(idx / ((long long)NY * NZC));
    int gi = x_offset + lx;
    double kx = (gi <= NX/2) ? (double)gi : (double)(gi - NX);
    double ky = (j  <= NY/2) ? (double)j  : (double)(j  - NY);
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

// Real-space cross product: V_r × rot_r → rot_r  (padded layout, double* stride 2*NZC)
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

// Add terms to rhs: rhs += visc + f  (spectral layout)
__global__ void kernel_add_to_rhs(GCplx* rhs1, GCplx* rhs2, GCplx* rhs3,
                                   const GCplx* visc1, const GCplx* visc2, const GCplx* visc3,
                                   const GCplx* f1, const GCplx* f2, const GCplx* f3,
                                   long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    rhs1[idx].x += visc1[idx].x + f1[idx].x;
    rhs1[idx].y += visc1[idx].y + f1[idx].y;
    rhs2[idx].x += visc2[idx].x + f2[idx].x;
    rhs2[idx].y += visc2[idx].y + f2[idx].y;
    rhs3[idx].x += visc3[idx].x + f3[idx].x;
    rhs3[idx].y += visc3[idx].y + f3[idx].y;
}

// RK4 intermediate: V = orig + alpha * k  (spectral layout)
__global__ void kernel_rk4_axpy(GCplx* V1, GCplx* V2, GCplx* V3,
                                  const GCplx* o1, const GCplx* o2, const GCplx* o3,
                                  const GCplx* k1, const GCplx* k2, const GCplx* k3,
                                  double alpha, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    V1[idx].x = o1[idx].x + alpha * k1[idx].x;
    V1[idx].y = o1[idx].y + alpha * k1[idx].y;
    V2[idx].x = o2[idx].x + alpha * k2[idx].x;
    V2[idx].y = o2[idx].y + alpha * k2[idx].y;
    V3[idx].x = o3[idx].x + alpha * k3[idx].x;
    V3[idx].y = o3[idx].y + alpha * k3[idx].y;
}

// RK4 final update (spectral layout)
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
    V1[idx].x = o1[idx].x + dtd6*(k1v1[idx].x + 2.0*k2v1[idx].x + 2.0*k3v1[idx].x + k4v1[idx].x);
    V1[idx].y = o1[idx].y + dtd6*(k1v1[idx].y + 2.0*k2v1[idx].y + 2.0*k3v1[idx].y + k4v1[idx].y);
    V2[idx].x = o2[idx].x + dtd6*(k1v2[idx].x + 2.0*k2v2[idx].x + 2.0*k3v2[idx].x + k4v2[idx].x);
    V2[idx].y = o2[idx].y + dtd6*(k1v2[idx].y + 2.0*k2v2[idx].y + 2.0*k3v2[idx].y + k4v2[idx].y);
    V3[idx].x = o3[idx].x + dtd6*(k1v3[idx].x + 2.0*k2v3[idx].x + 2.0*k3v3[idx].x + k4v3[idx].x);
    V3[idx].y = o3[idx].y + dtd6*(k1v3[idx].y + 2.0*k2v3[idx].y + 2.0*k3v3[idx].y + k4v3[idx].y);
}

// Squared pointwise error |V_r - V_exact|² → scratch  (padded real read, linear scratch write)
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

// |ik·V_c| per spectral mode → scratch  (spectral layout → linear scratch)
__global__ void kernel_div_abs(const GCplx* V1, const GCplx* V2, const GCplx* V3,
                                 double* scratch, int nx_local, int x_offset) {
    long long nc_local = (long long)nx_local * NY * NZC;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    int kz = (int)(idx % NZC);
    int j  = (int)((idx / NZC) % NY);
    int lx = (int)(idx / ((long long)NY * NZC));
    int gi = x_offset + lx;
    double kx = (gi <= NX/2) ? (double)gi : (double)(gi - NX);
    double ky = (j  <= NY/2) ? (double)j  : (double)(j  - NY);
    double kzd = (double)kz;
    double dr = -(kx*V1[idx].y + ky*V2[idx].y + kzd*V3[idx].y);
    double di =   kx*V1[idx].x + ky*V2[idx].x + kzd*V3[idx].x;
    scratch[idx] = sqrt(dr*dr + di*di);
}

// Copy spectral content from in-place buf (GCplx*) → per-GPU regular array
__global__ void kernel_copy_cplx(const GCplx* src, GCplx* dst, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    dst[idx] = src[idx];
}

// Restore in-place buf from backup: dst_buf (GCplx*) = src array
__global__ void kernel_restore_cplx(const GCplx* src, GCplx* dst, long long nc_local) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    dst[idx] = src[idx];
}

// ============================================================
// Multi-GPU state
// ============================================================
struct MGPUState {
    int nGPUs;
    int gpu_ids[MAX_GPUS];
    int nx_local;          // NX / nGPUs  (X slabs per GPU)
    long long nc_local;    // nx_local * NY * NZC  (complex elements per GPU)
    long long nr_local;    // nx_local * NY * NZ   (real elements per GPU, unpadded)

    // In-place cuFFTXt buffers (velocity, vorticity, work/forcing)
    cudaLibXtDesc* V1_buf;
    cudaLibXtDesc* V2_buf;
    cudaLibXtDesc* V3_buf;
    cudaLibXtDesc* rot1_buf;
    cudaLibXtDesc* rot2_buf;
    cudaLibXtDesc* rot3_buf;
    cudaLibXtDesc* work1_buf;
    cudaLibXtDesc* work2_buf;
    cudaLibXtDesc* work3_buf;

    // Per-GPU spectral-only arrays (regular cudaMalloc, size nc_local)
    GCplx* visc1_c[MAX_GPUS], *visc2_c[MAX_GPUS], *visc3_c[MAX_GPUS];
    GCplx* f1_c[MAX_GPUS],    *f2_c[MAX_GPUS],    *f3_c[MAX_GPUS];
    GCplx* rhs1_c[MAX_GPUS],  *rhs2_c[MAX_GPUS],  *rhs3_c[MAX_GPUS];
    GCplx* k1v1[MAX_GPUS], *k2v1[MAX_GPUS], *k3v1[MAX_GPUS], *k4v1[MAX_GPUS];
    GCplx* k1v2[MAX_GPUS], *k2v2[MAX_GPUS], *k3v2[MAX_GPUS], *k4v2[MAX_GPUS];
    GCplx* k1v3[MAX_GPUS], *k2v3[MAX_GPUS], *k3v3[MAX_GPUS], *k4v3[MAX_GPUS];
    GCplx* V1_orig[MAX_GPUS], *V2_orig[MAX_GPUS], *V3_orig[MAX_GPUS];

    // Per-GPU scratch for reductions (size max(nc_local, nr_local))
    double* d_scratch[MAX_GPUS];
};

static void alloc_per_gpu_cplx(MGPUState& s, GCplx** arr) {
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        CUDA_CHECK(cudaMalloc(&arr[g], s.nc_local * sizeof(GCplx)));
    }
}

static void alloc_mgpu(MGPUState& s, cufftHandle plan_r2c) {
    // In-place cuFFTXt buffers
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V1_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V2_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.V3_buf,    CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot1_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot2_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.rot3_buf,  CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work1_buf, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work2_buf, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &s.work3_buf, CUFFT_XT_FORMAT_INPLACE));

    // Per-GPU spectral-only arrays
    alloc_per_gpu_cplx(s, s.visc1_c); alloc_per_gpu_cplx(s, s.visc2_c); alloc_per_gpu_cplx(s, s.visc3_c);
    alloc_per_gpu_cplx(s, s.f1_c);    alloc_per_gpu_cplx(s, s.f2_c);    alloc_per_gpu_cplx(s, s.f3_c);
    alloc_per_gpu_cplx(s, s.rhs1_c);  alloc_per_gpu_cplx(s, s.rhs2_c);  alloc_per_gpu_cplx(s, s.rhs3_c);
    alloc_per_gpu_cplx(s, s.k1v1);    alloc_per_gpu_cplx(s, s.k2v1);
    alloc_per_gpu_cplx(s, s.k3v1);    alloc_per_gpu_cplx(s, s.k4v1);
    alloc_per_gpu_cplx(s, s.k1v2);    alloc_per_gpu_cplx(s, s.k2v2);
    alloc_per_gpu_cplx(s, s.k3v2);    alloc_per_gpu_cplx(s, s.k4v2);
    alloc_per_gpu_cplx(s, s.k1v3);    alloc_per_gpu_cplx(s, s.k2v3);
    alloc_per_gpu_cplx(s, s.k3v3);    alloc_per_gpu_cplx(s, s.k4v3);
    alloc_per_gpu_cplx(s, s.V1_orig); alloc_per_gpu_cplx(s, s.V2_orig); alloc_per_gpu_cplx(s, s.V3_orig);

    // Per-GPU scratch (large enough for both real and complex reductions)
    long long scratch_size = std::max(s.nc_local, s.nr_local);
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        CUDA_CHECK(cudaMalloc(&s.d_scratch[g], scratch_size * sizeof(double)));
    }
}

// Synchronize all GPUs — required between user kernels (NULL/stream 0) and cuFFTXt calls
// (internal streams). cuFFTXt multi-GPU uses per-GPU internal streams separate from stream 0,
// so without explicit sync, user kernel writes and cuFFT reads/writes on the same buffer race.
static void sync_all_gpus(const MGPUState& s) {
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// ============================================================
// Nonlinear term: V × rot(V) → rhs_c  (per-GPU, 9 FFTs equivalent)
//
// Sequence:
//  1. Spectral vorticity: rot_buf_c = ik × V_buf_c  (written directly, subFormat hack)
//  2. Z2D(V_buf) [destroys spectral V, gives padded real]
//  3. Z2D(rot_buf) [subFormat forced to 3, gives padded real vorticity]
//  4. Real cross product V_r × rot_r → rot_buf (real)
//  5. D2Z(rot_buf) → spectral cross product
//  6. Copy rot_buf complex → rhs_c, scale by 1/N
// ============================================================
static void compute_nonlinear(cufftHandle plan_r2c, cufftHandle plan_c2r,
                               MGPUState& s) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    const int nGPUs = s.nGPUs;
    const long long nc_local = s.nc_local;
    const long long nr_local = s.nr_local;
    const int nx_local = s.nx_local;

    // 1. Spectral vorticity written directly into rot_buf (using complex layout)
    //    Then subFormat hacked to 3 so cuFFTXt accepts them as Z2D input.
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * nx_local;
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_compute_rot<<<grid_c, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            (GCplx*)gpu_ptr(s.rot1_buf, g), (GCplx*)gpu_ptr(s.rot2_buf, g), (GCplx*)gpu_ptr(s.rot3_buf, g),
            nx_local, x_offset);
    }
    // subFormat hack: we wrote complex data directly → tell cuFFTXt the buffers are in shuffled format
    force_z2d_format(s.rot1_buf);
    force_z2d_format(s.rot2_buf);
    force_z2d_format(s.rot3_buf);

    // SYNC [A]: kernel_compute_rot (stream 0) reads V_buf and writes rot_buf.
    //   cuFFTXt Z2D below uses internal streams that do NOT automatically wait for stream 0.
    //   Without this sync: Z2D could start overwriting V_buf while rot kernel is still reading it,
    //   or Z2D on rot_buf could start before the rot kernel finishes writing rot_buf.
    sync_all_gpus(s);

    // 2. IFFT velocity V_buf (V_buf.subFormat should be 3 from previous D2Z)
    //    Gives padded real velocity in V_buf.
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V3_buf, s.V3_buf));

    // 3. IFFT vorticity rot_buf (subFormat forced to 3 above)
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.rot1_buf, s.rot1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.rot2_buf, s.rot2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.rot3_buf, s.rot3_buf));

    // SYNC [B]: cuFFTXt Z2D is asynchronous. kernel_cross_product below reads the Z2D results
    //   (real V and real rot). Without sync, it would read stale/partial data.
    sync_all_gpus(s);

    // 4. Real-space cross product V × rot → rot_buf  (padded layout)
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int grid_r = (int)((nr_local + BLOCK - 1) / BLOCK);
        kernel_cross_product<<<grid_r, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf, g), (double*)gpu_ptr(s.V2_buf, g), (double*)gpu_ptr(s.V3_buf, g),
            (double*)gpu_ptr(s.rot1_buf, g), (double*)gpu_ptr(s.rot2_buf, g), (double*)gpu_ptr(s.rot3_buf, g),
            nx_local);
    }
    // After Z2D, rot_buf.subFormat = 2 (INPLACE, linear/real) → valid for D2Z

    // SYNC [C]: kernel_cross_product (stream 0) writes rot_buf.
    //   D2Z below reads rot_buf on cuFFT internal streams. Must wait for kernel to finish.
    sync_all_gpus(s);

    // 5. FFT cross product result rot_buf → spectral
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.rot1_buf, s.rot1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.rot2_buf, s.rot2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.rot3_buf, s.rot3_buf));

    // SYNC [D]: D2Z is asynchronous. kernel_copy_cplx below reads rot_buf (spectral result).
    sync_all_gpus(s);

    // 6. Copy spectral cross product to rhs_c and scale
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.rot1_buf, g), s.rhs1_c[g], nc_local);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.rot2_buf, g), s.rhs2_c[g], nc_local);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.rot3_buf, g), s.rhs3_c[g], nc_local);
        kernel_scale_cplx<<<grid_c, BLOCK>>>(s.rhs1_c[g], nc_local, inv_N);
        kernel_scale_cplx<<<grid_c, BLOCK>>>(s.rhs2_c[g], nc_local, inv_N);
        kernel_scale_cplx<<<grid_c, BLOCK>>>(s.rhs3_c[g], nc_local, inv_N);
    }
}

// ============================================================
// Full RHS = nl + visc + f, projected to divergence-free
//
// V_buf must be in spectral (subFormat=3) on entry.
// V_buf is destroyed (consumed by Z2D in compute_nonlinear).
// Output: rhs{1,2,3}_c per GPU.
// ============================================================
static void compute_rhs(cufftHandle plan_r2c, cufftHandle plan_c2r,
                        MGPUState& s, double t) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    const int nGPUs = s.nGPUs;
    const long long nc_local = s.nc_local;
    const long long nr_local = s.nr_local;
    const int nx_local = s.nx_local;

    // 1. Viscous: visc_c = -k² V_buf_c  (BEFORE nonlinear which destroys V_buf)
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * nx_local;
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_compute_viscous<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf, g), s.visc1_c[g], nx_local, x_offset);
        kernel_compute_viscous<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf, g), s.visc2_c[g], nx_local, x_offset);
        kernel_compute_viscous<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf, g), s.visc3_c[g], nx_local, x_offset);
    }

    // SYNC [E]: kernel_compute_viscous (stream 0) reads V_buf.
    //   compute_nonlinear below will call Z2D on V_buf (cuFFT internal streams, writes V_buf).
    //   Race: viscous read vs Z2D write. Note: viscous and the subsequent kernel_compute_rot
    //   are both on stream 0 (ordered), so this one sync covers both before Z2D.
    sync_all_gpus(s);

    // 2. Nonlinear: V × rot(V) → rhs_c  (V_buf destroyed inside via Z2D)
    compute_nonlinear(plan_r2c, plan_c2r, s);

    // 3. Forcing: fill work_buf real, FFT, scale → f_c
    //    subFormat hack: work_buf may be 3 from previous D2Z, force to 2 for D2Z input
    force_d2z_format(s.work1_buf);
    force_d2z_format(s.work2_buf);
    force_d2z_format(s.work3_buf);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * nx_local;
        int grid_r = (int)((nr_local + BLOCK - 1) / BLOCK);
        kernel_fill_forcing<<<grid_r, BLOCK>>>(
            (double*)gpu_ptr(s.work1_buf, g), (double*)gpu_ptr(s.work2_buf, g),
            (double*)gpu_ptr(s.work3_buf, g), nx_local, x_offset, t);
    }
    // SYNC [F]: kernel_fill_forcing (stream 0) writes work_buf.
    //   D2Z below reads work_buf on cuFFT internal streams.
    sync_all_gpus(s);
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.work1_buf, s.work1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.work2_buf, s.work2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.work3_buf, s.work3_buf));
    // SYNC [G]: D2Z is asynchronous. kernel_copy_cplx reads work_buf (spectral result).
    sync_all_gpus(s);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.work1_buf, g), s.f1_c[g], nc_local);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.work2_buf, g), s.f2_c[g], nc_local);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.work3_buf, g), s.f3_c[g], nc_local);
        kernel_scale_cplx<<<grid_c, BLOCK>>>(s.f1_c[g], nc_local, inv_N);
        kernel_scale_cplx<<<grid_c, BLOCK>>>(s.f2_c[g], nc_local, inv_N);
        kernel_scale_cplx<<<grid_c, BLOCK>>>(s.f3_c[g], nc_local, inv_N);
    }

    // 4. rhs += visc + f
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_add_to_rhs<<<grid_c, BLOCK>>>(
            s.rhs1_c[g], s.rhs2_c[g], s.rhs3_c[g],
            s.visc1_c[g], s.visc2_c[g], s.visc3_c[g],
            s.f1_c[g], s.f2_c[g], s.f3_c[g], nc_local);
    }

    // 5. Project rhs to divergence-free space
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * nx_local;
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_make_div_free<<<grid_c, BLOCK>>>(
            s.rhs1_c[g], s.rhs2_c[g], s.rhs3_c[g], nx_local, x_offset);
    }
}

// Restore V_buf from V_orig per-GPU arrays and set subFormat=3 for Z2D
static void restore_v_buf(MGPUState& s) {
    const int nGPUs = s.nGPUs;
    const long long nc_local = s.nc_local;
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_restore_cplx<<<grid_c, BLOCK>>>(s.V1_orig[g], (GCplx*)gpu_ptr(s.V1_buf, g), nc_local);
        kernel_restore_cplx<<<grid_c, BLOCK>>>(s.V2_orig[g], (GCplx*)gpu_ptr(s.V2_buf, g), nc_local);
        kernel_restore_cplx<<<grid_c, BLOCK>>>(s.V3_orig[g], (GCplx*)gpu_ptr(s.V3_buf, g), nc_local);
    }
    force_z2d_format(s.V1_buf);
    force_z2d_format(s.V2_buf);
    force_z2d_format(s.V3_buf);
}

// Save V_buf (spectral) to V_orig per-GPU arrays
static void save_v_orig(MGPUState& s) {
    const int nGPUs = s.nGPUs;
    const long long nc_local = s.nc_local;
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int grid_c = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf, g), s.V1_orig[g], nc_local);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf, g), s.V2_orig[g], nc_local);
        kernel_copy_cplx<<<grid_c, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf, g), s.V3_orig[g], nc_local);
    }
}

// ============================================================
// RK4 time integration
// On entry: V_buf is in spectral format (subFormat=3).
// On exit:  V_buf is in spectral format (subFormat=3).
// ============================================================
static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r,
                      MGPUState& s, double t) {
    const int nGPUs = s.nGPUs;
    const long long nc_local = s.nc_local;

    // Save V^n
    save_v_orig(s);

    // k1 = RHS(V^n, t)
    compute_rhs(plan_r2c, plan_c2r, s, t);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c[g], s.k1v1[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c[g], s.k1v2[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c[g], s.k1v3[g], nc_local);
    }

    // V = V_orig + τ/2 * k1
    // restore_v_buf copies V_orig→V_buf on stream 0 and sets subFormat=3.
    // rk4_axpy then overwrites V_buf on stream 0 (same stream, ordered after restore).
    // subFormat stays 3 (host-side field unaffected by GPU kernel).
    restore_v_buf(s);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_rk4_axpy<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            s.V1_orig[g], s.V2_orig[g], s.V3_orig[g],
            s.k1v1[g], s.k1v2[g], s.k1v3[g], 0.5*TAU, nc_local);
    }
    // subFormat already 3 from restore_v_buf — no need to force again

    // k2 = RHS(V^n + τ/2*k1, t + τ/2)
    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c[g], s.k2v1[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c[g], s.k2v2[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c[g], s.k2v3[g], nc_local);
    }

    // V = V_orig + τ/2 * k2
    restore_v_buf(s);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_rk4_axpy<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            s.V1_orig[g], s.V2_orig[g], s.V3_orig[g],
            s.k2v1[g], s.k2v2[g], s.k2v3[g], 0.5*TAU, nc_local);
    }

    // k3 = RHS(V^n + τ/2*k2, t + τ/2)
    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c[g], s.k3v1[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c[g], s.k3v2[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c[g], s.k3v3[g], nc_local);
    }

    // V = V_orig + τ * k3
    restore_v_buf(s);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_rk4_axpy<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            s.V1_orig[g], s.V2_orig[g], s.V3_orig[g],
            s.k3v1[g], s.k3v2[g], s.k3v3[g], TAU, nc_local);
    }

    // k4 = RHS(V^n + τ*k3, t + τ)
    compute_rhs(plan_r2c, plan_c2r, s, t + TAU);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c[g], s.k4v1[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c[g], s.k4v2[g], nc_local);
        kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c[g], s.k4v3[g], nc_local);
    }

    // V^{n+1} = V_orig + τ/6*(k1 + 2k2 + 2k3 + k4)
    restore_v_buf(s);
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_rk4_update<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            s.V1_orig[g], s.V2_orig[g], s.V3_orig[g],
            s.k1v1[g], s.k2v1[g], s.k3v1[g], s.k4v1[g],
            s.k1v2[g], s.k2v2[g], s.k3v2[g], s.k4v2[g],
            s.k1v3[g], s.k2v3[g], s.k3v3[g], s.k4v3[g],
            TAU / 6.0, nc_local);
    }
    // V_buf is now spectral (complex layout), subFormat was set to 3 by restore_v_buf
    // Update was done in-place on the complex data; subFormat stays 3 ✓
}

// ============================================================
// Diagnostics: L2 error and max|div V|
// V_buf is spectral on entry (subFormat=3).
// V_buf is IFFT'd to real, error computed, then re-FFT'd to restore spectral state.
// ============================================================
static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
                                 MGPUState& s, double t,
                                 double& L2_err, double& max_div) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    const int nGPUs = s.nGPUs;
    const long long nc_local = s.nc_local;
    const long long nr_local = s.nr_local;
    const int nx_local = s.nx_local;

    // max|div V| in spectral space (before IFFT)
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * nx_local;
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_div_abs<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            s.d_scratch[g], nx_local, x_offset);
    }
    max_div = 0.0;
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<double> sp(s.d_scratch[g]);
        double local_max = thrust::reduce(thrust::device, sp, sp + nc_local, 0.0, thrust::maximum<double>());
        max_div = max(max_div, local_max);
    }

    // IFFT V_buf → real space for L2 error
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorZ2D(plan_c2r, s.V3_buf, s.V3_buf));

    // SYNC [H]: Z2D is asynchronous. kernel_error_sq reads V_buf (real, padded).
    sync_all_gpus(s);

    // Compute squared error per grid point → d_scratch (linear, nr_local elements)
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * nx_local;
        int gr = (int)((nr_local + BLOCK - 1) / BLOCK);
        kernel_error_sq<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf, g), (double*)gpu_ptr(s.V2_buf, g), (double*)gpu_ptr(s.V3_buf, g),
            s.d_scratch[g], nx_local, x_offset, t);
    }
    double err_sq = 0.0;
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::device_ptr<double> sp(s.d_scratch[g]);
        err_sq += thrust::reduce(thrust::device, sp, sp + nr_local, 0.0);
    }
    L2_err = sqrt(err_sq * DX * DY * DZ);

    // Re-FFT V_buf → restore spectral state  (subFormat becomes 3 after D2Z)
    // V_buf.subFormat = 2 after Z2D above → valid for D2Z
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V3_buf, s.V3_buf));
    // SYNC [I]: D2Z is asynchronous. kernel_scale_cplx reads/writes V_buf.
    sync_all_gpus(s);
    // Renormalize (D2Z is not normalized)
    #pragma omp parallel for num_threads(nGPUs)
    for (int g = 0; g < nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf, g), nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf, g), nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf, g), nc_local, inv_N);
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    // Query available GPUs
    int nGPUs_avail = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs_avail));
    if (nGPUs_avail < 1) { fprintf(stderr, "No CUDA devices found.\n"); return 1; }

    MGPUState s;
    s.nGPUs = nGPUs_avail;
    for (int g = 0; g < s.nGPUs; g++) s.gpu_ids[g] = g;

    if (NX % s.nGPUs != 0) {
        fprintf(stderr, "NX=%d must be divisible by nGPUs=%d\n", NX, s.nGPUs);
        return 1;
    }
    s.nx_local  = NX / s.nGPUs;
    s.nc_local  = (long long)s.nx_local * NY * NZC;
    s.nr_local  = (long long)s.nx_local * NY * NZ;

    cout << "============================================================" << endl;
    cout << "  Navier-Stokes Solver - cuFFTXt Multi-GPU Version" << endl;
    cout << "============================================================" << endl;
    cout << "GPUs: " << s.nGPUs << endl;
    for (int g = 0; g < s.nGPUs; g++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, s.gpu_ids[g]));
        cout << "  GPU " << g << ": " << prop.name << endl;
    }
    cout << "Grid: " << NX << " x " << NY << " x " << NZ
         << "  (nx_local=" << s.nx_local << " per GPU)" << endl;
    cout << "Steps: " << NT_RUN << " / " << NT_TOTAL << ", dt = " << TAU << endl;
    cout << "============================================================" << endl;

    // Create cuFFTXt plans
    cout << "Creating cuFFTXt plans..." << endl;
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    CUFFT_CHECK(cufftXtSetGPUs(plan_r2c, s.nGPUs, s.gpu_ids));
    CUFFT_CHECK(cufftXtSetGPUs(plan_c2r, s.nGPUs, s.gpu_ids));
    size_t workSizes[MAX_GPUS];
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, workSizes));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, workSizes));

    // Allocate all GPU arrays
    cout << "Allocating GPU memory..." << endl;
    alloc_mgpu(s, plan_r2c);

    // Initial conditions: fill V_buf real, then D2Z → spectral
    cout << "Setting initial conditions..." << endl;
    force_d2z_format(s.V1_buf);
    force_d2z_format(s.V2_buf);
    force_d2z_format(s.V3_buf);
    #pragma omp parallel for num_threads(s.nGPUs)
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * s.nx_local;
        int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);
        kernel_fill_velocity<<<gr, BLOCK>>>(
            (double*)gpu_ptr(s.V1_buf, g), (double*)gpu_ptr(s.V2_buf, g), (double*)gpu_ptr(s.V3_buf, g),
            s.nx_local, x_offset, 0.0);
    }
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V1_buf, s.V1_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V2_buf, s.V2_buf));
    CUFFT_CHECK(cufftXtExecDescriptorD2Z(plan_r2c, s.V3_buf, s.V3_buf));
    // Normalize
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
    #pragma omp parallel for num_threads(s.nGPUs)
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf, g), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf, g), s.nc_local, inv_N);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf, g), s.nc_local, inv_N);
    }

    // Project initial condition to divergence-free space
    cout << "Projecting to divergence-free space..." << endl;
    #pragma omp parallel for num_threads(s.nGPUs)
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        int x_offset = g * s.nx_local;
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
        kernel_make_div_free<<<gc, BLOCK>>>(
            (GCplx*)gpu_ptr(s.V1_buf, g), (GCplx*)gpu_ptr(s.V2_buf, g), (GCplx*)gpu_ptr(s.V3_buf, g),
            s.nx_local, x_offset);
    }

    // V_buf is spectral, subFormat=3 (from D2Z above)
    // Initial diagnostics
    {
        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, s, 0.0, L2_err, max_div);
        cout << "\nInitial condition (t=0):" << endl;
        cout << "  L2 error:   " << scientific << L2_err << endl;
        cout << "  max|div V|: " << max_div << "\n" << endl;
    }

    // Time integration
    cout << "============================================================" << endl;
    cout << "  Time Integration (RK4)" << endl;
    cout << "============================================================" << endl;
    cout << setw(6) << "Step" << setw(12) << "Wall(s)"
         << setw(15) << "L2 Error" << setw(15) << "Max |div V|" << endl;
    cout << "------------------------------------------------------------------------" << endl;

    double t_wall_total = 0.0;
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaSetDevice(s.gpu_ids[0]));
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int it = 0; it <= NT_RUN; ++it) {
        double t_cur = it * TAU;

        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, s, t_cur, L2_err, max_div);
        cout << setw(6) << it
             << setw(12) << fixed << setprecision(4) << t_wall_total
             << setw(15) << scientific << setprecision(4) << L2_err
             << setw(15) << max_div << endl;

        if (it < NT_RUN) {
            CUDA_CHECK(cudaSetDevice(s.gpu_ids[0]));
            CUDA_CHECK(cudaEventRecord(ev_start));
            rk4_step(plan_r2c, plan_c2r, s, t_cur);
            // Sync all GPUs
            for (int g = 0; g < s.nGPUs; g++) {
                CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            CUDA_CHECK(cudaSetDevice(s.gpu_ids[0]));
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            t_wall_total += (double)ms * 1e-3;
        }
    }

    cout << "============================================================" << endl;
    cout << "  Total steps: " << NT_RUN << endl;
    cout << "  Total wall:  " << fixed << setprecision(4) << t_wall_total << " s" << endl;
    cout << "  Avg/step:    " << t_wall_total / NT_RUN << " s" << endl;
    cout << "============================================================" << endl;

    // Cleanup
    CUDA_CHECK(cudaSetDevice(s.gpu_ids[0]));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUFFT_CHECK(cufftXtFree(s.V1_buf));    CUFFT_CHECK(cufftXtFree(s.V2_buf));    CUFFT_CHECK(cufftXtFree(s.V3_buf));
    CUFFT_CHECK(cufftXtFree(s.rot1_buf));  CUFFT_CHECK(cufftXtFree(s.rot2_buf));  CUFFT_CHECK(cufftXtFree(s.rot3_buf));
    CUFFT_CHECK(cufftXtFree(s.work1_buf)); CUFFT_CHECK(cufftXtFree(s.work2_buf)); CUFFT_CHECK(cufftXtFree(s.work3_buf));
    for (int g = 0; g < s.nGPUs; g++) {
        CUDA_CHECK(cudaSetDevice(s.gpu_ids[g]));
        cudaFree(s.visc1_c[g]); cudaFree(s.visc2_c[g]); cudaFree(s.visc3_c[g]);
        cudaFree(s.f1_c[g]);    cudaFree(s.f2_c[g]);    cudaFree(s.f3_c[g]);
        cudaFree(s.rhs1_c[g]);  cudaFree(s.rhs2_c[g]);  cudaFree(s.rhs3_c[g]);
        cudaFree(s.k1v1[g]); cudaFree(s.k2v1[g]); cudaFree(s.k3v1[g]); cudaFree(s.k4v1[g]);
        cudaFree(s.k1v2[g]); cudaFree(s.k2v2[g]); cudaFree(s.k3v2[g]); cudaFree(s.k4v2[g]);
        cudaFree(s.k1v3[g]); cudaFree(s.k2v3[g]); cudaFree(s.k3v3[g]); cudaFree(s.k4v3[g]);
        cudaFree(s.V1_orig[g]); cudaFree(s.V2_orig[g]); cudaFree(s.V3_orig[g]);
        cudaFree(s.d_scratch[g]);
    }
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    return 0;
}
