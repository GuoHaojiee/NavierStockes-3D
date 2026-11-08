#include <iostream>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "common_mode.h"

using namespace std;

// Run mode / phase timing (see common_mode.h).
//   g_with_force = true  (verify): analytic forcing f every RHS (original behaviour)
//   g_with_force = false (bench) : f = 0; forcing fill and its 3 FFTs are skipped.
// Time scheme: classic RK4 in accumulated form (see kernel_rk4_stage); k is never stored.
static bool g_with_force = true;
static nsmode::GpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;

typedef cufftDoubleReal    GReal;
typedef cufftDoubleComplex GCplx;

// ============================================================
// Grid parameters — host-side (set at runtime from argv)
// ============================================================
static int    NX, NY, NZ, NZC, NT_TOTAL, NT_RUN;
static long long NR, NC_TOT;
static double LX, LY, LZ, DX, DY, DZ, TAU;

// Device constants (read-only from kernels)
__device__ __constant__ int    d_NX, d_NY, d_NZ, d_NZC;
__device__ __constant__ long long d_NR, d_NC_TOT;
__device__ __constant__ double d_DX, d_DY, d_DZ;

// CUDA launch config (computed at runtime)
constexpr int BLOCK  = 256;
static int GRID_R, GRID_C;

// ============================================================
// Error checking macros
// ============================================================
#define CUDA_CHECK(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); \
    } \
} while (0)

#define CUFFT_CHECK(e) do { \
    cufftResult _e = (e); \
    if (_e != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error at %s:%d: code %d\n", \
                __FILE__, __LINE__, (int)_e); exit(1); \
    } \
} while (0)

// ============================================================
// GPU array bundle
// ============================================================
struct GPUArrays {
    // Velocity: real space (no FFTW padding) + spectral space (stage value)
    GReal *V1_r, *V2_r, *V3_r;          // also holds the forcing f in verify mode
    GCplx *V1_c, *V2_c, *V3_c;          // destroyed by Z2D, rewritten by kernel_rk4_stage
    // Vorticity; after the forward FFT of the cross product rot_c holds nl
    GReal *rot1_r, *rot2_r, *rot3_r;
    GCplx *rot1_c, *rot2_c, *rot3_c;
    // -k^2 V (viscous term), written by kernel_rot_and_viscous
    GCplx *rhs1_c, *rhs2_c, *rhs3_c;
    // V^n and the RK4 accumulator
    GCplx *V1_c_orig, *V2_c_orig, *V3_c_orig;
    GCplx *acc1_c, *acc2_c, *acc3_c;
    // Scratch for the diagnostics (after the timed loop): shares rhs1_c's memory
    GReal *d_scratch;
};

template<typename T>
static void gmalloc(T** p, long long n) {
    CUDA_CHECK(cudaMalloc((void**)p, (size_t)n * sizeof(T)));
}

// ============================================================
// Analytical solution (Taylor-Green variant, __host__ __device__)
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
    // d²V1/dx² = d²V1/dy² = 9*coef*(c²-s)*cos(6z)
    // d²V1/dz² = -36*coef*cos(6z)
    return 2.0 * 9.0 * coef * (c*c - s) * cos(6.0*z)
           - 36.0 * coef * cos(6.0*z);
}
__host__ __device__ double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x, y, z, t);
}
__host__ __device__ double func_laplace_V3(double x, double y, double z, double t) {
    double s = sin(3.0*x + 3.0*y), c = cos(3.0*x + 3.0*y);
    double coef = (t*t + 1.0) * exp(s) * c;
    // d²V3/dx² = d²V3/dy² = -9*coef*((c²-s)-(2s+1))*sin(6z)
    // d²V3/dz² = 36*coef*sin(6z)
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
__host__ __device__ double func_rot3(double /*x*/, double /*y*/, double /*z*/, double /*t*/) {
    return 0.0;
}
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
    return func_dV1_dt(x,y,z,t) - func_laplace_V1(x,y,z,t)
           - func_vcr1(x,y,z,t) + func_grad_p1(x,y,z,t);
}
__host__ __device__ double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t) - func_laplace_V2(x,y,z,t)
           - func_vcr2(x,y,z,t) + func_grad_p2(x,y,z,t);
}
__host__ __device__ double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t) - func_laplace_V3(x,y,z,t)
           - func_vcr3(x,y,z,t) + func_grad_p3(x,y,z,t);
}

// ============================================================
// CUDA Kernels
// ============================================================

// Fill real arrays with analytical velocity at time t
__global__ void kernel_fill_velocity(GReal* V1_r, GReal* V2_r, GReal* V3_r, double t) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NR) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZ));
    double x = i * d_DX, y = j * d_DY, z = k * d_DZ;
    V1_r[idx] = func_V1(x, y, z, t);
    V2_r[idx] = func_V2(x, y, z, t);
    V3_r[idx] = func_V3(x, y, z, t);
}

// Scale complex array: A *= scale (normalization after forward FFT)
__global__ void kernel_scale_cplx(GCplx* A, long long total, double scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    A[idx].x *= scale;
    A[idx].y *= scale;
}

// Kernel A: rot_c = ik × V_c and rhs_c = -k² V_c in one pass
__global__ void kernel_rot_and_viscous(const GCplx* V1_c, const GCplx* V2_c, const GCplx* V3_c,
                                        GCplx* rot1_c, GCplx* rot2_c, GCplx* rot3_c,
                                        GCplx* rhs1_c, GCplx* rhs2_c, GCplx* rhs3_c) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NC_TOT) return;
    int k = (int)(idx % d_NZC);
    int j = (int)((idx / d_NZC) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZC));
    double kx = (i <= d_NX/2) ? (double)i : (double)(i - d_NX);
    double ky = (j <= d_NY/2) ? (double)j : (double)(j - d_NY);
    double kz = (double)k;
    double k2 = kx*kx + ky*ky + kz*kz;
    GCplx v1 = V1_c[idx], v2 = V2_c[idx], v3 = V3_c[idx];
    // i*(a+ib) = -b + ia
    rot1_c[idx].x = -(ky * v3.y - kz * v2.y);
    rot1_c[idx].y =   ky * v3.x - kz * v2.x;
    rot2_c[idx].x = -(kz * v1.y - kx * v3.y);
    rot2_c[idx].y =   kz * v1.x - kx * v3.x;
    rot3_c[idx].x = -(kx * v2.y - ky * v1.y);
    rot3_c[idx].y =   kx * v2.x - ky * v1.x;
    rhs1_c[idx].x = -k2 * v1.x;  rhs1_c[idx].y = -k2 * v1.y;
    rhs2_c[idx].x = -k2 * v2.x;  rhs2_c[idx].y = -k2 * v2.y;
    rhs3_c[idx].x = -k2 * v3.x;  rhs3_c[idx].y = -k2 * v3.y;
}

enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

// Kernel B: end of every RK4 stage (classic RK4, accumulated form), k in registers only.
//   k = P[ inv_N*nl + (-k^2 V + inv_N*f) ]     (f only in verify; sum order as before)
//   ACC_INIT: acc = base + a_acc*k;  ACC_ADD: acc += a_acc*k;  ACC_KEEP: acc unchanged
//   V = base + a_v*k
// Stage 1: base=V_orig INIT tau/6, a_v=tau/2 | 2: V_orig ADD tau/3, tau/2 |
// Stage 3: V_orig ADD tau/3, tau | 4: base=acc KEEP, a_v=tau/6.  k²=0 is left unprojected.
__global__ void kernel_rk4_stage(GCplx* V1, GCplx* V2, GCplx* V3,
                                  const GCplx* b1, const GCplx* b2, const GCplx* b3,
                                  GCplx* acc1, GCplx* acc2, GCplx* acc3,
                                  const GCplx* nl1, const GCplx* nl2, const GCplx* nl3,
                                  const GCplx* rhs1, const GCplx* rhs2, const GCplx* rhs3,
                                  const GCplx* f1, const GCplx* f2, const GCplx* f3,
                                  double inv_N, double a_acc, double a_v, int acc_mode) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NC_TOT) return;
    int k = (int)(idx % d_NZC);
    int j = (int)((idx / d_NZC) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZC));
    double kx = (i <= d_NX/2) ? (double)i : (double)(i - d_NX);
    double ky = (j <= d_NY/2) ? (double)j : (double)(j - d_NY);
    double kz = (double)k;
    double k2 = kx*kx + ky*ky + kz*kz;

    double s1x = rhs1[idx].x, s1y = rhs1[idx].y;
    double s2x = rhs2[idx].x, s2y = rhs2[idx].y;
    double s3x = rhs3[idx].x, s3y = rhs3[idx].y;
    if (f1) {   // f1..3 alias V1..3: read before V is written below
        s1x += f1[idx].x * inv_N;  s1y += f1[idx].y * inv_N;
        s2x += f2[idx].x * inv_N;  s2y += f2[idx].y * inv_N;
        s3x += f3[idx].x * inv_N;  s3y += f3[idx].y * inv_N;
    }
    double k1x = nl1[idx].x * inv_N + s1x, k1y = nl1[idx].y * inv_N + s1y;
    double k2x = nl2[idx].x * inv_N + s2x, k2y = nl2[idx].y * inv_N + s2y;
    double k3x = nl3[idx].x * inv_N + s3x, k3y = nl3[idx].y * inv_N + s3y;

    if (k2 >= 1e-10) {
        double div_r = -(kx*k1y + ky*k2y + kz*k3y);
        double div_i =   kx*k1x + ky*k2x + kz*k3x;
        double phi_r = div_r / (-k2);
        double phi_i = div_i / (-k2);
        k1x -= -kx * phi_i;  k1y -=  kx * phi_r;
        k2x -= -ky * phi_i;  k2y -=  ky * phi_r;
        k3x -= -kz * phi_i;  k3y -=  kz * phi_r;
    }

    GCplx c1 = b1[idx], c2 = b2[idx], c3 = b3[idx];
    if (acc_mode == ACC_INIT) {
        acc1[idx].x = c1.x + a_acc*k1x;  acc1[idx].y = c1.y + a_acc*k1y;
        acc2[idx].x = c2.x + a_acc*k2x;  acc2[idx].y = c2.y + a_acc*k2y;
        acc3[idx].x = c3.x + a_acc*k3x;  acc3[idx].y = c3.y + a_acc*k3y;
    } else if (acc_mode == ACC_ADD) {
        acc1[idx].x += a_acc*k1x;  acc1[idx].y += a_acc*k1y;
        acc2[idx].x += a_acc*k2x;  acc2[idx].y += a_acc*k2y;
        acc3[idx].x += a_acc*k3x;  acc3[idx].y += a_acc*k3y;
    }
    V1[idx].x = c1.x + a_v*k1x;  V1[idx].y = c1.y + a_v*k1y;
    V2[idx].x = c2.x + a_v*k2x;  V2[idx].y = c2.y + a_v*k2y;
    V3[idx].x = c3.x + a_v*k3x;  V3[idx].y = c3.y + a_v*k3y;
}

// Projection to divergence-free space (in-place):
//   div = ik·V,  phi = div/(-k²),  V -= ik*phi
__global__ void kernel_make_div_free(GCplx* V1_c, GCplx* V2_c, GCplx* V3_c) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NC_TOT) return;
    int k = (int)(idx % d_NZC);
    int j = (int)((idx / d_NZC) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZC));
    double kx = (i <= d_NX/2) ? (double)i : (double)(i - d_NX);
    double ky = (j <= d_NY/2) ? (double)j : (double)(j - d_NY);
    double kz = (double)k;
    double k2 = kx*kx + ky*ky + kz*kz;
    if (k2 < 1e-10) return;

    // div = ik·V: real = -(kx*V1.y + ky*V2.y + kz*V3.y), imag = kx*V1.x + ...
    double div_r = -(kx*V1_c[idx].y + ky*V2_c[idx].y + kz*V3_c[idx].y);
    double div_i =   kx*V1_c[idx].x + ky*V2_c[idx].x + kz*V3_c[idx].x;
    double phi_r = div_r / (-k2);
    double phi_i = div_i / (-k2);

    // V -= ik*phi: i*kx*(a+ib) = (-kx*b) + i*(kx*a)
    V1_c[idx].x -= -kx * phi_i;
    V1_c[idx].y -=  kx * phi_r;
    V2_c[idx].x -= -ky * phi_i;
    V2_c[idx].y -=  ky * phi_r;
    V3_c[idx].x -= -kz * phi_i;
    V3_c[idx].y -=  kz * phi_r;
}

// Real-space cross product: V_r × rot_r → overwrite rot_r
__global__ void kernel_cross_product(const GReal* V1_r, const GReal* V2_r, const GReal* V3_r,
                                      GReal* rot1_r, GReal* rot2_r, GReal* rot3_r) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NR) return;
    double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
    double w1 = rot1_r[idx], w2 = rot2_r[idx], w3 = rot3_r[idx];
    rot1_r[idx] = v2*w3 - v3*w2;
    rot2_r[idx] = v3*w1 - v1*w3;
    rot3_r[idx] = v1*w2 - v2*w1;
}

// Fill forcing real arrays at time t
__global__ void kernel_fill_forcing(GReal* w1_r, GReal* w2_r, GReal* w3_r, double t) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NR) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZ));
    double x = i * d_DX, y = j * d_DY, z = k * d_DZ;
    w1_r[idx] = func_f1(x, y, z, t);
    w2_r[idx] = func_f2(x, y, z, t);
    w3_r[idx] = func_f3(x, y, z, t);
}

// Compute squared error per grid point: |V_r - V_exact|²
__global__ void kernel_error_sq(const GReal* V1_r, const GReal* V2_r, const GReal* V3_r,
                                 GReal* err_r, double t) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NR) return;
    int k = (int)(idx % d_NZ);
    int j = (int)((idx / d_NZ) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZ));
    double x = i * d_DX, y = j * d_DY, z = k * d_DZ;
    double d1 = V1_r[idx] - func_V1(x, y, z, t);
    double d2 = V2_r[idx] - func_V2(x, y, z, t);
    double d3 = V3_r[idx] - func_V3(x, y, z, t);
    err_r[idx] = d1*d1 + d2*d2 + d3*d3;
}

// |v|^2 per grid point (bench-mode kinetic-energy diagnostic)
__global__ void kernel_energy_sq(const GReal* V1_r, const GReal* V2_r, const GReal* V3_r,
                                  GReal* e_r) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NR) return;
    double v1 = V1_r[idx], v2 = V2_r[idx], v3 = V3_r[idx];
    e_r[idx] = v1*v1 + v2*v2 + v3*v3;
}

// Compute |ik·V_c| (spectral divergence magnitude) per mode
__global__ void kernel_div_abs(const GCplx* V1_c, const GCplx* V2_c, const GCplx* V3_c,
                                GReal* div_r) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NC_TOT) return;
    int k = (int)(idx % d_NZC);
    int j = (int)((idx / d_NZC) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZC));
    double kx = (i <= d_NX/2) ? (double)i : (double)(i - d_NX);
    double ky = (j <= d_NY/2) ? (double)j : (double)(j - d_NY);
    double kz = (double)k;
    double dr = -(kx*V1_c[idx].y + ky*V2_c[idx].y + kz*V3_c[idx].y);
    double di =   kx*V1_c[idx].x + ky*V2_c[idx].x + kz*V3_c[idx].x;
    div_r[idx] = sqrt(dr*dr + di*di);
}

// ============================================================
// One RK4 stage (9 FFTs, +3 for the forcing in verify mode):
//   1. kernel A: rot_c = ik × V_c, rhs_c = -k² V_c
//   2. Z2D(V_c → V_r), Z2D(rot_c → rot_r)   (may destroy V_c, rot_c: not used afterwards)
//   3. cross product V_r × rot_r → rot_r
//   4. D2Z(rot_r → rot_c)                    (nl, not normalised)
//   5. verify: forcing → V_r, D2Z(V_r → V_c) (f, not normalised)
//   6. kernel B: scale + sum + projection + RK4 update → V_c (next stage value)
// ============================================================
static void rk4_stage(cufftHandle plan_r2c, cufftHandle plan_c2r,
                      GPUArrays& g, double t, int stage) {
    const double inv_N = 1.0 / ((double)NX * NY * NZ);

    kernel_rot_and_viscous<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c,
                                              g.rot1_c, g.rot2_c, g.rot3_c,
                                              g.rhs1_c, g.rhs2_c, g.rhs3_c);
    g_timer.mark(PH_SPEC);

    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V1_c, g.V1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V2_c, g.V2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V3_c, g.V3_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot1_c, g.rot1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot2_c, g.rot2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot3_c, g.rot3_r));
    g_timer.mark(PH_FFT);

    kernel_cross_product<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r,
                                             g.rot1_r, g.rot2_r, g.rot3_r);
    g_timer.mark(PH_PHYS);

    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot1_r, g.rot1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot2_r, g.rot2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot3_r, g.rot3_c));
    g_timer.mark(PH_FFT);

    if (g_with_force) {
        kernel_fill_forcing<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r, t);
        CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V1_r, g.V1_c));
        CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V2_r, g.V2_c));
        CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V3_r, g.V3_c));
        g_timer.mark(PH_FORCE);
    }

    const GCplx *b1 = g.V1_c_orig, *b2 = g.V2_c_orig, *b3 = g.V3_c_orig;
    double a_acc = 0.0, a_v = 0.0;
    int acc_mode = ACC_ADD;
    switch (stage) {
        case 1:  acc_mode = ACC_INIT; a_acc = TAU/6.0; a_v = 0.5*TAU; break;
        case 2:                       a_acc = TAU/3.0; a_v = 0.5*TAU; break;
        case 3:                       a_acc = TAU/3.0; a_v = TAU;     break;
        default: acc_mode = ACC_KEEP;                  a_v = TAU/6.0;
                 b1 = g.acc1_c; b2 = g.acc2_c; b3 = g.acc3_c;         break;
    }
    kernel_rk4_stage<<<GRID_C, BLOCK>>>(
        g.V1_c, g.V2_c, g.V3_c, b1, b2, b3, g.acc1_c, g.acc2_c, g.acc3_c,
        g.rot1_c, g.rot2_c, g.rot3_c, g.rhs1_c, g.rhs2_c, g.rhs3_c,
        g_with_force ? g.V1_c : nullptr, g_with_force ? g.V2_c : nullptr, g_with_force ? g.V3_c : nullptr,
        inv_N, a_acc, a_v, acc_mode);
    g_timer.mark(PH_SPEC);
}

// ============================================================
// RK4 time integration (spectral space)
// V^{n+1} = V^n + τ/6*(k1 + 2k2 + 2k3 + k4), accumulated stage by stage
// ============================================================
static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r,
                      GPUArrays& g, double t) {
    CUDA_CHECK(cudaMemcpy(g.V1_c_orig, g.V1_c, NC_TOT * sizeof(GCplx), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.V2_c_orig, g.V2_c, NC_TOT * sizeof(GCplx), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.V3_c_orig, g.V3_c, NC_TOT * sizeof(GCplx), cudaMemcpyDeviceToDevice));
    g_timer.mark(PH_SPEC);

    rk4_stage(plan_r2c, plan_c2r, g, t,           1);
    rk4_stage(plan_r2c, plan_c2r, g, t + 0.5*TAU, 2);
    rk4_stage(plan_r2c, plan_c2r, g, t + 0.5*TAU, 3);
    rk4_stage(plan_r2c, plan_c2r, g, t + TAU,     4);
}

// ============================================================
// Diagnostics: L2 error (verify) or kinetic energy 0.5*sum|v|^2 (bench),
// and max|div V|.
//
// V_c is destroyed by Z2D then restored by D2Z+scale.
// After this call V_c is consistent with V_r (to FFT round-trip precision).
// ============================================================
static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
                                 GPUArrays& g, double t, bool verify,
                                 double& L2_or_energy, double& max_div) {
    const double inv_N = 1.0 / ((double)NX * NY * NZ);

    // 1. V_c → V_r (Z2D, may destroy V_c)
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V1_c, g.V1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V2_c, g.V2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V3_c, g.V3_r));

    // 2. Squared error (verify) or |v|^2 (bench) → d_scratch, then reduce
    if (verify)
        kernel_error_sq<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r, g.d_scratch, t);
    else
        kernel_energy_sq<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r, g.d_scratch);
    thrust::device_ptr<double> sp(g.d_scratch);
    double sq = thrust::reduce(thrust::device, sp, sp + NR, 0.0);
    L2_or_energy = verify ? sqrt(sq * DX * DY * DZ) : 0.5 * sq;

    // 3. Restore V_c: V_r → V_c (D2Z) + normalize
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V1_r, g.V1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V2_r, g.V2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V3_r, g.V3_c));
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.V1_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.V2_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.V3_c, NC_TOT, inv_N);

    // 4. max|div V| in spectral space
    kernel_div_abs<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c, g.d_scratch);
    max_div = thrust::reduce(thrust::device, sp, sp + NC_TOT,
                              0.0, thrust::maximum<double>());
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " " << nsmode::usage_tail(false) << endl;
        cerr << "  Example: " << argv[0] << " 64 64 64 0.00001 100 bench" << endl;
        return 1;
    }
    nsmode::RunArgs rargs;
    {
        std::string perr;
        if (!nsmode::parse_tail_args(argc, argv, 6, /*allow_threads=*/false, rargs, perr)) {
            cerr << "Error: " << perr << "\nUsage: " << argv[0] << " "
                 << nsmode::usage_tail(false) << endl;
            return 1;
        }
    }
    const nsmode::RunMode mode = rargs.mode;
    g_with_force = (mode == nsmode::MODE_VERIFY);

    // Parse command-line parameters
    NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]);
    TAU = atof(argv[4]); NT_RUN = atoi(argv[5]);
    NZC = NZ / 2 + 1;
    NR  = (long long)NX * NY * NZ;
    NC_TOT = (long long)NX * NY * NZC;
    LX = LY = LZ = 2.0 * M_PI;
    DX = LX / NX; DY = LY / NY; DZ = LZ / NZ;
    GRID_R = (int)((NR     + BLOCK - 1) / BLOCK);
    GRID_C = (int)((NC_TOT + BLOCK - 1) / BLOCK);

    // Copy grid parameters to device constant memory
    cudaMemcpyToSymbol(d_NX, &NX, sizeof(int));
    cudaMemcpyToSymbol(d_NY, &NY, sizeof(int));
    cudaMemcpyToSymbol(d_NZ, &NZ, sizeof(int));
    cudaMemcpyToSymbol(d_NZC, &NZC, sizeof(int));
    cudaMemcpyToSymbol(d_NR, &NR, sizeof(long long));
    cudaMemcpyToSymbol(d_NC_TOT, &NC_TOT, sizeof(long long));
    cudaMemcpyToSymbol(d_DX, &DX, sizeof(double));
    cudaMemcpyToSymbol(d_DY, &DY, sizeof(double));
    cudaMemcpyToSymbol(d_DZ, &DZ, sizeof(double));

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cout << "============================================================" << endl;
    cout << "  Navier-Stokes Solver - cuFFT Single GPU Version" << endl;
    cout << "============================================================" << endl;
    cout << "GPU: " << prop.name << endl;
    printf("Grid: %d x %d x %d, dt=%.2e, steps=%d\n", NX, NY, NZ, TAU, NT_RUN);
    cout << "Domain: [0, 2π]³" << endl;
    cout << "Run mode: " << nsmode::mode_name(mode)
         << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << endl;

    // Memory estimate: 15 complex + 6 real full-size arrays (same in bench and verify)
    double mem_mb = (6.0 * NR * sizeof(GReal) + 15.0 * NC_TOT * sizeof(GCplx)) / (1024.0 * 1024.0);
    cout << "GPU memory (arrays): " << fixed << setprecision(1) << mem_mb << " MB" << endl;
    cout << "============================================================" << endl;

    // Create cuFFT plans (single plan reused for all transforms of same type)
    cout << "Creating cuFFT plans..." << endl;
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftPlan3d(&plan_r2c, NX, NY, NZ, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&plan_c2r, NX, NY, NZ, CUFFT_Z2D));

    // Allocate GPU arrays
    cout << "Allocating GPU memory..." << endl;
    GPUArrays g;
    gmalloc(&g.V1_r,  NR);   gmalloc(&g.V2_r,  NR);   gmalloc(&g.V3_r,  NR);
    gmalloc(&g.rot1_r,NR);   gmalloc(&g.rot2_r,NR);   gmalloc(&g.rot3_r,NR);
    gmalloc(&g.V1_c,     NC_TOT); gmalloc(&g.V2_c,     NC_TOT); gmalloc(&g.V3_c,     NC_TOT);
    gmalloc(&g.rot1_c,   NC_TOT); gmalloc(&g.rot2_c,   NC_TOT); gmalloc(&g.rot3_c,   NC_TOT);
    gmalloc(&g.rhs1_c,   NC_TOT); gmalloc(&g.rhs2_c,   NC_TOT); gmalloc(&g.rhs3_c,   NC_TOT);
    gmalloc(&g.V1_c_orig,NC_TOT); gmalloc(&g.V2_c_orig,NC_TOT); gmalloc(&g.V3_c_orig,NC_TOT);
    gmalloc(&g.acc1_c,   NC_TOT); gmalloc(&g.acc2_c,   NC_TOT); gmalloc(&g.acc3_c,   NC_TOT);
    // Diagnostics scratch (NR doubles, then NC_TOT doubles) reuses rhs1_c (2*NC_TOT doubles)
    g.d_scratch = (GReal*)g.rhs1_c;

    {
        const double GB = 1024.0 * 1024.0 * 1024.0;
        size_t mem_free = 0, mem_total = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
        printf("Memory report (per GPU):\n");
        printf("  Full-size arrays: 15 complex of %.3f GB, 6 real of %.3f GB\n",
               NC_TOT * sizeof(GCplx) / GB, NR * sizeof(GReal) / GB);
        printf("  GPU memory in use (cudaMemGetInfo): %.3f GB of %.3f GB\n",
               (mem_total - mem_free) / GB, mem_total / GB);
        fflush(stdout);
    }

    // Initial conditions (t = 0)
    cout << "Setting initial conditions..." << endl;
    kernel_fill_velocity<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r, 0.0);

    // Forward FFT: V_r → V_c, normalize  (V_c = DFT(V_r)/N)
    const double inv_N = 1.0 / ((double)NX * NY * NZ);
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V1_r, g.V1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V2_r, g.V2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.V3_r, g.V3_c));
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.V1_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.V2_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.V3_c, NC_TOT, inv_N);

    // Project initial condition to divergence-free space
    cout << "Projecting initial condition to divergence-free space..." << endl;
    kernel_make_div_free<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c);

    // ----------------------------------------------------------------
    // Time integration (RK4)
    // ----------------------------------------------------------------
    double t_wall_total = 0.0;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    g_timer.init();

    for (int it = 0; it < NT_RUN; ++it) {
        double t_cur = it * TAU;
        CUDA_CHECK(cudaEventRecord(ev_start));
        g_timer.start();
        rk4_step(plan_r2c, plan_c2r, g, t_cur);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        g_timer.collect();
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        t_wall_total += (double)ms * 1e-3;
    }

    g_timer.destroy();

    // Diagnostics (not timed)
    {
        double t_final = NT_RUN * TAU;
        double L2_or_E, max_div;
        const bool verify = (mode == nsmode::MODE_VERIFY);
        compute_diagnostics(plan_r2c, plan_c2r, g, t_final, verify, L2_or_E, max_div);
        if (verify)
            cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << L2_or_E << endl;
        else
            cout << "  Kinetic energy 0.5*sum|v|^2 (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(10) << L2_or_E << endl;
        cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): "
             << scientific << setprecision(4) << max_div << endl;
    }

    cout << "============================================================" << endl;
    cout << "  Timing Summary" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "  Total steps:     " << NT_RUN << endl;
    cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s" << endl;
    cout << "  Avg per step:    " << t_wall_total / NT_RUN << " s" << endl;
    nsmode::print_phase_summary(g_timer.acc, t_wall_total, NT_RUN, mode);
    cout << "============================================================" << endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));

    cudaFree(g.V1_r);      cudaFree(g.V2_r);      cudaFree(g.V3_r);
    cudaFree(g.rot1_r);    cudaFree(g.rot2_r);    cudaFree(g.rot3_r);
    cudaFree(g.V1_c);      cudaFree(g.V2_c);      cudaFree(g.V3_c);
    cudaFree(g.rot1_c);    cudaFree(g.rot2_c);    cudaFree(g.rot3_c);
    cudaFree(g.rhs1_c);    cudaFree(g.rhs2_c);    cudaFree(g.rhs3_c);   // d_scratch aliases rhs1_c
    cudaFree(g.V1_c_orig); cudaFree(g.V2_c_orig); cudaFree(g.V3_c_orig);
    cudaFree(g.acc1_c);    cudaFree(g.acc2_c);    cudaFree(g.acc3_c);

    return 0;
}
