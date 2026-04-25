#include <iostream>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

using namespace std;

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
    // Velocity: real space (no FFTW padding) + spectral space
    GReal *V1_r, *V2_r, *V3_r;
    GCplx *V1_c, *V2_c, *V3_c;
    // Vorticity buffers (reused each RHS call)
    GReal *rot1_r, *rot2_r, *rot3_r;
    GCplx *rot1_c, *rot2_c, *rot3_c;
    // Viscous term (spectral)
    GCplx *visc1_c, *visc2_c, *visc3_c;
    // Forcing: real work arrays + spectral result
    GReal *work1_r, *work2_r, *work3_r;
    GCplx *f1_c, *f2_c, *f3_c;
    // RK4 stages (k_i arrays, 4 stages × 3 components)
    GCplx *k1v1, *k1v2, *k1v3;
    GCplx *k2v1, *k2v2, *k2v3;
    GCplx *k3v1, *k3v2, *k3v3;
    GCplx *k4v1, *k4v2, *k4v3;
    // V^n backup for RK4
    GCplx *V1_c_orig, *V2_c_orig, *V3_c_orig;
    // Scratch for reductions (size NR, covers both NR and NC_TOT)
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

// Vorticity in spectral space: rot_c = ik × V_c
// (ik × V)_1 = i(ky*V3 - kz*V2), etc.
__global__ void kernel_compute_rot(const GCplx* V1_c, const GCplx* V2_c, const GCplx* V3_c,
                                    GCplx* rot1_c, GCplx* rot2_c, GCplx* rot3_c) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NC_TOT) return;
    int k = (int)(idx % d_NZC);
    int j = (int)((idx / d_NZC) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZC));
    double kx = (i <= d_NX/2) ? (double)i : (double)(i - d_NX);
    double ky = (j <= d_NY/2) ? (double)j : (double)(j - d_NY);
    double kz = (double)k;

    // i*(a+ib) = -b + ia, so real part of i*(ky*V3-kz*V2) = -(ky*V3.y - kz*V2.y)
    rot1_c[idx].x = -(ky * V3_c[idx].y - kz * V2_c[idx].y);
    rot1_c[idx].y =   ky * V3_c[idx].x - kz * V2_c[idx].x;
    rot2_c[idx].x = -(kz * V1_c[idx].y - kx * V3_c[idx].y);
    rot2_c[idx].y =   kz * V1_c[idx].x - kx * V3_c[idx].x;
    rot3_c[idx].x = -(kx * V2_c[idx].y - ky * V1_c[idx].y);
    rot3_c[idx].y =   kx * V2_c[idx].x - ky * V1_c[idx].x;
}

// Viscous term: visc_c = -k² V_c
__global__ void kernel_compute_viscous(const GCplx* V_c, GCplx* visc_c) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_NC_TOT) return;
    int k = (int)(idx % d_NZC);
    int j = (int)((idx / d_NZC) % d_NY);
    int i = (int)(idx / ((long long)d_NY * d_NZC));
    double kx = (i <= d_NX/2) ? (double)i : (double)(i - d_NX);
    double ky = (j <= d_NY/2) ? (double)j : (double)(j - d_NY);
    double kz = (double)k;
    double k2 = kx*kx + ky*ky + kz*kz;
    visc_c[idx].x = -k2 * V_c[idx].x;
    visc_c[idx].y = -k2 * V_c[idx].y;
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

// Add viscous and forcing terms to rhs (which already holds nl):
//   rhs += visc + f
__global__ void kernel_add_to_rhs(GCplx* rhs1, GCplx* rhs2, GCplx* rhs3,
                                   const GCplx* visc1, const GCplx* visc2, const GCplx* visc3,
                                   const GCplx* f1, const GCplx* f2, const GCplx* f3,
                                   long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    rhs1[idx].x += visc1[idx].x + f1[idx].x;
    rhs1[idx].y += visc1[idx].y + f1[idx].y;
    rhs2[idx].x += visc2[idx].x + f2[idx].x;
    rhs2[idx].y += visc2[idx].y + f2[idx].y;
    rhs3[idx].x += visc3[idx].x + f3[idx].x;
    rhs3[idx].y += visc3[idx].y + f3[idx].y;
}

// RK4 intermediate: V = orig + alpha * k
__global__ void kernel_rk4_axpy(GCplx* V1, GCplx* V2, GCplx* V3,
                                  const GCplx* orig1, const GCplx* orig2, const GCplx* orig3,
                                  const GCplx* k1, const GCplx* k2, const GCplx* k3,
                                  double alpha, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    V1[idx].x = orig1[idx].x + alpha * k1[idx].x;
    V1[idx].y = orig1[idx].y + alpha * k1[idx].y;
    V2[idx].x = orig2[idx].x + alpha * k2[idx].x;
    V2[idx].y = orig2[idx].y + alpha * k2[idx].y;
    V3[idx].x = orig3[idx].x + alpha * k3[idx].x;
    V3[idx].y = orig3[idx].y + alpha * k3[idx].y;
}

// RK4 final: V = orig + dtd6*(k1 + 2*k2 + 2*k3 + k4)
__global__ void kernel_rk4_update(GCplx* V1, GCplx* V2, GCplx* V3,
                                   const GCplx* orig1, const GCplx* orig2, const GCplx* orig3,
                                   const GCplx* k1v1, const GCplx* k2v1,
                                   const GCplx* k3v1, const GCplx* k4v1,
                                   const GCplx* k1v2, const GCplx* k2v2,
                                   const GCplx* k3v2, const GCplx* k4v2,
                                   const GCplx* k1v3, const GCplx* k2v3,
                                   const GCplx* k3v3, const GCplx* k4v3,
                                   double dtd6, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    V1[idx].x = orig1[idx].x + dtd6*(k1v1[idx].x + 2.0*k2v1[idx].x + 2.0*k3v1[idx].x + k4v1[idx].x);
    V1[idx].y = orig1[idx].y + dtd6*(k1v1[idx].y + 2.0*k2v1[idx].y + 2.0*k3v1[idx].y + k4v1[idx].y);
    V2[idx].x = orig2[idx].x + dtd6*(k1v2[idx].x + 2.0*k2v2[idx].x + 2.0*k3v2[idx].x + k4v2[idx].x);
    V2[idx].y = orig2[idx].y + dtd6*(k1v2[idx].y + 2.0*k2v2[idx].y + 2.0*k3v2[idx].y + k4v2[idx].y);
    V3[idx].x = orig3[idx].x + dtd6*(k1v3[idx].x + 2.0*k2v3[idx].x + 2.0*k3v3[idx].x + k4v3[idx].x);
    V3[idx].y = orig3[idx].y + dtd6*(k1v3[idx].y + 2.0*k2v3[idx].y + 2.0*k3v3[idx].y + k4v3[idx].y);
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
// Nonlinear term: V × rot(V) → rhs_c  (nl output)
//
// Algorithm (9 FFTs):
//   1. rot_c = ik × V_c              (spectral, reads V_c)
//   2. Z2D(V_c → V_r)                (destroys V_c — viscous must run first)
//   3. Z2D(rot_c → rot_r)            (destroys rot_c)
//   4. real cross product → rot_r    (overwrites)
//   5. D2Z(rot_r → rhs_c) + scale   (nl stored in rhs_c)
// ============================================================
static void compute_nonlinear(cufftHandle plan_r2c, cufftHandle plan_c2r,
                               GPUArrays& g,
                               GCplx* rhs1_c, GCplx* rhs2_c, GCplx* rhs3_c) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);

    // 1. Spectral vorticity (reads V_c)
    kernel_compute_rot<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c,
                                          g.rot1_c, g.rot2_c, g.rot3_c);
    // 2. V_c → V_r (may destroy V_c)
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V1_c, g.V1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V2_c, g.V2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V3_c, g.V3_r));
    // 3. rot_c → rot_r (destroys rot_c)
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot1_c, g.rot1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot2_c, g.rot2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot3_c, g.rot3_r));
    // 4. Real-space cross product V_r × rot_r → rot_r
    kernel_cross_product<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r,
                                             g.rot1_r, g.rot2_r, g.rot3_r);
    // 5. rot_r → rhs_c (as nl), normalize
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot1_r, rhs1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot2_r, rhs2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot3_r, rhs3_c));
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(rhs1_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(rhs2_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(rhs3_c, NC_TOT, inv_N);
}

// ============================================================
// Full RHS: nl + visc + f, projected to divergence-free space
// Output → rhs_c (= k_i in RK4)
// Note: V_c is destroyed inside (via Z2D in compute_nonlinear)
// ============================================================
static void compute_rhs(cufftHandle plan_r2c, cufftHandle plan_c2r,
                         GPUArrays& g,
                         GCplx* rhs1_c, GCplx* rhs2_c, GCplx* rhs3_c,
                         double t) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);

    // 1. Viscous: visc_c = -k² V_c  (MUST run before compute_nonlinear which destroys V_c)
    kernel_compute_viscous<<<GRID_C, BLOCK>>>(g.V1_c, g.visc1_c);
    kernel_compute_viscous<<<GRID_C, BLOCK>>>(g.V2_c, g.visc2_c);
    kernel_compute_viscous<<<GRID_C, BLOCK>>>(g.V3_c, g.visc3_c);

    // 2. Nonlinear: V × rot(V) → rhs_c  (V_c destroyed inside)
    compute_nonlinear(plan_r2c, plan_c2r, g, rhs1_c, rhs2_c, rhs3_c);

    // 3. Forcing: fill work_r, forward FFT, normalize
    kernel_fill_forcing<<<GRID_R, BLOCK>>>(g.work1_r, g.work2_r, g.work3_r, t);
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.work1_r, g.f1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.work2_r, g.f2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.work3_r, g.f3_c));
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.f1_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.f2_c, NC_TOT, inv_N);
    kernel_scale_cplx<<<GRID_C, BLOCK>>>(g.f3_c, NC_TOT, inv_N);

    // 4. Combine: rhs += visc + f  (rhs already has nl from step 2)
    kernel_add_to_rhs<<<GRID_C, BLOCK>>>(rhs1_c, rhs2_c, rhs3_c,
                                          g.visc1_c, g.visc2_c, g.visc3_c,
                                          g.f1_c, g.f2_c, g.f3_c, NC_TOT);
    // 5. Project: make rhs divergence-free
    kernel_make_div_free<<<GRID_C, BLOCK>>>(rhs1_c, rhs2_c, rhs3_c);
}

// ============================================================
// RK4 time integration (spectral space)
// V^{n+1} = V^n + τ/6*(k1 + 2k2 + 2k3 + k4)
// ============================================================
static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r,
                      GPUArrays& g, double t) {
    // Save V^n
    CUDA_CHECK(cudaMemcpy(g.V1_c_orig, g.V1_c, NC_TOT * sizeof(GCplx), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.V2_c_orig, g.V2_c, NC_TOT * sizeof(GCplx), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.V3_c_orig, g.V3_c, NC_TOT * sizeof(GCplx), cudaMemcpyDeviceToDevice));

    // k1 = RHS(V^n, t)  — V_c destroyed inside
    compute_rhs(plan_r2c, plan_c2r, g, g.k1v1, g.k1v2, g.k1v3, t);

    // V = V_orig + τ/2 * k1
    kernel_rk4_axpy<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c,
                                        g.V1_c_orig, g.V2_c_orig, g.V3_c_orig,
                                        g.k1v1, g.k1v2, g.k1v3, 0.5*TAU, NC_TOT);

    // k2 = RHS(V^n + τ/2*k1, t + τ/2)
    compute_rhs(plan_r2c, plan_c2r, g, g.k2v1, g.k2v2, g.k2v3, t + 0.5*TAU);

    // V = V_orig + τ/2 * k2
    kernel_rk4_axpy<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c,
                                        g.V1_c_orig, g.V2_c_orig, g.V3_c_orig,
                                        g.k2v1, g.k2v2, g.k2v3, 0.5*TAU, NC_TOT);

    // k3 = RHS(V^n + τ/2*k2, t + τ/2)
    compute_rhs(plan_r2c, plan_c2r, g, g.k3v1, g.k3v2, g.k3v3, t + 0.5*TAU);

    // V = V_orig + τ * k3
    kernel_rk4_axpy<<<GRID_C, BLOCK>>>(g.V1_c, g.V2_c, g.V3_c,
                                        g.V1_c_orig, g.V2_c_orig, g.V3_c_orig,
                                        g.k3v1, g.k3v2, g.k3v3, TAU, NC_TOT);

    // k4 = RHS(V^n + τ*k3, t + τ)
    compute_rhs(plan_r2c, plan_c2r, g, g.k4v1, g.k4v2, g.k4v3, t + TAU);

    // V^{n+1} = V^n + τ/6*(k1 + 2k2 + 2k3 + k4)
    kernel_rk4_update<<<GRID_C, BLOCK>>>(
        g.V1_c, g.V2_c, g.V3_c,
        g.V1_c_orig, g.V2_c_orig, g.V3_c_orig,
        g.k1v1, g.k2v1, g.k3v1, g.k4v1,
        g.k1v2, g.k2v2, g.k3v2, g.k4v2,
        g.k1v3, g.k2v3, g.k3v3, g.k4v3,
        TAU / 6.0, NC_TOT);
}

// ============================================================
// Diagnostics: L2 error and max|div V|
//
// V_c is destroyed by Z2D then restored by D2Z+scale.
// After this call V_c is consistent with V_r (to FFT round-trip precision).
// ============================================================
static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
                                 GPUArrays& g, double t,
                                 double& L2_err, double& max_div) {
    const double inv_N = 1.0 / (double)(NX * NY * NZ);

    // 1. V_c → V_r (Z2D, may destroy V_c)
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V1_c, g.V1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V2_c, g.V2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V3_c, g.V3_r));

    // 2. Squared error → d_scratch, then reduce
    kernel_error_sq<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r, g.d_scratch, t);
    thrust::device_ptr<double> sp(g.d_scratch);
    double err_sq = thrust::reduce(thrust::device, sp, sp + NR, 0.0);
    L2_err = sqrt(err_sq * DX * DY * DZ);

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
        cerr << "Usage: " << argv[0] << " NX NY NZ dt NSTEPS" << endl;
        cerr << "  Example: " << argv[0] << " 64 64 64 0.00001 100" << endl;
        return 1;
    }

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

    // Memory estimate
    double mem_mb = (10.0 * NR * sizeof(GReal)
                    + 27.0 * NC_TOT * sizeof(GCplx)) / (1024.0 * 1024.0);
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
    gmalloc(&g.work1_r,NR);  gmalloc(&g.work2_r,NR);  gmalloc(&g.work3_r,NR);
    gmalloc(&g.d_scratch,NR);

    gmalloc(&g.V1_c,    NC_TOT); gmalloc(&g.V2_c,    NC_TOT); gmalloc(&g.V3_c,    NC_TOT);
    gmalloc(&g.rot1_c,  NC_TOT); gmalloc(&g.rot2_c,  NC_TOT); gmalloc(&g.rot3_c,  NC_TOT);
    gmalloc(&g.visc1_c, NC_TOT); gmalloc(&g.visc2_c, NC_TOT); gmalloc(&g.visc3_c, NC_TOT);
    gmalloc(&g.f1_c,    NC_TOT); gmalloc(&g.f2_c,    NC_TOT); gmalloc(&g.f3_c,    NC_TOT);
    gmalloc(&g.k1v1,    NC_TOT); gmalloc(&g.k1v2,    NC_TOT); gmalloc(&g.k1v3,    NC_TOT);
    gmalloc(&g.k2v1,    NC_TOT); gmalloc(&g.k2v2,    NC_TOT); gmalloc(&g.k2v3,    NC_TOT);
    gmalloc(&g.k3v1,    NC_TOT); gmalloc(&g.k3v2,    NC_TOT); gmalloc(&g.k3v3,    NC_TOT);
    gmalloc(&g.k4v1,    NC_TOT); gmalloc(&g.k4v2,    NC_TOT); gmalloc(&g.k4v3,    NC_TOT);
    gmalloc(&g.V1_c_orig,NC_TOT); gmalloc(&g.V2_c_orig,NC_TOT); gmalloc(&g.V3_c_orig,NC_TOT);

    // Initial conditions (t = 0)
    cout << "Setting initial conditions..." << endl;
    kernel_fill_velocity<<<GRID_R, BLOCK>>>(g.V1_r, g.V2_r, g.V3_r, 0.0);

    // Forward FFT: V_r → V_c, normalize  (V_c = DFT(V_r)/N)
    const double inv_N = 1.0 / (double)(NX * NY * NZ);
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

    for (int it = 0; it < NT_RUN; ++it) {
        double t_cur = it * TAU;
        CUDA_CHECK(cudaEventRecord(ev_start));
        rk4_step(plan_r2c, plan_c2r, g, t_cur);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        t_wall_total += (double)ms * 1e-3;
    }

    {
        double t_final = NT_RUN * TAU;
        double L2_err, max_div;
        compute_diagnostics(plan_r2c, plan_c2r, g, t_final, L2_err, max_div);
        cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
             << scientific << setprecision(4) << L2_err << endl;
    }

    cout << "============================================================" << endl;
    cout << "  Timing Summary" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "  Total steps:     " << NT_RUN << endl;
    cout << "  Total wall time: " << fixed << setprecision(4) << t_wall_total << " s" << endl;
    cout << "  Avg per step:    " << t_wall_total / NT_RUN << " s" << endl;
    cout << "============================================================" << endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));

    cudaFree(g.V1_r);      cudaFree(g.V2_r);      cudaFree(g.V3_r);
    cudaFree(g.rot1_r);    cudaFree(g.rot2_r);    cudaFree(g.rot3_r);
    cudaFree(g.work1_r);   cudaFree(g.work2_r);   cudaFree(g.work3_r);
    cudaFree(g.d_scratch);
    cudaFree(g.V1_c);      cudaFree(g.V2_c);      cudaFree(g.V3_c);
    cudaFree(g.rot1_c);    cudaFree(g.rot2_c);    cudaFree(g.rot3_c);
    cudaFree(g.visc1_c);   cudaFree(g.visc2_c);   cudaFree(g.visc3_c);
    cudaFree(g.f1_c);      cudaFree(g.f2_c);      cudaFree(g.f3_c);
    cudaFree(g.k1v1);      cudaFree(g.k1v2);      cudaFree(g.k1v3);
    cudaFree(g.k2v1);      cudaFree(g.k2v2);      cudaFree(g.k2v3);
    cudaFree(g.k3v1);      cudaFree(g.k3v2);      cudaFree(g.k3v3);
    cudaFree(g.k4v1);      cudaFree(g.k4v2);      cudaFree(g.k4v3);
    cudaFree(g.V1_c_orig); cudaFree(g.V2_c_orig); cudaFree(g.V3_c_orig);

    return 0;
}
