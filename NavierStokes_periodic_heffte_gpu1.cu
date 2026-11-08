/**
 * NavierStokes_periodic_heffte_gpu1.cu
 * Navier-Stokes solver — heFFTe cufft backend, single node, single GPU
 *
 * Identical physics to NavierStokes_periodic_heffte_v2.cpp (CPU version).
 * Key differences vs CPU version:
 *   - Backend: heffte::backend::cufft  (GPU FFT via cuFFT)
 *   - Memory:  cudaMalloc device arrays instead of std::vector host arrays
 *   - Spectral ops: CUDA kernels instead of OpenMP loops
 *   - heFFTe backward does NOT destroy complex input (safe out-of-place)
 *
 * Normalization convention (same as CPU heFFTe version):
 *   forward scale::none  → V_c = DFT(V_r)  (non-normalized, = N·FFT_norm)
 *   backward scale::full → V_r = IDFT(V_c)/N  (auto-normalized physical value)
 *
 * Run: mpirun -np 1 ./navier_stokes_heffte_gpu1
 * Compile: make -f Makefile_heffte_gpu1
 */

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <heffte.h>
#include <mpi.h>
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
//   g_with_force = false (bench) : f = 0; forcing fill and its 3 forward FFTs are skipped.
// Time scheme: classic RK4 in accumulated form (see kernel_rk4_stage); k is never stored.
// heFFTe (cufft backend, no user stream) runs on the default stream, so the
// cudaEvent-based phase timer brackets its transforms correctly.
static bool g_with_force = true;
static nsmode::GpuPhaseTimer g_timer;
using nsmode::PH_FFT; using nsmode::PH_SPEC; using nsmode::PH_PHYS; using nsmode::PH_FORCE;

typedef double             GReal;
typedef cufftDoubleComplex GCplx;   // .x = real, .y = imag

// ============================================================
// Grid parameters — host-side (set at runtime from argv)
// ============================================================
static int    NX, NY, NZ, NT_TOTAL, NT_RUN;
static double LX, LY, LZ, DX, DY, DZ, TAU;

// Device constants (read-only from kernels)
__device__ __constant__ int    d_NX, d_NY;
__device__ __constant__ double d_DX, d_DY, d_DZ;

constexpr int BLOCK = 256;

// ============================================================
// Error macros
// ============================================================
#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

#define CUFFT_CHECK(e) do { cufftResult _e=(e); if(_e!=CUFFT_SUCCESS){ \
    fprintf(stderr,"cuFFT error %s:%d: %d\n",__FILE__,__LINE__,(int)_e); exit(1);} } while(0)

template<typename T>
static void gm(T** p, long long n) { CUDA_CHECK(cudaMalloc((void**)p, (size_t)n*sizeof(T))); }

// ============================================================
// Analytical solution  (Taylor-Green variant)
// ============================================================
__host__ __device__ double func_V1(double x,double y,double z,double t){return (t*t+1.)*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_V2(double x,double y,double z,double t){return (t*t+1.)*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_V3(double x,double y,double z,double t){return -(t*t+1.)*exp(sin(3.*x+3.*y))*cos(3.*x+3.*y)*sin(6.*z);}
__host__ __device__ double func_dV1_dt(double x,double y,double z,double t){return 2.*t*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_dV2_dt(double x,double y,double z,double t){return 2.*t*exp(sin(3.*x+3.*y))*cos(6.*z);}
__host__ __device__ double func_dV3_dt(double x,double y,double z,double t){return -2.*t*exp(sin(3.*x+3.*y))*cos(3.*x+3.*y)*sin(6.*z);}
__host__ __device__ double func_laplace_V1(double x,double y,double z,double t){
    double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);
    return (t*t+1.)*exp(s)*(18.*(c*c-s)-36.)*cos(6.*z);}
__host__ __device__ double func_laplace_V2(double x,double y,double z,double t){return func_laplace_V1(x,y,z,t);}
__host__ __device__ double func_laplace_V3(double x,double y,double z,double t){
    double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);
    return (t*t+1.)*exp(s)*c*((-18.)*((c*c-s)-(2.*s+1.))+36.)*sin(6.*z);}
__host__ __device__ double func_rot1(double x,double y,double z,double t){
    double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);
    return -(t*t+1.)*exp(s)*(3.*(c*c-s)-6.)*sin(6.*z);}
__host__ __device__ double func_rot2(double x,double y,double z,double t){
    double s=sin(3.*x+3.*y),c=cos(3.*x+3.*y);
    return -(t*t+1.)*exp(s)*(6.-3.*(c*c-s))*sin(6.*z);}
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
// CUDA kernels for real-space operations
// Box indexing for real space: flat idx → local (li,lj,lk) → global (gi,gj,gk)
//   li = idx % s0,  lj = (idx/s0)%s1,  lk = idx/(s0*s1)
//   gi = li+lo0,    gj = lj+lo1,        gk = lk+lo2
// ============================================================
__global__ void kernel_fill_velocity(GReal* V1, GReal* V2, GReal* V3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx % s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double x=(li+lo0)*d_DX, y=(lj+lo1)*d_DY, z=(lk+lo2)*d_DZ;
    V1[idx]=func_V1(x,y,z,0.); V2[idx]=func_V2(x,y,z,0.); V3[idx]=func_V3(x,y,z,0.);
}

__global__ void kernel_fill_forcing(GReal* W1, GReal* W2, GReal* W3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n, double t)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx % s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double x=(li+lo0)*d_DX, y=(lj+lo1)*d_DY, z=(lk+lo2)*d_DZ;
    W1[idx]=func_f1(x,y,z,t); W2[idx]=func_f2(x,y,z,t); W3[idx]=func_f3(x,y,z,t);
}

__global__ void kernel_cross_product(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* R1, GReal* R2, GReal* R3, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v1=V1[idx],v2=V2[idx],v3=V3[idx];
    double w1=R1[idx],w2=R2[idx],w3=R3[idx];
    R1[idx]=v2*w3-v3*w2; R2[idx]=v3*w1-v1*w3; R3[idx]=v1*w2-v2*w1;
}

__global__ void kernel_error_sq(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* err, int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n, double t)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx%s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double x=(li+lo0)*d_DX, y=(lj+lo1)*d_DY, z=(lk+lo2)*d_DZ;
    double d1=V1[idx]-func_V1(x,y,z,t), d2=V2[idx]-func_V2(x,y,z,t), d3=V3[idx]-func_V3(x,y,z,t);
    err[idx]=d1*d1+d2*d2+d3*d3;
}

// |v|^2 per point (bench-mode kinetic-energy diagnostic)
__global__ void kernel_energy_sq(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* out, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v1=V1[idx], v2=V2[idx], v3=V3[idx];
    out[idx]=v1*v1+v2*v2+v3*v3;
}

// ============================================================
// CUDA kernels for spectral-space operations
// heFFTe R2C along dim=2 (z): kz = gk ≥ 0 always
// Wavenumber folding: kx = gi≤NX/2 ? gi : gi-NX,  ky = gj≤NY/2 ? gj : gj-NY
// ============================================================
static __device__ __forceinline__ double kx_fold(int gi){return gi<=d_NX/2?(double)gi:(double)(gi-d_NX);}
static __device__ __forceinline__ double ky_fold(int gj){return gj<=d_NY/2?(double)gj:(double)(gj-d_NY);}

// Kernel A: rot = ik × V and rhs = -k² V in one pass (outbox layout, x fastest)
__global__ void kernel_rot_and_viscous(const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GCplx* R1, GCplx* R2, GCplx* R3, GCplx* H1, GCplx* H2, GCplx* H3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0),ky=ky_fold(lj+lo1),kz=(double)(lk+lo2);
    double k2=kx*kx+ky*ky+kz*kz;
    GCplx v1=V1[idx],v2=V2[idx],v3=V3[idx];
    R1[idx].x=-(ky*v3.y-kz*v2.y); R1[idx].y=ky*v3.x-kz*v2.x;
    R2[idx].x=-(kz*v1.y-kx*v3.y); R2[idx].y=kz*v1.x-kx*v3.x;
    R3[idx].x=-(kx*v2.y-ky*v1.y); R3[idx].y=kx*v2.x-ky*v1.x;
    H1[idx].x=-k2*v1.x; H1[idx].y=-k2*v1.y;
    H2[idx].x=-k2*v2.x; H2[idx].y=-k2*v2.y;
    H3[idx].x=-k2*v3.x; H3[idx].y=-k2*v3.y;
}

enum { ACC_INIT = 0, ACC_ADD = 1, ACC_KEEP = 2 };

// Kernel B: end of every RK4 stage (classic RK4, accumulated form), k in registers only.
// All spectral data are non-normalised DFTs (forward scale::none).
//   k = P[ nl + (-k^2 V + f) ]     (f only in verify, aliases V; sum order as before)
//   ACC_INIT: acc = base + a_acc*k;  ACC_ADD: acc += a_acc*k;  ACC_KEEP: acc unchanged
//   V = base + a_v*k
// Stage 1: base=V_orig INIT tau/6, a_v=tau/2 | 2: V_orig ADD tau/3, tau/2 |
// Stage 3: V_orig ADD tau/3, tau | 4: base=acc KEEP, a_v=tau/6.  k²=0 is left unprojected.
__global__ void kernel_rk4_stage(GCplx* V1, GCplx* V2, GCplx* V3,
    const GCplx* b1, const GCplx* b2, const GCplx* b3,
    GCplx* acc1, GCplx* acc2, GCplx* acc3,
    const GCplx* nl1, const GCplx* nl2, const GCplx* nl3,
    const GCplx* h1, const GCplx* h2, const GCplx* h3,
    const GCplx* f1, const GCplx* f2, const GCplx* f3,
    double a_acc, double a_v, int acc_mode,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0),ky=ky_fold(lj+lo1),kz=(double)(lk+lo2);
    double k2=kx*kx+ky*ky+kz*kz;
    double s1x=h1[idx].x, s1y=h1[idx].y, s2x=h2[idx].x, s2y=h2[idx].y, s3x=h3[idx].x, s3y=h3[idx].y;
    if(f1){ s1x+=f1[idx].x; s1y+=f1[idx].y; s2x+=f2[idx].x; s2y+=f2[idx].y; s3x+=f3[idx].x; s3y+=f3[idx].y; }
    double k1x=nl1[idx].x+s1x, k1y=nl1[idx].y+s1y;
    double k2x=nl2[idx].x+s2x, k2y=nl2[idx].y+s2y;
    double k3x=nl3[idx].x+s3x, k3y=nl3[idx].y+s3y;
    if(k2>=1e-10){
        double dr=-(kx*k1y+ky*k2y+kz*k3y), di=kx*k1x+ky*k2x+kz*k3x;
        double pr=dr/(-k2), pi=di/(-k2);
        k1x-=-kx*pi; k1y-=kx*pr;
        k2x-=-ky*pi; k2y-=ky*pr;
        k3x-=-kz*pi; k3y-=kz*pr;
    }
    GCplx c1=b1[idx],c2=b2[idx],c3=b3[idx];
    if(acc_mode==ACC_INIT){
        acc1[idx].x=c1.x+a_acc*k1x; acc1[idx].y=c1.y+a_acc*k1y;
        acc2[idx].x=c2.x+a_acc*k2x; acc2[idx].y=c2.y+a_acc*k2y;
        acc3[idx].x=c3.x+a_acc*k3x; acc3[idx].y=c3.y+a_acc*k3y;
    }else if(acc_mode==ACC_ADD){
        acc1[idx].x+=a_acc*k1x; acc1[idx].y+=a_acc*k1y;
        acc2[idx].x+=a_acc*k2x; acc2[idx].y+=a_acc*k2y;
        acc3[idx].x+=a_acc*k3x; acc3[idx].y+=a_acc*k3y;
    }
    V1[idx].x=c1.x+a_v*k1x; V1[idx].y=c1.y+a_v*k1y;
    V2[idx].x=c2.x+a_v*k2x; V2[idx].y=c2.y+a_v*k2y;
    V3[idx].x=c3.x+a_v*k3x; V3[idx].y=c3.y+a_v*k3y;
}

__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx%s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0), ky=ky_fold(lj+lo1), kz=(double)(lk+lo2);
    double k2=kx*kx+ky*ky+kz*kz;
    if (k2<1e-10) return;
    double dr=-(kx*V1[idx].y+ky*V2[idx].y+kz*V3[idx].y);
    double di=  kx*V1[idx].x+ky*V2[idx].x+kz*V3[idx].x;
    double pr=dr/(-k2), pi=di/(-k2);
    V1[idx].x-= -kx*pi;  V1[idx].y-=  kx*pr;
    V2[idx].x-= -ky*pi;  V2[idx].y-=  ky*pr;
    V3[idx].x-= -kz*pi;  V3[idx].y-=  kz*pr;
}

__global__ void kernel_div_abs(const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GReal* out, int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx%s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0), ky=ky_fold(lj+lo1), kz=(double)(lk+lo2);
    double dr=-(kx*V1[idx].y+ky*V2[idx].y+kz*V3[idx].y);
    double di=  kx*V1[idx].x+ky*V2[idx].x+kz*V3[idx].x;
    out[idx]=sqrt(dr*dr+di*di);
}

// ============================================================
// Structures and helpers
// ============================================================
struct BoxGPU {
    int lo[3], sz[3];
    long long n() const { return (long long)sz[0]*sz[1]*sz[2]; }
    int grid(int block) const { return (int)((n()+block-1)/block); }
};

static BoxGPU make_box(const heffte::box3d<>& b) {
    BoxGPU bg;
    for (int d=0;d<3;d++) { bg.lo[d]=b.low[d]; bg.sz[d]=b.size[d]; }
    return bg;
}

// Thin wrappers to call heFFTe forward/backward with cufftDoubleComplex* cast to std::complex<double>*.
// They take a persistent workspace, so heFFTe does not cudaMalloc/cudaFree one per FFT call
// (same scheme as the mgpu/multigpu versions).
template<typename FFT>
static void heffte_fwd(FFT& fft, GReal* d_r, GCplx* d_c, std::complex<double>* ws) {
    fft.forward(d_r, reinterpret_cast<std::complex<double>*>(d_c), ws, heffte::scale::none);
}
template<typename FFT>
static void heffte_bwd(FFT& fft, GCplx* d_c, GReal* d_r, std::complex<double>* ws) {
    fft.backward(reinterpret_cast<std::complex<double>*>(d_c), d_r, ws, heffte::scale::full);
}

// ============================================================
// GPU array bundle
// ============================================================
struct GPUArrays {
    GReal *V1_r, *V2_r, *V3_r;          // real velocity; also holds the forcing f in verify mode
    GReal *rot1_r, *rot2_r, *rot3_r;    // real vorticity, overwritten by the cross product
    GCplx *V1_c, *V2_c, *V3_c;          // stage value (spectral)
    GCplx *rot1_c, *rot2_c, *rot3_c;    // vorticity; after the forward FFT: nl
    GCplx *rhs1_c, *rhs2_c, *rhs3_c;    // -k^2 V (viscous term)
    GCplx *orig1_c, *orig2_c, *orig3_c; // V^n
    GCplx *acc1_c, *acc2_c, *acc3_c;    // RK4 accumulator
    GReal *scratch;                     // diagnostics only; shares rhs1_c's memory when it fits
    bool scratch_own = false;
};

static void alloc_arrays(GPUArrays& g, long long nr, long long nc){
    gm(&g.V1_r,nr);   gm(&g.V2_r,nr);   gm(&g.V3_r,nr);
    gm(&g.rot1_r,nr); gm(&g.rot2_r,nr); gm(&g.rot3_r,nr);
    gm(&g.V1_c,nc);   gm(&g.V2_c,nc);   gm(&g.V3_c,nc);
    gm(&g.rot1_c,nc); gm(&g.rot2_c,nc); gm(&g.rot3_c,nc);
    gm(&g.rhs1_c,nc); gm(&g.rhs2_c,nc); gm(&g.rhs3_c,nc);
    gm(&g.orig1_c,nc);gm(&g.orig2_c,nc);gm(&g.orig3_c,nc);
    gm(&g.acc1_c,nc); gm(&g.acc2_c,nc); gm(&g.acc3_c,nc);
    // Diagnostics run after the timed loop, when rhs is no longer needed: reuse rhs1
    // (2*nc doubles) if it holds max(nr,nc) doubles, otherwise allocate a separate scratch.
    if (2*nc >= max(nr,nc)) g.scratch = (GReal*)g.rhs1_c;
    else { gm(&g.scratch, max(nr,nc)); g.scratch_own = true; }
}

static void free_arrays(GPUArrays& g){
    cudaFree(g.V1_r);   cudaFree(g.V2_r);   cudaFree(g.V3_r);
    cudaFree(g.rot1_r); cudaFree(g.rot2_r); cudaFree(g.rot3_r);
    cudaFree(g.V1_c);   cudaFree(g.V2_c);   cudaFree(g.V3_c);
    cudaFree(g.rot1_c); cudaFree(g.rot2_c); cudaFree(g.rot3_c);
    cudaFree(g.rhs1_c); cudaFree(g.rhs2_c); cudaFree(g.rhs3_c);
    cudaFree(g.orig1_c);cudaFree(g.orig2_c);cudaFree(g.orig3_c);
    cudaFree(g.acc1_c); cudaFree(g.acc2_c); cudaFree(g.acc3_c);
    if (g.scratch_own) cudaFree(g.scratch);
}

// ============================================================
// Nonlinear term: V × rot(V) in spectral space (pseudospectral)
//
// heFFTe backward does NOT destroy complex input → no save/restore needed.
// Output nl1_c/nl2_c/nl3_c use scale::none (same DFT convention as V_c).
// ============================================================
// One RK4 stage (9 FFTs, +3 for the forcing in verify mode); heFFTe backward keeps its input.
template<typename FFT>
static void rk4_stage(FFT& fft, GPUArrays& g, const BoxGPU& br, const BoxGPU& bc,
    double t, int stage, std::complex<double>* ws)
{
    int gc=bc.grid(BLOCK), gr=br.grid(BLOCK);
    // 1. Kernel A: rot = ik × V, rhs = -k² V
    kernel_rot_and_viscous<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.rot1_c,g.rot2_c,g.rot3_c,
        g.rhs1_c,g.rhs2_c,g.rhs3_c, bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    g_timer.mark(PH_SPEC);
    // 2. Inverse FFT: V, rot → real space (scale::full: physical values)
    heffte_bwd(fft,g.V1_c,g.V1_r,ws);     heffte_bwd(fft,g.V2_c,g.V2_r,ws);     heffte_bwd(fft,g.V3_c,g.V3_r,ws);
    heffte_bwd(fft,g.rot1_c,g.rot1_r,ws); heffte_bwd(fft,g.rot2_c,g.rot2_r,ws); heffte_bwd(fft,g.rot3_c,g.rot3_r,ws);
    g_timer.mark(PH_FFT);
    // 3. Cross product V × rot → rot_r
    kernel_cross_product<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r, g.rot1_r,g.rot2_r,g.rot3_r, br.n());
    g_timer.mark(PH_PHYS);
    // 4. Forward FFT: rot_r → rot_c (nl, scale::none)
    heffte_fwd(fft,g.rot1_r,g.rot1_c,ws); heffte_fwd(fft,g.rot2_r,g.rot2_c,ws); heffte_fwd(fft,g.rot3_r,g.rot3_c,ws);
    g_timer.mark(PH_FFT);
    // 5. Forcing [verify only]: fill the free V_r, forward FFT into V_c
    if (g_with_force) {
        kernel_fill_forcing<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,
            br.lo[0],br.lo[1],br.lo[2], br.sz[0],br.sz[1],br.sz[2], br.n(), t);
        heffte_fwd(fft,g.V1_r,g.V1_c,ws); heffte_fwd(fft,g.V2_r,g.V2_c,ws); heffte_fwd(fft,g.V3_r,g.V3_c,ws);
        g_timer.mark(PH_FORCE);
    }
    // 6. Kernel B: sum + projection + RK4 update → V_c (next stage value)
    const GCplx *b1=g.orig1_c, *b2=g.orig2_c, *b3=g.orig3_c;
    double a_acc=0., a_v=0.; int acc_mode=ACC_ADD;
    switch (stage) {
        case 1:  acc_mode=ACC_INIT; a_acc=TAU/6.; a_v=0.5*TAU; break;
        case 2:                     a_acc=TAU/3.; a_v=0.5*TAU; break;
        case 3:                     a_acc=TAU/3.; a_v=TAU;     break;
        default: acc_mode=ACC_KEEP;               a_v=TAU/6.;
                 b1=g.acc1_c; b2=g.acc2_c; b3=g.acc3_c;        break;
    }
    kernel_rk4_stage<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, b1,b2,b3, g.acc1_c,g.acc2_c,g.acc3_c,
        g.rot1_c,g.rot2_c,g.rot3_c, g.rhs1_c,g.rhs2_c,g.rhs3_c,
        g_with_force ? g.V1_c : nullptr, g_with_force ? g.V2_c : nullptr, g_with_force ? g.V3_c : nullptr,
        a_acc, a_v, acc_mode, bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    g_timer.mark(PH_SPEC);
}

// RK4 time step (spectral space, non-normalized convention)
template<typename FFT>
static void rk4_step(FFT& fft, GPUArrays& g, const BoxGPU& br, const BoxGPU& bc, double t,
    std::complex<double>* ws)
{
    long long nc=bc.n();
    CUDA_CHECK(cudaMemcpy(g.orig1_c,g.V1_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.orig2_c,g.V2_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.orig3_c,g.V3_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    g_timer.mark(PH_SPEC);
    rk4_stage(fft,g,br,bc,t,         1,ws);
    rk4_stage(fft,g,br,bc,t+0.5*TAU, 2,ws);
    rk4_stage(fft,g,br,bc,t+0.5*TAU, 3,ws);
    rk4_stage(fft,g,br,bc,t+TAU,     4,ws);
}

template<typename FFT>
static pair<double,double> compute_diagnostics(FFT& fft, GPUArrays& g,
    const BoxGPU& br, const BoxGPU& bc, double t, std::complex<double>* ws, bool verify)
{
    int gc = bc.grid(BLOCK), gr = br.grid(BLOCK);
    // V_c → V_r (backward, does not destroy V_c)
    heffte_bwd(fft, g.V1_c, g.V1_r, ws);
    heffte_bwd(fft, g.V2_c, g.V2_r, ws);
    heffte_bwd(fft, g.V3_c, g.V3_r, ws);
    // L2 error (verify) or |v|^2 (bench)
    if (verify)
        kernel_error_sq<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r, g.scratch,
            br.lo[0],br.lo[1],br.lo[2], br.sz[0],br.sz[1],br.sz[2], br.n(), t);
    else
        kernel_energy_sq<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r, g.scratch, br.n());
    thrust::device_ptr<double> sp(g.scratch);
    double local_err = thrust::reduce(thrust::device, sp, sp+br.n());
    double global_err;
    MPI_Reduce(&local_err,&global_err,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    // max|div V|
    kernel_div_abs<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.scratch,
        bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    double local_div = thrust::reduce(thrust::device, sp, sp+bc.n(), 0., thrust::maximum<double>());
    double global_div;
    MPI_Reduce(&local_div,&global_div,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    // V_c is a non-normalised DFT (= N x normalised coefficient): divide by N so max|div V| is comparable
    return {verify ? sqrt(global_err*DX*DY*DZ) : 0.5*global_err, global_div/((double)NX*NY*NZ)};
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 6) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s %s\n", argv[0], nsmode::usage_tail(false));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    nsmode::RunArgs rargs;
    {
        std::string perr;
        if (!nsmode::parse_tail_args(argc, argv, 6, /*allow_threads=*/false, rargs, perr)) {
            if (rank == 0)
                fprintf(stderr, "Error: %s\nUsage: %s %s\n", perr.c_str(), argv[0], nsmode::usage_tail(false));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const nsmode::RunMode mode = rargs.mode;
    g_with_force = (mode == nsmode::MODE_VERIFY);
    NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]);
    TAU = atof(argv[4]); NT_RUN = atoi(argv[5]);
    NT_TOTAL = NT_RUN;
    LX = LY = LZ = 2.0 * M_PI;
    DX = LX / NX; DY = LY / NY; DZ = LZ / NZ;

    cudaMemcpyToSymbol(d_NX, &NX, sizeof(int));
    cudaMemcpyToSymbol(d_NY, &NY, sizeof(int));
    cudaMemcpyToSymbol(d_DX, &DX, sizeof(double));
    cudaMemcpyToSymbol(d_DY, &DY, sizeof(double));
    cudaMemcpyToSymbol(d_DZ, &DZ, sizeof(double));

    if (rank == 0)
        printf("Grid: %d x %d x %d, dt=%.2e, steps=%d\n", NX, NY, NZ, TAU, NT_RUN);

    // Single GPU
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (rank==0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes — heFFTe cuFFT Backend, Single GPU\n";
        cout << "============================================================\n";
        cout << "GPU: " << prop.name << "\n";
        cout << "MPI ranks: " << nprocs << "  GPU: " << prop.name << "\n";
        cout << "Run mode: " << nsmode::mode_name(mode)
             << (mode == nsmode::MODE_BENCH ? "  (f = 0, 36 FFT/step)" : "  (analytic f, 48 FFT/step)") << "\n";
        cout << "============================================================\n";
    }

    // ---- heFFTe setup ----
    {
    heffte::box3d<> world_r = {{0,0,0},{NX-1,NY-1,NZ-1}};
    heffte::box3d<> world_c = {{0,0,0},{NX-1,NY-1,NZ/2}};   // R2C along z → nz/2+1 modes
    auto pg = heffte::proc_setup_min_surface(world_r, nprocs);
    auto inboxes  = heffte::split_world(world_r, pg);
    auto outboxes = heffte::split_world(world_c, pg);
    heffte::box3d<> inbox_r  = inboxes[rank];
    heffte::box3d<> outbox_c = outboxes[rank];

    if (rank==0) cout << "heFFTe pencil grid: " << pg[0] << "x" << pg[1] << "x" << pg[2] << "\n";

    heffte::fft3d_r2c<heffte::backend::cufft> fft(inbox_r, outbox_c, 2, MPI_COMM_WORLD);

    BoxGPU br = make_box(inbox_r);
    BoxGPU bc = make_box(outbox_c);
    long long nr = br.n(), nc = bc.n();

    // ---- GPU memory ----
    GPUArrays g;
    alloc_arrays(g, nr, nc);

    // Persistent heFFTe workspace: avoids cudaMalloc/cudaFree on every FFT call
    size_t ws_size = fft.size_workspace();
    std::complex<double>* d_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ws, ws_size * sizeof(std::complex<double>)));
    {   // Memory report (per GPU, max over ranks)
        const double GB=1024.*1024.*1024.;
        size_t mf=0, mt=0; CUDA_CHECK(cudaMemGetInfo(&mf,&mt));
        double in[4]={ (double)(nc*sizeof(GCplx)), (double)(nr*sizeof(GReal)),
                       (double)(ws_size*sizeof(std::complex<double>)), (double)(mt-mf) }, mx[4];
        MPI_Allreduce(in,mx,4,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if (rank==0) {
            printf("Memory report (per GPU, max over ranks):\n");
            printf("  Full-size arrays: 15 complex of %.3f GB, 6 real of %.3f GB%s\n", mx[0]/GB, mx[1]/GB,
                   g.scratch_own ? " (+ separate diagnostics scratch)" : "");
            printf("  heFFTe workspace: %.3f GB\n", mx[2]/GB);
            printf("  GPU memory in use (max over ranks, cudaMemGetInfo): %.3f GB of %.3f GB\n", mx[3]/GB, mt/GB);
            fflush(stdout);
        }
    }
    if (rank==0) printf("  heFFTe workspace per rank: %.0f MB\n",
                        ws_size * sizeof(std::complex<double>) / (1024.*1024.));

    // ---- Initial condition ----
    {
    int gr = br.grid(BLOCK);
    kernel_fill_velocity<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,
        br.lo[0],br.lo[1],br.lo[2], br.sz[0],br.sz[1],br.sz[2], nr);
    }
    heffte_fwd(fft, g.V1_r, g.V1_c, d_ws);
    heffte_fwd(fft, g.V2_r, g.V2_c, d_ws);
    heffte_fwd(fft, g.V3_r, g.V3_c, d_ws);
    // Project initial condition to divergence-free space
    {
    int gc = bc.grid(BLOCK);
    kernel_make_div_free<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], nc);
    }

    double t_wall = 0.;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    g_timer.init();

    for (int it=0; it<NT_RUN; ++it) {
        double tc = it*TAU;
        CUDA_CHECK(cudaEventRecord(ev0));
        g_timer.start();
        rk4_step(fft, g, br, bc, tc, d_ws);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        g_timer.collect();
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,ev0,ev1));
        t_wall += (double)ms*1e-3;
    }
    double phase_max[nsmode::PH_COUNT];
    nsmode::reduce_phases_max(g_timer.acc, phase_max);
    g_timer.destroy();

    // Diagnostics (not timed)
    {
        double t_final = NT_RUN * TAU;
        const bool verify = (mode == nsmode::MODE_VERIFY);
        auto [e, d] = compute_diagnostics(fft, g, br, bc, t_final, d_ws, verify);
        if (rank==0) {
            if (verify)
                cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(4) << e << "\n";
            else
                cout << "  Kinetic energy 0.5*sum|v|^2 (t=" << fixed << setprecision(6) << t_final << "): "
                     << scientific << setprecision(10) << e << "\n";
            cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << d << "\n";
        }
    }
    if (rank==0) {
        cout << "============================================================\n";
        cout << "  Total steps:     " << NT_RUN << "\n";
        cout << "  Total wall time: " << fixed<<setprecision(4)<<t_wall << " s\n";
        cout << "  Avg per step:    " << t_wall/NT_RUN << " s\n" << flush;
        nsmode::print_phase_summary(phase_max, t_wall, NT_RUN, mode);
        cout << "============================================================\n";
    }

    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
    free_arrays(g);
    cudaFree(d_ws);
    }  // heFFTe fft destructs before MPI_Finalize

    MPI_Finalize();
    return 0;
}
