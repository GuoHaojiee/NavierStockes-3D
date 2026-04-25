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

using namespace std;

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

// ============================================================
// CUDA kernels for spectral-space operations
// heFFTe R2C along dim=2 (z): kz = gk ≥ 0 always
// Wavenumber folding: kx = gi≤NX/2 ? gi : gi-NX,  ky = gj≤NY/2 ? gj : gj-NY
// ============================================================
static __device__ __forceinline__ double kx_fold(int gi){return gi<=d_NX/2?(double)gi:(double)(gi-d_NX);}
static __device__ __forceinline__ double ky_fold(int gj){return gj<=d_NY/2?(double)gj:(double)(gj-d_NY);}

__global__ void kernel_compute_rot(
    const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GCplx* R1, GCplx* R2, GCplx* R3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx%s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0), ky=ky_fold(lj+lo1), kz=(double)(lk+lo2);
    // rot_1 = i(ky·V3 - kz·V2): real part = -(ky·V3.y - kz·V2.y)
    R1[idx].x = -(ky*V3[idx].y - kz*V2[idx].y);
    R1[idx].y =   ky*V3[idx].x - kz*V2[idx].x;
    R2[idx].x = -(kz*V1[idx].y - kx*V3[idx].y);
    R2[idx].y =   kz*V1[idx].x - kx*V3[idx].x;
    R3[idx].x = -(kx*V2[idx].y - ky*V1[idx].y);
    R3[idx].y =   kx*V2[idx].x - ky*V1[idx].x;
}

__global__ void kernel_compute_viscous(const GCplx* V, GCplx* visc,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int li=(int)(idx%s0), lj=(int)((idx/s0)%s1), lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0), ky=ky_fold(lj+lo1), kz=(double)(lk+lo2);
    double k2=kx*kx+ky*ky+kz*kz;
    visc[idx].x = -k2*V[idx].x;
    visc[idx].y = -k2*V[idx].y;
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

__global__ void kernel_add_rhs(GCplx* rhs1, GCplx* rhs2, GCplx* rhs3,
    const GCplx* visc1, const GCplx* visc2, const GCplx* visc3,
    const GCplx* f1, const GCplx* f2, const GCplx* f3, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    rhs1[idx].x+=visc1[idx].x+f1[idx].x; rhs1[idx].y+=visc1[idx].y+f1[idx].y;
    rhs2[idx].x+=visc2[idx].x+f2[idx].x; rhs2[idx].y+=visc2[idx].y+f2[idx].y;
    rhs3[idx].x+=visc3[idx].x+f3[idx].x; rhs3[idx].y+=visc3[idx].y+f3[idx].y;
}

__global__ void kernel_rk4_axpy(GCplx* V1, GCplx* V2, GCplx* V3,
    const GCplx* o1, const GCplx* o2, const GCplx* o3,
    const GCplx* k1, const GCplx* k2, const GCplx* k3,
    double alpha, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    V1[idx].x=o1[idx].x+alpha*k1[idx].x; V1[idx].y=o1[idx].y+alpha*k1[idx].y;
    V2[idx].x=o2[idx].x+alpha*k2[idx].x; V2[idx].y=o2[idx].y+alpha*k2[idx].y;
    V3[idx].x=o3[idx].x+alpha*k3[idx].x; V3[idx].y=o3[idx].y+alpha*k3[idx].y;
}

__global__ void kernel_rk4_update(GCplx* V1, GCplx* V2, GCplx* V3,
    const GCplx* o1, const GCplx* o2, const GCplx* o3,
    const GCplx* k1v1,const GCplx* k2v1,const GCplx* k3v1,const GCplx* k4v1,
    const GCplx* k1v2,const GCplx* k2v2,const GCplx* k3v2,const GCplx* k4v2,
    const GCplx* k1v3,const GCplx* k2v3,const GCplx* k3v3,const GCplx* k4v3,
    double dtd6, long long n)
{
    long long idx = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    V1[idx].x=o1[idx].x+dtd6*(k1v1[idx].x+2.*k2v1[idx].x+2.*k3v1[idx].x+k4v1[idx].x);
    V1[idx].y=o1[idx].y+dtd6*(k1v1[idx].y+2.*k2v1[idx].y+2.*k3v1[idx].y+k4v1[idx].y);
    V2[idx].x=o2[idx].x+dtd6*(k1v2[idx].x+2.*k2v2[idx].x+2.*k3v2[idx].x+k4v2[idx].x);
    V2[idx].y=o2[idx].y+dtd6*(k1v2[idx].y+2.*k2v2[idx].y+2.*k3v2[idx].y+k4v2[idx].y);
    V3[idx].x=o3[idx].x+dtd6*(k1v3[idx].x+2.*k2v3[idx].x+2.*k3v3[idx].x+k4v3[idx].x);
    V3[idx].y=o3[idx].y+dtd6*(k1v3[idx].y+2.*k2v3[idx].y+2.*k3v3[idx].y+k4v3[idx].y);
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

// Thin wrapper to call heFFTe forward/backward with cufftDoubleComplex* cast to std::complex<double>*
template<typename FFT>
static void heffte_fwd(FFT& fft, GReal* d_r, GCplx* d_c) {
    fft.forward(d_r, reinterpret_cast<std::complex<double>*>(d_c), heffte::scale::none);
}
template<typename FFT>
static void heffte_bwd(FFT& fft, GCplx* d_c, GReal* d_r) {
    fft.backward(reinterpret_cast<std::complex<double>*>(d_c), d_r, heffte::scale::full);
}

// ============================================================
// GPU array bundle
// ============================================================
struct GPUArrays {
    GReal *V1_r, *V2_r, *V3_r;
    GCplx *V1_c, *V2_c, *V3_c;
    GReal *rot1_r, *rot2_r, *rot3_r;
    GCplx *rot1_c, *rot2_c, *rot3_c;
    GCplx *visc1_c, *visc2_c, *visc3_c;
    GReal *work1_r, *work2_r, *work3_r;
    GCplx *f1_c, *f2_c, *f3_c;
    GCplx *k1v1,*k1v2,*k1v3, *k2v1,*k2v2,*k2v3;
    GCplx *k3v1,*k3v2,*k3v3, *k4v1,*k4v2,*k4v3;
    GCplx *tmp1_c, *tmp2_c, *tmp3_c;
    GCplx *div_c;
    GReal *scratch;
};

static void alloc_arrays(GPUArrays& g, long long nr, long long nc) {
    gm(&g.V1_r,nr);   gm(&g.V2_r,nr);   gm(&g.V3_r,nr);
    gm(&g.rot1_r,nr); gm(&g.rot2_r,nr); gm(&g.rot3_r,nr);
    gm(&g.work1_r,nr);gm(&g.work2_r,nr);gm(&g.work3_r,nr);
    gm(&g.scratch, max(nr,nc));
    gm(&g.V1_c,nc);   gm(&g.V2_c,nc);   gm(&g.V3_c,nc);
    gm(&g.rot1_c,nc); gm(&g.rot2_c,nc); gm(&g.rot3_c,nc);
    gm(&g.visc1_c,nc);gm(&g.visc2_c,nc);gm(&g.visc3_c,nc);
    gm(&g.f1_c,nc);   gm(&g.f2_c,nc);   gm(&g.f3_c,nc);
    gm(&g.k1v1,nc);gm(&g.k1v2,nc);gm(&g.k1v3,nc);
    gm(&g.k2v1,nc);gm(&g.k2v2,nc);gm(&g.k2v3,nc);
    gm(&g.k3v1,nc);gm(&g.k3v2,nc);gm(&g.k3v3,nc);
    gm(&g.k4v1,nc);gm(&g.k4v2,nc);gm(&g.k4v3,nc);
    gm(&g.tmp1_c,nc);gm(&g.tmp2_c,nc);gm(&g.tmp3_c,nc);
    gm(&g.div_c,nc);
}

static void free_arrays(GPUArrays& g) {
    cudaFree(g.V1_r);   cudaFree(g.V2_r);   cudaFree(g.V3_r);
    cudaFree(g.rot1_r); cudaFree(g.rot2_r); cudaFree(g.rot3_r);
    cudaFree(g.work1_r);cudaFree(g.work2_r);cudaFree(g.work3_r);
    cudaFree(g.scratch);
    cudaFree(g.V1_c);   cudaFree(g.V2_c);   cudaFree(g.V3_c);
    cudaFree(g.rot1_c); cudaFree(g.rot2_c); cudaFree(g.rot3_c);
    cudaFree(g.visc1_c);cudaFree(g.visc2_c);cudaFree(g.visc3_c);
    cudaFree(g.f1_c);   cudaFree(g.f2_c);   cudaFree(g.f3_c);
    cudaFree(g.k1v1);cudaFree(g.k1v2);cudaFree(g.k1v3);
    cudaFree(g.k2v1);cudaFree(g.k2v2);cudaFree(g.k2v3);
    cudaFree(g.k3v1);cudaFree(g.k3v2);cudaFree(g.k3v3);
    cudaFree(g.k4v1);cudaFree(g.k4v2);cudaFree(g.k4v3);
    cudaFree(g.tmp1_c);cudaFree(g.tmp2_c);cudaFree(g.tmp3_c);
    cudaFree(g.div_c);
}

// ============================================================
// Nonlinear term: V × rot(V) in spectral space (pseudospectral)
//
// heFFTe backward does NOT destroy complex input → no save/restore needed.
// Output nl1_c/nl2_c/nl3_c use scale::none (same DFT convention as V_c).
// ============================================================
template<typename FFT>
static void compute_nonlinear(FFT& fft, GPUArrays& g,
    GCplx* nl1, GCplx* nl2, GCplx* nl3,
    const BoxGPU& br, const BoxGPU& bc)
{
    int gc = bc.grid(BLOCK), gr = br.grid(BLOCK);
    // 1. Spectral vorticity: rot_c = ik × V_c
    kernel_compute_rot<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.rot1_c,g.rot2_c,g.rot3_c,
        bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    // 2. Inverse FFT: V_c → V_r and rot_c → rot_r (heFFTe backward, input preserved)
    heffte_bwd(fft, g.V1_c, g.V1_r); heffte_bwd(fft, g.V2_c, g.V2_r); heffte_bwd(fft, g.V3_c, g.V3_r);
    heffte_bwd(fft, g.rot1_c, g.rot1_r); heffte_bwd(fft, g.rot2_c, g.rot2_r); heffte_bwd(fft, g.rot3_c, g.rot3_r);
    // 3. Real-space cross product: V × rot → rot_r
    kernel_cross_product<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r, g.rot1_r,g.rot2_r,g.rot3_r, br.n());
    // 4. Forward FFT: rot_r → nl_c (scale::none, same DFT convention as V_c)
    heffte_fwd(fft, g.rot1_r, nl1); heffte_fwd(fft, g.rot2_r, nl2); heffte_fwd(fft, g.rot3_r, nl3);
}

// ============================================================
// Full RHS: nl + visc + f, projected to divergence-free space
// ============================================================
template<typename FFT>
static void compute_rhs(FFT& fft, GPUArrays& g,
    GCplx* rhs1, GCplx* rhs2, GCplx* rhs3,
    const BoxGPU& br, const BoxGPU& bc, double t)
{
    int gc = bc.grid(BLOCK), gr = br.grid(BLOCK);
    // 1. Viscous term (reads V_c, safe since backward doesn't destroy input)
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V1_c,g.visc1_c, bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V2_c,g.visc2_c, bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V3_c,g.visc3_c, bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
    // 2. Nonlinear term → rhs1/2/3 (used as nl workspace)
    compute_nonlinear(fft, g, rhs1, rhs2, rhs3, br, bc);
    // 3. Forcing term: fill real array, forward FFT
    kernel_fill_forcing<<<gr,BLOCK>>>(g.work1_r,g.work2_r,g.work3_r,
        br.lo[0],br.lo[1],br.lo[2], br.sz[0],br.sz[1],br.sz[2], br.n(), t);
    heffte_fwd(fft, g.work1_r, g.f1_c); heffte_fwd(fft, g.work2_r, g.f2_c); heffte_fwd(fft, g.work3_r, g.f3_c);
    // 4. Combine: rhs += visc + f
    kernel_add_rhs<<<gc,BLOCK>>>(rhs1,rhs2,rhs3, g.visc1_c,g.visc2_c,g.visc3_c, g.f1_c,g.f2_c,g.f3_c, bc.n());
    // 5. Project: make rhs divergence-free
    kernel_make_div_free<<<gc,BLOCK>>>(rhs1,rhs2,rhs3,
        bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], bc.n());
}

// ============================================================
// RK4 time step (spectral space, non-normalized convention)
// ============================================================
template<typename FFT>
static void rk4_step(FFT& fft, GPUArrays& g,
    const BoxGPU& br, const BoxGPU& bc, double t)
{
    long long nc = bc.n();
    int gc = bc.grid(BLOCK);
    // Save V^n
    CUDA_CHECK(cudaMemcpy(g.tmp1_c,g.V1_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.tmp2_c,g.V2_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.tmp3_c,g.V3_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    // k1
    compute_rhs(fft,g, g.k1v1,g.k1v2,g.k1v3, br,bc, t);
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.tmp1_c,g.tmp2_c,g.tmp3_c,
        g.k1v1,g.k1v2,g.k1v3, 0.5*TAU, nc);
    // k2
    compute_rhs(fft,g, g.k2v1,g.k2v2,g.k2v3, br,bc, t+0.5*TAU);
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.tmp1_c,g.tmp2_c,g.tmp3_c,
        g.k2v1,g.k2v2,g.k2v3, 0.5*TAU, nc);
    // k3
    compute_rhs(fft,g, g.k3v1,g.k3v2,g.k3v3, br,bc, t+0.5*TAU);
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.tmp1_c,g.tmp2_c,g.tmp3_c,
        g.k3v1,g.k3v2,g.k3v3, TAU, nc);
    // k4
    compute_rhs(fft,g, g.k4v1,g.k4v2,g.k4v3, br,bc, t+TAU);
    // V^{n+1}
    kernel_rk4_update<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        g.tmp1_c,g.tmp2_c,g.tmp3_c,
        g.k1v1,g.k2v1,g.k3v1,g.k4v1,
        g.k1v2,g.k2v2,g.k3v2,g.k4v2,
        g.k1v3,g.k2v3,g.k3v3,g.k4v3,
        TAU/6., nc);
}

// ============================================================
// Diagnostics: L2 error and max|div V|
// ============================================================
template<typename FFT>
static pair<double,double> compute_diagnostics(FFT& fft, GPUArrays& g,
    const BoxGPU& br, const BoxGPU& bc, double t)
{
    int gc = bc.grid(BLOCK), gr = br.grid(BLOCK);
    // V_c → V_r (backward, does not destroy V_c)
    heffte_bwd(fft, g.V1_c, g.V1_r);
    heffte_bwd(fft, g.V2_c, g.V2_r);
    heffte_bwd(fft, g.V3_c, g.V3_r);
    // L2 error
    kernel_error_sq<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r, g.scratch,
        br.lo[0],br.lo[1],br.lo[2], br.sz[0],br.sz[1],br.sz[2], br.n(), t);
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
    return {sqrt(global_err*DX*DY*DZ), global_div};
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
            fprintf(stderr, "Usage: %s NX NY NZ dt NSTEPS\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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
        cout << "Grid: " << NX << "x" << NY << "x" << NZ << "  MPI ranks: " << nprocs << "\n";
        cout << "Steps: " << NT_RUN << "/" << NT_TOTAL << "  dt=" << TAU << "\n";
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

    // ---- Initial condition ----
    {
    int gr = br.grid(BLOCK);
    kernel_fill_velocity<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,
        br.lo[0],br.lo[1],br.lo[2], br.sz[0],br.sz[1],br.sz[2], nr);
    }
    heffte_fwd(fft, g.V1_r, g.V1_c);
    heffte_fwd(fft, g.V2_r, g.V2_c);
    heffte_fwd(fft, g.V3_r, g.V3_c);
    // Project initial condition to divergence-free space
    {
    int gc = bc.grid(BLOCK);
    kernel_make_div_free<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        bc.lo[0],bc.lo[1],bc.lo[2], bc.sz[0],bc.sz[1],bc.sz[2], nc);
    }

    double t_wall = 0.;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));

    for (int it=0; it<NT_RUN; ++it) {
        double tc = it*TAU;
        CUDA_CHECK(cudaEventRecord(ev0));
        rk4_step(fft, g, br, bc, tc);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,ev0,ev1));
        t_wall += (double)ms*1e-3;
    }
    {
        double t_final = NT_RUN * TAU;
        auto [e, d] = compute_diagnostics(fft, g, br, bc, t_final);
        if (rank==0)
            cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << e << "\n";
    }
    if (rank==0) {
        cout << "============================================================\n";
        cout << "  Total wall time: " << fixed<<setprecision(4)<<t_wall << " s\n";
        cout << "  Avg per step:    " << t_wall/NT_RUN << " s\n";
        cout << "============================================================\n";
    }

    CUDA_CHECK(cudaEventDestroy(ev0)); CUDA_CHECK(cudaEventDestroy(ev1));
    free_arrays(g);
    }  // heFFTe fft destructs before MPI_Finalize

    MPI_Finalize();
    return 0;
}
