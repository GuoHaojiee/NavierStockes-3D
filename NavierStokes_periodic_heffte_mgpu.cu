/**
 * NavierStokes_periodic_heffte_mgpu.cu
 * Navier-Stokes solver — heFFTe cufft backend, single node, multi-GPU
 *
 * Each MPI rank controls one GPU on the same node.
 * GPU assignment: cudaSetDevice(rank % num_local_gpus)
 *
 * heFFTe automatically distributes work via 2D pencil decomposition across all
 * ranks/GPUs. Device-to-device MPI communication is handled internally by heFFTe
 * (requires CUDA-aware MPI, e.g., OpenMPI built with --with-cuda).
 *
 * Run: mpirun -np <ngpu> ./navier_stokes_heffte_mgpu NX NY NZ dt NSTEPS
 *
 * ── FIXES vs. original ──────────────────────────────────────────────────────
 * Bug 1 [Correctness]: cudaMemcpyToSymbol() was called BEFORE cudaSetDevice().
 *   All MPI processes start with GPU 0 as the default device, so ranks ≥ 1
 *   were writing constant memory to GPU 0 instead of their assigned GPU.
 *   Kernels on GPU 1,2,… then read uninitialised d_NX/d_NY/d_DX/d_DY/d_DZ.
 *   Fix: call cudaMemcpyToSymbol() AFTER cudaSetDevice().
 *
 * Bug 2 [OOM]: 28 complex device arrays were allocated per rank (12 for all
 *   four RK4 ki stages + 3 tmp + 1 unused div_c + 12 physics intermediates).
 *   For a 256³ grid with 2 GPUs each complex array ≈ 64 MB → ~1.8 GB just in
 *   complex buffers, plus real arrays and heFFTe internal workspace easily
 *   pushing into OOM territory on a shared/partially-occupied GPU.
 *   Fix: accumulate the weighted RK4 sum on-the-fly (ksum += w_i * k_i),
 *   reducing time-stepping complex arrays from 15 to 9, and remove the
 *   never-used div_c array.  Total saving: 7 × ~64 MB ≈ 448 MB per rank.
 * ────────────────────────────────────────────────────────────────────────────
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
typedef cufftDoubleComplex GCplx;

// ============================================================
// Grid parameters — host-side (set at runtime from argv)
// ============================================================
static int    NX, NY, NZ, NT_RUN;
static double LX, LY, LZ, DX, DY, DZ, TAU;

__device__ __constant__ int    d_NX, d_NY;
__device__ __constant__ double d_DX, d_DY, d_DZ;

constexpr int BLOCK = 256;

// ============================================================
// Error macros
// ============================================================
#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

template<typename T>
static void gm(T** p, long long n) { CUDA_CHECK(cudaMalloc((void**)p, (size_t)n*sizeof(T))); }

// ============================================================
// Analytical solution (Taylor-Green variant)
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
// CUDA kernels
// ============================================================
__global__ void kernel_fill_velocity(GReal* V1, GReal* V2, GReal* V3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double x=(li+lo0)*d_DX,y=(lj+lo1)*d_DY,z=(lk+lo2)*d_DZ;
    V1[idx]=func_V1(x,y,z,0.);V2[idx]=func_V2(x,y,z,0.);V3[idx]=func_V3(x,y,z,0.);
}
__global__ void kernel_fill_forcing(GReal* W1, GReal* W2, GReal* W3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n, double t)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double x=(li+lo0)*d_DX,y=(lj+lo1)*d_DY,z=(lk+lo2)*d_DZ;
    W1[idx]=func_f1(x,y,z,t);W2[idx]=func_f2(x,y,z,t);W3[idx]=func_f3(x,y,z,t);
}
__global__ void kernel_cross_product(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* R1, GReal* R2, GReal* R3, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    double v1=V1[idx],v2=V2[idx],v3=V3[idx],w1=R1[idx],w2=R2[idx],w3=R3[idx];
    R1[idx]=v2*w3-v3*w2;R2[idx]=v3*w1-v1*w3;R3[idx]=v1*w2-v2*w1;
}
__global__ void kernel_error_sq(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* err, int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n, double t)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double x=(li+lo0)*d_DX,y=(lj+lo1)*d_DY,z=(lk+lo2)*d_DZ;
    double d1=V1[idx]-func_V1(x,y,z,t),d2=V2[idx]-func_V2(x,y,z,t),d3=V3[idx]-func_V3(x,y,z,t);
    err[idx]=d1*d1+d2*d2+d3*d3;
}
static __device__ __forceinline__ double kx_fold(int gi){return gi<=d_NX/2?(double)gi:(double)(gi-d_NX);}
static __device__ __forceinline__ double ky_fold(int gj){return gj<=d_NY/2?(double)gj:(double)(gj-d_NY);}

__global__ void kernel_compute_rot(
    const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GCplx* R1, GCplx* R2, GCplx* R3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0),ky=ky_fold(lj+lo1),kz=(double)(lk+lo2);
    R1[idx].x=-(ky*V3[idx].y-kz*V2[idx].y); R1[idx].y=ky*V3[idx].x-kz*V2[idx].x;
    R2[idx].x=-(kz*V1[idx].y-kx*V3[idx].y); R2[idx].y=kz*V1[idx].x-kx*V3[idx].x;
    R3[idx].x=-(kx*V2[idx].y-ky*V1[idx].y); R3[idx].y=kx*V2[idx].x-ky*V1[idx].x;
}
__global__ void kernel_compute_viscous(const GCplx* V, GCplx* visc,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0),ky=ky_fold(lj+lo1),kz=(double)(lk+lo2);
    double k2=kx*kx+ky*ky+kz*kz;
    visc[idx].x=-k2*V[idx].x; visc[idx].y=-k2*V[idx].y;
}
__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
    int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0),ky=ky_fold(lj+lo1),kz=(double)(lk+lo2);
    double k2=kx*kx+ky*ky+kz*kz;
    if(k2<1e-10)return;
    double dr=-(kx*V1[idx].y+ky*V2[idx].y+kz*V3[idx].y);
    double di=  kx*V1[idx].x+ky*V2[idx].x+kz*V3[idx].x;
    double pr=dr/(-k2),pi=di/(-k2);
    V1[idx].x-=-kx*pi; V1[idx].y-= kx*pr;
    V2[idx].x-=-ky*pi; V2[idx].y-= ky*pr;
    V3[idx].x-=-kz*pi; V3[idx].y-= kz*pr;
}
__global__ void kernel_add_rhs(GCplx* r1, GCplx* r2, GCplx* r3,
    const GCplx* v1, const GCplx* v2, const GCplx* v3,
    const GCplx* f1, const GCplx* f2, const GCplx* f3, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    r1[idx].x+=v1[idx].x+f1[idx].x; r1[idx].y+=v1[idx].y+f1[idx].y;
    r2[idx].x+=v2[idx].x+f2[idx].x; r2[idx].y+=v2[idx].y+f2[idx].y;
    r3[idx].x+=v3[idx].x+f3[idx].x; r3[idx].y+=v3[idx].y+f3[idx].y;
}

// V = orig + a * k  (used for both intermediate RK4 stages and final update)
__global__ void kernel_rk4_axpy(GCplx* V1, GCplx* V2, GCplx* V3,
    const GCplx* o1, const GCplx* o2, const GCplx* o3,
    const GCplx* k1, const GCplx* k2, const GCplx* k3,
    double a, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    V1[idx].x=o1[idx].x+a*k1[idx].x; V1[idx].y=o1[idx].y+a*k1[idx].y;
    V2[idx].x=o2[idx].x+a*k2[idx].x; V2[idx].y=o2[idx].y+a*k2[idx].y;
    V3[idx].x=o3[idx].x+a*k3[idx].x; V3[idx].y=o3[idx].y+a*k3[idx].y;
}

// FIX (Bug 2): accumulate weighted ki into ksum on-the-fly instead of
// storing all four ki simultaneously.  ksum += w * k
__global__ void kernel_rk4_accum(GCplx* s1, GCplx* s2, GCplx* s3,
    const GCplx* k1, const GCplx* k2, const GCplx* k3,
    double w, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    s1[idx].x+=w*k1[idx].x; s1[idx].y+=w*k1[idx].y;
    s2[idx].x+=w*k2[idx].x; s2[idx].y+=w*k2[idx].y;
    s3[idx].x+=w*k3[idx].x; s3[idx].y+=w*k3[idx].y;
}

__global__ void kernel_div_abs(const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GReal* out, int lo0, int lo1, int lo2, int s0, int s1, int s2, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    int li=(int)(idx%s0),lj=(int)((idx/s0)%s1),lk=(int)(idx/((long long)s0*s1));
    double kx=kx_fold(li+lo0),ky=ky_fold(lj+lo1),kz=(double)(lk+lo2);
    double dr=-(kx*V1[idx].y+ky*V2[idx].y+kz*V3[idx].y);
    double di=  kx*V1[idx].x+ky*V2[idx].x+kz*V3[idx].x;
    out[idx]=sqrt(dr*dr+di*di);
}

// ============================================================
// Box helper and array management
// ============================================================
struct BoxGPU {
    int lo[3],sz[3];
    long long n()const{return(long long)sz[0]*sz[1]*sz[2];}
    int grid(int b)const{return(int)((n()+b-1)/b);}
};
static BoxGPU make_box(const heffte::box3d<>& b){
    BoxGPU bg; for(int d=0;d<3;d++){bg.lo[d]=b.low[d];bg.sz[d]=b.size[d];} return bg;
}

// FIX (Bug 2): removed k2v1..k4v3 (9 arrays) and div_c (1 array, never used).
// Added ksum1_c..ksum3_c for rolling RK4 accumulation.
// Total complex arrays: 28 → 21  (saving ~448 MB per rank for 256³ / 2 GPU).
struct GPUArrays {
    // Physical-space arrays
    GReal *V1_r, *V2_r, *V3_r;
    GReal *rot1_r, *rot2_r, *rot3_r;
    GReal *work1_r, *work2_r, *work3_r;
    GReal *scratch;                    // used for error/divergence reduction

    // Spectral-space: solution
    GCplx *V1_c, *V2_c, *V3_c;

    // Spectral-space: physics intermediates
    GCplx *rot1_c, *rot2_c, *rot3_c;
    GCplx *visc1_c, *visc2_c, *visc3_c;
    GCplx *f1_c, *f2_c, *f3_c;

    // Spectral-space: RK4 time-stepping (9 arrays instead of 15)
    GCplx *orig1_c, *orig2_c, *orig3_c; // save V at start of step
    GCplx *ksum1_c, *ksum2_c, *ksum3_c; // rolling weighted sum of ki
    GCplx *ktmp1_c, *ktmp2_c, *ktmp3_c; // output of current compute_rhs stage
};

static void alloc_arrays(GPUArrays& g, long long nr, long long nc){
    gm(&g.V1_r,nr);  gm(&g.V2_r,nr);  gm(&g.V3_r,nr);
    gm(&g.rot1_r,nr);gm(&g.rot2_r,nr);gm(&g.rot3_r,nr);
    gm(&g.work1_r,nr);gm(&g.work2_r,nr);gm(&g.work3_r,nr);
    gm(&g.scratch, max(nr,nc));

    gm(&g.V1_c,nc);  gm(&g.V2_c,nc);  gm(&g.V3_c,nc);
    gm(&g.rot1_c,nc);gm(&g.rot2_c,nc);gm(&g.rot3_c,nc);
    gm(&g.visc1_c,nc);gm(&g.visc2_c,nc);gm(&g.visc3_c,nc);
    gm(&g.f1_c,nc);  gm(&g.f2_c,nc);  gm(&g.f3_c,nc);

    gm(&g.orig1_c,nc);gm(&g.orig2_c,nc);gm(&g.orig3_c,nc);
    gm(&g.ksum1_c,nc);gm(&g.ksum2_c,nc);gm(&g.ksum3_c,nc);
    gm(&g.ktmp1_c,nc);gm(&g.ktmp2_c,nc);gm(&g.ktmp3_c,nc);
}

static void free_arrays(GPUArrays& g){
    cudaFree(g.V1_r);  cudaFree(g.V2_r);  cudaFree(g.V3_r);
    cudaFree(g.rot1_r);cudaFree(g.rot2_r);cudaFree(g.rot3_r);
    cudaFree(g.work1_r);cudaFree(g.work2_r);cudaFree(g.work3_r);
    cudaFree(g.scratch);
    cudaFree(g.V1_c);  cudaFree(g.V2_c);  cudaFree(g.V3_c);
    cudaFree(g.rot1_c);cudaFree(g.rot2_c);cudaFree(g.rot3_c);
    cudaFree(g.visc1_c);cudaFree(g.visc2_c);cudaFree(g.visc3_c);
    cudaFree(g.f1_c);  cudaFree(g.f2_c);  cudaFree(g.f3_c);
    cudaFree(g.orig1_c);cudaFree(g.orig2_c);cudaFree(g.orig3_c);
    cudaFree(g.ksum1_c);cudaFree(g.ksum2_c);cudaFree(g.ksum3_c);
    cudaFree(g.ktmp1_c);cudaFree(g.ktmp2_c);cudaFree(g.ktmp3_c);
}

template<typename FFT>
static void heffte_fwd(FFT& fft, GReal* r, GCplx* c){
    fft.forward(r, reinterpret_cast<std::complex<double>*>(c), heffte::scale::none);
}
template<typename FFT>
static void heffte_bwd(FFT& fft, GCplx* c, GReal* r){
    fft.backward(reinterpret_cast<std::complex<double>*>(c), r, heffte::scale::full);
}

// ============================================================
// Physics
// ============================================================
template<typename FFT>
static void compute_nonlinear(FFT& fft, GPUArrays& g,
    GCplx* nl1, GCplx* nl2, GCplx* nl3, const BoxGPU& br, const BoxGPU& bc)
{
    int gc=bc.grid(BLOCK), gr=br.grid(BLOCK);
    kernel_compute_rot<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        g.rot1_c,g.rot2_c,g.rot3_c,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],bc.n());
    heffte_bwd(fft,g.V1_c,g.V1_r);
    heffte_bwd(fft,g.V2_c,g.V2_r);
    heffte_bwd(fft,g.V3_c,g.V3_r);
    heffte_bwd(fft,g.rot1_c,g.rot1_r);
    heffte_bwd(fft,g.rot2_c,g.rot2_r);
    heffte_bwd(fft,g.rot3_c,g.rot3_r);
    kernel_cross_product<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,
        g.rot1_r,g.rot2_r,g.rot3_r,br.n());
    heffte_fwd(fft,g.rot1_r,nl1);
    heffte_fwd(fft,g.rot2_r,nl2);
    heffte_fwd(fft,g.rot3_r,nl3);
}

template<typename FFT>
static void compute_rhs(FFT& fft, GPUArrays& g,
    GCplx* r1, GCplx* r2, GCplx* r3,
    const BoxGPU& br, const BoxGPU& bc, double t)
{
    int gc=bc.grid(BLOCK), gr=br.grid(BLOCK);
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V1_c,g.visc1_c,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],bc.n());
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V2_c,g.visc2_c,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],bc.n());
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V3_c,g.visc3_c,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],bc.n());
    compute_nonlinear(fft,g,r1,r2,r3,br,bc);
    kernel_fill_forcing<<<gr,BLOCK>>>(g.work1_r,g.work2_r,g.work3_r,
        br.lo[0],br.lo[1],br.lo[2],br.sz[0],br.sz[1],br.sz[2],br.n(),t);
    heffte_fwd(fft,g.work1_r,g.f1_c);
    heffte_fwd(fft,g.work2_r,g.f2_c);
    heffte_fwd(fft,g.work3_r,g.f3_c);
    kernel_add_rhs<<<gc,BLOCK>>>(r1,r2,r3,
        g.visc1_c,g.visc2_c,g.visc3_c,g.f1_c,g.f2_c,g.f3_c,bc.n());
    kernel_make_div_free<<<gc,BLOCK>>>(r1,r2,r3,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],bc.n());
}

// FIX (Bug 2): memory-efficient RK4 — accumulate weighted sum on-the-fly.
//
// Algorithm (weights 1, 2, 2, 1):
//   orig  ← V_n
//   ksum  ← 0
//   ktmp  ← f(V_n,           t)          ;  ksum += 1*ktmp  ;  V ← orig + 0.5*dt*ktmp
//   ktmp  ← f(V_n+0.5*dt*k1, t+0.5*dt)  ;  ksum += 2*ktmp  ;  V ← orig + 0.5*dt*ktmp
//   ktmp  ← f(V_n+0.5*dt*k2, t+0.5*dt)  ;  ksum += 2*ktmp  ;  V ← orig +     dt*ktmp
//   ktmp  ← f(V_n+dt*k3,     t+dt)      ;  ksum +=   ktmp
//   V     ← orig + (dt/6)*ksum
//
// Complex arrays for time-stepping: 9 instead of 15.
template<typename FFT>
static void rk4_step(FFT& fft, GPUArrays& g,
    const BoxGPU& br, const BoxGPU& bc, double t)
{
    long long nc=bc.n();
    int gc=bc.grid(BLOCK);

    // ── Save original state ────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(g.orig1_c,g.V1_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.orig2_c,g.V2_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.orig3_c,g.V3_c,nc*sizeof(GCplx),cudaMemcpyDeviceToDevice));

    // ── Zero the accumulator ───────────────────────────────────────────────
    CUDA_CHECK(cudaMemset(g.ksum1_c,0,nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMemset(g.ksum2_c,0,nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMemset(g.ksum3_c,0,nc*sizeof(GCplx)));

    // ── Stage 1: k1 ────────────────────────────────────────────────────────
    compute_rhs(fft,g,g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,br,bc,t);
    kernel_rk4_accum<<<gc,BLOCK>>>(g.ksum1_c,g.ksum2_c,g.ksum3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,1.0,nc);
    // V ← V_n + 0.5*dt*k1
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        g.orig1_c,g.orig2_c,g.orig3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,0.5*TAU,nc);

    // ── Stage 2: k2 ────────────────────────────────────────────────────────
    compute_rhs(fft,g,g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,br,bc,t+0.5*TAU);
    kernel_rk4_accum<<<gc,BLOCK>>>(g.ksum1_c,g.ksum2_c,g.ksum3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,2.0,nc);
    // V ← V_n + 0.5*dt*k2
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        g.orig1_c,g.orig2_c,g.orig3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,0.5*TAU,nc);

    // ── Stage 3: k3 ────────────────────────────────────────────────────────
    compute_rhs(fft,g,g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,br,bc,t+0.5*TAU);
    kernel_rk4_accum<<<gc,BLOCK>>>(g.ksum1_c,g.ksum2_c,g.ksum3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,2.0,nc);
    // V ← V_n + dt*k3
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        g.orig1_c,g.orig2_c,g.orig3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,TAU,nc);

    // ── Stage 4: k4 ────────────────────────────────────────────────────────
    compute_rhs(fft,g,g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,br,bc,t+TAU);
    kernel_rk4_accum<<<gc,BLOCK>>>(g.ksum1_c,g.ksum2_c,g.ksum3_c,
        g.ktmp1_c,g.ktmp2_c,g.ktmp3_c,1.0,nc);

    // ── Final update: V_{n+1} = V_n + (dt/6)*(k1+2k2+2k3+k4) ─────────────
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        g.orig1_c,g.orig2_c,g.orig3_c,
        g.ksum1_c,g.ksum2_c,g.ksum3_c,TAU/6.0,nc);
}

template<typename FFT>
static pair<double,double> compute_diagnostics(FFT& fft, GPUArrays& g,
    const BoxGPU& br, const BoxGPU& bc, double t)
{
    int gc=bc.grid(BLOCK), gr=br.grid(BLOCK);
    heffte_bwd(fft,g.V1_c,g.V1_r);
    heffte_bwd(fft,g.V2_c,g.V2_r);
    heffte_bwd(fft,g.V3_c,g.V3_r);
    kernel_error_sq<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,g.scratch,
        br.lo[0],br.lo[1],br.lo[2],br.sz[0],br.sz[1],br.sz[2],br.n(),t);
    thrust::device_ptr<double> sp(g.scratch);
    double le=thrust::reduce(thrust::device,sp,sp+br.n()), ge;
    MPI_Reduce(&le,&ge,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    kernel_div_abs<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,g.scratch,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],bc.n());
    double ld=thrust::reduce(thrust::device,sp,sp+bc.n(),0.,thrust::maximum<double>()), gd;
    MPI_Reduce(&ld,&gd,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    return {sqrt(ge*DX*DY*DZ), gd};
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    if (argc < 6) {
        if (rank==0) fprintf(stderr,"Usage: %s NX NY NZ dt NSTEPS\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    NX=atoi(argv[1]); NY=atoi(argv[2]); NZ=atoi(argv[3]);
    TAU=atof(argv[4]); NT_RUN=atoi(argv[5]);
    LX=LY=LZ=2.0*M_PI; DX=LX/NX; DY=LY/NY; DZ=LZ/NZ;

    // ── GPU assignment ─────────────────────────────────────────────────────
    int num_gpus=0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int dev = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(dev));

    // FIX (Bug 1): cudaMemcpyToSymbol MUST come AFTER cudaSetDevice.
    // Before the fix these writes went to GPU 0 for every rank; kernels on
    // GPU 1+ then used uninitialised constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(d_NX,&NX,sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_NY,&NY,sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_DX,&DX,sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_DY,&DY,sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_DZ,&DZ,sizeof(double)));

    if (rank==0) printf("Grid: %d x %d x %d, dt=%.2e, steps=%d\n",NX,NY,NZ,TAU,NT_RUN);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    if (rank==0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes — heFFTe cuFFT Backend, Single Node Multi-GPU\n";
        cout << "============================================================\n";
        cout << "MPI ranks: " << nprocs << "  GPUs available per node: " << num_gpus << "\n";
    }
    cout << "  Rank " << rank << " → GPU " << dev << " (" << prop.name << ")\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) cout << "============================================================\n";

    {
    heffte::box3d<> world_r = {{0,0,0},{NX-1,NY-1,NZ-1}};
    heffte::box3d<> world_c = {{0,0,0},{NX-1,NY-1,NZ/2}};

    // Slab decomposition along X (8x1x1 instead of 2x2x2)
    // → 1 transpose per FFT instead of 3
    std::array<int,3> pg = {nprocs, 1, 1};
    auto inboxes  = heffte::split_world(world_r, pg);
    auto outboxes = heffte::split_world(world_c, pg);
    heffte::box3d<> inbox_r=inboxes[rank], outbox_c=outboxes[rank];

    if (rank==0) cout << "heFFTe slab grid: "<<pg[0]<<"x"<<pg[1]<<"x"<<pg[2]<<"\n";

    // Force GPU-aware MPI path + lighter reshape
    auto options = heffte::default_options<heffte::backend::cufft>();
    options.use_gpu_aware = true;
    options.use_reorder   = false;
    options.algorithm     = heffte::reshape_algorithm::alltoallv;

    heffte::fft3d_r2c<heffte::backend::cufft> fft(
        inbox_r, outbox_c, 2, MPI_COMM_WORLD, options);
    BoxGPU br=make_box(inbox_r), bc=make_box(outbox_c);
    long long nr=br.n(), nc=bc.n();

    // Print per-rank memory estimate
    if (rank==0) {
        double mb_real = 10.0 * nr * sizeof(GReal)  / (1024.*1024.);
        double mb_cplx = 21.0 * nc * sizeof(GCplx)  / (1024.*1024.);
        printf("  Estimated device alloc per rank: %.0f MB real + %.0f MB complex = %.0f MB\n",
               mb_real, mb_cplx, mb_real+mb_cplx);
    }

    GPUArrays g; alloc_arrays(g,nr,nc);

    // ── Initial condition ──────────────────────────────────────────────────
    kernel_fill_velocity<<<br.grid(BLOCK),BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,
        br.lo[0],br.lo[1],br.lo[2],br.sz[0],br.sz[1],br.sz[2],nr);
    heffte_fwd(fft,g.V1_r,g.V1_c);
    heffte_fwd(fft,g.V2_r,g.V2_c);
    heffte_fwd(fft,g.V3_r,g.V3_c);
    kernel_make_div_free<<<bc.grid(BLOCK),BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,
        bc.lo[0],bc.lo[1],bc.lo[2],bc.sz[0],bc.sz[1],bc.sz[2],nc);

    // ── Time loop ──────────────────────────────────────────────────────────
    double t_wall=0.;
    for(int it=0;it<NT_RUN;++it){
        double tc=it*TAU;
        double t0=MPI_Wtime();
        rk4_step(fft,g,br,bc,tc);
        CUDA_CHECK(cudaDeviceSynchronize());
        double dt_step=MPI_Wtime()-t0, dt_max;
        MPI_Allreduce(&dt_step,&dt_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        t_wall+=dt_max;
    }

    // ── Final diagnostics ──────────────────────────────────────────────────
    {
        double t_final=NT_RUN*TAU;
        auto [e,d]=compute_diagnostics(fft,g,br,bc,t_final);
        if(rank==0)
            cout << "  L2 error (t=" << fixed << setprecision(6) << t_final << "): "
                 << scientific << setprecision(4) << e << "\n";
    }
    if(rank==0){
        cout << "============================================================\n";
        cout << "  Total steps:     " << NT_RUN << "\n";
        cout << "  Total wall time: " << fixed << setprecision(4) << t_wall << " s\n";
        cout << "  Avg per step:    " << t_wall/NT_RUN << " s\n";
        cout << "============================================================\n";
    }
    free_arrays(g);
    }
    MPI_Finalize();
    return 0;
}
