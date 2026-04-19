/**
 * NavierStokes_periodic_cufftmp_mgpu.cu
 * Navier-Stokes solver — cuFFTMp, single node, multi-GPU
 *
 * cuFFTMp distributes 3D FFTs across multiple GPUs using NCCL/MPI.
 * Data layout: X-slab decomposition.
 *   Real space:    rank r owns [ix_start..ix_end] × [0..NY-1] × [0..NZ-1]
 *   Spectral space: rank r owns [ix_start..ix_end] × [0..NY-1] × [0..NZC-1]
 *   (Same X-slab for both; cuFFTMp handles internal redistribution)
 *
 * Normalization convention (same as cuFFT single GPU):
 *   V_c = DFT(V_r)/N  (normalized: scale by 1/N after each D2Z)
 *   Z2D is unnormalized: Z2D(V_c)*N = V_r → correct physical value
 *
 * Requirements:
 *   - CUDA 11.7+ with cuFFTMp support (header: cufftMp.h, lib: cufftMp)
 *   - CUDA-aware MPI or NCCL for GPU-to-GPU communication
 *   - Each MPI rank binds to one GPU: cudaSetDevice(rank % num_gpus)
 *
 * Run: mpirun -np <ngpu> ./navier_stokes_cufftmp_mgpu
 *
 * NOTE on spectral layout after D2Z:
 *   We specify output distribution as same X-slab (ix_start..ix_end, 0..NY-1, 0..NZC-1).
 *   cuFFTMp guarantees data is in this distribution after D2Z, handling any internal
 *   transposes transparently.
 *   Wavenumbers: kx = gi≤NX/2 ? gi : gi-NX,  ky folded similarly,  kz = kc (0..NZC-1)
 */

#include <cmath>
#include <iomanip>
#include <iostream>
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
constexpr int    NZC      = NZ/2 + 1;
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
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)
#define CUFFT_CHECK(e) do { cufftResult _e=(e); if(_e!=CUFFT_SUCCESS){ \
    fprintf(stderr,"cuFFT error %s:%d: %d\n",__FILE__,__LINE__,(int)_e); exit(1);} } while(0)

template<typename T>
static void gm(T** p, long long n){CUDA_CHECK(cudaMalloc((void**)p,(size_t)n*sizeof(T)));}

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
// CUDA kernels — X-slab layout with x_offset
//
// Real space:    V[lx][j][k],  lx in [0, nx_local), j in [0, NY), k in [0, NZ)
//                flat idx = lx*NY*NZ + j*NZ + k
//                global ix = lx + x_offset_r
//
// Spectral space: V_c[lx][j][kc], kc in [0, NZC)
//                flat idx = lx*NY*NZC + j*NZC + kc
//                global ix = lx + x_offset_c  (= x_offset_r for X-slab decomp)
//
// Normalization: V_c = DFT(V_r)/N  (normalized convention, same as cuFFT single GPU)
// ============================================================

__global__ void kernel_fill_velocity(GReal* V1, GReal* V2, GReal* V3,
    int nx_local, int x_offset, double t)
{
    long long nr_l = (long long)nx_local*NY*NZ;
    long long idx = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=nr_l) return;
    int k=(int)(idx%NZ), j=(int)((idx/NZ)%NY), lx=(int)(idx/((long long)NY*NZ));
    double x=(lx+x_offset)*DX, y=j*DY, z=k*DZ;
    V1[idx]=func_V1(x,y,z,t); V2[idx]=func_V2(x,y,z,t); V3[idx]=func_V3(x,y,z,t);
}

__global__ void kernel_fill_forcing(GReal* W1, GReal* W2, GReal* W3,
    int nx_local, int x_offset, double t)
{
    long long nr_l = (long long)nx_local*NY*NZ;
    long long idx = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (idx>=nr_l) return;
    int k=(int)(idx%NZ), j=(int)((idx/NZ)%NY), lx=(int)(idx/((long long)NY*NZ));
    double x=(lx+x_offset)*DX, y=j*DY, z=k*DZ;
    W1[idx]=func_f1(x,y,z,t); W2[idx]=func_f2(x,y,z,t); W3[idx]=func_f3(x,y,z,t);
}

__global__ void kernel_scale_cplx(GCplx* A, long long n, double sc) {
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return; A[idx].x*=sc; A[idx].y*=sc;
}

__global__ void kernel_cross_product(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* R1, GReal* R2, GReal* R3, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    double v1=V1[idx],v2=V2[idx],v3=V3[idx],w1=R1[idx],w2=R2[idx],w3=R3[idx];
    R1[idx]=v2*w3-v3*w2; R2[idx]=v3*w1-v1*w3; R3[idx]=v1*w2-v2*w1;
}

__global__ void kernel_error_sq(const GReal* V1, const GReal* V2, const GReal* V3,
    GReal* err, int nx_local, int x_offset, double t)
{
    long long nr_l=(long long)nx_local*NY*NZ;
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nr_l)return;
    int k=(int)(idx%NZ),j=(int)((idx/NZ)%NY),lx=(int)(idx/((long long)NY*NZ));
    double x=(lx+x_offset)*DX,y=j*DY,z=k*DZ;
    double d1=V1[idx]-func_V1(x,y,z,t),d2=V2[idx]-func_V2(x,y,z,t),d3=V3[idx]-func_V3(x,y,z,t);
    err[idx]=d1*d1+d2*d2+d3*d3;
}

// Spectral kernels — flat spectral idx: lx*NY*NZC + j*NZC + kc
// kx uses global ix = lx + x_offset (same slab offset as real space)
__global__ void kernel_compute_rot(
    const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GCplx* R1, GCplx* R2, GCplx* R3,
    int nx_local, int x_offset)
{
    long long nc_l=(long long)nx_local*NY*NZC;
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nc_l)return;
    int kc=(int)(idx%NZC),j=(int)((idx/NZC)%NY),lx=(int)(idx/((long long)NY*NZC));
    int gi=lx+x_offset;
    double kx=(gi<=NX/2)?(double)gi:(double)(gi-NX);
    double ky=(j<=NY/2)?(double)j:(double)(j-NY);
    double kz=(double)kc;
    R1[idx].x=-(ky*V3[idx].y-kz*V2[idx].y); R1[idx].y=ky*V3[idx].x-kz*V2[idx].x;
    R2[idx].x=-(kz*V1[idx].y-kx*V3[idx].y); R2[idx].y=kz*V1[idx].x-kx*V3[idx].x;
    R3[idx].x=-(kx*V2[idx].y-ky*V1[idx].y); R3[idx].y=kx*V2[idx].x-ky*V1[idx].x;
}

__global__ void kernel_compute_viscous(const GCplx* V, GCplx* visc,
    int nx_local, int x_offset)
{
    long long nc_l=(long long)nx_local*NY*NZC;
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nc_l)return;
    int kc=(int)(idx%NZC),j=(int)((idx/NZC)%NY),lx=(int)(idx/((long long)NY*NZC));
    int gi=lx+x_offset;
    double kx=(gi<=NX/2)?(double)gi:(double)(gi-NX);
    double ky=(j<=NY/2)?(double)j:(double)(j-NY);
    double kz=(double)kc,k2=kx*kx+ky*ky+kz*kz;
    visc[idx].x=-k2*V[idx].x; visc[idx].y=-k2*V[idx].y;
}

__global__ void kernel_make_div_free(GCplx* V1, GCplx* V2, GCplx* V3,
    int nx_local, int x_offset)
{
    long long nc_l=(long long)nx_local*NY*NZC;
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nc_l)return;
    int kc=(int)(idx%NZC),j=(int)((idx/NZC)%NY),lx=(int)(idx/((long long)NY*NZC));
    int gi=lx+x_offset;
    double kx=(gi<=NX/2)?(double)gi:(double)(gi-NX);
    double ky=(j<=NY/2)?(double)j:(double)(j-NY);
    double kz=(double)kc,k2=kx*kx+ky*ky+kz*kz;
    if(k2<1e-10)return;
    double dr=-(kx*V1[idx].y+ky*V2[idx].y+kz*V3[idx].y);
    double di=  kx*V1[idx].x+ky*V2[idx].x+kz*V3[idx].x;
    double pr=dr/(-k2),pi=di/(-k2);
    V1[idx].x-=-kx*pi; V1[idx].y-=kx*pr;
    V2[idx].x-=-ky*pi; V2[idx].y-=ky*pr;
    V3[idx].x-=-kz*pi; V3[idx].y-=kz*pr;
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

__global__ void kernel_rk4_update(GCplx* V1, GCplx* V2, GCplx* V3,
    const GCplx* o1, const GCplx* o2, const GCplx* o3,
    const GCplx* k1v1,const GCplx* k2v1,const GCplx* k3v1,const GCplx* k4v1,
    const GCplx* k1v2,const GCplx* k2v2,const GCplx* k3v2,const GCplx* k4v2,
    const GCplx* k1v3,const GCplx* k2v3,const GCplx* k3v3,const GCplx* k4v3,
    double d6, long long n)
{
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n)return;
    V1[idx].x=o1[idx].x+d6*(k1v1[idx].x+2.*k2v1[idx].x+2.*k3v1[idx].x+k4v1[idx].x);
    V1[idx].y=o1[idx].y+d6*(k1v1[idx].y+2.*k2v1[idx].y+2.*k3v1[idx].y+k4v1[idx].y);
    V2[idx].x=o2[idx].x+d6*(k1v2[idx].x+2.*k2v2[idx].x+2.*k3v2[idx].x+k4v2[idx].x);
    V2[idx].y=o2[idx].y+d6*(k1v2[idx].y+2.*k2v2[idx].y+2.*k3v2[idx].y+k4v2[idx].y);
    V3[idx].x=o3[idx].x+d6*(k1v3[idx].x+2.*k2v3[idx].x+2.*k3v3[idx].x+k4v3[idx].x);
    V3[idx].y=o3[idx].y+d6*(k1v3[idx].y+2.*k2v3[idx].y+2.*k3v3[idx].y+k4v3[idx].y);
}

__global__ void kernel_div_abs(const GCplx* V1, const GCplx* V2, const GCplx* V3,
    GReal* out, int nx_local, int x_offset)
{
    long long nc_l=(long long)nx_local*NY*NZC;
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nc_l)return;
    int kc=(int)(idx%NZC),j=(int)((idx/NZC)%NY),lx=(int)(idx/((long long)NY*NZC));
    int gi=lx+x_offset;
    double kx=(gi<=NX/2)?(double)gi:(double)(gi-NX);
    double ky=(j<=NY/2)?(double)j:(double)(j-NY);
    double kz=(double)kc;
    double dr=-(kx*V1[idx].y+ky*V2[idx].y+kz*V3[idx].y);
    double di=  kx*V1[idx].x+ky*V2[idx].x+kz*V3[idx].x;
    out[idx]=sqrt(dr*dr+di*di);
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
    GCplx *k1v1,*k1v2,*k1v3,*k2v1,*k2v2,*k2v3;
    GCplx *k3v1,*k3v2,*k3v3,*k4v1,*k4v2,*k4v3;
    GCplx *tmp1_c, *tmp2_c, *tmp3_c;
    GReal *scratch;
};
static void alloc_arrays(GPUArrays& g, long long nr, long long nc){
    gm(&g.V1_r,nr);gm(&g.V2_r,nr);gm(&g.V3_r,nr);
    gm(&g.rot1_r,nr);gm(&g.rot2_r,nr);gm(&g.rot3_r,nr);
    gm(&g.work1_r,nr);gm(&g.work2_r,nr);gm(&g.work3_r,nr);
    gm(&g.scratch,max(nr,nc));
    gm(&g.V1_c,nc);gm(&g.V2_c,nc);gm(&g.V3_c,nc);
    gm(&g.rot1_c,nc);gm(&g.rot2_c,nc);gm(&g.rot3_c,nc);
    gm(&g.visc1_c,nc);gm(&g.visc2_c,nc);gm(&g.visc3_c,nc);
    gm(&g.f1_c,nc);gm(&g.f2_c,nc);gm(&g.f3_c,nc);
    gm(&g.k1v1,nc);gm(&g.k1v2,nc);gm(&g.k1v3,nc);
    gm(&g.k2v1,nc);gm(&g.k2v2,nc);gm(&g.k2v3,nc);
    gm(&g.k3v1,nc);gm(&g.k3v2,nc);gm(&g.k3v3,nc);
    gm(&g.k4v1,nc);gm(&g.k4v2,nc);gm(&g.k4v3,nc);
    gm(&g.tmp1_c,nc);gm(&g.tmp2_c,nc);gm(&g.tmp3_c,nc);}
static void free_arrays(GPUArrays& g){
    cudaFree(g.V1_r);cudaFree(g.V2_r);cudaFree(g.V3_r);
    cudaFree(g.rot1_r);cudaFree(g.rot2_r);cudaFree(g.rot3_r);
    cudaFree(g.work1_r);cudaFree(g.work2_r);cudaFree(g.work3_r);cudaFree(g.scratch);
    cudaFree(g.V1_c);cudaFree(g.V2_c);cudaFree(g.V3_c);
    cudaFree(g.rot1_c);cudaFree(g.rot2_c);cudaFree(g.rot3_c);
    cudaFree(g.visc1_c);cudaFree(g.visc2_c);cudaFree(g.visc3_c);
    cudaFree(g.f1_c);cudaFree(g.f2_c);cudaFree(g.f3_c);
    cudaFree(g.k1v1);cudaFree(g.k1v2);cudaFree(g.k1v3);
    cudaFree(g.k2v1);cudaFree(g.k2v2);cudaFree(g.k2v3);
    cudaFree(g.k3v1);cudaFree(g.k3v2);cudaFree(g.k3v3);
    cudaFree(g.k4v1);cudaFree(g.k4v2);cudaFree(g.k4v3);
    cudaFree(g.tmp1_c);cudaFree(g.tmp2_c);cudaFree(g.tmp3_c);}

// ============================================================
// cuFFTMp plan setup helper
//
// Creates paired D2Z and Z2D plans with X-slab distribution.
// After D2Z: spectral data in same X-slab → cuFFTMp handles internal redistribution.
// Before Z2D: spectral data must be in the X-slab → cuFFTMp redistributes back.
// ============================================================
static void create_mp_plans(cufftHandle& plan_r2c, cufftHandle& plan_c2r,
    int nx_local, int x_start, int x_end, MPI_Comm comm, size_t* work_size)
{
    long long lo_r[3] = {x_start, 0, 0};
    long long hi_r[3] = {x_end,   NY-1, NZ-1};
    long long lo_c[3] = {x_start, 0, 0};
    long long hi_c[3] = {x_end,   NY-1, NZC-1};

    // R2C plan: real X-slab → complex X-slab
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &comm));
    CUFFT_CHECK(cufftXtSetDistribution(plan_r2c, 3,
        lo_r, hi_r, /*strides=*/nullptr,
        lo_c, hi_c, /*strides=*/nullptr));
    size_t ws_r2c;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));

    // C2R plan: complex X-slab → real X-slab
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &comm));
    CUFFT_CHECK(cufftXtSetDistribution(plan_c2r, 3,
        lo_c, hi_c, nullptr,
        lo_r, hi_r, nullptr));
    size_t ws_c2r;
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    if (work_size) *work_size = max(ws_r2c, ws_c2r);
}

// ============================================================
// Physics
// ============================================================

// Nonlinear term: V × rot(V)
// cuFFTMp Z2D (C2R) = unnormalized, result is V_r directly since V_c = DFT/N.
// V_c NOT destroyed by cuFFTMp Z2D (out-of-place), so viscous can be computed after.
static void compute_nonlinear(cufftHandle plan_r2c, cufftHandle plan_c2r,
    GPUArrays& g, GCplx* nl1, GCplx* nl2, GCplx* nl3,
    int nx_l, int x_off, long long nr_l, long long nc_l)
{
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    int gc=(int)((nc_l+BLOCK-1)/BLOCK), gr=(int)((nr_l+BLOCK-1)/BLOCK);

    // 1. Spectral vorticity: rot_c = ik × V_c
    kernel_compute_rot<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c, g.rot1_c,g.rot2_c,g.rot3_c, nx_l,x_off);

    // 2. V_c → V_r (Z2D, out-of-place: V_c preserved)
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V1_c, g.V1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V2_c, g.V2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.V3_c, g.V3_r));

    // 3. rot_c → rot_r
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot1_c, g.rot1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot2_c, g.rot2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r, g.rot3_c, g.rot3_r));

    // 4. Real-space cross product
    kernel_cross_product<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r, g.rot1_r,g.rot2_r,g.rot3_r, nr_l);

    // 5. rot_r → nl_c (D2Z + normalize by 1/N)
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot1_r, nl1));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot2_r, nl2));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.rot3_r, nl3));
    kernel_scale_cplx<<<gc,BLOCK>>>(nl1,nc_l,inv_N);
    kernel_scale_cplx<<<gc,BLOCK>>>(nl2,nc_l,inv_N);
    kernel_scale_cplx<<<gc,BLOCK>>>(nl3,nc_l,inv_N);
}

static void compute_rhs(cufftHandle plan_r2c, cufftHandle plan_c2r,
    GPUArrays& g, GCplx* r1, GCplx* r2, GCplx* r3,
    int nx_l, int x_off, long long nr_l, long long nc_l, double t)
{
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    int gc=(int)((nc_l+BLOCK-1)/BLOCK), gr=(int)((nr_l+BLOCK-1)/BLOCK);

    // Viscous: reads V_c (safe, V_c not destroyed by Z2D in nonlinear)
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V1_c,g.visc1_c,nx_l,x_off);
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V2_c,g.visc2_c,nx_l,x_off);
    kernel_compute_viscous<<<gc,BLOCK>>>(g.V3_c,g.visc3_c,nx_l,x_off);

    // Nonlinear → r1/r2/r3
    compute_nonlinear(plan_r2c,plan_c2r,g,r1,r2,r3,nx_l,x_off,nr_l,nc_l);

    // Forcing
    kernel_fill_forcing<<<gr,BLOCK>>>(g.work1_r,g.work2_r,g.work3_r,nx_l,x_off,t);
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.work1_r, g.f1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.work2_r, g.f2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c, g.work3_r, g.f3_c));
    kernel_scale_cplx<<<gc,BLOCK>>>(g.f1_c,nc_l,inv_N);
    kernel_scale_cplx<<<gc,BLOCK>>>(g.f2_c,nc_l,inv_N);
    kernel_scale_cplx<<<gc,BLOCK>>>(g.f3_c,nc_l,inv_N);

    // Combine rhs = nl + visc + f
    kernel_add_rhs<<<gc,BLOCK>>>(r1,r2,r3, g.visc1_c,g.visc2_c,g.visc3_c, g.f1_c,g.f2_c,g.f3_c, nc_l);

    // Project to divergence-free
    kernel_make_div_free<<<gc,BLOCK>>>(r1,r2,r3,nx_l,x_off);
}

static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r,
    GPUArrays& g, int nx_l, int x_off, long long nr_l, long long nc_l, double t)
{
    int gc=(int)((nc_l+BLOCK-1)/BLOCK);
    CUDA_CHECK(cudaMemcpy(g.tmp1_c,g.V1_c,nc_l*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.tmp2_c,g.V2_c,nc_l*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(g.tmp3_c,g.V3_c,nc_l*sizeof(GCplx),cudaMemcpyDeviceToDevice));
    compute_rhs(plan_r2c,plan_c2r,g,g.k1v1,g.k1v2,g.k1v3,nx_l,x_off,nr_l,nc_l,t);
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,g.tmp1_c,g.tmp2_c,g.tmp3_c,g.k1v1,g.k1v2,g.k1v3,0.5*TAU,nc_l);
    compute_rhs(plan_r2c,plan_c2r,g,g.k2v1,g.k2v2,g.k2v3,nx_l,x_off,nr_l,nc_l,t+0.5*TAU);
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,g.tmp1_c,g.tmp2_c,g.tmp3_c,g.k2v1,g.k2v2,g.k2v3,0.5*TAU,nc_l);
    compute_rhs(plan_r2c,plan_c2r,g,g.k3v1,g.k3v2,g.k3v3,nx_l,x_off,nr_l,nc_l,t+0.5*TAU);
    kernel_rk4_axpy<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,g.tmp1_c,g.tmp2_c,g.tmp3_c,g.k3v1,g.k3v2,g.k3v3,TAU,nc_l);
    compute_rhs(plan_r2c,plan_c2r,g,g.k4v1,g.k4v2,g.k4v3,nx_l,x_off,nr_l,nc_l,t+TAU);
    kernel_rk4_update<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,g.tmp1_c,g.tmp2_c,g.tmp3_c,
        g.k1v1,g.k2v1,g.k3v1,g.k4v1,g.k1v2,g.k2v2,g.k3v2,g.k4v2,
        g.k1v3,g.k2v3,g.k3v3,g.k4v3,TAU/6.,nc_l);
}

static pair<double,double> compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r,
    GPUArrays& g, int nx_l, int x_off, long long nr_l, long long nc_l, double t)
{
    const double inv_N=1.0/(double)(NX*NY*NZ);
    int gc=(int)((nc_l+BLOCK-1)/BLOCK), gr=(int)((nr_l+BLOCK-1)/BLOCK);
    // V_c → V_r (Z2D, preserves V_c since out-of-place)
    CUFFT_CHECK(cufftExecZ2D(plan_c2r,g.V1_c,g.V1_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r,g.V2_c,g.V2_r));
    CUFFT_CHECK(cufftExecZ2D(plan_c2r,g.V3_c,g.V3_r));
    // L2 error
    kernel_error_sq<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,g.scratch,nx_l,x_off,t);
    thrust::device_ptr<double> sp(g.scratch);
    double le=thrust::reduce(thrust::device,sp,sp+nr_l),ge;
    MPI_Reduce(&le,&ge,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    // max|div V|
    kernel_div_abs<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,g.scratch,nx_l,x_off);
    double ld=thrust::reduce(thrust::device,sp,sp+nc_l,0.,thrust::maximum<double>()),gd;
    MPI_Reduce(&ld,&gd,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    return{sqrt(ge*DX*DY*DZ),gd};
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank,nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // ---- GPU assignment: single node ----
    int num_gpus=0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int dev=rank%num_gpus;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));

    // ---- X-slab decomposition ----
    // Require NX divisible by nprocs for equal slabs.
    if (NX % nprocs != 0 && rank==0) {
        fprintf(stderr,"WARNING: NX=%d not divisible by nprocs=%d; last ranks may have different slab size.\n",NX,nprocs);
    }
    int nx_l = NX / nprocs;
    // Handle remainder: last rank gets extra rows
    int x_start = rank * nx_l;
    int nx_l_r  = (rank < nprocs-1) ? nx_l : NX - x_start;  // actual local NX
    int x_end   = x_start + nx_l_r - 1;
    long long nr_l = (long long)nx_l_r * NY * NZ;
    long long nc_l = (long long)nx_l_r * NY * NZC;

    if (rank==0) {
        cout << "============================================================\n";
        cout << "  Navier-Stokes — cuFFTMp, Single Node Multi-GPU\n";
        cout << "============================================================\n";
        cout << "MPI ranks: " << nprocs << "  GPUs: " << num_gpus << "\n";
        cout << "Grid: " << NX << "x" << NY << "x" << NZ << "\n";
        cout << "X-slab per rank: ~" << nx_l_r << " of " << NX << "\n";
        cout << "Steps: " << NT_RUN << "/" << NT_TOTAL << "  dt=" << TAU << "\n";
    }
    cout << "  Rank " << rank << " → GPU " << dev << " (" << prop.name << ")  x=[" << x_start << "," << x_end << "]\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) cout << "============================================================\n";

    // ---- cuFFTMp plans ----
    cufftHandle plan_r2c, plan_c2r;
    size_t work_size;
    create_mp_plans(plan_r2c, plan_c2r, nx_l_r, x_start, x_end, MPI_COMM_WORLD, &work_size);

    // ---- GPU memory ----
    GPUArrays g;
    alloc_arrays(g, nr_l, nc_l);

    // ---- Initial condition ----
    {
    int gr=(int)((nr_l+BLOCK-1)/BLOCK),gc=(int)((nc_l+BLOCK-1)/BLOCK);
    const double inv_N=1.0/(double)(NX*NY*NZ);
    kernel_fill_velocity<<<gr,BLOCK>>>(g.V1_r,g.V2_r,g.V3_r,nx_l_r,x_start,0.);
    CUFFT_CHECK(cufftExecD2Z(plan_r2c,g.V1_r,g.V1_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c,g.V2_r,g.V2_c));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c,g.V3_r,g.V3_c));
    kernel_scale_cplx<<<gc,BLOCK>>>(g.V1_c,nc_l,inv_N);
    kernel_scale_cplx<<<gc,BLOCK>>>(g.V2_c,nc_l,inv_N);
    kernel_scale_cplx<<<gc,BLOCK>>>(g.V3_c,nc_l,inv_N);
    // Project to divergence-free
    kernel_make_div_free<<<gc,BLOCK>>>(g.V1_c,g.V2_c,g.V3_c,nx_l_r,x_start);
    }

    {auto [e,d]=compute_diagnostics(plan_r2c,plan_c2r,g,nx_l_r,x_start,nr_l,nc_l,0.);
     if(rank==0) cout<<"\nInitial (t=0):\n  L2 error: "<<scientific<<e<<"  max|div V|: "<<d<<"\n\n";}

    // ---- Time integration ----
    if(rank==0){
        cout<<"============================================================\n"<<"  Time Integration (RK4)\n------------------------------------------------------------\n"
            <<setw(6)<<"Step"<<setw(12)<<"Wall(s)"<<setw(16)<<"L2 Error"<<setw(16)<<"max|div V|\n------------------------------------------------------------\n";}
    double t_wall=0.;
    for(int it=0;it<=NT_RUN;++it){
        double tc=it*TAU;
        auto [e,d]=compute_diagnostics(plan_r2c,plan_c2r,g,nx_l_r,x_start,nr_l,nc_l,tc);
        if(rank==0) cout<<setw(6)<<it<<setw(12)<<fixed<<setprecision(4)<<t_wall<<setw(16)<<scientific<<setprecision(4)<<e<<setw(16)<<d<<"\n";
        if(it<NT_RUN){
            double t0=MPI_Wtime();
            rk4_step(plan_r2c,plan_c2r,g,nx_l_r,x_start,nr_l,nc_l,tc);
            CUDA_CHECK(cudaDeviceSynchronize());
            double dt=MPI_Wtime()-t0,dtmax;
            MPI_Allreduce(&dt,&dtmax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
            t_wall+=dtmax;}}
    if(rank==0){cout<<"============================================================\n"
                    <<"  Total wall time: "<<fixed<<setprecision(4)<<t_wall<<" s\n"
                    <<"  Avg per step:    "<<t_wall/NT_RUN<<" s\n"
                    <<"============================================================\n";}

    free_arrays(g);
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
