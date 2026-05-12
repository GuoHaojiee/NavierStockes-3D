// NavierStokes_periodic_cufftmp_multinode.cu
// =============================================================================
// FIX: 摒弃手动修改 desc->subFormat 的 hack 方式。
// 
// 诊断逻辑：
// 1. cuFFTMp 内部可能维护一个状态机，手动修改 desc->subFormat (CUFFT_XT_FORMAT_INPLACE / SHUFFLED)
//    虽能影响单次执行，但在复杂的 RK4 步骤中，如果内部状态机与 descriptor 不一致，
//    会导致 NVSHMEM 在做 slab-transpose 时按错误的布局解读内存，从而产生 NaN。
// 2. 解决方案：让每个 Buffer 遵循 cuFFTMp 的自然状态流。
//    - rot_buf: 初始分配为 SHUFFLED，匹配 kernel 直接写入的谱空间数据布局。
//    - restore_v_buf: 通过一次“假 FFT”将 V_buf 的状态从 INPLACE (实空间) 驱动到 
//      SHUFFLED (谱空间)，然后覆盖数据，不再手动修改 subFormat。
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

// FFT 执行宏：不再手动修改 subFormat，完全信任 cuFFTMp 自身的状态转换。
#define FFT_FORWARD(plan, buf) CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_FORWARD))
#define FFT_INVERSE(plan, buf) CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_INVERSE))

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
// Kernels
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

__global__ void kernel_fill_test_real(double* V, int nx_local, int x_offset) {
    long long n_padded = (long long)nx_local * d_NY * (2*d_NZC);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_padded) return;
    int k_padded = (int)(idx % (2*d_NZC));
    int j        = (int)((idx / (2*d_NZC)) % d_NY);
    int lx       = (int)(idx / ((long long)d_NY * (2*d_NZC)));
    int gi = x_offset + lx;
    if (k_padded >= d_NZ) { V[idx] = 0.0; return; }
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
    int kz = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx = (int)(idx / ((long long)ny_local * d_NZC));
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
    int kz = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx = (int)(idx / ((long long)ny_local * d_NZC));
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
    int kz = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx = (int)(idx / ((long long)ny_local * d_NZC));
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
    int kz = (int)(idx % d_NZC);
    int local_y = (int)((idx / d_NZC) % ny_local);
    int gx = (int)(idx / ((long long)ny_local * d_NZC));
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
    rhs1[idx].x = visc1[idx].x + f1[idx].x;
    rhs1[idx].y = visc1[idx].y + f1[idx].y;
    rhs2[idx].x = visc2[idx].x + f2[idx].x;
    rhs2[idx].y = visc2[idx].y + f2[idx].y;
    rhs3[idx].x = visc3[idx].x + f3[idx].x;
    rhs3[idx].y = visc3[idx].y + f3[idx].y;
}

__global__ void kernel_rk4_update(GCplx* V1, GCplx* V2, GCplx* V3,
                                    const GCplx* o1, const GCplx* o2, const GCplx* o3,
                                    const GCplx* k1v1, const GCplx* k1v2, const GCplx* k1v3,
                                    const GCplx* k2v1, const GCplx* k2v2, const GCplx* k2v3,
                                    const GCplx* k3v1, const GCplx* k3v2, const GCplx* k3v3,
                                    const GCplx* k4v1, const GCplx* k4v2, const GCplx* k4v3,
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

    cudaLibXtDesc *V1_buf, *V2_buf, *V3_buf;
    cudaLibXtDesc *rot1_buf, *rot2_buf, *rot3_buf;

    GCplx *rhs1_c, *rhs2_c, *rhs3_c;
    GCplx *visc1_c, *visc2_c, *visc3_c;
    GCplx *f1_c, *f2_c, *f3_c;

    GCplx *V1_orig, *V2_orig, *V3_orig;
    GCplx *k1v1, *k1v2, *k1v3, *k2v1, *k2v2, *k2v3;
    GCplx *k3v1, *k3v2, *k3v3, *k4v1, *k4v2, *k4v3;

    double* scratch;
};

static void alloc_and_zero(cufftHandle plan, cudaLibXtDesc **d, cudaLibXtSubFormat format) {
    CUFFT_CHECK(cufftXtMalloc(plan, d, format));
    CUDA_CHECK(cudaMemset(gpu_ptr(*d), 0, gpu_size_bytes(*d)));
}

static void alloc_state(cufftHandle plan_r2c, State& s) {
    // V_buf 初始作为 INPLACE (实空间布局)
    alloc_and_zero(plan_r2c, &s.V1_buf, CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.V2_buf, CUFFT_XT_FORMAT_INPLACE);
    alloc_and_zero(plan_r2c, &s.V3_buf, CUFFT_XT_FORMAT_INPLACE);
    
    // rot_buf 初始设为 SHUFFLED，因为我们在 kernel_compute_rot 中直接按 Y-slab 复数写入
    alloc_and_zero(plan_r2c, &s.rot1_buf, CUFFT_XT_FORMAT_INPLACE_SHUFFLED);
    alloc_and_zero(plan_r2c, &s.rot2_buf, CUFFT_XT_FORMAT_INPLACE_SHUFFLED);
    alloc_and_zero(plan_r2c, &s.rot3_buf, CUFFT_XT_FORMAT_INPLACE_SHUFFLED);

    long long nc = s.nc_local;
    CUDA_CHECK(cudaMalloc(&s.rhs1_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.rhs2_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.rhs3_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.visc1_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.visc2_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.visc3_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.f1_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.f2_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.f3_c, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.V1_orig, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.V2_orig, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.V3_orig, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k1v1, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k1v2, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k1v3, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k2v1, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k2v2, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k2v3, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k3v1, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k3v2, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k3v3, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k4v1, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k4v2, nc*sizeof(GCplx)));
    CUDA_CHECK(cudaMalloc(&s.k4v3, nc*sizeof(GCplx)));

    long long nr_max = max(s.nr_local, s.nc_local);
    CUDA_CHECK(cudaMalloc(&s.scratch, nr_max*sizeof(double)));
}

static void free_state(State& s) {
    CUFFT_CHECK(cufftXtFree(s.V1_buf)); CUFFT_CHECK(cufftXtFree(s.V2_buf)); CUFFT_CHECK(cufftXtFree(s.V3_buf));
    CUFFT_CHECK(cufftXtFree(s.rot1_buf)); CUFFT_CHECK(cufftXtFree(s.rot2_buf)); CUFFT_CHECK(cufftXtFree(s.rot3_buf));
    cudaFree(s.rhs1_c); cudaFree(s.rhs2_c); cudaFree(s.rhs3_c);
    cudaFree(s.visc1_c); cudaFree(s.visc2_c); cudaFree(s.visc3_c);
    cudaFree(s.f1_c); cudaFree(s.f2_c); cudaFree(s.f3_c);
    cudaFree(s.k1v1); cudaFree(s.k1v2); cudaFree(s.k1v3);
    cudaFree(s.k2v1); cudaFree(s.k2v2); cudaFree(s.k2v3);
    cudaFree(s.k3v1); cudaFree(s.k3v2); cudaFree(s.k3v3);
    cudaFree(s.k4v1); cudaFree(s.k4v2); cudaFree(s.k4v3);
    cudaFree(s.V1_orig); cudaFree(s.V2_orig); cudaFree(s.V3_orig);
    cudaFree(s.scratch);
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

    kernel_fill_test_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.nx_local, s.x_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    FFT_FORWARD(plan_r2c, s.V1_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    FFT_INVERSE(plan_c2r, s.V1_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), n_padded, inv_N);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_test_error<<<gr, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.scratch, s.nx_local, s.x_offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<double> sp(s.scratch);
    double local_sq = thrust::reduce(thrust::device, sp, sp + nr_local, 0.0);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_sq * DX * DY * DZ);
}

// =============================================================================
// Solver components
// =============================================================================
static void compute_nonlinear(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);
    const double inv_N = 1.0/(double)(NX*NY*NZ);

    // V_buf 现在处于 SHUFFLED 状态 (来自 rk4_update 或 initial setup)。
    // 1. 计算 curl V -> 存入 rot_buf (SHUFFLED)
    kernel_compute_rot<<<gc, BLOCK>>>(
        (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
        (GCplx*)gpu_ptr(s.rot1_buf), (GCplx*)gpu_ptr(s.rot2_buf), (GCplx*)gpu_ptr(s.rot3_buf),
        s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 将 V 和 rot 转回实空间
    FFT_INVERSE(plan_c2r, s.V1_buf); FFT_INVERSE(plan_c2r, s.V2_buf); FFT_INVERSE(plan_c2r, s.V3_buf);
    FFT_INVERSE(plan_c2r, s.rot1_buf); FFT_INVERSE(plan_c2r, s.rot2_buf); FFT_INVERSE(plan_c2r, s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. 实空间叉乘 V x rot -> 存回 rot_buf
    kernel_cross_product<<<gr, BLOCK>>>(
        (double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
        (double*)gpu_ptr(s.rot1_buf), (double*)gpu_ptr(s.rot2_buf), (double*)gpu_ptr(s.rot3_buf),
        s.nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 将 rot_buf 转回谱空间
    FFT_FORWARD(plan_r2c, s.rot1_buf); FFT_FORWARD(plan_r2c, s.rot2_buf); FFT_FORWARD(plan_r2c, s.rot3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot1_buf), s.rhs1_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot2_buf), s.rhs2_c, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.rot3_buf), s.rhs3_c, nc);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs1_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs2_c, nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>(s.rhs3_c, nc, inv_N);
}

static void compute_rhs(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gp = (int)(((long long)s.nx_local*NY*2*NZC + BLOCK - 1) / BLOCK);

    compute_nonlinear(plan_r2c, plan_c2r, s, t);

    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.visc1_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.visc2_c, s.ny_local, s.y_offset);
    kernel_compute_viscous<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.visc3_c, s.ny_local, s.y_offset);

    kernel_fill_forcing<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
                                       s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    FFT_FORWARD(plan_r2c, s.V1_buf); FFT_FORWARD(plan_r2c, s.V2_buf); FFT_FORWARD(plan_r2c, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double inv_N = 1.0/(double)(NX*NY*NZ);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), nc, inv_N);
    kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), nc, inv_N);

    kernel_add_to_rhs<<<gc, BLOCK>>>(s.rhs1_c, s.rhs2_c, s.rhs3_c,
                                     s.visc1_c, s.visc2_c, s.visc3_c,
                                     (GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                     nc);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void save_v_orig(State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), s.V1_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V2_buf), s.V2_orig, s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V3_buf), s.V3_orig, s.nc_local);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 状态恢复：通过调用 FFT_FORWARD 强制驱动 descriptor 进入 SHUFFLED 状态。
static void restore_v_buf(cufftHandle plan_r2c, State& s) {
    int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);
    // V_buf 此时在 INPLACE (实空间) 状态。
    // 我们做一个“假” FFT 转换来同步 cuFFTMp 内部状态机，丢弃输出，然后写入 V_orig。
    FFT_FORWARD(plan_r2c, s.V1_buf);
    FFT_FORWARD(plan_r2c, s.V2_buf);
    FFT_FORWARD(plan_r2c, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel_copy_cplx<<<gc, BLOCK>>>(s.V1_orig, (GCplx*)gpu_ptr(s.V1_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V2_orig, (GCplx*)gpu_ptr(s.V2_buf), s.nc_local);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.V3_orig, (GCplx*)gpu_ptr(s.V3_buf), s.nc_local);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void rk4_step(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t) {
    long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);

    save_v_orig(s);
    compute_rhs(plan_r2c, plan_c2r, s, t);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k1v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k1v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k1v3, nc);

    // K2
    restore_v_buf(plan_r2c, s);
    kernel_rk4_update<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                     s.V1_orig, s.V2_orig, s.V3_orig,
                                     s.k1v1, s.k1v2, s.k1v3, s.k1v1, s.k1v2, s.k1v3, s.k1v1, s.k1v2, s.k1v3, s.k1v1, s.k1v2, s.k1v3,
                                     0.5*TAU, nc);
    kernel_make_div_free<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                        s.ny_local, s.y_offset);
    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k2v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k2v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k2v3, nc);

    // K3
    restore_v_buf(plan_r2c, s);
    kernel_rk4_update<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                     s.V1_orig, s.V2_orig, s.V3_orig,
                                     s.k2v1, s.k2v2, s.k2v3, s.k2v1, s.k2v2, s.k2v3, s.k2v1, s.k2v2, s.k2v3, s.k2v1, s.k2v2, s.k2v3,
                                     0.5*TAU, nc);
    kernel_make_div_free<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                        s.ny_local, s.y_offset);
    compute_rhs(plan_r2c, plan_c2r, s, t + 0.5*TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k3v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k3v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k3v3, nc);

    // K4
    restore_v_buf(plan_r2c, s);
    kernel_rk4_update<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                     s.V1_orig, s.V2_orig, s.V3_orig,
                                     s.k3v1, s.k3v2, s.k3v3, s.k3v1, s.k3v2, s.k3v3, s.k3v1, s.k3v2, s.k3v3, s.k3v1, s.k3v2, s.k3v3,
                                     TAU, nc);
    kernel_make_div_free<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                        s.ny_local, s.y_offset);
    compute_rhs(plan_r2c, plan_c2r, s, t + TAU);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs1_c, s.k4v1, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs2_c, s.k4v2, nc);
    kernel_copy_cplx<<<gc, BLOCK>>>(s.rhs3_c, s.k4v3, nc);

    // Final update
    restore_v_buf(plan_r2c, s);
    kernel_rk4_update<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                     s.V1_orig, s.V2_orig, s.V3_orig,
                                     s.k1v1, s.k1v2, s.k1v3, s.k2v1, s.k2v2, s.k2v3, s.k3v1, s.k3v2, s.k3v3, s.k4v1, s.k4v2, s.k4v3,
                                     TAU/6.0, nc);
    kernel_make_div_free<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                        s.ny_local, s.y_offset);
    // V_buf 现在仍然处于 SHUFFLED 状态。
}

static void compute_diagnostics(cufftHandle plan_r2c, cufftHandle plan_c2r, State& s, double t,
                                double& L2_err, double& max_div) {
    long long nc = s.nc_local;
    int gc = (int)((nc + BLOCK - 1) / BLOCK);
    int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);

    // V_buf 在 SHUFFLED 谱空间状态下计算散度
    kernel_div_abs<<<gc, BLOCK>>>((GCplx*)gpu_ptr(s.V1_buf), (GCplx*)gpu_ptr(s.V2_buf), (GCplx*)gpu_ptr(s.V3_buf),
                                   s.scratch, s.ny_local, s.y_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_ptr<double> sp(s.scratch);
    max_div = thrust::reduce(thrust::device, sp, sp + nc, 0.0, thrust::maximum<double>());

    // 转回实空间计算 L2 误差
    FFT_INVERSE(plan_c2r, s.V1_buf); FFT_INVERSE(plan_c2r, s.V2_buf); FFT_INVERSE(plan_c2r, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double inv_N = 1.0/(double)(NX*NY*NZ);
    long long n_padded = (long long)s.nx_local * NY * 2*NZC;
    int gp = (int)((n_padded + BLOCK - 1) / BLOCK);
    kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), n_padded, inv_N);
    kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V2_buf), n_padded, inv_N);
    kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V3_buf), n_padded, inv_N);

    kernel_error_sq<<<gr, BLOCK>>>((double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
                                    s.scratch, s.nx_local, s.x_offset, t);
    CUDA_CHECK(cudaDeviceSynchronize());
    double local_sq = thrust::reduce(thrust::device, sp, sp + s.nr_local, 0.0);
    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    L2_err = sqrt(global_sq * DX * DY * DZ);

    // 重新转回谱空间以进行下一步迭代
    FFT_FORWARD(plan_r2c, s.V1_buf); FFT_FORWARD(plan_r2c, s.V2_buf); FFT_FORWARD(plan_r2c, s.V3_buf);
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

    NX=128; NY=128; NZ=128; NT_RUN=5;
    LX=2.*M_PI; LY=2.*M_PI; LZ=2.*M_PI;
    DX=LX/NX; DY=LY/NY; DZ=LZ/NZ; TAU=1e-5; NZC=NZ/2+1;

    int ndev; cudaGetDeviceCount(&ndev);
    s.gpu = s.rank % ndev;
    CUDA_CHECK(cudaSetDevice(s.gpu));

    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    size_t work;
    int n[] = {NX, NY, NZ};
    CUFFT_CHECK(cufftXtMakePlanMany(plan_r2c, 3, n, NULL, 1, 1, CUDA_C_64F, NULL, 1, 1, CUDA_C_96F, 1, &work, CUFFT_FORWARD));
    CUFFT_CHECK(cufftXtMakePlanMany(plan_c2r, 3, n, NULL, 1, 1, CUDA_C_96F, NULL, 1, 1, CUDA_C_64F, 1, &work, CUFFT_INVERSE));

    cudaLibXtDesc *d_r2c; CUFFT_CHECK(cufftXtGetDescriptor(plan_r2c, &d_r2c, CUFFT_FORWARD));
    s.nx_local = (int)d_r2c->descriptor->size[0] / (NY * 2*NZC * sizeof(double));
    s.x_offset = (int)d_r2c->descriptor->displacement[0] / (NY * 2*NZC * sizeof(double));
    s.nr_local = (long long)s.nx_local * NY * NZ;

    cudaLibXtDesc *d_c2r; CUFFT_CHECK(cufftXtGetDescriptor(plan_c2r, &d_c2r, CUFFT_INVERSE));
    s.ny_local = (int)d_c2r->descriptor->size[0] / (NX * NZC * sizeof(GCplx));
    s.y_offset = (int)d_c2r->descriptor->displacement[0] / (NX * NZC * sizeof(GCplx));
    s.nc_local = (long long)s.ny_local * NX * NZC;

    CUDA_CHECK(cudaMemcpyToSymbol(d_NX, &NX, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_NY, &NY, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_NZ, &NZ, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_NZC, &NZC, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_DX, &DX, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_DY, &DY, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_DZ, &DZ, sizeof(double)));

    alloc_state(plan_r2c, s);

#if RUN_FFT_TEST
    if (s.rank == 0) cout << "Running FFT smoke test...\n";
    double err = fft_roundtrip_test(plan_r2c, plan_c2r, s);
    if (s.rank == 0) cout << "  Roundtrip L2: " << scientific << setprecision(4) << err << "\n";
    
    // === 关键诊断：验证 subFormat 手动强制是否在 cuFFTMp 上有效 ===
    {
        const long long n_padded = (long long)s.nx_local * NY * (2*NZC);
        const double inv_N = 1.0/(double)(NX*NY*NZ);
        int gp = (int)((n_padded + BLOCK - 1) / BLOCK);
        int gr = (int)((s.nr_local + BLOCK - 1) / BLOCK);
        int gc = (int)((s.nc_local + BLOCK - 1) / BLOCK);

        kernel_fill_test_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.nx_local, s.x_offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        FFT_FORWARD(plan_r2c, s.V1_buf);
        CUDA_CHECK(cudaDeviceSynchronize());

        GCplx* spec_save;
        CUDA_CHECK(cudaMalloc(&spec_save, s.nc_local * sizeof(GCplx)));
        CUDA_CHECK(cudaMemcpy(spec_save, gpu_ptr(s.V1_buf), s.nc_local*sizeof(GCplx), cudaMemcpyDeviceToDevice));

        FFT_INVERSE(plan_c2r, s.V1_buf);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));

        // Path B: 手动写入复数并强制 subFormat = SHUFFLED
        kernel_copy_cplx<<<gc, BLOCK>>>(spec_save, (GCplx*)gpu_ptr(s.V1_buf), s.nc_local);
        CUDA_CHECK(cudaDeviceSynchronize());
        s.V1_buf->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED; 

        CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, s.V1_buf, s.V1_buf, CUFFT_INVERSE));
        CUDA_CHECK(cudaDeviceSynchronize());
        kernel_scale_real<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), n_padded, inv_N);
        CUDA_CHECK(cudaDeviceSynchronize());
        kernel_test_error<<<gr, BLOCK>>>((double*)gpu_ptr(s.V1_buf), s.scratch, s.nx_local, s.x_offset);
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::device_ptr<double> sp(s.scratch);
        double local_sq = thrust::reduce(thrust::device, sp, sp + s.nr_local, 0.0);
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double l2_hack = sqrt(global_sq * DX * DY * DZ);
        if (s.rank == 0) {
            cout << "subFormat hack test L2 error: " << scientific << setprecision(4) 
                 << l2_hack << "  (should be < 1e-10 if hack works)\n" << flush;
        }
        cudaFree(spec_save);
        CUDA_CHECK(cudaMemset(gpu_ptr(s.V1_buf), 0, gpu_size_bytes(s.V1_buf)));
    }
#endif

    if (s.rank == 0) cout << "Initializing simulation...\n";
    int gp = (int)(((long long)s.nx_local*NY*2*NZC + BLOCK - 1) / BLOCK);
    kernel_fill_velocity<<<gp, BLOCK>>>((double*)gpu_ptr(s.V1_buf), (double*)gpu_ptr(s.V2_buf), (double*)gpu_ptr(s.V3_buf),
                                         s.nx_local, s.x_offset, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    FFT_FORWARD(plan_r2c, s.V1_buf); FFT_FORWARD(plan_r2c, s.V2_buf); FFT_FORWARD(plan_r2c, s.V3_buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (s.rank == 0) cout << "Starting time integration (" << NT_RUN << " steps)...\n";
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

    double t_final = NT_RUN * TAU;
    double L2_err, max_div;
    compute_diagnostics(plan_r2c, plan_c2r, s, t_final, L2_err, max_div);
    if (s.rank == 0) {
        cout << "  L2 error   (t=" << fixed << setprecision(6) << t_final << "): " << scientific << setprecision(4) << L2_err << "\n";
        cout << "  max|div V| (t=" << fixed << setprecision(6) << t_final << "): " << scientific << setprecision(4) << max_div << "\n";
        cout << "  Avg step wall-time: " << fixed << setprecision(4) << t_wall/NT_RUN << " s\n";
    }

    free_state(s);
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}