// probe_direction.cu
// =============================================================================
// 四方向探针测试 + Fix1 + Fix2
//
// Fix 1: cufftXtSetSubformatDefault 在 cufftMakePlan3d 之前调用
// Fix 2: 顺序严格为 Create → AttachComm → SetSubformatDefault → MakePlan3d
//
// TEST C: V(x,y,z) = 1.0          → 期望只有 (kx=0,ky=0,kz=0) = 1.0
// TEST D: V(x,y,z) = cos(2*2π*y/NY) → 期望 (kx=0,ky=2,kz=0) 和 conj = 0.5
// TEST E: V(x,y,z) = cos(2*2π*z/NZ) → 期望 (kx=0,ky=0,kz=2) = 0.5 (仅正频)
// TEST F: V(x,y,z) = cos(2*2π*x/NX) → 期望 (kx=2,ky=0,kz=0) 和 conj = 0.5
//
// 编译：
//   nvcc -O2 -std=c++17 probe_direction.cu -lcufftMp -lmpi -o probe_direction
//
// 运行：
//   mpirun -np 4 --allow-run-as-root ./probe_direction 128 128 128
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftMp.h>

typedef cufftDoubleComplex GCplx;
constexpr int BLOCK = 256;

static int NX, NY, NZ, NZC;
__device__ __constant__ int d_NX, d_NY, d_NZ, d_NZC;

// ---------------------------------------------------------------------------
// Error macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(e) do {                                                    \
    cudaError_t _e=(e);                                                       \
    if(_e!=cudaSuccess){                                                      \
        fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,                  \
                cudaGetErrorString(_e));                                       \
        MPI_Abort(MPI_COMM_WORLD,1);}                                         \
} while(0)
#define CUFFT_CHECK(e) do {                                                   \
    cufftResult _e=(e);                                                       \
    if(_e!=CUFFT_SUCCESS){                                                    \
        fprintf(stderr,"cuFFT %s:%d code=%d\n",__FILE__,__LINE__,(int)_e);   \
        MPI_Abort(MPI_COMM_WORLD,1);}                                         \
} while(0)

static inline void*  gpu_ptr(cudaLibXtDesc* d){ return d->descriptor->data[0]; }
static inline size_t gpu_bytes(cudaLibXtDesc* d){ return d->descriptor->size[0]; }

// subFormat helpers (同原始代码)
static inline void force_d2z(cudaLibXtDesc* d){
    d->subFormat = CUFFT_XT_FORMAT_INPLACE;
}
static inline void force_z2d(cudaLibXtDesc* d){
    d->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
}
#define FFT_FORWARD(plan,buf) do{ force_d2z(buf); \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_FORWARD)); }while(0)

// ---------------------------------------------------------------------------
// Fill kernels (X-slab 实空间，stride 2*NZC)
// ---------------------------------------------------------------------------

// TEST C: 常数 1.0
__global__ void kern_fill_const(double* V, int nx_local){
    long long np = (long long)nx_local * d_NY * (2*d_NZC);
    long long idx = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=np) return;
    int kp = (int)(idx % (2*d_NZC));
    V[idx] = (kp < d_NZ) ? 1.0 : 0.0;
}

// TEST D: cos(2π * KY0 * j / NY)，只 Y 变化
__global__ void kern_fill_cosY(double* V, int nx_local, int KY0){
    long long np = (long long)nx_local * d_NY * (2*d_NZC);
    long long idx = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=np) return;
    int kp = (int)(idx % (2*d_NZC));
    int j  = (int)((idx / (2*d_NZC)) % d_NY);
    if(kp >= d_NZ){ V[idx]=0.0; return; }
    V[idx] = cos(2.0*M_PI*(double)KY0*(double)j/(double)d_NY);
}

// TEST E: cos(2π * KZ0 * k / NZ)，只 Z 变化
__global__ void kern_fill_cosZ(double* V, int nx_local, int KZ0){
    long long np = (long long)nx_local * d_NY * (2*d_NZC);
    long long idx = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=np) return;
    int kp = (int)(idx % (2*d_NZC));
    if(kp >= d_NZ){ V[idx]=0.0; return; }
    V[idx] = cos(2.0*M_PI*(double)KZ0*(double)kp/(double)d_NZ);
}

// TEST F: cos(2π * KX0 * gi / NX)，只 X 变化
__global__ void kern_fill_cosX(double* V, int nx_local, int x_offset, int KX0){
    long long np = (long long)nx_local * d_NY * (2*d_NZC);
    long long idx = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=np) return;
    int kp = (int)(idx % (2*d_NZC));
    int lx = (int)(idx / ((long long)d_NY*(2*d_NZC)));
    int gi = x_offset + lx;
    if(kp >= d_NZ){ V[idx]=0.0; return; }
    V[idx] = cos(2.0*M_PI*(double)KX0*(double)gi/(double)d_NX);
}

// Scale complex
__global__ void kern_scale(GCplx* A, long long n, double s){
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;
    A[idx].x*=s; A[idx].y*=s;
}

// ---------------------------------------------------------------------------
// 单个探针的分析函数
//   expected_pos / expected_neg: 在 rank 有 y_offset=0 上的预测 idx
//   如果 expected_neg < 0 表示没有负频率（如 const 只有 pos）
//   Layout A [gx][local_y][kz] vs Layout B [local_y][gx][kz] 的预测都打印
// ---------------------------------------------------------------------------
struct ProbeExpect {
    // Layout A: idx = gx * ny_local * NZC + local_y * NZC + kz
    // Layout B: idx = local_y * NX * NZC + gx * NZC + kz
    long long A_pos, A_neg; // -1 = N/A
    long long B_pos, B_neg;
    double    amp;          // 期望幅值
    const char* name;
};

static void run_probe(cufftHandle plan_r2c,
                      cudaLibXtDesc* buf,
                      int rank, int nprocs,
                      int nx_local, int ny_local,
                      int x_offset, int y_offset,
                      long long nc_local,
                      long long n_padded,
                      const char* tag,
                      ProbeExpect ex)
{
    // FFT → normalize
    FFT_FORWARD(plan_r2c, buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    double inv_N = 1.0/(double)((long long)NX*NY*NZ);
    int gc = (int)((nc_local+BLOCK-1)/BLOCK);
    kern_scale<<<gc,BLOCK>>>((GCplx*)gpu_ptr(buf), nc_local, inv_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 把当前 rank 的 Y-slab 拷到 host
    std::vector<GCplx> host(nc_local);
    CUDA_CHECK(cudaMemcpy(host.data(), gpu_ptr(buf),
                          nc_local*sizeof(GCplx), cudaMemcpyDeviceToHost));

    // 全局收集分析（所有 rank 独立计算，最后 reduce）
    double local_max_other  = 0.0;
    long long local_leak_count = 0;
    // "expected" set: only meaningful on rank with y_offset==0
    // 其他 rank 上 expected idx 为 N/A（因为 ky=0 只在 y_offset=0 的 rank）
    auto is_expected = [&](long long idx, int this_rank_y_offset) -> bool {
        if(this_rank_y_offset != 0) return false; // ky=0 only on rank 0
        return (idx == ex.A_pos ||
                (ex.A_neg >= 0 && idx == ex.A_neg) ||
                idx == ex.B_pos ||
                (ex.B_neg >= 0 && idx == ex.B_neg));
    };

    for(long long i=0; i<nc_local; ++i){
        double mag = sqrt(host[i].x*host[i].x + host[i].y*host[i].y);
        if(!is_expected(i, y_offset) && mag > 1e-6){
            local_max_other = std::max(local_max_other, mag);
            ++local_leak_count;
        }
    }
    double global_max_other=0.0;
    long long global_leak_count=0;
    MPI_Allreduce(&local_max_other,  &global_max_other,  1, MPI_DOUBLE,   MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_leak_count, &global_leak_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // rank 0 打印 header + 期望模态的实际值
    if(rank == 0){
        printf("\n");
        printf("========================================\n");
        printf("  %s\n", tag);
        printf("  Expected amplitude: %.4f\n", ex.amp);
        printf("  ny_local=%d  NZC=%d  NX=%d\n", ny_local, NZC, NX);
        printf("  Layout-A predicted idx: pos=%lld  neg=%lld\n", ex.A_pos, ex.A_neg);
        printf("  Layout-B predicted idx: pos=%lld  neg=%lld\n", ex.B_pos, ex.B_neg);
        printf("----------------------------------------\n");

        // 打印 Layout-A pos
        if(ex.A_pos >= 0 && ex.A_pos < nc_local){
            double mag = sqrt(host[ex.A_pos].x*host[ex.A_pos].x +
                              host[ex.A_pos].y*host[ex.A_pos].y);
            printf("  [Layout-A pos] idx=%lld  V=(%+.8f,%+.8f)  |V|=%.8f  %s\n",
                   ex.A_pos, host[ex.A_pos].x, host[ex.A_pos].y, mag,
                   fabs(mag - ex.amp) < 1e-4 ? "✓ MATCH" : "✗ WRONG");
        }
        // 打印 Layout-A neg
        if(ex.A_neg >= 0 && ex.A_neg < nc_local){
            double mag = sqrt(host[ex.A_neg].x*host[ex.A_neg].x +
                              host[ex.A_neg].y*host[ex.A_neg].y);
            printf("  [Layout-A neg] idx=%lld  V=(%+.8f,%+.8f)  |V|=%.8f  %s\n",
                   ex.A_neg, host[ex.A_neg].x, host[ex.A_neg].y, mag,
                   fabs(mag - ex.amp) < 1e-4 ? "✓ MATCH" : "✗ WRONG");
        }
        // 打印 Layout-B pos（与 A 不同时才打）
        if(ex.B_pos != ex.A_pos && ex.B_pos >= 0 && ex.B_pos < nc_local){
            double mag = sqrt(host[ex.B_pos].x*host[ex.B_pos].x +
                              host[ex.B_pos].y*host[ex.B_pos].y);
            printf("  [Layout-B pos] idx=%lld  V=(%+.8f,%+.8f)  |V|=%.8f  %s\n",
                   ex.B_pos, host[ex.B_pos].x, host[ex.B_pos].y, mag,
                   fabs(mag - ex.amp) < 1e-4 ? "✓ MATCH (→ Layout B!)" : "");
        }
        if(ex.B_neg >= 0 && ex.B_neg != ex.A_neg && ex.B_neg >= 0 && ex.B_neg < nc_local){
            double mag = sqrt(host[ex.B_neg].x*host[ex.B_neg].x +
                              host[ex.B_neg].y*host[ex.B_neg].y);
            printf("  [Layout-B neg] idx=%lld  V=(%+.8f,%+.8f)  |V|=%.8f  %s\n",
                   ex.B_neg, host[ex.B_neg].x, host[ex.B_neg].y, mag,
                   fabs(mag - ex.amp) < 1e-4 ? "✓ MATCH (→ Layout B!)" : "");
        }
        printf("  Global leak_count=%lld  leak_max=%.4e\n",
               global_leak_count, global_max_other);
        // 结论
        bool A_ok = false, B_ok = false;
        {
            double mA = (ex.A_pos>=0&&ex.A_pos<nc_local) ?
                        sqrt(host[ex.A_pos].x*host[ex.A_pos].x+
                             host[ex.A_pos].y*host[ex.A_pos].y) : 0.0;
            double mB = (ex.B_pos!=ex.A_pos&&ex.B_pos>=0&&ex.B_pos<nc_local) ?
                        sqrt(host[ex.B_pos].x*host[ex.B_pos].x+
                             host[ex.B_pos].y*host[ex.B_pos].y) : 0.0;
            A_ok = fabs(mA - ex.amp) < 1e-4;
            B_ok = fabs(mB - ex.amp) < 1e-4;
        }
        if(A_ok && global_leak_count==0)
            printf("  >> RESULT: PASS (Layout A confirmed, no leakage)\n");
        else if(A_ok && global_leak_count>0)
            printf("  >> RESULT: PARTIAL – Layout-A idx matches but %.0f%% leakage\n",
                   100.0*global_max_other/ex.amp);
        else if(B_ok)
            printf("  >> RESULT: FAIL – Layout B idx matches (kernels use wrong layout)\n");
        else
            printf("  >> RESULT: FAIL – neither Layout A nor B idx matches (transpose broken?)\n");
        printf("========================================\n");
        fflush(stdout);
    }

    // 每个 rank 打印自己的 leak 详情（排序后前 5 条）
    MPI_Barrier(MPI_COMM_WORLD);
    for(int r=0; r<nprocs; ++r){
        if(r==rank){
            if(local_leak_count > 0){
                printf("  [rank %d y_off=%d] %lld leak entries, top 5:\n",
                       rank, y_offset, local_leak_count);
                // 找前5大泄漏
                std::vector<std::pair<double,long long>> leaks;
                for(long long i=0; i<nc_local; ++i){
                    if(is_expected(i,y_offset)) continue;
                    double mag=sqrt(host[i].x*host[i].x+host[i].y*host[i].y);
                    if(mag>1e-6) leaks.push_back({mag,i});
                }
                std::sort(leaks.rbegin(),leaks.rend());
                int shown=0;
                for(auto& [mag,i]: leaks){
                    long long A_gx = i/((long long)ny_local*NZC);
                    long long A_ly = (i/NZC)%ny_local;
                    long long A_kz = i%NZC;
                    long long B_ly = i/((long long)NX*NZC);
                    long long B_gx = (i/NZC)%NX;
                    long long B_kz = i%NZC;
                    printf("    idx=%lld |V|=%.4e  A:(gx=%lld gy=%lld kz=%lld)  B:(gx=%lld gy=%lld kz=%lld)\n",
                           i, mag,
                           A_gx, (long long)y_offset+A_ly, A_kz,
                           B_gx, (long long)y_offset+B_ly, B_kz);
                    if(++shown>=5){ printf("    ...\n"); break; }
                }
                fflush(stdout);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 清零 buffer
    CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_bytes(buf)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(argc < 4){
        if(rank==0) fprintf(stderr,"Usage: mpirun -np NP %s NX NY NZ\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    NX=atoi(argv[1]); NY=atoi(argv[2]); NZ=atoi(argv[3]);
    NZC = NZ/2+1;

    if(NX%nprocs!=0 || NY%nprocs!=0){
        if(rank==0) fprintf(stderr,"nprocs=%d must divide NX=%d and NY=%d\n",nprocs,NX,NY);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // GPU binding
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node_comm);
    int local_rank; MPI_Comm_rank(node_comm,&local_rank); MPI_Comm_free(&node_comm);
    int num_gpus=0; CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if(local_rank>=num_gpus){ fprintf(stderr,"rank %d: local_rank %d >= ngpu %d\n",rank,local_rank,num_gpus); MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(local_rank));

    // Partition
    int nx_local = NX/nprocs;
    int ny_local = NY/nprocs;
    int x_offset = rank*nx_local;
    int y_offset = rank*ny_local;
    long long nc_local = (long long)NX*ny_local*NZC;
    long long n_padded = (long long)nx_local*NY*(2*NZC);

    cudaMemcpyToSymbol(d_NX,&NX,sizeof(int));
    cudaMemcpyToSymbol(d_NY,&NY,sizeof(int));
    cudaMemcpyToSymbol(d_NZ,&NZ,sizeof(int));
    cudaMemcpyToSymbol(d_NZC,&NZC,sizeof(int));

    if(rank==0){
        printf("============================================================\n");
        printf("  Direction Probe: Fix1+Fix2 + TEST C/D/E/F\n");
        printf("  Grid: %d x %d x %d   NZC=%d   nprocs=%d\n",NX,NY,NZ,NZC,nprocs);
        printf("  ny_local=%d  nc_local=%lld\n",ny_local,nc_local);
        printf("============================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("  Rank %d -> GPU %d  X=[%d,%d)  Y=[%d,%d)\n",
           rank,local_rank,x_offset,x_offset+nx_local,y_offset,y_offset+ny_local);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // Fix 1 + Fix 2: 正确的 plan 创建顺序
    //   Create → AttachComm → SetSubformatDefault → MakePlan3d
    // =========================================================================
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));

    MPI_Comm world = MPI_COMM_WORLD;
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &world));

    // Fix 1: SetSubformatDefault 必须在 MakePlan3d 之前
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_r2c,
        CUFFT_XT_FORMAT_INPLACE,
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_c2r,
        CUFFT_XT_FORMAT_INPLACE_SHUFFLED,
        CUFFT_XT_FORMAT_INPLACE));

    size_t ws_r2c=0, ws_c2r=0;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    if(rank==0){
        printf("  plan created: Fix1+Fix2 applied (SetSubformatDefault BEFORE MakePlan3d)\n");
        fflush(stdout);
    }

    // 分配一个 buffer（INPLACE，同原始代码）
    cudaLibXtDesc* buf = nullptr;
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &buf, CUFFT_XT_FORMAT_INPLACE));
    CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_bytes(buf)));

    int gp = (int)((n_padded+BLOCK-1)/BLOCK);

    // =========================================================================
    // 预计算所有 expected idx（都在 y_offset=0 的 rank 上，即 rank 0）
    // KX0=KY0=KZ0=2
    // =========================================================================
    const int K = 2; // 探针频率

    // --- TEST C: V=1, 期望 (kx=0,ky=0,kz=0) = 1.0 ---
    // Layout A: idx = gx*ny_local*NZC + local_y*NZC + kz
    //           (0,0,0) → 0 * ny_local * NZC + 0 * NZC + 0 = 0
    // Layout B: idx = local_y*NX*NZC + gx*NZC + kz  = 0
    // A_pos == B_pos == 0
    ProbeExpect ex_C;
    ex_C.name = "TEST-C: V=1";
    ex_C.A_pos = 0;
    ex_C.A_neg = -1;
    ex_C.B_pos = 0;
    ex_C.B_neg = -1;
    ex_C.amp   = 1.0;

    // --- TEST D: V=cos(2π*K*j/NY), 期望 (kx=0,ky=K,kz=0) and (kx=0,ky=NY-K,kz=0) = 0.5 ---
    // (ky=K) lives on rank with y_offset <= K < y_offset+ny_local  → K=2 → rank 0 (y_offset=0)
    // local_y = K - y_offset = K (on rank 0)
    // Layout A: idx = gx*ny_local*NZC + local_y*NZC + kz = 0 + K*NZC + 0 = K*NZC
    // Layout B: idx = local_y*NX*NZC + gx*NZC + kz = K*NX*NZC + 0 + 0 = K*NX*NZC
    // conj (ky=NY-K) on rank with y_offset+ny_local-1 >= NY-K → rank nprocs-1 (y_offset=96)
    //   local_y = (NY-K) - y_offset_of_last_rank = (128-2)-96 = 30
    //   Layout A: 0 + 30*NZC + 0 = 30*NZC
    //   Layout B: 30*NX*NZC + 0  = 30*NX*NZC
    // (These are on different ranks; report will be on rank 0 only for pos, and rank nprocs-1 for neg)
    // For simplicity we only report pos here (both layouts, rank 0 side)
    ProbeExpect ex_D;
    ex_D.name = "TEST-D: V=cos(2*2pi*y/NY)";
    ex_D.A_pos = (long long)K * NZC;                // Layout A (gx=0,ly=K,kz=0)
    ex_D.A_neg = -1;                                 // conj on different rank
    ex_D.B_pos = (long long)K * NX * NZC;            // Layout B (ly=K,gx=0,kz=0)
    ex_D.B_neg = -1;
    ex_D.amp   = 0.5;

    // --- TEST E: V=cos(2π*K*k/NZ), 期望 (kx=0,ky=0,kz=K) = 0.5 ---
    // (kx=0,ky=0,kz=K) on rank 0 (y_offset=0), local_y=0
    // Layout A: idx = 0*ny_local*NZC + 0*NZC + K = K
    // Layout B: idx = 0*NX*NZC + 0*NZC + K = K
    // A == B here (gx=0, local_y=0, kz=K → same idx)
    // conj: for R2C, kz lives in [0, NZC); kz=K < NZC → only one entry
    ProbeExpect ex_E;
    ex_E.name = "TEST-E: V=cos(2*2pi*z/NZ)";
    ex_E.A_pos = (long long)K;
    ex_E.A_neg = -1;
    ex_E.B_pos = (long long)K; // same
    ex_E.B_neg = -1;
    ex_E.amp   = 0.5;

    // --- TEST F: V=cos(2π*K*gi/NX), 期望 (kx=K,ky=0,kz=0) and (kx=NX-K,ky=0,kz=0) = 0.5 ---
    // Both on rank 0 (ky=0 → local_y=0 → rank 0)
    // Layout A: idx = K*(ny_local*NZC) + 0 + 0 = K*ny_local*NZC
    //           neg: (NX-K)*ny_local*NZC
    // Layout B: idx = 0*NX*NZC + K*NZC + 0 = K*NZC
    //           neg: 0*NX*NZC + (NX-K)*NZC = (NX-K)*NZC
    ProbeExpect ex_F;
    ex_F.name = "TEST-F: V=cos(2*2pi*x/NX)";
    ex_F.A_pos = (long long)K         * ny_local * NZC;
    ex_F.A_neg = (long long)(NX-K)    * ny_local * NZC;
    ex_F.B_pos = (long long)K         * NZC;
    ex_F.B_neg = (long long)(NX-K)    * NZC;
    ex_F.amp   = 0.5;

    // =========================================================================
    // 运行 4 个探针
    // =========================================================================

    // TEST C
    if(rank==0){ printf("\n>>> Running TEST C (V=const 1.0) ...\n"); fflush(stdout); }
    CUDA_CHECK(cudaMemset(gpu_ptr(buf),0,gpu_bytes(buf)));
    kern_fill_const<<<gp,BLOCK>>>((double*)gpu_ptr(buf), nx_local);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_probe(plan_r2c, buf,
              rank, nprocs, nx_local, ny_local, x_offset, y_offset,
              nc_local, n_padded, ex_C.name, ex_C);

    // TEST D
    if(rank==0){ printf("\n>>> Running TEST D (V=cos(2y)) ...\n"); fflush(stdout); }
    CUDA_CHECK(cudaMemset(gpu_ptr(buf),0,gpu_bytes(buf)));
    kern_fill_cosY<<<gp,BLOCK>>>((double*)gpu_ptr(buf), nx_local, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_probe(plan_r2c, buf,
              rank, nprocs, nx_local, ny_local, x_offset, y_offset,
              nc_local, n_padded, ex_D.name, ex_D);

    // TEST E
    if(rank==0){ printf("\n>>> Running TEST E (V=cos(2z)) ...\n"); fflush(stdout); }
    CUDA_CHECK(cudaMemset(gpu_ptr(buf),0,gpu_bytes(buf)));
    kern_fill_cosZ<<<gp,BLOCK>>>((double*)gpu_ptr(buf), nx_local, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_probe(plan_r2c, buf,
              rank, nprocs, nx_local, ny_local, x_offset, y_offset,
              nc_local, n_padded, ex_E.name, ex_E);

    // TEST F
    if(rank==0){ printf("\n>>> Running TEST F (V=cos(2x)) ...\n"); fflush(stdout); }
    CUDA_CHECK(cudaMemset(gpu_ptr(buf),0,gpu_bytes(buf)));
    kern_fill_cosX<<<gp,BLOCK>>>((double*)gpu_ptr(buf), nx_local, x_offset, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    run_probe(plan_r2c, buf,
              rank, nprocs, nx_local, ny_local, x_offset, y_offset,
              nc_local, n_padded, ex_F.name, ex_F);

    // =========================================================================
    // 诊断总结
    // =========================================================================
    if(rank==0){
        printf("\n============================================================\n");
        printf("  DIAGNOSTIC MATRIX:\n");
        printf("  C=PASS D=PASS E=PASS F=FAIL → X transpose broken\n");
        printf("  C=PASS D=FAIL E=PASS F=FAIL → X+Y transpose broken\n");
        printf("  All PASS                    → transpose OK, bug elsewhere\n");
        printf("  All show Layout-B match      → spectral kernels use wrong layout\n");
        printf("  TEST F: Layout-A pos idx=%lld  Layout-B pos idx=%lld\n",
               ex_F.A_pos, ex_F.B_pos);
        printf("  If TEST F shows value at idx=%lld → Layout A (cuFFTMp standard)\n", ex_F.A_pos);
        printf("  If TEST F shows value at idx=%lld → Layout B (kernel indexing matches)\n", ex_F.B_pos);
        printf("============================================================\n");
        fflush(stdout);
    }

    CUFFT_CHECK(cufftXtFree(buf));
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
