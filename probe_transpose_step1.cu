// probe_transpose_step1.cu
// =============================================================================
// Step-1 Diagnostic: 常数信号探针，直接验证 cuFFTMp 分布式 all-to-all
// transpose 是否正常工作。
//
// 理论：
//   输入 V(x,y,z) = 1.0 (全局常数)
//   forward-FFT + 归一化后，只有 (kx=0, ky=0, kz=0) 的幅值应该 = 1.0。
//   其他所有模态 = 0.0。
//
// 判断标准：
//   V_hat(0,0,0) ≈ 1.0              → transpose 正常  ✓
//   V_hat(0,0,0) ≈ 1/nprocs = 0.25  → transpose 损坏，每 rank 只看到自己  ✗
//   V_hat(0,0,0) ≈ 任何其他值       → 其他问题
//
// 编译：
//   nvcc -O2 -std=c++17 probe_transpose_step1.cu \
//        -I/path/to/cufft/include \
//        -lcufftMp -lmpi -o probe_transpose_step1
//
// 运行（4 GPU）：
//   mpirun -np 4 ./probe_transpose_step1 128 128 128
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftMp.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

typedef cufftDoubleComplex GCplx;
constexpr int BLOCK = 256;

// ---------------------------------------------------------------------------
// Global grid dimensions (set from argv in main)
// ---------------------------------------------------------------------------
static int NX, NY, NZ, NZC;

__device__ __constant__ int d_NX, d_NY, d_NZ, d_NZC;

// ---------------------------------------------------------------------------
// Error macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(e) do {                                                   \
    cudaError_t _e = (e);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        MPI_Abort(MPI_COMM_WORLD, 1);                                        \
    }                                                                        \
} while(0)

#define CUFFT_CHECK(e) do {                                                  \
    cufftResult _e = (e);                                                    \
    if (_e != CUFFT_SUCCESS) {                                               \
        fprintf(stderr, "cuFFT error %s:%d: code %d\n",                     \
                __FILE__, __LINE__, (int)_e);                                \
        MPI_Abort(MPI_COMM_WORLD, 1);                                        \
    }                                                                        \
} while(0)

// ---------------------------------------------------------------------------
// Helpers: raw GPU pointer / size from a cudaLibXtDesc
// ---------------------------------------------------------------------------
static inline void*   gpu_ptr(cudaLibXtDesc* d) { return d->descriptor->data[0]; }
static inline size_t  gpu_bytes(cudaLibXtDesc* d){ return d->descriptor->size[0]; }

// ---------------------------------------------------------------------------
// subFormat helpers (same hack as original code)
// ---------------------------------------------------------------------------
static inline void force_d2z(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE;           // X-slab real
}
static inline void force_z2d(cudaLibXtDesc* d) {
    d->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;  // Y-slab complex
}

// FFT macros — identical to original code
#define FFT_FORWARD(plan, buf) do {                                          \
    force_d2z(buf);                                                          \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_FORWARD));    \
} while(0)

#define FFT_INVERSE(plan, buf) do {                                          \
    force_z2d(buf);                                                          \
    CUFFT_CHECK(cufftXtExecDescriptor((plan),(buf),(buf),CUFFT_INVERSE));    \
} while(0)

// ---------------------------------------------------------------------------
// Kernel: fill X-slab real buffer with constant 1.0
//   - Indices 0..NZ-1   along Z (real data)
//   - Indices NZ..2*NZC-1  along Z (R2C padding) → 0.0
// ---------------------------------------------------------------------------
__global__ void kernel_fill_const_one(double* V, int nx_local)
{
    long long n_padded = (long long)nx_local * d_NY * (2 * d_NZC);
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_padded) return;

    int k_padded = (int)(idx % (2 * d_NZC));
    V[idx] = (k_padded < d_NZ) ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// Kernel: scale Y-slab complex buffer by a scalar
// ---------------------------------------------------------------------------
__global__ void kernel_scale_cplx(GCplx* A, long long nc_local, double scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc_local) return;
    A[idx].x *= scale;
    A[idx].y *= scale;
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

    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np <NP> %s NX NY NZ\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    NX = atoi(argv[1]); NY = atoi(argv[2]); NZ = atoi(argv[3]);
    NZC = NZ / 2 + 1;

    if (NX % nprocs != 0 || NY % nprocs != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: nprocs=%d must divide NX=%d and NY=%d\n",
                    nprocs, NX, NY);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // -----------------------------------------------------------------------
    // GPU binding (same logic as original)
    // -----------------------------------------------------------------------
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                        rank, MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (local_rank >= num_gpus) {
        fprintf(stderr, "Rank %d: local_rank %d >= num_gpus %d\n",
                rank, local_rank, num_gpus);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank));

    // -----------------------------------------------------------------------
    // Per-rank partition
    // -----------------------------------------------------------------------
    int nx_local = NX / nprocs;
    int ny_local = NY / nprocs;
    int x_offset = rank * nx_local;
    int y_offset = rank * ny_local;
    long long nc_local = (long long)NX * ny_local * NZC;  // Y-slab complex
    long long n_padded = (long long)nx_local * NY * (2 * NZC); // X-slab padded

    // Copy constants to device
    cudaMemcpyToSymbol(d_NX,  &NX,  sizeof(int));
    cudaMemcpyToSymbol(d_NY,  &NY,  sizeof(int));
    cudaMemcpyToSymbol(d_NZ,  &NZ,  sizeof(int));
    cudaMemcpyToSymbol(d_NZC, &NZC, sizeof(int));

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Step-1 Transpose Probe\n");
        printf("  Grid: %d x %d x %d   nprocs=%d\n", NX, NY, NZ, nprocs);
        printf("  Input: V(x,y,z) = 1.0 everywhere\n");
        printf("  Expected result after FFT+normalize:\n");
        printf("    V_hat(0,0,0) = 1.0        [transpose OK]\n");
        printf("    V_hat(0,0,0) = %.6f  [transpose BROKEN]\n",
               1.0 / nprocs);
        printf("============================================================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("  Rank %d -> GPU %d  X=[%d,%d)  Y=[%d,%d)\n",
           rank, local_rank,
           x_offset, x_offset + nx_local,
           y_offset, y_offset + ny_local);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Create cuFFTMp plan (same as original)
    // -----------------------------------------------------------------------
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));

    MPI_Comm world = MPI_COMM_WORLD;
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &world));

    size_t ws_r2c = 0, ws_c2r = 0;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, NX, NY, NZ, CUFFT_D2Z, &ws_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, NX, NY, NZ, CUFFT_Z2D, &ws_c2r));

    // -----------------------------------------------------------------------
    // Allocate buffer (same as original: INPLACE format)
    // -----------------------------------------------------------------------
    cudaLibXtDesc* buf = nullptr;
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &buf, CUFFT_XT_FORMAT_INPLACE));
    CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_bytes(buf)));

    // -----------------------------------------------------------------------
    // =======================================================================
    // TEST A: No warmup — bare FFT_FORWARD on fresh buffer
    // =======================================================================
    // -----------------------------------------------------------------------
    if (rank == 0) {
        printf("\n--- TEST A: no warmup (fresh buffer) ---\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Fill constant 1.0
    {
        int gp = (int)((n_padded + BLOCK - 1) / BLOCK);
        kernel_fill_const_one<<<gp, BLOCK>>>((double*)gpu_ptr(buf), nx_local);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward FFT (uses force_d2z inside macro)
    FFT_FORWARD(plan_r2c, buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Normalize by 1/N
    {
        double inv_N = 1.0 / (double)(NX * NY * NZ);
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(buf), nc_local, inv_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Download Y-slab complex data
    {
        std::vector<GCplx> host(nc_local);
        CUDA_CHECK(cudaMemcpy(host.data(), gpu_ptr(buf),
                              nc_local * sizeof(GCplx), cudaMemcpyDeviceToHost));

        // -----------------------------------------------
        // Find (kx=0, ky=0, kz=0)
        // Layout A  [gx][local_y][kz]:
        //   index = gx * ny_local * NZC + local_y * NZC + kz
        //   (0,0,0) on rank with y_offset=0 → local_y=0 → idx=0
        // -----------------------------------------------
        GCplx mode000 = {0.0, 0.0};
        if (y_offset == 0) {        // rank holding ky=0
            mode000 = host[0];      // Layout A prediction: idx=0
        }

        // Broadcast mode000 from the rank that has it (rank 0 always has y_offset=0)
        MPI_Bcast(&mode000, sizeof(GCplx), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Maximum magnitude over ALL other modes on this rank
        double local_max_other = 0.0;
        for (long long i = 0; i < nc_local; ++i) {
            // Skip the (0,0,0) entry on rank 0
            if (y_offset == 0 && i == 0) continue;
            double mag = sqrt(host[i].x * host[i].x + host[i].y * host[i].y);
            local_max_other = std::max(local_max_other, mag);
        }
        double global_max_other = 0.0;
        MPI_Allreduce(&local_max_other, &global_max_other, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            double mag000 = sqrt(mode000.x * mode000.x + mode000.y * mode000.y);
            double expected_broken = 1.0 / nprocs;
            printf("  V_hat(0,0,0)  = (%+.10f, %+.10f)  |V| = %.10f\n",
                   mode000.x, mode000.y, mag000);
            printf("  Max other mode = %.6e  (should be ~0)\n", global_max_other);
            printf("\n");
            if (fabs(mag000 - 1.0) < 1e-6) {
                printf("  RESULT A: PASS ✓  V_hat(0,0,0) = 1.0 → transpose OK\n");
            } else if (fabs(mag000 - expected_broken) < 1e-4) {
                printf("  RESULT A: FAIL ✗  V_hat(0,0,0) ≈ 1/nprocs = %.6f\n",
                       expected_broken);
                printf("            All-to-all transpose is BROKEN.\n");
                printf("            Each rank only processed its own local data.\n");
            } else {
                printf("  RESULT A: UNEXPECTED  |V_hat(0,0,0)| = %.10f\n", mag000);
                printf("            (not 1.0 nor 1/nprocs -- investigate further)\n");
            }
            fflush(stdout);
        }

        // Per-rank detail printout (useful for diagnosing which ranks get zeros)
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0; r < nprocs; ++r) {
            if (r == rank) {
                // Find largest few entries on this rank
                printf("  [rank %d] y_offset=%d  nc_local=%lld\n",
                       rank, y_offset, nc_local);
                // Print top-5 entries by magnitude
                std::vector<std::pair<double,long long>> entries;
                for (long long i = 0; i < nc_local; ++i) {
                    double mag = sqrt(host[i].x*host[i].x + host[i].y*host[i].y);
                    if (mag > 1e-8)
                        entries.push_back({mag, i});
                }
                std::sort(entries.rbegin(), entries.rend());
                int shown = 0;
                for (auto& [mag, i] : entries) {
                    // Decode Layout-A [gx][local_y][kz]
                    long long gx     = i / ((long long)ny_local * NZC);
                    long long local_y= (i / NZC) % ny_local;
                    long long kz     = i % NZC;
                    long long gy     = y_offset + local_y;
                    printf("    idx=%lld  (gx=%lld gy=%lld kz=%lld)  "
                           "V=(%+.8f %+.8f)  |V|=%.8f\n",
                           i, gx, gy, kz, host[i].x, host[i].y, mag);
                    if (++shown >= 5) { printf("    ...\n"); break; }
                }
                if (entries.empty())
                    printf("    (all entries < 1e-8 on this rank)\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // -----------------------------------------------------------------------
    // =======================================================================
    // TEST B: WITH warmup — one zero-data FFT_FORWARD+FFT_INVERSE first
    //         (replicates the warmup_buffers call in the original code)
    // =======================================================================
    // -----------------------------------------------------------------------
    if (rank == 0) {
        printf("\n--- TEST B: with warmup (zero-data F+I cycle first) ---\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup: same as warmup_buffers() in original
    CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_bytes(buf)));
    FFT_FORWARD(plan_r2c, buf);
    FFT_INVERSE(plan_c2r, buf);
    CUDA_CHECK(cudaMemset(gpu_ptr(buf), 0, gpu_bytes(buf)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fill constant 1.0
    {
        int gp = (int)((n_padded + BLOCK - 1) / BLOCK);
        kernel_fill_const_one<<<gp, BLOCK>>>((double*)gpu_ptr(buf), nx_local);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Forward FFT
    FFT_FORWARD(plan_r2c, buf);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Normalize
    {
        double inv_N = 1.0 / (double)(NX * NY * NZ);
        int gc = (int)((nc_local + BLOCK - 1) / BLOCK);
        kernel_scale_cplx<<<gc, BLOCK>>>((GCplx*)gpu_ptr(buf), nc_local, inv_N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Download and analyze
    {
        std::vector<GCplx> host(nc_local);
        CUDA_CHECK(cudaMemcpy(host.data(), gpu_ptr(buf),
                              nc_local * sizeof(GCplx), cudaMemcpyDeviceToHost));

        GCplx mode000 = {0.0, 0.0};
        if (y_offset == 0)
            mode000 = host[0];
        MPI_Bcast(&mode000, sizeof(GCplx), MPI_BYTE, 0, MPI_COMM_WORLD);

        double local_max_other = 0.0;
        for (long long i = 0; i < nc_local; ++i) {
            if (y_offset == 0 && i == 0) continue;
            double mag = sqrt(host[i].x * host[i].x + host[i].y * host[i].y);
            local_max_other = std::max(local_max_other, mag);
        }
        double global_max_other = 0.0;
        MPI_Allreduce(&local_max_other, &global_max_other, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            double mag000 = sqrt(mode000.x * mode000.x + mode000.y * mode000.y);
            double expected_broken = 1.0 / nprocs;
            printf("  V_hat(0,0,0)  = (%+.10f, %+.10f)  |V| = %.10f\n",
                   mode000.x, mode000.y, mag000);
            printf("  Max other mode = %.6e  (should be ~0)\n", global_max_other);
            printf("\n");
            if (fabs(mag000 - 1.0) < 1e-6) {
                printf("  RESULT B: PASS ✓  V_hat(0,0,0) = 1.0 → transpose OK\n");
            } else if (fabs(mag000 - expected_broken) < 1e-4) {
                printf("  RESULT B: FAIL ✗  V_hat(0,0,0) ≈ 1/nprocs = %.6f\n",
                       expected_broken);
                printf("            All-to-all transpose is BROKEN.\n");
            } else {
                printf("  RESULT B: UNEXPECTED  |V_hat(0,0,0)| = %.10f\n", mag000);
            }
            fflush(stdout);
        }

        // Per-rank detail
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0; r < nprocs; ++r) {
            if (r == rank) {
                printf("  [rank %d] y_offset=%d  nc_local=%lld\n",
                       rank, y_offset, nc_local);
                std::vector<std::pair<double,long long>> entries;
                for (long long i = 0; i < nc_local; ++i) {
                    double mag = sqrt(host[i].x*host[i].x + host[i].y*host[i].y);
                    if (mag > 1e-8)
                        entries.push_back({mag, i});
                }
                std::sort(entries.rbegin(), entries.rend());
                int shown = 0;
                for (auto& [mag, i] : entries) {
                    long long gx     = i / ((long long)ny_local * NZC);
                    long long local_y= (i / NZC) % ny_local;
                    long long kz     = i % NZC;
                    long long gy     = y_offset + local_y;
                    printf("    idx=%lld  (gx=%lld gy=%lld kz=%lld)  "
                           "V=(%+.8f %+.8f)  |V|=%.8f\n",
                           i, gx, gy, kz, host[i].x, host[i].y, mag);
                    if (++shown >= 5) { printf("    ...\n"); break; }
                }
                if (entries.empty())
                    printf("    (all entries < 1e-8 on this rank)\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (rank == 0) {
        printf("\n============================================================\n");
        printf("  Interpretation guide:\n");
        printf("  A=PASS  B=PASS → transpose OK in both cases\n");
        printf("  A=FAIL  B=FAIL → transpose broken; subFormat hack not the cause\n");
        printf("                   → check NVSHMEM / MPI topology / environment\n");
        printf("  A=FAIL  B=PASS → warmup fixed it; warmup_buffers() is load-bearing\n");
        printf("                   → the lazy-init hypothesis is confirmed\n");
        printf("  A=PASS  B=FAIL → very unusual; report raw values above\n");
        printf("============================================================\n");
        fflush(stdout);
    }

    // Cleanup
    CUFFT_CHECK(cufftXtFree(buf));
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
    MPI_Finalize();
    return 0;
}
