/**
 * fftbench_cufftmp.cu — pure FFT benchmark for cuFFTMp (no Navier-Stokes physics).   [v2]
 *
 * Same FFT setup as NavierStokes_periodic_cufftmp_multinode.cu:
 *   D2Z: real X-slab (INPLACE) -> complex Y-slab (INPLACE_SHUFFLED), one global transpose
 *   Z2D: complex Y-slab -> real X-slab, one global transpose
 *   forward / inverse plans share one NVSHMEM work area (FB_SHARE_WS=0 disables).
 *
 * Question it answers: when the FFT leaves one node, where does the time go, and how much of
 * the InfiniBand bandwidth is really used?
 *
 * Measured (per single 3D FFT, max over ranks):
 *   [A] dist    : cuFFTMp forward / inverse on NF fields, as in the solver (one field per call).
 *                 Forward and inverse are printed separately: both must cost about the same,
 *                 otherwise one direction does more than one transpose.
 *   [B] local   : estimate of the purely local FFT work of the slab algorithm with plain cuFFT,
 *                 no communication: 2D R2C over (Y,Z) on the X-slab + 1D C2C along X on the
 *                 Y-slab (layout [X][Y/P][Z/2+1]).  An ESTIMATE: cuFFTMp's own local kernels
 *                 may be faster or slower than this.
 *   [C] a2a     : MPI_Alltoall (CUDA-aware MPI / UCX) on GPU buffers, same volume and per-pair
 *                 message size as one transpose = what an MPI-based library (heFFTe) can reach.
 *   [D] nvshmem : optional (FB_NVSHMEM_A2A=1) nvshmemx_alltoallmem_on_stream, same volume.
 *   [R] reshape : cuFFTMp standalone reshape X-slab -> Y-slab of the complex array = exactly one
 *                 global transpose and NO FFT at all.  The most direct bandwidth measurement.
 *
 * Inter-node data per node for one transpose (n nodes, V = NX*NY*(NZ/2+1)*16 B):
 *   D = V/n^2 * (n-1)        bandwidth = D / time
 *   from [A]          : lower bound (time contains the local FFT)
 *   from [A] - [B]    : local-corrected estimate
 *   from [R]          : direct (contains only pack + transfer)
 *
 * Documented switches worth testing on InfiniBand (set in the environment, recorded in the output):
 *   CUFFT_RESHAPE_USE_PACKING=1   cuFFTMp 11.2.6 API reference, cufftMpMakeReshape: "new experimental
 *                                 multi-node implementation ... improved performances over Infiniband"
 *   NVSHMEM_IB_SL=1               cuFFTMp "Performance considerations": adaptive routing
 *   NVSHMEM_IB_ENABLE_IBGDA=1     NVSHMEM GPU-initiated transport (needs driver/NIC support)
 *
 * Also reports GPU memory taken by plans / work area / buffers (plan memory).
 *
 * Usage: fftbench_cufftmp NX NY NZ [iters=20] [nfields=3]
 * Env:   FB_SHARE_WS=0|1 (default 1)  FB_CHECK=0  FB_SKIP_LOCAL=1  FB_SKIP_A2A=1  FB_SKIP_RESHAPE=1
 *        FB_NVSHMEM_A2A=1  FB_PEAK_NODE_GBS (default 200)
 * Build: -DFB_NO_NVSHMEM_A2A if nvshmemx_alltoallmem_on_stream is missing,
 *        -DFB_NO_RESHAPE     if the standalone reshape API does not compile with this cuFFTMp.
 *
 * Every phase prints its own "RES_x" line as soon as it is done, so a failure in a later phase
 * does not lose the earlier numbers; the full "RESULT" line comes at the end.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cufftMp.h>

#ifdef __CUDACC__
#define KLAUNCH(kernel, grid, block) kernel<<<(grid), (block)>>>
#else   // host-only syntax check / mock run: kernels are not executed
template <class... A> static void fb_noop(A&&...) {}
#define KLAUNCH(kernel, grid, block) fb_noop
#endif

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define CUFFT_CHECK(e) do { cufftResult _e=(e); if(_e!=CUFFT_SUCCESS){ \
    fprintf(stderr,"cuFFT error %s:%d: %d\n",__FILE__,__LINE__,(int)_e); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

static int g_rank = 0, g_np = 1;
static const double GB  = 1e9;            // decimal GB, as in the bandwidth table
static const double GiB = 1024.0*1024.0*1024.0;

static int env_int(const char* n, int def) { const char* v = getenv(n); return v ? atoi(v) : def; }
static double env_dbl(const char* n, double def) { const char* v = getenv(n); return v ? atof(v) : def; }
static bool env_on(const char* n) { const char* v = getenv(n); return v && *v && strcmp(v, "0") && strcmp(v, "false"); }

static double used_gib() {
    size_t f = 0, t = 0; CUDA_CHECK(cudaMemGetInfo(&f, &t));
    double u = (double)(t - f) / GiB, m;
    MPI_Allreduce(&u, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return m;
}
static double max_all(double v) { double m; MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return m; }
// true only if the step succeeded on every rank: keeps all ranks on the same branch
static bool all_ok(bool ok) { int a = ok ? 1 : 0, b = 0; MPI_Allreduce(&a, &b, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD); return b != 0; }

static inline void* gptr(cudaLibXtDesc* d) { return d->descriptor->data[0]; }
static inline size_t gbytes(cudaLibXtDesc* d) { return d->descriptor->size[0]; }

static void fwd(cufftHandle p, cudaLibXtDesc* b) {
    b->subFormat = CUFFT_XT_FORMAT_INPLACE;
    CUFFT_CHECK(cufftXtExecDescriptor(p, b, b, CUFFT_FORWARD));
}
static void inv(cufftHandle p, cudaLibXtDesc* b) {
    b->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    CUFFT_CHECK(cufftXtExecDescriptor(p, b, b, CUFFT_INVERSE));
}

// Test field on the real X-slab. In-place D2Z layout: row length 2*(NZ/2+1) doubles.
__global__ void k_fill(double* r, int nxl, int x0, int NX, int NY, int NZ, int NZP) {
    long long n = (long long)nxl * NY * NZ;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int z = (int)(i % NZ), y = (int)((i / NZ) % NY), x = (int)(i / ((long long)NZ * NY));
    double h = 2.0 * M_PI;
    double v = sin(h * (x + x0) / NX) * cos(2.0 * h * y / NY) + 0.5 * cos(3.0 * h * z / NZ);
    r[((long long)x * NY + y) * NZP + z] = v;
}
__global__ void k_err(const double* r, double* e, int nxl, int x0, int NX, int NY, int NZ, int NZP, double sc) {
    long long n = (long long)nxl * NY * NZ;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int z = (int)(i % NZ), y = (int)((i / NZ) % NY), x = (int)(i / ((long long)NZ * NY));
    double h = 2.0 * M_PI;
    double v = sin(h * (x + x0) / NX) * cos(2.0 * h * y / NY) + 0.5 * cos(3.0 * h * z / NZ);
    e[i] = fabs(r[((long long)x * NY + y) * NZP + z] * sc - v);
}
__global__ void k_max(const double* e, long long n, double* out) {
    __shared__ double s[256];
    double m = 0.0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        m = fmax(m, e[i]);
    s[threadIdx.x] = m; __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + k]); __syncthreads(); }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_np);
    if (argc < 4) {
        if (!g_rank) fprintf(stderr, "Usage: %s NX NY NZ [iters=20] [nfields=3]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int NX = atoi(argv[1]), NY = atoi(argv[2]), NZ = atoi(argv[3]);
    const int ITERS = argc > 4 ? atoi(argv[4]) : 20;
    const int NF    = argc > 5 ? atoi(argv[5]) : 3;
    const int NZC = NZ / 2 + 1, NZP = 2 * NZC;
    const int P = g_np;
    if (NX % P || NY % P) { if (!g_rank) fprintf(stderr, "nprocs=%d must divide NX and NY\n", P); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (ITERS < 1 || NF < 1) { if (!g_rank) fprintf(stderr, "iters and nfields must be >= 1\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    const int nxl = NX / P, nyl = NY / P, x0 = g_rank * nxl;

    // ---- node / GPU assignment (same as the solver) ----
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node_comm);
    int lrank, lsize; MPI_Comm_rank(node_comm, &lrank); MPI_Comm_size(node_comm, &lsize);
    MPI_Comm_free(&node_comm);
    int lsize_min; MPI_Allreduce(&lsize, &lsize_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const int nnodes = P / lsize;
    CUDA_CHECK(cudaSetDevice(lrank));
    CUDA_CHECK(cudaFree(0));

    const int share_ws = env_int("FB_SHARE_WS", 1);
    const double peak_node = env_dbl("FB_PEAK_NODE_GBS", 200.0);
    const double V = (double)NX * NY * NZC * 16.0;              // complex array, bytes
    const double D_node = nnodes > 1 ? V / ((double)nnodes * nnodes) * (nnodes - 1) : 0.0;
    const double D_gpu_all = V / ((double)P * P) * (P - 1);     // bytes each GPU sends to all other GPUs
    const double D_gpu_ib  = D_node / lsize;                    // bytes each GPU sends over InfiniBand

    // which documented switches are active in this run (same environment on all ranks)
    std::string knobs;
    if (env_on("CUFFT_RESHAPE_USE_PACKING")) knobs += "pack";
    if (env_int("NVSHMEM_IB_SL", 0) != 0)    knobs += std::string(knobs.empty() ? "" : "+") + "sl" + std::to_string(env_int("NVSHMEM_IB_SL", 0));
    if (env_on("NVSHMEM_IB_ENABLE_IBGDA"))   knobs += std::string(knobs.empty() ? "" : "+") + "ibgda";
    if (knobs.empty()) knobs = "default";

    if (!g_rank) {
        printf("============================================================\n");
        printf(" cuFFTMp pure FFT benchmark  %d x %d x %d   ranks=%d  nodes=%d  GPUs/node=%d\n", NX, NY, NZ, P, nnodes, lsize);
        printf(" iters=%d  fields per batch=%d  share_ws=%d  switches=%s\n", ITERS, NF, share_ws, knobs.c_str());
        printf(" V (complex array) = %.3f GB; per-pair message = %.2f MB; inter-node data per FFT: %.3f GB per node = %.3f GB per GPU\n",
               V / GB, V / ((double)P * P) / 1e6, D_node / GB, D_gpu_ib / GB);
        if (nnodes > 1)
            printf(" time on the wire for that volume at 100%% of %.0f GB/s per node: %.2f ms per FFT (no software can go below this)\n",
                   peak_node, D_node / peak_node / GB * 1e3);
        if (lsize_min != lsize) printf(" WARNING: uneven ranks per node; node count is approximate\n");
        printf("============================================================\n");
        fflush(stdout);
    }

    // ---- memory: context ----
    const double m_ctx = used_gib();

    // ---- plans (identical sequence to the solver) ----
    cufftHandle pf, pi;
    CUFFT_CHECK(cufftCreate(&pf)); CUFFT_CHECK(cufftCreate(&pi));
    MPI_Comm world = MPI_COMM_WORLD;
    CUFFT_CHECK(cufftMpAttachComm(pf, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftMpAttachComm(pi, CUFFT_COMM_MPI, &world));
    CUFFT_CHECK(cufftXtSetSubformatDefault(pf, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUFFT_CHECK(cufftXtSetSubformatDefault(pi, CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_XT_FORMAT_INPLACE));
    if (share_ws) { CUFFT_CHECK(cufftSetAutoAllocation(pf, 0)); CUFFT_CHECK(cufftSetAutoAllocation(pi, 0)); }
    size_t wsf = 0, wsi = 0;
    CUFFT_CHECK(cufftMakePlan3d(pf, NX, NY, NZ, CUFFT_D2Z, &wsf));
    CUFFT_CHECK(cufftMakePlan3d(pi, NX, NY, NZ, CUFFT_Z2D, &wsi));
    const double m_plans = used_gib();
    long long ws_shared = (long long)std::max(wsf, wsi);
    MPI_Allreduce(MPI_IN_PLACE, &ws_shared, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    void* wa = nullptr;
    if (share_ws && ws_shared > 0) {
        wa = nvshmem_malloc((size_t)ws_shared);
        if (!wa) { fprintf(stderr, "rank %d: nvshmem_malloc work area failed\n", g_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUFFT_CHECK(cufftSetWorkArea(pf, wa)); CUFFT_CHECK(cufftSetWorkArea(pi, wa));
    }
    const double m_wa = used_gib();

    std::vector<cudaLibXtDesc*> buf(NF, nullptr);
    for (int f = 0; f < NF; ++f) {
        CUFFT_CHECK(cufftXtMalloc(pf, &buf[f], CUFFT_XT_FORMAT_INPLACE));
        CUDA_CHECK(cudaMemset(gptr(buf[f]), 0, gbytes(buf[f])));
    }
    // warm-up every buffer (the solver needs this before relying on a set subFormat)
    for (int f = 0; f < NF; ++f) { fwd(pf, buf[f]); inv(pi, buf[f]); }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double m_desc = used_gib();
    const double buf_gib = (double)gbytes(buf[0]) / GiB;

    if (!g_rank) {
        printf("[mem] per GPU (max over ranks), GiB: context %.3f | after 2 plans %.3f (+%.3f) | after shared work area %.3f (+%.3f) | after %d buffers + warm-up %.3f (+%.3f, %.3f per buffer)\n",
               m_ctx, m_plans, m_plans - m_ctx, m_wa, m_wa - m_plans, NF, m_desc, m_desc - m_wa, buf_gib);
        printf("[mem] cufftMakePlan3d work size: D2Z %.3f GiB, Z2D %.3f GiB; shared area %.3f GiB\n",
               wsf / GiB, wsi / GiB, share_ws ? ws_shared / GiB : 0.0);
        printf("[mem] note: the 'after 2 plans' step also contains NVSHMEM initialisation (symmetric heap reservation)\n");
        fflush(stdout);
    }

    // ---- correctness check: fill, forward, inverse, compare (scale 1/N) ----
    if (env_int("FB_CHECK", 1)) {
        long long nr = (long long)nxl * NY * NZ;
        double* r = (double*)gptr(buf[0]);
        int gb = (int)((nr + 255) / 256);
        KLAUNCH(k_fill, gb, 256)(r, nxl, x0, NX, NY, NZ, NZP);
        fwd(pf, buf[0]); inv(pi, buf[0]);
        double *e, *pm; CUDA_CHECK(cudaMalloc((void**)&e, nr * sizeof(double))); CUDA_CHECK(cudaMalloc((void**)&pm, 1024 * sizeof(double)));
        KLAUNCH(k_err, gb, 256)(r, e, nxl, x0, NX, NY, NZ, NZP, 1.0 / ((double)NX * NY * NZ));
        KLAUNCH(k_max, 1024, 256)(e, nr, pm);
        std::vector<double> h(1024); CUDA_CHECK(cudaMemcpy(h.data(), pm, 1024 * sizeof(double), cudaMemcpyDeviceToHost));
        double em = *std::max_element(h.begin(), h.end()), eg = max_all(em);
        if (!g_rank) { printf("[check] forward+inverse max error = %.3e %s\n", eg, eg < 1e-10 ? "(OK)" : "(FAIL)"); fflush(stdout); }
        CUDA_CHECK(cudaFree(e)); CUDA_CHECK(cudaFree(pm));
        CUDA_CHECK(cudaMemset(r, 0, gbytes(buf[0])));
    }

    // ---- [A] distributed FFT ----
    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1)); CUDA_CHECK(cudaEventCreate(&e2));
    for (int w = 0; w < 2; ++w) for (int f = 0; f < NF; ++f) { fwd(pf, buf[f]); inv(pi, buf[f]); }
    CUDA_CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    double tf = 0, ti = 0, tw0 = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        CUDA_CHECK(cudaEventRecord(e0));
        for (int f = 0; f < NF; ++f) fwd(pf, buf[f]);
        CUDA_CHECK(cudaEventRecord(e1));
        for (int f = 0; f < NF; ++f) inv(pi, buf[f]);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaEventSynchronize(e2));
        float a, b; CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1)); CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
        tf += a * 1e-3; ti += b * 1e-3;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double twall = MPI_Wtime() - tw0;
    const double nfft = (double)ITERS * NF;
    const double t_fwd = max_all(tf / nfft), t_inv = max_all(ti / nfft), t_wall = max_all(twall / (2 * nfft));
    const double t_dist = 0.5 * (t_fwd + t_inv);
    if (!g_rank) {
        printf("[A dist ] per FFT: forward %.3f ms, inverse %.3f ms, mean %.3f ms (wall-clock mean %.3f ms)\n",
               t_fwd * 1e3, t_inv * 1e3, t_dist * 1e3, t_wall * 1e3);
        if (std::max(t_fwd, t_inv) > 1.5 * std::min(t_fwd, t_inv))
            printf("          WARNING: forward and inverse differ by more than 1.5x - one direction may do an extra transpose\n");
        if (t_wall > 1.2 * t_dist || t_dist > 1.2 * t_wall)
            printf("          WARNING: event time and wall-clock time differ by more than 20%% - trust the wall-clock value\n");
        printf("          36 FFT per step -> %.4f s/step of FFT\n", 36 * t_dist);
        if (nnodes > 1)
            printf("          bandwidth from this time (lower bound): %.1f GB/s per node = %.1f%% of %.0f\n",
                   D_node / t_dist / GB, 100 * D_node / t_dist / GB / peak_node, peak_node);
        printf("RES_A lib=cufftmp N=%dx%dx%d P=%d nodes=%d nf=%d knobs=%s t_fwd_ms=%.4f t_inv_ms=%.4f t_dist_ms=%.4f t_wall_ms=%.4f D_node_GB=%.4f B_dist=%.2f\n",
               NX, NY, NZ, P, nnodes, NF, knobs.c_str(), t_fwd * 1e3, t_inv * 1e3, t_dist * 1e3, t_wall * 1e3, D_node / GB,
               nnodes > 1 ? D_node / t_dist / GB : 0.0);
        fflush(stdout);
    }
    // the FFT buffers are not needed any more: release them before the other tests
    for (int f = 0; f < NF; ++f) { CUFFT_CHECK(cufftXtFree(buf[f])); buf[f] = nullptr; }

    // ---- [B] local FFT only (plain cuFFT, no communication) ----
    double t_loc = 0.0, t_locf = 0.0, t_loci = 0.0;
    if (!env_int("FB_SKIP_LOCAL", 0)) {
        long long nrl = (long long)nxl * NY * NZ, ncx = (long long)nxl * NY * NZC, ncy = (long long)NX * nyl * NZC;
        double* rin = nullptr; cufftDoubleComplex *cx = nullptr, *cy = nullptr;
        cufftHandle p2f = 0, p2i = 0, p1 = 0;
        int n2[2] = {NY, NZ}, inr[2] = {NY, NZ}, inc[2] = {NY, NZC};
        int n1[1] = {NX}, S = nyl * NZC, emb[1] = {NX};
        bool ok = cudaMalloc((void**)&rin, nrl * sizeof(double)) == cudaSuccess
               && cudaMalloc((void**)&cx, std::max(ncx, ncy) * sizeof(cufftDoubleComplex)) == cudaSuccess
               && cudaMalloc((void**)&cy, ncy * sizeof(cufftDoubleComplex)) == cudaSuccess;
        // 2D over (Y,Z) for each of the nxl X-planes; 1D along X on the Y-slab: stride = nyl*NZC, distance 1
        bool p_ok[3] = {false, false, false};
        if (ok) {
            p_ok[0] = cufftPlanMany(&p2f, 2, n2, inr, 1, NY * NZ, inc, 1, NY * NZC, CUFFT_D2Z, nxl) == CUFFT_SUCCESS;
            p_ok[1] = p_ok[0] && cufftPlanMany(&p2i, 2, n2, inc, 1, NY * NZC, inr, 1, NY * NZ, CUFFT_Z2D, nxl) == CUFFT_SUCCESS;
            p_ok[2] = p_ok[1] && cufftPlanMany(&p1, 1, n1, emb, S, 1, emb, S, 1, CUFFT_Z2Z, S) == CUFFT_SUCCESS;
            ok = p_ok[2];
        }
        if (all_ok(ok)) {
            CUDA_CHECK(cudaMemset(rin, 0, nrl * sizeof(double)));
            CUDA_CHECK(cudaMemset(cy, 0, ncy * sizeof(cufftDoubleComplex)));
            auto run_f = [&]() { CUFFT_CHECK(cufftExecD2Z(p2f, rin, cx)); CUFFT_CHECK(cufftExecZ2Z(p1, cy, cy, CUFFT_FORWARD)); };
            auto run_i = [&]() { CUFFT_CHECK(cufftExecZ2Z(p1, cy, cy, CUFFT_INVERSE)); CUFFT_CHECK(cufftExecZ2D(p2i, cx, rin)); };
            for (int w = 0; w < 2; ++w) { run_f(); run_i(); }
            CUDA_CHECK(cudaDeviceSynchronize());
            double a_s = 0, b_s = 0;
            for (int it = 0; it < ITERS; ++it) {
                CUDA_CHECK(cudaEventRecord(e0));
                for (int f = 0; f < NF; ++f) run_f();
                CUDA_CHECK(cudaEventRecord(e1));
                for (int f = 0; f < NF; ++f) run_i();
                CUDA_CHECK(cudaEventRecord(e2));
                CUDA_CHECK(cudaEventSynchronize(e2));
                float a, b; CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1)); CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
                a_s += a * 1e-3; b_s += b * 1e-3;
            }
            t_locf = max_all(a_s / nfft); t_loci = max_all(b_s / nfft); t_loc = 0.5 * (t_locf + t_loci);
            if (!g_rank) {
                printf("[B local] per FFT (2D R2C on X-slab + 1D C2C along X, no communication): forward %.3f ms, inverse %.3f ms, mean %.3f ms (%.0f%% of dist)\n",
                       t_locf * 1e3, t_loci * 1e3, t_loc * 1e3, 100.0 * t_loc / t_dist);
                printf("RES_B t_local_ms=%.4f\n", t_loc * 1e3);
                fflush(stdout);
            }
        } else if (!g_rank) {
            printf("[B local] skipped: not enough free GPU memory for the local test arrays / plans on at least one rank\n");
            fflush(stdout);
        }
        if (p_ok[0]) cufftDestroy(p2f);
        if (p_ok[1]) cufftDestroy(p2i);
        if (p_ok[2]) cufftDestroy(p1);
        if (rin) cudaFree(rin);
        if (cx) cudaFree(cx);
        if (cy) cudaFree(cy);
    }
    // local-corrected communication time; only meaningful if the local estimate is clearly smaller than dist
    const bool have_comm = t_loc > 0 && t_loc < 0.9 * t_dist;
    const double t_comm = have_comm ? t_dist - t_loc : 0.0;

    // ---- [C] MPI_Alltoall of the same volume (GPU buffers, CUDA-aware MPI) ----
    double t_a2a = 0.0;
    const size_t per_pair = (size_t)((long long)NX * nyl * NZC * 16 / P);   // = V/P^2 bytes
    if (!env_int("FB_SKIP_A2A", 0) && P > 1) {
        size_t cnt = per_pair / sizeof(double);
        double *sb = nullptr, *rb = nullptr;
        bool ok = cudaMalloc((void**)&sb, per_pair * P) == cudaSuccess && cudaMalloc((void**)&rb, per_pair * P) == cudaSuccess;
        if (all_ok(ok)) {
            CUDA_CHECK(cudaMemset(sb, 0, per_pair * P));
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int w = 0; w < 2; ++w) MPI_Alltoall(sb, (int)cnt, MPI_DOUBLE, rb, (int)cnt, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int it = 0; it < ITERS; ++it) MPI_Alltoall(sb, (int)cnt, MPI_DOUBLE, rb, (int)cnt, MPI_DOUBLE, MPI_COMM_WORLD);
            t_a2a = max_all((MPI_Wtime() - t0) / ITERS);
            if (!g_rank) {
                printf("[C a2a  ] MPI_Alltoall on GPU buffers, same volume (%.2f MB per pair): %.3f ms\n", per_pair / 1e6, t_a2a * 1e3);
                printf("RES_C t_a2a_ms=%.4f\n", t_a2a * 1e3);
                fflush(stdout);
            }
        } else if (!g_rank) {
            printf("[C a2a  ] skipped: not enough free GPU memory on at least one rank\n"); fflush(stdout);
        }
        if (sb) cudaFree(sb);
        if (rb) cudaFree(rb);
    }

    // ---- [D] optional NVSHMEM alltoall of the same volume ----
    double t_nv = 0.0;
#ifndef FB_NO_NVSHMEM_A2A
    if (env_int("FB_NVSHMEM_A2A", 0) && P > 1) {
        void* s = nvshmem_malloc(per_pair * P);
        void* d = nvshmem_malloc(per_pair * P);
        if (!all_ok(s && d)) {
            if (!g_rank) printf("[D nvshm] nvshmem_malloc of %.2f GiB failed: raise NVSHMEM_SYMMETRIC_SIZE\n", 2.0 * per_pair * P / GiB);
        } else {
            cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
            CUDA_CHECK(cudaMemset(s, 0, per_pair * P));
            for (int w = 0; w < 2; ++w) nvshmemx_alltoallmem_on_stream(NVSHMEM_TEAM_WORLD, d, s, per_pair, st);
            CUDA_CHECK(cudaStreamSynchronize(st)); MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int it = 0; it < ITERS; ++it) nvshmemx_alltoallmem_on_stream(NVSHMEM_TEAM_WORLD, d, s, per_pair, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            t_nv = max_all((MPI_Wtime() - t0) / ITERS);
            CUDA_CHECK(cudaStreamDestroy(st));
            if (!g_rank) { printf("[D nvshm] nvshmemx_alltoallmem_on_stream, same volume: %.3f ms\n", t_nv * 1e3); fflush(stdout); }
        }
        if (s) nvshmem_free(s);
        if (d) nvshmem_free(d);
    }
#endif

    // ---- [R] cuFFTMp standalone reshape: X-slab -> Y-slab of the complex array, nothing else ----
    double t_rs = 0.0, rs_ws_gib = 0.0;
#ifndef FB_NO_RESHAPE
    if (!env_int("FB_SKIP_RESHAPE", 0) && P > 1) {
        // boxes: lower corner inclusive, upper corner exclusive; strides in elements, C order (X slowest)
        const long long y0 = (long long)g_rank * nyl;
        long long lo_in[3]  = {x0, 0, 0},  hi_in[3]  = {(long long)x0 + nxl, NY, NZC};
        long long lo_out[3] = {0, y0, 0},  hi_out[3] = {NX, y0 + nyl, NZC};
        long long st_in[3]  = {(long long)NY * NZC, NZC, 1};
        long long st_out[3] = {(long long)nyl * NZC, NZC, 1};
        const size_t bytes = (size_t)nxl * NY * NZC * 16;        // = NX*nyl*NZC*16: both slabs have the same size
        cufftReshapeHandle rh;
        void *src = nullptr, *dst = nullptr, *scr = nullptr;
        size_t rws = 0;
        bool made = false;
        bool ok = cufftMpCreateReshape(&rh) == CUFFT_SUCCESS;
        if (ok) { made = true; ok = cufftMpAttachReshapeComm(rh, CUFFT_COMM_MPI, &world) == CUFFT_SUCCESS; }
        if (all_ok(ok)) {
            ok = cufftMpMakeReshape(rh, 16, 3, lo_in, hi_in, lo_out, hi_out, st_in, st_out) == CUFFT_SUCCESS;   // collective
            ok = ok && cufftMpGetReshapeSize(rh, &rws) == CUFFT_SUCCESS;
            if (all_ok(ok)) {
                long long rws_all = (long long)rws;
                MPI_Allreduce(MPI_IN_PLACE, &rws_all, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                rs_ws_gib = rws_all / GiB;
                src = nvshmem_malloc(bytes);
                dst = nvshmem_malloc(bytes);
                scr = nvshmem_malloc((size_t)std::max(rws_all, 16LL));
                if (all_ok(src && dst && scr)) {
                    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
                    CUDA_CHECK(cudaMemset(src, 0, bytes));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    MPI_Barrier(MPI_COMM_WORLD);
                    for (int w = 0; w < 2; ++w) {
                        CUFFT_CHECK(cufftMpExecReshapeAsync(rh, dst, src, scr, st));
                        CUDA_CHECK(cudaStreamSynchronize(st));
                        MPI_Barrier(MPI_COMM_WORLD);
                    }
                    double t0 = MPI_Wtime();
                    for (int it = 0; it < ITERS; ++it) {
                        CUFFT_CHECK(cufftMpExecReshapeAsync(rh, dst, src, scr, st));
                        CUDA_CHECK(cudaStreamSynchronize(st));
                        MPI_Barrier(MPI_COMM_WORLD);       // every rank has finished sending before the next round
                    }
                    t_rs = max_all((MPI_Wtime() - t0) / ITERS);
                    CUDA_CHECK(cudaStreamDestroy(st));
                    if (!g_rank) {
                        printf("[R resh ] cuFFTMp reshape X-slab -> Y-slab (one transpose, no FFT): %.3f ms; scratch %.3f GiB per GPU\n", t_rs * 1e3, rs_ws_gib);
                        if (nnodes > 1)
                            printf("          bandwidth: %.1f GB/s per node = %.1f%% of %.0f  (%.1f GB/s per GPU port)\n",
                                   D_node / t_rs / GB, 100 * D_node / t_rs / GB / peak_node, peak_node, D_gpu_ib / t_rs / GB);
                        printf("RES_R t_reshape_ms=%.4f B_reshape=%.2f\n", t_rs * 1e3, nnodes > 1 ? D_node / t_rs / GB : 0.0);
                        fflush(stdout);
                    }
                } else if (!g_rank) {
                    printf("[R resh ] skipped: nvshmem_malloc failed (%.2f GiB data x2 + %.2f GiB scratch per GPU)\n", bytes / GiB, rs_ws_gib);
                    fflush(stdout);
                }
                if (src) nvshmem_free(src);
                if (dst) nvshmem_free(dst);
                if (scr) nvshmem_free(scr);
            } else if (!g_rank) { printf("[R resh ] skipped: cufftMpMakeReshape / cufftMpGetReshapeSize failed\n"); fflush(stdout); }
        } else if (!g_rank) { printf("[R resh ] skipped: could not create the reshape handle\n"); fflush(stdout); }
        if (made) cufftMpDestroyReshape(rh);
    }
#endif

    // ---- summary ----
    if (!g_rank) {
        printf("------------------------------------------------------------\n");
        if (nnodes > 1) {
            auto bw = [&](double t) { return t > 0 ? D_node / t / GB : 0.0; };
            printf("Inter-node data per node per FFT: D = V/n^2*(n-1) = %.3f GB (peak %.0f GB/s per node)\n", D_node / GB, peak_node);
            printf("  B from dist time   (lower bound)       : %7.1f GB/s per node = %5.1f%% of peak\n", bw(t_dist), 100 * bw(t_dist) / peak_node);
            if (have_comm)
                printf("  B from dist - local (local-corrected)  : %7.1f GB/s per node = %5.1f%% of peak\n", bw(t_comm), 100 * bw(t_comm) / peak_node);
            else if (t_loc > 0)
                printf("  B from dist - local                    : not shown, the local estimate (%.3f ms) is not small against dist (%.3f ms)\n", t_loc * 1e3, t_dist * 1e3);
            if (t_rs > 0)
                printf("  B of the reshape alone (direct)        : %7.1f GB/s per node = %5.1f%% of peak\n", bw(t_rs), 100 * bw(t_rs) / peak_node);
            if (t_a2a > 0)
                printf("  B of MPI_Alltoall, same pattern        : %7.1f GB/s per node = %5.1f%% of peak\n", bw(t_a2a), 100 * bw(t_a2a) / peak_node);
            if (t_nv > 0)
                printf("  B of NVSHMEM alltoall                  : %7.1f GB/s per node = %5.1f%% of peak\n", bw(t_nv), 100 * bw(t_nv) / peak_node);
        } else {
            printf("Single node: data each GPU sends to the other GPUs per FFT = %.3f GB (NVLink/NVSwitch)\n", D_gpu_all / GB);
            printf("  per-GPU send rate from dist time : %.1f GB/s", D_gpu_all / t_dist / GB);
            if (have_comm) printf("; from dist - local: %.1f GB/s", D_gpu_all / t_comm / GB);
            if (t_rs > 0) printf("; reshape alone: %.1f GB/s", D_gpu_all / t_rs / GB);
            printf("\n");
        }
        printf("RESULT lib=cufftmp N=%dx%dx%d P=%d nodes=%d nf=%d knobs=%s t_fwd_ms=%.4f t_inv_ms=%.4f t_dist_ms=%.4f t_local_ms=%.4f t_comm_ms=%.4f t_reshape_ms=%.4f t_a2a_ms=%.4f t_nvshmem_ms=%.4f D_node_GB=%.4f B_dist=%.2f B_comm=%.2f B_reshape=%.2f B_a2a=%.2f mem_plans_GiB=%.3f mem_wa_GiB=%.3f mem_buf_GiB=%.3f reshape_ws_GiB=%.3f\n",
               NX, NY, NZ, P, nnodes, NF, knobs.c_str(), t_fwd * 1e3, t_inv * 1e3, t_dist * 1e3, t_loc * 1e3, t_comm * 1e3, t_rs * 1e3, t_a2a * 1e3, t_nv * 1e3,
               D_node / GB,
               nnodes > 1 ? D_node / t_dist / GB : 0.0,
               (nnodes > 1 && have_comm) ? D_node / t_comm / GB : 0.0,
               (nnodes > 1 && t_rs > 0) ? D_node / t_rs / GB : 0.0,
               (nnodes > 1 && t_a2a > 0) ? D_node / t_a2a / GB : 0.0,
               m_plans - m_ctx, m_wa - m_plans, buf_gib, rs_ws_gib);
        fflush(stdout);
    }

    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1)); CUDA_CHECK(cudaEventDestroy(e2));
    // Order as in the solver: the work area must be freed BEFORE cufftDestroy, which finalizes NVSHMEM.
    if (wa) nvshmem_free(wa);
    CUFFT_CHECK(cufftDestroy(pf)); CUFFT_CHECK(cufftDestroy(pi));
    MPI_Finalize();
    return 0;
}
