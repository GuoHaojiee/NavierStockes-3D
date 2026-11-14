/**
 * fftbench_heffte.cpp — pure FFT benchmark for heFFTe (no Navier-Stokes physics).   [v2]
 *
 * One source, two builds:
 *   GPU (cufft backend): nvcc -x cu ...            (default)
 *   CPU (fftw backend) : mpicxx -DFB_CPU ...
 *
 * Purpose: find which heFFTe settings explain the gap to cuFFT / cuFFTMp / FFTW.
 * Everything is switched with environment variables, so one binary covers the sweep.
 *
 * Usage: fftbench_heffte NX NY NZ [iters=20] [nfields=3]
 *
 * Env (defaults reproduce the solver NavierStokes_periodic_heffte_multigpu.cu):
 *   FB_IN   = xslab | yslab | zslab | xpencil | ypencil | zpencil | brick   real-space split (default xslab: {P,1,1})
 *   FB_OUT  = same choices                                                  spectral split   (default yslab: {1,P,1})
 *             slab    : one dimension split            ({P,1,1} = xslab)
 *             pencil  : two dimensions split, the named one is whole (zpencil = {p,q,1})
 *             brick   : heffte::proc_setup_min_surface = all three dimensions split as soon as P has
 *                       three factors (128 ranks on 512^3 -> 4x4x8).  This is what the solvers
 *                       heffte_v2 (CPU) and heffte_gpu1 call "pencil grid"; with bricks on both sides
 *                       heFFTe needs FOUR MPI reshapes per transform (see heffte_plan_logic.h).
 *   FB_R2C  = 0 | 1 | 2                        r2c direction (0 = x = fastest index) (default 2)
 *   NS_HEFFTE_ALGO    = alltoallv | alltoall | p2p | p2p_plined      (default alltoallv)
 *   NS_HEFFTE_REORDER = 0 | 1                  (default: library default: cufft 0, fftw 1)
 *   FB_PENCILS = 0 | 1                         plan_options.use_pencils (default 1)
 *   FB_GPU_AWARE = 0 | 1                       plan_options.use_gpu_aware (default 1)
 *   FB_BATCH = 0 | 1                           1: all nfields in ONE batched call (default 0)
 *   FB_CHECK = 0 | 1                           forward+backward correctness check (default 1)
 *   FB_PEAK_NODE_GBS                           for the bandwidth percentage (default 200)
 *
 * Note: the solver layout is xslab -> yslab with r2c along z (dim 2). In heFFTe index 0 is the
 * fastest, so splitting along x and doing r2c along z means both are on strided dimensions.
 * The alternative "zslab -> yslab with FB_R2C=0" keeps the r2c on the contiguous dimension.
 * (The same memory pattern is available to the solver WITHOUT changing the physics: declare the
 *  boxes with order {2,1,0}, i.e. z fastest, as all the non-heFFTe versions already store data.)
 *
 * What heFFTe 2.4.1 does internally (read from its sources, relevant for interpreting the numbers):
 *   - fft3d_r2c always runs three separate 1-D passes; it never merges them into a 2-D / 3-D plan
 *     (only the c2c class fft3d does that), so on one GPU it cannot match a native cuFFT 3-D plan.
 *   - use_reorder = 0: the pass along the MIDDLE memory dimension is issued as one backend call per
 *     plane of the slowest dimension (NZ/2+1 separate cuFFT calls in the solver layout).
 *   - use_reorder = 1: every pass is contiguous, at the price of extra transposes of the whole array.
 *   - the FFTW backend plans with FFTW_ESTIMATE and NULL arrays (see est_vs_meas.cpp).
 *
 * The program prints the processor grid of every stage and the number of MPI reshapes of the plan.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <array>
#include <complex>
#include <algorithm>
#include <mpi.h>
#include <heffte.h>

#ifndef FB_CPU
#include <cuda_runtime.h>
#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
using backend_t = heffte::backend::cufft;
#else
using backend_t = heffte::backend::fftw;
#endif

using cplx = std::complex<double>;
static int g_rank = 0, g_np = 1;
static const double GB  = 1e9;
static const double GiB = 1024.0 * 1024.0 * 1024.0;

static const char* env_str(const char* n, const char* def) { const char* v = getenv(n); return v ? v : def; }
static int env_int(const char* n, int def) { const char* v = getenv(n); return v ? atoi(v) : def; }
static double env_dbl(const char* n, double def) { const char* v = getenv(n); return v ? atof(v) : def; }
static double max_all(double v) { double m; MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return m; }
static void dev_sync() {
#ifndef FB_CPU
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}
static double used_gib() {
#ifndef FB_CPU
    size_t f = 0, t = 0; CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return max_all((double)(t - f) / GiB);
#else
    return 0.0;
#endif
}

// ---- memory helpers (device or host) ----
template<typename T> static T* alloc(long long n) {
    T* p = nullptr;
#ifndef FB_CPU
    CUDA_CHECK(cudaMalloc((void**)&p, (size_t)std::max(n, 1LL) * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, (size_t)std::max(n, 1LL) * sizeof(T)));
#else
    p = new T[std::max(n, 1LL)]();
#endif
    return p;
}
template<typename T> static void dealloc(T* p) {
#ifndef FB_CPU
    cudaFree(p);
#else
    delete[] p;
#endif
}
template<typename T> static void to_dev(T* d, const std::vector<T>& h) {
#ifndef FB_CPU
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
#else
    std::copy(h.begin(), h.end(), d);
#endif
}
template<typename T> static void to_host(std::vector<T>& h, const T* d) {
#ifndef FB_CPU
    CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
#else
    std::copy(d, d + h.size(), h.begin());
#endif
}

static std::array<int, 3> grid_of(const std::string& s, const heffte::box3d<>& world, int P) {
    int p = 1;                                   // most square factorisation P = p * q, p <= q
    for (int i = 1; (long long)i * i <= P; ++i) if (P % i == 0) p = i;
    const int q = P / p;
    if (s == "xslab") return {P, 1, 1};
    if (s == "yslab") return {1, P, 1};
    if (s == "zslab") return {1, 1, P};
    if (s == "xpencil") return {1, p, q};        // whole along x
    if (s == "ypencil") return {p, 1, q};        // whole along y
    if (s == "zpencil") return {p, q, 1};        // whole along z
    if (s == "brick") return heffte::proc_setup_min_surface(world, P);
    if (!g_rank) fprintf(stderr, "Unknown layout '%s' (xslab|yslab|zslab|xpencil|ypencil|zpencil|brick)\n", s.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return {P, 1, 1};
}
static heffte::reshape_algorithm algo_of(const char* a) {
    if (!strcmp(a, "alltoallv"))  return heffte::reshape_algorithm::alltoallv;
    if (!strcmp(a, "alltoall"))   return heffte::reshape_algorithm::alltoall;
    if (!strcmp(a, "p2p"))        return heffte::reshape_algorithm::p2p;
    if (!strcmp(a, "p2p_plined")) return heffte::reshape_algorithm::p2p_plined;
    if (!g_rank) fprintf(stderr, "Unknown NS_HEFFTE_ALGO '%s'\n", a);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return heffte::reshape_algorithm::alltoallv;
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
    const int P = g_np;

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node_comm);
    int lrank, lsize; MPI_Comm_rank(node_comm, &lrank); MPI_Comm_size(node_comm, &lsize);
    MPI_Comm_free(&node_comm);
    const int nnodes = P / lsize;
#ifndef FB_CPU
    CUDA_CHECK(cudaSetDevice(lrank));
    CUDA_CHECK(cudaFree(0));
#endif
    const double m_ctx = used_gib();
    { // scope: heFFTe objects must be destroyed before MPI_Finalize

    // ---- configuration ----
    const std::string in_s  = env_str("FB_IN", "xslab");
    const std::string out_s = env_str("FB_OUT", "yslab");
    const int r2c = env_int("FB_R2C", 2);
    int batch = env_int("FB_BATCH", 0);
    if (r2c < 0 || r2c > 2) { if (!g_rank) fprintf(stderr, "FB_R2C must be 0,1,2\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    std::array<int, 3> N = {NX, NY, NZ};
    std::array<int, 3> Nc = N; Nc[r2c] = N[r2c] / 2 + 1;
    heffte::box3d<> world_r({0, 0, 0}, {NX - 1, NY - 1, NZ - 1});
    heffte::box3d<> world_c({0, 0, 0}, {Nc[0] - 1, Nc[1] - 1, Nc[2] - 1});
    auto pg_in  = grid_of(in_s, world_r, P);
    auto pg_out = grid_of(out_s, world_r, P);   // "brick": same grid on both sides, taken from the real box, as in the solvers
    heffte::box3d<> inbox  = heffte::split_world(world_r, pg_in)[g_rank];
    heffte::box3d<> outbox = heffte::split_world(world_c, pg_out)[g_rank];

    auto opt = heffte::default_options<backend_t>();
    if (const char* a = getenv("NS_HEFFTE_ALGO")) opt.algorithm = algo_of(a);
    if (const char* r = getenv("NS_HEFFTE_REORDER")) opt.use_reorder = (atoi(r) != 0);
    opt.use_pencils   = env_int("FB_PENCILS", 1) != 0;
    opt.use_gpu_aware = env_int("FB_GPU_AWARE", 1) != 0;

    // The plan heFFTe will build for these boxes, obtained the same way as heFFTe's own benchmark
    // (benchmarks/speed3d.h): processor grid of every stage; a stage whose grid differs from the
    // previous one needs an MPI reshape.
    std::string grids_s; int n_reshape = 0;
    {
        heffte::logic_plan3d<int> lp = heffte::plan_operations<int>(
            {heffte::split_world(world_r, pg_in), heffte::split_world(world_c, pg_out)}, r2c, opt, g_rank);
        std::vector<std::array<int, 3>> grids = heffte::compute_grids(lp);
        for (size_t i = 0; i < grids.size(); ++i) {
            const bool same = i > 0 && grids[i] == grids[i - 1];
            if (i > 0 && !same) ++n_reshape;
            grids_s += same ? " -" : " (" + std::to_string(grids[i][0]) + "," + std::to_string(grids[i][1]) + "," + std::to_string(grids[i][2]) + ")";
        }
    }
    // heFFTe 2.4.1 (and current master): compute_transform() for the forward r2c case with NO reshape
    // after stage 0 offsets the 2nd/3rd pass of batch entry j by j*real_size instead of j*complex_size.
    // That is the single-rank, use_reorder=0 case: run it field by field instead of batched.
    if (batch && P == 1 && !opt.use_reorder) {
        if (!g_rank) printf(" NOTE: FB_BATCH=1 ignored on one rank without reorder (batched r2c offset bug in heFFTe 2.4.1)\n");
        batch = 0;
    }

    heffte::fft3d_r2c<backend_t> fft(inbox, outbox, r2c, MPI_COMM_WORLD, opt);
    const double m_plan = used_gib();

    const long long nin = fft.size_inbox(), nout = fft.size_outbox(), nws = fft.size_workspace();
    const double V = (double)Nc[0] * Nc[1] * Nc[2] * 16.0;
    const double D_node = nnodes > 1 ? V / ((double)nnodes * nnodes) * (nnodes - 1) : 0.0;
    const double peak_node = env_dbl("FB_PEAK_NODE_GBS", 200.0);

    if (!g_rank) {
        printf("============================================================\n");
#ifndef FB_CPU
        printf(" heFFTe (cufft) pure FFT benchmark  %d x %d x %d   ranks=%d  nodes=%d  ranks/node=%d\n", NX, NY, NZ, P, nnodes, lsize);
#else
        printf(" heFFTe (fftw) pure FFT benchmark  %d x %d x %d   ranks=%d  nodes=%d  ranks/node=%d\n", NX, NY, NZ, P, nnodes, lsize);
#endif
        printf(" in=%s {%d,%d,%d}  out=%s {%d,%d,%d}  r2c_dir=%d  algo=%s  reorder=%d  pencils=%d  gpu_aware=%d  batch=%d  nfields=%d  iters=%d\n",
               in_s.c_str(), pg_in[0], pg_in[1], pg_in[2], out_s.c_str(), pg_out[0], pg_out[1], pg_out[2], r2c,
               env_str("NS_HEFFTE_ALGO", "alltoallv(default)"), opt.use_reorder ? 1 : 0, opt.use_pencils ? 1 : 0,
               opt.use_gpu_aware ? 1 : 0, batch, NF, ITERS);
        printf(" rank0 inbox lo=(%d,%d,%d) size=(%d,%d,%d); outbox lo=(%d,%d,%d) size=(%d,%d,%d)\n",
               inbox.low[0], inbox.low[1], inbox.low[2], inbox.size[0], inbox.size[1], inbox.size[2],
               outbox.low[0], outbox.low[1], outbox.low[2], outbox.size[0], outbox.size[1], outbox.size[2]);
        printf(" heFFTe plan, processor grid per stage (in, fft0, fft1, fft2, out):%s  ->  %d MPI reshape(s) per transform\n",
               grids_s.c_str(), n_reshape);
        printf("============================================================\n");
        fflush(stdout);
    }

    // ---- buffers: NF fields contiguous (needed by the batched call) ----
    double* r  = alloc<double>(nin * NF);
    cplx*   c  = alloc<cplx>(nout * NF);
    const int wsb = batch ? NF : 1;
    cplx*   ws = alloc<cplx>(nws * wsb);
    const double m_buf = used_gib();
    if (!g_rank) {
#ifndef FB_CPU
        printf("[mem] per GPU (max over ranks), GiB: context %.3f | after plan %.3f (+%.3f) | after %d real + %d complex fields + workspace %.3f (+%.3f)\n",
               m_ctx, m_plan, m_plan - m_ctx, NF, NF, m_buf, m_buf - m_plan);
#endif
        printf("[mem] per rank: real field %.3f GiB, complex field %.3f GiB, heFFTe workspace %.3f GiB (x%d) = %.2f complex fields\n",
               nin * 8.0 / GiB, nout * 16.0 / GiB, nws * 16.0 / GiB, wsb, (double)nws * wsb / std::max(nout, 1LL));
        fflush(stdout);
    }

    auto do_fwd = [&]() {
        if (batch) fft.forward(NF, r, c, ws, heffte::scale::none);
        else for (int f = 0; f < NF; ++f) fft.forward(r + f * nin, c + f * nout, ws, heffte::scale::none);
    };
    auto do_bwd = [&]() {
        if (batch) fft.backward(NF, c, r, ws, heffte::scale::full);
        else for (int f = 0; f < NF; ++f) fft.backward(c + f * nout, r + f * nin, ws, heffte::scale::full);
    };

    // ---- correctness check on all fields ----
    if (env_int("FB_CHECK", 1)) {
        std::vector<double> h(nin * NF);
        const double tp = 2.0 * M_PI;
        for (int f = 0; f < NF; ++f) {
            long long i = 0;
            for (int k = inbox.low[2]; k <= inbox.high[2]; ++k)
                for (int j = inbox.low[1]; j <= inbox.high[1]; ++j)
                    for (int ii = inbox.low[0]; ii <= inbox.high[0]; ++ii, ++i)
                        h[f * nin + i] = sin(tp * ii / NX) * cos(2.0 * tp * j / NY) + 0.5 * cos((3.0 + f) * tp * k / NZ);
        }
        to_dev(r, h);
        do_fwd(); do_bwd(); dev_sync();
        std::vector<double> g(nin * NF); to_host(g, r);
        double e = 0.0;
        for (long long i = 0; i < nin * NF; ++i) e = std::max(e, std::fabs(g[i] - h[i]));
        e = max_all(e);
        if (!g_rank) { printf("[check] forward+backward max error = %.3e %s\n", e, e < 1e-10 ? "(OK)" : "(FAIL)"); fflush(stdout); }
#ifndef FB_CPU
        CUDA_CHECK(cudaMemset(r, 0, nin * NF * sizeof(double)));
        CUDA_CHECK(cudaMemset(c, 0, nout * NF * sizeof(cplx)));
#else
        std::fill(r, r + nin * NF, 0.0); std::fill(c, c + nout * NF, cplx(0.0, 0.0));
#endif
    }

    // ---- timing ----
    for (int w = 0; w < 2; ++w) { do_fwd(); do_bwd(); }
    dev_sync(); MPI_Barrier(MPI_COMM_WORLD);
    double tf = 0.0, tb = 0.0;
    for (int it = 0; it < ITERS; ++it) {
        double t0 = MPI_Wtime();
        do_fwd(); dev_sync();
        double t1 = MPI_Wtime();
        do_bwd(); dev_sync();
        double t2 = MPI_Wtime();
        tf += t1 - t0; tb += t2 - t1;
    }
    const double nfft = (double)ITERS * NF;
    const double t_fwd = max_all(tf / nfft), t_bwd = max_all(tb / nfft), t_mean = 0.5 * (t_fwd + t_bwd);

    if (!g_rank) {
        printf("[time] per FFT: forward %.3f ms, backward %.3f ms, mean %.3f ms  -> 36 FFT/step = %.4f s/step\n",
               t_fwd * 1e3, t_bwd * 1e3, t_mean * 1e3, 36 * t_mean);
        if (nnodes > 1)
            printf("[bw  ] if one slab transpose per FFT: D = V/n^2*(n-1) = %.3f GB per node -> %.1f GB/s = %.1f%% of %.0f (lower bound)\n",
                   D_node / GB, D_node / t_mean / GB, 100.0 * D_node / t_mean / GB / peak_node, peak_node);
        printf("RESULT lib=heffte_%s N=%dx%dx%d P=%d nodes=%d in=%s out=%s r2c=%d algo=%s reorder=%d pencils=%d gpu_aware=%d batch=%d nf=%d reshapes=%d t_fwd_ms=%.4f t_bwd_ms=%.4f t_mean_ms=%.4f step_fft_s=%.4f B_node=%.2f ws_GiB=%.3f mem_plan_GiB=%.3f\n",
#ifndef FB_CPU
               "gpu",
#else
               "cpu",
#endif
               NX, NY, NZ, P, nnodes, in_s.c_str(), out_s.c_str(), r2c, env_str("NS_HEFFTE_ALGO", "alltoallv"),
               opt.use_reorder ? 1 : 0, opt.use_pencils ? 1 : 0, opt.use_gpu_aware ? 1 : 0, batch, NF, n_reshape,
               t_fwd * 1e3, t_bwd * 1e3, t_mean * 1e3, 36 * t_mean, nnodes > 1 ? D_node / t_mean / GB : 0.0,
               nws * wsb * 16.0 / GiB, m_plan - m_ctx);
        fflush(stdout);
    }

    dealloc(r); dealloc(c); dealloc(ws);
    } // heFFTe plan destroyed here
    MPI_Finalize();
    return 0;
}
