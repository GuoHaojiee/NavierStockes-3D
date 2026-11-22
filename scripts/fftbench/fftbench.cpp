// fftbench.cpp — pure 3D FFT benchmark for the four CPU libraries, with EXACTLY the solvers' plan and execute
// code: the FFT setup lives in fftlib/ns_fft_<lib>.h and is included both by NavierStokes_periodic_<lib>.cpp
// and by this program (NS_DECOMP, NS_FFTW_PLAN, r2c direction, transposed output, heFFTe backend and
// NS_HEFFTE_REORDER / NS_HEFFTE_ALGO all behave as in the solver).
//
// Build: one binary per library, -DFFTB_LIB=1 (FFTW) | 2 (AccFFT) | 3 (P3DFFT) | 4 (heFFTe; plus
//        -DNS_HEFFTE_BACKEND=fftw|mkl); see scripts/fftbench/build_fftb.sh.
// Usage: fftb NX NY NZ [threads]
//
// One launch:
//   setup + plan (timed; P3DFFT: Cp3dfft_setup; heFFTe-FFTW: the FFTW plans are made at the first execution,
//   i.e. during the check / warm-up)
//   -> check: x -> forward -> backward -> normalise; max |result - x| (OK below 1e-10); also whether the
//      forward transform left its real input unchanged
//   -> FFTB_WARM (2) warm-up rounds (one pair = forward + backward per field); the last sets
//      K = ceil(FFTB_WINDOW (3 s) / t_pair)
//      (rank 0's decision, broadcast; FFTB_PAIRS=K fixes it)
//   -> FFTB_REP (3) repetitions of K pairs, each timed barrier to barrier (max over ranks)
//   -> optional FFTB_SPLIT=1: the local FFT work alone (no communication), see local_model() below
// FFTB_NF (3) fields, each with its own real input, complex and real output: pair p uses field p % NF, so the
// transforms do not run on one cache-resident buffer set (the solver cycles through 21 full-size arrays).
// The data path of a field: forward from the same real input, backward into a second real buffer,
// as the solver (which never feeds a backward result straight into the next forward): values stay bounded.
//
// Per repetition (rank 0):
//   FFTB_REP rep=<r> pairs=<K> window_s=<W> t_fwd_s=<f> t_bwd_s=<b> t_pair_s=<W/K> t_fft_s=<(f+2b)/3> finite=<yes|no>
//   W = wall time of the K pairs; f and b split W/K in the ratio of the forward / backward time measured per
//   call (each max over ranks). t_fft_s weights them as the solver's step does (bench: 12 forward + 24
//   backward = 36 FFTs per RK4 step), so 36 * t_fft_s is the FFT time per solver step.
// At the end: FFTB_SUMMARY ... t_fft_median_s=... t_fft_min_s=... t_fft_max_s=...
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

#ifndef FFTB_LIB
#error "compile with -DFFTB_LIB=1 (FFTW) | 2 (AccFFT) | 3 (P3DFFT) | 4 (heFFTe)"
#endif
#if FFTB_LIB == 1
#include "fftlib/ns_fft_fftw.h"
using Engine = nsfft::Fftw;
#elif FFTB_LIB == 2
#include "fftlib/ns_fft_accfft.h"
using Engine = nsfft::Accfft;
#elif FFTB_LIB == 3
#include "fftlib/ns_fft_p3dfft.h"
using Engine = nsfft::P3dfft;
#elif FFTB_LIB == 4
#include "fftlib/ns_fft_heffte.h"
using Engine = nsfft::Heffte;
#else
#error "FFTB_LIB must be 1..4"
#endif
#include <fftw3.h>      // local_model() (FFTB_SPLIT=1)

static int g_rank = 0, g_np = 1;
static double max_all(double v) { double m; MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return m; }
static double env_dbl(const char* n, double d) { const char* v = getenv(n); return (v && *v) ? atof(v) : d; }
static int env_int(const char* n, int d) { const char* v = getenv(n); return (v && *v) ? atoi(v) : d; }
// ---- binding report (always, rank 0 prints one FFTB_BIND line; FFTB_SHOWBIND=1 also the CPU lists of ranks 0 and 1)
// Per rank: CPUs in the union of its threads' affinity masks; CPU of every OpenMP thread (sched_getcpu inside a parallel region, after a
// short spin so the threads have settled); distinct CPUs and distinct physical cores (sysfs core_id +
// physical_package_id) among the threads. Per node: CPUs that are in the masks of two or more ranks.
// OK = no overlap between ranks and every thread on its own physical core.
static std::string cpu_list(const std::vector<int>& v) {
    std::string s; for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += std::to_string(v[i]); } return s;
}
static long core_key(int cpu) {
    auto rd = [](const std::string& p) { FILE* f = fopen(p.c_str(), "r"); int x = -1; if (f) { if (fscanf(f, "%d", &x) != 1) x = -1; fclose(f); } return x; };
    const std::string b = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const int core = rd(b + "core_id"), pkg = rd(b + "physical_package_id");
    return core < 0 ? -1 - cpu : (long)pkg * 100000 + core;
}
static void bind_report(int nthreads) {
    // the rank's CPUs = union of its threads' masks (with OMP_PROC_BIND the master thread's own mask is
    // only its place, not the whole domain the rank was given)
    cpu_set_t m; CPU_ZERO(&m);
    std::vector<int> tcpu(nthreads, -1);
    #pragma omp parallel num_threads(nthreads)
    {
        volatile double x = 0.0; for (int i = 0; i < 2000000; ++i) x += 1e-9 * i;
        cpu_set_t t; CPU_ZERO(&t); sched_getaffinity(0, sizeof t, &t);
        tcpu[omp_get_thread_num()] = sched_getcpu();
        #pragma omp critical
        CPU_OR(&m, &m, &t);
    }
    const int nmask = CPU_COUNT(&m);
    std::vector<int> d = tcpu; std::sort(d.begin(), d.end()); d.erase(std::unique(d.begin(), d.end()), d.end());
    std::vector<long> ck; for (int c : tcpu) ck.push_back(core_key(c));
    std::sort(ck.begin(), ck.end()); ck.erase(std::unique(ck.begin(), ck.end()), ck.end());
    // overlap of the masks of the ranks of one node
    MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
    int nr; MPI_Comm_size(node, &nr);
    std::vector<unsigned char> all((size_t)nr * sizeof m);
    MPI_Allgather(&m, sizeof m, MPI_BYTE, all.data(), sizeof m, MPI_BYTE, node);
    int overlap = 0;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        int k = 0; for (int r = 0; r < nr; ++r) if (CPU_ISSET(c, (cpu_set_t*)(all.data() + (size_t)r * sizeof m))) ++k;
        if (k > 1) ++overlap;
    }
    MPI_Comm_free(&node);
    int v[5] = {nmask, -nmask, (int)d.size(), (int)ck.size(), overlap}, g[5];
    MPI_Allreduce(v, g, 5, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    int ov; MPI_Allreduce(&overlap, &ov, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    const bool ok = ov == 0 && g[2] == nthreads && g[3] == nthreads;
    if (!g_rank) {
        printf("FFTB_BIND ranks/node=%d threads=%d mask_cpus=%d-%d thread_cpus_distinct_min=%d thread_cores_distinct_min=%d "
               "overlapping_cpus_per_node=%d %s\n", nr, nthreads, g[0], -g[1], g[2], g[3], ov, ok ? "OK" : "WARN");
        fflush(stdout);
    }
    const char* sb = getenv("FFTB_SHOWBIND");
    if (sb && !strcmp(sb, "1"))
        for (int r = 0; r < std::min(2, g_np); ++r) {
            if (g_rank == r) {
                std::vector<int> mk; for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &m)) mk.push_back(c);
                char h[64]; gethostname(h, sizeof h);
                printf("FFTB_BIND rank=%d host=%s mask=%s threads_on=%s\n", r, h, cpu_list(mk).c_str(), cpu_list(tcpu).c_str());
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}
// test field (several modes in every direction, not a single wave)
static double test_field(int i, int j, int k, int nx, int ny, int nz) {
    const double tp = 2.0 * M_PI, x = tp * i / nx, y = tp * j / ny, z = tp * k / nz;
    return sin(x) * cos(2.0 * y) + 0.5 * cos(3.0 * z) + 0.25 * sin(x + 2.0 * y - 3.0 * z) + 0.1;
}

// FFTB_SPLIT=1: the local FFT work of one rank with plain FFTW, no communication: batches of contiguous 1-D
// transforms over N^3/P points (r2c along the first, c2c along the other two dimensions; forward and
// backward), with the solver's thread count and planner rigor. It is the arithmetic of the 3-D FFT without
// the global transposes and without the libraries' local packing / strides, so t_fft - t_local is an upper
// estimate of the communication (+ packing) time. Returns {t_fwd, t_bwd} per transform.
static std::pair<double, double> local_model(int nx, int ny, int nz, int nthreads, unsigned rigor, double window) {
#if FFTB_LIB == 4
#if NS_HB_CAT(NS_HB_, NS_HEFFTE_BACKEND) == NS_HB_mkl
    fftw_init_threads();        // heFFTe-MKL did not initialise FFTW threads
#endif
#endif
    fftw_plan_with_nthreads(nthreads);
    const long ncx = nx / 2 + 1;
    const long l1 = std::max(1L, ((long)ny * nz + g_np - 1) / g_np);            // r2c lines of length nx
    const long nc_pts = ncx * l1;                                               // complex points after r2c
    const long l2 = std::max(1L, nc_pts / ny), l3 = std::max(1L, nc_pts / nz);  // c2c lines of length ny / nz
    double* r = fftw_alloc_real((size_t)nx * l1);
    fftw_complex* c = fftw_alloc_complex((size_t)std::max(nc_pts, std::max(l2 * ny, l3 * nz)));
    for (long i = 0; i < (long)nx * l1; ++i) r[i] = 0.001 * (i % 997);
    int n1[1] = {nx}, n2[1] = {ny}, n3[1] = {nz};
    fftw_plan pf1 = fftw_plan_many_dft_r2c(1, n1, (int)l1, r, nullptr, 1, nx, c, nullptr, 1, (int)ncx, rigor);
    fftw_plan pb1 = fftw_plan_many_dft_c2r(1, n1, (int)l1, c, nullptr, 1, (int)ncx, r, nullptr, 1, nx, rigor);
    fftw_plan pf2 = fftw_plan_many_dft(1, n2, (int)l2, c, nullptr, 1, ny, c, nullptr, 1, ny, FFTW_FORWARD, rigor);
    fftw_plan pb2 = fftw_plan_many_dft(1, n2, (int)l2, c, nullptr, 1, ny, c, nullptr, 1, ny, FFTW_BACKWARD, rigor);
    fftw_plan pf3 = fftw_plan_many_dft(1, n3, (int)l3, c, nullptr, 1, nz, c, nullptr, 1, nz, FFTW_FORWARD, rigor);
    fftw_plan pb3 = fftw_plan_many_dft(1, n3, (int)l3, c, nullptr, 1, nz, c, nullptr, 1, nz, FFTW_BACKWARD, rigor);
    for (long i = 0; i < (long)nx * l1; ++i) r[i] = 0.001 * (i % 997);       // planning may overwrite
    auto fwd = [&]() { fftw_execute(pf1); fftw_execute(pf2); fftw_execute(pf3); };
    auto bwd = [&]() { fftw_execute(pb3); fftw_execute(pb2); fftw_execute(pb1);
                       const double s = 1.0 / ((double)nx * ny * nz);
                       for (long i = 0; i < (long)nx * l1; ++i) r[i] *= s; };   // keep values bounded (not timed below)
    fwd(); bwd();
    double t0 = MPI_Wtime(); fwd(); bwd(); double tp = std::max(MPI_Wtime() - t0, 1e-6);
    int K = (int)std::ceil(window / max_all(tp)); MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD); K = std::max(K, 1);
    double tf = 0.0, tb = 0.0;
    for (int it = 0; it < K; ++it) {
        double a = MPI_Wtime(); fwd(); double b = MPI_Wtime();
        fftw_execute(pb3); fftw_execute(pb2); fftw_execute(pb1); double e = MPI_Wtime();
        tf += b - a; tb += e - b;
        const double s = 1.0 / ((double)nx * ny * nz);
        for (long i = 0; i < (long)nx * l1; ++i) r[i] *= s;
    }
    fftw_destroy_plan(pf1); fftw_destroy_plan(pb1); fftw_destroy_plan(pf2); fftw_destroy_plan(pb2);
    fftw_destroy_plan(pf3); fftw_destroy_plan(pb3);
    fftw_free(r); fftw_free(c);
    return {max_all(tf / K), max_all(tb / K)};
}

int main(int argc, char** argv) {
#if FFTB_LIB == 1
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);   // as NavierStokes_periodic_fftw.cpp
#else
    MPI_Init(&argc, &argv);
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_np);
    if (argc < 4) {
        if (!g_rank) fprintf(stderr, "Usage: %s NX NY NZ [threads]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int nx = atoi(argv[1]), ny = atoi(argv[2]), nz = atoi(argv[3]);
    const int nthreads = (argc > 4) ? atoi(argv[4]) : omp_get_max_threads();
    omp_set_num_threads(nthreads);
    const int REP = std::max(1, env_int("FFTB_REP", 3)), WARM = std::max(1, env_int("FFTB_WARM", 2));
    const double WINDOW = env_dbl("FFTB_WINDOW", 3.0);
    const int PAIRS = env_int("FFTB_PAIRS", 0), SPLIT = env_int("FFTB_SPLIT", 0), CHECK = env_int("FFTB_CHECK", 1);
    const int NF = std::max(1, env_int("FFTB_NF", 3));
    {
    Engine E;
    E.init(nthreads);
    E.decompose(nx, ny, nz);
    if (!g_rank) {
        printf("============================================================\n");
        printf(" fftbench: %s  %d x %d x %d  ranks=%d  threads/rank=%d  (solver plan/execute code: fftlib)\n",
               Engine::lib_name(), nx, ny, nz, g_np, nthreads);
        int in_par = 0;
        #pragma omp parallel
        {
            #pragma omp master
            in_par = omp_get_num_threads();
        }
        printf("Threads (runtime): omp_get_max_threads()=%d omp_get_num_threads(in parallel)=%d\n", omp_get_max_threads(), in_par);
        printf(" repetitions=%d warm-up rounds=%d fields=%d window=%.1f s%s%s\n", REP, WARM, NF, WINDOW,
               PAIRS > 0 ? " pairs fixed (FFTB_PAIRS)" : " pairs automatic", SPLIT ? "  local FFT model on (FFTB_SPLIT=1)" : "");
        printf("============================================================\n");
        fflush(stdout);
    }
    bind_report(nthreads);
    E.print_decomp();          // collective for heFFTe
    E.setup();                 // P3DFFT: Cp3dfft_setup (the planning)
    const size_t nr = E.nreal(), nc = E.ncplx();
    // NF independent fields (default 3, as the velocity / vorticity components): pair p uses field p % NF,
    // so consecutive transforms touch different memory as in the solver, not one cache-resident set
    std::vector<double*> R0(NF), R1(NF);
    std::vector<typename Engine::cplx*> C(NF);
    for (int f = 0; f < NF; ++f) { R0[f] = E.alloc_r(); R1[f] = E.alloc_r(); C[f] = E.alloc_c(); }
    double* r0 = R0[0]; double* r1 = R1[0]; typename Engine::cplx* c = C[0];
    {
        const double GB = 1024.0 * 1024.0 * 1024.0;
        double m[2] = {nr * 8.0, nc * 16.0}, g[2];
        MPI_Allreduce(m, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (!g_rank) { printf("FFTB buffers per rank (max): %d fields x (2 real of %.3f GB + 1 complex of %.3f GB)\n", NF, g[0] / GB, g[1] / GB); fflush(stdout); }
    }
    const double t_plan = E.plan(r0, c);

    auto fill = [&](double* r) {
        std::fill(r, r + nr, 0.0);
        E.for_real([&](size_t idx, int i, int j, int k) { r[idx] = test_field(i, j, k, nx, ny, nz); });
    };
    // ---- check: forward + backward, normalised, against the field; is the forward input preserved?
    double err = 0.0; int preserved = 1;
    if (CHECK) {
        fill(r0);
        E.forward(r0, c);
        E.backward(c, r1);
        const double s = E.roundtrip_norm();
        double e = 0.0; int pres = 1;
        E.for_real([&](size_t idx, int i, int j, int k) {
            const double f = test_field(i, j, k, nx, ny, nz);
            e = std::max(e, std::fabs(r1[idx] * s - f));
            if (r0[idx] != f) pres = 0;
        });
        err = max_all(e);
        int p; MPI_Allreduce(&pres, &p, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD); preserved = p;
    }
    const bool ok = !CHECK || err < 1e-10;
    if (!g_rank) {
        if (CHECK) printf("FFTB_CHECK max_err=%.3e %s forward_input_preserved=%s\n", err, ok ? "(OK)" : "(FAIL)", preserved ? "yes" : "no");
        else       printf("FFTB_CHECK skipped (FFTB_CHECK=0)\n");
        fflush(stdout);
    }
    for (int f = 0; f < NF; ++f) fill(R0[f]);

    // ---- warm-up; the last pair sets K
    double tpair = 0.0;
    for (int w = 0; w < WARM; ++w) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int f = 0; f < NF; ++f) { E.forward(R0[f], C[f]); E.backward(C[f], R1[f]); }
        MPI_Barrier(MPI_COMM_WORLD);
        tpair = max_all(MPI_Wtime() - t0) / NF;
    }
    int K = PAIRS > 0 ? PAIRS : (int)std::ceil(WINDOW / std::max(tpair, 1e-6));
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    K = std::max(K, 1);
    if (!g_rank) { printf("FFTB warm-up: last pair %.4f s -> %d pairs per repetition\n", tpair, K); fflush(stdout); }

    // ---- repetitions
    std::vector<double> tfft;
    for (int rep = 1; rep <= REP; ++rep) {
        if (!preserved) for (int f = 0; f < NF; ++f) fill(R0[f]);   // a library that overwrote its input: refill
        double tf = 0.0, tb = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        for (int it = 0; it < K; ++it) {
            const int f = it % NF;
            const double a = MPI_Wtime();
            E.forward(R0[f], C[f]);
            const double b = MPI_Wtime();
            E.backward(C[f], R1[f]);
            tb += MPI_Wtime() - b; tf += b - a;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double W = max_all(MPI_Wtime() - t0);
        const double tfm = max_all(tf), tbm = max_all(tb);
        const double tp = W / K, fr = (tfm + tbm) > 0 ? tfm / (tfm + tbm) : 0.5;
        const double t_fwd = tp * fr, t_bwd = tp * (1.0 - fr), t_fft = (t_fwd + 2.0 * t_bwd) / 3.0;
        int fin = 1;
        for (int f = 0; f < NF && f < K; ++f)
            for (size_t i = 0; i < nr; i += 4093) if (!std::isfinite(R1[f][i])) { fin = 0; break; }
        int finall; MPI_Allreduce(&fin, &finall, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        tfft.push_back(t_fft);
        if (!g_rank) {
            printf("FFTB_REP rep=%d pairs=%d window_s=%.4f t_fwd_s=%.6e t_bwd_s=%.6e t_pair_s=%.6e t_fft_s=%.6e finite=%s\n",
                   rep, K, W, t_fwd, t_bwd, tp, t_fft, finall ? "yes" : "no");
            fflush(stdout);
        }
    }
    double t_local = -1.0;
    if (SPLIT) {
        unsigned rigor = FFTW_MEASURE;
        const int rid = nsmode::plan_rigor_from_env();
        if (rid == nsmode::PLAN_ESTIMATE) rigor = FFTW_ESTIMATE; else if (rid == nsmode::PLAN_PATIENT) rigor = FFTW_PATIENT;
        std::pair<double, double> lt = local_model(nx, ny, nz, nthreads, rigor, std::max(1.0, WINDOW / 3.0));
        t_local = (lt.first + 2.0 * lt.second) / 3.0;
        if (!g_rank) {
            const double m = median(tfft);
            printf("FFTB_LOCAL t_fwd_s=%.6e t_bwd_s=%.6e t_local_s=%.6e  -> t_fft - t_local = %.6e s (%.0f%% of t_fft; communication + packing, estimate)\n",
                   lt.first, lt.second, t_local, m - t_local, m > 0 ? 100.0 * (m - t_local) / m : 0.0);
            fflush(stdout);
        }
    }
    if (!g_rank) {
        char tl[32] = "";
        if (t_local >= 0) snprintf(tl, sizeof tl, "%.6e", t_local);
        printf("FFTB_SUMMARY lib=%s grid=%d ranks=%d threads=%d settings=\"%s\" t_plan_s=%.4f check=%s max_err=%.3e pairs=%d "
               "t_fft_median_s=%.6e t_fft_min_s=%.6e t_fft_max_s=%.6e step_fft_s=%.4f t_local_s=%s\n",
               Engine::lib_name(), nx, g_np, nthreads, E.settings().c_str(), t_plan, CHECK ? (ok ? "OK" : "FAIL") : "skipped", err, K,
               median(tfft), *std::min_element(tfft.begin(), tfft.end()), *std::max_element(tfft.begin(), tfft.end()),
               36.0 * median(tfft), tl);
        fflush(stdout);
    }
    for (int f = 0; f < NF; ++f) { Engine::free_r(R0[f]); Engine::free_r(R1[f]); Engine::free_c(C[f]); }
    E.destroy();
    }
    MPI_Finalize();
    return 0;
}
