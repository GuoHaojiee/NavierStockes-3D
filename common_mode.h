// common_mode.h
// =============================================================================
// Shared helpers for all NavierStokes_periodic_* solvers:
//   1. Run-mode parsing  (verify | bench)
//   2. Per-phase timing  (FFT / spectral pointwise / physical pointwise / forcing)
//   3. Uniform printing of the phase breakdown
//
// Command line (all solvers):
//   ./prog NX NY NZ dt NSTEPS [OMP_THREADS] [mode]        (CPU versions)
//   ./prog NX NY NZ dt NSTEPS [mode]                      (GPU versions)
//
//   mode = verify (default) | bench     (also accepted: mode=bench, --mode=bench)
//
// Disambiguation rule for the optional trailing arguments (argv[6], argv[7]):
//   - a token consisting only of digits      -> OMP_THREADS (CPU versions only)
//   - one of the mode words above            -> run mode
//   - anything else                          -> error (program aborts)
//   The two token classes are disjoint, so "... 10 4 bench", "... 10 bench",
//   "... 10 4" and "... 10" are all unambiguous.  Each may be given at most once.
//
// Modes:
//   verify : analytic forcing f is evaluated every RHS call (original behaviour);
//            prints L2 error and max|div V| at the end.
//   bench  : f = 0.  The forcing fill, its 3 forward FFTs, copies and scaling
//            are skipped and the forcing-only buffers are not allocated.
//            36 FFTs per RK4 step instead of 48.  Prints kinetic energy
//            0.5*sum|v|^2 and max|div V| after timing (not timed).
//
// Phase timing:
//   Each solver calls timer.start() at the beginning of a time step and
//   timer.mark(PHASE) right after every region; the time since the previous
//   mark is attributed to PHASE.  FFT time of the forcing transforms is booked
//   under PH_FORCE (not PH_FFT), so PH_FFT is directly comparable between modes.
//   CPU: host wall clock.  GPU (one device per process): cudaEvent pairs on the
//   legacy default stream, collected after the step's final synchronisation.
// =============================================================================
#ifndef NS_COMMON_MODE_H
#define NS_COMMON_MODE_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace nsmode {

// -----------------------------------------------------------------------------
// Run mode
// -----------------------------------------------------------------------------
enum RunMode { MODE_VERIFY = 0, MODE_BENCH = 1 };

inline const char* mode_name(RunMode m) { return m == MODE_BENCH ? "bench" : "verify"; }

struct RunArgs {
    RunMode mode = MODE_VERIFY;
    int omp_threads = 0;          // 0 = not given on the command line
};

inline bool parse_mode_word(const char* s, RunMode& m) {
    if (std::strncmp(s, "--mode=", 7) == 0) s += 7;
    else if (std::strncmp(s, "mode=", 5) == 0) s += 5;
    if (std::strcmp(s, "verify") == 0) { m = MODE_VERIFY; return true; }
    if (std::strcmp(s, "bench")  == 0) { m = MODE_BENCH;  return true; }
    return false;
}

inline bool is_all_digits(const char* s) {
    if (!*s) return false;
    for (; *s; ++s) if (*s < '0' || *s > '9') return false;
    return true;
}

// Parse argv[first..argc-1].  Returns false and fills `err` on invalid input.
inline bool parse_tail_args(int argc, char** argv, int first, bool allow_threads,
                            RunArgs& out, std::string& err) {
    bool have_mode = false, have_threads = false;
    for (int i = first; i < argc; ++i) {
        const char* a = argv[i];
        RunMode m;
        if (parse_mode_word(a, m)) {
            if (have_mode) { err = std::string("mode given twice: ") + a; return false; }
            out.mode = m; have_mode = true;
        } else if (is_all_digits(a)) {
            if (!allow_threads) {
                err = std::string("unexpected numeric argument '") + a +
                      "' (this GPU version takes no OMP_THREADS; expected verify|bench)";
                return false;
            }
            if (have_threads) { err = std::string("OMP_THREADS given twice: ") + a; return false; }
            out.omp_threads = std::atoi(a);
            if (out.omp_threads <= 0) { err = "OMP_THREADS must be > 0"; return false; }
            have_threads = true;
        } else {
            err = std::string("unrecognised argument '") + a + "' (expected verify|bench" +
                  (allow_threads ? " or an integer OMP_THREADS)" : ")");
            return false;
        }
    }
    return true;
}

inline const char* usage_tail(bool allow_threads) {
    return allow_threads ? "NX NY NZ dt NSTEPS [OMP_THREADS] [verify|bench]"
                         : "NX NY NZ dt NSTEPS [verify|bench]";
}

// -----------------------------------------------------------------------------
// Phases
// -----------------------------------------------------------------------------
enum Phase { PH_FFT = 0, PH_SPEC = 1, PH_PHYS = 2, PH_FORCE = 3, PH_COUNT = 4 };

inline const char* phase_label(int p) {
    static const char* L[PH_COUNT] = {
        "FFT (incl. transpose/comm):",
        "Spectral pointwise:        ",
        "Physical pointwise:        ",
        "Forcing f (fill+3 FFT):    "
    };
    return L[p];
}

inline double wall_now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Host wall-clock phase timer (CPU solvers; also multi-GPU single-process
// solvers when every mark is preceded by a device synchronisation).
struct CpuPhaseTimer {
    double acc[PH_COUNT] = {0.0, 0.0, 0.0, 0.0};
    double t_last = 0.0;
    void start() { t_last = wall_now(); }
    void mark(Phase p) { double t = wall_now(); acc[p] += t - t_last; t_last = t; }
};

// Print the breakdown.  acc[] must already be totals over all steps
// (and reduced with MAX over ranks where applicable).
inline void print_phase_summary(const double acc[PH_COUNT], double total_wall, long nsteps,
                                RunMode mode) {
    if (nsteps <= 0) nsteps = 1;
    double sum = 0.0;
    std::printf("  Phase breakdown (avg per step, max over ranks):\n");
    for (int p = 0; p < PH_COUNT; ++p) {
        std::printf("    %s %.6f s\n", phase_label(p), acc[p] / nsteps);
        sum += acc[p];
    }
    std::printf("    Sum of phases:              %.6f s\n", sum / nsteps);
    std::printf("    Total (wall, per step):     %.6f s\n", total_wall / nsteps);
    std::printf("    FFTs per step:              %d\n", mode == MODE_BENCH ? 36 : 48);
    std::fflush(stdout);
}

// -----------------------------------------------------------------------------
// FFTW planner rigor from the environment (CPU solvers whose library takes it at run time)
//   NS_FFTW_PLAN = estimate | measure | patient | exhaustive      (default: measure)
// The environment is the same on every rank (srun exports it), so no broadcast is needed.
// Returns PLAN_INVALID for an unknown value; the caller aborts.
// -----------------------------------------------------------------------------
enum PlanRigor { PLAN_INVALID = -1, PLAN_ESTIMATE = 0, PLAN_MEASURE = 1, PLAN_PATIENT = 2, PLAN_EXHAUSTIVE = 3 };

inline int plan_rigor_from_env(int def = PLAN_MEASURE) {
    const char* s = std::getenv("NS_FFTW_PLAN");
    if (!s || !*s)                     return def;
    if (!std::strcmp(s, "estimate"))   return PLAN_ESTIMATE;
    if (!std::strcmp(s, "measure"))    return PLAN_MEASURE;
    if (!std::strcmp(s, "patient"))    return PLAN_PATIENT;
    if (!std::strcmp(s, "exhaustive")) return PLAN_EXHAUSTIVE;
    return PLAN_INVALID;
}

inline const char* plan_rigor_name(int r) {
    switch (r) {
        case PLAN_ESTIMATE:   return "ESTIMATE";
        case PLAN_MEASURE:    return "MEASURE";
        case PLAN_PATIENT:    return "PATIENT";
        case PLAN_EXHAUSTIVE: return "EXHAUSTIVE";
    }
    return "INVALID";
}

// -----------------------------------------------------------------------------
// Domain decomposition from the environment (CPU solvers AccFFT / P3DFFT / heFFTe)
//   NS_DECOMP = auto | slab | pencil | brick      (default: auto; brick = heFFTe only)
//   auto: slab when nprocs <= n_slab, otherwise pencil.
//   n_slab = number of grid points along the direction(s) a slab decomposition splits
//            (the slowest-varying direction; the caller passes the minimum over the
//            real-space and spectral-space split directions, so a slab never has empty
//            ranks).  For an N^3 grid this is simply N.
// The environment is the same on every rank (srun exports it), so no broadcast is needed.
// Returns DECOMP_INVALID for an unknown value; the caller aborts.  The caller also
// rejects values its library does not support (brick outside heFFTe).
// -----------------------------------------------------------------------------
enum Decomp { DECOMP_INVALID = -1, DECOMP_SLAB = 0, DECOMP_PENCIL = 1, DECOMP_BRICK = 2 };

inline const char* decomp_env_value() {
    const char* s = std::getenv("NS_DECOMP");
    return (s && *s) ? s : "auto (unset)";
}

inline int decomp_from_env(int nprocs, long n_slab) {
    const char* s = std::getenv("NS_DECOMP");
    if (!s || !*s || !std::strcmp(s, "auto")) return (nprocs <= n_slab) ? DECOMP_SLAB : DECOMP_PENCIL;
    if (!std::strcmp(s, "slab"))   return DECOMP_SLAB;
    if (!std::strcmp(s, "pencil")) return DECOMP_PENCIL;
    if (!std::strcmp(s, "brick"))  return DECOMP_BRICK;
    return DECOMP_INVALID;
}

inline const char* decomp_name(int d) {
    switch (d) {
        case DECOMP_SLAB:   return "slab";
        case DECOMP_PENCIL: return "pencil";
        case DECOMP_BRICK:  return "brick";
    }
    return "INVALID";
}

// One line, the same in every solver (scripts grep for "decomposition:").
inline void print_decomp(int d, const char* grid) {
    std::printf("decomposition: %s   process grid: %s   NS_DECOMP=%s\n", decomp_name(d), grid, decomp_env_value());
    std::fflush(stdout);
}

} // namespace nsmode

// -----------------------------------------------------------------------------
// MPI helper (only when mpi.h was included before this header)
// -----------------------------------------------------------------------------
#ifdef MPI_VERSION
namespace nsmode {
inline void reduce_phases_max(const double in[PH_COUNT], double out[PH_COUNT],
                              MPI_Comm comm = MPI_COMM_WORLD) {
    MPI_Allreduce(const_cast<double*>(in), out, PH_COUNT, MPI_DOUBLE, MPI_MAX, comm);
}
} // namespace nsmode
#endif

// -----------------------------------------------------------------------------
// heFFTe plan options from the environment (only when heffte.h was included first)
//   NS_HEFFTE_ALGO    = alltoallv | alltoall | p2p | p2p_plined   (default: the caller's)
//   NS_HEFFTE_REORDER = 0 | 1                                     (default: the caller's)
// Only algorithm and use_reorder are touched; every other option is left as given.
// Read on rank 0 and broadcast, so all ranks use the same plan. Rank 0 prints the result.
// -----------------------------------------------------------------------------
#if defined(HEFFTE_H) && defined(MPI_VERSION)
namespace nsmode {
inline const char* heffte_algo_name(heffte::reshape_algorithm a) {
    switch (a) {
        case heffte::reshape_algorithm::alltoallv:  return "alltoallv";
        case heffte::reshape_algorithm::alltoall:   return "alltoall";
        case heffte::reshape_algorithm::p2p:        return "p2p";
        case heffte::reshape_algorithm::p2p_plined: return "p2p_plined";
    }
    return "?";
}
inline void heffte_options_from_env(heffte::plan_options& opt, int rank, MPI_Comm comm = MPI_COMM_WORLD) {
    int v[3] = { (int)opt.algorithm, opt.use_reorder ? 1 : 0, 0 };   // v[2]: 1 = invalid value
    if (rank == 0) {
        if (const char* a = std::getenv("NS_HEFFTE_ALGO")) {
            if      (!std::strcmp(a, "alltoallv"))  v[0] = (int)heffte::reshape_algorithm::alltoallv;
            else if (!std::strcmp(a, "alltoall"))   v[0] = (int)heffte::reshape_algorithm::alltoall;
            else if (!std::strcmp(a, "p2p"))        v[0] = (int)heffte::reshape_algorithm::p2p;
            else if (!std::strcmp(a, "p2p_plined")) v[0] = (int)heffte::reshape_algorithm::p2p_plined;
            else { std::fprintf(stderr, "Error: NS_HEFFTE_ALGO=%s (expected alltoallv|alltoall|p2p|p2p_plined)\n", a); v[2] = 1; }
        }
        if (const char* r = std::getenv("NS_HEFFTE_REORDER")) {
            if      (!std::strcmp(r, "0")) v[1] = 0;
            else if (!std::strcmp(r, "1")) v[1] = 1;
            else { std::fprintf(stderr, "Error: NS_HEFFTE_REORDER=%s (expected 0|1)\n", r); v[2] = 1; }
        }
    }
    MPI_Bcast(v, 3, MPI_INT, 0, comm);
    if (v[2]) MPI_Abort(comm, 1);
    opt.algorithm   = (heffte::reshape_algorithm)v[0];
    opt.use_reorder = (v[1] != 0);
    if (rank == 0) {
        std::printf("heFFTe plan options: algorithm = %s (NS_HEFFTE_ALGO), use_reorder = %d (NS_HEFFTE_REORDER)\n",
                    heffte_algo_name(opt.algorithm), opt.use_reorder ? 1 : 0);
        std::fflush(stdout);
    }
}

// -----------------------------------------------------------------------------
// Slab layouts of the multi-GPU heFFTe solvers (heffte_mgpu, heffte_multigpu)
//   NS_HEFFTE_LAYOUT = zslab (default) | xslab
//   heFFTe box index 0 = x = fastest-varying, 2 = z = slowest (the solvers' real arrays: x fastest).
//   zslab: real space split along z {1,1,P}, r2c along x (dim 0, contiguous), spectral split along y {1,P,1}
//          -> complex box (NX/2+1) x NY x NZ, kx = 0..NX/2.  Needs NY, NZ >= P.  (cuFFTMp-like)
//   xslab: real space split along x {P,1,1}, r2c along z (dim 2), spectral split along y {1,P,1}
//          -> complex box NX x NY x (NZ/2+1), kz = 0..NZ/2.  Needs NX, NY >= P.  (the earlier layout)
//   Both need one MPI reshape per transform. The spectral kernels fold all three indices
//   (g <= N/2 ? g : g - N), which is the identity on the halved r2c dimension, so they are the
//   same for both layouts.
// Read on rank 0 and broadcast; an unknown value or a too small grid aborts.
// -----------------------------------------------------------------------------
struct HeffteSlab {
    std::string name;
    std::array<int, 3> pg, pg_c;        // processor grids: real space, spectral space
    int r2c;                            // r2c direction (heFFTe dimension)
    heffte::box3d<> world_r, world_c;
};
inline HeffteSlab heffte_slab_from_env(int NX, int NY, int NZ, int P, int rank, MPI_Comm comm = MPI_COMM_WORLD) {
    int v[2] = {0, 0};                  // v[0]: 0 zslab, 1 xslab; v[1]: 1 = invalid value
    if (rank == 0) {
        const char* s = std::getenv("NS_HEFFTE_LAYOUT");
        if (!s || !std::strcmp(s, "zslab")) v[0] = 0;
        else if (!std::strcmp(s, "xslab"))  v[0] = 1;
        else { std::fprintf(stderr, "Error: NS_HEFFTE_LAYOUT=%s (expected zslab|xslab)\n", s); v[1] = 1; }
    }
    MPI_Bcast(v, 2, MPI_INT, 0, comm);
    if (v[1]) MPI_Abort(comm, 1);
    const heffte::box3d<> world_r({0, 0, 0}, {NX - 1, NY - 1, NZ - 1});
    if (v[0] == 0) {
        if (NZ < P || NY < P) {
            if (rank == 0) std::fprintf(stderr, "Error: zslab needs NZ (%d) and NY (%d) >= number of ranks (%d)\n", NZ, NY, P);
            MPI_Abort(comm, 1);
        }
        return HeffteSlab{"zslab", {1, 1, P}, {1, P, 1}, 0, world_r,
                          heffte::box3d<>({0, 0, 0}, {NX / 2, NY - 1, NZ - 1})};
    }
    if (NX < P || NY < P) {
        if (rank == 0) std::fprintf(stderr, "Error: xslab needs NX (%d) and NY (%d) >= number of ranks (%d)\n", NX, NY, P);
        MPI_Abort(comm, 1);
    }
    return HeffteSlab{"xslab", {P, 1, 1}, {1, P, 1}, 2, world_r,
                      heffte::box3d<>({0, 0, 0}, {NX - 1, NY - 1, NZ / 2})};
}
// MPI reshapes per transform of the plan heFFTe builds for these boxes (as in heFFTe's
// benchmarks/speed3d.h): a stage whose processor grid differs from the previous one is a reshape.
inline int heffte_reshapes_per_fft(const std::vector<heffte::box3d<>>& in, const std::vector<heffte::box3d<>>& out,
                                   int r2c, const heffte::plan_options& opt, int rank) {
    heffte::logic_plan3d<int> lp = heffte::plan_operations<int>({in, out}, r2c, opt, rank);
    std::vector<std::array<int, 3>> grids = heffte::compute_grids(lp);
    int n = 0;
    for (size_t i = 1; i < grids.size(); ++i) if (grids[i] != grids[i - 1]) ++n;
    return n;
}
// One line for people and one for scripts (grep "^HEFFTE_PLAN").
inline void heffte_print_plan(const char* layout, const std::array<int, 3>& pg, const std::array<int, 3>& pg_c, int r2c,
                              const heffte::plan_options& opt, int reshapes, int rank) {
    if (rank != 0) return;
    static const char* dn[3] = {"x", "y", "z"};
    std::printf("heFFTe layout: %s, real %dx%dx%d, spectral %dx%dx%d, r2c along %s (dim %d), reorder = %d, "
                "algorithm = %s, MPI reshapes per FFT = %d\n",
                layout, pg[0], pg[1], pg[2], pg_c[0], pg_c[1], pg_c[2], dn[r2c], r2c, opt.use_reorder ? 1 : 0,
                heffte_algo_name(opt.algorithm), reshapes);
    std::printf("HEFFTE_PLAN layout=%s r2c_dir=%d reorder=%d algo=%s reshapes=%d\n",
                layout, r2c, opt.use_reorder ? 1 : 0, heffte_algo_name(opt.algorithm), reshapes);
    std::fflush(stdout);
}
} // namespace nsmode
#endif

// -----------------------------------------------------------------------------
// GPU event-based phase timer (one device per process)
// -----------------------------------------------------------------------------
#ifdef __CUDACC__
#include <cuda_runtime.h>
namespace nsmode {
struct GpuPhaseTimer {
    static const int MAXEV = 1024;
    cudaEvent_t ev[MAXEV];
    int ph[MAXEV];
    int n = 0;
    bool overflow = false;
    double acc[PH_COUNT] = {0.0, 0.0, 0.0, 0.0};

    void init() {
        for (int i = 0; i < MAXEV; ++i) cudaEventCreate(&ev[i]);
    }
    void destroy() {
        for (int i = 0; i < MAXEV; ++i) cudaEventDestroy(ev[i]);
    }
    // Record on the legacy default stream (stream 0), which is where all
    // solver kernels and FFT executions of these programs are issued.
    void start() { n = 0; cudaEventRecord(ev[0], 0); ph[0] = -1; n = 1; }
    void mark(Phase p) {
        if (n >= MAXEV) { overflow = true; return; }
        cudaEventRecord(ev[n], 0); ph[n] = (int)p; ++n;
    }
    // Call after the step's final cudaDeviceSynchronize().
    void collect() {
        for (int i = 1; i < n; ++i) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev[i - 1], ev[i]);
            acc[ph[i]] += 1e-3 * (double)ms;
        }
        n = 0;
    }
};
} // namespace nsmode
#endif

#endif // NS_COMMON_MODE_H
