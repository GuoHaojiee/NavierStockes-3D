// ns_fft_heffte.h — heFFTe 2.x (CPU) 3D r2c/c2r exactly as NavierStokes_periodic_heffte_v2.cpp uses it.
// Shared by the solver and scripts/fftbench/fftbench.cpp (one copy of the plan / execute code).
//
//   NS_DECOMP = auto | slab | pencil | brick (only the in/out processor grids change):
//     slab  : inbox {p,1,1} (x split), outbox {1,p,1} (y split), use_pencils=false, 1 reshape expected
//     pencil: inbox {p1,p2,1} (z pencils), outbox {1,p1,p2} (x pencils), p1*p2=p closest to square
//             (heffte::make_procgrid), use_pencils=true, 2 reshapes expected
//     brick : proc_setup_min_surface on both sides, usually 4 reshapes
//     the real number of MPI reshapes per transform is computed with plan_operations and printed
//   r2c along dim 2 (z): spectral box nx x ny x (nz/2+1); boxes in heFFTe order, x fastest
//   local FFT backend at compile time: -DNS_HEFFTE_BACKEND=fftw (default) | mkl
//   NS_HEFFTE_ALGO / NS_HEFFTE_REORDER override the backend's defaults (heFFTe 2.4.1: fftw reorder 1, mkl 0)
//   forward scale::none, backward scale::full: backward(forward(x)) = x
//   the FFTW backend creates its FFTW plans at the first execution: "heFFTe plan time" does not include them
//
// Order of calls (the solver's order): init(threads) -> decompose(nx,ny,nz) -> [banner: print_decomp()]
//   -> setup() -> allocate nreal()/ncplx() -> plan() -> forward/backward ... -> destroy() (before MPI_Finalize)
#ifndef NS_FFT_HEFFTE_H
#define NS_FFT_HEFFTE_H
#include <algorithm>
#include <array>
#include <complex>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <mpi.h>
#include <heffte.h>
#include "../common_mode.h"

// ---- local FFT backend: NS_HEFFTE_BACKEND = fftw | mkl (compile time) ----
#define NS_HB_fftw 1
#define NS_HB_mkl  2
#define NS_HB_CAT2(a, b) a##b
#define NS_HB_CAT(a, b) NS_HB_CAT2(a, b)
#define NS_HB_STR2(a) #a
#define NS_HB_STR(a) NS_HB_STR2(a)
#ifndef NS_HEFFTE_BACKEND
#define NS_HEFFTE_BACKEND fftw
#endif
#if NS_HB_CAT(NS_HB_, NS_HEFFTE_BACKEND) == NS_HB_fftw
#include <fftw3.h>
using FftBackend = heffte::backend::fftw;
#elif NS_HB_CAT(NS_HB_, NS_HEFFTE_BACKEND) == NS_HB_mkl
#ifndef Heffte_ENABLE_MKL
#error "NS_HEFFTE_BACKEND=mkl: this heFFTe was built without -DHeffte_ENABLE_MKL=ON"
#endif
#include <mkl_service.h>
using FftBackend = heffte::backend::mkl;
#else
#error "NS_HEFFTE_BACKEND must be fftw or mkl"
#endif
static const char* const kBackendName = NS_HB_STR(NS_HEFFTE_BACKEND);

namespace nsfft {
struct Heffte {
    using cplx = std::complex<double>;
    static const char* lib_name() { return "heFFTe"; }
    int rank = 0, nprocs = 1, nthreads = 1;
    int nx = 0, ny = 0, nz = 0;
    int decomp_id = nsmode::DECOMP_SLAB;
    std::array<int, 3> grid_in{{1, 1, 1}}, grid_out{{1, 1, 1}};
    std::vector<heffte::box3d<>> all_inboxes, all_outboxes;   // box3d has const members: kept in vectors
    const heffte::box3d<>& inbox() const { return all_inboxes[rank]; }
    const heffte::box3d<>& outbox() const { return all_outboxes[rank]; }
    heffte::plan_options options = heffte::default_options<FftBackend>();
    int n_reshape = 0;
    std::unique_ptr<heffte::fft3d_r2c<FftBackend>> fft;
    std::vector<cplx> ws;

    void init(int nt) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        nthreads = nt;
#if NS_HB_CAT(NS_HB_, NS_HEFFTE_BACKEND) == NS_HB_fftw
        // heFFTe's FFTW backend creates the FFTW plans at the first execution with the global thread
        // setting; without these two calls every rank's FFT uses one thread. Link libfftw3_omp (see the FFTW version).
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
#else
        // MKL DFTI threads: mkl_intel_thread takes OMP_NUM_THREADS; set explicitly, as for the FFTW backend
        mkl_set_num_threads(nthreads);
#endif
    }
    void decompose(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        // slab splits x in real space and y in spectral space
        decomp_id = nsmode::decomp_from_env(nprocs, std::min(nx, ny));
        if (decomp_id == nsmode::DECOMP_INVALID) {
            if (rank == 0) std::cerr << "Error: NS_DECOMP=" << nsmode::decomp_env_value()
                                     << " (heFFTe: expected auto|slab|pencil|brick)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (decomp_id == nsmode::DECOMP_SLAB && nprocs > std::min(nx, ny)) {
            if (rank == 0) std::cerr << "Error: slab needs nprocs <= min(nx, ny) = " << std::min(nx, ny)
                                     << ", got " << nprocs << "; use NS_DECOMP=pencil or auto\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    int decomp() const { return decomp_id; }
    // boxes, plan options and the reshape count (collective); prints the decomposition line on rank 0
    void print_decomp() {
        heffte::box3d<> world_r = {{0, 0, 0}, {nx - 1, ny - 1, nz - 1}};
        heffte::box3d<> world_c = {{0, 0, 0}, {nx - 1, ny - 1, nz / 2}};  // r2c halves z
        if (decomp_id == nsmode::DECOMP_SLAB) {
            grid_in  = {nprocs, 1, 1};
            grid_out = {1, nprocs, 1};
        } else if (decomp_id == nsmode::DECOMP_PENCIL) {
            std::array<int, 2> pg = heffte::make_procgrid(nprocs);   // p1*p2 = p, minimal surface = closest to square
            grid_in  = {pg[0], pg[1], 1};
            grid_out = {1, pg[0], pg[1]};
        } else {
            grid_in = grid_out = heffte::proc_setup_min_surface(world_r, nprocs);
        }
        all_inboxes  = heffte::split_world(world_r, grid_in);
        all_outboxes = heffte::split_world(world_c, grid_out);
        const heffte::box3d<>& inbox_r = inbox();
        const heffte::box3d<>& outbox_c = outbox();
        // library defaults of the backend (alltoallv; use_reorder fftw 1 / mkl 0) unless NS_HEFFTE_ALGO /
        // NS_HEFFTE_REORDER; use_pencils follows the decomposition (brick keeps the library default)
        options = heffte::default_options<FftBackend>();
        nsmode::heffte_options_from_env(options, rank);
        if (decomp_id == nsmode::DECOMP_SLAB)   options.use_pencils = false;
        if (decomp_id == nsmode::DECOMP_PENCIL) options.use_pencils = true;
        // the same planning as inside the fft3d_r2c constructor (collective), to count the MPI reshapes
        // per transform: a stage whose input and output boxes differ (box3d == ignores order) communicates
        n_reshape = 0;
        {
            auto lplan = heffte::plan_operations(heffte::mpi::gather_boxes(inbox_r, outbox_c, MPI_COMM_WORLD), 2,
                                                 heffte::set_options<FftBackend, true>(options), rank);
            for (int i = 0; i < 4; ++i) if (!heffte::match(lplan.in_shape[i], lplan.out_shape[i])) ++n_reshape;
        }
        if (rank == 0) {
            char grid[160];
            snprintf(grid, sizeof grid, "in %dx%dx%d, out %dx%dx%d, use_pencils=%d, MPI reshapes per FFT = %d",
                     grid_in[0], grid_in[1], grid_in[2], grid_out[0], grid_out[1], grid_out[2],
                     options.use_pencils ? 1 : 0, n_reshape);
            nsmode::print_decomp(decomp_id, grid);
        }
    }
    void setup() {}
    size_t nreal() const { return inbox().count(); }
    size_t ncplx() const { return outbox().count(); }
    double* alloc_r() const { return new double[nreal()](); }
    cplx*   alloc_c() const { return new cplx[ncplx()](); }
    static void free_r(double* p) { delete[] p; }
    static void free_c(cplx* p) { delete[] p; }
    // fft3d_r2c constructor: reshape plans + local FFT plans (MKL commits its DFTI descriptors here; the
    // FFTW backend plans at the first execution, not included). Also allocates the workspace.
    double plan(double* = nullptr, cplx* = nullptr) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        fft.reset(new heffte::fft3d_r2c<FftBackend>(inbox(), outbox(), 2, MPI_COMM_WORLD, options));
        MPI_Barrier(MPI_COMM_WORLD);
        double tp = MPI_Wtime() - t0;
        if (rank == 0) std::cout << "heFFTe plan time: " << tp << " s\n";
        ws.assign(fft->size_workspace(), cplx(0.0, 0.0));
        return tp;
    }
    size_t size_workspace() const { return fft->size_workspace(); }
    void forward(double* r, cplx* c) { fft->forward(r, c, ws.data(), heffte::scale::none); }
    void backward(cplx* c, double* r) { fft->backward(c, r, ws.data(), heffte::scale::full); }
    double roundtrip_norm() const { return 1.0; }
    // x fastest, global indices i, j, k
    template <class F> void for_real(F f) const {
        const heffte::box3d<>& inbox_r = inbox();
        for (int i = inbox_r.low[0]; i <= inbox_r.high[0]; ++i)
            for (int j = inbox_r.low[1]; j <= inbox_r.high[1]; ++j)
                for (int k = inbox_r.low[2]; k <= inbox_r.high[2]; ++k)
                    f((size_t)(i - inbox_r.low[0]) + (size_t)(j - inbox_r.low[1]) * inbox_r.size[0]
                          + (size_t)(k - inbox_r.low[2]) * inbox_r.size[0] * inbox_r.size[1], i, j, k);
    }
    std::string settings() const {
        return std::string("decomp=") + nsmode::decomp_name(decomp_id) + " backend=" + kBackendName
             + " reorder=" + (options.use_reorder ? "1" : "0") + " algo=" + nsmode::heffte_algo_name(options.algorithm)
             + " reshapes=" + std::to_string(n_reshape);
    }
    void destroy() { fft.reset(); std::vector<cplx>().swap(ws); }   // before MPI_Finalize (frees heFFTe's communicators)
};
} // namespace nsfft
#endif
