// ns_fft_accfft.h — AccFFT 3D r2c/c2r exactly as NavierStokes_periodic_accfft.cpp uses it.
// Shared by the solver and scripts/fftbench/fftbench.cpp (one copy of the plan / execute code).
//
//   NS_DECOMP = auto | slab | pencil
//     slab  : c_dims = {p,1}: AccFFT's oneD path (src/accfft.cpp: oneD = true when np[1]==1), 1 transpose per FFT
//     pencil: c_dims = {0,0}: the library picks P0 x P1 with MPI_Dims_create, 2 transposes per FFT
//   real space    [isize0][isize1][nz] (z fastest, stride = global nz), r2c along z
//   spectral      osize/ostart (x whole, y split by P0, z split by P1) for both decompositions
//   NS_FFTW_PLAN  estimate | measure (default) | patient (stock AccFFT treats PATIENT as MEASURE;
//                 -DNS_ACCFFT_PATCHED for the patched library)
//   NS_ACCFFT_PLAN  same values, AccFFT only; overrides NS_FFTW_PLAN here (the other libraries keep NS_FFTW_PLAN)
//   Transpose method (src/accfft.cpp, src/transpose.cpp of the library):
//     flag ACCFFT_MEASURE: T_Plan::which_fast_method times every candidate ONCE (one warm-up call + one timed call,
//       max over ranks) and keeps the fastest: method 1 = Isend/Irecv to all ranks, method 2 = MPI_Alltoall
//       (MPI_Alltoallv if the data is not evenly distributed), method 3 = par::Mpi_Alltoallv_dense, k-way exchange of
//       Isend/Irecv rounds, kway = p/2, p/4, ... > 4, async (kway_async=1) or not (only for p a power of two > 7).
//       Slab: T_plan_2 (forward) is measured, T_plan_2i (backward) gets -method. Pencil: T_plan_1 on the ranks with
//       coord[0] == 0, broadcast, copied to T_plan_1i / T_plan_2 / T_plan_2i.
//     any other flag (ESTIMATE, PATIENT in the stock library): method 2, kway 2, no measurement.
//     The chosen method is printed by rank 0 after the plan:
//       FFTB_ACCFFT_TPLAN sel=measure|default|fixed method=.. kway=.. kway_async=.. call=.. comm_size=.. measured=...
//     and is part of settings() (CSV lib_settings).
//   NS_ACCFFT_TMETHOD  auto (default: the library's choice) | 1 | 2 | 3:<kway>:a | 3:<kway>:s   (also m1, m2, m3k16a ...)
//     Fixes the method after planning by setting the public T_Plan fields method / kway / kway_async of the plan's
//     transpose plans (forward +method, backward -method), the same thing accfft_plan_dft_3d_r2c does after its
//     measurement; the library source is not changed. kway must split the communicator (p = c * kway^n, c < kway).
//   accfft_execute_c2r destroys its input; unnormalised both ways: backward(forward(x)) = N x
//   AccFFT keeps buffer sizes in int: more than 2 GB of FFT buffer per rank aborts.
//
// Order of calls (the solver's order): init(threads) -> decompose(nx,ny,nz) -> [banner: print_decomp()]
//   -> setup() -> allocate nreal()/ncplx() -> plan(any_r, any_c) -> forward/backward ... -> destroy()
#ifndef NS_FFT_ACCFFT_H
#define NS_FFT_ACCFFT_H
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <mpi.h>
#include <accfft.h>
#include "../common_mode.h"

namespace nsfft {
struct Accfft {
    using cplx = Complex;
    static const char* lib_name() { return "AccFFT"; }
    int rank = 0, nprocs = 1, nthreads = 1;
    int nx = 0, ny = 0, nz = 0, n[3] = {0, 0, 0};
    int decomp_id = nsmode::DECOMP_SLAB, rigor_id = nsmode::PLAN_MEASURE;
    int c_dims[2] = {0, 0};
    MPI_Comm c_comm = MPI_COMM_NULL;
    int isize[3], istart[3], osize[3], ostart[3];
    int alloc_max = 0;
    long local_size_r = 0, local_size_c = 0;
    accfft_plan* p = nullptr;
    std::string tplan_desc;               // chosen transpose method, for settings()

    void init(int nt) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        nthreads = nt;                     // accfft_init(nthreads) follows the local size in decompose()
    }
    void decompose(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_; n[0] = nx; n[1] = ny; n[2] = nz;
        // slab splits x in real space and y in spectral space: both need at least p points
        decomp_id = nsmode::decomp_from_env(nprocs, std::min(nx, ny));
        if (decomp_id != nsmode::DECOMP_SLAB && decomp_id != nsmode::DECOMP_PENCIL) {
            if (rank == 0) std::cerr << "Error: NS_DECOMP=" << nsmode::decomp_env_value()
                                     << " (AccFFT: expected auto|slab|pencil)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (decomp_id == nsmode::DECOMP_SLAB && nprocs > std::min(nx, ny)) {
            if (rank == 0) std::cerr << "Error: slab needs nprocs <= min(nx, ny) = " << std::min(nx, ny)
                                     << ", got " << nprocs << "; use NS_DECOMP=pencil or auto\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // slab: {p,1} -> np[1]==1, AccFFT oneD = true; pencil: {0,0} lets AccFFT choose the grid
        c_dims[0] = c_dims[1] = 0;
        if (decomp_id == nsmode::DECOMP_SLAB) { c_dims[0] = nprocs; c_dims[1] = 1; }
        accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
        alloc_max = accfft_local_size_dft_r2c(n, isize, istart, osize, ostart, c_comm);
        local_size_r = (long)isize[0] * isize[1] * nz;
        local_size_c = (long)osize[0] * osize[1] * osize[2];
        // AccFFT keeps buffer byte counts (alloc_max / alloc_local) in int: more than 2 GB per rank
        // overflows (a library limit). Stop here instead of running with wrongly sized buffers.
        {
            const long avg_bytes = (long)nx * ny * (nz / 2 + 1) * (long)sizeof(Complex) / nprocs;
            const long local_bytes = local_size_c * (long)sizeof(Complex);
            int bad_local = (avg_bytes >= 2147483647L || (long)alloc_max < local_bytes) ? 1 : 0, bad = 0;
            MPI_Allreduce(&bad_local, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (bad) {
                if (rank == 0)
                    std::cerr << "Error: AccFFT stores buffer sizes in int; this grid needs about "
                              << avg_bytes / 1048576L << " MB of FFT buffer per rank (limit 2048 MB). "
                              << "Use more MPI ranks.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        accfft_init(nthreads);
    }
    int decomp() const { return decomp_id; }
    void print_decomp() const {
        if (rank != 0) return;
        char grid[160];
        snprintf(grid, sizeof grid, "%dx%d (c_dims)%s", c_dims[0], c_dims[1],
                 (decomp_id == nsmode::DECOMP_PENCIL && (c_dims[0] == 1 || c_dims[1] == 1))
                     ? " -- MPI_Dims_create gave a 1D grid, AccFFT runs it as slab" : "");
        nsmode::print_decomp(decomp_id, grid);
        std::cout << "Real space local:     ["
                  << istart[0] << "+" << isize[0] << "] x ["
                  << istart[1] << "+" << isize[1] << "] x [0+" << isize[2] << "]\n";
        std::cout << "Spectral space local: ["
                  << ostart[0] << "+" << osize[0] << "] x ["
                  << ostart[1] << "+" << osize[1] << "] x ["
                  << ostart[2] << "+" << osize[2] << "]\n";
    }
    void setup() {}
    // allocation as in the solver: real local_size_r doubles, complex alloc_max bytes (accfft_alloc)
    size_t nreal() const { return (size_t)local_size_r; }
    size_t ncplx() const { return (size_t)alloc_max / sizeof(Complex); }
    double* alloc_r() const { return (double*)accfft_alloc(local_size_r * sizeof(double)); }
    cplx*   alloc_c() const { return (Complex*)accfft_alloc((long)alloc_max); }
    static void free_r(double* q) { accfft_free(q); }
    static void free_c(cplx* q) { accfft_free(q); }

    // one plan for all transforms of this size; planner flag NS_FFTW_PLAN. Returns the plan time (s).
    double plan(double* any_r, cplx* any_c) {
        const char* pvar = plan_var();
        rigor_id = nsmode::plan_rigor_from_env(nsmode::PLAN_MEASURE, pvar);
        if (rigor_id == nsmode::PLAN_INVALID || rigor_id == nsmode::PLAN_EXHAUSTIVE) {
            if (rank == 0) std::cerr << "Error: " << pvar << " must be estimate|measure|patient for AccFFT\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int fm = 0, fk = 0, fa = 0;        // NS_ACCFFT_TMETHOD, checked before the (long) planning
        const bool fixed = tmethod_from_env(fm, fk, fa);
        const unsigned acc_flags[] = { ACCFFT_ESTIMATE, ACCFFT_MEASURE, ACCFFT_PATIENT };
#ifdef NS_ACCFFT_PATCHED
        const char* acc_lib = "patched AccFFT: PATIENT -> FFTW_PATIENT";
#else
        const char* acc_lib = "stock AccFFT: PATIENT is treated as FFTW_MEASURE";
#endif
        if (rank == 0) std::cout << "AccFFT planner flag: ACCFFT_" << nsmode::plan_rigor_name(rigor_id)
                                 << " (" << pvar << "; " << acc_lib << ")\n";
        if (rank == 0) std::cout << "Creating AccFFT plan...\n";
        MPI_Barrier(c_comm);
        double t0 = MPI_Wtime();
        p = accfft_plan_dft_3d_r2c(n, any_r, (double*)any_c, c_comm, acc_flags[rigor_id]);
        MPI_Barrier(c_comm);
        double tp = MPI_Wtime() - t0;
        if (rank == 0) std::cout << "AccFFT plan time: " << tp << " s\n";
        const char* sel = fixed ? "fixed" : (acc_flags[rigor_id] == ACCFFT_MEASURE ? "measure" : "default");
        if (fixed) set_tmethod(fm, fk, fa);
        report_tplan(sel);
        return tp;
    }

    // ---- transpose method (see the header comment)
    static const char* plan_var() {
        const char* a = std::getenv("NS_ACCFFT_PLAN");
        return (a && *a) ? "NS_ACCFFT_PLAN" : "NS_FFTW_PLAN";
    }
    // the transpose plans of this plan: slab T_plan_2 / T_plan_2i, pencil also T_plan_1 / T_plan_1i
    std::vector<T_Plan<double>*> tplans() const {
        std::vector<T_Plan<double>*> v;
        if (!p) return v;
        if (p->T_plan_1) v.push_back(p->T_plan_1);
        if (p->T_plan_1i) v.push_back(p->T_plan_1i);
        if (p->T_plan_2) v.push_back(p->T_plan_2);
        if (p->T_plan_2i) v.push_back(p->T_plan_2i);
        return v;
    }
    static bool kway_splits(int np, int k) {     // par::Mpi_Alltoallv_dense: assert((np_new / kway) * kway == np_new)
        if (k < 2) return false;
        for (int q = np; q > 1;) { int kk = std::min(k, q); if (q % kk) return false; q /= kk; }
        return true;
    }
    static std::string method_label(int m, int k, int a) {
        char b[48];
        if (m == 3) snprintf(b, sizeof b, "m3k%d%s", k, a ? "a" : "s"); else snprintf(b, sizeof b, "m%d", m);
        return b;
    }
    // NS_ACCFFT_TMETHOD: false = auto (library's choice); aborts on an invalid value
    bool tmethod_from_env(int& m, int& k, int& a) const {
        const char* s = std::getenv("NS_ACCFFT_TMETHOD");
        if (!s || !*s || !std::strcmp(s, "auto")) return false;
        m = 0; k = 0; a = 1;
        char c = 0; int n1 = 0;
        if (!std::strcmp(s, "1") || !std::strcmp(s, "m1")) m = 1;
        else if (!std::strcmp(s, "2") || !std::strcmp(s, "m2")) m = 2;
        else if ((std::sscanf(s, "3:%d:%c%n", &k, &c, &n1) == 2 && s[n1] == 0) ||
                 (std::sscanf(s, "m3k%d%c%n", &k, &c, &n1) == 2 && s[n1] == 0)) {
            if (c == 'a' || c == 's') { m = 3; a = (c == 'a'); }
        }
        if (!m) {
            if (rank == 0) std::cerr << "Error: NS_ACCFFT_TMETHOD=" << s << " (auto | 1 | 2 | 3:<kway>:a | 3:<kway>:s, or m1 m2 m3k<kway>a m3k<kway>s)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (m == 1) k = nprocs;          // as which_fast_method sets it (unused by method 1)
        if (m == 2) k = 2;
        return true;
    }
    void set_tmethod(int m, int k, int a) {
        for (T_Plan<double>* t : tplans())
            if (m == 3 && !kway_splits((int)t->nprocs, k)) {
                if (rank == 0) std::cerr << "Error: NS_ACCFFT_TMETHOD: kway=" << k << " does not split a transpose communicator of "
                                         << t->nprocs << " ranks (p = c * kway^n, c < kway)\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        // forward plans +m, backward plans -m (accfft.cpp: T_plan_2i->method = -T_plan_2->method)
        for (T_Plan<double>* t : tplans()) {
            const bool inv = (t == p->T_plan_1i || t == p->T_plan_2i);
            t->method = inv ? -m : m; t->kway = k; t->kway_async = (a != 0);
        }
    }
    void report_tplan(const char* sel) {
        T_Plan<double>* t = p->oneD ? p->T_plan_2 : p->T_plan_1;   // the measured / primary plan
        if (!t) { tplan_desc = "tplan=none"; return; }
        const int m = std::abs(t->method);
        const char* call = m == 1 ? "Isend/Irecv" : m == 2 ? (t->is_evenly_distributed ? "MPI_Alltoall" : "MPI_Alltoallv")
                         : m == 3 ? (t->kway_async ? "kway-dense-async" : "kway-dense-sync") : "other";
        const std::string lab = method_label(m, t->kway, t->kway_async ? 1 : 0);
        // candidates timed by which_fast_method (rank 0's copy holds the max over ranks); empty unless measured here
        std::string meas;
        if (!std::strcmp(sel, "measure") && t->pwhich_f_time && (p->oneD || p->coord[0] == 0)) {
            for (const auto& e : *t->pwhich_f_time) {
                const int f = e.first;
                std::string l = (f == t->nprocs) ? "m1" : (f == 2) ? "m2" : method_label(3, std::abs(f), f > 0 ? 1 : 0);
                char b[64]; snprintf(b, sizeof b, "%s%s:%.3e", meas.empty() ? "" : ";", l.c_str(), e.second);
                meas += b;
            }
        }
        char d[256];
        snprintf(d, sizeof d, "tsel=%s tmethod=%s call=%s comm=%d", sel, lab.c_str(), call, (int)t->nprocs);
        tplan_desc = d;
        if (rank == 0) {
            std::printf("FFTB_ACCFFT_TPLAN sel=%s method=%d kway=%d kway_async=%d label=%s call=%s comm_size=%d evenly=%d%s measured=%s\n",
                        sel, t->method, (int)t->kway, t->kway_async ? 1 : 0, lab.c_str(), call, (int)t->nprocs,
                        t->is_evenly_distributed ? 1 : 0, p->oneD ? "" : " (pencil: row plan; column plans use the same method)",
                        meas.empty() ? "-" : meas.c_str());
            std::fflush(stdout);
        }
    }
    void forward(double* r, cplx* c) const { accfft_execute_r2c(p, r, c); }
    void backward(cplx* c, double* r) const { accfft_execute_c2r(p, c, r); }
    double roundtrip_norm() const { return 1.0 / ((double)nx * ny * nz); }
    template <class F> void for_real(F f) const {
        for (int i = 0; i < isize[0]; ++i)
            for (int j = 0; j < isize[1]; ++j)
                for (int k = 0; k < nz; ++k)
                    f((size_t)((long)i * isize[1] * nz + (long)j * nz + k), i + istart[0], j + istart[1], k);
    }
    std::string settings() const {
        return std::string("decomp=") + nsmode::decomp_name(decomp_id) + " plan=" + nsmode::plan_rigor_name(rigor_id) +
               (tplan_desc.empty() ? "" : " " + tplan_desc);
    }
    void destroy() {
        if (p) accfft_destroy_plan(p);
        p = nullptr;
        if (c_comm != MPI_COMM_NULL) MPI_Comm_free(&c_comm);
        accfft_cleanup();
    }
};
} // namespace nsfft
#endif
