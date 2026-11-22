// ns_fft_fftw.h — FFTW-MPI 3D r2c/c2r exactly as NavierStokes_periodic_fftw.cpp uses it.
// Shared by the solver and scripts/fftbench/fftbench.cpp (one copy of the plan / execute code).
//
//   slab only (x slabs); NS_DECOMP is ignored and only printed
//   real space    [x_local][y][2*(nz/2+1)] (padded, z fastest), x split
//   spectral      transposed output [y_local][x][nz/2+1] (FFTW_MPI_TRANSPOSED_OUT / _IN)
//   NS_FFTW_PLAN  estimate | measure (default) | patient | exhaustive
//   NS_PRINT_PLAN=1: rank 0 prints both plans (fftw_sprint_plan; -DNS_FFTW_NO_SPRINT: fftw_fprint_plan)
//   unnormalised both ways: backward(forward(x)) = N x
//
// Order of calls (the solver's order): init(threads) -> decompose(nx,ny,nz) -> [banner: print_decomp()]
//   -> setup() -> allocate nreal()/ncplx() -> plan(any_r, any_c) -> forward/backward ... -> destroy()
#ifndef NS_FFT_FFTW_H
#define NS_FFT_FFTW_H
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <fftw3.h>
#include "../common_mode.h"

namespace nsfft {
struct Fftw {
    using cplx = fftw_complex;
    static const char* lib_name() { return "FFTW"; }
    int rank = 0, nprocs = 1, nthreads = 1;
    ptrdiff_t nx = 0, ny = 0, nz = 0;
    ptrdiff_t alloc_local = 0, local_n0 = 0, local_0_start = 0, local_n1 = 0, local_1_start = 0;
    int rigor_id = nsmode::PLAN_MEASURE;
    fftw_plan r2c = nullptr, c2r = nullptr;

    // FFTW threads: fftw_init_threads before fftw_mpi_init, thread count before planning.
    // Link libfftw3_omp, not libfftw3_threads: the _threads workers inherit the main thread's
    // CPU affinity, which libgomp pins when OMP_PROC_BIND/OMP_PLACES are set -> serial FFT.
    void init(int nt) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        nthreads = nt;
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
        fftw_mpi_init();
    }
    // real space: x distributed [x_local][y][z]; spectral (transposed out): y distributed [y_local][x][z]
    void decompose(ptrdiff_t nx_, ptrdiff_t ny_, ptrdiff_t nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        alloc_local = fftw_mpi_local_size_3d_transposed(nx, ny, nz / 2 + 1, MPI_COMM_WORLD,
                                                        &local_n0, &local_0_start, &local_n1, &local_1_start);
    }
    int decomp() const { return nsmode::DECOMP_SLAB; }
    // FFTW MPI is slab only (x split); NS_DECOMP has no effect and is only printed
    void print_decomp() const {
        if (rank != 0) return;
        char pgrid[96];
        snprintf(pgrid, sizeof pgrid, "%dx1 (x slabs; FFTW MPI is slab only, NS_DECOMP ignored)", nprocs);
        nsmode::print_decomp(nsmode::DECOMP_SLAB, pgrid);
    }
    void setup() {}
    size_t nreal() const { return 2 * (size_t)alloc_local; }
    size_t ncplx() const { return (size_t)alloc_local; }
    double* alloc_r() const { return fftw_alloc_real(nreal()); }
    cplx*   alloc_c() const { return fftw_alloc_complex(ncplx()); }
    static void free_r(double* p) { fftw_free(p); }
    static void free_c(cplx* p) { fftw_free(p); }

    // Planner flag NS_FFTW_PLAN; two plans for all transforms (new-array execute API: all arrays come
    // from fftw_alloc_*, so any pair of buffers can be used to plan). Returns the planning time (s).
    double plan(double* any_r, cplx* any_c) {
        rigor_id = nsmode::plan_rigor_from_env();
        if (rigor_id == nsmode::PLAN_INVALID) {
            if (rank == 0) std::cerr << "Error: NS_FFTW_PLAN must be estimate|measure|patient|exhaustive" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const unsigned rigor_flags[] = { FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE };
        if (rank == 0) std::cout << "FFTW planner flag: FFTW_" << nsmode::plan_rigor_name(rigor_id)
                                 << " (NS_FFTW_PLAN)" << std::endl;
        if (rank == 0) std::cout << "Initializing FFTW plans..." << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        // spectral space stays transposed [y][x][z] (y distributed): no global transpose at the end of
        // the forward and the start of the backward transform
        r2c = fftw_mpi_plan_dft_r2c_3d(nx, ny, nz, any_r, any_c, MPI_COMM_WORLD, rigor_flags[rigor_id] | FFTW_MPI_TRANSPOSED_OUT);
        c2r = fftw_mpi_plan_dft_c2r_3d(nx, ny, nz, any_c, any_r, MPI_COMM_WORLD, rigor_flags[rigor_id] | FFTW_MPI_TRANSPOSED_IN);
        MPI_Barrier(MPI_COMM_WORLD);
        double tp = MPI_Wtime() - t0;
        if (rank == 0) std::cout << "FFTW planning time: " << tp << " s" << std::endl;
        print_plans();
        return tp;
    }
    // NS_PRINT_PLAN=1: rank 0 prints the forward and backward plan (the MPI transpose FFTW chose,
    // e.g. mpi-transpose-alltoall / -pairwise). FFTW < 3.3.5 has no fftw_sprint_plan: -DNS_FFTW_NO_SPRINT.
    void print_plans() const {
        if (!(rank == 0 && getenv("NS_PRINT_PLAN") && !strcmp(getenv("NS_PRINT_PLAN"), "1"))) return;
        const fftw_plan plans[2] = {r2c, c2r};
        const char* names[2] = {"forward r2c", "backward c2r"};
        for (int k = 0; k < 2; ++k) {
            std::cout << "FFTW PLAN: ---- " << names[k] << " (rank 0)" << std::endl;
#ifndef NS_FFTW_NO_SPRINT
            char* txt = fftw_sprint_plan(plans[k]);
#else
            char* txt = nullptr; size_t len = 0;
            FILE* ms = open_memstream(&txt, &len);
            if (ms) { fftw_fprint_plan(plans[k], ms); fclose(ms); }
#endif
            if (txt) {
                for (char* line = strtok(txt, "\n"); line; line = strtok(nullptr, "\n"))
                    std::cout << "FFTW PLAN: " << line << "\n";
                free(txt);
            }
            std::cout << std::flush;
        }
    }
    void forward(double* r, cplx* c) const { fftw_mpi_execute_dft_r2c(r2c, r, c); }
    void backward(cplx* c, double* r) const { fftw_mpi_execute_dft_c2r(c2r, c, r); }
    double roundtrip_norm() const { return 1.0 / ((double)nx * ny * nz); }
    // every local real point: f(index, global x, y, z index)
    template <class F> void for_real(F f) const {
        for (ptrdiff_t i = 0; i < local_n0; ++i)
            for (ptrdiff_t j = 0; j < ny; ++j)
                for (ptrdiff_t k = 0; k < nz; ++k)
                    f((size_t)((i * ny + j) * (2 * (nz / 2 + 1)) + k), (int)(local_0_start + i), (int)j, (int)k);
    }
    std::string settings() const {
        return std::string("decomp=slab plan=") + nsmode::plan_rigor_name(rigor_id);
    }
    void destroy() {
        if (r2c) fftw_destroy_plan(r2c);
        if (c2r) fftw_destroy_plan(c2r);
        r2c = c2r = nullptr;
    }
};
} // namespace nsfft
#endif
