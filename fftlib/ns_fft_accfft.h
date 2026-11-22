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
//   accfft_execute_c2r destroys its input; unnormalised both ways: backward(forward(x)) = N x
//   AccFFT keeps buffer sizes in int: more than 2 GB of FFT buffer per rank aborts.
//
// Order of calls (the solver's order): init(threads) -> decompose(nx,ny,nz) -> [banner: print_decomp()]
//   -> setup() -> allocate nreal()/ncplx() -> plan(any_r, any_c) -> forward/backward ... -> destroy()
#ifndef NS_FFT_ACCFFT_H
#define NS_FFT_ACCFFT_H
#include <algorithm>
#include <cstdio>
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
        rigor_id = nsmode::plan_rigor_from_env();
        if (rigor_id == nsmode::PLAN_INVALID || rigor_id == nsmode::PLAN_EXHAUSTIVE) {
            if (rank == 0) std::cerr << "Error: NS_FFTW_PLAN must be estimate|measure|patient for AccFFT\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const unsigned acc_flags[] = { ACCFFT_ESTIMATE, ACCFFT_MEASURE, ACCFFT_PATIENT };
#ifdef NS_ACCFFT_PATCHED
        const char* acc_lib = "patched AccFFT: PATIENT -> FFTW_PATIENT";
#else
        const char* acc_lib = "stock AccFFT: PATIENT is treated as FFTW_MEASURE";
#endif
        if (rank == 0) std::cout << "AccFFT planner flag: ACCFFT_" << nsmode::plan_rigor_name(rigor_id)
                                 << " (NS_FFTW_PLAN; " << acc_lib << ")\n";
        if (rank == 0) std::cout << "Creating AccFFT plan...\n";
        MPI_Barrier(c_comm);
        double t0 = MPI_Wtime();
        p = accfft_plan_dft_3d_r2c(n, any_r, (double*)any_c, c_comm, acc_flags[rigor_id]);
        MPI_Barrier(c_comm);
        double tp = MPI_Wtime() - t0;
        if (rank == 0) std::cout << "AccFFT plan time: " << tp << " s\n";
        return tp;
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
        return std::string("decomp=") + nsmode::decomp_name(decomp_id) + " plan=" + nsmode::plan_rigor_name(rigor_id);
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
