// ns_fft_p3dfft.h — P3DFFT 2.x (C interface) 3D r2c/c2r exactly as NavierStokes_periodic_p3dfft.cpp uses it.
// Shared by the solver and scripts/fftbench/fftbench.cpp (one copy of the setup / execute code).
//
//   NS_DECOMP = auto | slab | pencil; dims = {P1, P2}, P1 = dims[0] = iproc, P2 = dims[1] = jproc
//     slab  : dims = {1, p}: ftran/btran skip the X<->Y transpose ("if (iproc .gt. 1)"), 1 transpose per FFT;
//             real space split in z, spectral space in y
//     pencil: dims = MPI_Dims_create(p, 2), 2 transposes per FFT
//   layout (STRIDE1 undefined): real Nx x Ny/P1 x Nz/P2, spectral (Nx/2+1)/P1 x Ny/P2 x Nz, x fastest,
//   1-based global indices from Cp3dfft_get_dims; r2c along x
//   planner flag fixed when the library was built (-DNS_P3DFFT_PLAN="..." records it), library path
//   -DNS_P3DFFT_LIB="..."; the FFTW planning happens in Cp3dfft_setup ("P3DFFT setup (planning) time")
//   btran_c2r destroys its input; unnormalised both ways: backward(forward(x)) = N x
//
// Order of calls (the solver's order): init(threads) -> decompose(nx,ny,nz) -> [banner: print_decomp()]
//   -> setup() (= Cp3dfft_setup, the planning) -> allocate nreal()/ncplx() -> plan() (returns the setup
//   time) -> forward/backward ... -> destroy()
#ifndef NS_FFT_P3DFFT_H
#define NS_FFT_P3DFFT_H
#include <algorithm>
#include <complex>
#include <cstdio>
#include <iostream>
#include <string>
#include <mpi.h>
#include <omp.h>
#include <fftw3.h>
#include <p3dfft.h>              // no include guard in p3dfft.h: include it only here, not again in the solver
#include "../common_mode.h"

#ifndef NS_P3DFFT_PLAN
#define NS_P3DFFT_PLAN "unknown (library default is MEASURE)"
#endif
#ifndef NS_P3DFFT_LIB
#define NS_P3DFFT_LIB "not recorded (built without -DNS_P3DFFT_LIB)"
#endif

namespace nsfft {
struct P3dfft {
    using cplx = std::complex<double>;
    static const char* lib_name() { return "P3DFFT"; }
    int rank = 0, nprocs = 1, nthreads = 1;
    int nx = 0, ny = 0, nz = 0;
    int decomp_id = nsmode::DECOMP_SLAB;
    int dims[2] = {0, 0};
    int istart[3], iend[3], isize[3];   // real space local range (1-based global indices)
    int fstart[3], fend[3], fsize[3];   // spectral space local range
    int memsize[3];
    int idx_offset = 1;
    double t_setup = 0.0;

    // FFTW threads before Cp3dfft_setup, so that P3DFFT's FFTW plans use them (as FFTW / AccFFT)
    void init(int nt) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        nthreads = nt;
        omp_set_num_threads(nthreads);
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
    }
    void decompose(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        // slab {1,p} splits z in real space (slowest in Fortran order) and y in spectral space
        decomp_id = nsmode::decomp_from_env(nprocs, std::min(ny, nz));
        if (decomp_id != nsmode::DECOMP_SLAB && decomp_id != nsmode::DECOMP_PENCIL) {
            if (rank == 0) std::cerr << "Error: NS_DECOMP=" << nsmode::decomp_env_value()
                                     << " (P3DFFT: expected auto|slab|pencil)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (decomp_id == nsmode::DECOMP_SLAB && nprocs > std::min(ny, nz)) {
            if (rank == 0) std::cerr << "Error: slab needs nprocs <= min(ny, nz) = " << std::min(ny, nz)
                                     << ", got " << nprocs << "; use NS_DECOMP=pencil or auto\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        dims[0] = dims[1] = 0;
        if (decomp_id == nsmode::DECOMP_SLAB) { dims[0] = 1; dims[1] = nprocs; }
        else                                  MPI_Dims_create(nprocs, 2, dims);
    }
    int decomp() const { return decomp_id; }
    void print_decomp() const {
        if (rank != 0) return;
        char grid[160];
        snprintf(grid, sizeof grid, "%dx%d (dims = P1 x P2)%s", dims[0], dims[1],
                 (decomp_id == nsmode::DECOMP_PENCIL && (dims[0] == 1 || dims[1] == 1))
                     ? " -- MPI_Dims_create gave a 1D grid" : "");
        nsmode::print_decomp(decomp_id, grid);
    }
    // Cp3dfft_setup = the FFTW planning of P3DFFT (flag fixed at library build), then the local ranges
    void setup() {
        if (rank == 0) std::cout << "P3DFFT planner flag (fixed at library build): " << NS_P3DFFT_PLAN << "\n";
        if (rank == 0) std::cout << "P3DFFT library: " << NS_P3DFFT_LIB << "\n";
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        Cp3dfft_setup(dims, nx, ny, nz,
                      MPI_Comm_c2f(MPI_COMM_WORLD),
                      nx, ny, nz,
                      1,       // OW=1: btran_c2r may overwrite its input
                      memsize);
        MPI_Barrier(MPI_COMM_WORLD);
        t_setup = MPI_Wtime() - t0;
        if (rank == 0) std::cout << "P3DFFT setup (planning) time: " << t_setup << " s\n";
        // real space (conf=1) and spectral space (conf=2) local ranges (1-based global indices)
        int conf = 1;
        Cp3dfft_get_dims(istart, iend, isize, conf);
        conf = 2;
        Cp3dfft_get_dims(fstart, fend, fsize, conf);
        idx_offset = (istart[0] == 0) ? 0 : 1;
        if (rank == 0) {
            std::cout << "idx_offset = " << idx_offset << "\n";
            std::cout << "Real space:     ["
                      << istart[0] << "," << iend[0] << "] x ["
                      << istart[1] << "," << iend[1] << "] x ["
                      << istart[2] << "," << iend[2] << "]\n";
            std::cout << "Spectral space: ["
                      << fstart[0] << "," << fend[0] << "] x ["
                      << fstart[1] << "," << fend[1] << "] x ["
                      << fstart[2] << "," << fend[2] << "]\n";
        }
    }
    // real space: memsize[2] as the last dimension (with FFT padding)
    size_t nreal() const { return (size_t)isize[0] * isize[1] * memsize[2]; }
    size_t ncplx() const { return (size_t)fsize[0] * fsize[1] * fsize[2]; }
    double* alloc_r() const { return new double[nreal()](); }
    cplx*   alloc_c() const { return new cplx[ncplx()](); }
    static void free_r(double* p) { delete[] p; }
    static void free_c(cplx* p) { delete[] p; }
    double plan(double*, cplx*) const { return t_setup; }   // the planning was Cp3dfft_setup
    void forward(double* r, cplx* c) const {
        unsigned char op_f[] = "fft";
        Cp3dfft_ftran_r2c(r, reinterpret_cast<double*>(c), op_f);
    }
    void backward(cplx* c, double* r) const {
        unsigned char op_b[] = "tff";
        Cp3dfft_btran_c2r(reinterpret_cast<double*>(c), r, op_b);
    }
    double roundtrip_norm() const { return 1.0 / ((double)nx * ny * nz); }
    template <class F> void for_real(F f) const {
        for (int k = istart[2]; k <= iend[2]; ++k)
            for (int j = istart[1]; j <= iend[1]; ++j)
                for (int i = istart[0]; i <= iend[0]; ++i)
                    f((size_t)(i - istart[0]) + (size_t)(j - istart[1]) * isize[0]
                          + (size_t)(k - istart[2]) * isize[0] * isize[1],
                      i - idx_offset, j - idx_offset, k - idx_offset);
    }
    std::string settings() const {
        return std::string("decomp=") + nsmode::decomp_name(decomp_id) + " plan=" + NS_P3DFFT_PLAN;
    }
    void destroy() {}
};
} // namespace nsfft
#endif
