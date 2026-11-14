/**
 * fftbench_fftw.cpp — pure FFT benchmark for FFTW-MPI (slab), same algorithm as the solver
 * NavierStokes_periodic_fftw.cpp: r2c forward with FFTW_MPI_TRANSPOSED_OUT and c2r backward
 * with FFTW_MPI_TRANSPOSED_IN, i.e. one global transpose per FFT (no inverse transpose).
 *
 * Usage: fftbench_fftw NX NY NZ [iters=20] [nfields=3] [omp_threads=1]
 * Env:   FB_FLAG = estimate | measure | patient   (default measure)
 *        FB_CHECK = 0 | 1                          (default 1)
 *
 * FB_FLAG=estimate reproduces how heFFTe's FFTW backend plans (it always uses FFTW_ESTIMATE),
 * so the effect of the planner flag alone can be separated from heFFTe's communication.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <fftw3-mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double max_all(double v) { double m; MPI_Allreduce(&v, &m, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return m; }

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, P; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (argc < 4) { if (!rank) fprintf(stderr, "Usage: %s NX NY NZ [iters=20] [nfields=3] [omp=1]\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    const ptrdiff_t NX = atoi(argv[1]), NY = atoi(argv[2]), NZ = atoi(argv[3]);
    const int ITERS = argc > 4 ? atoi(argv[4]) : 20;
    const int NF    = argc > 5 ? atoi(argv[5]) : 3;
    const int NT    = argc > 6 ? atoi(argv[6]) : 1;
    const ptrdiff_t NZC = NZ / 2 + 1;

#ifdef FB_THREADS
    fftw_init_threads();
#endif
    fftw_mpi_init();
#ifdef FB_THREADS
    fftw_plan_with_nthreads(NT);
#endif

    const char* fs = getenv("FB_FLAG"); if (!fs) fs = "measure";
    unsigned flag = FFTW_MEASURE;
    if (!strcmp(fs, "estimate")) flag = FFTW_ESTIMATE;
    else if (!strcmp(fs, "patient")) flag = FFTW_PATIENT;
    else if (strcmp(fs, "measure")) { if (!rank) fprintf(stderr, "FB_FLAG must be estimate|measure|patient\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    ptrdiff_t lnx, lx0, lny, ly0;
    const ptrdiff_t alloc = fftw_mpi_local_size_3d_transposed(NX, NY, NZC, MPI_COMM_WORLD, &lnx, &lx0, &lny, &ly0);

    std::vector<double*> buf(NF);
    for (int f = 0; f < NF; ++f) { buf[f] = fftw_alloc_real(2 * alloc); std::fill(buf[f], buf[f] + 2 * alloc, 0.0); }
    double t_plan0 = MPI_Wtime();
    fftw_plan pf = fftw_mpi_plan_dft_r2c_3d(NX, NY, NZ, buf[0], (fftw_complex*)buf[0], MPI_COMM_WORLD, flag | FFTW_MPI_TRANSPOSED_OUT);
    fftw_plan pb = fftw_mpi_plan_dft_c2r_3d(NX, NY, NZ, (fftw_complex*)buf[0], buf[0], MPI_COMM_WORLD, flag | FFTW_MPI_TRANSPOSED_IN);
    double t_plan = max_all(MPI_Wtime() - t_plan0);
    for (int f = 0; f < NF; ++f) std::fill(buf[f], buf[f] + 2 * alloc, 0.0);   // planning may scribble

    if (!rank) {
        printf("============================================================\n");
        printf(" FFTW-MPI pure FFT benchmark  %td x %td x %td  ranks=%d  threads=%d  flag=%s  nfields=%d  iters=%d\n",
               NX, NY, NZ, P, NT, fs, NF, ITERS);
#ifndef FB_THREADS
        if (NT > 1) printf(" WARNING: built without -DFB_THREADS, omp_threads ignored\n");
#endif
        printf(" planning time %.2f s\n", t_plan);
        printf("============================================================\n");
        fflush(stdout);
    }

    const ptrdiff_t NZP = 2 * NZC;
    if (getenv("FB_CHECK") == nullptr || atoi(getenv("FB_CHECK"))) {
        double* r = buf[0];
        const double tp = 2.0 * M_PI;
        auto val = [&](ptrdiff_t x, ptrdiff_t y, ptrdiff_t z) { return sin(tp * x / NX) * cos(2.0 * tp * y / NY) + 0.5 * cos(3.0 * tp * z / NZ); };
        for (ptrdiff_t x = 0; x < lnx; ++x) for (ptrdiff_t y = 0; y < NY; ++y) for (ptrdiff_t z = 0; z < NZ; ++z)
            r[(x * NY + y) * NZP + z] = val(x + lx0, y, z);
        fftw_mpi_execute_dft_r2c(pf, r, (fftw_complex*)r);
        fftw_mpi_execute_dft_c2r(pb, (fftw_complex*)r, r);
        double e = 0.0, sc = 1.0 / ((double)NX * NY * NZ);
        for (ptrdiff_t x = 0; x < lnx; ++x) for (ptrdiff_t y = 0; y < NY; ++y) for (ptrdiff_t z = 0; z < NZ; ++z)
            e = std::max(e, std::fabs(r[(x * NY + y) * NZP + z] * sc - val(x + lx0, y, z)));
        e = max_all(e);
        if (!rank) { printf("[check] forward+backward max error = %.3e %s\n", e, e < 1e-10 ? "(OK)" : "(FAIL)"); fflush(stdout); }
        std::fill(r, r + 2 * alloc, 0.0);
    }

    auto do_f = [&]() { for (int f = 0; f < NF; ++f) fftw_mpi_execute_dft_r2c(pf, buf[f], (fftw_complex*)buf[f]); };
    auto do_b = [&]() { for (int f = 0; f < NF; ++f) fftw_mpi_execute_dft_c2r(pb, (fftw_complex*)buf[f], buf[f]); };
    for (int w = 0; w < 2; ++w) { do_f(); do_b(); }
    MPI_Barrier(MPI_COMM_WORLD);
    double tf = 0, tb = 0;
    for (int it = 0; it < ITERS; ++it) {
        double t0 = MPI_Wtime(); do_f(); double t1 = MPI_Wtime(); do_b(); double t2 = MPI_Wtime();
        tf += t1 - t0; tb += t2 - t1;
    }
    const double nfft = (double)ITERS * NF;
    const double t_f = max_all(tf / nfft), t_b = max_all(tb / nfft), t_m = 0.5 * (t_f + t_b);
    if (!rank) {
        printf("[time] per FFT: forward %.3f ms, backward %.3f ms, mean %.3f ms -> 36 FFT/step = %.4f s/step\n", t_f * 1e3, t_b * 1e3, t_m * 1e3, 36 * t_m);
        printf("RESULT lib=fftw N=%tdx%tdx%td P=%d threads=%d flag=%s nf=%d t_fwd_ms=%.4f t_bwd_ms=%.4f t_mean_ms=%.4f step_fft_s=%.4f plan_s=%.2f\n",
               NX, NY, NZ, P, NT, fs, NF, t_f * 1e3, t_b * 1e3, t_m * 1e3, 36 * t_m, t_plan);
        fflush(stdout);
    }
    fftw_destroy_plan(pf); fftw_destroy_plan(pb);
    for (int f = 0; f < NF; ++f) fftw_free(buf[f]);
    fftw_mpi_cleanup();
    MPI_Finalize();
    return 0;
}
