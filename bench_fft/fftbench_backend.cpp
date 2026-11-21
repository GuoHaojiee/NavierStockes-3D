// fftbench_backend.cpp — local FFT backends without any communication (one MPI rank).
//
// Question (supervisor): heFFTe supports two local backends, FFTW and MKL. Which one is faster when
// no communication is involved? Native FFTW and native MKL are measured as references.
//
// One 3-D r2c forward + c2r backward transform of N^3 doubles per iteration, out of place, the
// backward transform scaled by 1/N^3 (the solver's convention: forward scale::none, backward scale::full).
//   heffte         heffte::fft3d_r2c<backend>, the whole box on one rank (no MPI reshape), r2c along
//                  FB_R2C_DIR (default 2, as in NavierStokes_periodic_heffte_v2.cpp). The backend is
//                  fixed at compile time: -DFB_HEFFTE_BACKEND=fftw|mkl. heFFTe's FFTW backend plans
//                  1-D batches with FFTW_ESTIMATE (hard-coded in heFFTe 2.4.1).
//   fftw_measure   fftw_plan_dft_r2c_3d / c2r_3d, FFTW_MEASURE, backward followed by a 1/N^3 loop
//   fftw_estimate  the same with FFTW_ESTIMATE (shows how much of heFFTe-FFTW vs native is the planner)
//   mkl            DFTI, 3-D real, CCE storage, DFTI_BACKWARD_SCALE = 1/N^3
//
// Timing: for every repetition and method, iterations are repeated until the window (>= 2 s by
// default) is full; the time per transform is window / iterations. Repetitions are the OUTER loop
// (rep 1: all methods, rep 2: all methods, ...), so a slow period hits all methods equally.
// One untimed warm-up iteration per method (heFFTe creates its FFTW/MKL plans on first use).
// Check per method: max|backward(forward(x)) - x| / max|x| < 1e-12, and the Parseval sum of the
// spectrum must equal sum x^2 to 1e-12 (relative).
//
// Build: see scripts/intel/build_intel_solvers.sh (fb_backend_fftw, fb_backend_mkl). The program
// always links real FFTW AND MKL; -lfftw3 must come before the MKL libraries on the link line,
// because MKL's interface library also exports fftw_* wrappers. The library that actually provides
// fftw_execute / DftiComputeForward is printed at start-up ([libs]).
//
// Usage:  fb_backend N [reps=5] [window_s=2] [threads=1]
//   FB_METHODS=heffte,fftw_measure,fftw_estimate,mkl   (default: all, in this order)
// Output: one "RESULT lib=backend ..." line per method (median over reps), parsed by scripts/intel/backend_parse.sh
#include <algorithm>
#include <array>
#include <memory>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <fftw3.h>
#include <mkl.h>
#include <heffte.h>

#define FB_HB_fftw 1
#define FB_HB_mkl  2
#define FB_CAT2(a, b) a##b
#define FB_CAT(a, b) FB_CAT2(a, b)
#define FB_STR2(a) #a
#define FB_STR(a) FB_STR2(a)
#ifndef FB_HEFFTE_BACKEND
#define FB_HEFFTE_BACKEND fftw
#endif
#if FB_CAT(FB_HB_, FB_HEFFTE_BACKEND) == FB_HB_fftw
using HBackend = heffte::backend::fftw;
#elif FB_CAT(FB_HB_, FB_HEFFTE_BACKEND) == FB_HB_mkl
#ifndef Heffte_ENABLE_MKL
#error "FB_HEFFTE_BACKEND=mkl: this heFFTe was built without -DHeffte_ENABLE_MKL=ON"
#endif
using HBackend = heffte::backend::mkl;
#else
#error "FB_HEFFTE_BACKEND must be fftw or mkl"
#endif
static const std::string kHB = FB_STR(FB_HEFFTE_BACKEND);

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
// Shared library that provides a symbol at run time (dlsym, not &func: in the executable &func is the PLT stub)
static const char* lib_of(const char* name) {
    Dl_info di;
    void* sym = dlsym(RTLD_DEFAULT, name);
    return (sym && dladdr(sym, &di) && di.dli_fname) ? di.dli_fname : "? (not found by dlsym)";
}
static void dfti_check(MKL_LONG st, const char* what) {
    if (st && !DftiErrorClass(st, DFTI_NO_ERROR)) {
        fprintf(stderr, "DFTI error in %s: %s\n", what, DftiErrorMessage(st));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size();
    return m == 0 ? 0.0 : (m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]));
}

// One method: forward/backward callables over the shared arrays.
struct Method {
    std::string name;
    double plan_s = 0, first_s = 0, err = -1, parseval = 0;
    std::vector<double> fwd, bwd;      // per rep: seconds per transform
    std::vector<long> iters;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual ~Method() {}
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int P; MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (P != 1) { fprintf(stderr, "fb_backend: run with exactly one MPI rank (got %d)\n", P); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (argc < 2) { fprintf(stderr, "Usage: %s N [reps=5] [window_s=2] [threads=1]\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    const int n = atoi(argv[1]);
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const double window = argc > 3 ? atof(argv[3]) : 2.0;
    const int nt = argc > 4 ? atoi(argv[4]) : 1;
    const int r2c_dir = getenv("FB_R2C_DIR") ? atoi(getenv("FB_R2C_DIR")) : 2;
    std::string methods = getenv("FB_METHODS") ? getenv("FB_METHODS") : "heffte,fftw_measure,fftw_estimate,mkl";

    omp_set_num_threads(nt);
    fftw_init_threads();
    fftw_plan_with_nthreads(nt);
    mkl_set_num_threads(nt);

    const size_t nr = (size_t)n * n * n;
    const size_t nc = (size_t)n * n * (n / 2 + 1);
    const double scale = 1.0 / (double)nr;
    double* in   = fftw_alloc_real(nr);
    double* back = fftw_alloc_real(nr);
    std::complex<double>* out = reinterpret_cast<std::complex<double>*>(fftw_alloc_complex(nc));

    char mklver[256]; mkl_get_version_string(mklver, sizeof mklver);
    printf("============================================================\n");
    printf(" local FFT backends, one rank, %d^3 r2c+c2r, out of place, threads=%d, reps=%d, window>=%.1f s\n", n, nt, reps, window);
    printf(" heFFTe backend (compile time): %s   heFFTe r2c direction: %d\n", kHB.c_str(), r2c_dir);
    printf(" FFTW: %s\n MKL : %s\n", fftw_version, mklver);
    for (const char* f : {"fftw_execute", "fftw_plan_dft_r2c_3d", "fftw_plan_many_dft_r2c", "DftiComputeForward"})
        printf("[libs] %-22s from %s\n", f, lib_of(f));
    printf(" memory: real %.2f GiB x2, complex %.2f GiB\n", nr * 8.0 / (1 << 30), nc * 16.0 / (1 << 30));
    printf("============================================================\n");
    fflush(stdout);

    auto fill = [&]() {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nr; ++i) in[i] = std::sin(0.37 * (double)(i % 1009)) + 0.25 * std::cos(0.011 * (double)(i % 7919));
    };

    std::vector<Method*> M;
    std::vector<std::complex<double>> hws;   // heFFTe workspace

    // ---------------- heFFTe
    struct HM : Method {
        std::unique_ptr<heffte::fft3d_r2c<HBackend>> f; double* in; double* back; std::complex<double>* out; std::complex<double>* ws;
        void forward() override  { f->forward(in, out, ws, heffte::scale::none); }
        void backward() override { f->backward(out, back, ws, heffte::scale::full); }
    };
    // ---------------- native FFTW
    struct FM : Method {
        fftw_plan pf, pb; double* back; size_t nr; double scale;
        void forward() override  { fftw_execute(pf); }
        void backward() override {
            fftw_execute(pb);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nr; ++i) back[i] *= scale;
        }
    };
    // ---------------- native MKL (DFTI)
    struct KM : Method {
        DFTI_DESCRIPTOR_HANDLE hf, hb; double* in; double* back; std::complex<double>* out;
        void forward() override  { dfti_check(DftiComputeForward(hf, in, out), "ComputeForward"); }
        void backward() override { dfti_check(DftiComputeBackward(hb, out, back), "ComputeBackward"); }
    };

    size_t pos = 0;
    while (pos <= methods.size()) {
        size_t e = methods.find(',', pos); if (e == std::string::npos) e = methods.size();
        std::string m = methods.substr(pos, e - pos); pos = e + 1;
        if (m.empty()) continue;
        double t0 = now();
        if (m == "heffte") {
            auto* h = new HM; h->name = "heffte_" + kHB + (r2c_dir != 2 ? "_r2c" + std::to_string(r2c_dir) : "");
            if (r2c_dir < 0 || r2c_dir > 2) { fprintf(stderr, "FB_R2C_DIR must be 0, 1 or 2\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            std::array<int, 3> hi = {n - 1, n - 1, n - 1};
            hi[r2c_dir] = n / 2;
            heffte::box3d<> rb({0, 0, 0}, {n - 1, n - 1, n - 1});
            heffte::box3d<> cb({0, 0, 0}, hi);
            auto opt = heffte::default_options<HBackend>();
            h->f.reset(new heffte::fft3d_r2c<HBackend>(rb, cb, r2c_dir, MPI_COMM_WORLD, opt));
            hws.resize(h->f->size_workspace());
            h->in = in; h->back = back; h->out = out; h->ws = hws.data();
            printf(" heFFTe options: algorithm=%d use_reorder=%d use_pencils=%d; workspace %.2f GiB\n",
                   (int)opt.algorithm, opt.use_reorder ? 1 : 0, opt.use_pencils ? 1 : 0, hws.size() * 16.0 / (1 << 30));
            M.push_back(h);
        } else if (m == "fftw_measure" || m == "fftw_estimate") {
            auto* f = new FM; f->name = m; f->back = back; f->nr = nr; f->scale = scale;
            unsigned fl = (m == "fftw_measure") ? FFTW_MEASURE : FFTW_ESTIMATE;
            f->pf = fftw_plan_dft_r2c_3d(n, n, n, in, reinterpret_cast<fftw_complex*>(out), fl);
            f->pb = fftw_plan_dft_c2r_3d(n, n, n, reinterpret_cast<fftw_complex*>(out), back, fl);
            if (!f->pf || !f->pb) { fprintf(stderr, "FFTW planning failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
            M.push_back(f);
        } else if (m == "mkl") {
            auto* k = new KM; k->name = "mkl_dfti"; k->in = in; k->back = back; k->out = out;
            MKL_LONG dims[3] = {n, n, n};
            MKL_LONG rs[4] = {0, (MKL_LONG)n * n, n, 1};
            MKL_LONG cs[4] = {0, (MKL_LONG)n * (n / 2 + 1), n / 2 + 1, 1};
            dfti_check(DftiCreateDescriptor(&k->hf, DFTI_DOUBLE, DFTI_REAL, 3, dims), "create f");
            dfti_check(DftiSetValue(k->hf, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "placement f");
            dfti_check(DftiSetValue(k->hf, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "cce f");
            dfti_check(DftiSetValue(k->hf, DFTI_INPUT_STRIDES, rs), "istride f");
            dfti_check(DftiSetValue(k->hf, DFTI_OUTPUT_STRIDES, cs), "ostride f");
            dfti_check(DftiCommitDescriptor(k->hf), "commit f");
            dfti_check(DftiCreateDescriptor(&k->hb, DFTI_DOUBLE, DFTI_REAL, 3, dims), "create b");
            dfti_check(DftiSetValue(k->hb, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "placement b");
            dfti_check(DftiSetValue(k->hb, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "cce b");
            dfti_check(DftiSetValue(k->hb, DFTI_INPUT_STRIDES, cs), "istride b");
            dfti_check(DftiSetValue(k->hb, DFTI_OUTPUT_STRIDES, rs), "ostride b");
            dfti_check(DftiSetValue(k->hb, DFTI_BACKWARD_SCALE, scale), "scale b");
            dfti_check(DftiCommitDescriptor(k->hb), "commit b");
            M.push_back(k);
        } else {
            fprintf(stderr, "unknown method '%s' (heffte|fftw_measure|fftw_estimate|mkl)\n", m.c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        M.back()->plan_s = now() - t0;
        printf(" planned %-14s in %.3f s\n", M.back()->name.c_str(), M.back()->plan_s); fflush(stdout);
    }

    // warm-up + checks (planning overwrote the arrays, so fill afterwards)
    fill();
    double xmax = 0; for (size_t i = 0; i < nr; ++i) xmax = std::max(xmax, std::fabs(in[i]));
    for (Method* m : M) {
        double t0 = now(); m->forward(); m->first_s = now() - t0;
        // Parseval on the half spectrum (before backward: a multi-dimensional c2r destroys its input).
        // Weight 2 except k=0 and k=n/2 of the halved dimension; heFFTe's index order is dim 0 fastest,
        // FFTW/MKL here are row-major (last dimension fastest and halved).
        double s = 0; const size_t h = n / 2 + 1;
        const bool hef = m->name.rfind("heffte", 0) == 0;
        for (size_t idx = 0; idx < nc; ++idx) {
            size_t kh;
            if (!hef || r2c_dir == 0) kh = idx % h;
            else if (r2c_dir == 1)    kh = (idx / n) % h;
            else                      kh = idx / ((size_t)n * n);
            double w = (kh == 0 || (n % 2 == 0 && kh == (size_t)n / 2)) ? 1.0 : 2.0;
            s += w * std::norm(out[idx]);
        }
        m->parseval = s / (double)nr;          // sum|X|^2 / N = sum|x|^2
        t0 = now(); m->backward(); m->first_s += now() - t0;
        double e = 0;
        for (size_t i = 0; i < nr; ++i) e = std::max(e, std::fabs(back[i] - in[i]));
        m->err = e / xmax;
        printf("[check] %-14s first call %.3f s, round-trip max rel. error %.2e %s\n", m->name.c_str(), m->first_s,
               m->err, m->err < 1e-12 ? "(OK)" : "(FAIL)");
        fflush(stdout);
    }
    double pref = 0; for (size_t i = 0; i < nr; ++i) pref += in[i] * in[i];
    for (Method* m : M) {
        double d = std::fabs(m->parseval - pref) / pref;
        printf("[check] %-14s Parseval vs sum x^2 = %.12e: rel. diff %.2e %s\n", m->name.c_str(), pref, d, d < 1e-12 ? "(OK)" : "(FAIL)");
    }
    fflush(stdout);

    for (int r = 0; r < reps; ++r) {
        for (Method* m : M) {
            fill();
            long it = 0; double tf = 0, tb = 0, t0 = now(), ta, tbb;
            do {
                ta = now(); m->forward(); tbb = now(); m->backward();
                tf += tbb - ta; tb += now() - tbb; ++it;
            } while (now() - t0 < window);
            m->fwd.push_back(tf / it); m->bwd.push_back(tb / it); m->iters.push_back(it);
            printf("  rep %d %-14s iters %4ld  fwd %.4f s  bwd %.4f s  fwd+bwd %.4f s\n", r + 1, m->name.c_str(), it, tf / it, tb / it, (tf + tb) / it);
            fflush(stdout);
        }
    }
    for (Method* m : M) {
        std::vector<double> s(m->fwd.size());
        for (size_t i = 0; i < s.size(); ++i) s[i] = m->fwd[i] + m->bwd[i];
        auto mm = std::minmax_element(s.begin(), s.end());
        printf("RESULT lib=backend heffte_backend=%s method=%s N=%d threads=%d reps=%zu t_fwd_s=%.5f t_bwd_s=%.5f t_pair_s=%.5f t_pair_min_s=%.5f t_pair_max_s=%.5f plan_s=%.3f err=%.2e\n",
               kHB.c_str(), m->name.c_str(), n, nt, s.size(), median(m->fwd), median(m->bwd), median(s), *mm.first, *mm.second, m->plan_s, m->err);
    }
    fflush(stdout);
    for (Method* m : M) delete m;   // heFFTe objects before MPI_Finalize
    fftw_free(in); fftw_free(back); fftw_free(out);
    MPI_Finalize();
    return 0;
}
