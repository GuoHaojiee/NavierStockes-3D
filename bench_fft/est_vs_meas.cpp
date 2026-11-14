// est_vs_meas.cpp — what do heFFTe's FFTW plans cost compared with normally planned ones?
//
// heFFTe 2.4.1 (include/heffte_backend_fftw.h) creates every 1-D batch plan as
//     fftw_plan_many_dft     (1,&n,howmany, NULL,NULL,stride,dist,  NULL,NULL,stride,dist,  sign, FFTW_ESTIMATE)
//     fftw_plan_many_dft_r2c (1,&n,howmany, NULL,NULL,stride,rdist, NULL,NULL,stride,cdist,       FFTW_ESTIMATE)
// i.e. (1) flag FFTW_ESTIMATE and (2) NULL for both arrays, so FFTW sees in == out ("in-place"),
// and then executes the r2c plan OUT of place with fftw_execute_dft_r2c(plan, real, complex).
// This program times that against plans made on real arrays with ESTIMATE / MEASURE, prints the
// plans FFTW chose, and checks whether wisdom from a MEASURE plan is picked up by heFFTe-style
// planning (FFTW documents that MEASURE wisdom is used by later ESTIMATE planning).
//
// Build: g++ -O3 -std=c++17 est_vs_meas.cpp -lfftw3 -lm -o est_vs_meas      (add -I/-L for /opt/fftw3)
// Usage: est_vs_meas N HOWMANY [reps=20] [patient=0]
//   N        length of one 1-D transform (512 for 512^3)
//   HOWMANY  transforms per batch (512^3 on 128 ranks: 512^3/128/512 = 2048)
// To include memory-bandwidth contention start one copy per core (srun -n 128 ...): only copy 0 prints.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <string>
#include <fftw3.h>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
template <class F> static double best_of(int reps, F f) {
    double b = 1e30;
    for (int r = 0; r < reps; ++r) { double t0 = now(); f(); b = std::min(b, now() - t0); }
    return b;
}
static std::string plan_str(fftw_plan p) {            // one-line description of the chosen plan
    char* s = fftw_sprint_plan(p);
    std::string o;
    for (char* c = s; *c; ++c) o += (*c == '\n' ? ' ' : *c);
    free(s);
    std::string t;                                    // squeeze blanks
    for (char ch : o) if (!(ch == ' ' && !t.empty() && t.back() == ' ')) t += ch;
    return t.size() > 150 ? t.substr(0, 150) + " ..." : t;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s N HOWMANY [reps=20] [patient=0]\n", argv[0]); return 1; }
    // started as many copies by srun (one per core, to load the memory like the real run): only copy 0 prints
    for (const char* v : {"SLURM_PROCID", "PMIX_RANK", "PMI_RANK"})
        if (const char* r = getenv(v)) { if (atoi(r) != 0 && !freopen("/dev/null", "w", stdout)) return 1; break; }
    int n = atoi(argv[1]), hm = atoi(argv[2]), reps = argc > 3 ? atoi(argv[3]) : 20;
    bool patient = argc > 4 && atoi(argv[4]) != 0;
    int nc = n / 2 + 1;
    size_t tot = (size_t)n * hm;
    fftw_complex* a = fftw_alloc_complex(tot);                   // c2c, in place
    double* r = fftw_alloc_real(tot);                            // r2c input
    fftw_complex* c = fftw_alloc_complex((size_t)nc * hm);       // r2c output
    auto fill = [&]() {
        for (size_t i = 0; i < tot; ++i) { a[i][0] = (double)(i % 97) / 97.0; a[i][1] = (double)(i % 31) / 31.0; r[i] = (double)(i % 89) / 89.0; }
    };
    printf("N=%d  howmany=%d  (%.1f MB complex per batch)\n", n, hm, tot * 16.0 / 1e6);

    const char* layouts[] = {"contiguous: stride 1, dist N        (heFFTe use_reorder=1, default for fftw backend)",
                             "strided:    stride HOWMANY, dist 1  (heFFTe use_reorder=0)"};
    for (int lay = 0; lay < 2; ++lay) {
        const int stride = lay == 0 ? 1 : hm, dist = lay == 0 ? n : 1, cdist = lay == 0 ? nc : 1;
        printf("\n== %s\n", layouts[lay]);
        struct V { const char* name; unsigned flag; bool null_arrays; };
        V v[] = {{"heFFTe way: ESTIMATE + NULL arrays", FFTW_ESTIMATE, true},
                 {"ESTIMATE on real arrays           ", FFTW_ESTIMATE, false},
                 {"MEASURE  on real arrays           ", FFTW_MEASURE, false},
                 {"PATIENT  on real arrays           ", FFTW_PATIENT, false}};
        double base_c = 0, base_r = 0;
        for (int k = 0; k < (patient ? 4 : 3); ++k) {
            fftw_forget_wisdom();                                // every variant plans from scratch
            double tp0 = now();
            fftw_plan pc = fftw_plan_many_dft(1, &n, hm, v[k].null_arrays ? nullptr : a, nullptr, stride, dist,
                                              v[k].null_arrays ? nullptr : a, nullptr, stride, dist, FFTW_FORWARD, v[k].flag);
            fftw_plan pr = fftw_plan_many_dft_r2c(1, &n, hm, v[k].null_arrays ? nullptr : r, nullptr, stride, dist,
                                                  v[k].null_arrays ? nullptr : c, nullptr, stride, cdist, v[k].flag);
            double tplan = now() - tp0;
            if (!pc || !pr) { printf("%s : planning failed\n", v[k].name); continue; }
            fill();
            fftw_execute_dft(pc, a, a); fftw_execute_dft_r2c(pr, r, c);
            double tc = best_of(reps, [&]() { fftw_execute_dft(pc, a, a); });
            fill();
            double tr = best_of(reps, [&]() { fftw_execute_dft_r2c(pr, r, c); });
            if (k == 0) { base_c = tc; base_r = tr; }
            printf("%s | c2c %8.3f ms (%.2fx faster than heFFTe way) | r2c %8.3f ms (%.2fx) | planning %.2f s\n",
                   v[k].name, tc * 1e3, base_c / tc, tr * 1e3, base_r / tr, tplan);
            if (lay == 0 && k < 3) printf("      r2c plan: %s\n", plan_str(pr).c_str());
            fftw_destroy_plan(pc); fftw_destroy_plan(pr);
        }
        // Wisdom: plan once with MEASURE on real arrays, in the same in-place / aligned form heFFTe asks for,
        // then plan the heFFTe way (ESTIMATE + NULL). If wisdom is used, the heFFTe-style plan gets MEASURE speed.
        fftw_forget_wisdom();
        {
            fftw_plan w1 = fftw_plan_many_dft(1, &n, hm, a, nullptr, stride, dist, a, nullptr, stride, dist, FFTW_FORWARD, FFTW_MEASURE);
            fftw_plan pc = fftw_plan_many_dft(1, &n, hm, nullptr, nullptr, stride, dist, nullptr, nullptr, stride, dist, FFTW_FORWARD, FFTW_ESTIMATE);
            fill();
            fftw_execute_dft(pc, a, a);
            double tc = best_of(reps, [&]() { fftw_execute_dft(pc, a, a); });
            printf("heFFTe way after MEASURE wisdom    | c2c %8.3f ms (%.2fx faster than heFFTe way)\n", tc * 1e3, base_c / tc);
            fftw_destroy_plan(w1); fftw_destroy_plan(pc);
        }
    }
    fftw_free(a); fftw_free(r); fftw_free(c);
    return 0;
}
