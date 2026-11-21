#!/bin/bash
# backend_parse.sh <log> — table of the RESULT lines of backend_1proc (seconds per transform, median over the
#   repetitions; pair = forward + backward, with min-max), then the two ratios the supervisor asked for:
#   heFFTe-MKL / heFFTe-FFTW and native MKL / native FFTW MEASURE (< 1 = MKL faster).
L=${1:?usage: backend_parse.sh <log>}
grep -E "^\[check\].*FAIL|^\[libs\]" "$L" | sort | uniq -c
awk '
  function kv(k,   i, p) { for (i = 1; i <= NF; i++) { p = index($i, "="); if (p && substr($i, 1, p - 1) == k) return substr($i, p + 1) } return "" }
  /^RESULT lib=backend/ {
    if (!h++) printf "%5s %-11s %-17s %9s %9s %9s %-21s %8s %9s\n", "N", "binary", "method", "fwd", "bwd", "pair", "(min-max)", "plan_s", "err"
    n = kv("N"); b = "heffte_" kv("heffte_backend"); m = kv("method"); p = kv("t_pair_s")
    printf "%5s %-11s %-17s %9s %9s %9s (%s-%s) %8s %9s\n", n, b, m, kv("t_fwd_s"), kv("t_bwd_s"), p, kv("t_pair_min_s"), kv("t_pair_max_s"), kv("plan_s"), kv("err")
    if (m ~ /^heffte_/) t[n, m] = p; else { t[n, m] += p; c[n, m]++ }
    if (!(n in seen)) { seen[n] = 1; ord[++k] = n }
  }
  END {
    print ""
    for (i = 1; i <= k; i++) {
      n = ord[i]
      fm = c[n, "fftw_measure"] ? t[n, "fftw_measure"] / c[n, "fftw_measure"] : 0
      mk = c[n, "mkl_dfti"] ? t[n, "mkl_dfti"] / c[n, "mkl_dfti"] : 0
      if (t[n, "heffte_fftw"] && t[n, "heffte_mkl"]) printf "N=%-4s heFFTe-MKL / heFFTe-FFTW      = %.3f\n", n, t[n, "heffte_mkl"] / t[n, "heffte_fftw"]
      if (fm && mk) printf "N=%-4s native MKL / FFTW MEASURE    = %.3f   (native rows: mean of both binaries)\n", n, mk / fm
      if (fm && t[n, "heffte_fftw"]) printf "N=%-4s heFFTe-FFTW / FFTW MEASURE  = %.3f\n", n, t[n, "heffte_fftw"] / fm
      if (mk && t[n, "heffte_mkl"])  printf "N=%-4s heFFTe-MKL / native MKL     = %.3f\n", n, t[n, "heffte_mkl"] / mk
    }
  }' "$L"
