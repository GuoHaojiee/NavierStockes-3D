#!/bin/bash
# pl_parse.sh <log> — one line per configuration: median (min–max) over repetitions of
#   step time, FFT time per step, planning time; plus the number of runs whose energy matches.
# Reference kinetic energy for 512^3, dt=1e-6, 20 steps: 1.7933833444e+08 (set PL_EREF for other cases).
L=${1:?usage: pl_parse.sh <log>}
EREF=${PL_EREF:-1.7933833444e+08}
awk -v eref="$EREF" '
function flush() {
    if (key == "") return
    n[key]++; st[key, n[key]] = step; ff[key, n[key]] = fft; pl[key, n[key]] = plan
    if (energy == eref) ok[key]++
    if (rc != "0") bad[key]++
    if (!(key in seen)) { seen[key] = 1; order[++nk] = key; pflag[key] = flagline }
    key = ""
}
function med(arr, k, m,   i, j, t, v) {     # median of arr[k,1..m]
    for (i = 1; i <= m; i++) v[i] = arr[k, i]
    for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
    lo = v[1]; hi = v[m]
    return (m % 2) ? v[(m + 1) / 2] : (v[m / 2] + v[m / 2 + 1]) / 2
}
/^>>>/ { flush(); key = $3 " " $6; step = fft = plan = ""; energy = ""; rc = "?"; flagline = "" ; next }
/planner flag/                 { flagline = $0; sub(/^ */, "", flagline) }
/planning time:|plan time:|\(planning\) time:/ { plan = $NF; if (plan == "s") plan = $(NF-1) }
/Kinetic energy/               { energy = $NF }
/Avg per step:/                { step = $(NF-1) }
/FFT \(incl/                   { fft = $(NF-1) }
/^<<< exit=/                   { split($0, a, "="); rc = a[2]; flush() }
END {
    flush()
    printf "%-26s %-14s %4s  %-24s %-24s %-22s %s\n", "config", "plan", "runs", "step s: med (min-max)", "FFT s: med (min-max)", "planning s: med", "energy ok / failed"
    for (i = 1; i <= nk; i++) {
        k = order[i]; m = n[k]; split(k, p, " ")
        s = med(st, k, m); slo = lo; shi = hi
        f = med(ff, k, m); flo = lo; fhi = hi
        q = med(pl, k, m)
        printf "%-26s %-14s %4d  %.4f (%.4f-%.4f)   %.4f (%.4f-%.4f)   %-22.3f %d / %d\n", p[1], p[2], m, s, slo, shi, f, flo, fhi, q, ok[k] + 0, bad[k] + 0
    }
    print ""
    for (i = 1; i <= nk; i++) if (pflag[order[i]] != "") print order[i] ": " pflag[order[i]]
}' "$L"
grep -h "stride-1" "$L" | sort | uniq -c
