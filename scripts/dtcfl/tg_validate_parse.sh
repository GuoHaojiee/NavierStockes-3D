#!/bin/bash
# tg_validate_parse.sh — Taylor–Green Re=1600 run of the FFTW solver vs the 512^3 spectral reference.
#   bash tg_validate_parse.sh <log> [local partial log] [reference .gdiag]
# <log>: output of tg_validate.sbatch (or any run with NS_IC=tg NS_DT_PRINT=K): "DT step t dt umax E eps Ceff lim"
#        lines (E = 0.5<|u|^2>, eps = nu<|w|^2>, at the start of the step) and the "FINAL t E eps ..." line.
# Reference (scripts/dtcfl/ref/spectral_Re1600_512.gdiag): t E -dE/dt Omega, Omega = 0.5<|w|^2> -> 2 nu Omega = nu<|w|^2>.
# Output:
#   1. peak of eps = nu<|w|^2> and of -dE/dt (centered differences of E), at the print resolution and refined by a
#      parabola through the three points around the maximum; the same for the reference
#   2. E, eps and -dE/dt at t = 5, 8, 9, 10, 12 and their deviation from the reference (relative)
#   3. [local partial log] line by line comparison on the common steps (same step -> same t): E and eps
# The reference is sampled every 0.01, the run every 0.1: values at the run's t are interpolated linearly.
LOG=${1:?usage: tg_validate_parse.sh <log> [local partial log] [reference .gdiag]}
LOC=$2
REF=${3:-$(dirname "$0")/ref/spectral_Re1600_512.gdiag}
[ -f "$LOG" ] || { echo "no such file: $LOG"; exit 1; }
[ -f "$REF" ] || { echo "no such file: $REF"; exit 1; }

NU=$(grep -m1 -oE "DTCFL_CONFIG.* nu=[^ (]+" "$LOG" | grep -oE "nu=[^ (]+" | cut -d= -f2)
NU=${NU:-0.000625}
echo "== $LOG"
grep -m1 -E "^Grid:" "$LOG"
grep -m1 -E "^DTCFL_CONFIG" "$LOG"
grep -E "^(BLOWUP|DTCFL_SUMMARY|DTCFL_CAP_HIT)" "$LOG"
grep -m1 -E "Avg per step:" "$LOG"
echo "reference: $REF  (nu = $NU)"

# run series "t E eps", sorted by t, duplicate t removed (FINAL repeats the last DT time only if T was printed)
SER=$(awk '$1=="DT" && NF>=9 {print $3, $6, $7}
           $1=="FINAL" {for(i=2;i<=NF;i++){split($i,a,"="); v[a[1]]=a[2]} print v["t"], v["E"], v["eps"]}' "$LOG" \
      | sort -g -k1,1 | awk '!seen[$1+0]++')
[ -n "$SER" ] || { echo "no DT lines in $LOG"; exit 1; }

echo "$SER" | awk -v nu="$NU" -v ref="$REF" '
function interp(tt,   i) {           # reference values at tt -> rE, rD, rW
    if (tt <= rt[1]) { rE = re[1]; rD = rd[1]; rW = rw[1]; return }
    for (i = 2; i <= nr; i++) if (rt[i] >= tt) {
        f = (tt - rt[i-1]) / (rt[i] - rt[i-1])
        rE = re[i-1] + f * (re[i] - re[i-1]); rD = rd[i-1] + f * (rd[i] - rd[i-1]); rW = rw[i-1] + f * (rw[i] - rw[i-1]); return }
    rE = re[nr]; rD = rd[nr]; rW = rw[nr]
}
function peak(x, y, n, name, unit,   i, im, a, b, c, d, tp, yp) {  # max at print resolution + parabola refinement
    im = 1; for (i = 2; i <= n; i++) if (y[i] > y[im]) im = i
    tp = x[im]; yp = y[im]
    if (im > 1 && im < n) {
        a = x[im-1]; b = x[im]; c = x[im+1]
        d = (b - a) * (y[im] - y[im+1]) - (b - c) * (y[im] - y[im-1])
        if (d != 0) {
            tp = b - 0.5 * ((b - a)^2 * (y[im] - y[im+1]) - (b - c)^2 * (y[im] - y[im-1])) / d
            # value of the parabola at tp
            l0 = (tp - b) * (tp - c) / ((a - b) * (a - c)); l1 = (tp - a) * (tp - c) / ((b - a) * (b - c)); l2 = (tp - a) * (tp - b) / ((c - a) * (c - b))
            yp = l0 * y[im-1] + l1 * y[im] + l2 * y[im+1]
        }
    }
    printf "  %-24s %.6f at t = %6.3f   (refined %.6f at t = %6.3f)%s\n", name, y[im], x[im], yp, tp, unit
}
BEGIN {
    while ((getline line < ref) > 0) {
        if (line ~ /^#/ || line ~ /^[ \t]*$/) continue
        split(line, p, " "); nr++; rt[nr] = p[1]; re[nr] = p[2]; rd[nr] = p[3]; rw[nr] = 2 * nu * p[4]
    }
}
{ n++; t[n] = $1; E[n] = $2; W[n] = $3 }
END {
    # -dE/dt of the run: centered (one-sided at the ends; non-uniform spacing allowed)
    for (i = 1; i <= n; i++) {
        if (i == 1)      D[i] = -(E[2] - E[1]) / (t[2] - t[1])
        else if (i == n) D[i] = -(E[n] - E[n-1]) / (t[n] - t[n-1])
        else             D[i] = -(E[i+1] - E[i-1]) / (t[i+1] - t[i-1])
    }
    for (i = 1; i <= nr; i++) { rtt[i] = rt[i]; rdd[i] = rd[i]; rww[i] = rw[i] }
    printf "run: %d points, t = %.3f .. %.3f\n", n, t[1], t[n]
    print "-- 1. peak of the dissipation rate"
    peak(t, W, n, "run nu<|w|^2>:", "")
    peak(t, D, n, "run -dE/dt:", "")
    peak(rtt, rww, nr, "ref 2nu*Omega:", "")
    peak(rtt, rdd, nr, "ref -dE/dt:", "")
    print "-- 2. deviation from the reference (rel = run/ref - 1)"
    printf "  %5s %10s %10s %8s   %10s %10s %8s   %10s %10s %8s\n", "t", "E", "E_ref", "rel", "nu<w^2>", "2nuOm_ref", "rel", "-dE/dt", "ref", "rel"
    split("5 8 9 10 12", tl, " ")
    for (k = 1; k <= 5; k++) {
        tt = tl[k]; j = 0
        for (i = 1; i <= n; i++) if (t[i] > tt - 1e-6 && t[i] < tt + 1e-6) j = i
        if (!j) { printf "  %5.1f   (not in the run)\n", tt; continue }
        interp(tt)
        printf "  %5.1f %10.6f %10.6f %+7.2f%%   %10.6f %10.6f %+7.2f%%   %10.6f %10.6f %+7.2f%%\n", tt,
               E[j], rE, 100 * (E[j] / rE - 1), W[j], rW, 100 * (W[j] / rW - 1), D[j], rD, 100 * (D[j] / rD - 1)
    }
}'

[ -n "$LOC" ] || exit 0
[ -f "$LOC" ] || { echo "no such file: $LOC"; exit 1; }
echo "-- 3. line by line vs the local partial run $LOC (same step)"
awk 'NR == FNR { if ($1 == "DT" && NF >= 9) { lt[$2] = $3; lE[$2] = $6; lW[$2] = $7 } next }
     $1 == "DT" && NF >= 9 && ($2 in lt) {
         n++; dE = ($6 - lE[$2]) / lE[$2]; dW = ($7 - lW[$2]) / lW[$2]
         if (dE < 0) dE = -dE; if (dW < 0) dW = -dW
         if (dE > mE) { mE = dE; tE = $3 } if (dW > mW) { mW = dW; tW = $3 }
         if (lt[$2] != $3) badt++
         printf "  step %5d t=%7.3f  E %.10e / %.10e  |rel| %.2e   eps %.10e / %.10e  |rel| %.2e\n", $2, $3, $6, lE[$2], dE, $7, lW[$2], dW
     }
     END {
         if (!n) { print "  no common steps (different NS_DT_PRINT or dt?)"; exit }
         printf "  common lines %d (cluster / local); max |rel| E %.2e (t=%.2f), eps %.2e (t=%.2f)%s\n", n, mE, tE, mW, tW,
                badt ? sprintf("; WARNING: %d lines with a different t", badt) : ""
         print "  (round-off differences between toolchains / rank counts / FFTW plans grow during the transition; small values = same computation)"
     }' "$LOC" "$LOG"
