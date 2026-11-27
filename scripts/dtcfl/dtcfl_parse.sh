#!/bin/bash
# dtcfl_parse.sh — results of dtcfl.sbatch.
#   bash dtcfl_parse.sh <cfl jobid> <fixed jobid> <ref jobid> [probe jobid]
# Files: ${DTC_RESULTS:-/scratch/pr0208/guohaojie/results/dtcfl}/<jobid>/<label>.txt (labels cfl_C<C>, fixed_C<C>, ref, probe_C<C>).
# Output:
#   1. every run: steps (total, to t=10), wall time, s/step, dt_min / dt_max, steps limited by conv / visc / tend, blowup
#   2. per C: step ratio fixed/cfl to t=10 and t=20 (main metric, deterministic), wall-time ratio to t=20 (supporting)
#   3. per C: E and eps = nu<|w|^2> at t=10 and t=20, relative deviation from the reference run (a)
#   4. instability probe: completed / BLOWUP (t, last scanned steps); a BLOWUP at C < 0.9 is marked as possibly caused by
#      aliasing (no dealiasing), not by the CFL limit
#   5. CSV of the cfl and probe series: run,C,step,t,dt,umax,E,eps,Ceff,lim -> ${DTC_CSV:-<results>/dtcfl_series_<cfl jobid>.csv}
# t=10 values: E and eps come from the per-step DT lines (values at the start of each step), cubic (4-point Lagrange)
# interpolation at t=10 — the error is O(dt^4), far below the time-stepping errors compared here.  t=20: FINAL line
# (all runs end exactly at t=20: cfl cuts its last step, fixed uses dt = 20/NSTEPS).
# Steps to t=10: steps that start before t=10 (the step that crosses or reaches t=10 is counted).
CJ=${1:?usage: dtcfl_parse.sh <cfl jobid> <fixed jobid> <ref jobid> [probe jobid]}; FJ=${2:?fixed jobid}; RJ=${3:?ref jobid}; PJ=$4
RES=${DTC_RESULTS:-/scratch/pr0208/guohaojie/results/dtcfl}
CSV=${DTC_CSV:-$RES/dtcfl_series_$CJ.csv}
TA=10; TB=20

# stats <file>: one line "key=value ..." (missing values: NA)
stats() {
    awk -v ta=$TA '
    function lag(x, xs, ys, i0,   j, m, l, s) {   # 4-point Lagrange through points i0..i0+3
        s = 0
        for (j = 0; j < 4; j++) { l = 1; for (m = 0; m < 4; m++) if (m != j) l *= (x - xs[i0+m]) / (xs[i0+j] - xs[i0+m]); s += l * ys[i0+j] }
        return s
    }
    $1 == "DT" && NF >= 9 { n++; t[n] = $3; E[n] = $6; W[n] = $7; if ($3 < ta - 1e-9) s10++; if ($8 > cmax) cmax = $8 }
    $1 == "FINAL" { for (i = 2; i <= NF; i++) { split($i, a, "="); f[a[1]] = a[2] } }
    $1 == "DTCFL_SUMMARY" { for (i = 2; i <= NF; i++) { split($i, a, "="); s[a[1]] = a[2] } }
    $1 == "BLOWUP" { for (i = 2; i <= NF; i++) { split($i, a, "="); b[a[1]] = a[2] } }
    /Avg per step:/ { sps = $(NF-1) }
    END {
        e10 = "NA"; w10 = "NA"
        if (n >= 4 && t[n] > ta) {
            for (k = 1; k < n && t[k+1] <= ta; k++) ;
            i0 = k - 1; if (i0 < 1) i0 = 1; if (i0 + 3 > n) i0 = n - 3
            e10 = sprintf("%.12e", lag(ta, t, E, i0)); w10 = sprintf("%.12e", lag(ta, t, W, i0))
        }
        printf "steps=%s s10=%s wall=%s sps=%s dtmin=%s dtmax=%s conv=%s visc=%s tend=%s blowup=%s cap=%s tfin=%s E20=%s W20=%s E10=%s W10=%s cmax=%s bt=%s tlast=%s\n",
            (s["steps"] != "" ? s["steps"] : "NA"), (s10 != "" ? s10 : 0), (s["wall"] != "" ? s["wall"] : "NA"), (sps != "" ? sps : "NA"),
            (s["dt_min"] != "" ? s["dt_min"] : "NA"), (s["dt_maxused"] != "" ? s["dt_maxused"] : "NA"),
            (s["lim_conv"] != "" ? s["lim_conv"] : "NA"), (s["lim_visc"] != "" ? s["lim_visc"] : "NA"), (s["lim_tend"] != "" ? s["lim_tend"] : "NA"),
            (s["blowup"] != "" ? s["blowup"] : "NA"), (s["cap_hit"] != "" ? s["cap_hit"] : "NA"),
            (f["t"] != "" ? f["t"] : "NA"), (f["E"] != "" ? f["E"] : "NA"), (f["eps"] != "" ? f["eps"] : "NA"), e10, w10,
            (cmax != "" ? cmax : "NA"), (b["t"] != "" ? b["t"] : "NA"), (n ? t[n] : "NA")
    }' "$1"
}
get() { echo "$1" | grep -oE "(^| )$2=[^ ]+" | cut -d= -f2; }
ratio() { awk -v a="$1" -v b="$2" 'BEGIN { if (a == "NA" || b == "NA" || a == "" || b == "" || b + 0 == 0) print "NA"; else printf "%.3f", a / b }'; }
rel()   { awk -v a="$1" -v b="$2" 'BEGIN { if (a == "NA" || b == "NA" || a == "" || b == "" || b + 0 == 0) print "NA"; else printf "%+.2e", a / b - 1 }'; }

f_of() { echo "$RES/$1/$2.txt"; }
CS=$(ls "$RES/$CJ" 2>/dev/null | sed -nE 's/^cfl_C([0-9.]+)\.txt$/\1/p' | sort -g)
PS=$([ -n "$PJ" ] && ls "$RES/$PJ" 2>/dev/null | sed -nE 's/^probe_C([0-9.]+)\.txt$/\1/p' | sort -g)
[ -n "$CS" ] || { echo "no cfl_C*.txt in $RES/$CJ"; exit 1; }

echo "== dtcfl: cfl job $CJ, fixed job $FJ, ref job $RJ${PJ:+, probe job $PJ}   ($RES)"
grep -m1 "^Grid:" "$(f_of $CJ cfl_C$(echo $CS | awk '{print $1}'))"
echo "-- 1. runs"
printf "  %-14s %7s %7s %10s %9s %12s %12s %6s %6s %5s %s\n" run steps "to t=10" "wall [s]" "s/step" dt_min dt_max conv visc tend status
row() {   # row <label> <file>
    local st; [ -f "$2" ] || { printf "  %-14s (missing: %s)\n" "$1" "$2"; return; }
    st=$(stats "$2"); eval "S_$(echo $1 | tr . _)=\"\$st\""
    local status=ok
    [ "$(get "$st" blowup)" = 1 ] && status="BLOWUP t=$(get "$st" bt)"
    [ "$(get "$st" cap)" = 1 ] && status="STEP CAP t=$(get "$st" tlast)"
    [ "$(get "$st" steps)" = NA ] && status="no DTCFL_SUMMARY (did not finish)"
    printf "  %-14s %7s %7s %10s %9s %12s %12s %6s %6s %5s %s\n" "$1" $(get "$st" steps) $(get "$st" s10) $(get "$st" wall) $(get "$st" sps) \
           $(get "$st" dtmin) $(get "$st" dtmax) $(get "$st" conv) $(get "$st" visc) $(get "$st" tend) "$status"
}
row ref "$(f_of $RJ ref)"
for C in $CS; do row cfl_C$C "$(f_of $CJ cfl_C$C)"; row fixed_C$C "$(f_of $FJ fixed_C$C)"; done
for C in $PS; do row probe_C$C "$(f_of $PJ probe_C$C)"; done
echo "  (cfl: conv/visc/tend = steps whose dt was set by the convective bound / viscous bound / the cut to T_END;"
echo "   fixed runs scan every step as well (NS_DT_PRINT=1), so the wall times compare like for like)"

echo "-- 2. speed-up of cfl over the fixed baseline with the same C (fixed steps / cfl steps; wall: fixed / cfl, to t=20)"
printf "  %5s %9s %9s %9s %9s %9s %9s %11s\n" C "cfl t<10" "fix t<10" "ratio10" "cfl t<20" "fix t<20" "ratio20" "wall ratio"
for C in $CS; do
    k=$(echo $C | tr . _); eval "c=\$S_cfl_C$k f=\$S_fixed_C$k"
    printf "  %5s %9s %9s %9s %9s %9s %9s %11s\n" $C $(get "$c" s10) $(get "$f" s10) $(ratio $(get "$f" s10) $(get "$c" s10)) \
        $(get "$c" steps) $(get "$f" steps) $(ratio $(get "$f" steps) $(get "$c" steps)) $(ratio $(get "$f" wall) $(get "$c" wall))
done

echo "-- 3. deviation from the reference (a) (rel = run/ref - 1; eps = nu<|w|^2>)"
r=$S_ref
printf "  %-14s %12s %12s %12s %12s\n" "" "E(t=10)" "eps(t=10)" "E(t=20)" "eps(t=20)"
printf "  %-14s %12.6e %12.6e %12.6e %12.6e\n" "ref (a)" $(get "$r" E10) $(get "$r" W10) $(get "$r" E20) $(get "$r" W20) 2>/dev/null \
  || echo "  ref (a): missing"
for C in $CS; do
    k=$(echo $C | tr . _)
    for g in cfl fixed; do
        eval "x=\$S_${g}_C$k"
        printf "  %-14s %12s %12s %12s %12s\n" ${g}_C$C $(rel $(get "$x" E10) $(get "$r" E10)) $(rel $(get "$x" W10) $(get "$r" W10)) \
               $(rel $(get "$x" E20) $(get "$r" E20)) $(rel $(get "$x" W20) $(get "$r" W20))
    done
done

if [ -n "$PS" ]; then
    echo "-- 4. instability probe (T_END=12)"
    for C in $PS; do
        k=$(echo $C | tr . _); eval "p=\$S_probe_C$k"
        if [ "$(get "$p" blowup)" = 1 ]; then
            note=""; awk -v c=$C 'BEGIN { exit !(c < 0.9) }' && note="  [C < 0.9: possibly aliasing (no dealiasing), not the CFL limit]"
            echo "  C=$C: BLOWUP at t=$(get "$p" bt)$note; last scanned steps (step t dt umax E eps Ceff):"
            grep "^BLOWUP_HIST [0-9]" "$(f_of $PJ probe_C$C)" | sed 's/^BLOWUP_HIST/    /'
        elif [ "$(get "$p" cap)" = 1 ]; then echo "  C=$C: step cap reached at t=$(get "$p" tlast) (dt collapsed?)"
        elif [ "$(get "$p" steps)" = NA ]; then echo "  C=$C: did not finish (no DTCFL_SUMMARY)"
        else echo "  C=$C: stable to t=$(get "$p" tfin) in $(get "$p" steps) steps, max Ceff $(get "$p" cmax)"; fi
    done
fi
# a BLOWUP among the main cfl runs with C < 0.9
for C in $CS; do
    k=$(echo $C | tr . _); eval "x=\$S_cfl_C$k"
    [ "$(get "$x" blowup)" = 1 ] && awk -v c=$C 'BEGIN { exit !(c < 0.9) }' && {
        echo "  NOTE cfl_C$C: BLOWUP at C < 0.9 — possibly aliasing (no dealiasing), not the CFL limit; last scanned steps:"
        grep "^BLOWUP_HIST [0-9]" "$(f_of $CJ cfl_C$C)" | sed 's/^BLOWUP_HIST/    /'; }
done

echo "-- 5. series CSV"
{
    echo "run,C,step,t,dt,umax,E,eps,Ceff,lim"
    for C in $CS; do awk -v r=cfl -v c=$C '$1 == "DT" && NF >= 9 { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", r, c, $2, $3, $4, $5, $6, $7, $8, $9 }' "$(f_of $CJ cfl_C$C)"; done
    for C in $PS; do awk -v r=probe -v c=$C '$1 == "DT" && NF >= 9 { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", r, c, $2, $3, $4, $5, $6, $7, $8, $9 }' "$(f_of $PJ probe_C$C)"; done
} > "$CSV" && echo "  $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
