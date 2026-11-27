#!/bin/bash
# dtcfl_parse.sh — results of dtcfl.sbatch.
#   bash dtcfl_parse.sh <cfl jobid> <fixed jobid> <ref jobid> [probe jobid]
# Files: ${DTC_RESULTS:-/scratch/pr0208/guohaojie/results/dtcfl}/<jobid>/<label>.txt (labels cfl_C<C>, fixed_C<C>, ref, probe_C<C>).
# Output:
#   1. every run: steps (total, to t=10), wall time, s/step, dt_min / dt_max, steps limited by conv / visc / tend, blowup
#   2. per C: step ratio fixed/cfl to t=10 and t=20 (main metric, deterministic), wall-time ratio to t=20 (supporting)
#   3. per C: E and eps = nu<|w|^2> at t=10 and t=20, relative deviation from the reference run (a)
#   4. instability probe: E and eps at t=12 (FINAL) vs the reference interpolated at t=12; status
#        BLOWUP     (t, last scanned steps; at C < 0.9 marked as possibly caused by aliasing, not by the CFL limit)
#        throttled  ran to the end, but somewhere the adaptive dt was held far below the C=1 run:
#                   at the time t* of the run's smallest dt (cut last step excluded),
#                   dt_min / (C * dt_{C=1}(t*)) < 0.8, dt_{C=1} = dt of probe C=1.0 interpolated at t*
#                   (= max sum|u_i|/dx_i of the C=1 run over that of this run). The steps around t* are listed.
#        stable     otherwise
#      (the same ratio is shown for the main cfl runs where t* lies inside the C=1 run)
#   5. CSV of the cfl and probe series: run,C,step,t,dt,umax,E,eps,Ceff,lim -> ${DTC_CSV:-<results>/dtcfl_series_<cfl jobid>.csv}
#   6. CSVs for native pptx charts (${DTC_PPTX:-<results>/pptx_<cfl jobid>}/):
#        umax_dt_C<C>.csv  t on a uniform grid (step 0.1, 0..20, linear interpolation of the per-step series):
#                          t,umax_cfl,umax_fixed,dt_cfl,dt_fixed   (C = DTC_PLOT_C, default 0.9)
#        summary.csv       C,steps_cfl,steps_fixed,speedup,E20_dev_cfl_pct,eps20_dev_cfl_pct,E20_dev_fixed_pct,eps20_dev_fixed_pct
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
    awk -v ta=$TA -v tc=12 '
    function lag(x, xs, ys, i0,   j, m, l, s) {   # 4-point Lagrange through points i0..i0+3
        s = 0
        for (j = 0; j < 4; j++) { l = 1; for (m = 0; m < 4; m++) if (m != j) l *= (x - xs[i0+m]) / (xs[i0+j] - xs[i0+m]); s += l * ys[i0+j] }
        return s
    }
    function at(x,   k, i0) {                       # 4-point Lagrange of E, W at x -> IE, IW (NA outside)
        IE = "NA"; IW = "NA"
        if (n < 4 || x < t[1] || x > t[n]) return
        for (k = 1; k < n && t[k+1] <= x; k++) ;
        i0 = k - 1; if (i0 < 1) i0 = 1; if (i0 + 3 > n) i0 = n - 3
        IE = sprintf("%.12e", lag(x, t, E, i0)); IW = sprintf("%.12e", lag(x, t, W, i0))
    }
    $1 == "DT" && NF >= 9 { n++; t[n] = $3; E[n] = $6; W[n] = $7; if ($3 < ta - 1e-9) s10++; if ($8 > cmax) cmax = $8
                            if ($9 != "tend" && (dmin == "" || $4 + 0 < dmin)) { dmin = $4 + 0; tdmin = $3 } }
    $1 == "FINAL" { for (i = 2; i <= NF; i++) { split($i, a, "="); f[a[1]] = a[2] } }
    $1 == "DTCFL_SUMMARY" { for (i = 2; i <= NF; i++) { split($i, a, "="); s[a[1]] = a[2] } }
    $1 == "BLOWUP" { for (i = 2; i <= NF; i++) { split($i, a, "="); b[a[1]] = a[2] } }
    /Avg per step:/ { sps = $(NF-1) }
    END {
        e10 = "NA"; w10 = "NA"; e12 = "NA"; w12 = "NA"
        if (t[n] > ta) { at(ta); e10 = IE; w10 = IW }
        at(tc); e12 = IE; w12 = IW
        printf "E12=%s W12=%s tdmin=%s dmin=%s ", e12, w12, (tdmin != "" ? tdmin : "NA"), (dmin != "" ? dmin : "NA")
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

# dt of the C=1 run at time x (linear interpolation of its per-step series); empty when unavailable
P1=$([ -n "$PJ" ] && [ -f "$(f_of $PJ probe_C1.0)" ] && f_of $PJ probe_C1.0)
dt1_at() {
    [ -n "$P1" ] || return
    awk -v x="$1" '$1 == "DT" && NF >= 9 && $9 != "tend" { n++; t[n] = $3; d[n] = $4 }
        END { if (n < 2 || x < t[1] || x > t[n]) exit
              for (k = 1; k < n && t[k+1] < x; k++) ; f = (x - t[k]) / (t[k+1] - t[k]); printf "%.10e", d[k] + f * (d[k+1] - d[k]) }' "$P1"
}
# throttle <stats> <C>: ratio dt_min / (C * dt_{C=1}(t*)) or NA
throttle() {
    local d1; d1=$(dt1_at $(get "$1" tdmin))
    [ -n "$d1" ] && awk -v a=$(get "$1" dmin) -v c=$2 -v b=$d1 'BEGIN { printf "%.3f", a / (c * b) }' || echo NA
}
around() {   # around <file> <t*>: the steps around the smallest dt
    awk -v x="$2" '$1 == "DT" && NF >= 9 { n++; l[n] = sprintf("    %6d %9.4f %11.4e %10.4e %12.6e %12.6e %7.4f %s", $2, $3, $4, $5, $6, $7, $8, $9); t[n] = $3 }
        END { for (i = 1; i <= n; i++) if (t[i] == x) m = i
              print "      step         t          dt       umax            E          eps    Ceff lim"
              for (i = m - 6; i <= m + 6; i++) if (i >= 1 && i <= n) print l[i] (i == m ? "   <- dt_min" : "") }' "$1"
}
if [ -n "$PS" ]; then
    echo "-- 4. instability probe (T_END=12)"
    [ -n "$P1" ] || echo "  (no probe_C1.0 run: the throttled criterion is not available)"
    printf "  %5s %7s %12s %9s %12s %9s %10s %10s  %s\n" C steps dt_min "t(dtmin)" "C*dt1(t*)" ratio "E(12) dev" "eps(12)dev" status
    for C in $PS; do
        k=$(echo $C | tr . _); eval "p=\$S_probe_C$k"
        th=$(throttle "$p" $C)
        d1=$(dt1_at $(get "$p" tdmin)); c1=$([ -n "$d1" ] && awk -v c=$C -v b=$d1 'BEGIN { printf "%.4e", c * b }' || echo NA)
        if   [ "$(get "$p" blowup)" = 1 ]; then status="BLOWUP t=$(get "$p" bt)"
        elif [ "$(get "$p" cap)" = 1 ];    then status="STEP CAP t=$(get "$p" tlast)"
        elif [ "$(get "$p" steps)" = NA ]; then status="did not finish"
        elif awk -v r=$th 'BEGIN { exit !(r != "NA" && r < 0.8) }'; then status=throttled
        else status=stable; fi
        dE=$(rel $(get "$p" E20) $(get "$S_ref" E12)); dW=$(rel $(get "$p" W20) $(get "$S_ref" W12))
        case "$status" in stable|throttled) ;; *) dE=-; dW=- ;; esac     # FINAL is not at t=12
        printf "  %5s %7s %12s %9s %12s %9s %10s %10s  %s\n" $C $(get "$p" steps) $(get "$p" dmin) $(get "$p" tdmin) $c1 $th \
               $dE $dW "$status"
        eval "ST_$k=\"\$status\""
    done
    echo "  (dev = run/ref - 1 at t=12: FINAL line of the probe vs the reference interpolated at t=12)"
    for C in $PS; do
        k=$(echo $C | tr . _); eval "p=\$S_probe_C$k st=\$ST_$k"
        case "$st" in
        BLOWUP*)
            note=""; awk -v c=$C 'BEGIN { exit !(c < 0.9) }' && note="  [C < 0.9: possibly aliasing (no dealiasing), not the CFL limit]"
            echo "  C=$C: $st$note; last scanned steps (step t dt umax E eps Ceff):"
            grep "^BLOWUP_HIST [0-9]" "$(f_of $PJ probe_C$C)" | sed 's/^BLOWUP_HIST/    /' ;;
        throttled|stable)
            echo "  C=$C ($st): steps around the smallest dt (t=$(get "$p" tdmin))"
            around "$(f_of $PJ probe_C$C)" $(get "$p" tdmin) ;;
        esac
    done
fi
# main cfl runs: the same ratio, and a BLOWUP with C < 0.9
for C in $CS; do
    k=$(echo $C | tr . _); eval "x=\$S_cfl_C$k"
    th=$(throttle "$x" $C)
    [ "$th" != NA ] && echo "  cfl_C$C: dt_min $(get "$x" dmin) at t=$(get "$x" tdmin), dt_min / (C dt_{C=1}) = $th"
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

PP=${DTC_PPTX:-$RES/pptx_$CJ}; PC=${DTC_PLOT_C:-0.9}; mkdir -p "$PP"
echo "-- 6. CSVs for pptx charts ($PP)"
fc=$(f_of $CJ cfl_C$PC); ff=$(f_of $FJ fixed_C$PC)
if [ -f "$fc" ] && [ -f "$ff" ]; then
    awk -v h=0.1 -v T=$TB 'function val(r, x, c,   k, f) {       # linear interpolation of column c of run r at x
            if (x <= t[r,1]) return v[r,1,c]
            if (x >= t[r,n[r]]) return v[r,n[r],c]
            for (k = p[r]; k < n[r] && t[r,k+1] < x; k++) ; p[r] = k
            f = (x - t[r,k]) / (t[r,k+1] - t[r,k]); return v[r,k,c] + f * (v[r,k+1,c] - v[r,k,c]) }
        FNR == 1 { r++; p[r] = 1 }
        $1 == "DT" && NF >= 9 { m = ++n[r]; t[r,m] = $3; v[r,m,1] = $5; v[r,m,2] = $4 }    # 1 = umax, 2 = dt
        END { print "t,umax_cfl,umax_fixed,dt_cfl,dt_fixed"
              for (i = 0; i * h <= T + 1e-9; i++) { x = i * h
                  printf "%.1f,%.6e,%.6e,%.6e,%.6e\n", x, val(1, x, 1), val(2, x, 1), val(1, x, 2), val(2, x, 2) } }' "$fc" "$ff" > "$PP/umax_dt_C$PC.csv"
    echo "  $PP/umax_dt_C$PC.csv ($(($(wc -l < "$PP/umax_dt_C$PC.csv") - 1)) rows; cfl and fixed C=$PC, values at the start of the step, linear interpolation to t = 0, 0.1, ..., $TB)"
else
    echo "  missing $fc or $ff: no umax_dt_C$PC.csv"
fi
pct() { awk -v a="$1" -v b="$2" 'BEGIN { if (a == "NA" || b == "NA" || a == "" || b == "" || b + 0 == 0) print ""; else printf "%.4f", 100 * (a / b - 1) }'; }
{
    echo "C,steps_cfl,steps_fixed,speedup,E20_dev_cfl_pct,eps20_dev_cfl_pct,E20_dev_fixed_pct,eps20_dev_fixed_pct"
    for C in $CS; do
        k=$(echo $C | tr . _); eval "c=\$S_cfl_C$k f=\$S_fixed_C$k"
        echo "$C,$(get "$c" steps),$(get "$f" steps),$(ratio $(get "$f" steps) $(get "$c" steps)),$(pct $(get "$c" E20) $(get "$S_ref" E20)),$(pct $(get "$c" W20) $(get "$S_ref" W20)),$(pct $(get "$f" E20) $(get "$S_ref" E20)),$(pct $(get "$f" W20) $(get "$S_ref" W20))"
    done
} > "$PP/summary.csv"
echo "  $PP/summary.csv:"; sed 's/^/    /' "$PP/summary.csv"
