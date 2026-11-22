#!/bin/bash
# ct4_parse.sh <cmp_toolchain_4n / ct4_fftw_diag log>
#   1. ct_parse.sh (pl_parse table + every toolchain relative to gcc per library and layout), with the grid's
#      reference energy
#   2. best configuration per (library, grid) and toolchain: lowest median step time among the layouts whose runs
#      all have the reference energy; ratio to gcc's best and "ranges overlap" = the min-max ranges of the two
#      best configurations overlap (then the difference is within the run-to-run scatter)
#   3. UCX version really loaded per configuration (UCX: lines) and the UCX warnings
#   4. if the log has FFTW plans (NS_PRINT_PLAN=1): one line per run, step time next to the MPI transposes of the
#      forward and backward plans
#   5. runs with a non-zero exit code
L=${1:?usage: ct4_parse.sh <log>}
D=$(cd "$(dirname "$0")" && pwd)
EREF=$(grep -m1 -oE "energy_ref=[^ ]+" "$L" | cut -d= -f2)
[ -n "$EREF" ] || { echo "no 'energy_ref=' header line in $L"; exit 1; }
grep -m1 "^grid=" "$L"; grep -m1 "^IMN:" "$L"; grep -m1 "^GCC env" "$L"
grep -m1 "^intel_mn_env:" "$L"
echo
T=$(PL_EREF=$EREF bash $D/ct_parse.sh "$L")
echo "$T"
echo
echo "--- best configuration per (library, grid) and toolchain (only layouts with energy ok in every run)"
PL_EREF=$EREF bash $D/../../planner/pl_parse.sh "$L" | awk '
  NR == 1 || NF < 9 || $4 !~ /^[0-9.]+$/ { next }
  { n = split($1, p, "_"); if (n != 3) next
    lg = p[1]; tc = p[2]; lay = p[3]; runs = $3; ok = $(NF-2)
    rg = $5; gsub(/[()]/, "", rg); split(rg, mm, "-")
    if (!(lg in seen)) { seen[lg] = 1; ord[++nk] = lg }
    if (!((lg, tc) in tcs)) { tcs[lg, tc] = 1; tl[lg] = tl[lg] " " tc }
    if (ok + 0 < runs + 0) { bad[lg, tc] = bad[lg, tc] " " lay "(" ok "/" runs ")"; next }
    if (!((lg, tc) in best) || $4 < best[lg, tc]) { best[lg, tc] = $4; blay[lg, tc] = lay; blo[lg, tc] = mm[1]; bhi[lg, tc] = mm[2] } }
  END {
    printf "%-14s %-10s %-8s %10s %-19s %9s  %-14s %s\n", "library-grid", "toolchain", "layout", "s/step", "range (min-max)", "vs gcc", "ranges overlap", "energy failures"
    for (i = 1; i <= nk; i++) { k = ord[i]; m = split(tl[k], t, " ")
      # gcc first
      for (j = 1; j <= m; j++) if (t[j] == "gcc") { t[j] = t[1]; t[1] = "gcc" }
      for (j = 1; j <= m; j++) { c = t[j]; f = ((k, c) in bad) ? bad[k, c] : ""
        if (!((k, c) in best)) { printf "%-14s %-10s %-8s %10s %-19s %9s  %-14s %s\n", k, c, "-", "-", "-", "-", "-", f; continue }
        r = "-"; ov = "-"
        if (c != "gcc" && ((k, "gcc") in best)) {
          r = sprintf("%.3f", best[k, c] / best[k, "gcc"])
          ov = (blo[k, c] + 0 <= bhi[k, "gcc"] + 0 && blo[k, "gcc"] + 0 <= bhi[k, c] + 0) ? "yes" : "no" }
        printf "%-14s %-10s %-8s %10.4f %-19s %9s  %-14s %s\n", k, c, blay[k, c], best[k, c], blo[k, c] "-" bhi[k, c], r, ov, f }
      print "" } }'
echo "--- UCX version really loaded by rank 0 (per configuration: version x runs)"
awk '/^UCX: / { split($4, v, "="); c[$2 " " v[2]]++ } END { for (k in c) { split(k, a, " "); printf "  %-28s %-10s x%d\n", a[1], a[2], c[k] } }' "$L" | sort
n=$(grep -c "^WARNING: UCX" "$L"); echo "  UCX warnings: $n"; grep "^WARNING: UCX" "$L" | sed 's/^/    /'
if grep -q "^FFTW PLAN:" "$L"; then
    echo
    echo "--- FFTW: step time and the MPI transposes of the plans, one line per run (rank 0)"
    awk '
      /^=============== repetition/ { rep = $NF }
      /^>>>/ { lab = $3; step = "-"; fw = ""; bw = ""; dir = "" }
      /^FFTW PLAN: ---- forward/  { dir = "f" }
      /^FFTW PLAN: ---- backward/ { dir = "b" }
      /^FFTW PLAN:/ && /mpi-/ { while (match($0, /mpi-transpose-[a-z0-9]+|mpi-rdft2-[a-z0-9-]+/)) {
                                   w = substr($0, RSTART, RLENGTH); $0 = substr($0, RSTART + RLENGTH)
                                   if (dir == "f" && index(" " fw " ", " " w " ") == 0) fw = fw " " w
                                   if (dir == "b" && index(" " bw " ", " " w " ") == 0) bw = bw " " w } }
      /Avg per step:/ { step = $(NF-1) }
      /^<<< exit=/ { printf "  %-26s rep=%-2s step=%-8s fwd:%s | bwd:%s\n", lab, rep, step, (fw == "" ? " -" : fw), (bw == "" ? " -" : bw) }' "$L"
fi
echo
echo "--- retries (start failures, mn_lib.sh): $(grep -c '^RETRY' "$L") ; runs stopped: $(grep -c '^FAILED-RUN' "$L")"
grep -E "^(RETRY|FAILED-RUN)" "$L" | sed 's/^/  /'
echo
echo "--- runs with a non-zero exit code"
awk '/^>>>/ { l = $3 } /^<<< exit=/ { split($0, a, "="); if (a[2] != "0") print "  " l " exit=" a[2] }' "$L" | sort | uniq -c
