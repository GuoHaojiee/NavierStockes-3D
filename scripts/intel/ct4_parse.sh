#!/bin/bash
# ct4_parse.sh <cmp_toolchain_4n log> — ct_parse.sh (pl_parse table + intel/gcc per library and layout) with the
#   reference energy of the log's grid, then per (library, grid) the best configuration (lowest median step time
#   among the layouts whose runs all have the reference energy) of each toolchain and the ratio intel / gcc.
L=${1:?usage: ct4_parse.sh <log>}
D=$(cd "$(dirname "$0")" && pwd)
EREF=$(grep -m1 -oE "energy_ref=[^ ]+" "$L" | cut -d= -f2)
[ -n "$EREF" ] || { echo "no 'energy_ref=' header line in $L"; exit 1; }
grep -m1 "^grid=" "$L"; grep -m1 "^IMN:" "$L"
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
    if (!(lg in seen)) { seen[lg] = 1; ord[++nk] = lg }
    if (ok + 0 < runs + 0) { bad[lg, tc] = bad[lg, tc] " " lay "(" ok "/" runs ")"; next }
    if (!((lg, tc) in best) || $4 < best[lg, tc]) { best[lg, tc] = $4; blay[lg, tc] = lay } }
  END {
    printf "%-14s %-22s %-22s %10s  %s\n", "library-grid", "gcc best (s/step)", "intel best (s/step)", "intel/gcc", "energy failures"
    for (i = 1; i <= nk; i++) { k = ord[i]
      g = ((k, "gcc") in best) ? sprintf("%.4f %s", best[k, "gcc"], blay[k, "gcc"]) : "-"
      t = ((k, "intel") in best) ? sprintf("%.4f %s", best[k, "intel"], blay[k, "intel"]) : "-"
      r = (((k, "gcc") in best) && ((k, "intel") in best)) ? sprintf("%.3f", best[k, "intel"] / best[k, "gcc"]) : "-"
      f = ""; if ((k, "gcc") in bad) f = f "gcc:" bad[k, "gcc"] " "; if ((k, "intel") in bad) f = f "intel:" bad[k, "intel"]
      printf "%-14s %-22s %-22s %10s  %s\n", k, g, t, r, f } }'
echo
echo "--- runs with a non-zero exit code"
awk '/^>>>/ { l = $3 } /^<<< exit=/ { split($0, a, "="); if (a[2] != "0") print "  " l " exit=" a[2] }' "$L" | sort | uniq -c
