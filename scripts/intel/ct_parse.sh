#!/bin/bash
# ct_parse.sh <log> — cmp_toolchain results: the planner-study table (planner/pl_parse.sh: median (min-max) of step
#   time, FFT time, planning time, energy ok / failed), then each toolchain relative to gcc for the same library
#   and layout (ratio of the median step time; < 1 = faster than gcc).
# Reference kinetic energy 512^3, 20 steps: 1.7933833444e+08 (PL_EREF=<value> for another grid).
L=${1:?usage: ct_parse.sh <log>}
D=$(cd "$(dirname "$0")" && pwd)
T=$(bash $D/../../planner/pl_parse.sh "$L")
echo "$T"
echo
echo "$T" | awk '
  NR == 1 || NF < 9 || $4 !~ /^[0-9.]+$/ { next }
  { n = split($1, p, "_"); if (n < 3) next
    lib = p[1]; tc = p[2]; lay = p[n]; key = lib "_" lay
    med[$1] = $4; if (tc == "gcc") ref[key] = $4
    if (!(key in seen)) { seen[key] = 1; ord[++nk] = key }
    rows[key] = rows[key] " " $1; ok[$1] = $(NF-2) }
  END {
    printf "%-26s %10s %10s %8s  %s\n", "config", "step s", "gcc s", "vs gcc", "energy ok"
    for (i = 1; i <= nk; i++) {
      k = ord[i]; m = split(rows[k], r, " ")
      for (j = 1; j <= m; j++) {
        c = r[j]; g = ref[k]
        printf "%-26s %10.4f %10s %8s  %s\n", c, med[c], (g == "" ? "-" : sprintf("%.4f", g)), (g == "" ? "-" : sprintf("%.3f", med[c] / g)), ok[c]
      }
    }
  }'
# all runs must have printed the slab decomposition
echo
grep -h "^decomposition:" "$L" | sed 's/ NS_DECOMP=.*//' | sort | uniq -c
