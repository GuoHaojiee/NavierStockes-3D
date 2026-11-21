#!/bin/bash
# p3d_parse.sh <p3d_timing log> — planner/pl_parse.sh table (median (min-max) of step time over the repetitions,
#   FFT time, planning time, energy ok / failed; reference 1.7933833444e+08), then per decomposition and layout
#   every program relative to old-gcc and new relative to old of the same toolchain (ratio of median step time,
#   < 1 = faster), then the library line, P3DFFT's thread count and the decomposition each run printed.
L=${1:?usage: p3d_parse.sh <log>}
D=$(cd "$(dirname "$0")" && pwd)
T=$(bash $D/../../planner/pl_parse.sh "$L")
echo "$T"
echo
echo "$T" | awk '
  NR == 1 || NF < 9 || $4 !~ /^[0-9.]+$/ { next }
  { n = split($1, p, "_"); if (n != 3) next          # P3D-<decomp> _ <old|new>-<gcc|intel> _ <layout>
    key = p[1] "_" p[3]; split(p[2], b, "-"); age = b[1]; tc = b[2]
    med[$1] = $4; ok[$1] = $(NF-2); grp[$1] = key; t[key, age, tc] = $4
    if (!(key in seen)) { seen[key] = 1; ord[++nk] = key }
    rows[key] = rows[key] " " $1; prog[$1] = p[2] }
  END {
    printf "%-26s %10s %12s %12s  %s\n", "config", "step s", "vs old-gcc", "new / old", "energy ok"
    for (i = 1; i <= nk; i++) {
      k = ord[i]; m = split(rows[k], r, " ")
      for (j = 1; j <= m; j++) {
        c = r[j]; split(prog[c], b, "-"); g = t[k, "old", "gcc"]; o = t[k, "old", b[2]]
        printf "%-26s %10.4f %12s %12s  %s\n", c, med[c], (g == "" ? "-" : sprintf("%.3f", med[c] / g)),
               (b[1] != "new" || o == "" ? "-" : sprintf("%.3f", med[c] / o)), ok[c]
      }
      print ""
    }
  }'
echo "--- library of each program (static libp3dfft.a, as recorded at build time; old builds: not recorded)"
awk '/^>>>/ { split($3, p, "_"); prog = p[2] } /^P3DFFT library:/ { if (!(prog in s)) { s[prog] = 1; print "  " prog ": " $0 } }' "$L"
echo "--- threads per rank reported by P3DFFT (\"Running on N threads\"), per program and layout"
awk '/^>>>/ { split($3, p, "_"); k = p[2] "_" p[3] } /Running on/ { c[k " " $3]++ } END { for (x in c) print "  " x " threads: " c[x] " runs" }' "$L" | sort
echo "--- decompositions printed (label prefix must match)"
awk '/^>>>/ { split($3, p, "_"); w = p[1] } /^decomposition:/ { print "  " w "  ->  " $2 }' "$L" | sort | uniq -c
