#!/bin/bash
# cpu_final_parse.sh [csv] — summary of ~/guohaojie/results/cpu_final.csv (written by cpu_final.sbatch):
#   1. per (library, nodes, layout): runs, median (min-max) step time, median FFT and planning time, energy ok / runs
#   2. per (library, nodes): best layout = lowest median step time among the layouts whose runs ALL have the reference
#      energy and exit code 0
#   3. strong scaling: library x nodes (1, 2, 4), each cell the best layout's median step time, speedup and parallel
#      efficiency relative to the best 1-node configuration of the same library
#   4. runs with a wrong energy, a non-zero exit code, a retry (start failure) or a UCX version other than 1.17.0
#      on more than one node
# Filters (environment): CPF_COMMIT=<hash> (only rows of that commit), CPF_JOBS="<jobid> <jobid> ..."
CSV=${1:-$HOME/guohaojie/results/cpu_final.csv}
[ -f "$CSV" ] || { echo "no $CSV"; exit 1; }
awk -F, -v commit="${CPF_COMMIT:-}" -v jobs=" ${CPF_JOBS:-} " '
function sortv(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } }
function median(a, n) { sortv(a, n); return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
function libord(l) { return (l == "FFTW") ? 1 : (l == "AccFFT") ? 2 : (l == "P3DFFT") ? 3 : (l == "heFFTe") ? 4 : 5 }
NR == 1 { next }
commit != "" && $1 != commit { next }
jobs != "  " && index(jobs, " " $2 " ") == 0 { next }
{
    lib = $4; nn = $5; lay = $6 "x" $7; k = lib SUBSEP nn SUBSEP lay
    if (!(k in runs)) { keys[++nk] = k }
    runs[k]++
    if ($9 != "") { ns[k]++; st[k, ns[k]] = $9 + 0; ff[k, ns[k]] = $10 + 0; pl[k, ns[k]] = $11 + 0 }
    if ($13 == "yes") eok[k]++
    if ($16 != "0") bad[k]++
    if (index(" " uv[k] " ", " " $14 " ") == 0) uv[k] = uv[k] " " $14
    libs[lib] = 1; nodes[nn] = 1
    if ($13 != "yes" || $16 != "0" || $15 == "WARN" || $19 + 0 > 0)
        prob[++np] = sprintf("  %-7s nodes=%s %-6s rep=%s job=%s  energy=%s(%s) exit=%s ucx=%s%s%s%s", lib, nn, lay, $8, $2, $12, $13, $16, $14,
                             ($20 != "" ? "(" $20 ")" : ""), ($15 == "WARN" ? " UCX-WARN" : ""), ($19 + 0 > 0 ? " retries=" $19 : ""))
    nret += $19
}
END {
    # ---- 1. per configuration
    printf "%-7s %5s %-7s %4s  %-26s %10s %10s %9s  %s\n", "lib", "nodes", "layout", "runs", "step s: median (min-max)", "FFT s", "plan s", "energy", "ucx"
    # order: library, nodes, layout as found
    for (o = 1; o <= 5; o++) for (n = 1; n <= 4; n *= 2) for (i = 1; i <= nk; i++) {
        split(keys[i], p, SUBSEP); if (libord(p[1]) != o || p[2] != n) continue
        k = keys[i]; m = ns[k]
        if (m) {
            for (j = 1; j <= m; j++) { a[j] = st[k, j]; b[j] = ff[k, j]; c[j] = pl[k, j] }
            sortv(a, m); lo = a[1]; hi = a[m]; med[k] = median(a, m)
            mf = median(b, m); mp = median(c, m)
            s = sprintf("%.4f (%.4f-%.4f)", med[k], lo, hi)
        } else { s = "-"; mf = mp = 0 }
        printf "%-7s %5s %-7s %4d  %-26s %10.4f %10.3f %4d/%-4d %s\n", p[1], p[2], p[3], runs[k], s, mf, mp, eok[k] + 0, runs[k], uv[k]
        ok = (m && eok[k] == runs[k] && !bad[k])
        bk = p[1] SUBSEP p[2]
        if (ok && (!(bk in best) || med[k] < best[bk])) { best[bk] = med[k]; blay[bk] = p[3] }
    }
    # ---- 2. best layout
    print ""; print "--- best layout per (library, nodes) (only layouts with the reference energy and exit 0 in every run)"
    printf "%-7s %5s %-7s %10s\n", "lib", "nodes", "layout", "step s"
    for (o = 1; o <= 5; o++) for (l in libs) { if (libord(l) != o) continue
        for (n = 1; n <= 4; n *= 2) { bk = l SUBSEP n; if (bk in best) printf "%-7s %5d %-7s %10.4f\n", l, n, blay[bk], best[bk]
                                      else if (n in nodes) printf "%-7s %5d %-7s %10s\n", l, n, "-", "none valid" } }
    # ---- 3. strong scaling
    print ""; print "--- strong scaling (best layout per node count; speedup and efficiency relative to the best 1-node configuration)"
    printf "%-7s", "lib"; for (n = 1; n <= 4; n *= 2) printf " | %22s", n " node" (n > 1 ? "s" : ""); printf "\n"
    printf "%-7s", ""; for (n = 1; n <= 4; n *= 2) printf " | %8s %6s %6s", "s/step", "S", "E"; printf "\n"
    for (o = 1; o <= 5; o++) for (l in libs) { if (libord(l) != o) continue
        printf "%-7s", l; t1 = ((l SUBSEP 1) in best) ? best[l SUBSEP 1] : 0
        for (n = 1; n <= 4; n *= 2) { bk = l SUBSEP n
            if (bk in best) { if (t1 > 0) printf " | %8.4f %6.2f %5.0f%%", best[bk], t1 / best[bk], 100 * t1 / best[bk] / n
                              else printf " | %8.4f %6s %6s", best[bk], "-", "-" }
            else printf " | %8s %6s %6s", "-", "-", "-" }
        printf "\n" }
    print "  (S = T(1 node) / T(n nodes), E = S / n; layouts in the table above)"
    # ---- 4. problems
    print ""; print "--- runs with a wrong energy, a non-zero exit code, a retry or an unexpected UCX version (" np + 0 "; retries in total: " nret + 0 ")"
    for (i = 1; i <= np; i++) print prob[i]
}' "$CSV"
