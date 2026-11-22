#!/bin/bash
# toolchain_mn_parse.sh [csv] — summary of ~/guohaojie/results/toolchain_mn.csv (cmp_toolchain_mn.sbatch); works the
# same for p256_diag.csv (there "settings" starts with the configuration name).
#   1. per (library, grid, nodes, toolchain, layout, settings): runs, median (min-max) step time, FFT, planning,
#      energy ok / runs, retries, UCX versions
#   2. per (library, grid, nodes): best configuration of each toolchain (lowest median among configurations whose
#      runs ALL have the reference energy and exit 0, over all layouts AND settings), intel/gcc and "ranges overlap"
#      (min-max ranges overlap); when a toolchain ran several settings (GCC_COLL="def t4", IMN_*), the best layout of
#      every setting is listed below it ("  setting:")
#   3. per toolchain and grid: strong scaling, library x nodes 1/2/4, best s/step, speedup S and efficiency E
#      relative to that toolchain's best 1-node configuration
#   4. runs killed at the time limit (TIMEOUT), runs whose start failed also when repeated (FAILED-RUN), and the other
#      runs with a wrong energy, a non-zero exit code, a retry, or (Intel, >1 node) a UCX version other than 1.17.0
#   (status column; rows from before it: exit 124 = TIMEOUT)
# Filters: CTM_COMMIT=<hash>, CTM_JOBS="<jobid> ...".
CSV=${1:-$HOME/guohaojie/results/toolchain_mn.csv}
[ -f "$CSV" ] || { echo "no $CSV"; exit 1; }
awk -F, -v commit="${CTM_COMMIT:-}" -v jobs=" ${CTM_JOBS:-} " '
function sortv(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } }
function median(a, n) { sortv(a, n); return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
function lo(l) { return (l == "FFTW") ? 1 : (l == "AccFFT") ? 2 : (l == "P3DFFT") ? 3 : (l == "heFFTe") ? 4 : 5 }
NR == 1 { next }
commit != "" && $1 != commit { next }
jobs != "  " && index(jobs, " " $2 " ") == 0 { next }
{
    tc = $4; set = $5; lib = $6; g = $7; nn = $8; lay = $9 "x" $10
    k = lib SUBSEP g SUBSEP nn SUBSEP tc SUBSEP lay SUBSEP set
    if (!(k in runs)) keys[++nk] = k
    runs[k]++
    if ($12 != "") { ns[k]++; st[k, ns[k]] = $12 + 0; ff[k, ns[k]] = $13 + 0; pl[k, ns[k]] = $14 + 0 }
    if ($16 == "yes") eok[k]++
    if ($19 != "0") bad[k]++
    ret[k] += $18
    if (index(" " uv[k] " ", " " $17 " ") == 0) uv[k] = uv[k] " " $17
    libs[lib] = 1; grids[g] = 1; tcs[tc] = 1
    stt = $21; if (stt == "") stt = ($19 == "0") ? "OK" : ($19 == "124") ? "TIMEOUT" : "ERROR"
    if (stt == "TIMEOUT") nto[k]++
    tag = sprintf("  %-7s %5s n=%s %-5s %-6s rep=%s job=%s [%s]", lib, g, nn, tc, lay, $11, $2, set)
    if (stt == "TIMEOUT")    { tol[++nt] = tag " exit=" $19 (($18 + 0 > 0) ? " retries=" $18 : "") }
    if (stt == "FAILED-RUN") { frl[++nf] = tag " exit=" $19 " retries=" $18 }
    w = ""
    if (stt == "TIMEOUT" || stt == "FAILED-RUN") { libs[lib] = 1; grids[g] = 1; tcs[tc] = 1; next }
    if ($16 != "yes") w = w " energy=" ($15 == "" ? "missing" : $15)
    if ($19 != "0") w = w " exit=" $19
    if ($18 + 0 > 0) w = w " retries=" $18
    if (tc == "intel" && nn + 0 > 1 && $17 != "1.17.0") w = w " ucx=" $17 ($20 != "" ? "(" $20 ")" : "")
    if (w != "") prob[++np] = sprintf("  %-7s %5s n=%s %-5s %-6s rep=%s job=%s [%s]%s", lib, g, nn, tc, lay, $11, $2, set, w)
}
END {
    # ---- 1
    printf "%-7s %5s %2s %-5s %-6s %-56s %4s  %-26s %8s %8s %7s %4s %3s %s\n", "lib", "grid", "n", "tc", "layout", "settings", "runs", "step s: median (min-max)", "FFT s", "plan s", "energy", "retr", "TO", "ucx"
    for (o = 1; o <= 5; o++) for (gg in grids) for (n = 1; n <= 4; n *= 2) for (i = 1; i <= nk; i++) {
        split(keys[i], p, SUBSEP); if (lo(p[1]) != o || p[2] != gg || p[3] != n) continue
        k = keys[i]; m = ns[k]
        if (m) { for (j = 1; j <= m; j++) { a[j] = st[k, j]; b[j] = ff[k, j]; c[j] = pl[k, j] }
                 sortv(a, m); mn[k] = a[1]; mx[k] = a[m]; md[k] = median(a, m); mf = median(b, m); mp = median(c, m)
                 s = sprintf("%.4f (%.4f-%.4f)", md[k], mn[k], mx[k]) }
        else { s = "-"; mf = mp = 0 }
        printf "%-7s %5s %2s %-5s %-6s %-56s %4d  %-26s %8.4f %8.3f %3d/%-3d %4d %3d%s\n", p[1], p[2], p[3], p[4], p[5], substr(p[6], 1, 56), runs[k], s, mf, mp, eok[k] + 0, runs[k], ret[k], nto[k], uv[k]
        if (m && eok[k] == runs[k] && !bad[k]) {
            sk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4] SUBSEP p[6]
            if (!(sk in sbest)) { nsets[p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]]++; sl[p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]] = sl[p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]] SUBSEP p[6] }
            if (!(sk in sbest) || md[k] < sbest[sk]) { sbest[sk] = md[k]; smin[sk] = mn[k]; smax[sk] = mx[k]; slay[sk] = p[5] }
            bk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]
            if (!(bk in best) || md[k] < best[bk]) { best[bk] = md[k]; bmin[bk] = mn[k]; bmax[bk] = mx[k]; bcfg[bk] = p[5] " " p[6] } } }
    # ---- 2
    print ""; print "--- best configuration per (library, grid, nodes) and toolchain (all runs: reference energy, exit 0)"
    printf "%-7s %5s %2s  %-5s %-62s %9s %-19s %9s  %s\n", "lib", "grid", "n", "tc", "layout settings", "s/step", "range", "vs gcc", "ranges overlap"
    for (o = 1; o <= 5; o++) for (gg in grids) for (n = 1; n <= 4; n *= 2) {
        for (l in libs) { if (lo(l) != o) continue
            gk = l SUBSEP gg SUBSEP n SUBSEP "gcc"; any = 0
            for (t in tcs) { bk = l SUBSEP gg SUBSEP n SUBSEP t; if (!(bk in best)) continue; any = 1
                r = "-"; ov = "-"
                if (t != "gcc" && (gk in best)) { r = sprintf("%.3f", best[bk] / best[gk]); ov = (bmin[bk] <= bmax[gk] && bmin[gk] <= bmax[bk]) ? "yes" : "no" }
                printf "%-7s %5s %2d  %-5s %-62s %9.4f %-19s %9s  %s\n", l, gg, n, t, substr(bcfg[bk], 1, 62), best[bk], sprintf("%.4f-%.4f", bmin[bk], bmax[bk]), r, ov
                if (nsets[bk] > 1) { ms = split(sl[bk], ss, SUBSEP)
                    for (q = 2; q <= ms; q++) { sk = bk SUBSEP ss[q]
                        r2 = (t != "gcc" && (gk in best)) ? sprintf("%.3f", sbest[sk] / best[gk]) : sprintf("%.3f", sbest[sk] / best[bk])
                        printf "%-7s %5s %2s  %-5s   setting: %-51s %9.4f %-19s %9s\n", "", "", "", "", substr(slay[sk] " " ss[q], 1, 51), sbest[sk], sprintf("%.4f-%.4f", smin[sk], smax[sk]), r2 } } }
            if (any) print "" } }
    # ---- 3
    for (t in tcs) for (gg in grids) {
        print "--- strong scaling, " t ", grid " gg " (best configuration per node count; S, E relative to the best 1-node one)"
        printf "%-7s", "lib"; for (n = 1; n <= 4; n *= 2) printf " | %8s %5s %5s", n " node", "S", "E"; printf "\n"
        for (o = 1; o <= 5; o++) for (l in libs) { if (lo(l) != o) continue
            t1k = l SUBSEP gg SUBSEP 1 SUBSEP t; t1 = (t1k in best) ? best[t1k] : 0
            printf "%-7s", l
            for (n = 1; n <= 4; n *= 2) { bk = l SUBSEP gg SUBSEP n SUBSEP t
                if (bk in best) { if (t1 > 0) printf " | %8.4f %5.2f %4.0f%%", best[bk], t1 / best[bk], 100 * t1 / best[bk] / n
                                  else printf " | %8.4f %5s %5s", best[bk], "-", "-" }
                else printf " | %8s %5s %5s", "-", "-", "-" }
            printf "\n" }
        print "" }
    # ---- 4
    print "--- TIMEOUT: killed at the time limit (" nt + 0 ")"
    for (i = 1; i <= nt; i++) print tol[i]
    print ""
    print "--- FAILED-RUN: start failure also in the repeated run (" nf + 0 ")"
    for (i = 1; i <= nf; i++) print frl[i]
    print ""
    print "--- other runs with a wrong energy, a non-zero exit, a retry or an unexpected UCX version (" np + 0 ")"
    for (i = 1; i <= np; i++) print prob[i]
}' "$CSV"
