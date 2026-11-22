#!/bin/bash
# fftb_parse.sh [fftb.csv] — summary of the pure FFT benchmark (fftb.sbatch, fftb_p256.sbatch)
#   1. per (library, grid, nodes, toolchain, settings, layout): repetitions, median t_fft (min-max) per FFT,
#      median plan time, median local FFT time and the communication share (FFTB_SPLIT=1), check, UCX
#   2. Intel vs GCC per (library, grid, nodes): the best GCC configuration (lowest median over settings and layouts),
#      every Intel configuration against it (ratio, "ranges overlap" = the min-max ranges overlap, i.e. the
#      difference is within the scatter), and best Intel vs best GCC
#   3. strong scaling per (library, grid, toolchain): best configuration per node count, speedup and efficiency
#      relative to the smallest node count present
#   4. against the solver: rows of FFTB_SOLVER_CSV (default ~/guohaojie/results/toolchain_mn.csv; several files
#      allowed, space separated) with the same (library, grid, nodes, toolchain, layout, configuration name =
#      first word of settings): 36 x t_fft (the FFT time of one RK4 step in bench mode: 12 forward + 24 backward)
#      next to the solver's "FFT (incl. transpose/comm)" s/step and the ratio
#   5. launches / repetitions that are not OK (status, check), and multi-node Intel runs without UCX 1.17.0
# Only rows with status OK and check OK count in 1-4. Filters: FFTB_COMMIT=<hash>, FFTB_JOBS="<jobid> ...".
CSV=${1:-$HOME/guohaojie/results/fftb.csv}
[ -f "$CSV" ] || { echo "no $CSV"; exit 1; }
SOLVER=${FFTB_SOLVER_CSV:-$HOME/guohaojie/results/toolchain_mn.csv}
SF=(); for f in $SOLVER; do [ -f "$f" ] && SF+=("$f"); done
awk -F, -v commit="${FFTB_COMMIT:-}" -v jobs=" ${FFTB_JOBS:-} " -v nsolver=${#SF[@]} '
function sortv(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } }
function median(a, n) { sortv(a, n); return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
function libord(l) { return (l == "FFTW") ? 1 : (l == "AccFFT") ? 2 : (l == "P3DFFT") ? 3 : (l == "heFFTe") ? 4 : 5 }
function first(s,   a) { split(s, a, " "); return a[1] }
FNR == 1 { fileno++; if (fileno > 1 && $0 ~ /^commit,/) { for (i = 1; i <= NF; i++) sc[fileno, $i] = i }; next }
# ---------------- solver CSV files (toolchain_mn.csv layout, header-driven)
fileno > 1 {
    st = (sc[fileno, "status"] ? $(sc[fileno, "status"]) : ($(sc[fileno, "exit_code"]) == "0" ? "OK" : "ERROR"))
    if (st != "OK" || $(sc[fileno, "energy_ok"]) != "yes" || $(sc[fileno, "fft_s"]) == "") next
    k = $(sc[fileno, "lib"]) SUBSEP $(sc[fileno, "grid"]) SUBSEP $(sc[fileno, "nodes"]) SUBSEP $(sc[fileno, "toolchain"]) SUBSEP \
        $(sc[fileno, "ranks_per_node"]) "x" $(sc[fileno, "threads"]) SUBSEP first($(sc[fileno, "settings"]))
    sn[k]++; sv[k, sn[k]] = $(sc[fileno, "fft_s"]) + 0
    next
}
# ---------------- fftb.csv
commit != "" && $1 != commit { next }
jobs != "  " && index(jobs, " " $2 " ") == 0 { next }
{
    tc = $4; set = $5; lib = $6; g = $7; nn = $8; lay = $9 "x" $10; st = $17; chk = $15
    if (st != "OK" || chk != "OK" || $11 == "0" || $12 == "")
        prob[++np] = sprintf("  %-7s %5s n=%s %-5s %-6s rep=%-2s job=%s status=%s check=%s exit=%s retries=%s  %s", lib, g, nn, tc, lay, $11, $2, st, chk, $25, $24, set)
    else if (tc == "intel" && nn + 0 > 1 && $16 != "1.17.0")
        prob[++np] = sprintf("  %-7s %5s n=%s %-5s %-6s rep=%-2s job=%s UCX %s (%s)  %s", lib, g, nn, tc, lay, $11, $2, $16, $26, set)
    if (st != "OK" || chk != "OK" || $12 == "") next
    k = lib SUBSEP g SUBSEP nn SUBSEP tc SUBSEP set SUBSEP lay
    if (!(k in n)) keys[++nk] = k
    n[k]++; t[k, n[k]] = $12 + 0; pl[k, n[k]] = $13 + 0
    if ($14 != "") { nl[k]++; tl[k, nl[k]] = $14 + 0 }
    if (index(" " uv[k] " ", " " $16 " ") == 0) uv[k] = uv[k] " " $16
    libs[lib] = 1; grids[g] = 1; nodes[nn] = 1
}
END {
    # ---- 1. per configuration
    print "--- 1. per configuration (rows with status OK and check OK; t_fft per FFT, solver mix (fwd + 2 bwd)/3)"
    printf "%-7s %5s %5s %-5s %-6s %4s  %-34s %9s %11s %6s  %-8s %s\n", "lib", "grid", "nodes", "tc", "layout", "reps", "t_fft s: median (min-max)", "plan s", "t_local s", "comm%", "ucx", "settings"
    for (o = 1; o <= 5; o++) for (i = 1; i <= nk; i++) {
        split(keys[i], p, SUBSEP); if (libord(p[1]) != o) continue
        k = keys[i]; m = n[k]
        for (j = 1; j <= m; j++) { a[j] = t[k, j]; b[j] = pl[k, j] }
        sortv(a, m); lo[k] = a[1]; hi[k] = a[m]; med[k] = median(a, m); mp = median(b, m)
        loc = "-"; cm = "-"
        if (nl[k]) { for (j = 1; j <= nl[k]; j++) c[j] = tl[k, j]; ml = median(c, nl[k]); loc = sprintf("%.4e", ml); cm = sprintf("%.0f", 100 * (med[k] - ml) / med[k]) }
        printf "%-7s %5s %5s %-5s %-6s %4d  %.4e (%.4e-%.4e) %9.3f %11s %6s %-8s %s\n", p[1], p[2], p[3], p[4], p[6], m, med[k], lo[k], hi[k], mp, loc, cm, uv[k], p[5]
        bk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]
        if (!(bk in best) || med[k] < best[bk]) { best[bk] = med[k]; bkey[bk] = k }
    }
    # ---- 2. Intel vs GCC
    print ""; print "--- 2. Intel vs GCC per (library, grid, nodes): every Intel configuration against the best GCC one"
    printf "%-7s %5s %5s  %-44s %-6s %11s %11s %8s  %s\n", "lib", "grid", "nodes", "intel settings", "layout", "intel s", "best gcc s", "int/gcc", "ranges overlap"
    for (o = 1; o <= 5; o++) for (gi = 1; gi <= nk; gi++) {
        split(keys[gi], p, SUBSEP); if (libord(p[1]) != o || p[4] != "intel") continue
        gk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP "gcc"
        if (!(gk in best)) { printf "%-7s %5s %5s  %-44s %-6s %11.4e %11s %8s  %s\n", p[1], p[2], p[3], substr(p[5], 1, 44), p[6], med[keys[gi]], "-", "-", "-"; continue }
        k = keys[gi]; q = bkey[gk]
        ov = (lo[k] <= hi[q] && lo[q] <= hi[k]) ? "yes" : "no"
        mark = (bkey[p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP "intel"] == k) ? "  <- best intel" : ""
        printf "%-7s %5s %5s  %-44s %-6s %11.4e %11.4e %8.3f  %s%s\n", p[1], p[2], p[3], substr(p[5], 1, 44), p[6], med[k], best[gk], med[k] / best[gk], ov, mark
    }
    for (bk in best) { split(bk, p, SUBSEP); if (p[4] == "gcc") { split(bkey[bk], qq, SUBSEP); printf "  best gcc %-7s %5s n=%s: %s %s  %.4e\n", p[1], p[2], p[3], qq[6], qq[5], best[bk] } }
    # ---- 3. strong scaling
    print ""; print "--- 3. strong scaling: best configuration per node count (t_fft s, speedup S and efficiency E vs the smallest node count)"
    printf "%-7s %5s %-5s", "lib", "grid", "tc"; for (nn = 1; nn <= 4; nn *= 2) printf " | %24s", nn " node" (nn > 1 ? "s" : ""); printf "\n"
    for (o = 1; o <= 5; o++) for (l in libs) { if (libord(l) != o) continue
        for (g in grids) for (ti = 1; ti <= 2; ti++) { tc = (ti == 1) ? "gcc" : "intel"
            n0 = 0; for (nn = 1; nn <= 4; nn *= 2) if ((l SUBSEP g SUBSEP nn SUBSEP tc) in best) { n0 = nn; break }
            if (!n0) continue
            t0 = best[l SUBSEP g SUBSEP n0 SUBSEP tc]
            printf "%-7s %5s %-5s", l, g, tc
            for (nn = 1; nn <= 4; nn *= 2) { bk = l SUBSEP g SUBSEP nn SUBSEP tc
                if (bk in best) printf " | %.4e %5.2f %5.0f%%", best[bk], t0 / best[bk], 100 * t0 / best[bk] / (nn / n0)
                else printf " | %24s", "-" }
            printf "\n" } }
    print "  (S = T(n0) / T(n), E = S / (n / n0), n0 = smallest node count with data)"
    # ---- 4. against the solver
    print ""; print "--- 4. against the solver: 36 x t_fft vs the solver FFT s/step, same (lib, grid, nodes, toolchain, layout, configuration name)"
    if (!nsolver) print "  (no solver CSV found)"
    else {
        printf "%-7s %5s %5s %-5s %-6s %-14s %12s %12s %7s %5s\n", "lib", "grid", "nodes", "tc", "layout", "config", "36*t_fft s", "solver FFT s", "ratio", "runs"
        nm = 0
        for (o = 1; o <= 5; o++) for (i = 1; i <= nk; i++) {
            split(keys[i], p, SUBSEP); if (libord(p[1]) != o) continue
            k = keys[i]; sk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4] SUBSEP p[6] SUBSEP first(p[5])
            if (!(sk in sn)) continue
            m = sn[sk]; for (j = 1; j <= m; j++) a[j] = sv[sk, j]; ms = median(a, m); nm++
            printf "%-7s %5s %5s %-5s %-6s %-14s %12.4f %12.4f %7.3f %5d\n", p[1], p[2], p[3], p[4], p[6], first(p[5]), 36 * med[k], ms, 36 * med[k] / ms, m
        }
        if (!nm) print "  (no matching solver rows)"
    }
    # ---- 5. problems
    print ""; print "--- 5. not OK (" np + 0 ")"
    for (i = 1; i <= np; i++) print prob[i]
}' "$CSV" "${SF[@]}"
