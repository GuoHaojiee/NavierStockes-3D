#!/bin/bash
# fftb_parse.sh [fftb.csv] — summary of the pure FFT benchmark (fftb.sbatch, fftb_p256.sbatch)
#   1. per (library, grid, nodes, toolchain, settings, layout): repetitions, median t_fft (min-max) per FFT,
#      median plan time, median local FFT time and the communication share (FFTB_SPLIT=1), check, UCX
#   2. Intel vs GCC per (library, grid, nodes): the best GCC configuration (lowest median over settings and layouts),
#      every Intel configuration against it (ratio, "ranges overlap" = the min-max ranges overlap, i.e. the
#      difference is within the scatter), with 36 x t_fft (FFT s per RK4 step) of both
#   2b. table: best Intel vs best GCC per (library, grid, nodes): layout, configuration, 36 x t_fft median (min-max),
#      intel/gcc ratio, ranges overlap
#   1b. layouts per (library, grid, nodes, toolchain, configuration name): t_fft of every layout, the best one marked
#   1c. communication share per configuration (FFTB_SPLIT=1 runs): t_comm = t_fft - t_local, (t_fft - t_local)/t_fft
#   3. strong scaling per (library, grid, toolchain): best configuration (layout, settings) per node count,
#      speedup and efficiency relative to the smallest node count present; communication share of that configuration
#   3b. the same with one fixed layout for every node count (FFTB_FIXED_LAYOUT, default 128x1), for comparison
#   4. against the solver: rows of FFTB_SOLVER_CSV (default ~/guohaojie/results/toolchain_mn.csv; several files
#      allowed, space separated) with the same (library, grid, nodes, toolchain, layout, configuration name =
#      first word of settings): 36 x t_fft (the FFT time of one RK4 step in bench mode: 12 forward + 24 backward)
#      next to the solver's "FFT (incl. transpose/comm)" s/step and the ratio
#   5. launches / repetitions that are not OK (status, check), and multi-node Intel runs without UCX 1.17.0
#   6. variants (fftb_accfft_tune.sbatch): every Intel configuration of library FFTB_VAR_LIB (AccFFT) against the
#      reference FFTB_VAR_REF (GCC configuration name, GCC_COLL=t4) with the same grid, nodes and layout: launches,
#      transpose method, t_fft median (min-max), t_local, t_comm = t_fft - t_local, comm%, t_fft/ref, comm/ref comm,
#      ranges overlap, the median of each job (A / B rounds) separately
# AccFFT rows in 1 and 6 show the transpose method AccFFT used (tmethod from lib_settings, FFTB_ACCFFT_TPLAN):
#   m1 Isend/Irecv, m2 MPI_Alltoall(v), m3k<k>a|s k-way exchange async|sync; "m2x3" = 3 launches with m2.
# Exit code 127 (command not found) is shown as "missing binary" in 5.
# Only rows with status OK and check OK count in 1-4. Filters: FFTB_COMMIT=<hash>, FFTB_JOBS="<jobid> ...",
# FFTB_TC=gcc|intel. A scaling study (fftb_scale.sbatch, one job per node count): FFTB_JOBS="<j1> <j2> <j4>".
# Runs whose FFTB_BIND check said WARN (ranks sharing CPUs, threads sharing a core) are counted but marked "bindWARN"
# in 1 and listed in 5.
CSV=${1:-$HOME/guohaojie/results/fftb.csv}
[ -f "$CSV" ] || { echo "no $CSV"; exit 1; }
SOLVER=${FFTB_SOLVER_CSV:-$HOME/guohaojie/results/toolchain_mn.csv}
SF=(); for f in $SOLVER; do [ -f "$f" ] && SF+=("$f"); done
awk -F, -v commit="${FFTB_COMMIT:-}" -v jobs=" ${FFTB_JOBS:-} " -v tcf="${FFTB_TC:-}" -v fixl="${FFTB_FIXED_LAYOUT:-128x1}" -v nsolver=${#SF[@]} -v vlib="${FFTB_VAR_LIB:-AccFFT}" -v vref="${FFTB_VAR_REF:-GCC_COLL=t4}" '
function sortv(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } }
function median(a, n) { sortv(a, n); return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
function libord(l) { return (l == "FFTW") ? 1 : (l == "AccFFT") ? 2 : (l == "P3DFFT") ? 3 : (l == "heFFTe") ? 4 : 5 }
function first(s,   a) { split(s, a, " "); return a[1] }
function rpn(l,   a) { split(l, a, "x"); return a[1] + 0 }
function tmfmt(k,   a, m, i, r) { if (tml[k] == "") return ""; m = split(tml[k], a, "/"); r = ""
    for (i = 1; i <= m; i++) r = r (i > 1 ? "/" : "") a[i] "x" tmc[k, a[i]]; return r }
function jobmeds(k,   js, m, i, j, c, r) { m = split(jbl[k], js, " "); r = ""
    for (i = 1; i <= m; i++) { c = njb[k, js[i]]; for (j = 1; j <= c; j++) za[j] = tjb[k, js[i], j]; r = r (i > 1 ? " / " : "") sprintf("%.4e", median(za, c)) }
    return r }
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
tcf != "" && $4 != tcf { next }
{
    tc = $4; set = $5; lib = $6; g = $7; nn = $8; lay = $9 "x" $10; st = $17; chk = $15
    if ($25 == "127") nmb++
    if (st != "OK" || chk != "OK" || $11 == "0" || $12 == "")
        prob[++np] = sprintf("  %-7s %5s n=%s %-5s %-6s rep=%-2s job=%s status=%s check=%s exit=%s%s retries=%s  %s", lib, g, nn, tc, lay, $11, $2, st, chk, $25,
                             ($25 == "127" ? " (missing binary)" : ""), $24, set)
    else if (tc == "intel" && nn + 0 > 1 && $16 != "1.17.0")
        prob[++np] = sprintf("  %-7s %5s n=%s %-5s %-6s rep=%-2s job=%s UCX %s (%s)  %s", lib, g, nn, tc, lay, $11, $2, $16, $26, set)
    if ($27 == "WARN" && st == "OK" && $11 == "1")
        prob[++np] = sprintf("  %-7s %5s n=%s %-5s %-6s job=%s FFTB_BIND WARN (ranks share CPUs or threads share a core)  %s", lib, g, nn, tc, lay, $2, set)
    if (st != "OK" || chk != "OK" || $12 == "") next
    k = lib SUBSEP g SUBSEP nn SUBSEP tc SUBSEP set SUBSEP lay
    if ($27 == "WARN") bw[k] = 1
    if (!(lay in layseen)) { layseen[lay] = 1; lays[++nlay] = lay }
    if (!(k in n)) keys[++nk] = k
    n[k]++; t[k, n[k]] = $12 + 0; pl[k, n[k]] = $13 + 0
    if ($11 == "1") { nla[k]++
        if (lib == "AccFFT" && match($23, /tmethod=[^ ;]+/)) { tmv = substr($23, RSTART + 8, RLENGTH - 8)
            if (!((k, tmv) in tmc)) tml[k] = tml[k] (tml[k] == "" ? "" : "/") tmv; tmc[k, tmv]++ } }
    jb = $2; if (!((k, jb) in njb)) jbl[k] = jbl[k] " " jb; njb[k, jb]++; tjb[k, jb, njb[k, jb]] = $12 + 0
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
        if (nl[k]) { for (j = 1; j <= nl[k]; j++) c[j] = tl[k, j]; ml = median(c, nl[k]); loc = sprintf("%.4e", ml); cm = sprintf("%.0f", 100 * (med[k] - ml) / med[k])
                     tloc[k] = ml; cfr[k] = (med[k] - ml) / med[k] }
        printf "%-7s %5s %5s %-5s %-6s %4d  %.4e (%.4e-%.4e) %9.3f %11s %6s %-8s %s%s\n", p[1], p[2], p[3], p[4], p[6], m, med[k], lo[k], hi[k], mp, loc, cm, uv[k], p[5], ((k in bw) ? "  bindWARN" : "") (tml[k] != "" ? "  tm=" tmfmt(k) : "")
        gk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4] SUBSEP first(p[5])
        if (!(gk in gseen)) { gseen[gk] = 1; gord[++ng] = gk }
        if (!((gk, p[6]) in gmed) || med[k] < gmed[gk, p[6]]) { gmed[gk, p[6]] = med[k]; gkey[gk, p[6]] = k }
        if (!(gk in gbest) || med[k] < gbest[gk]) { gbest[gk] = med[k]; gblay[gk] = p[6] }
        fk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]
        if (p[6] == fixl && (!(fk in fbest) || med[k] < fbest[fk])) { fbest[fk] = med[k]; fbkey[fk] = k }
        bk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP p[4]
        if (!(bk in best) || med[k] < best[bk]) { best[bk] = med[k]; bkey[bk] = k }
    }
    # layouts in order of ranks per node, most ranks first
    for (i = 2; i <= nlay; i++) { tt = lays[i]; for (j = i - 1; j >= 1 && rpn(lays[j]) < rpn(tt); j--) lays[j + 1] = lays[j]; lays[j + 1] = tt }
    # ---- 1b. layouts
    print ""; print "--- 1b. layouts per (library, grid, nodes, toolchain, configuration): t_fft s per layout (comm% if measured), * = best"
    printf "%-7s %5s %5s %-5s %-12s", "lib", "grid", "nodes", "tc", "config"; for (i = 1; i <= nlay; i++) printf " %17s", lays[i]; printf "   best\n"
    for (o = 1; o <= 5; o++) for (gi = 1; gi <= ng; gi++) {
        split(gord[gi], p, SUBSEP); if (libord(p[1]) != o) continue
        gk = gord[gi]; printf "%-7s %5s %5s %-5s %-12s", p[1], p[2], p[3], p[4], substr(p[5], 1, 12)
        for (i = 1; i <= nlay; i++) {
            if (!((gk, lays[i]) in gmed)) { printf " %17s", "-"; continue }
            k = gkey[gk, lays[i]]; cs = (k in cfr) ? sprintf("(%2.0f%%)", 100 * cfr[k]) : "     "
            printf " %10.4e%s%s", gmed[gk, lays[i]], cs, (lays[i] == gblay[gk]) ? "*" : " "
        }
        printf "   %s\n", gblay[gk]
    }
    # ---- 1c. communication share
    print ""; print "--- 1c. communication share per configuration (FFTB_SPLIT=1): t_comm = t_fft - t_local (upper estimate: includes packing)"
    printf "%-7s %5s %5s %-5s %-6s %-12s %11s %11s %11s %6s\n", "lib", "grid", "nodes", "tc", "layout", "config", "t_fft s", "t_local s", "t_comm s", "comm%"
    nc1 = 0
    for (o = 1; o <= 5; o++) for (nn = 1; nn <= 4; nn *= 2) for (i = 1; i <= nk; i++) {
        split(keys[i], p, SUBSEP); k = keys[i]; if (libord(p[1]) != o || p[3] != nn || !(k in cfr)) continue
        nc1++
        printf "%-7s %5s %5s %-5s %-6s %-12s %11.4e %11.4e %11.4e %5.0f%%\n", p[1], p[2], p[3], p[4], p[6], substr(first(p[5]), 1, 12), med[k], tloc[k], med[k] - tloc[k], 100 * cfr[k]
    }
    if (!nc1) print "  (no runs with FFTB_SPLIT=1)"
    # ---- 2. Intel vs GCC
    print ""; print "--- 2. Intel vs GCC per (library, grid, nodes): every Intel configuration against the best GCC one"
    print "    (t_fft per FFT; 36*t = 36 x t_fft = FFT seconds per RK4 step, bench mode)"
    hasg = 0; for (bk in best) { split(bk, p, SUBSEP); if (p[4] == "gcc") hasg = 1 }
    if (!hasg) print "  (no GCC runs selected)"
    else printf "%-7s %5s %5s  %-40s %-6s %11s %9s %11s %9s %8s  %s\n", "lib", "grid", "nodes", "intel settings", "layout", "intel s", "36*t s", "best gcc s", "36*t s", "int/gcc", "ranges overlap"
    if (hasg) for (o = 1; o <= 5; o++) for (gi = 1; gi <= nk; gi++) {
        split(keys[gi], p, SUBSEP); if (libord(p[1]) != o || p[4] != "intel") continue
        gk = p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP "gcc"
        if (!(gk in best)) { printf "%-7s %5s %5s  %-40s %-6s %11.4e %9.4f %11s %9s %8s  %s\n", p[1], p[2], p[3], substr(p[5], 1, 40), p[6], med[keys[gi]], 36 * med[keys[gi]], "-", "-", "-", "-"; continue }
        k = keys[gi]; q = bkey[gk]
        ov = (lo[k] <= hi[q] && lo[q] <= hi[k]) ? "yes" : "no"
        mark = (bkey[p[1] SUBSEP p[2] SUBSEP p[3] SUBSEP "intel"] == k) ? "  <- best intel" : ""
        printf "%-7s %5s %5s  %-40s %-6s %11.4e %9.4f %11.4e %9.4f %8.3f  %s%s\n", p[1], p[2], p[3], substr(p[5], 1, 40), p[6], med[k], 36 * med[k], best[gk], 36 * best[gk], med[k] / best[gk], ov, mark
    }
    # ---- 2b. table: best Intel vs best GCC per (library, grid, nodes)
    if (hasg) {
        print ""; print "--- 2b. table: best Intel vs best GCC per (library, grid, nodes); 36*t = FFT s per RK4 step (median, min-max)"
        printf "%-7s %5s %5s | %-6s %-14s %-24s | %-6s %-14s %-24s | %8s  %s\n", "lib", "grid", "nodes", "layout", "gcc config", "gcc 36*t s (min-max)", "layout", "intel config", "intel 36*t s (min-max)", "int/gcc", "ranges overlap"
        for (o = 1; o <= 5; o++) for (gg in grids) for (nn = 1; nn <= 4; nn *= 2) for (l in libs) {
            if (libord(l) != o) continue
            gk = l SUBSEP gg SUBSEP nn SUBSEP "gcc"; ik = l SUBSEP gg SUBSEP nn SUBSEP "intel"
            if (!(gk in best) && !(ik in best)) continue
            gs = "-"; is = "-"; gl = gc = il = ic = "-"; r = "-"; ov = "-"
            if (gk in best) { q = bkey[gk]; split(q, qa, SUBSEP); gl = qa[6]; gc = first(qa[5]); gs = sprintf("%.4f (%.4f-%.4f)", 36 * med[q], 36 * lo[q], 36 * hi[q]) }
            if (ik in best) { k = bkey[ik]; split(k, ka, SUBSEP); il = ka[6]; ic = first(ka[5]); is = sprintf("%.4f (%.4f-%.4f)", 36 * med[k], 36 * lo[k], 36 * hi[k]) }
            if ((gk in best) && (ik in best)) { r = sprintf("%.3f", med[k] / med[q]); ov = (lo[k] <= hi[q] && lo[q] <= hi[k]) ? "yes" : "no" }
            printf "%-7s %5s %5s | %-6s %-14s %-24s | %-6s %-14s %-24s | %8s  %s\n", l, gg, nn, gl, substr(gc, 1, 14), gs, il, substr(ic, 1, 14), is, r, ov
        }
        print "  (int/gcc = ratio of the medians; ranges overlap = the min-max ranges of the two configurations overlap)"
    }
    # ---- 3. strong scaling (best configuration per node count) and 3b (fixed layout)
    for (pass = 1; pass <= 2; pass++) {
        print ""
        if (pass == 1) print "--- 3. strong scaling, best configuration per node count: t_fft s, layout, speedup S, efficiency E, comm%"
        else print "--- 3b. strong scaling with the fixed layout " fixl " (for comparison with 3)"
        printf "%-7s %5s %-5s", "lib", "grid", "tc"; for (nn = 1; nn <= 4; nn *= 2) printf " | %34s", nn " node" (nn > 1 ? "s" : ""); printf "\n"
        nrow = 0
        for (o = 1; o <= 5; o++) for (l in libs) { if (libord(l) != o) continue
            for (g in grids) for (ti = 1; ti <= 2; ti++) { tc = (ti == 1) ? "gcc" : "intel"
                n0 = 0; for (nn = 1; nn <= 4; nn *= 2) { bk = l SUBSEP g SUBSEP nn SUBSEP tc; if ((pass == 1 && (bk in best)) || (pass == 2 && (bk in fbest))) { n0 = nn; break } }
                if (!n0) continue
                bk = l SUBSEP g SUBSEP n0 SUBSEP tc; t0 = (pass == 1) ? best[bk] : fbest[bk]
                printf "%-7s %5s %-5s", l, g, tc; nrow++
                for (nn = 1; nn <= 4; nn *= 2) { bk = l SUBSEP g SUBSEP nn SUBSEP tc
                    if (pass == 1 && (bk in best)) { tv = best[bk]; kk = bkey[bk] } else if (pass == 2 && (bk in fbest)) { tv = fbest[bk]; kk = fbkey[bk] } else { printf " | %34s", "-"; continue }
                    split(kk, qs, SUBSEP); cs = (kk in cfr) ? sprintf("%3.0f%%", 100 * cfr[kk]) : "   -"
                    printf " | %.4e %-6s %5.2f %4.0f%% %s", tv, qs[6], t0 / tv, 100 * t0 / tv / (nn / n0), cs }
                printf "\n" } }
        if (!nrow) print "  (no data" (pass == 2 ? " with layout " fixl : "") ")"
    }
    print "  (S = T(n0) / T(n), E = S / (n / n0), n0 = smallest node count with data; the last column is the comm% of that configuration)"
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
    print ""; print "--- 5. not OK (" np + 0 (nmb ? "; " nmb " with exit 127 = missing binary: rebuild, build_fftb.sh must end with BUILD OK" : "") ")"
    for (i = 1; i <= np; i++) print prob[i]
    # ---- 6. variants against the reference configuration
    print ""; print "--- 6. " vlib " variants against " vref " (same grid, nodes, layout); comm = t_fft - t_local; per FFT"
    nvg = 0
    for (gg in grids) for (nn = 1; nn <= 4; nn *= 2) for (li = 1; li <= nlay; li++) {
        lay = lays[li]; rk = ""; nc = 0
        for (i = 1; i <= nk; i++) { split(keys[i], p, SUBSEP)
            if (p[1] != vlib || p[2] != gg || p[3] != nn || p[6] != lay) continue
            if (p[4] == "gcc" && first(p[5]) == vref && (rk == "" || med[keys[i]] < med[rk])) rk = keys[i] }
        for (i = 1; i <= nk; i++) { split(keys[i], p, SUBSEP)
            if (p[1] == vlib && p[2] == gg && p[3] == nn && p[6] == lay && p[4] == "intel") cand[++nc] = keys[i] }
        if (!nc) continue
        for (i = 2; i <= nc; i++) { tt = cand[i]; for (j = i - 1; j >= 1 && med[cand[j]] > med[tt]; j--) cand[j + 1] = cand[j]; cand[j + 1] = tt }
        nvg++
        printf "\n  %s %s, %s node%s, layout %s  (reference %s: %s)\n", vlib, gg, nn, (nn > 1 ? "s" : ""), lay, vref, (rk == "" ? "none" : sprintf("%.4e s", med[rk]))
        printf "  %-5s %-12s %-16s %3s  %-34s %10s %10s %5s %7s %8s %4s  %s\n", "tc", "config", "tmethod", "la", "t_fft s: median (min-max)", "t_local s", "t_comm s", "comm%", "t/ref", "comm/ref", "ovl", "per job"
        for (i = 0; i <= nc; i++) {
            k = (i == 0) ? rk : cand[i]; if (k == "") continue
            split(k, p, SUBSEP)
            lc = (k in tloc) ? sprintf("%10.4e", tloc[k]) : sprintf("%10s", "-"); cc = (k in tloc) ? sprintf("%10.4e", med[k] - tloc[k]) : sprintf("%10s", "-")
            cp = (k in cfr) ? sprintf("%4.0f%%", 100 * cfr[k]) : "    -"
            r1 = (rk != "" && i > 0) ? sprintf("%7.3f", med[k] / med[rk]) : sprintf("%7s", i == 0 ? "1" : "-")
            r2 = (rk != "" && i > 0 && (k in tloc) && (rk in tloc) && med[rk] > tloc[rk]) ? sprintf("%8.3f", (med[k] - tloc[k]) / (med[rk] - tloc[rk])) : sprintf("%8s", i == 0 ? "1" : "-")
            ov = (rk != "" && i > 0) ? ((lo[k] <= hi[rk] && lo[rk] <= hi[k]) ? "yes" : "no") : "-"
            printf "  %-5s %-12s %-16s %3d  %.4e (%.4e-%.4e) %s %s %5s %s %s %4s  %s\n", p[4], substr(first(p[5]), 1, 12), (tmfmt(k) == "" ? "-" : substr(tmfmt(k), 1, 16)), nla[k], med[k], lo[k], hi[k], lc, cc, cp, r1, r2, ov, jobmeds(k)
        }
    }
    if (!nvg) print "  (no " vlib " Intel runs)"
    else print "  (la = launches; t/ref and comm/ref against the reference row; ovl = min-max ranges overlap; per job = median of each job, in job order)"
}' "$CSV" "${SF[@]}"
