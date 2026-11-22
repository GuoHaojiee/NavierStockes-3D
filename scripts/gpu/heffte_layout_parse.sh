#!/bin/bash
# heffte_layout_parse.sh <hlc log or hlc_<job>.results> — summary of heffte_layout_check.sbatch
#   1. 64^3 verify: L2 per (program, GPUs, layout), OK when it equals the reference 6.0041e-10
#   2. 512^3 bench: per (program, GPUs, layout): runs, energy ok / runs, median step time (min-max),
#      median FFT time per step, the plan line (r2c direction, reorder, algorithm, reshapes per FFT)
#   3. zslab vs xslab per (program, GPUs): ratio of the median step times, and whether the min-max
#      ranges overlap (then the difference is within the run-to-run scatter)
#   4. runs that are not OK (wrong value, error exit, no result)
F=${1:?usage: heffte_layout_parse.sh <log or .results file>}
grep -h "^RESULT " "$F" | awk '
function kv(s,   n, i, a, b) { n = split(s, a, " "); for (i = 2; i <= n; i++) { split(a[i], b, "="); V[b[1]] = substr(a[i], length(b[1]) + 2) } }
function sortv(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } }
function median(a, n) { sortv(a, n); return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
{
    delete V; kv($0)
    k = V["prog"] SUBSEP V["gpus"] SUBSEP V["layout"]
    if (V["status"] != "OK") bad[++nb] = $0
    if (V["mode"] == "verify") { if (!(k in vl)) vord[++nv] = k; vl[k] = V["value"]; vs[k] = V["status"]; next }
    if (!(k in runs)) { bord[++nk] = k }
    runs[k]++
    if (V["status"] == "OK") eok[k]++
    if (V["step_s"] != "-" && V["step_s"] != "") { ns[k]++; st[k, ns[k]] = V["step_s"] + 0; ff[k, ns[k]] = V["fft_s"] + 0 }
    pl[k] = "r2c=" V["r2c_dir"] " reorder=" V["reorder"] " algo=" V["algo"] " reshapes=" V["reshapes"]
    bn = V["N"]; ref = V["ref"]
}
END {
    print "--- 64^3 verify (reference L2 6.0041e-10)"
    printf "  %-20s %4s %-7s %-12s %s\n", "program", "gpus", "layout", "L2", "status"
    for (i = 1; i <= nv; i++) { split(vord[i], p, SUBSEP); printf "  %-20s %4s %-7s %-12s %s\n", p[1], p[2], p[3], vl[vord[i]], vs[vord[i]] }
    print ""
    print "--- " bn "^3 bench (reference energy " ref "): step time per configuration"
    printf "  %-20s %4s %-7s %4s %6s  %-28s %10s  %s\n", "program", "gpus", "layout", "runs", "en.ok", "s/step: median (min-max)", "FFT s/step", "plan"
    for (i = 1; i <= nk; i++) {
        k = bord[i]; split(k, p, SUBSEP); m = ns[k]
        if (m) {
            for (j = 1; j <= m; j++) { a[j] = st[k, j]; b[j] = ff[k, j] }
            sortv(a, m); lo[k] = a[1]; hi[k] = a[m]; med[k] = median(a, m); mf = median(b, m)
            s = sprintf("%.4f (%.4f-%.4f)", med[k], lo[k], hi[k])
        } else { s = "-"; mf = 0 }
        printf "  %-20s %4s %-7s %4d %3d/%-2d  %-28s %10.4f  %s\n", p[1], p[2], p[3], runs[k], eok[k] + 0, runs[k], s, mf, pl[k]
    }
    print ""
    print "--- zslab vs xslab (median step time; only configurations with the reference energy in every run)"
    printf "  %-20s %4s %10s %10s %14s  %s\n", "program", "gpus", "zslab s", "xslab s", "zslab/xslab", "ranges overlap"
    for (i = 1; i <= nk; i++) {
        k = bord[i]; split(k, p, SUBSEP); if (p[3] != "zslab") continue
        x = p[1] SUBSEP p[2] SUBSEP "xslab"
        okz = (ns[k] && eok[k] == runs[k]); okx = ((x in runs) && ns[x] && eok[x] == runs[x])
        if (!okz || !okx) { printf "  %-20s %4s %10s %10s %14s  %s\n", p[1], p[2], okz ? sprintf("%.4f", med[k]) : "invalid", okx ? sprintf("%.4f", med[x]) : "invalid", "-", "-"; continue }
        ov = (lo[k] <= hi[x] && lo[x] <= hi[k]) ? "yes" : "no"
        printf "  %-20s %4s %10.4f %10.4f %14.3f  %s\n", p[1], p[2], med[k], med[x], med[k] / med[x], ov
    }
    print ""
    print "--- runs that are not OK (" nb + 0 ")"
    for (i = 1; i <= nb; i++) print "  " bad[i]
}'
