#!/bin/bash
# gpu_final_parse.sh [gpu_final.csv] — summary of gpu_final.sbatch
#   1. per configuration (mode, library, grid, nodes, GPUs): OK runs / runs, step s median (min-max), FFT s, non-FFT
#      s (step - fft, per run), spectral and physical pointwise s, GPU memory GB, energy check, other statuses (OOM ...)
#   2. same job: heFFTe / cuFFTMp (node, multi; single: heFFTe gpu1 / cuFFT) and cuFFTXt / cuFFTMp, per (job, grid,
#      nodes, GPUs): ratio of the medians of step, FFT and non-FFT time, and whether the step ranges overlap
#   3. "outside the FFT" by library: per (mode, grid, GPUs) the non-FFT time (step - fft), spectral, physical of every
#      library, and the ratio of the non-FFT time to the reference library's (cuFFTMp; single: cuFFT)
#   4. runs that are not OK (OOM listed separately as "expected" for 1024^3 on 1 GPU and 2048^3 on <= 8 GPUs)
# Only status OK rows count in 1-3 (energy_ok yes or close). Filters: GF_JOBS="<jobid> ...", GF_COMMIT=<hash>,
# GF_MODE=single|node|multi.
CSV=${1:-$HOME/guohaojie/results/gpu_final.csv}
[ -f "$CSV" ] || { echo "no $CSV"; exit 1; }
awk -F, -v jobs=" ${GF_JOBS:-} " -v commit="${GF_COMMIT:-}" -v modef="${GF_MODE:-}" '
function sortv(a, n,   i, j, t) { for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t } }
function median(a, n) { sortv(a, n); return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2 }
function modeord(m) { return (m == "single") ? 1 : (m == "node") ? 2 : 3 }
function libord(l) { return (l == "cufft") ? 1 : (l == "heffte_gpu1") ? 2 : (l == "cufftxt") ? 3 : (l == "cufftmp") ? 4 : (l == "heffte_mgpu") ? 5 : 6 }
function refl(m) { return (m == "single") ? "cufft" : "cufftmp" }
# stat <array name prefix via global arrays>: fills mm, ml, mh from vals[1..c]
function stat3(c,   i) { for (i = 1; i <= c; i++) zz[i] = vals[i]; sortv(zz, c); ml = zz[1]; mh = zz[c]; mm = median(zz, c) }
function fmt(x) { return (x == "") ? "-" : sprintf("%.4f", x) }
NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
commit != "" && $1 != commit { next }
jobs != "  " && index(jobs, " " $2 " ") == 0 { next }
modef != "" && $4 != modef { next }
{
    mode = $4; lib = $5; g = $6; nn = $7; gp = $8; st = $17; job = $2
    k = mode SUBSEP lib SUBSEP g SUBSEP nn SUBSEP gp
    if (!(k in tot)) keys[++nk] = k
    tot[k]++
    if (st != "OK") {
        other[k] = other[k] " " st
        expd = (st == "OOM" && ((g + 0 >= 1024 && gp + 0 == 1) || (g + 0 >= 2048 && gp + 0 <= 8)))
        prob[++np] = sprintf("  %-6s %-15s %5s n=%s gpus=%-3s rep=%s job=%s status=%s%s exit=%s retries=%s", mode, lib, g, nn, gp, $9, job, st, expd ? " (expected)" : "", $19, $20)
        next
    }
    ok[k]++; c = ok[k]
    st_[k, c] = $10 + 0; ff[k, c] = $11 + 0; nf[k, c] = $10 - $11; sp[k, c] = $12 + 0; ph[k, c] = $13 + 0; me[k, c] = $14
    if (index(" " eo[k] " ", " " $16 " ") == 0) eo[k] = eo[k] " " $16
    if ($21 != "" && pl[k] == "") pl[k] = $21
    # per job
    jk = job SUBSEP k
    if (!(jk in jn)) { jkeys[++njk] = jk }
    jn[jk]++; js[jk, jn[jk]] = $10 + 0; jf[jk, jn[jk]] = $11 + 0; jx[jk, jn[jk]] = $10 - $11
}
END {
    # ---- 1
    print "--- 1. per configuration (status OK runs; seconds per step; non-FFT = step - fft per run, median)"
    printf "%-6s %-15s %5s %5s %4s %5s  %-26s %-26s %-26s %8s %8s %7s  %-8s %s\n", "mode", "lib", "grid", "nodes", "gpus", "ok/n", "step s: median (min-max)", "fft s: median (min-max)", "non-FFT s: median (min-max)", "spectral", "physical", "mem GB", "energy", "other"
    for (o = 1; o <= 3; o++) for (lo_ = 1; lo_ <= 6; lo_++) for (i = 1; i <= nk; i++) {
        k = keys[i]; split(k, p, SUBSEP); if (modeord(p[1]) != o || libord(p[2]) != lo_) continue
        c = ok[k] + 0
        if (c == 0) { printf "%-6s %-15s %5s %5s %4s %2d/%-2d  %-26s %-26s %-26s %8s %8s %7s  %-8s %s\n", p[1], p[2], p[3], p[4], p[5], 0, tot[k], "-", "-", "-", "-", "-", "-", "-", other[k]; continue }
        for (j = 1; j <= c; j++) vals[j] = st_[k, j]; stat3(c); s1 = sprintf("%.4f (%.4f-%.4f)", mm, ml, mh); smed[k] = mm; slo[k] = ml; shi[k] = mh
        for (j = 1; j <= c; j++) vals[j] = ff[k, j]; stat3(c); s2 = sprintf("%.4f (%.4f-%.4f)", mm, ml, mh); fmed[k] = mm
        for (j = 1; j <= c; j++) vals[j] = nf[k, j]; stat3(c); s3 = sprintf("%.4f (%.4f-%.4f)", mm, ml, mh); xmed[k] = mm
        for (j = 1; j <= c; j++) vals[j] = sp[k, j]; stat3(c); spm[k] = mm
        for (j = 1; j <= c; j++) vals[j] = ph[k, j]; stat3(c); phm[k] = mm
        for (j = 1; j <= c; j++) vals[j] = me[k, j] + 0; stat3(c); mem = (me[k, 1] == "") ? "-" : sprintf("%.2f", mh)
        printf "%-6s %-15s %5s %5s %4s %2d/%-2d  %-26s %-26s %-26s %8.4f %8.4f %7s  %-8s %s\n", p[1], p[2], p[3], p[4], p[5], c, tot[k], s1, s2, s3, spm[k], phm[k], mem, substr(eo[k], 2), other[k]
    }
    print "  (mem GB = the largest \"GPU memory in use\" of the runs; energy = energy_ok values seen: yes = identical, close = rel. diff < 1e-9)"
    # ---- 2. same job ratios
    print ""; print "--- 2. same job: heFFTe / reference and cuFFTXt / cuFFTMp (ratio of the medians of that job; overlap = step ranges)"
    printf "%-10s %-6s %5s %5s %4s  %-26s %9s %9s %8s %8s %8s  %s\n", "job", "mode", "grid", "nodes", "gpus", "pair a / b", "step a s", "step b s", "step a/b", "fft a/b", "nonFFT", "overlap"
    n2 = 0
    for (i = 1; i <= njk; i++) {
        c = jn[jkeys[i]]
        for (j = 1; j <= c; j++) vals[j] = js[jkeys[i], j]; stat3(c); jm_s[jkeys[i]] = mm; jl_s[jkeys[i]] = ml; jh_s[jkeys[i]] = mh
        for (j = 1; j <= c; j++) vals[j] = jf[jkeys[i], j]; stat3(c); jm_f[jkeys[i]] = mm
        for (j = 1; j <= c; j++) vals[j] = jx[jkeys[i], j]; stat3(c); jm_x[jkeys[i]] = mm
    }
    for (o = 1; o <= 3; o++) for (i = 1; i <= njk; i++) {
        split(jkeys[i], q, SUBSEP); job = q[1]; mode = q[2]; lib = q[3]
        if (modeord(mode) != o || (lib !~ /^heffte/ && lib != "cufftxt")) continue
        rl = refl(mode); rk = job SUBSEP mode SUBSEP rl SUBSEP q[4] SUBSEP q[5] SUBSEP q[6]
        if (!(rk in jm_s)) continue
        a = jkeys[i]; n2++
        ov = (jl_s[a] <= jh_s[rk] && jl_s[rk] <= jh_s[a]) ? "yes" : "no"
        printf "%-10s %-6s %5s %5s %4s  %-26s %9.4f %9.4f %8.3f %8s %8s  %s\n", job, mode, q[4], q[5], q[6], lib " / " rl, jm_s[a], jm_s[rk], jm_s[a] / jm_s[rk],
               (jm_f[rk] > 0 ? sprintf("%8.3f", jm_f[a] / jm_f[rk]) : "-"), (jm_x[rk] > 0 ? sprintf("%8.3f", jm_x[a] / jm_x[rk]) : "-"), ov
    }
    if (!n2) print "  (no pairs: both libraries of a pair need OK runs in the same job)"
    else print "  (medians of that job; nonFFT = (step - fft) a/b; overlap = the min-max step ranges of a and b overlap)"
    # ---- 3. outside the FFT
    print ""; print "--- 3. outside the FFT (step - fft) by library, medians; vs ref = non-FFT / non-FFT of " "cuFFTMp (single: cuFFT)"
    printf "%-6s %5s %4s  %-15s %9s %9s %9s %9s %9s %7s\n", "mode", "grid", "gpus", "lib", "step", "fft", "non-FFT", "spectral", "physical", "vs ref"
    n3 = 0
    for (o = 1; o <= 3; o++) for (i = 1; i <= nk; i++) {
        k = keys[i]; split(k, p, SUBSEP); if (modeord(p[1]) != o || !(k in smed)) continue
        gk = p[1] SUBSEP p[3] SUBSEP p[4] SUBSEP p[5]
        if (gk in g3) continue; g3[gk] = 1
        rk = p[1] SUBSEP refl(p[1]) SUBSEP p[3] SUBSEP p[4] SUBSEP p[5]
        for (lo_ = 1; lo_ <= 6; lo_++) for (j = 1; j <= nk; j++) {
            k2 = keys[j]; split(k2, r, SUBSEP)
            if (libord(r[2]) != lo_ || !(k2 in smed) || r[1] != p[1] || r[3] != p[3] || r[4] != p[4] || r[5] != p[5]) continue
            vr = ((rk in xmed) && xmed[rk] > 0) ? sprintf("%7.3f", xmed[k2] / xmed[rk]) : "      -"
            printf "%-6s %5s %4s  %-15s %9.4f %9.4f %9.4f %9.4f %9.4f %s\n", r[1], r[3], r[5], r[2], smed[k2], fmed[k2], xmed[k2], spm[k2], phm[k2], vr; n3++
        }
    }
    if (!n3) print "  (no OK runs)"
    print "  (non-FFT = step - FFT phase; spectral / physical are the solver phase lines; their sum with the forcing phase is the non-FFT time)"
    # ---- 4
    print ""; print "--- 4. not OK (" np + 0 ")"
    for (i = 1; i <= np; i++) print prob[i]
}' "$CSV"
