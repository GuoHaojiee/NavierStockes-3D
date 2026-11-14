#!/bin/bash
# fb_parse.sh — turn the RESULT lines (and failures) of one or more fb logs into readable tables
#   bash fb_parse.sh ~/guohaojie/logs/fb1_*.out
#   bash fb_parse.sh $(ls -t ~/guohaojie/logs/fb2_*.out | head -1)
#
# cuFFTMp table (times in ms per FFT, B in GB/s per node, percentages relative to PEAK = 200 GB/s per node):
#   dist    = one distributed FFT (mean of forward and inverse)
#   local   = estimate of its local part (plain cuFFT, no communication)
#   reshape = one transpose alone (cuFFTMp standalone reshape, no FFT)
#   a2a     = MPI_Alltoall of the same volume on GPU buffers
#   B_dist  = D / dist            lower bound
#   B_comm  = D / (dist - local)  local-corrected
#   B_resh  = D / reshape         direct
# A row marked "partial" comes from a run that stopped after phase [A]; its later columns are empty.
PEAK=${FB_PEAK_NODE_GBS:-200}
for f in "$@"; do
  echo "=== $f"
  awk -v PEAK="$PEAK" '
    function kv(line, k,   a, n, i, p) { n = split(line, a, " "); for (i = 1; i <= n; i++) { p = index(a[i], "="); if (p > 0 && substr(a[i], 1, p-1) == k) return substr(a[i], p+1) } return "-" }
    function pct(b) { return (b == "-" || b + 0 == 0) ? "-" : sprintf("%.0f%%", 100 * b / PEAK) }
    function z(v) { return (v == "-" || v + 0 == 0) ? "-" : v }
    function mp_head() { if (!h1++) printf "%-14s %3s %5s %-12s %8s %8s %8s %8s %8s %8s %7s | %7s %5s %7s %5s %7s %5s %7s  %s\n", "N","P","nodes","switches","fwd","inv","dist","local","reshape","a2a","D_GB","B_dist","","B_comm","","B_resh","","B_a2a","" }
    function mp_row(l, note) {
      mp_head()
      printf "%-14s %3s %5s %-12s %8s %8s %8s %8s %8s %8s %7s | %7s %5s %7s %5s %7s %5s %7s  %s\n", kv(l,"N"),kv(l,"P"),kv(l,"nodes"),kv(l,"knobs"),kv(l,"t_fwd_ms"),kv(l,"t_inv_ms"),kv(l,"t_dist_ms"),z(kv(l,"t_local_ms")),z(kv(l,"t_reshape_ms")),z(kv(l,"t_a2a_ms")),kv(l,"D_node_GB"),z(kv(l,"B_dist")),pct(kv(l,"B_dist")),z(kv(l,"B_comm")),pct(kv(l,"B_comm")),z(kv(l,"B_reshape")),pct(kv(l,"B_reshape")),z(kv(l,"B_a2a")),note
    }
    function flush_partial() { if (resA != "" && !got) mp_row(resA, "partial"); resA = "" }
    /^>>> / { flush_partial(); cur = $0; got = 0; next }
    /^RES_A lib=cufftmp/ { resA = $0; next }
    /^RESULT lib=cufftmp/ { mp_row($0, ""); got = 1; resA = ""; next }
    /^RESULT lib=heffte/ {
      if (!h2++) printf "%-11s %-14s %3s %5s %-7s %-7s %3s %-10s %3s %3s %4s %9s %9s %9s %9s %8s %7s\n", "lib","N","P","nodes","in","out","r2c","algo","ro","bat","resh","fwd_ms","bwd_ms","mean_ms","step_s","B_node","ws_GiB"
      printf "%-11s %-14s %3s %5s %-7s %-7s %3s %-10s %3s %3s %4s %9s %9s %9s %9s %8s %7s\n", kv($0,"lib"),kv($0,"N"),kv($0,"P"),kv($0,"nodes"),kv($0,"in"),kv($0,"out"),kv($0,"r2c"),kv($0,"algo"),kv($0,"reorder"),kv($0,"batch"),kv($0,"reshapes"),kv($0,"t_fwd_ms"),kv($0,"t_bwd_ms"),kv($0,"t_mean_ms"),kv($0,"step_fft_s"),z(kv($0,"B_node")),kv($0,"ws_GiB")
      got = 1; next }
    /^RESULT lib=fftw/ {
      if (!h3++) printf "%-11s %-14s %4s %3s %-9s %6s %9s %9s %9s %9s %8s\n", "lib","N","P","thr","flag","transp","fwd_ms","bwd_ms","mean_ms","step_s","plan_s"
      tr = kv($0,"transposes"); if (tr == "-") tr = 1
      printf "%-11s %-14s %4s %3s %-9s %6s %9s %9s %9s %9s %8s\n", "fftw",kv($0,"N"),kv($0,"P"),kv($0,"threads"),kv($0,"flag"),tr,kv($0,"t_fwd_ms"),kv($0,"t_bwd_ms"),kv($0,"t_mean_ms"),kv($0,"step_fft_s"),kv($0,"plan_s")
      got = 1; next }
    /^heFFTe way|^ESTIMATE on real|^MEASURE  on real|^PATIENT  on real|^== (contiguous|strided)/ { print "  " $0; got = 1; next }
    /\[check\].*FAIL/ { print "  !! CHECK FAILED in: " cur }
    /WARNING/ { print "  !! " $0 "   <- " cur }
    /skipped:|IBGDA probe failed|NOTE: FB_BATCH/ { print "  .. " $0 }
    /out of memory|OOM|cudaErrorMemoryAllocation|CUDA error|cuFFT error|bad_alloc|MPI_ABORT|Abort|error:|CANCELLED|TIME LIMIT/ { if (cur != lastbad) { print "  !! error in: " cur; print "     " $0; lastbad = cur } }
    /^<<< exit=/ { split($0, e, "="); if (e[2] + 0 != 0 && cur != lastexit) { print "  !! exit code " e[2] " in: " cur; lastexit = cur } }
    END { flush_partial() }
  ' "$f"
done
