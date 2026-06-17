#!/bin/bash
# 扫所有 batch2_bind/b2b_*.log，提取数据成 CSV

LOG_DIR=/scratch/s02230673/benchmark_results/batch2_bind
OUT_CSV=${LOG_DIR}/batch2_bind_summary.csv

cd "$LOG_DIR"

echo "lib,nodes,gpus,rep,jobid,decomp,avg_step_s,total_wall_s,l2_error,gpu_mem_max_mib" > "$OUT_CSV"

for LOG in b2b_*n_heffte_bind_r*_*.log; do
    [[ "$LOG" == *_err_* ]] && continue
    
    # 文件名: b2b_<N>n_heffte_bind_r<R>_<jobid>.log
    BASE=$(basename "$LOG" .log)
    IFS='_' read -ra PARTS <<< "$BASE"
    # PARTS = [b2b, <N>n, heffte, bind, r<R>, <jobid>]
    NNODES=${PARTS[1]%n}
    REP=${PARTS[4]#r}
    JOBID=${PARTS[5]}
    GPUS=$((NNODES * 8))
    LIB="heffte_bind"
    
    AVG=$(grep "Avg per step" "$LOG" | awk '{print $4}')
    TOTAL=$(grep "Total wall time" "$LOG" | awk '{print $4}')
    L2=$(grep "L2 error" "$LOG" | head -1 | awk '{print $NF}')
    DECOMP=$(grep "slab grid" "$LOG" | head -1 | awk -F: '{print $2}' | tr -d ' ')
    [ -z "$DECOMP" ] && DECOMP="-"
    
    MEMMAX=$(grep -A 9 "GPU memory" "$LOG" | grep -E "^[0-9]+, [0-9]+ MiB" | awk -F, '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    [ -z "$MEMMAX" ] && MEMMAX="-"
    
    echo "${LIB},${NNODES},${GPUS},${REP},${JOBID},${DECOMP},${AVG},${TOTAL},${L2},${MEMMAX}" >> "$OUT_CSV"
done

echo "Wrote $OUT_CSV"
echo ""
echo "=== Sorted by gpus, rep ==="
(head -1 "$OUT_CSV"; tail -n +2 "$OUT_CSV" | sort -t, -k3,3n -k4,4n)
