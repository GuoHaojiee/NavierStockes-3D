#!/bin/bash
# 扫所有 b2_*.log，提取关键数据，输出 CSV

LOG_DIR=/scratch/s02230673/benchmark_results/batch2
OUT_CSV=${LOG_DIR}/batch2_summary.csv

cd "$LOG_DIR"

echo "lib,nodes,gpus,rep,jobid,decomp,avg_step_s,total_wall_s,l2_error,max_div,gpu_mem_max_mib" > "$OUT_CSV"

for LOG in b2_*n_*_r*_*.log; do
    # 跳过 _err_ 文件
    [[ "$LOG" == *_err_* ]] && continue
    
    # 文件名格式: b2_<N>n_<lib>_r<R>_<jobid>.log
    BASE=$(basename "$LOG" .log)
    # 用下划线切分
    IFS='_' read -ra PARTS <<< "$BASE"
    # PARTS = [b2, <N>n, <lib>, r<R>, <jobid>]
    NNODES=${PARTS[1]%n}
    LIB=${PARTS[2]}
    REP=${PARTS[3]#r}
    JOBID=${PARTS[4]}
    GPUS=$((NNODES * 8))
    
    # 从 log 抽数据
    AVG=$(grep "Avg per step" "$LOG" | awk '{print $4}')
    TOTAL=$(grep "Total wall time" "$LOG" | awk '{print $4}')
    L2=$(grep "L2 error" "$LOG" | head -1 | awk '{print $NF}')
    DIV=$(grep "max|div V|" "$LOG" | awk '{print $NF}')
    
    # 分解配置（cuFFTmp 没有，heFFTe 有 slab grid）
    DECOMP=$(grep -E "slab grid|pencil" "$LOG" | head -1 | awk -F: '{print $2}' | tr -d ' ')
    [ -z "$DECOMP" ] && DECOMP="-"
    
    # GPU 显存峰值
    MEMMAX=$(grep -A 9 "GPU memory" "$LOG" | grep -E "^[0-9]+, [0-9]+ MiB" | awk -F, '{print $2}' | awk '{print $1}' | sort -n | tail -1)
    [ -z "$MEMMAX" ] && MEMMAX="-"
    
    echo "${LIB},${NNODES},${GPUS},${REP},${JOBID},${DECOMP},${AVG},${TOTAL},${L2},${DIV},${MEMMAX}" >> "$OUT_CSV"
done

echo "Wrote $OUT_CSV"
echo ""
echo "=== Sorted by lib, gpus, rep ==="
(head -1 "$OUT_CSV"; tail -n +2 "$OUT_CSV" | sort -t, -k1,1 -k3,3n -k4,4n) | column -t -s,
