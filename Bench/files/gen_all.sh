#!/bin/bash
# 生成 20 个 sbatch 文件 + job_list.txt 序列文件

cd "$(dirname "$0")"

# 节点规模列表
NODES_LIST=(1 2 4 8 16)

# 清空旧的
rm -f b2_*.sbatch job_list.txt

# 顺序：先把所有 cuFFTmp 跑完，再跑 heFFTe（便于观察单库的扩展曲线）
# 每个规模 × 每个库 × 2 rep

for LIB in cufftmp heffte; do
    for N in "${NODES_LIST[@]}"; do
        for R in 1 2; do
            JOBNAME="b2_${N}n_${LIB}_r${R}"
            FILE="${JOBNAME}.sbatch"
            
            sed -e "s/__JOBNAME__/${JOBNAME}/g" \
                -e "s/__NNODES__/${N}/g" \
                "tmpl_${LIB}.sbatch" > "$FILE"
            chmod +x "$FILE"
            echo "$FILE" >> job_list.txt
        done
    done
done

echo "Generated $(wc -l < job_list.txt) sbatch files."
echo ""
echo "Execution order:"
cat -n job_list.txt
