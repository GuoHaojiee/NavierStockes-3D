#!/bin/bash
# 只生成 heFFTe + bind 的 10 个 sbatch

cd "$(dirname "$0")"

NODES_LIST=(1 2 4 8 16)

rm -f b2b_*.sbatch b2b_job_list.txt

for N in "${NODES_LIST[@]}"; do
    for R in 1 2; do
        JOBNAME="b2b_${N}n_heffte_bind_r${R}"
        FILE="${JOBNAME}.sbatch"
        
        sed -e "s/__JOBNAME__/${JOBNAME}/g" \
            -e "s/__NNODES__/${N}/g" \
            tmpl_heffte_bind.sbatch > "$FILE"
        chmod +x "$FILE"
        echo "$FILE" >> b2b_job_list.txt
    done
done

echo "Generated $(wc -l < b2b_job_list.txt) sbatch files."
echo ""
echo "Execution order:"
cat -n b2b_job_list.txt
