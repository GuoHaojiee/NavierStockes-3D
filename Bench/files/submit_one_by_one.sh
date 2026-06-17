#!/bin/bash
# Watcher: 一次只在队列里保留 1 个 job
# 前一个 job 从 squeue 消失（无论成功失败）后，自动提交下一个

cd "$(dirname "$0")"

LIST_FILE="job_list.txt"
LOG_DIR="/scratch/s02230673/benchmark_results/batch2"
WATCHER_LOG="${LOG_DIR}/watcher.log"
STATE_FILE="${LOG_DIR}/watcher_state.txt"

mkdir -p "$LOG_DIR"

if [ ! -f "$LIST_FILE" ]; then
    echo "ERROR: $LIST_FILE not found. Run gen_all.sh first."
    exit 1
fi

# 起始位置：从 state 文件读，或从 1 开始
if [ -f "$STATE_FILE" ]; then
    START_IDX=$(cat "$STATE_FILE")
    echo "$(date '+%F %T')  Resuming from index $START_IDX" | tee -a "$WATCHER_LOG"
else
    START_IDX=1
    echo "$(date '+%F %T')  Fresh start, index 1" | tee -a "$WATCHER_LOG"
fi

TOTAL=$(wc -l < "$LIST_FILE")

echo "$(date '+%F %T')  Total tasks: $TOTAL" | tee -a "$WATCHER_LOG"

IDX=0
while IFS= read -r SBATCH_FILE; do
    IDX=$((IDX + 1))
    
    if [ $IDX -lt $START_IDX ]; then
        continue
    fi
    
    if [ ! -f "$SBATCH_FILE" ]; then
        echo "$(date '+%F %T')  SKIP [${IDX}/${TOTAL}]: $SBATCH_FILE not found" | tee -a "$WATCHER_LOG"
        continue
    fi
    
    echo "" | tee -a "$WATCHER_LOG"
    echo "============================================================" | tee -a "$WATCHER_LOG"
    echo "$(date '+%F %T')  [${IDX}/${TOTAL}] Submitting: $SBATCH_FILE" | tee -a "$WATCHER_LOG"
    
    # 提交
    JOBID=$(sbatch --parsable "$SBATCH_FILE")
    SUBMIT_RC=$?
    
    if [ $SUBMIT_RC -ne 0 ] || [ -z "$JOBID" ]; then
        echo "$(date '+%F %T')  ERROR: sbatch failed for $SBATCH_FILE (rc=$SUBMIT_RC)" | tee -a "$WATCHER_LOG"
        echo "$(date '+%F %T')  Aborting watcher. Resume by deleting $STATE_FILE or set it to $IDX" | tee -a "$WATCHER_LOG"
        exit 1
    fi
    
    echo "$(date '+%F %T')  Submitted as JobID=$JOBID, waiting for completion..." | tee -a "$WATCHER_LOG"
    
    # 写入 state（下次从 IDX+1 继续）
    echo $((IDX + 1)) > "$STATE_FILE"
    
    # 轮询：等 jobid 从 squeue 消失
    while true; do
        # squeue 返回 JOBID 行数；0 表示已消失
        COUNT=$(squeue -h -j "$JOBID" 2>/dev/null | wc -l)
        if [ "$COUNT" -eq 0 ]; then
            break
        fi
        sleep 30
    done
    
    # 取最终状态
    FINAL_STATE=$(sacct -j "$JOBID" -X --format=State --noheader | head -1 | xargs)
    FINAL_EXIT=$(sacct -j "$JOBID" -X --format=ExitCode --noheader | head -1 | xargs)
    
    echo "$(date '+%F %T')  JobID=$JOBID finished: State=$FINAL_STATE ExitCode=$FINAL_EXIT" | tee -a "$WATCHER_LOG"
    
    # 任务之间间隔（让 /dev/shm 等资源彻底清理）
    sleep 20
    
done < "$LIST_FILE"

echo "" | tee -a "$WATCHER_LOG"
echo "============================================================" | tee -a "$WATCHER_LOG"
echo "$(date '+%F %T')  ALL DONE." | tee -a "$WATCHER_LOG"
echo "============================================================" | tee -a "$WATCHER_LOG"

# 清除 state，下次重跑会从头开始
rm -f "$STATE_FILE"
