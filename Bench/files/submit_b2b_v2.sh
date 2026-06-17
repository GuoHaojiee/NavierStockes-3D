#!/bin/bash
cd "$(dirname "$0")"

LIST_FILE="b2b_v2_job_list.txt"
LOG_DIR="/scratch/s02230673/benchmark_results/batch2_bind_v2"
WATCHER_LOG="${LOG_DIR}/watcher.log"
STATE_FILE="${LOG_DIR}/watcher_state.txt"

mkdir -p "$LOG_DIR"

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
    [ $IDX -lt $START_IDX ] && continue
    [ ! -f "$SBATCH_FILE" ] && continue
    
    echo "" | tee -a "$WATCHER_LOG"
    echo "============================================================" | tee -a "$WATCHER_LOG"
    echo "$(date '+%F %T')  [${IDX}/${TOTAL}] Submitting: $SBATCH_FILE" | tee -a "$WATCHER_LOG"
    
    JOBID=$(sbatch --parsable "$SBATCH_FILE")
    [ -z "$JOBID" ] && { echo "ERROR: sbatch failed" | tee -a "$WATCHER_LOG"; exit 1; }
    
    echo "$(date '+%F %T')  Submitted as JobID=$JOBID" | tee -a "$WATCHER_LOG"
    echo $((IDX + 1)) > "$STATE_FILE"
    
    while true; do
        COUNT=$(squeue -h -j "$JOBID" 2>/dev/null | wc -l)
        [ "$COUNT" -eq 0 ] && break
        sleep 30
    done
    
    FINAL_STATE=$(sacct -j "$JOBID" -X --format=State --noheader | head -1 | xargs)
    FINAL_EXIT=$(sacct -j "$JOBID" -X --format=ExitCode --noheader | head -1 | xargs)
    echo "$(date '+%F %T')  JobID=$JOBID finished: State=$FINAL_STATE ExitCode=$FINAL_EXIT" | tee -a "$WATCHER_LOG"
    sleep 20
done < "$LIST_FILE"

echo "$(date '+%F %T')  ALL DONE." | tee -a "$WATCHER_LOG"
rm -f "$STATE_FILE"
