#!/bin/bash
# Watcher for heFFTe + HCA bind retest (batch2_bind)

cd "$(dirname "$0")"

LIST_FILE="b2b_job_list.txt"
LOG_DIR="/scratch/s02230673/benchmark_results/batch2_bind"
WATCHER_LOG="${LOG_DIR}/watcher.log"
STATE_FILE="${LOG_DIR}/watcher_state.txt"

mkdir -p "$LOG_DIR"

if [ ! -f "$LIST_FILE" ]; then
    echo "ERROR: $LIST_FILE not found. Run gen_heffte_bind.sh first."
    exit 1
fi

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
    
    if [ ! -f "$SBATCH_FILE" ]; then
        echo "$(date '+%F %T')  SKIP [${IDX}/${TOTAL}]: $SBATCH_FILE not found" | tee -a "$WATCHER_LOG"
        continue
    fi
    
    echo "" | tee -a "$WATCHER_LOG"
    echo "============================================================" | tee -a "$WATCHER_LOG"
    echo "$(date '+%F %T')  [${IDX}/${TOTAL}] Submitting: $SBATCH_FILE" | tee -a "$WATCHER_LOG"
    
    JOBID=$(sbatch --parsable "$SBATCH_FILE")
    if [ $? -ne 0 ] || [ -z "$JOBID" ]; then
        echo "$(date '+%F %T')  ERROR: sbatch failed for $SBATCH_FILE" | tee -a "$WATCHER_LOG"
        exit 1
    fi
    
    echo "$(date '+%F %T')  Submitted as JobID=$JOBID, waiting for completion..." | tee -a "$WATCHER_LOG"
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

echo "" | tee -a "$WATCHER_LOG"
echo "$(date '+%F %T')  ALL DONE." | tee -a "$WATCHER_LOG"
rm -f "$STATE_FILE"
