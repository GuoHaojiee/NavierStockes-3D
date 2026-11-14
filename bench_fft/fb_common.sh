# fb_common.sh — launch helpers for the pure-FFT benchmarks (sourced by fb*.sbatch)
#
# CHECK ONCE before the first run: the srun lines below must match how your existing multi-node
# GPU jobs (e.g. the dD cuFFTMp runs) and CPU jobs (fl_scan) start inside the container:
#   grep -n "srun" ~/guohaojie/runs/dD*.sbatch ~/guohaojie/runs/fl_scan.sbatch | head
#   head -20 ~/guohaojie/build/bind_hca.sh
# If they differ, edit only the srun lines / the inner command in G() and C().

source ~/guohaojie/runs/common.sh          # defines $M, precheck, cleanup
BIN=/work/build/bin_fb
STEP_MIN=${FB_STEP_MIN:-12}                # a run that hangs is killed after this many minutes; the job goes on

# G <nodes> <ranks_per_node> "<VAR=val ...>" <program> <args...>   — one rank per GPU
# The variables are exported inside the container for every rank and printed in the ">>>" line,
# so every block of the log says exactly which switches were on. Returns the exit code of the run.
G() {
    local n=$1 rpn=$2 envs=$3 rc; shift 3
    echo ">>> [$(date +%H:%M:%S)] GPU nodes=$n ranks/node=$rpn env: ${envs:-none} | $*"
    srun --mpi=pmix --cpus-per-task=16 --time=$STEP_MIN -N $n --ntasks-per-node=$rpn $M bash -c "export NS_MODE=gpu; source /work/build/run_env.sh; export FB_DUMMY=1 $envs; \
        if [ -x /work/build/bind_hca.sh ]; then exec /work/build/bind_hca.sh $BIN/$*; else exec $BIN/$*; fi"
    rc=$?
    echo "<<< exit=$rc"
    echo
    return $rc
}
# C <ranks> <threads> "<VAR=val ...>" <program> <args...>   — CPU, one node
C() {
    local np=$1 nt=$2 envs=$3 rc; shift 3
    echo ">>> [$(date +%H:%M:%S)] CPU ranks=$np threads=$nt env: ${envs:-none} | $*"
    srun --mpi=pmix --cpu-bind=cores --time=$STEP_MIN -N 1 -n $np -c $nt $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close $envs; exec $BIN/$*"
    rc=$?
    echo "<<< exit=$rc"
    echo
    return $rc
}
