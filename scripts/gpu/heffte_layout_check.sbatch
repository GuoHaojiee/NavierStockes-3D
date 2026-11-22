#!/bin/bash
#SBATCH -J hlc
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 01:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/hlc_%j.out
# heffte_layout_check — the heFFTe GPU solvers with both slab layouts (NS_HEFFTE_LAYOUT), one node.
#   ns_heffte_gpu1     : 1 GPU (its own min-surface grid, no layout switch)
#   ns_heffte_mgpu     : 2 / 4 / 8 GPUs x zslab / xslab
#   ns_heffte_multigpu : 2 / 4 / 8 GPUs x zslab / xslab
# Every configuration:
#   64^3 verify, dt=5e-5, 10 steps, once      -> L2 must be     6.0041e-10
#   512^3 bench, dt=1e-6, 10 steps, HLC_REP x -> energy must be 1.7950250065e+08; step time compared
# NS_HEFFTE_ALGO / NS_HEFFTE_REORDER are left to the solvers' defaults (p2p_plined / 0 for mgpu and
# multigpu, alltoallv / 0 for gpu1) unless set in HLC_ENV, e.g. HLC_ENV="NS_HEFFTE_ALGO=alltoallv".
# Binaries: /work/build/bin_gpu_new from scripts/gpu/build_heffte_gpu.sh (override: HLC_BIN=...).
# Options: HLC_REP (3), HLC_NGPU ("2 4 8"), HLC_PROGS ("mgpu multigpu"), HLC_LAYOUTS ("zslab xslab"),
#          HLC_GPU1 (1: also run gpu1), HLC_BENCH_N (512), HLC_ENV (extra variables for every run).
# Submit:
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/gpu && sbatch heffte_layout_check.sbatch
# Results (the summary is also printed at the end of the log):
#   bash ~/guohaojie/src/NavierStockes-3D/scripts/gpu/heffte_layout_parse.sh $(ls -t ~/guohaojie/logs/hlc_*.out | head -1)
source ~/guohaojie/runs/common.sh          # $M, precheck, cleanup
precheck
BIN=${HLC_BIN:-/work/build/bin_gpu_new}
HBIN=${BIN/#\/work/$HOME/guohaojie}        # the same directory seen from outside the container
REP=${HLC_REP:-3}; NB=${HLC_BENCH_N:-512}
L2REF=6.0041e-10; EREF=${HLC_EREF:-1.7950250065e+08}
D=$(cd "$(dirname "$0")" 2>/dev/null && pwd); [ -f "$D/heffte_layout_parse.sh" ] || D=$HOME/guohaojie/src/NavierStockes-3D/scripts/gpu
echo "heffte_layout_check: commit $(git -C $HOME/guohaojie/src/NavierStockes-3D rev-parse --short HEAD 2>/dev/null) job ${SLURM_JOB_ID:-?} node(s) ${SLURM_JOB_NODELIST:-?}"
echo "binaries $BIN:"; ls -l $HBIN/ns_heffte_* 2>/dev/null | grep -v "\.log$" || echo "  (not visible from the host as $HBIN)"
echo "L2REF=$L2REF EREF=$EREF bench N=$NB rep=$REP extra env: ${HLC_ENV:-none}"
echo

# R <program> <ranks> <layout or -> <mode> <rep>
R() {
    local prog=$1 np=$2 lay=$3 mode=$4 rep=$5 envs="${HLC_ENV:-}" N DT ref out rc val st step fft plan
    [ "$lay" != "-" ] && envs="NS_HEFFTE_LAYOUT=$lay $envs"
    if [ $mode = verify ]; then N=64; DT=5e-5; ref=$L2REF; else N=$NB; DT=1e-6; ref=$EREF; fi
    echo ">>> [$(date +%H:%M:%S)] $prog gpus=$np layout=$lay $mode rep=$rep env: ${envs:-none} | $prog $N $N $N $DT 10 $mode"
    out=$(srun --mpi=pmix --cpus-per-task=16 --time=10 -N 1 --ntasks-per-node=$np $M bash -c "export NS_MODE=gpu; source /work/build/run_env.sh; \
        export $envs X_HLC=1; exec $BIN/$prog $N $N $N $DT 10 $mode" 2>&1)
    rc=$?
    echo "$out"; echo "<<< exit=$rc"; echo
    if [ $mode = verify ]; then val=$(echo "$out" | awk '/L2 error/ {print $NF; exit}')
    else val=$(echo "$out" | awk '/Kinetic energy/ {print $NF; exit}'); fi
    [ -n "$val" ] && [ "$val" = "$ref" ] && st=OK || st=MISMATCH
    [ $rc != 0 ] && st=ERROR
    step=$(echo "$out" | awk '/Avg per step:/ {print $(NF-1); exit}')
    fft=$(echo "$out" | awk '/FFT \(incl/ {print $(NF-1); exit}')
    plan=$(echo "$out" | grep -m1 '^HEFFTE_PLAN' | cut -d' ' -f3-)
    printf "RESULT prog=%s gpus=%s layout=%s mode=%s N=%s rep=%s value=%s ref=%s status=%s step_s=%s fft_s=%s exit=%s %s\n" \
        $prog $np $lay $mode $N $rep "${val:-none}" $ref $st "${step:--}" "${fft:--}" $rc "${plan:-plan=none}" | tee -a $RES
}
RES=$HOME/guohaojie/logs/hlc_${SLURM_JOB_ID:-local}.results; : > $RES

if [ "${HLC_GPU1:-1}" = 1 ]; then
    R ns_heffte_gpu1 1 - verify 1
    for r in $(seq $REP); do R ns_heffte_gpu1 1 - bench $r; done
fi
for p in ${HLC_PROGS:-mgpu multigpu}; do
    for np in ${HLC_NGPU:-2 4 8}; do
        for L in ${HLC_LAYOUTS:-zslab xslab}; do
            R ns_heffte_$p $np $L verify 1
        done
        # bench repetitions interleaved over the layouts, so a slow phase of the node hits both
        for r in $(seq $REP); do for L in ${HLC_LAYOUTS:-zslab xslab}; do
            R ns_heffte_$p $np $L bench $r
        done; done
    done
done

echo "================ summary"
bash $D/heffte_layout_parse.sh $RES
cleanup
