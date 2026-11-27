#!/bin/bash
#SBATCH -J dtcfl
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 02:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/dtcfl_%x_%j.out
# dtcfl — CFL-adaptive vs fixed time step, FFTW CPU solver, Taylor–Green Re=1600 (NS_NU=0.000625), NS_NYQ0=1, 256^3,
# bench (f = 0), Intel toolchain (v2 image, /work/build/bin_intel/ns_fftw), 1 node, FFTW 64x2, NS_FFTW_PLAN=measure.
# Every run prints a "DT" line every step (NS_DT_PRINT=1) so that dtcfl_parse.sh can read t=10 values from the series;
# the scan costs ~0.06 % of a step and is paid by cfl and fixed runs alike.
# One job per group (argument 1):
#   cfl   [Cs]         NS_DT_MODE=cfl, NS_T_END=20, C = ${Cs:-0.5 0.75 0.9}, step cap 20000
#   probe [Cs]         the same with T_END=12, C = ${Cs:-1.0 1.1 1.25} (instability probe), step cap 6000
#   ref                fixed dt = 0.00125, 16000 steps (T = 20): reference solution (a)
#   fixed <cfl jobid> [Cs]
#                      fixed baselines (b): per C, dt_min of the cfl run of job <cfl jobid> (DTCFL_SUMMARY dt_min,
#                      without the cut last step), NSTEPS = ceil(20/dt_min), dt = 20/NSTEPS (<= dt_min, ends at t=20).
#                      Reads files only, so it does not matter whether Slurm still knows the cfl job; submit it with
#                      --dependency=afterok:<cfl jobid> --kill-on-invalid-dep=yes right after the cfl job.
# Output: one file per run, $SCR/results/dtcfl/<jobid>/<label>.txt (label cfl_C0.5, probe_C1.1, ref, fixed_C0.9);
# the job log keeps the summary lines.  Analysis: dtcfl_parse.sh <cfl jobid> <fixed jobid> <ref jobid> [probe jobid].
# Expected (s = seconds per step, ~0.2-0.3 s on 64x2): cfl ~3400 steps, probe <= ~1700, ref 16000, fixed ~7700 steps
# (+ ~1 min MEASURE planning per run). Time limits on the sbatch command line (see README.md).
# Options: DTC_IMAGE (v2), DTC_N (256), DTC_LAYOUT (64x2), DTC_RESULTS ($SCR/results/dtcfl).
MODE=${1:?usage: dtcfl.sbatch cfl|probe|ref|fixed [...]}
export INTEL_IMAGE=${DTC_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
N=${DTC_N:-256}; L=${DTC_LAYOUT:-64x2}; PPN=${L%x*}; NT=${L#*x}
BIN=/work/build/bin_intel/ns_fftw; HBIN=$SCR/${BIN#/work/}
RES=${DTC_RESULTS:-$SCR/results/dtcfl}
OUT=$RES/$SLURM_JOB_ID; mkdir -p $OUT
BASE="NS_DECOMP=slab NS_FFTW_PLAN=measure NS_NYQ0=1 NS_IC=tg NS_NU=0.000625 NS_DT_PRINT=1"
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)
KEEP="^Grid:|^DTCFL_CONFIG|^DTCFL_SUMMARY|^DTCFL_CAP_HIT|^BLOWUP|^FINAL|Avg per step:|planning time|Scan \(|rror|abort"

echo "dtcfl: mode=$MODE job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, commit $COMMIT, image $INTEL_IMAGE, $N^3, layout $L"
echo "  results $OUT"
echo "  binary $BIN  $(ls -l --time-style=+%F_%R $HBIN 2>&1 | awk '{print $6}')"
grep -aq NS_NYQ0 $HBIN || { echo "ERROR: $HBIN has no NS_NYQ0 support: git pull and rebuild (scripts/intel/build_intel_solvers.sh)"; exit 1; }

# run <label> <extra env> <dt> <nsteps>
run() {
    local label=$1 envs=$2 dt=$3 ns=$4 t0 rc
    echo ">>> [$(date +%H:%M:%S)] $label: $BASE $envs  $N^3 dt=$dt NSTEPS=$ns"
    t0=$(date +%s)
    srun -N 1 -n 1 -c 128 --cpu-bind=none $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
        export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$NT:compact OMP_NUM_THREADS=$NT OMP_PLACES=cores \
               OMP_PROC_BIND=close KMP_BLOCKTIME=0 MKL_NUM_THREADS=$NT $BASE $envs; \
        exec mpirun -np $PPN $BIN $N $N $N $dt $ns $NT bench" > $OUT/$label.txt 2>&1
    rc=$?
    grep -E "$KEEP" $OUT/$label.txt | grep -v "^BLOWUP_HIST"
    grep "^BLOWUP_HIST" $OUT/$label.txt
    # exit 2 = BLOWUP or step cap (expected for some probes)
    echo "<<< $label exit=$rc ($([ $rc = 0 ] && echo ok || { [ $rc = 2 ] && echo 'BLOWUP / cap' || echo FAILED; })) elapsed $(( $(date +%s) - t0 )) s"
    echo
}

job_containers_init
case $MODE in
cfl|probe)
    if [ $MODE = cfl ]; then CS=${2:-"0.5 0.75 0.9"}; TE=20; CAP=20000; else CS=${2:-"1.0 1.1 1.25"}; TE=12; CAP=6000; fi
    for C in $CS; do run ${MODE}_C$C "NS_DT_MODE=cfl NS_CFL=$C NS_T_END=$TE" 0 $CAP; done ;;
ref)
    run ref "" 0.00125 16000 ;;
fixed)
    CJ=${2:?fixed needs the job id of the cfl job}; CS=${3:-"0.5 0.75 0.9"}
    for C in $CS; do
        f=$RES/$CJ/cfl_C$C.txt
        s=$(grep -m1 "^DTCFL_SUMMARY" $f 2>/dev/null)
        if [ -z "$s" ] || ! echo "$s" | grep -q "blowup=0 cap_hit=0"; then
            echo "ERROR: $f: no complete cfl run for C=$C (${s:-no DTCFL_SUMMARY}), fixed_C$C skipped"; echo; continue
        fi
        dtmin=$(echo "$s" | grep -oE "dt_min=[^ ]+" | cut -d= -f2)
        read ns dt < <(awk -v d=$dtmin 'BEGIN { n = int(20 / d); if (n * d < 20) n++; printf "%d %.17g\n", n, 20 / n }')
        echo "fixed_C$C: cfl job $CJ dt_min=$dtmin -> NSTEPS=$ns dt=$dt"
        run fixed_C$C "" $dt $ns
    done ;;
*) echo "ERROR: unknown mode $MODE (cfl|probe|ref|fixed)"; cleanup_all; exit 1 ;;
esac
echo "=============== files"; ls -l $OUT
cleanup_all
