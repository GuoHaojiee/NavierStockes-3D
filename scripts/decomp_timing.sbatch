#!/bin/bash
#SBATCH -J dt
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 01:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/dt_%x_%j.out
# decomp_timing — time per step of every CPU library x decomposition (NS_DECOMP), one node, p <= N.
#   Default: 512^3, dt=1e-6, 20 steps, bench, 128x1, NS_FFTW_PLAN=measure (heFFTe: ESTIMATE, hard-coded).
#   Reference kinetic energy (512^3, 20 steps): 1.7933833444e+08.
#   3 repetitions, repetitions are the OUTER loop: a slow period of the node hits every configuration equally.
#   Ratios between configurations should only be taken inside one job.
# Binaries: /work/build/bin_gcc from scripts/compile_all.sh (override with DT_BIN=...).
# Options (environment at sbatch time): DT_N (grid, default 512), DT_STEPS (20), DT_REP (3), DT_THREADS (1)
# Submit:
#   cd ~/guohaojie/src/NavierStockes-3D/scripts && sbatch decomp_timing.sbatch
# Results (same parser as the planner study):
#   bash ../planner/pl_parse.sh $(ls -t /scratch/pr0208/guohaojie/logs/dt_dt_*.out | head -1)
#   For another grid, pass the reference energy: PL_EREF=<value> bash ../planner/pl_parse.sh <log>
source ~/guohaojie/runs/common.sh          # $M, precheck, cleanup
precheck
BIN=${DT_BIN:-/work/build/bin_gcc}
N=${DT_N:-512}; DT=1e-6; NS=${DT_STEPS:-20}; REP=${DT_REP:-3}; NT=${DT_THREADS:-1}
NP=$((128 / NT))

# R <label> <NS_DECOMP or -> <program>
R() {
    local label=$1 d=$2 prog=$3 envs="NS_FFTW_PLAN=measure"
    [ "$d" != "-" ] && envs="$envs NS_DECOMP=$d"
    echo ">>> [$(date +%H:%M:%S)] ${label}_${NP}x${NT} ranks=$NP threads=$NT plan=measure | NS_DECOMP=$d $prog $N $N $N $DT $NS $NT bench"
    srun --mpi=pmix --cpu-bind=cores --time=20 -N 1 -n $NP -c $NT $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=$NT OMP_PLACES=cores OMP_PROC_BIND=close $envs; exec $BIN/$prog $N $N $N $DT $NS $NT bench" 2>&1 \
        | grep -E "^decomposition:|process grid|MPI reshapes|planner flag|planning time|plan time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror"
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}

for rep in $(seq 1 $REP); do
  echo "=============== repetition $rep"
  R FFTW_slab          -      ns_fftw
  R AccFFT_slab        slab   ns_accfft
  R AccFFT_pencil      pencil ns_accfft
  R P3DFFT_slab        slab   ns_p3dfft
  R P3DFFT_pencil      pencil ns_p3dfft
  R heFFTe_slab        slab   ns_heffte_v2
  R heFFTe_pencil      pencil ns_heffte_v2
  R heFFTe_brick       brick  ns_heffte_v2
done
cleanup
