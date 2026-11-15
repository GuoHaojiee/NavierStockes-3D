#!/bin/bash
#SBATCH -J pl
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 02:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/pl_%x_%j.out
# pl_scan — is FFTW_PATIENT really faster than MEASURE / ESTIMATE in the CPU solvers?
#   512^3, bench mode, 20 steps (reference kinetic energy 1.7933833444e+08), one node, 3 repetitions.
#   Repetitions are the OUTER loop, so a slow period of the node hits every flag equally.
#   Every run re-plans, so the spread also contains the plan-to-plan variation of MEASURE/PATIENT.
# One job per library; the three run in parallel on three nodes:
#   cd ~/guohaojie/src/NavierStockes-3D/planner
#   sbatch -J pl_fftw pl_scan.sbatch fftw; sbatch -J pl_accfft pl_scan.sbatch accfft; sbatch -J pl_p3dfft pl_scan.sbatch p3dfft
# Results:
#   bash pl_parse.sh $(ls -t ~/guohaojie/logs/pl_pl_fftw_*.out | head -1)      (same for accfft, p3dfft)
source ~/guohaojie/runs/common.sh          # $M, precheck, cleanup
precheck
BIN=/work/build/bin_pl
LIB=${1:?usage: sbatch pl_scan.sbatch fftw|accfft|p3dfft}
N=${PL_N:-512}; DT=1e-6; NS=${PL_STEPS:-20}; REP=${PL_REP:-3}

# R <label> <ranks> <threads> <NS_FFTW_PLAN or -> <program>
R() {
    local label=$1 np=$2 nt=$3 fl=$4 prog=$5 rc envs=""
    [ "$fl" != "-" ] && envs="NS_FFTW_PLAN=$fl"
    echo ">>> [$(date +%H:%M:%S)] $label ranks=$np threads=$nt plan=$fl | $prog $N $N $N $DT $NS $nt bench"
    srun --mpi=pmix --cpu-bind=cores --time=20 -N 1 -n $np -c $nt $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close $envs; exec $BIN/$prog $N $N $N $DT $NS $nt bench"
    rc=$?; echo "<<< exit=$rc"; echo
}

for rep in $(seq 1 $REP); do
  echo "=============== repetition $rep"
  case $LIB in
  fftw)
    for fl in estimate measure patient; do R "FFTW_64x2"  64 2 $fl ns_fftw; done
    for fl in estimate measure patient; do R "FFTW_128x1" 128 1 $fl ns_fftw; done ;;
  accfft)
    for fl in estimate measure patient; do R "AccFFT_patched_128x1" 128 1 $fl ns_accfft_pt; done
    R "AccFFT_stock_128x1" 128 1 patient ns_accfft ;;          # stock library: PATIENT is really MEASURE
  p3dfft)
    R "P3DFFT_container_128x1" 128 1 - ns_p3dfft_orig
    for fl in estimate measure patient; do R "P3DFFT_${fl}_128x1" 128 1 - ns_p3dfft_$fl; done ;;
  *) echo "unknown library $LIB"; break ;;
  esac
done
cleanup
