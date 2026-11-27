#!/bin/bash
#SBATCH -J tgval
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/tgval_%j.out
# tg_validate — physical check of the FFTW CPU solver before the dtcfl experiments:
#   Taylor–Green vortex, Re = 1600 (NS_NU=0.000625), NS_NYQ0=1, 256^3, fixed dt = 0.005, 2400 steps (T = 12), bench (f = 0),
#   "DT" line every 20 steps (0.1 time units): t, E = 0.5<|u|^2>, eps = nu<|w|^2>.
#   Compared with the 512^3 spectral reference (scripts/dtcfl/ref/, van Rees et al. 2011) by tg_validate_parse.sh.
#   No dealiasing: at Re = 1600 the grid is under-resolved, deviations from the reference are expected to come partly from that.
# Intel toolchain (v2 image, /work/build/bin_intel/ns_fftw from scripts/intel/build_intel_solvers.sh), 1 node,
# FFTW 64x2 (64 ranks x 2 threads, slab), NS_FFTW_PLAN=measure; launched like the 1-node runs of cpu_final.sbatch
# (one srun task, mpirun inside the image, fork + shm).
# Log: ~/guohaojie/logs/tgval_<jobid>.out (= /scratch/pr0208/guohaojie/logs).  Expected: MEASURE planning ~1 min,
# 2400 steps x ~0.2 s -> ~10 min.
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/dtcfl
#   sbatch --parsable tg_validate.sbatch | tee -a ~/guohaojie/logs/ids_1122.txt
#   bash tg_validate_parse.sh ~/guohaojie/logs/tgval_<jobid>.out [local partial run]
# Options: TGV_IMAGE (v2), TGV_N (256), TGV_DT (0.005), TGV_STEPS (2400), TGV_PRINT (20), TGV_LAYOUT (64x2).
export INTEL_IMAGE=${TGV_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
N=${TGV_N:-256}; DT=${TGV_DT:-0.005}; NS=${TGV_STEPS:-2400}; PR=${TGV_PRINT:-20}; L=${TGV_LAYOUT:-64x2}
PPN=${L%x*}; NT=${L#*x}
BIN=/work/build/bin_intel/ns_fftw
HBIN=$SCR/${BIN#/work/}
ENVS="NS_DECOMP=slab NS_FFTW_PLAN=measure NS_NYQ0=1 NS_IC=tg NS_NU=0.000625 NS_DT_PRINT=$PR"
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)

echo "tg_validate: job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, commit $COMMIT, image $INTEL_IMAGE"
echo "  $N^3 dt=$DT steps=$NS (T=$(awk -v a=$DT -v b=$NS 'BEGIN{print a*b}')) layout ${PPN}x$NT  $ENVS"
echo "  binary $BIN  $(ls -l --time-style=+%F_%R $HBIN 2>&1 | awk '{print $6}')"
# the binary must contain the NS_NYQ0 / NS_IC code (commit a8c98bf or later), otherwise it ignores the variables
if ! grep -aq NS_NYQ0 $HBIN; then
    echo "ERROR: $HBIN has no NS_NYQ0 support: git pull and rebuild (scripts/intel/build_intel_solvers.sh)"; exit 1
fi
job_containers_init

t0=$(date +%s)
srun -N 1 -n 1 -c 128 --cpu-bind=none --time=27 $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
    export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$NT:compact OMP_NUM_THREADS=$NT OMP_PLACES=cores \
           OMP_PROC_BIND=close KMP_BLOCKTIME=0 MKL_NUM_THREADS=$NT $ENVS; \
    exec mpirun -np $PPN $BIN $N $N $N $DT $NS $NT bench"
rc=$?
echo "<<< exit=$rc  elapsed $(( $(date +%s) - t0 )) s"
cleanup_all
