#!/bin/bash
#SBATCH -J fftb_scale
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 04:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fftb_%x_%j.out
# fftb_scale <nodes 1|2|4> <grid> — scaling scan with the pure FFT benchmark, Intel toolchain only:
#   FFTW, AccFFT, P3DFFT, heFFTe (MKL backend, NS_HEFFTE_REORDER=1); slab; MEASURE;
#   layouts per node FFTB_LAYOUTS ("128x1 64x2 32x4 16x8"; threads = 128 / ranks per node);
#   FFTB_SPLIT=1: local FFT time t_local and communication t_fft - t_local; 3 repetitions per launch -> fftb.csv.
#   The job must have that many nodes: sbatch -N <nodes> -J fftb_scale_<grid>_<nodes> ... fftb_scale.sbatch <nodes> <grid>
# Threads: OMP_NUM_THREADS = threads, OMP_PLACES=cores, OMP_PROC_BIND=close, KMP_BLOCKTIME=0, MKL_NUM_THREADS = threads.
#   1 node : one srun -n1 -c128 --cpu-bind=none, Intel mpirun inside the image pins with I_MPI_PIN_DOMAIN=<threads>:compact
#            (mn_lib.sh mn_intel), the OpenMP threads stay inside their rank's domain (close).
#   2/4 nodes: irun_mn, srun --cpu-bind=cores -c <threads> gives every rank <threads> cores.
#   Every launch prints FFTB_BIND (fftbench.cpp): ranks of a node must share no CPU and every thread must sit on its own
#   physical core; the result (OK | WARN) is the CSV bind column and fftb_parse.sh lists every WARN.
#   FFTB_SHOWBIND=1 also prints the CPU lists of ranks 0 and 1.
# Order: library, then layout (the four layouts of one library run back to back).
# Settings name in the CSV: single-node (1 node) / E6d (2, 4 nodes) as in toolchain_mn.csv; FFTB_INTEL_SETS to change.
# Time estimate for 1024^3 (per launch: container start, MEASURE planning, check, warm-up, 3 x >= 3 s, local model):
#   1 node ~4-6 min, 2 nodes ~3-4 min, 4 nodes ~2-3 min; 16 launches per job (4 libraries x 4 layouts):
#   1 node ~1.1-1.6 h (-t 04:00:00), 2 nodes ~0.8-1.1 h (-t 03:00:00), 4 nodes ~0.5-0.8 h (-t 02:00:00).
#   Time limit per launch: FFTB_TIMEOUT_FACTOR (3) x expected (1024^3: 10 / 6 / 4 min).
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/fftbench
#   sbatch -N 1 -J fftb_scale_1024_1 -t 04:00:00 fftb_scale.sbatch 1 1024
#   sbatch -N 2 -J fftb_scale_1024_2 -t 03:00:00 fftb_scale.sbatch 2 1024
#   sbatch -N 4 -J fftb_scale_1024_4 -t 02:00:00 fftb_scale.sbatch 4 1024
#   FFTB_JOBS="<job1> <job2> <job4>" bash fftb_parse.sh
NODES=${1:?usage: sbatch -N <n> fftb_scale.sbatch <1|2|4> <grid>}
GRID=${2:?usage: sbatch -N <n> fftb_scale.sbatch <1|2|4> <grid>}
export FFTB_TC=intel
export FFTB_LIBS=${FFTB_LIBS:-"fftw accfft p3dfft heffte"}
export FFTB_LAYOUTS=${FFTB_LAYOUTS:-"128x1 64x2 32x4 16x8"}
export FFTB_SPLIT=${FFTB_SPLIT:-1}
export FFTB_REP=${FFTB_REP:-3}
export FFTB_DECOMP=slab FFTB_PLAN=${FFTB_PLAN:-measure}
unset FFTB_CONFIGS
for L in $FFTB_LAYOUTS; do
    p=${L%x*}; t=${L#*x}
    [ $((p * t)) = 128 ] || echo "WARNING: layout $L uses $((p * t)) of 128 cores per node"
    [ $((NODES * p)) -le $GRID ] || { echo "ERROR: layout $L: $((NODES * p)) ranks > $GRID (slab needs ranks <= N)"; exit 1; }
done
exec bash ~/guohaojie/src/NavierStockes-3D/scripts/fftbench/fftb.sbatch $NODES $GRID
