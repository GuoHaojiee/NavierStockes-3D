#!/bin/bash
#SBATCH -J fftb_tc
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fftb_%x_%j.out
# fftb_tc <nodes 1|2|4> <grid> — toolchain comparison (GCC + Open MPI vs Intel + Intel MPI) with the pure FFT benchmark
#   (the solvers' step times of P3DFFT / heFFTe contain non-FFT work that is not usable for this comparison).
#   GCC  : /work/build/bin_fftb/gcc,   GCC_COLL ("def"; list allowed, e.g. "def t4"), heFFTe = FFTW backend
#          (the only one on the GCC side)
#   Intel: /work/build/bin_fftb/intel, FFTB_INTEL_SETS (default E6d for 2/4 nodes, single-node for 1 node, as the solver
#          comparison cmp_toolchain_mn), heFFTe = MKL backend + NS_HEFFTE_REORDER=1
#   FFTW, AccFFT, P3DFFT, heFFTe; slab; MEASURE; layouts FFTB_LAYOUTS ("128x1"); FFTB_SPLIT=1 (t_local, t_fft - t_local);
#   3 repetitions per launch; order: library, then the toolchains alternate (gcc settings, intel settings) per layout.
#   Results: ~/guohaojie/results/fftb.csv. All needed binaries are checked first (exit 2 if one is missing); the job
#   exits 3 if any run is not OK (fftb.sbatch).
# Time estimate for 1024^3 (per launch: container start, MEASURE planning, check, warm-up, 3 x >= 3 s, local model):
#   1 node ~4-6 min, 4 nodes ~2-3 min; 4 libraries x 2 toolchains = 8 launches per layout and GCC_COLL / Intel setting:
#   1 node ~35-50 min (-t 02:00:00), 4 nodes ~20-30 min (-t 01:30:00). Per launch time limit 3 x expected (10 / 4 min).
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/fftbench
#   sbatch -N 1 -J fftb_tc_1024_1 -t 02:00:00 fftb_tc.sbatch 1 1024
#   sbatch -N 4 -J fftb_tc_1024_4 -t 01:30:00 fftb_tc.sbatch 4 1024
#   FFTB_JOBS="<job1> <job4>" bash fftb_parse.sh      (sections 2 and 2b)
NODES=${1:?usage: sbatch -N <n> fftb_tc.sbatch <1|2|4> <grid>}
GRID=${2:?usage: sbatch -N <n> fftb_tc.sbatch <1|2|4> <grid>}
export FFTB_TC="gcc intel"
export FFTB_LIBS=${FFTB_LIBS:-"fftw accfft p3dfft heffte"}
export FFTB_LAYOUTS=${FFTB_LAYOUTS:-"128x1"}
export FFTB_SPLIT=${FFTB_SPLIT:-1}
export FFTB_REP=${FFTB_REP:-3}
export FFTB_DECOMP=slab FFTB_PLAN=${FFTB_PLAN:-measure}
export GCC_COLL=${GCC_COLL:-def}
unset FFTB_CONFIGS
[[ $GRID =~ ^[0-9]+$ ]] || { echo "ERROR: grid '$GRID' (one number)"; exit 1; }
for L in $FFTB_LAYOUTS; do
    p=${L%x*}; t=${L#*x}
    [ $((p * t)) = 128 ] || echo "WARNING: layout $L uses $((p * t)) of 128 cores per node"
    [ $((NODES * p)) -le $GRID ] || { echo "ERROR: layout $L: $((NODES * p)) ranks > $GRID (slab needs ranks <= N)"; exit 1; }
done
exec bash ~/guohaojie/src/NavierStockes-3D/scripts/fftbench/fftb.sbatch $NODES $GRID
