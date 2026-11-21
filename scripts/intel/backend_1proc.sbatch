#!/bin/bash
#SBATCH -J be1
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 01:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/be1_%j.out
# backend_1proc — local FFT backends without communication (supervisor: compare heFFTe-FFTW and heFFTe-MKL
#   on one process). One MPI rank, one thread, pinned to one core; the rest of the node is idle.
#   256^3 and 512^3 r2c forward + c2r backward (scaled), out of place, timing window >= 2 s, 5 repetitions,
#   median. Methods in every binary: heFFTe (<backend>), native FFTW MEASURE, native FFTW ESTIMATE, native MKL DFTI.
#   fb_backend_fftw = heFFTe built with FFTW, fb_backend_mkl = heFFTe built with MKL. The native rows appear
#   in both binaries: they must agree (consistency check of the node / of the two runs).
#   Source: bench_fft/fftbench_backend.cpp; built by build_intel_solvers.sh (Intel container, icx FFTW, MKL).
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/backend_1proc.sbatch
#   bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/backend_parse.sh $(ls -t ~/guohaojie/logs/be1_*.out | head -1)
# Options: BE_SIZES ("256 512"), BE_REPS (5), BE_WINDOW (2), BE_THREADS (1), BE_R2C0 (1: extra heFFTe rows, r2c along dim 0)
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
IBIN=/work/build/bin_intel
SIZES=${BE_SIZES:-"256 512"}; REPS=${BE_REPS:-5}; WIN=${BE_WINDOW:-2}; NT=${BE_THREADS:-1}
for n in $SIZES; do
  for b in fb_backend_fftw fb_backend_mkl; do
    echo ">>> [$(date +%H:%M:%S)] $b N=$n reps=$REPS window=$WIN threads=$NT"
    srun -N 1 -n 1 -c 128 --cpu-bind=none --time=25 $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
        export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$NT:compact OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT \
               OMP_PLACES=cores OMP_PROC_BIND=close KMP_BLOCKTIME=0; exec mpirun -np 1 $IBIN/$b $n $REPS $WIN $NT"
    echo "<<< exit=$?"; echo
  done
done
# extra rows: heFFTe with the r2c direction along dim 0 (contiguous in heFFTe's layout) instead of the
# solver's dim 2 — shows how much of a heFFTe-vs-native gap is the strided r2c direction
if [ "${BE_R2C0:-1}" = 1 ]; then
  for n in $SIZES; do
    for b in fb_backend_fftw fb_backend_mkl; do
      echo ">>> [$(date +%H:%M:%S)] $b N=$n FB_R2C_DIR=0 heffte only"
      srun -N 1 -n 1 -c 128 --cpu-bind=none --time=10 $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
          export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$NT:compact OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT \
                 OMP_PLACES=cores OMP_PROC_BIND=close KMP_BLOCKTIME=0 FB_R2C_DIR=0 FB_METHODS=heffte; exec mpirun -np 1 $IBIN/$b $n $REPS $WIN $NT"
      echo "<<< exit=$?"; echo
    done
  done
fi
cleanup_all
