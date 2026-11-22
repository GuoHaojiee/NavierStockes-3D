#!/bin/bash
#SBATCH -J fftb_build
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:40:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fftb_build_%x_%j.out
# build_fftb.sbatch <gcc|intel> — scripts/fftbench/build_fftb.sh as a batch job (no interactive srun), so that the
#   benchmark jobs can depend on it: sbatch --dependency=afterok:<this job> ...
#   The job's exit code is build_fftb.sh's: 0 only if the log ends with "BUILD OK" (every expected binary present and
#   built from this source); otherwise non-zero and afterok jobs do not start (they stay DependencyNeverSatisfied:
#   scancel them).
#   gcc   -> GCC container ($M),            /work/build/bin_fftb/gcc
#   intel -> Intel container, image v2 ($MI with INTEL_IMAGE=v2, the image the runs use), /work/build/bin_fftb/intel
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/fftbench && sbatch -J fftb_build_intel build_fftb.sbatch intel
MODE=${1:?usage: sbatch build_fftb.sbatch gcc|intel}
export INTEL_IMAGE=${FFTB_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
case $MODE in gcc) IMG=$M;; intel) IMG=$MI;; *) echo "usage: build_fftb.sbatch gcc|intel"; exit 2;; esac
[ -n "$IMG" ] || { echo "ERROR: no container options for $MODE"; exit 2; }
echo "build_fftb $MODE  commit $(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null)  container: $IMG"
srun -N1 -n1 -c16 --time=35 $IMG bash /work/src/NavierStockes-3D/scripts/fftbench/build_fftb.sh $MODE
rc=$?
srun -N1 -n1 --time=2 bash -c "for n in \$(enroot list 2>/dev/null | grep -E '_j${SLURM_JOB_ID}\$'); do enroot remove -f \$n >/dev/null 2>&1; done; true" 2>/dev/null
echo "build_fftb $MODE exit code $rc"
exit $rc
