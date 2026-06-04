#!/bin/bash
#SBATCH --job-name=test_1gpu_cufft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/s02230673/benchmark_results/gpu_test/1gpu_cufft_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/gpu_test/1gpu_cufft_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/gpu_test

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=256; NY=256; NZ=256
DT=1e-6; NSTEPS=10

echo "############################################################"
echo "#  test_1gpu_cufft   JobID=${SLURM_JOB_ID}"
echo "#  Node: $SLURM_NODELIST   Grid: ${NX}^3  steps=$NSTEPS"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# cuFFT 单 GPU 不需要 MPI
srun --nodes=1 --ntasks=1 --gres=gpu:1 \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         ./ns_cufft $NX $NY $NZ $DT $NSTEPS
     "

EXIT=$?
echo ""
echo "#  test_1gpu_cufft exit=$EXIT at $(date '+%F %T')"
exit $EXIT
