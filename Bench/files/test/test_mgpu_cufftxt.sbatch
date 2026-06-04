#!/bin/bash
#SBATCH --job-name=test_mgpu_cufftxt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/s02230673/benchmark_results/gpu_test/mgpu_cufftxt_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/gpu_test/mgpu_cufftxt_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/gpu_test

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=512; NY=512; NZ=512
DT=1e-6; NSTEPS=10

echo "############################################################"
echo "#  test_mgpu_cufftxt   JobID=${SLURM_JOB_ID}"
echo "#  Node: $SLURM_NODELIST   Grid: ${NX}^3  steps=$NSTEPS"
echo "#  cuFFTXt: single process drives all visible GPUs via NVLink"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# cuFFTXt 单进程管多卡,不用 MPI;依赖 CUDA_VISIBLE_DEVICES 让进程看到 4 张卡
srun --nodes=1 --ntasks=1 --gres=gpu:4 \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
         ./ns_cufftxt $NX $NY $NZ $DT $NSTEPS
     "

EXIT=$?
echo ""
echo "#  test_mgpu_cufftxt exit=$EXIT at $(date '+%F %T')"
exit $EXIT
