#!/bin/bash
#SBATCH --job-name=test_1gpu_heffte
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/s02230673/benchmark_results/gpu_test/1gpu_heffte_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/gpu_test/1gpu_heffte_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/gpu_test

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=256; NY=256; NZ=256
DT=1e-6; NSTEPS=10

echo "############################################################"
echo "#  test_1gpu_heffte   JobID=${SLURM_JOB_ID}"
echo "#  Node: $SLURM_NODELIST   Grid: ${NX}^3  steps=$NSTEPS"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# heFFTe 单 GPU 走 MPI(1 个 rank),节点内只用 GPU 直连传输
srun --mpi=pmix \
     --nodes=1 --ntasks=1 --gres=gpu:1 \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export UCX_TLS=cuda_copy,cuda_ipc,sm,self
         export UCX_MEMTYPE_CACHE=n
         ./ns_heffte_gpu1 $NX $NY $NZ $DT $NSTEPS
     "

EXIT=$?
echo ""
echo "#  test_1gpu_heffte exit=$EXIT at $(date '+%F %T')"
exit $EXIT
