#!/bin/bash
#SBATCH --job-name=test_mgpu_heffte
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/s02230673/benchmark_results/gpu_test/mgpu_heffte_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/gpu_test/mgpu_heffte_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/gpu_test

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=512; NY=512; NZ=512
DT=1e-6; NSTEPS=10
NGPU=4

echo "############################################################"
echo "#  test_mgpu_heffte   JobID=${SLURM_JOB_ID}"
echo "#  Node: $SLURM_NODELIST   Grid: ${NX}^3  steps=$NSTEPS  GPUs=$NGPU"
echo "#  heFFTe: one MPI rank per GPU, intra-node NVLink (CUDA-IPC)"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# 单节点多 GPU heFFTe:每 GPU 一个 MPI rank,节点内走 NVLink(cuda_ipc)
# 单节点不跨网络,所以不需要 HCA 绑定
srun --mpi=pmix \
     --nodes=1 --ntasks=${NGPU} --ntasks-per-node=${NGPU} --gres=gpu:${NGPU} \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export UCX_TLS=cuda_copy,cuda_ipc,sm,self
         export UCX_MEMTYPE_CACHE=n
         export LD_LIBRARY_PATH=/opt/heffte/lib:\$LD_LIBRARY_PATH
         ./ns_heffte_mgpu $NX $NY $NZ $DT $NSTEPS
     "

EXIT=$?
echo ""
echo "#  test_mgpu_heffte exit=$EXIT at $(date '+%F %T')"
exit $EXIT
