#!/bin/bash
#============================================================
#  multinode_cufftmp.sbatch
#  多节点多 GPU: cuFFTMp (NVSHMEM)
#  修改 --nodes 即可切换 1/2/4/8/16 节点(每节点 8 GPU)
#  提交: sbatch multinode_cufftmp.sbatch
#============================================================
#SBATCH --job-name=mn_cufftmp
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/multinode/mn_cufftmp_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/multinode/mn_cufftmp_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/multinode

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
EXEC=navier_stokes_cufftmp_multinode_docker

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  mn_cufftmp   JobID=${SLURM_JOB_ID}"
echo "#  Nodes: $SLURM_NNODES ($SLURM_NODELIST)   GPUs total: $((SLURM_NNODES*8))"
echo "#  Grid: ${NX}^3   steps=$NSTEPS   dt=$DT"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export NVSHMEM_SYMMETRIC_SIZE=1073741824
         export NVSHMEM_REMOTE_TRANSPORT=ibrc
         export NVSHMEM_BOOTSTRAP=MPI
         export UCX_TLS=rc_verbs,ud_verbs,cuda_copy,cuda_ipc,sm,self
         export UCX_MEMTYPE_CACHE=n
         ./$EXEC $NX $NY $NZ $DT $NSTEPS
         if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
             echo '--- GPU memory ---'
             nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
         fi
     "

EXIT=$?
echo ""
echo "#  mn_cufftmp exit=$EXIT at $(date '+%F %T')"
exit $EXIT
