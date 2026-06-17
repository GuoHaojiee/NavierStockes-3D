#!/bin/bash
#SBATCH --job-name=b2_2n_cufftmp_r1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/batch2/b2_2n_cufftmp_r1_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/batch2/b2_2n_cufftmp_r1_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/batch2

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  b2_2n_cufftmp_r1   JobID=${SLURM_JOB_ID}"
echo "#  Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
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
         ./navier_stokes_cufftmp_multinode_docker $NX $NY $NZ $DT $NSTEPS
         if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
             echo '--- GPU memory ---'
             nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
         fi
     "

EXIT=$?
echo ""
echo "#  b2_2n_cufftmp_r1 exit=$EXIT at $(date '+%F %T')"
exit $EXIT
