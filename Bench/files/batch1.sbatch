#!/bin/bash
#SBATCH --job-name=b1_sanity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/batch1/b1_rep%a_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/batch1/b1_rep%a_err_%j.log
#SBATCH --array=1-2

mkdir -p /scratch/s02230673/benchmark_results/batch1

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=10
REP=${SLURM_ARRAY_TASK_ID}

echo "############################################################"
echo "#  Batch 1 sanity baseline  -  Rep ${REP}/2"
echo "#  JobID : ${SLURM_JOB_ID}"
echo "#  Nodes : $SLURM_NNODES ($SLURM_NODELIST)"
echo "#  Grid  : ${NX}^3   steps=$NSTEPS   dt=$DT"
echo "#  Start : $(date '+%F %T')"
echo "############################################################"

# ============================================================
# Test 1: cuFFTmp
# ============================================================
echo ""
echo "============================================================"
echo "  [Rep ${REP}] Test 1/2: cuFFTmp   $(date '+%F %T')"
echo "============================================================"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         echo \"[\$(hostname)] rank=\${SLURM_PROCID} local=\${SLURM_LOCALID}\"
         export NVSHMEM_SYMMETRIC_SIZE=4294967296
         export NVSHMEM_REMOTE_TRANSPORT=ibrc
         export NVSHMEM_BOOTSTRAP=MPI
         export UCX_TLS=rc_verbs,ud_verbs,cuda_copy,cuda_ipc,sm,self
         export UCX_MEMTYPE_CACHE=n
         ./navier_stokes_cufftmp_multinode_docker $NX $NY $NZ $DT $NSTEPS
         
         if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
             echo '--- GPU memory snapshot (cuFFTmp) ---'
             nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
             echo '--- end snapshot ---'
         fi
     "

echo ""
echo "  [Rep ${REP}] cuFFTmp done at $(date '+%F %T')"
sleep 5

# ============================================================
# Test 2: heFFTe
# ============================================================
echo ""
echo "============================================================"
echo "  [Rep ${REP}] Test 2/2: heFFTe   $(date '+%F %T')"
echo "============================================================"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export UCX_TLS=rc_mlx5,dc_mlx5,ud_mlx5,cuda_copy,cuda_ipc,gdr_copy,sm,self
         unset UCX_MEMTYPE_CACHE
         unset UCX_RNDV_THRESH
         export LD_LIBRARY_PATH=/opt/heffte/lib:\$LD_LIBRARY_PATH
         
         if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
             echo '--- env check ---'
             ldd ./navier_stokes_heffte_gpuaware | grep -E 'heffte|cufft|mpi' | head -10
             echo 'UCX_TLS='\$UCX_TLS
             echo '--- end env check ---'
         fi
         
         ./navier_stokes_heffte_gpuaware $NX $NY $NZ $DT $NSTEPS
         
         if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
             echo '--- GPU memory snapshot (heFFTe) ---'
             nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
             echo '--- end snapshot ---'
         fi
     "

echo ""
echo "############################################################"
echo "#  Batch 1 Rep ${REP} DONE at $(date '+%F %T')"
echo "############################################################"
