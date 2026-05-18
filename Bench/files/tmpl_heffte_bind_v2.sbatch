#!/bin/bash
#SBATCH --job-name=__JOBNAME__
#SBATCH --nodes=__NNODES__
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/batch2_bind_v2/__JOBNAME___%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/batch2_bind_v2/__JOBNAME___err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/batch2_bind_v2

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
EXEC=navier_stokes_heffte_gpuaware

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  __JOBNAME__   JobID=${SLURM_JOB_ID}"
echo "#  Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "#  Grid: ${NX}^3   steps=$NSTEPS   dt=$DT"
echo "#  Start: $(date '+%F %T')"
echo "#  Mode : heFFTe + HCA binding (v2: with verbs fallback)"
echo "############################################################"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc '
         cd '"$NS_DIR"'
         # 加 rc_verbs/ud_verbs 作为 fallback，避免大规模跨节点 mlx5 路由问题
         export UCX_TLS=rc_mlx5,dc_mlx5,ud_mlx5,rc_verbs,ud_verbs,cuda_copy,cuda_ipc,gdr_copy,sm,self
         unset UCX_MEMTYPE_CACHE
         unset UCX_RNDV_THRESH
         export LD_LIBRARY_PATH=/opt/heffte/lib:$LD_LIBRARY_PATH
         
         if [ "$SLURM_PROCID" = "0" ]; then
             echo "--- env check ---"
             ldd ./'"$EXEC"' | grep -E "heffte|cufft|mpi" | head -10
             echo "UCX_TLS=$UCX_TLS"
             echo "--- end env check ---"
         fi
         
         ./bind_hca.sh ./'"$EXEC"' '"$NX $NY $NZ $DT $NSTEPS"'
         
         RC=$?
         if [ "$SLURM_PROCID" = "0" ]; then
             echo "--- GPU memory ---"
             nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
         fi
         exit $RC
     '

EXIT=$?
echo ""
echo "#  __JOBNAME__ exit=$EXIT at $(date '+%F %T')"
exit $EXIT
