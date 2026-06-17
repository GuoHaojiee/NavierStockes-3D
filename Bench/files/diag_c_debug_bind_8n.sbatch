#!/bin/bash
#SBATCH --job-name=diag_c_debug_bind_8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/diag/diag_c_debug_bind_8n_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/diag/diag_c_debug_bind_8n_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/diag

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
EXEC=navier_stokes_heffte_gpuaware

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  DIAG C: original bind_hca.sh + UCX_LOG_LEVEL=info"
echo "#  Expect to FAIL — purpose is to capture which transport UCX tried"
echo "#  JobID=${SLURM_JOB_ID}  Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "############################################################"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc '
         cd '"$NS_DIR"'
         export UCX_TLS=rc_mlx5,dc_mlx5,ud_mlx5,cuda_copy,cuda_ipc,gdr_copy,sm,self
         unset UCX_MEMTYPE_CACHE
         unset UCX_RNDV_THRESH
         export LD_LIBRARY_PATH=/opt/heffte/lib:$LD_LIBRARY_PATH

         # 打开 UCX info 级日志
         export UCX_LOG_LEVEL=info
         # 每个进程独立 log 文件, 避免合并 stderr 互相覆盖
         export UCX_LOG_FILE=/home/user/benchmark_results/diag/ucx_${SLURM_JOB_ID}_r${SLURM_PROCID}.log

         if [ "$SLURM_PROCID" = "0" ]; then
             echo "--- env check ---"
             echo "UCX_TLS=$UCX_TLS"
             echo "UCX_LOG_LEVEL=$UCX_LOG_LEVEL"
             echo "UCX_LOG_FILE pattern=/home/user/benchmark_results/diag/ucx_${SLURM_JOB_ID}_r*.log"
             echo "--- end env check ---"
         fi

         ./bind_hca.sh ./'"$EXEC"' '"$NX $NY $NZ $DT $NSTEPS"'
         exit $?
     '

EXIT=$?
echo ""
echo "#  DIAG C exit=$EXIT at $(date '+%F %T')"
exit $EXIT
