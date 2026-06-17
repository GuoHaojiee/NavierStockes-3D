#!/bin/bash
#SBATCH --job-name=b2b_8n_heffte_bind_excn29_r1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn29
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/batch2_bind_excn29/b2b_8n_heffte_bind_excn29_r1_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/batch2_bind_excn29/b2b_8n_heffte_bind_excn29_r1_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/batch2_bind_excn29

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
EXEC=navier_stokes_heffte_gpuaware

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  b2b_8n_heffte_bind_excn29_r1   JobID=${SLURM_JOB_ID}"
echo "#  Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "#  Grid: ${NX}^3   steps=$NSTEPS   dt=$DT"
echo "#  Start: $(date '+%F %T')"
echo "#  Mode : heFFTe + HCA binding"
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
echo "#  b2b_8n_heffte_bind_excn29_r1 exit=$EXIT at $(date '+%F %T')"
exit $EXIT
