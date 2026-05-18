#!/bin/bash
#SBATCH --job-name=diag_a_no_bind_8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/diag/diag_a_no_bind_8n_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/diag/diag_a_no_bind_8n_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/diag

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
EXEC=navier_stokes_heffte_gpuaware

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  DIAG A: NO HCA BIND (UCX auto-select)"
echo "#  JobID=${SLURM_JOB_ID}  Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc '
         cd '"$NS_DIR"'
         export UCX_TLS=rc_mlx5,dc_mlx5,ud_mlx5,cuda_copy,cuda_ipc,gdr_copy,sm,self
         unset UCX_MEMTYPE_CACHE
         unset UCX_RNDV_THRESH
         unset UCX_NET_DEVICES
         unset HCOLL_MAIN_IB
         export LD_LIBRARY_PATH=/opt/heffte/lib:$LD_LIBRARY_PATH

         if [ "$SLURM_PROCID" = "0" ]; then
             echo "--- env check ---"
             echo "UCX_TLS=$UCX_TLS"
             echo "UCX_NET_DEVICES=${UCX_NET_DEVICES:-<unset>}"
             echo "HCOLL_MAIN_IB=${HCOLL_MAIN_IB:-<unset>}"
             echo "--- end env check ---"
         fi

         ./'"$EXEC"' '"$NX $NY $NZ $DT $NSTEPS"'
         exit $?
     '

EXIT=$?
echo ""
echo "#  DIAG A exit=$EXIT at $(date '+%F %T')"
exit $EXIT
