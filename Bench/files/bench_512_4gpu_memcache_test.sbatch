#!/bin/bash
#SBATCH --job-name=h512_4_memtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/h512_4_memtest_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/h512_4_memtest_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

DT=1e-6

echo "Start: $(date '+%F %T')   Node: $SLURM_NODELIST"

# ============================================================
# Run 0: UCX transport probe (256^3, 2 steps)
# ============================================================
echo ""
echo "############################################################"
echo "##  Run 0: UCX transport probe (256^3, 2 steps)"
echo "############################################################"
srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export LD_LIBRARY_PATH=/opt/heffte/lib:\$LD_LIBRARY_PATH
         export UCX_TLS=cuda_copy,cuda_ipc,sm,self
         export UCX_MEMTYPE_CACHE=n
         export UCX_LOG_LEVEL=info
         export UCX_PROTO_INFO=y
         if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
             echo '--- env check ---'
             ldd ./ns_heffte_mgpu | grep -E 'heffte|cufft|mpi' | head -5
             echo 'UCX_TLS='\$UCX_TLS
             echo 'UCX_MEMTYPE_CACHE='\${UCX_MEMTYPE_CACHE:-<unset>}
             echo '--- end env check ---'
         fi
         ./ns_heffte_mgpu 256 256 256 $DT 2
     "

# ============================================================
# Run A: baseline UCX_MEMTYPE_CACHE=n  (512^3, 10 steps)
# ============================================================
echo ""
echo "############################################################"
echo "##  Run A: baseline UCX_MEMTYPE_CACHE=n  (512^3, 10 steps)"
echo "############################################################"
srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export LD_LIBRARY_PATH=/opt/heffte/lib:\$LD_LIBRARY_PATH
         export UCX_TLS=cuda_copy,cuda_ipc,sm,self
         export UCX_MEMTYPE_CACHE=n
         ./ns_heffte_mgpu 512 512 512 $DT 10
     "

# ============================================================
# Run B: UCX_MEMTYPE_CACHE default (enabled)  — single variable change
# ============================================================
echo ""
echo "############################################################"
echo "##  Run B: UCX_MEMTYPE_CACHE default (enabled)  (512^3, 10 steps)"
echo "############################################################"
srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc "
         cd $NS_DIR
         export LD_LIBRARY_PATH=/opt/heffte/lib:\$LD_LIBRARY_PATH
         export UCX_TLS=cuda_copy,cuda_ipc,sm,self
         unset UCX_MEMTYPE_CACHE
         ./ns_heffte_mgpu 1024 1024 1024 $DT 2
     "

echo ""
echo "End: $(date '+%F %T')"
