#!/bin/bash
#SBATCH --job-name=hyb_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/s02230673/benchmark_results/hyb_smoke_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/hyb_smoke_err_%j.log

CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
HPCX=/home/user/.local/nvhpc/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19

ENV="\
export OPAL_PREFIX=${HPCX}/ompi; \
export PMIX_INSTALL_PREFIX=\$OPAL_PREFIX; \
export LD_LIBRARY_PATH=${HPCX}/ompi/lib:${HPCX}/ucx/lib:${HPCX}/ucc/lib:${HPCX}/hcoll/lib:${HPCX}/sharp/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:/home/user/.local/lib:/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH; \
unset UCX_NET_DEVICES UCX_IB_REG_METHODS UCX_RNDV_THRESH UCX_MAX_RNDV_RAILS UCX_ASYNC_MAX_EVENTS UCX_UNIFIED_MODE UCX_WARN_UNUSED_ENV_VARS; \
export UCX_TLS=sm,self,posix,sysv; \
export OMPI_MCA_pml=ucx; \
export OMPI_MCA_osc=ucx; \
export OMPI_MCA_coll=^hcoll; \
export OMPI_MCA_mtl=^ofi; \
export OMP_NUM_THREADS=2; \
unset OMP_PROC_BIND OMP_PLACES; \
export OMP_DISPLAY_ENV=true"

echo "===== Test A: SLURM default cpuset, OMP unbound ====="
echo "    --cpus-per-task=2 让 SLURM 给每 rank 一个 2-CPU cpuset"
echo "    OMP 没有绑定指令，由 OS 在 cpuset 内调度"
echo ""
srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -c "$ENV; cd $NS_DIR && exec ./ns_accfft_hpcx 128 128 128 1e-5 5 2"
