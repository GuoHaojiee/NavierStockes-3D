#!/bin/bash
#============================================================
#  multinode_heffte_bind.sbatch
#  多节点多 GPU: heFFTe + 手动 HCA 绑定 (论文 8.3.2)
#  必须存在可执行的 bind_hca.sh 于 NS_DIR 下
#  修改 --nodes 即可切换 1/2/4/8/16 节点(每节点 8 GPU)
#  提交: sbatch multinode_heffte_bind.sbatch
#============================================================
#SBATCH --job-name=mn_heffte_bind
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/s02230673/benchmark_results/multinode/mn_heffte_bind_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/multinode/mn_heffte_bind_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/multinode

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
EXEC=navier_stokes_heffte_gpuaware

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=5

echo "############################################################"
echo "#  mn_heffte_bind   JobID=${SLURM_JOB_ID}"
echo "#  Nodes: $SLURM_NNODES ($SLURM_NODELIST)   GPUs total: $((SLURM_NNODES*8))"
echo "#  Grid: ${NX}^3   steps=$NSTEPS   dt=$DT"
echo "#  Mode : heFFTe + HCA binding (v2: with verbs fallback)"
echo "#  Start: $(date '+%F %T')"
echo "############################################################"

srun --mpi=pmix \
     --container-image=$CONTAINER \
     --container-mounts="$MOUNTS" \
     bash -lc '
         cd '"$NS_DIR"'
         # verbs fallback,避免大规模跨节点 mlx5 路由问题(原 v2 模板逻辑,保留)
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

         # HCA 绑定:按 SLURM_LOCALID 选最近网卡,然后 exec 可执行文件
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
echo "#  mn_heffte_bind exit=$EXIT at $(date '+%F %T')"
exit $EXIT
