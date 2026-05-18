#!/bin/bash
#SBATCH --job-name=b2a_scale_v2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/batch2/b2a_v2_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/batch2/b2a_v2_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results/batch2

CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

NX=1024; NY=1024; NZ=1024
DT=1e-6; NSTEPS=10

echo "############################################################"
echo "#  Batch 2a v2: Strong scaling  -  1 + 2 nodes  (with --exact)"
echo "#  JobID : ${SLURM_JOB_ID}"
echo "#  Allocated: $SLURM_NNODES nodes ($SLURM_NODELIST)"
echo "#  Start : $(date '+%F %T')"
echo "############################################################"

CONFIGS=("1:8:8" "2:8:16")

run_cufftmp() {
    local NNODES=$1
    local TASKS_PER_NODE=$2
    local TOTAL_TASKS=$3
    local REP=$4
    
    echo ""
    echo "============================================================"
    echo "  cuFFTmp: ${NNODES}node × ${TASKS_PER_NODE}GPU = ${TOTAL_TASKS} GPU   Rep ${REP}/2   $(date '+%F %T')"
    echo "============================================================"
    
    srun --mpi=pmix \
         --nodes=$NNODES \
         --ntasks=$TOTAL_TASKS \
         --ntasks-per-node=$TASKS_PER_NODE \
         --gres=gpu:$TASKS_PER_NODE \
         --exact \
         --container-image=$CONTAINER \
         --container-mounts="$MOUNTS" \
         bash -lc "
             cd $NS_DIR
             export NVSHMEM_SYMMETRIC_SIZE=4294967296
             export NVSHMEM_REMOTE_TRANSPORT=ibrc
             export NVSHMEM_BOOTSTRAP=MPI
             export UCX_TLS=rc_verbs,ud_verbs,cuda_copy,cuda_ipc,sm,self
             export UCX_MEMTYPE_CACHE=n
             ./navier_stokes_cufftmp_multinode_docker $NX $NY $NZ $DT $NSTEPS
         "
    
    local EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "!!! cuFFTmp ${NNODES}node×${TASKS_PER_NODE}GPU Rep ${REP} FAILED ($EXIT)"
        return $EXIT
    fi
    echo "  Done at $(date '+%F %T')"
    sleep 10
}

run_heffte() {
    local NNODES=$1
    local TASKS_PER_NODE=$2
    local TOTAL_TASKS=$3
    local REP=$4
    
    echo ""
    echo "============================================================"
    echo "  heFFTe : ${NNODES}node × ${TASKS_PER_NODE}GPU = ${TOTAL_TASKS} GPU   Rep ${REP}/2   $(date '+%F %T')"
    echo "============================================================"
    
    srun --mpi=pmix \
         --nodes=$NNODES \
         --ntasks=$TOTAL_TASKS \
         --ntasks-per-node=$TASKS_PER_NODE \
         --gres=gpu:$TASKS_PER_NODE \
         --exact \
         --container-image=$CONTAINER \
         --container-mounts="$MOUNTS" \
         bash -lc "
             cd $NS_DIR
             export UCX_TLS=rc_mlx5,dc_mlx5,ud_mlx5,cuda_copy,cuda_ipc,gdr_copy,sm,self
             unset UCX_MEMTYPE_CACHE
             unset UCX_RNDV_THRESH
             export LD_LIBRARY_PATH=/opt/heffte/lib:\$LD_LIBRARY_PATH
             ./navier_stokes_heffte_gpuaware $NX $NY $NZ $DT $NSTEPS
         "
    
    local EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "!!! heFFTe ${NNODES}node×${TASKS_PER_NODE}GPU Rep ${REP} FAILED ($EXIT)"
        return $EXIT
    fi
    echo "  Done at $(date '+%F %T')"
    sleep 10
}

echo ""
echo "############################################################"
echo "#  PHASE 1: cuFFTmp"
echo "############################################################"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r N T G <<< "$CFG"
    for REP in 1 2; do
        run_cufftmp $N $T $G $REP || exit $?
    done
done

echo ""
echo "############################################################"
echo "#  PHASE 2: heFFTe"
echo "############################################################"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r N T G <<< "$CFG"
    for REP in 1 2; do
        run_heffte $N $T $G $REP || exit $?
    done
done

echo ""
echo "############################################################"
echo "#  Batch 2a v2 ALL DONE at $(date '+%F %T')"
echo "############################################################"
