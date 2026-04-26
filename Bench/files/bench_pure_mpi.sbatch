#!/bin/bash
#============================================================
#  bench_pure_mpi.sbatch — Pure MPI: 128 进程 × 1 OMP线程
#  提交（登录节点）：sbatch bench_pure_mpi.sbatch
#============================================================
#SBATCH --job-name=ns3d_pure_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=64
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_pure_mpi_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_pure_mpi_err_%j.log

# ── UCX 通信优化（InfiniBand，与学长脚本一致）───────────────
export UCX_ASYNC_MAX_EVENTS=512
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_IB_REG_METHODS=odp
export UCX_RNDV_THRESH=8192
export UCX_MAX_RNDV_RAILS=1
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc_x,sm,self
export UCX_UNIFIED_MODE=y

# ── MPI 优化 ────────────────────────────────────────────────
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl="^openib,tcp"
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_orte_quit_timeout=10
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
export OMPI_MCA_coll_tuned_bcast_algorithm=1
export OMPI_MCA_coll_tuned_allreduce_algorithm=2
export PMIX_MCA_gds=hash
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1

# ── 路径配置 ────────────────────────────────────────────────
CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
RESULT_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_pure_mpi

# 显式 CPU 映射：0,1,2,...,127
CPU_MAP=$(seq -s , 0 127)

mkdir -p $RESULT_DIR/logs/{fftw,accfft,p3dfft,heffte}/{128,256,512}
CSV=$RESULT_DIR/summary_pure_mpi.csv
echo "solver,grid,mode,np,omp_threads,avg_step_s,l2_error,status" > $CSV

echo "============================================================"
echo "  NS3D Benchmark — pure_mpi (NP=128, OMP=1)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

SOLVERS=("ns_fftw:fftw" "ns_accfft:accfft" "ns_p3dfft:p3dfft" "ns_heffte_v2:heffte")
GRIDS=(128 256 512)
NSTEPS=10; DT=1e-5; TIMEOUT=900

for solver_entry in "${SOLVERS[@]}"; do
    exe="${solver_entry%%:*}"
    name="${solver_entry##*:}"
    for grid in "${GRIDS[@]}"; do
        LOG=$RESULT_DIR/logs/${name}/${grid}/pure_mpi.log
        echo ""
        echo "── [$(date '+%H:%M:%S')] $name  ${grid}^3  pure_mpi ──"

        timeout $TIMEOUT \
            srun \
              --mpi=pmix \
              --network=host \
              --cpu-bind=map_cpu:${CPU_MAP} \
              --distribution=block:block:cyclic \
              --hint=nomultithread \
              --mem-bind=local \
              --container-image=$CONTAINER \
              --container-mounts="$MOUNTS" \
              --no-container-entrypoint \
              $NS_DIR/$exe $grid $grid $grid $DT $NSTEPS 1 \
            > $LOG 2>&1

        ec=$?
        avg=$(grep "Avg per step:" $LOG | awk '{print $(NF-1)}')
        l2=$(grep "L2 error" $LOG | tail -1 | grep -oP '[0-9.]+e[+\-][0-9]+')

        if   [[ $ec -eq 0   ]]; then st="OK"
        elif [[ $ec -eq 139 ]]; then st="OK_SEGFAULT"
        elif [[ $ec -eq 124 ]]; then st="TIMEOUT"; avg="TIMEOUT"
        else                         st="ERROR(exit=$ec)"; avg="ERROR"
        fi

        echo "  -> $st | avg=${avg}s | L2=$l2"
        echo "${name},${grid},pure_mpi,128,1,${avg:-PARSE_ERR},${l2:-N/A},${st}" >> $CSV
    done
done

echo ""
echo "============================================================"
echo "  pure_mpi 完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
cat $CSV
