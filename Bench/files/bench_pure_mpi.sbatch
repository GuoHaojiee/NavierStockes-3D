#!/bin/bash
#============================================================
#  bench_pure_mpi.sbatch — Pure MPI: 128 进程 × 1 OMP线程
#
#  设计说明：
#    完全照抄你已验证的 heFFTe 多节点 sbatch 的模式：
#    srun + bash -c "export LD_LIBRARY_PATH=容器路径; exec 程序"
#    这样程序用容器自己的 OpenMPI，不会碰宿主机 HPC-X
#
#    sbatch 里的循环每次调用一个 srun（一个测试）
#    srun 启动 128 个容器实例 = 128 个 MPI rank
#    MPI_Init() 通过 SLURM PMI 协调所有 rank
#
#  提交（登录节点）：sbatch bench_pure_mpi.sbatch
#============================================================
#SBATCH --job-name=ns3d_pure_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_pure_mpi_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_pure_mpi_err_%j.log

# ── 结果目录 ────────────────────────────────────────────────
RESULT_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_pure_mpi
mkdir -p $RESULT_DIR/logs/{fftw,accfft,p3dfft,heffte}/{128,256,512}
CSV=$RESULT_DIR/summary_pure_mpi.csv
echo "solver,grid,mode,np,omp_threads,avg_step_s,l2_error,status" > $CSV

echo "============================================================"
echo "  NS3D Benchmark — pure_mpi (NP=128, OMP=1)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── 容器配置 ────────────────────────────────────────────────
CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

# 容器内环境（与 heFFTe 多节点 sbatch 一致）
ENV_SETUP="export LD_LIBRARY_PATH=/home/user/.local/lib:/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH; \
           export OMPI_MCA_btl='^ofi,openib'; \
           export OMPI_MCA_pml=ob1; \
           export OMP_NUM_THREADS=1"

SOLVERS=("ns_fftw:fftw" "ns_accfft:accfft" "ns_p3dfft:p3dfft" "ns_heffte_v2:heffte")
GRIDS=(128 256 512)
NSTEPS=10; DT=1e-5; TIMEOUT=900

# ── 主循环：每次 srun = 一个测试 ────────────────────────────
for solver_entry in "${SOLVERS[@]}"; do
    exe="${solver_entry%%:*}"
    name="${solver_entry##*:}"
    for grid in "${GRIDS[@]}"; do
        LOG=$RESULT_DIR/logs/${name}/${grid}/pure_mpi.log
        echo ""
        echo "── [$(date '+%H:%M:%S')] $name  ${grid}^3  pure_mpi ──"

        # 128 个 srun task → 128 个容器实例 → 128 个 MPI rank
        timeout $TIMEOUT \
            srun \
              --container-image=$CONTAINER \
              --container-mounts="$MOUNTS" \
              bash -c "$ENV_SETUP; cd $NS_DIR && exec ./$exe $grid $grid $grid $DT $NSTEPS 1" \
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
