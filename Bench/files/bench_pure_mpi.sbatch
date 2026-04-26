#!/bin/bash
#============================================================
#  bench_pure_mpi.sbatch — Pure MPI: 128 进程 × 1 OMP线程
#
#  使用重新编译的 _hpcx 版本（链接 /home/user/.local/nvhpc HPC-X）
#  RPATH 已写入二进制，srun 启动时自动加载正确的 libmpi.so
#  此方式与 SLURM PMIx 兼容，同时支持多节点扩展
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

RESULT_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_pure_mpi
mkdir -p $RESULT_DIR/logs/{fftw,accfft,p3dfft,heffte}/{128,256,512}
CSV=$RESULT_DIR/summary_pure_mpi.csv
echo "solver,grid,mode,np,omp_threads,avg_step_s,l2_error,status" > $CSV

echo "============================================================"
echo "  NS3D Benchmark — pure_mpi (NP=128, OMP=1)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

# LD_LIBRARY_PATH：RPATH 已处理 libmpi，这里补充 NVHPC 编译器运行时库
# （ldd 显示需要 /opt/nvidia/hpc_sdk/.../compilers/lib 里的 libnvomp 等）
ENV="export LD_LIBRARY_PATH=\
/home/user/.local/nvhpc/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi/lib:\
/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:\
/home/user/.local/lib:\
/usr/local/cuda-12.2/lib64:\
\$LD_LIBRARY_PATH; \
export OMP_NUM_THREADS=1"

# 使用重新编译的 _hpcx 版本
SOLVERS=("ns_fftw_hpcx:fftw" "ns_accfft_hpcx:accfft" "ns_p3dfft_hpcx:p3dfft" "ns_heffte_v2:heffte")
GRIDS=(128 256 512)
NSTEPS=10; DT=1e-5; TIMEOUT=900

for solver_entry in "${SOLVERS[@]}"; do
    exe="${solver_entry%%:*}"
    name="${solver_entry##*:}"
    for grid in "${GRIDS[@]}"; do
        LOG=$RESULT_DIR/logs/${name}/${grid}/pure_mpi.log
        echo ""
        echo "── [$(date '+%H:%M:%S')] $name  ${grid}^3  pure_mpi ──"

        # srun 启动 128 个容器实例 = 128 个 MPI rank
        # 二进制 RPATH 指向正确的 HPC-X，PMIx 兼容
        timeout $TIMEOUT \
            srun \
              --container-image=$CONTAINER \
              --container-mounts="$MOUNTS" \
              bash -c "$ENV; cd $NS_DIR && exec ./$exe $grid $grid $grid $DT $NSTEPS 1" \
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
