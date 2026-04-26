#!/bin/bash
#============================================================
#  bench_gpu_mgpu.sbatch — 单节点多 GPU FFT 性能对比
#
#  对比：cuFFTXt vs heFFTe(优化版: UCX + slab + persistent workspace)
#  GPU 数：2 / 4 / 8        网格：256³ / 512³ / 1024³
#
#  cuFFTXt: 单进程多卡，CUDA_VISIBLE_DEVICES 控制卡数
#  heFFTe : mpirun + UCX/CUDA-IPC + slab 分解 + 持久 workspace
#
#  注意：1024³ 在 2 GPU 下 heFFTe 可能 OOM（每卡 ~128 GB），
#        脚本会捕获错误并继续，结果在 CSV 中标 ERROR
#
#  提交：sbatch bench_gpu_mgpu.sbatch
#============================================================
#SBATCH --job-name=ns3d_gpu_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_gpu_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_gpu_err_%j.log

#---------- 结果目录 ----------
RESULT_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_gpu_mgpu
mkdir -p $RESULT_DIR/logs/{cufftxt,heffte}/{256,512,1024}
CSV=$RESULT_DIR/summary_gpu_mgpu.csv
echo "solver,grid,ngpu,avg_step_s,total_time_s,l2_error,status" > $CSV

echo "============================================================"
echo "  NS3D GPU Benchmark — cuFFTXt vs heFFTe (optimized)"
echo "  Job:    $SLURM_JOB_ID    Node: $SLURM_NODELIST"
echo "  Start:  $(date '+%F %T')"
echo "  Output: $RESULT_DIR"
echo "============================================================"

#---------- 容器与环境 ----------
CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

# 单引号确保 $LD_LIBRARY_PATH 不在外层 shell 展开，
# 留到容器内 bash -c 处理时再展开（跟模板用 \$ 等价但更干净）
ENV_SETUP='export LD_LIBRARY_PATH=/home/user/.local/nvhpc/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:/home/user/.local/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH'

#---------- 测试矩阵 ----------
GPU_COUNTS=(2 4 8)
GRIDS=(256 512 1024)
NSTEPS=10
DT=1e-5
TIMEOUT=900    # 1024³ on 8 GPU 大概 100s，900s 足够

# heFFTe MPI 标志（你验证过的优化组合）
HEFFTE_MPI_FLAGS="--mca pml ucx --mca btl ^openib \
-x UCX_TLS=cuda_copy,cuda_ipc,sm,self \
-x UCX_MEMTYPE_CACHE=n \
-x LD_LIBRARY_PATH \
-x CUDA_VISIBLE_DEVICES"

#---------- 解析辅助 ----------
parse_avg()   { grep "Avg per step:"     "$1" | tail -1 | awk '{print $(NF-1)}'; }
parse_total() { grep "Total wall time:"  "$1" | tail -1 | awk '{print $(NF-1)}'; }
parse_l2()    { grep -E "L2 error"       "$1" | tail -1 | grep -oP '[0-9.]+e[+\-][0-9]+'; }

run_in_container() {
    # $1 = 在容器内 NS_DIR 下要执行的命令字符串
    local cmd="$1"
    srun -n 1 \
        --container-image=$CONTAINER \
        --container-mounts="$MOUNTS" \
        bash -c "$ENV_SETUP; cd $NS_DIR && $cmd"
}

classify() {
    # $1 = exit code; 设置全局 ST 变量
    case $1 in
        0)   ST="OK" ;;
        124) ST="TIMEOUT" ;;
        137) ST="OOM_KILLED" ;;
        139) ST="SEGFAULT" ;;
        *)   ST="ERROR(exit=$1)" ;;
    esac
}

#---------- 主循环 ----------
for ngpu in "${GPU_COUNTS[@]}"; do
    GPU_LIST=$(seq -s, 0 $((ngpu-1)))   # "0,1" / "0,1,2,3" / "0,1,2,3,4,5,6,7"

    for grid in "${GRIDS[@]}"; do
        echo ""
        echo "############################################################"
        echo "  [$(date '+%H:%M:%S')]  CONFIG  GPUs=$ngpu  Grid=${grid}^3"
        echo "  Visible devices: $GPU_LIST"
        echo "############################################################"

        # =============== cuFFTXt ===============
        LOG=$RESULT_DIR/logs/cufftxt/${grid}/ngpu${ngpu}.log
        echo "── cuFFTXt ${grid}³ on ${ngpu} GPU(s) ──"

        timeout $TIMEOUT run_in_container \
            "CUDA_VISIBLE_DEVICES=$GPU_LIST ./ns_cufftxt $grid $grid $grid $DT $NSTEPS" \
            > $LOG 2>&1
        classify $?

        avg=$(parse_avg $LOG); tot=$(parse_total $LOG); l2=$(parse_l2 $LOG)
        echo "   $ST | avg=${avg:-N/A}s  total=${tot:-N/A}s  L2=${l2:-N/A}"
        echo "cufftxt,${grid},${ngpu},${avg:-PARSE_ERR},${tot:-N/A},${l2:-N/A},${ST}" >> $CSV

        # =============== heFFTe (优化版) ===============
        LOG=$RESULT_DIR/logs/heffte/${grid}/ngpu${ngpu}.log
        echo "── heFFTe  ${grid}³ on ${ngpu} GPU(s) ──"

        timeout $TIMEOUT run_in_container \
            "CUDA_VISIBLE_DEVICES=$GPU_LIST mpirun -np $ngpu $HEFFTE_MPI_FLAGS ./ns_heffte_mgpu $grid $grid $grid $DT $NSTEPS" \
            > $LOG 2>&1
        classify $?

        avg=$(parse_avg $LOG); tot=$(parse_total $LOG); l2=$(parse_l2 $LOG)
        echo "   $ST | avg=${avg:-N/A}s  total=${tot:-N/A}s  L2=${l2:-N/A}"
        echo "heffte,${grid},${ngpu},${avg:-PARSE_ERR},${tot:-N/A},${l2:-N/A},${ST}" >> $CSV
    done
done

echo ""
echo "============================================================"
echo "  Done: $(date '+%F %T')"
echo "  CSV:  $CSV"
echo "============================================================"
column -t -s, $CSV
