#!/bin/bash
#============================================================
#  bench_gpu_mgpu_v2.sbatch — 单节点多 GPU FFT 性能对比（新容器）
#
#  vs v1 的差异：
#    1) 容器换成 ns3d_v2.sqsh（heFFTe 在 /opt/heffte/lib，多节点验证过）
#    2) heFFTe 启动方式从 mpirun -np 改成 srun --mpi=pmix --ntasks=N
#       （绕开容器 root-mpirun 限制，跟多节点脚本统一）
#    3) UCX env 简化：移除 UCX_MEMTYPE_CACHE=n（实测无收益，0.535s vs 0.528s）
#    4) cuFFTXt 调用方式不变（单进程多卡，CUDA_VISIBLE_DEVICES 控制）
#    5) ENV_SETUP 里 LD_LIBRARY_PATH 前置 /opt/heffte/lib
#
#  提交：sbatch bench_gpu_mgpu_v2.sbatch
#============================================================
#SBATCH --job-name=ns3d_gpu_bench_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_gpu_v2_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_gpu_v2_err_%j.log

#---------- 结果目录 ----------
RESULT_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_gpu_mgpu_v2
mkdir -p $RESULT_DIR/logs/{cufftxt,heffte}/{256,512,1024}
CSV=$RESULT_DIR/summary_gpu_mgpu_v2.csv
echo "solver,grid,ngpu,avg_step_s,total_time_s,l2_error,status" > $CSV

echo "============================================================"
echo "  NS3D GPU Benchmark v2 — cuFFTXt vs heFFTe (ns3d_v2 container)"
echo "  Job:    $SLURM_JOB_ID    Node: $SLURM_NODELIST"
echo "  Start:  $(date '+%F %T')"
echo "  Output: $RESULT_DIR"
echo "============================================================"

#---------- 容器与环境 ----------
CONTAINER=/scratch/s02230673/ns3d_v2.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

#---------- 测试矩阵 ----------
GPU_COUNTS=(2 4 8)
GRIDS=(256 512 1024)
NSTEPS=5
DT=1e-6
TIMEOUT=900

#---------- 解析辅助 ----------
parse_avg()   { grep "Avg per step:"     "$1" | tail -1 | awk '{print $(NF-1)}'; }
parse_total() { grep "Total wall time:"  "$1" | tail -1 | awk '{print $(NF-1)}'; }
parse_l2()    { grep -E "L2 error"       "$1" | tail -1 | grep -oP '[0-9.]+e[+\-][0-9]+'; }

classify() {
    case $1 in
        0)   ST="OK" ;;
        124) ST="TIMEOUT" ;;
        137) ST="KILLED(SIGKILL)" ;;
        139) ST="SEGFAULT" ;;
        *)   ST="ERROR(exit=$1)" ;;
    esac
}

#---------- cuFFTXt 运行（单进程多卡）----------
run_cufftxt() {
    local ngpu=$1 grid=$2 logfile=$3 gpu_list=$4

    timeout $TIMEOUT srun -n 1 \
        --container-image=$CONTAINER \
        --container-mounts="$MOUNTS" \
        bash -lc "
            cd $NS_DIR
            export LD_LIBRARY_PATH=/opt/heffte/lib:/home/user/.local/nvhpc/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:/home/user/.local/lib:/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH
            export CUDA_VISIBLE_DEVICES=$gpu_list
            ./ns_cufftxt $grid $grid $grid $DT $NSTEPS
        " > $logfile 2>&1
    return $?
}

#---------- heFFTe 运行（srun --mpi=pmix）----------
run_heffte() {
    local ngpu=$1 grid=$2 logfile=$3 gpu_list=$4

    timeout $TIMEOUT srun --mpi=pmix \
        --ntasks=$ngpu --ntasks-per-node=$ngpu --exact --gres=gpu:$ngpu \
        --container-image=$CONTAINER \
        --container-mounts="$MOUNTS" \
        bash -lc "
            cd $NS_DIR
            export LD_LIBRARY_PATH=/opt/heffte/lib:/home/user/.local/nvhpc/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:/home/user/.local/lib:/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH
            export UCX_TLS=cuda_copy,cuda_ipc,sm,self
            export CUDA_VISIBLE_DEVICES=$gpu_list
            ./ns_heffte_mgpu $grid $grid $grid $DT $NSTEPS
        " > $logfile 2>&1
    return $?
}

#---------- 主循环 ----------
for ngpu in "${GPU_COUNTS[@]}"; do
    GPU_LIST=$(seq -s, 0 $((ngpu-1)))

    for grid in "${GRIDS[@]}"; do
        echo ""
        echo "############################################################"
        echo "  [$(date '+%H:%M:%S')]  CONFIG  GPUs=$ngpu  Grid=${grid}^3"
        echo "  Visible devices: $GPU_LIST"
        echo "############################################################"

        # ============ cuFFTXt ============
        LOG=$RESULT_DIR/logs/cufftxt/${grid}/ngpu${ngpu}.log
        echo "── cuFFTXt ${grid}³ on ${ngpu} GPU(s) ──"
        run_cufftxt $ngpu $grid $LOG $GPU_LIST
        classify $?
        avg=$(parse_avg $LOG); tot=$(parse_total $LOG); l2=$(parse_l2 $LOG)
        echo "   $ST | avg=${avg:-N/A}s  total=${tot:-N/A}s  L2=${l2:-N/A}"
        echo "cufftxt,${grid},${ngpu},${avg:-PARSE_ERR},${tot:-N/A},${l2:-N/A},${ST}" >> $CSV

        # ============ heFFTe ============
        LOG=$RESULT_DIR/logs/heffte/${grid}/ngpu${ngpu}.log
        echo "── heFFTe  ${grid}³ on ${ngpu} GPU(s) ──"
        run_heffte $ngpu $grid $LOG $GPU_LIST
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
