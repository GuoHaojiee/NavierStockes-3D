#!/bin/bash
#============================================================
#  benchmark_inside.sh  —  容器内执行，由三个 sbatch 分别调用
#
#  用法: bash benchmark_inside.sh <MODE>
#    MODE: pure_mpi | hybrid | pure_omp
#
#  由 sbatch 通过 --export 传入的环境变量:
#    SLURM_JOB_ID   — 用于结果目录命名
#    BENCH_ROOT     — 结果根目录（容器内路径，默认 /home/user/benchmark_results）
#============================================================

MODE=${1:-"pure_mpi"}

# ════════════════════════════════════════════════════════════
#  1. 环境变量
# ════════════════════════════════════════════════════════════
export LD_LIBRARY_PATH=/home/user/.local/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export OMPI_MCA_btl=^ofi,openib
export OMPI_MCA_pml=ob1

NS_DIR=/home/user/guo/NavierStockes-3D
BENCH_ROOT=${BENCH_ROOT:-/home/user/benchmark_results}
RESULT_DIR=${BENCH_ROOT}/job_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}_${MODE}
SUMMARY_CSV=${RESULT_DIR}/summary_${MODE}.csv
SUMMARY_TABLE=${RESULT_DIR}/summary_${MODE}_table.txt

mkdir -p ${RESULT_DIR}/logs

# ════════════════════════════════════════════════════════════
#  2. 根据 MODE 确定 NP、OMP、mpirun binding
#
#  binding 参数与 SLURM 申请严格对齐：
#    pure_mpi  → --ntasks=128 --cpus-per-task=1
#                mpirun: --map-by slot --bind-to core
#
#    hybrid    → --ntasks=64  --cpus-per-task=2
#                mpirun: --map-by slot:PE=2 --bind-to core
#                每个 MPI 进程绑定到连续 2 个核，OMP 线程在其上展开
#
#    pure_omp  → --ntasks=1   --cpus-per-task=128
#                mpirun: --bind-to none
#                单进程，128 个 OMP 线程使用全部核心
# ════════════════════════════════════════════════════════════
case "$MODE" in
    pure_mpi)
        NP=128; OMP=1
        MODE_DESC="Pure MPI: 128 进程 × 1 OMP线程"
        MPIRUN_BIND="--map-by slot --bind-to core"
        ;;
    hybrid)
        NP=64;  OMP=2
        MODE_DESC="Hybrid: 64 进程 × 2 OMP线程"
        MPIRUN_BIND="--map-by slot:PE=2 --bind-to core"
        ;;
    pure_omp)
        NP=1;   OMP=128
        MODE_DESC="Pure OMP: 1 进程 × 128 OMP线程"
        MPIRUN_BIND="--bind-to none"
        ;;
    *)
        echo "ERROR: Unknown MODE='$MODE'. Use: pure_mpi | hybrid | pure_omp"
        exit 1
        ;;
esac

export OMP_NUM_THREADS=$OMP

echo "============================================================"
echo "  NS3D CPU Benchmark — MODE: $MODE"
echo "  $MODE_DESC"
echo "  Node    : $(hostname)"
echo "  CPUs    : $(nproc)"
echo "  NP=$NP  OMP=$OMP"
echo "  Binding : $MPIRUN_BIND"
echo "  Start   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results : $RESULT_DIR"
echo "============================================================"

echo "solver,grid,mode,np,omp_threads,avg_step_s,l2_error,status" > $SUMMARY_CSV

# ════════════════════════════════════════════════════════════
#  3. 测试矩阵
# ════════════════════════════════════════════════════════════
SOLVERS=(
    "fftw:ns_fftw"
    "accfft:ns_accfft"
    "p3dfft:ns_p3dfft"
    "heffte:ns_heffte_v2"
)
GRIDS=(128 256 512)
NSTEPS=10
DT=1e-5
TIMEOUT_SEC=900      # 15 min，防止 pure_omp 大网格跑死

# ════════════════════════════════════════════════════════════
#  4. 单次测试函数
# ════════════════════════════════════════════════════════════
run_one_test() {
    local solver_name=$1
    local solver_exe=$2
    local grid=$3

    local log_dir=${RESULT_DIR}/logs/${solver_name}/${grid}
    mkdir -p $log_dir
    local log_file=${log_dir}/${MODE}.log

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  [$(date '+%H:%M:%S')] $solver_name | ${grid}^3 | $MODE"
    echo "  mpirun -np $NP $MPIRUN_BIND"
    echo "────────────────────────────────────────────────────────"

    {
        echo "# ======================================================"
        echo "# Solver  : $solver_name ($solver_exe)"
        echo "# Grid    : ${grid}^3"
        echo "# Mode    : $MODE  (NP=$NP, OMP=$OMP)"
        echo "# Binding : $MPIRUN_BIND"
        echo "# DT=$DT  NSTEPS=$NSTEPS"
        echo "# Start   : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# ======================================================"
    } > $log_file

    timeout ${TIMEOUT_SEC} \
        mpirun -np $NP \
               $MPIRUN_BIND \
               ${NS_DIR}/${solver_exe} \
               $grid $grid $grid $DT $NSTEPS $OMP \
        >> $log_file 2>&1

    local exit_code=$?
    echo "# Finish  : $(date '+%Y-%m-%d %H:%M:%S')" >> $log_file
    echo "# ExitCode: $exit_code"                   >> $log_file

    local avg_time="N/A"
    local l2_err="N/A"
    local status="UNKNOWN"

    if [[ $exit_code -eq 0 || $exit_code -eq 139 ]]; then
        # 0   = 正常; 139 = segfault (p3dfft 退出时已知问题，结果有效)
        avg_time=$(grep -m1 "Avg per step:" $log_file \
                   | awk '{print $(NF-1)}')
        l2_err=$(grep "L2 error" $log_file \
                 | tail -1 \
                 | grep -oP '[\d]+\.[\d]+e[+\-][\d]+')
        [[ -z "$avg_time" ]] && avg_time="PARSE_ERR"
        [[ -z "$l2_err"   ]] && l2_err="PARSE_ERR"
        status="OK"
        [[ $exit_code -eq 139 ]] && status="OK_SEGFAULT"
    elif [[ $exit_code -eq 124 ]]; then
        status="TIMEOUT(>${TIMEOUT_SEC}s)"; avg_time="TIMEOUT"
    else
        status="ERROR(exit=$exit_code)";    avg_time="ERROR"
    fi

    echo "  ✔ avg=$avg_time s | L2=$l2_err | Status=$status"
    echo "${solver_name},${grid},${MODE},${NP},${OMP},${avg_time},${l2_err},${status}" \
        >> $SUMMARY_CSV
}

# ════════════════════════════════════════════════════════════
#  5. 主循环：4 solver × 3 grid = 12 组
# ════════════════════════════════════════════════════════════
cd $NS_DIR
total=0

for solver_entry in "${SOLVERS[@]}"; do
    solver_name="${solver_entry%%:*}"
    solver_exe="${solver_entry##*:}"

    if [[ ! -f "${NS_DIR}/${solver_exe}" ]]; then
        echo "⚠  WARNING: ${solver_exe} not found, skipping."
        for grid in "${GRIDS[@]}"; do
            echo "${solver_name},${grid},${MODE},${NP},${OMP},N/A,N/A,NOT_FOUND" >> $SUMMARY_CSV
        done
        continue
    fi

    for grid in "${GRIDS[@]}"; do
        run_one_test "$solver_name" "$solver_exe" "$grid"
        (( total++ ))
    done
done

# ════════════════════════════════════════════════════════════
#  6. 格式化汇总表
# ════════════════════════════════════════════════════════════
python3 - "$SUMMARY_CSV" "$SUMMARY_TABLE" "$MODE" "$NP" "$OMP" << 'PYEOF'
import csv, sys

csv_path, table_path, mode, np_val, omp_val = sys.argv[1:6]
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

SOLVERS = ['fftw', 'accfft', 'p3dfft', 'heffte']
GRIDS   = ['128', '256', '512']

def lookup(rows, solver, grid, field):
    for r in rows:
        if r['solver'] == solver and r['grid'] == grid:
            return r.get(field, 'N/A')
    return 'N/A'

def fmt_time(v):
    if v in ('N/A','ERROR','TIMEOUT','PARSE_ERR','NOT_FOUND'):
        return f"[{v[:10]:^10}]"
    try:    return f"{float(v):>8.4f} s"
    except: return f"{v:>10}"

lines = []
lines.append("=" * 62)
lines.append(f"  NS3D CPU Benchmark — {mode.upper()}")
lines.append(f"  NP={np_val} × OMP={omp_val}   |   10 steps, dt=1e-5")
lines.append("=" * 62)
lines.append(f"  {'Solver':<10}  {'128³':>12}  {'256³':>12}  {'512³':>12}")
lines.append(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
for solver in SOLVERS:
    t128 = fmt_time(lookup(rows, solver, '128', 'avg_step_s'))
    t256 = fmt_time(lookup(rows, solver, '256', 'avg_step_s'))
    t512 = fmt_time(lookup(rows, solver, '512', 'avg_step_s'))
    lines.append(f"  {solver:<10}  {t128:>12}  {t256:>12}  {t512:>12}")
lines.append("")
lines.append("  L2 Error:")
lines.append(f"  {'Solver':<10}  {'128³':>12}  {'256³':>12}  {'512³':>12}")
lines.append(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
for solver in SOLVERS:
    e128 = lookup(rows, solver, '128', 'l2_error')
    e256 = lookup(rows, solver, '256', 'l2_error')
    e512 = lookup(rows, solver, '512', 'l2_error')
    lines.append(f"  {solver:<10}  {e128:>12}  {e256:>12}  {e512:>12}")
lines.append("=" * 62)

table_str = "\n".join(lines) + "\n"
print(table_str)
with open(table_path, 'w') as f:
    f.write(table_str)
PYEOF

echo ""
echo "============================================================"
echo "  MODE=$MODE 完成  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Tested : $total"
echo "  Logs   : $RESULT_DIR/logs/"
echo "  CSV    : $SUMMARY_CSV"
echo "  Table  : $SUMMARY_TABLE"
echo "============================================================"
cat $SUMMARY_CSV
