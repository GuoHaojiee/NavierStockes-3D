#!/bin/bash
#============================================================
#  bench_pure_omp_debug.sbatch — 失败配置定向诊断
#  目标:  p3dfft@512  /  heffte@256  /  heffte@512
#  假设:  exit=135 (SIGBUS) 由 /dev/shm 不足引起
#============================================================
#SBATCH --job-name=ns3d_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_debug_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_debug_err_%j.log

DEBUG_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_debug
mkdir -p $DEBUG_DIR/{env,p3dfft_512,heffte_256,heffte_512}

CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D
HPCX=/home/user/.local/nvhpc/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19

# ---- 基础环境（与 bench_pure_omp.sbatch 一致）----
ENV_BASE="\
export OPAL_PREFIX=${HPCX}/ompi; \
export PMIX_INSTALL_PREFIX=\$OPAL_PREFIX; \
export LD_LIBRARY_PATH=${HPCX}/ompi/lib:${HPCX}/ucx/lib:${HPCX}/ucc/lib:${HPCX}/hcoll/lib:${HPCX}/sharp/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:/home/user/.local/lib:/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH; \
unset UCX_NET_DEVICES UCX_IB_REG_METHODS UCX_RNDV_THRESH UCX_MAX_RNDV_RAILS UCX_ASYNC_MAX_EVENTS UCX_UNIFIED_MODE UCX_WARN_UNUSED_ENV_VARS; \
export UCX_TLS=sm,self,posix,sysv; \
export OMPI_MCA_pml=ucx; \
export OMPI_MCA_osc=ucx; \
export OMPI_MCA_coll=^hcoll; \
export OMPI_MCA_mtl=^ofi; \
export OMP_NUM_THREADS=128; \
export OMP_PROC_BIND=spread; \
export OMP_PLACES={0:128}"

#============================================================
#  Step 0: 容器内资源探查（最关键的一步！）
#============================================================
echo "============================================================"
echo "  [Step 0] 容器内资源探查"
echo "============================================================"
srun --container-image=$CONTAINER --container-mounts="$MOUNTS" \
     bash -c '
echo "=== /dev/shm 大小 ==="
df -h /dev/shm
echo
echo "=== mount /dev/shm ==="
mount | grep shm
echo
echo "=== ulimit -a ==="
ulimit -a
echo
echo "=== /proc/meminfo (关键项) ==="
grep -E "MemTotal|MemAvailable|MemFree|Shmem|HugePages" /proc/meminfo
echo
echo "=== /proc/sys/kernel ==="
echo "shmmax: $(cat /proc/sys/kernel/shmmax)"
echo "shmall: $(cat /proc/sys/kernel/shmall)"
echo "core_pattern: $(cat /proc/sys/kernel/core_pattern)"
' > $DEBUG_DIR/env/probe.log 2>&1
echo "→ $DEBUG_DIR/env/probe.log"

#============================================================
#  失败配置列表 -> exe:grid:name
#============================================================
CONFIGS=(
    "ns_p3dfft_hpcx:512:p3dfft_512"
    "ns_heffte_v2:256:heffte_256"
    "ns_heffte_v2:512:heffte_512"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r EXE GRID NAME <<< "$cfg"
    OUT=$DEBUG_DIR/$NAME

    echo ""
    echo "============================================================"
    echo "  $NAME  (exe=$EXE, grid=${GRID}^3)"
    echo "============================================================"

    #--------------------------------------------------------
    # [A] 复现 + 详细信号信息  (NSTEPS=1 看是不是 init/第一次 transpose 就挂)
    #--------------------------------------------------------
    echo "--- [A] 复现 + NSTEPS=1 + ulimit core ---"
    timeout 300 srun --mpi=pmix \
      --container-image=$CONTAINER --container-mounts="$MOUNTS" \
      bash -c "$ENV_BASE; \
        ulimit -c unlimited; ulimit -s unlimited; \
        cd $NS_DIR; \
        echo '--- shm before ---'; df -h /dev/shm; \
        echo '--- mem before ---'; grep -E 'MemAvail|Shmem' /proc/meminfo; \
        echo '--- run ---'; \
        ./$EXE $GRID $GRID $GRID 1e-5 1 128; \
        ec=\$?; \
        echo \"--- exit=\$ec  signal=\$((ec-128)) ---\"; \
        echo '--- shm after ---'; df -h /dev/shm" \
      > $OUT/A_step1.log 2>&1
    echo "    exit=$?  ->  $OUT/A_step1.log"

    #--------------------------------------------------------
    # [B] 把 OMP 砍到 16，排除 OMP 私有内存放大
    #--------------------------------------------------------
    echo "--- [B] OMP_NUM_THREADS=16 ---"
    ENV_OMP16="${ENV_BASE//OMP_NUM_THREADS=128/OMP_NUM_THREADS=16}"
    ENV_OMP16="${ENV_OMP16//OMP_PLACES=\{0:128\}/OMP_PLACES=\{0:16\}}"
    timeout 900 srun --mpi=pmix \
      --container-image=$CONTAINER --container-mounts="$MOUNTS" \
      bash -c "$ENV_OMP16; cd $NS_DIR; ./$EXE $GRID $GRID $GRID 1e-5 3 16" \
      > $OUT/B_omp16.log 2>&1
    echo "    exit=$?  ->  $OUT/B_omp16.log"

    #--------------------------------------------------------
    # [C] 关掉 vader/sm shm 传输, 强制 self only —— 看是不是 OMPI shm 路径
    #--------------------------------------------------------
    echo "--- [C] UCX_TLS=self only (绕开 sm/posix/sysv) ---"
    ENV_NOSHM="${ENV_BASE//UCX_TLS=sm,self,posix,sysv/UCX_TLS=self}"
    timeout 600 srun --mpi=pmix \
      --container-image=$CONTAINER --container-mounts="$MOUNTS" \
      bash -c "$ENV_NOSHM; cd $NS_DIR; ./$EXE $GRID $GRID $GRID 1e-5 1 128" \
      > $OUT/C_self_only.log 2>&1
    echo "    exit=$?  ->  $OUT/C_self_only.log"

    #--------------------------------------------------------
    # [D] 显式禁用 vader BTL,  shmem mmap 改成 sysv
    #--------------------------------------------------------
    echo "--- [D] OMPI 强制 sysv shm 而非 mmap ---"
    timeout 600 srun --mpi=pmix \
      --container-image=$CONTAINER --container-mounts="$MOUNTS" \
      bash -c "$ENV_BASE; \
        export OMPI_MCA_btl='^vader,sm'; \
        export OMPI_MCA_shmem=sysv; \
        cd $NS_DIR; ./$EXE $GRID $GRID $GRID 1e-5 1 128" \
      > $OUT/D_sysv_shm.log 2>&1
    echo "    exit=$?  ->  $OUT/D_sysv_shm.log"

    #--------------------------------------------------------
    # [E] UCX 详细日志, 抓最后一次内存事件
    #--------------------------------------------------------
    echo "--- [E] UCX_LOG_LEVEL=info ---"
    timeout 300 srun --mpi=pmix \
      --container-image=$CONTAINER --container-mounts="$MOUNTS" \
      bash -c "$ENV_BASE; export UCX_LOG_LEVEL=info; \
        cd $NS_DIR; ./$EXE $GRID $GRID $GRID 1e-5 1 128 2>&1 | tail -300" \
      > $OUT/E_ucxlog.log 2>&1
    echo "    exit=$?  ->  $OUT/E_ucxlog.log"
done

echo ""
echo "============================================================"
echo "  调试完成: $(date '+%H:%M:%S')"
echo "  输出目录: $DEBUG_DIR"
echo "============================================================"
ls -laR $DEBUG_DIR
