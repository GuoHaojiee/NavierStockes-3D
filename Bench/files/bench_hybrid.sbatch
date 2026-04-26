#!/bin/bash
#============================================================
#  bench_hybrid.sbatch
#  Hybrid 模式: 64 进程 × 2 OMP线程
#
#  SLURM 申请与测试配置严格匹配:
#    ntasks=64        ← MPI 进程数
#    cpus-per-task=2  ← 每进程 2 核（供 2 个 OMP 线程使用）
#    总核数 = 64 × 2 = 128
#
#  提交（仅在登录节点）:
#    sbatch bench_hybrid.sbatch
#============================================================
#SBATCH --job-name=ns3d_hybrid
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_hybrid_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_hybrid_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results

echo "Job $SLURM_JOB_ID  hybrid  start $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: $SLURM_NODELIST"

srun --ntasks=1 \
  --container-image=/scratch/s02230673/ns3d.sqsh \
  --container-mounts="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw" \
  --export=ALL,SLURM_JOB_ID=$SLURM_JOB_ID,BENCH_ROOT=/home/user/benchmark_results \
  bash /home/user/benchmark_inside.sh hybrid

echo "Job $SLURM_JOB_ID  hybrid  finish $(date '+%Y-%m-%d %H:%M:%S')"
