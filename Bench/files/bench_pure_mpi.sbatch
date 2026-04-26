#!/bin/bash
#============================================================
#  bench_pure_mpi.sbatch
#  Pure MPI 模式: 128 进程 × 1 OMP线程
#
#  SLURM 申请与测试配置严格匹配:
#    ntasks=128       ← MPI 进程数
#    cpus-per-task=1  ← 每进程 1 核（无 OMP）
#
#  提交（仅在登录节点）:
#    sbatch bench_pure_mpi.sbatch
#============================================================
#SBATCH --job-name=ns3d_pure_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_pure_mpi_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_pure_mpi_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results

echo "Job $SLURM_JOB_ID  pure_mpi  start $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: $SLURM_NODELIST"

srun --ntasks=1 \
  --container-image=/scratch/s02230673/ns3d.sqsh \
  --container-mounts="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw" \
  --export=ALL,SLURM_JOB_ID=$SLURM_JOB_ID,BENCH_ROOT=/home/user/benchmark_results \
  bash /home/user/benchmark_inside.sh pure_mpi

echo "Job $SLURM_JOB_ID  pure_mpi  finish $(date '+%Y-%m-%d %H:%M:%S')"
