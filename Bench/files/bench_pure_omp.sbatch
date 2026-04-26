#!/bin/bash
#============================================================
#  bench_pure_omp.sbatch
#  Pure OMP 模式: 1 进程 × 128 OMP线程
#
#  SLURM 申请与测试配置严格匹配:
#    ntasks=1          ← 1 个 MPI 进程
#    cpus-per-task=128 ← 全部 128 核给这一个进程用
#
#  注意: 这些 FFT 库主要为 MPI 设计，纯 OMP 模式
#        在大网格（512^3）可能极慢，已设置 15 分钟超时
#
#  提交（仅在登录节点）:
#    sbatch bench_pure_omp.sbatch
#============================================================
#SBATCH --job-name=ns3d_pure_omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_pure_omp_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_pure_omp_err_%j.log

mkdir -p /scratch/s02230673/benchmark_results

echo "Job $SLURM_JOB_ID  pure_omp  start $(date '+%Y-%m-%d %H:%M:%S')"
echo "Node: $SLURM_NODELIST"

srun --ntasks=1 \
  --container-image=/scratch/s02230673/ns3d.sqsh \
  --container-mounts="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw" \
  --export=ALL,SLURM_JOB_ID=$SLURM_JOB_ID,BENCH_ROOT=/home/user/benchmark_results \
  bash /home/user/benchmark_inside.sh pure_omp

echo "Job $SLURM_JOB_ID  pure_omp  finish $(date '+%Y-%m-%d %H:%M:%S')"
