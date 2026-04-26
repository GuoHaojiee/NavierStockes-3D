#!/bin/bash
#============================================================
#  bench_hybrid.sbatch — Hybrid: 64 进程 × 2 OMP线程
#  提交（登录节点）：sbatch bench_hybrid.sbatch
#============================================================
#SBATCH --job-name=ns3d_hybrid
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/s02230673/benchmark_results/slurm_hybrid_%j.log
#SBATCH --error=/scratch/s02230673/benchmark_results/slurm_hybrid_err_%j.log

RESULT_DIR=/scratch/s02230673/benchmark_results/job_${SLURM_JOB_ID}_hybrid
mkdir -p $RESULT_DIR/logs/{fftw,accfft,p3dfft,heffte}/{128,256,512}
CSV=$RESULT_DIR/summary_hybrid.csv
echo "solver,grid,mode,np,omp_threads,avg_step_s,l2_error,status" > $CSV

echo "============================================================"
echo "  NS3D Benchmark — hybrid (NP=64, OMP=2)"
echo "  Job: $SLURM_JOB_ID   Node: $SLURM_NODELIST"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

CONTAINER=/scratch/s02230673/ns3d.sqsh
MOUNTS="/usr/local/lib/slurm:/usr/local/lib/slurm:ro,/scratch/s02230673:/home/user:rw"
NS_DIR=/home/user/guo/NavierStockes-3D

ENV_SETUP="export LD_LIBRARY_PATH=/home/user/.local/lib:/usr/local/cuda-12.2/lib64:\$LD_LIBRARY_PATH; \
           export OMPI_MCA_btl='^ofi,openib'; \
           export OMPI_MCA_pml=ob1; \
           export OMP_NUM_THREADS=2; \
           export OMP_PROC_BIND=spread; \
           export OMP_PLACES=cores"

SOLVERS=("ns_fftw:fftw" "ns_accfft:accfft" "ns_p3dfft:p3dfft" "ns_heffte_v2:heffte")
GRIDS=(128 256 512)
NSTEPS=10; DT=1e-5; TIMEOUT=900

for solver_entry in "${SOLVERS[@]}"; do
    exe="${solver_entry%%:*}"
    name="${solver_entry##*:}"
    for grid in "${GRIDS[@]}"; do
        LOG=$RESULT_DIR/logs/${name}/${grid}/hybrid.log
        echo ""
        echo "── [$(date '+%H:%M:%S')] $name  ${grid}^3  hybrid ──"

        # 64 个 srun task → 64 个容器实例 → 64 个 MPI rank
        # 每个 rank 内 OMP_NUM_THREADS=2，共 128 核
        timeout $TIMEOUT \
            srun \
              --container-image=$CONTAINER \
              --container-mounts="$MOUNTS" \
              bash -c "$ENV_SETUP; cd $NS_DIR && exec ./$exe $grid $grid $grid $DT $NSTEPS 2" \
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
        echo "${name},${grid},hybrid,64,2,${avg:-PARSE_ERR},${l2:-N/A},${st}" >> $CSV
    done
done

echo ""
echo "============================================================"
echo "  hybrid 完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
cat $CSV
