#!/bin/bash
#SBATCH -J p3dv
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:20:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/p3dv_%j.out
# p3d_verify — the two rebuilt P3DFFT solvers (build_p3dfft_opt.sh), 64^3 verify, dt=5e-5, 10 steps, 8 ranks x 1 thread,
#   slab and pencil. Reference L2 = 6.0041e-10 (same as scripts/verify_decomp.sbatch).
#   new-gcc   /work/build/bin_p3d/gcc/ns_p3dfft     GCC container, Open MPI, srun --mpi=pmix
#   new-intel /work/build/bin_p3d/intel/ns_p3dfft   Intel container, Intel MPI, mpirun (fork + shm) in one task
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/p3dfft/p3d_verify.sbatch
#   grep -E "^RESULT|^P3DFFT library" $(ls -t ~/guohaojie/logs/p3dv_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh     # $M, $MI, precheck, cleanup_all
precheck
N=64; DT=5e-5; NS=10; NP=8; REF=6.0041e-10
B=/work/build/bin_p3d
KEEP="^decomposition:|P3DFFT library|P3DFFT planner|Running on|L2 error|max\|div|rror|abort"

# V <label> gcc|intel <binary> <slab|pencil>
V() {
    local label=$1 tc=$2 bin=$3 d=$4 out rc l2 dec st
    echo ">>> [$(date +%H:%M:%S)] $label ranks=$NP NS_DECOMP=$d | $bin $N $N $N $DT $NS 1 verify"
    if [ $tc = gcc ]; then
        out=$(srun --mpi=pmix --cpu-bind=cores --time=10 -N 1 -n $NP -c 1 $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
            export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close NS_DECOMP=$d; exec $bin $N $N $N $DT $NS 1 verify" 2>&1); rc=$?
    else
        out=$(srun -N 1 -n 1 -c 128 --cpu-bind=none --time=10 $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
            export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=1:compact OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close \
                   KMP_BLOCKTIME=0 NS_DECOMP=$d; exec mpirun -np $NP $bin $N $N $N $DT $NS 1 verify" 2>&1); rc=$?
    fi
    echo "$out" | grep -E "$KEEP"; echo "<<< exit=$rc"; echo
    dec=$(echo "$out" | grep -m1 '^decomposition:')
    l2=$(echo "$out" | grep -m1 'L2 error' | awk '{print $NF}')
    [ "$l2" = "$REF" ] && [ $rc = 0 ] && st=OK || st=MISMATCH
    printf "RESULT %-10s NS_DECOMP=%-6s L2=%-11s %-8s | %s\n" $label $d "${l2:-none}" $st "$dec" | tee -a $SCR/logs/p3dv_$SLURM_JOB_ID.res
}

echo "binaries: $(ls -l --time-style=+%F_%R $SCR/build/bin_p3d/gcc/ns_p3dfft $SCR/build/bin_p3d/intel/ns_p3dfft 2>&1 | awk '{print $6, $7}' | tr '\n' ' ')"
for d in slab pencil; do
    V new-gcc   gcc   $B/gcc/ns_p3dfft   $d
    V new-intel intel $B/intel/ns_p3dfft $d
done
echo "=============== summary (all 4 must be OK)"
cat $SCR/logs/p3dv_$SLURM_JOB_ID.res; rm -f $SCR/logs/p3dv_$SLURM_JOB_ID.res
cleanup_all
