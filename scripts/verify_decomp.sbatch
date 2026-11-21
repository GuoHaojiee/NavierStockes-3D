#!/bin/bash
#SBATCH -J vd
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:40:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/vd_%x_%j.out
# verify_decomp — every CPU library x every decomposition (NS_DECOMP), 64^3 verify, one node.
#   Reference (dt=5e-5, 10 steps): L2 = 6.0041e-10 for every library and every decomposition.
#   8 ranks  : slab, pencil (and brick for heFFTe) explicitly, auto (-> slab since 8 <= 64)
#   128 ranks: auto (-> pencil since 128 > 64)
#   FFTW MPI is slab only: run once at 8 ranks (prints "decomposition: slab").
# Binaries: /work/build/bin_gcc from scripts/compile_all.sh (override with VD_BIN=...).
# Submit:
#   cd ~/guohaojie/src/NavierStockes-3D/scripts && sbatch verify_decomp.sbatch
# Results (one line per run, ending in OK or MISMATCH):
#   grep ^RESULT $(ls -t /scratch/pr0208/guohaojie/logs/vd_vd_*.out | head -1)
source ~/guohaojie/runs/common.sh          # $M, precheck, cleanup
precheck
BIN=${VD_BIN:-/work/build/bin_gcc}
N=64; DT=5e-5; NS=10; REF=6.0041e-10

# R <ranks> <NS_DECOMP or -> <program>
R() {
    local np=$1 d=$2 prog=$3 envs="" out l2 dec st
    [ "$d" != "-" ] && envs="NS_DECOMP=$d"
    echo ">>> [$(date +%H:%M:%S)] $prog ranks=$np NS_DECOMP=$d | $prog $N $N $N $DT $NS 1 verify"
    out=$(srun --mpi=pmix --cpu-bind=cores --time=10 -N 1 -n $np -c 1 $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close $envs; exec $BIN/$prog $N $N $N $DT $NS 1 verify" 2>&1)
    echo "<<< exit=$?"; echo "$out"; echo
    dec=$(echo "$out" | grep -m1 '^decomposition:')
    l2=$(echo "$out" | grep -m1 'L2 error' | awk '{print $NF}')
    [ "$l2" = "$REF" ] && st=OK || st=MISMATCH
    printf "RESULT %-13s np=%-3s NS_DECOMP=%-6s L2=%-11s %-8s | %s\n" $prog $np $d "${l2:-none}" $st "$dec"
}

R 8 - ns_fftw
for d in slab pencil auto; do R 8 $d ns_accfft; done;      R 128 auto ns_accfft
for d in slab pencil auto; do R 8 $d ns_p3dfft; done;      R 128 auto ns_p3dfft
for d in slab pencil brick auto; do R 8 $d ns_heffte_v2; done; R 128 auto ns_heffte_v2
cleanup
