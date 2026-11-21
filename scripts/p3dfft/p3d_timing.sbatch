#!/bin/bash
#SBATCH -J p3dt
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/p3dt_%j.out
# p3d_timing — old vs rebuilt (-O3 + OpenMP) P3DFFT, GCC/NVHPC + Open MPI vs Intel + Intel MPI, one node.
#   512^3, dt=1e-6, 20 steps, bench; kinetic energy must be 1.7933833444e+08 (counted by the parser).
#   programs:  old-gcc    /work/build/bin_gcc/ns_p3dfft        (scripts/compile_all.sh, /opt/p3dfft_v2: -g, no -mp)
#              old-intel  /work/build/bin_intel/ns_p3dfft      (build_intel_solvers.sh, /work/opt_intel/p3dfft: -g)
#              new-gcc    /work/build/bin_p3d/gcc/ns_p3dfft    (build_p3dfft_opt.sh gcc:   -O3 -tp=zen2 -mp)
#              new-intel  /work/build/bin_p3d/intel/ns_p3dfft  (build_p3dfft_opt.sh intel: -O3 -march=core-avx2, -qopenmp)
#   decompositions slab, pencil (NS_DECOMP); layouts 128x1, 64x2; 3 repetitions, repetition = OUTER loop;
#   then once per program pure OpenMP 1x128 (slab, NUMA interleave: numactl --interleave=all in the GCC container,
#   /work/tools/numactl in the Intel container).
#   Labels: P3D-<decomp>_<old|new>-<gcc|intel>_<ranks>x<threads>   (pl_parse.sh / p3d_parse.sh).
# Time: 4 programs x 2 decompositions x 2 layouts x 3 = 48 runs of ~1-3 min (container start + MEASURE planning +
#   20 steps), + 4 OpenMP runs (old-gcc ~12 min at 33.6 s/step, the others a few minutes) = ~1.3-2.8 h -> 3 h.
#   To split it into two jobs:  sbatch --export=ALL,P3T_DECOMPS=slab   -t 01:45:00 p3d_timing.sbatch
#                               sbatch --export=ALL,P3T_DECOMPS=pencil,P3T_OMP=0 -t 01:30:00 p3d_timing.sbatch
# Options: P3T_N (512), P3T_STEPS (20), P3T_REP (3), P3T_LAYOUTS ("128x1 64x2"), P3T_DECOMPS ("slab pencil"),
#   P3T_PROGS ("old-gcc new-gcc old-intel new-intel"), P3T_OMP (1 = run the 1x128 OpenMP case), P3T_OMP_THREADS (128), P3T_NUMALIB
#   (directories with libnuma for /work/tools/numactl, default "/work/tools:/work/tools/lib").
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/p3dfft && sbatch p3d_timing.sbatch
#   bash p3d_parse.sh $(ls -t ~/guohaojie/logs/p3dt_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh     # $M, $MI, precheck, cleanup_all
precheck
N=${P3T_N:-512}; DT=1e-6; NS=${P3T_STEPS:-20}; REP=${P3T_REP:-3}
LAYOUTS=${P3T_LAYOUTS:-"128x1 64x2"}; DECOMPS=${P3T_DECOMPS:-"slab pencil"}
PROGS=${P3T_PROGS:-"old-gcc new-gcc old-intel new-intel"}; OMPRUN=${P3T_OMP:-1}
NUMALIB=${P3T_NUMALIB:-/work/tools:/work/tools/lib}
KEEP="^decomposition:|P3DFFT library|P3DFFT planner|Running on|OpenMP threads|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|abort|numactl|MPI startup\(\): +(Rank|Intel|[0-9]+ )"

bin_of() { case $1 in old-gcc) echo /work/build/bin_gcc/ns_p3dfft;; old-intel) echo /work/build/bin_intel/ns_p3dfft;;
                      new-gcc) echo /work/build/bin_p3d/gcc/ns_p3dfft;; new-intel) echo /work/build/bin_p3d/intel/ns_p3dfft;; esac; }

# G <label> <ranks> <threads> <decomp> <binary> [numa]     GCC container, Open MPI, srun launches the ranks
G() {
    local label=$1 np=$2 nt=$3 d=$4 bin=$5 numa="" tl=20
    [ -n "$6" ] && { numa="numactl --interleave=all"; tl=30; }
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$np threads=$nt plan=measure | NS_DECOMP=$d $numa $bin $N $N $N $DT $NS $nt bench"
    srun --mpi=pmix --cpu-bind=cores --time=$tl -N 1 -n $np -c $nt $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close NS_DECOMP=$d; \
        [ -n '$numa' ] && { command -v numactl >/dev/null || { echo 'numactl not found in the GCC container'; exit 9; }; }; \
        exec $numa $bin $N $N $N $DT $NS $nt bench" 2>&1 | grep -E "$KEEP"
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}
# I <label> <ranks> <threads> <decomp> <binary> [numa]     Intel container, Intel MPI, mpirun (fork + shm) in one task
I() {
    local label=$1 np=$2 nt=$3 d=$4 bin=$5 numa="" tl=20 dbg=""
    [ -n "$6" ] && { numa="/work/tools/numactl --interleave=all"; tl=30; }
    [ -z "$IDBG_DONE" ] && { dbg="I_MPI_DEBUG=4"; IDBG_DONE=1; }       # pin map of the first Intel run only
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$np threads=$nt plan=measure | NS_DECOMP=$d $numa $bin $N $N $N $DT $NS $nt bench"
    srun -N 1 -n 1 -c 128 --cpu-bind=none --time=$tl $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
        export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$nt:compact OMP_NUM_THREADS=$nt OMP_PLACES=cores \
               OMP_PROC_BIND=close KMP_BLOCKTIME=0 NS_DECOMP=$d $dbg; \
        [ -n '$numa' ] && { export LD_LIBRARY_PATH=$NUMALIB:\$LD_LIBRARY_PATH; /work/tools/numactl --interleave=all true || { echo 'numactl: /work/tools/numactl does not run (libnuma? set P3T_NUMALIB)'; exit 9; }; }; \
        exec mpirun -np $np $numa $bin $N $N $N $DT $NS $nt bench" 2>&1 | grep -E "$KEEP" | awk '/MPI startup\(\): +[0-9]/ { if (++k > 4) next } 1'
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}
# P <program> <decomp> <ranks> <threads> [numa]
P() {
    local p=$1 d=$2 np=$3 nt=$4 label="P3D-${2}_${1}_${3}x${4}"
    case $p in *-gcc) G $label $np $nt $d $(bin_of $p) $5;; *-intel) I $label $np $nt $d $(bin_of $p) $5;; esac
}

echo "N=$N steps=$NS reps=$REP layouts=$LAYOUTS decomps=$DECOMPS programs=$PROGS omp=$OMPRUN node=$(hostname)"
for p in $PROGS; do b=$(bin_of $p); echo "  $p: $(ls -l --time-style=+%F_%R $SCR/${b#/work/} 2>&1 | awk '{print $6, $7}')"; done
for rep in $(seq 1 $REP); do
  echo "=============== repetition $rep"
  for L in $LAYOUTS; do
    for d in $DECOMPS; do
      for p in $PROGS; do P $p $d ${L%x*} ${L#*x}; done
    done
  done
done
if [ "$OMPRUN" = 1 ]; then
  echo "=============== pure OpenMP 1x128 (once per program, NUMA interleave)"
  for p in $PROGS; do P $p slab 1 ${P3T_OMP_THREADS:-128} numa; done
fi
cleanup_all
