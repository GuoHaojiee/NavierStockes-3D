#!/bin/bash
#SBATCH -J ct4
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/ct4_%x_%j.out
# cmp_toolchain_4n — GCC toolchain (GCC container, Open MPI, srun --mpi=pmix) vs Intel toolchain (v2 image, Intel MPI
#   started by irun_mn = hello_2node E6d_a2a3 settings, KMP_BLOCKTIME=0), 4 nodes, the CPU solvers.
#   Libraries: FFTW, AccFFT, P3DFFT, heFFTe; NS_DECOMP=slab, NS_FFTW_PLAN=measure; per node 128x1 and 64x2
#   (512 ranks x 1 thread, 256 ranks x 2 threads); bench mode; 2 repetitions, repetition = OUTER loop, the two
#   toolchains alternate (gcc, intel, gcc, intel ...).
#     gcc    /work/build/bin_gcc/{ns_fftw,ns_accfft,ns_heffte_v2}, /work/build/bin_p3d/gcc/ns_p3dfft
#     intel  /work/build/bin_intel/{ns_fftw,ns_accfft}, ns_heffte_v2_mkl + NS_HEFFTE_REORDER=1,
#            /work/build/bin_p3d/intel/ns_p3dfft
#   Argument = grid (one grid per job, the parser needs one reference energy per log):
#     dry    64^3, dt=5e-5, 10 steps, energy 3.3526466678e+05, one run per library and toolchain, 4 nodes x 16x8
#            (slab needs ranks <= N: 64 ranks)  — run this first
#     512    512^3, dt=1e-6, 20 steps, energy 1.7933833444e+08
#     1024   1024^3, dt=1e-6, 10 steps, energy 1.4360200052e+09
# Estimate: 4 libraries x 2 toolchains x 2 layouts x 2 repetitions = 32 runs per grid.
#   512^3 : ~1-1.5 min per run (container start on 4 nodes, MEASURE planning, 20 short steps) -> 35-50 min
#   1024^3: ~2-4 min per run (planning and memory 8x larger, steps ~1-2 s)                   -> 65-130 min
#   together 1.7-3 h, too close to 3 h: one job per grid.
#     sbatch -J ct4dry   -t 00:30:00 cmp_toolchain_4n.sbatch dry
#     sbatch -J ct4_512  -t 01:30:00 cmp_toolchain_4n.sbatch 512
#     sbatch -J ct4_1024 -t 03:00:00 cmp_toolchain_4n.sbatch 1024
# Results:  bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4_512_*.out | head -1)
# Options (environment at sbatch time): CT4_REP (2), CT4_LAYOUTS ("128x1 64x2"), CT4_LIBS ("fftw accfft p3dfft heffte"),
#   CT4_IMAGE (v2), CT4_STEPS, and the irun_mn switches IMN_ALLTOALLV (1|2), IMN_ALLTOALL (3), IMN_PIN (slurm|intel).
GRID=${1:?usage: sbatch cmp_toolchain_4n.sbatch dry|512|1024}
export INTEL_IMAGE=${CT4_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
NN=${SLURM_JOB_NUM_NODES:-4}
case $GRID in
dry)  N=64;   DT=5e-5; NS=${CT4_STEPS:-10}; EREF=3.3526466678e+05; REP=1;             LAYOUTS=${CT4_LAYOUTS:-"16x8"} ;;
512)  N=512;  DT=1e-6; NS=${CT4_STEPS:-20}; EREF=1.7933833444e+08; REP=${CT4_REP:-2}; LAYOUTS=${CT4_LAYOUTS:-"128x1 64x2"} ;;
1024) N=1024; DT=1e-6; NS=${CT4_STEPS:-10}; EREF=1.4360200052e+09; REP=${CT4_REP:-2}; LAYOUTS=${CT4_LAYOUTS:-"128x1 64x2"} ;;
*) echo "unknown grid $GRID (dry|512|1024)"; exit 1 ;;
esac
LIBS=${CT4_LIBS:-"fftw accfft p3dfft heffte"}
GBIN=/work/build/bin_gcc; IBIN=/work/build/bin_intel; PBIN=/work/build/bin_p3d
COMMON="NS_FFTW_PLAN=measure NS_DECOMP=slab"
KEEP="^decomposition:|process grid|MPI reshapes|heFFTe backend|heFFTe plan options|planner flag|P3DFFT library|planning time|plan time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|abort|intel_mn_env:|BIND |MPI startup\(\): +(Rank|Intel|libfabric provider|[0-9]+ )"
TL=$([ $GRID = 1024 ] && echo 30 || echo 15)          # srun time limit per run, minutes

# G <label> <ranks/node> <threads> "<extra env>" <binary>    GCC container, Open MPI, srun --mpi=pmix
G() {
    local label=$1 ppn=$2 nt=$3 envs="$COMMON $4" bin=$5
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$((NN * ppn)) threads=$nt plan=measure | nodes=$NN ppn=$ppn $envs $bin $N $N $N $DT $NS $nt bench"
    srun --mpi=pmix --cpu-bind=cores --time=$TL -N $NN --ntasks-per-node=$ppn -c $nt $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close $envs; exec $bin $N $N $N $DT $NS $nt bench" 2>&1 | grep -E "$KEEP"
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}
# I <label> <ranks/node> <threads> "<extra env>" <binary>    v2 image, Intel MPI, irun_mn (E6d settings)
I() {
    local label=$1 ppn=$2 nt=$3 envs="$COMMON $4" bin=$5
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$((NN * ppn)) threads=$nt plan=measure | nodes=$NN ppn=$ppn KMP_BLOCKTIME=0 $envs $bin $N $N $N $DT $NS $nt bench"
    local first=""; [ -z "$IFIRST_DONE" ] && { first=1; IFIRST_DONE=1; }
    IRUN_SRUN_OPTS="--time=$TL" IMN_DEBUG=${first:+4} IMN_SHOWBIND=$first \
    irun_mn $NN $ppn $nt "export KMP_BLOCKTIME=0 MKL_NUM_THREADS=$nt $envs; exec $bin $N $N $N $DT $NS $nt bench" 2>&1 \
        | grep -E "$KEEP" | awk '/MPI startup\(\): +[0-9]/ { if (++k > 4) next } 1'
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}
# R <lib> <layout> — one gcc run and one intel run of a library; label <Lib>-<N>_<toolchain>_<layout>
R() {
    local lib=$1 L=$2 ppn=${2%x*} nt=${2#*x}
    case $lib in
    fftw)   G FFTW-${N}_gcc_$L     $ppn $nt "" $GBIN/ns_fftw
            I FFTW-${N}_intel_$L   $ppn $nt "" $IBIN/ns_fftw ;;
    accfft) G AccFFT-${N}_gcc_$L   $ppn $nt "" $GBIN/ns_accfft
            I AccFFT-${N}_intel_$L $ppn $nt "" $IBIN/ns_accfft ;;
    p3dfft) G P3DFFT-${N}_gcc_$L   $ppn $nt "" $PBIN/gcc/ns_p3dfft
            I P3DFFT-${N}_intel_$L $ppn $nt "" $PBIN/intel/ns_p3dfft ;;
    heffte) G heFFTe-${N}_gcc_$L   $ppn $nt "" $GBIN/ns_heffte_v2
            I heFFTe-${N}_intel_$L $ppn $nt "NS_HEFFTE_REORDER=1" $IBIN/ns_heffte_v2_mkl ;;
    esac
}

echo "grid=$GRID N=$N dt=$DT steps=$NS energy_ref=$EREF reps=$REP layouts(per node)=$LAYOUTS libs=$LIBS nodes=$NN ($SLURM_JOB_NODELIST)"
echo "IMN: ALLTOALL=${IMN_ALLTOALL-3} ALLTOALLV=${IMN_ALLTOALLV:-default} PIN=${IMN_PIN:-slurm}"
for b in $SCR/build/bin_gcc/ns_fftw $SCR/build/bin_gcc/ns_accfft $SCR/build/bin_gcc/ns_heffte_v2 $SCR/build/bin_p3d/gcc/ns_p3dfft \
         $SCR/build/bin_intel/ns_fftw $SCR/build/bin_intel/ns_accfft $SCR/build/bin_intel/ns_heffte_v2_mkl $SCR/build/bin_p3d/intel/ns_p3dfft; do
    [ -x $b ] && echo "  $(ls -l --time-style=+%F_%R $b | awk '{print $6, $7}')" || echo "  MISSING $b"
done
[ -f $SCR/opt_ucx/hpcx/lib/libucp.so.0 ] || echo "WARNING: $SCR/opt_ucx/hpcx not staged (stage_hpcx_ucx.sh): Intel runs use the image's UCX 1.12.1"
for rep in $(seq 1 $REP); do
  echo "=============== repetition $rep"
  for L in $LAYOUTS; do
    for lib in $LIBS; do R $lib $L; done
  done
done
cleanup_all
