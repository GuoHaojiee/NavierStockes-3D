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
#   (512 ranks x 1 thread, 256 ranks x 2 threads); bench mode; 2 repetitions, repetition = OUTER loop, the
#   toolchains alternate (gcc, intel, gcc, intel ...).
#     gcc    /work/build/bin_gcc/{ns_fftw,ns_accfft,ns_heffte_v2}, /work/build/bin_p3d/gcc/ns_p3dfft
#     intel  /work/build/bin_intel/{ns_fftw,ns_accfft}, ns_heffte_v2_mkl + NS_HEFFTE_REORDER=1,
#            /work/build/bin_p3d/intel/ns_p3dfft
#   Both sides can be tuned the same way (results of hello_2node R/RT/E6/E7):
#     GCC    every OMPI_MCA_* variable of the sbatch environment is passed to the ranks unchanged, AFTER run_env.sh
#            (e.g. --export=ALL,OMPI_MCA_coll_tuned_use_dynamic_rules=1,OMPI_MCA_coll_tuned_alltoall_algorithm=2);
#            CT4_GCC_ENV="UCX_NET_DEVICES=..." adds other variables; GCC_COLL=def|t4 (t4: tuned alltoall algorithm 4,
#            dynamic rules, ucc and hcoll off; alltoallv default). All of them are printed at the start.
#   Reliability (mn_lib.sh): job-specific container names, time limit per run (3 x expected, >= 3 min), stop and
#   one retry on a broken container / MPI start ("RETRY" lines, counted by ct4_parse.sh).
#     Intel  intel_mn_env.sh switches IMN_ALLTOALL (3), IMN_ALLTOALLV (unset), ...; CT4_INTEL_A2A="3 def" runs the
#            Intel side once per I_MPI_ADJUST_ALLTOALL setting (labels intel-a3, intel-def; one setting: label intel).
#   Argument = grid (one grid per job, the parser needs one reference energy per log):
#     dry    64^3, dt=5e-5, 10 steps, energy 3.3526466678e+05, one run per library and toolchain, 4 nodes x 16x8
#            (slab needs ranks <= N: 64 ranks)  — run this first
#     512    512^3, dt=1e-6, 10 steps, energy 1.7950250065e+08
#     1024   1024^3, dt=1e-6, 10 steps, energy 1.4360200052e+09
#   UCX: MPI rank 0 of every run prints the UCX it really loaded (scripts/cpu_final/ucx_probe.c via LD_PRELOAD); the
#   log gets "UCX: <label> rep=<r> version=<v>" and, for Intel on more than one node without 1.17.0,
#   "WARNING: UCX <v> loaded, expected 1.17.0".
# Estimate: 4 libraries x 2 toolchains x 2 layouts x 2 repetitions = 32 runs per grid.
#   512^3 : ~1-1.5 min per run (container start on 4 nodes, MEASURE planning, 10 short steps) -> 35-50 min
#   1024^3: ~2-4 min per run (planning and memory 8x larger, steps ~1-2 s)                   -> 65-130 min
#   together 1.7-3 h, too close to 3 h: one job per grid.
#     sbatch -J ct4dry   -t 00:30:00 cmp_toolchain_4n.sbatch dry
#     sbatch -J ct4_512  -t 01:30:00 cmp_toolchain_4n.sbatch 512
#     sbatch -J ct4_1024 -t 03:00:00 cmp_toolchain_4n.sbatch 1024
# Results:  bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4_512_*.out | head -1)
# Options (environment at sbatch time): CT4_REP (2), CT4_LAYOUTS ("128x1 64x2"), CT4_LIBS ("fftw accfft p3dfft heffte"),
#   CT4_IMAGE (v2), CT4_STEPS, CT4_INTEL_A2A, CT4_GCC_ENV, CT4_PRINT_PLAN=1 (FFTW prints its plans, NS_PRINT_PLAN=1),
#   the irun_mn switches IMN_ALLTOALLV (1|2), IMN_ALLTOALL (3), IMN_PIN (slurm|intel), and OMPI_MCA_*.
GRID=${1:?usage: sbatch cmp_toolchain_4n.sbatch dry|512|1024}
export INTEL_IMAGE=${CT4_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/mn_lib.sh
precheck
NN=${SLURM_JOB_NUM_NODES:-4}
case $GRID in
dry)  N=64;   DT=5e-5; NS=${CT4_STEPS:-10}; EREF=3.3526466678e+05; REP=1;             LAYOUTS=${CT4_LAYOUTS:-"16x8"} ;;
512)  N=512;  DT=1e-6; NS=${CT4_STEPS:-10}; EREF=1.7950250065e+08; REP=${CT4_REP:-2}; LAYOUTS=${CT4_LAYOUTS:-"128x1 64x2"} ;;
1024) N=1024; DT=1e-6; NS=${CT4_STEPS:-10}; EREF=1.4360200052e+09; REP=${CT4_REP:-2}; LAYOUTS=${CT4_LAYOUTS:-"128x1 64x2"} ;;
*) echo "unknown grid $GRID (dry|512|1024)"; exit 1 ;;
esac
[ "$NS" != 10 ] && [ $GRID != dry ] && echo "WARNING: CT4_STEPS=$NS: the reference energy $EREF is for 10 steps"
LIBS=${CT4_LIBS:-"fftw accfft p3dfft heffte"}
INTEL_A2A=${CT4_INTEL_A2A:-""}          # "" = one Intel run with intel_mn_env.sh defaults (label intel)
GBIN=/work/build/bin_gcc; IBIN=/work/build/bin_intel; PBIN=/work/build/bin_p3d
COMMON="NS_FFTW_PLAN=measure NS_DECOMP=slab"; [ "$CT4_PRINT_PLAN" = 1 ] && COMMON="$COMMON NS_PRINT_PLAN=1"
KEEP="^decomposition:|process grid|MPI reshapes|heFFTe backend|heFFTe plan options|planner flag|P3DFFT library|Running on|Threads \(runtime\)|planning time|plan time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|abort|intel_mn_env:|BIND |UCXPROBE|OMPI_MCA_|FFTW PLAN: (----|.*mpi-)|MPI startup\(\): +(Rank|Intel|libfabric provider|[0-9]+ )"
TMP=$SCR/logs/ct4_$SLURM_JOB_ID.d; WTMP=/work/logs/ct4_$SLURM_JOB_ID.d; mkdir -p $TMP
# GCC side: OMPI_MCA_* of the sbatch environment (+ CT4_GCC_ENV), exported after run_env.sh
GCC_COLL=${GCC_COLL:-def}
GENV="$(mn_gcc_coll $GCC_COLL) $(env | grep -E '^OMPI_MCA_' | sort | tr '\n' ' ') ${CT4_GCC_ENV:-}"
[[ "$GENV" == *BAD_GCC_COLL* ]] && { echo "ERROR: GCC_COLL=$GCC_COLL (def|t4)"; exit 1; }

# job-specific containers, created once per node; UCX probe built in both containers (mn_lib.sh)
job_containers_init
mn_init $TMP $WTMP

# ucx_check <label> <side> <raw file>
ucx_check() {
    mn_extract $3 $EREF
    echo "UCX: $1 rep=$rep version=$MN_UCXV${MN_UCXWHY:+ reason=$MN_UCXWHY}"
    if [ $2 = intel ] && [ $NN -gt 1 ] && [ "$MN_UCXV" != 1.17.0 ]; then echo "WARNING: UCX $MN_UCXV loaded, expected 1.17.0 ($1 rep=$rep${MN_UCXWHY:+, $MN_UCXWHY})"; fi
}
# G <label> <ranks/node> <threads> "<extra env>" <binary>    GCC container, Open MPI, srun --mpi=pmix
G() {
    local label=$1 ppn=$2 nt=$3 envs="$COMMON $4" bin=$5 raw=$TMP/$1.r$rep.raw rc
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$((NN * ppn)) threads=$nt plan=measure | nodes=$NN ppn=$ppn $envs ${GENV:+[$GENV]} $bin $N $N $N $DT $NS $nt bench"
    local tmo; tmo=$(mn_timeout $GRID $NN ${ppn}x$nt)
    mn_run $label $raw $tmo mn_gcc $NN $ppn $nt $(( tmo / 60 + 2 )) "$envs $GENV" $bin $N $N $N $DT $NS $nt bench
    rc=$MN_RC
    grep -E "$KEEP" $raw; ucx_check $label gcc $raw
    echo "<<< exit=$rc"; echo
}
# I <label> <ranks/node> <threads> "<extra env>" <binary> [I_MPI_ADJUST_ALLTOALL: 3|def]    v2 image, irun_mn
I() {
    local label=$1 ppn=$2 nt=$3 envs="$COMMON $4" bin=$5 a2a=$6 raw=$TMP/$1.r$rep.raw rc imn
    case "$a2a" in "") imn=${IMN_ALLTOALL-3};; def) imn="";; *) imn=$a2a;; esac
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$((NN * ppn)) threads=$nt plan=measure | nodes=$NN ppn=$ppn KMP_BLOCKTIME=0 IMN_ALLTOALL=${imn:-default} $envs $bin $N $N $N $DT $NS $nt bench"
    local first=""; [ -z "$IFIRST_DONE" ] && { first=1; IFIRST_DONE=1; }
    local tmo; tmo=$(mn_timeout $GRID $NN ${ppn}x$nt)
    IMN_ALLTOALL=$imn IMN_DEBUG=${first:+4} IMN_SHOWBIND=$first \
        mn_run $label $raw $tmo mn_intel $NN $ppn $nt $(( tmo / 60 + 2 )) "$envs" $bin $N $N $N $DT $NS $nt bench
    rc=$MN_RC
    grep -E "$KEEP" $raw | awk '/MPI startup\(\): +[0-9]/ { if (++k > 4) next } 1'; ucx_check $label intel $raw
    echo "<<< exit=$rc"; echo
}
# II <Lib> <layout> <ppn> <nt> "<extra env>" <binary> — the Intel side, once per CT4_INTEL_A2A setting
II() {
    local s
    if [ -z "$INTEL_A2A" ]; then I $1-${N}_intel_$2 $3 $4 "$5" $6
    else for s in $INTEL_A2A; do
             if [ $s = def ]; then I $1-${N}_intel-def_$2 $3 $4 "$5" $6 def; else I $1-${N}_intel-a${s}_$2 $3 $4 "$5" $6 $s; fi
         done; fi
}
# R <lib> <layout> — one gcc run and the Intel run(s) of a library; label <Lib>-<N>_<toolchain>_<layout>
R() {
    local lib=$1 L=$2 ppn=${2%x*} nt=${2#*x}
    case $lib in
    fftw)   G FFTW-${N}_gcc_$L     $ppn $nt "" $GBIN/ns_fftw
            II FFTW   $L $ppn $nt "" $IBIN/ns_fftw ;;
    accfft) G AccFFT-${N}_gcc_$L   $ppn $nt "" $GBIN/ns_accfft
            II AccFFT $L $ppn $nt "" $IBIN/ns_accfft ;;
    p3dfft) G P3DFFT-${N}_gcc_$L   $ppn $nt "" $PBIN/gcc/ns_p3dfft
            II P3DFFT $L $ppn $nt "" $PBIN/intel/ns_p3dfft ;;
    heffte) G heFFTe-${N}_gcc_$L   $ppn $nt "" $GBIN/ns_heffte_v2
            II heFFTe $L $ppn $nt "NS_HEFFTE_REORDER=1" $IBIN/ns_heffte_v2_mkl ;;
    esac
}

echo "grid=$GRID N=$N dt=$DT steps=$NS energy_ref=$EREF reps=$REP layouts(per node)=$LAYOUTS libs=$LIBS nodes=$NN ($SLURM_JOB_NODELIST)"
echo "IMN: ALLTOALL=${IMN_ALLTOALL-3} ALLTOALLV=${IMN_ALLTOALLV:-default} PIN=${IMN_PIN:-slurm} CT4_INTEL_A2A=${INTEL_A2A:-<intel_mn_env default>}"
echo "GCC env (after run_env.sh): GCC_COLL=$GCC_COLL -> ${GENV:-<none: Open MPI defaults of run_env.sh>}"
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
