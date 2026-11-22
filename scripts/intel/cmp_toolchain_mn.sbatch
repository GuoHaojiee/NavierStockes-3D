#!/bin/bash
#SBATCH -J ctmn
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/ctmn_%x_%j.out
# cmp_toolchain_mn <nodes 1|2|4> <grid 512|1024|dry> — GCC toolchain (GCC container, Open MPI) vs Intel toolchain
#   (v2 image, Intel MPI), the four CPU solvers, slab, FFTW MEASURE. The job must have that many nodes:
#   sbatch -N <nodes> ... cmp_toolchain_mn.sbatch <nodes> <grid>
#   gcc    srun --mpi=pmix; /work/build/bin_gcc/{ns_fftw,ns_accfft,ns_heffte_v2}, /work/build/bin_p3d/gcc/ns_p3dfft;
#          Open MPI settings: GCC_COLL=def|t4 (t4 = tuned alltoall algorithm 4 with dynamic rules, ucc and hcoll off;
#          alltoallv stays default) and CTM_GCC_ENV="VAR=val ..." (e.g. UCX_NET_DEVICES=...)
#   intel  1 node: mpirun inside the image (fork + shm); 2, 4 nodes: irun_mn (intel_mn_env.sh, E6d: pmi2, mlx,
#          HPC-X UCX 1.17); settings from IMN_* (IMN_ALLTOALL, IMN_ALLTOALLV, IMN_UCX_TLS, IMN_EXTRA, IMN_PIN ...);
#          KMP_BLOCKTIME=0; /work/build/bin_intel/{ns_fftw,ns_accfft}, ns_heffte_v2_mkl + NS_HEFFTE_REORDER=1,
#          /work/build/bin_p3d/intel/ns_p3dfft
#   per node 128x1 and 64x2; 3 repetitions (outer loop), toolchains alternate (gcc, intel, gcc, intel ...)
#   grids: 512   512^3,  dt=1e-6, 10 steps, energy 1.7950250065e+08
#          1024  1024^3, dt=1e-6, 10 steps, energy 1.4360200052e+09
#          dry   64^3,   dt=5e-5, 10 steps, energy 3.3526466678e+05, 1 run per library and toolchain,
#                1 node 64x2, 2 nodes 32x4 per node, 4 nodes 16x8 (slab: ranks <= 64); CSV toolchain_mn_dry.csv
# Reliability (mn_lib.sh): job-specific container names (created once per node at the start, removed at the end);
#   every run has a time limit of 3 x its expected time (at least 3 min) and is stopped at once on "container start
#   failed", "enroot-switchroot", "inconsistent client PMI version" or "internal_Init_thread", then repeated once
#   (RETRY line, retries column).
# Every run appends one line to ~/guohaojie/results/toolchain_mn.csv:
#   commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,step_s,fft_s,plan_s,energy,
#   energy_ok,ucx_version,retries,exit_code,ucx_reason
#   (ucx_version: libucp really loaded by MPI rank 0, ucx_probe.c; "none" for single-node shm; the reason when unknown)
# Time estimate: 4 libraries x 2 toolchains x 2 layouts x 3 repetitions = 48 runs per (nodes, grid); expected time per
#   run (mn_lib.sh table; container start + MEASURE planning + 10 steps):
#     grid  nodes  per run   total     sbatch -t (x1.3)
#     512   1      2.5 min   2.0 h     02:40:00
#     512   2      2 min     1.6 h     02:10:00
#     512   4      1.5 min   1.2 h     01:40:00
#     1024  1      15 min    12 h      16:00:00
#     1024  2      10 min    8 h       10:30:00
#     1024  4      6 min     4.8 h     06:30:00
#     dry   1, 2   1 min     8 runs    00:30:00
#   All below 20 h, so one job per (nodes, grid). To split anyway: CTM_LIBS="fftw accfft" / "p3dfft heffte".
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/intel
#   sbatch -N 1 -J ctmn_dry1 -t 00:30:00 cmp_toolchain_mn.sbatch 1 dry
#   sbatch -N 4 -J ctmn_1024_4 -t 06:30:00 cmp_toolchain_mn.sbatch 4 1024
#   bash toolchain_mn_parse.sh ; python3 plot_toolchain_mn.py
# Options: CTM_LIBS, CTM_LAYOUTS, CTM_REP (3), CTM_CSV, CTM_IMAGE (v2), MN_TMO_SCALE (1), GCC_COLL (def), CTM_GCC_ENV.
NODES=${1:?usage: sbatch -N <n> cmp_toolchain_mn.sbatch <1|2|4> <512|1024|dry>}
GRID=${2:?usage: sbatch -N <n> cmp_toolchain_mn.sbatch <1|2|4> <512|1024|dry>}
export INTEL_IMAGE=${CTM_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/mn_lib.sh
precheck
[ "${SLURM_JOB_NUM_NODES:-0}" = "$NODES" ] || { echo "ERROR: the job has ${SLURM_JOB_NUM_NODES:-?} nodes, argument says $NODES (sbatch -N $NODES ...)"; exit 1; }
case $GRID in
dry)  N=64;   DT=5e-5; NS=10; EREF=3.3526466678e+05; REP=1
      case $NODES in 1) LAY="64x2";; 2) LAY="32x4";; *) LAY="16x8";; esac
      CSV=${CTM_CSV:-$HOME/guohaojie/results/toolchain_mn_dry.csv} ;;
512)  N=512;  DT=1e-6; NS=10; EREF=1.7950250065e+08; REP=${CTM_REP:-3}; LAY="128x1 64x2"; CSV=${CTM_CSV:-$HOME/guohaojie/results/toolchain_mn.csv} ;;
1024) N=1024; DT=1e-6; NS=10; EREF=1.4360200052e+09; REP=${CTM_REP:-3}; LAY="128x1 64x2"; CSV=${CTM_CSV:-$HOME/guohaojie/results/toolchain_mn.csv} ;;
*) echo "ERROR: grid 512|1024|dry"; exit 1 ;;
esac
LAYOUTS=${CTM_LAYOUTS:-$LAY}
LIBS=${CTM_LIBS:-"fftw accfft p3dfft heffte"}
GCC_COLL=${GCC_COLL:-def}
GENV="$(mn_gcc_coll $GCC_COLL) ${CTM_GCC_ENV:-}"
[[ "$GENV" == *BAD_GCC_COLL* ]] && { echo "ERROR: GCC_COLL=$GCC_COLL (def|t4)"; exit 1; }
COMMON="NS_FFTW_PLAN=measure NS_DECOMP=slab"
GSET="GCC_COLL=$GCC_COLL${CTM_GCC_ENV:+ $CTM_GCC_ENV}"
if [ $NODES = 1 ]; then ISET="single-node mpirun fork+shm"
else ISET="E6d$(for v in IMN_ALLTOALL IMN_ALLTOALLV IMN_UCX_TLS IMN_EXTRA IMN_PIN IMN_UCX_DIR IMN_NET_DEVICES; do [ -n "${!v+x}" ] && printf ' %s=%s' $v "${!v}"; done)"; fi
TMP=$SCR/logs/ctmn_$SLURM_JOB_ID.d; mkdir -p $TMP "$(dirname "$CSV")"
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)
NODELIST=$(echo "${SLURM_JOB_NODELIST:-?}" | tr ',' ';')
KEEP="^decomposition:|process grid|heFFTe backend|planner flag|P3DFFT library|Running on|Threads \(runtime\)|planning time|plan time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|abort|intel_mn_env:|GCCENV:|UCXPROBE|MPI startup\(\): +(Intel|libfabric provider)"
[ -f "$CSV" ] || echo "commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,step_s,fft_s,plan_s,energy,energy_ok,ucx_version,retries,exit_code,ucx_reason" > "$CSV"

echo "cmp_toolchain_mn: nodes=$NODES ($SLURM_JOB_NODELIST) grid=$GRID N=$N dt=$DT steps=$NS energy_ref=$EREF reps=$REP layouts(per node)=$LAYOUTS libs=$LIBS"
echo "commit $COMMIT   csv $CSV"
echo "gcc settings:   $GSET   -> env: ${GENV:-<run_env.sh only>}"
echo "intel settings: $ISET"
for b in bin_gcc/ns_fftw bin_gcc/ns_accfft bin_gcc/ns_heffte_v2 bin_p3d/gcc/ns_p3dfft bin_intel/ns_fftw bin_intel/ns_accfft bin_intel/ns_heffte_v2_mkl bin_p3d/intel/ns_p3dfft; do
    [ -x $SCR/build/$b ] && echo "  $(ls -l --time-style=+%F_%R $SCR/build/$b | awk '{print $6, $7}')" || echo "  MISSING $SCR/build/$b"
done
[ $NODES -gt 1 ] && { [ -f $SCR/opt_ucx/hpcx/lib/libucp.so.0 ] || echo "WARNING: HPC-X UCX not staged (stage_hpcx_ucx.sh)"; }
job_containers_init
mn_init $TMP /work/logs/ctmn_$SLURM_JOB_ID.d

bin_of() {   # <toolchain> <lib>
    case $1:$2 in
    gcc:fftw) echo /work/build/bin_gcc/ns_fftw;; gcc:accfft) echo /work/build/bin_gcc/ns_accfft;;
    gcc:p3dfft) echo /work/build/bin_p3d/gcc/ns_p3dfft;; gcc:heffte) echo /work/build/bin_gcc/ns_heffte_v2;;
    intel:fftw) echo /work/build/bin_intel/ns_fftw;; intel:accfft) echo /work/build/bin_intel/ns_accfft;;
    intel:p3dfft) echo /work/build/bin_p3d/intel/ns_p3dfft;; intel:heffte) echo /work/build/bin_intel/ns_heffte_v2_mkl;;
    esac
}
name_of() { case $1 in fftw) echo FFTW;; accfft) echo AccFFT;; p3dfft) echo P3DFFT;; heffte) echo heFFTe;; esac; }

# X <toolchain> <lib> <layout>
X() {
    local tc=$1 lib=$2 L=$3 ppn=${3%x*} nt=${3#*x} bin envs label raw tmo tl set
    bin=$(bin_of $tc $lib); envs=$COMMON
    [ $tc = intel ] && [ $lib = heffte ] && envs="$envs NS_HEFFTE_REORDER=1"
    label="$(name_of $lib)-${N}_${tc}_n${NODES}_$L"; raw=$TMP/$label.r$rep.raw
    tmo=$(mn_timeout $GRID $NODES $L); tl=$(( tmo / 60 + 2 ))
    if [ $tc = gcc ]; then set=$GSET; envs="$envs $GENV"; else set=$ISET; fi
    echo ">>> [$(date +%H:%M:%S)] $label ranks=$((NODES * ppn)) threads=$nt plan=measure | rep=$rep limit=${tmo}s $set | $envs $bin $N $N $N $DT $NS $nt bench"
    mn_run $label $raw $tmo mn_$tc $NODES $ppn $nt $tl "$envs" $bin $N $N $N $DT $NS $nt bench
    grep -E "$KEEP" $raw | awk '!/UCXPROBE/ || !u++'
    mn_extract $raw $EREF
    echo "UCX: $label rep=$rep version=$MN_UCXV${MN_UCXWHY:+ reason=$MN_UCXWHY}"
    if [ $tc = intel ] && [ $NODES -gt 1 ] && [ "$MN_UCXV" != 1.17.0 ]; then echo "WARNING: UCX $MN_UCXV loaded, expected 1.17.0 ($label rep=$rep${MN_UCXWHY:+, $MN_UCXWHY})"; fi
    [ "$MN_EOK" = no ] && echo "WARNING: kinetic energy ${MN_EN:-missing}, reference $EREF"
    echo "<<< exit=$MN_RC"
    echo "$COMMIT,$SLURM_JOB_ID,$NODELIST,$tc,$(echo "$set" | tr ',' ';'),$(name_of $lib),$N,$NODES,$ppn,$nt,$rep,$MN_STEP,$MN_FFT,$MN_PLAN,$MN_EN,$MN_EOK,$MN_UCXV,$MN_RETRIES,$MN_RC,$MN_UCXWHY" \
        | tee -a "$CSV" | sed 's/^/CSV /'
    echo
}

for rep in $(seq 1 $REP); do
    echo "=============== repetition $rep"
    for L in $LAYOUTS; do
        for lib in $LIBS; do X gcc $lib $L; X intel $lib $L; done
    done
done
echo "=============== this job's CSV lines: $(grep -c ",$SLURM_JOB_ID," "$CSV")   retries: $(grep -c '^RETRY' $SCR/logs/ctmn_${SLURM_JOB_NAME}_$SLURM_JOB_ID.out 2>/dev/null)"
cleanup_all
