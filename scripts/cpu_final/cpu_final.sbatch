#!/bin/bash
#SBATCH -J cpuf
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 02:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/cpuf_%x_%j.out
# cpu_final — the final CPU measurements: Intel toolchain only (icpx + Intel MPI, v2 image), slab, 512^3.
#   Libraries   FFTW    /work/build/bin_intel/ns_fftw
#               AccFFT  /work/build/bin_intel/ns_accfft
#               P3DFFT  /work/build/bin_p3d/intel/ns_p3dfft
#               heFFTe  /work/build/bin_intel/ns_heffte_v2_mkl, NS_HEFFTE_REORDER=1
#   NS_DECOMP=slab NS_FFTW_PLAN=measure KMP_BLOCKTIME=0 MKL_NUM_THREADS=<threads>; 512^3, dt=1e-6, 10 steps, bench,
#   reference kinetic energy 1.7950250065e+08. 3 repetitions, repetition = OUTER loop.
#   1 node   128x1, 64x2, 1x128 (pure OpenMP): one srun task, mpirun inside the image (fork + shm), as the single-node
#            Intel runs of scripts/intel/cmp_toolchain.sbatch; 1x128 with /work/tools/numactl --interleave=all,
#            OMP_PLACES=cores OMP_PROC_BIND=spread.
#   2, 4     per node 128x1 and 64x2, irun_mn (intel_common.sh, hello_2node E6d settings: pmi2, mlx, HPC-X UCX 1.17).
# Arguments: <nodes 1|2|4> [dry]. The job must have that many nodes: sbatch -N <nodes> ... cpu_final.sbatch <nodes>
#   dry  64^3, dt=5e-5, 10 steps, energy 3.3526466678e+05, one run per library; 1 node: 64x2 and 1x128,
#        2 nodes: 32x4 per node (slab: ranks <= 64); results go to cpu_final_dry.csv
# Every run appends one CSV line to ~/guohaojie/results/cpu_final.csv (dry: cpu_final_dry.csv; CPF_CSV=...):
#   commit,jobid,nodelist,lib,nodes,ranks_per_node,threads,rep,step_s,fft_s,plan_s,energy,energy_ok,ucx_version,
#   ucx_warn,exit_code,omp_threads,mkl_threads,retries,ucx_reason
#   Reliability (scripts/intel/mn_lib.sh): job-specific container names, time limit per run (3 x expected, >= 3 min),
#   a broken container / MPI start is stopped and repeated once (RETRY line, retries column).
#   ucx_version: the libucp really mapped by MPI rank 0 (LD_PRELOAD scripts/cpu_final/ucx_probe.so), "none" when
#   no UCX was loaded (single node, shm). Multi-node runs expect 1.17.0, otherwise ucx_warn=WARN and the log says
#   "WARNING: UCX <version> loaded, expected 1.17.0". omp_threads / mkl_threads: printed by the solver at run time.
# Time estimate (512^3, 10 steps; container start + MEASURE planning + steps):
#   1 node : 4 libs x 3 layouts x 3 = 36 runs x 1.5-3 min (1x128 slower to plan)  -> 55-110 min -> -t 02:30:00
#   2 nodes: 4 x 2 x 3 = 24 runs x 1-2 min                                         -> 25-50 min  -> -t 01:15:00
#   4 nodes: 24 runs x 1-1.5 min                                                   -> 25-40 min  -> -t 01:00:00
#   dry    : 8 (1 node) or 4 (2 nodes) short runs                                  ->  ~10 min   -> -t 00:20:00
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/cpu_final
#   sbatch -N 1 -J cpuf1 -t 02:30:00 cpu_final.sbatch 1
#   bash cpu_final_parse.sh ~/guohaojie/results/cpu_final.csv
# Options: CPF_REP (3), CPF_LIBS ("fftw accfft p3dfft heffte"), CPF_LAYOUTS, CPF_IMAGE (v2), CPF_CSV,
#   CPF_NUMALIB (directories with libnuma for /work/tools/numactl, default "/work/tools:/work/tools/lib"),
#   and the irun_mn switches (IMN_ALLTOALLV ...).
NODES=${1:?usage: sbatch -N <n> cpu_final.sbatch <1|2|4> [dry]}
MODE=${2:-final}
export INTEL_IMAGE=${CPF_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/mn_lib.sh
precheck
[ "${SLURM_JOB_NUM_NODES:-0}" = "$NODES" ] || { echo "ERROR: the job has ${SLURM_JOB_NUM_NODES:-?} nodes, argument says $NODES (sbatch -N $NODES ...)"; exit 1; }
case $NODES in 1|2|4) ;; *) echo "ERROR: nodes must be 1, 2 or 4"; exit 1 ;; esac

if [ "$MODE" = dry ]; then
    N=64; DT=5e-5; NS=10; EREF=3.3526466678e+05; REP=1
    if [ $NODES = 1 ]; then LAYOUTS="64x2 1x128"; else LAYOUTS="$((64 / NODES))x$((128 * NODES / 64))"; fi
    CSV=${CPF_CSV:-$HOME/guohaojie/results/cpu_final_dry.csv}
else
    N=512; DT=1e-6; NS=10; EREF=1.7950250065e+08; REP=${CPF_REP:-3}
    if [ $NODES = 1 ]; then LAYOUTS="128x1 64x2 1x128"; else LAYOUTS="128x1 64x2"; fi
    CSV=${CPF_CSV:-$HOME/guohaojie/results/cpu_final.csv}
fi
LAYOUTS=${CPF_LAYOUTS:-$LAYOUTS}
LIBS=${CPF_LIBS:-"fftw accfft p3dfft heffte"}
NUMALIB=${CPF_NUMALIB:-/work/tools:/work/tools/lib}
COMMON="NS_DECOMP=slab NS_FFTW_PLAN=measure"
IE="source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1"
TMP=$SCR/logs/cpuf_$SLURM_JOB_ID.d; WTMP=/work/logs/cpuf_$SLURM_JOB_ID.d; mkdir -p $TMP "$(dirname "$CSV")"
MN_NUMALIB=$NUMALIB
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)
NODELIST=$(echo "${SLURM_JOB_NODELIST:-?}" | tr ',' ';')
KEEP="^decomposition:|process grid|heFFTe backend|heFFTe plan options|planner flag|P3DFFT library|Running on|OpenMP threads|Threads \(runtime\)|planning time|plan time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|abort|UCXPROBE|intel_mn_env:|numactl|MPI startup\(\): +(Intel|libfabric provider)"
[ -f "$CSV" ] || echo "commit,jobid,nodelist,lib,nodes,ranks_per_node,threads,rep,step_s,fft_s,plan_s,energy,energy_ok,ucx_version,ucx_warn,exit_code,omp_threads,mkl_threads,retries,ucx_reason" > "$CSV"
# a CSV from before the retries column: extend the header only (old rows simply have no value there)
head -1 "$CSV" | grep -q ',retries' || sed -i '1s/$/,retries,ucx_reason/' "$CSV"

bin_of() { case $1 in fftw) echo /work/build/bin_intel/ns_fftw;; accfft) echo /work/build/bin_intel/ns_accfft;;
                      p3dfft) echo /work/build/bin_p3d/intel/ns_p3dfft;; heffte) echo /work/build/bin_intel/ns_heffte_v2_mkl;; esac; }
name_of() { case $1 in fftw) echo FFTW;; accfft) echo AccFFT;; p3dfft) echo P3DFFT;; heffte) echo heFFTe;; esac; }
extra_of() { [ $1 = heffte ] && echo "NS_HEFFTE_REORDER=1"; }

echo "cpu_final: mode=$MODE nodes=$NODES ($SLURM_JOB_NODELIST) N=$N dt=$DT steps=$NS energy_ref=$EREF reps=$REP layouts(per node)=$LAYOUTS"
echo "commit $COMMIT   image $INTEL_IMAGE   csv $CSV"
for l in $LIBS; do b=$(bin_of $l); echo "  $(name_of $l): $b  $(ls -l --time-style=+%F_%R $SCR/${b#/work/} 2>&1 | awk '{print $6}') $(extra_of $l)"; done
[ $NODES -gt 1 ] && { [ -f $SCR/opt_ucx/hpcx/lib/libucp.so.0 ] && echo "HPC-X UCX: $(grep -m1 version $SCR/opt_ucx/hpcx/VERSION.txt)" || echo "WARNING: $SCR/opt_ucx/hpcx not staged (stage_hpcx_ucx.sh)"; }
# job-specific containers created once per node; the UCX probe built in the image (mn_lib.sh)
job_containers_init
mn_init $TMP $WTMP
# numactl in the image (1x128): show that it runs with the library path
if [ $NODES = 1 ] && [[ " $LAYOUTS " == *" 1x"* ]]; then
    srun -N1 -n1 --time=2 $MI bash -c "export LD_LIBRARY_PATH=$NUMALIB:\$LD_LIBRARY_PATH; /work/tools/numactl --interleave=all true && echo 'numactl: /work/tools/numactl --interleave=all OK' || echo 'numactl: FAILED (set CPF_NUMALIB)'"
fi

# X <lib> <ranks per node> <threads> <rep>
X() {
    local lib=$1 ppn=$2 nt=$3 rep=$4 bin extra label raw rc bind numa=""
    bin=$(bin_of $lib); extra=$(extra_of $lib); label="$(name_of $lib)_n${NODES}_${ppn}x${nt}"
    raw=$TMP/$label.r$rep.raw
    echo ">>> [$(date +%H:%M:%S)] $label ranks=$((NODES * ppn)) threads=$nt plan=measure | rep=$rep nodes=$NODES $COMMON $extra $bin $N $N $N $DT $NS $nt bench"
    # time limit 3 x expected (>= 3 min); a broken container / MPI start is stopped and repeated once (mn_lib.sh)
    local tmo; tmo=$(mn_timeout $([ $MODE = dry ] && echo dry || echo 512) $NODES ${ppn}x$nt)
    mn_run $label $raw $tmo mn_intel $NODES $ppn $nt $(( tmo / 60 + 2 )) "$COMMON $extra" $bin $N $N $N $DT $NS $nt bench
    rc=$MN_RC
    grep -E "$KEEP" $raw
    echo "<<< exit=$rc"
    # ---- CSV line
    local step fft plan en eok uv uw ompt mklt
    mn_extract $raw $EREF
    step=$MN_STEP; fft=$MN_FFT; plan=$MN_PLAN; en=$MN_EN; eok=$MN_EOK; uv=$MN_UCXV
    echo "UCX: $label rep=$rep version=$uv${MN_UCXWHY:+ reason=$MN_UCXWHY}"
    uw=""
    if [ $NODES -gt 1 ] && [ "$uv" != 1.17.0 ]; then uw=WARN; echo "WARNING: UCX $uv loaded, expected 1.17.0${MN_UCXWHY:+ ($MN_UCXWHY)}"; fi
    ompt=$(grep -m1 -oE "omp_get_max_threads\(\)=[0-9]+" $raw | cut -d= -f2)
    mklt=$(grep -m1 -oE "mkl_get_max_threads\(\)=[0-9]+" $raw | cut -d= -f2)
    [ -n "$ompt" ] && [ "$ompt" != "$nt" ] && echo "WARNING: omp_get_max_threads()=$ompt, expected $nt"
    [ -n "$mklt" ] && [ "$mklt" != "$nt" ] && echo "WARNING: mkl_get_max_threads()=$mklt, expected $nt"
    [ "$eok" = no ] && echo "WARNING: kinetic energy ${en:-missing}, reference $EREF"
    echo "$COMMIT,$SLURM_JOB_ID,$NODELIST,$(name_of $lib),$NODES,$ppn,$nt,$rep,$step,$fft,$plan,$en,$eok,$uv,$uw,$rc,$ompt,$mklt,$MN_RETRIES,$MN_UCXWHY" | tee -a "$CSV" | sed 's/^/CSV /'
    echo
}

for rep in $(seq 1 $REP); do
    echo "=============== repetition $rep"
    for L in $LAYOUTS; do
        for lib in $LIBS; do X $lib ${L%x*} ${L#*x} $rep; done
    done
done
echo "=============== this job's CSV lines"
grep ",$SLURM_JOB_ID," "$CSV"
cleanup_all
