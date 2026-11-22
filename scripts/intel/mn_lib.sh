# mn_lib.sh — sourced by the sbatch scripts (after intel_common.sh, outside the containers): one solver run with a
# time limit, early detection of a broken start and one retry, the two launch methods, result extraction.
# Used by cmp_toolchain_mn.sbatch, p256_diag.sbatch, cmp_toolchain_4n.sbatch and scripts/cpu_final/cpu_final.sbatch.
#
#   mn_init <tmpdir on /scratch> <the same path under /work>
#       builds the UCX probe in both containers (scripts/cpu_final/ucx_probe.c, mpicc, wraps MPI_Finalize)
#   mn_timeout <grid> <nodes> <layout>          -> seconds: 3 x expected (table below), at least 180 (MN_TMO_SCALE)
#   mn_gcc_coll <def|t4>                        -> Open MPI environment for GCC_COLL
#   mn_run <label> <raw file> <timeout s> <command...>
#       runs the command (a shell function or program) in the background, output to <raw>; every 5 s looks for
#       $MN_FAILPAT (broken container start or MPI start); then, or at the time limit, kills it. A run killed for a
#       start failure is repeated once ("RETRY <label> rep=<r> attempt=2 reason=<...>" in the log).
#       Sets MN_RC (exit code; 124 = time limit), MN_REASON, MN_RETRIES.
#   mn_gcc   <nodes> <ppn> <threads> <time limit min> "<env>" <binary> <args...>   GCC container, Open MPI, srun pmix
#   mn_intel <nodes> <ppn> <threads> <time limit min> "<env>" <binary> <args...>   v2 image: 1 node mpirun fork+shm,
#                                                                                   more nodes irun_mn (E6d, IMN_*)
#   mn_extract <raw> <reference energy>
#       sets MN_STEP MN_FFT MN_PLAN MN_EN MN_EOK MN_UCXV MN_UCXWHY MN_OMPT
MN_FAILPAT="container start failed|enroot-switchroot|inconsistent client PMI version|internal_Init_thread"

# expected seconds per run (container start, MEASURE planning, 10 steps); grid x nodes x ranks-per-node
mn_expected() {
    local g=$1 n=$2 ppn=${3%x*}
    case $g in
    64|dry) echo 60 ;;
    512)  case $n in 1) [ $ppn = 1 ] && echo 300 || echo 150;; 2) echo 120;; *) echo 90;; esac ;;
    1024) case $n in 1) [ $ppn = 1 ] && echo 1500 || echo 900;; 2) echo 600;; *) echo 360;; esac ;;
    *) echo 600 ;;
    esac
}
mn_timeout() {
    local e t; e=$(mn_expected "$@"); t=$(awk -v e=$e -v s=${MN_TMO_SCALE:-1} 'BEGIN { t = 3 * e * s; print (t < 180 ? 180 : int(t)) }')
    echo $t
}
mn_gcc_coll() {
    case ${1:-def} in
    def) echo "" ;;
    t4)  echo "OMPI_MCA_coll_tuned_use_dynamic_rules=1 OMPI_MCA_coll_tuned_alltoall_algorithm=4 OMPI_MCA_coll_ucc_enable=0 OMPI_MCA_coll_hcoll_enable=0" ;;
    *)   echo "BAD_GCC_COLL=$1" ;;
    esac
}

mn_init() {
    MN_TMP=$1; MN_WTMP=$2; mkdir -p $MN_TMP
    local src=/work/src/NavierStockes-3D/scripts/cpu_final/ucx_probe.c
    srun -N1 -n1 --time=5 $M bash -c "source /etc/profile.d/ns3d_env.sh >/dev/null 2>&1; mpicc -O2 -shared -fPIC $src -o $MN_WTMP/ucx_probe_gcc.so -ldl" \
        || echo "WARNING: UCX probe for the GCC side not built"
    srun -N1 -n1 --time=5 $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; mpicc -O2 -shared -fPIC $src -o $MN_WTMP/ucx_probe_intel.so -ldl" \
        || echo "WARNING: UCX probe for the Intel side not built"
    MN_PG=""; [ -f $MN_TMP/ucx_probe_gcc.so ] && MN_PG=$MN_WTMP/ucx_probe_gcc.so
    MN_PI=""; [ -f $MN_TMP/ucx_probe_intel.so ] && MN_PI=$MN_WTMP/ucx_probe_intel.so
    echo "UCX probe: gcc ${MN_PG:-none}  intel ${MN_PI:-none}"
}

# kill a background command and everything it started (srun, irun_mn's subshell ...)
mn_kill() {
    local pid=$1 c
    for c in $(pgrep -P $pid 2>/dev/null); do mn_kill $c; done
    kill -TERM $pid 2>/dev/null
}
mn_run_once() {
    local raw=$1 tmo=$2; shift 2
    "$@" > $raw 2>&1 &
    local pid=$! t=0
    MN_REASON=""
    while kill -0 $pid 2>/dev/null; do
        sleep 5; t=$((t + 5))
        if grep -qE "$MN_FAILPAT" $raw 2>/dev/null; then
            MN_REASON="start failure: $(grep -m1 -oE "$MN_FAILPAT" $raw)"; mn_kill $pid; sleep 5; pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; break
        fi
        if [ $t -ge $tmo ]; then MN_REASON="time limit ${tmo}s"; mn_kill $pid; sleep 5; pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; break; fi
    done
    wait $pid 2>/dev/null; MN_RC=$?
    case "$MN_REASON" in "time limit"*) MN_RC=124 ;; "start failure"*) [ $MN_RC = 0 ] && MN_RC=125 ;; esac
    # a start failure can also show up only at the end (srun returned quickly)
    [ -z "$MN_REASON" ] && [ $MN_RC != 0 ] && grep -qE "$MN_FAILPAT" $raw && MN_REASON="start failure: $(grep -m1 -oE "$MN_FAILPAT" $raw)"
    return 0
}
mn_run() {
    local label=$1 raw=$2 tmo=$3; shift 3
    MN_RETRIES=0
    mn_run_once $raw $tmo "$@"
    if [[ "$MN_REASON" == "start failure"* ]]; then
        echo "RETRY $label rep=${rep:-?} attempt=2 reason=${MN_REASON#start failure: } (exit $MN_RC)"
        cp $raw $raw.try1
        MN_RETRIES=1
        mn_run_once $raw $tmo "$@"
    fi
    [ -n "$MN_REASON" ] && echo "FAILED-RUN $label rep=${rep:-?} reason=$MN_REASON exit=$MN_RC"
    return 0
}

# mn_gcc <nodes> <ppn> <threads> <time min> "<env>" <binary> <args...>
mn_gcc() {
    local nn=$1 ppn=$2 nt=$3 tl=$4 envs=$5 bin=$6; shift 6
    local pre=""; [ -n "$MN_PG" ] && pre="export LD_PRELOAD=$MN_PG;"
    srun --mpi=pmix --cpu-bind=cores --time=$tl -N $nn --ntasks-per-node=$ppn -c $nt $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close $envs; \
        [ \${SLURM_PROCID:-0} = 0 ] && { echo \"GCCENV: \$(env | grep -E '^OMPI_MCA_coll|^UCX_(NET|TLS)' | sort | tr '\\n' ' ')\"; }; \
        $pre exec $bin $*"
}
# mn_intel <nodes> <ppn> <threads> <time min> "<env>" <binary> <args...>
mn_intel() {
    local nn=$1 ppn=$2 nt=$3 tl=$4 envs=$5 bin=$6; shift 6
    if [ $nn = 1 ]; then
        local bind=close numa=""
        [ $ppn = 1 ] && [ $nt -ge 64 ] && { bind=spread; numa="/work/tools/numactl --interleave=all"; }
        srun -N 1 -n 1 -c 128 --cpu-bind=none --time=$tl $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
            export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$nt:compact OMP_NUM_THREADS=$nt OMP_PLACES=cores \
                   OMP_PROC_BIND=$bind KMP_BLOCKTIME=0 MKL_NUM_THREADS=$nt $envs; \
            [ -n '$numa' ] && export LD_LIBRARY_PATH=${MN_NUMALIB:-/work/tools:/work/tools/lib}:\$LD_LIBRARY_PATH; \
            exec mpirun -np $ppn ${MN_PI:+-genv LD_PRELOAD $MN_PI} $numa $bin $*"
    else
        IRUN_SRUN_OPTS="--time=$tl" irun_mn $nn $ppn $nt "export KMP_BLOCKTIME=0 MKL_NUM_THREADS=$nt $envs ${MN_PI:+LD_PRELOAD=$MN_PI}; exec $bin $*"
    fi
}

mn_extract() {
    local raw=$1 eref=$2
    MN_STEP=$(awk '/Avg per step:/ {print $(NF-1); exit}' $raw)
    MN_FFT=$(awk '/FFT \(incl/ {print $(NF-1); exit}' $raw)
    MN_PLAN=$(awk '/planning time:|plan time:|\(planning\) time:/ {p = $NF; if (p == "s") p = $(NF-1); print p; exit}' $raw)
    MN_EN=$(awk '/Kinetic energy/ {print $NF; exit}' $raw)
    [ -n "$MN_EN" ] && [ "$MN_EN" = "$eref" ] && MN_EOK=yes || MN_EOK=no
    local l; l=$(grep -m1 "^UCXPROBE" $raw)
    if [ -n "$l" ]; then
        MN_UCXV=$(echo "$l" | grep -oE "version=[^ ]+" | cut -d= -f2)
        MN_UCXWHY=$(echo "$l" | grep -oE "reason=[^ ]+" | cut -d= -f2)
    else
        MN_UCXV=unknown
        if grep -q "UCXPROBE" $raw; then MN_UCXWHY=probe-line-garbled
        elif [ -z "$MN_STEP" ]; then MN_UCXWHY=run-did-not-finish
        else MN_UCXWHY=no-probe-line-rank0-output-missing; fi
    fi
    MN_OMPT=$(grep -m1 -oE "omp_get_max_threads\(\)=[0-9]+" $raw | cut -d= -f2)
}
