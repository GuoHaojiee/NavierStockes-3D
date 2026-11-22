#!/bin/bash
#SBATCH -J gpu_final
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/gpu_final_%x_%j.out
# gpu_final <single|node|multi> <nodes> [dry] — final GPU measurements of the solvers (replaces the dB/dC/dD data of
#   early November; there heFFTe ran with xslab and reorder=1 because runs/common.sh's HX sets NS_HEFFTE_REORDER=1;
#   HX is NOT used here).
#   single (1 node, 1 GPU)   : cuFFT (bin_gcc/ns_cufft), heFFTe gpu1 (bin_gpu_new/ns_heffte_gpu1);
#                              128^3 256^3 512^3 1024^3 (1024^3 is expected to run out of memory: recorded as OOM)
#   node   (1 node, 2/4/8 GPUs): cuFFTXt (bin_gcc/ns_cufftxt, one process; NS_XT_THREADS=6 on 8 GPUs, 1 on 2/4),
#                              cuFFTMp (bin_gcc/ns_cufftmp_multinode), heFFTe mgpu (bin_gpu_new/ns_heffte_mgpu); 512^3 1024^3
#   multi  (1/2/4 nodes x 8 GPUs, bind_hca.sh): cuFFTMp, heFFTe multigpu (bin_gpu_new/ns_heffte_multigpu); 1024^3 2048^3
#   heFFTe mgpu / multigpu: NS_HEFFTE_LAYOUT=zslab NS_HEFFTE_ALGO=p2p_plined NS_HEFFTE_REORDER=0 (the solvers' defaults,
#   set explicitly); gpu1: NS_HEFFTE_ALGO=alltoallv NS_HEFFTE_REORDER=0 (its defaults). Every NS_* variable coming from
#   the submitting shell is removed inside the container before these are set; rank 0 prints GF_ENV (what the run got).
#   bench mode, dt=1e-6, 10 steps, energy references (10 digits): 128^3 2.8047265727e+06 256^3 2.2437812582e+07
#   (CPU FFTW, same dt and steps; GF_EREF_<N> to override) 512^3 1.7950250065e+08 1024^3 1.4360200052e+09
#   2048^3 1.1488160042e+10. energy_ok: yes (identical) | close (relative difference < 1e-9) | no | noref.
#   dry: every (library, GPU count) of the mode once, 64^3 verify, dt=5e-5, 10 steps: L2 must be 6.0041e-10.
#        CSV gpu_final_dry.csv (the energy column then holds the L2 value). Exit 0 only if all match.
# Order: repetition outermost (GF_REP, 3), then grid, then GPU count, then the libraries alternate; so a slow phase of
#   the node hits every library, and each (grid, GPU count) has all libraries within a few minutes of each other.
# Launches: scripts/intel/mn_lib.sh mn_run (time limit, one retry on a broken container / MPI start), job-specific
#   container names (intel_common.sh), srun --mpi=pmix --cpus-per-task=16, one rank per GPU; multi: /work/build/bind_hca.sh.
#   Single-process programs (cuFFT, gpu1, cuFFTXt) get CUDA_VISIBLE_DEVICES=0..g-1. Time limit per launch:
#   GF_TIMEOUT_FACTOR (3) x expected (64^3 1.5 min; <=256^3 2 min; 512^3 3 min; 1024^3 5 min; 2048^3 10 min), >= 3 min.
# Start: the path, size and date of every binary, then for every run the ">>>" line (program, ranks, environment),
#   the plan lines (HEFFTE_PLAN / heFFTe layout, NS_XT_THREADS, GPUs) and the results.
# CSV ~/guohaojie/results/gpu_final.csv (GF_CSV), one line per run:
#   commit,jobid,nodelist,mode,lib,grid,nodes,gpus,rep,step_s,fft_s,spectral_s,physical_s,gpu_mem_GB,energy,energy_ok,status,
#   env,exit_code,retries,heffte_plan
#   status: OK | OOM (out-of-memory message in the output) | TIMEOUT | FAILED-RUN | ERROR | NO-RESULT | MISMATCH (energy / L2)
#   gpu_mem_GB: "GPU memory in use (max over ranks/GPUs, cudaMemGetInfo)"
# Exit code: 0 all OK (OOM counts as OK for single 1024^3 and multi 2048^3 on 1 node, the expected ones), 2 a binary
#   is missing (nothing run), 3 something else not OK.
# Options: GF_REP (3), GF_GRIDS, GF_LIBS, GF_NGPU (node: "2 4 8"), GF_ENV (extra VAR=val for every run),
#   GF_BIN_GCC (/work/build/bin_gcc), GF_BIN_NEW (/work/build/bin_gpu_new), GF_CSV.
# Submit (see scripts/gpu/README.md for the whole chain and the time estimates):
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/gpu
#   sbatch -N 1 -J gf_dry_node -t 00:40:00 gpu_final.sbatch node 1 dry
#   sbatch -N 1 -J gf_node -t 03:00:00 gpu_final.sbatch node 1
#   sbatch -N 4 -J gf_multi4 -t 01:30:00 gpu_final.sbatch multi 4
MODE=${1:?usage: sbatch -N <n> gpu_final.sbatch <single|node|multi> <nodes> [dry]}
NODES=${2:?usage: sbatch -N <n> gpu_final.sbatch <single|node|multi> <nodes> [dry]}
DRY=${3:-}
case $MODE in single|node) [ "$NODES" = 1 ] || { echo "ERROR: $MODE runs on 1 node"; exit 1; };; multi) ;; *) echo "ERROR: mode '$MODE' (single|node|multi)"; exit 1;; esac
[[ $NODES =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: nodes '$NODES'"; exit 1; }
[ -z "$DRY" ] || [ "$DRY" = dry ] || { echo "ERROR: third argument '$DRY' (dry or nothing)"; exit 1; }
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/mn_lib.sh
unset HX
precheck
[ "${SLURM_JOB_NUM_NODES:-0}" = "$NODES" ] || { echo "ERROR: the job has ${SLURM_JOB_NUM_NODES:-?} nodes, argument says $NODES (sbatch -N $NODES ...)"; exit 1; }
export CTM_TIMEOUT_FACTOR=${GF_TIMEOUT_FACTOR:-3}
BG=${GF_BIN_GCC:-/work/build/bin_gcc}; BN=${GF_BIN_NEW:-/work/build/bin_gpu_new}
case $MODE in
single) LIBS=${GF_LIBS:-"cufft heffte_gpu1"};          NG=1;                       GRIDS=${GF_GRIDS:-"128 256 512 1024"} ;;
node)   LIBS=${GF_LIBS:-"cufftxt cufftmp heffte_mgpu"}; NG=${GF_NGPU:-"2 4 8"};     GRIDS=${GF_GRIDS:-"512 1024"} ;;
multi)  LIBS=${GF_LIBS:-"cufftmp heffte_multigpu"};     NG=$((8 * NODES));          GRIDS=${GF_GRIDS:-"1024 2048"} ;;
esac
REP=${GF_REP:-3}; RMODE=bench; DT=1e-6
CSV=${GF_CSV:-$HOME/guohaojie/results/gpu_final.csv}
if [ "$DRY" = dry ]; then GRIDS=64; REP=1; RMODE=verify; DT=5e-5; CSV=${GF_CSV:-$HOME/guohaojie/results/gpu_final_dry.csv}; fi
L2REF=6.0041e-10
eref() {
    local v=GF_EREF_$1
    [ -n "${!v:-}" ] && { echo ${!v}; return; }
    case $1 in 128) echo 2.8047265727e+06;; 256) echo 2.2437812582e+07;; 512) echo 1.7950250065e+08;;
               1024) echo 1.4360200052e+09;; 2048) echo 1.1488160042e+10;; *) echo none;; esac
}
bin_of() {
    case $1 in cufft) echo $BG/ns_cufft;; cufftxt) echo $BG/ns_cufftxt;; cufftmp) echo $BG/ns_cufftmp_multinode;;
               heffte_gpu1|heffte_mgpu|heffte_multigpu) echo $BN/ns_$1;; *) echo NONE;; esac
}
env_of() {   # <lib> <gpus>
    case $1 in
    cufftxt) [ $2 -ge 8 ] && echo NS_XT_THREADS=6 || echo NS_XT_THREADS=1 ;;
    heffte_gpu1) echo NS_HEFFTE_ALGO=alltoallv NS_HEFFTE_REORDER=0 ;;
    heffte_mgpu|heffte_multigpu) echo NS_HEFFTE_LAYOUT=zslab NS_HEFFTE_ALGO=p2p_plined NS_HEFFTE_REORDER=0 ;;
    *) echo "" ;;
    esac
}
gf_expected() {   # <grid> -> seconds per launch
    case $1 in 64) echo 90;; 128|256) echo 120;; 512) echo 180;; 1024) echo 300;; *) echo 600;; esac
}
# out of memory: CUDA, cuFFT CUFFT_ALLOC_FAILED (code 2), heFFTe / C++ allocation, NVSHMEM heap
OOMPAT="out of memory|cudaErrorMemoryAllocation|CUFFT_ALLOC_FAILED|: code 2$|bad_alloc|nvshmem.*(alloc|heap|memory)|NVSHMEM.*(alloc|heap|memory)"
KEEP="^HEFFTE_PLAN|heFFTe layout|NS_XT_THREADS =|^GPUs:|GPU memory in use|Kinetic energy|L2 error|Avg per step|FFT \(incl|Spectral pointwise|Physical pointwise|^GF_ENV|[Ee]rror|out of memory|Abort|abort"

for l in $LIBS; do [ "$(bin_of $l)" = NONE ] && { echo "ERROR: library '$l' (cufft heffte_gpu1 cufftxt cufftmp heffte_mgpu heffte_multigpu)"; exit 1; }; done
for g in $GRIDS; do [[ $g =~ ^[0-9]+$ ]] || { echo "ERROR: grid '$g'"; exit 1; }; done
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)
NODELIST=$(echo "${SLURM_JOB_NODELIST:-?}" | tr ',' ';')
echo "gpu_final: mode=$MODE${DRY:+ (dry)} nodes=$NODES ($SLURM_JOB_NODELIST) gpus=$NG grids=$GRIDS libs=$LIBS reps=$REP $RMODE dt=$DT 10 steps"
echo "commit $COMMIT   csv $CSV   time limit $CTM_TIMEOUT_FACTOR x expected (>= 180 s)   HX not used   extra env: ${GF_ENV:-none}"
miss=0
for l in $LIBS; do
    w=$(bin_of $l); h=$SCR${w#/work}
    if [ -f $h ] && [ -x $h ]; then echo "  binary $l: $w  ($(stat -c '%s bytes, %y' $h | cut -d. -f1))"
    else echo "ERROR: missing binary $h ($l)"; miss=$((miss + 1)); fi
done
if [ $MODE = multi ]; then
    [ -x $SCR/build/bind_hca.sh ] && echo "  bind_hca: /work/build/bind_hca.sh" || { echo "ERROR: missing $SCR/build/bind_hca.sh"; miss=$((miss + 1)); }
fi
[ $miss -gt 0 ] && { echo "ERROR: $miss binary/binaries missing -> nothing run"; exit 2; }
for l in $LIBS; do for g in $NG; do e=$(env_of $l $g); echo "  settings $l gpus=$g: ${e:-library defaults}"; done; done

HDR="commit,jobid,nodelist,mode,lib,grid,nodes,gpus,rep,step_s,fft_s,spectral_s,physical_s,gpu_mem_GB,energy,energy_ok,status,env,exit_code,retries,heffte_plan"
mkdir -p "$(dirname "$CSV")"; [ -f "$CSV" ] || echo "$HDR" > "$CSV"
TMP=$SCR/logs/gpu_final_${SLURM_JOB_ID:-local}.d; mkdir -p $TMP
MN_PREWARM=0 job_containers_init
t0=$(date +%s); srun --ntasks-per-node=1 --time=20 $M true >/dev/null 2>$TMP/prewarm.err || srun --ntasks-per-node=1 --time=20 $M true >/dev/null 2>>$TMP/prewarm.err \
    || echo "WARNING: could not create the container on all nodes (see $TMP/prewarm.err)"
echo "container ready on all nodes in $(( $(date +%s) - t0 )) s"
echo

# gf_launch <lib> <gpus> <time limit min> "<env>" <binary> <args...>
gf_launch() {
    local lib=$1 g=$2 tl=$3 envs=$4 bin=$5; shift 5
    local nn=1 rpn pre="" vis="" inner
    case $lib in
    cufft|heffte_gpu1|cufftxt) rpn=1; vis="CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((g - 1)))" ;;
    *) nn=$(( (g + 7) / 8 )); rpn=$(( g / nn )) ;;
    esac
    [ $MODE = multi ] && pre=/work/build/bind_hca.sh
    inner="for v in \$(env | sed -n 's/^\(NS_[A-Za-z0-9_]*\)=.*/\1/p'); do unset \$v; done; export NS_MODE=gpu; source /work/build/run_env.sh; \
        export $vis $envs GF_RUN=1; \
        [ \${SLURM_PROCID:-0} = 0 ] && echo \"GF_ENV: \$(env | grep -E '^(NS_|CUDA_VISIBLE_DEVICES|NVSHMEM_|UCX_TLS|UCX_NET)' | sort | tr '\n' ' ')\"; \
        exec $pre $bin $*"
    srun --mpi=pmix -N $nn --ntasks-per-node=$rpn --cpus-per-task=16 --time=$tl $M bash -c "$inner"
}
NBAD=0
# X <lib> <grid> <gpus> <rep>
X() {
    local lib=$1 N=$2 g=$3 rep=$4 bin envs label raw tmo tl val ref eok st mem spec phys plan nn
    bin=$(bin_of $lib); envs="$(env_of $lib $g)${GF_ENV:+ $GF_ENV}"; envs=${envs# }
    nn=1; case $lib in cufft|heffte_gpu1|cufftxt) ;; *) nn=$(( (g + 7) / 8 ));; esac
    label="${lib}-${N}_g${g}_n${nn}_r${rep}"; raw=$TMP/$label.raw
    tmo=$(awk -v e=$(gf_expected $N) -v f=$CTM_TIMEOUT_FACTOR 'BEGIN { t = f * e; print (t < 180 ? 180 : int(t)) }'); tl=$(( tmo / 60 + 2 ))
    echo ">>> [$(date +%H:%M:%S)] $label $RMODE | limit=${tmo}s env: ${envs:-none} | $bin $N $N $N $DT 10 $RMODE"
    mn_run $label $raw $tmo gf_launch $lib $g $tl "$envs" $bin $N $N $N $DT 10 $RMODE
    grep -E "$KEEP" $raw | grep -v "^ *$" | awk '!seen[$0]++' | head -40
    mn_extract $raw none
    if [ $RMODE = verify ]; then val=$(awk '/L2 error/ {print $NF; exit}' $raw); ref=$L2REF
    else val=$MN_EN; ref=$(eref $N); fi
    if [ -z "$val" ]; then eok=no
    elif [ "$ref" = none ]; then eok=noref
    elif [ "$val" = "$ref" ]; then eok=yes
    else eok=$(awk -v a=$val -v b=$ref 'BEGIN { d = (a - b) / b; if (d < 0) d = -d; print (d < 1e-9) ? "close" : "no" }'); fi
    spec=$(awk '/Spectral pointwise:/ {print $(NF-1); exit}' $raw)
    phys=$(awk '/Physical pointwise:/ {print $(NF-1); exit}' $raw)
    mem=$(grep -m1 "GPU memory in use" $raw | sed -E 's/.*: *([0-9.]+) GB of.*/\1/')
    plan=$(grep -m1 "^HEFFTE_PLAN" $raw | cut -d' ' -f2- | tr ',' ';')
    st=$MN_STATUS
    if [ $st != OK ] && grep -qE "$OOMPAT" $raw; then st=OOM
    elif [ $st = OK ] && [ -z "$val" ]; then st=NO-RESULT
    elif [ $st = OK ] && [ $eok = no ]; then st=MISMATCH; fi
    [ $st = OOM ] && echo "OOM: $(grep -m1 -E "$OOMPAT" $raw | cut -c1-200)"
    echo "<<< exit=$MN_RC status=$st value=${val:-none} ref=$ref ok=$eok step=${MN_STEP:--} fft=${MN_FFT:--} mem=${mem:--} GB"
    echo "$COMMIT,${SLURM_JOB_ID:-local},$NODELIST,$MODE,$lib,$N,$nn,$g,$rep,$MN_STEP,$MN_FFT,$spec,$phys,$mem,$val,$eok,$st,$(echo "$envs" | tr ',' ';'),$MN_RC,$MN_RETRIES,$plan" \
        | tee -a "$CSV" | sed 's/^/CSV /'
    # OOM is the expected result of 1024^3 on one GPU and of 2048^3 on one node; anything else not OK counts
    if [ $st != OK ]; then
        local exp=0
        [ $N -ge 1024 ] && [ $g = 1 ] && exp=1
        [ $N -ge 2048 ] && [ $g -le 8 ] && exp=1
        if [ $st = OOM ] && [ $exp = 1 ]; then echo "(OOM expected here)"
        else NBAD=$((NBAD + 1)); fi
    fi
    echo
}

for rep in $(seq $REP); do
    echo "=============== repetition $rep / $REP"
    for N in $GRIDS; do for g in $NG; do for lib in $LIBS; do
        X $lib $N $g $rep
    done; done; done
done
echo "=============== this job's CSV lines: $(grep -c ",${SLURM_JOB_ID:-local}," "$CSV")   retries: $(grep -c '^RETRY' $SCR/logs/gpu_final_${SLURM_JOB_NAME}_$SLURM_JOB_ID.out 2>/dev/null)"
cleanup_all
D=$HOME/guohaojie/src/NavierStockes-3D/scripts/gpu
[ -f $D/gpu_final_parse.sh ] && GF_JOBS=${SLURM_JOB_ID:-local} bash $D/gpu_final_parse.sh "$CSV"
if [ $NBAD -gt 0 ]; then echo "gpu_final: $NBAD run(s) not OK -> exit 3"; exit 3; fi
echo "gpu_final: all runs OK"
exit 0
