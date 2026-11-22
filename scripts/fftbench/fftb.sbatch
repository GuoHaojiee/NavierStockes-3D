#!/bin/bash
#SBATCH -J fftb
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 02:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fftb_%x_%j.out
# fftb <nodes 1|2|4> <grid(s): 512 | 1024 | "512 1024" | dry> — pure 3D FFT benchmark (fftbench.cpp, the solvers'
#   plan / execute code from fftlib/), the four CPU libraries, GCC and Intel toolchains, launched with mn_lib.sh
#   (job-specific containers, time limit, start-failure retry, UCX probe). The job must have that many nodes:
#   sbatch -N <nodes> ... fftb.sbatch <nodes> <grid>
# One launch per (grid, library, configuration); inside it: plan (timed), check, 2 warm-up rounds, FFTB_REP (3)
#   repetitions of K pairs (K so that a repetition takes >= FFTB_WINDOW = 3 s); see fftbench.cpp.
#   dry: 64^3, FFTB_REP=1, window 0.5 s, layouts with <= 64 ranks (slab): 1 node 64x2, 2 nodes 32x4, 4 nodes 16x8;
#        CSV fftb_dry.csv. Shows that every binary starts, plans and passes the check (FFTB_CHECK ... (OK)).
# Configurations (each one launch per grid and library):
#   default: for every layout in FFTB_LAYOUTS ("128x1"): GCC once per GCC_COLL setting ("def"; list e.g. "def t4"),
#            then Intel once per FFTB_INTEL_SETS entry ("name:VAR=val VAR=val;name2:..."; default one entry: E6d for
#            2/4 nodes, single-node for 1 node, i.e. the solvers' default Intel settings). Toolchains: FFTB_TC ("gcc intel").
#   FFTB_INTEL_SETS format: entries separated by ";", each "name:VAR=val VAR=val ..." (name = configuration name, the
#            first word of the CSV settings column; "name:" alone = the default E6d / single-node settings).
#            IMN_* variables go to intel_mn_env.sh (IMN_ALLTOALL, IMN_ALLTOALLV, IMN_UCX_TLS, IMN_EXTRA, ...;
#            IMN_ALLTOALL= empty = Intel MPI's default alltoall); every other variable (NS_ACCFFT_TMETHOD, NS_ACCFFT_PLAN,
#            I_MPI_..., UCX_...) is exported in the run's environment after intel_mn_env.sh, so it wins.
#            Example: FFTB_INTEL_SETS="E6d:;a2a_def:IMN_ALLTOALL=;a2av1:IMN_ALLTOALLV=1;est:NS_ACCFFT_PLAN=estimate"
#   per library, Intel only: FFTB_INTEL_ENV_FFTW / _ACCFFT / _P3DFFT / _HEFFTE = "VAR=val ..." added to every Intel run
#            of that library (after the configuration's variables), e.g. FFTB_INTEL_ENV_ACCFFT="I_MPI_ADJUST_ALLTOALLV=1".
#            Shown in the ">>>" line and in the CSV settings column; the mpi_env column has what the program saw.
#   explicit: FFTB_CONFIGS = lines "name|gcc|layout|GCC_COLL=<x> [VAR=val ...]" or "name|intel|layout|IMN_...=... "
#            (as p256_diag.sbatch; see fftb_p256.sbatch).
#   The CSV settings column is "<name>[ <variables>] NS_DECOMP=<d>"; its first word is the configuration name, the
#   key used by fftb_parse.sh to match solver runs (toolchain_mn.csv: GCC_COLL=<x> / E6d / single-node;
#   p256_diag.csv: i_a_base, g_def, ...).
# Binaries /work/build/bin_fftb/<gcc|intel>/ (build_fftb.sh); Intel heFFTe = fftb_heffte_mkl with NS_HEFFTE_REORDER=1
#   (as cmp_toolchain_mn). Libraries: FFTB_LIBS ("fftw accfft p3dfft heffte").
# Settings for every run: NS_DECOMP=$FFTB_DECOMP (slab), NS_FFTW_PLAN=$FFTB_PLAN (measure), FFTB_REP, FFTB_WINDOW,
#   FFTB_NF (3 fields), FFTB_SPLIT (0; 1: also the local FFT time -> t_local_s), FFTB_ENV (extra VAR=val for all runs).
# Time limit per launch: FFTB_TIMEOUT_FACTOR (3) x expected (64^3 1 min; 512^3 2 min; 1024^3 1 node 10 min,
#   2 nodes 6 min, 4 nodes 4 min), at least 3 min.
# CSV ~/guohaojie/results/fftb.csv (FFTB_CSV), one line per repetition:
#   commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,t_fft_s,t_plan_s,t_local_s,
#   check,ucx_version,status,t_fwd_s,t_bwd_s,pairs,max_err,nf,lib_settings,retries,exit_code,ucx_reason,bind,mpi_env
#   lib_settings: the library's settings() (AccFFT: tsel=measure|default|fixed tmethod=m1|m2|m3k<k>a|s call=...)
#   mpi_env: FFTB_MPIENV of rank 0 (I_MPI_ADJUST_*, I_MPI_FABRICS, UCX_TLS, OMPI_MCA_coll*, NS_ACCFFT_*, ...; "," -> ";")
#   bind: OK | WARN from the FFTB_BIND line (ranks of a node share no CPU, every thread on its own physical core)
#   t_fft_s = (t_fwd + 2 t_bwd)/3: the solver mix (12 forward + 24 backward per RK4 step), 36 t_fft_s = FFT s/step.
#   status: OK | TIMEOUT | FAILED-RUN | ERROR (mn_lib.sh) | NO-RESULT (exit 0 but no FFTB_REP line) | CHECK-FAIL |
#           MISSING-BINARY (exit 127). Before anything runs, the job checks that every binary it needs exists (exit 2 if not).
#   A launch without results writes one line with rep=0.
# Exit code: 0 = every run OK (status OK, check OK); 2 = a needed binary is missing (nothing run); 3 = some run not OK.
#   So --dependency=afterok:<dry-run job> starts the long jobs only after a clean dry run.
# Submit:
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/fftbench
#   sbatch -N 1 -J fftb_dry1 -t 00:40:00 fftb.sbatch 1 dry
#   sbatch -N 2 -J fftb_512_2 -t 02:00:00 --export=ALL,GCC_COLL="def t4" fftb.sbatch 2 512
#   bash fftb_parse.sh            (FFTB_JOBS="<jobid> ..." to select)
NODES=${1:?usage: sbatch -N <n> fftb.sbatch <1|2|4> <grid(s)|dry>}
GRIDS=${2:?usage: sbatch -N <n> fftb.sbatch <1|2|4> <grid(s)|dry>}
export INTEL_IMAGE=${FFTB_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/mn_lib.sh
precheck
[ "${SLURM_JOB_NUM_NODES:-0}" = "$NODES" ] || { echo "ERROR: the job has ${SLURM_JOB_NUM_NODES:-?} nodes, argument says $NODES (sbatch -N $NODES ...)"; exit 1; }
export CTM_TIMEOUT_FACTOR=${FFTB_TIMEOUT_FACTOR:-3}       # printed by mn_run on a timeout
REP=${FFTB_REP:-3}; WINDOW=${FFTB_WINDOW:-3}; LAY="${FFTB_LAYOUTS:-128x1}"
if [ "$GRIDS" = dry ]; then
    GRIDS=64; REP=${FFTB_REP:-1}; WINDOW=${FFTB_WINDOW:-0.5}
    case $NODES in 1) LAY="${FFTB_LAYOUTS:-64x2}";; 2) LAY="${FFTB_LAYOUTS:-32x4}";; *) LAY="${FFTB_LAYOUTS:-16x8}";; esac
    CSV=${FFTB_CSV:-$HOME/guohaojie/results/fftb_dry.csv}
else
    CSV=${FFTB_CSV:-$HOME/guohaojie/results/fftb.csv}
fi
for g in $GRIDS; do [[ $g =~ ^[0-9]+$ ]] || { echo "ERROR: grid '$g' (a number, a quoted list, or dry)"; exit 1; }; done
LIBS=${FFTB_LIBS:-"fftw accfft p3dfft heffte"}
DECOMP=${FFTB_DECOMP:-slab}; PLAN=${FFTB_PLAN:-measure}; NF=${FFTB_NF:-3}; SPLIT=${FFTB_SPLIT:-0}
COMMON="NS_FFTW_PLAN=$PLAN NS_DECOMP=$DECOMP FFTB_REP=$REP FFTB_WINDOW=$WINDOW FFTB_NF=$NF FFTB_SPLIT=$SPLIT ${FFTB_ENV:-}"
DEFI=$([ $NODES = 1 ] && echo single-node || echo E6d)

# ---- configurations: name|toolchain|layout|settings
CONFIGS=()
if [ -n "${FFTB_CONFIGS:-}" ]; then
    while IFS= read -r line; do line=$(echo "$line" | sed 's/^ *//;s/ *$//'); [ -n "$line" ] && CONFIGS+=("$line"); done <<< "$FFTB_CONFIGS"
else
    for L in $LAY; do
        if [[ " ${FFTB_TC:-gcc intel} " == *" gcc "* ]]; then
            for gc in ${GCC_COLL:-def}; do CONFIGS+=("GCC_COLL=$gc|gcc|$L|GCC_COLL=$gc${FFTB_GCC_ENV:+ $FFTB_GCC_ENV}"); done
        fi
        if [[ " ${FFTB_TC:-gcc intel} " == *" intel "* ]]; then
            IFS=';' read -ra ISETS <<< "${FFTB_INTEL_SETS:-$DEFI:}"
            for s in "${ISETS[@]}"; do [ -n "$s" ] && CONFIGS+=("${s%%:*}|intel|$L|${s#*:}"); done
        fi
    done
fi
for c in "${CONFIGS[@]}"; do
    IFS='|' read -r cn tc L set <<< "$c"
    case $tc in gcc|intel) ;; *) echo "ERROR: configuration '$c': toolchain gcc|intel"; exit 1;; esac
    [[ $L =~ ^[0-9]+x[0-9]+$ ]] || { echo "ERROR: configuration '$c': layout <ranks>x<threads>"; exit 1; }
    if [ $tc = gcc ]; then g=$(echo "$set" | grep -oE "GCC_COLL=[a-z0-9]+" | cut -d= -f2); [[ "$(mn_gcc_coll ${g:-def})" == *BAD_GCC_COLL* ]] && { echo "ERROR: GCC_COLL=$g (def|t4)"; exit 1; }; fi
done

TMP=$SCR/logs/fftb_$SLURM_JOB_ID.d; mkdir -p $TMP "$(dirname "$CSV")"
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)
NODELIST=$(echo "${SLURM_JOB_NODELIST:-?}" | tr ',' ';')
HDR="commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,t_fft_s,t_plan_s,t_local_s,check,ucx_version,status,t_fwd_s,t_bwd_s,pairs,max_err,nf,lib_settings,retries,exit_code,ucx_reason,bind,mpi_env"
[ -f "$CSV" ] || echo "$HDR" > "$CSV"
# a CSV from before the bind column: extend the header only (old rows: bind empty)
head -1 "$CSV" | grep -q ',bind' || sed -i '1s/$/,bind/' "$CSV"
head -1 "$CSV" | grep -q ',mpi_env' || sed -i '1s/$/,mpi_env/' "$CSV"
KEEP="^decomposition:|process grid|planner flag|P3DFFT library|Threads \(runtime\)|planning time|plan time|\(planning\) time|^FFTB|rror|abort|intel_mn_env:|GCCENV:|UCXPROBE|MPI startup\(\): +(Intel|libfabric provider)"
echo "fftb: nodes=$NODES ($SLURM_JOB_NODELIST) grids=$GRIDS libs=$LIBS reps=$REP window=${WINDOW}s fields=$NF split=$SPLIT"
echo "commit $COMMIT   csv $CSV   common: $COMMON   time limit: $CTM_TIMEOUT_FACTOR x expected (>= 180 s)"
for c in "${CONFIGS[@]}"; do echo "  config $c"; done
for lib in $LIBS; do v=FFTB_INTEL_ENV_$(echo $lib | tr a-z A-Z); [ -n "${!v:-}" ] && echo "  $v=${!v} (Intel runs of $lib)"; done
bin_of() {   # <toolchain> <lib>
    case $1:$2 in
    gcc:heffte) echo /work/build/bin_fftb/gcc/fftb_heffte;; intel:heffte) echo /work/build/bin_fftb/intel/fftb_heffte_mkl;;
    *) echo /work/build/bin_fftb/$1/fftb_$2;;
    esac
}
# every binary this job will run (toolchains of CONFIGS x FFTB_LIBS) must exist and be executable; otherwise stop
# here with exit code 2, before any container is started (instead of one exit-127 run per configuration)
TCS=$(for c in "${CONFIGS[@]}"; do IFS='|' read -r cn tc L set <<< "$c"; echo $tc; done | sort -u)
miss=0
for tc in $TCS; do for lib in $LIBS; do
    w=$(bin_of $tc $lib); h=$SCR${w#/work}
    if [ -f $h ] && [ -x $h ]; then echo "  binary $(ls -l --time-style=+%F_%R $h | awk '{print $6, $7}')"
    else echo "ERROR: missing binary $h (toolchain $tc, library $lib)"; miss=$((miss + 1)); fi
done; done
if [ $miss -gt 0 ]; then
    echo "ERROR: $miss binary/binaries missing -> nothing run. Build: scripts/fftbench/build_fftb.sh gcc|intel (it must end with BUILD OK)"
    exit 2
fi
[ $NODES -gt 1 ] && { [ -f $SCR/opt_ucx/hpcx/lib/libucp.so.0 ] || echo "WARNING: HPC-X UCX not staged (stage_hpcx_ucx.sh)"; }
job_containers_init
mn_init $TMP /work/logs/fftb_$SLURM_JOB_ID.d

name_of() { case $1 in fftw) echo FFTW;; accfft) echo AccFFT;; p3dfft) echo P3DFFT;; heffte) echo heFFTe;; esac; }
fftb_expected() {   # <grid> <nodes> -> seconds per launch (container start, planning, check, warm-up, repetitions)
    local g=$1 n=$2
    if [ $g -le 128 ]; then echo 60; elif [ $g -le 512 ]; then echo 120
    else case $n in 1) echo 600;; 2) echo 360;; *) echo 240;; esac; fi
}
intel_cfg() {   # intel_cfg "<IMN_* settings>" <mn_intel arguments>: only this configuration's IMN_* (subshell)
    local set=$1; shift
    ( unset IMN_ALLTOALL IMN_ALLTOALLV IMN_UCX_TLS IMN_EXTRA IMN_PIN; [ -n "$set" ] && export $set; mn_intel "$@" )
}
kv() { echo "$2" | grep -oE "(^| )$1=[^ ]*" | head -1 | sed "s/^ *$1=//"; }

# X <config> <lib> <grid>
X() {
    local cn tc L set ppn nt bin label raw tmo tl envs sets e t setcol libset chk err line r st n=0
    IFS='|' read -r cn tc L set <<< "$1"; ppn=${L%x*}; nt=${L#*x}
    local lib=$2 N=$3
    bin=$(bin_of $tc $lib); envs=$COMMON
    [ $tc = intel ] && [ $lib = heffte ] && envs="$envs NS_HEFFTE_REORDER=1"
    # Intel: IMN_* of the configuration go to intel_cfg (intel_mn_env.sh), everything else into the run environment,
    # then the per-library variables FFTB_INTEL_ENV_<LIB>
    local imn="" oth="" w lv libenv=""
    if [ $tc = intel ]; then
        for w in $set; do case $w in IMN_*) imn="$imn $w";; *=*) oth="$oth $w";; esac; done
        lv=FFTB_INTEL_ENV_$(echo $lib | tr a-z A-Z); libenv=${!lv:-}
        envs="$envs$oth${libenv:+ $libenv}"
    fi
    label="$(name_of $lib)-${N}_${tc}-${cn}_n${NODES}_$L"; raw=$TMP/$label.raw
    e=$(fftb_expected $N $NODES)
    tmo=$(awk -v e=$e -v f=$CTM_TIMEOUT_FACTOR 'BEGIN { t = f * e; print (t < 180 ? 180 : int(t)) }'); tl=$(( tmo / 60 + 2 ))
    setcol="$cn"; [ -n "$set" ] && [ "$set" != "$cn" ] && setcol="$cn $set"; setcol="$setcol${libenv:+ $libenv} NS_DECOMP=$DECOMP"
    echo ">>> [$(date +%H:%M:%S)] $label ranks=$((NODES * ppn)) threads=$nt | limit=${tmo}s $setcol | $bin $N $N $N $nt"
    if [ $tc = gcc ]; then
        t=$(echo "$set" | grep -oE "GCC_COLL=[a-z0-9]+" | cut -d= -f2)
        envs="$envs $(mn_gcc_coll ${t:-def}) $(echo "$set" | sed -E 's/GCC_COLL=[a-z0-9]+//')"
        mn_run $label $raw $tmo mn_gcc $NODES $ppn $nt $tl "$envs" $bin $N $N $N $nt
    else
        mn_run $label $raw $tmo intel_cfg "${imn# }" $NODES $ppn $nt $tl "$envs" $bin $N $N $N $nt
    fi
    grep -E "$KEEP" $raw | awk '!/UCXPROBE/ || !u++'
    mn_extract $raw 0
    if ! grep -q "^UCXPROBE" $raw; then grep -q "^FFTB_SUMMARY" $raw && MN_UCXWHY=no-probe-line-rank0-output-missing || MN_UCXWHY=run-did-not-finish; fi
    echo "UCX: $label version=$MN_UCXV${MN_UCXWHY:+ reason=$MN_UCXWHY}"
    [ $tc = intel ] && [ $NODES -gt 1 ] && [ "$MN_UCXV" != 1.17.0 ] && echo "WARNING: UCX $MN_UCXV loaded, expected 1.17.0 ($label${MN_UCXWHY:+, $MN_UCXWHY})"
    line=$(grep -m1 "^FFTB_SUMMARY" $raw)
    libset=$(echo "$line" | grep -oE 'settings="[^"]*"' | sed 's/settings="//;s/"$//' | tr ',' ';')
    chk=$(kv check "$line"); err=$(kv max_err "$line")
    local tplan tloc bind; tplan=$(kv t_plan_s "$line"); tloc=$(kv t_local_s "$line")
    bind=$(grep -m1 "^FFTB_BIND ranks/node" $raw | awk '{print $NF}')
    local menv; menv=$(grep -m1 "^FFTB_MPIENV " $raw | sed 's/^FFTB_MPIENV //' | tr ',' ';')
    echo "<<< exit=$MN_RC"
    [ "$MN_RC" = 127 ] && echo "WARNING: exit 127 = command not found (missing binary $bin?)"
    while read -r r; do
        [ -z "$r" ] && continue; n=$((n + 1))
        st=$MN_STATUS; [ "$st" = OK ] && [ "$chk" = FAIL ] && st=CHECK-FAIL
        echo "$COMMIT,$SLURM_JOB_ID,$NODELIST,$tc,$(echo "$setcol" | tr ',' ';'),$(name_of $lib),$N,$NODES,$ppn,$nt,$(kv rep "$r"),$(kv t_fft_s "$r"),$tplan,$tloc,$chk,$MN_UCXV,$st,$(kv t_fwd_s "$r"),$(kv t_bwd_s "$r"),$(kv pairs "$r"),$err,$NF,$libset,$MN_RETRIES,$MN_RC,$MN_UCXWHY,$bind,$menv" \
            | tee -a "$CSV" | sed 's/^/CSV /'
    done <<< "$(grep "^FFTB_REP " $raw)"
    if [ $n = 0 ]; then
        st=$MN_STATUS; [ "$st" = OK ] && st=NO-RESULT; [ "$MN_RC" = 127 ] && st=MISSING-BINARY
        echo "$COMMIT,$SLURM_JOB_ID,$NODELIST,$tc,$(echo "$setcol" | tr ',' ';'),$(name_of $lib),$N,$NODES,$ppn,$nt,0,,$tplan,,${chk:-none},$MN_UCXV,$st,,,,$err,$NF,$libset,$MN_RETRIES,$MN_RC,$MN_UCXWHY,$bind,$menv" \
            | tee -a "$CSV" | sed 's/^/CSV /'
    fi
    echo
}

for N in $GRIDS; do
    echo "=============== grid $N"
    for lib in $LIBS; do
        for c in "${CONFIGS[@]}"; do X "$c" $lib $N; done
    done
done
echo "=============== this job's CSV lines: $(grep -c ",$SLURM_JOB_ID," "$CSV")   retries: $(grep -c '^RETRY' $SCR/logs/fftb_${SLURM_JOB_NAME}_$SLURM_JOB_ID.out 2>/dev/null)"
# exit code for --dependency=afterok chains: 0 only if every line of this job has status OK and check OK
NBAD=$(awk -F, -v j=$SLURM_JOB_ID '$2 == j && ($17 != "OK" || $15 != "OK")' "$CSV" | wc -l)
cleanup_all
if [ "$NBAD" -gt 0 ]; then echo "fftb: $NBAD CSV line(s) of this job not OK (status/check) -> exit 3"; exit 3; fi
echo "fftb: all runs OK"
exit 0
