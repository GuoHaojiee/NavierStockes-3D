#!/bin/bash
#SBATCH -J p256
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/p256_%x_%j.out
# p256_diag — why are FFTW and P3DFFT slow with Intel MPI at p = 256 ranks (irun_mn, E6d, HPC-X UCX 1.17)?
#   512^3 2 nodes: FFTW 128x1 7.22 s/step vs 64x2 1.99 s; 4 nodes 64x2 4.60 s vs 128x1 2.13 s; GCC + Open MPI 4 nodes
#   64x2 1.05 s. FFTW chose mpi-transpose-pairwise.
# 2 nodes, 512^3, dt=1e-6, 10 steps, energy 1.7950250065e+08; FFTW and P3DFFT; 3 repetitions (outer loop),
# configurations alternate:
#   intel 128x1 (p=256)  i_a_base     E6d with I_MPI_ADJUST_ALLTOALL not set
#                        i_b_rc       + IMN_UCX_TLS=rc_mlx5,sm,self
#                        i_c_dc       + IMN_UCX_TLS=dc_mlx5,sm,self
#                        i_d_rndvinf  + IMN_EXTRA="UCX_RNDV_THRESH=inf"
#                        i_e_rndv4k   + IMN_EXTRA="UCX_RNDV_THRESH=4096"
#                        i_f_pinintel + IMN_PIN=intel (Intel MPI pins, srun does not bind)
#   intel 64x2  (p=128)  i_a_base
#   gcc   128x1 (p=256)  g_def (Open MPI default: coll ucc) and g_t4 (GCC_COLL=t4: tuned alltoall 4, ucc/hcoll off)
# 2 libraries x 9 configurations x 3 = 54 runs x ~1.5-2.5 min (p=256 Intel runs ~7 s/step) -> 1.5-2.3 h.
# Every run is a line in ~/guohaojie/results/p256_diag.csv (columns as toolchain_mn.csv; settings starts with the
# configuration name); runs are stopped and repeated once on a broken start (mn_lib.sh).
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/intel && sbatch p256_diag.sbatch
#   bash toolchain_mn_parse.sh ~/guohaojie/results/p256_diag.csv
export INTEL_IMAGE=${P256_IMAGE:-v2}
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/mn_lib.sh
precheck
NODES=${SLURM_JOB_NUM_NODES:-2}; N=512; DT=1e-6; NS=10; EREF=1.7950250065e+08; REP=${P256_REP:-3}
LIBS=${P256_LIBS:-"fftw p3dfft"}
CSV=${P256_CSV:-$HOME/guohaojie/results/p256_diag.csv}
COMMON="NS_FFTW_PLAN=measure NS_DECOMP=slab"
TMP=$SCR/logs/p256_$SLURM_JOB_ID.d; mkdir -p $TMP "$(dirname "$CSV")"
COMMIT=$(git -C $SCR/src/NavierStockes-3D log -1 --format=%h 2>/dev/null || echo unknown)
NODELIST=$(echo "${SLURM_JOB_NODELIST:-?}" | tr ',' ';')
KEEP="^decomposition:|planner flag|P3DFFT library|Running on|Threads \(runtime\)|planning time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|abort|intel_mn_env:|GCCENV:|UCXPROBE|MPI startup\(\): +(Intel|libfabric provider)"
[ -f "$CSV" ] || echo "commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,step_s,fft_s,plan_s,energy,energy_ok,ucx_version,retries,exit_code,ucx_reason" > "$CSV"
# configuration: name | toolchain | layout | IMN_* or GCC_COLL settings
CONFIGS=(
  "i_a_base|intel|128x1|IMN_ALLTOALL="
  "i_b_rc|intel|128x1|IMN_ALLTOALL= IMN_UCX_TLS=rc_mlx5,sm,self"
  "i_c_dc|intel|128x1|IMN_ALLTOALL= IMN_UCX_TLS=dc_mlx5,sm,self"
  "i_d_rndvinf|intel|128x1|IMN_ALLTOALL= IMN_EXTRA=UCX_RNDV_THRESH=inf"
  "i_e_rndv4k|intel|128x1|IMN_ALLTOALL= IMN_EXTRA=UCX_RNDV_THRESH=4096"
  "i_f_pinintel|intel|128x1|IMN_ALLTOALL= IMN_PIN=intel"
  "i_a_base|intel|64x2|IMN_ALLTOALL="
  "g_def|gcc|128x1|GCC_COLL=def"
  "g_t4|gcc|128x1|GCC_COLL=t4"
)
echo "p256_diag: nodes=$NODES ($SLURM_JOB_NODELIST) N=$N steps=$NS energy_ref=$EREF reps=$REP libs=$LIBS  commit $COMMIT  csv $CSV"
for c in "${CONFIGS[@]}"; do echo "  config $c"; done
job_containers_init
mn_init $TMP /work/logs/p256_$SLURM_JOB_ID.d

bin_of() { case $1:$2 in gcc:fftw) echo /work/build/bin_gcc/ns_fftw;; gcc:p3dfft) echo /work/build/bin_p3d/gcc/ns_p3dfft;;
                          intel:fftw) echo /work/build/bin_intel/ns_fftw;; intel:p3dfft) echo /work/build/bin_p3d/intel/ns_p3dfft;; esac; }
name_of() { case $1 in fftw) echo FFTW;; p3dfft) echo P3DFFT;; esac; }

# intel_cfg "<IMN_* settings>" <mn_intel arguments> — only this configuration's IMN_* (subshell)
intel_cfg() {
    local set=$1; shift
    ( unset IMN_ALLTOALL IMN_ALLTOALLV IMN_UCX_TLS IMN_EXTRA IMN_PIN; export $set; mn_intel "$@" )
}
# C <config string> <lib>
C() {
    local cname tc L set ppn nt bin label raw tmo tl envs
    IFS='|' read -r cname tc L set <<< "$1"; ppn=${L%x*}; nt=${L#*x}
    bin=$(bin_of $tc $2); envs=$COMMON
    label="$(name_of $2)-${N}_${tc}-${cname}_n${NODES}_$L"; raw=$TMP/$label.r$rep.raw
    tmo=$(mn_timeout 512 $NODES $L); tl=$(( tmo / 60 + 2 ))
    echo ">>> [$(date +%H:%M:%S)] $label ranks=$((NODES * ppn)) threads=$nt plan=measure | rep=$rep limit=${tmo}s $set"
    if [ $tc = gcc ]; then
        envs="$envs $(mn_gcc_coll ${set#GCC_COLL=})"
        mn_run $label $raw $tmo mn_gcc $NODES $ppn $nt $tl "$envs" $bin $N $N $N $DT $NS $nt bench
    else
        mn_run $label $raw $tmo intel_cfg "$set" $NODES $ppn $nt $tl "$envs" $bin $N $N $N $DT $NS $nt bench
    fi
    grep -E "$KEEP" $raw | awk '!/UCXPROBE/ || !u++'
    mn_extract $raw $EREF
    echo "UCX: $label rep=$rep version=$MN_UCXV${MN_UCXWHY:+ reason=$MN_UCXWHY}"
    [ $tc = intel ] && [ "$MN_UCXV" != 1.17.0 ] && echo "WARNING: UCX $MN_UCXV loaded, expected 1.17.0 ($label rep=$rep${MN_UCXWHY:+, $MN_UCXWHY})"
    [ "$MN_EOK" = no ] && echo "WARNING: kinetic energy ${MN_EN:-missing}, reference $EREF"
    echo "<<< exit=$MN_RC"
    echo "$COMMIT,$SLURM_JOB_ID,$NODELIST,$tc,$(echo "$cname $set" | tr ',' ';'),$(name_of $2),$N,$NODES,$ppn,$nt,$rep,$MN_STEP,$MN_FFT,$MN_PLAN,$MN_EN,$MN_EOK,$MN_UCXV,$MN_RETRIES,$MN_RC,$MN_UCXWHY" \
        | tee -a "$CSV" | sed 's/^/CSV /'
    echo
}
for rep in $(seq 1 $REP); do
    echo "=============== repetition $rep"
    for lib in $LIBS; do
        for c in "${CONFIGS[@]}"; do C "$c" $lib; done
    done
done
cleanup_all
