#!/bin/bash
#SBATCH -J hello2
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 01:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/hello2_%j.out
# hello_2node — can Intel MPI in the Intel container run on 2 nodes x 128 ranks? Tries several launchers
# and fabrics, one after another. Every message size is a SEPARATE srun (MPI_Init only once per process);
# sizes 8192, 65536, 262144 bytes per rank pair. Every srun ends with
#     RESULT variant=<name> func=alltoall|alltoallv bytes=<size> status=OK|FAIL|SKIP rc=<exit code> a2a_us=... GBs_node_offnode=...
# Sizes per rank pair: 8 KB, 64 KB, 256 KB, 512 KB, 1 MB. hello_a2a measures MPI_Alltoall or, with mode a2av,
# MPI_Alltoallv with uniform counts (as P3DFFT / heFFTe / AccFFT call it); HFUNC=a2a|a2av per variant.
# If the smallest size fails, the larger ones of that variant are skipped. A hung srun is killed after 200 s.
# Files in logs/hello2_<job>.d/: <variant>.<bytes>.raw = full output of every srun; for a FAILED srun also
# <variant>.<bytes>.txt = the "MPI startup"/PMI lines (I_MPI_DEBUG=5), the libfabric log (FI_LOG_LEVEL=warn, E group)
# and the last 20 error lines.
#   0  discovery: PMI libraries on the host and inside the container, srun inside the container?
#   R  reference: Open MPI (GCC container, srun --mpi=pmix) — the fallback if all Intel variants fail
#   A  srun --mpi=pmi2 (the host has no libpmi2.so):
#        A_cont<i>_pmi2   I_MPI_PMI_LIBRARY=<each libpmi2.so found in the container>
#        A1_pmi2          Intel MPI's own PMI-2 client (I_MPI_PMI=pmi2 I_MPI_PMI2=yes, no I_MPI_PMI_LIBRARY), shm:ofi
#        A1_pmi2_ppn2     the same with 2 ranks per node (does the shm failure depend on the rank count?)
#        A1_ofi_only      the same, I_MPI_FABRICS=ofi (no shared memory), FI_PROVIDER=mlx
#   B1 srun --mpi=pmix + I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=<each libpmix.so found in the container>
#   B2 srun --mpi=pmix + the HOST's PMIx 3.2.3 (Slurm plugin pmix_v3): /opt/pmix-3.2.3 mounted at the same path
#      (its MCA components are found at the compiled-in prefix); dependencies the container does not have are
#      copied from the host to logs/hello2_<job>.d/hostpmix, mounted at /hostpmix; FI_PROVIDER = mlx, then verbs.
#      (NOT /usr/mpi/gcc/openmpi-*/lib/libpmix.so: that one belongs to the host's Open MPI.)
#   D  the PMI client libraries installed in the image by rebuild_container.sh (paths from /opt/ns3d_pmi_libs.txt):
#        D1_pmi2lib_<prov>       srun --mpi=pmi2, I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=<libpmi2.so.0 of the image (Slurm client)>
#        D2_pmixlib_<prov>       srun --mpi=pmix, I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=<libpmix.so.2 of the image>
#        D3_pmi2lib_ppn2_<prov>  as D1 with 2 ranks per node
#      I_MPI_FABRICS=shm:ofi, <prov> = FI_PROVIDER mlx, then verbs. Skipped when the image has no such library.
#   E  the D1 launcher (srun --mpi=pmi2, I_MPI_PMI=pmi2, I_MPI_PMI_LIBRARY=libpmi2.so.0 of the image, shm:ofi) with
#      FI_PROVIDER_PATH = $I_MPI_ROOT/opt/mpi/libfabric/lib/prov only (no system libfabric directory), FI_LOG_LEVEL=warn:
#        E1_verbs_rxm    FI_PROVIDER="verbs;ofi_rxm"
#        E2_psm3         FI_PROVIDER=psm3 (Intel's libpsm3-fi.so, runs over the verbs device)
#        E3_mlx_ucxtls   FI_PROVIDER=mlx, UCX_TLS=rc,ud,sm,self UCX_NET_DEVICES=mlx5_2:1 (NIC of GPU0; E3_NET_DEVICES=...),
#                        rank 0 prints ucx_info -v
#        E4_tcp          FI_PROVIDER=tcp — diagnosis: if only this works, launch and PMI are fine and only the
#                        high-speed network layer is missing
#   E5 as E3 (D1 pmi2 launcher, FI_PROVIDER=mlx, Intel provider directory only), only the UCX settings change;
#      UCX_LOG_LEVEL=info, rank 0 prints ucx_info -v; the transports/devices UCX chose go to <variant>.<bytes>.txt
#      (also when the run works) and into the RESULT line as ucx=<tl/dev,...>:
#        E5a_mlx5tls   UCX_TLS=rc_mlx5,ud_mlx5,sm,self (as run_env.sh for Open MPI, CPU mode), no UCX_NET_DEVICES
#        E5b_dc        UCX_TLS=dc_mlx5,sm,self, no UCX_NET_DEVICES
#        E5c_default   UCX_TLS and UCX_NET_DEVICES unset (UCX defaults)
#        E5d_alldev    UCX_TLS=rc_mlx5,ud_mlx5,sm,self, UCX_NET_DEVICES=mlx5_0:1,...,mlx5_9:1 (8 HCAs; E5D_NET_DEVICES=...)
#   E6 as E5d_alldev (v2 image), UCX of HPC-X 2.19 (the one Open MPI uses; staged by stage_hpcx_ucx.sh in
#      /work/opt_ucx/hpcx): LD_LIBRARY_PATH=/work/opt_ucx/hpcx/lib:..., UCX_MODULE_DIR=/work/opt_ucx/hpcx/lib/ucx
#        E6a_hpcxucx         only that
#        E6b_hpcxucx_relax   + UCX_IB_PCI_RELAXED_ORDERING=on
#        E6c_a2a2            + I_MPI_ADJUST_ALLTOALL=2 (Isend/Irecv + waitall)
#        E6d_a2a3            + I_MPI_ADJUST_ALLTOALL=3 (pairwise exchange)
#        E6e_a2a4            + I_MPI_ADJUST_ALLTOALL=4 (Plum)
#        E6f_mlxtls          + FI_MLX_TLS=rc_mlx5,ud_mlx5,sm,self
#      hello_a2a rank 0 prints "MAPS <path>" of the UCX libraries it really mapped; if E6a does not run with the
#      HPC-X libucp (missing stage, dependency or version problem), E6b-f fall back to the image's UCX 1.12.1.
# Every RESULT line ends with ucx_version=<version printed by ucx_info -v> (R: Open MPI's ucx_info) and, when
# hello_a2a printed it, ucx_lib=<directory of the libucp really loaded>.
# Summary: one row per (variant, function, size), then per (function, size) the best Open MPI setting (R*, RT*) and
# the best Intel setting (all other groups) and their ratio (target >= 0.8).
#   RT Open MPI tuning (GCC container, srun --mpi=pmix, run_env.sh CPU mode). Before the runs: ucx_info -v, all
#      UCX_* and OMPI_MCA_* variables, and the coll components hcoll / ucc / tuned from ompi_info.
#        RT_<func>_def_<dev>   no coll setting (Open MPI default; for a2a the same as R_ompi_pmix)
#        RT_a2a_t<k>_<dev>     OMPI_MCA_coll_tuned_use_dynamic_rules=1, coll_tuned_alltoall_algorithm = 0, 2, 3, 4
#        RT_a2av_t<k>_<dev>    the same with coll_tuned_alltoallv_algorithm = 0, 1, 2
#      For the t<k> variants hcoll and ucc are switched off (OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_coll_ucc_enable=0),
#      otherwise a higher-priority component would take the collective and the tuned algorithm would never run;
#      RT_HCOLL_OFF=0 keeps them. <dev>: d = UCX default devices, 8h = the 8 HCAs of E5d/E6 (RT_DEVS="d 8h").
#   E7 Intel, E6d settings (intel_mn_env.sh: pmi2, mlx, HPC-X UCX 1.17, I_MPI_ADJUST_ALLTOALL=3):
#        E7_a2a_e6d            MPI_Alltoall (E6d at the new sizes)
#        E7_a2av_def           MPI_Alltoallv, I_MPI_ADJUST_ALLTOALLV unset
#        E7_a2av_1, E7_a2av_2  MPI_Alltoallv, I_MPI_ADJUST_ALLTOALLV=1 / 2
#   C  mpirun -bootstrap slurm inside the container (hydra starts its proxies with srun; needs srun in the container)
#   F  fabrics, with the first launcher that worked: I_MPI_FABRICS=shm:ofi with FI_PROVIDER = (auto) mlx verbs tcp
# Every Intel variant also reports the PMI API Intel MPI really used ("MPI startup(): PMI API: ..." with I_MPI_DEBUG=5):
#   RESULT ... pmi_api=<pmi|pmi2|pmix>; PMI_API_MISMATCH(expected=...) when it is not the one the variant asked for.
# hello_a2a: MPI_Alltoall, data checked; build it with build_intel_solvers.sh.
# Argument: the Intel image (default: the one in $MI, i.e. ns3d_intel.sqsh); "v2" = containers/ns3d_intel_v2.sqsh.
# HELLO_GROUPS selects the groups (default "R A B D E E5 E6 C F"; "R E5 E6" = reference, E5d for comparison, HPC-X UCX), e.g. only the reference and the E group on the new image:
#   sbatch --export=ALL,HELLO_GROUPS="R RT E6 E7" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
#   sbatch --export=ALL,HELLO_GROUPS="R E" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
#   sbatch --export=ALL,HELLO_GROUPS="R D F" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch
#   sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
[ -n "$1" ] && export INTEL_IMAGE=$1                   # image choice, see intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
GROUPS_=${HELLO_GROUPS:-"R RT A B D E E5 E6 E7 C F"}
want() { [[ " $GROUPS_ " == *" $1 "* ]]; }
IE="source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1"
HB=/work/build/bin_intel/hello_a2a
TMP=$SCR/logs/hello2_$SLURM_JOB_ID.d; mkdir -p $TMP     # = /work/logs/hello2_$SLURM_JOB_ID.d in the containers
WTMP=/work/logs/hello2_$SLURM_JOB_ID.d
SIZES="8192:20 65536:10 262144:5 524288:4 1048576:3"   # bytes per pair : iterations (1 MB: 512 MB buffers per rank)
RES=$TMP/results.txt; : > $RES                         # RESULT lines (the job's .out file is stdout: grep cannot read it)
res() { echo "RESULT $*" | tee -a $RES; }
FIRST_OK=""                                            # "<srun options>|<inner environment>" of the first launcher that worked

# T <variant> <srun options> <inner command>
#   The inner command runs inside the container on every rank; @A@ is replaced by "<bytes> <iterations>".
#   One srun per message size. Returns 0 if all sizes are OK.
T() {
    local v=$1 so=$2 cmd=$3 si sz it c rc st l f
    for si in $SIZES; do
        sz=${si%:*}; it=${si#*:}; c=${cmd//@A@/$sz $it ${HFUNC:-a2a}}; f=$TMP/$v.$sz.raw
        echo ">>> [$(date +%H:%M:%S)] $v bytes=$sz | srun $so ... | $c"
        timeout 200 srun $so --time=3 --kill-on-bad-exit=1 bash -c "$c" > $f 2>&1; rc=$?
        if [ $sz = 8192 ]; then   # launcher/fabric details once per variant
            grep -E "HELLO_|node leader|MPI startup\(\): (libfabric|Load|I_MPI_|Rank)|provider|rror|abort|PMI" $f | head -20
        else
            grep -E "HELLO_|rror|abort" $f | head -5
        fi
        if grep -q "^HELLO_OK" $f && [ $rc = 0 ]; then st=OK; else
            st=FAIL; tail -5 $f
            { echo "# $v bytes=$sz rc=$rc   srun $so ... | $c"
              echo "# ---- MPI startup / PMI lines"
              grep -E "MPI startup|PMI|pmi" $f | sort | uniq -c | sort -rn | head -60
              if grep -q "libfabric:" $f; then
                  echo "# ---- libfabric log (FI_LOG_LEVEL=warn; pid/time stripped, counted)"
                  grep "libfabric:" $f | sed -E 's/libfabric:[0-9]+:[0-9]+:/libfabric:/' | sort | uniq -c | sort -rn | head -40
              fi
              echo "# ---- last 20 error lines"
              grep -iE "error|fail|abort|unable|cannot|denied|not found|killed|signal" $f | tail -20
              echo "# ---- last 20 lines"
              tail -20 $f; } > $TMP/$v.$sz.txt
            echo "    failure details: $TMP/$v.$sz.txt"
        fi
        l=$(grep "^HELLO_OK" $f | tail -1)
        # UCX: what it really selected (UCX_LOG_LEVEL=info), written to <variant>.<bytes>.txt also for a run that worked
        local ux=""
        if [ -n "$UCXLOG" ]; then
            { [ $st = OK ] && echo "# $v bytes=$sz rc=$rc status=OK   srun $so ... | $c"
              echo "# ---- ucx_info -v (rank 0)"
              grep -E "^# " $f | head -6
              echo "# ---- UCX transport / device selection (UCX_LOG_LEVEL=info; time, host, pid stripped; counted)"
              grep -E "UCX +(INFO|WARN|ERROR)|ucp_ep|transport|ep_cfg|lane" $f | grep -v "^>>>" \
                | sed -E 's/^\[[0-9.]+\] +\[[^]]+\] +//' | sort | uniq -c | sort -rn | head -60
              echo "# ---- transport/device pairs seen"
              grep -oE "[a-z_0-9]+/(mlx5_[0-9]+:[0-9]+|memory|lo|ib[0-9a-z]*|eth[0-9a-z]*|en[0-9a-z]*)" $f | sort | uniq -c | sort -rn | head -20
            } >> $TMP/$v.$sz.txt
            ux=$(grep -oE "(rc|ud|dc|rc_mlx5|ud_mlx5|dc_mlx5|rc_verbs|ud_verbs|tcp)/[a-z0-9_]+(:[0-9]+)?" $f | sort -u | head -12 | tr '\n' ',' | sed 's/,$//')
            ux="ucx=${ux:-none}"
            echo "    UCX selection: ${ux#ucx=}   (details: $TMP/$v.$sz.txt)"
        fi
        # PMI API actually used by Intel MPI (I_MPI_DEBUG >= 2 prints it); EXPECT_API = what the variant asked for
        local api pa=""
        api=$(grep -m1 -o "PMI API: *[A-Za-z0-9_]*" $f | awk '{print $NF}')
        if [ -n "$EXPECT_API" ] || [ -n "$api" ]; then
            pa="pmi_api=${api:-none}"
            if [ -n "$EXPECT_API" ] && [ "$api" != "$EXPECT_API" ]; then
                pa="$pa PMI_API_MISMATCH(expected=$EXPECT_API)"; echo "    WARNING: Intel MPI used PMI API '${api:-none}', not $EXPECT_API"
            fi
        fi
        # UCX version (ucx_info -v of rank 0: "Library version: X" since 1.13, "UCT version=X" in 1.12) and the libucp
        # directory hello_a2a really mapped; UCXV_FALLBACK = version known from elsewhere (R: Open MPI's ucx_info)
        local uv ul uf
        uv=$(grep -m1 -oE "(Library version: *|UCT version=)[0-9]+\.[0-9]+(\.[0-9]+)?" $f | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?$")
        ul=$(grep -m1 -E "^MAPS .*/libucp\.so" $f | awk '{print $2}')
        uf="ucx_version=${uv:-${UCXV_FALLBACK:-unknown}}"; [ -n "$ul" ] && uf="$uf ucx_lib=$(dirname $ul)"
        local fn; fn=$(echo "$l" | grep -o 'func=[^ ]*' | cut -d= -f2); [ -z "$fn" ] && { [ "${HFUNC:-a2a}" = a2av ] && fn=alltoallv || fn=alltoall; }
        res "variant=$v func=$fn bytes=$sz status=$st rc=$rc $(echo "$l" | grep -o 'a2a_us=[^ ]*') $(echo "$l" | grep -o 'GBs_node_offnode=[^ ]*') $pa $ux $uf"; echo
        [ $st = OK ] || { [ "$si" != "${SIZES##* }" ] && echo "    (larger sizes of $v skipped)"; return 1; }
    done
    return 0
}
SR2="-N 2 --ntasks-per-node=128 -c 1 --cpu-bind=cores"
HELLO="$HB @A@"
IFAB="I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5"

# ---------------------------------------------------------------- 0) discovery
echo "=============== 0) discovery"
echo "--- host $(hostname) ($(. /etc/os-release; echo $PRETTY_NAME)): srun $(srun --version 2>/dev/null); srun --mpi=list:"; srun --mpi=list 2>&1 | tr '\n' ' '; echo
echo "--- PMI client libraries on the host:"
find /usr/lib64 /usr/lib /usr/local/lib /opt/slurm* /usr/local/slurm* -maxdepth 4 \( -name 'libpmi2.so*' -o -name 'libpmix.so*' -o -name 'libpmi.so*' \) 2>/dev/null | sort -u
echo "--- inside the Intel container (${INTEL_IMAGE:-image of \$MI}):"
srun -N1 -n1 --time=3 $MI bash -c "$IE; cat /opt/ns3d_pmi_libs.txt > $WTMP/manifest.txt 2>/dev/null; . /etc/os-release; echo \"\$PRETTY_NAME  I_MPI_ROOT=\$I_MPI_ROOT\"; ls -l $HB; command -v srun && srun --version || echo 'no srun inside the container'; \
    find / \( -path /proc -o -path /sys -o -path /dev -o -path /work -o -path /scratch \) -prune -o \( -name 'libpmi2.so*' -o -name 'libpmix.so*' -o -name 'libpmi.so*' \) -print 2>/dev/null | sort -u > $WTMP/pmilibs.txt; cat $WTMP/pmilibs.txt; \
    fi_info -l 2>/dev/null | grep -E '^[a-z]' | tr '\n' ' '; echo"
# hello_a2a prints "MAPS <lib>" since the E6 group: rebuild it in the Intel image when the source is newer
if [ ! -x $SCR/build/bin_intel/hello_a2a ] || [ $SCR/src/NavierStockes-3D/scripts/intel/hello_a2a.c -nt $SCR/build/bin_intel/hello_a2a ]; then
    echo "--- rebuilding $HB (source newer or binary missing)"
    srun -N1 -n1 --time=3 $MI bash -c "$IE; mpicc -O2 /work/src/NavierStockes-3D/scripts/intel/hello_a2a.c -o $HB" \
        || echo "ERROR: could not build $HB"
fi
PMI2=$(grep -E 'libpmi2\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
PMIX=$(grep -E 'libpmix\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
echo "candidates pmi2: ${PMI2:-none}   pmix: ${PMIX:-none}"
# libraries installed by rebuild_container.sh (manifest), else the usual Ubuntu paths if the search found them
D_PMI2=$(sed -n 's/^PMI2_LIB=//p' $TMP/manifest.txt 2>/dev/null); D_PMIX=$(sed -n 's/^PMIX_LIB=//p' $TMP/manifest.txt 2>/dev/null)
[ -z "$D_PMI2" ] && D_PMI2=$(grep -E '/libpmi2\.so\.0$' $TMP/pmilibs.txt 2>/dev/null | head -1)
[ -z "$D_PMIX" ] && D_PMIX=$(grep -E '/libpmix\.so\.2$' $TMP/pmilibs.txt 2>/dev/null | head -1)
echo "image manifest /opt/ns3d_pmi_libs.txt: $(tr '\n' ' ' < $TMP/manifest.txt 2>/dev/null || echo none)"
echo "D variants: pmi2 lib ${D_PMI2:-none}   pmix lib ${D_PMIX:-none}"

# Open MPI hello_a2a (GCC container), built once per job
build_ompi_hello() {
    [ -n "$OMPI_HELLO_OK" ] && return 0
    srun -N1 -n1 --time=3 $M bash -c "source /etc/profile.d/ns3d_env.sh; mpicc -O2 /work/src/NavierStockes-3D/scripts/intel/hello_a2a.c -o /work/build/bin_gcc/hello_a2a_ompi" \
        && OMPI_HELLO_OK=1
}
HBO=/work/build/bin_gcc/hello_a2a_ompi

if want R; then
# ---------------------------------------------------------------- R) Open MPI reference (GCC container)
echo "=============== R) Open MPI reference, GCC container"
# the UCX Open MPI uses (HPC-X), for comparison with the E5/E6 RESULT lines
srun -N1 -n1 --time=3 $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; echo \"ucx_info: \$(command -v ucx_info)\"; ucx_info -v; \
    p=\$(ompi_info --path libdir --parsable 2>/dev/null | cut -d: -f3); ls \$p/openmpi/mca_pml_ucx.so >/dev/null 2>&1 && ldd \$p/openmpi/mca_pml_ucx.so | grep -E 'libuc[pst]'; \
    env | grep -E '^UCX_' | sort" > $TMP/ompi_ucx.txt 2>&1
echo "--- Open MPI (GCC container) UCX:"; sed 's/^/    /' $TMP/ompi_ucx.txt
OMPI_UCXV=$(grep -m1 -oE "(Library version: *|UCT version=)[0-9]+\.[0-9]+(\.[0-9]+)?" $TMP/ompi_ucx.txt | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?$")
build_ompi_hello \
  && UCXV_FALLBACK=$OMPI_UCXV T R_ompi_pmix "--mpi=pmix $SR2 $M" "export NS_MODE=cpu; source /work/build/run_env.sh; /work/build/bin_gcc/hello_a2a_ompi @A@"
fi   # group R

if want RT; then
# ---------------------------------------------------------------- RT) Open MPI coll tuning (GCC container)
echo "=============== RT) Open MPI: coll tuned algorithms for alltoall / alltoallv, UCX devices"
srun -N1 -n1 --time=3 $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
    echo '--- ucx_info -v'; ucx_info -v; \
    echo '--- UCX_* and OMPI_MCA_* (run_env.sh, CPU mode)'; env | grep -E '^(UCX_|OMPI_MCA_)' | sort; \
    echo '--- coll components (ompi_info)'; ompi_info | grep -E 'MCA coll'; \
    for c in hcoll ucc tuned; do echo \"--- coll \$c: enable / priority / algorithm parameters\"; \
        ompi_info --param coll \$c --level 9 2>/dev/null | grep -E \"coll_\${c}_(enable|priority)|coll_tuned_(use_dynamic_rules|alltoallv?_algorithm)\\\"\" | sed 's/^ *//'; done" \
    > $TMP/ompi_coll.txt 2>&1
sed 's/^/    /' $TMP/ompi_coll.txt
[ -z "$OMPI_UCXV" ] && OMPI_UCXV=$(grep -m1 -oE "(Library version: *|UCT version=)[0-9]+\.[0-9]+(\.[0-9]+)?" $TMP/ompi_coll.txt | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?$")
RTOFF="OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_coll_ucc_enable=0"; [ "${RT_HCOLL_OFF:-1}" = 0 ] && RTOFF=""
echo "tuned variants: dynamic rules on${RTOFF:+, hcoll and ucc off ($RTOFF)}"
build_ompi_hello || echo "ERROR: could not build $HBO"
for dev in ${RT_DEVS:-d 8h}; do
  dv=""; [ $dev = 8h ] && dv="export UCX_NET_DEVICES=$E5DEV;"
  for spec in "a2a|def|" "a2a|t0|0" "a2a|t2|2" "a2a|t3|3" "a2a|t4|4" "a2av|def|" "a2av|t0|0" "a2av|t1|1" "a2av|t2|2"; do
    fn=${spec%%|*}; tag=$(echo $spec | cut -d'|' -f2); alg=${spec##*|}
    if [ -z "$alg" ]; then mca=""
    elif [ $fn = a2a ]; then mca="export OMPI_MCA_coll_tuned_use_dynamic_rules=1 OMPI_MCA_coll_tuned_alltoall_algorithm=$alg $RTOFF;"
    else mca="export OMPI_MCA_coll_tuned_use_dynamic_rules=1 OMPI_MCA_coll_tuned_alltoallv_algorithm=$alg $RTOFF;"; fi
    HFUNC=$fn UCXV_FALLBACK=$OMPI_UCXV T RT_${fn}_${tag}_$dev "--mpi=pmix $SR2 $M" \
      "export NS_MODE=cpu; source /work/build/run_env.sh; $mca $dv [ \${SLURM_PROCID:-0} = 0 ] && env | grep -E '^(OMPI_MCA_coll|UCX_NET)' | sort | tr '\\n' ' ' && echo; $HBO @A@"
  done
done
fi   # group RT

if want A; then
# ---------------------------------------------------------------- A) srun --mpi=pmi2
echo "=============== A) srun --mpi=pmi2"
EXPECT_API=pmi2
i=0
for lib in $PMI2; do
  i=$((i+1)); e="export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$lib; unset I_MPI_HYDRA_BOOTSTRAP"
  T A_cont${i}_pmi2 "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
done
# Intel MPI's own PMI-2 client: no I_MPI_PMI_LIBRARY. The last run (A0, I_MPI_PMI=pmi2 only) printed "PMI API: pmi"
# and failed in MPIDU_Init_shm_init; I_MPI_PMI2=yes is the older switch for the same thing, set as well.
# Check the "PMI API:" line in the log: it must say pmi2.
e="export I_MPI_PMI=pmi2 I_MPI_PMI2=yes; unset I_MPI_PMI_LIBRARY I_MPI_HYDRA_BOOTSTRAP"
T A1_pmi2 "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
# 2 ranks per node (not used as the launcher for F: F runs 128 per node)
T A1_pmi2_ppn2 "--mpi=pmi2 -N 2 --ntasks-per-node=2 -c 1 --cpu-bind=cores $MI" "$IE; $e; export $IFAB; $HELLO"
# no shared memory: OFI (mlx) also inside a node
eo="$e; export I_MPI_FABRICS=ofi FI_PROVIDER=mlx"
T A1_ofi_only "--mpi=pmi2 $SR2 $MI" "$IE; export $IFAB; $eo; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$eo"}
fi   # group A

if want B; then
# ---------------------------------------------------------------- B1) srun --mpi=pmix, libpmix of the container
echo "=============== B1) srun --mpi=pmix, libpmix from the container"
EXPECT_API=pmix
i=0
for lib in $PMIX; do
  i=$((i+1)); e="export I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=$lib; unset I_MPI_HYDRA_BOOTSTRAP"
  T B1_pmix_cont$i "--mpi=pmix $SR2 $MI" "$IE; $e; export $IFAB; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmix $SR2 $MI|$e"}
done
[ -z "$PMIX" ] && res "variant=B1_pmix_cont bytes=- status=SKIP rc=- (no libpmix.so inside the container)"

# ---------------------------------------------------------------- B2) srun --mpi=pmix, libpmix of the HOST
echo "=============== B2) srun --mpi=pmix, host PMIx mounted at its own path"
HPMIX=${HOST_PMIX:-/opt/pmix-3.2.3/lib/libpmix.so.2.2.33}
HPROOT=$(dirname $(dirname $HPMIX))                    # /opt/pmix-3.2.3
STG=$TMP/hostpmix
if [ -f "$HPMIX" ]; then
  rm -rf $STG; mkdir -p $STG
  echo "host $HPMIX, prefix $HPROOT: $(ls $HPROOT/lib | tr '\n' ' ')"
  echo "MCA components: $(ls $HPROOT/lib/pmix 2>/dev/null | wc -l) files in $HPROOT/lib/pmix"
  # the existing --container-mounts of $MI, plus the PMIx prefix (same path) and the staging directory
  ADD="$HPROOT:$HPROOT:ro,$STG:/hostpmix:ro"
  if [[ "$MI" == *--container-mounts=* ]]; then MIH=$(echo "$MI" | sed "s|--container-mounts=\([^ ]*\)|--container-mounts=\1,$ADD|")
  else MIH="$MI --container-mounts=$ADD"; fi
  echo "MIH = $MIH"
  # dependencies (libevent, libhwloc, ...) the container cannot resolve -> copied from the host to /hostpmix;
  # repeated for the dependencies of the copied libraries
  cat > $TMP/chk_pmix.sh <<EOF
source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/hostpmix:$HPROOT/lib:\$LD_LIBRARY_PATH
for f in $HPMIX $HPROOT/lib/pmix/*.so /hostpmix/*.so*; do
  [ -f "\$f" ] || continue
  LD_TRACE_LOADED_OBJECTS=1 /lib64/ld-linux-x86-64.so.2 "\$f" 2>&1 | grep "=> not found" | awk '{print \$1}'
done | sort -u
EOF
  # where the host itself finds a library: ldconfig cache, else what ldd of the host PMIx libraries says
  hostlib() { ldconfig -p | awk -v s=$1 '$1 == s && /x86-64/ {print $NF; exit}' | grep . ||
              ldd $HPMIX $HPROOT/lib/pmix/*.so 2>/dev/null | awk -v s=$1 '$1 == s && $3 ~ /^\// {print $3; exit}'; }
  for pass in 1 2 3 4; do
    miss=$(srun -N1 -n1 --time=3 $MIH bash $WTMP/chk_pmix.sh 2>/dev/null)
    [ -z "$miss" ] && { echo "pass $pass: all dependencies resolve inside the container"; break; }
    echo "pass $pass: missing in the container: $(echo $miss)"
    for s in $miss; do
      p=$(hostlib $s)
      if [ -n "$p" ]; then cp -L $p $STG/$s && echo "  + $s  <- $p"; else echo "  ! $s not on the host either"; fi
    done
  done
  echo "staged in /hostpmix: $(ls $STG | tr '\n' ' ')"
  e="export I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=$HPROOT/lib/libpmix.so.2 LD_LIBRARY_PATH=/hostpmix:$HPROOT/lib:\$LD_LIBRARY_PATH; unset I_MPI_HYDRA_BOOTSTRAP"
  for prov in mlx verbs; do
    T B2_pmix_host_$prov "--mpi=pmix $SR2 $MIH" "$IE; $e; export $IFAB FI_PROVIDER=$prov; $HELLO" \
      && FIRST_OK=${FIRST_OK:-"--mpi=pmix $SR2 $MIH|$e"}
  done
else
  res "variant=B2_pmix_host bytes=- status=SKIP rc=- (no $HPMIX on the host; set HOST_PMIX=...)"
fi
fi   # group B

if want D; then
# ---------------------------------------------------------------- D) PMI libraries of the (rebuilt) image
echo "=============== D) PMI client libraries of the image (rebuild_container.sh)"
if [ -n "$D_PMI2" ]; then
  e="export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$D_PMI2; unset I_MPI_HYDRA_BOOTSTRAP"
  for prov in mlx verbs; do
    EXPECT_API=pmi2 T D1_pmi2lib_$prov "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB FI_PROVIDER=$prov; $HELLO" \
      && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
  done
else
  res "variant=D1_pmi2lib bytes=- status=SKIP rc=- (no libpmi2.so.0 in the image: use the image of rebuild_container.sh)"
fi
if [ -n "$D_PMIX" ]; then
  e="export I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=$D_PMIX; unset I_MPI_HYDRA_BOOTSTRAP"
  for prov in mlx verbs; do
    EXPECT_API=pmix T D2_pmixlib_$prov "--mpi=pmix $SR2 $MI" "$IE; $e; export $IFAB FI_PROVIDER=$prov; $HELLO" \
      && FIRST_OK=${FIRST_OK:-"--mpi=pmix $SR2 $MI|$e"}
  done
else
  res "variant=D2_pmixlib bytes=- status=SKIP rc=- (no libpmix.so.2 in the image: use the image of rebuild_container.sh)"
fi
if [ -n "$D_PMI2" ]; then   # 2 ranks per node: not used as the launcher for F (F runs 128 per node)
  e="export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$D_PMI2; unset I_MPI_HYDRA_BOOTSTRAP"
  for prov in mlx verbs; do
    EXPECT_API=pmi2 T D3_pmi2lib_ppn2_$prov "--mpi=pmi2 -N 2 --ntasks-per-node=2 -c 1 --cpu-bind=cores $MI" "$IE; $e; export $IFAB FI_PROVIDER=$prov; $HELLO"
  done
fi
fi   # group D

# ---- shared by the E and E5 groups: the D1 launcher with Intel's libfabric providers only
E_PMI2=${D_PMI2:-/usr/lib/x86_64-linux-gnu/libpmi2.so.0}
E3NET=${E3_NET_DEVICES:-mlx5_2:1}
E5DEV=${E5D_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}
# \$I_MPI_ROOT is expanded inside the container (after intel_env.sh)
e="export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$E_PMI2 FI_PROVIDER_PATH=\$I_MPI_ROOT/opt/mpi/libfabric/lib/prov FI_LOG_LEVEL=warn; unset I_MPI_HYDRA_BOOTSTRAP"
# printed once, by rank 0: which provider files exist and that the system directory is gone from the path
pinfo="[ \${SLURM_PROCID:-0} = 0 ] && { echo \"FI_PROVIDER_PATH=\$FI_PROVIDER_PATH: \$(ls \$FI_PROVIDER_PATH 2>&1 | tr '\\n' ' ')\"; }"

if want E; then
# ---------------------------------------------------------------- E) D1 launcher, Intel's own libfabric providers only
echo "=============== E) srun --mpi=pmi2 + libpmi2.so.0, FI_PROVIDER_PATH = Intel libfabric providers only"
for v in "E1_verbs_rxm|export FI_PROVIDER='verbs;ofi_rxm'" \
         "E2_psm3|export FI_PROVIDER=psm3" \
         "E3_mlx_ucxtls|export FI_PROVIDER=mlx UCX_TLS=rc,ud,sm,self UCX_NET_DEVICES=$E3NET; [ \${SLURM_PROCID:-0} = 0 ] && ucx_info -v" \
         "E4_tcp|export FI_PROVIDER=tcp"; do
  name=${v%%|*}; pe=${v#*|}
  EXPECT_API=pmi2 T $name "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $pe; $pinfo; $HELLO" \
    && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
done
fi   # group E

if want E5; then
# ---------------------------------------------------------------- E5) as E3 (mlx), UCX settings only
echo "=============== E5) mlx provider (E3 launcher), UCX transport / device settings"
ux0="export FI_PROVIDER=mlx UCX_LOG_LEVEL=info; [ \${SLURM_PROCID:-0} = 0 ] && ucx_info -v"
for v in "E5a_mlx5tls|export UCX_TLS=rc_mlx5,ud_mlx5,sm,self; unset UCX_NET_DEVICES" \
         "E5b_dc|export UCX_TLS=dc_mlx5,sm,self; unset UCX_NET_DEVICES" \
         "E5c_default|unset UCX_TLS UCX_NET_DEVICES" \
         "E5d_alldev|export UCX_TLS=rc_mlx5,ud_mlx5,sm,self UCX_NET_DEVICES=$E5DEV"; do
  name=${v%%|*}; pe=${v#*|}
  UCXLOG=1 EXPECT_API=pmi2 T $name "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $ux0; $pe; $pinfo; $HELLO" \
    && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
done
fi   # group E5

# ---- shared by E6 and E7: E5d_alldev settings + HPC-X UCX
HUCX=/work/opt_ucx/hpcx
e5d="export FI_PROVIDER=mlx UCX_LOG_LEVEL=info UCX_TLS=rc_mlx5,ud_mlx5,sm,self UCX_NET_DEVICES=$E5DEV"
uh="export LD_LIBRARY_PATH=$HUCX/lib:\$LD_LIBRARY_PATH UCX_MODULE_DIR=$HUCX/lib/ucx PATH=$HUCX/bin:\$PATH"
# rank 0: version of the ucx_info first in PATH (HPC-X when staged) and the libucp it loads
uinfo="[ \${SLURM_PROCID:-0} = 0 ] && { echo \"ucx_info: \$(command -v ucx_info)\"; ucx_info -v; }"

if want E6; then
# ---------------------------------------------------------------- E6) E5d_alldev + UCX of HPC-X 2.19
echo "=============== E6) E5d_alldev settings with the UCX of HPC-X (stage_hpcx_ucx.sh -> /work/opt_ucx/hpcx)"
if [ -f $SCR/opt_ucx/hpcx/lib/libucp.so.0 ]; then
  echo "staged: $(grep -m1 -E 'version' $SCR/opt_ucx/hpcx/VERSION.txt)   check: $(grep -E '^(OK|FAIL)' $SCR/opt_ucx/hpcx/CHECK.txt 2>/dev/null | tr '\n' ' ' || echo 'not run')"
  UCXLOG=1 EXPECT_API=pmi2 T E6a_hpcxucx "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $e5d; $uh; $uinfo; $pinfo; $HELLO"
  if grep -q "^MAPS .*/opt_ucx/hpcx/lib/libucp" $TMP/E6a_hpcxucx.8192.raw 2>/dev/null && grep -q "^HELLO_OK" $TMP/E6a_hpcxucx.8192.raw; then
    echo "E6a: hello_a2a mapped $(grep -m1 '^MAPS .*libucp' $TMP/E6a_hpcxucx.8192.raw | awk '{print $2}') — E6b-f use the HPC-X UCX"
  else
    echo "NOTE: E6a did not run with the HPC-X libucp ($(grep -m1 '^MAPS .*libucp' $TMP/E6a_hpcxucx.8192.raw 2>/dev/null | awk '{print $2}')) — E6b-f fall back to the image's UCX"
    uh="true"
  fi
else
  res "variant=E6a_hpcxucx bytes=- status=SKIP rc=- (no $SCR/opt_ucx/hpcx/lib/libucp.so.0: run stage_hpcx_ucx.sh) ucx_version=-"
  echo "NOTE: E6b-f use the image's UCX"
  uh="true"
fi
for v in "E6b_hpcxucx_relax|export UCX_IB_PCI_RELAXED_ORDERING=on" \
         "E6c_a2a2|export I_MPI_ADJUST_ALLTOALL=2" \
         "E6d_a2a3|export I_MPI_ADJUST_ALLTOALL=3" \
         "E6e_a2a4|export I_MPI_ADJUST_ALLTOALL=4" \
         "E6f_mlxtls|export FI_MLX_TLS=rc_mlx5,ud_mlx5,sm,self"; do
  name=${v%%|*}; pe=${v#*|}
  UCXLOG=1 EXPECT_API=pmi2 T $name "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $e5d; $uh; $pe; $uinfo; $pinfo; $HELLO"
done
fi   # group E6

if want E7; then
# ---------------------------------------------------------------- E7) E6d settings, MPI_Alltoallv algorithms
echo "=============== E7) E6d settings (HPC-X UCX, I_MPI_ADJUST_ALLTOALL=3), MPI_Alltoallv with I_MPI_ADJUST_ALLTOALLV"
uh7=$uh; [ -f $SCR/opt_ucx/hpcx/lib/libucp.so.0 ] || { uh7=true; echo "WARNING: HPC-X UCX not staged: E7 uses the image's UCX"; }
e6d="$IE; $e; export $IFAB; $e5d; $uh7; export I_MPI_ADJUST_ALLTOALL=3"
for spec in "E7_a2a_e6d|a2a|unset I_MPI_ADJUST_ALLTOALLV" "E7_a2av_def|a2av|unset I_MPI_ADJUST_ALLTOALLV" \
            "E7_a2av_1|a2av|export I_MPI_ADJUST_ALLTOALLV=1" "E7_a2av_2|a2av|export I_MPI_ADJUST_ALLTOALLV=2"; do
  name=${spec%%|*}; fn=$(echo "$spec" | cut -d'|' -f2); pe=${spec##*|}
  HFUNC=$fn UCXLOG=1 EXPECT_API=pmi2 T $name "--mpi=pmi2 $SR2 $MI" "$e6d; $pe; $uinfo; $pinfo; $HELLO"
done
fi   # group E7

if want C; then
# ---------------------------------------------------------------- C) mpirun -bootstrap slurm inside the container
echo "=============== C) mpirun -bootstrap slurm (one launcher task, proxies via srun inside the container)"
EXPECT_API=""
EH="export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS='--overlap $MI'"
HYD="mpirun -np 256 -ppn 128 $HB @A@"
T C_hydra_slurm "-N 1 -n 1 -c 1 --overlap $MI" "$IE; command -v srun >/dev/null || { echo 'no srun inside the container'; exit 3; }; \
    $EH; export $IFAB; $HYD" && FIRST_OK=${FIRST_OK:-"hydra"}
fi   # group C

if want F; then
# ---------------------------------------------------------------- F) fabrics, first working launcher
echo "=============== F) fabrics (launcher: ${FIRST_OK%%|*})"
case "$FIRST_OK" in *I_MPI_PMI=pmix*) EXPECT_API=pmix;; *I_MPI_PMI=pmi2*) EXPECT_API=pmi2;; *) EXPECT_API="";; esac
for prov in auto mlx verbs tcp; do
  fp="export FI_PROVIDER=$prov"; [ $prov = auto ] && fp="unset FI_PROVIDER"
  if [ -z "$FIRST_OK" ]; then
    break
  elif [ "$FIRST_OK" = hydra ]; then
    T F_$prov "-N 1 -n 1 -c 1 --overlap $MI" "$IE; $EH; export $IFAB; $fp; $HYD"
  else
    T F_$prov "${FIRST_OK%%|*}" "$IE; export $IFAB; ${FIRST_OK#*|}; $fp; $HELLO"
  fi
done
[ -z "$FIRST_OK" ] && echo "no Intel MPI launcher worked; fallback: multi-node CPU runs with Open MPI (Intel MPI only helps FFTW)"
fi   # group F

echo "=============== summary"
cat $RES
echo "--- per (variant, function, size); side: ompi = R*/RT* (GCC container, Open MPI), intel = the other groups"
awk '
  function kv(name,   i, a) { for (i = 2; i <= NF; i++) { split($i, a, "="); if (a[1] == name) return a[2] } return "" }
  { v = kv("variant"); f = kv("func"); b = kv("bytes"); s = kv("status"); if (f == "") f = "alltoall"
    if (b == "-" || b == "") next
    side = (v ~ /^R/) ? "ompi" : "intel"
    printf "%-24s %-6s %-10s %8s %-5s %10s %8s %s\n", v, side, f, b, s, kv("a2a_us"), kv("GBs_node_offnode"), kv("ucx_version") }
  BEGIN { printf "%-24s %-6s %-10s %8s %-5s %10s %8s %s\n", "variant", "side", "func", "bytes", "stat", "a2a_us", "GB/s/node", "ucx" }' $RES
echo
echo "--- best setting per side for every (function, size) (lowest alltoall time = highest off-node GB/s/node); intel/ompi >= 0.8 is the target"
awk '
  function kv(name,   i, a) { for (i = 2; i <= NF; i++) { split($i, a, "="); if (a[1] == name) return a[2] } return "" }
  { v = kv("variant"); f = kv("func"); b = kv("bytes"); s = kv("status"); g = kv("GBs_node_offnode"); u = kv("a2a_us"); if (f == "") f = "alltoall"
    if (s != "OK" || u == "") next
    side = (v ~ /^R/) ? "ompi" : "intel"; k = f SUBSEP b
    if (!(k in seen)) { seen[k] = 1; keys[++n] = k }
    if (!((k, side) in tus) || u + 0 < tus[k, side]) { tus[k, side] = u + 0; best[k, side] = g + 0; bv[k, side] = v } }
  END {
    printf "%-10s %8s   %-24s %9s   %-24s %9s  %s\n", "func", "bytes", "best Open MPI", "GB/s", "best Intel", "GB/s", "intel/ompi"
    # sort keys by function, then size
    for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) {
      split(keys[i], x, SUBSEP); split(keys[j], y, SUBSEP)
      if (x[1] > y[1] || (x[1] == y[1] && x[2] + 0 > y[2] + 0)) { t = keys[i]; keys[i] = keys[j]; keys[j] = t } }
    for (i = 1; i <= n; i++) { k = keys[i]; split(k, x, SUBSEP)
      ho = ((k, "ompi") in tus); ht = ((k, "intel") in tus)
      r = (ho && ht) ? tus[k, "ompi"] / tus[k, "intel"] : 0      # = GB/s intel / GB/s ompi
      printf "%-10s %8s   %-24s %9s   %-24s %9s  %s\n", x[1], x[2],
        (ho ? bv[k, "ompi"] : "-"), (ho ? sprintf("%.2f", best[k, "ompi"]) : "-"),
        (ht ? bv[k, "intel"] : "-"), (ht ? sprintf("%.2f", best[k, "intel"]) : "-"),
        (r ? sprintf("%.2f%s", r, (r < 0.8 ? "  below 0.8" : "")) : "-") } }' $RES
cleanup_all
