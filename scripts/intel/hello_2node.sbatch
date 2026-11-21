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
#     RESULT variant=<name> bytes=<size> status=OK|FAIL|SKIP rc=<exit code> a2a_us=... GBs_node_offnode=...
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
#   C  mpirun -bootstrap slurm inside the container (hydra starts its proxies with srun; needs srun in the container)
#   F  fabrics, with the first launcher that worked: I_MPI_FABRICS=shm:ofi with FI_PROVIDER = (auto) mlx verbs tcp
# Every Intel variant also reports the PMI API Intel MPI really used ("MPI startup(): PMI API: ..." with I_MPI_DEBUG=5):
#   RESULT ... pmi_api=<pmi|pmi2|pmix>; PMI_API_MISMATCH(expected=...) when it is not the one the variant asked for.
# hello_a2a: MPI_Alltoall, data checked; build it with build_intel_solvers.sh.
# Argument: the Intel image (default: the one in $MI, i.e. ns3d_intel.sqsh); "v2" = containers/ns3d_intel_v2.sqsh.
# HELLO_GROUPS selects the groups (default "R A B D E C F"), e.g. only the reference and the E group on the new image:
#   sbatch --export=ALL,HELLO_GROUPS="R E" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
#   sbatch --export=ALL,HELLO_GROUPS="R D F" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch
#   sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
[ -n "$1" ] && export INTEL_IMAGE=$1                   # image choice, see intel_common.sh
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
GROUPS_=${HELLO_GROUPS:-"R A B D E C F"}
want() { [[ " $GROUPS_ " == *" $1 "* ]]; }
IE="source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1"
HB=/work/build/bin_intel/hello_a2a
TMP=$SCR/logs/hello2_$SLURM_JOB_ID.d; mkdir -p $TMP     # = /work/logs/hello2_$SLURM_JOB_ID.d in the containers
WTMP=/work/logs/hello2_$SLURM_JOB_ID.d
SIZES="8192:20 65536:10 262144:5"                      # bytes per pair : iterations
RES=$TMP/results.txt; : > $RES                         # RESULT lines (the job's .out file is stdout: grep cannot read it)
res() { echo "RESULT $*" | tee -a $RES; }
FIRST_OK=""                                            # "<srun options>|<inner environment>" of the first launcher that worked

# T <variant> <srun options> <inner command>
#   The inner command runs inside the container on every rank; @A@ is replaced by "<bytes> <iterations>".
#   One srun per message size. Returns 0 if all sizes are OK.
T() {
    local v=$1 so=$2 cmd=$3 si sz it c rc st l f
    for si in $SIZES; do
        sz=${si%:*}; it=${si#*:}; c=${cmd//@A@/$sz $it}; f=$TMP/$v.$sz.raw
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
        # PMI API actually used by Intel MPI (I_MPI_DEBUG >= 2 prints it); EXPECT_API = what the variant asked for
        local api pa=""
        api=$(grep -m1 -o "PMI API: *[A-Za-z0-9_]*" $f | awk '{print $NF}')
        if [ -n "$EXPECT_API" ] || [ -n "$api" ]; then
            pa="pmi_api=${api:-none}"
            if [ -n "$EXPECT_API" ] && [ "$api" != "$EXPECT_API" ]; then
                pa="$pa PMI_API_MISMATCH(expected=$EXPECT_API)"; echo "    WARNING: Intel MPI used PMI API '${api:-none}', not $EXPECT_API"
            fi
        fi
        res "variant=$v bytes=$sz status=$st rc=$rc $(echo "$l" | grep -o 'a2a_us=[^ ]*') $(echo "$l" | grep -o 'GBs_node_offnode=[^ ]*') $pa"; echo
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
[ -x $SCR/build/bin_intel/hello_a2a ] || { echo "ERROR: $SCR/build/bin_intel/hello_a2a missing — run build_intel_solvers.sh first"; }
PMI2=$(grep -E 'libpmi2\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
PMIX=$(grep -E 'libpmix\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
echo "candidates pmi2: ${PMI2:-none}   pmix: ${PMIX:-none}"
# libraries installed by rebuild_container.sh (manifest), else the usual Ubuntu paths if the search found them
D_PMI2=$(sed -n 's/^PMI2_LIB=//p' $TMP/manifest.txt 2>/dev/null); D_PMIX=$(sed -n 's/^PMIX_LIB=//p' $TMP/manifest.txt 2>/dev/null)
[ -z "$D_PMI2" ] && D_PMI2=$(grep -E '/libpmi2\.so\.0$' $TMP/pmilibs.txt 2>/dev/null | head -1)
[ -z "$D_PMIX" ] && D_PMIX=$(grep -E '/libpmix\.so\.2$' $TMP/pmilibs.txt 2>/dev/null | head -1)
echo "image manifest /opt/ns3d_pmi_libs.txt: $(tr '\n' ' ' < $TMP/manifest.txt 2>/dev/null || echo none)"
echo "D variants: pmi2 lib ${D_PMI2:-none}   pmix lib ${D_PMIX:-none}"

if want R; then
# ---------------------------------------------------------------- R) Open MPI reference (GCC container)
echo "=============== R) Open MPI reference, GCC container"
srun -N1 -n1 --time=3 $M bash -c "source /etc/profile.d/ns3d_env.sh; mpicc -O2 /work/src/NavierStockes-3D/scripts/intel/hello_a2a.c -o /work/build/bin_gcc/hello_a2a_ompi" \
  && T R_ompi_pmix "--mpi=pmix $SR2 $M" "export NS_MODE=cpu; source /work/build/run_env.sh; /work/build/bin_gcc/hello_a2a_ompi @A@"
fi   # group R

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
  /lib64/ld-linux-x86-64.so.2 --list "\$f" 2>&1 | grep "=> not found" | awk '{print \$1}'
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

if want E; then
# ---------------------------------------------------------------- E) D1 launcher, Intel's own libfabric providers only
echo "=============== E) srun --mpi=pmi2 + libpmi2.so.0, FI_PROVIDER_PATH = Intel libfabric providers only"
E_PMI2=${D_PMI2:-/usr/lib/x86_64-linux-gnu/libpmi2.so.0}
E3NET=${E3_NET_DEVICES:-mlx5_2:1}
# \$I_MPI_ROOT is expanded inside the container (after intel_env.sh)
e="export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$E_PMI2 FI_PROVIDER_PATH=\$I_MPI_ROOT/opt/mpi/libfabric/lib/prov FI_LOG_LEVEL=warn; unset I_MPI_HYDRA_BOOTSTRAP"
# printed once, by rank 0: which provider files exist and that the system directory is gone from the path
pinfo="[ \${SLURM_PROCID:-0} = 0 ] && { echo \"FI_PROVIDER_PATH=\$FI_PROVIDER_PATH: \$(ls \$FI_PROVIDER_PATH 2>&1 | tr '\\n' ' ')\"; }"
for v in "E1_verbs_rxm|export FI_PROVIDER='verbs;ofi_rxm'" \
         "E2_psm3|export FI_PROVIDER=psm3" \
         "E3_mlx_ucxtls|export FI_PROVIDER=mlx UCX_TLS=rc,ud,sm,self UCX_NET_DEVICES=$E3NET; [ \${SLURM_PROCID:-0} = 0 ] && ucx_info -v" \
         "E4_tcp|export FI_PROVIDER=tcp"; do
  name=${v%%|*}; pe=${v#*|}
  EXPECT_API=pmi2 T $name "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $pe; $pinfo; $HELLO" \
    && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
done
fi   # group E

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
echo "--- working variants: alltoall time [us] / off-node bandwidth per node [GB/s] for 8192 65536 262144 bytes per pair"
awk '
  { v=""; b=""; s=""; t="-"; g="-"
    for (i = 2; i <= NF; i++) { split($i, kv, "="); if (kv[1]=="variant") v=kv[2]; else if (kv[1]=="bytes") b=kv[2];
      else if (kv[1]=="status") s=kv[2]; else if (kv[1]=="a2a_us") t=kv[2]; else if (kv[1]=="GBs_node_offnode") g=kv[2] }
    if (!(v in seen)) { seen[v]=1; order[++n]=v }
    if (s=="OK") { r[v, b]=sprintf("%9s / %-6s", t, g); ok[v]++ } }
  END { printf "%-22s %20s %20s %20s\n", "variant", "8192", "65536", "262144"
        for (k = 1; k <= n; k++) { v=order[k]; if (!ok[v]) continue
          printf "%-22s", v; split("8192 65536 262144", B, " ")
          for (j = 1; j <= 3; j++) printf " %20s", ((v, B[j]) in r) ? r[v, B[j]] : "FAIL"; printf "\n" } }' $RES
cleanup_all
