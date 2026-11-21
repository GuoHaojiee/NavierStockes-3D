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
#   0  discovery: PMI libraries on the host and inside the container, srun inside the container?
#   R  reference: Open MPI (GCC container, srun --mpi=pmix) — the fallback if all Intel variants fail
#   A  srun --mpi=pmi2 + I_MPI_PMI_LIBRARY=<each libpmi2.so found in the container>, and Intel MPI's built-in client
#   B1 srun --mpi=pmix + I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=<each libpmix.so found in the container>
#   B2 srun --mpi=pmix + the HOST's libpmix.so.2.5.2 (the one Slurm's pmix plugin uses), mounted at /hostpmix
#      together with its dependencies that the container does not have; FI_PROVIDER = mlx, then verbs
#   C  mpirun -bootstrap slurm inside the container (hydra starts its proxies with srun; needs srun in the container)
#   F  fabrics, with the first launcher that worked: I_MPI_FABRICS=shm:ofi with FI_PROVIDER = (auto) mlx verbs tcp
# hello_a2a: MPI_Alltoall, data checked; build it with build_intel_solvers.sh.
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch
#   sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
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
        sz=${si%:*}; it=${si#*:}; c=${cmd//@A@/$sz $it}; f=$TMP/$v.$sz.txt
        echo ">>> [$(date +%H:%M:%S)] $v bytes=$sz | srun $so ... | $c"
        timeout 200 srun $so --time=3 --kill-on-bad-exit=1 bash -c "$c" > $f 2>&1; rc=$?
        if [ $sz = 8192 ]; then   # launcher/fabric details once per variant
            grep -E "HELLO_|node leader|MPI startup\(\): (libfabric|Load|I_MPI_|Rank)|provider|rror|abort|PMI" $f | head -20
        else
            grep -E "HELLO_|rror|abort" $f | head -5
        fi
        if grep -q "^HELLO_OK" $f && [ $rc = 0 ]; then st=OK; else st=FAIL; tail -5 $f; fi
        l=$(grep "^HELLO_OK" $f | tail -1)
        res "variant=$v bytes=$sz status=$st rc=$rc $(echo "$l" | grep -o 'a2a_us=[^ ]*') $(echo "$l" | grep -o 'GBs_node_offnode=[^ ]*')"; echo
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
echo "--- inside the Intel container:"
srun -N1 -n1 --time=3 $MI bash -c "$IE; . /etc/os-release; echo \"\$PRETTY_NAME  I_MPI_ROOT=\$I_MPI_ROOT\"; ls -l $HB; command -v srun && srun --version || echo 'no srun inside the container'; \
    find / \( -path /proc -o -path /sys -o -path /dev -o -path /work -o -path /scratch \) -prune -o \( -name 'libpmi2.so*' -o -name 'libpmix.so*' -o -name 'libpmi.so*' \) -print 2>/dev/null | sort -u > $WTMP/pmilibs.txt; cat $WTMP/pmilibs.txt; \
    fi_info -l 2>/dev/null | grep -E '^[a-z]' | tr '\n' ' '; echo"
[ -x $SCR/build/bin_intel/hello_a2a ] || { echo "ERROR: $SCR/build/bin_intel/hello_a2a missing — run build_intel_solvers.sh first"; }
PMI2=$(grep -E 'libpmi2\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
PMIX=$(grep -E 'libpmix\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
echo "candidates pmi2: ${PMI2:-none}   pmix: ${PMIX:-none}"

# ---------------------------------------------------------------- R) Open MPI reference (GCC container)
echo "=============== R) Open MPI reference, GCC container"
srun -N1 -n1 --time=3 $M bash -c "source /etc/profile.d/ns3d_env.sh; mpicc -O2 /work/src/NavierStockes-3D/scripts/intel/hello_a2a.c -o /work/build/bin_gcc/hello_a2a_ompi" \
  && T R_ompi_pmix "--mpi=pmix $SR2 $M" "export NS_MODE=cpu; source /work/build/run_env.sh; /work/build/bin_gcc/hello_a2a_ompi @A@"

# ---------------------------------------------------------------- A) srun --mpi=pmi2
echo "=============== A) srun --mpi=pmi2"
i=0
for lib in $PMI2; do
  i=$((i+1)); e="export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$lib; unset I_MPI_HYDRA_BOOTSTRAP"
  T A${i}_pmi2 "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}
done
# without I_MPI_PMI_LIBRARY (Intel MPI's built-in PMI-2 client, if it has one)
e="export I_MPI_PMI=pmi2; unset I_MPI_PMI_LIBRARY I_MPI_HYDRA_BOOTSTRAP"
T A0_pmi2_builtin "--mpi=pmi2 $SR2 $MI" "$IE; $e; export $IFAB; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmi2 $SR2 $MI|$e"}

# ---------------------------------------------------------------- B1) srun --mpi=pmix, libpmix of the container
echo "=============== B1) srun --mpi=pmix, libpmix from the container"
i=0
for lib in $PMIX; do
  i=$((i+1)); e="export I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=$lib; unset I_MPI_HYDRA_BOOTSTRAP"
  T B1_pmix_cont$i "--mpi=pmix $SR2 $MI" "$IE; $e; export $IFAB; $HELLO" && FIRST_OK=${FIRST_OK:-"--mpi=pmix $SR2 $MI|$e"}
done
[ -z "$PMIX" ] && res "variant=B1_pmix_cont bytes=- status=SKIP rc=- (no libpmix.so inside the container)"

# ---------------------------------------------------------------- B2) srun --mpi=pmix, libpmix of the HOST
echo "=============== B2) srun --mpi=pmix, host libpmix mounted at /hostpmix"
HPMIX=${HOST_PMIX:-/usr/lib/x86_64-linux-gnu/libpmix.so.2.5.2}
STG=$TMP/hostpmix
if [ -f "$HPMIX" ]; then
  rm -rf $STG; mkdir -p $STG
  cp -L $HPMIX $STG/; hp=$(basename $HPMIX)
  so=$(objdump -p $HPMIX 2>/dev/null | awk '/SONAME/ {print $2}'); so=${so:-libpmix.so.2}
  [ "$so" != "$hp" ] && ln -s $hp $STG/$so
  echo "host $HPMIX -> $(readlink -f $HPMIX), SONAME $so"
  # PMIx MCA components (dlopen'ed from the install prefix compiled into the library, which the container does not have)
  HPENV=""
  for d in $(dirname $(readlink -f $HPMIX))/pmix /usr/lib/x86_64-linux-gnu/pmix2/lib/pmix /usr/lib/x86_64-linux-gnu/pmix/lib/pmix; do
    if ls $d/*.so >/dev/null 2>&1; then
      mkdir -p $STG/mca; cp -L $d/*.so $STG/mca/; HPENV="PMIX_MCA_mca_base_component_path=/hostpmix/mca"
      echo "PMIx components from $d: $(ls $STG/mca | wc -l) files"; break
    fi
  done
  [ -z "$HPENV" ] && echo "no PMIx component directory found on the host (components built into libpmix?)"
  # the existing --container-mounts of $MI plus the staging directory
  if [[ "$MI" == *--container-mounts=* ]]; then MIH=$(echo "$MI" | sed "s|--container-mounts=\([^ ]*\)|--container-mounts=\1,$STG:/hostpmix:ro|")
  else MIH="$MI --container-mounts=$STG:/hostpmix:ro"; fi
  echo "MIH = $MIH"
  # dependencies the container cannot resolve -> copied from the host; repeated for the dependencies of those
  cat > $TMP/chk_pmix.sh <<'EOF'
source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/hostpmix:$LD_LIBRARY_PATH
LDSO=/lib64/ld-linux-x86-64.so.2
for f in /hostpmix/*.so* /hostpmix/mca/*.so; do
  [ -f "$f" ] || continue
  $LDSO --list "$f" 2>&1 | grep "not found" | awk '{print $1}'
done | sort -u
EOF
  for pass in 1 2 3 4; do
    miss=$(srun -N1 -n1 --time=3 $MIH bash $WTMP/chk_pmix.sh 2>/dev/null)
    [ -z "$miss" ] && { echo "pass $pass: all dependencies resolve inside the container"; break; }
    echo "pass $pass: missing in the container: $(echo $miss)"
    for s in $miss; do
      p=$(ldconfig -p | awk -v s=$s '$1 == s && /x86-64/ {print $NF; exit}')
      if [ -n "$p" ]; then cp -L $p $STG/$s && echo "  + $s  <- $p"; else echo "  ! $s not on the host either"; fi
    done
  done
  echo "staged in /hostpmix: $(ls $STG | tr '\n' ' ')"
  e="export I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=/hostpmix/$hp LD_LIBRARY_PATH=/hostpmix:\$LD_LIBRARY_PATH $HPENV; unset I_MPI_HYDRA_BOOTSTRAP"
  for prov in mlx verbs; do
    T B2_pmix_host_$prov "--mpi=pmix $SR2 $MIH" "$IE; $e; export $IFAB FI_PROVIDER=$prov; $HELLO" \
      && FIRST_OK=${FIRST_OK:-"--mpi=pmix $SR2 $MIH|$e"}
  done
else
  res "variant=B2_pmix_host bytes=- status=SKIP rc=- (no $HPMIX on the host; set HOST_PMIX=...)"
fi

# ---------------------------------------------------------------- C) mpirun -bootstrap slurm inside the container
echo "=============== C) mpirun -bootstrap slurm (one launcher task, proxies via srun inside the container)"
EH="export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS='--overlap $MI'"
HYD="mpirun -np 256 -ppn 128 $HB @A@"
T C_hydra_slurm "-N 1 -n 1 -c 1 --overlap $MI" "$IE; command -v srun >/dev/null || { echo 'no srun inside the container'; exit 3; }; \
    $EH; export $IFAB; $HYD" && FIRST_OK=${FIRST_OK:-"hydra"}

# ---------------------------------------------------------------- F) fabrics, first working launcher
echo "=============== F) fabrics (launcher: ${FIRST_OK%%|*})"
for prov in auto mlx verbs tcp; do
  fp="export FI_PROVIDER=$prov"; [ $prov = auto ] && fp="unset FI_PROVIDER"
  if [ -z "$FIRST_OK" ]; then
    break
  elif [ "$FIRST_OK" = hydra ]; then
    T F_$prov "-N 1 -n 1 -c 1 --overlap $MI" "$IE; $EH; export $IFAB; $fp; $HYD"
  else
    T F_$prov "${FIRST_OK%%|*}" "$IE; ${FIRST_OK#*|}; export $IFAB; $fp; $HELLO"
  fi
done
[ -z "$FIRST_OK" ] && echo "no Intel MPI launcher worked; fallback: multi-node CPU runs with Open MPI (Intel MPI only helps FFTW)"

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
