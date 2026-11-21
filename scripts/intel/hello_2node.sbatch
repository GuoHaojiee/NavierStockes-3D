#!/bin/bash
#SBATCH -J hello2
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:45:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/hello2_%j.out
# hello_2node — can Intel MPI in the Intel container run on 2 nodes x 128 ranks? Tries several launchers
# and fabrics, one after another; every attempt ends with
#     RESULT variant=<name> status=OK|FAIL|SKIP rc=<exit code> a2a_us=... GBs_node=...
# A hung attempt is killed after 3 minutes and the job goes on.
#   0  discovery: PMI libraries on the host and inside the container, srun inside the container?
#   R  reference: Open MPI (GCC container, srun --mpi=pmix) — the fallback if all Intel variants fail
#   A  srun --mpi=pmi2 + I_MPI_PMI_LIBRARY=<each libpmi2.so found>
#   B  srun --mpi=pmix + I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=<each libpmix.so found>
#   C  mpirun -bootstrap slurm inside the container (hydra starts its proxies with srun; needs srun in the container)
#   F  fabrics, with the first launcher that worked: I_MPI_FABRICS=shm:ofi with FI_PROVIDER = (auto) mlx verbs tcp
# hello_a2a: MPI_Alltoall of 8 KiB and 64 KiB per pair, data checked; build it with build_intel_solvers.sh.
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch
#   grep -E "^RESULT|^>>>" $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
IE="source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1"
HB=/work/build/bin_intel/hello_a2a
TMP=$SCR/logs/hello2_$SLURM_JOB_ID.d; mkdir -p $TMP
FIRST_OK=""

# T <variant> <srun options> <inner command>     (inner command runs inside the container on every rank)
T() {
    local v=$1 so=$2 cmd=$3 rc st
    echo ">>> [$(date +%H:%M:%S)] $v | srun $so ... | $cmd"
    timeout 200 srun $so --time=3 --kill-on-bad-exit=1 bash -c "$cmd" > $TMP/$v.txt 2>&1; rc=$?
    grep -E "HELLO_|node leader|MPI startup\(\): (libfabric|Load|I_MPI_|Rank)|provider|rror|abort|PMI" $TMP/$v.txt | head -20
    if grep -q "^HELLO_OK" $TMP/$v.txt && [ $rc = 0 ]; then st=OK; else st=FAIL; tail -5 $TMP/$v.txt; fi
    local l; l=$(grep "^HELLO_OK" $TMP/$v.txt | tail -1)
    echo "RESULT variant=$v status=$st rc=$rc $(echo "$l" | grep -o 'a2a_us=[^ ]*') $(echo "$l" | grep -o 'GBs_node_offnode=[^ ]*')"; echo
    [ $st = OK ]
}
SR2="-N 2 --ntasks-per-node=128 -c 1 --cpu-bind=cores"
HELLO="$HB 8192 20 && $HB 65536 10"

# ---------------------------------------------------------------- 0) discovery
echo "=============== 0) discovery"
echo "--- host $(hostname): srun $(srun --version 2>/dev/null); srun --mpi=list:"; srun --mpi=list 2>&1 | tr '\n' ' '; echo
echo "--- PMI client libraries on the host:"
find /usr/lib64 /usr/lib /usr/local/lib /opt/slurm* /usr/local/slurm* -maxdepth 4 \( -name 'libpmi2.so*' -o -name 'libpmix.so*' -o -name 'libpmi.so*' \) 2>/dev/null | sort -u
echo "--- inside the Intel container:"
srun -N1 -n1 --time=3 $MI bash -c "$IE; echo I_MPI_ROOT=\$I_MPI_ROOT; ls -l $HB; command -v srun && srun --version || echo 'no srun inside the container'; \
    find / \( -path /proc -o -path /sys -o -path /dev -o -path /work -o -path /scratch \) -prune -o \( -name 'libpmi2.so*' -o -name 'libpmix.so*' -o -name 'libpmi.so*' \) -print 2>/dev/null | sort -u > /work/logs/hello2_$SLURM_JOB_ID.d/pmilibs.txt; cat /work/logs/hello2_$SLURM_JOB_ID.d/pmilibs.txt; \
    fi_info -l 2>/dev/null | grep -E '^[a-z]' | tr '\n' ' '; echo"
[ -x $SCR/build/bin_intel/hello_a2a ] || { echo "ERROR: $SCR/build/bin_intel/hello_a2a missing — run build_intel_solvers.sh first"; }
PMI2=$(grep -E 'libpmi2\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
PMIX=$(grep -E 'libpmix\.so' $TMP/pmilibs.txt 2>/dev/null | head -3)
echo "candidates pmi2: ${PMI2:-none}   pmix: ${PMIX:-none}"

# ---------------------------------------------------------------- R) Open MPI reference (GCC container)
echo "=============== R) Open MPI reference, GCC container"
srun -N1 -n1 --time=3 $M bash -c "source /etc/profile.d/ns3d_env.sh; mpicc -O2 /work/src/NavierStockes-3D/scripts/intel/hello_a2a.c -o /work/build/bin_gcc/hello_a2a_ompi" \
  && T R_ompi_pmix "--mpi=pmix $SR2 $M" "export NS_MODE=cpu; source /work/build/run_env.sh; /work/build/bin_gcc/hello_a2a_ompi 8192 20 && /work/build/bin_gcc/hello_a2a_ompi 65536 10"

# ---------------------------------------------------------------- A) srun --mpi=pmi2
echo "=============== A) srun --mpi=pmi2"
i=0
for lib in $PMI2; do
  i=$((i+1))
  T A${i}_pmi2 "--mpi=pmi2 $SR2 $MI" "$IE; export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$lib I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5; unset I_MPI_HYDRA_BOOTSTRAP; $HELLO" \
    && FIRST_OK=${FIRST_OK:-"--mpi=pmi2|I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=$lib"}
done
# without I_MPI_PMI_LIBRARY (Intel MPI's built-in PMI-2 client, if it has one)
T A0_pmi2_builtin "--mpi=pmi2 $SR2 $MI" "$IE; export I_MPI_PMI=pmi2 I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5; unset I_MPI_PMI_LIBRARY I_MPI_HYDRA_BOOTSTRAP; $HELLO" \
  && FIRST_OK=${FIRST_OK:-"--mpi=pmi2|I_MPI_PMI=pmi2"}

# ---------------------------------------------------------------- B) srun --mpi=pmix
echo "=============== B) srun --mpi=pmix"
i=0
for lib in $PMIX; do
  i=$((i+1))
  T B${i}_pmix "--mpi=pmix $SR2 $MI" "$IE; export I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=$lib I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5; unset I_MPI_HYDRA_BOOTSTRAP; $HELLO" \
    && FIRST_OK=${FIRST_OK:-"--mpi=pmix|I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=$lib"}
done
[ -z "$PMIX" ] && echo "RESULT variant=B_pmix status=SKIP rc=- (no libpmix.so inside the container)"

# ---------------------------------------------------------------- C) mpirun -bootstrap slurm inside the container
echo "=============== C) mpirun -bootstrap slurm (one launcher task, proxies via srun inside the container)"
T C_hydra_slurm "-N 1 -n 1 -c 1 --overlap $MI" "$IE; command -v srun >/dev/null || { echo 'no srun inside the container'; exit 3; }; \
    export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS='--overlap $MI' I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5; \
    mpirun -np 256 -ppn 128 $HB 8192 20 && mpirun -np 256 -ppn 128 $HB 65536 10" \
  && FIRST_OK=${FIRST_OK:-"hydra"}

# ---------------------------------------------------------------- F) fabrics, first working launcher
echo "=============== F) fabrics (launcher: ${FIRST_OK:-none worked})"
if [ -n "$FIRST_OK" ] && [ "$FIRST_OK" != hydra ]; then
  mpiopt=${FIRST_OK%%|*}; pmienv=${FIRST_OK#*|}
  for prov in auto mlx verbs tcp; do
    fp="FI_PROVIDER=$prov"; [ $prov = auto ] && fp=""
    T F_$prov "$mpiopt $SR2 $MI" "$IE; export $pmienv I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5 $fp; [ $prov = auto ] && unset FI_PROVIDER; unset I_MPI_HYDRA_BOOTSTRAP; $HELLO"
  done
elif [ "$FIRST_OK" = hydra ]; then
  for prov in auto mlx verbs tcp; do
    fp="FI_PROVIDER=$prov"; [ $prov = auto ] && fp=""
    T F_$prov "-N 1 -n 1 -c 1 --overlap $MI" "$IE; export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS='--overlap $MI' I_MPI_FABRICS=shm:ofi I_MPI_DEBUG=5 $fp; \
        mpirun -np 256 -ppn 128 $HB 8192 20 && mpirun -np 256 -ppn 128 $HB 65536 10"
  done
else
  echo "no Intel MPI launcher worked; fallback: multi-node CPU runs with Open MPI (Intel MPI only helps FFTW)"
fi
echo "=============== summary"
grep "^RESULT" $SCR/logs/hello2_$SLURM_JOB_ID.out
cleanup_all
