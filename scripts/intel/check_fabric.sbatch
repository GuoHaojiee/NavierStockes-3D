#!/bin/bash
#SBATCH -J cfab
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:15:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/cfab_%j.out
# check_fabric — is the InfiniBand stack usable INSIDE the Intel image? One node, one task.
#   ibv_devinfo          lists mlx5_* devices                (rdma-core, libibverbs, ibverbs-providers)
#   ucx_info -d          transports (rc/dc/ud ... on mlx5)   (libucx0, ucx-utils)
#   fi_info -p mlx       Intel MPI's libfabric, mlx provider (needs UCX)
#   fi_info -p verbs     Intel MPI's libfabric, verbs provider (needs libibverbs)
# Each check prints   CHECK <name> OK|FAIL   and the first lines of its output.
# Argument: the image (default v2 = containers/ns3d_intel_v2.sqsh from rebuild_container.sh; "old" = the image in $MI).
#   sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/check_fabric.sbatch
#   grep -E "^CHECK|^image" $(ls -t ~/guohaojie/logs/cfab_*.out | head -1)
IMG=${1:-v2}
[ "$IMG" != old ] && export INTEL_IMAGE=$IMG
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
CHK=$SCR/logs/cfab_$SLURM_JOB_ID.sh
cat > $CHK <<'EOF'
source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1
. /etc/os-release
echo "image: $PRETTY_NAME   I_MPI_ROOT=$I_MPI_ROOT   FI_PROVIDER_PATH=${FI_PROVIDER_PATH:-<unset>}"
echo "manifest: $(tr '\n' ' ' < /opt/ns3d_pmi_libs.txt 2>/dev/null || echo 'none (old image?)')"
echo "/dev/infiniband: $(ls /dev/infiniband 2>&1 | tr '\n' ' ')"
echo "fi_info: $(command -v fi_info)   $(fi_info --version 2>&1 | tr '\n' ' ')"
# C <name> <ok-pattern> <command...>   OK = command exits 0 and its output matches the pattern
C() {
    local n=$1 pat=$2 out rc; shift 2
    out=$("$@" 2>&1); rc=$?
    if [ $rc = 0 ] && echo "$out" | grep -qiE "$pat"; then echo "CHECK $n OK"; else echo "CHECK $n FAIL (rc=$rc)"; fi
    echo "$out" | head -${HEADN:-15} | sed 's/^/    /'; echo
}
C ibv_devinfo   'hca_id: *mlx5'       ibv_devinfo
HEADN=25 C ucx_transports 'transport'  bash -c "ucx_info -d | grep -i transport | sort | uniq -c"
C fi_info_mlx   'provider: *mlx'      fi_info -p mlx
C fi_info_verbs 'provider: *verbs'    fi_info -p verbs
echo "--- libraries the providers need:"
for l in libibverbs.so.1 libmlx5.so.1 librdmacm.so.1 libucp.so.0 libuct.so.0; do
    printf "    %-16s %s\n" $l "$(ldconfig -p | awk -v l=$l '$1 == l {print $NF; exit}')"
done
EOF
echo "=============== check_fabric  image: ${INTEL_IMAGE:-image of \$MI}  node: $(hostname)"
srun -N1 -n1 --time=10 $MI bash /work/logs/cfab_$SLURM_JOB_ID.sh
rm -f $CHK
cleanup_all
