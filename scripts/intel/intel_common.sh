# intel_common.sh — sourced on the LOGIN node / in sbatch scripts (outside the container).
# Defines $M (GCC container, from ~/guohaojie/runs/common.sh), $MI (Intel container), precheck, cleanup_all.
# $MI is derived from $M by replacing ns3d_gcc with ns3d_intel (image and pyxis container name).
# If your $M does not contain "ns3d_gcc", set MI yourself before sourcing, e.g.
#   export MI="--container-image=/scratch/pr0208/guohaojie/ns3d_intel.sqsh --container-name=ns3d_intel --container-mounts=/scratch/pr0208/guohaojie:/work"
source ~/guohaojie/runs/common.sh          # $M, precheck, cleanup
if [ -z "$MI" ]; then
    MI=$(echo "$M" | sed 's/ns3d_gcc/ns3d_intel/g')
    if [ "$MI" = "$M" ]; then echo "intel_common.sh: \$M has no 'ns3d_gcc' ($M); set MI by hand (see the header of this file)"; MI=""; fi
fi
export MI
SCR=/scratch/pr0208/guohaojie
EXCL=cn34,cn18,cn25
# remove the pyxis containers of this job on every node (GCC via common.sh's cleanup, Intel here)
cleanup_all() {
    cleanup 2>/dev/null
    srun --ntasks-per-node=1 --time=2 bash -c "enroot remove -f pyxis_ns3d_intel >/dev/null 2>&1; true" 2>/dev/null
}
echo "M  = $M"
echo "MI = $MI"
