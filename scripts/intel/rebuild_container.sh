#!/bin/bash
# rebuild_container.sh — new Intel image = the existing ns3d_intel.sqsh + rdma-core/libibverbs, UCX and the
# PMI-2 / PMIx client libraries (apt, Ubuntu 22.04). oneAPI is NOT reinstalled. The old image is not touched.
#   in : /scratch/pr0208/guohaojie/containers/ns3d_intel.sqsh      (RC_OLD)
#   out: /scratch/pr0208/guohaojie/containers/ns3d_intel_v2.sqsh   (RC_NEW; refuses to overwrite unless RC_FORCE=1,
#        and never writes RC_OLD)
#   The packages are installed by rebuild_inside.sh (same directory), run as root in a writable copy.
# Two methods:
#   RC_METHOD=enroot (default) — on a COMPUTE node, outside any container:
#       enroot create + enroot start --root --rw + enroot export
#       srun -p batch --exclude=cn34,cn18,cn25 -N1 -n1 -c16 --time=01:00:00 bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/rebuild_container.sh
#   RC_METHOD=pyxis — from the LOGIN node; this script calls srun itself:
#       srun --container-image=<old> --container-remap-root --container-writable --container-save=<new> ...
#       RC_METHOD=pyxis bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/rebuild_container.sh
# Log: ~/guohaojie/logs/rebuild_container_<date>.log
SCR=/scratch/pr0208/guohaojie
HERE=$(cd "$(dirname "$0")" && pwd)
OLD=${RC_OLD:-$SCR/containers/ns3d_intel.sqsh}
NEW=${RC_NEW:-$SCR/containers/ns3d_intel_v2.sqsh}
METHOD=${RC_METHOD:-enroot}
LOG=${RC_LOG:-$SCR/logs/rebuild_container_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "rebuild_container: $OLD -> $NEW   method=$METHOD   host=$(hostname)   log=$LOG"

[ -f "$OLD" ] || { echo "FAIL: $OLD not found"; exit 1; }
[ "$(readlink -f "$OLD")" = "$(readlink -f "$NEW")" ] && { echo "FAIL: output = input ($NEW)"; exit 1; }
if [ -e "$NEW" ]; then
    if [ "$RC_FORCE" = 1 ]; then echo "RC_FORCE=1: replacing $NEW"; rm -f "$NEW"
    else echo "FAIL: $NEW exists (move it away or RC_FORCE=1)"; exit 1; fi
fi
[ -f $HERE/rebuild_inside.sh ] || { echo "FAIL: $HERE/rebuild_inside.sh missing"; exit 1; }
df -h "$(dirname "$NEW")" | tail -1 | awk '{print "free space for the new image: " $4}'
t0=$(date +%s)

case $METHOD in
enroot)
    command -v enroot >/dev/null || { echo "FAIL: no enroot here; run this on a compute node (srun ... bash $0)"; exit 1; }
    NAME=ns3d_intel_v2_build_$$
    echo "enroot data path: $(enroot -v 2>/dev/null | head -1); ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-<default>}"
    trap 'enroot remove -f $NAME >/dev/null 2>&1' EXIT
    enroot create -n $NAME "$OLD" || { echo "FAIL enroot create"; exit 1; }
    enroot start --root --rw --mount $HERE:/rc_scripts $NAME bash /rc_scripts/rebuild_inside.sh
    rc=$?
    [ $rc = 0 ] || { echo "FAIL rebuild_inside.sh (rc=$rc) — image NOT exported"; exit 1; }
    enroot export -o "$NEW" $NAME || { echo "FAIL enroot export"; rm -f "$NEW"; exit 1; }
    ;;
pyxis)
    command -v srun >/dev/null || { echo "FAIL: no srun"; exit 1; }
    # no --container-name: a named container of the old image must not be reused
    srun -p batch --exclude=cn34,cn18,cn25 -N1 -n1 -c16 --time=01:00:00 \
         --container-image="$OLD" --container-remap-root --container-writable --container-save="$NEW" \
         --container-mounts=$HERE:/rc_scripts bash /rc_scripts/rebuild_inside.sh
    rc=$?
    # pyxis saves the image even if the command failed: do not keep a half-done image
    [ $rc = 0 ] || { echo "FAIL rebuild_inside.sh (rc=$rc) — removing $NEW"; rm -f "$NEW"; exit 1; }
    ;;
*) echo "FAIL: RC_METHOD=enroot|pyxis"; exit 1 ;;
esac

[ -s "$NEW" ] || { echo "FAIL: $NEW not written"; exit 1; }
echo "--- $(ls -l "$OLD")"
echo "--- $(ls -l "$NEW")"
echo "OK   $NEW  ($(( ($(date +%s) - t0) / 60 )) min)"
