#!/bin/bash
# stage_hpcx_ucx.sh — copy the UCX of HPC-X 2.19 (GCC container, the one Open MPI uses) to /work/opt_ucx/hpcx,
# so that Intel MPI's mlx provider in the Intel v2 image can load it instead of Ubuntu's UCX 1.12.1.
#   /work/opt_ucx/hpcx/lib/       libucp libuct libucs libucm (+ the other libuc*.so* of HPC-X), symlinks kept
#   /work/opt_ucx/hpcx/lib/ucx/   UCX plugins (UCX_MODULE_DIR)
#   /work/opt_ucx/hpcx/bin/       ucx_info
#   /work/opt_ucx/hpcx/VERSION.txt  ucx_info -v of HPC-X
#   /work/opt_ucx/hpcx/CHECK.txt    dependency check in the Intel v2 image
# Plugins whose missing dependencies are only GPU / shared-memory kernel-module libraries (CUDA, ROCm, Level Zero,
# gdrcopy, knem, xpmem) are moved to lib/ucx.unresolved/ (UCX would only warn about them); any other missing
# dependency — of a core library, ucx_info or an IB plugin — is a FAIL.
#
# Usage, from the LOGIN node (two sruns: copy in the GCC container, check in the Intel v2 image):
#   bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/stage_hpcx_ucx.sh
# or each step inside its container:
#   ... $M  bash /work/src/NavierStockes-3D/scripts/intel/stage_hpcx_ucx.sh copy
#   ... $MI bash /work/src/NavierStockes-3D/scripts/intel/stage_hpcx_ucx.sh check     ($MI of INTEL_IMAGE=v2)
# Overrides: HPCX_UCX_SRC (HPC-X ucx directory), STAGE_DST (/work/opt_ucx/hpcx), STAGE_IMAGE (v2).
MODE=${1:-all}
DST=${STAGE_DST:-/work/opt_ucx/hpcx}
HERE=$(cd "$(dirname "$0")" && pwd)
LDSO=/lib64/ld-linux-x86-64.so.2

case $MODE in
# ------------------------------------------------------------------ login node: both steps
all)
    export INTEL_IMAGE=${STAGE_IMAGE:-v2}
    source $HERE/intel_common.sh
    echo "=============== copy (GCC container)"
    srun -p batch --exclude=$EXCL -N1 -n1 -c4 --gres=gpu:1 --time=00:15:00 $M bash /work/src/NavierStockes-3D/scripts/intel/stage_hpcx_ucx.sh copy \
        || { echo "FAIL copy"; exit 1; }
    echo "=============== check ($INTEL_IMAGE)"
    srun -p batch --exclude=$EXCL -N1 -n1 -c4 --time=00:15:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/stage_hpcx_ucx.sh check
    exit $? ;;

# ------------------------------------------------------------------ GCC container: copy
copy)
    [ -f /etc/profile.d/ns3d_env.sh ] && source /etc/profile.d/ns3d_env.sh >/dev/null 2>&1
    SRCU=${HPCX_UCX_SRC:-/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ucx}
    [ -d $SRCU/lib ] || SRCU=${HPCX_UCX_DIR:-$SRCU}
    [ -f $SRCU/lib/libucp.so ] || [ -f $SRCU/lib/libucp.so.0 ] || { echo "FAIL: no HPC-X UCX in $SRCU (set HPCX_UCX_SRC)"; exit 1; }
    echo "HPC-X UCX: $SRCU"
    rm -rf $DST.tmp && mkdir -p $DST.tmp/lib $DST.tmp/bin || exit 1
    cp -a $SRCU/lib/libuc*.so* $DST.tmp/lib/ || exit 1
    [ -d $SRCU/lib/ucx ] && cp -a $SRCU/lib/ucx $DST.tmp/lib/
    cp -a $SRCU/bin/ucx_info $DST.tmp/bin/ 2>/dev/null || echo "NOTE: no $SRCU/bin/ucx_info"
    LD_LIBRARY_PATH=$SRCU/lib:$LD_LIBRARY_PATH $SRCU/bin/ucx_info -v > $DST.tmp/VERSION.txt 2>&1
    echo "source: $SRCU" >> $DST.tmp/VERSION.txt
    rm -rf $DST && mv $DST.tmp $DST || exit 1
    echo "--- ucx_info -v (HPC-X):"; sed 's/^/    /' $DST/VERSION.txt
    echo "--- $DST/lib: $(ls $DST/lib | grep -v '^ucx$' | tr '\n' ' ')"
    echo "--- $DST/lib/ucx: $(ls $DST/lib/ucx 2>/dev/null | tr '\n' ' ')"
    echo "OK   copy  ($(grep -m1 -oE '(Library version: *|UCT version=)[0-9.]+' $DST/VERSION.txt))" ;;

# ------------------------------------------------------------------ Intel v2 image: dependency check
check)
    [ -f $(dirname "$0")/intel_env.sh ] && source $(dirname "$0")/intel_env.sh >/dev/null 2>&1
    [ -d $DST/lib ] || { echo "FAIL: $DST/lib missing (run the copy step first)"; exit 1; }
    . /etc/os-release; echo "image: $PRETTY_NAME   system UCX: $(ucx_info -v 2>/dev/null | grep -m1 -oE '(Library version: *|UCT version=)[0-9.]+')"
    export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
    # move back plugins set aside by an earlier check, so the check is repeatable
    [ -d $DST/lib/ucx.unresolved ] && { mv $DST/lib/ucx.unresolved/* $DST/lib/ucx/ 2>/dev/null; rmdir $DST/lib/ucx.unresolved; }
    OPTIONAL='cuda|cudart|nvidia|nvml|gdrapi|gdrcopy|knem|xpmem|hsa|rocm|amdhip|ze_loader|level_zero'
    st=0; nmove=0
    {
    echo "# dependency check of $DST in $PRETTY_NAME, $(date '+%F %T')"
    for f in $DST/lib/*.so* $DST/bin/ucx_info $DST/lib/ucx/*.so*; do
        [ -f "$f" ] && [ ! -L "$f" ] || continue
        # plugins link each other (rdmacm -> ib): UCX loads them globally, so their own directory counts as found
        miss=$(LD_LIBRARY_PATH=$DST/lib/ucx:$LD_LIBRARY_PATH LD_TRACE_LOADED_OBJECTS=1 $LDSO "$f" 2>&1 | grep -E "not found|cannot open" | sed -E 's/^\s+//' | sed -E 's/ => not found//' | sort -u)
        [ -z "$miss" ] && continue
        case $f in
        */lib/ucx/*)
            if ! echo "$miss" | grep -qvE "$OPTIONAL"; then
                mkdir -p $DST/lib/ucx.unresolved; mv "$f" $DST/lib/ucx.unresolved/; nmove=$((nmove + 1))
                echo "SET ASIDE $(basename $f): needs $(echo $miss | tr '\n' ' ') (GPU / kernel-module plugin, not used here)"
            else
                echo "FAIL plugin $(basename $f): $(echo "$miss" | tr '\n' ';')"; st=1
            fi ;;
        *)  echo "FAIL $(basename $f): $(echo "$miss" | tr '\n' ';')"; st=1 ;;
        esac
    done
    [ $st = 0 ] && echo "OK   all core libraries, ucx_info and the remaining plugins resolve ($nmove plugins set aside)"
    } | tee $DST/CHECK.txt
    grep -q "^FAIL" $DST/CHECK.txt && st=1
    echo "--- HPC-X ucx_info in this image (UCX_MODULE_DIR=$DST/lib/ucx):"
    export UCX_MODULE_DIR=$DST/lib/ucx
    $DST/bin/ucx_info -v 2>&1 | sed 's/^/    /'
    tr=$($DST/bin/ucx_info -d 2>/dev/null | grep -E "Transport:" | awk '{print $NF}' | sort | uniq -c | tr '\n' ' ')
    echo "    transports: $tr"
    if echo "$tr" | grep -q "rc_mlx5"; then echo "OK   rc_mlx5 visible to the HPC-X UCX in this image"
    else echo "FAIL rc_mlx5 not visible (IB plugin missing or no device)"; st=1; fi
    [ $st = 0 ] && echo "OK   check" || echo "FAIL check (see $DST/CHECK.txt)"
    exit $st ;;
*) echo "usage: stage_hpcx_ucx.sh [all|copy|check]"; exit 2 ;;
esac
