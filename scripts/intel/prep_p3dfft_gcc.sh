#!/bin/bash
# prep_p3dfft_gcc.sh — run in the GCC container (it has autoreconf/libtoolize; the Intel container does not).
# Clones P3DFFT into the work directory of build_intel_libs.sh and generates configure there.
#   source scripts/intel/intel_common.sh
#   srun -p batch --exclude=$EXCL -N1 -n1 -c4 --gres=gpu:1 --time=00:10:00 $M bash /work/src/NavierStockes-3D/scripts/intel/prep_p3dfft_gcc.sh
set -e
WRK=${INTEL_WRK:-/work/src/intel_libs}
mkdir -p $WRK && cd $WRK
[ -d p3dfft ] || git clone -q --depth 1 https://github.com/sdsc/p3dfft.git p3dfft
cd p3dfft && autoreconf -fi
ls -l configure && echo "OK   p3dfft configure generated in $WRK/p3dfft"
