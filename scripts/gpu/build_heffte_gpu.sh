#!/bin/bash
# build_heffte_gpu.sh — build the three heFFTe GPU solvers (gpu1, mgpu, multigpu) inside the GCC/NVHPC container,
# with the flags of scripts/compile_all.sh, into a SEPARATE directory (default /work/build/bin_gpu_new), so the
# CPU programs in /work/build/bin_gcc that running jobs use are never touched.
# Run on a compute node:
#   source ~/guohaojie/runs/common.sh
#   srun -N1 -n1 -c16 --time=20 --exclude=cn34,cn18,cn25 $M bash /work/src/NavierStockes-3D/scripts/gpu/build_heffte_gpu.sh
# Each binary has its compiler log next to it: <OUT>/ns_heffte_<v>.log (look there when a line says FAIL).
# The new binary is first written as <name>.new and renamed only after a successful link, so a failed build
# leaves an older binary of the same name in place (and a job that is using it is not disturbed).
source /etc/profile.d/ns3d_env.sh
SRC=${HG_SRC:-/work/src/NavierStockes-3D}; OUT=${HG_OUT:-/work/build/bin_gpu_new}
case "$OUT" in */bin_gcc|*/bin_gcc/) echo "refusing to build into $OUT (used by the CPU jobs)"; exit 1 ;; esac
mkdir -p $OUT; cd $SRC || exit 1
MATH=$NVHPC_ROOT/math_libs/12.5/targets/x86_64-linux
export CPATH=/opt/fftw3/include:$MATH/include:$HEFFTE_DIR/include:${CPATH:-}
export LIBRARY_PATH=/opt/fftw3/lib:$MATH/lib:$HEFFTE_DIR/lib:${LIBRARY_PATH:-}
RPN="-Xlinker -rpath,/opt/fftw3/lib -Xlinker -rpath,$HEFFTE_DIR/lib -Xlinker -rpath,$MATH/lib"
echo "source $(git -C $SRC rev-parse --short HEAD 2>/dev/null || echo ?)  nvcc: $(nvcc --version | tail -1)  heFFTe: $HEFFTE_DIR  -> $OUT"
rc=0
for v in gpu1 mgpu multigpu; do
    name=ns_heffte_$v
    if nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx -Xcompiler -fopenmp NavierStokes_periodic_heffte_$v.cu \
            -lheffte -lcufft -lm $RPN -o $OUT/$name.new > $OUT/$name.log 2>&1; then
        mv -f $OUT/$name.new $OUT/$name; echo "OK   $name"
    else
        rm -f $OUT/$name.new; echo "FAIL $name (see $OUT/$name.log)"; rc=1
    fi
done
ls -l $OUT
exit $rc
