#!/bin/bash
# accfft_gpu_check.sh — CHECK ONLY: can the GPU part of AccFFT be built with the container's CUDA 12.5?
# Configures a copy of the AccFFT source with BUILD_GPU=ON in a throw-away directory and builds the GPU
# libraries and the GPU examples. Installs nothing; /opt/accfft, the solvers and bin_gcc are not touched.
# Run on a compute node (no GPU needed for building):
#   source ~/guohaojie/runs/common.sh
#   srun -N1 -n1 -c16 --time=30 --exclude=cn34,cn18,cn25 $M bash /work/src/NavierStockes-3D/scripts/gpu/accfft_gpu_check.sh
# Source: ACCFFT_SRC (default: the first of /work/src/accfft /opt/src/accfft /opt/accfft-src that has CMakeLists.txt).
# cuFFT: NVHPC keeps it in math_libs, not in the CUDA toolkit directory, so FindCUDA leaves CUDA_cufft_LIBRARY
#   NOTFOUND (the examples use it through cuda_add_cufft_to_target). The script looks for libcufft.so under
#   /opt/nvidia/hpc_sdk (CUFFT_SEARCH to change), takes the one under math_libs/12.5, and passes it explicitly.
# Configurations (each one: configure, then make of the libraries, then make of every GPU example):
#   both: BUILD_GPU=ON, CUDA_TOOLKIT_ROOT_DIR=$NVHPC_ROOT/cuda/12.5, CUDA_cufft_LIBRARY=<the libcufft.so found>
#   A  AccFFT's own arch list (compute_60/70/75 and compute_35, no sm_80)
#      expected: the library build fails on compute_35 (CUDA 12 dropped it)
#   B  CUDA_NVCC_FLAGS = sm_80 only
#   (There is no configuration "libraries only": AccFFT's CMakeLists has no option to leave out the examples
#    — steps/step1 is always added, steps 2-5 with EXCLUDE_FROM_ALL — so libraries and examples are built as
#    separate make targets and reported separately instead.)
# Result lines: "CHECK <config> <stage> OK|FAIL|SKIP"; on FAIL the last 30 lines of that log are printed.
# Logs: /work/tmp/accfft_gpu_check/<config>_{configure,libs,<example>}.log
source /etc/profile.d/ns3d_env.sh
W=${ACCFFT_CHECK_DIR:-/work/tmp/accfft_gpu_check}; rm -rf $W; mkdir -p $W
SRC=${ACCFFT_SRC:-}
if [ -z "$SRC" ]; then for d in /work/src/accfft /opt/src/accfft /opt/accfft-src; do [ -f $d/CMakeLists.txt ] && { SRC=$d; break; }; done; fi
[ -f "$SRC/CMakeLists.txt" ] || { echo "CHECK source FAIL: no AccFFT source found (set ACCFFT_SRC=/work/... ; the local copy is ~/src/accfft on the workstation)"; exit 1; }

# ---- cuFFT library of NVHPC
CS=${CUFFT_SEARCH:-/opt/nvidia/hpc_sdk}
echo "libcufft.so found under $CS:"
ALL=$(find -L $CS -name "libcufft.so" 2>/dev/null | sort); echo "$ALL" | sed 's/^/    /'
CUFFT=$(echo "$ALL" | grep "/math_libs/12.5/" | head -1)
if [ -z "$CUFFT" ]; then echo "CHECK cufft FAIL: no libcufft.so under $CS/.../math_libs/12.5"; exit 1; fi
echo "CHECK cufft OK: $CUFFT"
CUFFT_INC=$(cd $(dirname $CUFFT)/../include 2>/dev/null && pwd)
[ -f "$CUFFT_INC/cufft.h" ] || CUFFT_INC=$(dirname $(find -L $(dirname $CUFFT)/../.. -name cufft.h 2>/dev/null | head -1) 2>/dev/null)
echo "cufft.h: ${CUFFT_INC:-not found}/cufft.h"

CUDA_ROOT=${CUDA_ROOT:-$NVHPC_ROOT/cuda/12.5}
export CPATH=/opt/fftw3/include:${CUFFT_INC:+$CUFFT_INC:}${CPATH:-}
export LIBRARY_PATH=/opt/fftw3/lib:$(dirname $CUFFT):${LIBRARY_PATH:-}
echo "source: $SRC ($(git -C $SRC log -1 --format='%h %ad' --date=short 2>/dev/null || echo 'no git'))"
echo "cmake: $(cmake --version | head -1)   nvcc: $(nvcc --version | tail -1)   host: $(mpicxx --version | head -1)"
echo "CUDA_ROOT=$CUDA_ROOT  FFTW: ${ACCFFT_FFTW:-/opt/fftw3}"
cp -r $SRC $W/src; rm -rf $W/src/build*
COMMON="-DBUILD_GPU=ON -DFFTW_ROOT=${ACCFFT_FFTW:-/opt/fftw3} -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT -DCUDA_cufft_LIBRARY=$CUFFT"
LIBS="accfft_gpu accfft_utils_gpu"
EXAMPLES="step1_gpu step1_gpuf step2_gpu step2_gpuf step3_gpu step3_gpuf step4_gpu step5_gpu step5_gpuf"

chk() {   # chk <label> <log> <command...>
    local name=$1 log=$2; shift 2
    if "$@" > $log 2>&1; then echo "CHECK $name OK" | tee -a $W/checks.txt; return 0; fi
    echo "CHECK $name FAIL ($log)" >> $W/checks.txt
    echo "CHECK $name FAIL ($log), last 30 lines:"; tail -30 $log | sed 's/^/    | /'; return 1
}
run_config() {   # run_config <name> <cmake options...>
    local c=$1 e nok=0 nfail=0; shift
    echo; echo "==================== configuration $c: $*"
    mkdir -p $W/b$c; cd $W/b$c
    chk "$c configure" $W/${c}_configure.log cmake $W/src $COMMON "$@"
    grep -E "^(CUDA_VERSION|CUDA_TOOLKIT_ROOT_DIR|CUDA_cufft_LIBRARY|CUDA_NVCC_FLAGS|CUDA_NVCC_EXECUTABLE)[:=]" CMakeCache.txt 2>/dev/null | sed 's/^/    /'
    if [ ! -f Makefile ]; then printf "CHECK %s libs SKIP (no Makefile generated)\nCHECK %s examples SKIP (no Makefile generated)\n" $c $c | tee -a $W/checks.txt; return; fi
    chk "$c libs ($LIBS)" $W/${c}_libs.log make -j16 $LIBS && \
        echo "    compiler warnings: $(grep -ci warning $W/${c}_libs.log)"
    for e in $EXAMPLES; do
        if chk "$c example $e" $W/${c}_$e.log make -j16 $e; then nok=$((nok + 1)); else nfail=$((nfail + 1)); fi
    done
    echo "CHECK $c examples: $nok OK, $nfail FAIL" | tee -a $W/checks.txt
    ls -l $W/b$c/libaccfft_gpu* $W/b$c/libaccfft_utils_gpu* 2>/dev/null | sed 's/^/    /'
}

run_config A
run_config B "-DCUDA_NVCC_FLAGS=-gencode;arch=compute_80,code=sm_80"

echo; echo "==================== summary"
cat $W/checks.txt
echo "done; logs in $W"
