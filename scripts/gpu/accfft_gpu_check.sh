#!/bin/bash
# accfft_gpu_check.sh — CHECK ONLY: can the GPU part of AccFFT be built with the container's CUDA 12.5?
# Configures a copy of the AccFFT source with BUILD_GPU=ON in a throw-away directory and builds the GPU
# libraries and the step1_gpu example. Installs nothing; /opt/accfft, the solvers and bin_gcc are not touched.
# Run on a compute node (no GPU needed for building):
#   source ~/guohaojie/runs/common.sh
#   srun -N1 -n1 -c16 --time=30 --exclude=cn34,cn18,cn25 $M bash /work/src/NavierStockes-3D/scripts/gpu/accfft_gpu_check.sh
# Source: ACCFFT_SRC (default: the first of /work/src/accfft /opt/src/accfft /opt/accfft-src that has CMakeLists.txt).
# Result lines: "CHECK <step> OK|FAIL (log)". Logs under /work/tmp/accfft_gpu_check/ (cmake_*.log, make_*.log).
# Attempts (each a separate configure, so the report says which of the known problems matter):
#   A  as shipped: only BUILD_GPU=ON          (expected to fail: the default arch list has compute_35,
#                                               which CUDA 12 no longer supports; and no sm_80)
#   B  + CUDA_NVCC_FLAGS for sm_80, CUDA_TOOLKIT_ROOT_DIR, and CMAKE_POLICY_VERSION_MINIMUM=3.5
#      (needed if cmake >= 4.0, harmless before)
#   then with B: make accfft_gpu accfft_utils_gpu (the libraries), then step1_gpu (example; its CMakeLists
#   links only CUDA_LIBRARIES = cudart, so it may miss the cuFFT symbols), then step1_gpu with -lcufft added.
source /etc/profile.d/ns3d_env.sh
W=${ACCFFT_CHECK_DIR:-/work/tmp/accfft_gpu_check}; rm -rf $W; mkdir -p $W
SRC=${ACCFFT_SRC:-}
if [ -z "$SRC" ]; then for d in /work/src/accfft /opt/src/accfft /opt/accfft-src; do [ -f $d/CMakeLists.txt ] && { SRC=$d; break; }; done; fi
[ -f "$SRC/CMakeLists.txt" ] || { echo "CHECK source FAIL: no AccFFT source found (set ACCFFT_SRC=/work/... ; the local copy is ~/src/accfft on the workstation)"; exit 1; }
CUDA_ROOT=${CUDA_ROOT:-$NVHPC_ROOT/cuda/12.5}
MATH=$NVHPC_ROOT/math_libs/12.5/targets/x86_64-linux
export CPATH=/opt/fftw3/include:$MATH/include:${CPATH:-}
export LIBRARY_PATH=/opt/fftw3/lib:$MATH/lib:${LIBRARY_PATH:-}
echo "source: $SRC ($(git -C $SRC log -1 --format='%h %ad' --date=short 2>/dev/null || echo 'no git'))"
echo "cmake: $(cmake --version | head -1)   nvcc: $(nvcc --version | tail -1)   host: $(mpicxx --version | head -1)"
echo "CUDA_ROOT=$CUDA_ROOT  cufft: $(ls $MATH/lib/libcufft.so 2>/dev/null || echo 'not found in math_libs')  FFTW: /opt/fftw3"
cp -r $SRC $W/src; rm -rf $W/src/build*
FFTWOPT="-DFFTW_ROOT=${ACCFFT_FFTW:-/opt/fftw3}"
chk() { local name=$1 log=$2; shift 2; if "$@" > $log 2>&1; then echo "CHECK $name OK"; return 0; else echo "CHECK $name FAIL ($log)"; grep -m5 -iE "error|fatal|undefined" $log | sed 's/^/    /'; return 1; fi; }

# A: as shipped
mkdir -p $W/bA; cd $W/bA
chk "A-configure(as shipped)" $W/cmake_A.log cmake $W/src -DBUILD_GPU=ON $FFTWOPT -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  && chk "A-build-libs(as shipped)" $W/make_A.log make -j16 accfft_gpu
grep -m1 -oE "Unsupported gpu architecture '[^']+'" $W/make_A.log 2>/dev/null | sed 's/^/    -> /'

# B: sm_80 and the toolkit path
mkdir -p $W/bB; cd $W/bB
chk "B-configure(sm_80)" $W/cmake_B.log cmake $W/src -DBUILD_GPU=ON $FFTWOPT -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT "-DCUDA_NVCC_FLAGS=-gencode;arch=compute_80,code=sm_80" || exit 1
grep -E "CUDA_VERSION|CUDA_TOOLKIT_ROOT_DIR|CUDA_CUFFT_LIBRARIES|CUDA_NVCC_EXECUTABLE" CMakeCache.txt | sed 's/^/    /'
chk "B-build-libs(accfft_gpu accfft_utils_gpu)" $W/make_B_libs.log make -j16 accfft_gpu accfft_utils_gpu
grep -c -i "warning" $W/make_B_libs.log | sed 's/^/    compiler warnings: /'
chk "B-build-step1_gpu(as shipped link line)" $W/make_B_step1.log make -j16 step1_gpu \
  || { cmake . "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed -L$MATH/lib -lcufft" > $W/cmake_B2.log 2>&1
       chk "B-build-step1_gpu(+ -lcufft)" $W/make_B_step1_cufft.log make -j16 step1_gpu; }
ls -l $W/bB/libaccfft_gpu* $W/bB/libaccfft_utils_gpu* $W/bB/steps/step1/step1_gpu 2>/dev/null
echo "done; logs in $W"
