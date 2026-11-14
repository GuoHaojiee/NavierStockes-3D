#!/bin/bash
# build_fb.sh — build the pure-FFT benchmarks inside the GCC/NVHPC container (same flags as scripts/compile_all.sh)
# Run on a compute node, e.g.:
#   source ~/guohaojie/runs/common.sh
#   srun -N1 -n1 -c16 --exclude=cn34,cn18 $M bash /work/src/NavierStockes-3D/bench_fft/build_fb.sh
source /etc/profile.d/ns3d_env.sh
SRC=${FB_SRC:-/work/src/NavierStockes-3D/bench_fft}; OUT=${FB_OUT:-/work/build/bin_fb}; mkdir -p $OUT; cd $SRC || exit 1
NVLIB=$NVHPC_ROOT/compilers/lib
MATH=$NVHPC_ROOT/math_libs/12.5/targets/x86_64-linux
export CPATH=/opt/fftw3/include:$MATH/include:$HEFFTE_DIR/include:${CPATH:-}
export LIBRARY_PATH=/opt/fftw3/lib:$MATH/lib:$HEFFTE_DIR/lib:${LIBRARY_PATH:-}
RPC="-Wl,-rpath,/opt/fftw3/lib -Wl,-rpath,$HEFFTE_DIR/lib -Wl,-rpath,$NVLIB -Wl,-rpath,$MATH/lib"
RPN="-Xlinker -rpath,/opt/fftw3/lib -Xlinker -rpath,$HEFFTE_DIR/lib -Xlinker -rpath,$MATH/lib"
c(){ name=$1; shift; if "$@" -o $OUT/$name > $OUT/$name.log 2>&1; then echo "OK   $name"; else echo "FAIL $name (see $OUT/$name.log)"; return 1; fi; }

MP="-I$CUFFTMP_HOME/include/cufftmp -I$NVSHMEM_HOME/include -I$MPI_HOME/include \
    -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib -L$MPI_HOME/lib -lcufftMp -lnvshmem_host -lcufft -lmpi -lm \
    -Xlinker -rpath,$CUFFTMP_HOME/lib -Xlinker -rpath,$NVSHMEM_HOME/lib"
# cuFFTMp: full build first. If it fails, retry without the two optional tests, one at a time
# (the log of every failed attempt is kept as fb_cufftmp.tryN.log — send it if all three fail).
NV="nvcc -O3 -std=c++17 -arch=sm_80"
c fb_cufftmp $NV fftbench_cufftmp.cu $MP \
 || { cp $OUT/fb_cufftmp.log $OUT/fb_cufftmp.try1.log; echo "     retry without the optional NVSHMEM alltoall test [D]";
      c fb_cufftmp $NV -DFB_NO_NVSHMEM_A2A fftbench_cufftmp.cu $MP; } \
 || { cp $OUT/fb_cufftmp.log $OUT/fb_cufftmp.try2.log; echo "     retry also without the standalone reshape test [R]";
      c fb_cufftmp $NV -DFB_NO_NVSHMEM_A2A -DFB_NO_RESHAPE fftbench_cufftmp.cu $MP; }

# heFFTe GPU (cufft backend) and CPU (fftw backend); host-only code, built like ns_heffte_*
c fb_heffte_gpu nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx -Xcompiler -fopenmp fftbench_heffte.cpp \
   -lheffte -lcufft -lm $RPN
c fb_heffte_cpu nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx -Xcompiler -fopenmp -DFB_CPU fftbench_heffte.cpp \
   -lheffte -lfftw3_omp -lfftw3 -lfftw3f -lm $RPN

# FFTW-MPI slab (same transform as ns_fftw), with OpenMP threads
c fb_fftw mpicxx -O3 -std=c++17 -fopenmp -DFB_THREADS fftbench_fftw.cpp -lfftw3_mpi -lfftw3_omp -lfftw3 -lm $RPC
# serial FFTW planning test (how heFFTe's FFTW backend plans vs. normal planning)
c est_vs_meas g++ -O3 -std=c++17 est_vs_meas.cpp -lfftw3 -lm $RPC
ls -l $OUT
