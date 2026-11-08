#!/bin/bash
source /etc/profile.d/ns3d_env.sh
SRC=/work/src/NavierStockes-3D; OUT=/work/build/bin_gcc; cd $SRC
NVLIB=$NVHPC_ROOT/compilers/lib
MATH=$NVHPC_ROOT/math_libs/12.5/targets/x86_64-linux
export CPATH=/opt/fftw3/include:$MATH/include:$HEFFTE_DIR/include:${CPATH:-}
export LIBRARY_PATH=/opt/fftw3/lib:$MATH/lib:$HEFFTE_DIR/lib:${LIBRARY_PATH:-}
RPC="-Wl,-rpath,/opt/fftw3/lib -Wl,-rpath,$HEFFTE_DIR/lib -Wl,-rpath,$NVLIB -Wl,-rpath,$MATH/lib"   # for mpicxx/gcc
RPN="-Xlinker -rpath,/opt/fftw3/lib -Xlinker -rpath,$HEFFTE_DIR/lib -Xlinker -rpath,$MATH/lib"        # for nvcc
c(){ name=$1; shift; if "$@" -o $OUT/$name > $OUT/$name.log 2>&1; then echo "OK   $name"; else echo "FAIL $name (see bin_gcc/$name.log)"; fi; }

c ns_fftw    mpicxx -O3 -std=c++17 -fopenmp -fno-math-errno NavierStokes_periodic_fftw.cpp \
             -lfftw3_mpi -lfftw3_omp -lfftw3 -lpthread -lm $RPC
c ns_accfft  mpicxx -O3 -std=c++17 -fopenmp -fno-math-errno NavierStokes_periodic_accfft.cpp -I$ACCFFT_DIR/include -L$ACCFFT_DIR/lib \
             -laccfft -laccfft_utils -lfftw3_mpi -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lpthread -lm $RPC
c ns_heffte_v2 nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx -Xcompiler -fopenmp -Xcompiler -fno-math-errno NavierStokes_periodic_heffte_v2.cpp \
             -lheffte -lfftw3_omp -lfftw3 -lfftw3f -lm $RPN
c ns_cufft   nvcc -O3 -std=c++17 -arch=sm_80 NavierStokes_periodic_cufft.cu -lcufft -lm $RPN
c ns_cufftxt nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx NavierStokes_periodic_cufftxt.cu -lcufft -lm $RPN
for v in gpu1 mgpu multigpu; do
c ns_heffte_$v nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx -Xcompiler -fopenmp NavierStokes_periodic_heffte_$v.cu \
             -lheffte -lcufft -lm $RPN
done
for v in mgpu multinode; do
c ns_cufftmp_$v nvcc -O3 -std=c++17 -arch=sm_80 NavierStokes_periodic_cufftmp_$v.cu \
             -I$CUFFTMP_HOME/include/cufftmp -I$NVSHMEM_HOME/include -I$MPI_HOME/include \
             -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib -L$MPI_HOME/lib -lcufftMp -lnvshmem_host -lcufft -lmpi -lm \
             -Xlinker -rpath,$CUFFTMP_HOME/lib -Xlinker -rpath,$NVSHMEM_HOME/lib
done
# P3DFFT：与论文一致，用 nvc++（HPC-X 默认），避免 OpenMP 运行库混用
# nvc++ 不接受 -fno-math-errno，因此这一行不加
c ns_p3dfft env OMPI_CXX=nvc++ mpicxx -O3 -std=c++17 -mp NavierStokes_periodic_p3dfft.cpp -I$P3DFFT_HOME/include \
             -L$P3DFFT_HOME/lib -lp3dfft -lfftw3_mpi -lfftw3_threads -lfftw3 -lfftw3f -lnvf -lmpi_mpifh -lm \
             -Wl,-rpath,/opt/fftw3/lib
