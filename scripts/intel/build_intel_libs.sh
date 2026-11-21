#!/bin/bash
# build_intel_libs.sh — P3DFFT 2.x and heFFTe 2.4.1 with the Intel toolchain, INSIDE the Intel container.
#
#   /work/opt_intel/p3dfft        P3DFFT 2.x, same steps as build/build_p3dfft.sh (git clone sdsc/p3dfft,
#                                 autoreconf -fi, configure --enable-fftw --with-fftw=<FFTW> --enable-openmp),
#                                 plus --enable-intel, FC = mpiifx (or mpif90 with I_MPI_F90=ifx).
#                                 Planner MEASURE (configure default), stride1 off (default). FFTW = icx build /opt/fftw3.
#   /work/opt_intel/heffte_fftw   heFFTe 2.4.1, FFTW backend (/opt/fftw3), -O3 -march=znver2
#   /work/opt_intel/heffte_mkl    heFFTe 2.4.1, MKL backend only (-DHeffte_ENABLE_MKL=ON), -O3 -march=znver2;
#                                 skipped with a message when the container has no MKL
#   logs: /work/opt_intel/logs/<name>.log
#
# Run on a compute node (16 cores are enough):
#   source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
#   srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=01:00:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/build_intel_libs.sh
# Only some libraries:  ... $MI env IL_ONLY="heffte_fftw heffte_mkl" bash .../build_intel_libs.sh   (names: p3dfft heffte_fftw heffte_mkl)
source "$(dirname "$0")/intel_env.sh"
WRK=${INTEL_WRK:-/work/src/intel_libs}; J=${J:-16}
LOGDIR=$IOPT/logs; mkdir -p $IOPT $LOGDIR $WRK
want() { [ -z "$IL_ONLY" ] || [[ " $IL_ONLY " == *" $1 "* ]]; }

# Fortran compiler for P3DFFT: mpiifx if present, otherwise mpif90 driven to ifx by I_MPI_F90
if command -v mpiifx >/dev/null 2>&1; then IFC=mpiifx; else IFC=mpif90; fi
echo "compilers: mpicc -> $(mpicc -show 2>/dev/null | awk '{print $1}'), mpicxx -> $(mpicxx -show 2>/dev/null | awk '{print $1}'), FC=$IFC -> $($IFC -show 2>/dev/null | awk '{print $1}')"
echo "FFTW: $IFFTW   MKLROOT: ${MKLROOT:-<empty>}   output: $IOPT"
case "$(mpicxx -show 2>/dev/null | awk '{print $1}')" in icpx*) ;; *) echo "WARNING: mpicxx does not call icpx; set I_MPI_CXX=icpx";; esac

# ---------------------------------------------------------------- P3DFFT 2.x
build_p3dfft() {
    (
    set -e
    cd $WRK
    [ -d p3dfft ] || git clone -q --depth 1 https://github.com/sdsc/p3dfft.git p3dfft
    rm -rf p3dfft_intel $IOPT/p3dfft && cp -r p3dfft p3dfft_intel && cd p3dfft_intel
    echo "p3dfft commit: $(git -C ../p3dfft log -1 --format='%h %cd' 2>/dev/null)"
    autoreconf -fi
    # same LIBS as build_p3dfft.sh; a library the icx FFTW does not have is left out (printed)
    local libs="-L$IFFTW/lib -lfftw3"
    for l in fftw3f fftw3_mpi; do
        if ls $IFFTW/lib/lib$l.* >/dev/null 2>&1; then libs="$libs -l$l"; else echo "NOTE: $IFFTW/lib has no lib$l, not linked"; fi
    done
    echo "./configure --prefix=$IOPT/p3dfft --enable-intel --enable-fftw --with-fftw=$IFFTW --enable-openmp FC=$IFC F77=$IFC CC=mpicc CXX=mpicxx LIBS=\"$libs\" CPPFLAGS=\"-I$IFFTW/include\""
    ./configure --prefix=$IOPT/p3dfft --enable-intel --enable-fftw --with-fftw=$IFFTW --enable-openmp \
      FC=$IFC F77=$IFC CC=mpicc CXX=mpicxx LIBS="$libs" CPPFLAGS="-I$IFFTW/include"
    grep -E "define (INTEL|OPENMP|FFTW|ESTIMATE|MEASURE|PATIENT|STRIDE1)" config.h || true
    grep -q "define MEASURE 1" config.h || { echo "ERROR: MEASURE not defined"; exit 1; }
    ! grep -q "define STRIDE1 1" config.h || { echo "ERROR: STRIDE1 defined"; exit 1; }
    grep -q "define INTEL 1" config.h || { echo "ERROR: INTEL not defined"; exit 1; }
    make -j1 -C build
    make -C build install
    mkdir -p $IOPT/p3dfft/include; cp include/*.h $IOPT/p3dfft/include/; cp build/*.mod $IOPT/p3dfft/include/ 2>/dev/null || true
    ls -l $IOPT/p3dfft/lib
    )
}
want p3dfft && c lib_p3dfft build_p3dfft

# ---------------------------------------------------------------- heFFTe 2.4.1
build_heffte() {   # $1 = fftw | mkl
    (
    set -e
    local b=$1 opts
    cd $WRK
    [ -d heffte-2.4.1 ] || git clone -q --branch v2.4.1 --depth 1 https://github.com/icl-utk-edu/heffte.git heffte-2.4.1
    if [ $b = fftw ]; then
        # FindHeffteFFTW.cmake requires double AND single precision FFTW
        for l in fftw3 fftw3f; do ls $IFFTW/lib/lib$l.so >/dev/null 2>&1 || { echo "ERROR: heFFTe needs $IFFTW/lib/lib$l.so"; exit 1; }; done
        opts="-DHeffte_ENABLE_FFTW=ON -DFFTW_ROOT=$IFFTW"
    else                   opts="-DHeffte_ENABLE_MKL=ON -DMKL_ROOT=$MKLROOT"; fi
    rm -rf heffte_b_$b $IOPT/heffte_$b
    cmake -S heffte-2.4.1 -B heffte_b_$b -DCMAKE_INSTALL_PREFIX=$IOPT/heffte_$b -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_FLAGS="-O3 -march=znver2" \
          -DBUILD_SHARED_LIBS=ON $opts
    grep -E "Heffte_ENABLE_(FFTW|MKL)|Heffte_MKL_LIBRARIES|FFTW_LIBRARIES|CMAKE_CXX_COMPILER:" heffte_b_$b/CMakeCache.txt || true
    make -C heffte_b_$b -j$J
    make -C heffte_b_$b install
    grep -E "define Heffte_ENABLE_(FFTW|MKL)" $IOPT/heffte_$b/include/heffte_config.h
    ls -l $IOPT/heffte_$b/lib
    )
}
want heffte_fftw && c lib_heffte_fftw build_heffte fftw
if want heffte_mkl; then
    if have_mkl; then c lib_heffte_mkl build_heffte mkl
    else echo "SKIP heffte_mkl: no MKL in this container (MKLROOT='${MKLROOT}', no mkl_dfti.h, /opt/intel/oneapi/mkl missing)"; fi
fi
echo "--- $IOPT"; ls $IOPT
