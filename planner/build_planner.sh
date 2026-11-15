#!/bin/bash
# build_planner.sh — FFTW planner study (ESTIMATE / MEASURE / PATIENT), CPU solvers, GCC/NVHPC container.
#
# What is built (everything outside the container image, nothing in bin_gcc is touched):
#   /work/opt_pl/accfft_pt            AccFFT with a one-line fix: stock AccFFT maps ACCFFT_PATIENT to
#                                     FFTW_MEASURE (src/accfft.cpp: "if ESTIMATE ... else MEASURE").
#   /work/opt_pl/p3dfft_<flag>        P3DFFT 2.x three times; its planner flag is a compile-time
#                                     configure option (--enable-estimate / default measure / --enable-patient).
#   /work/build/bin_pl/ns_fftw        planner flag from NS_FFTW_PLAN at run time
#   /work/build/bin_pl/ns_accfft      container AccFFT (to show that its PATIENT == MEASURE)
#   /work/build/bin_pl/ns_accfft_pt   patched AccFFT, NS_FFTW_PLAN at run time
#   /work/build/bin_pl/ns_p3dfft_orig container P3DFFT (reference: must equal p3dfft_measure)
#   /work/build/bin_pl/ns_p3dfft_{estimate,measure,patient}
#
# P3DFFT layout: the solver must be linked against a library configured like the container one.
# Pass the configure arguments of the container build in P3DFFT_CONF (see logs/build_p3dfft2.log).
# P3DFFT only:  ... $M env PL_ONLY_P3DFFT=1 P3DFFT_CONF="..." bash .../build_planner.sh
#
# Run on a compute node:
#   source ~/guohaojie/runs/common.sh
#   srun -p batch --exclude=cn34,cn18 -N1 -n1 -c16 --time=01:00:00 $M bash /work/src/NavierStockes-3D/planner/build_planner.sh
# Only the solvers again (libraries already built):  ... $M env PL_SOLVERS_ONLY=1 bash .../build_planner.sh
source /etc/profile.d/ns3d_env.sh
SRC=/work/src/NavierStockes-3D; OPT=/work/opt_pl; OUT=/work/build/bin_pl; WRK=/work/src/planner_libs
FFTWD=/opt/fftw3; J=${J:-16}
mkdir -p $OPT $OUT $WRK
c(){ name=$1; shift; if "$@" > $OUT/$name.log 2>&1; then echo "OK   $name"; else echo "FAIL $name (see $OUT/$name.log)"; return 1; fi; }

echo "container: ACCFFT_DIR=$ACCFFT_DIR  P3DFFT_HOME=$P3DFFT_HOME  P3DFFT_CONF='${P3DFFT_CONF:-}'"

if [ -z "$PL_SOLVERS_ONLY" ]; then
# ---------------------------------------------------------------- AccFFT, patched
build_accfft() {
    cd $WRK || return 1
    [ -d accfft ] || git clone -q https://github.com/amirgholami/accfft.git accfft || return 1
    rm -rf accfft_pt && cp -r accfft accfft_pt && cd accfft_pt || return 1
    # both r2c (accfft_plan_dft_3d_r2c) and c2c planners contain the same if/else
    perl -0pi -e 's/\telse\n\t\tfftw_flags = FFTW_MEASURE;/\telse if (flags == ACCFFT_PATIENT)   \/* NS3D patch *\/\n\t\tfftw_flags = FFTW_PATIENT;\n\telse\n\t\tfftw_flags = FFTW_MEASURE;/g' src/accfft.cpp
    local n; n=$(grep -c "NS3D patch" src/accfft.cpp)
    echo "  AccFFT patch applied at $n places (expected 2)"; [ "$n" = 2 ] || return 1
    mkdir -p b && cd b && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$OPT/accfft_pt -DFFTW_ROOT=$FFTWD -DBUILD_GPU=false -DBUILD_SHARED=true \
             -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS="-O3 -march=znver2" && \
    make -j$J accfft accfft_utils && make install
}
[ -z "$PL_ONLY_P3DFFT" ] && c lib_accfft_pt build_accfft

# ---------------------------------------------------------------- P3DFFT x3
build_p3dfft() {   # $1 = estimate | measure | patient
    local f=$1 opt=""
    [ $f = estimate ] && opt=--enable-estimate
    [ $f = patient  ] && opt=--enable-patient
    cd $WRK || return 1
    [ -d p3dfft ] || git clone -q https://github.com/sdsc/p3dfft.git p3dfft || return 1
    rm -rf p3dfft_$f && cp -r p3dfft p3dfft_$f && cd p3dfft_$f || return 1
    # P3DFFT_CONF: the configure arguments of the original container build (without --prefix and without
    # --enable-estimate/--enable-patient), e.g. taken from logs/build_p3dfft2.log. Default = a guess.
    # eval, so quoted values with spaces in P3DFFT_CONF (e.g. FCFLAGS="-O3 -Mextend") stay one argument
    local conf=${P3DFFT_CONF:-"--enable-pgi --enable-fftw --with-fftw=$FFTWD FC=${P3DFFT_FC:-mpifort} CC=mpicc"}
    echo "  ./configure --prefix=$OPT/p3dfft_$f $opt $conf"
    eval ./configure --prefix=$OPT/p3dfft_$f $opt $conf && make -j$J && make install
}
for f in estimate measure patient; do c lib_p3dfft_$f build_p3dfft $f; done
fi

# ---------------------------------------------------------------- solvers (same flags as scripts/compile_all.sh)
cd $SRC || exit 1
RPC="-Wl,-rpath,$FFTWD/lib"
ACCLIBS="-laccfft -laccfft_utils -lfftw3_mpi -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lpthread -lm"
P3INC="-I$FFTWD/include"
P3LIBS="-lp3dfft -lfftw3_mpi -lfftw3_threads -lfftw3 -lfftw3f -lnvf -lmpi_mpifh -lm"

if [ -z "$PL_ONLY_P3DFFT" ]; then
c ns_fftw   mpicxx -O3 -std=c++17 -fopenmp -fno-math-errno NavierStokes_periodic_fftw.cpp -I$FFTWD/include -L$FFTWD/lib \
            -lfftw3_mpi -lfftw3_omp -lfftw3 -lpthread -lm $RPC -o $OUT/ns_fftw
c ns_accfft mpicxx -O3 -std=c++17 -fopenmp -fno-math-errno NavierStokes_periodic_accfft.cpp -I$FFTWD/include -L$FFTWD/lib \
            -I$ACCFFT_DIR/include -L$ACCFFT_DIR/lib $ACCLIBS $RPC -Wl,-rpath,$ACCFFT_DIR/lib -o $OUT/ns_accfft
c ns_accfft_pt mpicxx -O3 -std=c++17 -fopenmp -fno-math-errno -DNS_ACCFFT_PATCHED NavierStokes_periodic_accfft.cpp \
            -I$FFTWD/include -L$FFTWD/lib -I$OPT/accfft_pt/include -L$OPT/accfft_pt/lib $ACCLIBS $RPC \
            -Wl,-rpath,$OPT/accfft_pt/lib -o $OUT/ns_accfft_pt
fi
# P3DFFT: nvc++ as in compile_all.sh (no -fno-math-errno for nvc++)
c ns_p3dfft_orig env OMPI_CXX=nvc++ mpicxx -O3 -std=c++17 -mp '-DNS_P3DFFT_PLAN="container build"' \
            NavierStokes_periodic_p3dfft.cpp $P3INC -I$P3DFFT_HOME/include -L$P3DFFT_HOME/lib -L$FFTWD/lib $P3LIBS $RPC \
            -o $OUT/ns_p3dfft_orig
for f in estimate measure patient; do
  F=$(echo $f | tr a-z A-Z)
  c ns_p3dfft_$f env OMPI_CXX=nvc++ mpicxx -O3 -std=c++17 -mp "-DNS_P3DFFT_PLAN=\"FFTW_$F\"" \
            NavierStokes_periodic_p3dfft.cpp $P3INC -I$OPT/p3dfft_$f/include -L$OPT/p3dfft_$f/lib -L$FFTWD/lib $P3LIBS $RPC \
            -o $OUT/ns_p3dfft_$f
done
ls -l $OUT | grep -v "\.log"
