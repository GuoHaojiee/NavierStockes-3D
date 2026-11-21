#!/bin/bash
# build_intel_solvers.sh — the CPU solvers with the Intel toolchain (icpx + Intel MPI), INSIDE the Intel container.
# Output: /work/build/bin_intel/
#   ns_fftw              FFTW-MPI            (/opt/fftw3, icx)
#   ns_accfft            AccFFT              (/opt/accfft, icx)
#   ns_p3dfft            P3DFFT 2.x          (/work/opt_intel/p3dfft, ifx/icx, MEASURE)
#   ns_heffte_v2_fftw    heFFTe 2.4.1, -DNS_HEFFTE_BACKEND=fftw  (/work/opt_intel/heffte_fftw)
#   ns_heffte_v2_mkl     heFFTe 2.4.1, -DNS_HEFFTE_BACKEND=mkl   (/work/opt_intel/heffte_mkl; only if it was built)
#   fb_backend_fftw      bench_fft/fftbench_backend.cpp, heFFTe-FFTW + native FFTW + native MKL (one rank)
#   fb_backend_mkl       the same with heFFTe-MKL
#   hello_a2a            scripts/intel/hello_a2a.c (multi-node launch test)
# Flags as in scripts/compile_all.sh (-O3 -std=c++17 -fopenmp -fno-math-errno), plus $NS_ICPX_FP
# (default -fp-model=precise: icpx defaults to -fp-model=fast, which may change the last digits of
# the kinetic energy; set NS_ICPX_FP="" to build with the icpx default).
# P3DFFT is linked with -lfftw3_threads, exactly like ns_p3dfft in compile_all.sh.
#
#   source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
#   srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/build_intel_solvers.sh
source "$(dirname "$0")/intel_env.sh"
LOGDIR=$IBIN; mkdir -p $IBIN; cd $SRC || exit 1
FP=${NS_ICPX_FP--fp-model=precise}
CXXF="-O3 -std=c++17 -fopenmp -fno-math-errno $FP"
echo "mpicxx -> $(mpicxx -show | awk '{print $1}')   flags: $CXXF   source: $SRC ($(git -C $SRC log -1 --format=%h 2>/dev/null))"
RPF="-Wl,-rpath,$IFFTW/lib"
FFTWL="-I$IFFTW/include -L$IFFTW/lib"

c ns_fftw   mpicxx $CXXF NavierStokes_periodic_fftw.cpp $FFTWL -lfftw3_mpi -lfftw3_omp -lfftw3 -lpthread -lm $RPF -o $IBIN/ns_fftw
c ns_accfft mpicxx $CXXF NavierStokes_periodic_accfft.cpp $FFTWL -I$IACCFFT/include -L$IACCFFT/lib \
            -laccfft -laccfft_utils -lfftw3_mpi -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lpthread -lm \
            $RPF -Wl,-rpath,$IACCFFT/lib -o $IBIN/ns_accfft

# ---- P3DFFT: C++ main + Fortran library -> Fortran runtime (ifx) and Intel MPI Fortran bindings
P3=$IOPT/p3dfft
FORTLIB=$(dirname "$(dirname "$(command -v ifx 2>/dev/null || echo /x/x)")")/lib
P3LIBS="-lp3dfft -lfftw3_mpi -lfftw3_threads -lfftw3 -lfftw3f -lm"
if [ -f $P3/lib/libp3dfft.a ]; then
  # 1st try: icpx links, Fortran runtime named explicitly; 2nd try: let mpiifx link (-nofor-main)
  c ns_p3dfft mpicxx $CXXF -DNS_P3DFFT_PLAN="\"FFTW_MEASURE\"" NavierStokes_periodic_p3dfft.cpp -I$P3/include $FFTWL -L$P3/lib \
            $P3LIBS -lmpifort -L$FORTLIB -lifcoremt -lifport -limf -lsvml -lintlc $RPF -Wl,-rpath,$FORTLIB -o $IBIN/ns_p3dfft \
  || { cp $IBIN/ns_p3dfft.log $IBIN/ns_p3dfft.try1.log; echo "     retry: compile with mpicxx, link with mpiifx -nofor-main"
       c ns_p3dfft bash -c "mpicxx $CXXF -DNS_P3DFFT_PLAN='\"FFTW_MEASURE\"' -c NavierStokes_periodic_p3dfft.cpp -I$P3/include $FFTWL -o $IBIN/ns_p3dfft.o && \
            mpiifx -nofor-main -qopenmp $IBIN/ns_p3dfft.o -L$P3/lib $FFTWL $P3LIBS -lstdc++ $RPF -o $IBIN/ns_p3dfft"; }
else
  echo "SKIP ns_p3dfft: $P3/lib/libp3dfft.a not found (run build_intel_libs.sh first)"
fi

# ---- heFFTe, two backends
H=$IOPT/heffte_fftw
if [ -f $H/lib/libheffte.so ]; then
  c ns_heffte_v2_fftw mpicxx $CXXF -DNS_HEFFTE_BACKEND=fftw NavierStokes_periodic_heffte_v2.cpp -I$H/include $FFTWL -L$H/lib \
            -lheffte -lfftw3_omp -lfftw3 -lfftw3f -lm -Wl,-rpath,$H/lib $RPF -o $IBIN/ns_heffte_v2_fftw
  have_mkl && c fb_backend_fftw mpicxx $CXXF -DFB_HEFFTE_BACKEND=fftw -DMKL_ILP64 bench_fft/fftbench_backend.cpp -I$H/include $FFTWL \
            -I$MKLROOT/include -L$H/lib -lfftw3_omp -lfftw3 -lheffte $MKL_LINK -ldl -lm -Wl,-rpath,$H/lib $RPF -o $IBIN/fb_backend_fftw
else
  echo "SKIP ns_heffte_v2_fftw, fb_backend_fftw: $H/lib/libheffte.so not found"
fi
H=$IOPT/heffte_mkl
if [ -f $H/lib/libheffte.so ] && have_mkl; then
  c ns_heffte_v2_mkl mpicxx $CXXF -DNS_HEFFTE_BACKEND=mkl -DMKL_ILP64 NavierStokes_periodic_heffte_v2.cpp -I$H/include -I$MKLROOT/include \
            -L$H/lib -lheffte $MKL_LINK -lm -Wl,-rpath,$H/lib -o $IBIN/ns_heffte_v2_mkl
  c fb_backend_mkl mpicxx $CXXF -DFB_HEFFTE_BACKEND=mkl -DMKL_ILP64 bench_fft/fftbench_backend.cpp -I$H/include $FFTWL \
            -I$MKLROOT/include -L$H/lib -lfftw3_omp -lfftw3 -lheffte $MKL_LINK -ldl -lm -Wl,-rpath,$H/lib $RPF -o $IBIN/fb_backend_mkl
else
  echo "SKIP ns_heffte_v2_mkl, fb_backend_mkl: no $H/lib/libheffte.so or no MKL"
fi

c hello_a2a mpicc -O2 scripts/intel/hello_a2a.c -o $IBIN/hello_a2a
ls -l $IBIN | grep -v "\.log$"
