#!/bin/bash
# build_fftb.sh gcc|intel — the pure-FFT benchmark (scripts/fftbench/fftbench.cpp) for the four CPU libraries,
# with the compilers, flags and libraries of the solver binaries the comparisons use:
#   gcc    (GCC/NVHPC container)  as scripts/compile_all.sh (bin_gcc) and build_p3dfft_opt.sh gcc (bin_p3d/gcc)
#          fftb_fftw fftb_accfft fftb_heffte (nvcc -ccbin mpicxx, FFTW backend)  fftb_p3dfft (nvc++ -mp, /work/opt_p3d/gcc)
#   intel  (Intel container)      as scripts/intel/build_intel_solvers.sh (bin_intel) and build_p3dfft_opt.sh intel
#          fftb_fftw fftb_accfft fftb_heffte_mkl fftb_heffte_fftw  fftb_p3dfft (/work/opt_p3d/intel)
# Output: /work/build/bin_fftb/<gcc|intel>/ (FFTB_OUT to change) — never bin_gcc, bin_intel or bin_p3d, which running
# solver jobs use. One log per binary next to it (<name>.log); a binary is replaced only after a successful link.
#   source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh      # $M, $MI, $EXCL
#   srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $M  bash /work/src/NavierStockes-3D/scripts/fftbench/build_fftb.sh gcc
#   srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $MI bash /work/src/NavierStockes-3D/scripts/fftbench/build_fftb.sh intel
# Off-cluster test: FFTB_LOCAL=1 bash build_fftb.sh gnu   (system mpicxx, ~/.local libraries)
MODE=${1:?usage: build_fftb.sh gcc|intel}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${NS_SRC:-$(cd "$HERE/../.." && pwd)}
R=${FFTB_ROOT:-/work}
OUT=${FFTB_OUT:-$R/build/bin_fftb/$MODE}
case "$OUT" in */bin_gcc|*/bin_intel|*/bin_p3d*) echo "refusing to build into $OUT (solver binaries)"; exit 1 ;; esac
mkdir -p $OUT; cd $SRC || exit 1
B=scripts/fftbench/fftbench.cpp
n=0; nfail=0
# fb_build <name> <command...>: build to $OUT/<name>.new (the -o is added here), rename on success.
# Its own name on purpose: scripts/intel/intel_env.sh (sourced in intel mode) defines c(), which runs the command
# WITHOUT -o and logs to $LOGDIR. Before df87e92+1 this script was called c() too and the Intel c() replaced it:
# every target printed OK, the binaries went to ./a.out, nothing reached $OUT and the count was "0 of 0".
fb_build() {
    local name=$1; shift; n=$((n + 1))
    if "$@" -o $OUT/$name.new > $OUT/$name.log 2>&1 && [ -x $OUT/$name.new ]; then mv -f $OUT/$name.new $OUT/$name; echo "OK   $name"
    else rm -f $OUT/$name.new; echo "FAIL $name (see $OUT/$name.log)"; nfail=$((nfail + 1)); return 1; fi
}
EXPECT=()      # binaries this mode must produce; checked at the end
case $MODE in
gcc)
    source /etc/profile.d/ns3d_env.sh
    NVLIB=$NVHPC_ROOT/compilers/lib
    MATH=$NVHPC_ROOT/math_libs/12.5/targets/x86_64-linux
    export CPATH=/opt/fftw3/include:$MATH/include:$HEFFTE_DIR/include:${CPATH:-}
    export LIBRARY_PATH=/opt/fftw3/lib:$MATH/lib:$HEFFTE_DIR/lib:${LIBRARY_PATH:-}
    RPC="-Wl,-rpath,/opt/fftw3/lib -Wl,-rpath,$HEFFTE_DIR/lib -Wl,-rpath,$NVLIB -Wl,-rpath,$MATH/lib"
    RPN="-Xlinker -rpath,/opt/fftw3/lib -Xlinker -rpath,$HEFFTE_DIR/lib -Xlinker -rpath,$MATH/lib"
    G="-O3 -std=c++17 -fopenmp -fno-math-errno -I$SRC"
    EXPECT=(fftb_fftw fftb_accfft fftb_heffte fftb_p3dfft)
    fb_build fftb_fftw   mpicxx $G -DFFTB_LIB=1 $B -lfftw3_mpi -lfftw3_omp -lfftw3 -lpthread -lm $RPC
    fb_build fftb_accfft mpicxx $G -DFFTB_LIB=2 $B -I$ACCFFT_DIR/include -L$ACCFFT_DIR/lib \
                  -laccfft -laccfft_utils -lfftw3_mpi -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lpthread -lm $RPC
    fb_build fftb_heffte nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpicxx -Xcompiler -fopenmp -Xcompiler -fno-math-errno -I$SRC \
                  -DFFTB_LIB=4 $B -lheffte -lfftw3_omp -lfftw3 -lfftw3f -lm $RPN
    P3=$R/opt_p3d/gcc
    if [ -f $P3/lib/libp3dfft.a ]; then
        fb_build fftb_p3dfft env OMPI_CXX=nvc++ mpicxx -O3 -std=c++17 -mp -I$SRC -DFFTB_LIB=3 '-DNS_P3DFFT_PLAN="FFTW_MEASURE"' \
                  "-DNS_P3DFFT_LIB=\"$P3/lib/libp3dfft.a (gcc)\"" $B -I$P3/include -I/opt/fftw3/include -L$P3/lib -L/opt/fftw3/lib \
                  -lp3dfft -lfftw3_mpi -lfftw3_threads -lfftw3 -lfftw3f -lm -lnvf -lmpi_mpifh -Wl,-rpath,/opt/fftw3/lib
    else echo "SKIP fftb_p3dfft: $P3/lib/libp3dfft.a not found (scripts/p3dfft/build_p3dfft_opt.sh gcc)"; fi ;;
intel)
    source "$HERE/../intel/intel_env.sh"
    FP=${NS_ICPX_FP--fp-model=precise}
    CXXF="-O3 -std=c++17 -fopenmp -fno-math-errno $FP -I$SRC"
    RPF="-Wl,-rpath,$IFFTW/lib"; FFTWL="-I$IFFTW/include -L$IFFTW/lib"
    EXPECT=(fftb_fftw fftb_accfft fftb_heffte_fftw fftb_heffte_mkl fftb_p3dfft)
    fb_build fftb_fftw   mpicxx $CXXF -DFFTB_LIB=1 $B $FFTWL -lfftw3_mpi -lfftw3_omp -lfftw3 -lpthread -lm $RPF
    fb_build fftb_accfft mpicxx $CXXF -DFFTB_LIB=2 $B $FFTWL -I$IACCFFT/include -L$IACCFFT/lib \
                  -laccfft -laccfft_utils -lfftw3_mpi -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lpthread -lm $RPF -Wl,-rpath,$IACCFFT/lib
    H=$IOPT/heffte_fftw
    [ -f $H/lib/libheffte.so ] && fb_build fftb_heffte_fftw mpicxx $CXXF -DFFTB_LIB=4 -DNS_HEFFTE_BACKEND=fftw $B -I$H/include $FFTWL -L$H/lib \
                  -lheffte -lfftw3_omp -lfftw3 -lfftw3f -lm -Wl,-rpath,$H/lib $RPF
    H=$IOPT/heffte_mkl
    if [ -f $H/lib/libheffte.so ] && have_mkl; then
        # FFTW only for the FFTB_SPLIT local model (as fb_backend_mkl links both)
        fb_build fftb_heffte_mkl mpicxx $CXXF -DFFTB_LIB=4 -DNS_HEFFTE_BACKEND=mkl -DMKL_ILP64 $B -I$H/include -I$MKLROOT/include $FFTWL -L$H/lib \
                  -lheffte -lfftw3_omp -lfftw3 $MKL_LINK -lm -Wl,-rpath,$H/lib $RPF
    else echo "SKIP fftb_heffte_mkl: no $H/lib/libheffte.so or no MKL"; fi
    P3=$R/opt_p3d/intel
    if [ -f $P3/lib/libp3dfft.a ]; then
        FORTLIB=$(dirname "$(dirname "$(command -v ifx 2>/dev/null || echo /x/x)")")/lib
        P3LIBS="-lp3dfft -lfftw3_mpi -lfftw3_threads -lfftw3 -lfftw3f -lm"
        PD=('-DNS_P3DFFT_PLAN="FFTW_MEASURE"' "-DNS_P3DFFT_LIB=\"$P3/lib/libp3dfft.a (intel)\"")
        # 1st try: icpx links with the ifx runtime; on failure (not "no binary there": an older one may exist)
        # 2nd try: compile with mpicxx, link with mpiifx -nofor-main. The pair counts as one build step.
        if ! fb_build fftb_p3dfft mpicxx $CXXF -DFFTB_LIB=3 "${PD[@]}" $B -I$P3/include $FFTWL -L$P3/lib $P3LIBS \
                  -lmpifort -L$FORTLIB -lifcoremt -lifport -limf -lsvml -lintlc $RPF -Wl,-rpath,$FORTLIB; then
            cp $OUT/fftb_p3dfft.log $OUT/fftb_p3dfft.try1.log
            echo "     retry: compile with mpicxx, link with mpiifx -nofor-main (1st try: $OUT/fftb_p3dfft.try1.log)"
            n=$((n - 1)); nfail=$((nfail - 1))
            if mpicxx $CXXF -DFFTB_LIB=3 "${PD[@]}" -c $B -I$P3/include $FFTWL -o $OUT/fftb_p3dfft.o > $OUT/fftb_p3dfft.try2.log 2>&1; then
                fb_build fftb_p3dfft mpiifx -nofor-main -qopenmp $OUT/fftb_p3dfft.o -L$P3/lib $FFTWL $P3LIBS -lstdc++ $RPF
            else
                n=$((n + 1)); nfail=$((nfail + 1)); echo "FAIL fftb_p3dfft (compile step, see $OUT/fftb_p3dfft.try2.log)"
            fi
        fi
    else echo "SKIP fftb_p3dfft: $P3/lib/libp3dfft.a not found (scripts/p3dfft/build_p3dfft_opt.sh intel)"; fi ;;
gnu)   # off-cluster test only
    L=${FFTB_PREFIX:-$HOME/.local}
    G="-O3 -std=c++17 -fopenmp -fno-math-errno -I$SRC"
    EXPECT=(fftb_fftw fftb_accfft fftb_heffte fftb_p3dfft)
    fb_build fftb_fftw   mpicxx $G -DFFTB_LIB=1 $B -lfftw3_mpi -lfftw3_omp -lfftw3 -lm
    fb_build fftb_accfft mpicxx $G -DFFTB_LIB=2 $B -I$L/include -L$L/lib -laccfft -laccfft_utils -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lm
    fb_build fftb_p3dfft mpicxx $G -DFFTB_LIB=3 $B -I$L/include -L$L/lib -lp3dfft -lfftw3_mpi -lfftw3_omp -lfftw3 -lfftw3f -lgfortran -lmpi_mpifh -lm
    fb_build fftb_heffte mpicxx $G -DFFTB_LIB=4 $B -I$L/include -L$L/lib -lheffte -Wl,-rpath,$L/lib -lfftw3_omp -lfftw3 -lfftw3f -lm ;;
*) echo "usage: build_fftb.sh gcc|intel"; exit 2 ;;
esac
echo "source $(git -C $SRC log -1 --format=%h 2>/dev/null || echo ?) -> $OUT: $((n - nfail)) of $n build steps OK"
ls -l $OUT | grep -v "\.log$"
# every expected binary must exist and be executable (independent of what the build steps printed)
miss=0
for b in "${EXPECT[@]}"; do
    if [ -x $OUT/$b ] && [ -f $OUT/$b ]; then echo "CHECK $b present"; else echo "FAIL  $b missing in $OUT"; miss=$((miss + 1)); fi
done
if [ $miss -gt 0 ] || [ ${#EXPECT[@]} = 0 ]; then echo "BUILD FAILED: $miss of ${#EXPECT[@]} expected binaries missing"; exit 1; fi
# a failed step keeps an older binary of that name: present, but not from this source
if [ $nfail -gt 0 ]; then echo "BUILD FAILED: $nfail build step(s) failed (an older binary of that name may still be there)"; exit 1; fi
echo "BUILD OK: all ${#EXPECT[@]} expected binaries present"
