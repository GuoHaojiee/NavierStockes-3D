#!/bin/bash
# check_env.sh — what does the Intel container offer? Run this first; build_intel_libs.sh depends on the answers.
#   source ~/guohaojie/scripts_intel_common.sh   (or: source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh)
#   srun -p batch --exclude=cn34,cn18,cn25 -N1 -n1 -c4 --time=00:10:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/check_env.sh
echo "=== environment file candidates (files that set I_MPI_CXX)"
grep -ls "I_MPI_CXX" /etc/profile.d/* /etc/*.sh /opt/*.sh /work/build/*.sh 2>/dev/null || echo "  none found"
source "$(dirname "$0")/intel_env.sh"
v(){ printf "%-10s " "$1"; if command -v "$1" >/dev/null 2>&1; then echo "$(command -v $1)"; "$@" 2>&1 | head -2 | sed 's/^/           /'; else echo "NOT FOUND"; fi; }
echo "=== compilers"
v icx --version; v icpx --version; v ifx --version; v ifort --version; v gfortran --version
echo "=== MPI wrappers (I_MPI_CC=$I_MPI_CC I_MPI_CXX=$I_MPI_CXX I_MPI_F90=$I_MPI_F90)"
v mpirun --version; v mpiifx -v; v mpif90 -v; v mpicxx -v
for w in mpicc mpicxx mpif90 mpiifx; do command -v $w >/dev/null && echo "  $w -show: $($w -show 2>&1 | cut -c1-150)"; done
echo "=== MKL"
echo "  MKLROOT=${MKLROOT:-<empty>}"
ls -d /opt/intel/oneapi/mkl/* 2>/dev/null | sed 's/^/  /'
have_mkl && echo "  mkl_dfti.h: $MKLROOT/include/mkl_dfti.h  (heffte_mkl WILL be built)" || echo "  mkl_dfti.h NOT found  (heffte_mkl will be skipped)"
ls $MKLROOT/lib/libmkl_intel_ilp64.so* $MKLROOT/lib/libmkl_intel_thread.so* $MKLROOT/lib/libmkl_core.so* 2>/dev/null | sed 's/^/  /'
echo "  libiomp5: $(find /opt/intel -name 'libiomp5.so' 2>/dev/null | head -2 | tr '\n' ' ')"
echo "=== cmake / autotools / git (heFFTe 2.4.1 needs cmake >= 3.19)"
v cmake --version; v autoreconf --version; v libtoolize --version; v git --version; v make --version
echo "=== FFTW / AccFFT in the container"
echo "  $IFFTW/lib:"; ls $IFFTW/lib 2>/dev/null | grep -E "\.so$|\.a$" | tr '\n' ' '; echo
ls $IFFTW/include/fftw3.h 2>/dev/null | sed 's/^/  /'
echo "  $IACCFFT:"; ls $IACCFFT/lib $IACCFFT/include 2>/dev/null | tr '\n' ' '; echo
ls -d /work/opt_im/* 2>/dev/null | sed 's/^/  /'
echo "=== Intel MPI: PMI libraries (for srun --mpi=pmi2/pmix + I_MPI_PMI_LIBRARY)"
d=$(dirname $(dirname $(command -v mpirun 2>/dev/null) 2>/dev/null) 2>/dev/null)
echo "  I_MPI_ROOT=${I_MPI_ROOT:-<empty>}  (mpirun in $d)"
find ${I_MPI_ROOT:-$d} /opt/intel /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/lib/slurm /usr/lib64/slurm \
     -maxdepth 4 \( -name 'libpmi*.so*' -o -name 'libpmix*.so*' -o -name 'mpi_pmi*.so' -o -name 'mpi_pmix*.so' \) 2>/dev/null | sort -u | sed 's/^/  /'
echo "=== Intel MPI fabric providers (libfabric)"
v fi_info --version
command -v fi_info >/dev/null && fi_info -l 2>/dev/null | grep -E "^[a-z]" | tr '\n' ' '; echo
ls ${I_MPI_ROOT:-$d}/opt/mpi/libfabric/lib/prov 2>/dev/null | tr '\n' ' '; echo
ls /dev/infiniband 2>/dev/null | tr '\n' ' '; echo
echo "=== numactl"
ls -l /work/tools/numactl 2>/dev/null || echo "  /work/tools/numactl not found"
echo "=== end of check_env"
