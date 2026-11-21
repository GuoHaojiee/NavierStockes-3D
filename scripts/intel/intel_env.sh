# intel_env.sh — sourced INSIDE the Intel container (ns3d_intel.sqsh) by every scripts/intel/* script.
# Paths can be overridden from the environment (that is also how the scripts were tested off-cluster).
# container environment file (check_env.sh shows candidates); default: the first that exists.
# build/build_intel.sh writes /etc/profile.d/ns3d_intel.sh.
if [ -z "$INTEL_ENV_FILE" ]; then
    for f in /etc/profile.d/ns3d_env.sh /etc/profile.d/ns3d_intel.sh; do [ -f $f ] && { INTEL_ENV_FILE=$f; break; }; done
    INTEL_ENV_FILE=${INTEL_ENV_FILE:-/etc/profile.d/ns3d_env.sh}
fi
if [ -f "$INTEL_ENV_FILE" ]; then source "$INTEL_ENV_FILE"; else echo "WARNING: environment file $INTEL_ENV_FILE not found (set INTEL_ENV_FILE=...)"; fi
export I_MPI_CC=${I_MPI_CC:-icx} I_MPI_CXX=${I_MPI_CXX:-icpx} I_MPI_F90=${I_MPI_F90:-ifx} I_MPI_FC=${I_MPI_FC:-ifx} I_MPI_F77=${I_MPI_F77:-ifx}
SRC=${NS_SRC:-/work/src/NavierStockes-3D}
IOPT=${INTEL_OPT:-/work/opt_intel}                # libraries built by build_intel_libs.sh
IFFTW=${INTEL_FFTW:-/opt/fftw3}                   # FFTW built with icx (container)
IACCFFT=${INTEL_ACCFFT:-/opt/accfft}              # AccFFT built with icx (container)
IBIN=${INTEL_BIN:-/work/build/bin_intel}          # solvers and benchmarks built by build_intel_solvers.sh
if [ -z "$MKLROOT" ] && [ -d /opt/intel/oneapi/mkl/latest ]; then export MKLROOT=/opt/intel/oneapi/mkl/latest; fi
# MKL as heFFTe links it (FindHeffteMKL.cmake, Intel compiler): ILP64 interface + Intel OpenMP threading
MKL_LINK="-L$MKLROOT/lib -L$MKLROOT/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -Wl,-rpath,$MKLROOT/lib -Wl,-rpath,$MKLROOT/lib/intel64"
have_mkl() { [ -n "$MKLROOT" ] && [ -f "$MKLROOT/include/mkl_dfti.h" ]; }
# c <name> <command...> — run a build step, log to $LOGDIR/<name>.log, print OK/FAIL (as in planner/build_planner.sh)
c(){ local name=$1; shift; if "$@" > $LOGDIR/$name.log 2>&1; then echo "OK   $name"; else echo "FAIL $name (see $LOGDIR/$name.log)"; return 1; fi; }
