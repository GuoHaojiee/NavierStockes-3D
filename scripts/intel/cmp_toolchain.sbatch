#!/bin/bash
#SBATCH -J ct
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 02:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/ct_%x_%j.out
# cmp_toolchain — GCC toolchain (Open MPI, GCC container) vs Intel toolchain (icpx + Intel MPI, Intel container),
#   per CPU library, one node. 512^3, dt=1e-6, 20 steps, bench, 128x1 and 64x2, 3 repetitions (OUTER loop).
#   All libraries NS_DECOMP=slab, NS_FFTW_PLAN=measure (P3DFFT: MEASURE fixed in the library; heFFTe: ESTIMATE, hard-coded).
#   Reference kinetic energy 1.7933833444e+08 — the parser counts the runs that match.
#   Config labels <LIB>_<toolchain>_<ranks>x<threads>, toolchain:
#     gcc        Open MPI, binaries in /work/build/bin_gcc (scripts/compile_all.sh; override with CT_GBIN)
#     intel      Intel MPI, /work/build/bin_intel (build_intel_solvers.sh); heFFTe: FFTW backend
#     intelmkl   heFFTe, MKL backend, library defaults (use_reorder = 0 for MKL in heFFTe 2.4.1)
#     intelmklr1 heFFTe, MKL backend, NS_HEFFTE_REORDER=1 (same plan options as the FFTW backend)
# One library per job, so the four can run in parallel on four nodes (ratios only inside one job):
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/intel
#   for l in fftw accfft p3dfft heffte; do sbatch -J ct_$l cmp_toolchain.sbatch $l; done
# Results:
#   for l in fftw accfft p3dfft heffte; do bash ct_parse.sh $(ls -t ~/guohaojie/logs/ct_ct_${l}_*.out | head -1); done
# Options (environment at sbatch time): CT_N (512), CT_STEPS (20), CT_REP (3), CT_LAYOUTS ("128x1 64x2"),
#   CT_EREF (reference energy for another grid), CT_ISRUN (how the single Intel launcher task is started)
source ~/guohaojie/src/NavierStockes-3D/scripts/intel/intel_common.sh
precheck
LIB=${1:?usage: sbatch cmp_toolchain.sbatch fftw|accfft|p3dfft|heffte}
GBIN=${CT_GBIN:-/work/build/bin_gcc}; IBIN=/work/build/bin_intel
N=${CT_N:-512}; DT=1e-6; NS=${CT_STEPS:-20}; REP=${CT_REP:-3}; LAYOUTS=${CT_LAYOUTS:-"128x1 64x2"}
ISRUN=${CT_ISRUN:-"srun -N 1 -n 1 -c 128 --cpu-bind=none"}
COMMON="NS_FFTW_PLAN=measure NS_DECOMP=slab"
KEEP="^decomposition:|process grid|MPI reshapes|heFFTe backend|heFFTe plan options|planner flag|planning time|plan time|\(planning\) time|Kinetic energy|Avg per step:|FFT \(incl|rror|MPI startup\(\): +(Rank|Intel|[0-9]+ )"

# G <label> <ranks> <threads> "<extra env>" <program>       GCC container, Open MPI, srun launches the ranks
G() {
    local label=$1 np=$2 nt=$3 envs="$COMMON $4" prog=$5
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$np threads=$nt plan=measure | $envs $GBIN/$prog $N $N $N $DT $NS $nt bench"
    srun --mpi=pmix --cpu-bind=cores --time=20 -N 1 -n $np -c $nt $M bash -c "export NS_MODE=cpu; source /work/build/run_env.sh; \
        export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close $envs; exec $GBIN/$prog $N $N $N $DT $NS $nt bench" 2>&1 | grep -E "$KEEP"
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}
# I <label> <ranks> <threads> "<extra env>" <program>       Intel container, Intel MPI, mpirun (fork + shm) in one task
I() {
    local label=$1 np=$2 nt=$3 envs="$COMMON $4" prog=$5 dbg=""
    [ -z "$IDBG_DONE" ] && { dbg="I_MPI_DEBUG=4"; IDBG_DONE=1; }       # pin map of the first Intel run only
    echo ">>> [$(date +%H:%M:%S)] ${label} ranks=$np threads=$nt plan=measure | $envs $IBIN/$prog $N $N $N $DT $NS $nt bench"
    $ISRUN --time=20 $MI bash -c "source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1; \
        export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm I_MPI_PIN_DOMAIN=$nt:compact OMP_NUM_THREADS=$nt OMP_PLACES=cores \
               OMP_PROC_BIND=close KMP_BLOCKTIME=0 MKL_NUM_THREADS=$nt $dbg $envs; \
        exec mpirun -np $np $IBIN/$prog $N $N $N $DT $NS $nt bench" 2>&1 | grep -E "$KEEP" | awk '/MPI startup\(\): +[0-9]/ { if (++k > 4) next } 1'
    echo "<<< exit=${PIPESTATUS[0]}"; echo
}

echo "library=$LIB  N=$N steps=$NS reps=$REP layouts=$LAYOUTS  node=$(hostname)"
echo "GCC binaries: $(ls -l --time-style=+%F_%R $SCR/build/bin_gcc/ns_fftw $SCR/build/bin_gcc/ns_accfft $SCR/build/bin_gcc/ns_p3dfft $SCR/build/bin_gcc/ns_heffte_v2 2>&1 | awk '{print $6, $7}' | tr '\n' ' ')"
echo "Intel binaries: $(ls $SCR/build/bin_intel 2>&1 | grep -v log | tr '\n' ' ')"
for rep in $(seq 1 $REP); do
  echo "=============== repetition $rep"
  for L in $LAYOUTS; do
    np=${L%x*}; nt=${L#*x}
    case $LIB in
    fftw)   G FFTW_gcc_$L     $np $nt "" ns_fftw
            I FFTW_intel_$L   $np $nt "" ns_fftw ;;
    accfft) G AccFFT_gcc_$L   $np $nt "" ns_accfft
            I AccFFT_intel_$L $np $nt "" ns_accfft ;;
    p3dfft) G P3DFFT_gcc_$L   $np $nt "" ns_p3dfft
            I P3DFFT_intel_$L $np $nt "" ns_p3dfft ;;
    heffte) G heFFTe_gcc_$L        $np $nt "" ns_heffte_v2
            I heFFTe_intel_$L      $np $nt "" ns_heffte_v2_fftw
            I heFFTe_intelmkl_$L   $np $nt "" ns_heffte_v2_mkl
            I heFFTe_intelmklr1_$L $np $nt "NS_HEFFTE_REORDER=1" ns_heffte_v2_mkl ;;
    *) echo "unknown library $LIB"; break 2 ;;
    esac
  done
done
cleanup_all
