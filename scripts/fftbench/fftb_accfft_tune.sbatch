#!/bin/bash
#SBATCH -J fftb_accfft_tune
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fftb_%x_%j.out
# fftb_accfft_tune <A|B> [nodes 4] [grid 1024] — AccFFT only, Intel MPI variants against GCC t4 in the same job
#   (pure FFT benchmark; the Intel slowdown of AccFFT on 4 nodes is in the communication, the local FFT time is the same).
#   Layouts FFTB_TUNE_LAYOUTS ("32x4 64x2"); slab; MEASURE; FFTB_SPLIT=1 (t_local, comm = t_fft - t_local); 3 repetitions.
#   Configurations per layout (configuration name = first word of the CSV settings column):
#     GCC_COLL=t4        GCC + Open MPI, coll t4 (reference)
#     E6d                Intel E6d: I_MPI_ADJUST_ALLTOALL=3
#     a2a_def            IMN_ALLTOALL= (Intel MPI's default alltoall algorithm)
#     a2a1 / a2a2        I_MPI_ADJUST_ALLTOALL=1 / 2
#     a2av1 / a2av2      E6d + I_MPI_ADJUST_ALLTOALLV=1 / 2
#     ucx_dc             E6d + UCX_TLS=dc_mlx5,sm,self
#     plan_est           E6d + NS_ACCFFT_PLAN=estimate (AccFFT only: FFTW_ESTIMATE local plans and transpose method 2,
#                        no transpose measurement)
#     tm_<method>        E6d + NS_ACCFFT_TMETHOD=<method> (MEASURE plan, then the transpose method fixed), methods
#                        FFTB_TUNE_TMETHODS ("m1 m2 m3k8a m3k8s m3k16a m3k16s m3k32a m3k32s m3k64a m3k64s"; "" = none):
#                        m1 Isend/Irecv, m2 MPI_Alltoall, m3k<k>a|s k-way exchange async|sync (AccFFT's candidates)
#   A: in each layout GCC first, then the Intel variants in the order above; B: the exact reverse of A (layouts too),
#      so that a drift during the job does not favour one side. Run both (B afterany A) and parse them together.
#   Every launch prints FFTB_ACCFFT_TPLAN (the chosen transpose method, the measured candidates) and FFTB_MPIENV.
# Time estimate (4 nodes, 1024^3; per launch container start, MEASURE planning incl. the transpose candidates, check,
#   warm-up, 3 x >= 3 s, local model ~2-3 min): 2 layouts x (1 + 8 + 10) = 38 launches ~1.3-1.9 h (-t 03:00:00);
#   without the tm_ variants (FFTB_TUNE_TMETHODS="") 18 launches ~0.6-0.9 h. Per launch limit 3 x 4 min.
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/fftbench
#   A=$(sbatch --parsable -J fftb_acc_tune_A fftb_accfft_tune.sbatch A)
#   B=$(sbatch --parsable -d afterany:$A -J fftb_acc_tune_B fftb_accfft_tune.sbatch B)
#   FFTB_JOBS="$A $B" bash fftb_parse.sh        (section 6: Intel variants against GCC t4)
ORDER=${1:?usage: sbatch fftb_accfft_tune.sbatch A|B [nodes] [grid]}
NODES=${2:-4}; GRID=${3:-1024}
case $ORDER in A|B) ;; *) echo "ERROR: order '$ORDER' (A = GCC first, B = Intel first, reverse order)"; exit 1;; esac
[[ $GRID =~ ^[0-9]+$ ]] || { echo "ERROR: grid '$GRID' (one number)"; exit 1; }
LAYS=${FFTB_TUNE_LAYOUTS:-"32x4 64x2"}
TMS=${FFTB_TUNE_TMETHODS-"m1 m2 m3k8a m3k8s m3k16a m3k16s m3k32a m3k32s m3k64a m3k64s"}
for L in $LAYS; do
    [[ $L =~ ^[0-9]+x[0-9]+$ ]] || { echo "ERROR: layout '$L'"; exit 1; }
    p=${L%x*}; t=${L#*x}
    [ $((p * t)) = 128 ] || echo "WARNING: layout $L uses $((p * t)) of 128 cores per node"
    [ $((NODES * p)) -le $GRID ] || { echo "ERROR: layout $L: $((NODES * p)) ranks > $GRID (slab needs ranks <= N)"; exit 1; }
done
for m in $TMS; do [[ $m =~ ^(m1|m2|m3k[0-9]+[as])$ ]] || { echo "ERROR: FFTB_TUNE_TMETHODS entry '$m' (m1 m2 m3k<k>a m3k<k>s)"; exit 1; }; done
LINES=()
for L in $LAYS; do
    LINES+=("GCC_COLL=t4|gcc|$L|GCC_COLL=t4")
    LINES+=("E6d|intel|$L|")
    LINES+=("a2a_def|intel|$L|IMN_ALLTOALL=")
    LINES+=("a2a1|intel|$L|IMN_ALLTOALL=1")
    LINES+=("a2a2|intel|$L|IMN_ALLTOALL=2")
    LINES+=("a2av1|intel|$L|IMN_ALLTOALLV=1")
    LINES+=("a2av2|intel|$L|IMN_ALLTOALLV=2")
    LINES+=("ucx_dc|intel|$L|IMN_UCX_TLS=dc_mlx5,sm,self")
    LINES+=("plan_est|intel|$L|NS_ACCFFT_PLAN=estimate")
    for m in $TMS; do LINES+=("tm_$m|intel|$L|NS_ACCFFT_TMETHOD=$m"); done
done
CFG=""
if [ $ORDER = A ]; then for ((i = 0; i < ${#LINES[@]}; i++)); do CFG+="${LINES[i]}"$'\n'; done
else for ((i = ${#LINES[@]} - 1; i >= 0; i--)); do CFG+="${LINES[i]}"$'\n'; done; fi
export FFTB_CONFIGS="$CFG"
export FFTB_LIBS=accfft
export FFTB_SPLIT=${FFTB_SPLIT:-1} FFTB_REP=${FFTB_REP:-3}
export FFTB_DECOMP=slab FFTB_PLAN=${FFTB_PLAN:-measure}
echo "fftb_accfft_tune: order $ORDER, nodes $NODES, grid $GRID, layouts $LAYS, ${#LINES[@]} launches"
exec bash ~/guohaojie/src/NavierStockes-3D/scripts/fftbench/fftb.sbatch $NODES $GRID
