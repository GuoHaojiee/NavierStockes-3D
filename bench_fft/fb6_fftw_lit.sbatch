#!/bin/bash
#SBATCH -J fb6
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 00:40:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb6_%j.out
# fb6 — why do the heFFTe authors report heFFTe >= FFTW while we measure FFTW 25 % ahead?
#   In their public benchmark suite (github.com/icl-utk-edu/fiber, hpfft_backend_fftw.h) FFTW-MPI is
#   planned with FFTW_ESTIMATE and WITHOUT FFTW_MPI_TRANSPOSED_OUT: natural-order output, so FFTW
#   transposes back = TWO global transposes per transform. Our solver uses transposed output (ONE).
#   This job reproduces their FFTW configuration next to ours, and heFFTe in three layouts
#   (bricks = default of heFFTe's own benchmark, pencils, slabs), all in one job on one node.
#   Needs the rebuilt fb_fftw (FB_TRANSPOSED switch):  build_fb.sh first.
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb6_fftw_lit.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb6_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

N=512; IT=10; NF=3
H="NS_HEFFTE_ALGO=p2p_plined"
for rep in 1 2 3; do                                   # three passes: run-to-run spread of every row
    C 128 1 "FB_FLAG=measure  FB_TRANSPOSED=1" fb_fftw $N $N $N $IT $NF 1      # ours: 1 transpose, MEASURE
    C 128 1 "FB_FLAG=estimate FB_TRANSPOSED=0" fb_fftw $N $N $N $IT $NF 1      # authors' benchmark: 2 transposes, ESTIMATE
    C 128 1 "FB_FLAG=measure  FB_TRANSPOSED=0" fb_fftw $N $N $N $IT $NF 1      # 2 transposes, MEASURE: effect of the transpose alone
    C 128 1 "$H FB_IN=brick   FB_OUT=brick   FB_R2C=2" fb_heffte_cpu $N $N $N $IT $NF
    C 128 1 "$H FB_IN=zpencil FB_OUT=xpencil FB_R2C=2" fb_heffte_cpu $N $N $N $IT $NF
    C 128 1 "$H FB_IN=zslab   FB_OUT=yslab   FB_R2C=0" fb_heffte_cpu $N $N $N $IT $NF
done

cleanup
