#!/bin/bash
#SBATCH -J fb8
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 01:15:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb8_%j.out
# fb8 — like-for-like FFTW vs heFFTe on one node, 512^3, 128 ranks.
#
#   What was checked in the sources (2026-11-14):
#   * ICL-UT-22-02, Fig. 3.4 (256^3 c2c, 1-4 Summit nodes, one rank per core): the harness gives every
#     library the SAME pencil distribution at the beginning and at the end, tuning phases disabled,
#     one forward transform timed. Under these conditions heFFTe is ~15-30 % ahead of FFTW.
#   * In the public harness (github.com/icl-utk-edu/fiber) FFTW-MPI is never planned with
#     FFTW_MPI_TRANSPOSED_OUT in any of the 280 commits, and c2c always uses FFTW_ESTIMATE.
#     The public FFTW backend is hard-wired to 4x4x4, so it is not the exact code behind the figures.
#
#   fb6 compared FFTW "same distribution out" (2 global transposes) with heFFTe zslab -> yslab
#   (output in a DIFFERENT distribution, 1 reshape). That pair is not like-for-like.
#   Here both libraries are run in both classes:
#     class B  output in the same distribution as the input   (what the authors' harness does)
#     class A  output distribution free                       (what our solver needs)
#   The heFFTe log line "-> N MPI reshape(s) per transform" gives the reshape count of every row.
#
#   Uses the existing binaries (fb_fftw, fb_heffte_cpu). No rebuild needed.
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb8_same_dist.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb8_*.out | head -1)
#   grep -E ">>>|MPI reshape" $(ls -t ~/guohaojie/logs/fb8_*.out | head -1) | head -20
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

N=512; IT=10; NF=3
P2P="NS_HEFFTE_ALGO=p2p_plined"
A2A="NS_HEFFTE_ALGO=alltoallv"

# 0) sanity: the two new heFFTe layouts must pass the forward+inverse check on a small grid
C 128 1 "$P2P FB_IN=xpencil FB_OUT=xpencil FB_R2C=0 FB_PENCILS=1" fb_heffte_cpu 128 128 128 3 1
C 128 1 "$P2P FB_IN=zslab   FB_OUT=zslab   FB_R2C=0 FB_PENCILS=0" fb_heffte_cpu 128 128 128 3 1

for rep in 1 2 3 4 5; do                               # five passes: median and spread of every row
    # ---- class B: output in the SAME distribution as the input ----
    C 128 1 "FB_FLAG=estimate FB_TRANSPOSED=0" fb_fftw $N $N $N $IT $NF 1                                   # FFTW, no planning
    C 128 1 "FB_FLAG=measure  FB_TRANSPOSED=0" fb_fftw $N $N $N $IT $NF 1                                   # FFTW, planned
    C 128 1 "$P2P FB_IN=xpencil FB_OUT=xpencil FB_R2C=0 FB_PENCILS=1" fb_heffte_cpu $N $N $N $IT $NF        # heFFTe "r=Pencils"
    C 128 1 "$P2P FB_IN=xpencil FB_OUT=xpencil FB_R2C=0 FB_PENCILS=0" fb_heffte_cpu $N $N $N $IT $NF        # heFFTe "r=Slabs"
    C 128 1 "$A2A FB_IN=xpencil FB_OUT=xpencil FB_R2C=0 FB_PENCILS=1" fb_heffte_cpu $N $N $N $IT $NF        # heFFTe library defaults
    C 128 1 "$P2P FB_IN=zslab   FB_OUT=zslab   FB_R2C=0 FB_PENCILS=0" fb_heffte_cpu $N $N $N $IT $NF        # heFFTe, same slab in/out as FFTW
    # ---- class A: output distribution free (reference rows, same as fb6) ----
    C 128 1 "FB_FLAG=measure  FB_TRANSPOSED=1" fb_fftw $N $N $N $IT $NF 1                                   # FFTW, transposed output
    C 128 1 "$P2P FB_IN=zslab   FB_OUT=yslab   FB_R2C=0 FB_PENCILS=0" fb_heffte_cpu $N $N $N $IT $NF        # heFFTe, 1 reshape
done

cleanup
