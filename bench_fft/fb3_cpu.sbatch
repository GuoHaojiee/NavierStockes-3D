#!/bin/bash
#SBATCH -J fb3
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 01:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb3_%j.out
# fb3 — CPU pure FFT, 512^3, one node: why is heFFTe twice as slow as FFTW in the solver?
#   The solver heffte_v2 takes its boxes from proc_setup_min_surface = 4x4x8 BRICKS for 128 ranks,
#   and bricks on both sides cost FOUR MPI reshapes per transform (FFTW slab: one transpose).
#   The log prints "N MPI reshape(s) per transform" for every heFFTe run.
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb3_cpu.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb3_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

N=512; IT=10; NF=3
# FFTW-MPI reference (measure = what the solver uses)
C 128 1 "FB_FLAG=measure" fb_fftw $N $N $N $IT $NF 1

# heFFTe, 128 ranks: bricks (= solver today, 4 reshapes) / pencils (2) / slabs (1), two reshape algorithms
for A in alltoallv p2p_plined; do
    C 128 1 "NS_HEFFTE_ALGO=$A FB_IN=brick   FB_OUT=brick   FB_R2C=2" fb_heffte_cpu $N $N $N $IT $NF   # solver today
    C 128 1 "NS_HEFFTE_ALGO=$A FB_IN=zpencil FB_OUT=xpencil FB_R2C=2" fb_heffte_cpu $N $N $N $IT $NF   # like AccFFT / P3DFFT
    C 128 1 "NS_HEFFTE_ALGO=$A FB_IN=xslab   FB_OUT=yslab   FB_R2C=2" fb_heffte_cpu $N $N $N $IT $NF   # like FFTW; no change of the physics
    C 128 1 "NS_HEFFTE_ALGO=$A FB_IN=zslab   FB_OUT=yslab   FB_R2C=0" fb_heffte_cpu $N $N $N $IT $NF
done
C 128 1 "NS_HEFFTE_ALGO=p2p_plined FB_IN=xslab FB_OUT=yslab FB_R2C=2 FB_BATCH=1" fb_heffte_cpu $N $N $N $IT $NF

# How much do heFFTe's FFTW plans (FFTW_ESTIMATE + NULL arrays) cost for the local 1-D FFTs?
# 128 copies at once, so the memory traffic is as in the real run; only copy 0 prints.
C 128 1 "" est_vs_meas 512 2048 20

# FFTW-MPI planner flag (for FFTW-MPI the flag also selects the transpose algorithm, so this is
# NOT the heFFTe effect; it only shows how sensitive FFTW-MPI itself is). PATIENT last: slow planning.
C 128 1 "FB_FLAG=estimate" fb_fftw $N $N $N $IT $NF 1
C 64  2 "FB_FLAG=measure"  fb_fftw $N $N $N $IT $NF 2
C 128 1 "FB_FLAG=patient"  fb_fftw $N $N $N $IT $NF 1

cleanup
