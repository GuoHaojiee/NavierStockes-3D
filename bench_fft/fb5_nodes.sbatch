#!/bin/bash
#SBATCH -J fb5
#SBATCH -p batch
#SBATCH -N 16
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 00:40:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb5_%j.out
# fb5 — the one job worth asking 16 nodes for (about 20-30 minutes): pure cuFFTMp FFT on 1..16 nodes.
#   It checks the prediction made from the 2- and 4-node data (fb1 / fb4) where it matters:
#     1024^3: does 8 / 16 nodes beat one node at all?   2048^3: does 4 -> 8 -> 16 keep scaling ~1.7x?
#   With only 8 nodes available: sbatch -N 8 fb5_nodes.sbatch   (the 16-node rows are then skipped)
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb5_nodes.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb5_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

MAXN=${SLURM_JOB_NUM_NODES:-16}
IT=10; NF=3
G 2 8 "FB_SKIP_A2A=1" fb_cufftmp 256 256 256 5 1                      # sanity, check on

# largest node counts first: if the job is cut short, the new information is already in the log
for n in 16 8 4 2 1; do
    [ $n -le $MAXN ] || continue
    G $n 8 "FB_SKIP_A2A=1" fb_cufftmp 1024 1024 1024 $IT $NF
done
for n in 16 8 4 2; do
    [ $n -le $MAXN ] || continue
    G $n 8 "FB_CHECK=0 FB_SKIP_A2A=1" fb_cufftmp 2048 2048 2048 $IT $NF
done
# weak scaling, continued from fb4 (about 4.1e9 points per node): 3200^3 on 8 nodes, 4096^3 on 16
[ 8 -le $MAXN ]  && G 8 8  "FB_CHECK=0 FB_SKIP_A2A=1" fb_cufftmp 3200 3200 3200 5 1
[ 16 -le $MAXN ] && G 16 8 "FB_CHECK=0 FB_SKIP_A2A=1" fb_cufftmp 4096 4096 4096 5 1
# heFFTe in its best layout on the same node counts (1024^3)
L2="FB_IN=zslab FB_OUT=yslab FB_R2C=0 NS_HEFFTE_ALGO=p2p_plined NS_HEFFTE_REORDER=0"
for n in 16 8; do
    [ $n -le $MAXN ] || continue
    G $n 8 "$L2" fb_heffte_gpu 1024 1024 1024 $IT $NF
done

cleanup
