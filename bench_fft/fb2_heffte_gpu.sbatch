#!/bin/bash
#SBATCH -J fb2
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 02:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb2_%j.out
# fb2 — heFFTe GPU parameter sweep (pure FFT).
#   L1: in=xslab r2c=2  (current solver: split x, r2c along z = the slowest memory index)
#   L2: in=zslab r2c=0  (split the slowest index, r2c along the fastest = contiguous)
#   A) one GPU: there is no communication at all, so everything above cuFFT's time is heFFTe's local work
#   B-D) 8 / 32 GPUs: the library default first, then p2p_plined with both layouts and reorder 0 / 1, then batching
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb2_heffte_gpu.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb2_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

IT=20; NF=3
L1="FB_IN=xslab FB_OUT=yslab FB_R2C=2"
L2="FB_IN=zslab FB_OUT=yslab FB_R2C=0"

# A) single GPU 512^3 (compare with fb1 part 1: cuFFTMp on one GPU, same grid)
for L in "$L1" "$L2"; do for R in 0 1; do
    G 1 1 "$L NS_HEFFTE_REORDER=$R" fb_heffte_gpu 512 512 512 $IT $NF
done; done

sweep() {   # sweep <nodes> <N> <iters> <extra env>
    local n=$1 N=$2 it=$3 X=$4
    G $n 8 "$L1 NS_HEFFTE_ALGO=alltoallv NS_HEFFTE_REORDER=0 $X" fb_heffte_gpu $N $N $N $it $NF      # library default
    for L in "$L1" "$L2"; do for R in 0 1; do
        G $n 8 "$L NS_HEFFTE_ALGO=p2p_plined NS_HEFFTE_REORDER=$R $X" fb_heffte_gpu $N $N $N $it $NF
    done; done
    for L in "$L1" "$L2"; do
        G $n 8 "$L NS_HEFFTE_ALGO=p2p_plined NS_HEFFTE_REORDER=1 FB_BATCH=1 $X" fb_heffte_gpu $N $N $N $it $NF
    done
}
sweep 1 1024 $IT ""              # B) 1 node, 8 GPUs
sweep 4 1024 $IT ""              # C) 4 nodes
sweep 4 2048 10 "FB_CHECK=0"     # D) 4 nodes, 2048^3 (a configuration that runs out of memory shows as an error: fine)

cleanup
