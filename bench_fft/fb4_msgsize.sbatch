#!/bin/bash
#SBATCH -J fb4
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 01:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb4_%j.out
# fb4 — how does the time of ONE transpose depend on the amount of data and on the number of GPUs?
#   fb1 gave 5 multi-node points; they fit  t = t0 + D/B  with B ~ 163 GB/s per node and a fixed
#   cost t0 that grows with the number of GPUs (~0.4 ms on 2 nodes, ~1.1 ms on 4 nodes).
#   That fixed cost decides whether 1024^3 can ever gain from 8-16 nodes, so it has to be measured
#   properly: many grid sizes, 1/2/4 nodes, and fewer GPUs per node to separate "nodes" from "GPUs".
#   Uses the existing fb_cufftmp binary (phases [A] and [R] only). No rebuild needed.
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb4_msgsize.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb4_*.out | head -1)
# Reading the table: with fewer than 8 GPUs per node only that many IB ports are in use, so compare
# B_resh / (GPUs per node) = GB/s per port (25 nominal, 24.7 measured), not the percentage of 200.
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

IT=10
X="FB_CHECK=0 FB_SKIP_LOCAL=1 FB_SKIP_A2A=1"
run() { G $1 $2 "$X" fb_cufftmp $3 $3 $3 $IT 1; }     # run <nodes> <GPUs per node> <N>

# 0) sanity with the correctness check on
G 2 8 "FB_SKIP_LOCAL=1 FB_SKIP_A2A=1" fb_cufftmp 256 256 256 5 1

# 1) main series: 8 GPUs per node, 1 / 2 / 4 nodes, seven grid sizes (all divisible by 32)
for N in 256 512 768 1024 1280 1536 2048; do
    for n in 1 2 4; do run $n 8 $N; done
done

# 2) the three reference rows again, twice more: run-to-run spread of the numbers used in the model
for rep in 1 2; do
    for N in 1024 2048; do for n in 2 4; do run $n 8 $N; done; done
done

# 3) same number of GPUs on a different number of nodes: is the fixed cost per GPU or per node?
for N in 512 1024 1536; do
    run 2 4 $N      #  8 GPUs on 2 nodes   (compare with 1 node x 8)
    run 4 2 $N      #  8 GPUs on 4 nodes
    run 4 4 $N      # 16 GPUs on 4 nodes   (compare with 2 nodes x 8)
    run 2 2 $N      #  4 GPUs on 2 nodes
done

# 4) weak scaling at about 4.2e9 grid points per node: 1600^3 on 1 node, 2048^3 on 2, 2560^3 on 4.
#    Here the local estimate [B] is kept, so the table shows local + transpose for each.
#    (The solver with its 15 arrays fits all three: 57-60 GiB per GPU.)
W="FB_CHECK=0 FB_SKIP_A2A=1"
G 1 8 "$W" fb_cufftmp 1600 1600 1600 $IT 1
G 2 8 "$W" fb_cufftmp 2048 2048 2048 $IT 1
G 4 8 "$W" fb_cufftmp 2560 2560 2560 $IT 1

cleanup
