#!/bin/bash
#SBATCH -J fb1
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH -c 16
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18
#SBATCH -t 01:30:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fb1_%j.out
# fb1 — cuFFTMp pure FFT on 1 / 2 / 4 nodes: why the step gets slower when the job leaves one node.
#   part 1-3: time of one FFT, of its local part, of one transpose alone -> used share of the IB bandwidth
#   part 4  : the multi-node switches from NVIDIA's documentation that we have never tried
#   sbatch ~/guohaojie/src/NavierStockes-3D/bench_fft/fb1_cufftmp.sbatch
#   bash   ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_parse.sh $(ls -t ~/guohaojie/logs/fb1_*.out | head -1)
source ~/guohaojie/src/NavierStockes-3D/bench_fft/fb_common.sh
precheck

IT=20; NF=3
# 0) small sanity run: the correctness check must say OK
G 1 8 "" fb_cufftmp 256 256 256 5 $NF

# 1) one GPU, 512^3 = the same number of points per GPU as 1024^3 on 8 GPUs: pure local FFT cost
G 1 1 "" fb_cufftmp 512 512 512 $IT $NF

# 2) 1024^3 on 1 / 2 / 4 nodes (1 node = NVLink only)
for n in 1 2 4; do G $n 8 "" fb_cufftmp 1024 1024 1024 $IT $NF; done

# 3) 2048^3 on 2 / 4 nodes, and on 1 node with a single field (NVLink-only reference; the solver itself does not fit there)
for n in 2 4; do G $n 8 "FB_CHECK=0" fb_cufftmp 2048 2048 2048 $IT $NF; done
G 1 8 "FB_CHECK=0 FB_SKIP_LOCAL=1 FB_SKIP_A2A=1" fb_cufftmp 2048 2048 2048 10 1

# 4) documented switches, 2 and 4 nodes. Compare t_dist / t_reshape with the "default" rows of parts 2-3.
#    The correctness check stays on for 1024^3: an experimental path must still give the right answer.
knob() {   # knob "<VAR=val ...>"
    for n in 2 4; do
        G $n 8 "$1 FB_SKIP_LOCAL=1 FB_SKIP_A2A=1" fb_cufftmp 1024 1024 1024 $IT $NF
        G $n 8 "$1 FB_SKIP_LOCAL=1 FB_SKIP_A2A=1 FB_CHECK=0" fb_cufftmp 2048 2048 2048 10 $NF
    done
}
knob "CUFFT_RESHAPE_USE_PACKING=1"
knob "NVSHMEM_IB_SL=1"

# 5) NVSHMEM GPU-initiated transport (IBGDA). It needs driver / NIC support that may be missing here,
#    so: one short probe first with a 4-minute limit; the full set only if the probe works.
STEP_MIN=4
if G 2 8 "NVSHMEM_IB_ENABLE_IBGDA=1 FB_SKIP_LOCAL=1 FB_SKIP_A2A=1" fb_cufftmp 512 512 512 5 1; then
    STEP_MIN=12
    knob "NVSHMEM_IB_ENABLE_IBGDA=1"
    knob "CUFFT_RESHAPE_USE_PACKING=1 NVSHMEM_IB_ENABLE_IBGDA=1"
else
    echo "IBGDA probe failed: transport not usable on this system, skipping the IBGDA runs"
fi

cleanup
