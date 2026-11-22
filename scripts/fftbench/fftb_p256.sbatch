#!/bin/bash
#SBATCH -J fftb_p256
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 03:00:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/fftb_%x_%j.out
# fftb_p256 — the p = 256 question (FFTW and P3DFFT slow with Intel MPI at 2 nodes x 128 ranks) with the pure FFT
#   benchmark instead of the solver: 2 nodes, 128x1, 512^3 and 1024^3, FFTW and P3DFFT, slab, MEASURE.
#   The same 7 configurations as p256_diag.sbatch (same names, so fftb_parse.sh can put the solver numbers of
#   p256_diag.csv next to these):
#     i_a_base      Intel E6d with I_MPI_ADJUST_ALLTOALL not set
#     i_b_rc        + IMN_UCX_TLS=rc_mlx5,sm,self
#     i_c_dc        + IMN_UCX_TLS=dc_mlx5,sm,self
#     i_d_rndvinf   + IMN_EXTRA=UCX_RNDV_THRESH=inf
#     i_e_rndv4k    + IMN_EXTRA=UCX_RNDV_THRESH=4096
#     i_f_pinintel  + IMN_PIN=intel (Intel MPI pins, srun does not bind)
#     g_def         GCC + Open MPI default (coll ucc)
#   Order: grid, library, configuration (the Intel variants and GCC alternate within each library).
#   One launch per (grid, library, configuration) = 2 x 2 x 7 = 28 launches; inside each: plan, check, warm-up,
#   3 repetitions of >= 3 s (fftbench.cpp).
# Time estimate (per launch: container start ~0.5 min + MEASURE planning + check/warm-up + 3 x >= 3 s):
#   512^3   ~1-2 min  (14 launches: 15-30 min; the slow p=256 Intel cases have ~0.2 s per FFT)
#   1024^3  ~3-5 min  (14 launches: 45-70 min; MEASURE planning of a 1024^3 slab FFT dominates)
#   total about 1-1.7 h; -t 03:00:00 leaves room for timeouts (limit per launch 3 x expected: 6 min / 18 min).
# Results: ~/guohaojie/results/fftb.csv (settings = configuration name + variables); log fftb_fftb_p256_<job>.out
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/fftbench && sbatch fftb_p256.sbatch
#   FFTB_SOLVER_CSV=~/guohaojie/results/p256_diag.csv FFTB_JOBS=<jobid> bash fftb_parse.sh
# Options: FFTB_P256_GRIDS ("512 1024"), FFTB_LIBS ("fftw p3dfft"), FFTB_REP, FFTB_SPLIT=1 (also the local FFT time).
export FFTB_LIBS=${FFTB_LIBS:-"fftw p3dfft"}
export FFTB_CONFIGS="
i_a_base|intel|128x1|IMN_ALLTOALL=
i_b_rc|intel|128x1|IMN_ALLTOALL= IMN_UCX_TLS=rc_mlx5,sm,self
i_c_dc|intel|128x1|IMN_ALLTOALL= IMN_UCX_TLS=dc_mlx5,sm,self
i_d_rndvinf|intel|128x1|IMN_ALLTOALL= IMN_EXTRA=UCX_RNDV_THRESH=inf
i_e_rndv4k|intel|128x1|IMN_ALLTOALL= IMN_EXTRA=UCX_RNDV_THRESH=4096
i_f_pinintel|intel|128x1|IMN_ALLTOALL= IMN_PIN=intel
g_def|gcc|128x1|GCC_COLL=def
"
exec bash ~/guohaojie/src/NavierStockes-3D/scripts/fftbench/fftb.sbatch ${SLURM_JOB_NUM_NODES:-2} "${FFTB_P256_GRIDS:-512 1024}"
