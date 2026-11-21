#!/bin/bash
#SBATCH -J ct4fd
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH -c 1
#SBATCH --exclusive
#SBATCH --exclude=cn34,cn18,cn25
#SBATCH -t 00:50:00
#SBATCH -o /scratch/pr0208/guohaojie/logs/ct4_%x_%j.out
# ct4_fftw_diag — why is FFTW Intel 64x2 on 4 nodes slow and unstable (2.83-4.68 s in ct4_512)?
#   FFTW only, 512^3, 10 steps, 4 nodes; gcc and Intel x 128x1 / 64x2; the Intel side twice:
#   I_MPI_ADJUST_ALLTOALL=3 (intel-a3, the E6d setting) and the Intel MPI default (intel-def); 3 repetitions;
#   NS_PRINT_PLAN=1: rank 0 prints the forward and backward FFTW plans, i.e. the MPI transpose FFTW chose
#   (mpi-transpose-alltoall / -pairwise / -recurse), which can differ between runs because MEASURE times them.
#   18 runs x 1-1.5 min -> ~20-30 min.
#   It is cmp_toolchain_4n.sbatch 512 with CT4_LIBS=fftw CT4_REP=3 CT4_INTEL_A2A="3 def" CT4_PRINT_PLAN=1;
#   OMPI_MCA_* / CT4_GCC_ENV pass through as there.
#   cd ~/guohaojie/src/NavierStockes-3D/scripts/intel && sbatch ct4_fftw_diag.sbatch
#   bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4fd_*.out | head -1)
#   (the last table of ct4_parse.sh: one line per run, step time next to the transposes of the plans)
export CT4_LIBS=fftw CT4_REP=${CT4_REP:-3} CT4_INTEL_A2A=${CT4_INTEL_A2A:-"3 def"} CT4_PRINT_PLAN=1
exec bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/cmp_toolchain_4n.sbatch 512
