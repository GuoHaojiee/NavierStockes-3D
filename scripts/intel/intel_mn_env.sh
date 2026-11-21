# intel_mn_env.sh — sourced INSIDE the Intel v2 image (ns3d_intel_v2.sqsh) on every rank of a multi-node Intel MPI
# job that srun starts with --mpi=pmi2 (see irun_mn in intel_common.sh). It is the setting that worked best in
# hello_2node.sbatch, E6d_a2a3: 2.72 / 0.96 / 0.93 x Open MPI off-node bandwidth at 8 KB / 64 KB / 256 KB.
#   PMI       srun --mpi=pmi2, I_MPI_PMI=pmi2, I_MPI_PMI_LIBRARY = Slurm's libpmi2.so.0 of the image
#   fabric    I_MPI_FABRICS=shm:ofi, FI_PROVIDER=mlx, FI_PROVIDER_PATH = Intel's libfabric providers only
#   UCX       HPC-X 2.19 UCX 1.17 (stage_hpcx_ucx.sh -> /work/opt_ucx/hpcx) first in LD_LIBRARY_PATH, UCX_MODULE_DIR,
#             UCX_TLS=rc_mlx5,ud_mlx5,sm,self, UCX_NET_DEVICES = the 8 HCAs
#   alltoall  I_MPI_ADJUST_ALLTOALL=3 (pairwise exchange)
# Switches (set them before sourcing, or pass them to irun_mn's environment):
#   IMN_ALLTOALL   (3)      I_MPI_ADJUST_ALLTOALL; "" = Intel MPI default
#   IMN_ALLTOALLV  (unset)  I_MPI_ADJUST_ALLTOALLV, 1 or 2 (heFFTe and AccFFT use MPI_Alltoallv)
#   IMN_UCX_DIR    (/work/opt_ucx/hpcx; "system" = the image's UCX 1.12.1)
#   IMN_NET_DEVICES (mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1)
#   IMN_PIN        (slurm)  slurm: srun --cpu-bind binds the ranks, I_MPI_PIN=0 (same binding as the Open MPI runs);
#                           intel: Intel MPI pins, I_MPI_PIN_DOMAIN=<threads>:compact (needs OMP_NUM_THREADS)
#   IMN_DEBUG      (unset)  I_MPI_DEBUG level (4: pin map and libfabric provider)
#   IMN_PMI2_LIB   (/usr/lib/x86_64-linux-gnu/libpmi2.so.0)
source /work/src/NavierStockes-3D/scripts/intel/intel_env.sh >/dev/null 2>&1

export I_MPI_PMI=pmi2 I_MPI_PMI_LIBRARY=${IMN_PMI2_LIB:-/usr/lib/x86_64-linux-gnu/libpmi2.so.0}
unset I_MPI_HYDRA_BOOTSTRAP
export I_MPI_FABRICS=shm:ofi FI_PROVIDER=mlx FI_PROVIDER_PATH=$I_MPI_ROOT/opt/mpi/libfabric/lib/prov
export UCX_TLS=rc_mlx5,ud_mlx5,sm,self
export UCX_NET_DEVICES=${IMN_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1}

IMN_UCX_DIR=${IMN_UCX_DIR:-/work/opt_ucx/hpcx}
if [ "$IMN_UCX_DIR" != system ]; then
    if [ -f $IMN_UCX_DIR/lib/libucp.so.0 ]; then
        export LD_LIBRARY_PATH=$IMN_UCX_DIR/lib:$LD_LIBRARY_PATH UCX_MODULE_DIR=$IMN_UCX_DIR/lib/ucx
    else
        [ "${SLURM_PROCID:-0}" = 0 ] && echo "intel_mn_env.sh: WARNING: no $IMN_UCX_DIR/lib/libucp.so.0 (run stage_hpcx_ucx.sh); using the image's UCX 1.12.1" >&2
    fi
fi

IMN_ALLTOALL=${IMN_ALLTOALL-3}
if [ -n "$IMN_ALLTOALL" ]; then export I_MPI_ADJUST_ALLTOALL=$IMN_ALLTOALL; else unset I_MPI_ADJUST_ALLTOALL; fi
case "${IMN_ALLTOALLV:-}" in
    "") unset I_MPI_ADJUST_ALLTOALLV ;;
    1|2) export I_MPI_ADJUST_ALLTOALLV=$IMN_ALLTOALLV ;;
    *) [ "${SLURM_PROCID:-0}" = 0 ] && echo "intel_mn_env.sh: WARNING: IMN_ALLTOALLV=$IMN_ALLTOALLV ignored (1 or 2)" >&2; unset I_MPI_ADJUST_ALLTOALLV ;;
esac

if [ "${IMN_PIN:-slurm}" = intel ]; then
    export I_MPI_PIN=1 I_MPI_PIN_DOMAIN=${OMP_NUM_THREADS:-1}:compact
else
    export I_MPI_PIN=0
fi
[ -n "$IMN_DEBUG" ] && export I_MPI_DEBUG=$IMN_DEBUG

# one line on rank 0, so every log shows what was used
if [ "${SLURM_PROCID:-0}" = 0 ]; then
    echo "intel_mn_env: PMI=$I_MPI_PMI ($I_MPI_PMI_LIBRARY) FABRICS=$I_MPI_FABRICS FI_PROVIDER=$FI_PROVIDER UCX=${UCX_MODULE_DIR:-image 1.12.1} UCX_TLS=$UCX_TLS ALLTOALL=${I_MPI_ADJUST_ALLTOALL:-default} ALLTOALLV=${I_MPI_ADJUST_ALLTOALLV:-default} PIN=${IMN_PIN:-slurm}"
fi
