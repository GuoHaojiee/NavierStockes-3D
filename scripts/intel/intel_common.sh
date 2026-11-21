# intel_common.sh — sourced on the LOGIN node / in sbatch scripts (outside the container).
# Defines $M (GCC container, from ~/guohaojie/runs/common.sh), $MI (Intel container), precheck, cleanup_all.
# $MI is derived from $M by replacing ns3d_gcc with ns3d_intel (image and pyxis container name).
# If your $M does not contain "ns3d_gcc", set MI yourself before sourcing, e.g.
#   export MI="--container-image=/scratch/pr0208/guohaojie/ns3d_intel.sqsh --container-name=ns3d_intel --container-mounts=/scratch/pr0208/guohaojie:/work"
source ~/guohaojie/runs/common.sh          # $M, precheck, cleanup
if [ -z "$MI" ]; then
    MI=$(echo "$M" | sed 's/ns3d_gcc/ns3d_intel/g')
    if [ "$MI" = "$M" ]; then echo "intel_common.sh: \$M has no 'ns3d_gcc' ($M); set MI by hand (see the header of this file)"; MI=""; fi
fi
SCR=/scratch/pr0208/guohaojie
# INTEL_IMAGE=<.sqsh path> or v2 (= $SCR/containers/ns3d_intel_v2.sqsh, made by rebuild_container.sh): use another
# Intel image. The pyxis container name becomes the image basename, so a container of the old image is not reused.
INTEL_CNAME=ns3d_intel
if [ -n "$INTEL_IMAGE" ]; then
    [ "$INTEL_IMAGE" = v2 ] && INTEL_IMAGE=$SCR/containers/ns3d_intel_v2.sqsh
    [ -f "$INTEL_IMAGE" ] || echo "intel_common.sh: WARNING: image $INTEL_IMAGE not found"
    INTEL_CNAME=$(basename "$INTEL_IMAGE" .sqsh)
    MI=$(echo "$MI" | sed -e "s|--container-image=[^ ]*|--container-image=$INTEL_IMAGE|" -e "s|--container-name=[^ ]*|--container-name=$INTEL_CNAME|")
fi
export MI INTEL_IMAGE INTEL_CNAME
EXCL=cn34,cn18,cn25
# remove the pyxis containers of this job on every node (GCC via common.sh's cleanup, Intel here)
cleanup_all() {
    cleanup 2>/dev/null
    srun --ntasks-per-node=1 --time=2 bash -c "for n in pyxis_ns3d_intel pyxis_$INTEL_CNAME; do enroot remove -f \$n >/dev/null 2>&1; done; true" 2>/dev/null
}
# irun_mn <nodes> <ranks per node> <threads per rank> <command...>
#   Multi-node Intel MPI run in the Intel image ($MI; use INTEL_IMAGE=v2), settings of intel_mn_env.sh
#   (hello_2node E6d_a2a3: srun --mpi=pmi2 + libpmi2.so.0, shm:ofi, mlx, HPC-X UCX 1.17, I_MPI_ADJUST_ALLTOALL=3).
#   The command runs on every rank after intel_mn_env.sh, with OMP_NUM_THREADS=<threads>, OMP_PLACES=cores,
#   OMP_PROC_BIND=close; srun binds each rank to <threads> cores (--cpu-bind=cores).
#   Environment passed on: IMN_ALLTOALL IMN_ALLTOALLV IMN_UCX_DIR IMN_NET_DEVICES IMN_PIN IMN_DEBUG IMN_PMI2_LIB,
#   IRUN_SRUN_OPTS (extra srun options, e.g. "--time=20"), IMN_SHOWBIND=1 (ranks 0,1 print their cpu list).
#   Example:  irun_mn 4 128 1 "exec /work/build/bin_intel/ns_fftw 512 512 512 1e-6 20 1 bench"
irun_mn() {
    local nn=$1 ppn=$2 nt=$3; shift 3
    local pass="" v
    for v in IMN_ALLTOALL IMN_ALLTOALLV IMN_UCX_DIR IMN_NET_DEVICES IMN_PIN IMN_DEBUG IMN_PMI2_LIB; do
        [ -n "${!v+x}" ] && pass="$pass $v='${!v}'"
    done
    local bind=""
    [ -n "$IMN_SHOWBIND" ] && bind="[ \${SLURM_PROCID:-0} -lt 2 ] && echo \"BIND rank=\$SLURM_PROCID host=\$(hostname) \$(grep Cpus_allowed_list /proc/self/status)\";"
    srun --mpi=pmi2 -N $nn --ntasks-per-node=$ppn -c $nt --cpu-bind=cores $IRUN_SRUN_OPTS $MI bash -c \
        "export OMP_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close $pass; \
         source /work/src/NavierStockes-3D/scripts/intel/intel_mn_env.sh; $bind $*"
}
echo "M  = $M"
echo "MI = $MI"
[ -n "$INTEL_IMAGE" ] && echo "   (INTEL_IMAGE=$INTEL_IMAGE)"
