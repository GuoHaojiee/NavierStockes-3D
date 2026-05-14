# NavierStokes 3D — 编译与运行指南

## 基本信息

| 项目 | 说明 |
|---|---|
| 集群 | MSU-270，SLURM batch 分区 |
| 计算节点 | A100 SXM4-80GB × 8，CUDA 12.2 |
| 用户目录 | `/scratch/s02230673`（即 `$HOME`） |
| 代码目录 | `$HOME/guo/NavierStockes-3D/` |
| 容器镜像 | `$HOME/ns3d_v2.sqsh`（当前生产版本） |

---

## 一、每次登录后的标准流程

### 1. 申请计算节点（必须带 GPU）

```bash
srun --partition=batch --nodes=1 --ntasks=4 \
     --cpus-per-task=8 --gres=gpu:4 \
     --time=02:00:00 --pty bash
```

### 2. 创建并进入容器

> enroot 实例是**节点本地**的，每次登录到新节点都需要重新创建。

```bash
ulimit -n 65536
enroot create --name ns3d /scratch/s02230673/ns3d_v2.sqsh
enroot start --rw \
    --mount "/usr/local/lib/slurm:/usr/local/lib/slurm" \
    --mount "/scratch/s02230673:/home/user" \
    ns3d bash -l
```

> `bash -l` 会自动 source `/etc/profile.d/ns3d_env.sh`，无需手动设置环境变量。

### 3. 验证环境

```bash
nvidia-smi | head -4          # 确认 GPU 可见
echo $MPI_HOME                # 应输出 NVHPC HPC-X 路径
echo $HEFFTE_DIR              # 应输出 /opt/heffte
nvcc --version | head -1      # 应输出 NVHPC 24.7
cd /home/user/guo/NavierStockes-3D
```

---

## 二、环境变量（容器内已自动加载）

容器内 `/etc/profile.d/ns3d_env.sh` 包含以下配置，**无需手动执行**：

```bash
export NVHPC=/home/user/.local/nvhpc
export NVHPC_VERSION=24.7
_NVHPC_ARCH=$NVHPC/Linux_x86_64/$NVHPC_VERSION

export CUFFTMP_HOME=$_NVHPC_ARCH/math_libs/12.5/targets/x86_64-linux
export NVSHMEM_HOME=$_NVHPC_ARCH/comm_libs/12.5/nvshmem
export MPI_HOME=$_NVHPC_ARCH/comm_libs/12.5/hpcx/hpcx-2.19/ompi
export OPAL_PREFIX=$MPI_HOME

export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_SYMMETRIC_SIZE=4294967296   # 4GB，cuFFTMp 必须
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export UCX_TLS=rc_verbs,ud_verbs,cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n

export FFTW3_DIR=/opt/fftw3
export HEFFTE_DIR=/opt/heffte
export ACCFFT_DIR=/opt/accfft
export P3DFFT_HOME=/opt/p3dfft_v2

export LD_LIBRARY_PATH=\
/opt/ucx-1.17/lib:\
/opt/fftw3/lib:\
/opt/heffte/lib:\
/opt/accfft/lib:\
/opt/p3dfft_v2/lib:\
$CUFFTMP_HOME/lib:\
$_NVHPC_ARCH/REDIST/math_libs/12.5/targets/x86_64-linux/lib:\
$NVSHMEM_HOME/lib:\
$MPI_HOME/lib:\
$_NVHPC_ARCH/compilers/lib:\
/home/user/.local/lib

export PATH=$_NVHPC_ARCH/compilers/bin:$MPI_HOME/bin:/opt/ucx-1.17/bin:$PATH
```

---

## 三、编译命令

所有编译在容器内、代码目录下执行。

### CPU 版本

```bash
# FFTW3（MPI 多进程）
$MPI_HOME/bin/mpicxx -O3 -std=c++17 -fopenmp \
    NavierStokes_periodic_fftw.cpp \
    -I/opt/fftw3/include -L/opt/fftw3/lib \
    -lfftw3_threads -lfftw3 -lfftw3_mpi -lpthread -lm \
    -Xlinker -rpath,/opt/fftw3/lib \
    -o ns_fftw

# AccFFT（MPI 多进程）
$MPI_HOME/bin/mpicxx -O3 -std=c++17 -fopenmp \
    NavierStokes_periodic_accfft.cpp \
    -I/opt/accfft/include -I/opt/fftw3/include \
    -L/opt/accfft/lib -L/opt/fftw3/lib \
    -laccfft -laccfft_utils \
    -lfftw3_threads -lfftw3 -lfftw3_mpi \
    -lfftw3f_threads -lfftw3f \
    -lmpi_cxx -lpthread -lm \
    -Xlinker -rpath,/opt/accfft/lib \
    -Xlinker -rpath,/opt/fftw3/lib \
    -o ns_accfft

# P3DFFT v2（MPI 多进程）
NVHPC_COMPLIB=$NVHPC/Linux_x86_64/$NVHPC_VERSION/compilers/lib
$MPI_HOME/bin/mpicxx -O3 -std=c++17 -fopenmp \
    NavierStokes_periodic_p3dfft.cpp \
    -I$P3DFFT_HOME/include -I/opt/fftw3/include \
    -L$P3DFFT_HOME/lib -L/opt/fftw3/lib -L$NVHPC_COMPLIB \
    -lp3dfft \
    -lfftw3_threads -lfftw3 -lfftw3_mpi \
    -lfftw3f_threads -lfftw3f \
    -lnvf -lnvhpcatm -lnvcpumath \
    -lmpi_cxx -lmpi_mpifh -lpthread -lm \
    -Xlinker -rpath,$P3DFFT_HOME/lib \
    -Xlinker -rpath,/opt/fftw3/lib \
    -Xlinker -rpath,$NVHPC_COMPLIB \
    -o ns_p3dfft

# heFFTe CPU（MPI 多进程）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin $MPI_HOME/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I$HEFFTE_DIR/include -I/opt/fftw3/include \
    NavierStokes_periodic_heffte_v2.cpp \
    -L$HEFFTE_DIR/lib -lheffte \
    -L/opt/fftw3/lib -lfftw3 -lfftw3f -lm \
    -Xlinker -rpath,$HEFFTE_DIR/lib \
    -Xlinker -rpath,/opt/fftw3/lib \
    -o ns_heffte_v2
```

### GPU 单节点版本

```bash
# cuFFT（单 GPU）
nvcc -O3 -std=c++17 -arch=sm_80 \
    NavierStokes_periodic_cufft.cu \
    -lcufft -lcudart -lm \
    -o ns_cufft

# cuFFT XT（多 GPU，无需 MPI）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin $MPI_HOME/bin/mpicxx \
    NavierStokes_periodic_cufftxt.cu \
    -lcufft -lcudart -lm \
    -o ns_cufftxt

# heFFTe（单 GPU）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin $MPI_HOME/bin/mpicxx \
    -I$HEFFTE_DIR/include \
    NavierStokes_periodic_heffte_gpu1.cu \
    -L$HEFFTE_DIR/lib -lheffte \
    -lcufft -lcudart -lm \
    -Xlinker -rpath,$HEFFTE_DIR/lib \
    -o ns_heffte_gpu1

# heFFTe（多 GPU 单节点）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin $MPI_HOME/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I$HEFFTE_DIR/include \
    NavierStokes_periodic_heffte_mgpu.cu \
    -L$HEFFTE_DIR/lib -lheffte \
    -lcufft -lcudart -lm \
    -Xlinker -rpath,$HEFFTE_DIR/lib \
    -o ns_heffte_mgpu
```

### GPU 多节点版本

```bash
# heFFTe 多节点
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin $MPI_HOME/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I$HEFFTE_DIR/include \
    NavierStokes_periodic_heffte_multigpu.cu \
    -L$HEFFTE_DIR/lib -lheffte \
    -lcufft -lcudart -lm \
    -Xlinker -rpath,$HEFFTE_DIR/lib \
    -o navier_stokes_heffte_multigpu_docker2

# cuFFTMp 单节点多 GPU
# ⚠️  只用 -lnvshmem_host，禁止同时链接 libnvshmem.a（会导致 error 5）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -I$CUFFTMP_HOME/include/cufftmp \
    -I$NVSHMEM_HOME/include \
    -I$MPI_HOME/include \
    -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib -L$MPI_HOME/lib \
    -lcufftMp -lnvshmem_host -lcufft -lcudart -lmpi -lm \
    -Xlinker -rpath,$CUFFTMP_HOME/lib \
    -Xlinker -rpath,$NVSHMEM_HOME/lib \
    NavierStokes_periodic_cufftmp_mgpu.cu \
    -o navier_stokes_cufftmp_mgpu_docker

# cuFFTMp 多节点
nvcc -O3 -std=c++17 -arch=sm_80 \
    -I$CUFFTMP_HOME/include/cufftmp \
    -I$NVSHMEM_HOME/include \
    -I$MPI_HOME/include \
    -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib -L$MPI_HOME/lib \
    -lcufftMp -lnvshmem_host -lcufft -lcudart -lmpi -lm \
    -Xlinker -rpath,$CUFFTMP_HOME/lib \
    -Xlinker -rpath,$NVSHMEM_HOME/lib \
    NavierStokes_periodic_cufftmp_multinode.cu \
    -o navier_stokes_cufftmp_multinode_docker
```

---

## 四、冒烟测试

编译完成后，在单节点交互环境（4 GPU）内执行以下测试，全部应输出 `L2 error ≈ 1.17e-03`。

### CPU 版本

```bash
mpirun -np 4 ./ns_fftw      32 32 32 1e-4 10 4
mpirun -np 4 ./ns_accfft    32 32 32 1e-4 10 4
mpirun -np 4 ./ns_p3dfft    32 32 32 1e-4 10 4
mpirun -np 4 ./ns_heffte_v2 32 32 32 1e-4 10 4
```

### GPU 单节点版本

```bash
./ns_cufft   32 32 32 1e-4 10        # 单 GPU
./ns_cufftxt 32 32 32 1e-4 10        # 多 GPU（自动使用节点所有 GPU）

mpirun -np 4 ./ns_heffte_gpu1 32 32 32 1e-4 10
mpirun -np 4 ./ns_heffte_mgpu 32 32 32 1e-4 10
mpirun -np 4 ./navier_stokes_cufftmp_mgpu_docker 32 32 32 1e-4 10
```

### GPU 多节点版本（需 sbatch）

```bash
sbatch $HOME/run_multinode.sh       # heFFTe 多节点
sbatch $HOME/run_cufftmp_multi.sh   # cuFFTMp 多节点
```

---

## 五、编译状态总览

| 可执行文件 | 源文件 | 后端 | 节点 | GPU | 状态 |
|---|---|---|---|---|---|
| `ns_fftw` | fftw.cpp | FFTW3 | 多 | 无 | ✅ |
| `ns_accfft` | accfft.cpp | AccFFT | 多 | 无 | ✅ |
| `ns_p3dfft` | p3dfft.cpp | P3DFFT v2 | 多 | 无 | ✅ |
| `ns_heffte_v2` | heffte_v2.cpp | heFFTe | 多 | 无 | ✅ |
| `ns_cufft` | cufft.cu | cuFFT | 单 | 1 | ✅ |
| `ns_cufftxt` | cufftxt.cu | cuFFT XT | 单 | 多 | ✅ |
| `ns_heffte_gpu1` | heffte_gpu1.cu | heFFTe | 单 | 1 | ✅ |
| `ns_heffte_mgpu` | heffte_mgpu.cu | heFFTe | 单 | 多 | ✅ 0.011s/step |
| `navier_stokes_heffte_multigpu_docker2` | heffte_multigpu.cu | heFFTe | 多 | 多 | ✅ 0.357s/step |
| `navier_stokes_cufftmp_mgpu_docker` | cufftmp_mgpu.cu | cuFFTMp | 单 | 多 | ✅ |
| `navier_stokes_cufftmp_multinode_docker` | cufftmp_multinode.cu | cuFFTMp | 多 | 多 | ✅ |

---

## 六、常见问题

**Q: 进入新节点后容器不存在？**
重新执行第一节的 `enroot create` 步骤，enroot 实例是节点本地的。

**Q: nvidia-smi 显示 "No devices were found"？**
当前 srun 没有申请 GPU（`--gres=gpu:4`），或 srun session 已过期。重新申请节点。

**Q: cuFFTMp 报 error 5（CUFFT_INTERNAL_ERROR）？**
编译时混入了 `libnvshmem.a` 静态库。只保留 `-lnvshmem_host`，删除所有 `libnvshmem.a` 引用后重新编译。

**Q: MPI 报 "not enough slots"？**
在没有 SLURM GPU 分配的节点内测试时，加 `--oversubscribe`。正式测试应通过 sbatch 提交。

**Q: UCX WARN cuda_copy/cuda_ipc not available？**
在无 GPU 的环境下运行时的正常提示，不影响 CPU 版本运行。有 GPU 时该警告消失。
