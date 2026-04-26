# NavierStokes 3D — 编译与运行指南

## 环境信息

| 项目 | 说明 |
|---|---|
| 集群 | MSU-270，SLURM batch 分区 |
| 计算节点 | A100 SXM4 80GB × 8，CUDA 12.2 |
| 用户目录 | `/scratch/s02230673`（即 `$HOME`） |
| 代码目录 | `$HOME/guo/NavierStockes-3D/` |
| 容器镜像 | `$HOME/ns3d.sqsh` |

---

## 一、进入容器（编译前必须执行）

```bash
# 1. 申请计算节点
srun --partition=batch --nodes=1 --ntasks=4 \
     --cpus-per-task=4 --gres=gpu:4 \
     --time=02:00:00 --pty bash

# 2. 创建并进入容器
ulimit -n 65536
enroot create --name ns3d $HOME/ns3d.sqsh
enroot start --rw \
    --mount "/usr/local/lib/slurm:/usr/local/lib/slurm" \
    --mount "/scratch/s02230673:/home/user" \
    ns3d bash

# 3. 设置环境变量
export NVHPC=/home/user/.local/nvhpc
export NVHPC_VERSION=24.7
export PATH=$NVHPC/Linux_x86_64/$NVHPC_VERSION/compilers/bin:$PATH
export CUFFTMP_HOME=$NVHPC/Linux_x86_64/$NVHPC_VERSION/math_libs/12.5/targets/x86_64-linux
export NVSHMEM_HOME=$NVHPC/Linux_x86_64/$NVHPC_VERSION/comm_libs/12.5/nvshmem
export LD_LIBRARY_PATH=$CUFFTMP_HOME/lib:$NVSHMEM_HOME/lib:/home/user/.local/lib:$LD_LIBRARY_PATH

# 4. 进入代码目录
cd /home/user/guo/NavierStockes-3D
```

---

## 二、编译命令

### CPU 版本

```bash
# FFTW3
mpicxx -O3 -std=c++17 -fopenmp \
    NavierStokes_periodic_fftw.cpp \
    -lfftw3_threads -lfftw3 -lfftw3_mpi -lpthread -lm \
    -o ns_fftw

# AccFFT
mpicxx -O3 -std=c++17 -fopenmp \
    NavierStokes_periodic_accfft.cpp \
    -I/home/user/.local/include \
    -L/home/user/.local/lib \
    -laccfft -laccfft_utils \
    -lfftw3_threads -lfftw3 -lfftw3_mpi \
    -lmpi_cxx -lpthread -lm \
    -Xlinker -rpath,/home/user/.local/lib \
    -o ns_accfft

# P3DFFT
P3DFFT_HOME=/home/user/.local/p3dfft_v2
mpicxx -O3 -std=c++17 -fopenmp \
    NavierStokes_periodic_p3dfft.cpp \
    -I$P3DFFT_HOME/include \
    -I/home/user/.local/include \
    -L$P3DFFT_HOME/lib \
    -L/home/user/.local/lib \
    -lp3dfft \
    -lfftw3_threads -lfftw3 -lfftw3_mpi \
    -lfftw3f_threads -lfftw3f \
    -lmpi_cxx -lmpi_mpifh -lgfortran \
    -lpthread -lm \
    -Xlinker -rpath,$P3DFFT_HOME/lib \
    -Xlinker -rpath,/home/user/.local/lib \
    -o ns_p3dfft

# heFFTe v2 (CPU)
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I/home/user/.local/include \
    -I/usr/local/cuda-12.2/include \
    NavierStokes_periodic_heffte_v2.cpp \
    -L/home/user/.local/lib -lheffte \
    -L/usr/local/cuda-12.2/lib64 -lcufft -lcudart \
    -lfftw3 -lfftw3f -lm \
    -Xlinker -rpath,/home/user/.local/lib \
    -o ns_heffte_v2
```

### GPU 单节点版本

```bash
# cuFFT（单 GPU）
nvcc -O3 -std=c++17 -arch=sm_80 \
    NavierStokes_periodic_cufft.cu \
    -lcufft -lcudart -lm \
    -o ns_cufft

# cuFFT XT（多 GPU 单节点）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    NavierStokes_periodic_cufftxt.cu \
    -lcufft -lcudart -lm \
    -o ns_cufftxt

# heFFTe（单 GPU）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    -I/home/user/.local/include \
    NavierStokes_periodic_heffte_gpu1.cu \
    -L/home/user/.local/lib -lheffte \
    -L/usr/local/cuda-12.2/lib64 -lcufft -lcudart -lm \
    -Xlinker -rpath,/home/user/.local/lib \
    -o ns_heffte_gpu1

# heFFTe（多 GPU 单节点）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I/home/user/.local/include \
    NavierStokes_periodic_heffte_mgpu.cu \
    -L/home/user/.local/lib -lheffte \
    -L/usr/local/cuda-12.2/lib64 -lcufft -lcudart -lm \
    -Xlinker -rpath,/home/user/.local/lib \
    -o ns_heffte_mgpu
```

### GPU 多节点版本

```bash
# heFFTe 多节点（已验证，0.357s/step）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I/home/user/.local/include \
    NavierStokes_periodic_heffte_multigpu.cu \
    -L/home/user/.local/lib -lheffte \
    -L/usr/local/cuda-12.2/lib64 -lcufft -lcudart -lm \
    -Xlinker -rpath,/home/user/.local/lib \
    -o navier_stokes_heffte_multigpu_docker2

# cuFFTMp 单节点（❌ error 5，待修）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I$CUFFTMP_HOME/include/cufftmp \
    -I$CUFFTMP_HOME/include \
    -I$NVSHMEM_HOME/include \
    NavierStokes_periodic_cufftmp_mgpu.cu \
    -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib \
    -L/usr/local/cuda-12.2/lib64 \
    -lcufftMp -lnvshmem_host \
    $NVSHMEM_HOME/lib/libnvshmem.a \
    -lcufft -lcudart -lmpi -lm \
    -Xlinker -rpath,$CUFFTMP_HOME/lib \
    -Xlinker -rpath,$NVSHMEM_HOME/lib \
    -o navier_stokes_cufftmp_mgpu_docker

# cuFFTMp 多节点（❌ error 4，待修）
nvcc -O3 -std=c++17 -arch=sm_80 \
    -ccbin /usr/bin/mpicxx \
    -Xcompiler -fopenmp \
    -I$CUFFTMP_HOME/include/cufftmp \
    -I$CUFFTMP_HOME/include \
    -I$NVSHMEM_HOME/include \
    NavierStokes_periodic_cufftmp_multinode.cu \
    -L$CUFFTMP_HOME/lib -L$NVSHMEM_HOME/lib \
    -L/usr/local/cuda-12.2/lib64 \
    -lcufftMp -lnvshmem_host \
    $NVSHMEM_HOME/lib/libnvshmem.a \
    -lcufft -lcudart -lmpi -lm \
    -Xlinker -rpath,$CUFFTMP_HOME/lib \
    -Xlinker -rpath,$NVSHMEM_HOME/lib \
    -o navier_stokes_cufftmp_multinode_docker
```

---

## 三、运行方式

### 交互运行（计算节点上）

```bash
# CPU / 单 GPU
./ns_fftw
./ns_cufft

# 多进程 CPU
mpirun -np 4 ./ns_fftw
mpirun -np 4 ./ns_accfft
mpirun -np 4 ./ns_p3dfft
mpirun -np 4 ./ns_heffte_v2

# 多 GPU 单节点
mpirun -np 4 ./ns_cufftxt
mpirun -np 4 ./ns_heffte_mgpu
```

### sbatch 提交（多节点）

```bash
sbatch $HOME/run_multinode.sh        # heFFTe 多节点
sbatch $HOME/run_cufftmp_single.sh   # cuFFTMp 单节点
sbatch $HOME/run_cufftmp_multi.sh    # cuFFTMp 多节点
```

---

## 四、编译状态总览

| 可执行文件 | 源文件 | 后端 | 节点 | GPU | 状态 |
|---|---|---|---|---|---|
| `ns_fftw` | fftw.cpp | FFTW3 | 多 | 无 | ✅ |
| `ns_accfft` | accfft.cpp | AccFFT | 多 | 无 | ✅ |
| `ns_p3dfft` | p3dfft.cpp | P3DFFT | 多 | 无 | ✅ |
| `ns_heffte_v2` | heffte_v2.cpp | heFFTe | 多 | 无 | ✅ |
| `ns_cufft` | cufft.cu | cuFFT | 单 | 1 | ✅ |
| `ns_cufftxt` | cufftxt.cu | cuFFT XT | 单 | 多 | ✅ |
| `ns_heffte_gpu1` | heffte_gpu1.cu | heFFTe | 单 | 1 | ✅ |
| `ns_heffte_mgpu` | heffte_mgpu.cu | heFFTe | 单 | 多 | ✅ |
| `navier_stokes_heffte_multigpu_docker2` | heffte_multigpu.cu | heFFTe | 多 | 多 | ✅ 0.357s/step |
| `navier_stokes_cufftmp_mgpu_docker` | cufftmp_mgpu.cu | cuFFTMp | 单 | 多 | ❌ error 5 |
| `navier_stokes_cufftmp_multinode_docker` | cufftmp_multinode.cu | cuFFTMp | 多 | 多 | ❌ error 4 |
