# Intel 工具链对比（icpx + Intel MPI vs GCC/NVHPC + Open MPI）

目的：给"CPU 版统一用 Intel 编译器 + Intel MPI"提供依据。已知：编译器本身对速度影响 <1%；Intel MPI 让
FFTW 快约 20%，AccFFT 无差别。这里补上 P3DFFT 和 heFFTe，并比较 heFFTe 的两个本地后端
（FFTW / MKL；导师的规则：差别不大用 FFTW，MKL 明显更快就用 MKL）。

| 文件 | 在哪里运行 | 作用 |
|---|---|---|
| `intel_common.sh` | 登录节点 / sbatch | 由 `~/guohaojie/runs/common.sh` 的 `$M` 推出 `$MI`（`ns3d_gcc` → `ns3d_intel`），`cleanup_all` |
| `intel_env.sh` | Intel 容器内 | 环境文件、路径（`/work/opt_intel`、`/work/build/bin_intel`）、MKL 链接参数、`c()` |
| `check_env.sh` | Intel 容器内 | icpx/ifx/mpiifx 版本、MKL、cmake、PMI 库、libfabric provider |
| `build_intel_libs.sh` | Intel 容器内 | P3DFFT 2.x（ifx，MEASURE，无 stride1），heFFTe 2.4.1 ×2（FFTW / MKL 后端） |
| `build_intel_solvers.sh` | Intel 容器内 | 4 个求解器（heFFTe 两个后端）、`fb_backend_{fftw,mkl}`、`hello_a2a` |
| `hello_2node.sbatch` + `hello_a2a.c` | 2 节点 | Intel MPI 多节点启动方式（含宿主机 libpmix）+ fabric 扫描；Open MPI 参照 |
| `rebuild_container.sh` + `rebuild_inside.sh` | 计算节点（宿主机，不在容器里） | 在旧镜像基础上用 apt 装 rdma-core/libibverbs、UCX、libpmi2-0、libpmix2，生成 `containers/ns3d_intel_v2.sqsh` |
| `check_fabric.sbatch` | 1 节点 | 新镜像里的 ibv_devinfo、ucx_info、`fi_info -p mlx`、`fi_info -p verbs` |
| `cmp_toolchain.sbatch` + `ct_parse.sh` | 1 节点 | 每个库 gcc vs intel，512³，128×1 / 64×2，各 3 次 |
| `backend_1proc.sbatch` + `backend_parse.sh` | 1 节点 | 单进程单线程：heFFTe-FFTW / heFFTe-MKL / 原生 FFTW / 原生 MKL |

heFFTe 后端：`NavierStokes_periodic_heffte_v2.cpp` 用 `-DNS_HEFFTE_BACKEND=fftw|mkl` 在编译时选择，
启动时打印 `heFFTe backend: ...`。heFFTe 2.4.1 的默认 `use_reorder` 两个后端不同（fftw 1，mkl 0），
所以 `cmp_toolchain` 对 MKL 跑两行：库默认（`intelmkl`）和 `NS_HEFFTE_REORDER=1`（`intelmklr1`，与 FFTW 后端同样的选项）。

编译选项与 `scripts/compile_all.sh` 相同（`-O3 -std=c++17 -fopenmp -fno-math-errno`），另加
`-fp-model=precise`（icpx 默认 fast；本地 64³ 两种都得到参考值，但 512³ 没验证过）。用 icpx 默认：
`NS_ICPX_FP= bash build_intel_solvers.sh`。Intel 版 P3DFFT 和 GCC 版一样链接 `-lfftw3_threads`。

## 本地测过的部分（oneAPI 2026.1，Intel MPI 2021.18，MKL 2026.1，icx 编的 FFTW 3.3.10；没有 ifx）

- heFFTe 2.4.1 的 FFTW 后端、MKL 后端都用 `build_intel_libs.sh` 编通（MKL：`mkl_intel_ilp64 + mkl_intel_thread + iomp5`）。
- `ns_fftw`、`ns_heffte_v2_fftw`、`ns_heffte_v2_mkl`，4 进程 × 2 线程，slab：64³ verify L2 = 6.0041e-10，
  bench 动能 = 3.3526466678e+05（与参考值相同）；MKL 加 `NS_HEFFTE_REORDER=1` 也一样。
- `fb_backend_*`：往返误差 ~1e-15，Parseval 检查通过；`fftw_*` 符号确实来自 libfftw3，没有被 MKL 自带的 FFTW3 包装函数顶替（`[libs]` 行）。
- `hello_a2a`：单节点 8 进程通过。
- 没测：P3DFFT（本机没有 ifx）、AccFFT、所有 Slurm/容器部分。

## 按顺序运行

```bash
cd ~/guohaojie/src/NavierStockes-3D && git pull --ff-only
source scripts/intel/intel_common.sh          # 打印 M= 和 MI=；MI 应当是 M 把 ns3d_gcc 换成 ns3d_intel
```

**1. 环境检查**

```bash
srun -p batch --exclude=$EXCL -N1 -n1 -c4 --time=00:10:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/check_env.sh
```
应当看到：第一段列出设置 `I_MPI_CXX` 的环境文件（如果不是 `/etc/profile.d/ns3d_env.sh`，以后所有命令前加
`INTEL_ENV_FILE=<那个文件>`，sbatch 用 `export INTEL_ENV_FILE=...` 之后再提交）；`icpx 2025.3.2`；`ifx` 有版本号；
`mpiifx -show` 以 `ifx` 开头；`mpicxx -show` 以 `icpx` 开头；MKL 一段有 `mkl_dfti.h ... WILL be built`；
`cmake ≥ 3.19`；`autoreconf`、`git` 存在；`/opt/fftw3/lib` 里有 `libfftw3.so libfftw3f.so libfftw3_omp.so libfftw3_threads.so libfftw3_mpi.so`；
PMI 一段列出 `libpmi2.so*` / `libpmix.so*` 的位置。把整段输出发给我。

**2. 编库**（约 15–25 分钟）

```bash
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=01:00:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/build_intel_libs.sh
```
应当看到 `OK   lib_p3dfft`、`OK   lib_heffte_fftw`、`OK   lib_heffte_mkl`（没有 MKL 时是 `SKIP heffte_mkl: ...`）。
`lib_p3dfft.log` 里应有 `#define INTEL 1`、`#define MEASURE 1`、`#define OPENMP 1`，没有 `STRIDE1`。
单独重编一个：`... $MI env IL_ONLY="p3dfft" bash .../build_intel_libs.sh`。

**3. 编求解器**

```bash
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $MI bash /work/src/NavierStockes-3D/scripts/intel/build_intel_solvers.sh
```
应当看到 8 个 `OK`：`ns_fftw ns_accfft ns_p3dfft ns_heffte_v2_fftw fb_backend_fftw ns_heffte_v2_mkl fb_backend_mkl hello_a2a`。
`ns_p3dfft` 如果第一次链接失败会自动用 `mpiifx -nofor-main` 重试（第一次的日志留在 `ns_p3dfft.try1.log`）。

**4. 多节点启动**（2 节点，约 15–40 分钟）

```bash
sbatch scripts/intel/hello_2node.sbatch
sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
```
每个消息大小（每对进程 8192、65536、262144 字节）单独一次 srun；最小的失败，这个方式更大的就跳过。
`RESULT variant=R_ompi_pmix bytes=... status=OK` 是 Open MPI 参照。Intel MPI 的方式：
`A1_pmi2`（srun --mpi=pmi2，Intel MPI 自带的 PMI-2 客户端；日志里 `PMI API:` 应为 pmi2）、`A1_pmi2_ppn2`（每节点 2 个进程）、
`A1_ofi_only`（`I_MPI_FABRICS=ofi`，不用共享内存）、`B1_pmix_cont*`（容器里的 libpmix）、
`B2_pmix_host_{mlx,verbs}`（宿主机 PMIx 3.2.3：`/opt/pmix-3.2.3` 原路径挂进容器，容器里缺的依赖拷到
`logs/hello2_<job>.d/hostpmix/` 挂到 `/hostpmix`；日志里 `pass N: missing in the container: ...` 和 `staged in /hostpmix:`
说明拷了什么）、`C_hydra_slurm`。失败的每次 srun 在 `logs/hello2_<job>.d/<variant>.<bytes>.txt` 里留下
MPI startup/PMI 行和最后 20 行错误（完整输出在 `.raw`）。
任何一个三种大小都 `status=OK`，就说明 Intel MPI 能跨节点；`F_*` 用第一个成功的方式扫 FI_PROVIDER。
summary 最后的表：每个成功的方式、每个大小的 alltoall 时间（µs）/ 每节点跨节点带宽（GB/s），和 `R_ompi_pmix` 那行比。
日志里 `MPI startup(): libfabric provider: ...` 说明实际用了哪个 provider。
全部 `FAIL`：多节点 CPU 用 Open MPI（理由：Intel MPI 只对 FFTW 有收益），把日志发给我。

**5. 工具链对比**（4 个作业并行，4 个节点，每个约 40–80 分钟）

```bash
cd scripts/intel
for l in fftw accfft p3dfft heffte; do sbatch -J ct_$l cmp_toolchain.sbatch $l; done
for l in fftw accfft p3dfft heffte; do bash ct_parse.sh $(ls -t ~/guohaojie/logs/ct_ct_${l}_*.out | head -1); done
```
应当看到：每个配置 `runs 3`，`energy ok / failed` 为 `3 / 0`（动能 1.7933833444e+08）；最后一段每一行都是
`decomposition: slab`；第二张表的 `vs gcc` 列是相对 GCC 的倍数（<1 = Intel 更快）。日志开头 `MPI startup(): ... {0}` 几行是
第一次 Intel 运行的绑核（rank 0 → `{0}`，128×1 时 rank 1 → `{1}`）。
`bin_gcc` 必须是 120dfd1（NS_DECOMP）之后编的——日志开头打印了 `bin_gcc` 可执行文件的日期；否则先运行 `scripts/compile_all.sh`。

**6. 后端单进程测试**（约 20–30 分钟）

```bash
sbatch backend_1proc.sbatch
bash backend_parse.sh $(ls -t ~/guohaojie/logs/be1_*.out | head -1)
```
应当看到：没有 `FAIL` 行；`[libs] fftw_execute from /opt/fftw3/lib/libfftw3.so.3`，`DftiComputeForward from .../libmkl_intel_ilp64.so.*`；
每个 N 有 8 行 + 2 行 `_r2c0`；两个二进制里的原生 FFTW / MKL 行差别应当在几个百分点以内；最后是比值
`heFFTe-MKL / heFFTe-FFTW`（导师要的数）和 `native MKL / FFTW MEASURE`。
本地（笔记本 CPU，128³，仅供参考）：heFFTe-MKL / heFFTe-FFTW ≈ 0.5，native MKL / FFTW ≈ 1.07，
heFFTe-FFTW 比原生 FFTW 慢 ~3.8 倍——集群上的数字以 EPYC 为准。

## Intel MPI 跨节点：新镜像 ns3d_intel_v2

### 为什么要新镜像

旧镜像不能跨节点，有三个原因：

- 容器里没有 libpmi2.so。设了 `I_MPI_PMI=pmi2`，Intel MPI 仍然用 PMI-1（日志里是 `PMI API: pmi`）。
- 挂载宿主机的 PMIx 3.2.3 后，libpmix.so.2 加载不了。
- 没有 UCX 和 libibverbs，所以 `fi_getinfo()` 失败，mlx 和 verbs 两种 provider 都用不了。

`rebuild_container.sh` 以旧镜像为基础（不重装 oneAPI），用 apt 安装以下包：

- `rdma-core libibverbs1 ibverbs-providers ibverbs-utils librdmacm1`
- `libucx0 ucx-utils`
- `libpmi2-0`：Slurm 21.08 的 PMI-2 客户端，提供 `libpmi2.so.0`
- `libpmix2`：PMIx 4.1.2，提供 `libpmix.so.2`

装完写 `/opt/ns3d_pmi_libs.txt`，里面是这两个库的路径；`hello_2node.sbatch` 的 D 组读它。旧镜像不动。

### 本机测过的部分

- 在本机 Docker 的 Ubuntu 22.04 镜像里跑了 `rebuild_inside.sh`：9 个包都能装上，4 个库都能找到，`/opt/ns3d_pmi_libs.txt` 也写出来了。
- `rebuild_container.sh` 的保护检查都测过：旧镜像不存在、输出等于输入、新镜像已存在、没有 enroot，这几种情况都会报 FAIL 并停下。
- `hello_2node.sbatch v2` 用假 srun 跑通了：镜像切换、manifest 读取、D 组、PMI API 检查、分组选择。
- `check_fabric` 的检查脚本在本机跑通了流程。本机没有 InfiniBand，所以 4 项都是 FAIL。

**没测**：enroot 和 pyxis 生成镜像（需要计算节点能上网用 apt）、真实的 PMI-2/PMIx 握手、InfiniBand。

### 运行顺序

**1. 生成新镜像**（约 15–30 分钟）

在计算节点的宿主机上运行，不在容器里：

```bash
srun -p batch --exclude=cn34,cn18,cn25 -N1 -n1 -c16 --time=01:00:00 bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/rebuild_container.sh
```

成功时应看到：

- 每个包一行版本号；没有的包打印 `SKIP <包>: ...`；
- `OK libpmi2.so … libpmi2.so.0`、`OK libpmix.so … libpmix.so.2`、`OK libibverbs.so …`、`OK libucp.so …`；
- `OK /etc/profile.d/ns3d_intel.sh still there`；
- 最后一行 `OK /scratch/…/containers/ns3d_intel_v2.sqsh (N min)`。

日志在 `~/guohaojie/logs/rebuild_container_*.log`。如果 enroot 失败（比如节点本地空间不够），从登录节点改用 pyxis 方法：`RC_METHOD=pyxis bash …/rebuild_container.sh`。

**2. 检查网络栈**（约 2 分钟）

```bash
sbatch ~/guohaojie/src/NavierStockes-3D/scripts/intel/check_fabric.sbatch
grep -E "^CHECK|^image|^manifest" $(ls -t ~/guohaojie/logs/cfab_*.out | head -1)
```

4 行都应是 `OK`：

- `CHECK ibv_devinfo OK`：列出 `hca_id: mlx5_*`；
- `CHECK ucx_transports OK`：有 rc、dc、ud 等传输；
- `CHECK fi_info_mlx OK`；
- `CHECK fi_info_verbs OK`。

加参数 `old` 可以看旧镜像的对比结果，那边应该全是 FAIL。

**3. 跨节点**（约 10–20 分钟）

```bash
sbatch --export=ALL,HELLO_GROUPS="R D F" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
```

不加 `HELLO_GROUPS` 时会跑全部组（R A B D C F）。

成功时应看到：

- `D variants: pmi2 lib /usr/lib/x86_64-linux-gnu/libpmi2.so.0   pmix lib /usr/lib/x86_64-linux-gnu/libpmix.so.2`；
- `R_ompi_pmix` 三个大小都是 OK；
- D1 或 D2 至少有一个三个大小都是 `status=OK`，并且 `pmi_api=pmi2`（D1）或 `pmi_api=pmix`（D2），没有 `PMI_API_MISMATCH`；
- summary 表里 D 和 F 行的 `GBs_node_offnode` 和 `R_ompi_pmix` 同一个数量级。

如果失败：

- 看 `logs/hello2_<job>.d/D*_*.8192.txt`：里面有 MPI startup/PMI 行和最后 20 行错误；
- D3（每节点 2 个进程）成功而 D1 失败：问题和每节点进程数有关；
- D2 失败而 D1 成功：多半是 PMIx 4.1.2 客户端和 Slurm 的 PMIx 3.2.3 服务端不兼容。

### E 组：D1 的启动方式 + 只用 Intel 自带的 libfabric provider

D1 在 v2 镜像上已经用上了 PMI-2（日志里 `PMI API: pmi2`，共享内存的错误也没有了），但网络这一层还不通：

- verbs：`fi_getinfo() failed: No data available`；
- mlx：`get_ep_names: OFI get endpoint name failed: Invalid argument`。

E 组用和 D1 一样的启动方式：`srun --mpi=pmi2`，`I_MPI_PMI=pmi2`，`I_MPI_PMI_LIBRARY=/usr/lib/x86_64-linux-gnu/libpmi2.so.0`，`I_MPI_FABRICS=shm:ofi`。另外：

- `FI_PROVIDER_PATH` 只留 `$I_MPI_ROOT/opt/mpi/libfabric/lib/prov`，去掉系统的 libfabric 目录；
- `FI_LOG_LEVEL=warn`，失败时 `<variant>.8192.txt` 里有去掉 pid 并计数后的 libfabric 日志。

四个变体：

| 变体 | FI_PROVIDER | 说明 |
|---|---|---|
| E1_verbs_rxm | `verbs;ofi_rxm` | |
| E2_psm3 | `psm3` | Intel 自带的 `libpsm3-fi.so`，走 verbs 设备 |
| E3_mlx_ucxtls | `mlx` | 另设 `UCX_TLS=rc,ud,sm,self`、`UCX_NET_DEVICES=mlx5_2:1`（GPU0 对应的网卡，可用 `E3_NET_DEVICES=...` 改）；rank 0 打印 `ucx_info -v` |
| E4_tcp | `tcp` | 诊断用：如果只有它能通，说明启动和 PMI 都没问题，只剩高速网络这一层 |

运行：

```bash
sbatch --export=ALL,HELLO_GROUPS="R E" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
```

rank 0 会先打印一行 `FI_PROVIDER_PATH=…/libfabric/lib/prov: libmlx-fi.so libpsm3-fi.so librxm-fi.so libtcp-fi.so libverbs-…`，确认 provider 目录。

### E5 组：mlx 能通，但带宽太低

E 组的结果：

| 变体 | 结果 | 262144 字节时每节点带宽 |
|---|---|---|
| E3_mlx_ucxtls | 跨节点成功，`pmi_api=pmi2` | 2.19 GB/s |
| E4_tcp | 成功 | 6.47 GB/s |
| E1_verbs_rxm | 失败（rc=137） | — |
| E2_psm3 | 失败（rc=143） | — |
| Open MPI（参照） | 成功 | 30.75 GB/s |

E3 能跨节点，但带宽只有 Open MPI 的 1/14，比 tcp 还低。

E5 组的启动方式和 E3 完全一样（D1 的 pmi2 启动，`FI_PROVIDER=mlx`，只用 Intel 自带的 provider 目录），只改 UCX 设置：

| 变体 | UCX_TLS | UCX_NET_DEVICES |
|---|---|---|
| E5a_mlx5tls | `rc_mlx5,ud_mlx5,sm,self`（和 Open MPI 的 run_env.sh CPU 模式相同） | 不设 |
| E5b_dc | `dc_mlx5,sm,self` | 不设 |
| E5c_default | 不设（UCX 默认） | 不设 |
| E5d_alldev | `rc_mlx5,ud_mlx5,sm,self` | `mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1`（可用 `E5D_NET_DEVICES=...` 改） |

每个变体都设 `UCX_LOG_LEVEL=info`，rank 0 打印 `ucx_info -v`。不管成功还是失败，每次运行都会写 `<variant>.<bytes>.txt`，里面有：

- ucx_info 的版本；
- UCX 的 INFO 行（去掉时间、主机、pid 后计数），也就是 `ep_cfg[...]: tag(rc_mlx5/mlx5_2:1 ...)`，即实际选中的传输方式和设备；
- 出现过的「传输方式/设备」组合。

RESULT 行末尾另有 `ucx=rc_mlx5/mlx5_2:1,...`。

```bash
sbatch --export=ALL,HELLO_GROUPS="R E5" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
grep -h "transport/device pairs" -A6 ~/guohaojie/logs/hello2_<job>.d/E5*.262144.txt
```

### E6 组：换成 HPC-X 的 UCX（1.17.0）

E5d_alldev 能跨节点，但带宽和 Open MPI（R_ompi_pmix）比：

| 消息大小 | Intel / Open MPI |
|---|---|
| 8 KB | 0.67 倍 |
| 64 KB | 1.16 倍 |
| 256 KB | 0.52 倍（15.9 对 30.8 GB/s/节点） |

两边的 UCX 版本不同：

- v2 镜像：Ubuntu 22.04 自带的 1.12.1；
- Open MPI：HPC-X 2.19 自带的 1.17.0（`/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ucx`）。

**1. stage_hpcx_ucx.sh**（在登录节点运行，脚本自己发起两次 srun）

```bash
bash ~/guohaojie/src/NavierStockes-3D/scripts/intel/stage_hpcx_ucx.sh
```

- **copy（GCC 容器）**：把 HPC-X 的 `lib/libuc*.so*`、`lib/ucx/` 插件目录和 `bin/ucx_info` 复制到 `/work/opt_ucx/hpcx/`，`ucx_info -v` 的输出写进 `VERSION.txt`。
  成功时打印 `OK copy (Library version: 1.17.0)`。
- **check（v2 镜像）**：在 v2 镜像里对每个库查依赖（`LD_TRACE_LOADED_OBJECTS=1 ld.so`，和 ldd 同样的办法）。
  - 只缺 CUDA、gdrcopy、xpmem、knem、ROCm 等 GPU 或内核模块库的插件，移到 `lib/ucx.unresolved/`，UCX 不再去加载它们；
  - 核心库、ucx_info 或 IB 插件缺依赖则判 FAIL；
  - 最后用 HPC-X 的 ucx_info 看 `rc_mlx5` 是否可见。

  成功时应看到：
  - 若干行 `SET ASIDE libuct_cuda… / libucm_cuda… / libuct_xpmem…`；
  - `OK all core libraries, ucx_info and the remaining plugins resolve`；
  - transports 一行里有 `rc_mlx5`、`ud_mlx5`、`dc_mlx5`；
  - `OK rc_mlx5 visible …`，最后 `OK check`。

  检查结果写进 `/work/opt_ucx/hpcx/CHECK.txt`。如果出现 `FAIL plugin libuct_ib…: libmlx5.so.1: version 'MLX5_1.xx' not found`，说明 HPC-X 的 UCX 需要的 rdma-core 比镜像里 Ubuntu 的 rdma-core 39 新，把这一行发给我。

**2. hello_2node 的 E6 组**

E6 全部以 E5d_alldev 的设置为基础（v2 镜像），每个变体测 3 种大小：

| 变体 | 在 E6a 基础上另加 |
|---|---|
| E6a_hpcxucx | —（`LD_LIBRARY_PATH` 最前面放 `/work/opt_ucx/hpcx/lib`，`UCX_MODULE_DIR=/work/opt_ucx/hpcx/lib/ucx`） |
| E6b_hpcxucx_relax | `UCX_IB_PCI_RELAXED_ORDERING=on` |
| E6c_a2a2 | `I_MPI_ADJUST_ALLTOALL=2`（Isend/Irecv + waitall） |
| E6d_a2a3 | `I_MPI_ADJUST_ALLTOALL=3`（pairwise exchange） |
| E6e_a2a4 | `I_MPI_ADJUST_ALLTOALL=4`（Plum） |
| E6f_mlxtls | `FI_MLX_TLS=rc_mlx5,ud_mlx5,sm,self` |

- **确认用的是哪个 UCX**：hello_a2a 的 rank 0 在 MPI_Init 之后打印 `MAPS <路径>`，也就是进程实际映射的 libucp/libuct/libfabric/libmpi。源码比二进制新时，hello_2node 会在 Intel 镜像里自动重编 hello_a2a。
- **退回自带 UCX**：如果 E6a 没用上 HPC-X 的 libucp（没 stage、缺依赖或版本问题），E6b–f 退回镜像自带的 UCX 1.12.1，并打印一行 NOTE。
- **RESULT 行**：末尾都有 `ucx_version=<ucx_info -v 的版本>`；hello_a2a 打印了 MAPS 时，另有 `ucx_lib=<实际加载的 libucp 所在目录>`。原来的 `pmi_api=` 和 `ucx=` 字段保留。
- **R_ompi_pmix**：本身不变。另外在它之前打印 Open MPI 用的 UCX：`ucx_info -v`、`mca_pml_ucx.so` 链接的 libucp，以及 `run_env.sh` 设置的 `UCX_*` 变量。

```bash
sbatch --export=ALL,HELLO_GROUPS="R E5 E6" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1)
```

**达标标准**

- 256 KB 时 Intel 的每节点跨节点带宽 ≥ 0.8 × Open MPI（Open MPI 是 30.8 GB/s，所以约 ≥ 25 GB/s/节点）；
- 8 KB 和 64 KB 都 ≥ 0.8 × Open MPI。

summary 最后一张表 `off-node bandwidth relative to R_ompi_pmix` 列出每个变体在三种大小下相对 Open MPI 的倍数、256 KB 时的 GB/s、ucx 版本，以及 `PASS` 或 `below target`。

本机测过的部分：

- copy 步骤在本机 NVHPC 24.7 Docker 镜像里跑通，HPC-X 的 UCX 是 1.17.0；
- check 步骤在没有 CUDA 和 IB 的 Debian 镜像里跑通：5 个 GPU/xpmem 插件被放到一边，IB 插件因为缺 libibverbs 被判 FAIL；
- E6 用假 srun 跑通：rank 0 的 MAPS 显示加载的是 `opt_ucx/hpcx/lib/libucp.so.0.0.0`，所以 E6b–f 都用了 HPC-X，`ucx_version=1.17.0`；
- 判定表用虚构的数据测过。

没测：v2 镜像里的真实依赖情况（rdma-core 39 对 HPC-X UCX 是否够新）、InfiniBand 和带宽。

另外修了一个 bug：B2 的依赖检查原来用 `ld.so --list`，它遇到第一个缺的库就报 "cannot open shared object file" 并退出，不会逐个打印 "not found"，所以以前报的 "all dependencies resolve" 并不可靠。现在改成 `LD_TRACE_LOADED_OBJECTS=1 ld.so`，和 ldd 的做法一样。

## 多节点 Intel MPI 的固定设置：intel_mn_env.sh + irun_mn

hello_2node 里最好的是 E6d_a2a3，相对 Open MPI 的带宽为 2.72 / 0.96 / 0.93 倍（8 KB / 64 KB / 256 KB）。这套设置现在固定下来，以后所有多节点 Intel 作业都用它：

- `intel_mn_env.sh`：在容器内 source，包含 E6d 的全部设置：
  - pmi2 启动，`I_MPI_PMI_LIBRARY=libpmi2.so.0`；
  - `I_MPI_FABRICS=shm:ofi`，`FI_PROVIDER=mlx`，只用 Intel 自带的 provider 目录；
  - HPC-X 的 UCX 1.17（`/work/opt_ucx/hpcx`）；
  - `UCX_TLS=rc_mlx5,ud_mlx5,sm,self`，8 块网卡；
  - `I_MPI_ADJUST_ALLTOALL=3`。

  rank 0 会打印一行 `intel_mn_env: …`，记录实际用了哪些设置。开关：

  | 开关 | 默认 | 作用 |
  |---|---|---|
  | `IMN_ALLTOALLV` | 不设 | 设为 1 或 2 时导出 `I_MPI_ADJUST_ALLTOALLV`；heFFTe 和 AccFFT 用 MPI_Alltoallv |
  | `IMN_ALLTOALL` | 3 | 设为空就用 Intel MPI 的默认值 |
  | `IMN_UCX_DIR` | `/work/opt_ucx/hpcx` | 设为 `system` 用镜像自带的 UCX 1.12.1 |
  | `IMN_NET_DEVICES` | 8 块网卡 | UCX_NET_DEVICES |
  | `IMN_PIN` | `slurm` | `slurm`：由 srun 的 `--cpu-bind=cores` 绑核，`I_MPI_PIN=0`，和 Open MPI 一样；`intel`：用 `I_MPI_PIN_DOMAIN` |
  | `IMN_DEBUG` | 不设 | I_MPI_DEBUG 级别 |

- `intel_common.sh` 里的 `irun_mn <节点数> <每节点进程数> <每进程线程数> "<命令>"`：
  - 执行 `srun --mpi=pmi2 -N … --ntasks-per-node=… -c … --cpu-bind=cores $MI`；
  - 每个 rank 上设 `OMP_NUM_THREADS`、`OMP_PLACES=cores`、`OMP_PROC_BIND=close`，然后 source intel_mn_env.sh，再执行命令；
  - `IMN_*` 会传进容器；`IRUN_SRUN_OPTS` 给 srun 加参数；`IMN_SHOWBIND=1` 时 rank 0、1 打印各自的 CPU 列表。
  - 用之前要 `export INTEL_IMAGE=v2`，再 source intel_common.sh。

## cmp_toolchain_4n：4 节点，求解器，gcc 对 intel

- **gcc**：GCC 容器，Open MPI，`srun --mpi=pmix`。程序是 `bin_gcc` 里的 ns_fftw、ns_accfft、ns_heffte_v2，以及 `bin_p3d/gcc/ns_p3dfft`。
- **intel**：v2 镜像，用 `irun_mn` 启动，`KMP_BLOCKTIME=0`。程序是 `bin_intel` 里的 ns_fftw、ns_accfft，`ns_heffte_v2_mkl` 加 `NS_HEFFTE_REORDER=1`，以及 `bin_p3d/intel/ns_p3dfft`。

四个库都用 `NS_DECOMP=slab`、`NS_FFTW_PLAN=measure`。每节点布局 128×1 和 64×2，即 512 个进程 × 1 线程，或 256 个进程 × 2 线程。每个配置跑 2 次，重复放外层循环，gcc 和 intel 交替。

标签格式是 `<库>-<网格>_<gcc|intel>_<每节点布局>`，例如 `heFFTe-1024_intel_64x2`。ct_parse.sh 和 pl_parse.sh 都能直接解析。

**时长估算**：每个网格 32 次运行（4 个库 × 2 种工具链 × 2 种布局 × 2 次）。

- 512³：每次约 1–1.5 分钟，共约 35–50 分钟；
- 1024³：每次约 2–4 分钟，共约 65–130 分钟。

两个加起来 1.7–3 小时，太接近 3 小时，所以每个网格单独一个作业。另外，ct4_parse.sh 每份日志只能用一个动能参考值，这也要求每个网格一份日志。

| 参数 | 网格 | 时间步 | 动能参考 |
|---|---|---|---|
| `dry` | 64³，dt=5e-5，10 步，每个库每种工具链 1 次；4 节点 × 每节点 16 进程 × 8 线程（slab 要求进程数不超过 N，即最多 64） | | 3.3526466678e+05 |
| `512` | 512³，dt=1e-6 | 20 步 | 1.7933833444e+08 |
| `1024` | 1024³，dt=1e-6 | 10 步 | 1.4360200052e+09 |

```bash
cd ~/guohaojie/src/NavierStockes-3D/scripts/intel
sbatch -J ct4dry   -t 00:30:00 cmp_toolchain_4n.sbatch dry
bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4dry_*.out | head -1)
# dry 通过后：
sbatch -J ct4_512  -t 01:30:00 cmp_toolchain_4n.sbatch 512
sbatch -J ct4_1024 -t 03:00:00 cmp_toolchain_4n.sbatch 1024
bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4_512_*.out | head -1)
bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4_1024_*.out | head -1)
```

ct4_parse.sh 的输出依次是：

- ct_parse 的表：每个配置的 step 时间中位数、`vs gcc`、`energy ok`；
- `best configuration per (library, grid) and toolchain`：每种工具链在能量全对的布局里取最快的一个，最后一列是 intel/gcc；
- 退出码不为 0 的运行。

**dry 通过的标志**：

- 8 行 `energy ok / failed` 都是 `1 / 0`；
- 都是 `decomposition: slab   process grid: 1x64`；
- 第一次 Intel 运行打印 `intel_mn_env: … UCX=/work/opt_ucx/hpcx/lib/ucx … ALLTOALL=3`、`MPI startup(): libfabric provider: mlx`，以及两行 BIND；
- 没有退出码不为 0 的运行。

本机测过的部分：

- 用假 srun（4 个 rank，Open MPI，8 个程序都指向本机的 P3DFFT 程序）跑通了 dry：能量 3.3526466678e+05 全对，ct4_parse 三张表都正常；
- `IMN_ALLTOALLV=2` 能传到每个 rank，非法值 7 被忽略。

没测：v2 镜像里真实的 4 节点运行、各库的 Intel 程序、带宽和耗时。

## 对等调优的 4 节点对比

ct4_512 的结果不可比，原因有三：

- Intel 用的是 hello 调出来的 E6d 设置，Open MPI 一边没有调过；
- FFTW Intel 64×2 异常慢而且不稳定（2.83–4.68 s）；
- GCC 的 P3DFFT 在 4 节点上几乎不加速。

下面的改动让两边可以用同样的办法调优，再做对比。

### 1. hello_2node：Alltoallv、更大的消息、Open MPI 调优（RT）和 Intel Alltoallv（E7）

- **hello_a2a** 多了第三个参数 `a2a|a2av`。`a2av` 用均匀计数调用 MPI_Alltoallv，和 P3DFFT、heFFTe、AccFFT 的调用方式一样。HELLO_OK 行多了 `func=`。
- **消息大小**：每对进程 8 KB、64 KB、256 KB、512 KB、1 MB。1 MB 时每个进程的两个缓冲区共 512 MB，每节点约 64 GB。
- **RT 组**（GCC 容器，Open MPI）：
  - 运行前打印 `ucx_info -v`、所有 `UCX_*` 和 `OMPI_MCA_*` 变量，以及 ompi_info 里 hcoll、ucc、tuned 三个 coll 组件的 enable、priority 和算法参数（写在 `hello2_<job>.d/ompi_coll.txt`）。
  - 变体名为 `RT_<a2a|a2av>_<def|t<k>>_<d|8h>`：
    - `def`：不设任何 coll 参数；
    - `t<k>`：`OMPI_MCA_coll_tuned_use_dynamic_rules=1`，alltoall 用算法 0、2、3、4，alltoallv 用算法 0、1、2；
    - `d`：UCX 默认网卡；`8h`：和 E5d/E6 一样的 8 块网卡。`RT_DEVS` 选择跑哪一种或两种。
  - **我自己加的一项**：`t<k>` 变体同时设了 `OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_coll_ucc_enable=0`。否则优先级更高的 hcoll 或 ucc 会接管这个集合操作，tuned 的算法设置根本不生效。`RT_HCOLL_OFF=0` 可以保留 hcoll 和 ucc。`def` 变体保持 Open MPI 的原样。
- **E7 组**（Intel，E6d 设置）：`E7_a2a_e6d`（Alltoall，用来在新的消息大小上和 RT 对比）、`E7_a2av_def`、`E7_a2av_1`、`E7_a2av_2`（Alltoallv，`I_MPI_ADJUST_ALLTOALLV` 分别为不设、1、2）。
- **summary**：
  - 先按 (变体, 函数, 大小) 逐行列出；
  - 再对每个 (函数, 大小) 给出 Open MPI 一边（R*、RT*）的最佳设置和 Intel 一边的最佳设置，以及 intel/ompi 带宽比，低于 0.8 时标出。

```bash
sbatch --export=ALL,HELLO_GROUPS="R RT E6 E7" ~/guohaojie/src/NavierStockes-3D/scripts/intel/hello_2node.sbatch v2
sed -n '/=== summary/,$p' $(ls -t ~/guohaojie/logs/hello2_*.out | head -1) | grep -v "^RESULT"
cat ~/guohaojie/logs/hello2_<job>.d/ompi_coll.txt
```

### 2. FFTW 求解器：NS_PRINT_PLAN=1

设了 `NS_PRINT_PLAN=1` 时，rank 0 在建完 plan 后把正、逆 plan 打印出来，每行前缀 `FFTW PLAN:`。从中能看到 FFTW 选了哪种 MPI 转置：`mpi-transpose-alltoall`、`-pairwise` 或 `-recurse`。

- 用的是 `fftw_sprint_plan`；FFTW 版本太旧没有这个函数时，编译加 `-DNS_FFTW_NO_SPRINT`，改用 `fftw_fprint_plan`；
- 不设这个变量时行为不变。

需要重编两边的 ns_fftw：GCC 用 `scripts/compile_all.sh`，Intel 用 `build_intel_solvers.sh`。

### 3. cmp_toolchain_4n.sbatch

- **GCC 一边**：sbatch 环境里所有 `OMPI_MCA_*` 变量在 `run_env.sh` 之后原样传给各进程；`CT4_GCC_ENV="UCX_NET_DEVICES=…"` 可以再加别的变量。日志开头打印 `GCC env (after run_env.sh): …`，rank 0 也会打印它实际看到的 `OMPI_MCA_coll*` 和 `UCX_NET*`。
- **Intel 一边**：`CT4_INTEL_A2A="3 def"` 会让 Intel 一边按每种 `I_MPI_ADJUST_ALLTOALL` 设置各跑一次，标签为 `intel-a3`、`intel-def`；不设时只跑一种，标签为 `intel`。
- **步数和参考动能**：步数改为 10。512³ 的参考动能为 1.7950250065e+08，1024³ 仍为 1.4360200052e+09。
- **UCX 版本**：每次运行都用 `scripts/cpu_final/ucx_probe.c` 记录 rank 0 实际加载的 UCX，日志里一行 `UCX: <标签> rep=<r> version=<v>`。Intel 多节点运行不是 1.17.0 时，打印 `WARNING: UCX <v> loaded, expected 1.17.0`。
- **ct4_parse.sh**：
  - 最佳配置表现在每个工具链一行，带 `range (min-max)`、相对 gcc 的倍数，以及 `ranges overlap`（与 gcc 最佳配置的 min–max 是否重叠；重叠说明差别在运行间波动之内）；
  - 新增 UCX 版本一节；
  - 日志里有 FFTW plan 时，每次运行输出一行：step 时间和正、逆 plan 的转置方式。

### 4. ct4_fftw_diag.sbatch

只测 FFTW，512³，4 节点。gcc 和 Intel 各跑 128×1 和 64×2；Intel 一边分 `I_MPI_ADJUST_ALLTOALL=3` 和 Intel 默认两种。每组 3 次，`NS_PRINT_PLAN=1`。共 18 次运行，约 20–30 分钟。实际上就是用 `CT4_LIBS=fftw CT4_REP=3 CT4_INTEL_A2A="3 def" CT4_PRINT_PLAN=1` 运行 `cmp_toolchain_4n.sbatch 512`。

```bash
cd ~/guohaojie/src/NavierStockes-3D/scripts/intel && sbatch ct4_fftw_diag.sbatch
bash ct4_parse.sh $(ls -t ~/guohaojie/logs/ct4_ct4fd_*.out | head -1)
```

### 本机测过的部分

- hello_a2a 的 a2a 和 a2av 两种模式，1 MB 时数据校验都通过；
- `NS_PRINT_PLAN`：`sprint` 和 `fprint` 两种实现都打印出 `mpi-transpose-pairwise`，默认不打印；
- hello_2node 的 R、RT、E7 三组用假 srun 跑通。本机的 ompi_info 能列出 coll 组件和 tuned 的参数（本机没有 hcoll 和 ucc），summary 两张表都正常；
- cmp_toolchain_4n 用假 srun 跑通（FFTW、两种 Intel ALLTOALL 设置、打印 plan、OMPI_MCA 透传），ct4_parse 各节都正常。

没测：真实的 2 节点和 4 节点运行、hcoll 和 ucc 的实际状态、带宽和耗时。
