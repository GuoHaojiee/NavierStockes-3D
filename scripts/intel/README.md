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
