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
