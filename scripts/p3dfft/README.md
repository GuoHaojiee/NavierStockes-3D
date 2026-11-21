# P3DFFT 重编（-O3 + OpenMP）

## 问题

已有的 P3DFFT 库有两个编译问题：

- **没有优化**：`FCFLAGS` 只有 `-g`，这是 autoconf 给非 gfortran 编译器的默认值。nvfortran 和 ifx 只拿到 `-g` 时按 `-O0` 编译，所以 Fortran 部分（打包、解包、本地转置）没有优化。
- **GCC 容器版没有 OpenMP**：`build/build_p3dfft.sh` 和 `planner/build_planner.sh` 都没加 `--enable-pgi`，configure 因此不加 `-mp`，`!$OMP` 指令被忽略。nm 只找到 2 个 OpenMP 符号（Intel 版有 327 个），每个进程只用 1 个线程。纯 OpenMP 1×128 要 33.6 s/步。

## 文件

| 文件 | 在哪里运行 | 作用 |
|---|---|---|
| `build_p3dfft_opt.sh gcc\|intel` | GCC 容器 / Intel 容器 | 编库，装到 `/work/opt_p3d/<gcc\|intel>`；编求解器，输出到 `/work/build/bin_p3d/<gcc\|intel>/ns_p3dfft`。不动已有的库。 |
| `p3d_verify.sbatch` | 1 节点 | 两个新程序，slab 和 pencil，64³ verify，8 个进程；L2 必须是 6.0041e-10 |
| `p3d_timing.sbatch` | 1 节点 | 旧/新 × gcc/intel，slab/pencil，128×1 和 64×2 各 3 次；每个程序再跑一次纯 OpenMP 1×128 |
| `p3d_parse.sh` | 登录节点 | pl_parse 的表，加上相对 old-gcc 的倍数和新/旧的倍数 |

### 编译参数

- **gcc**：
  - 编译器：`OMPI_FC=nvfortran OMPI_CC=nvc OMPI_CXX=nvc++`，先 `autoreconf -fi`；
  - configure：`--enable-pgi --enable-fftw --with-fftw=/opt/fftw3 --enable-openmp`，`FCFLAGS=CFLAGS="-O3 -tp=zen2 -mp"`；
  - LIBS、CPPFLAGS 和 build_p3dfft.sh 一样。
- **intel**：
  - 写法同 `build_intel_libs.sh`：`--enable-intel`，`FC=mpiifx`，configure 用 `prep_p3dfft_gcc.sh` 生成的那份；
  - `FCFLAGS=CFLAGS="-O3 -march=core-avx2"`，`-qopenmp` 由 configure 自己加；
  - 不用 `-xHost` / `-xCORE-AVX2`，这两个编出的程序在 AMD 上运行会失败。
- **检查**：脚本打印 `build/Makefile` 里的 FC、FCFLAGS、FDFLAGS、CFLAGS，以及 `nm libp3dfft.a | grep -ciE "__kmpc|_mp_|omp_"` 的计数。以下任一条不满足就判 `FAIL`：
  - FCFLAGS 里有 `-O3`；
  - OpenMP 符号数大于 20（阈值用 `P3O_MIN_OMP` 改）；
  - config.h 里有 MEASURE、OPENMP，没有 STRIDE1。
- **启动时的打印**：求解器打印 `P3DFFT library: <libp3dfft.a 路径> (<模式>: FC=…, FCFLAGS=…, OpenMP symbols N)`，来自新加的宏 `NS_P3DFFT_LIB`。旧程序打印 `not recorded`。

## 本机测过的部分

- 本机没有 nvfortran 和 ifx，只能用第三种模式 `gnu`（只供本机测试：gfortran + Open MPI，`--enable-gnu`，`-O3 -fallow-argument-mismatch`）把整个流程跑了一遍：
  - configure 参数、config.h 检查、Makefile 打印、nm 计数（37）和 FAIL 判定都正常；
  - 求解器能编出来；
  - 64³ 8 进程 slab 和 pencil 的 L2 都是 6.0041e-10，4×2 线程也一样；P3DFFT 打印 `Running on 2 threads`。
- 三个 sbatch 用一个假 srun 在本机跑通了：32³，4×1、2×2、1×4，然后用 p3d_parse 解析。
- **没测**：nvfortran 和 ifx 的实际编译（`-tp=zen2 -mp`、`-march=core-avx2`、`--enable-pgi` 的 FDFLAGS）、Intel 版的链接、容器、Slurm、numactl。

## 超算上的运行顺序

```bash
cd ~/guohaojie/src/NavierStockes-3D && git pull --ff-only
source scripts/intel/intel_common.sh            # $M $MI $EXCL
```

**1. GCC 版**（约 10–20 分钟）

```bash
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --gres=gpu:1 --time=00:40:00 $M bash /work/src/NavierStockes-3D/scripts/p3dfft/build_p3dfft_opt.sh gcc
```

成功时应看到：
- `OK   gcc_lib`；
- `FC = mpif90`、`FCFLAGS = -O3 -tp=zen2 -mp`，`FDFLAGS` 里有 `-DOPENMP -mp -DMEASURE -DFFTW`；
- OpenMP 符号数是几百（Intel 旧版是 327），然后是 `OK  gcc_check`；
- `OK   gcc_solver`；
- ldd 显示 `libnvomp.so` 和 `libnvf.so`。

日志在 `~/guohaojie/opt_p3d/logs/gcc_lib.log`。

**2. Intel 版**（约 10–20 分钟）

需要 `/work/src/intel_libs/p3dfft/configure`，它由 `prep_p3dfft_gcc.sh` 生成。第 1 步也在 `/work/src/p3d_opt/p3dfft` 生成一份，脚本会自动找这两处。

```bash
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:40:00 $MI bash /work/src/NavierStockes-3D/scripts/p3dfft/build_p3dfft_opt.sh intel
```

成功时应看到：
- `OK   intel_lib`；
- `FC = mpiifx`、`FCFLAGS = -O3 -march=core-avx2`，`FDFLAGS` 里有 `-qopenmp`；
- OpenMP 符号数是几百，然后是 `OK  intel_check`；
- `OK   intel_solver`；如果第一次链接失败，会出现 `retry: … mpiifx -nofor-main`，之后仍是 `OK   intel_solver`；
- ldd 显示 `libiomp5.so`。

**3. 正确性**（约 5 分钟）

```bash
sbatch scripts/p3dfft/p3d_verify.sbatch
grep -E "^RESULT|^P3DFFT library" $(ls -t ~/guohaojie/logs/p3dv_*.out | head -1)
```

成功时应看到 4 行 `RESULT … L2=6.0041e-10 OK`：new-gcc/new-intel × slab/pencil，pencil 的进程网格是 4x2。

**4. 计时**（3 小时，单节点）

```bash
cd scripts/p3dfft && sbatch p3d_timing.sbatch
bash p3d_parse.sh $(ls -t ~/guohaojie/logs/p3dt_*.out | head -1)
```

运行次数：4 个程序 × 2 种分解 × 2 种布局 × 3 次 = 48 次，每次 1–3 分钟；再加 4 次纯 OpenMP，其中 old-gcc 约 12 分钟。合计约 1.3–2.8 小时，所以设了 3 小时。

想拆成两个作业：

```bash
sbatch --export=ALL,P3T_DECOMPS=slab -t 01:45:00 p3d_timing.sbatch
sbatch --export=ALL,P3T_DECOMPS=pencil,P3T_OMP=0 -t 01:30:00 p3d_timing.sbatch
```

成功时应看到：
- 第一张表每行 `energy ok / failed` 为 `3 / 0`，1×128 行为 `1 / 0`；
- 第二张表：`vs old-gcc` 是相对 old-gcc 的倍数，`new / old` 是新库相对同一工具链旧库的倍数，都是小于 1 表示更快。预期 new-gcc 1×128 远快于 old-gcc 的 33.6 s/步；
- `Running on N threads` 一段：new-gcc、new-intel、old-intel 在 64×2 下是 2 threads，在 1×128 下是 128 threads。old-gcc 预期在所有布局下都是 1 threads，这直接证明旧库没有 OpenMP。原因是 P3DFFT 的 `init_plan.F90` 在 `!$OMP PARALLEL` 区里调用 `omp_get_num_threads()`，而没有 `-mp` 时这条指令被忽略，函数在并行区外调用只返回 1；
- decompositions 一段：`P3D-slab -> slab`、`P3D-pencil -> pencil`。

只解析 step 表也可以用 `bash planner/pl_parse.sh <log>`。
