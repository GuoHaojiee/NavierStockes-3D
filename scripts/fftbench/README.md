# scripts/fftbench — pure 3D FFT benchmark (CPU)

For quick scans of configurations and scaling. The final numbers still come from the solvers.

## Same code as the solvers

The plan and execute code of every CPU library now lives in one header per library. Each header is included by the
solver **and** by `fftbench.cpp`:

| header | solver | what it covers |
|---|---|---|
| `fftlib/ns_fft_fftw.h`   | `NavierStokes_periodic_fftw.cpp`      | FFTW-MPI slab, `FFTW_MPI_TRANSPOSED_OUT/IN`, `NS_FFTW_PLAN`, `NS_PRINT_PLAN` |
| `fftlib/ns_fft_accfft.h` | `NavierStokes_periodic_accfft.cpp`    | `NS_DECOMP` slab/pencil (c_dims), int-overflow check, `NS_FFTW_PLAN` |
| `fftlib/ns_fft_p3dfft.h` | `NavierStokes_periodic_p3dfft.cpp`    | `NS_DECOMP` slab/pencil (dims), `Cp3dfft_setup` (= planning), r2c along x |
| `fftlib/ns_fft_heffte.h` | `NavierStokes_periodic_heffte_v2.cpp` | `NS_DECOMP` slab/pencil/brick, `NS_HEFFTE_BACKEND` fftw/mkl, `NS_HEFFTE_REORDER/ALGO`, reshape count |

The refactored solvers print exactly what they printed before. This was checked locally with every decomposition: the full
output, timing lines masked, is identical to the previous binaries, and so are the 64³ L2 and energy values.
(Since the transpose-method report, the AccFFT solver prints one more line, `FFTB_ACCFFT_TPLAN ...`. 64³ L2 is still
6.0041e-10 for slab and pencil.)

## One launch of `fftb_<lib> NX NY NZ [threads]`

1. **Setup and plan.** The plan is timed and reported separately (`t_plan_s`).
   - P3DFFT: this is `Cp3dfft_setup`.
   - heFFTe with the FFTW backend creates its FFTW plans at the first execution, so they are not in `t_plan_s`.
2. **Check.** forward → backward → normalise, then the maximum error against the input field: `FFTB_CHECK max_err=... (OK)`.
   The limit is 1e-10. The same line says whether the forward transform kept its input.
3. **Warm-up.** `FFTB_WARM` (2) rounds. The last round sets K, so that a repetition takes at least `FFTB_WINDOW` (3 s).
   `FFTB_PAIRS=K` fixes K instead.
4. **Repetitions.** `FFTB_REP` (3) repetitions of K forward + backward pairs, each timed from barrier to barrier (maximum
   over ranks). Pairs cycle through `FFTB_NF` (3) fields, so they do not stay on one cache-resident buffer set.
   - Each repetition prints `FFTB_REP ... t_fft_s=...`.
   - **t_fft_s = (t_fwd + 2·t_bwd)/3**: the solver's mix (12 forward + 24 backward per RK4 step in bench mode), so
     **36 × t_fft_s = FFT time per solver step**.
5. **Binding.** At start-up `FFTB_BIND ... OK|WARN` reports whether the ranks of a node share a CPU and whether
   every OpenMP thread sits on its own physical core (sysfs `core_id`). It becomes the CSV `bind` column.
   `FFTB_SHOWBIND=1` also prints the CPU lists of ranks 0 and 1.
6. **`FFTB_SPLIT=1`** additionally times the local FFT work with plain FFTW: batched 1-D transforms over N³/P points,
   same thread count, no communication. `t_fft − t_local` is an upper estimate of the communication + packing time.

## Cluster

Everything goes through `sbatch` with dependencies:
- **`build_fftb.sbatch gcc|intel`** runs `build_fftb.sh` in the container. It exits 0 only when the build ends with
  `BUILD OK`, meaning every expected binary is present and built from this source.
- **`fftb.sbatch`** (and the wrappers, which `exec` it) checks every binary it needs before starting anything (exit 2),
  and exits 3 if any run was not OK. So `afterok` on a dry run gates the long jobs.

```bash
cd ~/guohaojie/src/NavierStockes-3D && git pull --ff-only && cd scripts/fftbench
B=$(sbatch --parsable -J fftb_build_intel build_fftb.sbatch intel)
D1=$(sbatch --parsable -d afterok:$B -N 1 -J fftb_dry1 -t 00:40:00 fftb.sbatch 1 dry)
D2=$(sbatch --parsable -d afterok:$B -N 2 -J fftb_dry2 -t 00:40:00 fftb.sbatch 2 dry)
sbatch -d afterok:$D2 fftb_p256.sbatch
bash fftb_parse.sh ~/guohaojie/results/fftb_dry.csv
```

`build_fftb.sh` (before 2026-11-22): in intel mode its build helper `c()` was replaced by the `c()` of
`scripts/intel/intel_env.sh`. Every step printed OK, but the binaries went to `./a.out` and nothing reached
`bin_fftb/intel` ("0 of 0 built", exit 0). The helper is now `fb_build`, and the script checks the expected binaries at the end.

## fftb.sbatch `<nodes> <grid | "512 1024" | dry>`

| variable | default | meaning |
|---|---|---|
| `FFTB_LIBS` | `fftw accfft p3dfft heffte` | libraries |
| `FFTB_LAYOUTS` | `128x1` (dry: 64x2 / 32x4 / 16x8) | ranks per node x threads, list |
| `FFTB_TC` | `gcc intel` | toolchains |
| `GCC_COLL` | `def` | GCC settings, list (`"def t4"`): one launch each |
| `FFTB_GCC_ENV` | — | extra variables for the GCC side |
| `FFTB_INTEL_SETS` | `E6d:` (1 node: `single-node:`) | Intel settings, list; one launch each (format below) |
| `FFTB_INTEL_ENV_FFTW` / `_ACCFFT` / `_P3DFFT` / `_HEFFTE` | — | extra `VAR=val ...` for every Intel run of that library |
| `FFTB_CONFIGS` | — | explicit list of lines `name\|gcc\|layout\|GCC_COLL=x ...` / `name\|intel\|layout\|IMN_...` (replaces the four above) |
| `FFTB_DECOMP` | `slab` | NS_DECOMP |
| `FFTB_PLAN` | `measure` | NS_FFTW_PLAN |
| `FFTB_REP` / `FFTB_WINDOW` / `FFTB_NF` | 3 / 3 s / 3 | repetitions, minimum window per repetition, fields |
| `FFTB_SPLIT` | 0 | 1: also the local FFT time (`t_local_s`) |
| `FFTB_ENV` | — | extra VAR=val for every run |
| `FFTB_TIMEOUT_FACTOR` | 3 | time limit = factor x expected (1 / 2 / 4-10 min), at least 3 min |
| `FFTB_CSV` | `~/guohaojie/results/fftb.csv` (dry: `fftb_dry.csv`) | |

**`FFTB_INTEL_SETS` format.**
- Entries are separated by `;`. Each entry is `name:VAR=val VAR=val ...`.
- `name` becomes the configuration name, which is the first word of the CSV `settings` column.
- `name:` with nothing after it means the default settings (E6d, or single-node on 1 node).
- `IMN_*` variables go to `scripts/intel/intel_mn_env.sh`:
  - `IMN_ALLTOALL=` with an empty value means Intel MPI's default alltoall.
  - `IMN_ALLTOALLV=1|2`, `IMN_UCX_TLS`, `IMN_EXTRA` are passed the same way.
- Every other variable (for example `NS_ACCFFT_TMETHOD`, `NS_ACCFFT_PLAN`, `I_MPI_...`) is exported after intel_mn_env.sh, so it takes precedence.

```bash
FFTB_INTEL_SETS="E6d:;a2a_def:IMN_ALLTOALL=;a2av1:IMN_ALLTOALLV=1;est:NS_ACCFFT_PLAN=estimate"
FFTB_INTEL_ENV_ACCFFT="I_MPI_ADJUST_ALLTOALLV=1"     # Intel AccFFT runs only; shown in ">>>" and in settings
```

- Rank 0 prints `FFTB_MPIENV`, the `I_MPI_ADJUST_*`, `I_MPI_FABRICS`, `UCX_TLS`, `OMPI_MCA_coll*`, `NS_ACCFFT_*` and `NS_HEFFTE_*` values the program actually got.
- These values go into the CSV `mpi_env` column (column 28, with `,` replaced by `;`).
- intel_mn_env.sh's own line is printed before these variables are exported. For a variable set this way it can still say `default`, so check `FFTB_MPIENV` instead.

Exit code: 0 all OK; 2 a binary missing (nothing run); 3 some run not OK (status or check). Exit 127 of a run: status MISSING-BINARY, shown as "missing binary" by the parser.

- Launches go through `scripts/intel/mn_lib.sh`: job-specific containers, time limit, one retry on a broken start, UCX probe.
- The Intel heFFTe binary is `fftb_heffte_mkl` with `NS_HEFFTE_REORDER=1`, as in `cmp_toolchain_mn`.
- CSV: one line per repetition. A launch that produced no repetition writes one line with `rep=0`:
  `commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,t_fft_s,t_plan_s,t_local_s,check,ucx_version,status,`
  `t_fwd_s,t_bwd_s,pairs,max_err,nf,lib_settings,retries,exit_code,ucx_reason,bind,mpi_env`.
- `settings` = configuration name (first word) + its variables + `NS_DECOMP`.
- `status` is one of: OK, TIMEOUT, FAILED-RUN, ERROR, NO-RESULT, CHECK-FAIL.

## fftb_scale.sbatch `<nodes> <grid>` (Intel only, scaling scan)

- Four libraries: FFTW, AccFFT, P3DFFT, heFFTe (MKL backend, `NS_HEFFTE_REORDER=1`); slab, MEASURE, `FFTB_SPLIT=1`,
  3 repetitions per launch.
- Layouts `FFTB_LAYOUTS` (`"128x1 64x2 32x4 16x8"`), with threads = 128 / ranks per node.
- Thread settings: `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, `KMP_BLOCKTIME=0`, `MKL_NUM_THREADS` = threads.
- Binding:
  - 1 node: Intel `mpirun` pins with `I_MPI_PIN_DOMAIN=<threads>:compact`.
  - 2 and 4 nodes: `srun --cpu-bind=cores -c <threads>`.
  - Either way the `FFTB_BIND` line checks the result.

```bash
sbatch -N 1 -J fftb_scale_1024_1 -t 04:00:00 fftb_scale.sbatch 1 1024
sbatch -N 2 -J fftb_scale_1024_2 -t 03:00:00 fftb_scale.sbatch 2 1024
sbatch -N 4 -J fftb_scale_1024_4 -t 02:00:00 fftb_scale.sbatch 4 1024
FFTB_JOBS="<job1> <job2> <job4>" bash fftb_parse.sh
```

## fftb_tc.sbatch `<nodes> <grid>` (toolchain comparison)

GCC (`bin_fftb/gcc`; `GCC_COLL` list; heFFTe with the FFTW backend) against Intel (`bin_fftb/intel`; `FFTB_INTEL_SETS`,
default E6d / single-node; heFFTe with MKL and `NS_HEFFTE_REORDER=1`).
- Four libraries, slab, MEASURE, layouts `FFTB_LAYOUTS` (`128x1`), `FFTB_SPLIT=1`, 3 repetitions per launch.
- Within each library the toolchains alternate.
- Result tables: sections 2 and 2b of the parser.

```bash
sbatch -N 1 -J fftb_tc_1024_1 -t 02:00:00 fftb_tc.sbatch 1 1024
sbatch -N 4 -J fftb_tc_1024_4 -t 01:30:00 fftb_tc.sbatch 4 1024
FFTB_JOBS="<job1> <job4>" bash fftb_parse.sh
```

## AccFFT transpose method

The AccFFT library code is `src/accfft.cpp` and `src/transpose.cpp`.

In the slab path (`oneD`), `accfft_plan_dft_3d_r2c` with `ACCFFT_MEASURE` calls `T_Plan::which_fast_method` on the forward transpose plan `T_plan_2`.
- Each candidate is timed **once**, after one warm-up call. The time used is the maximum over the ranks.
- The fastest candidate is kept. The candidates are:
  - **m1**: `MPI_Isend`/`MPI_Irecv` to every rank.
  - **m2**: `MPI_Alltoall`, or `MPI_Alltoallv` when the data is not evenly distributed. For 1024³ on 128 or 256 ranks it is `MPI_Alltoall`, so `I_MPI_ADJUST_ALLTOALLV` has no effect there.
  - **m3k\<k\>a|s**: `par::Mpi_Alltoallv_dense`, a k-way exchange in Isend/Irecv rounds, async or sync, with kway = p/2, p/4, … > 4. These are only tried when p is a power of two > 7.
- The backward plan gets `-method`.

`ACCFFT_ESTIMATE` (and `ACCFFT_PATIENT` in the stock library) do not measure. They take m2.
- `ACCFFT_ESTIMATE` also makes the local FFTW plans `FFTW_ESTIMATE`.

After the plan, rank 0 prints:
```
FFTB_ACCFFT_TPLAN sel=measure|default|fixed method=.. kway=.. kway_async=.. label=m3k16a call=kway-dense-async comm_size=128 evenly=1 measured=m3k16a:1.2e-02;m2:...
```
- `measured` lists the candidates' single measurements, sorted by time.
- The same information is in `lib_settings` (`tsel=... tmethod=... call=...`).
- The solver binary prints the line too, because it uses the same header.

| variable | values | effect |
|---|---|---|
| `NS_ACCFFT_TMETHOD` | `auto` (default) \| `1` \| `2` \| `3:<kway>:a` \| `3:<kway>:s` (or `m1`, `m2`, `m3k16a`, …) | fixes the method after planning |
| `NS_ACCFFT_PLAN` | `estimate` \| `measure` \| `patient` | AccFFT only: replaces `NS_FFTW_PLAN`. The other libraries keep `NS_FFTW_PLAN`; the `FFTB_SPLIT` local model of the AccFFT benchmark follows it. |

How `NS_ACCFFT_TMETHOD` works:
- It sets the public `T_Plan` fields `method` / `kway` / `kway_async` of the plan's transpose plans: forward `+m`, backward `-m`. In pencil mode it also sets `T_plan_1` / `T_plan_1i`.
- This is exactly what `accfft_plan_dft_3d_r2c` does after its own measurement. The library source is not changed.
- kway must split the communicator (`p = c * kway^n`, c < kway). Otherwise the run stops with an error.
- The library's methods 5–8 (`transpose_v5..v8`, chosen by the old `which_method`) are not offered. The r2c plan never selects them.

## fftb_accfft_tune.sbatch `<A|B> [nodes 4] [grid 1024]`

This script measures only AccFFT, in layouts `FFTB_TUNE_LAYOUTS` (`32x4 64x2`), with slab, MEASURE, `FFTB_SPLIT=1` and 3 repetitions.

Configurations per layout:

| name | settings |
|---|---|
| `GCC_COLL=t4` | GCC reference |
| `E6d` | Intel, I_MPI_ADJUST_ALLTOALL=3 |
| `a2a_def` | Intel MPI default alltoall |
| `a2a1`, `a2a2` | ALLTOALL=1 / 2 |
| `a2av1`, `a2av2` | E6d + ALLTOALLV=1 / 2 |
| `ucx_dc` | E6d + UCX_TLS=dc_mlx5,sm,self |
| `plan_est` | E6d + NS_ACCFFT_PLAN=estimate |
| `tm_<m>` | E6d + NS_ACCFFT_TMETHOD=\<m\>, for each m in `FFTB_TUNE_TMETHODS` (`m1 m2 m3k8a m3k8s m3k16a m3k16s m3k32a m3k32s m3k64a m3k64s`; `""` = none) |

Run order:
- **A**: in each layout, GCC first, then the Intel variants.
- **B**: the exact reverse of A.

That makes 38 launches, about 1.3–1.9 h on 4 nodes at 1024³ (`-t 03:00:00`). Without the `tm_` variants it is 18 launches, about 0.6–0.9 h.

```bash
A=$(sbatch --parsable -J fftb_acc_tune_A fftb_accfft_tune.sbatch A)
B=$(sbatch --parsable -d afterany:$A -J fftb_acc_tune_B fftb_accfft_tune.sbatch B)
FFTB_JOBS="$A $B" bash fftb_parse.sh | sed -n '/--- 6\./,$p'
```

## fftb_parse.sh `[csv]`

The parser prints five sections:
1. Per (library, grid, nodes, toolchain, settings, layout): the median and range.
   - **1b**: every layout of a configuration side by side, with the best one marked.
   - **1c**: the communication share (t_fft − t_local)/t_fft per configuration.
2. Intel against the best GCC configuration: t_fft and 36 × t_fft (FFT s per RK4 step), ratio, and whether the
   ranges overlap. **2b**: one row per (library, grid, nodes) with the best GCC and the best Intel configuration,
   36 × t_fft median (min–max), intel/gcc and overlap, ready for a table.
3. Strong scaling with the best layout per node count (speedup, efficiency, comm%).
   **3b**: the same with the fixed layout `FFTB_FIXED_LAYOUT` (128x1).
4. Against the solver: 36 × t_fft next to the solver's `FFT (incl. transpose/comm)` s/step. It uses the rows of
   `FFTB_SOLVER_CSV` (default `toolchain_mn.csv`) with the same library, grid, nodes, toolchain, layout and
   configuration name. The configuration name is the first word of settings: `GCC_COLL=def`, `E6d`, `single-node`,
   or the `p256_diag` names `i_a_base` … `g_def`.
5. Everything that is not OK.
6. Variants: every Intel configuration of `FFTB_VAR_LIB` (AccFFT) against `FFTB_VAR_REF` (`GCC_COLL=t4`), with the same grid, nodes and layout.
   - Columns: launches, transpose method, t_fft median (min–max), t_local, t_comm, comm%, t/ref, comm/ref comm, ranges overlap.
   - One median per job, so the A and B rounds can be compared.

AccFFT rows in sections 1 and 6 show `tm=`, the transpose method per launch. `m2x3` means 3 launches used m2.

Filters: `FFTB_JOBS`, `FFTB_COMMIT`, `FFTB_TC`. Runs with a binding WARN are marked and listed in section 5.
