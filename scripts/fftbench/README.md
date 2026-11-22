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
5. **`FFTB_SPLIT=1`** additionally times the local FFT work with plain FFTW: batched 1-D transforms over N³/P points,
   same thread count, no communication. `t_fft − t_local` is an upper estimate of the communication + packing time.

## Cluster

```bash
cd ~/guohaojie/src/NavierStockes-3D && git pull --ff-only
source scripts/intel/intel_common.sh                       # $M, $MI, $EXCL
# build into build/bin_fftb/{gcc,intel} (never bin_gcc / bin_intel / bin_p3d)
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $M  bash /work/src/NavierStockes-3D/scripts/fftbench/build_fftb.sh gcc
srun -p batch --exclude=$EXCL -N1 -n1 -c16 --time=00:30:00 $MI bash /work/src/NavierStockes-3D/scripts/fftbench/build_fftb.sh intel
cd scripts/fftbench
sbatch -N 1 -J fftb_dry1 -t 00:40:00 fftb.sbatch 1 dry     # 64^3, every binary: check OK?
sbatch -N 2 -J fftb_dry2 -t 00:40:00 fftb.sbatch 2 dry
bash fftb_parse.sh ~/guohaojie/results/fftb_dry.csv
sbatch fftb_p256.sbatch                                     # p=256 question, 2 nodes, 512^3 + 1024^3
FFTB_SOLVER_CSV=~/guohaojie/results/p256_diag.csv bash fftb_parse.sh
```

## fftb.sbatch `<nodes> <grid | "512 1024" | dry>`

| variable | default | meaning |
|---|---|---|
| `FFTB_LIBS` | `fftw accfft p3dfft heffte` | libraries |
| `FFTB_LAYOUTS` | `128x1` (dry: 64x2 / 32x4 / 16x8) | ranks per node x threads, list |
| `FFTB_TC` | `gcc intel` | toolchains |
| `GCC_COLL` | `def` | GCC settings, list (`"def t4"`): one launch each |
| `FFTB_GCC_ENV` | — | extra variables for the GCC side |
| `FFTB_INTEL_SETS` | `E6d:` (1 node: `single-node:`) | Intel settings, list `"name:VAR=val VAR=val;name2:..."` with IMN_* variables; one launch each |
| `FFTB_CONFIGS` | — | explicit list of lines `name\|gcc\|layout\|GCC_COLL=x ...` / `name\|intel\|layout\|IMN_...` (replaces the four above) |
| `FFTB_DECOMP` | `slab` | NS_DECOMP |
| `FFTB_PLAN` | `measure` | NS_FFTW_PLAN |
| `FFTB_REP` / `FFTB_WINDOW` / `FFTB_NF` | 3 / 3 s / 3 | repetitions, minimum window per repetition, fields |
| `FFTB_SPLIT` | 0 | 1: also the local FFT time (`t_local_s`) |
| `FFTB_ENV` | — | extra VAR=val for every run |
| `FFTB_TIMEOUT_FACTOR` | 3 | time limit = factor x expected (1 / 2 / 4-10 min), at least 3 min |
| `FFTB_CSV` | `~/guohaojie/results/fftb.csv` (dry: `fftb_dry.csv`) | |

- Launches go through `scripts/intel/mn_lib.sh`: job-specific containers, time limit, one retry on a broken start, UCX probe.
- The Intel heFFTe binary is `fftb_heffte_mkl` with `NS_HEFFTE_REORDER=1`, as in `cmp_toolchain_mn`.
- CSV: one line per repetition. A launch that produced no repetition writes one line with `rep=0`:
  `commit,jobid,nodelist,toolchain,settings,lib,grid,nodes,ranks_per_node,threads,rep,t_fft_s,t_plan_s,t_local_s,check,ucx_version,status,`
  `t_fwd_s,t_bwd_s,pairs,max_err,nf,lib_settings,retries,exit_code,ucx_reason`.
- `settings` = configuration name (first word) + its variables + `NS_DECOMP`.
- `status` is one of: OK, TIMEOUT, FAILED-RUN, ERROR, NO-RESULT, CHECK-FAIL.

## fftb_parse.sh `[csv]`

The parser prints five sections:
1. Per (library, grid, nodes, toolchain, settings, layout): the median and range.
2. Intel against the best GCC configuration: ratio and whether the ranges overlap.
3. Strong scaling.
4. Against the solver: 36 × t_fft next to the solver's `FFT (incl. transpose/comm)` s/step. It uses the rows of
   `FFTB_SOLVER_CSV` (default `toolchain_mn.csv`) with the same library, grid, nodes, toolchain, layout and
   configuration name. The configuration name is the first word of settings: `GCC_COLL=def`, `E6d`, `single-node`,
   or the `p256_diag` names `i_a_base` … `g_def`.
5. Everything that is not OK.

Filters: `FFTB_JOBS`, `FFTB_COMMIT`.
