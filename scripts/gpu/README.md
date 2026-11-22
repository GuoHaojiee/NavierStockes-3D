# scripts/gpu

| file | what |
|---|---|
| `build_heffte_gpu.sh` | builds `ns_heffte_gpu1`, `ns_heffte_mgpu`, `ns_heffte_multigpu` into `/work/build/bin_gpu_new` (never `bin_gcc`; `HG_OUT=` to change). One log per binary: `bin_gpu_new/ns_heffte_<v>.log`. A failed build keeps the old binary. |
| `heffte_layout_check.sbatch` | one node, 8 GPUs: gpu1 on 1 GPU, mgpu and multigpu on 2/4/8 GPUs × `zslab`/`xslab`; 64³ verify once (L2 = 6.0041e-10), 512³ bench (dt=1e-6, 10 steps, energy 1.7950250065e+08) 3 times; summary at the end. |
| `heffte_layout_parse.sh` | the summary, from the log or `~/guohaojie/logs/hlc_<job>.results` |
| `gpu_final.sbatch` | final GPU measurements, `<single\|node\|multi> <nodes> [dry]` (below); CSV `~/guohaojie/results/gpu_final.csv` |
| `gpu_final_parse.sh` | its summary: per configuration, heFFTe / cuFFTMp in the same job, time outside the FFT by library |
| `plot_gpu_final.py` | three figures (single, node, multi) from the CSV |
| `accfft_gpu_check.sh` | check only: tries to build AccFFT's GPU libraries with the container's CUDA 12.5 in `/work/tmp/accfft_gpu_check` (installs nothing) |

## heFFTe GPU layout switches (mgpu, multigpu)

| variable | values | default |
|---|---|---|
| `NS_HEFFTE_LAYOUT` | `zslab` (real z slabs, r2c along x, spectral y slabs; complex box (N/2+1)×N×N) · `xslab` (the earlier layout: x slabs, r2c along z, y slabs) | `zslab` |
| `NS_HEFFTE_ALGO` | `alltoallv` `alltoall` `p2p` `p2p_plined` | `p2p_plined` (gpu1: `alltoallv`) |
| `NS_HEFFTE_REORDER` | `0` `1` | `0` (also gpu1) |

Rank 0 prints the plan:
```
heFFTe layout: zslab, real 1x1x8, spectral 1x8x1, r2c along x (dim 0), reorder = 0, algorithm = p2p_plined, MPI reshapes per FFT = 1
HEFFTE_PLAN layout=zslab r2c_dir=0 reorder=0 algo=p2p_plined reshapes=1
```
zslab needs NY, NZ ≥ ranks; xslab needs NX, NY ≥ ranks.

## heffte_layout_check options

`HLC_REP` (3) · `HLC_NGPU` ("2 4 8") · `HLC_PROGS` ("mgpu multigpu") · `HLC_LAYOUTS` ("zslab xslab") · `HLC_GPU1` (1) ·
`HLC_BENCH_N` (512) · `HLC_ENV` (extra variables for every run, e.g. `"NS_HEFFTE_ALGO=alltoallv"`) · `HLC_BIN` (/work/build/bin_gpu_new).

## gpu_final.sbatch — final GPU runs

These runs replace the dB/dC/dD data from early November. In that data heFFTe used xslab with reorder=1, because `HX` in `runs/common.sh` sets `NS_HEFFTE_REORDER=1`.
- `HX` is not used here.
- Every `NS_*` variable from the submitting shell is removed inside the container. Only the settings below are set, and rank 0 prints `GF_ENV:` with what the run actually got.

| mode | nodes × GPUs | programs | grids |
|---|---|---|---|
| `single` | 1 × 1 | cuFFT `bin_gcc/ns_cufft`, heFFTe gpu1 `bin_gpu_new/ns_heffte_gpu1` (alltoallv, reorder 0) | 128 256 512 1024 (1024³: OOM expected) |
| `node` | 1 × 2/4/8 | cuFFTXt `bin_gcc/ns_cufftxt` (NS_XT_THREADS 6 on 8 GPUs, else 1), cuFFTMp `bin_gcc/ns_cufftmp_multinode`, heFFTe mgpu (zslab, p2p_plined, reorder 0) | 512 1024 |
| `multi` | 1/2/4 × 8, `bind_hca.sh` | cuFFTMp, heFFTe multigpu (zslab, p2p_plined, reorder 0) | 1024 2048 |

**Run settings**
- bench mode, dt=1e-6, 10 steps, 3 repetitions.
- The repetition is the outer loop; within it the libraries alternate for each (grid, GPU count).
- Energy references (10 digits):

| grid | energy | source |
|---|---|---|
| 128³ | 2.8047265727e+06 | computed with CPU FFTW, same dt and steps |
| 256³ | 2.2437812582e+07 | computed with CPU FFTW, same dt and steps |
| 512³ | 1.7950250065e+08 | given |
| 1024³ | 1.4360200052e+09 | given |
| 2048³ | 1.1488160042e+10 | given |

- `GF_EREF_<N>` overrides a reference.
- `energy_ok` is `yes` (identical), `close` (relative difference < 1e-9) or `no` (then the status is MISMATCH).

**dry**
- `dry` runs every (library, GPU count) of the mode once at 64³ in verify mode (dt=5e-5, 10 steps). L2 must be 6.0041e-10.
- Output goes to `gpu_final_dry.csv`. The exit code is 0 only if every run matches, so `afterok` gates the real jobs.

**Launches and statuses**
- Launches go through `scripts/intel/mn_lib.sh` `mn_run`: time limit 3 × expected, one retry on a broken start, and container names that include the job id.
- Status is one of OK, OOM, TIMEOUT, FAILED-RUN, ERROR, NO-RESULT, MISMATCH.
- The job exits 3 if anything is not OK. OOM at 1024³ on one GPU and at 2048³ on one node is expected and does not count.

**CSV columns:** `commit,jobid,nodelist,mode,lib,grid,nodes,gpus,rep,step_s,fft_s,spectral_s,physical_s,gpu_mem_GB,energy,energy_ok,status`, then `env,exit_code,retries,heffte_plan`.

```bash
cd ~/guohaojie/src/NavierStockes-3D && git pull --ff-only && cd scripts/gpu
DS=$(sbatch --parsable -N 1 -J gf_dry_single -t 00:30:00 gpu_final.sbatch single 1 dry)
DN=$(sbatch --parsable -N 1 -J gf_dry_node   -t 00:40:00 gpu_final.sbatch node 1 dry)
DM=$(sbatch --parsable -N 4 -J gf_dry_multi4 -t 00:30:00 gpu_final.sbatch multi 4 dry)
S1=$(sbatch --parsable -d afterok:$DS -N 1 -J gf_single -t 01:00:00 gpu_final.sbatch single 1)
N1=$(sbatch --parsable -d afterok:$DN -N 1 -J gf_node   -t 02:30:00 gpu_final.sbatch node 1)
M1=$(sbatch --parsable -d afterok:$DM -N 1 -J gf_multi1 -t 01:30:00 gpu_final.sbatch multi 1)
M2=$(sbatch --parsable -d afterok:$DM -N 2 -J gf_multi2 -t 01:30:00 gpu_final.sbatch multi 2)
M4=$(sbatch --parsable -d afterok:$DM -N 4 -J gf_multi4 -t 01:30:00 gpu_final.sbatch multi 4)
GF_JOBS="$S1 $N1 $M1 $M2 $M4" bash gpu_final_parse.sh
python3 plot_gpu_final.py --jobs "$S1 $N1 $M1 $M2 $M4" --out ~/guohaojie/results/plots_gpu_final
```

`gpu_final_parse.sh` filters: `GF_JOBS`, `GF_COMMIT`, `GF_MODE`. Its sections:
1. Per configuration: median (min–max) of step, FFT and non-FFT (step − fft), plus spectral, physical, memory and statuses.
2. heFFTe / cuFFTMp and cuFFTXt / cuFFTMp (single: heFFTe gpu1 / cuFFT) in the same job.
3. Time outside the FFT by library, against cuFFTMp (single: cuFFT).
4. Not OK.

`plot_gpu_final.py` draws:
- Solid lines: time per step.
- Dashed lines with open markers: the FFT phase.
- Points are medians; error bars span min–max.
