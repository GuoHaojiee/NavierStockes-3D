# scripts/gpu

| file | what |
|---|---|
| `build_heffte_gpu.sh` | builds `ns_heffte_gpu1`, `ns_heffte_mgpu`, `ns_heffte_multigpu` into `/work/build/bin_gpu_new` (never `bin_gcc`; `HG_OUT=` to change). One log per binary: `bin_gpu_new/ns_heffte_<v>.log`. A failed build keeps the old binary. |
| `heffte_layout_check.sbatch` | one node, 8 GPUs: gpu1 on 1 GPU, mgpu and multigpu on 2/4/8 GPUs × `zslab`/`xslab`; 64³ verify once (L2 = 6.0041e-10), 512³ bench (dt=1e-6, 10 steps, energy 1.7950250065e+08) 3 times; summary at the end. |
| `heffte_layout_parse.sh` | the summary, from the log or `~/guohaojie/logs/hlc_<job>.results` |
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
