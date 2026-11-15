# FFTW planner flag study (ESTIMATE / MEASURE / PATIENT), CPU solvers

Rule: every library uses the most thorough planner it supports, provided measurements show it is
not slower. heFFTe stays on ESTIMATE (hard-coded in heFFTe 2.x, no option to change it).

| Library | How the flag is chosen | PATIENT possible |
|---|---|---|
| FFTW (`ns_fftw`) | `NS_FFTW_PLAN=estimate\|measure\|patient\|exhaustive` at run time (default measure) | yes |
| AccFFT (`ns_accfft_pt`) | `NS_FFTW_PLAN=estimate\|measure\|patient` | only with the patched library: stock AccFFT turns `ACCFFT_PATIENT` into `FFTW_MEASURE` |
| P3DFFT (`ns_p3dfft_<flag>`) | fixed when the library is configured (`--enable-estimate`, default measure, `--enable-patient`) | yes, separate library build |
| heFFTe | `FFTW_ESTIMATE` hard-coded | no |

Every solver prints the flag and the planning time; results do not depend on the flag
(64³ verify: L2 = 6.0041e-10 for all flags, checked locally for FFTW and AccFFT).

```bash
# build (once)
source ~/guohaojie/runs/common.sh
srun -p batch --exclude=cn34,cn18 -N1 -n1 -c16 --time=01:00:00 $M bash /work/src/NavierStockes-3D/planner/build_planner.sh
# measure (three nodes in parallel)
cd ~/guohaojie/src/NavierStockes-3D/planner
sbatch -J pl_fftw pl_scan.sbatch fftw; sbatch -J pl_accfft pl_scan.sbatch accfft; sbatch -J pl_p3dfft pl_scan.sbatch p3dfft
# results
bash pl_parse.sh $(ls -t ~/guohaojie/logs/pl_pl_fftw_*.out | head -1)
```
