#!/usr/bin/env python3
"""
collect_summary.py — 合并三个 mode 的 CSV，生成总汇总表
在登录节点运行（不在容器内），登录节点有 python3

用法：
  python3 collect_summary.py /scratch/s02230673/benchmark_results
"""
import sys, os, csv, glob

root = sys.argv[1] if len(sys.argv) > 1 else "/scratch/s02230673/benchmark_results"

latest = {}
for f in sorted(glob.glob(os.path.join(root, "job_*", "summary_*.csv"))):
    mode = os.path.basename(f).replace("summary_","").replace(".csv","")
    latest[mode] = f

if not latest:
    print(f"No summary CSVs found under {root}/job_*/")
    sys.exit(1)

print("Using:")
for m,f in latest.items():
    print(f"  {m:20s} -> {f}")

all_rows = []
for f in latest.values():
    with open(f) as fh:
        all_rows.extend(csv.DictReader(fh))

out_csv = os.path.join(root, "combined_summary.csv")
with open(out_csv,"w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["solver","grid","mode","np","omp_threads","avg_step_s","l2_error","status"])
    w.writeheader(); w.writerows(all_rows)

SOLVERS = ['fftw','accfft','p3dfft','heffte']
GRIDS   = ['128','256','512']

def get(rows, solver, grid, mode, field):
    for r in rows:
        if r['solver']==solver and r['grid']==grid and r['mode']==mode:
            return r.get(field,'N/A')
    return 'N/A'

def ft(v):
    try:    return f"{float(v):>8.4f}s"
    except: return f"[{str(v)[:8]:^8}]"

W = 72
lines = ["="*W,
         "  NS3D CPU Benchmark — Combined: Avg Time per Step",
         "  Single node 128 cores | 10 steps | dt=1e-5",
         "="*W]

for grid in GRIDS:
    lines += [f"\n  Grid: {grid}^3",
              f"  {'Solver':<10} {'PureMPI(128x1)':>16} {'Hybrid(64x2)':>16} {'PureOMP(1x128)':>16}",
              f"  {'-'*10} {'-'*16} {'-'*16} {'-'*16}"]
    for s in SOLVERS:
        pm = ft(get(all_rows,s,grid,'pure_mpi','avg_step_s'))
        hy = ft(get(all_rows,s,grid,'hybrid','avg_step_s'))
        po = ft(get(all_rows,s,grid,'pure_omp','avg_step_s'))
        lines.append(f"  {s:<10} {pm:>16} {hy:>16} {po:>16}")

lines += ["\n"+"─"*W,"  L2 Error (pure_mpi)","─"*W,
          f"  {'Solver':<10} {'128^3':>14} {'256^3':>14} {'512^3':>14}",
          f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14}"]
for s in SOLVERS:
    e = [get(all_rows,s,g,'pure_mpi','l2_error') for g in GRIDS]
    lines.append(f"  {s:<10} {e[0]:>14} {e[1]:>14} {e[2]:>14}")
lines.append("="*W)

table = "\n".join(lines)+"\n"
print("\n"+table)

out_table = os.path.join(root, "combined_table.txt")
with open(out_table,"w") as fh:
    fh.write(table)

print(f"CSV   -> {out_csv}")
print(f"Table -> {out_table}")
