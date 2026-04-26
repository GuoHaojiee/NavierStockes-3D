#!/usr/bin/env python3
"""
collect_summary.py — 合并三个 mode 的 CSV，生成总汇总表

在所有 sbatch job 完成后，在登录节点运行：
  python3 collect_summary.py /scratch/s02230673/benchmark_results

会在 benchmark_results/ 下生成：
  combined_summary.csv
  combined_table.txt
"""

import sys, os, csv, glob

root = sys.argv[1] if len(sys.argv) > 1 else "/scratch/s02230673/benchmark_results"

# 找所有 summary_*.csv（取最新 job 的）
csv_files = sorted(glob.glob(os.path.join(root, "job_*", "summary_*.csv")))
if not csv_files:
    print(f"No summary CSVs found under {root}/job_*/")
    sys.exit(1)

# 按 mode 取最新（job_ID 越大越新）
latest = {}
for f in csv_files:
    mode = os.path.basename(f).replace("summary_", "").replace(".csv", "")
    latest[mode] = f   # sorted 保证最新在后

print("Using CSV files:")
for mode, f in latest.items():
    print(f"  {mode:15s} → {f}")

all_rows = []
for mode, f in latest.items():
    with open(f) as fh:
        for row in csv.DictReader(fh):
            all_rows.append(row)

# 写合并 CSV
out_csv = os.path.join(root, "combined_summary.csv")
with open(out_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh,
        fieldnames=["solver","grid","mode","np","omp_threads",
                    "avg_step_s","l2_error","status"])
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nCombined CSV → {out_csv}")

# ── 格式化总表 ────────────────────────────────────────────
SOLVERS = ['fftw', 'accfft', 'p3dfft', 'heffte']
GRIDS   = ['128', '256', '512']
MODES   = ['pure_mpi', 'hybrid', 'pure_omp']
MODE_HEADER = {
    'pure_mpi' : 'PureMPI\n(128×1)',
    'hybrid'   : 'Hybrid\n(64×2)',
    'pure_omp' : 'PureOMP\n(1×128)',
}

def lookup(rows, solver, grid, mode, field):
    for r in rows:
        if r['solver']==solver and r['grid']==grid and r['mode']==mode:
            return r.get(field, 'N/A')
    return 'N/A'

def fmt(v):
    if v in ('N/A','ERROR','TIMEOUT','PARSE_ERR','NOT_FOUND',''):
        return f"[{v[:7]:^7}]"
    try:    return f"{float(v):.4f}s"
    except: return v[:9]

lines = []
W = 88
lines.append("=" * W)
lines.append("  NS3D CPU Benchmark — Combined Results: Avg Time per Step (seconds)")
lines.append("  Single node, 128 cores  |  10 steps  |  dt=1e-5")
lines.append("=" * W)

for grid in GRIDS:
    lines.append(f"\n  ┌── Grid: {grid}³ ─────────────────────────────────────────────────────────────┐")
    lines.append(f"  │ {'Solver':<8} │ {'PureMPI (128×1)':^17} │ {'Hybrid (64×2)':^17} │ {'PureOMP (1×128)':^17} │")
    lines.append(f"  │ {'-'*8}─┼─{'-'*17}─┼─{'-'*17}─┼─{'-'*17}─┤")
    for solver in SOLVERS:
        pm  = fmt(lookup(all_rows, solver, grid, 'pure_mpi',  'avg_step_s'))
        hy  = fmt(lookup(all_rows, solver, grid, 'hybrid',    'avg_step_s'))
        po  = fmt(lookup(all_rows, solver, grid, 'pure_omp',  'avg_step_s'))
        spm = lookup(all_rows, solver, grid, 'pure_mpi', 'status')
        shy = lookup(all_rows, solver, grid, 'hybrid',   'status')
        spo = lookup(all_rows, solver, grid, 'pure_omp', 'status')
        flag = lambda s: "✔" if s.startswith("OK") else "✘"
        lines.append(
            f"  │ {solver:<8} │ {pm:>8} {flag(spm):^8}   │"
            f" {hy:>8} {flag(shy):^8}   │"
            f" {po:>8} {flag(spo):^8}   │"
        )
    lines.append(f"  └{'─'*(W-4)}┘")

lines.append("\n" + "─" * W)
lines.append("  L2 Error Correctness Check (pure_mpi mode)")
lines.append("─" * W)
lines.append(f"  {'Solver':<10} {'128³':>14} {'256³':>14} {'512³':>14}")
lines.append(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14}")
for solver in SOLVERS:
    e = [lookup(all_rows, solver, g, 'pure_mpi', 'l2_error') for g in GRIDS]
    lines.append(f"  {solver:<10} {e[0]:>14} {e[1]:>14} {e[2]:>14}")

lines.append("=" * W)

table_str = "\n".join(lines) + "\n"
print(table_str)

out_table = os.path.join(root, "combined_table.txt")
with open(out_table, "w") as fh:
    fh.write(table_str)
print(f"Combined table → {out_table}")
