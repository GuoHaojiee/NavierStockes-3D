#!/usr/bin/env python3
"""plot_cpu_final.py [csv] [outdir] — two figures from ~/guohaojie/results/cpu_final.csv (cpu_final.sbatch).

  fig_cpu_1node.(png|pdf)    one node: grouped bars, 4 libraries x 3 modes (128x1, 64x2, 1x128), median time per
                             step, value on top of each bar
  fig_cpu_scaling.(png|pdf)  strong scaling: nodes 1/2/4, time per step (log axis), one line per library; every point
                             is the best layout at that node count (lowest median among the layouts whose runs all
                             have the reference energy and exit code 0 — the same rule as cpu_final_parse.sh)
Academic style: Calibri (fallback Carlito / DejaVu Sans), dark blue 1F4E79, dark red 8B1E1E, grey; no titles or notes.
Filters: CPF_COMMIT=<hash>, CPF_JOBS="<jobid> ...".
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

BLUE, RED, GREY = "#1F4E79", "#8B1E1E", "#808080"
LIBS = ["FFTW", "AccFFT", "P3DFFT", "heFFTe"]
MODES = [("128", "1", "128×1"), ("64", "2", "64×2"), ("1", "128", "1×128")]
NODES = [1, 2, 4]


def setup_style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Calibri", "Carlito", "Cambria", "Caladea") if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def load(path):
    commit = os.environ.get("CPF_COMMIT", "")
    jobs = set(os.environ.get("CPF_JOBS", "").split())
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if commit and r["commit"] != commit:
                continue
            if jobs and r["jobid"] not in jobs:
                continue
            rows.append(r)
    return rows


def configs(rows):
    """(lib, nodes, ppn, threads) -> dict(median, valid)"""
    g = defaultdict(list)
    for r in rows:
        g[(r["lib"], int(r["nodes"]), r["ranks_per_node"], r["threads"])].append(r)
    out = {}
    for k, rs in g.items():
        steps = [float(r["step_s"]) for r in rs if r["step_s"]]
        valid = bool(steps) and all(r["energy_ok"] == "yes" and r["exit_code"] == "0" for r in rs)
        out[k] = {"median": statistics.median(steps) if steps else None, "valid": valid}
    return out


def best(cfg, lib, nodes):
    c = [(v["median"], k) for k, v in cfg.items() if k[0] == lib and k[1] == nodes and v["valid"]]
    return min(c) if c else None


def fig_1node(cfg, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [BLUE, RED, GREY]
    w = 0.26
    for j, (ppn, nt, lab) in enumerate(MODES):
        xs, ys = [], []
        for i, lib in enumerate(LIBS):
            v = cfg.get((lib, 1, ppn, nt))
            if v and v["median"] is not None:
                xs.append(i + (j - 1) * w)
                ys.append(v["median"])
        bars = ax.bar(xs, ys, w, color=colors[j], label=lab, edgecolor="none")
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}" if y < 10 else f"{y:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(LIBS)))
    ax.set_xticklabels(LIBS)
    ax.set_ylabel("Time per step, s")
    ax.legend(ncol=3, loc="upper left")
    ax.margins(y=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_cpu_1node.{ext}"))
    plt.close(fig)


def fig_scaling(cfg, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    style = {"FFTW": (BLUE, "o", "-"), "AccFFT": (RED, "s", "-"), "P3DFFT": (GREY, "^", "-"), "heFFTe": (BLUE, "D", "--")}
    for lib in LIBS:
        pts = [(n, best(cfg, lib, n)) for n in NODES]
        pts = [(n, b[0]) for n, b in pts if b]
        if not pts:
            continue
        c, m, ls = style[lib]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=m, linestyle=ls, linewidth=1.4,
                markersize=5, label=lib)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(NODES)
    ax.set_xticklabels([str(n) for n in NODES])
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time per step, s")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_cpu_scaling.{ext}"))
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/guohaojie/results/cpu_final.csv")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(path))
    os.makedirs(out, exist_ok=True)
    setup_style()
    cfg = configs(load(path))
    fig_1node(cfg, out)
    fig_scaling(cfg, out)
    print(f"written: {out}/fig_cpu_1node.(png|pdf), {out}/fig_cpu_scaling.(png|pdf)")


if __name__ == "__main__":
    main()
