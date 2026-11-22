#!/usr/bin/env python3
"""plot_toolchain_mn.py [csv] [outdir] — strong scaling from ~/guohaojie/results/toolchain_mn.csv (cmp_toolchain_mn).

One figure per grid, fig_toolchain_mn_<grid>.(png|pdf): nodes 1/2/4 on the x axis, time per step (log axis) on the y
axis, one colour/marker per library, Intel solid, GCC dashed. Every point is the best configuration at that node count
(lowest median among the layouts/settings whose runs all have the reference energy and exit code 0, the same rule as
toolchain_mn_parse.sh). Academic style, no titles or notes; colours 1F4E79, 8B1E1E, grey.
Filters: CTM_COMMIT=<hash>, CTM_JOBS="<jobid> ...".
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

BLUE, RED, GREY, BLACK = "#1F4E79", "#8B1E1E", "#808080", "#000000"
LIBS = ["FFTW", "AccFFT", "P3DFFT", "heFFTe"]
STYLE = {"FFTW": (BLUE, "o"), "AccFFT": (RED, "s"), "P3DFFT": (GREY, "^"), "heFFTe": (BLUE, "D")}
TC = {"intel": ("-", "Intel"), "gcc": ("--", "GCC")}
NODES = [1, 2, 4]


def setup_style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Calibri", "Carlito", "Cambria", "Caladea") if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def load(path):
    commit = os.environ.get("CTM_COMMIT", "")
    jobs = set(os.environ.get("CTM_JOBS", "").split())
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f)
                if (not commit or r["commit"] == commit) and (not jobs or r["jobid"] in jobs)]


def best_points(rows):
    """(grid, toolchain, lib, nodes) -> best median"""
    g = defaultdict(list)
    for r in rows:
        g[(r["grid"], r["toolchain"], r["lib"], int(r["nodes"]), r["ranks_per_node"], r["threads"], r["settings"])].append(r)
    best = {}
    for k, rs in g.items():
        steps = [float(r["step_s"]) for r in rs if r["step_s"]]
        if not steps or not all(r["energy_ok"] == "yes" and r["exit_code"] == "0" for r in rs):
            continue
        m = statistics.median(steps)
        bk = k[:4]
        if bk not in best or m < best[bk]:
            best[bk] = m
    return best


def plot_grid(best, grid, out):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for lib in LIBS:
        c, mk = STYLE[lib]
        for tc, (ls, _) in TC.items():
            pts = [(n, best[(grid, tc, lib, n)]) for n in NODES if (grid, tc, lib, n) in best]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, linestyle=ls, linewidth=1.4,
                        markersize=5, markerfacecolor=c if tc == "intel" else "white")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(NODES)
    ax.set_xticklabels([str(n) for n in NODES])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Time per step, s")
    handles = [Line2D([], [], color=STYLE[l][0], marker=STYLE[l][1], linestyle="none", label=l) for l in LIBS]
    handles += [Line2D([], [], color=BLACK, linestyle=TC[t][0], label=TC[t][1]) for t in TC]
    ax.legend(handles=handles, ncol=2, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_toolchain_mn_{grid}.{ext}"))
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/guohaojie/results/toolchain_mn.csv")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(path))
    os.makedirs(out, exist_ok=True)
    setup_style()
    best = best_points(load(path))
    grids = sorted({k[0] for k in best}, key=lambda g: int(g) if g.isdigit() else 0)
    for g in grids:
        plot_grid(best, g, out)
        print(f"written: {out}/fig_toolchain_mn_{g}.(png|pdf)")


if __name__ == "__main__":
    main()
