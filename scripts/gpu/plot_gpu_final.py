#!/usr/bin/env python3
"""plot_gpu_final.py [gpu_final.csv] [--jobs "<id> <id> ..."] [--out DIR] [--fmt pdf,png]

Three figures from gpu_final.sbatch's CSV (status OK rows only), data only (no titles, no annotations):
  gpu_final_single.<fmt>  1 GPU: time per step vs grid size, cuFFT and heFFTe gpu1
  gpu_final_node.<fmt>    1 node: time per step vs GPUs (2/4/8), one panel per grid, cuFFTXt, cuFFTMp, heFFTe mgpu
  gpu_final_multi.<fmt>   8 GPUs per node: time per step vs nodes, one panel per grid, cuFFTMp, heFFTe multigpu
Solid line, filled marker: whole step; dashed line, open marker: its FFT phase ("FFT (incl. transpose/comm)").
Points are medians, error bars the min-max range of the repetitions. A configuration without OK runs (OOM) has no point.
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# fixed colour per library (validated palette: blue, orange, aqua); markers give a second channel
STYLE = {
    "cufft":           ("cuFFT",          "#2a78d6", "o"),
    "cufftmp":         ("cuFFTMp",        "#2a78d6", "o"),
    "heffte_gpu1":     ("heFFTe",         "#eb6834", "s"),
    "heffte_mgpu":     ("heFFTe",         "#eb6834", "s"),
    "heffte_multigpu": ("heFFTe",         "#eb6834", "s"),
    "cufftxt":         ("cuFFTXt",        "#1baf7a", "^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path, jobs):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if jobs and r["jobid"] not in jobs:
                continue
            if r.get("status") != "OK" or not r.get("step_s"):
                continue
            rows.append(r)
    return rows


def agg(rows, mode, xkey, panel_key=None):
    """{panel: {lib: [(x, med, lo, hi, fmed, flo, fhi)]}}"""
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ([], []))))
    for r in rows:
        if r["mode"] != mode:
            continue
        p = r[panel_key] if panel_key else ""
        s, f = d[p][r["lib"]][int(r[xkey])]
        s.append(float(r["step_s"]))
        if r.get("fft_s"):
            f.append(float(r["fft_s"]))
    out = {}
    for p, libs in d.items():
        out[p] = {}
        for lib, xs in libs.items():
            pts = []
            for x in sorted(xs):
                s, f = xs[x]
                fm = (statistics.median(f), min(f), max(f)) if f else (None, None, None)
                pts.append((x, statistics.median(s), min(s), max(s)) + fm)
            out[p][lib] = pts
    return out


def style_axes(ax, xlabel, xticks):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def draw(ax, series, order):
    for lib in order:
        if lib not in series:
            continue
        name, col, mk = STYLE[lib]
        pts = series[lib]
        x = [p[0] for p in pts]
        m = [p[1] for p in pts]
        err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        ax.errorbar(x, m, yerr=err, color=col, marker=mk, markersize=6, linewidth=2, capsize=3,
                    markeredgecolor="white", markeredgewidth=1, label=f"{name}, step")
        fp = [p for p in pts if p[4] is not None]
        if fp:
            ax.errorbar([p[0] for p in fp], [p[4] for p in fp],
                        yerr=[[p[4] - p[5] for p in fp], [p[6] - p[4] for p in fp]],
                        color=col, marker=mk, markersize=6, linewidth=1.5, linestyle="--", capsize=3,
                        markerfacecolor="white", markeredgecolor=col, label=f"{name}, FFT")


def legend(fig, axes):
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    if h:
        fig.legend(h, l, loc="upper center", ncol=max(1, len(l) // 2), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 1.0))


def save(fig, out, name, fmts):
    for fm in fmts:
        p = os.path.join(out, f"{name}.{fm}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.expanduser("~/guohaojie/results/gpu_final.csv"))
    ap.add_argument("--jobs", default="", help="job ids, space separated (default: all rows)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--fmt", default="pdf,png")
    a = ap.parse_args()
    rows = load(a.csv, set(a.jobs.split()))
    fmts = [f for f in a.fmt.split(",") if f]
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})

    # 1. single GPU
    d = agg(rows, "single", "grid").get("", {})
    if d:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        draw(ax, d, ["cufft", "heffte_gpu1"])
        grids = sorted({p[0] for v in d.values() for p in v})
        style_axes(ax, "grid size N (N³ points)", grids)
        ax.set_ylabel("time per step (s)", color=INK)
        legend(fig, [ax])
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        save(fig, a.out, "gpu_final_single", fmts)
    else:
        print("no OK single rows")

    # 2. one node, 2/4/8 GPUs; 3. several nodes, 8 GPUs each
    for mode, xkey, xlabel, order, name in (
            ("node", "gpus", "GPUs (one node)", ["cufftxt", "cufftmp", "heffte_mgpu"], "gpu_final_node"),
            ("multi", "nodes", "nodes (8 GPUs each)", ["cufftmp", "heffte_multigpu"], "gpu_final_multi")):
        dd = agg(rows, mode, xkey, "grid")
        if not dd:
            print(f"no OK {mode} rows")
            continue
        panels = sorted(dd, key=int)
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.4), squeeze=False)
        axes = axes[0]
        for ax, g in zip(axes, panels):
            draw(ax, dd[g], order)
            xs = sorted({p[0] for v in dd[g].values() for p in v})
            style_axes(ax, xlabel, xs)
            ax.text(0.03, 0.04, f"{g}³", transform=ax.transAxes, color=INK, fontsize=10)
        axes[0].set_ylabel("time per step (s)", color=INK)
        legend(fig, axes)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        save(fig, a.out, name, fmts)


if __name__ == "__main__":
    main()
