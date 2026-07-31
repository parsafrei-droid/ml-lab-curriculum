#!/usr/bin/env python3
"""Dashboard for the project's compute history: hardware used and compute per phase."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "compute_summary.json")))
OUT = os.path.join(HERE, "compute_dashboard.png")
H = 3600.0

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
})

fig = plt.figure(figsize=(15.5, 7.2), dpi=200)
gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.135, right=0.975, top=0.700, bottom=0.115)

fig.text(0.017, 0.962, "Compute history — curriculum learning for tabular foundation models",
         fontsize=21, fontweight="bold", color=INK, va="top")
fig.text(0.017, 0.905,
         f"{S['n_slurm_jobs']} Slurm jobs on BwUniCluster  +  {S['n_kaggle_runs']} Kaggle notebook runs   ·   "
         f"{S['first_job']} to {S['last_job']}  ({S['span_days']} days, {S['active_days']} with activity)",
         fontsize=12.5, color=INK2, va="top")

for i, (val, lab) in enumerate([
    (f"{S['total_seconds']/H:.1f} h", "total GPU run time"),
    (f"{S['n_slurm_jobs'] + S['n_kaggle_runs']}", "total runs"),
    (f"{S['queue_seconds']/H:.0f} h", "spent waiting in the queue"),
]):
    x = 0.017 + i * 0.185
    fig.text(x, 0.855, val, fontsize=27, fontweight="bold",
             color=BLUE if i < 2 else ORANGE, va="top")
    fig.text(x, 0.788, lab, fontsize=11, color=MUTED, va="top")

# ------------------------------------------------- (a) GPU hours by hardware
ax = fig.add_subplot(gs[0, 0])
g = {k: v for k, v in S["by_gpu"].items() if v["sec"] > 0.05 * H}
items = sorted(g.items(), key=lambda x: x[1]["sec"])
labels = [k.replace(" (Kaggle)", "") for k, _ in items]
vals = [v["sec"] / H for _, v in items]
ns = [v["n"] for _, v in items]
cols = [ORANGE if "T4" in k else BLUE for k, _ in items]
b = ax.barh(labels, vals, color=cols, height=0.58)
for r, v, n in zip(b, vals, ns):
    ax.text(v + max(vals) * 0.022, r.get_y() + r.get_height() / 2,
            f"{v:.2f} h   ({n} runs)", va="center", fontsize=12, color=INK)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.yaxis.grid(False)
ax.set_axisbelow(True)
ax.tick_params(length=0, labelsize=11.5)
ax.set_xlim(0, max(vals) * 1.36)
ax.set_xlabel("GPU run time (hours)", fontsize=11.5, color=INK2, labelpad=8)
ax.set_title("a.  Where the compute ran", fontsize=15, fontweight="bold",
             color=INK, loc="left", pad=13)
ax.legend(handles=[Patch(facecolor=BLUE, label="BwUniCluster (Slurm)"),
                   Patch(facecolor=ORANGE, label="Kaggle (notebook)")],
          loc="lower right", frameon=False, fontsize=11.5, labelcolor=INK2)

# ------------------------------------------------------ (b) hours per phase
ax = fig.add_subplot(gs[0, 1])
w = sorted(S["by_wave"].items())
short = ["Setup &\nGPU probes", "First Wave\n(TabICL)", "Second Wave\n(nanoTabPFN)", "Third Wave\n(modded-nano)"]
vals = [v["sec"] / H for _, v in w]
ns = [v["n"] for _, v in w]
b = ax.bar(short, vals, color=BLUE, width=0.54)
kag = S["kaggle_seconds"] / H
ax.bar(short[2], kag, bottom=vals[2], color=ORANGE, width=0.54)
for i, (r, v, n) in enumerate(zip(b, vals, ns)):
    top = v + (kag if i == 2 else 0)
    extra = f"\n+{kag:.1f} h Kaggle" if i == 2 else ""
    ax.text(r.get_x() + r.get_width() / 2, top + max(vals) * 0.035,
            f"{v:.1f} h\n{n} jobs{extra}", ha="center", va="bottom",
            fontsize=11.5, color=INK, linespacing=1.5)
ax.set_ylim(0, max(vals) * 1.44)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.tick_params(axis="x", labelsize=11.5, colors=INK2)
ax.set_ylabel("GPU run time (hours)", fontsize=11.5, color=INK2, labelpad=8)
ax.set_title("b.  Compute per project phase", fontsize=15, fontweight="bold",
             color=INK, loc="left", pad=13)

fig.text(0.017, 0.030,
         "Source: Slurm accounting (sacct) and per-run cum_time_s from the Kaggle notebook logs.",
         fontsize=10, color=MUTED)

fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.3)
print("wrote", OUT)
