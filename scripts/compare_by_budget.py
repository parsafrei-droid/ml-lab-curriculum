"""Per-budget comparison figures + a convergence curve across training budgets.

Runs are named <scenario>[_e<epochs>]_s<seed>. We read the real step count from
each meta.json, so 2k / 5k / 10k / the paper's 2.5k all slot in automatically.

Produces, in experiments/:
  comparison_<label>_efficiency.png   - one per budget (ROC-AUC vs wall-clock)
  comparison_convergence.png          - ROC-AUC vs steps for the key scenarios

    python scripts/compare_by_budget.py
"""

import glob
import json
import pathlib
import re
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent.parent
OUT = BASE / "experiments"
KEY = ["baseline", "curriculum_combined", "curriculum_reverse"]  # convergence lines


def load():
    runs = []
    for m in glob.glob(str(BASE) + "/results/*_s*/meta.json"):
        d = json.loads(open(m).read())
        base = re.sub(r"(_e\d+)?_s\d+$", "", d["name"])
        r = {"base": base, "steps": d.get("total_steps"), "t": d.get("elapsed_s")}
        sc = m.replace("meta.json", "tabarena_scores.json")
        try:
            r["auc"] = json.loads(open(sc).read())["mean_roc_auc"]
        except Exception:
            r["auc"] = None
        runs.append(r)
    return runs


def ms(v):
    v = [x for x in v if x is not None]
    return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0) if v else (None, 0.0)


def label(steps):
    return f"{steps / 1000:g}k" if steps and steps >= 1000 else str(steps)


def colour(base):
    if base == "baseline":
        return "tab:red"
    if base == "paper_small":
        return "tab:purple"
    return "tab:green"


def efficiency(runs, steps):
    g = defaultdict(lambda: defaultdict(list))
    # this budget's runs, plus paper_small as a fixed reference in every figure
    for r in (x for x in runs if x["steps"] == steps or x["base"] == "paper_small"):
        g[r["base"]]["t"].append(r["t"])
        g[r["base"]]["auc"].append(r["auc"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for b in g:
        tm, ts = ms(g[b]["t"])
        am, ae = ms(g[b]["auc"])
        if tm is None or am is None:
            continue
        ax.errorbar(tm, am, xerr=ts, yerr=ae, fmt="o", ms=9, capsize=4, color=colour(b))
        ax.annotate(b.replace("curriculum_", ""), (tm, am),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set(xlabel="wall-clock compute (s, mean over seeds)", ylabel="TabArena ROC-AUC",
           title=f"{label(steps)} steps: compute efficiency (up-and-left = better)")
    fig.tight_layout()
    fig.savefig(OUT / f"comparison_{label(steps)}_efficiency.png", dpi=120)
    plt.close()


def convergence(runs):
    # distinct colours so the legend is readable (efficiency plots use red/green)
    line_colour = {"baseline": "tab:red", "curriculum_combined": "tab:green",
                   "curriculum_reverse": "tab:orange"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    drawn = False
    for b in KEY:
        pts = defaultdict(list)
        for r in runs:
            if r["base"] == b and r["auc"] is not None:
                pts[r["steps"]].append(r["auc"])
        if not pts:
            continue
        xs = sorted(pts)
        ys = [ms(pts[x])[0] for x in xs]
        es = [ms(pts[x])[1] for x in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=b.replace("curriculum_", ""),
                    color=line_colour.get(b), lw=2.2 if b == "baseline" else 1.8)
        drawn = True
    if not drawn:
        plt.close()
        return
    # paper_small as a horizontal reference line, if it's been run
    paper = [r["auc"] for r in runs if r["base"] == "paper_small" and r["auc"] is not None]
    if paper:
        ax.axhline(ms(paper)[0], ls="--", c="tab:purple", lw=1.4, label="paper_small (ref)")
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set(xlabel="training steps", ylabel="TabArena ROC-AUC",
           title="Convergence: does the curriculum gap hold with more training?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "comparison_convergence.png", dpi=120)
    plt.close()


def main():
    runs = load()
    budgets = sorted({r["steps"] for r in runs if r["steps"]})
    if not budgets:
        print("no runs found under results/")
        return
    print("budgets found:", ", ".join(label(b) for b in budgets))
    for s in budgets:
        efficiency(runs, s)
    convergence(runs)
    print(f"saved per-budget efficiency + convergence figures -> {OUT}/")


if __name__ == "__main__":
    main()
