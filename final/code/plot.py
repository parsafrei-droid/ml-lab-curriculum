import argparse
import csv
import glob
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

BASELINE_COLOR = "#3b6fd4"
CURRICULUM_COLOR = "#e8862e"


def read_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run_config(run_dir):
    path = pathlib.Path(run_dir) / "config.yaml"
    return yaml.safe_load(open(path)) if path.exists() else {}


# Same FLOP model as train.py, recomputed per step from the feature counts a run
# actually processed, so we can put the curves on a compute axis after the fact.
def cumulative_flops(feats, cfg):
    e = cfg.get("embedding_size", 96)
    h = cfg.get("hidden_size", 192)
    layers = cfg.get("layers", 3)
    accum = cfg.get("grad_accum", 32)
    npts = cfg.get("num_datapoints", 200)
    interval = cfg.get("eval_every", 100)
    total = 0.0
    out = []
    for feat in feats:
        cols = feat + 1
        per = cols * npts * npts * e + npts * cols * cols * e + 2 * npts * cols * e * h
        total += 3.0 * layers * per * accum * interval
        out.append(total)
    return out


# Our runs come in two vintages: early ones stored a bare AUC per dataset, later ones
# stored a dict of metrics. We read both.
def dataset_entry(value):
    if isinstance(value, dict):
        return value.get("roc_auc"), value.get("n_classes")
    return value, None


def group_runs(results_dir, skip=("_a100",)):
    groups = {}
    for path in glob.glob(str(pathlib.Path(results_dir) / "*" / "log.csv")):
        run_dir = pathlib.Path(path).parent
        if any(s in run_dir.name for s in skip):
            continue
        key = re.sub(r"_s\d+$", "", run_dir.name)
        groups.setdefault(key, []).append(run_dir)
    return groups


def load_scores(results_dir, skip=("_a100",)):
    scores = {}
    for path in glob.glob(str(pathlib.Path(results_dir) / "*" / "tabarena_scores.json")):
        name = pathlib.Path(path).parent.name
        if any(s in name for s in skip):
            continue
        key = re.sub(r"_s\d+$", "", name)
        try:
            scores.setdefault(key, []).append(json.load(open(path)))
        except Exception:
            continue
    return scores


# The headline figure. We find the best score the baseline ever reaches, then measure how
# much sooner the curriculum gets there. That gap is the arrow.
def iso_quality_curve(results_dir, out_dir, xcol, xlabel, unit, fname, scale=1.0):
    runs = {}
    for key, tag in (("baseline", "Baseline"), ("curriculum_features", "Curriculum")):
        matches = sorted(glob.glob(str(pathlib.Path(results_dir) / f"{key}_a100_s*" / "log.csv")))
        if not matches:
            matches = sorted(glob.glob(str(pathlib.Path(results_dir) / f"{key}_s*" / "log.csv")))
        if not matches:
            return
        rows = read_log(matches[0])
        x = np.array([float(r[xcol]) for r in rows]) / scale
        y = np.array([float(r["val_auc"]) for r in rows])
        runs[key] = (x, y, tag)

    bx, by, _ = runs["baseline"]
    cx, cy, _ = runs["curriculum_features"]
    target = by.max()
    reach = lambda xs, ys: next((xi for xi, yi in zip(xs, ys) if yi >= target), None)
    t_base, t_curr = reach(bx, by), reach(cx, cy)

    fig = plt.figure(figsize=(11.5, 5.4))
    ax = fig.add_axes([0.08, 0.13, 0.56, 0.78])
    for key, color in (("baseline", BASELINE_COLOR), ("curriculum_features", CURRICULUM_COLOR)):
        x, y, tag = runs[key]
        ax.plot(x, y, color=color, lw=2.4, marker="o", ms=3, mec="none", label=tag)

    if t_base and t_curr:
        ax.axhline(target, ls=":", c="#999999", lw=1)
        ax.annotate("", xy=(t_base, target), xytext=(t_curr, target),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6))
        ax.text(0.5 * (t_base + t_curr), target + 0.004,
                f"x = {t_base - t_curr:.0f} {unit}  ({t_base / t_curr:.1f}x sooner)",
                ha="center", va="bottom", fontsize=10, color="#333333")
        for tx in (t_curr, t_base):
            ax.plot([tx, tx], [by.min() - 0.01, target], ls=":", c="#bbbbbb", lw=1, zorder=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROC-AUC (validation)")
    ax.set_title("Same dump, same total compute: the curriculum reaches the\n"
                 "baseline's best quality far sooner", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.13)

    info = ["Dump info", "- fixed pool, 80,000 datasets", "- 2-60 features",
            "- 200 datapoints per table", "", "1. Baseline: random order of",
            "   the datasets", "2. Curriculum: order the",
            "   datasets according to number", "   of features (few -> many)"]
    fig.text(0.68, 0.88, "\n".join(info), va="top", ha="left", fontsize=11, family="monospace")
    fig.savefig(out_dir / fname, dpi=300)
    plt.close(fig)
    print(f"saved -> {out_dir / fname}")


# Validation AUC against compute instead of steps, averaged over seeds.
def compute_efficiency(results_dir, out_dir, fname="compute_efficiency.png"):
    groups = group_runs(results_dir)
    keys = [("baseline", "baseline (random order)", BASELINE_COLOR),
            ("curriculum_features", "feature curriculum (few to many)", CURRICULUM_COLOR)]
    if not all(k in groups for k, _, _ in keys):
        return
    plt.figure(figsize=(7.5, 5))
    for key, label, color in keys:
        xs, ys = [], []
        for run_dir in groups[key]:
            rows = read_log(run_dir / "log.csv")
            feats = [float(r["mean_features"]) for r in rows]
            xs.append(cumulative_flops(feats, run_config(run_dir)))
            ys.append([float(r["val_auc"]) if r["val_auc"] != "" else np.nan for r in rows])
        x, y, s = np.mean(xs, axis=0), np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
        plt.plot(x, y, marker="o", ms=4, lw=2, color=color, label=label)
        plt.fill_between(x, y - s, y + s, alpha=0.18, color=color, linewidth=0)
    plt.xscale("log")
    plt.xlabel("cumulative training compute - estimated FLOPs (log scale)")
    plt.ylabel("validation ROC-AUC (shared set)")
    plt.title("Same data, same total compute: the curriculum reaches good\n"
              "performance at a fraction of the compute")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()
    print(f"saved -> {out_dir / fname}")


# Where every ordering landed on TabArena, averaged over seeds.
def tabarena_summary(results_dir, out_dir, fname="tabarena_roc_auc.png"):
    scores = {k: [d["mean_roc_auc"] for d in v] for k, v in load_scores(results_dir).items()}
    scores = {k: v for k, v in scores.items() if all(x is not None for x in v)}
    if not scores:
        return
    items = sorted(scores.items(), key=lambda kv: np.mean(kv[1]))
    labels = [k for k, _ in items]
    means = np.array([np.mean(v) for _, v in items])
    stds = np.array([np.std(v) for _, v in items])
    base = np.mean(scores["baseline"]) if "baseline" in scores else means.mean()
    colors = ["#8a8f98" if k == "baseline" else (CURRICULUM_COLOR if means[i] >= base else "#c2544d")
              for i, k in enumerate(labels)]
    y = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.axvline(base, ls=":", c="#8a8f98", lw=1, label="baseline")
    plt.errorbar(means, y, xerr=stds, fmt="o", ms=7, capsize=4, lw=0, elinewidth=1.5,
                 ecolor="#aaaaaa", mfc="white", mec="none")
    for i in range(len(labels)):
        plt.plot(means[i], y[i], "o", ms=7, color=colors[i])
        plt.text(means[i], y[i] + 0.22, f"{means[i]:.3f}", ha="center", fontsize=9)
    plt.yticks(y, labels)
    plt.xlabel("TabArena mean ROC-AUC  (3 seeds, dot = mean, bar = std)")
    plt.xlim(means.min() - stds.max() - 0.02, means.max() + stds.max() + 0.02)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()
    print(f"saved -> {out_dir / fname}")


# Same ranking, but split into the binary and multiclass tasks to check it is not an
# artefact of which datasets got scored.
def tabarena_split(results_dir, out_dir, classes, fname="tabarena_binary_vs_all.png"):
    scopes = [("all", "#333333"), ("binary", BASELINE_COLOR), ("multiclass", "#c2544d")]
    groups = {}
    for key, docs in load_scores(results_dir).items():
        for doc in docs:
            per = doc.get("per_dataset", {})
            if not per:
                continue
            vals = {}
            for name, entry in per.items():
                auc, k = dataset_entry(entry)
                if auc is None:
                    continue
                vals[name] = (auc, k if k is not None else classes.get(name))
            picks = {
                "all": [a for a, _ in vals.values()],
                "binary": [a for a, k in vals.values() if k == 2],
                "multiclass": [a for a, k in vals.values() if k and k > 2],
            }
            groups.setdefault(key, {s: [] for s, _ in scopes})
            for s, _ in scopes:
                if picks[s]:
                    groups[key][s].append(np.mean(picks[s]))
    if not groups:
        return
    order = sorted(groups, key=lambda k: np.mean(groups[k]["all"]))
    y = np.arange(len(order))
    plt.figure(figsize=(8.5, 5.5))
    for j, (scope, color) in enumerate(scopes):
        means = np.array([np.mean(groups[k][scope]) for k in order])
        stds = np.array([np.std(groups[k][scope]) for k in order])
        plt.errorbar(means, y + (j - 1) * 0.24, xerr=stds, fmt="o", ms=6, capsize=3, lw=0,
                     elinewidth=1.2, color=color, label=scope)
    if "baseline" in groups:
        plt.axvline(np.mean(groups["baseline"]["all"]), ls=":", c="#888888", lw=1)
    plt.yticks(y, [k.replace("curriculum_", "") for k in order])
    plt.xlabel("TabArena ROC-AUC  (3 seeds, dot = mean, bar = std)")
    plt.title("Does the result depend on the evaluation subset?")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()
    print(f"saved -> {out_dir / fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(pathlib.Path(__file__).parent / "results"))
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent / "figures"))
    parser.add_argument("--classes", default=None)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = {}
    if args.classes and pathlib.Path(args.classes).exists():
        classes = {t["name"]: t["n_classes"]
                   for t in json.load(open(args.classes))["tasks"]}

    iso_quality_curve(args.results, out_dir, "cum_time_s", "Pretraining time (s)", "s",
                      "time_curve.png")
    iso_quality_curve(args.results, out_dir, "cum_flops", "Pretraining FLOPs", "PFLOP",
                      "flops_curve.png", scale=1e15)
    compute_efficiency(args.results, out_dir)
    tabarena_summary(args.results, out_dir)
    tabarena_split(args.results, out_dir, classes)


if __name__ == "__main__":
    main()
