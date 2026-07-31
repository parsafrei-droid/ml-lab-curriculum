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

BASE = pathlib.Path(__file__).parent


def run_config(run_dir):
    path = pathlib.Path(run_dir) / "config.yaml"
    return yaml.safe_load(open(path)) if path.exists() else {}


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


def read_log(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, name):
    return [float(r.get(name, "")) if r.get(name, "") != "" else np.nan for r in rows]


def plot_run(out_dir):
    out_dir = pathlib.Path(out_dir)
    rows = read_log(out_dir / "log.csv")
    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(steps, col(rows, "val_auc"), marker="o", ms=3)
    ax[0].set_xlabel("pretraining step")
    ax[0].set_ylabel("validation ROC-AUC")
    ax[1].plot(steps, col(rows, "val_loss"), marker="o", ms=3, color="tab:orange")
    ax[1].set_xlabel("pretraining step")
    ax[1].set_ylabel("validation loss")
    fig.suptitle(out_dir.name)
    fig.tight_layout()
    fig.savefig(out_dir / "curve.png", dpi=120)
    plt.close(fig)


def load_runs():
    runs = {}
    for path in glob.glob(str(BASE / "results" / "*" / "log.csv")):
        name = pathlib.Path(path).parent.name
        key = re.sub(r"_s\d+$", "", name)
        runs.setdefault(key, []).append(read_log(path))
    return runs


def series(run_list, name):
    steps = [int(r["step"]) for r in run_list[0]]
    values = np.array([col(rows, name) for rows in run_list])
    return steps, np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def compare(metric, ylabel, fname):
    runs = load_runs()
    if not runs:
        print("no runs found in results/")
        return
    plt.figure(figsize=(7, 5))
    for key in sorted(runs):
        steps, m, s = series(runs[key], metric)
        line, = plt.plot(steps, m, marker="o", ms=3, label=key)
        plt.fill_between(steps, m - s, m + s, alpha=0.15, color=line.get_color())
    plt.xlabel("pretraining step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / fname, dpi=120)
    plt.close()
    print(f"saved -> {out / fname}")


def compare_vs_flops(fname):
    dirs = {}
    for path in glob.glob(str(BASE / "results" / "*" / "log.csv")):
        run_dir = pathlib.Path(path).parent
        key = re.sub(r"_s\d+$", "", run_dir.name)
        dirs.setdefault(key, []).append(run_dir)
    if not dirs:
        return
    plt.figure(figsize=(7, 5))
    for key in sorted(dirs):
        curves = []
        for run_dir in dirs[key]:
            rows = read_log(run_dir / "log.csv")
            feats = [float(r["mean_features"]) for r in rows]
            x = cumulative_flops(feats, run_config(run_dir))
            y = [float(r["val_auc"]) if r["val_auc"] != "" else np.nan for r in rows]
            curves.append((x, y))
        x = np.mean([c[0] for c in curves], axis=0)
        y = np.nanmean([c[1] for c in curves], axis=0)
        plt.plot(x, y, marker="o", ms=3, label=key)
    plt.xscale("log")
    plt.xlabel("cumulative FLOPs (estimated, log scale)")
    plt.ylabel("validation ROC-AUC (shared set)")
    plt.legend()
    plt.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / fname, dpi=120)
    plt.close()
    print(f"saved -> {out / fname}")


def flops_headline(fname, keys=("baseline", "curriculum_features"),
                   labels=("baseline (random order)", "feature curriculum (easy to hard)"),
                   colors=("#8a8f98", "#3b7dd8")):
    dirs = {}
    for path in glob.glob(str(BASE / "results" / "*" / "log.csv")):
        run_dir = pathlib.Path(path).parent
        k = re.sub(r"_s\d+$", "", run_dir.name)
        dirs.setdefault(k, []).append(run_dir)
    plt.figure(figsize=(7.5, 5))
    for key, label, color in zip(keys, labels, colors):
        if key not in dirs:
            continue
        xs, ys = [], []
        for run_dir in dirs[key]:
            rows = read_log(run_dir / "log.csv")
            feats = [float(r["mean_features"]) for r in rows]
            xs.append(cumulative_flops(feats, run_config(run_dir)))
            ys.append([float(r["val_auc"]) if r["val_auc"] != "" else np.nan for r in rows])
        x = np.mean(xs, axis=0)
        y = np.nanmean(ys, axis=0)
        s = np.nanstd(ys, axis=0)
        plt.plot(x, y, marker="o", ms=4, lw=2, color=color, label=label)
        plt.fill_between(x, y - s, y + s, alpha=0.18, color=color, linewidth=0)
    plt.xscale("log")
    plt.xlabel("cumulative training compute — estimated FLOPs (log scale)")
    plt.ylabel("validation ROC-AUC (shared set)")
    plt.title("Same data, same total compute: the curriculum reaches good\nperformance at a fraction of the compute")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(True, which="major", axis="both", alpha=0.15)
    plt.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / fname, dpi=140)
    plt.close()
    print(f"saved -> {out / fname}")


def tabarena_summary(fname="tabarena_roc_auc.png"):
    scores = {}
    for path in glob.glob(str(BASE / "results" / "*" / "tabarena_scores.json")):
        key = re.sub(r"_s\d+$", "", pathlib.Path(path).parent.name)
        try:
            scores.setdefault(key, []).append(json.load(open(path))["mean_roc_auc"])
        except Exception:
            continue
    if not scores:
        return
    items = sorted(scores.items(), key=lambda kv: np.mean(kv[1]))
    labels = [k for k, _ in items]
    means = np.array([np.mean(v) for _, v in items])
    stds = np.array([np.std(v) for _, v in items])
    base = np.mean(scores["baseline"]) if "baseline" in scores else means.mean()
    colors = ["#8a8f98" if k == "baseline" else ("#3b7dd8" if means[i] >= base else "#c2544d")
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
    lo, hi = means.min() - stds.max(), means.max() + stds.max()
    plt.xlim(lo - 0.02, hi + 0.02)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / fname, dpi=140)
    plt.close()
    print(f"saved -> {out / fname}")


def tabarena_split(fname="tabarena_binary_vs_all.png"):
    classes = {t["name"]: t["n_classes"]
               for t in json.load(open(BASE / "figures" / "regime_gap.json"))["tasks"]}
    scopes = [("all", "#333333"), ("binary", "#3b7dd8"), ("multiclass", "#c2544d")]
    groups = {}
    for path in glob.glob(str(BASE / "results" / "*" / "tabarena_scores.json")):
        key = re.sub(r"_s\d+$", "", pathlib.Path(path).parent.name)
        per = json.load(open(path)).get("per_dataset", {})
        if not per:
            continue
        picks = {
            "all": list(per.values()),
            "binary": [v for n, v in per.items() if classes.get(n) == 2],
            "multiclass": [v for n, v in per.items() if classes.get(n, 0) > 2],
        }
        groups.setdefault(key, {s: [] for s, _ in scopes})
        for s, _ in scopes:
            groups[key][s].append(np.mean(picks[s]))
    if not groups:
        return
    order = sorted(groups, key=lambda k: np.mean(groups[k]["all"]))
    y = np.arange(len(order))
    plt.figure(figsize=(8.5, 5.5))
    for j, (scope, color) in enumerate(scopes):
        off = (j - 1) * 0.24
        means = np.array([np.mean(groups[k][scope]) for k in order])
        stds = np.array([np.std(groups[k][scope]) for k in order])
        plt.errorbar(means, y + off, xerr=stds, fmt="o", ms=6, capsize=3, lw=0,
                     elinewidth=1.2, color=color, label=scope)
    if "baseline" in groups:
        plt.axvline(np.mean(groups["baseline"]["all"]), ls=":", c="#888888", lw=1)
    plt.yticks(y, [k.replace("curriculum_", "") for k in order])
    plt.xlabel("TabArena ROC-AUC  (3 seeds, dot = mean, bar = std)")
    plt.title("Does the result depend on the evaluation subset? (binary vs all vs multiclass)")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / fname, dpi=140)
    plt.close()
    print(f"saved -> {out / fname}")


def main():
    tabarena_summary("tabarena_roc_auc.png")
    tabarena_split("tabarena_binary_vs_all.png")
    flops_headline("compute_efficiency.png")
    compare("val_auc", "validation ROC-AUC (shared set)", "val_auc.png")
    compare("val_loss", "validation loss (shared set)", "val_loss.png")
    compare("mean_features", "features per step", "feature_ramp.png")
    compare("mean_context", "in-context training examples per step", "context_ramp.png")
    compare_vs_flops("val_auc_vs_flops.png")


if __name__ == "__main__":
    main()
