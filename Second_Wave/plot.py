import csv
import glob
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = pathlib.Path(__file__).parent


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
    runs = load_runs()
    if not runs:
        return
    plt.figure(figsize=(7, 5))
    for key in sorted(runs):
        rows = runs[key][0]
        x = col(rows, "cum_flops")
        y = col(rows, "val_auc")
        plt.plot(x, y, marker="o", ms=3, label=key)
    plt.xlabel("cumulative FLOPs (estimated)")
    plt.ylabel("validation ROC-AUC (shared set)")
    plt.legend()
    plt.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / fname, dpi=120)
    plt.close()
    print(f"saved -> {out / fname}")


def main():
    compare("val_auc", "validation ROC-AUC (shared set)", "val_auc.png")
    compare("val_loss", "validation loss (shared set)", "val_loss.png")
    compare("mean_features", "features per step", "feature_ramp.png")
    compare("mean_context", "in-context training examples per step", "context_ramp.png")
    compare_vs_flops("val_auc_vs_flops.png")


if __name__ == "__main__":
    main()
