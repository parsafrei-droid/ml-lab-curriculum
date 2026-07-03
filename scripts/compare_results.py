"""Pull every scenario's results together into the poster figures.

Runs are named <scenario>_s<seed> (e.g. baseline_s42, baseline_s43, ...). We
group by scenario and report mean +/- std across seeds, so the comparison has
error bars instead of single noisy points.

After everyone has run their scenarios (train + eval_tabarena) and committed
results/, one person runs this. It scans results/*/ and produces:

    experiments/comparison_val_loss.png   - shared-validation loss curves (comparable)
    experiments/comparison_roc_auc.png    - mean TabArena ROC-AUC, error bars
    experiments/comparison_summary.csv    - the numbers, one row per scenario

Missing pieces are skipped, so you can run it early for partial results.

    python scripts/compare_results.py
"""

import csv
import json
import pathlib
import re
import statistics
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import matplotlib.pyplot as plt

RESULTS = BASE / "results"
OUT = BASE / "experiments"


def load_runs():
    """One dict per scenario folder that has at least a meta.json."""
    runs = []
    for d in sorted(RESULTS.glob("*")):
        if not (d / "meta.json").exists():
            continue
        run = {"name": d.name, "meta": json.loads((d / "meta.json").read_text())}

        val_csv = d / "val.csv"
        if val_csv.exists():
            v_epochs, v_losses = [], []
            with val_csv.open() as f:
                for row in csv.DictReader(f):
                    v_epochs.append(int(row["epoch"]))
                    v_losses.append(float(row["val_loss"]))
            run["val_epochs"], run["val_losses"] = v_epochs, v_losses

        scores_path = d / "tabarena_scores.json"
        if scores_path.exists():
            run["roc_auc"] = json.loads(scores_path.read_text())["mean_roc_auc"]
        runs.append(run)
    return runs


def base_name(name):
    """baseline_s42 -> baseline ; leaves un-seeded names untouched."""
    return re.sub(r"_s\d+$", "", name)


def group_runs(runs):
    groups = {}
    for r in runs:
        groups.setdefault(base_name(r["name"]), []).append(r)
    return groups


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    return sum(vals) / len(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def plot_val_loss(groups):
    """One averaged validation-loss curve per scenario (the comparable metric)."""
    have = {b: rs for b, rs in groups.items() if any(r.get("val_losses") for r in rs)}
    if not have:
        print("  (no val.csv yet - rerun training so it logs the shared validation loss)")
        return
    plt.figure(figsize=(7, 5))
    for base, rs in sorted(have.items()):
        curves = [r["val_losses"] for r in rs if r.get("val_losses")]
        length = min(len(c) for c in curves)
        mean_curve = [sum(c[i] for c in curves) / len(curves) for i in range(length)]
        epochs = next(r["val_epochs"] for r in rs if r.get("val_epochs"))[:length]
        plt.plot(epochs, mean_curve, marker="o", ms=3, label=base)
    plt.xlabel("epoch")
    plt.ylabel("validation loss (shared set, mean over seeds)")
    plt.title("Comparable validation loss by scenario")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "comparison_val_loss.png", dpi=120)
    plt.close()


def plot_roc_auc(groups):
    names, means, stds = [], [], []
    for base, rs in sorted(groups.items()):
        m, s = mean_std([r.get("roc_auc") for r in rs])
        if m is not None:
            names.append(base)
            means.append(m)
            stds.append(s)
    if not names:
        print("  (no tabarena_scores.json yet - run eval_tabarena.py first)")
        return
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color="tab:blue")
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("mean TabArena ROC-AUC (error bars = std over seeds)")
    plt.title("Final performance by scenario")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "comparison_roc_auc.png", dpi=120)
    plt.close()


def rnd(x):
    return round(x, 4) if x is not None else ""


def write_summary(groups):
    cols = ["scenario", "n_seeds", "val_loss_mean", "val_loss_std",
            "val_acc_mean", "roc_auc_mean", "roc_auc_std"]
    with (OUT / "comparison_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        print("\n" + " | ".join(f"{c:>13}" for c in cols))
        for base, rs in sorted(groups.items()):
            vl_m, vl_s = mean_std([r["meta"].get("final_val_loss") for r in rs])
            va_m, _ = mean_std([r["meta"].get("final_val_acc") for r in rs])
            au_m, au_s = mean_std([r.get("roc_auc") for r in rs])
            row = [base, len(rs), rnd(vl_m), rnd(vl_s), rnd(va_m), rnd(au_m), rnd(au_s)]
            w.writerow(row)
            print(" | ".join(f"{str(v):>13}" for v in row))


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        print(f"no results found under {RESULTS}/ - run scripts/run.py first")
        return
    groups = group_runs(runs)
    print(f"found {len(runs)} run(s) in {len(groups)} scenario(s): {', '.join(sorted(groups))}")
    plot_val_loss(groups)
    plot_roc_auc(groups)
    write_summary(groups)
    print(f"\nsaved figures + summary -> {OUT}/")


if __name__ == "__main__":
    main()
