"""Pull every scenario's results together into the poster figures.

After all three of us have run our scenarios (and their eval_tabarena.py), one
person runs this. It scans results/*/ and produces:

    experiments/comparison_loss.png      - all loss curves on one axis
    experiments/comparison_tabarena.png  - mean TabArena ROC-AUC per scenario
    experiments/comparison_summary.csv   - one row per scenario, the numbers

The whole poster comparison is these three files. Missing pieces are skipped, so
you can run it early to see partial results.

    python scripts/compare_results.py
"""

import csv
import json
import pathlib
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
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        run = {"name": d.name, "dir": d, "meta": json.loads(meta_path.read_text())}

        loss_csv = d / "loss.csv"
        if loss_csv.exists():
            epochs, losses = [], []
            with loss_csv.open() as f:
                for row in csv.DictReader(f):
                    epochs.append(int(row["epoch"]))
                    losses.append(float(row["loss"]))
            run["epochs"], run["losses"] = epochs, losses

        scores_path = d / "tabarena_scores.json"
        if scores_path.exists():
            run["scores"] = json.loads(scores_path.read_text())
        runs.append(run)
    return runs


def plot_loss(runs):
    have = [r for r in runs if r.get("losses")]
    if not have:
        return
    plt.figure(figsize=(7, 5))
    for r in have:
        plt.plot(r["epochs"], r["losses"], marker="o", ms=3, label=r["name"])
    plt.xlabel("epoch")
    plt.ylabel("mean loss")
    plt.title("Training loss by scenario")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "comparison_loss.png", dpi=120)
    plt.close()


def plot_tabarena(runs):
    have = [r for r in runs if r.get("scores")]
    if not have:
        print("  (no tabarena_scores.json yet - run eval_tabarena.py first)")
        return
    names = [r["name"] for r in have]
    aucs = [r["scores"]["mean_roc_auc"] for r in have]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, aucs, color="tab:blue")
    for b, a in zip(bars, aucs):
        plt.text(b.get_x() + b.get_width() / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("mean TabArena ROC-AUC")
    plt.title("Final performance by scenario")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "comparison_tabarena.png", dpi=120)
    plt.close()


def write_summary(runs):
    cols = ["name", "total_steps", "elapsed_s", "sec_per_step", "peak_gpu_gb",
            "final_loss", "mean_roc_auc"]
    with (OUT / "comparison_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        print("\n" + " | ".join(f"{c:>13}" for c in cols))
        for r in runs:
            m = r["meta"]
            auc = r["scores"]["mean_roc_auc"] if r.get("scores") else ""
            row = [r["name"], m.get("total_steps"), m.get("elapsed_s"),
                   m.get("sec_per_step"), m.get("peak_gpu_gb"),
                   m.get("final_loss"), round(auc, 4) if auc != "" else ""]
            w.writerow(row)
            print(" | ".join(f"{str(v):>13}" for v in row))


def main():
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        print(f"no results found under {RESULTS}/ - run scripts/run.py first")
        return
    print(f"found {len(runs)} run(s): {', '.join(r['name'] for r in runs)}")
    plot_loss(runs)
    plot_tabarena(runs)
    write_summary(runs)
    print(f"\nsaved figures + summary -> {OUT}/")


if __name__ == "__main__":
    main()
