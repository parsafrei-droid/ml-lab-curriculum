import json
import pathlib
import sys

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent.parent / "TFM-Playground"))
sys.path.insert(0, str(BASE.parent.parent / "tabicl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openml
from openml.tasks import TaskType

from tfmplayground.evaluation import TABARENA_TASKS

POOL = {"features": (2, 60), "context": (20, 180), "classes": (2, 10)}
MAX_N_SAMPLES = 5000
MAX_N_FEATURES = 500


def collect():
    rows = []
    for task_id in TABARENA_TASKS:
        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue
            dataset = task.get_dataset(download_data=False)
            q = dataset.qualities
            n_features = q["NumberOfFeatures"]
            n_samples = q["NumberOfInstances"]
            if n_features > MAX_N_FEATURES or n_samples > MAX_N_SAMPLES:
                continue
            rows.append({
                "task_id": task_id,
                "name": dataset.name,
                "n_features": int(n_features),
                "n_samples": int(n_samples),
                "n_classes": int(q.get("NumberOfClasses") or 0),
                "n_train": int(round(0.9 * n_samples)),
            })
            print(f"  {dataset.name:28s} feat {int(n_features):4d}  rows {int(n_samples):5d}  "
                  f"classes {int(q.get('NumberOfClasses') or 0):2d}", flush=True)
        except Exception as e:
            print(f"  task {task_id}: skipped ({type(e).__name__})", flush=True)
    return rows


def summarise(rows, key):
    v = np.array([r[key] for r in rows])
    return {
        "min": int(v.min()), "q25": int(np.percentile(v, 25)), "median": int(np.median(v)),
        "q75": int(np.percentile(v, 75)), "max": int(v.max()),
    }


def figure(rows):
    panels = [
        ("n_features", "number of features", POOL["features"]),
        ("n_train", "in-context training examples", POOL["context"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, label, span) in zip(axes, panels):
        v = [r[key] for r in rows]
        ax.hist(v, bins=20, color="tab:blue", alpha=0.75, label="TabArena tasks")
        ax.axvspan(span[0], span[1], color="tab:orange", alpha=0.25, label="pretraining pool")
        ax.set_xlabel(label)
        ax.set_ylabel("tasks")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
    fig.suptitle("Evaluation regime (TabArena) vs pretraining regime (our pool)")
    fig.tight_layout()
    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "regime_gap.png", dpi=300)
    plt.close(fig)
    print(f"saved -> {out / 'regime_gap.png'}")


def main():
    cached = BASE / "figures" / "regime_gap.json"
    if "--replot" in sys.argv and cached.exists():
        figure(json.loads(cached.read_text())["tasks"])
        return
    print(f"profiling {len(TABARENA_TASKS)} TabArena tasks "
          f"(same filters as evaluate.py: <={MAX_N_SAMPLES} rows, <={MAX_N_FEATURES} features)", flush=True)
    rows = collect()
    if not rows:
        print("no tasks collected")
        return
    summary = {k: summarise(rows, k) for k in ("n_features", "n_samples", "n_train", "n_classes")}
    binary = sum(1 for r in rows if r["n_classes"] == 2) / len(rows)
    summary["n_tasks"] = len(rows)
    summary["binary_share"] = round(binary, 3)
    summary["pool_regime"] = POOL
    (BASE / "figures").mkdir(exist_ok=True)
    (BASE / "figures" / "regime_gap.json").write_text(json.dumps(
        {"summary": summary, "tasks": rows}, indent=2))
    print("\n=== evaluation regime ===")
    for k, s in summary.items():
        if isinstance(s, dict) and "median" in s:
            print(f"{k:12s} min {s['min']:6d} | q25 {s['q25']:6d} | median {s['median']:6d} "
                  f"| q75 {s['q75']:6d} | max {s['max']:6d}")
    print(f"tasks scored: {len(rows)} | binary share: {binary * 100:.0f}%")
    print(f"pretraining pool: features {POOL['features']}, in-context {POOL['context']}, classes {POOL['classes']}")
    figure(rows)


if __name__ == "__main__":
    main()
