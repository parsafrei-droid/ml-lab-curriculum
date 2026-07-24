import argparse
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "TFM-Playground"))
sys.path.insert(0, str(BASE.parent / "tabicl"))

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

from tfmplayground.evaluation import TABARENA_TASKS, get_openml_predictions
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.utils import get_default_device


def per_dataset_metrics(y_true, y_pred, y_proba):
    classes = np.unique(y_true)
    k = len(classes)
    m = {"n_classes": int(k)}
    try:
        if k == 2:
            m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            m["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    except Exception:
        m["roc_auc"] = None
    try:
        m["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
    except Exception:
        m["log_loss"] = None
    try:
        m["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        m["balanced_acc"] = None
    return m


def aggregate(per, keep):
    subset = {n: m for n, m in per.items() if keep(m)}
    out = {"n_datasets": len(subset)}
    for metric in ("roc_auc", "log_loss", "balanced_acc"):
        vals = [m[metric] for m in subset.values() if m[metric] is not None]
        out[metric] = float(sum(vals) / len(vals)) if vals else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-n-samples", type=int, default=5000)
    args = parser.parse_args()

    classifier = NanoTabPFNClassifier(model=args.checkpoint, device=get_default_device())
    print(f"evaluating {args.checkpoint} on {len(TABARENA_TASKS)} tasks", flush=True)

    predictions = get_openml_predictions(
        model=classifier, classification=True, tasks=TABARENA_TASKS,
        max_n_samples=args.max_n_samples,
    )

    per = {}
    for name, (y_true, y_pred, y_proba) in predictions.items():
        per[name] = per_dataset_metrics(y_true, y_pred, y_proba)
        m = per[name]
        print(f"  {name:30s} k={m['n_classes']:<2d} auc={m['roc_auc']!s:>7.7} "
              f"logloss={m['log_loss']!s:>7.7} bacc={m['balanced_acc']!s:>7.7}", flush=True)

    summary = {
        "all": aggregate(per, lambda m: True),
        "binary": aggregate(per, lambda m: m["n_classes"] == 2),
        "multiclass": aggregate(per, lambda m: m["n_classes"] > 2),
    }
    out = {
        "checkpoint": args.checkpoint,
        "mean_roc_auc": summary["all"]["roc_auc"],
        "summary": summary,
        "per_dataset": per,
    }
    out_path = pathlib.Path(args.checkpoint).parent / "tabarena_scores.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== summary (mean over datasets) ===", flush=True)
    for scope, s in summary.items():
        print(f"{scope:11s} n={s['n_datasets']:<3d} roc_auc={s['roc_auc']!s:>7.7} "
              f"log_loss={s['log_loss']!s:>7.7} balanced_acc={s['balanced_acc']!s:>7.7}", flush=True)
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
