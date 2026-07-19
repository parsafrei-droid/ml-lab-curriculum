"""Evaluate a trained checkpoint on TabArena (or the toy tasks) and save scores.

    python scripts/eval_tabarena.py --checkpoint results/scenario_A/checkpoint.pth
    python scripts/eval_tabarena.py --checkpoint results/scenario_A/checkpoint.pth --tasks tabarena

--tasks toy      -> 3 small sklearn-ish datasets, fast, for a sanity check.
--tasks tabarena -> every classification task in TABARENA_CLASSIFICATION_TASKS
                    (below) with <= --max-n-features columns (default 120);
                    tasks over --max-n-samples rows are stratified-subsampled
                    down to that budget (seeded by --subsample-seed) rather
                    than skipped, so nothing is dropped for being too tall.
                    Each task is evaluated ONCE.

Output is backward compatible with the original (pre-subsampling) version of
this script: "mean_roc_auc" and "per_dataset" are still there, same shape,
and equal to the "evaluate everything, once" pass (what used to be the only
pass). On top of that we now also report two means computed as subsets of
that same per-dataset scores, no re-evaluation:
  mean_roc_auc_binary - the subset that's binary classification.
  mean_roc_auc_16     - the subset that was NOT subsampled (originally <=
                        --max-n-samples rows) - i.e. what the model saw at
                        its real size, not a shrunk copy. At the defaults
                        (max-n-features=120, max-n-samples=5000) this is
                        exactly the ~16-dataset set the very first version of
                        this script (skip-only, no subsampling) evaluated.

TABARENA_CLASSIFICATION_TASKS is a hardcoded (task_id, dataset_name,
n_features, is_binary) table for tfmplayground.evaluation.TABARENA_TASKS's
classification tasks, rather than re-deriving task-type/feature-count/class-
count from OpenML metadata every run. Two reasons: (1) fewer OpenML round
trips per run (no more metadata probes for the ~13 non-classification tasks,
or the wide ones we're about to skip anyway) - the shared ~/.cache/openml
directory raced and corrupted under concurrent jobs twice already this
project; (2) is_binary becomes a ground-truth fact instead of something
inferred after training from how many classes survived into y_train, which
subsampling could silently shrink (a genuinely 3-class task could get
mistagged binary if a class dropped out of the train subsample).

Regenerate the table if TABARENA_TASKS changes upstream:

    import openml
    from openml.tasks import TaskType
    from tfmplayground.evaluation import TABARENA_TASKS
    for tid in TABARENA_TASKS:
        task = openml.tasks.get_task(tid, download_splits=False)
        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        ds = task.get_dataset(download_data=False)
        print(tid, ds.name, int(ds.qualities["NumberOfFeatures"]), len(task.class_labels) == 2)

Writes results/<name>/tabarena_scores.json.
"""

import argparse
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "TFM-Playground"))
sys.path.insert(0, str(BASE / "tabicl"))

import numpy as np
import openml
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.utils import get_default_device

# (task_id, dataset_name, n_features, is_binary) - see module docstring.
# Generated from tfmplayground.evaluation.TOY_TASKS_CLASSIFICATION.
TOY_CLASSIFICATION_TASKS = [
    (59, "iris", 5, False),
    (2382, "wine", 14, False),
    (9946, "wdbc", 31, True),
]

# Same, generated from tfmplayground.evaluation.TABARENA_TASKS.
TABARENA_CLASSIFICATION_TASKS = [
    (363613, "Amazon_employee_access", 10, True),
    (363614, "anneal", 39, False),
    (363616, "APSFailure", 171, True),
    (363618, "bank-marketing", 14, True),
    (363619, "Bank_Customer_Churn", 11, True),
    (363620, "Bioresponse", 1777, True),
    (363621, "blood-transfusion-service-center", 5, True),
    (363623, "churn", 20, True),
    (363624, "coil2000_insurance_policies", 86, True),
    (363626, "credit-g", 21, True),
    (363627, "credit_card_clients_default", 24, True),
    (363628, "customer_satisfaction_in_airline", 22, True),
    (363629, "diabetes", 9, True),
    (363630, "Diabetes130US", 48, True),
    (363632, "E-CommereShippingData", 11, True),
    (363671, "Fitness_Club", 7, True),
    (363673, "GiveMeSomeCredit", 11, True),
    (363674, "hazelnut-spread-contaminant-detection", 31, True),
    (363676, "heloc", 24, True),
    (363677, "hiva_agnostic", 1618, False),
    (363679, "HR_Analytics_Job_Change_of_Data_Scientists", 13, True),
    (363681, "in_vehicle_coupon_recommendation", 25, True),
    (363682, "Is-this-a-good-customer", 14, True),
    (363683, "kddcup09_appetency", 213, True),
    (363684, "Marketing_Campaign", 26, True),
    (363685, "maternal_health_risk", 7, False),
    (363689, "NATICUSdroid", 87, True),
    (363691, "online_shoppers_intention", 18, True),
    (363694, "polish_companies_bankruptcy", 65, True),
    (363696, "qsar-biodeg", 42, True),
    (363699, "SDSS17", 12, False),
    (363700, "seismic-bumps", 16, True),
    (363702, "splice", 61, False),
    (363704, "students_dropout_and_academic_success", 37, False),
    (363706, "taiwanese_bankruptcy_prediction", 95, True),
    (363707, "website_phishing", 10, False),
    (363711, "MIC", 112, False),
    (363712, "jm1", 22, True),
]


def _subsample_indices(indices, labels, n_keep, rng):
    """Shrink `indices` to `n_keep` entries, stratifying by `labels` when possible.

    Uniform random subsampling can wipe out a rare class on an imbalanced
    dataset, which is especially bad for ROC-AUC on a small test split. We
    stratify by label and only fall back to uniform sampling when a class is
    too small to stratify on (sklearn raises ValueError in that case).
    """
    if n_keep >= len(indices):
        return indices
    indices = np.asarray(indices)
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        kept, _ = train_test_split(
            indices, train_size=n_keep, stratify=np.asarray(labels), random_state=seed
        )
        return kept
    except ValueError:
        pass  # a class has too few members to stratify on - fall back below
    rng2 = np.random.default_rng(seed)
    return rng2.choice(indices, size=n_keep, replace=False)


@torch.no_grad()
def evaluate_classification_tasks(model, resolved_tasks, *, max_n_features=120, max_n_samples=5_000, subsample_seed=0):
    """Score `model` on each (task_id, name, n_features, is_binary) in `resolved_tasks`, once each.

    Tasks with more than `max_n_features` columns are skipped outright (a
    model's feature axis is fixed by what it was trained on). Tasks with more
    than `max_n_samples` rows are stratified-subsampled down to that budget
    (seeded by `subsample_seed`) rather than skipped.

    Returns: dict of dataset name -> dict(y_true, y_pred, y_proba, is_binary,
    was_subsampled). was_subsampled is False iff the task's original row count
    was already <= max_n_samples (nothing was thrown away for that task).
    is_binary is the caller-supplied ground truth, not inferred from what
    classes happened to survive subsampling into the training split.
    """
    rng = np.random.default_rng(subsample_seed)
    out = {}

    for task_id, name, n_features, is_binary in resolved_tasks:
        if n_features > max_n_features:
            continue  # skip task, too many features

        try:
            task = openml.tasks.get_task(task_id, download_splits=False)
            dataset = task.get_dataset(download_data=False)
            X, y, _categorical_indicator, _attribute_names = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )
            train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0)

            # Fit on the FULL label column before any subsampling below, so a class
            # that subsampling drops from one split can never look "unseen" to the
            # other split's .transform(). Fitting on y_train alone (the previous
            # approach) could raise "y contains previously unseen labels" whenever
            # a rare class survived into the (independently subsampled) test split
            # but not the train one.
            label_encoder = LabelEncoder().fit(y)
            n_classes_total = len(label_encoder.classes_)

            # A model's classification head is architecturally fixed-width at
            # training time (num_outputs, from the schedule's max max_classes) -
            # it CANNOT represent more classes than that, no matter what the
            # real dataset has. NanoTabPFNClassifier.predict_proba silently
            # clips (out[:, :self.num_classes] on a narrower tensor just
            # returns what's there, no error), so a too-wide task doesn't fail
            # until much later, confusingly, inside roc_auc_score. Check the
            # model's actual output width up front and skip cleanly instead.
            if n_classes_total > getattr(model.model, "num_outputs", n_classes_total):
                print(f"  {name:40s} SKIPPED - task has {n_classes_total} classes, "
                      f"model only outputs {model.model.num_outputs}", flush=True)
                continue

            n_total = len(train_indices) + len(test_indices)
            was_subsampled = n_total > max_n_samples
            if was_subsampled:
                keep_frac = max_n_samples / n_total
                n_train_keep = max(1, round(len(train_indices) * keep_frac))
                n_test_keep = max(1, round(len(test_indices) * keep_frac))
                train_indices = _subsample_indices(train_indices, y.iloc[train_indices], n_train_keep, rng)
                test_indices = _subsample_indices(test_indices, y.iloc[test_indices], n_test_keep, rng)

            X_train = X.iloc[train_indices].to_numpy()
            y_train = label_encoder.transform(y.iloc[train_indices])
            X_test = X.iloc[test_indices].to_numpy()
            y_test = label_encoder.transform(y.iloc[test_indices])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # binary classification
                y_proba = y_proba[:, 1]

            out[name] = {
                "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba,
                "is_binary": is_binary, "was_subsampled": was_subsampled,
            }
        except Exception as e:
            # One bad task must not take down every other task's results with
            # it - print why and move on (see e.g. slurm-e_paper_binary_e2500_
            # s42-5903007.out: an uncaught error on task #2 lost all 38 results).
            print(f"  {name:40s} FAILED - {type(e).__name__}: {e}", flush=True)
            continue

    return out


def _mean(scores, names):
    # names may include tasks that failed scoring (not present in `scores`,
    # e.g. an exception in main()'s scoring loop) - only average what scored.
    values = [scores[n] for n in names if n in scores]
    if not values:
        return None
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint.pth")
    parser.add_argument("--tasks", choices=["toy", "tabarena"], default="toy")
    parser.add_argument(
        "--max-n-samples", type=int, default=5_000,
        help="subsample (with --subsample-seed) tasks with more rows than this rather than "
             "skipping them - datapoint attention is O(n^2) in rows and large tasks OOM the "
             "GPU (default 5000). Tasks at or under this are the mean_roc_auc_16 subset.",
    )
    parser.add_argument(
        "--max-n-features", type=int, default=120,
        help="SKIP (never subsample) tasks with more columns than this, since a model's "
             "feature axis is fixed by what it was trained on. Default 120 keeps the two "
             "widest of the original 16-dataset legacy set (splice=61, MIC=112 features) "
             "while excluding the pathologically wide tasks that OOM the big architecture "
             "(Bioresponse=1777, hiva_agnostic=1618, kddcup09_appetency=213, APSFailure=171)",
    )
    parser.add_argument(
        "--subsample-seed", type=int, default=0,
        help="random seed for subsampling tasks larger than --max-n-samples",
    )
    parser.add_argument(
        "--output-name", default="tabarena_scores.json",
        help="filename (under the checkpoint's directory) to write scores to",
    )
    args = parser.parse_args()

    device = get_default_device()
    # NanoTabPFNClassifier reads the architecture straight out of the checkpoint.
    classifier = NanoTabPFNClassifier(model=args.checkpoint, device=device)

    resolved_tasks = TOY_CLASSIFICATION_TASKS if args.tasks == "toy" else TABARENA_CLASSIFICATION_TASKS
    print(f"evaluating {args.checkpoint} on {len(resolved_tasks)} {args.tasks} task(s) once each..."
          f" (max_n_samples={args.max_n_samples}, max_n_features={args.max_n_features})", flush=True)

    predictions = evaluate_classification_tasks(
        classifier, resolved_tasks, max_n_features=args.max_n_features,
        max_n_samples=args.max_n_samples, subsample_seed=args.subsample_seed,
    )

    scores = {}
    for name, p in predictions.items():
        try:
            scores[name] = float(roc_auc_score(p["y_true"], p["y_proba"], multi_class="ovr"))
        except Exception as e:
            print(f"  {name:40s} SCORING FAILED - {type(e).__name__}: {e}", flush=True)
            continue
        print(f"  {name:40s} roc_auc={scores[name]:.4f}"
              f"  (binary={p['is_binary']}, subsampled={p['was_subsampled']})", flush=True)

    # only names that actually scored (a task can be in `predictions` but
    # missing from `scores` if it failed in the try/except above)
    binary_names = sorted(n for n, p in predictions.items() if p["is_binary"] and n in scores)
    sixteen_names = sorted(n for n, p in predictions.items() if not p["was_subsampled"] and n in scores)

    mean_auc = _mean(scores, list(scores.keys()))
    mean_binary = _mean(scores, binary_names)
    mean_16 = _mean(scores, sixteen_names)

    # "mean_roc_auc"/"per_dataset" match the original (pre-subsampling) script's
    # output shape exactly, so existing readers (compare_results.py,
    # compare_by_budget.py, plot_lr_sweep.py) keep working unmodified against
    # files written by this version.
    out = {
        "checkpoint": args.checkpoint, "task_set": args.tasks,
        "max_n_samples": args.max_n_samples, "max_n_features": args.max_n_features,
        "subsample_seed": args.subsample_seed,
        "mean_roc_auc": mean_auc, "per_dataset": scores,
        "mean_roc_auc_binary": mean_binary, "binary_datasets": binary_names,
        "mean_roc_auc_16": mean_16, "under_max_n_samples_datasets": sixteen_names,
    }

    out_path = pathlib.Path(args.checkpoint).parent / args.output_name
    out_path.write_text(json.dumps(out, indent=2))

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    print(f"\nmean roc_auc: all={_fmt(mean_auc)} ({len(scores)} ds), "
          f"binary={_fmt(mean_binary)} ({len(binary_names)} ds), "
          f"16={_fmt(mean_16)} ({len(sixteen_names)} ds)  ->  {out_path}")


if __name__ == "__main__":
    main()
