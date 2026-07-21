"""Rank-based comparison across multiple results/<name>/tabarena_scores.json runs.

A plain per-dataset mean (what mean_roc_auc etc. already give you) treats a
dataset every model scores ~0.52 on (no headroom - see e.g. Diabetes130US,
~9% minority class) the same as one every model scores ~0.95 on: equal
weight in the average despite wildly different intrinsic difficulty and
room to move. TabArena's own leaderboard aggregates by RANK instead, for
exactly this reason - "how did this model do relative to the others on
this dataset" isn't sensitive to the dataset's absolute scale the way a
raw mean is. This script computes that: for each dataset all the given
runs share, rank the runs (1 = best), then average each run's rank across
datasets.

Only datasets present in EVERY given run are used (set intersection) - a
run that skipped a task (wrong max-n-features, wrong max-n-samples, or a
model that can't represent that many classes) can't be ranked on it.
Same for metrics: mean_roc_auc/per_dataset always exists; balanced-accuracy
and log-loss only exist in runs evaluated with the current eval_tabarena.py
(older tabarena_scores.json files won't have per_dataset_balanced_accuracy/
per_dataset_log_loss) - those metrics are silently skipped if any given run
lacks them, rather than dropping runs or crashing.

    python scripts/rank_compare.py paper_binary_e10000 curriculum_combined_binary_e10000 \\
        curriculum_noise_layers_binary_e10000 curriculum_noise_layers_binary_early_ramp_e10000

    python scripts/rank_compare.py --output experiments/rank_compare_binary_10k.json \\
        paper_binary_e10000 curriculum_combined_binary_e10000 ...
"""

import argparse
import json
import pathlib

BASE = pathlib.Path(__file__).parent.parent

# (json key for per-dataset values, display name, higher_is_better)
METRICS = [
    ("per_dataset", "roc_auc", True),
    ("per_dataset_balanced_accuracy", "balanced_accuracy", True),
    ("per_dataset_log_loss", "log_loss", False),
]


def load(name):
    path = BASE / "results" / name / "tabarena_scores.json"
    if not path.exists():
        raise SystemExit(f"no tabarena_scores.json for '{name}' (looked in {path})")
    return json.loads(path.read_text())


def rank_metric(per_dataset_by_run, names, higher_is_better):
    """per_dataset_by_run: {run_name: {dataset: value}}. Returns (per-dataset ranks,
    average rank per run), using only datasets every run has a value for."""
    shared = set.intersection(*(set(d.keys()) for d in per_dataset_by_run.values()))
    if not shared:
        return {}, {}
    per_dataset_ranks = {}
    avg_rank = {n: [] for n in names}
    for ds in sorted(shared):
        values = [(n, per_dataset_by_run[n][ds]) for n in names]
        values.sort(key=lambda nv: nv[1], reverse=higher_is_better)
        ranks = {n: i + 1 for i, (n, _v) in enumerate(values)}
        per_dataset_ranks[ds] = ranks
        for n in names:
            avg_rank[n].append(ranks[n])
    avg_rank = {n: sum(v) / len(v) for n, v in avg_rank.items()}
    return per_dataset_ranks, avg_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="result dir names under results/ (>= 2)")
    parser.add_argument("--output", default=None, help="optional path to also write a JSON summary to")
    args = parser.parse_args()
    if len(args.runs) < 2:
        raise SystemExit("need at least 2 runs to rank against each other")

    data = {name: load(name) for name in args.runs}

    print(f"comparing {len(args.runs)} runs: {', '.join(args.runs)}\n")

    summary = {"runs": args.runs, "metrics": {}}
    for key, label, higher_is_better in METRICS:
        per_dataset_by_run = {}
        missing = []
        for name in args.runs:
            pd = data[name].get(key)
            if not pd:
                missing.append(name)
            else:
                per_dataset_by_run[name] = pd
        if missing:
            print(f"[{label}] skipped - missing from: {', '.join(missing)} "
                  f"(re-run eval_tabarena.py to backfill)\n")
            continue

        per_dataset_ranks, avg_rank = rank_metric(per_dataset_by_run, args.runs, higher_is_better)
        n_shared = len(per_dataset_ranks)
        if n_shared == 0:
            print(f"[{label}] skipped - no datasets shared by every run\n")
            continue

        direction = "higher is better" if higher_is_better else "lower is better"
        print(f"[{label}] ({direction}, {n_shared} shared datasets)")
        ordered = sorted(avg_rank.items(), key=lambda nv: nv[1])
        for rank_pos, (name, ar) in enumerate(ordered, start=1):
            raw_mean = sum(per_dataset_by_run[name].values()) / len(per_dataset_by_run[name])
            wins = sum(1 for ranks in per_dataset_ranks.values() if ranks[name] == 1)
            print(f"  {rank_pos}. {name:45s} avg_rank={ar:.2f}   "
                  f"raw_mean={raw_mean:.4f}   wins={wins}/{n_shared}")
        print()

        summary["metrics"][label] = {
            "n_shared_datasets": n_shared,
            "higher_is_better": higher_is_better,
            "avg_rank": avg_rank,
            "per_dataset_ranks": per_dataset_ranks,
        }

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
