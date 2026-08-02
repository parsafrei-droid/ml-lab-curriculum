import argparse
import collections
import json
import pathlib

import numpy as np
from scipy.stats import wilcoxon

SEEDS = ("s42", "s1", "s2")


def load(results_dir, run):
    path = pathlib.Path(results_dir) / run / "tabarena_scores.json"
    doc = json.load(open(path))
    out = {}
    for name, v in doc["per_dataset"].items():
        auc = v["roc_auc"] if isinstance(v, dict) else v
        if auc is not None:
            out[name] = auc
    return out


# The comparison is paired: for a given seed, the curriculum run and the baseline run share
# the same pool and the same initialisation, so the only difference is the order. Taking the
# difference first cancels most of the seed-to-seed noise, which is why the spread here is
# much tighter than the spread of the raw scores.
def compare(results_dir, treat, ctrl):
    per_seed = []
    per_dataset = collections.defaultdict(list)
    for s in SEEDS:
        t, c = load(results_dir, f"{treat}_{s}"), load(results_dir, f"{ctrl}_{s}")
        common = sorted(set(t) & set(c))
        per_seed.append(np.mean([t[k] for k in common]) - np.mean([c[k] for k in common]))
        for k in common:
            per_dataset[k].append(t[k] - c[k])

    deltas = np.array([np.mean(v) for v in per_dataset.values()])
    stat, p = wilcoxon(deltas)
    return {
        "per_seed": per_seed,
        "seeds_won": int(sum(d > 0 for d in per_seed)),
        "mean_delta": float(np.mean(per_seed)),
        "delta_spread": float(np.std(per_seed)),
        "datasets_improved": int((deltas > 0).sum()),
        "n_datasets": int(len(deltas)),
        "median_dataset_delta": float(np.median(deltas)),
        "wilcoxon_p": float(p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(pathlib.Path(__file__).parent / "results"))
    parser.add_argument("--control", default="baseline")
    parser.add_argument("--treatments", nargs="+",
                        default=["curriculum_features", "curriculum_context_reverse"])
    args = parser.parse_args()

    for treat in args.treatments:
        r = compare(args.results, treat, args.control)
        print(f"\n=== {treat} vs {args.control} ===")
        for s, d in zip(SEEDS, r["per_seed"]):
            print(f"  seed {s:4s}  delta {d:+.4f}")
        print(f"  seeds won            {r['seeds_won']}/{len(SEEDS)}")
        print(f"  mean delta           {r['mean_delta']:+.4f}  (spread {r['delta_spread']:.4f})")
        print(f"  datasets improved    {r['datasets_improved']}/{r['n_datasets']}")
        print(f"  median dataset delta {r['median_dataset_delta']:+.4f}")
        print(f"  Wilcoxon p           {r['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
