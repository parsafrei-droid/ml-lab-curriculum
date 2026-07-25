"""Sample random (lr, num_datapoints_max) configs from the nanoTabPFN paper's
own search space (Table 1, Appendix A) and materialize them as training YAMLs,
to test: does curriculum pretraining help more the further a config drifts
from the paper's tuned optimum?

Only lr and num_datapoints are randomized. Everything else - architecture
(heads/embedding_size/hidden_size/layers), batch_size, steps, seed, and the
feature/class ceiling - stays fixed at paper_binary.yaml's exact values, so
every trial isolates "how much does curriculum help at this lr/num_datapoints"
against the same paper-matched control (already run, see paper_binary.yaml /
curriculum_noise_layers_binary_early_ramp.yaml).

batch_size (Table 1's effective_batch_size) is deliberately NOT swept even
though it's also in Table 1: this project separately found it's the single
biggest lever on baseline quality on its own (PRESENTATION.md section 10),
so mixing it in here would confound "tuning sensitivity" with "compute cost
per step" instead of isolating the former. weight_decay is excluded because
it's dead code - TFM-Playground/tfmplayground/train.py hardcodes
weight_decay=0.0 and never reads it from config, so sampling it would
silently change nothing.

For each trial i, two YAMLs are written, identical except for lr/num_datapoints
and name:
  paper_binary_hp<i>_s<seed>.yaml   - no curriculum (copy of paper_binary.yaml)
  early_ramp_hp<i>_s<seed>.yaml     - curriculum   (copy of
                                       curriculum_noise_layers_binary_early_ramp.yaml)
so scripts/hp_random_search.sh can train both arms per sampled config and
scripts/rank_hp_sweep.py can compare them pairwise.

--train-seed controls only the training seed (weight init / batch order) -
it does NOT affect which (lr, num_datapoints) get sampled (that's
--sample-seed, kept fixed by default so re-running with a different
--train-seed reproduces the exact same 10 configs under a different seed,
e.g. to check the sweep's findings aren't a single-seed fluke - matches
this project's usual 42/43/44 seed convention). Writes
experiments/configs/hp_sweep/manifest_s<seed>.csv, so scripts/rank_hp_sweep.py
can find every seed's manifest independently.

    python scripts/sample_hp_configs.py
    python scripts/sample_hp_configs.py --train-seed 43   # same 10 configs, new training seed
    python scripts/sample_hp_configs.py --n 20 --sample-seed 0   # a genuinely different draw
"""

import argparse
import math
import pathlib
import random

import yaml

BASE = pathlib.Path(__file__).parent.parent
CONFIGS = BASE / "experiments" / "configs"
OUT = CONFIGS / "hp_sweep"

BASELINE_SRC = CONFIGS / "paper_binary.yaml"
CURRICULUM_SRC = CONFIGS / "curriculum_noise_layers_binary_early_ramp.yaml"

# Table 1 (Appendix A) search space - only the two axes this sweep varies.
LR_LOW, LR_HIGH = 1e-4, 5e-2   # log scale, per the paper
NDP_LOW, NDP_HIGH = 50, 300    # num_datapoints_max, linear (not marked log scale in Table 1)

def sample_configs(n, sample_seed):
    rng = random.Random(sample_seed)
    log_lo, log_hi = math.log10(LR_LOW), math.log10(LR_HIGH)
    return [(10 ** rng.uniform(log_lo, log_hi), rng.randint(NDP_LOW, NDP_HIGH))  # randint is inclusive
            for _ in range(n)]


def write_variant(src_path, trial_idx, lr, num_datapoints, prefix, train_seed):
    with open(src_path) as f:
        cfg = yaml.safe_load(f)
    cfg["name"] = f"{prefix}_hp{trial_idx}_s{train_seed}"
    cfg["seed"] = train_seed
    cfg["lr"] = lr
    cfg["num_datapoints"] = num_datapoints
    # steps/batch_size/heads/embedding_size/hidden_size/layers/schedule: untouched,
    # copied straight from src_path
    out_path = OUT / f"{prefix}_hp{trial_idx}_s{train_seed}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of random configs to sample")
    parser.add_argument("--sample-seed", type=int, default=42,
                         help="RNG seed for the (lr, num_datapoints) sampling itself - kept fixed "
                              "by default so a different --train-seed still reproduces the same "
                              "10 configs")
    parser.add_argument("--train-seed", type=int, default=42,
                         help="training seed baked into every config (weight init / batch order) - "
                              "does not affect which hyperparameters get sampled")
    args = parser.parse_args()

    assert BASELINE_SRC.exists(), f"missing {BASELINE_SRC}"
    assert CURRICULUM_SRC.exists(), f"missing {CURRICULUM_SRC}"
    OUT.mkdir(parents=True, exist_ok=True)

    configs = sample_configs(args.n, args.sample_seed)

    manifest = ["trial,lr,num_datapoints,baseline_name,curriculum_name"]
    for i, (lr, ndp) in enumerate(configs):
        b = write_variant(BASELINE_SRC, i, lr, ndp, "paper_binary", args.train_seed)
        c = write_variant(CURRICULUM_SRC, i, lr, ndp, "early_ramp", args.train_seed)
        manifest.append(f"{i},{lr:.6g},{ndp},{b.stem},{c.stem}")
        print(f"[{i}] lr={lr:.6g}  num_datapoints={ndp}  -> {b.name}, {c.name}")

    manifest_path = OUT / f"manifest_s{args.train_seed}.csv"
    manifest_path.write_text("\n".join(manifest) + "\n")
    print(f"\nwrote {2 * args.n} configs + {manifest_path.name} to {OUT}")


if __name__ == "__main__":
    main()
