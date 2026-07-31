"""Compare scripts/hp_random_search.sh's paired trials, across every seed
that's been run: does curriculum help more the further (lr, num_datapoints)
drifts from the paper's tuned optimum, and does that hold up across seeds?

Reads every experiments/configs/hp_sweep/manifest_s<seed>.csv (written by
scripts/sample_hp_configs.py --train-seed <seed>) for each trial's sampled
lr/num_datapoints - the same 10 configs are reused across seeds by design
(--sample-seed stays fixed; only --train-seed changes), so trial hp<i> means
the same (lr, num_datapoints) in every seed. For each (trial, seed) pair it
loads both results/paper_binary_hp<i>_s<seed>/meta.json (no curriculum) and
results/early_ramp_hp<i>_s<seed>/meta.json (curriculum) and compares
final_val_acc/final_val_loss - the free, comparable proxy every run already
logs (FixedValidationCallback, one shared synthetic set) - plus, once
scripts/eval_pending_checkpoints.sh has scored the checkpoint,
results/<name>/tabarena_scores.json's mean_roc_auc_binary - the real,
full-TabArena ground truth. Also prints results/paper_binary_e2500_s42 vs
results/curriculum_noise_layers_binary_early_ramp_e2500_s42 - the already-run
pair AT the paper's literal optimum - as a "zero mistuning" anchor row.

Rows are grouped by trial (sorted by distance from the paper's optimum lr in
log space, most mistuned first) with one row per seed, followed by that
trial's mean delta across seeds - so a trial's headline effect (e.g. hp0's
lr-rescue) can be read alongside how much it moves seed to seed, not just as
a single-seed number.

    python scripts/rank_hp_sweep.py
"""

import csv
import json
import math
import pathlib
import re

BASE = pathlib.Path(__file__).parent.parent
RESULTS = BASE / "results"
HP_SWEEP_DIR = BASE / "experiments" / "configs" / "hp_sweep"
PAPER_OPTIMUM_LR = 0.003892  # Table 1's optimum, for the "distance from tuned" sort

REFERENCE_PAIR = ("paper_binary_e2500_s42", "curriculum_noise_layers_binary_early_ramp_e2500_s42")

MANIFEST_RE = re.compile(r"^manifest_s(\d+)\.csv$")


def load_meta(name):
    p = RESULTS / name / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_final_toy_auc(name):
    p = RESULTS / name / "toy_tabarena.csv"
    if not p.exists():
        return None
    with p.open() as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    return float(rows[-1][1])


def load_tabarena_auc(name):
    p = RESULTS / name / "tabarena_scores.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("mean_roc_auc_binary", d.get("mean_roc_auc"))


def load_all(name):
    return load_meta(name), load_final_toy_auc(name), load_tabarena_auc(name)


def find_seeds():
    seeds = {}
    for p in sorted(HP_SWEEP_DIR.glob("manifest_s*.csv")):
        m = MANIFEST_RE.match(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        with p.open() as f:
            seeds[seed] = list(csv.DictReader(f))
    return seeds


def print_row(label, lr, ndp, base, curr):
    def fmt(meta, tab):
        if meta is None or meta.get("final_val_acc") is None:
            return f"{'--':>8} {'--':>8}"
        tab_s = f"{tab:.4f}" if tab is not None else "n/a"
        return f"{meta['final_val_acc']:8.4f} {tab_s:>8}"

    lr_s = f"{lr:.6g}" if lr is not None else "paper opt"
    ndp_s = f"{ndp}" if ndp is not None else "154"
    bm, _, btab = base
    cm, _, ctab = curr
    print(f"{label:>16} {lr_s:>12} {ndp_s:>5}  |  {fmt(bm, btab)}  |  {fmt(cm, ctab)}")


def main():
    seeds = find_seeds()
    if not seeds:
        raise SystemExit(f"no manifest_s<seed>.csv files in {HP_SWEEP_DIR} - run scripts/sample_hp_configs.py first")

    n_trials = len(next(iter(seeds.values())))
    seed_list = sorted(seeds)

    header = (f"{'trial / seed':>16} {'lr':>12} {'ndp':>5}  |  {'base_va':>8} {'base_tab':>8}"
              f"  |  {'curr_va':>8} {'curr_tab':>8}")
    print(header)
    print("-" * len(header))
    print_row("paper-opt (s42)", None, None, load_all(REFERENCE_PAIR[0]), load_all(REFERENCE_PAIR[1]))
    print("-" * len(header))

    # trial -> (lr, ndp); assumes every seed sampled the same configs (--sample-seed fixed)
    trial_hp = {int(r["trial"]): (float(r["lr"]), int(r["num_datapoints"])) for r in seeds[seed_list[0]]}
    order = sorted(trial_hp, key=lambda t: -abs(math.log10(trial_hp[t][0]) - math.log10(PAPER_OPTIMUM_LR)))

    all_deltas = []       # every (trial, seed) tabarena delta - for the overall win-rate
    all_deltas_va = []    # proxy fallback
    per_trial_deltas = {t: [] for t in order}

    for trial in order:
        lr, ndp = trial_hp[trial]
        for seed in seed_list:
            base = load_all(f"paper_binary_hp{trial}_s{seed}")
            curr = load_all(f"early_ramp_hp{trial}_s{seed}")
            print_row(f"hp{trial} (s{seed})", lr, ndp, base, curr)
            bm, _, btab = base
            cm, _, ctab = curr
            if bm and cm and bm.get("final_val_acc") is not None and cm.get("final_val_acc") is not None:
                all_deltas_va.append(cm["final_val_acc"] - bm["final_val_acc"])
            if btab is not None and ctab is not None:
                d = ctab - btab
                all_deltas.append(d)
                per_trial_deltas[trial].append(d)
        if per_trial_deltas[trial]:
            ds = per_trial_deltas[trial]
            mean_d = sum(ds) / len(ds)
            spread = f"[{min(ds):+.3f}, {max(ds):+.3f}]" if len(ds) > 1 else ""
            print(f"{'  mean Δ AUC':>16} {'':>12} {'':>5}  |  {'':>8} {'':>8}  |  {'':>8} {mean_d:+8.4f}  {spread}")
        print()

    print("-" * len(header))
    if all_deltas_va:
        wins_va = sum(1 for d in all_deltas_va if d > 0)
        print(f"[proxy]         curriculum wins on val_acc: {wins_va}/{len(all_deltas_va)} "
              f"(trial,seed) pairs (mean Δ={sum(all_deltas_va) / len(all_deltas_va):+.4f})")
    if all_deltas:
        wins = sum(1 for d in all_deltas if d > 0)
        print(f"[real TabArena] curriculum wins on mean_roc_auc_binary: {wins}/{len(all_deltas)} "
              f"(trial,seed) pairs across {len(seed_list)} seed(s) {seed_list} "
              f"(mean Δ={sum(all_deltas) / len(all_deltas):+.4f})")
        print(f"\nper-trial mean Δ AUC across seeds, most lr-mistuned first:")
        for trial in order:
            ds = per_trial_deltas[trial]
            if not ds:
                continue
            mean_d = sum(ds) / len(ds)
            print(f"  hp{trial:<2} lr={trial_hp[trial][0]:.4g}  n={trial_hp[trial][1]:<4}  "
                  f"mean Δ={mean_d:+.4f}  (n_seeds={len(ds)})")
    else:
        print("\n(no tabarena_scores.json found yet - run scripts/submit_eval_pending.sh with the hp-sweep names)")

    print(f"\n{n_trials} trials x {len(seed_list)} seed(s) {seed_list}. Rows sorted by "
          f"|log10(lr) - log10(paper_optimum_lr)| descending.")


if __name__ == "__main__":
    main()
