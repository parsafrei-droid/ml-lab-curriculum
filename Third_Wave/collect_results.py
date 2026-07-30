"""Collect the four Third-Wave runs into one summary.

Ours (runs 1-2) log to Second_Wave/results/<name>/log.csv with val_auc and cum_time_s.
Modded (runs 3-4) logs one line per epoch to its experiment dir, ending with
"record time in mins: X" when it reaches the 0.8068 target.

The two models are NOT put on a shared axis: our val_auc is synthetic (~0.59), modded's
is TabArena (~0.807). We compare within a model, on wall-clock.
"""

import csv
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
SW = REPO / "Second_Wave" / "results"
MODDED = REPO / "Third_Wave" / "modded-nanotabpfn" / "workdir" / "experiments"
JACKPOT = 0.8068462330697953


def read_ours(name):
    """val_auc vs cum_time_s from our train.py log."""
    p = SW / name / "log.csv"
    if not p.exists():
        return None
    traj = []
    with open(p) as f:
        for row in csv.DictReader(f):
            traj.append(
                {
                    "step": int(row["step"]),
                    "val_auc": float(row["val_auc"]),
                    "cum_time_s": float(row["cum_time_s"]),
                    "mean_features": float(row["mean_features"]),
                }
            )
    if not traj:
        return None
    best = max(traj, key=lambda r: r["val_auc"])
    out = {
        "run": name,
        "trajectory": traj,
        "final_val_auc": traj[-1]["val_auc"],
        "best_val_auc": best["val_auc"],
        "total_wall_clock_s": traj[-1]["cum_time_s"],
    }
    scores = SW / name / "tabarena_scores.json"
    if scores.exists():
        with open(scores) as f:
            out["tabarena"] = json.load(f)
    return out


EPOCH_RE = re.compile(r"e:(\d+)/\d+.*?t_t:([\d.]+)s avg_roc_auc:([\d.eE+-]+)")


def read_modded(name):
    """Parse modded's own log: per-epoch training time (t_t) and TabArena avg ROC AUC."""
    root = MODDED / name
    if not root.exists():
        return None
    logs = sorted(root.glob("*/*-log.txt"))
    if not logs:
        return None
    log = logs[-1]  # most recent run of this name
    traj, record_mins, gpu = [], None, None
    for line in open(log, errors="replace"):
        m = EPOCH_RE.search(line)
        if m:
            traj.append(
                {
                    "epoch": int(m.group(1)),
                    "cum_train_s": float(m.group(2)),
                    "avg_roc_auc": float(m.group(3)),
                }
            )
        # the log starts with an echo of train_nano.py's own source, so match the real
        # emitted line ("record time in mins: 1.23") and not the print0(...) that makes it
        m2 = re.match(r"record time in mins:\s*([\d.]+)\s*$", line.strip())
        if m2:
            record_mins = float(m2.group(1))
        if gpu is None:
            g = re.search(r"(NVIDIA [A-Za-z0-9]+(?: [A-Za-z0-9]+)*?)\s+(?:On|Off)\b", line)
            if g:
                gpu = g.group(1).strip()
    if not traj:
        return None
    # time to first reach the target, using modded's own training-time accumulator
    hit = next((r for r in traj if r["avg_roc_auc"] >= JACKPOT), None)
    best = max(traj, key=lambda r: r["avg_roc_auc"])
    return {
        # when a run does not reach the target, time-to-best is the honest stand-in
        "best_epoch": best["epoch"],
        "time_to_best_s": best["cum_train_s"],
        "run": name,
        "log": str(log.relative_to(REPO)),
        "gpu": gpu,
        "trajectory": traj,
        "final_avg_roc_auc": traj[-1]["avg_roc_auc"],
        "best_avg_roc_auc": max(r["avg_roc_auc"] for r in traj),
        "epochs_run": traj[-1]["epoch"],
        "total_train_s": traj[-1]["cum_train_s"],
        "reached_target": hit is not None,
        "time_to_target_s": hit["cum_train_s"] if hit else None,
        "epoch_at_target": hit["epoch"] if hit else None,
        "record_time_mins": record_mins,
    }


def main():
    out = {
        "note": (
            "Compare within a model on wall-clock: run1 vs run2 (ours, synthetic val_auc) "
            "and run3 vs run4 (modded, TabArena ROC-AUC). Different metrics/scales - do not "
            "put all four on one y-axis."
        ),
        "target_roc_auc": JACKPOT,
        "runs": {},
    }
    for key, name in [("run1_ours_baseline", "baseline_a100_s42"),
                      ("run2_ours_curriculum", "curriculum_features_a100_s42")]:
        r = read_ours(name)
        if r:
            out["runs"][key] = r
    for key, name in [("run3_modded_baseline", "tw_baseline"),
                      ("run4_modded_curriculum", "tw_curriculum")]:
        r = read_modded(name)
        if r:
            out["runs"][key] = r

    # If neither modded run reached 0.807, "time to target" is undefined for both. The
    # fair within-model comparison is then iso-quality: pick the best AUC that BOTH runs
    # achieved, and compare how long each took to first get there.
    r3 = out["runs"].get("run3_modded_baseline")
    r4 = out["runs"].get("run4_modded_curriculum")
    if r3 and r4:
        common = min(r3["best_avg_roc_auc"], r4["best_avg_roc_auc"])

        def first_at(r, thr):
            hit = next((p for p in r["trajectory"] if p["avg_roc_auc"] >= thr), None)
            return hit["cum_train_s"] if hit else None

        t3, t4 = first_at(r3, common), first_at(r4, common)
        out["iso_quality"] = {
            "roc_auc": common,
            "note": "highest TabArena ROC-AUC both modded runs reached; times are training-only (t_t)",
            "run3_train_s": t3,
            "run4_train_s": t4,
            "curriculum_speedup": (t3 / t4) if (t3 and t4) else None,
        }

    dest = REPO / "Third_Wave" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    for key, r in out["runs"].items():
        if "modded" in key:
            tt = f"{r['time_to_target_s']:.1f}s" if r["time_to_target_s"] else "not reached"
            print(f"{key:24s} epochs={r['epochs_run']:4d} best_auc={r['best_avg_roc_auc']:.4f} "
                  f"train_s={r['total_train_s']:7.1f} time_to_target={tt}")
        else:
            print(f"{key:24s} final_val_auc={r['final_val_auc']:.4f} "
                  f"wall_clock_s={r['total_wall_clock_s']:7.1f}")
    print(f"\nwrote {dest / 'summary.json'}")


if __name__ == "__main__":
    main()
