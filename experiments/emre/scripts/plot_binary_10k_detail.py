"""Two supporting figures for the binary 10k (exact-baseline) comparison,
built from the real config files and the real per-seed results - not
illustrative sketches.

  binary_10k_schedule.png - the late-ramp (curriculum_noise_layers_binary.yaml)
                            and early-ramp (…_early_ramp.yaml) schedules,
                            visualized as step functions over actual training
                            steps: num_layers, hidden_dim, noise_std. Answers
                            "how is the curriculum config actually built."
                            Thresholds are read from those two YAML files (as
                            fractions of their own `steps:` config value, then
                            rescaled to ACTUAL_STEPS below - these configs were
                            authored at steps=2500 but every result compared
                            in the report was actually run at --steps 10000,
                            which rescales the schedule thresholds
                            proportionally; see run.py's --steps handling).
                            paper_binary.yaml (the baseline, no ramp) is
                            overlaid as a flat dashed reference at its
                            implicit ceiling - it never overrides num_layers/
                            hidden_dim/noise_std in its own schedule, so it
                            trains at TabICL's own defaults the whole run,
                            which are exactly the values both ramps climb to
                            (max_mean: num_layers=6, hidden_dim=130,
                            noise_std=0.3 - tabicl/src/tabicl/prior/
                            _prior_config.py's DEFAULT_SAMPLED_HP).
  binary_10k_seeds.png    - exact per-seed TabArena AUC for baseline vs
                            late-ramp vs early-ramp at 10,000 steps (dots, not
                            just mean +/- std), read from each run's
                            tabarena_scores.json.

    python scripts/plot_binary_10k_detail.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import yaml

BASE = pathlib.Path(__file__).parent.parent
CONFIGS = BASE / "experiments" / "configs"
RESULTS = BASE / "results"
OUT = BASE / "experiments"

BASE_COLOR = "#2a78d6"
LATE_COLOR = "#eb6834"
EARLY_COLOR = "#1baf7a"

# Every result in the report was run at this step budget (--steps 10000
# override), not the configs' own authored steps=2500 - rescale to match.
ACTUAL_STEPS = 10000

# paper_binary.yaml's implicit ceiling (see module docstring) - it holds here
# for the entire run, since it never ramps these knobs at all.
BASELINE_CEILING = {"num_layers": 6, "hidden_dim": 130, "noise_std": 0.3}


def schedule_in_steps(cfg_path, actual_steps=ACTUAL_STEPS):
    cfg = yaml.safe_load(open(cfg_path))
    authored_total = cfg["steps"]
    scale = actual_steps / authored_total
    sched = {int(k): v for k, v in cfg["schedule"].items()}
    return {round(step * scale): knobs for step, knobs in sorted(sched.items())}


def step_series(step_schedule, key, end_step=ACTUAL_STEPS):
    """Turn {step: {key: val, ...}} sparse updates into a full step series."""
    xs, ys = [], []
    current = None
    for s in sorted(step_schedule):
        if key in step_schedule[s]:
            current = step_schedule[s][key]
        if current is None:
            continue
        xs.append(s)
        ys.append(current)
    xs.append(end_step)
    ys.append(current)
    return xs, ys


def plot_schedule():
    late = schedule_in_steps(CONFIGS / "curriculum_noise_layers_binary.yaml")
    early = schedule_in_steps(CONFIGS / "curriculum_noise_layers_binary_early_ramp.yaml")

    fig, axes = plt.subplots(3, 2, figsize=(9, 6.5), sharex=True)
    for col, (label, sched, color) in enumerate([
        ("Late ramp", late, LATE_COLOR),
        ("Early ramp", early, EARLY_COLOR),
    ]):
        for row, key in enumerate(["num_layers", "hidden_dim", "noise_std"]):
            ax = axes[row, col]
            xs, ys = step_series(sched, key)
            ax.step(xs, ys, where="post", color=color, lw=2.2, label="curriculum")
            ax.fill_between(xs, ys, step="post", color=color, alpha=0.12)
            ax.axhline(BASELINE_CEILING[key], color=BASE_COLOR, lw=1.6, ls="--",
                       label="baseline")
            ax.set_xticks([i * 1000 for i in range(11)])
            ax.grid(axis="x", color="#000000", alpha=0.06, lw=0.8)
            if key == "noise_std":
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(key, fontsize=9)
            ax.set_xlim(0, ACTUAL_STEPS)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    for ax in axes[-1]:
        ax.set_xlabel("training step", fontsize=9)
        ax.set_xticklabels([f"{i * 1000}" for i in range(11)], fontsize=7.5, rotation=45)
    fig.suptitle("How the curriculum schedule is built", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "binary_10k_schedule.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def load_seed_aucs(name_pattern_names):
    out = []
    for name in name_pattern_names:
        d = json.loads((RESULTS / name / "tabarena_scores.json").read_text())
        out.append(d.get("mean_roc_auc_binary", d.get("mean_roc_auc")))
    return out


def plot_seeds():
    groups = [
        ("Baseline\n(exact paper recipe)", ["paper_binary_e10000", "paper_binary_e10000_s43", "paper_binary_e10000_s44"], BASE_COLOR),
        ("Late ramp", ["curriculum_noise_layers_binary_e10000", "curriculum_noise_layers_binary_e10000_s43", "curriculum_noise_layers_binary_e10000_s44"], LATE_COLOR),
        ("Early ramp", ["curriculum_noise_layers_binary_early_ramp_e10000", "curriculum_noise_layers_binary_early_ramp_e10000_s43", "curriculum_noise_layers_binary_early_ramp_e10000_s44"], EARLY_COLOR),
    ]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for i, (label, names, color) in enumerate(groups):
        aucs = load_seed_aucs(names)
        seeds = [42, 43, 44]
        jitter = [-0.08, 0, 0.08]
        for auc, seed, jx in zip(aucs, seeds, jitter):
            ax.scatter(i + jx, auc, color=color, s=70, zorder=3, edgecolor="white", linewidth=0.6)
            ax.annotate(f"s{seed}", (i + jx, auc), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color="#666")
        mean = sum(aucs) / len(aucs)
        ax.plot([i - 0.18, i + 0.18], [mean, mean], color=color, lw=2.5, zorder=2)
        ax.annotate(f"mean {mean:.4f}", (i, mean), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, fontweight="bold", color=color)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("TabArena binary ROC-AUC (10,000 steps)")
    ax.set_title("Per seed results")
    ax.set_xlim(-0.5, len(groups) - 0.5)
    fig.tight_layout()
    fig.savefig(OUT / "binary_10k_seeds.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot_schedule()
    plot_seeds()
    print(f"wrote {OUT / 'binary_10k_schedule.png'}")
    print(f"wrote {OUT / 'binary_10k_seeds.png'}")
