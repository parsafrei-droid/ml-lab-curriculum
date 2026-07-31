import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent

AUTHORED_STEPS = 2500
ACTUAL_STEPS = 10000
SCALE = ACTUAL_STEPS / AUTHORED_STEPS
CEILING = 0.3

EARLY_RAMP = {
    0: 0.001, 100: 0.01, 200: 0.1, 300: 0.3,
    400: 0.001, 500: 0.01, 600: 0.1, 700: 0.3,
    800: 0.001, 900: 0.01, 1000: 0.1, 1100: 0.3,
}

BASELINE_COLOR = "#3b6fd4"
CURRICULUM_COLOR = "#e8862e"

NOTE = [
    "Noise curriculum (early ramp)",
    "",
    "noise_std is a TabICL prior knob:",
    "more noise -> harder synthetic data.",
    "",
    "- Baseline holds it at the ceiling",
    "  (0.3) for the whole run.",
    "- Curriculum ramps it up inside each",
    "  num_layers block:",
    "  0.001 -> 0.01 -> 0.1 -> 0.3",
    "  (4 levels, log-spaced - the sampler",
    "  is log-scaled, so equal decade steps).",
    "- Ramp finishes at ~44% of training,",
    "  then holds at the ceiling for the",
    "  remaining ~56%.",
    "",
    "Result (binary, 10k steps, 3 seeds):",
    "+0.008 TabArena AUC over the exact",
    "paper baseline, ~29% less train time.",
]


def step_series(sched):
    xs, ys = [], []
    current = None
    for s in sorted(sched):
        current = sched[s]
        xs.append(round(s * SCALE))
        ys.append(current)
    xs.append(ACTUAL_STEPS)
    ys.append(current)
    return xs, ys


def make(fname):
    xs, ys = step_series(EARLY_RAMP)

    fig = plt.figure(figsize=(11.5, 5.0))
    ax = fig.add_axes([0.08, 0.14, 0.52, 0.74])

    ax.step(xs, ys, where="post", color=CURRICULUM_COLOR, lw=2.6, label="Curriculum")
    ax.fill_between(xs, ys, step="post", color=CURRICULUM_COLOR, alpha=0.12)
    ax.axhline(CEILING, color=BASELINE_COLOR, lw=2.0, ls="--", label="Baseline")

    ax.set_yscale("log")
    ax.set_xlim(0, ACTUAL_STEPS)
    ax.set_xticks([i * 2000 for i in range(6)])
    ax.set_xlabel("Training step")
    ax.set_ylabel("noise_std (log scale)")
    ax.set_title("The noise curriculum: baseline trains at full noise from step 0,\n"
                 "the curriculum climbs to it", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, which="both", alpha=0.13)

    fig.text(0.65, 0.9, "\n".join(NOTE), va="top", ha="left",
             fontsize=10, family="monospace")

    out = BASE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / fname, dpi=140)
    plt.close(fig)
    print(f"saved -> {out / fname}")


if __name__ == "__main__":
    make("noise_schedule.png")
