"""Compare the four 10k-step binary-model runs on real TabArena binary data.

Runs compared (all: paper's 3-layer/96-emb/4-head/192-hidden model, lr
0.003892, batch 32, 10000 steps, evaluated on the same 27 real binary
TabArena datasets - see experiments/configs/*.yaml for the exact recipes):

  paper_binary                       - no curriculum: fixed at the paper's
                                        4-feature/binary prior the whole run
  curriculum_combined_binary         - features+noise+layers+hidden_dim ramp
                                        together, 3 stages (0/35%/70%)
  curriculum_noise_layers_binary     - features fixed at 4 the whole run;
                                        noise+layers+hidden_dim ramp nested
                                        (3 layer-blocks x 3 noise sub-stages),
                                        ramp completes at 68% of the run
  curriculum_noise_layers_binary_early_ramp - same knobs, finer 4-level log-
                                        spaced noise ramp, completes at 44%
                                        of the run (more time held at target)

Produces experiments/binary_10k_comparison.png: a three-panel figure -
  top           per-dataset ROC-AUC, one row per binary TabArena dataset,
                sorted hardest (lowest mean AUC across the 4 runs) to
                easiest, with dataset size/feature-count/class-imbalance
                alongside and the winning model marked per row
  bottom-left   how many of the 27 datasets each model actually won
  bottom-right  AUC vs. class imbalance - is performance tracking a concrete
                difficulty axis, and does that differ by model

Dataset metadata (n_features, n_rows, minority-class fraction) is hardcoded
below - fetched once from OpenML task qualities (see eval_tabarena.py's own
TABARENA_CLASSIFICATION_TASKS table for the same task IDs) - not re-queried
at plot time, so this script runs offline against existing results/ output.

    python scripts/compare_binary_10k.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

BASE = pathlib.Path(__file__).parent.parent
OUT = BASE / "experiments" / "binary_10k_comparison.png"

# ---------------------------------------------------------------------------
# Runs, in a fixed display order and a fixed categorical color per run (the
# same 4-color order everywhere in this figure - never re-cycled). Colors are
# the dataviz skill's validated categorical slots 1-4 (blue/green/magenta/
# yellow), chosen for CVD-safe adjacent contrast in this exact order.
RUNS = [
    # (result dir, column/axis label, short unique label, color)
    ("paper_binary_e10000", "paper_binary\n(no curriculum)", "paper_binary", "#2a78d6"),
    ("curriculum_combined_binary_e10000", "combined\n(all-at-once ramp)", "combined (all-at-once)", "#008300"),
    ("curriculum_noise_layers_binary_e10000", "noise_layers\n(late ramp)", "noise_layers (late ramp)", "#e87ba4"),
    ("curriculum_noise_layers_binary_early_ramp_e10000", "noise_layers\n(early ramp)", "noise_layers (early ramp)", "#eda100"),
]

# Sequential blue ramp for the AUC heatmap fill (dataviz skill palette.md,
# steps 250-650) - magnitude gets ONE hue, light=low/dark=high, never the
# categorical colors above (those mean "which model", not "how good").
AUC_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "auc_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"]
)

# (n_features, n_rows, minority-class fraction) - from OpenML task qualities
# for the exact task IDs in eval_tabarena.py's TABARENA_CLASSIFICATION_TASKS.
DATASET_META = {
    "Amazon_employee_access": (10, 32769, 0.058),
    "Bank_Customer_Churn": (11, 10000, 0.204),
    "Diabetes130US": (48, 71518, 0.088),
    "E-CommereShippingData": (11, 10999, 0.403),
    "Fitness_Club": (7, 1500, 0.303),
    "GiveMeSomeCredit": (11, 150000, 0.067),
    "HR_Analytics_Job_Change_of_Data_Scientists": (13, 19158, 0.249),
    "Is-this-a-good-customer": (14, 1723, 0.114),
    "Marketing_Campaign": (26, 2240, 0.149),
    "NATICUSdroid": (87, 7491, 0.350),
    "bank-marketing": (14, 45211, 0.117),
    "blood-transfusion-service-center": (5, 748, 0.238),
    "churn": (20, 5000, 0.141),
    "coil2000_insurance_policies": (86, 9822, 0.060),
    "credit-g": (21, 1000, 0.300),
    "credit_card_clients_default": (24, 30000, 0.221),
    "customer_satisfaction_in_airline": (22, 129880, 0.453),
    "diabetes": (9, 768, 0.349),
    "hazelnut-spread-contaminant-detection": (31, 2400, 0.500),
    "heloc": (24, 10459, 0.478),
    "in_vehicle_coupon_recommendation": (25, 12684, 0.432),
    "jm1": (22, 10885, 0.193),
    "online_shoppers_intention": (18, 12330, 0.155),
    "polish_companies_bankruptcy": (65, 5910, 0.069),
    "qsar-biodeg": (42, 1054, 0.337),
    "seismic-bumps": (16, 2584, 0.066),
    "taiwanese_bankruptcy_prediction": (95, 6819, 0.032),
}


def load_scores():
    scores = {}
    for run_name, _label, _short, _color in RUNS:
        path = BASE / "results" / run_name / "tabarena_scores.json"
        scores[run_name] = json.loads(path.read_text())["per_dataset"]
    return scores


def main():
    scores = load_scores()
    run_names = [r[0] for r in RUNS]
    run_labels = [r[1] for r in RUNS]
    run_short = [r[2] for r in RUNS]
    run_colors = [r[3] for r in RUNS]

    datasets = sorted(scores[run_names[0]].keys())
    assert all(set(scores[r].keys()) == set(datasets) for r in run_names), \
        "the 4 runs did not score the same dataset set - not directly comparable"

    # hardest (lowest mean AUC across the 4 runs) first
    mean_auc = {d: np.mean([scores[r][d] for r in run_names]) for d in datasets}
    datasets = sorted(datasets, key=lambda d: mean_auc[d])

    matrix = np.array([[scores[r][d] for r in run_names] for d in datasets])
    winners = matrix.argmax(axis=1)  # which run won each row

    n = len(datasets)
    fig = plt.figure(figsize=(14, 0.30 * n + 6.6))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[0.30 * n, 4.2], hspace=0.20, wspace=0.30,
        left=0.30, right=0.975, top=0.905, bottom=0.07,
    )
    ax_heat = fig.add_subplot(gs[0, :])
    ax_wins = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    # --- top: per-dataset AUC heatmap + metadata sidebar ------------------
    im = ax_heat.imshow(matrix, aspect="auto", cmap=AUC_CMAP, vmin=0.5, vmax=1.0)
    ax_heat.set_xticks(range(len(run_names)))
    ax_heat.set_xticklabels(run_labels, fontsize=8.3)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(datasets, fontsize=8.5)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # small color swatch under each column label, tying this column back to
    # the same model identity used (as categorical color) in the panels below.
    # Positioned in axes-fraction coordinates (not data coords) so it stays
    # put regardless of row count / gridspec proportions.
    n_cols = len(run_names)
    for j, color in enumerate(run_colors):
        x0 = (j + 0.5) / (n_cols + 3.5)  # +3.5 leaves room for the metadata sidebar
        ax_heat.add_patch(Rectangle((x0 - 0.5 / (n_cols + 3.5), 1.012), 1.0 / (n_cols + 3.5), 0.014,
                                     facecolor=color, edgecolor="none", clip_on=False,
                                     transform=ax_heat.transAxes))

    # cell values; bold + dark outline on the row's winning model
    for i in range(n):
        for j in range(n_cols):
            val = matrix[i, j]
            is_winner = j == winners[i]
            text_color = "white" if val > 0.80 else "#0b0b0b"
            ax_heat.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.8,
                         color=text_color, fontweight="bold" if is_winner else "normal")
            if is_winner:
                ax_heat.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                             edgecolor="#0b0b0b", linewidth=1.6))

    # metadata sidebar (plain text, no second color scale - avoids stacking
    # two different magnitude encodings in one axis)
    meta_x = n_cols - 0.5 + 0.35
    labels = ["n_features", "n_rows", "minority %"]
    for k, lab in enumerate(labels):
        ax_heat.text(meta_x + k * 1.15, 1.008, lab, ha="left", va="bottom", fontsize=7.5,
                     color="#52514e", rotation=20, transform=ax_heat.get_yaxis_transform())
    for i, d in enumerate(datasets):
        nf, nr, mf = DATASET_META[d]
        row_txt = f"{nf:>4d}        {nr:>7,d}        {mf * 100:>4.1f}%"
        ax_heat.text(meta_x, i, row_txt, ha="left", va="center", fontsize=7.6,
                     color="#33322f", family="monospace")
    ax_heat.set_xlim(-0.5, meta_x + 3.6)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.012, pad=0.012)
    cbar.set_label("ROC-AUC", fontsize=8, color="#52514e")
    cbar.ax.tick_params(labelsize=7.5)

    ax_heat.set_title(
        "TabArena binary datasets, hardest → easiest (by mean AUC across the 4 runs)  —  "
        "bold + outlined cell = best model for that row",
        fontsize=10, loc="left", color="#0b0b0b", pad=32,
    )

    # --- bottom-left: win count per model ----------------------------------
    win_counts = [int((winners == j).sum()) for j in range(len(run_names))]
    bars = ax_wins.bar(range(len(run_names)), win_counts, color=run_colors, width=0.6)
    ax_wins.set_xticks(range(len(run_names)))
    ax_wins.set_xticklabels(run_labels, fontsize=8)
    ax_wins.set_ylabel("datasets won (of 27)", fontsize=9, color="#52514e")
    ax_wins.set_title("Best model, by dataset count", fontsize=10, loc="left")
    for spine in ("top", "right"):
        ax_wins.spines[spine].set_visible(False)
    ax_wins.spines["left"].set_color("#c9c8c0")
    ax_wins.spines["bottom"].set_color("#c9c8c0")
    ax_wins.tick_params(colors="#52514e", labelsize=8)
    ax_wins.grid(axis="y", color="#e7e6e0", linewidth=0.8, zorder=0)
    ax_wins.set_axisbelow(True)
    for rect, count in zip(bars, win_counts):
        ax_wins.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3,
                     str(count), ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                     color="#0b0b0b")
    mean_row = [f"{np.mean([scores[r][d] for d in datasets]):.4f}" for r in run_names]

    # --- bottom-right: AUC vs class imbalance ------------------------------
    markers = ["o", "s", "^", "D"]
    minority = np.array([DATASET_META[d][2] for d in datasets])
    for j, (run, label, short, color) in enumerate(RUNS):
        ax_scatter.scatter(minority, matrix[:, j], s=26, color=color, marker=markers[j],
                            alpha=0.85, edgecolor="white", linewidth=0.4, label=short)
    ax_scatter.set_xlabel("minority-class fraction (0 = very imbalanced)", fontsize=9, color="#52514e")
    ax_scatter.set_ylabel("ROC-AUC", fontsize=9, color="#52514e")
    ax_scatter.set_title("Does performance track class imbalance?", fontsize=10, loc="left")
    for spine in ("top", "right"):
        ax_scatter.spines[spine].set_visible(False)
    ax_scatter.spines["left"].set_color("#c9c8c0")
    ax_scatter.spines["bottom"].set_color("#c9c8c0")
    ax_scatter.tick_params(colors="#52514e", labelsize=8)
    ax_scatter.grid(color="#e7e6e0", linewidth=0.8, zorder=0)
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(fontsize=7.3, frameon=False, loc="lower right")

    fig.suptitle(
        "Binary nanoTabPFN @ 10,000 steps — curriculum ordering vs. the paper's fixed recipe",
        fontsize=13, y=0.985, color="#0b0b0b",
    )
    fig.text(0.5, 0.955, "all 27 real binary TabArena datasets this checkpoint family can score",
              fontsize=9.5, color="#52514e", ha="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    # console summary
    print("\nwin counts:", dict(zip(run_labels, win_counts)))
    print("mean AUC:", dict(zip(run_labels, mean_row)))


if __name__ == "__main__":
    main()
