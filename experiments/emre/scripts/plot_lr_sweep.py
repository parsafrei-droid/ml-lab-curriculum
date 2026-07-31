"""Plot ROC-AUC vs learning rate for baseline vs curriculum, on our model.

Two questions in one figure:
  - Is our baseline just under-tuned on lr? (does it climb toward the curriculum
    at a better lr, or stay well below?)
  - Is the curriculum less lr-sensitive? (a flatter, higher curve would support
    "curriculum reduces the need to tune")

Reads results/<scenario>_lr<value>_s<seed>/ (see scripts/lr_sweep.sh).

    python scripts/plot_lr_sweep.py
"""

import glob
import json
import pathlib
import re
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).parent.parent
OUT = BASE / "experiments"
NAME = re.compile(r"^(.*)_lr([0-9.eE+-]+)_s\d+$")


def load():
    g = defaultdict(lambda: defaultdict(list))
    for m in glob.glob(str(BASE) + "/results/*_lr*_s*/meta.json"):
        name = json.loads(open(m).read())["name"]
        mt = NAME.match(name)
        if not mt:
            continue
        scenario, lr = mt.group(1), float(mt.group(2))
        sc = m.replace("meta.json", "tabarena_scores.json")
        try:
            g[scenario][lr].append(json.loads(open(sc).read())["mean_roc_auc"])
        except Exception:
            pass
    return g


def ms(v):
    return (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else 0.0) if v else (None, 0.0)


def main():
    g = load()
    if not g:
        print("no lr-sweep runs found (results/<scenario>_lr<value>_s<seed>/) - run scripts/lr_sweep.sh")
        return
    plt.figure(figsize=(8, 5.5))
    for scenario in sorted(g):
        lrs = sorted(g[scenario])
        ys = [ms(g[scenario][x])[0] for x in lrs]
        es = [ms(g[scenario][x])[1] for x in lrs]
        colour = "tab:red" if scenario == "baseline" else "tab:green"
        plt.errorbar(lrs, ys, yerr=es, marker="o", capsize=4,
                     label=scenario.replace("curriculum_", ""), color=colour, lw=2)
    plt.xscale("log")
    plt.axhline(0.5, ls=":", c="gray", lw=1)
    plt.xlabel("learning rate (log scale)")
    plt.ylabel("TabArena ROC-AUC")
    plt.title("Do you need to tune? AUC vs learning rate (our model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "comparison_lr_sweep.png", dpi=120)
    print(f"saved -> {OUT / 'comparison_lr_sweep.png'}")


if __name__ == "__main__":
    main()
