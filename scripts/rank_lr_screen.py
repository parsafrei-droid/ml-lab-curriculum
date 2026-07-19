"""Rank scripts/lr_screen.sh's candidates by their synthetic-validation proxy.

Reads results/<scenario>_screen_lr<value>_s<seed>/meta.json (no
tabarena_scores.json needed - lr_screen.sh never calls eval_tabarena.py) and
prints each scenario's lr values sorted by final_val_loss (lower is better),
averaged over seeds when there's more than one.

    python scripts/rank_lr_screen.py
"""

import glob
import json
import pathlib
import re
import statistics
from collections import defaultdict

BASE = pathlib.Path(__file__).parent.parent
NAME = re.compile(r"^(.*)_screen_lr([0-9.eE+-]+)_s\d+$")


def load():
    g = defaultdict(lambda: defaultdict(list))  # scenario -> lr -> [(val_loss, val_acc), ...]
    for m in glob.glob(str(BASE) + "/results/*_screen_lr*_s*/meta.json"):
        d = json.loads(open(m).read())
        mt = NAME.match(d["name"])
        if not mt:
            continue
        scenario, lr = mt.group(1), float(mt.group(2))
        if d.get("final_val_loss") is None:
            continue  # run didn't finish
        g[scenario][lr].append((d["final_val_loss"], d["final_val_acc"]))
    return g


def main():
    g = load()
    if not g:
        print("no lr_screen.sh runs found (results/<scenario>_screen_lr<value>_s<seed>/) - "
              "run scripts/lr_screen.sh first")
        return

    for scenario in sorted(g):
        print(f"\n=== {scenario} ===")
        rows = []
        for lr, vals in g[scenario].items():
            losses = [v[0] for v in vals]
            accs = [v[1] for v in vals]
            rows.append((lr, statistics.mean(losses), statistics.mean(accs), len(vals)))
        rows.sort(key=lambda r: r[1])  # best (lowest) val_loss first
        print(f"  {'lr':>10s}  {'val_loss':>10s}  {'val_acc':>9s}  {'n_seeds':>7s}")
        for lr, vl, va, n in rows:
            print(f"  {lr:>10g}  {vl:>10.4f}  {va:>9.4f}  {n:>7d}")
        best = rows[0]
        print(f"  -> best by val_loss: lr={best[0]:g} (val_loss={best[1]:.4f}, val_acc={best[2]:.4f})")


if __name__ == "__main__":
    main()
