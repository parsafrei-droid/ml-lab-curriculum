# Curriculum Pretraining for nanoTabPFN

ML Lab 2026, University of Freiburg. Parsa Rasouli, Emre Ozturk, Omid Frei.
Supervisors: Alexander Pfefferle, Dominika Wozniak.

Does the **order** of the synthetic tables matter when pretraining a tabular foundation
model? It does, but not because "easy first" is magic. What decides the outcome is the
regime the curriculum **ends** in.

**Start here: [`final/`](final/)** contains the figures, the write-up and the code to
reproduce the main result.

- [`final/README.md`](final/README.md) - what we found and how to run it
- [`final/FINDINGS.md`](final/FINDINGS.md) - the full write-up
- [`final/COMPUTE.md`](final/COMPUTE.md) - what the project cost
- [`experiments/`](experiments/) - everything we tried on the way, including what failed

## Setup

Two upstream repos are expected next to this one, plus a shared virtualenv:

```bash
git clone <TFM-Playground> TFM-Playground
git clone <tabicl> tabicl
python -m venv .venv && source .venv/bin/activate
pip install -e TFM-Playground -e tabicl
```

Then:

```bash
cd final/code && bash run.sh
```
