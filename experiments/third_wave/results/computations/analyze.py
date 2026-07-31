#!/usr/bin/env python3
"""Aggregate the full compute history: Slurm accounting + Kaggle notebook logs."""
import csv, glob, io, json, os, subprocess, datetime as dt
from collections import defaultdict

REPO = "/pfs/data6/home/fr/fr_fr/fr_or51/projects/ml-lab-curriculum"
KAGGLE = os.path.join(REPO, "Second_Wave/results/kaggle_run")
SACCT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sacct_all.psv")

SACCT_CMD = ["sacct", "-X", "-S", "2026-06-01", "-E", "now", "-P",
             "--format=JobID,JobName%60,Partition,State,Elapsed,ElapsedRaw,"
             "Start,End,Submit,NodeList,AllocTRES%80,ExitCode,ReqTRES%60"]

def load_sacct():
    """Read the cached dump if present, else query Slurm directly."""
    if os.path.exists(SACCT):
        return open(SACCT)
    return io.StringIO(subprocess.run(SACCT_CMD, capture_output=True, text=True,
                                      check=True).stdout)

# Partition -> (hardware label, cluster). Node lists confirmed via `scontrol show partition`.
PART2GPU = {
    "gpu_a100_short":  ("NVIDIA A100 80GB", "BwUniCluster 2.0 (uc2)"),
    "gpu_a100_il":     ("NVIDIA A100 80GB", "BwUniCluster 2.0 (uc2)"),
    "dev_gpu_a100_il": ("NVIDIA A100 80GB", "BwUniCluster 2.0 (uc2)"),
    "gpu_h100":        ("NVIDIA H100 94GB", "BwUniCluster 3.0 (uc3)"),
    "gpu_h100_short":  ("NVIDIA H100 94GB", "BwUniCluster 3.0 (uc3)"),
    "dev_gpu_h100":    ("NVIDIA H100 94GB", "BwUniCluster 3.0 (uc3)"),
    "gpu_h100_il":     ("NVIDIA H100 94GB", "BwUniCluster 2.0 (uc2)"),
    "gpu_mi300":       ("AMD MI300",        "BwUniCluster 3.0 (uc3)"),
    "cpu":             ("CPU only",         "BwUniCluster 2.0 (uc2)"),
}

def parse(s):
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def wave_of(d, name):
    """Project phase, keyed on submit date (the waves are cleanly separated in time)."""
    if d <= dt.date(2026, 6, 30):
        return "0. Setup & GPU probes"
    if d <= dt.date(2026, 7, 6):
        return "1. First Wave (TabICL / TFM-Playground)"
    if d <= dt.date(2026, 7, 29):
        return "2. Second Wave (nanoTabPFN feature curriculum)"
    return "3. Third Wave (modded-nanoTabPFN 2x2)"

rows = list(csv.DictReader(load_sacct(), delimiter="|"))

jobs = []
for r in rows:
    st = r["State"].split(" by ")[0].strip()
    submit, start, end = parse(r["Submit"]), parse(r["Start"]), parse(r["End"])
    elapsed = int(r["ElapsedRaw"] or 0)
    queue = (start - submit).total_seconds() if (start and submit) else None
    gpu, cluster = PART2GPU.get(r["Partition"], ("unknown", "unknown"))
    jobs.append(dict(
        jobid=r["JobID"], name=r["JobName"], part=r["Partition"], state=st,
        elapsed=elapsed, queue=queue, submit=submit, start=start, end=end,
        node=r["NodeList"], gpu=gpu, cluster=cluster,
        wave=wave_of(submit.date(), r["JobName"]) if submit else "unknown",
    ))

# ---- Kaggle notebook runs -------------------------------------------------
kaggle = []
for f in sorted(glob.glob(os.path.join(KAGGLE, "*log.csv"))):
    rr = list(csv.DictReader(open(f)))
    t = float(rr[-1]["cum_time_s"])
    kaggle.append(dict(run=os.path.basename(f).replace("results__", "").replace("__log.csv", ""),
                       steps=int(rr[-1]["step"]), elapsed=t))

out = {}
out["n_slurm_jobs"] = len(jobs)
out["slurm_seconds"] = sum(j["elapsed"] for j in jobs)
out["n_kaggle_runs"] = len(kaggle)
out["kaggle_seconds"] = sum(k["elapsed"] for k in kaggle)
out["total_seconds"] = out["slurm_seconds"] + out["kaggle_seconds"]
out["queue_seconds"] = sum(j["queue"] for j in jobs if j["queue"] is not None)
out["n_with_queue"] = sum(1 for j in jobs if j["queue"] is not None)

# ---- by GPU type (the requested table) ------------------------------------
by_gpu = defaultdict(lambda: dict(n=0, sec=0.0, queue=0.0, cluster="", _cl=defaultdict(float)))
for j in jobs:
    e = by_gpu[j["gpu"]]
    e["n"] += 1; e["sec"] += j["elapsed"]
    # attribute the cluster label by where the time actually went, not by last-seen
    e["_cl"][j["cluster"]] += j["elapsed"] + 1e-6
    if j["queue"] is not None:
        e["queue"] += j["queue"]
for e in by_gpu.values():
    e["cluster"] = max(e["_cl"].items(), key=lambda x: x[1])[0]
    del e["_cl"]
by_gpu["NVIDIA Tesla T4 (Kaggle)"] = dict(
    n=len(kaggle), sec=sum(k["elapsed"] for k in kaggle), queue=0.0, cluster="Kaggle notebooks")
out["by_gpu"] = {k: dict(v) for k, v in by_gpu.items()}

# ---- by partition ---------------------------------------------------------
by_part = defaultdict(lambda: dict(n=0, sec=0.0, queue=0.0, nq=0))
for j in jobs:
    e = by_part[j["part"]]
    e["n"] += 1; e["sec"] += j["elapsed"]
    if j["queue"] is not None:
        e["queue"] += j["queue"]; e["nq"] += 1
out["by_partition"] = {k: dict(v) for k, v in by_part.items()}

# ---- by wave --------------------------------------------------------------
by_wave = defaultdict(lambda: dict(n=0, sec=0.0, queue=0.0, gpus=set(), d0=None, d1=None))
for j in jobs:
    e = by_wave[j["wave"]]
    e["n"] += 1; e["sec"] += j["elapsed"]; e["gpus"].add(j["gpu"])
    if j["queue"] is not None:
        e["queue"] += j["queue"]
    d = j["submit"].date()
    e["d0"] = d if e["d0"] is None else min(e["d0"], d)
    e["d1"] = d if e["d1"] is None else max(e["d1"], d)
out["by_wave"] = {k: dict(n=v["n"], sec=v["sec"], queue=v["queue"],
                          gpus=sorted(v["gpus"]), d0=str(v["d0"]), d1=str(v["d1"]))
                  for k, v in by_wave.items()}

# ---- outcomes -------------------------------------------------------------
by_state = defaultdict(lambda: dict(n=0, sec=0.0))
for j in jobs:
    by_state[j["state"]]["n"] += 1
    by_state[j["state"]]["sec"] += j["elapsed"]
out["by_state"] = {k: dict(v) for k, v in by_state.items()}

# ---- worst queue waits ----------------------------------------------------
q = sorted([j for j in jobs if j["queue"]], key=lambda x: -x["queue"])[:10]
out["worst_queue"] = [dict(jobid=j["jobid"], name=j["name"], part=j["part"],
                           queue_h=j["queue"] / 3600, run_h=j["elapsed"] / 3600) for j in q]

out["kaggle_runs"] = kaggle
dates = sorted(j["submit"].date() for j in jobs if j["submit"])
out["first_job"] = str(dates[0]); out["last_job"] = str(dates[-1])
out["active_days"] = len(set(dates))
out["span_days"] = (dates[-1] - dates[0]).days + 1

json.dump(out, open(os.path.join(os.path.dirname(__file__), "compute_summary.json"), "w"),
          indent=2, default=str)

# ---- print ----------------------------------------------------------------
H = 3600
print(f"Slurm jobs      : {out['n_slurm_jobs']}")
print(f"Kaggle runs     : {out['n_kaggle_runs']}")
print(f"Slurm run time  : {out['slurm_seconds']/H:8.2f} h")
print(f"Kaggle run time : {out['kaggle_seconds']/H:8.2f} h")
print(f"TOTAL run time  : {out['total_seconds']/H:8.2f} h")
print(f"Queue wait      : {out['queue_seconds']/H:8.2f} h  ({out['n_with_queue']} jobs)")
print(f"Span            : {out['first_job']} .. {out['last_job']}  "
      f"({out['span_days']} days, {out['active_days']} active)")
print("\n-- BY GPU TYPE --")
for k, v in sorted(out["by_gpu"].items(), key=lambda x: -x[1]["sec"]):
    print(f"{k:<28} {v['n']:>4} runs  {v['sec']/H:8.2f} h   queue {v['queue']/H:8.2f} h  [{v['cluster']}]")
print("\n-- BY WAVE --")
for k, v in sorted(out["by_wave"].items()):
    print(f"{k:<48} {v['n']:>4} jobs {v['sec']/H:7.2f} h  queue {v['queue']/H:7.2f} h  {v['d0']}..{v['d1']}")
print("\n-- BY PARTITION --")
for k, v in sorted(out["by_partition"].items(), key=lambda x: -x[1]["sec"]):
    mq = v["queue"] / v["nq"] / 60 if v["nq"] else 0
    print(f"{k:<18} {v['n']:>4} jobs {v['sec']/H:7.2f} h  queue {v['queue']/H:7.2f} h (mean {mq:6.1f} min)")
print("\n-- OUTCOMES --")
for k, v in sorted(out["by_state"].items(), key=lambda x: -x[1]["n"]):
    print(f"{k:<12} {v['n']:>4} jobs  {v['sec']/H:7.2f} h")
print("\n-- WORST QUEUE WAITS --")
for w in out["worst_queue"][:6]:
    print(f"{w['jobid']:>9} {w['name']:<16} {w['part']:<16} waited {w['queue_h']:6.2f} h, ran {w['run_h']:.2f} h")
