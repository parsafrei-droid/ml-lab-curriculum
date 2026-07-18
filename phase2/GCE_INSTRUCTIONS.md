# Google Compute Engine (GCE) — setup & usage for this project

> Source: supervisor's GCE onboarding instructions. Saved here for reference.
> **Adapted run steps for THIS project are in the second half of this file.**

## ⚠️ The one rule that costs money if you forget it
**ALWAYS STOP YOUR INSTANCE AFTER USE.** Never leave it running idle. A running
GPU VM burns the coupon even when nothing is training. Stop it and wait until it
has fully shut down.

---

## Setup (from the supervisor's instructions)

1. **ALWAYS STOP YOUR INSTANCES AFTER USAGE.** Never leave them on without
   running something.
2. Email your supervisor stating you need a GCE account, including
   `<unique_name_of_your_final_project>` and **all** `<gmail>` addresses of the
   group members.
3. Log in with your gmail account and go to <https://console.cloud.google.com/>.
4. You should now see a project at the top left with
   `<unique_name_of_your_final_project>`. If not, repeat Step 2.
5. **One** student in the group activates a **$25 coupon** that serves as the
   billing account.
6. Change the billing account of the project to **"Billing Account for
   Education"**.
   <https://cloud.google.com/billing/docs/how-to/modify-project>
7. When you finish the coupon of the last team member (~$10 remaining), activate a
   new coupon.

## Computing

8. Click **"Go to Compute Engine"**.
9. Select the instance `deeplab2022-instance` (the name may differ depending on the
   supervisor who created it) and click **"Start"** (three-dots menu) to launch it.
10. The instance is a preconfigured VM: **4 CPU cores, 15 GB memory, 128 GB SSD,
    1 NVIDIA K80 GPU, Ubuntu 18.04, CUDA 11.3, NVIDIA drivers.**
11. Click **"SSH"** to open a terminal and configure the machine (you have sudo
    rights). You are now ready to work.
12. When finished, **"Stop"** the instance and wait until it shuts down.

## Optional

13. To access the instance from a terminal outside the console, install the
    `gcloud` CLI on your laptop and connect it to your Google account.
14. To copy data to the instance from outside, use `gcloud compute scp` or GitHub.
15. If you need a GUI, follow the VNC-server tutorial.

---

## ⚠️ Important reality check for THIS project (read before running)

The VM's GPU is an **NVIDIA K80 (~11–12 GB VRAM)** — that is **less** memory than
the Colab T4 (14.5 GB) that already hit CUDA out-of-memory on the baseline config
(`max_features=100, batch_size=32`). So on the K80 the same config will OOM too.

Also: **Ubuntu 18.04 + CUDA 11.3** is old. The K80 is a legacy GPU that recent
PyTorch builds have dropped support for. The preinstalled environment may need a
specific older torch. We handle this in the run steps below.

Because of both issues, the K80 VM is **not clearly better than Colab** for this
job. Consider it a fallback, or use it with the reduced-batch config below.

---

## Adapted run steps for THIS project (baseline_default_prior)

Once you are SSH'd into the started instance:

```bash
# 0. confirm the GPU and CUDA are visible
nvidia-smi

# 1. get the code (repo is private; use a token or make it public first)
git clone --branch curriculum-fixes --single-branch \
    https://github.com/parsafrei-droid/ml-lab-curriculum.git
cd ml-lab-curriculum

# 2. run the setup (clones pinned deps, installs, applies the tabicl patch, stubs)
python3 phase2/colab_setup.py
#    NOTE: colab_setup.py hardcodes /content/ml-lab-curriculum (a Colab path).
#    On the GCE VM the repo is NOT under /content — see "GCE fix" note below.

# 3. pretrain the baseline (reduced batch so it fits the K80's smaller VRAM)
python3 scripts/run.py --config phase2/configs/baseline_default_prior.yaml \
    --epochs 50 --name baseline_default_prior_e50_s42

# 4. evaluate on TabArena
python3 scripts/eval_tabarena.py \
    --checkpoint results/baseline_default_prior_e50_s42/checkpoint.pth \
    --tasks tabarena

# 5. STOP THE INSTANCE from the console when done.
```

### GCE fix needed before this works
`phase2/colab_setup.py` currently assumes the Colab path `/content/ml-lab-curriculum`.
On the GCE VM the repo lives wherever you cloned it (e.g. `~/ml-lab-curriculum`), so
that script must be made path-agnostic first: add a `--repo` argument
or auto-detect the repo root before running on GCE.

### Batch-size / OOM fix
The K80 will OOM on `batch_size: 32`. Either lower `batch_size` in
`phase2/configs/baseline_default_prior.yaml` (e.g. to 8 or 4) — which does NOT
change the prior, only the training batching — or set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running.
