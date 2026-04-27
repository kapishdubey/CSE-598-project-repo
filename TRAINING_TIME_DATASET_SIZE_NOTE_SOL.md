# Training time vs dataset size (SOL) — what actually matters

This note explains why “X GB of data” does **not** reliably predict how long 10 epochs will take, and gives the exact SOL command patterns we used for baseline/improved training and Option‑A evaluation.

---

## 1) Why “GB on disk” ≠ “training time”

Training time per epoch is dominated by two factors:

1. **How many training samples the dataset loader produces** (not how many files/GB exist)
2. **Seconds per batch** (compute + I/O), which can vary by **10×** based on dataloader workers, backbone, AMP, etc.

Roughly:

\[
\text{epoch\_time} \approx \text{num\_batches} \times \text{sec\_per\_batch}
\]

where:
- `num_batches ≈ (#samples / batch_size)`
- `sec_per_batch` depends on GPU compute, model size, AMP, dataloader parallelism, storage, etc.

### 1.1 The “real dataset size” for training is the loader’s sample count

In this repo, the dataset loader reports:

> `Loading N lidars from M folders`

That `N` is the number of *training samples* actually produced (after windowing, skipping first/last frames, etc.). This is far more predictive than GB.

Example from our baseline:
- **Training samples**: `Loading 28812 lidars from 16 folders`
- **Batch size**: 12
- `num_batches ≈ 28812 / 12 = 2401` (matches tqdm: `0/2401`)

### 1.2 Seconds per batch is strongly affected by dataloader workers

If `num_workers=0` in the single‑GPU path, the GPU stalls waiting on Python I/O.

We observed a **~15× speedup** when switching the single‑GPU dataloader to:
- `num_workers=8`
- `persistent_workers=True`

So a “smaller dataset” can still train slower if `num_workers=0`.

### 1.3 Other major speed factors

- **Backbone/model size**: fewer parameters → faster batches
  - baseline RegNetY transfuser: ~168M trainable params
  - improved EffNetV2‑S transfuser: ~50M params
- **AMP (`--use_amp 1`)**: can speed up, but can also introduce NaNs if not handled carefully (see NaN doc).
- **Batch size & accumulation**: affects throughput and stability
- **Storage**: BeeGFS first‑touch and small-file metadata can bottleneck

---

## 2) What to ask/check in a training log (for time estimates)

### 2.1 Sample count + batch count

Look for:
- `Loading N lidars from M folders`
- tqdm total batches (e.g. `0/2401`)

You can estimate batches:
- `batches ≈ N / batch_size`

### 2.2 Speed (sec/it or it/s)

tqdm shows either:
- `6.37s/it` (seconds per batch), or
- `2.30it/s` (iterations per second → invert to get seconds per batch)

Then:
- `epoch_time ≈ batches × sec_per_batch`

Example:
- `2401 batches × 0.43 s/it ≈ 1032 s ≈ 17.2 min/epoch`

---

## 3) Our SOL training workflow (baseline/improved)

### 3.1 Persistent terminal (tmux)

```bash
tmux new -s improved
# detach: Ctrl+b then d
# reattach: tmux attach -t improved
```

### 3.2 Training via sbatch (recommended)

We used sbatch scripts so the job runs even if you disconnect. Typical pattern:

```bash
sbatch /scratch/$USER/transfuser/repo/train_<name>.sbatch
squeue -u $USER
tail -f /scratch/$USER/transfuser/log/slurm-<JOBID>.err   # tqdm progress
```

### 3.3 Env activation pattern (important on SOL)

```bash
module purge
module load mamba/latest
source \"$(conda info --base)/etc/profile.d/conda.sh\"
conda activate tfuse
export PATH=\"$HOME/.conda/envs/tfuse/bin:$PATH\"
export LD_LIBRARY_PATH=\"$HOME/.conda/envs/tfuse/lib:${LD_LIBRARY_PATH:-}\"
hash -r
```

---

## 4) Option‑A evaluation workflow (held‑out towns)

We built a test suite of 3 “test cases” (Town03/Town05/Town07) using symlinks:

```bash
BASE=/scratch/$USER/transfuser
DATA=$BASE/data
CASES=$BASE/eval_cases
mkdir -p $CASES

for T in Town03 Town05 Town07; do
  mkdir -p $CASES/case_${T}/left_dataset_23_11
  mkdir -p $CASES/case_${T}/right_dataset_23_11

  ln -sfn $DATA/left_dataset_23_11/Routes_routes_10mshortroutes_${T}_Scenario9_Seed0 \
          $CASES/case_${T}/left_dataset_23_11/

  ln -sfn $DATA/right_dataset_23_11/Routes_routes_10mshortroutes_${T}_Scenario8junction_Seed1000 \
          $CASES/case_${T}/right_dataset_23_11/
done
```

Then we evaluated a checkpoint on each case and wrote:
- `baseline_eval_town03_05_07.json`
- `improved_eval_town03_05_07*.json`

Metrics reported:
- per-town `loss_total`
- across-town mean/std/stderr for `loss_total` and key components

---

## 5) Quick checklist for someone else training on a “smaller” dataset

1. Record disk size (for context only):
   ```bash
   du -sh /scratch/$USER/transfuser/data
   ```
2. Record the **loader sample count**:
   - from logs: `Loading N lidars from M folders`
3. Confirm single‑GPU dataloader uses workers:
   ```bash
   grep -n \"num_workers\" team_code_transfuser/train.py | head -25
   ```
4. Record actual per-epoch speed from tqdm:
   - `s/it` or `it/s`

With those four, you can predict total wall time far better than “GB”.

---

## 6) Full Option‑A evaluation script (baseline + improved)

This is the exact script used for our Option‑A evaluation on the 3 held‑out towns
(Town03/Town05/Town07). It:

- loads one checkpoint,
- evaluates it on each case directory under `/scratch/$USER/transfuser/eval_cases/case_<TownXX>/`,
- prints per-town `loss_total`,
- prints across-town mean/std/stderr for key losses,
- writes a JSON with per-case metrics + summary.

### 6.1 Baseline script (RegNetY 032; CKPT = `model_10.pth`)

Run on a GPU node (recommended). It automatically selects `BATCH_SIZE=12` on
A100-80GB and `BATCH_SIZE=6` otherwise.

```bash
python - <<'PY'
import os, math, json, statistics
import torch

REPO_TC = "/scratch/kdubey3/transfuser/repo/team_code_transfuser"
CKPT = "/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/model_10.pth"
CASES_ROOT = "/scratch/kdubey3/transfuser/eval_cases"
CASES = ["Town03", "Town05", "Town07"]
NUM_WORKERS = 8

import sys
sys.path.insert(0, REPO_TC)
from config import GlobalConfig
from model import LidarCenterNet

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
else:
    vram_gb = 0.0

BATCH_SIZE = 12 if vram_gb >= 70 else 6
print("Device:", device, "VRAM_GB:", round(vram_gb, 1), "BATCH_SIZE:", BATCH_SIZE)

def load_model(cfg):
    net = LidarCenterNet(
        cfg, device, backbone=cfg.backbone,
        image_architecture="regnety_032",
        lidar_architecture="regnety_032",
        use_velocity=False,
    )
    sd = torch.load(CKPT, map_location=device)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net

def build_weights(cfg):
    return {k: float(cfg.detailed_losses_weights[i]) for i, k in enumerate(cfg.detailed_losses)}

@torch.no_grad()
def eval_case(case_name):
    root_dir = os.path.join(CASES_ROOT, f"case_{case_name}")
    cfg = GlobalConfig(root_dir=root_dir, setting="all")
    cfg.backbone = "transFuser"
    cfg.use_target_point_image = True
    cfg.use_point_pillars = False
    cfg.augment = False  # deterministic eval

    from data import CARLA_Data
    ds = CARLA_Data(root=cfg.train_data, config=cfg, shared_dict=None)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True
    )

    net = load_model(cfg)
    w = build_weights(cfg)

    sums = {k: 0.0 for k in cfg.detailed_losses}
    total_sum = 0.0
    n_batches = 0

    for batch in dl:
        losses = net(
            batch["rgb"].to(device, dtype=torch.float32),
            batch["lidar"].to(device, dtype=torch.float32),
            ego_waypoint=batch["ego_waypoint"].to(device, dtype=torch.float32),
            target_point=batch["target_point"].to(device, dtype=torch.float32),
            target_point_image=batch["target_point_image"].to(device, dtype=torch.float32),
            ego_vel=batch["speed"].to(device, dtype=torch.float32).reshape(-1, 1),
            bev=batch["bev"].to(device, dtype=torch.long),
            label=batch["label"].to(device, dtype=torch.float32),
            depth=batch["depth"].to(device, dtype=torch.float32),
            semantic=batch["semantic"].squeeze(1).to(device, dtype=torch.long),
            num_points=batch.get("num_points", None),
        )

        total = 0.0
        for k, v in losses.items():
            if k not in sums:
                continue
            val = float((w.get(k, 0.0) * v).mean().item())
            sums[k] += val
            total += val
        total_sum += total
        n_batches += 1

    out = {k: sums[k] / n_batches for k in sums}
    out["loss_total"] = total_sum / n_batches
    out["batches"] = n_batches
    out["case"] = case_name
    return out

results = []
for c in CASES:
    r = eval_case(c)
    print(f"Case {c}: batches={r['batches']} loss_total={r['loss_total']:.6f}")
    results.append(r)

keys = [k for k in results[0].keys() if k.startswith("loss_")]
summary = {}
print("\n=== Across-case summary (mean, std, stderr) over 3 test cases ===")
for k in sorted(keys):
    vals = [r[k] for r in results]
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    stderr = std / math.sqrt(len(vals)) if len(vals) > 0 else 0.0
    summary[k] = {"mean": mean, "std": std, "stderr": stderr}
    if k in ("loss_total", "loss_wp", "loss_bev", "loss_depth", "loss_semantic"):
        print(f"{k:>12}: mean={mean:.6f} std={std:.6f} stderr={stderr:.6f}")

out_path = os.path.join(CASES_ROOT, "baseline_eval_town03_05_07.json")
with open(out_path, "w") as f:
    json.dump({"per_case": results, "summary": summary}, f, indent=2)
print("\nWrote:", out_path)
PY
```

### 6.2 Improved script changes (EffNetV2‑S)

For the improved model, keep the script identical except:

1. Change the checkpoint:
   - `CKPT = "/scratch/kdubey3/transfuser/log/improved_run_<JOBID>/improved/model_<E>.pth"`
2. Change the model architectures in `load_model()`:
   - `image_architecture="tf_efficientnetv2_s_in21ft1k"`
   - `lidar_architecture="tf_efficientnetv2_s_in21ft1k"`
3. Change the output JSON name:
   - `improved_eval_town03_05_07.json` (or add `_epoch2` if evaluating `model_2.pth`)

