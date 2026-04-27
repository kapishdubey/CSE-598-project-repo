# Improved run NaN investigation (Option‑A eval)

This note documents the issue we hit when evaluating the **improved** TransFuser training run on SOL, where **all evaluation losses became NaN**. It also records the diagnostics we ran, likely root causes, and concrete next steps to permanently fix the problem. Finally, it includes the exact evaluation script used for the improved run.

## Summary (what happened)

- **Baseline** training + Option‑A evaluation on 3 held‑out towns (Town03/Town05/Town07) produced **finite** losses and plots.
- **Improved** training finished and produced checkpoints `model_1.pth` … `model_10.pth`.
- Option‑A evaluation of the **improved** checkpoint `model_10.pth` produced:
  - `loss_total = nan` for Town03, Town05, Town07
  - all component losses (wp, bev, depth, semantic, CenterNet losses) were also NaN
  - the written JSON was therefore unusable for comparisons.

## Improved training context (the run we evaluated)

- **SLURM job**: `51900975` (COMPLETED)
- **Logdir**: `/scratch/kdubey3/transfuser/log/improved_run_51900975/improved`
- **Checkpoint evaluated (initially)**: `model_10.pth`
- **Backbones**:
  - `image_architecture = tf_efficientnetv2_s_in21ft1k`
  - `lidar_architecture  = tf_efficientnetv2_s_in21ft1k`
- **Recipe flags used (from args.txt)**:
  - `--use_amp 1`
  - `--use_cosine_lr 1 --warmup_epochs 1`
  - `--grad_clip 1.0`
  - `--uncertainty_weights 1`
  - `--batch_size 12 --epochs 10`

## Observed failure symptoms

### 1) Evaluation outputs were NaN

During evaluation:

- Town03, Town05, Town07 all printed `loss_total=nan`
- Summary mean/std/stderr were all NaN

This implies **at least one** component loss became NaN in the forward pass, and then NaNs propagated into the total.

### 2) First‑batch per‑loss check: everything was non‑finite

We ran a first‑batch diagnostic on Town03. Output:

- `missing_keys: 256 unexpected_keys: 5`
- Every loss term was non‑finite on the very first batch:
  - `loss_wp`, `loss_bev`, `loss_depth`, `loss_semantic`
  - CenterNet losses: `loss_center_heatmap`, `loss_wh`, `loss_offset`, `loss_yaw_class`, `loss_yaw_res`, `loss_velocity`, `loss_brake`

Because the NaNs appeared immediately, this strongly suggests the **weights** were already corrupted (NaN/Inf), rather than the loss computation gradually producing NaNs during evaluation.

### 3) Checkpoint inspection: thousands of tensors contain NaN/Inf

We scanned checkpoint tensors for NaN/Inf:

- `NaN/Inf tensors in checkpoint: 3005`
- Examples included:
  - `log_vars.wp`, `log_vars.bev`, `log_vars.depth`, `log_vars.semantic`, `log_vars.detection`
  - large portions of `_model.image_encoder...` weights and BN running stats

This confirms the training run diverged and then saved corrupted weights.

### 4) Which epoch diverged?

We counted bad tensors per checkpoint:

```
model_1.pth  bad_tensors: 0
model_2.pth  bad_tensors: 0
model_3.pth  bad_tensors: 3005
model_4.pth  bad_tensors: 3005
...
model_10.pth bad_tensors: 3005
```

So divergence happened between **epoch 2 → epoch 3**. The run is salvageable for comparisons by evaluating `model_2.pth` (last known finite checkpoint), but we still want a permanent fix.

## Likely root causes (most probable → less probable)

### A) AMP (FP16) instability in CenterNet heatmap/focal loss (very likely)

The training recipe enables AMP (`--use_amp 1`). The repo’s detection head uses CenterNet‑style losses (heatmap focal / gaussian focal). These are well known to underflow/overflow in FP16.

The teammate plan itself warned:
> “The CenterNet focal loss (GaussianFocalLoss) can underflow in FP16… force FP32 for that part.”

**Expected symptom**: training looks normal for a while, then suddenly produces NaN gradients/weights, often starting in detection head / BN statistics and quickly propagating.

### B) Uncertainty‑weighted loss parameters (`log_vars`) exploding (likely, but often secondary)

We enabled uncertainty weighting (`--uncertainty_weights 1`), which introduces learnable `log_vars.*`. If any task loss spikes (e.g., due to AMP instability), the gradients on `log_vars` can explode, and the model may quickly diverge.

Note: this does not mean uncertainty weighting is “wrong”; it may just require a stable underlying loss computation (especially with AMP).

### C) LR schedule ordering warning (possible contributor)

Training emitted:
> “Detected call of lr_scheduler.step() before optimizer.step()…”

This can change the early LR schedule. It’s not usually the sole cause of NaNs, but it can increase risk.

### D) Architecture mismatch between eval model and checkpoint (less likely as the *main* cause here)

We saw `missing_keys: 256 unexpected_keys: 5`, indicating a mismatch between how the model is constructed in eval and what was saved. However, since the checkpoint itself contains NaNs/Inf in thousands of tensors, the primary cause is still training divergence. We should still ensure eval config matches `args.txt` exactly.

## Suggested checks (fast triage)

1) **Evaluate `model_2.pth`** (finite) instead of `model_10.pth` to produce an “improved vs baseline” comparison immediately.
2) Run short stability A/B tests (2–3 epochs) to isolate the trigger:
   - AMP OFF, uncertainty ON: `--use_amp 0 --uncertainty_weights 1`
   - AMP ON, uncertainty OFF: `--use_amp 1 --uncertainty_weights 0`
   - If AMP ON triggers divergence, implement FP32 for the heatmap/focal loss.

## Permanent fix recommendations (code‑level)

### 1) Force FP32 for CenterNet heatmap/focal loss when AMP is enabled

In `team_code_transfuser/train.py` AMP path, keep autocast for forward, but compute the fragile detection loss block in full precision:

- wrap `loss_bbox = self.head.loss(...)` inside `with autocast(enabled=False): ...`
- or force only the heatmap/focal components to FP32 (if you want more speed).

### 2) Add NaN/Inf guards during training

Before optimizer/scaler step:

- if `not torch.isfinite(loss)`: skip step and log a warning
- after unscale, optionally check gradients are finite before stepping

This prevents writing corrupted checkpoints and gives immediate visibility.

### 3) Consider stabilizing uncertainty parameters

If needed:

- clamp `log_vars` to a safe range (e.g. `[-10, 10]`)
- or reduce LR for `log_vars` via parameter groups

These are secondary to fixing AMP focal‑loss precision.

## Evaluation scripts

### A) Improved Option‑A evaluation script (EffNetV2‑S)

This is the exact script we used for the **improved** evaluation, with EfficientNetV2‑S hardcoded. It evaluates Town03/05/07 cases under `/scratch/kdubey3/transfuser/eval_cases/` and writes JSON output.

To evaluate `model_2.pth` (last finite checkpoint), set `CKPT` to `.../model_2.pth` and write to a new JSON filename (recommended: `improved_eval_town03_05_07_epoch2.json`).

```python
import os, math, json, statistics
import torch

REPO_TC = "/scratch/kdubey3/transfuser/repo/team_code_transfuser"
CKPT = "/scratch/kdubey3/transfuser/log/improved_run_51900975/improved/model_10.pth"
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
        image_architecture="tf_efficientnetv2_s_in21ft1k",
        lidar_architecture="tf_efficientnetv2_s_in21ft1k",
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
    cfg.augment = False

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

out_path = os.path.join(CASES_ROOT, "improved_eval_town03_05_07.json")
with open(out_path, "w") as f:
    json.dump({"per_case": results, "summary": summary}, f, indent=2)
print("\nWrote:", out_path)
```

## Recommended immediate next action

1) Re‑run the improved eval using `model_2.pth` and write:
   - `improved_eval_town03_05_07_epoch2.json`
2) Generate baseline‑vs‑improved plots and a results table for slides.
3) Implement the permanent training stability fix (FP32 heatmap/focal loss under AMP + NaN guards), then re‑train to get a clean 10‑epoch improved checkpoint.

