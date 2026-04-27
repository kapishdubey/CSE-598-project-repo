# Baseline Training + Option-A Evaluation Report (SOL) — TransFuser (2022)

This document summarizes the **baseline** training run and the **Option A** (held-out, loss-based) evaluation performed on ASU SOL for the CSE 598 final project.

---

## 1) Baseline training run (SOL)

### 1.1 SLURM job metadata

- **Job ID**: `51857823`
- **State**: `COMPLETED`
- **Exit code**: `0:0`
- **Elapsed**: `02:36:23`
- **Node**: `sg039`
- **GPU**: `NVIDIA A100-SXM4-80GB (80GB)`

### 1.2 Training arguments used (from `args.txt`)

Path:

- `/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/args.txt`

Key args (verbatim):

```json
{
  "id": "transfuser_sol",
  "epochs": 10,
  "lr": 0.0001,
  "batch_size": 12,
  "logdir": "/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol",
  "setting": "all",
  "root_dir": "/scratch/kdubey3/transfuser/data",
  "schedule": 1,
  "schedule_reduce_epoch_01": 7,
  "schedule_reduce_epoch_02": 9,
  "backbone": "transFuser",
  "image_architecture": "regnety_032",
  "lidar_architecture": "regnety_032",
  "use_velocity": 0,
  "n_layer": 4,
  "wp_only": 0,
  "use_target_point_image": 1,
  "use_point_pillars": 0,
  "parallel_training": 0,
  "val_every": 100,
  "no_bev_loss": 0,
  "sync_batch_norm": 0,
  "zero_redundancy_optimizer": 0,
  "use_disk_cache": 0
}
```

### 1.3 Baseline outputs (checkpoints + logs)

Run directory:

- `/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/`

Final checkpoint artifacts:

- **Model**: `/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/model_10.pth` (643 MB)
- **Optimizer**: `/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/optimizer_10.pth` (1.3 GB)

TensorBoard event file(s):

- `/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/events.out.tfevents.`*

### 1.4 What data was used for training (type + coverage)

**Data type**: CARLA logged expert-imitation dataset (generated via privileged autopilot) with:

- RGB images
- LiDAR point clouds (converted to BEV histogram features at load time)
- topdown/BEV semantic map
- depth + semantic segmentation labels
- detection labels (CenterNet-style heatmap/wh/offset/yaw)
- measurements metadata (ego + target point)

**Training root**:

- `/scratch/kdubey3/transfuser/data`

**Dataset directories included**:

- `/scratch/kdubey3/transfuser/data/left_dataset_23_11`
- `/scratch/kdubey3/transfuser/data/right_dataset_23_11`

**Towns covered (8 towns)**:

- Town01, Town02, Town03, Town04, Town05, Town06, Town07, Town10HD

**Scenario coverage (as encoded by folder naming)**:

- **Left dataset**: `Scenario9` (`..._Scenario9_Seed0`)
- **Right dataset**: `Scenario8junction` (`..._Scenario8junction_Seed1000`)

**Top-level folder count**:

- 8 (left) + 8 (right) = **16 town folders total**

**Route subfolder counts** (total ≈ **673 routes**):

- Left (Scenario9): **323 routes** total across 8 towns
- Right (Scenario8junction): **350 routes** total across 8 towns
- Combined: **673 routes**

**Sample count created by dataset windowing**:

- Training log printed: `Loading 28812 lidars from 16 folders`
  - i.e., **28,812 training samples** (each sample corresponds to a valid indexed frame window in the dataset class).

**Example per-route frame count**:

- Example route had **41** LiDAR frames (`lidar/*.npy`).

### 1.5 Amount of data (GB)

Measured with `du -sh`:

- `/scratch/kdubey3/transfuser/data/left_dataset_23_11`: **17G**
- `/scratch/kdubey3/transfuser/data/right_dataset_23_11`: **15G**
- `/scratch/kdubey3/transfuser/data` (total): **32G**

---

## 2) Baseline evaluation (Option A: held-out towns, loss-based)

### 2.1 What Option A “results” mean

Option A evaluates the trained checkpoint on **held-out logged data** and reports the same **supervised losses** used in training (e.g., waypoint loss, BEV loss, depth loss, semantic loss, detection losses).

Important: these metrics are **not CARLA driving scores** (no collision/route-completion metrics). They are **loss-based generalization metrics**: how well the network predicts expert labels on unseen towns/routes from the same data-generation regime.

### 2.2 Test cases used (3 held-out towns)

We evaluated on 3 test cases (each test case is one town, including both left+right scenario folders):

- **Town03**
- **Town05**
- **Town07**

Test-case roots (symlinked into an eval folder):

- `/scratch/kdubey3/transfuser/eval_cases/case_Town03/`
- `/scratch/kdubey3/transfuser/eval_cases/case_Town05/`
- `/scratch/kdubey3/transfuser/eval_cases/case_Town07/`

### 2.3 Checkpoint evaluated

- `/scratch/kdubey3/transfuser/log/run_51857823/transfuser_sol/model_10.pth`

### 2.4 Evaluation outputs

JSON results:

- `/scratch/kdubey3/transfuser/eval_cases/baseline_eval_town03_05_07.json`

Plots (generated from that JSON):

- `/scratch/kdubey3/transfuser/eval_cases/plots/baseline_loss_total_by_town.png`
- `/scratch/kdubey3/transfuser/eval_cases/plots/baseline_component_means_stderr.png`

### 2.5 Baseline Option-A results (mean/std/stderr across 3 test cases)

Across-case summary over the 3 held-out towns:

- `loss_total`: **mean 1.259721**, std 0.119866, stderr 0.069205
- `loss_wp`: mean 0.150623, std 0.030095, stderr 0.017375
- `loss_bev`: mean 0.247726, std 0.023751, stderr 0.013713
- `loss_depth`: mean 0.298997, std 0.045654, stderr 0.026358
- `loss_semantic`: mean 0.117594, std 0.023111, stderr 0.013343

Interpretation highlights:

- Performance differs by town (Town05 notably easier than Town03/Town07 in loss_total).
- Among the plotted components, depth + BEV are the largest contributors (under the current weighting).

---

## 3) Next steps after teammate merges improvements into `georgieee03/transfuser`

### 3.1 Train the “improved” model

After merging:

- training-recipe changes (`train.py`, e.g. AMP / grad accumulation / cosine LR / clipping)
- uncertainty-weighted loss (your PR, `--uncertainty_weights`)

We will run a new training job on the same training root:

- `/scratch/kdubey3/transfuser/data`

Fair comparison choices (pick one and keep it consistent):

- **Same epochs** (e.g. 10) on the same data → compare final test losses directly (simplest).
- Or **same wall-clock budget** → shows speed/efficiency gains (AMP, accumulation).

### 3.2 Re-evaluate on the exact same Option-A test suite

Run the same Option-A evaluation on the **improved** checkpoint using the same towns:

- Town03, Town05, Town07

Produce:

- `improved_eval_town03_05_07.json` (same schema as baseline JSON)
- baseline vs improved plots:
  - loss_total by town (overlay or side-by-side)
  - component means ± stderr (baseline vs improved)

### 3.3 Report improvement

For slides, report baseline vs improved:

- `loss_total` mean/std/stderr across the same 3 test cases
- component losses (wp, bev, depth, semantic) mean±stderr
- (optional) training throughput: epoch time / it/s, especially if AMP is used

