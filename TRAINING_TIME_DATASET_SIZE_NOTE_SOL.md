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

