# ECGEncoderTransformer

Frozen ECG encoder (MedTVT xresnet1d **or** Symile ResNet18 PL checkpoint) over each waveform in the anchor lookback window, then a **causal transformer**, then dual **MLP** heads for s2f/p2f **severity change** (3-class).

## Data (default)

`data/p2f_or_s2f_ecg_catalog_labeled.csv` — built by:

```bash
python data/enrich_ecg_catalog_anchor_labels.py
```

One training sample = one anchor; `ecg_seq` is all ECGs with `hours_ecg_to_anchor ∈ (12, 24]`, time-sorted. Failed WFDB loads are **skipped** (not kept as invalid mask slots).

## ECG checkpoint

`run_train.sh` / `config.resolve_ecg_ckpt_path()` picks:

1. `MedTVT-R1/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt` → **xresnet1d** SignalEncoder (512-d)
2. Else newest `*.ckpt` → **Symile** ResNet18 (1024-d, auto `ecg_dim=1024`)

Pass `--ecg_ckpt` to override.

## Train

```bash
# Smoke test (login node, no GPU ok for tiny subset)
python ECGEncoderTransformer/validate_data.py
python ECGEncoderTransformer/test_pool_anchor_index.py

# Diagnostics
python ECGEncoderTransformer/diagnose_masks.py
python ECGEncoderTransformer/debug_nan_loss.py --max_batches 300
python ECGEncoderTransformer/diagnose_collapse.py --max_samples 2000 --epochs 5
python ECGEncoderTransformer/verify_collapse_fix.py --max_samples 2000 --mini_steps 120
python ECGEncoderTransformer/verify_nan_fix.py

# Slurm GPU (stable defaults from config.py — do NOT copy EHR/CXR p2f_weight=10)
sbatch ECGEncoderTransformer/run_train.sh --max_samples 5000 --epochs 10

# Explicit stable hyperparams (recommended if overriding CLI)
sbatch ECGEncoderTransformer/run_train.sh --max_samples 5000 --epochs 10 \
  --lr 1e-4 --p2f_loss_weight 1.0 --max_grad_norm 1.0 --anchor_pool mean
```

**Do not** pass `--p2f_loss_weight 10` or `--lr 5e-4` (EHR/CXR defaults); on sparse p2f mini-batches this often yields mid-epoch NaN and collapsed predictions.

## Collapse / NaN (Slurm job failure mode)

If training logs show `non-finite loss` mid-epoch, then `param_l2=nan` and val predictions collapse to one class, weights were corrupted — not a benign majority-class optimum.

**Fixes in this repo (causal encoder unchanged):**

- Causal attention uses **split** `key_padding_mask` + bool causal mask (same as CXREncoderTransformer), not float `-inf` combined mask
- Default **`anchor_pool=mean`**: classify from **mean-pooled ECG token** states; learnable anchor slot still attends causally but is not the sole pooled vector
- Symile encoder `target_time` matches `ecg_target_len` (no 5000↔1000 upsample)
- Train loop rollback on non-finite loss/grad; `best.pt` only when val loss is finite **and** ≥2 unique s2f preds
- Safer defaults: `LR=1e-4`, `P2F_LOSS_WEIGHT=1.0`, `MAX_GRAD_NORM=1.0`, `LABEL_SMOOTHING=0.05`, class weights clipped to `[0.25, 4.0]`
- `collate_ecg_window_batch` sanitizes non-finite waveforms

Reproduce vs verify:

```bash
python ECGEncoderTransformer/diagnose_collapse.py --max_samples 2000 --epochs 5
# Expect experiment F_causal_mean_pool in diagnose_collapse.json to pass

sbatch ECGEncoderTransformer/run_train.sh --max_samples 5000 --epochs 10
# Expect: train skipped non-finite loss=0, val pred_s2f unique_classes=3, finite param_l2
```

## Architecture

```
ecg_seq [B,T,12,L]
  → frozen ECG encoder (only valid timesteps) → [B,T,D]
  → Linear → d_model + sinusoidal pos
  → causal EncoderBlocks (key_padding_mask + bool causal mask)
  → optional learnable anchor slot at end (context only when anchor_pool=mean)
  → mean-pool valid ECG tokens (default) or last-pool (anchor slot if enabled)
  → head_s2f / head_p2f (MLP → 3 logits each)
```

Trainable: `proj`, transformer layers, `head_s2f`, `head_p2f`, `miss_ecg`, `anchor_slot`.  
Frozen by default: full `ecg_enc`.

## Config

See `config.py` — `ECG_TARGET_LEN=1000`, `ANCHOR_POOL=mean`, `INCLUDE_ANCHOR_SLOT=True`, `USE_CLASS_WEIGHTS=True`.
