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

# Diagnostics
python ECGEncoderTransformer/diagnose_masks.py
python ECGEncoderTransformer/debug_nan_loss.py --max_batches 300
python ECGEncoderTransformer/diagnose_collapse.py --max_samples 2000 --epochs 5
python ECGEncoderTransformer/verify_collapse_fix.py --max_samples 2000 --mini_steps 120

# Slurm GPU
sbatch ECGEncoderTransformer/run_train.sh

# Debug subset (recommended stable hyperparams)
sbatch ECGEncoderTransformer/run_train.sh --max_samples 5000 --epochs 10
```

## Collapse / NaN (Slurm job failure mode)

If training logs show `non-finite loss` mid-epoch, then `param_l2=nan` and val predictions are **100% class 0**, the model weights were corrupted (not a benign majority-class optimum). Typical trigger: `p2f_loss_weight=10` + inverse-frequency class weights + `lr=5e-4` on sparse p2f mini-batches.

**Fixes in this repo (encoder stays frozen; architecture unchanged):**

- Combined causal+padding attention mask (`build_combined_attn_mask`) and logits clamp in `model.py`
- Train loop weight rollback on non-finite loss/grad; only save `best.pt` when val loss is finite
- Safer defaults in `config.py`: `LR=1e-4`, `P2F_LOSS_WEIGHT=1.0`, `MAX_GRAD_NORM=1.0`, class weights clipped to `[0.25, 4.0]`
- `collate_ecg_window_batch` sanitizes non-finite waveforms

Reproduce vs verify:

```bash
# Legacy unstable settings (expect collapse / NaN on 2k subset)
python ECGEncoderTransformer/diagnose_collapse.py --max_samples 2000 --epochs 5
# Check output/diagnose_collapse/diagnose_collapse.json — experiment A_legacy_defaults

# Full train with new defaults
sbatch ECGEncoderTransformer/run_train.sh --max_samples 5000 --epochs 10
# Expect: val pred_s2f unique_classes=3, train skipped non-finite loss=0
```

## Architecture

```
ecg_seq [B,T,12,L]
  → frozen ECG encoder (only valid timesteps) → [B,T,D]
  → Linear → d_model + sinusoidal pos
  → causal TransformerEncoderBlocks (key_padding_mask + bool causal mask)
  → optional learnable anchor slot at end; pool last valid (= anchor)
  → head_s2f / head_p2f (MLP → 3 logits each)
```

Trainable: `proj`, transformer layers, `head_s2f`, `head_p2f`, `miss_ecg`, `anchor_slot`.  
Frozen by default: full `ecg_enc`.

## Config

See `config.py` — `ECG_TARGET_LEN=1000`, `INCLUDE_ANCHOR_SLOT=True`, `USE_CLASS_WEIGHTS=True`.
