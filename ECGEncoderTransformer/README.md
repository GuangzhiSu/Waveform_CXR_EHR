# ECGEncoderTransformer

Frozen **baseline2** ECG encoder (`experiment1(old)/baseline/signal_encoder.py` → `models.encoders.ecg.SignalEncoder`, xresnet1d101 + MedTVT checkpoint) over each waveform in the anchor lookback window, then a **causal transformer**, then dual **MLP** heads for s2f/p2f **severity change** (3-class).

## Data (default)

`data/p2f_or_s2f_ecg_catalog_labeled.csv` — built by:

```bash
python data/enrich_ecg_catalog_anchor_labels.py
```

One training sample = one anchor; `ecg_seq` is all ECGs with `hours_ecg_to_anchor ∈ (12, 24]`, time-sorted.

## Train

```bash
# Smoke test (login node, no GPU ok for tiny subset)
python ECGEncoderTransformer/validate_data.py

# Slurm GPU
sbatch ECGEncoderTransformer/run_train.sh

# Debug subset
sbatch ECGEncoderTransformer/run_train.sh --max_samples 500 --epochs 3
```

## Architecture

```
ecg_seq [B,T,12,L]
  → frozen SignalEncoder (per timestep) → [B,T,512]
  → Linear → d_model + sinusoidal pos
  → causal TransformerEncoderBlocks
  → pool last valid timestep (or mean)
  → head_s2f / head_p2f (MLP → 3 logits each)
```

Trainable: `proj`, `pos` (buffer), transformer layers, `head_s2f`, `head_p2f`, `miss_ecg`.  
Frozen by default: full `ecg_enc` (same as baseline2 `freeze_encoder=True`).

## Config

See `config.py` — `ECG_CKPT` under MedTVT `CKPTS/best_valid_all_increase_with_augment_epoch_3.pt`, `ECG_TARGET_LEN=1000` (matches baseline `load_ecg`).
