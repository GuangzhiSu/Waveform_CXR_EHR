# CXREncoderTransformer

**Frozen ViT CXR encoder → causal Transformer → dual MLP heads**  
Predict anchor-time **s2f / p2f severity change (12–24h)** (3-class each, masked CE).

## Entry point (run this)

From repo root:

```bash
sbatch legacy/encoder_transformers/CXREncoderTransformer/run_train.sh
```

Debug (small subset):

```bash
sbatch legacy/encoder_transformers/CXREncoderTransformer/run_train.sh --max_samples 5000 --epochs 15
```

## Collapse / single-class predictions

If val logs show `train_loss ≈ 1.099` (random 3-class baseline), `val pred_s2f` stuck on one class, or accuracy exactly equal to majority baseline, the model was not using CXR pixels (see `diagnose_collapse` experiment B).

**Fixes in this repo:**

- Combined causal+padding attention mask (`build_combined_attn_mask`) and logits clamp in `model.py` (aligned with `ECGEncoderTransformer`)
- Train loop: class-weight clip `[0.25, 4.0]`, non-finite loss/grad rollback, `unique_classes` in epoch logs
- Safer defaults: `LR=1e-4`, `P2F_LOSS_WEIGHT=1.0`, `MAX_GRAD_NORM=1.0`
- `collate_cxr_window_batch` drops anchors with zero loadable CXR slots

Verify before full Slurm train:

```bash
python legacy/encoder_transformers/CXREncoderTransformer/test_pool_anchor_index.py
python legacy/encoder_transformers/CXREncoderTransformer/verify_collapse_fix.py --max_samples 2000 --mini_steps 80
python legacy/encoder_transformers/CXREncoderTransformer/diagnose_collapse.py --max_samples 3000 --epochs 5
```

Expect: `VERIFIED=True`, diagnose B `acc_drop >= 0.05`, val `unique_classes(s2f/p2f) >= 2`.

## Files

| File | Role |
|------|------|
| [`run_train.sh`](../../legacy/encoder_transformers/CXREncoderTransformer/run_train.sh) | **Slurm entry** — submit with `sbatch` |
| [`train.py`](../../legacy/encoder_transformers/CXREncoderTransformer/train.py) | Training loop |
| [`model.py`](../../legacy/encoder_transformers/CXREncoderTransformer/model.py) | `CXREncoderTransformer` |
| [`config.py`](../../legacy/encoder_transformers/CXREncoderTransformer/config.py) | Defaults |
| [`verify_collapse_fix.py`](../../legacy/encoder_transformers/CXREncoderTransformer/verify_collapse_fix.py) | Quick check: multi-class preds + image sensitivity |
| [`diagnose_collapse.py`](../../legacy/encoder_transformers/CXREncoderTransformer/diagnose_collapse.py) | Collapse root-cause experiments |
| [`output/`](../../legacy/encoder_transformers/CXREncoderTransformer/output/) | `best.pt`, `results.json` |

## Data

| Role | Default path |
|------|----------------|
| **Training data (default)** | `data/p2f_or_s2f_cxr_catalog_labeled.csv` |
| Raw CXR catalog (runtime window mode) | `data/p2f_or_s2f_cxr_catalog.csv` |

**Default (`CXRLabeledCatalogDataset`):** reads labeled CSV, groups by `anchor_index`, one sample = one anchor + its pre-matched CXRs in `(12h, 24h]` after each CXR. Labels from the same file.

Generate labeled CSV:

```bash
python data/enrich_cxr_catalog_anchor_labels.py
```

Validate loading:

```bash
python legacy/encoder_transformers/CXREncoderTransformer/validate_data.py --max_samples 32
```

Legacy runtime window: `sbatch legacy/encoder_transformers/CXREncoderTransformer/run_train.sh --use_runtime_catalog`.

## Model vs EHRWindowTransformer `CXRWindowTransformer`

| | CXRWindowTransformer | **CXREncoderTransformer** |
|--|----------------------|---------------------------|
| Transformer | Bidirectional | **Causal** |
| Classification head | Linear | **MLP** |
| Folder | `legacy/ehr_temporal/EHRWindowTransformer/` | **`legacy/encoder_transformers/CXREncoderTransformer/`** |

## Architecture

```text
CXR seq [t-24h, t-12h]
  → frozen ViT (CXREncoder)
  → Linear proj + positional encoding
  → causal Transformer ×4
  → anchor pool (last)
  → head_s2f MLP / head_p2f MLP → 3 classes each
```
