# CXREncoderTransformer

**Frozen ViT CXR encoder → causal Transformer → dual MLP heads**  
Predict anchor-time **s2f / p2f severity change (12–24h)** (3-class each, masked CE).

## Entry point (run this)

From repo root:

```bash
sbatch CXREncoderTransformer/run_train.sh
```

Debug (small subset):

```bash
sbatch CXREncoderTransformer/run_train.sh --max_samples 5000 --epochs 5
```

## Files

| File | Role |
|------|------|
| [`run_train.sh`](run_train.sh) | **Slurm entry** — submit with `sbatch` |
| [`train.py`](train.py) | Training loop |
| [`model.py`](model.py) | `CXREncoderTransformer` |
| [`config.py`](config.py) | Defaults |
| [`output/`](output/) | `best.pt`, `results.json` |

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
python CXREncoderTransformer/validate_data.py --max_samples 32
```

Legacy runtime window: `sbatch CXREncoderTransformer/run_train.sh --use_runtime_catalog`.

## Model vs EHRWindowTransformer `CXRWindowTransformer`

| | CXRWindowTransformer | **CXREncoderTransformer** |
|--|----------------------|---------------------------|
| Transformer | Bidirectional | **Causal** |
| Classification head | Linear | **MLP** |
| Folder | `EHRWindowTransformer/` | **`CXREncoderTransformer/`** |

## Architecture

```text
CXR seq [t-24h, t-12h]
  → frozen ViT (CXREncoder)
  → Linear proj + positional encoding
  → causal Transformer ×4
  → anchor pool (last)
  → head_s2f MLP / head_p2f MLP → 3 classes each
```
