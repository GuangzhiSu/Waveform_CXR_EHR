# Fusion Baseline: Predict Oxygenation from ECG + CXR

**Goal**: Use ECG waveform and CXR image data to predict oxygenation at the timestamp. EHR oxygenation (e.g. SpO₂) serves as **ground truth** to evaluate prediction accuracy.

- **Model input**: ECG + CXR only
- **Model output**: Predicted oxygenation (continuous, regression)
- **Ground truth**: EHR oxygenation at the same timestamp (spo2, PaO₂, P/F ratio, etc.)

## Architecture

- **CXR Encoder**: ViT (google/vit-base-patch16-224-in21k)
- **Signal Encoder**: xresnet1d101 (ECG, from MedTVT-R1)
- **Fusion**: Concatenate → MLP → 1 (oxygenation value)

No EHR encoder—EHR is used only as labels.

## Paths (set in `run_baseline.sh`)

| Purpose | Path |
|---------|------|
| EHR/matched CSV | `../cxr_supertable_waveform_matched.csv` |
| CXR images | `/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg` |
| CXR metadata | `.../mimic-cxr-2.0.0-metadata.csv.gz` |
| ECG waveforms | Full path in `wf_File_Path` column |
| ECG encoder | `MedTVT-R1/CKPTS/best_valid_all_increase_with_augment_epoch_3.pt` |
| CXR encoder (ViT) | `MedTVT-R1/CKPTS/vit-base-patch16-224` |

## Oxygenation Targets (EHR Ground Truth)

| Column | Description | Availability |
|--------|-------------|--------------|
| `spo2` | SpO₂ (pulse ox) | ~1,984 rows |
| `partial_pressure_of_oxygen_(pao2)` | PaO₂ (ABG) | ~691 rows |
| `s2f_vent_fio2` | SpO₂/FiO₂ ratio | ~418 rows |
| `p2f_vent_fio2` | PaO₂/FiO₂ (P/F) ratio | ~152 rows |

Default: `spo2` (best coverage).

## Usage

### Quick run (recommended)

```bash
cd Waveform_CXR_EHR/baseline
./run_baseline.sh
```

The script sets paths for ECG, CXR, EHR CSV, and encoder checkpoints. Override with args:

```bash
./run_baseline.sh --epochs 100 --batch_size 32
```

### Manual run

```bash
cd Waveform_CXR_EHR/baseline
conda activate MedTVT-R1
python train.py --output_dir ./output
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--target_col` | `spo2` | EHR column for ground truth oxygenation |
| `--freeze_encoders` | True | Freeze CXR/signal encoders |
| `--no_freeze` | - | Finetune all encoders |
| `--batch_size` | 16 | Batch size |
| `--epochs` | 50 | Training epochs |
| `--lr` | 1e-3 | Learning rate |

### Metrics

- **MSE**: Training/validation loss
- **MAE**: Mean absolute error (primary)
- **RMSE**: Root mean squared error

Best checkpoint is saved by lowest validation MAE.
