# ECG-CXR Patient-Temporal Contrastive Learning

Test, **in latent space only** (no image generation, no pixel-level prediction),
whether an **ECG** (one beat or a short sequence) can be aligned with / predict a
patient's **future chest X-ray** embedding. CXR is acquired infrequently while
ECG is dense, and lung changes may show up on CXR only late — so we ask whether
the ECG already carries a learnable signal about the *future* CXR latent state.

All embeddings are frozen: CXR from **Bio-ViL-T**, ECG from **ECG-CoCa**. Only
small projection / Transformer / predictor heads are trained.

## Staged experiments (Exp 1 → Exp 4)

The framework is deliberately staged from **ECG-only** to **multimodal fusion**,
so we can tell whether a fusion model truly uses the ECG rather than leaning on a
`CXR_t1` shortcut. Every experiment uses the **same two contrastive losses**
(`cross_patient_loss + λ·temporal_loss`, λ=0.2) — no delta / label /
reconstruction / generation losses.

| Step | Name | Input → target | Key pieces |
|------|------|----------------|------------|
| 1 | `exp1a_single_ecg_cross` | single ECG → CXR @ +9–15h | proj only, **cross loss only** |
| 1 | `exp1b_single_ecg_combined` | single ECG → CXR @ +9–15h | proj only, cross + 0.2·temporal |
| 2 | `exp2_single_ecg_predictor` | single ECG + Δt → future CXR | predictor `g`, time embedding |
| 3 | `exp3a_seq_ecg_meanpool` | ECG sequence → CXR_t2 | Transformer + mean pool + `g` |
| 3 | `exp3b_seq_ecg_future_query` | ECG sequence + future query → CXR_t2 | learnable future-time query |
| 4 | `exp4c_fusion_cxr1_ecgseq` | **CXR_t1 + ECG seq → CXR_t2** | fusion `g` (the target model) |
| 4 | `exp4a_ecg_only` | ECG seq → CXR_t2 | shortcut control **A** |
| 4 | `exp4b_cxr_only` | CXR_t1 → CXR_t2 | shortcut control **B** |
| 4 | `exp4d_fusion_shuffled_ecg` | CXR_t1 + *other-patient* ECG → CXR_t2 | shortcut control **D** |
| 4 | `exp4e_fusion_zeroed_ecg` | CXR_t1 + *zeroed* ECG → CXR_t2 | shortcut control **E** |

Read off the shortcut controls: C ≫ B means ECG adds information; C ≈ B means the
model leans on the CXR_t1 shortcut; D ≈ B means it relies on the *real* ECG;
D ≈ C or E ≈ B means the ECG is being ignored.

Steps 1–2 use **single-ECG** pairs (`ECG at t → CXR at t+9–15h`, built by
`build_single_ecg_pairs.py`); steps 3–4 reuse the **sequence** interval pairs
(`build_pairs.py`). Run order and a unified results table are produced by
`run_experiments.py`.

```bash
# 1. single-ECG pairs (Exp 1/2). --restrict_to_cache reuses already-embedded ids.
python ECGCXRPatientTemporal/build_single_ecg_pairs.py            # full set (needs precompute)
# 2. (only if new ids) merge-precompute over BOTH pairs files, reusing the cache:
sbatch ECGCXRPatientTemporal/run_precompute.sh \
       --pairs ECGCXRPatientTemporal/cache/patient_temporal_pairs.json \
               ECGCXRPatientTemporal/cache/single_ecg_pairs.json --merge
# 3. run all staged experiments + unified table -> output_staged/results_table.csv
sbatch ECGCXRPatientTemporal/run_staged.sh
#    or a subset:  --only step1 step2   /   --only exp4c_fusion_cxr1_ecgseq exp4b_cxr_only
```

---

## Original fusion view (Experiment 4)

Each fusion sample is an *interval*:

```
(patient_id, t1, t2, CXR_t1, ECG_{t1:t2}, CXR_t2)
```

The model builds a fused query embedding `q_{t1->t2} = f(CXR_t1, ECG_interval)`
and is trained to pull it toward the true `CXR_t2` embedding `c_t2`.

```
CXR_t1 ─┐                                  CXR_t2
        ├─► Bio-ViL-T (frozen) ─► MLP proj ─► c_t1 (L2)        │
        │                                                      ▼
ECG_{t1:t2} ─► ECG-CoCa (frozen) ─► +rel-time ─► 3-layer Transformer ─► h_ecg
        │                                                      │
        └────────────── concat(c_t1, h_ecg) ─► fusion MLP ─► q (L2)
                                                               │
                                  S = q @ c_t2ᵀ / temperature  (B×B)
```

Bio-ViL-T and ECG-CoCa are **frozen**; only the CXR projection, ECG temporal
Transformer, and fusion MLP are trained.

## Encoders

| Modality | Encoder | Source | Frozen feature |
|----------|---------|--------|----------------|
| CXR | Bio-ViL-T image model | [`microsoft/BiomedVLP-BioViL-T`](https://huggingface.co/microsoft/BiomedVLP-BioViL-T) | `img_embedding` (512-d) |
| ECG | ECG-CoCa ECG tower | [`PKUDigitalHealth/ECG-R1`](https://github.com/PKUDigitalHealth/ECG-R1) (ckpt from ECG-Chat) | pooled latent (512-d) |

The vendored ECG-CoCa code lives in `external/ECG-R1/`. Only the ECG tower is
built (the CoCa text tower needs `ncbi/MedCPT-Query-Encoder` and is skipped).
Python deps for the encoders (`health_multimodal`, `gdown`, `pydicom`,
`SimpleITK`) are installed into the workspace-local `pylibs/` prefix because the
`MedTVT-R1` conda env is read-only on this cluster; `setup_env.sh` wires them up.

## Pipeline

```bash
# 0. Environment (activates MedTVT-R1 + workspace pylibs + ECG-R1)
source ECGCXRPatientTemporal/setup_env.sh

# 1. Download frozen encoder weights -> checkpoints/
bash ECGCXRPatientTemporal/download_weights.sh
#    Bio-ViL-T: auto from HuggingFace.
#    ECG-CoCa : Google Drive file id 1wOKYfkb-Nep0WzYZz9-n66oTzp_4cky7
#               (cpt_wfep_epoch_20.pt). This Drive file is often rate-limited
#               ("Too many users ... recently"); if so, retry later (~24h reset).

# 2. Build patient-temporal pairs from the CXR + ECG catalogs -> cache/
bash ECGCXRPatientTemporal/run_build_pairs.sh

# 3. Precompute frozen CXR/ECG embeddings (GPU) -> cache/
sbatch ECGCXRPatientTemporal/run_precompute.sh
#   or locally: python ECGCXRPatientTemporal/precompute_embeddings.py

# 4. Train (GPU)
sbatch ECGCXRPatientTemporal/run_train.sh --loss_mode combined

# 5. Ablation: only cross, only temporal, cross + 0.2*temporal
sbatch ECGCXRPatientTemporal/run_ablation.sh
```

## Data extraction

By default, the scripts still point at the older p2f/s2f modality catalogs
(`data/p2f_or_s2f_cxr_catalog.csv`, `data/p2f_or_s2f_ecg_catalog.csv`) for
backward compatibility with the first runs. For the contrastive-only question,
build full non-EHR-restricted catalogs first:

```bash
python ECGCXRPatientTemporal/build_full_catalogs.py
bash ECGCXRPatientTemporal/run_build_pairs.sh \
  --cxr_csv data/ecg_cxr_full_cxr_catalog.csv \
  --ecg_csv data/ecg_cxr_full_ecg_catalog.csv \
  --target_out ECGCXRPatientTemporal/cache_full/seq_target_pairs.json \
  --t1_out ECGCXRPatientTemporal/cache_full/patient_temporal_pairs.json \
  --skip_cxr_path_check
bash ECGCXRPatientTemporal/run_build_single_pairs.sh \
  --cxr_csv data/ecg_cxr_full_cxr_catalog.csv \
  --ecg_csv data/ecg_cxr_full_ecg_catalog.csv \
  --out ECGCXRPatientTemporal/cache_full/single_ecg_pairs.json \
  --skip_cxr_path_check
sbatch ECGCXRPatientTemporal/run_precompute.sh \
  --pairs ECGCXRPatientTemporal/cache_full/patient_temporal_pairs.json \
          ECGCXRPatientTemporal/cache_full/seq_target_pairs.json \
          ECGCXRPatientTemporal/cache_full/single_ecg_pairs.json \
  --merge
sbatch ECGCXRPatientTemporal/run_staged.sh \
  --pairs ECGCXRPatientTemporal/cache_full/patient_temporal_pairs.json \
  --seq_target_pairs ECGCXRPatientTemporal/cache_full/seq_target_pairs.json \
  --single_pairs ECGCXRPatientTemporal/cache_full/single_ecg_pairs.json
```

For each
patient (`subject_id`), CXRs are sorted by time (one node per distinct
timestamp) and paired `(t1, t2)` up to `MAX_SKIP` steps apart, within
`[MIN_INTERVAL_HOURS, MAX_INTERVAL_HOURS]`, keeping all same-patient ECGs whose
base time is in `(t1, t2]` (`MIN/MAX_ECGS_PER_INTERVAL`). CXR jpg paths follow
the MIMIC-CXR-JPG layout (`get_cxr_path`), ECG waveforms are read with `wfdb`.

## Batch sampling & losses

Batches are sampled as **N patients × K intervals** (default N=16, K=2) so each
batch has both cross-patient and same-patient (temporal) negatives. With
`S = q @ c_t2ᵀ / τ` (B×B), the positive for row `i` is always the diagonal:

- **cross_patient_loss**: valid columns `{j : j==i OR patient_j != patient_i}`
  (same patient's other intervals are *ignored*, not negatives).
- **temporal_loss**: valid columns `{j : patient_j == patient_i}` (different
  patients ignored; rows with no same-patient other interval are skipped).

Masked columns get `-inf`; per-row cross-entropy uses the diagonal as target.
`loss = w_cross · cross + w_temporal · temporal`. `τ = 0.07` (optionally
`--learnable_temperature`). Default `w = (1, 0.2)`; ablation modes:
`cross → (1, 0)`, `temporal → (0, 1)`, `combined → (1, 0.2)`.

## Evaluation (retrieval)

- **Cross-patient retrieval**: rank the true `CXR_t2` among *all* unique
  `CXR_t2` in the split → Recall@1/5/10, MRR.
- **Within-patient temporal retrieval**: rank the true `CXR_t2` among the *same
  patient's* candidates → Temporal Recall@1, Temporal MRR (queries with ≥2
  candidates).

Splits are **by patient** (no patient leakage across train/val/test).

## Files

| File | Role |
|------|------|
| `config.py` | Paths + hyper-parameters |
| `env_setup.py` / `setup_env.sh` | Wire pylibs + ECG-R1 + skimage stub |
| `download_weights.sh` | Fetch Bio-ViL-T + ECG-CoCa checkpoints |
| `encoders/biovil_t.py` | Frozen Bio-ViL-T CXR encoder |
| `encoders/ecg_coca.py` | Frozen ECG-CoCa ECG encoder |
| `build_pairs.py` | Build sequence interval samples (Exp 3/4) from catalogs |
| `build_single_ecg_pairs.py` | Build single-ECG → future-CXR pairs (Exp 1/2, 9–15h) |
| `precompute_embeddings.py` | Cache frozen CXR/ECG embeddings (`--pairs ... --merge`) |
| `experiments.py` | Staged experiment registry (Exp 1A/1B/2/3A/3B/4 + controls A–E) |
| `staged_dataset.py` | Unified single/sequence dataset (+ zero/shuffle ECG controls) |
| `staged_model.py` | Configurable model (cxr_t1 / single-or-seq ECG / `g` / time / query) |
| `engine.py` | Generic train + eval loop reused by every staged experiment |
| `run_experiments.py` | Run staged experiments → unified `results_table.csv`/`.json` |
| `sampler.py` | N-patients × K-intervals sampler |
| `losses.py` | `cross_patient_loss`, `temporal_loss` (masked InfoNCE) |
| `metrics.py` | Cross-patient + within-patient temporal retrieval metrics |
| `dataset.py` / `model.py` / `train.py` | Original single-config fusion path (Exp 4 only) |

## Goal

Verify whether the **cross-patient loss** teaches `q` to discriminate the right
future CXR across patients, and whether adding the **temporal loss** improves
fine-grained within-patient time-point discrimination.
