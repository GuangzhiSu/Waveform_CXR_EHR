# ECG-CXR Patient-Temporal Contrastive Learning

Test, **in latent space only** (no image generation, no pixel-level prediction),
whether an **ECG** (one beat or a short sequence) can be aligned with / predict a
patient's **future chest X-ray** embedding. CXR is acquired infrequently while
ECG is dense, and lung changes may show up on CXR only late — so we ask whether
the ECG already carries a learnable signal about the *future* CXR latent state.

All embeddings are frozen: CXR from **Bio-ViL-T**, ECG from **ECG-CoCa**. Only
small projection / Transformer / predictor heads are trained.

## Layout

```text
ECGCXRPatientTemporal/
├── *.py                       # current experiment code
├── jobs/                      # CPU / Slurm launchers
├── encoders/                  # frozen Bio-ViL-T / ECG-CoCa wrappers
├── external/ECG-R1/           # local ECG-CoCa source checkout, ignored by git
└── artifacts/                 # generated caches, checkpoints, outputs, logs, pylibs
```

Large generated files are kept under `artifacts/` so the source directory stays
readable for new work. The large artifact contents, encoder checkpoints, local
`pylibs/`, and `external/ECG-R1/` checkout are ignored by git; only layout
markers and source/docs are tracked.

## Staged experiments

The framework is deliberately staged from **ECG-only** to **multimodal fusion**,
so we can tell whether a fusion model truly uses the ECG rather than leaning on a
`CXR_t1` shortcut. The default `ALL_IN_ORDER` run covers Exp 1-4. Exp 5/6 are
registered follow-up groups that must be requested with `--only`.

All runs use the same contrastive loss family: `cross_patient_loss` plus optional
`λ·temporal_loss` (`λ=0.2` for combined runs). There are no delta / label /
reconstruction / generation losses; Exp 1A is the deliberate cross-only check.

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
| opt-in | `exp5a_proj_tx_crossattn_norm` | CXR_t1 query + ECG tokens → CXR_t2 | CXR query cross-attends ECG tokens, residual norm |
| opt-in | `exp5b_proj_add_norm` | CXR_t1 + pooled ECG → CXR_t2 | project, add, normalize |
| opt-in | `exp5c_weighted_attn_pool` | CXR_t1 + ECG tokens → CXR_t2 | single-linear weighted pooling over CXR/ECG tokens |
| opt-in | `exp6a_cxr_residual_ecg` | CXR_t1 base + ECG residual → CXR_t2 | CXR-only base with gated ECG residual |

Read off the shortcut controls: C ≫ B means ECG adds information; C ≈ B means the
model leans on the CXR_t1 shortcut; D ≈ B means it relies on the *real* ECG;
D ≈ C or E ≈ B means the ECG is being ignored.

Steps 1–2 use **single-ECG** pairs (`ECG at t → CXR at t+9–15h`, built by
`build_single_ecg_pairs.py`); steps 3–4 reuse the **sequence** pair files built
by `build_seq_pairs.py` through `jobs/run_build_pairs.sh`. Run order and a
unified results table are produced by `run_experiments.py`.

```bash
# 1. sequence pairs (Exp 3/4) -> patient_temporal_pairs.json + seq_target_pairs.json
bash ECGCXRPatientTemporal/jobs/run_build_pairs.sh
# 2. single-ECG pairs (Exp 1/2). --restrict_to_cache reuses already-embedded ids.
bash ECGCXRPatientTemporal/jobs/run_build_single_pairs.sh
# 3. (only if new ids) merge-precompute over all staged pairs files, reusing the cache:
sbatch ECGCXRPatientTemporal/jobs/run_precompute.sh \
       --pairs ECGCXRPatientTemporal/artifacts/cache/default/patient_temporal_pairs.json \
               ECGCXRPatientTemporal/artifacts/cache/default/seq_target_pairs.json \
               ECGCXRPatientTemporal/artifacts/cache/default/single_ecg_pairs.json \
       --merge
# 4. run default staged experiments (Exp 1-4) + unified table -> artifacts/outputs/staged/results_table.csv
sbatch ECGCXRPatientTemporal/jobs/run_staged.sh
#    or a subset:  --only step1 step2   /   --only exp4c_fusion_cxr1_ecgseq exp4b_cxr_only
#    optional follow-ups: --only fusion_schemes   /   --only improve
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
`SimpleITK`) are installed into the workspace-local `artifacts/pylibs/` prefix
because the `MedTVT-R1` conda env is read-only on this cluster; `setup_env.sh`
wires them up.

## Pipeline

```bash
# 0. Environment (activates MedTVT-R1 + artifacts/pylibs + ECG-R1)
source ECGCXRPatientTemporal/setup_env.sh

# 1. Download frozen encoder weights -> artifacts/checkpoints/
bash ECGCXRPatientTemporal/download_weights.sh
#    Bio-ViL-T: auto from HuggingFace.
#    ECG-CoCa : Google Drive file id 1wOKYfkb-Nep0WzYZz9-n66oTzp_4cky7
#               (cpt_wfep_epoch_20.pt). This Drive file is often rate-limited
#               ("Too many users ... recently"); if so, retry later (~24h reset).

# 2. Build sequence + single-ECG pairs from the CXR + ECG catalogs -> artifacts/cache/default/
bash ECGCXRPatientTemporal/jobs/run_build_pairs.sh
bash ECGCXRPatientTemporal/jobs/run_build_single_pairs.sh

# 3. Precompute frozen CXR/ECG embeddings (GPU) -> artifacts/cache/default/
sbatch ECGCXRPatientTemporal/jobs/run_precompute.sh \
  --pairs ECGCXRPatientTemporal/artifacts/cache/default/patient_temporal_pairs.json \
          ECGCXRPatientTemporal/artifacts/cache/default/seq_target_pairs.json \
          ECGCXRPatientTemporal/artifacts/cache/default/single_ecg_pairs.json \
  --merge
#   or locally: python ECGCXRPatientTemporal/precompute_embeddings.py

# 4. Train (GPU)
sbatch ECGCXRPatientTemporal/jobs/run_train.sh --loss_mode combined

# 5. Ablation: only cross, only temporal, cross + 0.2*temporal
sbatch ECGCXRPatientTemporal/jobs/run_ablation.sh
```

## Data extraction

By default, the scripts still point at the older p2f/s2f modality catalogs
(`data/p2f_or_s2f_cxr_catalog.csv`, `data/p2f_or_s2f_ecg_catalog.csv`) for
backward compatibility with the first runs. For the contrastive-only question,
build full non-EHR-restricted catalogs first:

```bash
python ECGCXRPatientTemporal/build_full_catalogs.py
bash ECGCXRPatientTemporal/jobs/run_build_pairs.sh \
  --cxr_csv data/ecg_cxr_full_cxr_catalog.csv \
  --ecg_csv data/ecg_cxr_full_ecg_catalog.csv \
  --target_out ECGCXRPatientTemporal/artifacts/cache/full/seq_target_pairs.json \
  --t1_out ECGCXRPatientTemporal/artifacts/cache/full/patient_temporal_pairs.json \
  --skip_cxr_path_check
bash ECGCXRPatientTemporal/jobs/run_build_single_pairs.sh \
  --cxr_csv data/ecg_cxr_full_cxr_catalog.csv \
  --ecg_csv data/ecg_cxr_full_ecg_catalog.csv \
  --out ECGCXRPatientTemporal/artifacts/cache/full/single_ecg_pairs.json \
  --skip_cxr_path_check
sbatch ECGCXRPatientTemporal/jobs/run_precompute.sh \
  --pairs ECGCXRPatientTemporal/artifacts/cache/full/patient_temporal_pairs.json \
          ECGCXRPatientTemporal/artifacts/cache/full/seq_target_pairs.json \
          ECGCXRPatientTemporal/artifacts/cache/full/single_ecg_pairs.json \
  --merge
sbatch ECGCXRPatientTemporal/jobs/run_staged.sh \
  --pairs ECGCXRPatientTemporal/artifacts/cache/full/patient_temporal_pairs.json \
  --seq_target_pairs ECGCXRPatientTemporal/artifacts/cache/full/seq_target_pairs.json \
  --single_pairs ECGCXRPatientTemporal/artifacts/cache/full/single_ecg_pairs.json
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
| `runtime.py` | Shared small runtime helpers (`get_device`, `set_seed`) |
| `env_setup.py` / `setup_env.sh` | Wire `artifacts/pylibs` + ECG-R1 + skimage stub |
| `download_weights.sh` | Fetch Bio-ViL-T + ECG-CoCa checkpoints into `artifacts/checkpoints/` |
| `jobs/_common.sh` | Shared project-root, log-dir, and environment setup for job scripts |
| `jobs/*.sh` | CPU/Slurm launchers; each wrapper only runs its task-specific command |
| `artifacts/` | Generated caches, checkpoints, logs, local pylibs, and experiment outputs |
| `encoders/biovil_t.py` | Frozen Bio-ViL-T CXR encoder |
| `encoders/ecg_coca.py` | Frozen ECG-CoCa ECG encoder |
| `build_pairs.py` | Shared CXR/ECG catalog loaders plus original Exp 4 pair builder |
| `build_seq_pairs.py` | Build Exp 3 `seq_target_pairs.json` and Exp 4 `patient_temporal_pairs.json` together |
| `build_single_ecg_pairs.py` | Build single-ECG → future-CXR pairs (Exp 1/2, 9–15h) |
| `precompute_embeddings.py` | Cache frozen CXR/ECG embeddings (`--pairs ... --merge`) |
| `experiments.py` | Staged registry (default Exp 1-4, plus opt-in `fusion_schemes` / `improve`) |
| `staged_dataset.py` | Unified single/sequence dataset (+ zero/shuffle ECG controls) |
| `staged_model.py` | Configurable model (cxr_t1 / single-or-seq ECG / `g` / time / query / special fusion heads) |
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
