# Waveform_CXR_EHR

Waveform, CXR and EHR data matching and processing.

## Pipeline

### Step 1: CXR–Supertable Matching (`experiment1/run_full_match.py`)

Matches CXR studies (from MIMIC-CXR metadata) with EHR supertables by patient and time.

- **Input**: MIMIC-CXR metadata, MIMIC-IV admissions, supertables (PKL per `hadm_id`)
- **Output**: `experiment1/cxr_supertable_matched.csv` / `.pkl` — CXR + EHR supertable rows
- **Time matching**: `_date_int` (YYYYMMDD) + `_hour` — same patient, same date, same hour

### Step 2: Waveform Matching (`experiment1/merge_cxr_waveform.py`)

Merges the CXR+supertable data with ECG waveform data.

- **Input**: `experiment1/cxr_supertable_matched.csv`, MIMIC4 waveform matched paths CSV
- **Output**: `experiment1/cxr_supertable_waveform_matched.csv` / `.pkl` — CXR + EHR + Waveform
- **Time matching**: Same as `run_full_match.py` (`_date_int`, `_hour`) for consistency
- **Match logic**: `(subject_id, _date_int, _hour)` — inner join, rows with waveform only
- **Multi-waveform**: One CXR can match multiple waveforms in the same hour; all pairs are kept

### Step 3 (p2f): Enrich with CXR/ECG Lookback (`enrich_p2f_cxr_ecg_lookback.py`)

Uses experiment1 CXR/ECG sources but **different match logic**: lookback window [ref-12h, ref-6h].

- **Input**: `p2f_vent_fio2_valid_rows.csv` (from `extract_p2f_rows.py`), experiment1 CXR/ECG CSVs
- **Output**: `p2f_vent_fio2_enriched.csv` — p2f rows with CXR_signal, ECG_signal, dicom_id, wf_File_Path, etc.
- **Match logic**: For each p2f row, find most recent CXR (supertable_datetime) and ECG (wf_Base_Time) in [row_time-12h, row_time-6h], same hadm_id

## File Description

| File | Description |
|------|-------------|
| `experiment1/run_full_match.py` | CXR–supertable matching with checkpoint/resume |
| `experiment1/merge_cxr_waveform.py` | CXR+supertable–waveform merge |
| `experiment1/visualization.ipynb` | Data exploration and visualization |
| `experiment1/cxr_supertable_matched.csv` / `.pkl` | Step 1 output — CXR + EHR (no waveform) |
| `experiment1/cxr_supertable_waveform_matched.csv` / `.pkl` | Step 2 output — CXR + EHR + Waveform |
| `extract_p2f_rows.py` | Extract rows with valid p2f_vent_fio2 from supertables |
| `enrich_p2f_cxr_ecg_lookback.py` | Enrich p2f rows with CXR/ECG in 6–12h lookback window |

## Output: `cxr_supertable_waveform_matched.csv`

- **Rows**: One row per CXR–waveform pair (only rows with a waveform match)
- **Columns**: ~179 columns
  - CXR/EHR supertable: demographics, labs, vitals, vent, scores (SOFA, SIRS), etc.
  - Waveform (`wf_*`): `wf_Study_ID`, `wf_File_Name`, `wf_Base_Time`, `wf_End_Time`, `wf_DurationMin`, `wf_sigLen`, `wf_ECG_Time`, `wf_stayHours`, `wf_File_Path`

## Usage

1. Run CXR–supertable matching (generates `experiment1/cxr_supertable_matched.csv`):
   ```bash
   cd experiment1 && conda activate MedTVT-R1 && python run_full_match.py
   ```

2. Run waveform merge (requires `experiment1/cxr_supertable_matched.csv`):
   ```bash
   cd experiment1 && python merge_cxr_waveform.py
   ```

3. (Optional) For p2f enrichment with CXR/ECG lookback:
   ```bash
   python extract_p2f_rows.py   # generates p2f_vent_fio2_valid_rows.csv
   python enrich_p2f_cxr_ecg_lookback.py  # generates p2f_vent_fio2_enriched.csv
   ```

Data files are not included due to size. Generate them using the pipeline above.
