# Waveform_CXR_EHR

Waveform, CXR and EHR data matching and processing.

## Pipeline

### Step 1: CXR–Supertable Matching (`run_full_match.py`)

Matches CXR studies (from MIMIC-CXR metadata) with EHR supertables by patient and time.

- **Input**: MIMIC-CXR metadata, MIMIC-IV admissions, supertables (PKL per `hadm_id`)
- **Output**: `cxr_supertable_matched.csv` / `.pkl` — CXR + EHR supertable rows
- **Time matching**: `_date_int` (YYYYMMDD) + `_hour` — same patient, same date, same hour

### Step 2: Waveform Matching (`merge_cxr_waveform.py`)

Merges the CXR+supertable data with ECG waveform data.

- **Input**: `cxr_supertable_matched.csv`, MIMIC4 waveform matched paths CSV
- **Output**: `cxr_supertable_waveform_matched.csv` / `.pkl` — CXR + EHR + Waveform
- **Time matching**: Same as `run_full_match.py` (`_date_int`, `_hour`) for consistency
- **Match logic**: `(subject_id, _date_int, _hour)` — inner join, rows with waveform only
- **Multi-waveform**: One CXR can match multiple waveforms in the same hour; all pairs are kept

## File Description

| File | Description |
|------|-------------|
| `run_full_match.py` | CXR–supertable matching with checkpoint/resume |
| `merge_cxr_waveform.py` | CXR+supertable–waveform merge |
| `visualization.ipynb` | Data exploration and visualization |
| `cxr_supertable_matched.csv` / `.pkl` | Step 1 output — CXR + EHR (no waveform) |
| `cxr_supertable_waveform_matched.csv` / `.pkl` | Step 2 output — CXR + EHR + Waveform |

## Output: `cxr_supertable_waveform_matched.csv`

- **Rows**: One row per CXR–waveform pair (only rows with a waveform match)
- **Columns**: ~179 columns
  - CXR/EHR supertable: demographics, labs, vitals, vent, scores (SOFA, SIRS), etc.
  - Waveform (`wf_*`): `wf_Study_ID`, `wf_File_Name`, `wf_Base_Time`, `wf_End_Time`, `wf_DurationMin`, `wf_sigLen`, `wf_ECG_Time`, `wf_stayHours`, `wf_File_Path`

## Usage

1. Run CXR–supertable matching (generates `cxr_supertable_matched.csv`):
   ```bash
   conda activate MedTVT-R1 && python run_full_match.py
   ```

2. Run waveform merge (requires `cxr_supertable_matched.csv`):
   ```bash
   python merge_cxr_waveform.py
   ```

Data files are not included due to size. Generate them using the pipeline above.
