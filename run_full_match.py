#!/usr/bin/env python
"""
Standalone script for full CXR-supertable matching.
Supports checkpoint/resume: re-run after interrupt to continue from last save.
Run with: conda activate MedTVT-R1 && nohup python run_full_match.py >> run_full_match.log 2>&1 &
"""
import json
import pickle
import signal
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

METADATA_PATH = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz"
ADMISSIONS_PATH = "/hpc/group/kamaleswaranlab/mimic_iv/mimic-iv-3.1-decompress/hosp/admissions.csv"
SUPERTABLES_DIR = Path("/hpc/group/kamaleswaranlab/mimic_iv/sepy_output/mimic-supertables/Supertables")
OUTPUT_CSV = "/work/gs285/cxr_supertable_matched.csv"
OUTPUT_PKL = "/work/gs285/cxr_supertable_matched.pkl"
CHECKPOINT_FILE = "/work/gs285/run_full_match_checkpoint.json"
TEMP_CSV = "/work/gs285/run_full_match_partial.csv"
CHECKPOINT_INTERVAL = 200  # Save more often to reduce lost progress

# For signal handler to save checkpoint on kill/interrupt
_exit_state = {"processed": None, "accumulated": None}


def _dedupe_cols(df):
    """Remove duplicate column names (supertables can have them)."""
    if df is None or df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()]


def load_supertable_pkl(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except (RuntimeError, ImportError) as e:
        if "empty_like" in str(e) or "numpy" in str(e).lower():
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                csv_path = tmp.name
            cmd = 'import pickle,sys,pandas as pd\npkl,csv=sys.argv[1],sys.argv[2]\nwith open(pkl,"rb") as f: d=pickle.load(f)\npd.DataFrame(d).to_csv(csv,index=True)\n'
            r = subprocess.run(["conda", "run", "-n", "base", "python", "-c", cmd, str(pkl_path), csv_path],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(f"subprocess failed: {r.stderr}")
            return pd.read_csv(csv_path, index_col=0)
        raise


def process_one_hadm(hadm_id, cxr_by_hadm, supertables_dir, load_fn):
    """Process single hadm_id, return list of matched rows."""
    rows_list = []
    pkl_path = supertables_dir / f"{hadm_id}.pkl"
    if not pkl_path.exists():
        return rows_list
    try:
        st = load_fn(pkl_path)
    except Exception:
        return rows_list

    if not isinstance(st, pd.DataFrame) or st.empty:
        return rows_list

    if not isinstance(st.index, pd.DatetimeIndex):
        st.index = pd.to_datetime(st.index)

    st = st.copy()
    st["_date_int"] = st.index.year * 10000 + st.index.month * 100 + st.index.day
    st["_hour"] = st.index.hour

    for _, row in cxr_by_hadm.get_group(hadm_id).iterrows():
        mask = (st["_date_int"] == row["_date_int"]) & (st["_hour"] == row["_hour"])
        matched = st.loc[mask]
        if len(matched) > 0:
            m = matched.iloc[0:1].copy().drop(columns=["_date_int", "_hour"])
            m["dicom_id"] = row["dicom_id"]
            m["subject_id"] = row["subject_id"]
            m["hadm_id"] = hadm_id
            m["supertable_datetime"] = matched.index[0]
            rows_list.append(m)
    return rows_list


def load_checkpoint():
    """Return (processed_hadm_ids set, partial_df or None)."""
    if not Path(CHECKPOINT_FILE).exists():
        return set(), None
    try:
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        processed = set(cp["processed_hadm_ids"])
        partial_df = None
        if Path(TEMP_CSV).exists():
            partial_df = _dedupe_cols(pd.read_csv(TEMP_CSV))
        return processed, partial_df
    except Exception as e:
        print(f"Checkpoint load failed: {e}, starting fresh", flush=True)
        return set(), None


def save_checkpoint(processed_hadm_ids, partial_df):
    """Save progress for resume."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_hadm_ids": sorted(processed_hadm_ids)}, f)
    if partial_df is not None and len(partial_df) > 0:
        partial_df.to_csv(TEMP_CSV, index=False)


def _emergency_save(signum=None, frame=None):
    """Save checkpoint on SIGTERM/SIGINT (kill, Ctrl+C), then exit."""
    state = _exit_state
    if state["processed"] is not None:
        try:
            partial = _dedupe_cols(pd.concat(state["accumulated"], ignore_index=True)) if state["accumulated"] else pd.DataFrame()
            save_checkpoint(state["processed"], partial)
            sig = "SIGTERM" if signum == signal.SIGTERM else "SIGINT" if signum == signal.SIGINT else "unknown"
            print(f"\n[{sig}] Checkpoint saved before exit. Re-run to resume.", flush=True)
        except Exception as e:
            print(f"\nCheckpoint save failed: {e}", file=sys.stderr, flush=True)
    sys.exit(130 if signum == signal.SIGINT else 143)


def main():
    signal.signal(signal.SIGTERM, _emergency_save)
    signal.signal(signal.SIGINT, _emergency_save)
    print("Loading metadata and admissions...", flush=True)
    meta = pd.read_csv(METADATA_PATH)
    admissions = pd.read_csv(ADMISSIONS_PATH, parse_dates=["admittime", "dischtime"])
    print(f"Metadata: {len(meta):,} CXR records", flush=True)
    print(f"Admissions: {len(admissions):,} records", flush=True)

    def studydatetime(row):
        sd, st = int(row["StudyDate"]), row["StudyTime"]
        y, m, d = sd // 10000, (sd % 10000) // 100, sd % 100
        h = int(st // 10000)
        return pd.Timestamp(year=y, month=m, day=d, hour=h, minute=0, second=0)

    meta = meta.copy()
    meta["_cxr_dt"] = meta.apply(studydatetime, axis=1)
    meta["_date_int"] = meta["StudyDate"].astype(int)
    meta["_hour"] = (meta["StudyTime"] // 10000).astype(int)

    meta_adm = meta.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime"]], on="subject_id")
    meta_adm = meta_adm[(meta_adm["admittime"] <= meta_adm["_cxr_dt"]) & (meta_adm["dischtime"] >= meta_adm["_cxr_dt"])]
    cxr_df = meta_adm.drop_duplicates(subset=["dicom_id"], keep="first")[
        ["subject_id", "StudyDate", "StudyTime", "dicom_id", "hadm_id", "_date_int", "_hour"]
    ]

    existing_hadm = {int(f.stem) for f in SUPERTABLES_DIR.glob("*.pkl")}
    cxr_df = cxr_df[cxr_df["hadm_id"].isin(existing_hadm)]
    cxr_by_hadm = cxr_df.groupby("hadm_id")
    hadm_ids = list(cxr_by_hadm.groups.keys())
    print(f"CXRs to match: {len(cxr_df):,}", flush=True)
    print(f"hadm_ids to process: {len(hadm_ids):,}", flush=True)

    processed, partial_df = load_checkpoint()
    if processed:
        hadm_ids = [h for h in hadm_ids if h not in processed]
        print(f"Resuming: skipping {len(processed):,} already done, {len(hadm_ids):,} remaining", flush=True)

    accumulated = [] if partial_df is None else [partial_df]
    iterator = tqdm(hadm_ids, desc="Matching")
    _exit_state["processed"] = processed
    _exit_state["accumulated"] = accumulated

    try:
        for i, hadm_id in enumerate(iterator):
            new_rows = process_one_hadm(hadm_id, cxr_by_hadm, SUPERTABLES_DIR, load_supertable_pkl)
            if new_rows:
                accumulated.append(_dedupe_cols(pd.concat(new_rows, ignore_index=True)))
            processed.add(hadm_id)
            _exit_state["accumulated"] = accumulated  # keep in sync for signal handler

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                partial_df = _dedupe_cols(pd.concat(accumulated, ignore_index=True)) if accumulated else pd.DataFrame()
                save_checkpoint(processed, partial_df)
                if hasattr(iterator, 'set_postfix_str'):
                    iterator.set_postfix_str(f"checkpoint {len(partial_df):,} rows")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr, flush=True)
        partial_df = _dedupe_cols(pd.concat(accumulated, ignore_index=True)) if accumulated else pd.DataFrame()
        save_checkpoint(processed, partial_df)
        print("Checkpoint saved. Re-run to resume.", flush=True)
        raise

    matched_full = _dedupe_cols(pd.concat(accumulated, ignore_index=True)) if accumulated else pd.DataFrame()
    print(f"Total matches: {len(matched_full):,} rows", flush=True)

    matched_full.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV: {OUTPUT_CSV}", flush=True)

    r = subprocess.run(
        ["conda", "run", "-n", "base", "python", "-c",
         f"import pandas as pd; pd.read_csv('{OUTPUT_CSV}').to_pickle('{OUTPUT_PKL}'); print('Saved pkl:', '{OUTPUT_PKL}')"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"Warning: pkl save failed: {r.stderr}", file=sys.stderr)
    else:
        print(f"Saved pkl: {OUTPUT_PKL}", flush=True)
    print(f"DataFrame columns: {len(matched_full.columns)}", flush=True)

    # Remove checkpoint on success
    for p in [CHECKPOINT_FILE, TEMP_CSV]:
        if Path(p).exists():
            Path(p).unlink()
            print(f"Removed checkpoint: {p}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
