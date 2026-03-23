#!/usr/bin/env python
"""
Extract all rows where p2f_vent_fio2 has a value from Supertables.
Runs sequentially; appends each found row to CSV and shows progress.
Run: conda run -n base python extract_p2f_rows.py
"""
import json
import pickle
import pandas as pd
from pathlib import Path

SUPERTABLES_DIR = Path("/hpc/group/kamaleswaranlab/mimic_iv/sepy_output/mimic-supertables/Supertables")
OUTPUT_CSV = "/work/gs285/Waveform_CXR_EHR/p2f_vent_fio2_valid_rows.csv"
CHECKPOINT_FILE = "/work/gs285/Waveform_CXR_EHR/extract_p2f_checkpoint.json"
COL = "p2f_vent_fio2"


def _dedupe_cols(df):
    """Remove duplicate column names."""
    if df is None or df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()]


def load_checkpoint():
    """Return (last_processed_index, output_columns or None)."""
    if not Path(CHECKPOINT_FILE).exists():
        return -1, None
    try:
        with open(CHECKPOINT_FILE) as f:
            cp = json.load(f)
        idx = cp.get("last_processed_index", -1)
        columns = cp.get("output_columns")
        return idx, columns
    except Exception as e:
        print(f"Checkpoint load failed: {e}, starting fresh", flush=True)
        return -1, None


def save_checkpoint(last_processed_index, output_columns):
    """Save progress for resume."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_processed_index": last_processed_index,
            "output_columns": output_columns,
        }, f, indent=0)


def main():
    pkl_files = sorted(SUPERTABLES_DIR.glob("*.pkl"), key=lambda p: p.name)
    total_files = len(pkl_files)

    start_idx, output_columns = load_checkpoint()
    output_exists = Path(OUTPUT_CSV).exists()
    write_header = not output_exists
    mode = "w" if write_header else "a"

    if output_exists and start_idx < 0:
        existing = sum(1 for _ in open(OUTPUT_CSV)) - 1
        print(f"CSV 已存在 ({existing:,} 条)，但无 checkpoint。", flush=True)
        print("无法断点续跑。若要重跑请先删除或重命名 CSV。", flush=True)
        print("若认为任务已完成，可忽略。", flush=True)
        return

    total_rows = sum(1 for _ in open(OUTPUT_CSV)) - 1 if output_exists else 0
    for i, p in enumerate(pkl_files):
        if i <= start_idx:
            continue

        try:
            with open(str(p), "rb") as f:
                st = pickle.load(f)
        except Exception:
            continue

        if not isinstance(st, pd.DataFrame) or st.empty or COL not in st.columns:
            save_checkpoint(i, output_columns)
            print(f"  第 {i+1}/{total_files} 个 pkl，已找到 {total_rows} 条", flush=True)
            continue

        mask = st[COL].notna()
        valid_rows = st.loc[mask]
        if len(valid_rows) == 0:
            save_checkpoint(i, output_columns)
            print(f"  第 {i+1}/{total_files} 个 pkl，已找到 {total_rows} 条", flush=True)
            continue

        hadm_id = int(p.stem)
        valid_rows = valid_rows.copy()
        valid_rows["hadm_id"] = hadm_id
        valid_rows = valid_rows.reset_index()
        valid_rows = _dedupe_cols(valid_rows)

        if output_columns is None:
            output_columns = list(valid_rows.columns)

        aligned = valid_rows.reindex(columns=output_columns)
        aligned.to_csv(OUTPUT_CSV, mode=mode, header=write_header, index=False)
        mode = "a"
        write_header = False
        total_rows += len(aligned)

        save_checkpoint(i, output_columns)
        print(f"  第 {i+1}/{total_files} 个 pkl，已找到 {total_rows} 条 ✓", flush=True)

    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        print(f"Removed checkpoint: {CHECKPOINT_FILE}", flush=True)

    print(f"\n完成。共保存 {total_rows:,} 条到 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
