#!/usr/bin/env python
"""
Analyze p2f_vent_fio2 missing values, range, and ARDS classification in Supertables.
Uses multiprocessing; ~545k pkl files take ~10–20 min (depends on CPU cores).
Note: MedTVT-R1 env numpy incompatible with some pkl; falls back to base env on failure.
Run: python analyze_p2f_supertables.py
"""
import pickle
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

SUPERTABLES_DIR = Path("/hpc/group/kamaleswaranlab/mimic_iv/sepy_output/mimic-supertables/Supertables")
COL = "p2f_vent_fio2"


def _load_pkl(pkl_path):
    """Same as visualization: on pickle failure, use base env subprocess to convert to CSV and read"""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except (RuntimeError, ImportError) as e:
        if "empty_like" in str(e) or "numpy" in str(e).lower():
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                csv_path = tmp.name
            cmd = 'import pickle,sys,pandas as pd\npkl,csv=sys.argv[1],sys.argv[2]\nwith open(pkl,"rb") as f: d=pickle.load(f)\npd.DataFrame(d).to_csv(csv,index=True)\n'
            r = subprocess.run(["conda", "run", "-n", "base", "python", "-c", cmd, str(pkl_path), csv_path],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return None
            return pd.read_csv(csv_path, index_col=0)
        return None
    except Exception:
        return None


def _process_chunk(args):
    """Process a batch of pkl paths, return aggregated stats. paths is a list of string paths."""
    paths_chunk, col = args
    valid_count = 0
    null_count = 0
    mild = mod = sev = above = 0
    vmin = float("inf")
    vmax = float("-inf")
    sum_v = 0.0

    for p in paths_chunk:
        try:
            st = _load_pkl(str(p))
            if st is None or not isinstance(st, pd.DataFrame) or st.empty or col not in st.columns:
                continue
            s = st[col]
            valid = s.dropna()
            valid_count += len(valid)
            null_count += len(s) - len(valid)
            for v in valid:
                try:
                    fv = float(v)
                    sum_v += fv
                    if fv > 300:
                        above += 1
                    elif 200 < fv <= 300:
                        mild += 1
                    elif 100 < fv <= 200:
                        mod += 1
                    elif fv <= 100:
                        sev += 1
                    if fv < vmin:
                        vmin = fv
                    if fv > vmax:
                        vmax = fv
                except (TypeError, ValueError):
                    pass
        except Exception:
            continue

    return {
        "valid": valid_count,
        "null": null_count,
        "mild": mild,
        "moderate": mod,
        "severe": sev,
        "above_mild": above,
        "vmin": vmin if vmin != float("inf") else None,
        "vmax": vmax if vmax != float("-inf") else None,
        "sum_v": sum_v,
    }


def main():
    pkl_files = sorted(SUPERTABLES_DIR.glob("*.pkl"), key=lambda p: p.name)
    n = len(pkl_files)
    print(f"Found {n:,} supertable pkl files")
    if n == 0:
        return

    n_proc = max(1, cpu_count() - 1)
    chunk_size = max(1, n // (n_proc * 10))
    # Use string paths to avoid multiprocessing Path serialization issues
    pkl_paths = [str(p) for p in pkl_files]
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append((pkl_paths[i : i + chunk_size], COL))

    print(f"Using {n_proc} processes, {len(chunks)} batches, ~{chunk_size} files per batch")
    results = []
    running_valid = 0
    with ProcessPoolExecutor(max_workers=n_proc) as ex:
        futures = {ex.submit(_process_chunk, c): c for c in chunks}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            running_valid += r["valid"]
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  Progress: {done}/{len(futures)} batches, valid count: {running_valid:,}", flush=True)

    # Aggregate
    total_valid = sum(r["valid"] for r in results)
    total_null = sum(r["null"] for r in results)
    mild = sum(r["mild"] for r in results)
    mod = sum(r["moderate"] for r in results)
    sev = sum(r["severe"] for r in results)
    above = sum(r["above_mild"] for r in results)
    sum_v = sum(r["sum_v"] for r in results)
    vmins = [r["vmin"] for r in results if r["vmin"] is not None]
    vmaxs = [r["vmax"] for r in results if r["vmax"] is not None]
    gmin = min(vmins) if vmins else None
    gmax = max(vmaxs) if vmaxs else None

    total = total_valid + total_null
    print("\n" + "=" * 60)
    print("[p2f_vent_fio2 Statistics]")
    print("=" * 60)
    print(f"Total rows: {total:,}")
    print(f"Valid: {total_valid:,}")
    print(f"Missing: {total_null:,}")
    print(f"Missing %: {100 * total_null / total:.2f}%" if total > 0 else "N/A")

    if total_valid > 0 and gmin is not None:
        mean_v = sum_v / total_valid
        print(f"\nValue range: min={gmin:.2f}, max={gmax:.2f}, mean={mean_v:.2f} (median omitted)")
        print(f"\nBy ARDS severity (PaO2/FiO2):")
        print("  Mild (200 < P/F ≤ 300):     ", mild, "  rows")
        print("  Moderate (100 < P/F ≤ 200): ", mod, "  rows")
        print("  Severe (P/F ≤ 100):         ", sev, "  rows")
        print("  Above Mild (P/F > 300):     ", above, "  rows")
    else:
        print("\nNo valid p2f_vent_fio2 values")


if __name__ == "__main__":
    main()
