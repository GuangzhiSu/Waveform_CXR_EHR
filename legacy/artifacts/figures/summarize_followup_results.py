#!/usr/bin/env python3
"""Print follow-up experiment test metrics from results.json (for README refresh)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("EHR TR Fix-D", "47837615", ROOT / "EHREncoderTransformer/output_fixD/results.json"),
    ("EHR TR Fix-E", "47837616", ROOT / "EHREncoderTransformer/output_fixE/results.json"),
    ("EmbedPred Exp-D", "47837617", ROOT / "EHREncoderTransformerEmbedPred/output_twophase_expD/finetune/results.json"),
    ("EmbedPred Exp-E", "47837618", ROOT / "EHREncoderTransformerEmbedPred/output_twophase_expE/finetune/results.json"),
    ("Window-Fix", "47837619", ROOT / "EHRWindowTransformer/output_direct_window_fix/results.json"),
    ("CXR-Fix-A", "47837620", ROOT / "CXREncoderTransformer/output_fixA/results.json"),
    ("CXR-Fix-B", "47837621", ROOT / "CXREncoderTransformer/output_fixB/results.json"),
    ("ECG-Fix-A", "47837622", ROOT / "ECGEncoderTransformer/output_fixA/results.json"),
    ("ECG-Fix-B", "47837623", ROOT / "ECGEncoderTransformer/output_fixB/results.json"),
]


def _acc(path: Path) -> tuple[float | None, float | None, str]:
    if not path.is_file():
        alt = path.parent.parent / "results.json"
        if alt.is_file():
            path = alt
        else:
            return None, None, "pending"
    d = json.loads(path.read_text())
    ts = d.get("test_summary") or d.get("test") or {}
    if isinstance(ts, dict) and "acc_s2f" not in ts and "test" in d:
        ts = d["test"]
    s2f = ts.get("acc_s2f")
    p2f = ts.get("acc_p2f")
    return s2f, p2f, "done"


def main() -> None:
    print(f"{'Experiment':<20} {'Job':<10} {'acc_s2f':>8} {'acc_p2f':>8}  status")
    print("-" * 60)
    for name, jid, path in RUNS:
        s2f, p2f, status = _acc(path)
        s2f_s = f"{100 * s2f:.1f}%" if s2f is not None else "—"
        p2f_s = f"{100 * p2f:.1f}%" if p2f is not None else "—"
        print(f"{name:<20} {jid:<10} {s2f_s:>8} {p2f_s:>8}  {status}")


if __name__ == "__main__":
    main()
