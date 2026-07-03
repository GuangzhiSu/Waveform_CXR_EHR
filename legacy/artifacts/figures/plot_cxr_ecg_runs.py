#!/usr/bin/env python3
"""Plot CXR/ECG EncoderTransformer training curves and test accuracy."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT_DIR = Path(__file__).resolve().parent / "cxr_ecg_enc_tr"

_RE_CXR_EPOCH = re.compile(
    r"Epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\)\s+"
    r"val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_TEST = re.compile(
    r"Test \(summary\): loss=([\d.]+)\s+acc_s2f=([\d.]+)\s+acc_p2f=([\d.]+)"
)

CXR_LOG = LOGS / "cxr-enc-tr-47646619.out"
CXR_RESULTS_FULL = ROOT / "CXREncoderTransformer/output/results.json"
ECG_RESULTS = ROOT / "ECGEncoderTransformer/output/results.json"


def parse_cxr_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    for m in _RE_CXR_EPOCH.finditer(text):
        epochs.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(5)),
                "val_acc_s2f": float(m.group(6)),
                "val_acc_p2f": float(m.group(7)),
            }
        )
    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {"loss": float(test_m.group(1)), "acc_s2f": float(test_m.group(2)), "acc_p2f": float(test_m.group(3))}
    return {"name": "CXR enc tr (5k)", "job_id": "47646619", "epochs": epochs, "test": test}


def _test_from_results(path: Path) -> dict | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    ts = d.get("test_summary") or d.get("test") or {}
    return {
        "loss": ts.get("loss"),
        "acc_s2f": ts.get("acc_s2f"),
        "acc_p2f": ts.get("acc_p2f"),
        "use_class_weights": d.get("use_class_weights"),
    }


def plot_cxr_curves(data: dict, out_path: Path) -> None:
    ep = data["epochs"]
    if not ep:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"CXREncoderTransformer — job {data['job_id']} (max_samples=5000)", fontsize=11)
    xs = [e["epoch"] for e in ep]
    axes[0].plot(xs, [e["train_loss"] for e in ep], label="train")
    axes[0].plot(xs, [e["val_loss"] for e in ep], "--", label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, [e["val_acc_s2f"] for e in ep], color="#1f77b4", label="s2f")
    axes[1].plot(xs, [e["val_acc_p2f"] for e in ep], color="#ff7f0e", label="p2f")
    axes[1].axhline(0.6591, linestyle="--", alpha=0.4, color="#1f77b4")
    axes[1].axhline(0.4968, linestyle="--", alpha=0.4, color="#ff7f0e")
    axes[1].set_title("Val accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    t = data.get("test") or {}
    axes[2].bar(["s2f", "p2f"], [t.get("acc_s2f", 0), t.get("acc_p2f", 0)], color=["#1f77b4", "#ff7f0e"])
    axes[2].set_title("Test accuracy")
    axes[2].set_ylim(0, 0.75)
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(entries: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [e["label"] for e in entries]
    s2f = [e.get("test", {}).get("acc_s2f", np.nan) for e in entries]
    p2f = [e.get("test", {}).get("acc_p2f", np.nan) for e in entries]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, s2f, w, label="Test acc s2f", color="#1f77b4")
    ax.bar(x + w / 2, p2f, w, label="Test acc p2f", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.85)
    ax.set_title("CXR / ECG EncoderTransformer — Test Accuracy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_fix_log(pattern: str, label_prefix: str) -> tuple[dict | None, dict | None]:
    matches = sorted(LOGS.glob(pattern))
    if not matches:
        return None, None
    path = matches[-1]
    jid = path.stem.split("-")[-1]
    data = parse_cxr_log(path)
    data["name"] = f"{label_prefix} ({jid})"
    data["job_id"] = jid
    return data, {"label": f"{label_prefix}\n{jid}", "test": data.get("test")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    if CXR_LOG.is_file():
        cxr = parse_cxr_log(CXR_LOG)
        plot_cxr_curves(cxr, OUT_DIR / "training_curves_cxr_47646619.png")
        summary.append({"label": "CXR 5k\n47646619", "test": cxr["test"]})
    else:
        cxr = None

    cxr_full = _test_from_results(CXR_RESULTS_FULL)
    if cxr_full:
        summary.append({"label": "CXR full\noutput", "test": cxr_full})

    ecg = _test_from_results(ECG_RESULTS)
    if ecg:
        summary.append({"label": "ECG full\noutput", "test": ecg})

    for pattern, prefix, out_key in (
        ("cxr-fixA-*.out", "CXR-Fix-A", "cxr_fixA"),
        ("cxr-fixB-*.out", "CXR-Fix-B", "cxr_fixB"),
    ):
        data, entry = _parse_fix_log(pattern, prefix)
        if data and data.get("epochs"):
            plot_cxr_curves(data, OUT_DIR / f"training_curves_{out_key}.png")
        if entry:
            summary.append(entry)

    for pattern, prefix, out_sub in (
        ("ecg-fixA-*.out", "ECG-Fix-A", "output_fixA"),
        ("ecg-fixB-*.out", "ECG-Fix-B", "output_fixB"),
    ):
        matches = sorted(LOGS.glob(pattern))
        if matches:
            t = _test_from_results(ROOT / f"ECGEncoderTransformer/{out_sub}/results.json")
            if t:
                jid = matches[-1].stem.split("-")[-1]
                summary.append({"label": f"{prefix}\n{jid}", "test": t})

    metrics = {"cxr_5k": cxr, "summary": summary}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    if summary:
        plot_summary_bar(summary, OUT_DIR / "test_accuracy_compare.png")

    print(f"Wrote figures to {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
