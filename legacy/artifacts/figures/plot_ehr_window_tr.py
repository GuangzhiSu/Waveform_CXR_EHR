#!/usr/bin/env python3
"""Plot EHRWindowTransformer (direct window) training curves."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOG = LOGS / "ehr-window-transformer-47103601.out"
RESULTS = ROOT / "EHRWindowTransformer/output_direct_window/results.json"
OUT_DIR = Path(__file__).resolve().parent / "ehr_window_transformer"

_RE_EPOCH = re.compile(
    r"Epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+"
    r"val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_TEST = re.compile(r"Test: loss=([\d.]+)\s+acc_s2f=([\d.]+)\s+acc_p2f=([\d.]+)")

S2F_MAJORITY = 0.6825
P2F_MAJORITY = 0.5106


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    for m in _RE_EPOCH.finditer(text):
        epochs.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
                "val_acc_s2f": float(m.group(4)),
                "val_acc_p2f": float(m.group(5)),
            }
        )
    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {"loss": float(test_m.group(1)), "acc_s2f": float(test_m.group(2)), "acc_p2f": float(test_m.group(3))}
    return {"job_id": "47103601", "epochs": epochs, "test": test}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG.is_file():
        print(f"Missing log: {LOG}")
        return
    data = parse_log(LOG)
    if RESULTS.is_file():
        data["results_json"] = json.loads(RESULTS.read_text())

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(data, f, indent=2)

    ep = data["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("EHRWindowTransformer (direct) — job 47103601", fontsize=12)

    xs = [e["epoch"] for e in ep]
    axes[0, 0].plot(xs, [e["train_loss"] for e in ep], label="train")
    axes[0, 0].plot(xs, [e["val_loss"] for e in ep], "--", label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xs, [e["val_acc_s2f"] for e in ep], label="s2f", color="#1f77b4")
    axes[0, 1].plot(xs, [e["val_acc_p2f"] for e in ep], label="p2f", color="#ff7f0e")
    axes[0, 1].axhline(S2F_MAJORITY, linestyle="--", alpha=0.4, color="#1f77b4")
    axes[0, 1].axhline(P2F_MAJORITY, linestyle="--", alpha=0.4, color="#ff7f0e")
    axes[0, 1].set_title("Val accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    t = data.get("test") or {}
    axes[1, 0].bar(["s2f", "p2f"], [t.get("acc_s2f", 0), t.get("acc_p2f", 0)], color=["#1f77b4", "#ff7f0e"])
    axes[1, 0].axhline(S2F_MAJORITY, linestyle="--", alpha=0.4)
    axes[1, 0].axhline(P2F_MAJORITY, linestyle="--", alpha=0.4)
    axes[1, 0].set_ylim(0, 0.75)
    axes[1, 0].set_title("Test accuracy")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05, 0.5,
        f"Test loss={t.get('loss', float('nan')):.4f}\n"
        f"acc_s2f={t.get('acc_s2f', float('nan')):.4f}  (maj {S2F_MAJORITY:.4f})\n"
        f"acc_p2f={t.get('acc_p2f', float('nan')):.4f}  (maj {P2F_MAJORITY:.4f})",
        fontsize=11, family="monospace", va="center",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves_47103601.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'training_curves_47103601.png'}")

    fix_logs = sorted(LOGS.glob("ehr-window-fix-*.out"))
    if fix_logs:
        fix_log = fix_logs[-1]
        jid = fix_log.stem.split("-")[-1]
        fix_data = parse_log(fix_log)
        fix_results = ROOT / "EHRWindowTransformer/output_direct_window_fix/results.json"
        if fix_results.is_file():
            fix_data["results_json"] = json.loads(fix_results.read_text())
        ep2 = fix_data["epochs"]
        if ep2:
            fig, ax = plt.subplots(figsize=(8, 4))
            xs2 = [e["epoch"] for e in ep2]
            ax.plot(xs2, [e["val_acc_s2f"] for e in ep2], label="s2f")
            ax.plot(xs2, [e["val_acc_p2f"] for e in ep2], label="p2f")
            ax.set_title(f"Window-Fix ({jid}) — val accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"training_curves_window_fix_{jid}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {OUT_DIR / f'training_curves_window_fix_{jid}.png'}")


if __name__ == "__main__":
    main()
