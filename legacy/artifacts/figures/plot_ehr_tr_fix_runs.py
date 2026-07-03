#!/usr/bin/env python3
"""Visualize EHREncoderTransformer baseline vs Fix-A/C training curves."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT_DIR = Path(__file__).resolve().parent / "ehr_tr_fix_runs"

_RE_EPOCH = re.compile(
    r"Epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\)\s+"
    r"(?:train_ce_uw_s2f=[\d.]+\s+train_ce_uw_p2f=[\d.]+\s+)?"
    r"val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_UW = re.compile(
    r"Epoch (\d+)/\d+\s+train_loss=[\d.]+\s+\(s2f=[\d.]+\s+p2f=[\d.]+\)\s+"
    r"train_ce_uw_s2f=([\d.]+)\s+train_ce_uw_p2f=([\d.]+)"
)
_RE_DIAG = re.compile(
    r"train diagnostics: last_batch_grad_norm=([\d.]+)\s+param_l2=([\d.]+)"
)
_RE_TEST = re.compile(
    r"Test \(summary\): loss=([\d.]+)\s+acc_s2f=([\d.]+)\s+acc_p2f=([\d.]+)"
)

S2F_MAJORITY = 0.6825
P2F_MAJORITY = 0.5106

RUNS = [
    {
        "key": "baseline",
        "log": LOGS / "ehr-symile-tr-47745355.out",
        "results": ROOT / "EHREncoderTransformer/output/results.json",
        "label": "Baseline\n(47745355)",
        "short": "Baseline",
        "color": "#7f7f7f",
    },
    {
        "key": "fixA",
        "log": LOGS / "ehr-tr-fixA-47789415.out",
        "results": ROOT / "EHREncoderTransformer/output_fixA/results.json",
        "label": "Fix-A\n(47789415)",
        "short": "Fix-A",
        "color": "#1f77b4",
    },
    {
        "key": "fixC",
        "log": LOGS / "ehr-tr-fixC-47789417.out",
        "results": ROOT / "EHREncoderTransformer/output_fixC/results.json",
        "label": "Fix-C (best)\n(47789417)",
        "short": "Fix-C",
        "color": "#2ca02c",
    },
]

_FOLLOWUP_SPECS = [
    ("ehr-tr-fixD-*.out", "fixD", "output_fixD", "Fix-D", "#9467bd"),
    ("ehr-tr-fixE-*.out", "fixE", "output_fixE", "Fix-E", "#8c564b"),
]


def _discover_followup_runs() -> list[dict]:
    extra = []
    for pattern, key, out_sub, short, color in _FOLLOWUP_SPECS:
        matches = sorted(LOGS.glob(pattern))
        if not matches:
            continue
        log = matches[-1]
        jid = log.stem.split("-")[-1]
        extra.append(
            {
                "key": key,
                "log": log,
                "results": ROOT / f"EHREncoderTransformer/{out_sub}/results.json",
                "label": f"{short}\n({jid})",
                "short": short,
                "color": color,
            }
        )
    return extra


def parse_log(path: Path, meta: dict) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    uw_by_ep: dict[int, tuple[float, float]] = {}
    for m in _RE_EPOCH.finditer(text):
        epochs.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_s2f": float(m.group(3)),
                "train_p2f": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_acc_s2f": float(m.group(6)),
                "val_acc_p2f": float(m.group(7)),
            }
        )
    for m in _RE_UW.finditer(text):
        uw_by_ep[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    for e in epochs:
        if e["epoch"] in uw_by_ep:
            e["train_ce_uw_s2f"], e["train_ce_uw_p2f"] = uw_by_ep[e["epoch"]]
    diags = list(_RE_DIAG.finditer(text))
    for i, d in enumerate(diags):
        if i < len(epochs):
            epochs[i]["grad_norm"] = float(d.group(1))
            epochs[i]["param_l2"] = float(d.group(2))
    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {"loss": float(test_m.group(1)), "acc_s2f": float(test_m.group(2)), "acc_p2f": float(test_m.group(3))}
    return {**meta, "job_id": path.stem.split("-")[-1], "epochs": epochs, "test": test}


def _xy(epochs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([e["epoch"] for e in epochs], dtype=np.int64),
        np.array([e.get(key, np.nan) for e in epochs], dtype=np.float64),
    )


def plot_compare(runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("EHREncoderTransformer — Baseline vs Fix (val curves)", fontsize=12)
    for r in runs:
        ep = r["epochs"]
        if not ep:
            continue
        c, lbl = r["color"], r["short"]
        for ax, key, title in zip(
            axes.flat,
            ["val_acc_s2f", "val_acc_p2f", "val_loss", "train_loss"],
            ["Val acc s2f", "Val acc p2f", "Val loss", "Train loss"],
        ):
            x, y = _xy(ep, key)
            ax.plot(x, y, label=lbl, color=c, linewidth=1.6)
    axes[0, 0].axhline(S2F_MAJORITY, color="#1f77b4", linestyle="--", alpha=0.4)
    axes[0, 1].axhline(P2F_MAJORITY, color="#ff7f0e", linestyle="--", alpha=0.4)
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_individual(run: dict, out_path: Path) -> None:
    ep = run["epochs"]
    if not ep:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{run['short']} — job {run['job_id']}", fontsize=12)
    c = run["color"]
    x, y = _xy(ep, "train_loss")
    axes[0, 0].plot(x, y, label="train", color=c)
    x, y = _xy(ep, "val_loss")
    axes[0, 0].plot(x, y, "--", label="val", color=c, alpha=0.65)
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if ep[0].get("train_ce_uw_s2f") is not None:
        x, y = _xy(ep, "train_ce_uw_s2f")
        axes[0, 1].plot(x, y, label="uw s2f", color="#1f77b4")
        x, y = _xy(ep, "train_ce_uw_p2f")
        axes[0, 1].plot(x, y, label="uw p2f", color="#ff7f0e")
        axes[0, 1].set_title("Unweighted train CE")
    else:
        x, y = _xy(ep, "train_s2f")
        axes[0, 1].plot(x, y, label="s2f", color="#1f77b4")
        x, y = _xy(ep, "train_p2f")
        axes[0, 1].plot(x, y, label="p2f", color="#ff7f0e")
        axes[0, 1].set_title("Train CE (weighted)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    x, y = _xy(ep, "val_acc_s2f")
    axes[1, 0].plot(x, y, label="s2f", color="#1f77b4")
    x, y = _xy(ep, "val_acc_p2f")
    axes[1, 0].plot(x, y, label="p2f", color="#ff7f0e")
    axes[1, 0].axhline(S2F_MAJORITY, color="#1f77b4", linestyle="--", alpha=0.35)
    axes[1, 0].axhline(P2F_MAJORITY, color="#ff7f0e", linestyle="--", alpha=0.35)
    axes[1, 0].set_title("Val accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    if ep[0].get("param_l2") is not None:
        x, y = _xy(ep, "param_l2")
        axes[1, 1].plot(x, y, label="param_l2", color=c)
        ax2 = axes[1, 1].twinx()
        x, y = _xy(ep, "grad_norm")
        ax2.plot(x, y, label="grad_norm", color="#d62728", alpha=0.7)
        axes[1, 1].set_title("param_l2 / grad_norm")
        axes[1, 1].legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        axes[1, 1].axis("off")
        t = run.get("test") or {}
        axes[1, 1].text(
            0.05, 0.5,
            f"Test acc_s2f={t.get('acc_s2f', float('nan')):.4f}\n"
            f"Test acc_p2f={t.get('acc_p2f', float('nan')):.4f}",
            fontsize=11, family="monospace", va="center",
        )
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _test_from_run(run: dict) -> dict:
    t = run.get("test")
    if t:
        return t
    rj = run.get("results_json") or {}
    ts = rj.get("test_summary") or rj.get("test") or {}
    if ts.get("acc_s2f") is not None:
        return ts
    return {}


def plot_test_bar(runs: list[dict], out_path: Path) -> None:
    bar_runs = [r for r in runs if _test_from_run(r).get("acc_s2f") is not None]
    if not bar_runs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["short"] for r in bar_runs]
    s2f = [_test_from_run(r).get("acc_s2f", np.nan) for r in bar_runs]
    p2f = [_test_from_run(r).get("acc_p2f", np.nan) for r in bar_runs]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, s2f, w, label="Test acc s2f", color="#1f77b4")
    ax.bar(x + w / 2, p2f, w, label="Test acc p2f", color="#ff7f0e")
    ax.axhline(S2F_MAJORITY, color="#1f77b4", linestyle="--", alpha=0.5)
    ax.axhline(P2F_MAJORITY, color="#ff7f0e", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.82)
    ax.set_title("EHREncoderTransformer Test Accuracy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed = []
    for spec in RUNS + _discover_followup_runs():
        if not spec["log"].is_file():
            print(f"SKIP missing log: {spec['log']}")
            continue
        run = parse_log(spec["log"], {k: spec[k] for k in ("key", "label", "short", "color")})
        if spec["results"].is_file():
            run["results_json"] = json.loads(spec["results"].read_text())
            if run.get("test") is None:
                ts = run["results_json"].get("test_summary") or run["results_json"].get("test") or {}
                if ts.get("acc_s2f") is not None:
                    run["test"] = {
                        "loss": ts.get("loss"),
                        "acc_s2f": ts.get("acc_s2f"),
                        "acc_p2f": ts.get("acc_p2f"),
                    }
        parsed.append(run)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(parsed, f, indent=2, default=str)

    plot_compare(parsed, OUT_DIR / "training_curves_compare.png")
    plot_test_bar(parsed, OUT_DIR / "test_accuracy_compare.png")
    for run in parsed:
        plot_individual(run, OUT_DIR / f"training_curves_{run['key']}.png")

    print(f"Wrote figures to {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        if p.suffix == ".png":
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
