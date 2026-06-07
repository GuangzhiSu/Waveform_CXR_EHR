#!/usr/bin/env python3
"""Visualize EmbedPred accuracy-fix experiments (baseline vs expA/B/C)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT_DIR = Path(__file__).resolve().parent / "ehr_embedpred_exp_runs"

_RE_PRETRAIN = re.compile(
    r"\[Pretrain\] Epoch (\d+)/\d+\s+train_embed=([\d.]+)\s+val_embed=([\d.]+)"
)
_RE_FINETUNE = re.compile(
    r"\[Finetune\] Epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\)\s+"
    r".*?val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_TEST = re.compile(
    r"Test \(summary\): loss=([\d.]+)\s+acc_s2f=([\d.]+)\s+acc_p2f=([\d.]+)"
)
_RE_SCHEDULE = re.compile(
    r"class_weights=(\w+)\s+class_weight_mode=(\w+)\s+label_smoothing=([\d.]+)"
    r".*?pretrain_resume=(\w+)\s+pretrain_min_epochs=(\d+)"
    r".*?p2f_weight=([\d.]+).*?lr=([\d.e+-]+)"
    r"|p2f_weight=([\d.]+).*?lr=([\d.e+-]+)"
)
_RE_FINETUNE_EP = re.compile(r"finetune_epochs=(\d+)")
_RE_LR = re.compile(r"lr=([\d.e+-]+)")

RUNS = [
    {
        "key": "baseline",
        "log": LOGS / "ehr-symile-embed-2ph-47748678.out",
        "results": ROOT / "EHREncoderTransformerEmbedPred/output_twophase/results.json",
        "label": "Baseline\n(47748678)",
        "short": "Baseline",
        "color": "#7f7f7f",
    },
    {
        "key": "expA",
        "log": LOGS / "ehr-embed-expA-47753034.out",
        "results": ROOT / "EHREncoderTransformerEmbedPred/output_twophase_expA/results.json",
        "label": "Exp A\n(47753034)",
        "short": "Exp A",
        "color": "#d62728",
    },
    {
        "key": "expB",
        "log": LOGS / "ehr-embed-expB-47753016.out",
        "results": ROOT / "EHREncoderTransformerEmbedPred/output_twophase_expB/results.json",
        "label": "Exp B\n(47753016)",
        "short": "Exp B",
        "color": "#1f77b4",
    },
    {
        "key": "expC",
        "log": LOGS / "ehr-embed-expC-47753017.out",
        "results": ROOT / "EHREncoderTransformerEmbedPred/output_twophase_expC/results.json",
        "label": "Exp C (best)\n(47753017)",
        "short": "Exp C",
        "color": "#2ca02c",
    },
]

S2F_MAJORITY = 0.6825
P2F_MAJORITY = 0.5106


def parse_two_phase_log(path: Path, meta: dict) -> dict[str, Any]:
    text = path.read_text()
    job_id = path.stem.split("-")[-1]

    pretrain = []
    for m in _RE_PRETRAIN.finditer(text):
        pretrain.append(
            {
                "epoch": int(m.group(1)),
                "train_embed": float(m.group(2)),
                "val_embed": float(m.group(3)),
            }
        )

    finetune = []
    for m in _RE_FINETUNE.finditer(text):
        finetune.append(
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

    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {
            "loss": float(test_m.group(1)),
            "acc_s2f": float(test_m.group(2)),
            "acc_p2f": float(test_m.group(3)),
        }

    best_ft_m = re.search(
        r"\[Finetune\] Early stopping.*best epoch (\d+), best metric=([\d.]+)", text
    )
    best_pre_m = re.search(
        r"\[Pretrain\] Early stopping.*best epoch (\d+), best metric=([\d.]+)", text
    )

    cfg = {
        "class_weights": "?",
        "class_weight_mode": "?",
        "label_smoothing": "?",
        "pretrain_resume": "?",
        "pretrain_min_epochs": "?",
        "p2f_weight": "?",
        "lr": "?",
        "finetune_epochs": "?",
    }
    for line in text.splitlines():
        if "class_weights=" in line:
            m = re.search(
                r"class_weights=(\w+)\s+class_weight_mode=(\w+)\s+label_smoothing=([\d.]+)"
                r".*?pretrain_resume=(\w+)\s+pretrain_min_epochs=(\d+)"
                r".*?p2f_weight=([\d.]+)",
                line,
            )
            if m:
                cfg.update(
                    {
                        "class_weights": m.group(1),
                        "class_weight_mode": m.group(2),
                        "label_smoothing": m.group(3),
                        "pretrain_resume": m.group(4),
                        "pretrain_min_epochs": m.group(5),
                        "p2f_weight": m.group(6),
                    }
                )
        if "finetune_epochs=" in line and "Schedule:" in line:
            m = _RE_FINETUNE_EP.search(line)
            if m:
                cfg["finetune_epochs"] = m.group(1)
        if line.strip().startswith("Parameters:") and "lr=" in line:
            m = _RE_LR.search(line)
            if m:
                cfg["lr"] = m.group(1)

    return {
        **meta,
        "job_id": job_id,
        "pretrain_epochs": pretrain,
        "finetune_epochs": finetune,
        "best_pretrain_epoch": int(best_pre_m.group(1)) if best_pre_m else None,
        "best_finetune_epoch": int(best_ft_m.group(1)) if best_ft_m else None,
        "best_finetune_val_loss": float(best_ft_m.group(2)) if best_ft_m else None,
        "test": test,
        "config": cfg,
    }


def _xy(epochs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([e["epoch"] for e in epochs], dtype=np.int64),
        np.array([e.get(key, np.nan) for e in epochs], dtype=np.float64),
    )


def plot_test_accuracy_compare(runs: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [r["label"] for r in runs]
    s2f = [r.get("test", {}).get("acc_s2f", np.nan) for r in runs]
    p2f = [r.get("test", {}).get("acc_p2f", np.nan) for r in runs]
    x = np.arange(len(names))
    w = 0.35
    bars_s = ax.bar(x - w / 2, s2f, w, label="Test acc s2f", color="#1f77b4", edgecolor="white")
    bars_p = ax.bar(x + w / 2, p2f, w, label="Test acc p2f", color="#ff7f0e", edgecolor="white")
    ax.axhline(S2F_MAJORITY, color="#1f77b4", linestyle="--", alpha=0.55, linewidth=1.2)
    ax.axhline(P2F_MAJORITY, color="#ff7f0e", linestyle="--", alpha=0.55, linewidth=1.2)
    ax.text(len(names) - 0.5, S2F_MAJORITY + 0.008, f"s2f majority ({S2F_MAJORITY:.1%})", fontsize=8, color="#1f77b4")
    ax.text(len(names) - 0.5, P2F_MAJORITY + 0.008, f"p2f majority ({P2F_MAJORITY:.1%})", fontsize=8, color="#ff7f0e")

    for bar, val in zip(bars_s, s2f):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_p, p2f):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.82)
    ax.set_title("EmbedPred Test Accuracy — Baseline vs Training-Fix Experiments")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_finetune_compare(runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Finetune Phase — Validation Curves (all runs)", fontsize=12)

    for r in runs:
        ft = r["finetune_epochs"]
        if not ft:
            continue
        c = r["color"]
        lbl = r["short"]
        x, y = _xy(ft, "val_acc_s2f")
        axes[0, 0].plot(x, y, label=lbl, color=c, linewidth=1.6)
        x, y = _xy(ft, "val_acc_p2f")
        axes[0, 1].plot(x, y, label=lbl, color=c, linewidth=1.6)
        x, y = _xy(ft, "val_loss")
        axes[1, 0].plot(x, y, label=lbl, color=c, linewidth=1.6)
        x, y = _xy(ft, "train_loss")
        axes[1, 1].plot(x, y, label=lbl, color=c, linewidth=1.6)
        if r.get("best_finetune_epoch"):
            for ax in axes.flat:
                ax.axvline(r["best_finetune_epoch"], color=c, linestyle=":", alpha=0.35, linewidth=1)

    axes[0, 0].axhline(S2F_MAJORITY, color="#1f77b4", linestyle="--", alpha=0.4)
    axes[0, 1].axhline(P2F_MAJORITY, color="#ff7f0e", linestyle="--", alpha=0.4)
    axes[0, 0].set_title("Val accuracy — s2f")
    axes[0, 1].set_title("Val accuracy — p2f")
    axes[1, 0].set_title("Val loss")
    axes[1, 1].set_title("Train loss")
    for ax in axes.flat:
        ax.set_xlabel("Finetune epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_individual_two_phase(run: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{run['short']} — job {run['job_id']}", fontsize=12)

    pre = run["pretrain_epochs"]
    ft = run["finetune_epochs"]
    c = run["color"]

    if pre:
        x, y = _xy(pre, "train_embed")
        axes[0, 0].plot(x, y, label="train", color="#9467bd")
        x, y = _xy(pre, "val_embed")
        axes[0, 0].plot(x, y, "--", label="val", color="#d62728")
    axes[0, 0].set_title(f"Phase 1 — Embed Pretrain ({len(pre)} ep ran)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    if ft:
        x, y = _xy(ft, "train_loss")
        axes[0, 1].plot(x, y, label="train", color=c)
        x, y = _xy(ft, "val_loss")
        axes[0, 1].plot(x, y, "--", label="val", color=c, alpha=0.65)
        if run.get("best_finetune_epoch"):
            axes[0, 1].axvline(run["best_finetune_epoch"], color=c, linestyle=":", alpha=0.5)
    axes[0, 1].set_title("Phase 2 — Cls Finetune")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    if ft:
        x, y = _xy(ft, "val_acc_s2f")
        axes[1, 0].plot(x, y, label="s2f", color="#1f77b4")
        axes[1, 0].axhline(S2F_MAJORITY, color="#1f77b4", linestyle="--", alpha=0.4)
        x, y = _xy(ft, "val_acc_p2f")
        axes[1, 0].plot(x, y, label="p2f", color="#ff7f0e")
        axes[1, 0].axhline(P2F_MAJORITY, color="#ff7f0e", linestyle="--", alpha=0.4)
    axes[1, 0].set_title("Val Accuracy (finetune)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    t = run.get("test") or {}
    cfg = run.get("config") or {}
    summary = (
        f"Config:\n"
        f"  class_weights={cfg.get('class_weights')}  mode={cfg.get('class_weight_mode')}\n"
        f"  label_smoothing={cfg.get('label_smoothing')}  pretrain_resume={cfg.get('pretrain_resume')}\n"
        f"  p2f_weight={cfg.get('p2f_weight')}  lr={cfg.get('lr')}  finetune_ep={cfg.get('finetune_epochs')}\n\n"
        f"Best finetune: epoch {run.get('best_finetune_epoch')}  "
        f"val_loss={run.get('best_finetune_val_loss')}\n"
        f"Test: acc_s2f={t.get('acc_s2f', float('nan')):.4f}  "
        f"acc_p2f={t.get('acc_p2f', float('nan')):.4f}"
    )
    axes[1, 1].text(0.04, 0.5, summary, fontsize=10, va="center", family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_best(baseline: dict, best: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Before vs After Training Fixes (Baseline → Exp C)", fontsize=12)

    metrics = [
        ("acc_s2f", "S2F test accuracy", S2F_MAJORITY, "#1f77b4"),
        ("acc_p2f", "P2F test accuracy", P2F_MAJORITY, "#ff7f0e"),
        ("loss", "Test loss", None, "#2ca02c"),
    ]
    for ax, (key, title, maj, color) in zip(axes, metrics):
        b = baseline.get("test", {}).get(key, np.nan)
        e = best.get("test", {}).get(key, np.nan)
        bars = ax.bar(["Baseline", "Exp C"], [b, e], color=["#7f7f7f", "#2ca02c"], edgecolor="white", width=0.55)
        if maj is not None:
            ax.axhline(maj, color=color, linestyle="--", alpha=0.5)
        for bar, val in zip(bars, [b, e]):
            if np.isfinite(val):
                fmt = f"{val:.1%}" if key.startswith("acc") else f"{val:.4f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt, ha="center", va="bottom", fontsize=10)
        if key.startswith("acc"):
            delta = e - b
            ax.annotate(f"Δ {delta:+.1%}", xy=(1, e), xytext=(1.15, (b + e) / 2),
                        fontsize=10, color="#2ca02c", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if key.startswith("acc"):
            ax.set_ylim(0, 0.82)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results_path: Path, out_path: Path, title: str) -> None:
    if not results_path.is_file():
        return
    data = json.loads(results_path.read_text())
    test = data.get("test") or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=12)
    class_names = ["change_0", "change_1", "change_2"]

    for ax, head, color in zip(axes, ("s2f", "p2f"), ("Blues", "Oranges")):
        block = test.get(head) or {}
        cm = np.array(block.get("confusion_matrix") or [], dtype=np.int64)
        if cm.size == 0:
            ax.set_visible(False)
            continue
        acc = block.get("accuracy", float("nan"))
        maj = block.get("majority_baseline", float("nan"))
        im = ax.imshow(cm, cmap=color, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{head.upper()}  acc={acc:.3f}  maj={maj:.3f}")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_config_table(runs: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    cols = ["Run", "class_weights", "p2f_w", "lr", "finetune_ep", "acc_s2f", "acc_p2f"]
    rows = []
    for r in runs:
        cfg = r.get("config") or {}
        t = r.get("test") or {}
        rows.append(
            [
                r["short"],
                f"{cfg.get('class_weights')} ({cfg.get('class_weight_mode')})",
                str(cfg.get("p2f_weight")),
                str(cfg.get("lr")),
                str(cfg.get("finetune_epochs")),
                f"{t.get('acc_s2f', float('nan')):.3f}" if t else "—",
                f"{t.get('acc_p2f', float('nan')):.3f}" if t else "—",
            ]
        )
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    ax.set_title("Experiment Config & Test Accuracy Summary", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parsed = []
    for spec in RUNS:
        if not spec["log"].is_file():
            print(f"SKIP missing log: {spec['log']}")
            continue
        run = parse_two_phase_log(
            spec["log"],
            {"key": spec["key"], "label": spec["label"], "short": spec["short"], "color": spec["color"]},
        )
        if spec["results"].is_file():
            run["results_json"] = json.loads(spec["results"].read_text())
        parsed.append(run)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(parsed, f, indent=2, default=str)

    plot_test_accuracy_compare(parsed, OUT_DIR / "test_accuracy_compare.png")
    plot_finetune_compare(parsed, OUT_DIR / "finetune_curves_compare.png")
    plot_config_table(parsed, OUT_DIR / "experiment_summary_table.png")

    baseline = next(r for r in parsed if r["key"] == "baseline")
    best = next(r for r in parsed if r["key"] == "expC")
    plot_baseline_vs_best(baseline, best, OUT_DIR / "baseline_vs_expC.png")

    for run in parsed:
        plot_individual_two_phase(run, OUT_DIR / f"training_curves_{run['key']}.png")

    plot_confusion_matrices(
        ROOT / "EHREncoderTransformerEmbedPred/output_twophase_expC/results.json",
        OUT_DIR / "confusion_matrices_expC.png",
        "Exp C — Test Confusion Matrices (best run)",
    )

    print(f"Wrote figures to {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
