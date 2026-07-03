#!/usr/bin/env python3
"""Parse Symile EHR encoder Slurm logs and plot training curves (figures/ style)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT_DIR = Path(__file__).resolve().parent / "ehr_symile_47745355_47745356_47748678"

_RE_TR = re.compile(
    r"Epoch (\d+)/\d+\s+"
    r"train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\)\s+"
    r"val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_EMBED = re.compile(
    r"Epoch (\d+)/\d+\s+"
    r"train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\s+embed=([\d.]+)\)\s+"
    r".*?val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)\s+val_embed=([\d.]+)"
)
_RE_PRETRAIN = re.compile(
    r"\[Pretrain\] Epoch (\d+)/\d+\s+train_embed=([\d.]+)\s+val_embed=([\d.]+)"
)
_RE_FINETUNE = re.compile(
    r"\[Finetune\] Epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\)\s+"
    r".*?val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_DIAG = re.compile(
    r"train diagnostics: last_batch_grad_norm=([\d.]+)\s+param_l2=([\d.]+)"
)
_RE_PRETRAIN_DIAG = re.compile(r"last_grad_norm=([\d.]+)")
_RE_BEST_TR = re.compile(r"best epoch (\d+), best val_loss=([\d.]+)")
_RE_BEST_FINETUNE = re.compile(r"\[Finetune\] Early stopping.*best epoch (\d+), best metric=([\d.]+)")
_RE_BEST_PRETRAIN_EMBED = re.compile(r"\[Pretrain\].*")  # use min val_embed from epochs
_RE_TEST = re.compile(
    r"Test \(summary\): loss=([\d.]+)\s+acc_s2f=([\d.]+)\s+acc_p2f=([\d.]+)(?:\s+embed=([\d.]+))?"
)


def _attach_diags(epochs: list[dict], text: str, diag_re: re.Pattern) -> None:
    diags = list(diag_re.finditer(text))
    for i, d in enumerate(diags):
        if i < len(epochs):
            if "grad_norm" in epochs[i] or "grad_norm" not in epochs[i]:
                epochs[i]["grad_norm"] = float(d.group(1))
            if diag_re == _RE_DIAG and d.lastindex >= 2:
                epochs[i]["param_l2"] = float(d.group(2))


def parse_tr_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    for m in _RE_TR.finditer(text):
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
    _attach_diags(epochs, text, _RE_DIAG)
    best_m = _RE_BEST_TR.search(text)
    best_epoch = int(best_m.group(1)) if best_m else None
    best_val = float(best_m.group(2)) if best_m else None
    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {
            "loss": float(test_m.group(1)),
            "acc_s2f": float(test_m.group(2)),
            "acc_p2f": float(test_m.group(3)),
        }
    return {
        "name": "EHREncoderTransformer (symile-tr)",
        "job_id": "47745355",
        "schedule": "single_phase_cls",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": test,
    }


def parse_embed_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    for m in _RE_EMBED.finditer(text):
        epochs.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_s2f": float(m.group(3)),
                "train_p2f": float(m.group(4)),
                "train_embed": float(m.group(5)),
                "val_loss": float(m.group(6)),
                "val_acc_s2f": float(m.group(7)),
                "val_acc_p2f": float(m.group(8)),
                "val_embed": float(m.group(9)),
            }
        )
    _attach_diags(epochs, text, _RE_DIAG)
    best_m = re.search(r"best epoch (\d+), best val_loss=([\d.]+)", text)
    best_epoch = int(best_m.group(1)) if best_m else None
    best_val = float(best_m.group(2)) if best_m else None
    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {
            "loss": float(test_m.group(1)),
            "acc_s2f": float(test_m.group(2)),
            "acc_p2f": float(test_m.group(3)),
            "embed": float(test_m.group(4)) if test_m.group(4) else None,
        }
    return {
        "name": "EmbedPred (symile-embed, joint)",
        "job_id": "47745356",
        "schedule": "single_phase_joint",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": test,
    }


def parse_embed_2ph_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    pretrain = []
    for m in _RE_PRETRAIN.finditer(text):
        pretrain.append(
            {
                "epoch": int(m.group(1)),
                "train_embed": float(m.group(2)),
                "val_embed": float(m.group(3)),
            }
        )
    pretrain_diags = [float(m.group(1)) for m in _RE_PRETRAIN_DIAG.finditer(text)]
    for i, g in enumerate(pretrain_diags):
        if i < len(pretrain):
            pretrain[i]["grad_norm"] = g

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

    best_pre = min(pretrain, key=lambda e: e["val_embed"]) if pretrain else None
    best_ft_m = _RE_BEST_FINETUNE.search(text)
    best_ft_epoch = int(best_ft_m.group(1)) if best_ft_m else None
    best_ft_val = float(best_ft_m.group(2)) if best_ft_m else None

    test_m = _RE_TEST.search(text)
    test = None
    if test_m:
        test = {
            "loss": float(test_m.group(1)),
            "acc_s2f": float(test_m.group(2)),
            "acc_p2f": float(test_m.group(3)),
        }

    return {
        "name": "EmbedPred (symile-embed-2ph)",
        "job_id": "47748678",
        "schedule": "two_phase",
        "pretrain_epochs": pretrain,
        "finetune_epochs": finetune,
        "best_pretrain_epoch": best_pre["epoch"] if best_pre else None,
        "best_pretrain_val_embed": best_pre["val_embed"] if best_pre else None,
        "best_finetune_epoch": best_ft_epoch,
        "best_finetune_val_loss": best_ft_val,
        "test": test,
    }


def _xy(epochs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([e["epoch"] for e in epochs], dtype=np.int64),
        np.array([e.get(key, np.nan) for e in epochs], dtype=np.float64),
    )


def plot_compare(tr: dict, embed: dict, embed2ph: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Symile EHR Encoders — jobs 47745355 (TR) / 47745356 (Embed joint) / 47748678 (Embed 2ph finetune)",
        fontsize=12,
    )

    colors = {
        "tr": "#1f77b4",
        "embed": "#ff7f0e",
        "2ph": "#2ca02c",
    }

    def plot3(ax, key_tr, key_emb, key_2ph, title, ylabel, *, val_key=None):
        x, y = _xy(tr["epochs"], key_tr)
        ax.plot(x, y, label="TR", color=colors["tr"], linewidth=1.5)
        x, y = _xy(embed["epochs"], key_emb)
        ax.plot(x, y, label="Embed joint", color=colors["embed"], linewidth=1.5)
        x, y = _xy(embed2ph["finetune_epochs"], key_2ph)
        ax.plot(x, y, label="Embed 2ph (finetune)", color=colors["2ph"], linewidth=1.5)
        if val_key:
            x, y = _xy(tr["epochs"], val_key)
            ax.plot(x, y, "--", color=colors["tr"], alpha=0.65)
            x, y = _xy(embed["epochs"], val_key)
            ax.plot(x, y, "--", color=colors["embed"], alpha=0.65)
            x, y = _xy(embed2ph["finetune_epochs"], val_key)
            ax.plot(x, y, "--", color=colors["2ph"], alpha=0.65)
        for log, c, attr in [
            (tr, colors["tr"], "best_epoch"),
            (embed, colors["embed"], "best_epoch"),
            (embed2ph, colors["2ph"], "best_finetune_epoch"),
        ]:
            be = log.get(attr)
            if be:
                ax.axvline(be, color=c, linestyle=":", alpha=0.45, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax = axes[0, 0]
    plot3(ax, "train_loss", "train_loss", "train_loss", "Total / Combined Loss", "Loss", val_key="val_loss")

    plot3(axes[0, 1], "train_s2f", "train_s2f", "train_s2f", "Train CE — s2f", "Loss")
    plot3(axes[0, 2], "train_p2f", "train_p2f", "train_p2f", "Train CE — p2f", "Loss")
    plot3(axes[1, 0], "val_acc_s2f", "val_acc_s2f", "val_acc_s2f", "Val Accuracy — s2f", "Accuracy")
    plot3(axes[1, 1], "val_acc_p2f", "val_acc_p2f", "val_acc_p2f", "Val Accuracy — p2f", "Accuracy")

    ax = axes[1, 2]
    x, y = _xy(embed["epochs"], "train_embed")
    ax.plot(x, y, label="Embed joint train", color=colors["embed"])
    x, y = _xy(embed["epochs"], "val_embed")
    ax.plot(x, y, "--", label="Embed joint val", color=colors["embed"], alpha=0.7)
    x, y = _xy(embed2ph["pretrain_epochs"], "train_embed")
    ax.plot(x, y, label="2ph pretrain train", color="#9467bd")
    x, y = _xy(embed2ph["pretrain_epochs"], "val_embed")
    ax.plot(x, y, "--", label="2ph pretrain val", color="#d62728", alpha=0.8)
    if embed.get("best_epoch"):
        ax.axvline(embed["best_epoch"], color=colors["embed"], linestyle=":", alpha=0.4)
    if embed2ph.get("best_pretrain_epoch"):
        ax.axvline(embed2ph["best_pretrain_epoch"], color="#9467bd", linestyle=":", alpha=0.4)
    ax.set_title("Embed Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_individual(log: dict, out_path: Path, *, two_phase: bool = False) -> None:
    if two_phase:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{log['name']} (job {log['job_id']})", fontsize=12)
        pre = log["pretrain_epochs"]
        ft = log["finetune_epochs"]
        x, y = _xy(pre, "train_embed")
        axes[0, 0].plot(x, y, label="train", color="#9467bd")
        x, y = _xy(pre, "val_embed")
        axes[0, 0].plot(x, y, "--", label="val", color="#d62728")
        axes[0, 0].set_title("Phase 1 — Embed Pretrain (100 ep)")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        x, y = _xy(ft, "train_loss")
        axes[0, 1].plot(x, y, label="train", color="#2ca02c")
        x, y = _xy(ft, "val_loss")
        axes[0, 1].plot(x, y, "--", label="val", color="#2ca02c", alpha=0.65)
        if log.get("best_finetune_epoch"):
            axes[0, 1].axvline(log["best_finetune_epoch"], color="#2ca02c", linestyle=":", alpha=0.5)
        axes[0, 1].set_title("Phase 2 — Cls Finetune")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        x, y = _xy(ft, "val_acc_s2f")
        axes[1, 0].plot(x, y, label="s2f", color="#1f77b4")
        x, y = _xy(ft, "val_acc_p2f")
        axes[1, 0].plot(x, y, label="p2f", color="#ff7f0e")
        axes[1, 0].set_title("Val Accuracy (finetune)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].axis("off")
        t = log.get("test") or {}
        summary = (
            f"Best pretrain: epoch {log.get('best_pretrain_epoch')}  "
            f"val_embed={log.get('best_pretrain_val_embed'):.4f}\n"
            f"Best finetune: epoch {log.get('best_finetune_epoch')}  "
            f"val_loss={log.get('best_finetune_val_loss'):.4f}\n"
            f"Test: loss={t.get('loss', float('nan')):.4f}  "
            f"acc_s2f={t.get('acc_s2f', float('nan')):.4f}  "
            f"acc_p2f={t.get('acc_p2f', float('nan')):.4f}"
        )
        axes[1, 1].text(0.05, 0.5, summary, fontsize=11, va="center", family="monospace")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{log['name']} (job {log['job_id']})", fontsize=12)
        ep = log["epochs"]
        x, y = _xy(ep, "train_loss")
        axes[0, 0].plot(x, y, label="train", color="#1f77b4")
        x, y = _xy(ep, "val_loss")
        axes[0, 0].plot(x, y, "--", label="val", color="#1f77b4", alpha=0.65)
        if log.get("best_epoch"):
            axes[0, 0].axvline(log["best_epoch"], color="#1f77b4", linestyle=":", alpha=0.5)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        x, y = _xy(ep, "train_s2f")
        axes[0, 1].plot(x, y, label="train s2f", color="#1f77b4")
        x, y = _xy(ep, "train_p2f")
        axes[0, 1].plot(x, y, label="train p2f", color="#ff7f0e")
        axes[0, 1].set_title("Train CE")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        x, y = _xy(ep, "val_acc_s2f")
        axes[1, 0].plot(x, y, label="s2f", color="#1f77b4")
        x, y = _xy(ep, "val_acc_p2f")
        axes[1, 0].plot(x, y, label="p2f", color="#ff7f0e")
        axes[1, 0].set_title("Val Accuracy")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].axis("off")
        t = log.get("test") or {}
        extra = ""
        if "train_embed" in (ep[0] if ep else {}):
            extra = f"\n(embed loss in joint training)"
        summary = (
            f"Best epoch: {log.get('best_epoch')}  val_loss={log.get('best_val_loss')}\n"
            f"Test: loss={t.get('loss', float('nan')):.4f}  "
            f"acc_s2f={t.get('acc_s2f', float('nan')):.4f}  "
            f"acc_p2f={t.get('acc_p2f', float('nan')):.4f}{extra}"
        )
        axes[1, 1].text(0.05, 0.5, summary, fontsize=11, va="center", family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_summary(tr: dict, embed: dict, embed2ph: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["TR\n47745355", "Embed joint\n47745356", "Embed 2ph\n47748678"]
    s2f = [
        tr.get("test", {}).get("acc_s2f", np.nan),
        embed.get("test", {}).get("acc_s2f", np.nan),
        embed2ph.get("test", {}).get("acc_s2f", np.nan),
    ]
    p2f = [
        tr.get("test", {}).get("acc_p2f", np.nan),
        embed.get("test", {}).get("acc_p2f", np.nan),
        embed2ph.get("test", {}).get("acc_p2f", np.nan),
    ]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, s2f, w, label="Test acc s2f", color="#1f77b4")
    ax.bar(x + w / 2, p2f, w, label="Test acc p2f", color="#ff7f0e")
    ax.axhline(0.6819, color="#1f77b4", linestyle="--", alpha=0.4, label="s2f majority (~0.68)")
    ax.axhline(0.5106, color="#ff7f0e", linestyle="--", alpha=0.4, label="p2f majority (~0.51)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy Comparison (Symile preprocessing)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_cached_metrics() -> dict | None:
    cache = OUT_DIR / "metrics.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_metrics()
    tr = parse_tr_log(LOGS / "ehr-symile-tr-47745355.out")
    if LOGS.joinpath("ehr-symile-embed-47745356.out").is_file():
        embed = parse_embed_log(LOGS / "ehr-symile-embed-47745356.out")
    elif cached and cached.get("embed_log", {}).get("epochs"):
        print("Using cached metrics for symile-embed 47745356 (log deleted)")
        embed = cached["embed_log"]
    else:
        embed = {"name": "EHREncoderTransformerEmbedPred", "job_id": "47745356", "epochs": [], "test": None}
    embed2ph = parse_embed_2ph_log(LOGS / "ehr-symile-embed-2ph-47748678.out")

    metrics = {
        "tr_log": tr,
        "embed_log": embed,
        "embed_2ph_log": embed2ph,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_compare(tr, embed, embed2ph, OUT_DIR / "training_curves_compare.png")
    plot_individual(tr, OUT_DIR / "training_curves_symile_tr_47745355.png")
    plot_individual(embed, OUT_DIR / "training_curves_symile_embed_47745356.png")
    plot_individual(embed2ph, OUT_DIR / "training_curves_symile_embed_2ph_47748678.png", two_phase=True)
    plot_test_summary(tr, embed, embed2ph, OUT_DIR / "test_accuracy_compare.png")

    print(f"Wrote figures to {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
