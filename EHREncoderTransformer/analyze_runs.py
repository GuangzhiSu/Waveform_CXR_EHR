#!/usr/bin/env python3
"""Compare EHREncoderTransformer vs EmbedPred runs: training curves, norm stats, t-SNE."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
_EMBED_EXP = PROJECT_ROOT / "EHREncoderTransformerEmbedPred"
for _p in (
    PROJECT_ROOT,
    PROJECT_ROOT / "BaselineExperiment",
    PROJECT_ROOT / "EHRTrend",
    _EXP,
):
    if _p.is_dir():
        sys.path.insert(0, str(_p))


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from config import (  # noqa: E402
    ANCHOR_POOL,
    BATCH_SIZE,
    D_MODEL,
    DROPOUT,
    EMBED_DIM,
    ENRICHED_CSV,
    HEAD_DROPOUT,
    LOOKBACK_MAX_HOURS,
    LOOKBACK_MIN_HOURS,
    MAX_SEQ_LENGTH,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    P2F_OR_S2F_CSV,
    SCHEMA_CSV,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from model import EHREncoderTransformer  # noqa: E402
from train import _stratify_labels_from_dataset, collate_anchor_batch  # noqa: E402

sys.path.insert(0, str(_EMBED_EXP))
_embed_model_mod = _import_module("ehr_embed_model", _EMBED_EXP / "model.py")
_embed_ds_mod = _import_module("ehr_embed_dataset", _EMBED_EXP / "anchor_embed_dataset.py")
EHREncoderTransformerEmbedPred = _embed_model_mod.EHREncoderTransformerEmbedPred
EHRAnchorEmbedDataset = _embed_ds_mod.EHRAnchorEmbedDataset

CLASS_NAMES = [f"change_{i}" for i in range(NUM_CLASSES)]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

_RE_EPOCH_TR = re.compile(
    r"Epoch (\d+)/\d+\s+"
    r"train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\)\s+"
    r"val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)"
)
_RE_EPOCH_EMBED = re.compile(
    r"Epoch (\d+)/\d+\s+"
    r"train_loss=([\d.]+)\s+\(s2f=([\d.]+)\s+p2f=([\d.]+)\s+embed=([\d.]+)\)\s+"
    r".*?val_loss=([\d.]+)\s+val_acc_s2f=([\d.]+)\s+val_acc_p2f=([\d.]+)\s+val_embed=([\d.]+)"
)
_RE_DIAG = re.compile(
    r"train diagnostics: last_batch_grad_norm=([\d.]+)\s+param_l2=([\d.]+)"
)
_RE_BEST_EPOCH = re.compile(r"\(best checkpoint epoch (\d+)\)")


def parse_tr_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    for m in _RE_EPOCH_TR.finditer(text):
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
    diags = list(_RE_DIAG.finditer(text))
    for i, d in enumerate(diags):
        if i < len(epochs):
            epochs[i]["grad_norm"] = float(d.group(1))
            epochs[i]["param_l2"] = float(d.group(2))
    best_m = _RE_BEST_EPOCH.search(text)
    best_epoch = int(best_m.group(1)) if best_m else None
    return {"name": "EHREncoderTransformer", "epochs": epochs, "best_epoch": best_epoch}


def parse_embed_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    epochs = []
    for m in _RE_EPOCH_EMBED.finditer(text):
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
    diags = list(_RE_DIAG.finditer(text))
    for i, d in enumerate(diags):
        if i < len(epochs):
            epochs[i]["grad_norm"] = float(d.group(1))
            epochs[i]["param_l2"] = float(d.group(2))
    best_m = _RE_BEST_EPOCH.search(text)
    best_epoch = int(best_m.group(1)) if best_m else None
    return {"name": "EHREncoderTransformerEmbedPred", "epochs": epochs, "best_epoch": best_epoch}


def _epochs_to_arrays(epochs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([e["epoch"] for e in epochs], dtype=np.int64)
    ys = np.array([e.get(key, np.nan) for e in epochs], dtype=np.float64)
    return xs, ys


def plot_training_curves(tr_log: dict, embed_log: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("EHR Encoder Training Curves (jobs 47410738 vs 47414328)", fontsize=13)

    tr_ep = tr_log["epochs"]
    emb_ep = embed_log["epochs"]
    tr_best = tr_log.get("best_epoch")
    emb_best = embed_log.get("best_epoch")

    def _plot(ax, key_tr, key_emb, title, ylabel):
        x1, y1 = _epochs_to_arrays(tr_ep, key_tr)
        x2, y2 = _epochs_to_arrays(emb_ep, key_emb or key_tr)
        ax.plot(x1, y1, label="TR", color="#1f77b4", linewidth=1.5)
        ax.plot(x2, y2, label="EmbedPred", color="#ff7f0e", linewidth=1.5)
        if tr_best:
            ax.axvline(tr_best, color="#1f77b4", linestyle="--", alpha=0.5, linewidth=1)
        if emb_best:
            ax.axvline(emb_best, color="#ff7f0e", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[0, 0]
    x1, y1 = _epochs_to_arrays(tr_ep, "train_loss")
    x2, y2 = _epochs_to_arrays(emb_ep, "train_loss")
    ax.plot(x1, y1, label="TR train", color="#1f77b4")
    ax.plot(x2, y2, label="EmbedPred train", color="#ff7f0e")
    x1v, y1v = _epochs_to_arrays(tr_ep, "val_loss")
    x2v, y2v = _epochs_to_arrays(emb_ep, "val_loss")
    ax.plot(x1v, y1v, "--", label="TR val", color="#1f77b4", alpha=0.7)
    ax.plot(x2v, y2v, "--", label="EmbedPred val", color="#ff7f0e", alpha=0.7)
    if tr_best:
        ax.axvline(tr_best, color="#1f77b4", linestyle=":", alpha=0.4)
    if emb_best:
        ax.axvline(emb_best, color="#ff7f0e", linestyle=":", alpha=0.4)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    _plot(axes[0, 1], "train_s2f", "train_s2f", "Train CE — s2f", "Loss")
    _plot(axes[0, 2], "train_p2f", "train_p2f", "Train CE — p2f", "Loss")
    _plot(axes[1, 0], "val_acc_s2f", "val_acc_s2f", "Val Accuracy — s2f", "Accuracy")
    _plot(axes[1, 1], "val_acc_p2f", "val_acc_p2f", "Val Accuracy — p2f", "Accuracy")

    ax = axes[1, 2]
    x, y = _epochs_to_arrays(emb_ep, "train_embed")
    ax.plot(x, y, label="train embed", color="#ff7f0e")
    x, y = _epochs_to_arrays(emb_ep, "val_embed")
    ax.plot(x, y, "--", label="val embed", color="#d62728")
    if emb_best:
        ax.axvline(emb_best, color="#ff7f0e", linestyle="--", alpha=0.5)
    ax.set_title("EmbedPred — Embed Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_norm_curves(
    tr_log: dict,
    embed_log: dict,
    norm_stats: dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Training Dynamics & anchor_vec L2 Norm (best checkpoint, val set)", fontsize=12)

    for log, color, label in (
        (tr_log, "#1f77b4", "TR"),
        (embed_log, "#ff7f0e", "EmbedPred"),
    ):
        ep = log["epochs"]
        x, y = _epochs_to_arrays(ep, "param_l2")
        axes[0, 0].plot(x, y, label=label, color=color)
        x, y = _epochs_to_arrays(ep, "grad_norm")
        axes[0, 1].plot(x, y, label=label, color=color)
        if log.get("best_epoch"):
            axes[0, 0].axvline(log["best_epoch"], color=color, linestyle="--", alpha=0.4)
            axes[0, 1].axvline(log["best_epoch"], color=color, linestyle="--", alpha=0.4)

    axes[0, 0].set_title("param_l2 (trainable weights)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("last_batch_grad_norm")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    ax = axes[1, 0]
    emb_ep = embed_log["epochs"]
    x, y = _epochs_to_arrays(emb_ep, "train_embed")
    ax.plot(x, y, label="train embed", color="#ff7f0e")
    x, y = _epochs_to_arrays(emb_ep, "val_embed")
    ax.plot(x, y, "--", label="val embed", color="#d62728")
    if embed_log.get("best_epoch"):
        ax.axvline(embed_log["best_epoch"], color="#ff7f0e", linestyle="--", alpha=0.4)
    ax.set_title("EmbedPred embed loss (MSE proxy)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    tr_norms = norm_stats["tr"].get("all_norms", np.array([]))
    emb_norms = norm_stats["embed"].get("all_norms", np.array([]))
    if len(tr_norms) and len(emb_norms):
        bins = np.linspace(
            min(tr_norms.min(), emb_norms.min()),
            max(tr_norms.max(), emb_norms.max()),
            40,
        )
        ax.hist(tr_norms, bins=bins, alpha=0.55, label=f"TR (n={len(tr_norms)})", color="#1f77b4", density=True)
        ax.hist(
            emb_norms,
            bins=bins,
            alpha=0.55,
            label=f"EmbedPred (n={len(emb_norms)})",
            color="#ff7f0e",
            density=True,
        )
        ax.set_title("||anchor_vec||_2 on full val set")
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No checkpoint data\n(use without --skip_embedding)", ha="center", va="center")
        ax.set_title("||anchor_vec||_2 on full val set")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Boxplot by s2f class
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for name, key, color in (
        ("TR", "tr", "#1f77b4"),
        ("EmbedPred", "embed", "#ff7f0e"),
    ):
        for cls in range(NUM_CLASSES):
            norms = norm_stats[key]["norms_by_s2f"].get(str(cls), [])
            if norms:
                data.append(norms)
                labels.append(f"{name}\n{CLASS_NAMES[cls]}")
    if data:
        try:
            bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True)
        except TypeError:
            bp = ax2.boxplot(data, labels=labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#1f77b4" if i < NUM_CLASSES else "#ff7f0e")
            patch.set_alpha(0.6)
    ax2.set_title("anchor_vec L2 norm by s2f class (val set)")
    ax2.set_ylabel("||anchor_vec||_2")
    ax2.tick_params(axis="x", labelsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    box_path = out_path.parent / "anchor_vec_norm_by_s2f_boxplot.png"
    fig2.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"])
    else:
        model.load_state_dict(raw)


def _build_val_dataset(seed: int) -> tuple[np.ndarray, EHRAnchorEmbedDataset, Subset]:
    enr = ENRICHED_CSV if Path(ENRICHED_CSV).is_file() else None
    full_ds = EHRAnchorEmbedDataset(
        anchor_source_csv=P2F_OR_S2F_CSV,
        history_csv=P2F_OR_S2F_CSV,
        schema_csv=SCHEMA_CSV,
        enriched_csv=enr,
        lookback_min_hours=LOOKBACK_MIN_HOURS,
        lookback_max_hours=LOOKBACK_MAX_HOURS,
        include_anchor_row=True,
    )
    y = _stratify_labels_from_dataset(full_ds)
    test_split = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    _, idx_val, _ = stratified_train_val_test_indices(y, TRAIN_SPLIT, VAL_SPLIT, test_split, seed)
    val_ds = make_subset(full_ds, idx_val)
    return idx_val, full_ds, val_ds


def _val_label_arrays(val_ds: Subset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = val_ds.dataset
    indices = np.asarray(val_ds.indices, dtype=np.int64)
    s2f = np.array(
        [int(base.anchor_s2f_cls[i]) if base.anchor_has_s2f[i] else -1 for i in indices],
        dtype=np.int64,
    )
    p2f = np.array(
        [int(base.anchor_p2f_cls[i]) if base.anchor_has_p2f[i] else -1 for i in indices],
        dtype=np.int64,
    )
    p2f_ok = np.array([bool(base.anchor_has_p2f[i]) for i in indices], dtype=bool)
    return s2f, p2f, p2f_ok


def _norm_stats_from_norms(
    all_norms: np.ndarray,
    s2f_labels: np.ndarray,
) -> dict[str, Any]:
    norms_by_s2f: dict[str, list[float]] = {}
    valid_s2f = s2f_labels >= 0
    for cls in range(NUM_CLASSES):
        mask = valid_s2f & (s2f_labels == cls)
        norms_by_s2f[str(cls)] = all_norms[mask].tolist()
    return {
        "mean": float(all_norms.mean()),
        "std": float(all_norms.std()),
        "p5": float(np.percentile(all_norms, 5)),
        "p95": float(np.percentile(all_norms, 95)),
        "all_norms": all_norms,
        "norms_by_s2f": norms_by_s2f,
    }


def _build_model(
    model_cls,
    input_dim: int,
    ckpt_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = model_cls(
        input_dim=input_dim,
        embed_dim=EMBED_DIM,
        d_model=D_MODEL,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        head_dropout=HEAD_DROPOUT,
        num_classes=NUM_CLASSES,
        max_seq_length=MAX_SEQ_LENGTH,
        anchor_pool=ANCHOR_POOL,
    ).to(device)
    _load_checkpoint(model, ckpt_path)
    model.eval()
    return model


@torch.no_grad()
def _run_val_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_local_indices: Optional[set[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    all_norms: list[float] = []
    s2f_labels: list[np.ndarray] = []
    p2f_labels: list[np.ndarray] = []
    p2f_valid: list[np.ndarray] = []
    collected: dict[int, np.ndarray] = {}
    offset = 0

    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        anchor_vec, _ = model.forward_transformer(b["ehr_seq"], b["ehr_mask"])
        av = anchor_vec.cpu().numpy()
        norms = np.linalg.norm(av, axis=1)
        all_norms.extend(norms.tolist())
        s2f_labels.append(b["anchor_s2f"].cpu().numpy())
        p2f_labels.append(b["anchor_p2f"].cpu().numpy())
        p2f_valid.append(b["anchor_has_p2f"].cpu().numpy())

        if collect_local_indices is not None:
            bsz = av.shape[0]
            for j in range(bsz):
                local_i = offset + j
                if local_i in collect_local_indices:
                    collected[local_i] = av[j]
            offset += bsz

    return (
        np.asarray(all_norms, dtype=np.float64),
        np.concatenate(s2f_labels, axis=0),
        np.concatenate(p2f_labels, axis=0),
        np.concatenate(p2f_valid, axis=0),
        collected,
    )


def _stratified_sample_indices(
    s2f_labels: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    valid = s2f_labels >= 0
    idx_all = np.where(valid)[0]
    y = s2f_labels[idx_all]
    if len(idx_all) <= n:
        return idx_all
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    sub_idx, _ = next(splitter.split(idx_all, y))
    return idx_all[sub_idx]


def _plot_tsne_panel(
    vecs_tr: np.ndarray,
    vecs_emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    seed: int,
    perplexity: float,
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12)
    meta = {}

    for ax, vecs, name in (
        (axes[0], vecs_tr, "EHREncoderTransformer"),
        (axes[1], vecs_emb, "EHREncoderTransformerEmbedPred"),
    ):
        n = vecs.shape[0]
        perp = min(perplexity, max(5.0, (n - 1) / 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        xy = tsne.fit_transform(vecs)
        meta[name] = {"n": n, "perplexity": perp}
        for cls in range(NUM_CLASSES):
            mask = labels == cls
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=CLASS_COLORS[cls],
                label=f"{CLASS_NAMES[cls]} (n={int(mask.sum())})",
                s=8,
                alpha=0.6,
                edgecolors="none",
            )
        ax.set_title(name)
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return meta


def run_embedding_analysis(
    ckpt_tr: Path,
    ckpt_embed: Path,
    out_dir: Path,
    tsne_n: int,
    seed: int,
    batch_size: int,
    perplexity: float,
    skip_tsne: bool,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding analysis on {device}")

    idx_val, full_ds, val_ds = _build_val_dataset(seed)
    s2f_val, p2f_val, p2f_ok_val = _val_label_arrays(val_ds)
    print(f"  Val set: n={len(val_ds)}")

    collect_indices: set[int] = set()
    if not skip_tsne:
        s2f_sample = _stratified_sample_indices(s2f_val, tsne_n, seed)
        collect_indices.update(int(i) for i in s2f_sample)
        p2f_idx = np.where(p2f_ok_val & (p2f_val >= 0))[0]
        if len(p2f_idx) > tsne_n:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=tsne_n, random_state=seed)
            sub_idx, _ = next(splitter.split(p2f_idx, p2f_val[p2f_idx]))
            p2f_idx = p2f_idx[sub_idx]
        collect_indices.update(int(i) for i in p2f_idx)

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anchor_batch,
    )

    print("  Running TR inference on val set...")
    model_tr = _build_model(EHREncoderTransformer, full_ds.input_dim, ckpt_tr, device)
    norms_tr, s2f_tr, p2f_tr, p2f_ok_tr, collected_tr = _run_val_inference(
        model_tr, loader, device, collect_indices or None
    )
    del model_tr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  Running EmbedPred inference on val set...")
    model_emb = _build_model(EHREncoderTransformerEmbedPred, full_ds.input_dim, ckpt_embed, device)
    norms_emb, s2f_emb, p2f_emb, _, collected_emb = _run_val_inference(
        model_emb, loader, device, collect_indices or None
    )
    del model_emb
    if device.type == "cuda":
        torch.cuda.empty_cache()

    norm_stats_full = {
        "tr": _norm_stats_from_norms(norms_tr, s2f_tr),
        "embed": _norm_stats_from_norms(norms_emb, s2f_emb),
    }

    tsne_meta: dict[str, Any] = {}
    if not skip_tsne:
        s2f_sample = _stratified_sample_indices(s2f_val, tsne_n, seed)
        vecs_tr_sub = np.stack([collected_tr[int(i)] for i in s2f_sample], axis=0)
        vecs_emb_sub = np.stack([collected_emb[int(i)] for i in s2f_sample], axis=0)
        labels_sub = s2f_val[s2f_sample]

        tsne_meta["s2f"] = _plot_tsne_panel(
            vecs_tr_sub,
            vecs_emb_sub,
            labels_sub,
            f"t-SNE anchor_vec — s2f (n={len(s2f_sample)}, val stratified)",
            out_dir / "tsne_anchor_vec_s2f.png",
            seed,
            perplexity,
        )

        p2f_idx = np.where(p2f_ok_tr & (p2f_tr >= 0))[0]
        if len(p2f_idx) > tsne_n:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=tsne_n, random_state=seed)
            sub_idx, _ = next(splitter.split(p2f_idx, p2f_tr[p2f_idx]))
            p2f_idx = p2f_idx[sub_idx]

        vecs_tr_p = np.stack([collected_tr[int(i)] for i in p2f_idx], axis=0)
        vecs_emb_p = np.stack([collected_emb[int(i)] for i in p2f_idx], axis=0)
        tsne_meta["p2f"] = _plot_tsne_panel(
            vecs_tr_p,
            vecs_emb_p,
            p2f_tr[p2f_idx],
            f"t-SNE anchor_vec — p2f (n={len(p2f_idx)}, val)",
            out_dir / "tsne_anchor_vec_p2f.png",
            seed,
            perplexity,
        )

    norm_stats_json = {}
    for key in ("tr", "embed"):
        ns = norm_stats_full[key]
        norms_by_s2f_json = {}
        for cls, vals in ns["norms_by_s2f"].items():
            norms_by_s2f_json[cls] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
            }
        norm_stats_json[key] = {
            "mean": ns["mean"],
            "std": ns["std"],
            "p5": ns["p5"],
            "p95": ns["p95"],
            "norms_by_s2f": norms_by_s2f_json,
        }

    return {
        "norm_stats": norm_stats_json,
        "norm_stats_full": norm_stats_full,
        "tsne": tsne_meta,
        "val_n": len(idx_val),
    }


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Analyze EHR encoder training runs")
    p.add_argument("--log_tr", type=str, default="logs/ehr-enc-tr-47410738.out")
    p.add_argument("--log_embed", type=str, default="logs/ehr-enc-embed-47414328.out")
    p.add_argument("--ckpt_tr", type=str, default="EHREncoderTransformer/output/best.pt")
    p.add_argument("--ckpt_embed", type=str, default="EHREncoderTransformerEmbedPred/output/best.pt")
    p.add_argument("--out_dir", type=str, default="figures/ehr_enc_47410738_47414328")
    p.add_argument("--tsne_n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--skip_tsne", action="store_true")
    p.add_argument("--skip_embedding", action="store_true", help="Only parse logs and plot curves")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_tr_path = PROJECT_ROOT / args.log_tr
    log_embed_path = PROJECT_ROOT / args.log_embed
    ckpt_tr = PROJECT_ROOT / args.ckpt_tr
    ckpt_embed = PROJECT_ROOT / args.ckpt_embed

    print(f"Parsing {log_tr_path}")
    tr_log = parse_tr_log(log_tr_path)
    print(f"Parsing {log_embed_path}")
    embed_log = parse_embed_log(log_embed_path)
    print(f"  TR epochs: {len(tr_log['epochs'])}, best={tr_log.get('best_epoch')}")
    print(f"  EmbedPred epochs: {len(embed_log['epochs'])}, best={embed_log.get('best_epoch')}")

    plot_training_curves(tr_log, embed_log, out_dir / "training_curves.png")
    print(f"Wrote {out_dir / 'training_curves.png'}")

    embed_result: dict[str, Any] = {}
    norm_stats_for_plot: dict[str, Any] = {
        "tr": {"all_norms": np.array([]), "norms_by_s2f": {}},
        "embed": {"all_norms": np.array([]), "norms_by_s2f": {}},
    }

    if not args.skip_embedding:
        if not ckpt_tr.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_tr}")
        if not ckpt_embed.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_embed}")

        embed_result = run_embedding_analysis(
            ckpt_tr,
            ckpt_embed,
            out_dir,
            args.tsne_n,
            args.seed,
            args.batch_size,
            args.perplexity,
            args.skip_tsne,
        )
        norm_stats_for_plot = embed_result.pop("norm_stats_full")

    plot_norm_curves(tr_log, embed_log, norm_stats_for_plot, out_dir / "anchor_vec_norm_curves.png")
    print(f"Wrote {out_dir / 'anchor_vec_norm_curves.png'}")

    metrics = {
        "tr_log": {k: v for k, v in tr_log.items() if k != "epochs"},
        "embed_log": {k: v for k, v in embed_log.items() if k != "epochs"},
        "tr_epochs": tr_log["epochs"],
        "embed_epochs": embed_log["epochs"],
        "embedding_analysis": embed_result,
    }
    metrics_path = out_dir / "metrics.json"

    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"Wrote {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
