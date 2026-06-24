#!/usr/bin/env python3
"""t-SNE of EHREncoderTransformer anchor_vec: Fix-C vs baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
_EXP = ROOT / "EHREncoderTransformer"
_OUT = Path(__file__).resolve().parent / "ehr_tr_fix_runs"

for _p in (ROOT, ROOT / "BaselineExperiment", ROOT / "EHRTrend", _EXP):
    sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from config import (  # noqa: E402
    ANCHOR_POOL,
    D_MODEL,
    DROPOUT,
    EMBED_DIM,
    ENRICHED_CSV,
    HEAD_DROPOUT,
    INCLUDE_ANCHOR_ROW,
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
from ehr_symile_dataset import EHRNextStepDatasetSymile  # noqa: E402
from model import EHREncoderTransformer  # noqa: E402
from train import _stratify_labels_from_dataset, collate_anchor_batch  # noqa: E402

CLASS_NAMES = [f"change_{i}" for i in range(NUM_CLASSES)]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
FIX_C_CKPT = _EXP / "output_fixC/best_acc.pt"


def _resolve_baseline_ckpt() -> Path:
    for p in (
        _EXP / "output/best_loss.pt",
        _EXP / "output/best_acc.pt",
        _EXP / "output/best.pt",
        _EXP / "output/last.pt",
    ):
        if p.is_file():
            return p
    return _EXP / "output/best.pt"


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"])
    else:
        model.load_state_dict(raw)


def _build_val_loader() -> tuple:
    enr = ENRICHED_CSV if Path(ENRICHED_CSV).is_file() else None
    full_ds = EHRNextStepDatasetSymile(
        anchor_source_csv=P2F_OR_S2F_CSV,
        history_csv=P2F_OR_S2F_CSV,
        schema_csv=SCHEMA_CSV,
        enriched_csv=enr,
        lookback_min_hours=LOOKBACK_MIN_HOURS,
        lookback_max_hours=LOOKBACK_MAX_HOURS,
        include_anchor_row=INCLUDE_ANCHOR_ROW,
    )
    y = _stratify_labels_from_dataset(full_ds)
    test_split = 1.0 - TRAIN_SPLIT - VAL_SPLIT
    idx_train, idx_val, _ = stratified_train_val_test_indices(y, TRAIN_SPLIT, VAL_SPLIT, test_split, SEED)
    full_ds.fit_preprocess(idx_train)
    val_ds = make_subset(full_ds, idx_val)
    base = val_ds.dataset
    indices = np.asarray(val_ds.indices, dtype=np.int64)
    s2f = np.array(
        [int(base.anchor_s2f_cls[i]) if base.anchor_has_s2f[i] else -1 for i in indices], dtype=np.int64
    )
    p2f = np.array(
        [int(base.anchor_p2f_cls[i]) if base.anchor_has_p2f[i] else -1 for i in indices], dtype=np.int64
    )
    p2f_ok = np.array([bool(base.anchor_has_p2f[i]) for i in indices], dtype=bool)
    return val_ds, full_ds, s2f, p2f, p2f_ok, base.input_dim


def _stratified_sample(labels: np.ndarray, n: int, seed: int, valid_mask=None) -> np.ndarray:
    if valid_mask is None:
        valid_mask = labels >= 0
    idx_all = np.where(valid_mask)[0]
    if len(idx_all) <= n:
        return idx_all
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    sub_idx, _ = next(splitter.split(idx_all, labels[idx_all]))
    return idx_all[sub_idx]


def _run_tsne(vecs: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    perp = min(perplexity, max(5.0, (vecs.shape[0] - 1) / 3))
    return TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto").fit_transform(vecs)


def _scatter(ax, xy, labels, title: str) -> None:
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], c=CLASS_COLORS[cls], label=f"{CLASS_NAMES[cls]} (n={int(mask.sum())})", s=10, alpha=0.55, edgecolors="none")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


@torch.no_grad()
def _infer(
    ckpt: Path,
    input_dim: int,
    full_ds: EHRNextStepDatasetSymile,
    union_global: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = EHREncoderTransformer(
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
    _load_checkpoint(model, ckpt)
    model.eval()
    loader = DataLoader(
        make_subset(full_ds, union_global),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anchor_batch,
    )
    print(f"  Inference n={len(union_global)}: {ckpt.name}")
    vecs = []
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        anchor_vec, _ = model.forward_transformer(b["ehr_seq"], b["ehr_mask"])
        vecs.append(anchor_vec.cpu().numpy())
    return np.concatenate(vecs, axis=0)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=FIX_C_CKPT)
    p.add_argument("--baseline_ckpt", type=Path, default=None)
    p.add_argument("--tsne_n", type=int, default=3000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", type=Path, default=_OUT)
    args = p.parse_args()
    if args.baseline_ckpt is None:
        args.baseline_ckpt = _resolve_baseline_ckpt()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Baseline ckpt: {args.baseline_ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"t-SNE EHR-TR on {device}")

    val_ds, full_ds, s2f_val, p2f_val, p2f_ok, input_dim = _build_val_loader()
    s2f_local = _stratified_sample(s2f_val, args.tsne_n, args.seed)
    p2f_local = _stratified_sample(p2f_val, args.tsne_n, args.seed, valid_mask=p2f_ok & (p2f_val >= 0))
    val_indices = np.asarray(val_ds.indices, dtype=np.int64)
    union_local = np.array(sorted(set(s2f_local.tolist()) | set(p2f_local.tolist())), dtype=np.int64)
    union_global = val_indices[union_local]
    local_to_pos = {int(loc): i for i, loc in enumerate(union_local)}

    vecs_fix = _infer(args.ckpt, input_dim, full_ds, union_global, device, args.batch_size)
    vecs_s2f = np.stack([vecs_fix[local_to_pos[int(loc)]] for loc in s2f_local])
    labels_s2f = s2f_val[s2f_local]
    vecs_p2f = np.stack([vecs_fix[local_to_pos[int(loc)]] for loc in p2f_local])
    labels_p2f = p2f_val[p2f_local]

    meta = {"ckpt": str(args.ckpt), "seed": args.seed}

    for head, vecs, labels in (("s2f", vecs_s2f, labels_s2f), ("p2f", vecs_p2f, labels_p2f)):
        xy = _run_tsne(vecs, args.seed, args.perplexity)
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(ax, xy, labels, f"Fix-C — t-SNE anchor_vec ({head})")
        fig.savefig(args.out_dir / f"tsne_fixC_{head}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, vecs, labels, head in zip(axes, [vecs_s2f, vecs_p2f], [labels_s2f, labels_p2f], ["s2f", "p2f"]):
        xy = _run_tsne(vecs, args.seed, args.perplexity)
        _scatter(ax, xy, labels, head.upper())
    fig.suptitle("EHREncoderTransformer Fix-C — t-SNE anchor_vec", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_dir / "tsne_fixC_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if args.baseline_ckpt.is_file():
        vecs_base = _infer(args.baseline_ckpt, input_dim, full_ds, union_global, device, args.batch_size)
        vecs_s2f_b = np.stack([vecs_base[local_to_pos[int(loc)]] for loc in s2f_local])
        vecs_p2f_b = np.stack([vecs_base[local_to_pos[int(loc)]] for loc in p2f_local])
        for head, va, vb, la in (
            ("s2f", vecs_s2f_b, vecs_s2f, labels_s2f),
            ("p2f", vecs_p2f_b, vecs_p2f, labels_p2f),
        ):
            combined = np.concatenate([va, vb], axis=0)
            xy = _run_tsne(combined, args.seed, args.perplexity)
            n_a = len(va)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            _scatter(axes[0], xy[:n_a], la, "Baseline")
            _scatter(axes[1], xy[n_a:], la, "Fix-C")
            fig.suptitle(f"Baseline vs Fix-C — t-SNE ({head})")
            fig.tight_layout()
            fig.savefig(args.out_dir / f"tsne_baseline_vs_fixC_{head}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        meta["baseline_ckpt"] = str(args.baseline_ckpt)

    with open(args.out_dir / "tsne_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote t-SNE to {args.out_dir}")


if __name__ == "__main__":
    main()
