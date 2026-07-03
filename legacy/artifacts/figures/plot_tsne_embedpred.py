#!/usr/bin/env python3
"""t-SNE of anchor_vec embeddings from EmbedPred Exp C (val set, stratified sample)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
_EXP = ROOT / "EHREncoderTransformerEmbedPred"
_OUT = Path(__file__).resolve().parent / "ehr_embedpred_exp_runs"

for _p in (ROOT, ROOT / "BaselineExperiment", ROOT / "EHRTrend", _EXP):
    sys.path.insert(0, str(_p))

from classification_utils import make_subset, stratified_train_val_test_indices  # noqa: E402
from anchor_embed_dataset import EHRAnchorEmbedDataset  # noqa: E402
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
from model import EHREncoderTransformerEmbedPred  # noqa: E402
from train import _stratify_labels_from_dataset, collate_anchor_batch  # noqa: E402

CLASS_NAMES = [f"change_{i}" for i in range(NUM_CLASSES)]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
DEFAULT_CKPT = _EXP / "output_twophase_expC/finetune/best_acc.pt"
BASELINE_CKPT = _EXP / "output_twophase/finetune/best.pt"


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        model.load_state_dict(raw["model"])
    else:
        model.load_state_dict(raw)


def _build_val_loader(batch_size: int = 64) -> tuple[Subset, np.ndarray, np.ndarray, np.ndarray, int]:
    enr = ENRICHED_CSV if Path(ENRICHED_CSV).is_file() else None
    full_ds = EHRAnchorEmbedDataset(
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
    idx_train, idx_val, _ = stratified_train_val_test_indices(
        y, TRAIN_SPLIT, VAL_SPLIT, test_split, SEED
    )
    full_ds.fit_preprocess(idx_train)
    val_ds = make_subset(full_ds, idx_val)
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
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anchor_batch,
    )
    return val_ds, s2f, p2f, p2f_ok, base.input_dim


def _stratified_sample(labels: np.ndarray, n: int, seed: int, *, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if valid_mask is None:
        valid_mask = labels >= 0
    idx_all = np.where(valid_mask)[0]
    y = labels[idx_all]
    if len(idx_all) <= n:
        return idx_all
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    sub_idx, _ = next(splitter.split(idx_all, y))
    return idx_all[sub_idx]


@torch.no_grad()
def _collect_anchor_vecs(
    model: EHREncoderTransformerEmbedPred,
    loader: DataLoader,
    device: torch.device,
    sample_indices: set[int],
) -> dict[int, np.ndarray]:
    collected: dict[int, np.ndarray] = {}
    offset = 0
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        anchor_vec, _ = model.forward_transformer(b["ehr_seq"], b["ehr_mask"])
        av = anchor_vec.cpu().numpy()
        bsz = av.shape[0]
        for j in range(bsz):
            local_i = offset + j
            if local_i in sample_indices:
                collected[local_i] = av[j]
        offset += bsz
    return collected


def _run_tsne(vecs: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    n = vecs.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(vecs)


def _scatter_tsne(ax, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=CLASS_COLORS[cls],
            label=f"{CLASS_NAMES[cls]} (n={int(mask.sum())})",
            s=10,
            alpha=0.55,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(True, alpha=0.2)


def plot_tsne_single(
    vecs: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    seed: int,
    perplexity: float,
) -> dict:
    xy = _run_tsne(vecs, seed, perplexity)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(title, fontsize=12)
    _scatter_tsne(ax, xy, labels, "anchor_vec (d_model=256)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"n": int(len(labels)), "perplexity": float(min(perplexity, max(5.0, (len(labels) - 1) / 3)))}


def plot_tsne_compare(
    vecs_a: np.ndarray,
    labels_a: np.ndarray,
    vecs_b: np.ndarray,
    labels_b: np.ndarray,
    name_a: str,
    name_b: str,
    head: str,
    out_path: Path,
    seed: int,
    perplexity: float,
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"t-SNE anchor_vec — {head.upper()} severity_change (val, stratified)", fontsize=12)
    meta = {}
    for ax, vecs, labels, name in (
        (axes[0], vecs_a, labels_a, name_a),
        (axes[1], vecs_b, labels_b, name_b),
    ):
        xy = _run_tsne(vecs, seed, perplexity)
        meta[name] = {"n": int(len(labels)), "perplexity": float(min(perplexity, max(5.0, (len(labels) - 1) / 3)))}
        _scatter_tsne(ax, xy, labels, name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return meta


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--baseline_ckpt", type=Path, default=BASELINE_CKPT)
    p.add_argument("--compare_baseline", action="store_true")
    p.add_argument("--tsne_n", type=int, default=3000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", type=Path, default=_OUT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"t-SNE on {device}")

    val_ds, s2f_val, p2f_val, p2f_ok, input_dim = _build_val_loader(args.batch_size)
    s2f_local = _stratified_sample(s2f_val, args.tsne_n, args.seed)
    p2f_local = _stratified_sample(p2f_val, args.tsne_n, args.seed, valid_mask=p2f_ok & (p2f_val >= 0))
    # Map val-local indices -> global anchor indices for Subset loader
    val_indices = np.asarray(val_ds.indices, dtype=np.int64)
    s2f_global = val_indices[s2f_local]
    p2f_global = val_indices[p2f_local]
    union_local = np.array(sorted(set(s2f_local.tolist()) | set(p2f_local.tolist())), dtype=np.int64)
    union_global = val_indices[union_local]
    local_to_pos = {int(loc): i for i, loc in enumerate(union_local)}

    subset_loader = DataLoader(
        make_subset(val_ds.dataset, union_global),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_anchor_batch,
    )

    @torch.no_grad()
    def _infer(ckpt: Path) -> np.ndarray:
        model = EHREncoderTransformerEmbedPred(
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
        print(f"  Inference on n={len(union_global)} anchors: {ckpt.name}")
        all_vecs: list[np.ndarray] = []
        for batch in subset_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            anchor_vec, _ = model.forward_transformer(b["ehr_seq"], b["ehr_mask"])
            all_vecs.append(anchor_vec.detach().cpu().numpy())
        return np.concatenate(all_vecs, axis=0)

    print("Collecting anchor_vec (Exp C)...")
    vecs_all = _infer(args.ckpt)
    vecs_s2f = np.stack([vecs_all[local_to_pos[int(loc)]] for loc in s2f_local], axis=0)
    labels_s2f = s2f_val[s2f_local]
    vecs_p2f = np.stack([vecs_all[local_to_pos[int(loc)]] for loc in p2f_local], axis=0)
    labels_p2f = p2f_val[p2f_local]

    meta: dict = {
        "ckpt": str(args.ckpt),
        "seed": args.seed,
        "tsne_n_s2f": int(len(s2f_local)),
        "tsne_n_p2f": int(len(p2f_local)),
    }

    print("Running t-SNE (s2f)...")
    meta["s2f_expC"] = plot_tsne_single(
        vecs_s2f,
        labels_s2f,
        f"Exp C — t-SNE anchor_vec by s2f label (n={len(s2f_local)})",
        args.out_dir / "tsne_expC_s2f.png",
        args.seed,
        args.perplexity,
    )

    print("Running t-SNE (p2f)...")
    meta["p2f_expC"] = plot_tsne_single(
        vecs_p2f,
        labels_p2f,
        f"Exp C — t-SNE anchor_vec by p2f label (n={len(p2f_local)})",
        args.out_dir / "tsne_expC_p2f.png",
        args.seed,
        args.perplexity,
    )

    # Combined 1x2 panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Exp C (best) — t-SNE of transformer anchor_vec on val set", fontsize=12)
    for ax, vecs, labels, head in (
        (axes[0], vecs_s2f, labels_s2f, "s2f"),
        (axes[1], vecs_p2f, labels_p2f, "p2f"),
    ):
        xy = _run_tsne(vecs, args.seed, args.perplexity)
        _scatter_tsne(ax, xy, labels, f"{head.upper()} severity_change")
    fig.tight_layout()
    fig.savefig(args.out_dir / "tsne_expC_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if args.compare_baseline and args.baseline_ckpt.is_file():
        print("Collecting anchor_vec (baseline)...")
        vecs_base = _infer(args.baseline_ckpt)
        vecs_s2f_b = np.stack([vecs_base[local_to_pos[int(loc)]] for loc in s2f_local], axis=0)
        vecs_p2f_b = np.stack([vecs_base[local_to_pos[int(loc)]] for loc in p2f_local], axis=0)
        print("Running t-SNE compare (s2f)...")
        meta["compare_s2f"] = plot_tsne_compare(
            vecs_s2f_b,
            labels_s2f,
            vecs_s2f,
            labels_s2f,
            "Baseline",
            "Exp C",
            "s2f",
            args.out_dir / "tsne_baseline_vs_expC_s2f.png",
            args.seed,
            args.perplexity,
        )
        print("Running t-SNE compare (p2f)...")
        meta["compare_p2f"] = plot_tsne_compare(
            vecs_p2f_b,
            labels_p2f,
            vecs_p2f,
            labels_p2f,
            "Baseline",
            "Exp C",
            "p2f",
            args.out_dir / "tsne_baseline_vs_expC_p2f.png",
            args.seed,
            args.perplexity,
        )

    with open(args.out_dir / "tsne_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote t-SNE figures to {args.out_dir}")
    for name in (
        "tsne_expC_s2f.png",
        "tsne_expC_p2f.png",
        "tsne_expC_combined.png",
        "tsne_baseline_vs_expC_s2f.png",
        "tsne_baseline_vs_expC_p2f.png",
    ):
        path = args.out_dir / name
        if path.is_file():
            print(f"  {name}")


if __name__ == "__main__":
    main()
