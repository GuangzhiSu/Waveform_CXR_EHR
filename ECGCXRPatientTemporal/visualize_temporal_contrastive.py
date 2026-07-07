#!/usr/bin/env python3
"""Visual diagnostics for temporal contrastive ECG-CXR experiments.

The default input is the temporal-loss-only frontal run:

    artifacts/outputs/frontal_methods/weighted_temporal_only_b32

It reads per-epoch ``results.json`` histories, optional ``train_dynamics.csv``
files, and (unless disabled) the best checkpoints to build separation/rank
diagnostics on the held-out split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
sys.path.insert(0, str(EXP_DIR))

try:
    import env_setup  # noqa: F401
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config as C  # noqa: E402
from experiments import ExperimentSpec  # noqa: E402
from runtime import get_device  # noqa: E402
from staged_dataset import StagedData, StagedDataset, collate_fn  # noqa: E402
from staged_model import StagedModel  # noqa: E402


DEFAULT_RUN_DIR = EXP_DIR / "artifacts" / "outputs" / "frontal_methods" / "weighted_temporal_only_b32"
DEFAULT_PAIRS = EXP_DIR / "artifacts" / "cache" / "frontal_t2_24h_ecg12h" / "patient_temporal_pairs.json"
DEFAULT_METHODS = [
    ("exp4b_cxr_only", "CXR-only"),
    ("exp5c_weighted_attn_pool", "Real ECG"),
    ("exp5c_weighted_attn_pool_shuffled", "Shuffled ECG"),
    ("exp5c_weighted_attn_pool_zeroed", "Zeroed ECG"),
]
COLORS = {
    "exp4b_cxr_only": "#4C78A8",
    "exp5c_weighted_attn_pool": "#54A24B",
    "exp5c_weighted_attn_pool_shuffled": "#F58518",
    "exp5c_weighted_attn_pool_zeroed": "#B279A2",
}


def _finite(v: Any, default: float = np.nan) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _nested(d: dict, *keys: str, default: float = np.nan) -> float:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return _finite(cur, default)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rolling(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=max(2, window // 10)).mean()


def _method_specs(raw: list[str] | None, run_dir: Path) -> list[dict]:
    if not raw:
        pairs = DEFAULT_METHODS
    else:
        pairs = []
        for token in raw:
            if "=" in token:
                name, label = token.split("=", 1)
            elif ":" in token:
                name, label = token.split(":", 1)
            else:
                name, label = token, token
            pairs.append((name.strip(), label.strip()))

    methods = []
    for name, label in pairs:
        out_dir = run_dir / name
        result_path = out_dir / "results.json"
        if not result_path.is_file():
            print(f"WARNING: skipping missing run {result_path}", file=sys.stderr)
            continue
        result = _load_json(result_path)
        methods.append(
            {
                "name": name,
                "label": label,
                "dir": out_dir,
                "result_path": result_path,
                "result": result,
                "color": COLORS.get(name, "#666666"),
            }
        )
    if not methods:
        raise SystemExit(f"No valid runs found under {run_dir}")
    return methods


def _test_metrics_table(methods: list[dict]) -> pd.DataFrame:
    rows = []
    for m in methods:
        res = m["result"]
        rows.append(
            {
                "method": m["name"],
                "label": m["label"],
                "best_epoch": res.get("best_epoch"),
                "best_val_monitor": res.get("best_val_monitor"),
                "test_temporal_R@1": _nested(res, "test", "temporal", "temporal_recall@1"),
                "test_temporal_R@5": _nested(res, "test", "temporal", "temporal_recall@5"),
                "test_temporal_MRR": _nested(res, "test", "temporal", "temporal_mrr"),
                "test_temporal_n": _nested(res, "test", "temporal", "n_queries"),
                "test_cross_R@1": _nested(res, "test", "cross_patient", "recall@1"),
                "test_cross_R@5": _nested(res, "test", "cross_patient", "recall@5"),
                "test_cross_MRR": _nested(res, "test", "cross_patient", "mrr"),
                "test_cross_median_rank": _nested(res, "test", "cross_patient", "median_rank"),
            }
        )
    return pd.DataFrame(rows)


def _epoch_frame(methods: list[dict]) -> pd.DataFrame:
    rows = []
    for m in methods:
        for item in m["result"].get("history", []):
            train = item.get("train", {})
            val = item.get("val", {})
            rows.append(
                {
                    "method": m["name"],
                    "label": m["label"],
                    "epoch": int(item.get("epoch", 0)),
                    "train_loss": _finite(train.get("loss")),
                    "train_temporal_loss": _finite(train.get("temporal_loss")),
                    "train_cross_loss": _finite(train.get("cross_patient_loss")),
                    "train_temporal_batch_R@1": _finite(train.get("temporal_batch_top1")),
                    "train_temporal_batch_R@5": _finite(train.get("temporal_batch_top5")),
                    "train_cross_batch_R@1": _finite(train.get("cross_patient_batch_top1")),
                    "val_temporal_R@1": _nested(val, "temporal", "temporal_recall@1"),
                    "val_temporal_MRR": _nested(val, "temporal", "temporal_mrr"),
                    "val_cross_R@1": _nested(val, "cross_patient", "recall@1"),
                    "val_cross_MRR": _nested(val, "cross_patient", "mrr"),
                    "monitor": _finite(item.get("monitor")),
                    "temperature": _finite(item.get("temperature")),
                }
            )
    return pd.DataFrame(rows)


def plot_test_metrics(df: pd.DataFrame, methods: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    metric_pairs = [("test_temporal_R@1", "Temporal Recall@1"), ("test_temporal_MRR", "Temporal MRR")]
    labels = [m["label"] for m in methods]
    colors = [m["color"] for m in methods]

    for ax, (col, title) in zip(axes, metric_pairs):
        vals = [float(df.loc[df["label"] == label, col].iloc[0]) for label in labels]
        bars = ax.bar(np.arange(len(vals)), vals, color=colors, width=0.72)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, 0.82)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Test score")
    fig.suptitle("Temporal loss only: held-out within-patient retrieval", fontsize=12, y=1.03)
    _savefig(fig, out_path)


def plot_epoch_curves(df: pd.DataFrame, methods: list[dict], out_path: Path) -> None:
    panels = [
        ("val_temporal_R@1", "Validation temporal Recall@1", "Recall@1"),
        ("val_temporal_MRR", "Validation temporal MRR", "MRR"),
        ("train_temporal_loss", "Train temporal InfoNCE loss", "Loss"),
        ("train_temporal_batch_R@1", "Train in-batch temporal Recall@1", "Batch Recall@1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    axes = axes.ravel()
    for ax, (col, title, ylabel) in zip(axes, panels):
        for m in methods:
            sub = df[df["method"] == m["name"]].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[col], color=m["color"], linewidth=2.0, label=m["label"])
            best_epoch = m["result"].get("best_epoch")
            if best_epoch is not None and best_epoch in set(sub["epoch"]):
                best_val = sub.loc[sub["epoch"] == best_epoch, col].iloc[0]
                if np.isfinite(best_val):
                    ax.scatter([best_epoch], [best_val], color=m["color"], s=44,
                               marker="*", edgecolor="black", linewidth=0.35, zorder=5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.7)
    axes[-2].set_xlabel("Epoch")
    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_path)


def plot_cross_epoch_curves(df: pd.DataFrame, methods: list[dict], out_path: Path) -> None:
    panels = [
        ("val_cross_R@1", "Validation cross-patient Recall@1", "Recall@1"),
        ("val_cross_MRR", "Validation cross-patient MRR", "MRR"),
        ("train_cross_batch_R@1", "Train in-batch cross-patient Recall@1", "Batch Recall@1"),
        ("train_cross_loss", "Cross-patient loss term (not optimized here)", "Loss term"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    axes = axes.ravel()
    for ax, (col, title, ylabel) in zip(axes, panels):
        for m in methods:
            sub = df[df["method"] == m["name"]].sort_values("epoch")
            if not sub.empty:
                ax.plot(sub["epoch"], sub[col], color=m["color"], linewidth=2.0, label=m["label"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.7)
    axes[-2].set_xlabel("Epoch")
    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_path)


def plot_train_dynamics(methods: list[dict], out_path: Path, rolling_window: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True)
    plotted = False
    for m in methods:
        path = m["dir"] / "train_dynamics.csv"
        if not path.is_file():
            continue
        d = pd.read_csv(path)
        if d.empty:
            continue
        plotted = True
        x = d["global_step"]
        axes[0].plot(x, _rolling(d["train_temporal_R@1"], rolling_window),
                     color=m["color"], linewidth=1.9, label=m["label"])
        axes[1].plot(x, _rolling(d["temporal_loss"], rolling_window),
                     color=m["color"], linewidth=1.9, label=m["label"])
        axes[2].plot(x, _rolling(d["train_cross_patient_R@1"], rolling_window),
                     color=m["color"], linewidth=1.9, label=m["label"])
    if not plotted:
        plt.close(fig)
        return
    titles = [
        f"Rolling train temporal Recall@1 (window={rolling_window} logged steps)",
        f"Rolling train temporal loss (window={rolling_window} logged steps)",
        f"Rolling train cross-patient Recall@1 (window={rolling_window} logged steps)",
    ]
    ylabels = ["Batch Recall@1", "Loss", "Batch Recall@1"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linewidth=0.7)
    axes[-1].set_xlabel("Optimizer step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_path)


def _pairs_file(spec: ExperimentSpec, args: argparse.Namespace) -> str:
    return {
        "single": str(args.single_pairs),
        "seq_target": str(args.seq_target_pairs),
        "seq_t1": str(args.pairs),
    }[spec.pairs_kind]


def _load_staged_data(spec: ExperimentSpec, args: argparse.Namespace, cache: dict[str, StagedData]) -> StagedData:
    key = f"{spec.pairs_kind}:{_pairs_file(spec, args)}"
    if key not in cache:
        cache[key] = StagedData(
            pairs_json=_pairs_file(spec, args),
            kind=spec.data_kind(),
            cxr_emb_npy=str(args.cxr_emb),
            cxr_ids_json=str(args.cxr_ids),
            ecg_emb_npy=str(args.ecg_emb),
            ecg_ids_json=str(args.ecg_ids),
            seed=args.seed,
            train_split=C.TRAIN_SPLIT,
            val_split=C.VAL_SPLIT,
            test_split=C.TEST_SPLIT,
        )
    return cache[key]


def _dataset_for_split(spec: ExperimentSpec, data: StagedData, split: str, seed: int) -> StagedDataset:
    offset = {"train": 0, "val": 1, "test": 2}[split]
    return StagedDataset(
        data,
        data.split_indices[split],
        ecg_perturb=spec.ecg_perturb,
        seed=seed + offset,
    )


def _build_model(spec: ExperimentSpec, result: dict, data: StagedData, ckpt_path: Path,
                 device: torch.device) -> StagedModel:
    cfg = result.get("model_config", {})
    model = StagedModel(
        spec,
        cxr_dim=data.cxr_emb.shape[1],
        ecg_dim=data.ecg_emb.shape[1],
        proj_dim=int(cfg.get("proj_dim", C.PROJ_DIM)),
        cxr_proj_hidden=int(cfg.get("cxr_proj_hidden", C.CXR_PROJ_HIDDEN)),
        d_model=int(cfg.get("d_model", C.D_MODEL)),
        ecg_tx_layers=int(cfg.get("ecg_tx_layers", C.ECG_TX_LAYERS)),
        ecg_tx_heads=int(cfg.get("ecg_tx_heads", C.ECG_TX_HEADS)),
        ecg_tx_mlp_ratio=float(cfg.get("ecg_tx_mlp_ratio", C.ECG_TX_MLP_RATIO)),
        fusion_hidden=int(cfg.get("fusion_hidden", C.FUSION_HIDDEN)),
        time_emb_dim=int(cfg.get("time_emb_dim", C.TIME_EMB_DIM)),
        dropout=float(cfg.get("dropout", C.DROPOUT)),
        temperature=float(cfg.get("temperature", C.TEMPERATURE)),
        learnable_temperature=bool(cfg.get("learnable_temperature", C.LEARNABLE_TEMPERATURE)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _collect_queries(model: StagedModel, dataset: StagedDataset, device: torch.device,
                     batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    qs, pids, c2_rows = [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        q, _, _ = model.encode(b)
        qs.append(q.detach().float().cpu().numpy())
        pids.append(batch["patient_id"].numpy())
        c2_rows.append(batch["c2_row"].numpy())
    return np.concatenate(qs, axis=0), np.concatenate(pids), np.concatenate(c2_rows)


@torch.no_grad()
def _project_gallery(model: StagedModel, cxr_emb: np.ndarray, rows: np.ndarray,
                     device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(rows), batch_size):
        sel = rows[start:start + batch_size]
        x = torch.from_numpy(cxr_emb[sel].astype(np.float32)).to(device)
        chunks.append(model.cxr_proj(x).detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.proj_dim), dtype=np.float32)


def _rank(scores: np.ndarray, target_idx: int) -> int:
    return int((scores > scores[target_idx]).sum()) + 1


def _compute_model_diagnostics(method: dict, args: argparse.Namespace, data_cache: dict[str, StagedData],
                               device: torch.device) -> tuple[pd.DataFrame, dict]:
    spec = ExperimentSpec(**method["result"]["spec"])
    data = _load_staged_data(spec, args, data_cache)
    dataset = _dataset_for_split(spec, data, args.split, args.seed)
    ckpt_path = method["dir"] / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model = _build_model(spec, method["result"], data, ckpt_path, device)

    q, pids, c2_rows = _collect_queries(model, dataset, device, args.batch_size)
    gallery_rows, target_gidx = np.unique(c2_rows, return_inverse=True)
    gallery = _project_gallery(model, data.cxr_emb, gallery_rows, device, args.batch_size)

    gallery_patient = np.full(len(gallery_rows), -1, dtype=np.int64)
    for gi, pid in zip(target_gidx, pids):
        gallery_patient[int(gi)] = int(pid)

    sims = (q @ gallery.T).astype(np.float32)
    pat_to_gidx: dict[int, list[int]] = {}
    for gi, pid in enumerate(gallery_patient):
        pat_to_gidx.setdefault(int(pid), []).append(int(gi))

    rng = np.random.RandomState(args.seed)
    rows = []
    for i, pid in enumerate(pids):
        pid = int(pid)
        target = int(target_gidx[i])
        same_patient = np.asarray(pat_to_gidx.get(pid, []), dtype=np.int64)
        if same_patient.size < 2:
            continue
        same_neg = same_patient[same_patient != target]
        if same_neg.size == 0:
            continue
        cross_cand = np.flatnonzero(gallery_patient != pid)
        random_cross = np.nan
        hard_cross = np.nan
        if cross_cand.size:
            random_cross = float(sims[i, rng.choice(cross_cand)])
            hard_cross = float(np.max(sims[i, cross_cand]))
        sub_scores = sims[i, same_patient]
        target_pos = int(np.where(same_patient == target)[0][0])
        temporal_rank = int((sub_scores > sub_scores[target_pos]).sum()) + 1
        cross_rank = _rank(sims[i], target)
        pos_sim = float(sims[i, target])
        hard_temp = float(np.max(sims[i, same_neg]))
        rows.append(
            {
                "method": method["name"],
                "label": method["label"],
                "query_index": i,
                "patient_id": pid,
                "target_gallery_index": target,
                "target_cxr_row": int(gallery_rows[target]),
                "positive_similarity": pos_sim,
                "hard_temporal_negative_similarity": hard_temp,
                "mean_temporal_negative_similarity": float(np.mean(sims[i, same_neg])),
                "random_cross_patient_negative_similarity": random_cross,
                "hard_cross_patient_negative_similarity": hard_cross,
                "temporal_margin_vs_hard_negative": pos_sim - hard_temp,
                "temporal_rank": temporal_rank,
                "cross_patient_rank": cross_rank,
                "n_temporal_candidates": int(same_patient.size),
            }
        )

    context = {
        "q": q,
        "pids": pids,
        "c2_rows": c2_rows,
        "gallery": gallery,
        "gallery_rows": gallery_rows,
        "gallery_patient": gallery_patient,
        "target_gidx": target_gidx,
        "sims": sims,
        "pat_to_gidx": pat_to_gidx,
    }
    return pd.DataFrame(rows), context


def _diagnostic_summary(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, label), sub in diag.groupby(["method", "label"], sort=False):
        ranks = sub["temporal_rank"].to_numpy(dtype=float)
        margins = sub["temporal_margin_vs_hard_negative"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "label": label,
                "n_temporal_queries": int(len(sub)),
                "temporal_R@1_from_ranks": float(np.mean(ranks <= 1)) if len(ranks) else np.nan,
                "temporal_R@5_from_ranks": float(np.mean(ranks <= 5)) if len(ranks) else np.nan,
                "temporal_MRR_from_ranks": float(np.mean(1.0 / ranks)) if len(ranks) else np.nan,
                "median_temporal_rank": float(np.median(ranks)) if len(ranks) else np.nan,
                "mean_positive_similarity": float(sub["positive_similarity"].mean()),
                "mean_hard_temporal_negative_similarity": float(sub["hard_temporal_negative_similarity"].mean()),
                "mean_random_cross_negative_similarity": float(sub["random_cross_patient_negative_similarity"].mean()),
                "mean_margin_vs_hard_temporal_negative": float(np.mean(margins)) if len(margins) else np.nan,
                "fraction_positive_above_hard_temporal_negative": float(np.mean(margins > 0)) if len(margins) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_rank_cdf(diag: pd.DataFrame, methods: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.0))
    max_rank = int(min(20, max(2, np.nanmax(diag["temporal_rank"].to_numpy()))))
    xs = np.arange(1, max_rank + 1)
    for m in methods:
        sub = diag[diag["method"] == m["name"]]
        if sub.empty:
            continue
        ranks = sub["temporal_rank"].to_numpy(dtype=float)
        ys = np.asarray([(ranks <= x).mean() for x in xs])
        ax.step(xs, ys, where="post", color=m["color"], linewidth=2.2, label=m["label"])
        ax.scatter([1], [ys[0]], color=m["color"], s=38, zorder=4)
    ax.set_xlabel("Temporal rank threshold k")
    ax.set_ylabel("Fraction with true CXR_t2 rank <= k")
    ax.set_title("Within-patient temporal retrieval rank CDF")
    ax.set_ylim(0.35, 1.01)
    ax.set_xticks(xs)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    _savefig(fig, out_path)


def plot_similarity_separation(diag: pd.DataFrame, methods: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    series_names = [
        ("positive_similarity", "Positive", "#54A24B"),
        ("hard_temporal_negative_similarity", "Hard temporal neg", "#E45756"),
        ("random_cross_patient_negative_similarity", "Random cross neg", "#4C78A8"),
    ]
    positions = []
    xticks = []
    xticklabels = []
    for idx, m in enumerate(methods):
        sub = diag[diag["method"] == m["name"]]
        if sub.empty:
            continue
        base = idx * 4.0
        vals = [sub[col].dropna().to_numpy() for col, _, _ in series_names]
        pos = [base, base + 0.9, base + 1.8]
        violins = ax.violinplot(vals, positions=pos, widths=0.72, showmeans=False,
                                showmedians=True, showextrema=False)
        for body, (_, _, color) in zip(violins["bodies"], series_names):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        violins["cmedians"].set_color("black")
        violins["cmedians"].set_linewidth(1.2)
        positions.extend(pos)
        xticks.append(base + 0.9)
        xticklabels.append(m["label"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=18, ha="right")
    ax.set_ylabel("Cosine similarity in learned space")
    ax.set_title("Positive vs negative similarity distributions")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", alpha=0.55, label=lab)
                       for _, lab, c in series_names],
              frameon=False, loc="best")

    ax = axes[1]
    margin_vals = []
    labels = []
    colors = []
    for m in methods:
        sub = diag[diag["method"] == m["name"]]
        if sub.empty:
            continue
        margin_vals.append(sub["temporal_margin_vs_hard_negative"].dropna().to_numpy())
        labels.append(m["label"])
        colors.append(m["color"])
    bp = ax.boxplot(margin_vals, patch_artist=True, labels=labels, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Positive sim - hardest temporal negative sim")
    ax.set_title("Temporal separation margin")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    y_top = ax.get_ylim()[1]
    for idx, vals in enumerate(margin_vals, start=1):
        frac = float(np.mean(vals > 0)) if len(vals) else np.nan
        ax.text(idx, y_top, f"{frac:.2f}", ha="center", va="top", fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_path)


def _pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    explained = (s[:2] ** 2) / np.sum(s ** 2)
    return coords, explained


def plot_pca_pairs(method_name: str, diag: pd.DataFrame, context: dict, out_path: Path,
                   max_pairs: int, seed: int) -> None:
    sub = diag[diag["method"] == method_name].copy()
    if sub.empty:
        return
    rng = np.random.RandomState(seed)
    if len(sub) > max_pairs:
        sub = sub.iloc[rng.choice(len(sub), size=max_pairs, replace=False)].copy()
    query_idx = sub["query_index"].to_numpy(dtype=int)
    target_idx = sub["target_gallery_index"].to_numpy(dtype=int)
    q = context["q"][query_idx]
    c = context["gallery"][target_idx]
    coords, explained = _pca2(np.vstack([q, c]))
    q2 = coords[: len(sub)]
    c2 = coords[len(sub):]
    success = sub["temporal_rank"].to_numpy(dtype=int) == 1

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for i in range(len(sub)):
        ax.plot([q2[i, 0], c2[i, 0]], [q2[i, 1], c2[i, 1]],
                color="#444444", alpha=0.14, linewidth=0.75)
    ax.scatter(q2[success, 0], q2[success, 1], s=28, c="#54A24B", marker="o",
               alpha=0.8, label="Query, rank 1")
    ax.scatter(c2[success, 0], c2[success, 1], s=34, c="#54A24B", marker="^",
               alpha=0.8, label="CXR_t2 target, rank 1")
    ax.scatter(q2[~success, 0], q2[~success, 1], s=28, c="#E45756", marker="o",
               alpha=0.8, label="Query, rank > 1")
    ax.scatter(c2[~success, 0], c2[~success, 1], s=34, c="#E45756", marker="^",
               alpha=0.8, label="CXR_t2 target, rank > 1")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title("Real ECG: paired query and target embeddings (PCA)")
    ax.grid(alpha=0.2, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _savefig(fig, out_path)


def plot_similarity_heatmap(method_name: str, diag: pd.DataFrame, context: dict, out_path: Path,
                            heatmap_patients: int, max_queries: int, max_cols: int) -> None:
    sub = diag[diag["method"] == method_name].copy()
    if sub.empty:
        return
    counts = sub.groupby("patient_id").size().sort_values(ascending=False)
    patient_ids = [int(p) for p in counts.head(heatmap_patients).index]
    if not patient_ids:
        return
    rows_per_patient = max(2, max_queries // max(1, len(patient_ids)))
    cols_per_patient = max(2, max_cols // max(1, len(patient_ids)))

    selected_query_indices = []
    selected_gallery_indices = []
    row_group_counts = []
    col_group_counts = []
    for pid in patient_ids:
        psub = sub[sub["patient_id"] == pid].sort_values(["temporal_rank", "query_index"])
        qidx = psub["query_index"].head(rows_per_patient).to_numpy(dtype=int).tolist()
        selected_query_indices.extend(qidx)
        row_group_counts.append(len(qidx))

        candidates = list(context["pat_to_gidx"].get(pid, []))
        target_needed = psub["target_gallery_index"].head(rows_per_patient).astype(int).tolist()
        ordered = []
        for g in target_needed + candidates:
            if g not in ordered:
                ordered.append(int(g))
            if len(ordered) >= cols_per_patient:
                break
        selected_gallery_indices.extend(ordered)
        col_group_counts.append(len(ordered))

    if not selected_query_indices or not selected_gallery_indices:
        return
    matrix = context["sims"][selected_query_indices][:, selected_gallery_indices]
    lo, hi = np.percentile(matrix, [2, 98])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo, hi = float(np.min(matrix)), float(np.max(matrix))
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=lo, vmax=hi)
    target_lookup = {g: j for j, g in enumerate(selected_gallery_indices)}
    target_gidx = context["target_gidx"]
    for row_pos, qidx in enumerate(selected_query_indices):
        col_pos = target_lookup.get(int(target_gidx[qidx]))
        if col_pos is not None:
            ax.scatter([col_pos], [row_pos], marker="s", facecolors="none",
                       edgecolors="black", s=44, linewidths=0.9)

    y = -0.5
    for n in row_group_counts[:-1]:
        y += n
        ax.axhline(y, color="black", linewidth=0.7, alpha=0.45)
    x = -0.5
    for n in col_group_counts[:-1]:
        x += n
        ax.axvline(x, color="black", linewidth=0.7, alpha=0.45)

    row_centers = np.cumsum(row_group_counts) - np.asarray(row_group_counts) / 2 - 0.5
    col_centers = np.cumsum(col_group_counts) - np.asarray(col_group_counts) / 2 - 0.5
    group_labels = [f"P{i + 1}" for i in range(len(patient_ids))]
    ax.set_yticks(row_centers)
    ax.set_yticklabels(group_labels)
    ax.set_xticks(col_centers)
    ax.set_xticklabels(group_labels)
    ax.set_xlabel("Candidate CXR_t2 gallery groups")
    ax.set_ylabel("Query groups")
    ax.set_title("Real ECG: same-patient temporal similarity blocks")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Cosine similarity")
    fig.tight_layout()
    _savefig(fig, out_path)


def _write_json_records(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Run specs as name=Label. Defaults to CXR-only, real, shuffled, zeroed.")
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                    help="Sequence pairs with CXR_t1 for the target run.")
    ap.add_argument("--seq_target_pairs", type=Path, default=Path(C.SEQ_TARGET_PAIRS_JSON))
    ap.add_argument("--single_pairs", type=Path, default=Path(C.SINGLE_ECG_PAIRS_JSON))
    ap.add_argument("--cxr_emb", type=Path, default=Path(C.CXR_EMB_NPY))
    ap.add_argument("--cxr_ids", type=Path, default=Path(C.CXR_IDS_JSON))
    ap.add_argument("--ecg_emb", type=Path, default=Path(C.ECG_EMB_NPY))
    ap.add_argument("--ecg_ids", type=Path, default=Path(C.ECG_IDS_JSON))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--rolling_window", type=int, default=25)
    ap.add_argument("--skip_model_diagnostics", action="store_true")
    ap.add_argument("--pca_method", default="exp5c_weighted_attn_pool")
    ap.add_argument("--heatmap_method", default="exp5c_weighted_attn_pool")
    ap.add_argument("--max_pca_pairs", type=int, default=220)
    ap.add_argument("--heatmap_patients", type=int, default=6)
    ap.add_argument("--max_heatmap_queries", type=int, default=72)
    ap.add_argument("--max_heatmap_cols", type=int, default=72)
    return ap.parse_args()


def main() -> None:
    args = build_args()
    out_dir = args.output_dir or (args.run_dir / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = _method_specs(args.methods, args.run_dir)
    test_df = _test_metrics_table(methods)
    epoch_df = _epoch_frame(methods)
    test_df.to_csv(out_dir / "test_metrics_summary.csv", index=False)
    epoch_df.to_csv(out_dir / "epoch_history_long.csv", index=False)

    plot_test_metrics(test_df, methods, out_dir / "test_temporal_metrics_bar.png")
    plot_epoch_curves(epoch_df, methods, out_dir / "epoch_temporal_learning_curves.png")
    plot_cross_epoch_curves(epoch_df, methods, out_dir / "epoch_cross_patient_curves.png")
    plot_train_dynamics(methods, out_dir / "train_dynamics_rolling.png", args.rolling_window)

    output_files = [
        out_dir / "test_metrics_summary.csv",
        out_dir / "epoch_history_long.csv",
        out_dir / "test_temporal_metrics_bar.png",
        out_dir / "epoch_temporal_learning_curves.png",
        out_dir / "epoch_cross_patient_curves.png",
        out_dir / "train_dynamics_rolling.png",
    ]

    if not args.skip_model_diagnostics:
        device = get_device(args.device)
        print(f"Running checkpoint diagnostics on {args.split} split with device={device}")
        data_cache: dict[str, StagedData] = {}
        diag_frames = []
        contexts = {}
        for m in methods:
            print(f"  diagnostics: {m['name']}")
            diag, context = _compute_model_diagnostics(m, args, data_cache, device)
            diag_frames.append(diag)
            contexts[m["name"]] = context
        diag_df = pd.concat(diag_frames, ignore_index=True)
        diag_summary = _diagnostic_summary(diag_df)
        diag_df.to_csv(out_dir / "contrastive_query_diagnostics.csv", index=False)
        diag_summary.to_csv(out_dir / "contrastive_diagnostics_summary.csv", index=False)
        _write_json_records(diag_summary, out_dir / "contrastive_diagnostics_summary.json")

        plot_rank_cdf(diag_df, methods, out_dir / "temporal_rank_cdf.png")
        plot_similarity_separation(diag_df, methods, out_dir / "similarity_separation.png")
        if args.pca_method in contexts:
            plot_pca_pairs(args.pca_method, diag_df, contexts[args.pca_method],
                           out_dir / "real_ecg_pca_positive_pairs.png",
                           args.max_pca_pairs, args.seed)
        if args.heatmap_method in contexts:
            plot_similarity_heatmap(args.heatmap_method, diag_df, contexts[args.heatmap_method],
                                    out_dir / "real_ecg_similarity_heatmap.png",
                                    args.heatmap_patients, args.max_heatmap_queries,
                                    args.max_heatmap_cols)
        output_files.extend([
            out_dir / "contrastive_query_diagnostics.csv",
            out_dir / "contrastive_diagnostics_summary.csv",
            out_dir / "contrastive_diagnostics_summary.json",
            out_dir / "temporal_rank_cdf.png",
            out_dir / "similarity_separation.png",
            out_dir / "real_ecg_pca_positive_pairs.png",
            out_dir / "real_ecg_similarity_heatmap.png",
        ])

    manifest = {
        "run_dir": str(args.run_dir),
        "output_dir": str(out_dir),
        "pairs": str(args.pairs),
        "methods": [{"name": m["name"], "label": m["label"]} for m in methods],
        "files": [str(p) for p in output_files if p.exists()],
    }
    with open(out_dir / "visualization_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    output_files.append(out_dir / "visualization_manifest.json")

    print(f"Wrote visualizations to {out_dir}")
    for path in output_files:
        if path.exists():
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
