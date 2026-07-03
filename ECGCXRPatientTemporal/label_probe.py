"""Frozen-embedding label probes for ECG/CXR temporal contrastive models.

The contrastive model is frozen. For each staged sample, we extract the query
embedding ``q`` (default) that was trained to align to ``CXR_t2`` and predict
structured CXR annotations from ``CXR_t2``. The label CSV is aggregated to a
multi-label finding-presence target per ``study_id``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from engine import load_staged_data, set_seed  # noqa: E402
from experiments import REGISTRY, ExperimentSpec  # noqa: E402
from staged_dataset import StagedDataset, collate_fn  # noqa: E402
from staged_model import StagedModel  # noqa: E402

DEFAULT_LABEL_CSV = (
    "/hpc/dctrl/ma618/temporal_labels/"
    "mimic_structured_labels_explicit_comparison_except_absent_text_cue/"
    "all_progession_with_prior.csv"
)


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _checkpoint_path(args, spec) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return Path(args.contrastive_output_dir) / spec.name / "best.pt"


def _load_spec_from_checkpoint(path: Path, fallback: ExperimentSpec) -> ExperimentSpec:
    ckpt = torch.load(path, map_location="cpu")
    spec_dict = ckpt.get("spec")
    if spec_dict:
        return ExperimentSpec(**spec_dict)
    return fallback


def _load_contrastive_model(spec: ExperimentSpec, args, data, device):
    ckpt_path = _checkpoint_path(args, spec)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    model = StagedModel(
        spec,
        cxr_dim=int(cfg.get("cxr_dim", data.cxr_emb.shape[1])),
        ecg_dim=int(cfg.get("ecg_dim", data.ecg_emb.shape[1])),
        proj_dim=int(cfg.get("proj_dim", args.proj_dim)),
        cxr_proj_hidden=int(cfg.get("cxr_proj_hidden", C.CXR_PROJ_HIDDEN)),
        d_model=int(cfg.get("d_model", args.d_model)),
        ecg_tx_layers=int(cfg.get("ecg_tx_layers", args.ecg_tx_layers)),
        ecg_tx_heads=int(cfg.get("ecg_tx_heads", C.ECG_TX_HEADS)),
        ecg_tx_mlp_ratio=float(cfg.get("ecg_tx_mlp_ratio", C.ECG_TX_MLP_RATIO)),
        fusion_hidden=int(cfg.get("fusion_hidden", C.FUSION_HIDDEN)),
        time_emb_dim=int(cfg.get("time_emb_dim", C.TIME_EMB_DIM)),
        dropout=float(cfg.get("dropout", C.DROPOUT)),
        temperature=float(cfg.get("temperature", args.temperature)),
        learnable_temperature=bool(cfg.get("learnable_temperature", args.learnable_temperature)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_path


def build_label_table(label_csv: str, study_ids: set[str], uncertain_positive: bool) -> pd.DataFrame:
    cols = ["study_id", "finding", "label"]
    df = pd.read_csv(label_csv, usecols=cols)
    df["study_id"] = df["study_id"].astype(str)
    df = df[df["study_id"].isin(study_ids)].copy()
    if df.empty:
        raise RuntimeError("No label rows matched the staged CXR_t2 study ids.")
    positive = df["label"].eq("present")
    if uncertain_positive:
        positive = positive | df["label"].eq("uncertain")
    df["target"] = positive.astype(np.float32)
    table = df.groupby(["study_id", "finding"], sort=True)["target"].max().unstack(fill_value=0.0)
    table = table.astype(np.float32)
    table.columns = [str(c) for c in table.columns]
    return table


@torch.no_grad()
def extract_embeddings(model, dataset, data, cxr_ids: list[str], label_table: pd.DataFrame,
                       device, batch_size: int, embedding: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    xs, ys, rows, pids = [], [], [], []
    model.eval()
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        q, c2, c1 = model.encode(b)
        if embedding == "q":
            z = q
        elif embedding == "c2":
            z = c2
        elif embedding == "c1":
            if c1 is None:
                raise RuntimeError("Requested c1 embedding, but this experiment has no CXR_t1.")
            z = c1
        else:
            raise ValueError(f"Unknown embedding={embedding!r}")

        z = z.float().cpu().numpy()
        c2_rows = batch["c2_row"].cpu().numpy()
        batch_pids = batch["patient_id"].cpu().numpy()
        for i, row in enumerate(c2_rows):
            study_id = cxr_ids[int(row)]
            if study_id not in label_table.index:
                continue
            xs.append(z[i])
            ys.append(label_table.loc[study_id].to_numpy(dtype=np.float32))
            rows.append(int(row))
            pids.append(int(batch_pids[i]))

    if not xs:
        raise RuntimeError("No labeled samples found after joining CXR_t2 ids to label table.")
    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.asarray(rows, dtype=np.int64),
        np.asarray(pids, dtype=np.int64),
    )


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), (
        (val_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    probs = 1.0 / (1.0 + np.exp(-logits))
    per_auc, per_ap = [], []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        if np.unique(yj).size < 2:
            continue
        per_auc.append(float(roc_auc_score(yj, probs[:, j])))
        per_ap.append(float(average_precision_score(yj, probs[:, j])))

    metrics = {
        "n_samples": int(y_true.shape[0]),
        "n_labels": int(y_true.shape[1]),
        "macro_auroc": float(np.mean(per_auc)) if per_auc else float("nan"),
        "macro_auprc": float(np.mean(per_ap)) if per_ap else float("nan"),
        "labels_with_both_classes": int(len(per_auc)),
    }
    if np.unique(y_true.reshape(-1)).size >= 2:
        metrics["micro_auroc"] = float(roc_auc_score(y_true.reshape(-1), probs.reshape(-1)))
        metrics["micro_auprc"] = float(average_precision_score(y_true.reshape(-1), probs.reshape(-1)))
    pred = (probs >= threshold).astype(np.float32)
    metrics["macro_f1@0.5"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    metrics["micro_f1@0.5"] = float(f1_score(y_true, pred, average="micro", zero_division=0))
    return metrics


def _loader(x, y, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _eval_loss_and_logits(model, x, y, batch_size: int, device, criterion):
    loader = _loader(x, y, batch_size, shuffle=False)
    losses, logits = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            losses.append(float(criterion(out, yb).item()) * xb.size(0))
            logits.append(out.float().cpu())
    return sum(losses) / max(len(x), 1), torch.cat(logits).numpy()


def train_probe(kind: str, train_x, train_y, val_x, val_y, test_x, test_y, args, out_dir: Path):
    device = get_device(args.device)
    in_dim, out_dim = train_x.shape[1], train_y.shape[1]
    if kind == "linear":
        probe = LinearProbe(in_dim, out_dim)
    elif kind == "mlp":
        probe = MLPProbe(in_dim, args.hidden_dim, out_dim, args.dropout)
    else:
        raise ValueError(kind)
    probe = probe.to(device)

    pos = train_y.sum(axis=0)
    neg = train_y.shape[0] - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, args.max_pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight.astype(np.float32)).to(device))
    opt = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)

    best_score, best_epoch, patience = -1.0, -1, 0
    history = []
    best_path = out_dir / f"best_{kind}.pt"
    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        total = 0.0
        n = 0
        for xb, yb in _loader(train_x, train_y, args.probe_batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(probe(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), args.max_grad_norm)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        val_loss, val_logits = _eval_loss_and_logits(
            probe, val_x, val_y, args.probe_batch_size, device, criterion)
        val_metrics = compute_metrics(val_y, val_logits)
        score = val_metrics.get("macro_auprc", float("nan"))
        score = score if score == score else -val_loss
        history.append({
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_loss": val_loss,
            "val": val_metrics,
        })
        print(f"    [{kind} E{epoch:03d}] train_loss={history[-1]['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_macro_AUPRC={val_metrics['macro_auprc']:.4f} "
              f"val_macro_AUROC={val_metrics['macro_auroc']:.4f}")

        if score > best_score + 1e-5:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({"model": probe.state_dict(), "kind": kind, "epoch": epoch}, best_path)
        else:
            patience += 1
            if patience >= args.probe_patience:
                print(f"    [{kind}] early stop @E{epoch} (best E{best_epoch})")
                break

    probe.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_loss, test_logits = _eval_loss_and_logits(
        probe, test_x, test_y, args.probe_batch_size, device, criterion)
    val_loss, val_logits = _eval_loss_and_logits(
        probe, val_x, val_y, args.probe_batch_size, device, criterion)
    result = {
        "kind": kind,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "val_loss": val_loss,
        "val": compute_metrics(val_y, val_logits),
        "test_loss": test_loss,
        "test": compute_metrics(test_y, test_logits),
        "history": history,
        "checkpoint": str(best_path),
    }
    with open(out_dir / f"results_{kind}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def plot_probe_history(results: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for kind, res in results.items():
            hist = res["history"]
            epochs = [h["epoch"] for h in hist]
            auprc = [h["val"]["macro_auprc"] for h in hist]
            ax.plot(epochs, auprc, marker="o", linewidth=1.5, label=kind)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation macro AUPRC")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "probe_val_macro_auprc.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)
    except Exception as e:
        print(f"  WARNING: failed to plot probe history: {e}")
        return None


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp5a_proj_tx_crossattn_norm",
                    choices=sorted(REGISTRY.keys()))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--contrastive_output_dir",
        default=str(C.OUTPUTS_DIR / "fusion_schemes_full"),
    )
    ap.add_argument("--label_csv", default=DEFAULT_LABEL_CSV)
    ap.add_argument("--output_dir", default=str(C.OUTPUTS_DIR / "label_probe"))
    ap.add_argument("--embedding", default="q", choices=["q", "c1", "c2"])
    ap.add_argument("--uncertain_positive", action="store_true")
    ap.add_argument("--min_train_positives", type=int, default=10)
    ap.add_argument("--probe", default="both", choices=["linear", "mlp", "both"])

    ap.add_argument("--pairs", default=str(C.CACHE_ROOT / "full" / "patient_temporal_pairs.json"))
    ap.add_argument("--seq_target_pairs", default=str(C.CACHE_ROOT / "full" / "seq_target_pairs.json"))
    ap.add_argument("--single_pairs", default=str(C.CACHE_ROOT / "full" / "single_ecg_pairs.json"))
    ap.add_argument("--cxr_emb", default=C.CXR_EMB_NPY)
    ap.add_argument("--cxr_ids", default=C.CXR_IDS_JSON)
    ap.add_argument("--ecg_emb", default=C.ECG_EMB_NPY)
    ap.add_argument("--ecg_ids", default=C.ECG_IDS_JSON)
    ap.add_argument("--seed", type=int, default=C.SEED)

    ap.add_argument("--proj_dim", type=int, default=C.PROJ_DIM)
    ap.add_argument("--d_model", type=int, default=C.D_MODEL)
    ap.add_argument("--ecg_tx_layers", type=int, default=C.ECG_TX_LAYERS)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--learnable_temperature", action="store_true",
                    default=C.LEARNABLE_TEMPERATURE)

    ap.add_argument("--device", default="auto")
    ap.add_argument("--extract_batch_size", type=int, default=512)
    ap.add_argument("--probe_batch_size", type=int, default=512)
    ap.add_argument("--probe_epochs", type=int, default=80)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_weight_decay", type=float, default=1e-4)
    ap.add_argument("--probe_patience", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_pos_weight", type=float, default=50.0)
    ap.add_argument("--no_standardize", action="store_true")
    return ap.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)
    device = get_device(args.device)
    fallback_spec = REGISTRY[args.experiment]
    ckpt_path = _checkpoint_path(args, fallback_spec)
    spec = _load_spec_from_checkpoint(ckpt_path, fallback_spec)
    out_dir = Path(args.output_dir) / spec.name / args.embedding
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Label probe: {spec.name} embedding={args.embedding} device={device} ===")
    print(f"  checkpoint: {ckpt_path}")
    data = load_staged_data(spec, args)
    cxr_ids = json.load(open(args.cxr_ids))
    used_rows = {int(data.pairs[i]["c2"]) for split in data.split_indices.values() for i in split}
    used_study_ids = {cxr_ids[r] for r in used_rows}
    label_table = build_label_table(args.label_csv, used_study_ids, args.uncertain_positive)
    print(f"  label studies matched: {len(label_table):,}; raw labels={label_table.shape[1]}")

    model, loaded_ckpt = _load_contrastive_model(spec, args, data, device)
    splits = {}
    for split_name in ("train", "val", "test"):
        ds = StagedDataset(
            data, data.split_indices[split_name], ecg_perturb=spec.ecg_perturb,
            seed=args.seed + {"train": 0, "val": 1, "test": 2}[split_name])
        splits[split_name] = extract_embeddings(
            model, ds, data, cxr_ids, label_table, device, args.extract_batch_size,
            args.embedding)
        print(f"  {split_name}: labeled samples={splits[split_name][0].shape[0]:,}")

    train_x, train_y, _, _ = splits["train"]
    val_x, val_y, _, _ = splits["val"]
    test_x, test_y, _, _ = splits["test"]

    pos = train_y.sum(axis=0)
    neg = train_y.shape[0] - pos
    keep = (pos >= args.min_train_positives) & (neg > 0)
    if not np.any(keep):
        raise RuntimeError("No labels survived min_train_positives filtering.")
    label_names = list(label_table.columns[keep])
    train_y, val_y, test_y = train_y[:, keep], val_y[:, keep], test_y[:, keep]
    print(f"  labels kept: {len(label_names)} / {label_table.shape[1]}")

    if not args.no_standardize:
        train_x, val_x, test_x = standardize(train_x, val_x, test_x)

    probe_kinds = ["linear", "mlp"] if args.probe == "both" else [args.probe]
    results = {}
    for kind in probe_kinds:
        print(f"\n--- Training {kind} probe ---")
        results[kind] = train_probe(
            kind, train_x, train_y, val_x, val_y, test_x, test_y, args, out_dir)
        tm = results[kind]["test"]
        print(f"  [{kind}] TEST macro_AUPRC={tm['macro_auprc']:.4f} "
              f"macro_AUROC={tm['macro_auroc']:.4f} micro_F1={tm['micro_f1@0.5']:.4f}")

    summary = {
        "experiment": spec.name,
        "embedding": args.embedding,
        "checkpoint": str(loaded_ckpt),
        "label_csv": args.label_csv,
        "uncertain_positive": bool(args.uncertain_positive),
        "label_names": label_names,
        "n_labels": len(label_names),
        "n_samples": {
            "train": int(train_x.shape[0]),
            "val": int(val_x.shape[0]),
            "test": int(test_x.shape[0]),
        },
        "results": results,
        "plot": plot_probe_history(results, out_dir),
        "args": vars(args),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
