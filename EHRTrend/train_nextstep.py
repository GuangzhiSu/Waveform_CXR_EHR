"""Train EHR next-step transformer with joint anchor + per-step disc losses."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BaselineExperiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: F401,F403
from classification_utils import make_subset, stratified_train_val_test_indices
from ehr_nextstep_dataset import EHRNextStepDataset
from forward_mlp_model import ForwardChangeRowModel
from model_nextstep import EHRNextStepModel


def load_pretrained_row_encoder_and_disc_heads(
    model: EHRNextStepModel,
    ckpt_path: str,
    input_dim: int,
    device: torch.device,
) -> None:
    """Load ``ForwardChangeRowModel`` weights into ``row_encoder`` + ``disc_s2f``/``disc_p2f`` and freeze them."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ref = ForwardChangeRowModel(
        input_dim=input_dim,
        embed_dim=model.embed_dim,
        num_classes=model.num_classes,
        dropout=0.2,
    )
    ref.load_state_dict(ckpt["model"], strict=True)
    model.row_encoder.load_state_dict(ref.encoder.state_dict())
    model.disc_s2f.load_state_dict(ref.head_s2f.state_dict())
    model.disc_p2f.load_state_dict(ref.head_p2f.state_dict())
    for m in (model.row_encoder, model.disc_s2f, model.disc_p2f):
        for p in m.parameters():
            p.requires_grad = False


def _collect_window_lengths(dataset) -> np.ndarray:
    """Per-anchor count of EHR rows in [t-24h, t-12h]."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return np.array(
            [int(base._window_indices(int(i)).size) for i in dataset.indices],
            dtype=np.int32,
        )
    return np.array(
        [int(dataset._window_indices(i).size) for i in range(len(dataset))],
        dtype=np.int32,
    )


def _log_seq_length_histogram(lengths: np.ndarray, out_dir: Path) -> dict:
    n_bins = min(40, max(1, int(lengths.max())))
    counts, edges = np.histogram(lengths, bins=n_bins)
    payload = {
        "n_anchors": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "median": float(np.median(lengths)),
        "mean": float(lengths.mean()),
        "histogram_counts": [int(x) for x in counts],
        "histogram_bin_edges": [float(x) for x in edges],
    }
    print(
        f"  seq_len n: min={payload['min']} median={payload['median']:.0f} "
        f"max={payload['max']} mean={payload['mean']:.1f}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "seq_length_histogram.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def collate_nextstep_batch(batch):
    lengths = [b["ehr_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    feat = batch[0]["ehr_seq"].shape[1]
    bsz = len(batch)
    seq = torch.zeros(bsz, max_len, feat, dtype=torch.float32)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    s2f_step = torch.full((bsz, max_len), -1, dtype=torch.long)
    p2f_step = torch.full((bsz, max_len), -1, dtype=torch.long)
    s2f_step_valid = torch.zeros(bsz, max_len, dtype=torch.bool)
    p2f_step_valid = torch.zeros(bsz, max_len, dtype=torch.bool)
    anchor_s2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_p2f = torch.full((bsz,), -1, dtype=torch.long)
    anchor_has_s2f = torch.zeros(bsz, dtype=torch.bool)
    anchor_has_p2f = torch.zeros(bsz, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["ehr_seq"].shape[0]
        seq[i, :t] = b["ehr_seq"]
        mask[i, :t] = True
        s2f_step[i, :t] = b["s2f_step"]
        p2f_step[i, :t] = b["p2f_step"]
        s2f_step_valid[i, :t] = b["s2f_step_valid"]
        p2f_step_valid[i, :t] = b["p2f_step_valid"]
        c = b["anchor_s2f_cls"]
        anchor_s2f[i] = c if c >= 0 else -1
        c2 = b["anchor_p2f_cls"]
        anchor_p2f[i] = c2 if c2 >= 0 else -1
        anchor_has_s2f[i] = bool(b["anchor_has_s2f"])
        anchor_has_p2f[i] = bool(b["anchor_has_p2f"])
    return {
        "ehr_seq": seq,
        "ehr_mask": mask,
        "anchor_s2f": anchor_s2f,
        "anchor_p2f": anchor_p2f,
        "anchor_has_s2f": anchor_has_s2f,
        "anchor_has_p2f": anchor_has_p2f,
        "s2f_step": s2f_step,
        "p2f_step": p2f_step,
        "s2f_step_valid": s2f_step_valid,
        "p2f_step_valid": p2f_step_valid,
    }


def _stratify_labels_from_dataset(ds: EHRNextStepDataset) -> np.ndarray:
    n = len(ds)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ds.anchor_has_p2f[i] and ds.anchor_p2f_cls[i] >= 0:
            y[i] = int(ds.anchor_p2f_cls[i])
        elif ds.anchor_has_s2f[i] and ds.anchor_s2f_cls[i] >= 0:
            y[i] = 3 + int(ds.anchor_s2f_cls[i])
        else:
            y[i] = 0
    return y


def loss_next_step(
    z0: torch.Tensor, next_padded: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    bsz, t, d = z0.shape
    if t < 2:
        return z0.new_tensor(0.0)
    target = z0[:, 1:, :].detach()
    pred = next_padded[:, : t - 1, :]
    pair_ok = (mask[:, :-1] & mask[:, 1:]).float().unsqueeze(-1)
    diff = (pred - target) ** 2
    num = (pair_ok * diff).sum()
    den = (pair_ok.sum() * d).clamp(min=1.0)
    return num / den


def forward_and_loss(
    model: EHRNextStepModel,
    batch: dict,
    l_next: float,
    l_anc: float,
    l_disc: float,
    use_step_disc_loss: bool = True,
) -> tuple:
    ehr = batch["ehr_seq"]
    m = batch["ehr_mask"]
    device = ehr.device
    log_s2f_a, log_p2f_a, next_p, s2f_s, p2f_s, z0, _h = model(ehr, m, return_embeddings=True)
    l_n = loss_next_step(z0, next_p, m)
    s2f_tgt = batch["anchor_s2f"].to(device)
    p2f_tgt = batch["anchor_p2f"].to(device)
    s2f_use = batch["anchor_has_s2f"].to(device) & (s2f_tgt >= 0)
    p2f_use = batch["anchor_has_p2f"].to(device) & (p2f_tgt >= 0)
    la_s = log_s2f_a.new_tensor(0.0)
    la_p = log_s2f_a.new_tensor(0.0)
    if s2f_use.any():
        la_s = F.cross_entropy(log_s2f_a[s2f_use], s2f_tgt[s2f_use])
    if p2f_use.any():
        la_p = F.cross_entropy(log_p2f_a[p2f_use], p2f_tgt[p2f_use])
    l_a = la_s + la_p

    ld = log_s2f_a.new_tensor(0.0)
    if use_step_disc_loss:
        vs = batch["s2f_step_valid"].to(device) & (batch["s2f_step"].to(device) >= 0)
        vp = batch["p2f_step_valid"].to(device) & (batch["p2f_step"].to(device) >= 0)
        if vs.any():
            ld = ld + F.cross_entropy(s2f_s[vs], batch["s2f_step"].to(device)[vs])
        if vp.any():
            ld = ld + F.cross_entropy(p2f_s[vp], batch["p2f_step"].to(device)[vp])
    total = l_next * l_n + l_anc * l_a + l_disc * ld
    return total, l_n.detach(), l_a.detach(), ld.detach(), log_s2f_a, log_p2f_a


@torch.no_grad()
def eval_epoch(model, loader, device, l_next, l_anc, l_disc, use_step_disc_loss: bool = True) -> dict:
    model.eval()
    tot = 0.0
    n_batches = 0
    s2f_ok = 0.0
    s2f_n = 0
    p2f_ok = 0.0
    p2f_n = 0
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        loss, _, _, _, log_s2f_a, log_p2f_a = forward_and_loss(
            model, b, l_next, l_anc, l_disc, use_step_disc_loss=use_step_disc_loss
        )
        tot += float(loss)
        n_batches += 1
        s2f_t = b["anchor_s2f"]
        p2f_t = b["anchor_p2f"]
        su = b["anchor_has_s2f"] & (s2f_t >= 0)
        pu = b["anchor_has_p2f"] & (p2f_t >= 0)
        if su.any():
            s2f_ok += (log_s2f_a[su].argmax(1) == s2f_t[su]).float().sum().item()
            s2f_n += int(su.sum())
        if pu.any():
            p2f_ok += (log_p2f_a[pu].argmax(1) == p2f_t[pu]).float().sum().item()
            p2f_n += int(pu.sum())
    return {
        "loss": tot / max(n_batches, 1),
        "acc_anchor_s2f": s2f_ok / s2f_n if s2f_n else float("nan"),
        "acc_anchor_p2f": p2f_ok / p2f_n if p2f_n else float("nan"),
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"EHR next-step training. Device: {device}")

    enr = None if args.no_enriched else args.enriched_csv
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        print(f"  No enriched join (path missing or empty): {enr!r}")
        enr = None

    full_ds = EHRNextStepDataset(
        anchor_source_csv=args.anchor_csv,
        history_csv=args.history_csv,
        schema_csv=args.schema_csv,
        enriched_csv=enr,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
    )
    n_all = len(full_ds)
    if args.max_samples and args.max_samples < n_all:
        import random

        rng = random.Random(args.seed)
        idxs = list(range(n_all))
        rng.shuffle(idxs)
        idxs = idxs[: args.max_samples]
        full_ds = Subset(full_ds, idxs)
        print(f"  Subset max_samples={args.max_samples} (from {n_all})")

    base = full_ds.dataset if isinstance(full_ds, Subset) else full_ds
    input_dim = base.input_dim
    y = _stratify_labels_from_dataset(base)
    if isinstance(full_ds, Subset):
        y = y[np.array(full_ds.indices, dtype=np.int64)]

    test_split = 1.0 - args.train_split - args.val_split
    idx_train, idx_val, idx_test = stratified_train_val_test_indices(
        y, args.train_split, args.val_split, test_split, args.seed
    )
    train_ds = make_subset(full_ds, idx_train)
    val_ds = make_subset(full_ds, idx_val)
    test_ds = make_subset(full_ds, idx_test)
    print(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_seq_length_histogram(_collect_window_lengths(full_ds), out_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_nextstep_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_nextstep_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_nextstep_batch,
    )

    model = EHRNextStepModel(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        anchor_pool=args.anchor_pool,
        disc_head_dropout=0.2,
    ).to(device)

    ckpt_path = (args.pretrained_forward_mlp_ckpt or "").strip()
    use_step_disc_loss = True
    if ckpt_path:
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"--pretrained_forward_mlp_ckpt not found: {ckpt_path}")
        load_pretrained_row_encoder_and_disc_heads(model, ckpt_path, input_dim, device)
        use_step_disc_loss = False
        print(
            f"  Loaded frozen ForwardChangeRowModel row encoder + disc heads from {ckpt_path}; "
            "step CE omitted from loss (no grad through frozen path). Training proj/transformer/head_next/anchor heads."
        )
        if args.lambda_disc != 0:
            print(f"  Note: lambda_disc={args.lambda_disc} has no effect while step disc loss is disabled.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(args.epochs):
        model.train()
        tr_sum = 0.0
        for batch in train_loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, _, _, _, _, _ = forward_and_loss(
                model,
                b,
                args.lambda_next,
                args.lambda_anchor,
                args.lambda_disc,
                use_step_disc_loss=use_step_disc_loss,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_sum += float(loss)
        tr = tr_sum / max(len(train_loader), 1)
        st = eval_epoch(
            model,
            val_loader,
            device,
            args.lambda_next,
            args.lambda_anchor,
            args.lambda_disc,
            use_step_disc_loss=use_step_disc_loss,
        )
        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={tr:.4f}  val_loss={st['loss']:.4f}  "
            f"val_acc_s2f={st['acc_anchor_s2f']:.4f}  val_acc_p2f={st['acc_anchor_p2f']:.4f}"
        )
        improved = st["loss"] < best_val - args.early_stop_min_delta
        if improved:
            best_val = st["loss"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": best_val},
                out_dir / "best.pt",
            )
        else:
            epochs_no_improve += 1

        if (
            args.early_stop_patience > 0
            and epochs_no_improve >= args.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}/{args.epochs} "
                f"(val_loss did not improve by >{args.early_stop_min_delta} for {args.early_stop_patience} epochs; "
                f"best epoch {best_epoch + 1}, best val_loss={best_val:.4f})"
            )
            stopped_early = True
            break

    torch.save(model.state_dict(), out_dir / "last.pt")
    if (out_dir / "best.pt").is_file():
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    stt = eval_epoch(
        model,
        test_loader,
        device,
        args.lambda_next,
        args.lambda_anchor,
        args.lambda_disc,
        use_step_disc_loss=use_step_disc_loss,
    )
    print(f"Test: loss={stt['loss']:.4f}  acc_s2f={stt['acc_anchor_s2f']:.4f}  acc_p2f={stt['acc_anchor_p2f']:.4f}")
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "test_loss": stt["loss"],
                "test_acc_anchor_s2f": stt["acc_anchor_s2f"],
                "test_acc_anchor_p2f": stt["acc_anchor_p2f"],
                "pretrained_forward_mlp_ckpt": ckpt_path or None,
                "frozen_row_encoder_and_disc_heads": bool(ckpt_path),
                "use_step_disc_loss": use_step_disc_loss,
                "best_val_loss": best_val,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
                "stopped_early": stopped_early,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_delta": args.early_stop_min_delta,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--history_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument(
        "--enriched_csv",
        default=NEXTSTEP_ENRICHED_CSV,
        help="p2f_vent_fio2_enriched for subject_id join; use with --no_enriched to force hadm_id only.",
    )
    p.add_argument("--no_enriched", action="store_true", help="Do not join subject_id from enriched CSV")
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--d_model", type=int, default=NEXTSTEP_D_MODEL)
    p.add_argument("--num_transformer_layers", type=int, default=NEXTSTEP_NUM_TRANSFORMER_LAYERS)
    p.add_argument("--num_heads", type=int, default=NEXTSTEP_NUM_HEADS)
    p.add_argument("--dropout", type=float, default=NEXTSTEP_DROPOUT)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--anchor_pool", type=str, default=NEXTSTEP_ANCHOR_POOL, choices=["last", "mean"])
    p.add_argument("--lambda_next", type=float, default=NEXTSTEP_LAMBDA_NEXT)
    p.add_argument("--lambda_anchor", type=float, default=NEXTSTEP_LAMBDA_ANCHOR)
    p.add_argument("--lambda_disc", type=float, default=NEXTSTEP_LAMBDA_DISC)
    p.add_argument(
        "--pretrained_forward_mlp_ckpt",
        default="",
        help="If set, load train_forward_mlp best.pt into row_encoder + disc heads (frozen); "
        "step CE skipped so loss trains proj/causal transformer/head_next/anchor heads only.",
    )
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "output_nextstep"))
    p.add_argument("--max_samples", type=int, default=0, help="If >0, use random subset for faster dev")
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=NEXTSTEP_EARLY_STOP_PATIENCE,
        help="Stop if val_loss does not improve (by min_delta) for this many epochs; 0 disables.",
    )
    p.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=NEXTSTEP_EARLY_STOP_MIN_DELTA,
        help="Minimum val_loss decrease to count as improvement.",
    )
    a = p.parse_args()
    if a.max_samples is None:
        a.max_samples = 0
    main(a)
