"""Train multimodal (EHR+CXR+ECG) next-step transformer."""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_exp_old = PROJECT_ROOT / "experiment1(old)"
for _p in (PROJECT_ROOT, PROJECT_ROOT / "BaselineExperiment", _exp_old):
    if _p.is_dir():
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: F401,F403
from classification_utils import make_subset, stratified_train_val_test_indices
from model_multimodal_nextstep import MultimodalNextStepModel
from multimodal_forward_mlp_model import MultimodalForwardMLPModel
from multimodal_nextstep_dataset import MultimodalNextStepDataset


def load_pretrained_multimodal_forward_stack(
    model: MultimodalNextStepModel,
    ckpt_path: str,
    input_dim: int,
    device: torch.device,
    ehr_embed_dim: int,
    cxr_dim: int,
    ecg_dim: int,
    fuse_dim: int,
    num_classes: int,
    dropout: float,
    vit_path: str,
    freeze_cxr: bool,
    ecg_ckpt_path: Optional[str],
    freeze_ecg: bool,
    ecg_sig_len: int,
) -> None:
    """Load ``MultimodalForwardMLPModel`` into encoders, miss tokens, projs, ``disc_s2f``/``disc_p2f``; freeze."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ref = MultimodalForwardMLPModel(
        input_dim=input_dim,
        ehr_embed_dim=ehr_embed_dim,
        cxr_dim=cxr_dim,
        ecg_dim=ecg_dim,
        fuse_dim=fuse_dim,
        num_classes=num_classes,
        dropout=dropout,
        vit_path=vit_path,
        freeze_cxr=freeze_cxr,
        ecg_ckpt_path=ecg_ckpt_path,
        freeze_ecg=freeze_ecg,
        ecg_sig_len=ecg_sig_len,
    ).to(device)
    ref.load_state_dict(sd, strict=True)
    model.ehr_enc.load_state_dict(ref.ehr_enc.state_dict())
    model.cxr_enc.load_state_dict(ref.cxr_enc.state_dict())
    model.ecg_enc.load_state_dict(ref.ecg_enc.state_dict())
    model.miss_cxr.data.copy_(ref.miss_cxr.data)
    model.miss_ecg.data.copy_(ref.miss_ecg.data)
    model.proj_e.load_state_dict(ref.proj_e.state_dict())
    model.proj_x.load_state_dict(ref.proj_x.state_dict())
    model.proj_s.load_state_dict(ref.proj_s.state_dict())
    model.disc_s2f.load_state_dict(ref.disc_s2f.state_dict())
    model.disc_p2f.load_state_dict(ref.disc_p2f.state_dict())
    for m in (
        model.ehr_enc,
        model.cxr_enc,
        model.ecg_enc,
        model.proj_e,
        model.proj_x,
        model.proj_s,
        model.disc_s2f,
        model.disc_p2f,
    ):
        for p in m.parameters():
            p.requires_grad = False
    model.miss_cxr.requires_grad = False
    model.miss_ecg.requires_grad = False


def _collect_window_lengths(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return np.array(
            [max(1, int(base._window_indices(int(i)).size)) for i in dataset.indices],
            dtype=np.int32,
        )
    return np.array(
        [max(1, int(dataset._window_indices(i).size)) for i in range(len(dataset))],
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


def collate_multimodal_batch(batch):
    lengths = [b["ehr_seq"].shape[0] for b in batch]
    max_len = max(lengths)
    feat = batch[0]["ehr_seq"].shape[1]
    ecg_L = batch[0]["ecg_seq"].shape[-1]
    bsz = len(batch)

    ehr_seq = torch.zeros(bsz, max_len, feat, dtype=torch.float32)
    ecg_seq = torch.zeros(bsz, max_len, 12, ecg_L, dtype=torch.float32)
    cxr_seq = torch.zeros(bsz, max_len, 3, 224, 224, dtype=batch[0]["cxr_seq"].dtype)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    cxr_mask = torch.zeros(bsz, max_len, dtype=torch.bool)
    ecg_mask = torch.zeros(bsz, max_len, dtype=torch.bool)

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
        ehr_seq[i, :t] = b["ehr_seq"]
        cxr_seq[i, :t] = b["cxr_seq"].float()
        ecg_seq[i, :t] = b["ecg_seq"].float()
        mask[i, :t] = True
        cxr_mask[i, :t] = b["cxr_mask"]
        ecg_mask[i, :t] = b["ecg_mask"]
        s2f_step[i, :t] = b["s2f_step"]
        p2f_step[i, :t] = b["p2f_step"]
        s2f_step_valid[i, :t] = b["s2f_step_valid"]
        p2f_step_valid[i, :t] = b["p2f_step_valid"]
        anchor_s2f[i] = b["anchor_s2f_cls"] if b["anchor_s2f_cls"] >= 0 else -1
        anchor_p2f[i] = b["anchor_p2f_cls"] if b["anchor_p2f_cls"] >= 0 else -1
        anchor_has_s2f[i] = bool(b["anchor_has_s2f"])
        anchor_has_p2f[i] = bool(b["anchor_has_p2f"])

    return {
        "ehr_seq": ehr_seq,
        "cxr_seq": cxr_seq,
        "ecg_seq": ecg_seq,
        "ehr_mask": mask,
        "cxr_mask": cxr_mask,
        "ecg_mask": ecg_mask,
        "anchor_s2f": anchor_s2f,
        "anchor_p2f": anchor_p2f,
        "anchor_has_s2f": anchor_has_s2f,
        "anchor_has_p2f": anchor_has_p2f,
        "s2f_step": s2f_step,
        "p2f_step": p2f_step,
        "s2f_step_valid": s2f_step_valid,
        "p2f_step_valid": p2f_step_valid,
    }


def _stratify_labels_from_mm_dataset(ds: MultimodalNextStepDataset) -> np.ndarray:
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


def loss_next_fused(
    fused: torch.Tensor, next_padded: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    bsz, t, d = fused.shape
    if t < 2:
        return fused.new_tensor(0.0)
    target = fused[:, 1:, :].detach()
    pred = next_padded[:, : t - 1, :]
    pair_ok = (mask[:, :-1] & mask[:, 1:]).float().unsqueeze(-1)
    diff = (pred - target) ** 2
    num = (pair_ok * diff).sum()
    den = (pair_ok.sum() * d).clamp(min=1.0)
    return num / den


def forward_and_loss(model, batch, l_next, l_anc, l_disc, use_step_disc_loss: bool = True):
    device = batch["ehr_seq"].device
    log_s2f_a, log_p2f_a, next_p, s2f_s, p2f_s, fused, _h, _fp = model(
        batch["ehr_seq"],
        batch["cxr_seq"].to(device),
        batch["ecg_seq"].to(device),
        batch["ehr_mask"],
        batch["cxr_mask"].to(device),
        batch["ecg_mask"].to(device),
        return_embeddings=True,
    )
    l_n = loss_next_fused(fused, next_p, batch["ehr_mask"])

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
        s2f_t, p2f_t = b["anchor_s2f"], b["anchor_p2f"]
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
    print(f"Multimodal next-step training. Device: {device}")

    enr = None if args.no_enriched else args.enriched_csv
    if enr and (not str(enr).strip() or not Path(enr).is_file()):
        print(f"  No enriched join: {enr!r}")
        enr = None

    meta = args.metadata_path
    if meta and not Path(meta).is_file():
        meta = None

    full_ds = MultimodalNextStepDataset(
        anchor_source_csv=args.anchor_csv,
        history_csv=args.history_csv,
        schema_csv=args.schema_csv,
        label_lookup_csv=args.label_lookup_csv,
        enriched_csv=enr,
        cxr_root=args.cxr_root,
        metadata_path=meta,
        lookback_min_hours=args.lookback_min_hours,
        lookback_max_hours=args.lookback_max_hours,
        ecg_target_len=args.ecg_target_len,
        cxr_split=args.cxr_split,
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
    y = _stratify_labels_from_mm_dataset(base)
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
        collate_fn=collate_multimodal_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal_batch,
    )

    ecg_ckpt = args.ecg_ckpt if args.ecg_ckpt and Path(args.ecg_ckpt).is_file() else None
    model = MultimodalNextStepModel(
        input_dim=input_dim,
        ehr_embed_dim=args.embed_dim,
        cxr_dim=args.cxr_dim,
        ecg_dim=args.ecg_dim,
        fuse_dim=args.fuse_dim,
        d_model=args.d_model,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
        max_seq_length=args.max_seq_length,
        anchor_pool=args.anchor_pool,
        vit_path=args.vit_path,
        freeze_cxr=args.freeze_cxr,
        ecg_ckpt_path=ecg_ckpt,
        freeze_ecg=args.freeze_ecg,
        ecg_sig_len=args.ecg_target_len,
    ).to(device)

    mm_ckpt = (args.pretrained_mm_forward_mlp_ckpt or "").strip()
    use_step_disc_loss = True
    if mm_ckpt:
        if not Path(mm_ckpt).is_file():
            raise FileNotFoundError(f"--pretrained_mm_forward_mlp_ckpt not found: {mm_ckpt}")
        load_pretrained_multimodal_forward_stack(
            model,
            mm_ckpt,
            input_dim,
            device,
            ehr_embed_dim=args.embed_dim,
            cxr_dim=args.cxr_dim,
            ecg_dim=args.ecg_dim,
            fuse_dim=args.fuse_dim,
            num_classes=args.num_classes,
            dropout=args.dropout,
            vit_path=args.vit_path,
            freeze_cxr=args.freeze_cxr,
            ecg_ckpt_path=ecg_ckpt,
            freeze_ecg=args.freeze_ecg,
            ecg_sig_len=args.ecg_target_len,
        )
        use_step_disc_loss = False
        print(
            f"  Loaded frozen multimodal forward MLP from {mm_ckpt}; step disc CE omitted. "
            "Training fuse_in / Transformer / head_next / anchor heads."
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
                "pretrained_mm_forward_mlp_ckpt": mm_ckpt or None,
                "frozen_mm_encoders_proj_disc": bool(mm_ckpt),
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
    p.add_argument("--history_csv", default=MULTIMODAL_HISTORY_CSV)
    p.add_argument("--label_lookup_csv", default=P2F_OR_S2F_CSV)
    p.add_argument("--schema_csv", default=SCHEMA_CSV)
    p.add_argument("--enriched_csv", default=NEXTSTEP_ENRICHED_CSV)
    p.add_argument("--no_enriched", action="store_true")
    p.add_argument("--cxr_root", default=CXR_ROOT_DEFAULT)
    p.add_argument("--metadata_path", default=METADATA_PATH_DEFAULT)
    p.add_argument("--ecg_ckpt", default=ECG_CKPT_DEFAULT)
    p.add_argument("--vit_path", default=VIT_PATH_DEFAULT)
    p.add_argument("--lookback_min_hours", type=int, default=LOOKBACK_MIN_HOURS)
    p.add_argument("--lookback_max_hours", type=int, default=LOOKBACK_MAX_HOURS)
    p.add_argument("--ecg_target_len", type=int, default=ECG_TARGET_LEN_DEFAULT)
    p.add_argument("--cxr_split", type=str, default="train")
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--cxr_dim", type=int, default=CXR_DIM_DEFAULT)
    p.add_argument("--ecg_dim", type=int, default=ECG_DIM_DEFAULT)
    p.add_argument("--fuse_dim", type=int, default=FUSE_DIM_DEFAULT)
    p.add_argument("--d_model", type=int, default=NEXTSTEP_D_MODEL)
    p.add_argument("--num_transformer_layers", type=int, default=NEXTSTEP_NUM_TRANSFORMER_LAYERS)
    p.add_argument("--num_heads", type=int, default=NEXTSTEP_NUM_HEADS)
    p.add_argument("--dropout", type=float, default=NEXTSTEP_DROPOUT)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--anchor_pool", type=str, default=NEXTSTEP_ANCHOR_POOL, choices=["last", "mean"])
    p.add_argument("--freeze_cxr", action="store_true", default=True)
    p.add_argument("--no_freeze_cxr", action="store_false", dest="freeze_cxr")
    p.add_argument("--freeze_ecg", action="store_true", default=True)
    p.add_argument("--no_freeze_ecg", action="store_false", dest="freeze_ecg")
    p.add_argument("--lambda_next", type=float, default=NEXTSTEP_LAMBDA_NEXT)
    p.add_argument("--lambda_anchor", type=float, default=NEXTSTEP_LAMBDA_ANCHOR)
    p.add_argument("--lambda_disc", type=float, default=NEXTSTEP_LAMBDA_DISC)
    p.add_argument(
        "--pretrained_mm_forward_mlp_ckpt",
        default="",
        help="If set, load train_multimodal_forward_mlp best.pt into encoders+proj+disc (frozen); "
        "step disc CE omitted (train fuse_in / causal stack / head_next / anchor heads).",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--train_split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "output_multimodal_nextstep"),
    )
    p.add_argument("--max_samples", type=int, default=0)
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
    if not a.max_samples:
        a.max_samples = 0
    main(a)
