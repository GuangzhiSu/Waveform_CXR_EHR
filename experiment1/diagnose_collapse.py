"""
Diagnose prediction collapse: check encoder features vs head output.
Identifies whether collapse happens at (1) encoder or (2) head.

Usage:
  python diagnose_collapse.py --model baseline
  python diagnose_collapse.py --model baseline2
  python diagnose_collapse.py --model baseline3
  python diagnose_collapse.py --model all   # run all three
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline.dataset import WaveformCXREHRDataset


def collate_baseline(batch):
    cxr = torch.stack([b["cxr"] for b in batch])
    signal = torch.stack([b["signal"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    return {"cxr": cxr, "signal": signal, "target": target}


def collate_baseline2(batch):
    signal = torch.stack([b["signal"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    return {"signal": signal, "target": target}


def collate_baseline3(batch):
    cxr = torch.stack([b["cxr"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    return {"cxr": cxr, "target": target}


def compute_stats(name, feats, targets):
    """feats: (N, D), targets: (N,)"""
    feats = np.asarray(feats, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n, d = feats.shape

    # Overall variance
    feat_std = np.std(feats)
    feat_std_per_dim = np.std(feats, axis=0)
    nz_dims = np.sum(feat_std_per_dim > 1e-6)

    # Correlation of each dim with target
    if targets.std() > 1e-8:
        max_corr, n_corr = 0.0, 0
        for i in range(d):
            if feat_std_per_dim[i] > 1e-8:
                c = np.corrcoef(feats[:, i], targets)[0, 1]
                c = 0.0 if np.isnan(c) else c
                max_corr = max(max_corr, abs(c))
                n_corr += 1 if abs(c) > 0.1 else 0
            else:
                pass  # constant dim: no correlation
    else:
        max_corr, n_corr = np.nan, 0

    return {
        "name": name,
        "feat_std": feat_std,
        "feat_std_per_dim_mean": np.mean(feat_std_per_dim),
        "feat_std_per_dim_max": np.max(feat_std_per_dim),
        "n_dims_with_variance": nz_dims,
        "max_corr_with_target": max_corr,
        "n_dims_corr_gt_01": n_corr,
    }


def diagnose_model(model_name, device, args):
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {model_name}")
    print(f"{'='*60}")

    proj = Path(__file__).resolve().parent
    ckpt_path = proj / model_name / "output" / "best.pt"
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    # Load dataset
    load_cxr = model_name in ["baseline", "baseline3"]
    load_signal = model_name in ["baseline", "baseline2"]
    full_ds = WaveformCXREHRDataset(
        csv_path=args.csv_path,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        target_col=args.target_col,
        split="test",
        load_cxr=load_cxr,
        load_signal=load_signal,
    )
    n = len(full_ds)
    n_train = int(n * args.train_split)
    n_val = int(n * args.val_split)
    n_test = n - n_train - n_val
    _, _, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.max_samples is not None:
        test_ds = torch.utils.data.Subset(test_ds, range(min(args.max_samples, len(test_ds))))
        print(f"  (using {len(test_ds)} samples for quick diagnosis)")

    if model_name == "baseline":
        collate = collate_baseline
        from baseline.model import FusionBaseline
        from baseline.config import VIT_PATH, ECG_CKPT, HIDDEN_DIM, FREEZE_ENCODERS
        model = FusionBaseline(
            hidden_dim=HIDDEN_DIM,
            vit_path=VIT_PATH,
            ecg_ckpt_path=ECG_CKPT if os.path.exists(ECG_CKPT) else None,
            freeze_encoders=True,
        )
    elif model_name == "baseline2":
        collate = collate_baseline2
        from baseline2.model import ECGOnlyBaseline
        from baseline2.config import ECG_CKPT, HIDDEN_DIM, FREEZE_ENCODERS
        model = ECGOnlyBaseline(
            hidden_dim=HIDDEN_DIM,
            ecg_ckpt_path=ECG_CKPT if os.path.exists(ECG_CKPT) else None,
            freeze_encoder=True,
        )
    else:
        collate = collate_baseline3
        from baseline3.model import CXROnlyBaseline
        from baseline3.config import VIT_PATH, HIDDEN_DIM, FREEZE_ENCODERS
        model = CXROnlyBaseline(
            hidden_dim=HIDDEN_DIM,
            vit_path=VIT_PATH,
            freeze_encoder=True,
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Collect features and predictions
    cxr_feats_list, sig_feats_list, head_inputs_list, preds_list, targets_list = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            target = batch["target"].cpu().numpy()
            targets_list.append(target)

            if model_name == "baseline":
                cxr = batch["cxr"].to(device)
                signal = batch["signal"].to(device)
                cxr_feat = model.cxr_encoder(cxr)
                sig_feat = model.signal_encoder(signal)
                fused = torch.cat([cxr_feat, sig_feat], dim=-1)
                pred = model.head(fused).squeeze(-1)

                cxr_feats_list.append(cxr_feat.cpu().numpy())
                sig_feats_list.append(sig_feat.cpu().numpy())
                head_inputs_list.append(fused.cpu().numpy())
            elif model_name == "baseline2":
                signal = batch["signal"].to(device)
                sig_feat = model.signal_encoder(signal)
                pred = model.head(sig_feat).squeeze(-1)

                sig_feats_list.append(sig_feat.cpu().numpy())
                head_inputs_list.append(sig_feat.cpu().numpy())
            else:
                cxr = batch["cxr"].to(device)
                cxr_feat = model.cxr_encoder(cxr)
                pred = model.head(cxr_feat).squeeze(-1)

                cxr_feats_list.append(cxr_feat.cpu().numpy())
                head_inputs_list.append(cxr_feat.cpu().numpy())

            preds_list.append(pred.cpu().numpy())

    # Concatenate
    cxr_feats = np.concatenate(cxr_feats_list, axis=0) if cxr_feats_list else None
    sig_feats = np.concatenate(sig_feats_list, axis=0) if sig_feats_list else None
    head_inputs = np.concatenate(head_inputs_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Print diagnostics
    print(f"\n  [1] ENCODER OUTPUT (frozen pretrained)")
    if cxr_feats is not None:
        s = compute_stats("CXR encoder", cxr_feats, targets)
        print(f"      CXR feat:  std={s['feat_std']:.4f}  dims_with_var={s['n_dims_with_variance']}/512  max_corr_target={s['max_corr_with_target']:.4f}")
    if sig_feats is not None:
        s = compute_stats("ECG encoder", sig_feats, targets)
        print(f"      ECG feat:  std={s['feat_std']:.4f}  dims_with_var={s['n_dims_with_variance']}/512  max_corr_target={s['max_corr_with_target']:.4f}")

    print(f"\n  [2] HEAD INPUT (what the MLP head receives)")
    s = compute_stats("Head input", head_inputs, targets)
    print(f"      std={s['feat_std']:.4f}  dims_with_var={s['n_dims_with_variance']}  max_corr_target={s['max_corr_with_target']:.4f}")

    print(f"\n  [3] HEAD OUTPUT (final prediction)")
    pred_std = np.std(preds)
    pred_unique = len(np.unique(preds.round(decimals=4)))
    print(f"      std(pred)={pred_std:.6f}  unique_values≈{pred_unique}  range=[{preds.min():.4f}, {preds.max():.4f}]")

    # Verdict
    # Key: std_per_dim across SAMPLES - if 0, encoder output is same for all samples
    head_std_per_dim = np.std(head_inputs, axis=0)
    n_head_dims_var = np.sum(head_std_per_dim > 1e-6)
    print(f"\n  [4] DIAGNOSIS")
    if n_head_dims_var < 10:
        print(f"      *** COLLAPSE at ENCODER ***")
        print(f"      Encoder output is nearly identical across samples ({n_head_dims_var} of {head_inputs.shape[1]} dims vary).")
        print(f"      -> Frozen encoder not discriminative for oxygenation. Task may not align with pretraining.")
    elif pred_std < 0.01:
        print(f"      *** COLLAPSE at HEAD ***")
        print(f"      Encoder produces diverse features ({n_head_dims_var} dims vary) but head output is constant.")
        print(f"      -> Head learned to ignore input. Check head init, LR, or try larger head capacity.")
    else:
        print(f"      No full collapse: {n_head_dims_var} dims vary, pred_std={pred_std:.4f}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "baseline2", "baseline3", "all"], default="all")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--cxr_root", default="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg")
    parser.add_argument("--metadata_path", default="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--target_col", default="p2f_vent_fio2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick test")
    args = parser.parse_args()

    proj = Path(__file__).resolve().parent
    if args.csv_path is None:
        args.csv_path = str(proj / "cxr_supertable_waveform_matched.csv")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = ["baseline", "baseline2", "baseline3"] if args.model == "all" else [args.model]
    for m in models:
        diagnose_model(m, device, args)

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
