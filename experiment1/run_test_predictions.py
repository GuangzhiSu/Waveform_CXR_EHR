"""
Run test set only, save per-sample predictions vs ground truth.
Usage:
  python run_test_predictions.py --model baseline [--output predictions_baseline.csv]
  python run_test_predictions.py --model baseline2 [--output predictions_baseline2.csv]
  python run_test_predictions.py --model baseline3 [--output predictions_baseline3.csv]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline.dataset import WaveformCXREHRDataset


def collate_fn_baseline(batch):
    """Baseline: CXR + ECG."""
    cxr = torch.stack([b["cxr"] for b in batch])
    signal = torch.stack([b["signal"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    ids = [b["dicom_id"] for b in batch]
    return {"cxr": cxr, "signal": signal, "target": target, "dicom_id": ids}


def collate_fn_baseline2(batch):
    """Baseline2: ECG-only."""
    signal = torch.stack([b["signal"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    ids = [b["dicom_id"] for b in batch]
    return {"signal": signal, "target": target, "dicom_id": ids}


def collate_fn_baseline3(batch):
    """Baseline3: CXR-only."""
    cxr = torch.stack([b["cxr"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    ids = [b["dicom_id"] for b in batch]
    return {"cxr": cxr, "target": target, "dicom_id": ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "baseline2", "baseline3"], required=True)
    parser.add_argument("--output", default=None, help="Output CSV path (default: predictions_{model}.csv)")
    parser.add_argument("--checkpoint", default=None, help="Path to best.pt (default: {model}/output/best.pt)")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--cxr_root", default="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg")
    parser.add_argument("--metadata_path", default="/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv.gz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--target_col", default="p2f_vent_fio2")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    proj = Path(__file__).resolve().parent
    if args.csv_path is None:
        args.csv_path = str(proj / "cxr_supertable_waveform_matched.csv")
    if args.checkpoint is None:
        args.checkpoint = proj / args.model / "output" / "best.pt"
    else:
        args.checkpoint = Path(args.checkpoint)
    if args.output is None:
        args.output = proj / f"predictions_{args.model}.csv"
    else:
        args.output = Path(args.output)

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}, Device: {device}")

    # Dataset (same split logic as training)
    load_cxr = args.model in ["baseline", "baseline3"]
    load_signal = args.model in ["baseline", "baseline2"]
    full_ds = WaveformCXREHRDataset(
        csv_path=args.csv_path,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        target_col=args.target_col,
        split="test",  # CenterCrop for consistent eval
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
    print(f"Test samples: {n_test}")

    if args.model == "baseline":
        collate = collate_fn_baseline
        from baseline.model import FusionBaseline
        from baseline.config import VIT_PATH, ECG_CKPT, HIDDEN_DIM, FREEZE_ENCODERS
        model = FusionBaseline(
            hidden_dim=HIDDEN_DIM,
            vit_path=VIT_PATH,
            ecg_ckpt_path=ECG_CKPT if os.path.exists(ECG_CKPT) else None,
            freeze_encoders=FREEZE_ENCODERS,
        )
    elif args.model == "baseline2":
        collate = collate_fn_baseline2
        from baseline2.model import ECGOnlyBaseline
        from baseline2.config import ECG_CKPT, HIDDEN_DIM, FREEZE_ENCODERS
        model = ECGOnlyBaseline(
            hidden_dim=HIDDEN_DIM,
            ecg_ckpt_path=ECG_CKPT if os.path.exists(ECG_CKPT) else None,
            freeze_encoder=FREEZE_ENCODERS,
        )
    else:
        collate = collate_fn_baseline3
        from baseline3.model import CXROnlyBaseline
        from baseline3.config import VIT_PATH, HIDDEN_DIM, FREEZE_ENCODERS
        model = CXROnlyBaseline(
            hidden_dim=HIDDEN_DIM,
            vit_path=VIT_PATH,
            freeze_encoder=FREEZE_ENCODERS,
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    rows = []
    sample_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["dicom_id"]
            target = batch["target"].cpu().numpy()

            if args.model == "baseline":
                pred = model(batch["cxr"].to(device), batch["signal"].to(device))
            elif args.model == "baseline2":
                pred = model(batch["signal"].to(device))
            else:
                pred = model(batch["cxr"].to(device))

            pred = pred.cpu().numpy()
            for i, (did, t, p) in enumerate(zip(ids, target, pred)):
                err = float(p) - float(t)
                rows.append({
                    "sample_idx": sample_idx,
                    "dicom_id": did,
                    "target": round(float(t), 4),
                    "pred": round(float(p), 4),
                    "error": round(err, 4),
                    "abs_error": round(abs(err), 4),
                })
                sample_idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions to {args.output}")

    # Summary
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print("\nFirst 10 samples:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
