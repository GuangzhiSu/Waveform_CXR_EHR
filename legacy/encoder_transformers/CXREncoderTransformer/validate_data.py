"""Smoke-test CXRLabeledCatalogDataset loading (no training)."""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(PROJECT_ROOT / "EHRWindowTransformer"))

from common import collate_cxr_window_batch  # noqa: E402
from config import *  # noqa: F401,F403,E402
from cxr_labeled_dataset import CXRLabeledCatalogDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cxr_labeled_csv", default=CXR_CATALOG_LABELED_CSV)
    p.add_argument("--cxr_root", default=CXR_ROOT)
    p.add_argument("--metadata_path", default=METADATA_PATH)
    p.add_argument("--max_samples", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--no_hours_filter", action="store_true")
    args = p.parse_args()

    print("=== CXRLabeledCatalogDataset validate ===")
    ds = CXRLabeledCatalogDataset(
        labeled_csv=args.cxr_labeled_csv,
        cxr_root=args.cxr_root,
        metadata_path=args.metadata_path,
        require_hours_in_window=not args.no_hours_filter,
    )
    n = min(args.max_samples, len(ds))
    loader = DataLoader(
        range(n),
        batch_size=args.batch_size,
        collate_fn=lambda idxs: collate_cxr_window_batch([ds[int(i)] for i in idxs]),
    )
    valid_slots = 0
    total_slots = 0
    for batch in loader:
        m = batch["cxr_mask"]
        total_slots += m.numel()
        valid_slots += int(m.sum())
        print(
            f"  batch cxr_seq={tuple(batch['cxr_seq'].shape)}  "
            f"valid_cxr={int(m.sum())}/{m.numel()}  "
            f"n_s2f={int(batch['anchor_has_s2f'].sum())}  n_p2f={int(batch['anchor_has_p2f'].sum())}"
        )
    print(f"OK: {n} anchor samples; loadable slots {valid_slots}/{total_slots}")
    if valid_slots == 0:
        print("WARNING: no loadable CXR on this node — check cxr_root / metadata_path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
