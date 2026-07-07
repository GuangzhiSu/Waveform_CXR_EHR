"""Precompute & cache frozen Bio-ViL-T (CXR) and ECG-CoCa (ECG) embeddings.

Since both encoders are frozen, we run them once over every unique CXR / ECG
referenced by the pairs file and cache the embeddings to disk. Training then
operates purely on these cached vectors (fast, no heavy encoders in the loop).

Disk loading (PIL decode + transform for CXR, waveform read for ECG) is the
bottleneck, so we parallelize it with a multi-worker ``DataLoader`` while the
frozen encoder runs batched forward passes on the GPU in the main process.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from io_utils import load_ecg  # noqa: E402
from runtime import get_device  # noqa: E402


# --------------------------------------------------------------------------- #
# Parallel-loading datasets (CPU work happens in DataLoader workers).
# --------------------------------------------------------------------------- #
class _CXRLoadDataset(Dataset):
    def __init__(self, ids, cxr_meta, transform, crop):
        self.ids = ids
        self.meta = cxr_meta
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        did = self.ids[i]
        try:
            img = Image.open(self.meta[did]["path"]).convert("L")
            return did, self.transform(img), True
        except Exception:
            return did, torch.zeros(3, self.crop, self.crop), False


class _ECGLoadDataset(Dataset):
    def __init__(self, ids, ecg_meta, sig_len):
        self.ids = ids
        self.meta = ecg_meta
        self.sig_len = sig_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        eid = self.ids[i]
        try:
            sig = load_ecg(self.meta[eid]["path"], target_len=self.sig_len).float()
        except Exception:
            sig = torch.zeros(12, self.sig_len)
        ok = bool(float(sig.abs().mean()) >= 1e-8)
        return eid, sig, ok


def _collate(batch):
    ids = [b[0] for b in batch]
    tensors = torch.stack([b[1] for b in batch])
    flags = [b[2] for b in batch]
    return ids, tensors, flags


def precompute_cxr(cxr_meta, ckpt, device, batch_size, num_workers, max_items=None):
    from encoders.biovil_t import BioViLTCXREncoder, BIOVIL_T_CENTER_CROP, get_biovil_t_transform

    enc = BioViLTCXREncoder(ckpt).to(device).eval()
    ids = [d for d, m in cxr_meta.items() if m.get("path_ok", True)]
    if max_items:
        ids = ids[:max_items]
    print(f"  CXR: embedding {len(ids):,} unique images (dim={enc.feat_dim}, workers={num_workers})",
          flush=True)

    ds = _CXRLoadDataset(ids, cxr_meta, get_biovil_t_transform(), BIOVIL_T_CENTER_CROP)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=(device.type == "cuda"))

    out_ids, out_vecs, n_fail, seen = [], [], 0, 0
    for bids, x, flags in loader:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            emb = enc(x).float().cpu().numpy()
        for j, ok in enumerate(flags):
            if ok:
                out_ids.append(bids[j])
                out_vecs.append(emb[j])
            else:
                n_fail += 1
        seen += len(bids)
        if seen % 4000 < batch_size:
            print(f"    CXR {seen:,}/{len(ids):,} (failed={n_fail})", flush=True)
    vecs = np.stack(out_vecs).astype(np.float32) if out_vecs else np.zeros((0, enc.feat_dim), np.float32)
    print(f"  CXR done: {len(out_ids):,} embedded, {n_fail} failed to load", flush=True)
    return out_ids, vecs


def precompute_ecg(ecg_meta, ckpt, config_path, device, batch_size, num_workers,
                   sig_len, max_items=None):
    from encoders.ecg_coca import ECGCoCaEncoder

    enc = ECGCoCaEncoder(ckpt, config_path).to(device).eval()
    ids = list(ecg_meta.keys())
    if max_items:
        ids = ids[:max_items]
    print(f"  ECG: embedding {len(ids):,} unique waveforms "
          f"(dim={enc.feat_dim}, len={sig_len}, workers={num_workers})", flush=True)

    ds = _ECGLoadDataset(ids, ecg_meta, sig_len)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=(device.type == "cuda"))

    out_ids, out_vecs, n_fail, seen = [], [], 0, 0
    for bids, x, flags in loader:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            emb = enc(x).float().cpu().numpy()
        for j, ok in enumerate(flags):
            if ok:
                out_ids.append(bids[j])
                out_vecs.append(emb[j])
            else:
                n_fail += 1
        seen += len(bids)
        if seen % 4000 < batch_size:
            print(f"    ECG {seen:,}/{len(ids):,} (failed={n_fail})", flush=True)
    vecs = np.stack(out_vecs).astype(np.float32) if out_vecs else np.zeros((0, enc.feat_dim), np.float32)
    print(f"  ECG done: {len(out_ids):,} embedded, {n_fail} failed/empty", flush=True)
    return out_ids, vecs


def _load_existing(ids_json, emb_npy):
    """Return (id->vec dict) for an existing cache, or empty if absent."""
    if not (Path(ids_json).is_file() and Path(emb_npy).is_file()):
        return {}
    ids = json.load(open(ids_json))
    vecs = np.load(emb_npy)
    return {i: vecs[k] for k, i in enumerate(ids)}


def _cache_paths(cache_dir: str | Path) -> dict[str, str]:
    cache_dir = Path(cache_dir)
    return {
        "cxr_emb": str(cache_dir / "cxr_emb.npy"),
        "cxr_ids": str(cache_dir / "cxr_ids.json"),
        "ecg_emb": str(cache_dir / "ecg_emb.npy"),
        "ecg_ids": str(cache_dir / "ecg_ids.json"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=[C.PAIRS_JSON],
                    help="One or more pairs files; their CXR/ECG meta are unioned.")
    ap.add_argument("--cache_dir", default=str(C.CACHE_DIR),
                    help="Directory for cxr/ecg embedding npy + id json outputs.")
    ap.add_argument("--merge", action="store_true",
                    help="Reuse the existing cache and only embed ids not already cached.")
    ap.add_argument("--biovil_ckpt", default=C.BIOVIL_T_CKPT)
    ap.add_argument("--ecg_ckpt", default=C.ECG_COCA_CKPT)
    ap.add_argument("--ecg_config", default=C.ECG_COCA_CONFIG)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cxr_batch_size", type=int, default=128)
    ap.add_argument("--ecg_batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--sig_len", type=int, default=C.ECG_SIG_LEN)
    ap.add_argument("--max_items", type=int, default=None, help="Cap per-modality for smoke tests.")
    ap.add_argument("--skip_cxr", action="store_true")
    ap.add_argument("--skip_ecg", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    paths = _cache_paths(args.cache_dir)
    print(f"=== precompute_embeddings (device={device}) ===", flush=True)
    cxr_meta: dict = {}
    ecg_meta: dict = {}
    for pf in args.pairs:
        data = json.load(open(pf))
        for k, v in data["cxr_meta"].items():
            cxr_meta.setdefault(k, v)
        for k, v in data["ecg_meta"].items():
            ecg_meta.setdefault(k, v)
    print(f"  union over {len(args.pairs)} pairs file(s): "
          f"unique CXR={len(cxr_meta):,}, unique ECG={len(ecg_meta):,}", flush=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_cxr:
        cached = _load_existing(paths["cxr_ids"], paths["cxr_emb"]) if args.merge else {}
        todo = {k: v for k, v in cxr_meta.items() if k not in cached}
        print(f"  CXR: {len(cached):,} cached, {len(todo):,} new to embed", flush=True)
        new_ids, new_vecs = precompute_cxr(todo, args.biovil_ckpt, device,
                                           args.cxr_batch_size, args.num_workers, args.max_items)
        ids = list(cached.keys()) + new_ids
        vecs = (np.stack([cached[i] for i in cached]) if cached else
                np.zeros((0, new_vecs.shape[1] if len(new_vecs) else C.CXR_FEAT_DIM), np.float32))
        vecs = np.concatenate([vecs, new_vecs], axis=0) if len(new_vecs) else vecs
        np.save(paths["cxr_emb"], vecs.astype(np.float32))
        json.dump(ids, open(paths["cxr_ids"], "w"))
        print(f"  Saved {paths['cxr_emb']} {vecs.shape}", flush=True)

    if not args.skip_ecg:
        cached = _load_existing(paths["ecg_ids"], paths["ecg_emb"]) if args.merge else {}
        todo = {k: v for k, v in ecg_meta.items() if k not in cached}
        print(f"  ECG: {len(cached):,} cached, {len(todo):,} new to embed", flush=True)
        new_ids, new_vecs = precompute_ecg(todo, args.ecg_ckpt, args.ecg_config, device,
                                           args.ecg_batch_size, args.num_workers,
                                           args.sig_len, args.max_items)
        ids = list(cached.keys()) + new_ids
        vecs = (np.stack([cached[i] for i in cached]) if cached else
                np.zeros((0, new_vecs.shape[1] if len(new_vecs) else C.ECG_FEAT_DIM), np.float32))
        vecs = np.concatenate([vecs, new_vecs], axis=0) if len(new_vecs) else vecs
        np.save(paths["ecg_emb"], vecs.astype(np.float32))
        json.dump(ids, open(paths["ecg_ids"], "w"))
        print(f"  Saved {paths['ecg_emb']} {vecs.shape}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
