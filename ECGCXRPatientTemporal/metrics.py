"""Retrieval evaluation: cross-patient and within-patient temporal.

Gallery = unique CXR_t2 embeddings (deduped by cached-embedding row id) so that
identical target images do not appear as multiple competing targets.

  * cross-patient retrieval: rank the correct CXR_t2 among ALL unique CXR_t2 in
    the eval split. Reports Recall@1/5/10 and MRR.
  * within-patient temporal retrieval: rank the correct CXR_t2 among the same
    patient's CXR_t2 candidates only. Reports Temporal Recall@1 and Temporal MRR
    (over queries whose patient has >= 2 distinct CXR_t2 candidates).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import collate_fn as _default_collate_fn


@torch.no_grad()
def _collect(model, dataset, device, batch_size: int, collate_fn):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    qs, pids, c2_rows = [], [], []
    model.eval()
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        q, _, _ = model.encode(b)
        qs.append(q.float().cpu())
        pids.append(batch["patient_id"])
        c2_rows.append(batch["c2_row"])
    return (torch.cat(qs), torch.cat(pids).numpy(), torch.cat(c2_rows).numpy())


@torch.no_grad()
def _gallery(model, cxr_emb: np.ndarray, gallery_rows: np.ndarray, device, batch_size: int):
    vecs = []
    for s in range(0, len(gallery_rows), batch_size):
        rows = gallery_rows[s:s + batch_size]
        x = torch.from_numpy(cxr_emb[rows].astype(np.float32)).to(device)
        vecs.append(model.cxr_proj(x).float().cpu())
    return torch.cat(vecs) if vecs else torch.zeros(0)


def _rank_of_target(sims_row: np.ndarray, target_idx: int) -> int:
    """1-based rank of the target (ties broken pessimistically)."""
    target_score = sims_row[target_idx]
    return int((sims_row > target_score).sum()) + 1


@torch.no_grad()
def evaluate_retrieval(model, dataset, cxr_emb: np.ndarray, device,
                       batch_size: int = 256, collate_fn=None) -> dict:
    if collate_fn is None:
        collate_fn = _default_collate_fn
    if len(dataset) == 0:
        return {}
    q, pids, c2_rows = _collect(model, dataset, device, batch_size, collate_fn)

    # Unique gallery of CXR_t2 (by embedding row id).
    gallery_rows, inv = np.unique(c2_rows, return_inverse=True)
    target_gidx = inv  # per-query index into gallery_rows
    gallery = _gallery(model, cxr_emb, gallery_rows, device, batch_size)  # (G, P)
    # patient id per gallery entry (a CXR_t2 row belongs to a single patient)
    gallery_patient = np.empty(len(gallery_rows), dtype=np.int64)
    gallery_patient[target_gidx] = pids

    sims = (q @ gallery.t()).numpy()  # (Nq, G)
    Nq = sims.shape[0]

    # ---- cross-patient retrieval (full gallery) ----
    r1 = r5 = r10 = 0
    mrr = 0.0
    ranks = []
    for i in range(Nq):
        rank = _rank_of_target(sims[i], target_gidx[i])
        ranks.append(rank)
        r1 += rank <= 1
        r5 += rank <= 5
        r10 += rank <= 10
        mrr += 1.0 / rank
    cross = {
        "n_queries": Nq, "gallery_size": len(gallery_rows),
        "recall@1": r1 / Nq, "recall@5": r5 / Nq, "recall@10": r10 / Nq, "mrr": mrr / Nq,
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
    }

    # ---- within-patient temporal retrieval ----
    # group gallery indices by patient
    pat_to_gidx: dict[int, list] = {}
    for g, pid in enumerate(gallery_patient):
        pat_to_gidx.setdefault(int(pid), []).append(g)

    t_r1 = t_r5 = 0
    t_mrr = 0.0
    t_n = 0
    for i in range(Nq):
        pid = int(pids[i])
        cand = pat_to_gidx.get(pid, [])
        if len(cand) < 2:
            continue  # no within-patient temporal negative
        cand = np.asarray(cand)
        sub = sims[i, cand]
        tpos = int(np.where(cand == target_gidx[i])[0][0])
        rank = int((sub > sub[tpos]).sum()) + 1
        t_r1 += rank <= 1
        t_r5 += rank <= 5
        t_mrr += 1.0 / rank
        t_n += 1
    temporal = {
        "n_queries": t_n,
        "temporal_recall@1": (t_r1 / t_n) if t_n else float("nan"),
        "temporal_recall@5": (t_r5 / t_n) if t_n else float("nan"),
        "temporal_mrr": (t_mrr / t_n) if t_n else float("nan"),
    }

    return {"cross_patient": cross, "temporal": temporal}
