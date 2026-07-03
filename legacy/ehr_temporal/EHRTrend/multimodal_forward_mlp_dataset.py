"""Single-timestep EHR+CXR+ECG aligned to enriched row; forward [t+12h,t+24h] s2f/p2f change labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import FeatureSpec, _parse_mapping, _to_bool
from forward_mlp_dataset import _assign_group, _compute_forward_labels_for_group
from multimodal_nextstep_dataset import _to_has_flag

_BE = Path(__file__).resolve().parents[1] / "BaselineExperiment"
for _p in (_BE, _BE / "CXRUni"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cxr_classification.dataset import (  # noqa: E402
    _first_non_empty_study_id,
    _norm_dicom_id,
    get_cxr_path,
    load_cxr,
)
from ECGUni.dataset import load_ecg, normalize_ecg_per_lead  # noqa: E402


class MultimodalForwardMLPDataset(Dataset):
    """
    One sample = anchor time ``t`` from ``anchor_csv`` (same keys as next-step), EHR percentiles
    from that row, CXR/ECG from ``enriched_csv`` row at ``(hadm_id, index)``. Labels = forward
    severity change in ``[t+12h, t+24h]`` (same encoding as ``EHRForwardChangeDataset``).
    """

    def __init__(
        self,
        anchor_csv: str,
        enriched_csv: str,
        schema_csv: str,
        enriched_csv_for_group: Optional[str] = None,
        cxr_root: str = "/hpc/group/kamaleswaranlab/mimic_cxr/mimic_cxr_jpg",
        metadata_path: Optional[str] = None,
        forward_min_hours: int = 12,
        forward_max_hours: int = 24,
        ecg_target_len: int = 5000,
        cxr_split: str = "train",
        imagenet_normalize: bool = True,
        normalize_ecg_per_lead: bool = True,
    ):
        self.cxr_root = cxr_root
        self.metadata_path = metadata_path
        self.ecg_target_len = int(ecg_target_len)
        self.cxr_split = cxr_split
        self.imagenet_normalize = imagenet_normalize
        self.normalize_ecg_per_lead = normalize_ecg_per_lead

        main = pd.read_csv(anchor_csv, low_memory=False)
        for c in ("hadm_id", "index", "has_s2f_vent_fio2", "has_p2f_vent_fio2", "s2f_vent_fio2_severity", "p2f_vent_fio2_severity"):
            if c not in main.columns:
                raise ValueError(f"anchor_csv missing column: {c}")

        main["hadm_id"] = pd.to_numeric(main["hadm_id"], errors="coerce")
        main["_ref_time"] = pd.to_datetime(main["index"], errors="coerce")
        main = main[main["hadm_id"].notna() & main["_ref_time"].notna()].copy()
        main["hadm_id"] = main["hadm_id"].astype(np.int64)
        grp_src = enriched_csv_for_group or enriched_csv
        main = _assign_group(main, grp_src)

        modal = pd.read_csv(enriched_csv, low_memory=False)
        for c in ("hadm_id", "index", "dicom_id", "subject_id", "wf_File_Path"):
            if c not in modal.columns:
                raise ValueError(f"enriched_csv missing column: {c}")
        modal = modal.copy()
        modal["hadm_id"] = pd.to_numeric(modal["hadm_id"], errors="coerce")
        modal["_ref_time"] = pd.to_datetime(modal["index"], errors="coerce")
        modal = modal[modal["hadm_id"].notna() & modal["_ref_time"].notna()].copy()
        modal["hadm_id"] = modal["hadm_id"].astype(np.int64)
        keep = ["hadm_id", "_ref_time", "dicom_id", "subject_id", "wf_File_Path"]
        modal = modal[keep].drop_duplicates(subset=["hadm_id", "_ref_time"], keep="first")

        if metadata_path and os.path.isfile(metadata_path):
            meta = pd.read_csv(metadata_path, usecols=["dicom_id", "study_id"])
            meta = meta.drop_duplicates(subset=["dicom_id"], keep="first")
            meta["dicom_id"] = meta["dicom_id"].map(_norm_dicom_id)
            modal["dicom_id"] = modal["dicom_id"].map(_norm_dicom_id)
            modal = modal.merge(meta[["dicom_id", "study_id"]], on="dicom_id", how="left")
        elif "study_id" not in modal.columns:
            modal["study_id"] = np.nan

        mcols = [c for c in ("dicom_id", "subject_id", "wf_File_Path", "study_id") if c in modal.columns]
        modal_m = modal[["hadm_id", "_ref_time"] + mcols].rename(columns={c: f"_m_{c}" for c in mcols})
        merged = main.merge(modal_m, on=["hadm_id", "_ref_time"], how="left")

        schema = pd.read_csv(schema_csv)
        use_schema = schema[schema["use_as_input"].map(_to_bool)].copy()
        specs: List[FeatureSpec] = []
        common_cols = set(merged.columns)
        for _, r in use_schema.iterrows():
            f = str(r["Features"])
            if f not in common_cols:
                continue
            specs.append(
                FeatureSpec(
                    name=f,
                    mapping=_parse_mapping(r.get("onehot_mapping")),
                    default_raw=str(r.get("imputation_params_default_impute", "median")),
                )
            )
        if not specs:
            raise ValueError("No schema input features in merged anchor/enriched rows")

        self.feature_specs = specs
        self.feature_cols = [s.name for s in specs]
        self.input_dim = len(self.feature_cols)
        self.df = merged.reset_index(drop=True)

        self.num = self._numeric_frame(self.df)
        self.fill_values = self._build_fill_values()
        self.sorted_values = self._build_sorted_values()
        self.x_pct = self._to_percentiles(self.num)

        t_ns = self.df["_ref_time"].astype("int64").to_numpy()
        grp = self.df["_group_id"].to_numpy(dtype=np.int64)
        s_sev = pd.to_numeric(self.df["s2f_vent_fio2_severity"], errors="coerce").to_numpy()
        p_sev = pd.to_numeric(self.df["p2f_vent_fio2_severity"], errors="coerce").to_numpy()
        has_s = self.df["has_s2f_vent_fio2"].map(_to_has_flag).to_numpy(dtype=bool)
        has_p = self.df["has_p2f_vent_fio2"].map(_to_has_flag).to_numpy(dtype=bool)

        order = np.lexsort((t_ns, grp))
        g_sorted = grp[order]
        t_sorted = t_ns[order]
        s_lab = np.full(len(self.df), -1, dtype=np.int64)
        p_lab = np.full(len(self.df), -1, dtype=np.int64)
        uniq, starts = np.unique(g_sorted, return_index=True)
        for j, _gid in enumerate(uniq):
            a = starts[j]
            b = starts[j + 1] if j + 1 < len(starts) else len(g_sorted)
            sl = order[a:b]
            tt = t_sorted[a:b]
            s_lab[sl] = _compute_forward_labels_for_group(
                tt, s_sev[sl], has_s[sl], forward_min_hours, forward_max_hours
            )
            p_lab[sl] = _compute_forward_labels_for_group(
                tt, p_sev[sl], has_p[sl], forward_min_hours, forward_max_hours
            )

        self.s_forward = s_lab
        self.p_forward = p_lab
        self.has_s = has_s
        self.has_p = has_p

        n = len(self.df)
        print(
            f"  MultimodalForwardMLPDataset: n={n:,}, ehr_features={self.input_dim}, "
            f"forward=[t+{forward_min_hours}h, t+{forward_max_hours}h]"
        )
        sup_s = int((self.has_s & (self.s_forward >= 0)).sum())
        sup_p = int((self.has_p & (self.p_forward >= 0)).sum())
        print(f"    supervised rows: s2f_head={sup_s:,}, p2f_head={sup_p:,}")

    def _numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for s in self.feature_specs:
            col = df[s.name]
            if s.mapping is not None:
                out[s.name] = col.astype(str).map(s.mapping)
            else:
                out[s.name] = pd.to_numeric(col, errors="coerce")
        x = pd.DataFrame(out)
        return x.replace([np.inf, -np.inf], np.nan)

    def _build_fill_values(self) -> Dict[str, float]:
        fills = {}
        for s in self.feature_specs:
            col = self.num[s.name]
            med = float(col.median()) if col.notna().any() else 0.0
            d = str(s.default_raw).strip().lower()
            if d in {"", "nan", "median"}:
                fills[s.name] = med
            else:
                try:
                    fills[s.name] = float(d)
                except ValueError:
                    fills[s.name] = med
        return fills

    def _build_sorted_values(self) -> Dict[str, np.ndarray]:
        out = {}
        for s in self.feature_specs:
            v = self.num[s.name].dropna().to_numpy(dtype=np.float64)
            if v.size == 0:
                out[s.name] = np.array([self.fill_values[s.name]], dtype=np.float64)
            else:
                out[s.name] = np.sort(v)
        return out

    def _to_percentiles(self, xdf: pd.DataFrame) -> np.ndarray:
        n = len(xdf)
        o = np.zeros((n, self.input_dim), dtype=np.float32)
        for j, s in enumerate(self.feature_specs):
            vals = xdf[s.name].to_numpy(dtype=np.float64)
            fill = self.fill_values[s.name]
            vals = np.where(np.isfinite(vals), vals, fill)
            arr = self.sorted_values[s.name]
            idx = np.searchsorted(arr, vals, side="right")
            o[:, j] = (idx / max(len(arr), 1)).astype(np.float32)
        return o

    def _resize_ecg(self, sig: torch.Tensor) -> torch.Tensor:
        c, L = sig.shape
        T = self.ecg_target_len
        if L == T:
            return sig
        if L > T:
            start = max(0, (L - T) // 2)
            return sig[:, start : start + T]
        out = torch.zeros(c, T, dtype=sig.dtype, device=sig.device)
        out[:, :L] = sig
        return out

    def _load_modalities(self, row: pd.Series):
        dicom_id = row.get("_m_dicom_id", row.get("dicom_id"))
        subject_id = row.get("_m_subject_id", row.get("subject_id"))
        study_id = row.get("_m_study_id", row.get("study_id", np.nan))
        if pd.isna(study_id) or (isinstance(study_id, str) and str(study_id).strip() == ""):
            study_id = _first_non_empty_study_id(row)
        has_cxr = pd.notna(dicom_id) and pd.notna(subject_id)
        if has_cxr:
            dicom_id = _norm_dicom_id(dicom_id)
            path = get_cxr_path(dicom_id, int(subject_id), study_id, self.cxr_root)
            has_cxr = bool(path and os.path.isfile(path))
            if has_cxr:
                cxr = load_cxr(path, self.cxr_split, imagenet_normalize=self.imagenet_normalize)
            else:
                cxr = torch.zeros(3, 224, 224)
        else:
            cxr = torch.zeros(3, 224, 224)
            has_cxr = False

        wf = row.get("_m_wf_File_Path", row.get("wf_File_Path"))
        has_ecg = pd.notna(wf) and str(wf).strip() and os.path.isfile(str(wf).strip())
        if has_ecg:
            sig = load_ecg(str(wf).strip())
            if self.normalize_ecg_per_lead:
                sig = normalize_ecg_per_lead(sig)
            sig = self._resize_ecg(sig.float())
        else:
            sig = torch.zeros(12, self.ecg_target_len)
            has_ecg = False
        return cxr, sig, has_cxr, has_ecg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        cxr, ecg, m_cxr, m_ecg = self._load_modalities(row)
        sy = int(self.s_forward[idx])
        py = int(self.p_forward[idx])
        return {
            "ehr": torch.from_numpy(self.x_pct[idx]).float(),
            "cxr": cxr,
            "ecg": ecg,
            "cxr_valid": m_cxr,
            "ecg_valid": m_ecg,
            "s2f_y": sy,
            "p2f_y": py,
            "s2f_valid": bool(self.has_s[idx]) and sy >= 0,
            "p2f_valid": bool(self.has_p[idx]) and py >= 0,
        }
