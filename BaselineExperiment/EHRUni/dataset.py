"""
Dataset for EHR-only ARDS severity classification.
Loads EHR features, imputes missing with median, normalizes.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EHRClassificationDataset(Dataset):
    """Dataset: EHR features -> ARDS severity class (0=Severe, 1=Moderate, 2=Mild)."""

    def __init__(self, csv_path, feature_cols, scaler=None):
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.df = self.df[self.df["p2f_class"].notna()].copy()
        self.df["p2f_class"] = self.df["p2f_class"].astype(int)
        self.df = self.df.reset_index(drop=True)

        # Select columns that exist (deduplicate)
        self.feature_cols = list(dict.fromkeys(c for c in feature_cols if c in self.df.columns))
        missing = set(feature_cols) - set(self.feature_cols)
        if missing:
            print(f"  Warning: {len(missing)} cols missing: {list(missing)[:5]}...")

        # Encode gender if present
        if "gender" in self.feature_cols and "gender" in self.df.columns:
            self.df["gender"] = self.df["gender"].map({"Male": 0, "Female": 1}).fillna(-1).astype(float)

        X = self.df[self.feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        # Median imputation
        self.medians = X.median()
        X = X.fillna(self.medians)
        X = X.fillna(0)
        # Clip and sanitize
        X = X.clip(-1e10, 1e10)
        X = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if scaler is not None:
            self.scaler = scaler
            self.X = scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(X)
        self.X = np.nan_to_num(self.X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        self.input_dim = self.X.shape[1]
        self.labels = self.df["p2f_class"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        label = int(self.labels[idx])
        return {"ehr": x, "label": label}
