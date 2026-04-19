"""ECG-only temporal baseline: per-waveform encoder + pooled stats + 3-layer MLP classifier."""
import torch
import torch.nn as nn

from models.encoders import build_ecg_encoder


class ClassificationHead(nn.Module):
    """3-layer MLP head for 3-class ARDS severity classification."""

    def __init__(self, input_dim, num_classes=3, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for j, m in enumerate(linears):
            if j == len(linears) - 1:
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ECGTemporalClassificationBaseline(nn.Module):
    """Encode each ECG in window, pool embeddings, classify."""

    def __init__(
        self,
        num_classes=3,
        hidden_dim=512,
        ecg_ckpt_path=None,
        ecg_encoder_kind: str = "cnn",
        freeze_encoder=True,
        pooling_stats=("mean", "median", "max", "min", "std"),
    ):
        super().__init__()
        self.ecg_encoder_kind = ecg_encoder_kind.lower()
        if self.ecg_encoder_kind == "transformer":
            self.signal_encoder = build_ecg_encoder(
                "transformer",
                hidden_dim=hidden_dim,
            )
        else:
            self.signal_encoder = build_ecg_encoder(
                "cnn",
                hidden_dim=hidden_dim,
                ckpt_path=ecg_ckpt_path,
                freeze=freeze_encoder,
            )
        self.hidden_dim = hidden_dim
        self.pooling_stats = tuple(pooling_stats)
        self.head = ClassificationHead(
            hidden_dim * len(self.pooling_stats),
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

    def _pool_one(self, z_valid: torch.Tensor):
        outs = []
        for s in self.pooling_stats:
            if s == "mean":
                outs.append(z_valid.mean(dim=0))
            elif s == "median":
                outs.append(z_valid.median(dim=0).values)
            elif s == "max":
                outs.append(z_valid.max(dim=0).values)
            elif s == "min":
                outs.append(z_valid.min(dim=0).values)
            elif s == "std":
                outs.append(z_valid.std(dim=0, unbiased=False))
            else:
                raise ValueError(f"Unsupported pooling stat: {s}")
        return torch.cat(outs, dim=0)

    def forward(self, signal_seq: torch.Tensor, signal_mask: torch.Tensor):
        b, t, c, l = signal_seq.shape
        if self.ecg_encoder_kind == "transformer":
            z = self.signal_encoder(signal_seq, attention_mask=signal_mask)
        else:
            x = signal_seq.view(b * t, c, l)
            z = self.signal_encoder(x).view(b, t, self.hidden_dim)

        pooled = []
        for i in range(b):
            valid = z[i][signal_mask[i]]
            if valid.size(0) == 0:
                valid = z[i, :1]
            pooled.append(self._pool_one(valid))
        pooled = torch.stack(pooled, dim=0)
        return self.head(pooled)
