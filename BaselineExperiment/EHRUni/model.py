"""EHR-only temporal classifier: unified EHR encoder + pooled stats + head."""
import torch
import torch.nn as nn

from models.encoders.ehr import build_ehr_encoder


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 256, dropout: float = 0.3):
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
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class EHRClassificationBaseline(nn.Module):
    """
    Input: padded EHR sequence (B, T, F) + mask (B, T).
    Encode each row -> (B, T, D), then pool stats over valid rows and classify.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        embed_dim: int = 256,
        pooling_stats=("mean", "median", "max", "min", "std"),
        encoder_kind: str = "mlp",
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder_kind = encoder_kind.lower()
        self.encoder = build_ehr_encoder(self.encoder_kind, input_dim=input_dim, embed_dim=embed_dim)
        self.pooling_stats = tuple(pooling_stats)
        self.embed_dim = embed_dim
        self.head = ClassificationHead(
            input_dim=embed_dim * len(self.pooling_stats),
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
        )

    def _pool_one(self, x_valid: torch.Tensor):
        outs = []
        for s in self.pooling_stats:
            if s == "mean":
                outs.append(x_valid.mean(dim=0))
            elif s == "median":
                outs.append(x_valid.median(dim=0).values)
            elif s == "max":
                outs.append(x_valid.max(dim=0).values)
            elif s == "min":
                outs.append(x_valid.min(dim=0).values)
            elif s == "std":
                outs.append(x_valid.std(dim=0, unbiased=False))
            else:
                raise ValueError(f"Unsupported pooling stat: {s}")
        return torch.cat(outs, dim=0)

    def forward(self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor):
        b, t, f = ehr_seq.shape
        if self.encoder_kind == "transformer":
            z = self.encoder(ehr_seq, attention_mask=ehr_mask)
        elif self.encoder_kind == "contrastive":
            x = ehr_seq.view(b * t, f)
            z = self.encoder(x, normalize=False).view(b, t, self.embed_dim)
        else:
            x = ehr_seq.view(b * t, f)
            z = self.encoder(x).view(b, t, self.embed_dim)

        pooled = []
        for i in range(b):
            valid = z[i][ehr_mask[i]]
            if valid.size(0) == 0:
                valid = z[i, :1]
            pooled.append(self._pool_one(valid))
        pooled = torch.stack(pooled, dim=0)
        return self.head(pooled)


class EHRClassificationAverageBaseline(nn.Module):
    """
    Input: padded EHR sequence (B, T, F) + mask (B, T).
    Encode each row -> (B, T, D), then average valid rows and classify.
    This variant does not concatenate multiple pooling statistics.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        embed_dim: int = 256,
        encoder_kind: str = "mlp",
    ):
        super().__init__()
        self.encoder_kind = encoder_kind.lower()
        self.encoder = build_ehr_encoder(self.encoder_kind, input_dim=input_dim, embed_dim=embed_dim)
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, ehr_seq: torch.Tensor, ehr_mask: torch.Tensor):
        b, t, f = ehr_seq.shape
        if self.encoder_kind == "transformer":
            z = self.encoder(ehr_seq, attention_mask=ehr_mask)
        elif self.encoder_kind == "contrastive":
            x = ehr_seq.view(b * t, f)
            z = self.encoder(x, normalize=False).view(b, t, self.embed_dim)
        else:
            x = ehr_seq.view(b * t, f)
            z = self.encoder(x).view(b, t, self.embed_dim)

        pooled = []
        for i in range(b):
            valid = z[i][ehr_mask[i]]
            if valid.size(0) == 0:
                valid = z[i, :1]
            pooled.append(valid.mean(dim=0))
        pooled = torch.stack(pooled, dim=0)
        return self.classifier(pooled)
