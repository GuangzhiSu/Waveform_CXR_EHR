"""
EHR-only baseline for ARDS severity classification.
3-layer NN (no pretrained encoder) to convert EHR -> embedding, then MLP classification head.
"""
import torch
import torch.nn as nn


class EHREncoder(nn.Module):
    """3-layer neural network: EHR raw features -> embedding."""

    def __init__(self, input_dim, hidden_dim=256, embed_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.net(x)  # (B, embed_dim)


class ClassificationHead(nn.Module):
    """MLP head for 3-class ARDS severity classification."""

    def __init__(self, input_dim, num_classes=3, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
    """EHR-only: 3-layer NN encoder -> embedding -> MLP -> class."""

    def __init__(
        self,
        input_dim,
        num_classes=3,
        embed_dim=256,
        hidden_dim=512,
    ):
        super().__init__()
        self.encoder = EHREncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.head = ClassificationHead(
            input_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim // 2,
        )

    def forward(self, ehr):
        emb = self.encoder(ehr)  # (B, embed_dim)
        return self.head(emb)  # (B, num_classes)
