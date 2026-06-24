"""Frozen Bio-ViL-T CXR image encoder wrapper.

Loads ``microsoft/BiomedVLP-BioViL-T`` image model (ResNet50 multi-image backbone
+ projector) from a local checkpoint and exposes a single-image embedding.

We use the global image embedding (``img_embedding``, 512-d) as the frozen CXR
feature; a trainable MLP projection on top lives in the contrastive model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

# Ensure pylibs / skimage-stub are wired before importing health_multimodal.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env_setup  # noqa: E402,F401

from health_multimodal.image.model.model import ImageModel  # noqa: E402
from health_multimodal.image.model.types import ImageEncoderType  # noqa: E402
from health_multimodal.image.data.transforms import (  # noqa: E402
    create_chest_xray_transform_for_inference,
)

BIOVIL_T_RESIZE = 512
BIOVIL_T_CENTER_CROP = 448
JOINT_FEATURE_SIZE = 128


def get_biovil_t_transform():
    """Canonical Bio-ViL-T inference transform (grayscale -> Resize/CenterCrop -> 3ch)."""
    return create_chest_xray_transform_for_inference(
        resize=BIOVIL_T_RESIZE, center_crop_size=BIOVIL_T_CENTER_CROP
    )


class BioViLTCXREncoder(nn.Module):
    """Frozen Bio-ViL-T image encoder. forward(x)->(B, 512) global img embedding."""

    feat_dim = 512

    def __init__(self, ckpt_path: str, freeze: bool = True):
        super().__init__()
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"Bio-ViL-T checkpoint not found: {ckpt_path}. "
                "Run download_weights.sh (downloads from HuggingFace)."
            )
        self.model = ImageModel(
            img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
            joint_feature_size=JOINT_FEATURE_SIZE,
            pretrained_model_path=ckpt_path,
        )
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.transform = get_biovil_t_transform()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) preprocessed CXR tensor -> (B, 512)."""
        out = self.model(x)
        emb = out.img_embedding
        return torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    def load_image(self, path: str) -> torch.Tensor:
        """Load a CXR jpg as grayscale and apply the Bio-ViL-T transform -> (3, H, W)."""
        img = Image.open(path).convert("L")
        return self.transform(img)
