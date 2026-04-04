"""Lightweight baseline: CXR + Signal + EHR encoders + MLP.

Import ``CXREncoder`` from ``baseline.cxr_encoder`` (or this package's lazy exports) so
CXR-only code does not load ``baseline.model`` and the MedTVT ``llama`` stack.
"""
from .cxr_encoder import CXREncoder

__all__ = [
    "CXREncoder",
    "FusionBaseline",
    "SignalEncoder",
    "EHREncoder",
    "WaveformCXREHRDataset",
]


def __getattr__(name):
    if name == "FusionBaseline":
        from .model import FusionBaseline

        return FusionBaseline
    if name == "SignalEncoder":
        from .model import SignalEncoder

        return SignalEncoder
    if name == "EHREncoder":
        from .model import EHREncoder

        return EHREncoder
    if name == "WaveformCXREHRDataset":
        from .dataset import WaveformCXREHRDataset

        return WaveformCXREHRDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
