"""Lightweight baseline: CXR + Signal + EHR encoders + MLP."""
from .model import FusionBaseline, CXREncoder, SignalEncoder, EHREncoder
from .dataset import WaveformCXREHRDataset
