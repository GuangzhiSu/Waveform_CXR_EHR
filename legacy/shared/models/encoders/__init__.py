"""Unified encoder entrypoints for BaselineExperiment models.

Imports are lazy so that EHR-only training does not pull in ECG (medtvt_paths) or CXR deps.
"""

from __future__ import annotations

__all__ = [
    "CXREncoder",
    "SignalEncoder",
    "ECGTransformerEncoder",
    "build_ecg_encoder",
    "EHREncoder",
    "EHRMLPEncoder",
    "EHRTransformerEncoder",
    "EHRContrastiveEncoder",
    "build_ehr_encoder",
]


def __getattr__(name: str):
    if name == "CXREncoder":
        from .cxr import CXREncoder

        return CXREncoder
    if name in ("ECGTransformerEncoder", "SignalEncoder", "build_ecg_encoder"):
        from . import ecg as _ecg

        return getattr(_ecg, name)
    if name in (
        "EHREncoder",
        "EHRMLPEncoder",
        "EHRTransformerEncoder",
        "EHRContrastiveEncoder",
        "build_ehr_encoder",
    ):
        from . import ehr as _ehr

        return getattr(_ehr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
