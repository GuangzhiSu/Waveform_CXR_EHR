"""Runtime path / dependency wiring for the ECG-CXR patient-temporal experiment.

This module must be imported *before* any ``health_multimodal`` or ``ecg_coca``
import. It:

1. Puts the workspace-local pip prefix (``artifacts/pylibs/``) on ``sys.path`` so the
   ``health_multimodal`` (Bio-ViL-T) and ``gdown`` packages installed there are
   importable inside the ``MedTVT-R1`` conda env (whose site-packages is
   read-only on this cluster).
2. Adds the vendored ``external/ECG-R1`` repo so ``ecg_coca`` (ECG-CoCa) imports.
3. Installs a lightweight ``skimage`` stub so importing ``health_multimodal.image``
   does not pull in scikit-image (only used by an image-IO path we do not need).

Importing this module is idempotent.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parent
PYLIBS = EXP_DIR / "artifacts" / "pylibs"
ECG_R1_DIR = EXP_DIR / "external" / "ECG-R1"


def _add_pylibs() -> None:
    # Match the active interpreter's site-packages dir name (pythonX.Y).
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sp = PYLIBS / "lib" / ver / "site-packages"
    for p in (sp, PYLIBS / "lib" / "python3.9" / "site-packages"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _add_ecg_r1() -> None:
    if ECG_R1_DIR.is_dir() and str(ECG_R1_DIR) not in sys.path:
        sys.path.insert(0, str(ECG_R1_DIR))


def _stub_skimage() -> None:
    if "skimage" in sys.modules:
        return
    try:
        import skimage  # noqa: F401  (real package available – nothing to stub)
        return
    except Exception:
        pass
    sk = types.ModuleType("skimage")
    skio = types.ModuleType("skimage.io")
    sk.io = skio
    sys.modules["skimage"] = sk
    sys.modules["skimage.io"] = skio


def setup() -> None:
    _add_pylibs()
    _add_ecg_r1()
    _stub_skimage()


setup()
