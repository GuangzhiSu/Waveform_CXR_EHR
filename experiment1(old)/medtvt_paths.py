"""
Resolve MedTVT-R1 repository root (directory that contains the `llama/` Python package).

Layout changed: MedTVT-R1 may live next to this repo, inside it, or elsewhere.
Override with env: MEDTVT_ROOT or MEDTVT_R1_ROOT (absolute path to MedTVT-R1 checkout).
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_medtvt_root() -> str:
    """Return absolute path to MedTVT-R1 checkout. Raises FileNotFoundError if none found."""
    for key in ("MEDTVT_ROOT", "MEDTVT_R1_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            p = Path(raw).expanduser().resolve()
            if p.is_dir():
                return str(p)

    # experiment1(old)/medtvt_paths.py -> parents[1] = Waveform_CXR_EHR (repo root)
    exp_root = Path(__file__).resolve().parent
    repo_root = exp_root.parent

    candidates = [
        repo_root / "MedTVT-R1",
        repo_root.parent / "MedTVT-R1",
        Path("/hpc/group/kamaleswaranlab/MedTVT-R1"),
    ]
    for c in candidates:
        if c.is_dir():
            return str(c.resolve())

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "MedTVT-R1 not found (need directory containing llama/). "
        "Set MEDTVT_ROOT=/path/to/MedTVT-R1 or place checkout at one of: "
        f"{tried}"
    )


def ensure_medtvt_on_syspath() -> str:
    """Add MedTVT-R1 root to sys.path and return it."""
    import sys

    root = resolve_medtvt_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
