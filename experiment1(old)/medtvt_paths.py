"""
Resolve MedTVT-R1 repository root (directory that contains the ``llama/`` Python package).

Some checkouts only ship part of ``llama/`` (e.g. Cardio_llama) and omit ECG files. This module
can copy ``__init__.py``, ``xresnet1d_101.py``, and ``lab_encoder.py`` from the vendored bundle
next to this file (``medtvt_llama_ecg_bundle/``, from keke-nice/MedTVT-R1) into ``<MEDTVT>/llama/``
when that directory is writable.

``ensure_medtvt_on_syspath`` moves the resolved root to ``sys.path[0]`` and verifies
``llama.xresnet1d_101`` imports.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Vendored upstream: https://github.com/keke-nice/MedTVT-R1/tree/master/llama
# Do not bundle upstream ``__init__.py`` (it imports ``llama.llama`` which many partial checkouts lack).
_BUNDLE_DIR = Path(__file__).resolve().parent / "medtvt_llama_ecg_bundle"
_BUNDLE_PY_FILES = ("xresnet1d_101.py", "lab_encoder.py")
_MINIMAL_LLAMA_INIT = (
    '"""``llama`` package marker for ECG baselines (auto-written by medtvt_paths)."""\n'
)


def _has_xresnet_module(llama_dir: Path) -> bool:
    if not llama_dir.is_dir():
        return False
    if (llama_dir / "xresnet1d_101.py").is_file():
        return True
    pkg = llama_dir / "xresnet1d_101"
    if pkg.is_dir() and any(pkg.glob("*.py")):
        return True
    return False


def _is_usable_medtvt_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    llama_dir = root / "llama"
    return llama_dir.is_dir() and _has_xresnet_module(llama_dir)


def _ensure_minimal_llama_init(llama: Path) -> bool:
    """Avoid upstream ``__init__.py`` that requires ``llama.py`` when only ECG modules are present."""
    init_p = llama / "__init__.py"
    if not init_p.is_file():
        try:
            init_p.write_text(_MINIMAL_LLAMA_INIT, encoding="utf-8")
            return True
        except OSError:
            return False
    try:
        txt = init_p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if "from .llama import" in txt and not (llama / "llama.py").is_file():
        try:
            init_p.write_text(_MINIMAL_LLAMA_INIT, encoding="utf-8")
            print(
                "medtvt_paths: replaced llama/__init__.py (needed llama.py; not present) "
                "with minimal package marker",
                file=sys.stderr,
            )
            return True
        except OSError:
            return False
    return False


def _sync_llama_ecg_bundle_into_medtvt(medtvt_root: Path) -> bool:
    """
    Copy missing ECG / lab modules from ``medtvt_llama_ecg_bundle`` into ``medtvt_root/llama``.
    Returns True if any file was written/copied.
    """
    llama = medtvt_root / "llama"
    if not llama.is_dir() or not _BUNDLE_DIR.is_dir():
        return False
    changed = _ensure_minimal_llama_init(llama)
    need_x = not (llama / "xresnet1d_101.py").is_file()
    if need_x:
        for name in _BUNDLE_PY_FILES:
            src = _BUNDLE_DIR / name
            dst = llama / name
            if not src.is_file():
                continue
            try:
                shutil.copy2(src, dst)
                changed = True
            except OSError:
                continue
    if changed:
        print(
            f"medtvt_paths: synced ECG/lab llama modules from repo bundle into {llama}",
            file=sys.stderr,
        )
    return changed


def _try_medtvt_root(p: Path) -> str | None:
    """If ``p`` looks like MedTVT-R1 (has ``llama/``), sync bundle then return path if usable."""
    try:
        p = p.resolve()
    except OSError:
        return None
    if not p.is_dir() or not (p / "llama").is_dir():
        return None
    _sync_llama_ecg_bundle_into_medtvt(p)
    if _is_usable_medtvt_root(p):
        return str(p)
    return None


def _medtvt_root_from_path_hint(hint: str | None) -> str | None:
    if not hint or not str(hint).strip():
        return None
    s = str(hint).strip()
    if s.startswith("google/") or s.startswith("facebook/"):
        return None
    p = Path(s).expanduser()
    try:
        p = p.resolve()
    except OSError:
        return None
    if not p.exists():
        return None
    cur = p if p.is_dir() else p.parent
    for _ in range(32):
        got = _try_medtvt_root(cur)
        if got is not None:
            return got
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def resolve_medtvt_root(*path_hints: str | None) -> str:
    """Return absolute path to MedTVT-R1 checkout. Raises FileNotFoundError if none found."""
    for key in ("MEDTVT_ROOT", "MEDTVT_R1_ROOT"):
        raw = os.environ.get(key, "").strip()
        if raw:
            got = _try_medtvt_root(Path(raw).expanduser())
            if got is not None:
                return got

    for key in ("VIT_PATH", "ECG_CKPT"):
        raw = os.environ.get(key, "").strip()
        found = _medtvt_root_from_path_hint(raw or None)
        if found is not None:
            return found

    for h in path_hints:
        found = _medtvt_root_from_path_hint(h)
        if found is not None:
            return found

    exp_root = Path(__file__).resolve().parent
    repo_root = exp_root.parent

    candidates = [
        repo_root / "MedTVT-R1",
        repo_root.parent / "MedTVT-R1",
        Path("/hpc/group/kamaleswaranlab/MedTVT-R1"),
    ]
    for c in candidates:
        got = _try_medtvt_root(c)
        if got is not None:
            return got

    tried = ", ".join(str(c) for c in candidates)
    bundle = _BUNDLE_DIR
    raise FileNotFoundError(
        "MedTVT-R1 not found or llama/ is incomplete (need llama/xresnet1d_101.py for ECG). "
        f"Tried standard locations: {tried}. "
        "Set MEDTVT_ROOT, or pass --vit_path/--ecg_ckpt under your MedTVT-R1 tree. "
        "If your checkout is missing ECG files, copy from: "
        f"{bundle} into <MedTVT-R1>/llama/ (writable install), or git pull keke-nice/MedTVT-R1."
    )


def ensure_medtvt_on_syspath(*path_hints: str | None) -> str:
    import importlib

    root = resolve_medtvt_root(*path_hints)
    rp = str(Path(root).resolve())
    while rp in sys.path:
        sys.path.remove(rp)
    root_alt = str(Path(root))
    if root_alt != rp:
        while root_alt in sys.path:
            sys.path.remove(root_alt)
    sys.path.insert(0, rp)

    try:
        importlib.import_module("llama.xresnet1d_101")
    except ModuleNotFoundError as e:
        # torch / fastai missing in env — do not mask as MedTVT layout error
        if getattr(e, "name", None) in (
            "torch",
            "fastai",
            "fastai.layers",
            "fastai.core",
        ):
            raise
        raise ImportError(
            f"Cannot import llama.xresnet1d_101 with MedTVT at sys.path[0]={rp!r}. "
            f"Expected: {Path(rp) / 'llama' / 'xresnet1d_101.py'} "
            f"(needs torch + fastai). Bundle: {_BUNDLE_DIR}. "
            "If a PyPI package named `llama` exists: pip uninstall llama"
        ) from e

    return rp
