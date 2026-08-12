"""
backend.py — pick pyFANTOM's CUDA or CPU backend at import time.
================================================================

`agents/to_agent_both.py` hard-binds the GPU in two ways:

    from pyFANTOM.CUDA import (...)      # module-level, fails with no GPU
    import cupy as cp                    # inside the BC / force selectors

This module centralises both so `to_agent_lite.py` is the same code running on
whichever backend is present.

    TO_BACKEND = auto | cuda | cpu       (default: auto)

`auto` uses CUDA when cupy imports *and* a device is actually visible — the
important distinction, since cupy imports fine on a machine with no GPU and
only fails when you touch a device.

Exports
-------
    BACKEND            "cuda" or "cpu"
    xp                 cupy or numpy — drop-in for the `cp` alias
    to_numpy(x)        device array -> numpy (no-op on CPU)
    HAS_LOCAL_FILTER   False on CPU; LocalFilter is CUDA-only
    <pyFANTOM symbols> StructuredMesh3D, MultiGrid, MinimumCompliance, ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as _np

__all__ = ["BACKEND", "xp", "to_numpy", "HAS_LOCAL_FILTER", "describe",
           "StructuredMesh3D", "GeneralMesh", "StructuredStiffnessKernel",
           "UniformStiffnessKernel", "GeneralStiffnessKernel", "MultiGrid", "CG",
           "SPSOLVE", "FiniteElement", "StructuredFilter3D", "GeneralFilter",
           "LocalFilter", "MinimumCompliance", "PGD", "MMA", "OC"]

# --------------------------------------------------------------------------- #
# make pyFANTOM importable (mirrors pipeline.py's path handling)
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
# Two supported layouts:
#   nested  — TO-Agents/website_lite/backend.py, agents/ one level up
#   flat    — repo root, agents/ right here (the standalone lite repo)
_env_root = os.environ.get("TO_AGENTS_ROOT", "").strip()
if _env_root:
    _REPO_ROOT = Path(_env_root).resolve()
elif (_HERE / "agents").is_dir():
    _REPO_ROOT = _HERE
else:
    _REPO_ROOT = _HERE.parent
_REPO_ROOT = _REPO_ROOT.resolve()

# The vendored checkout shadows itself: `pyFANTOM/pyFANTOM` on sys.path makes
# `import pyFANTOM` resolve to the inner package with no Physics submodule.
sys.path[:] = [p for p in sys.path if "pyFANTOM/pyFANTOM" not in p]
# Only prepend a vendored copy if one actually exists — when pyFANTOM is
# pip-installed (the standalone repo / Colab), site-packages already has it and
# inserting a non-existent path would just mask the real install.
_PYFANTOM = os.environ.get("PYFANTOM_PATH", str(_REPO_ROOT / "pyFANTOM"))
if Path(_PYFANTOM).is_dir() and _PYFANTOM not in sys.path:
    sys.path.insert(0, _PYFANTOM)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# choose the backend
# --------------------------------------------------------------------------- #
# Minimum free VRAM before we are willing to call CUDA usable. The default is
# deliberately conservative: a shared login-node GPU often has a few hundred MiB
# free, which passes a toy allocation and then dies on the first real solve.
MIN_VRAM_MB = float(os.environ.get("TO_MIN_VRAM_MB", "3072"))


def _cuda_usable() -> tuple[bool, str]:
    """cupy importing is not enough — a device has to actually be there AND
    have memory free. On a shared login node the GPU is frequently full, which
    is a different problem from having no GPU at all; say which."""
    try:
        import cupy as cp
    except Exception as e:
        return False, f"cupy not importable ({type(e).__name__})"
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        return False, f"no CUDA driver/device ({type(e).__name__})"
    if n < 1:
        return False, "cupy imports but no CUDA device is visible"
    try:
        cp.ones(1).sum()          # force a real allocation + kernel launch
    except Exception as e:
        if "OutOfMemory" in type(e).__name__:
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                return False, (f"GPU present but FULL — {free/2**20:.0f} MiB free "
                               f"of {total/2**20:.0f} MiB (someone else is using it)")
            except Exception:
                return False, "GPU present but out of memory (in use by another process)"
        return False, f"GPU present but unusable ({type(e).__name__})"
    # A 1-element allocation succeeding proves almost nothing: a shared GPU can
    # have 100 MiB free and still pass it, then OOM on the first real solve.
    # Require a usable floor before claiming CUDA.
    try:
        free, total = cp.cuda.runtime.memGetInfo()
    except Exception:
        return True, f"{n} CUDA device(s) (free VRAM unknown)"
    free_mb = free / 2**20
    if free_mb < MIN_VRAM_MB:
        return False, (f"GPU present but only {free_mb:.0f} MiB free "
                       f"(< TO_MIN_VRAM_MB={MIN_VRAM_MB:.0f}) — falling back to CPU")
    return True, f"{n} CUDA device(s), {free_mb:.0f} MiB free"


_requested = os.environ.get("TO_BACKEND", "auto").strip().lower()
if _requested not in ("auto", "cuda", "cpu"):
    raise ValueError(f"TO_BACKEND must be auto|cuda|cpu, got {_requested!r}")

_ok, _why = _cuda_usable()
if _requested == "cuda":
    if not _ok:
        raise RuntimeError(f"TO_BACKEND=cuda but CUDA is unavailable: {_why}")
    BACKEND = "cuda"
elif _requested == "cpu":
    BACKEND = "cpu"
else:
    BACKEND = "cuda" if _ok else "cpu"

_REASON = _why

# --------------------------------------------------------------------------- #
# bind the array module and the pyFANTOM symbols
# --------------------------------------------------------------------------- #
if BACKEND == "cuda":
    import cupy as xp                                       # noqa: F401
    from pyFANTOM.CUDA import (                             # noqa: F401
        StructuredMesh3D, GeneralMesh, StructuredStiffnessKernel,
        UniformStiffnessKernel, GeneralStiffnessKernel, MultiGrid, CG, SPSOLVE,
        FiniteElement, StructuredFilter3D, GeneralFilter, LocalFilter,
        MinimumCompliance, PGD, MMA, OC,
    )
    HAS_LOCAL_FILTER = True
else:
    xp = _np                                                # noqa: F401
    from pyFANTOM.CPU import (                              # noqa: F401
        StructuredMesh3D, GeneralMesh, StructuredStiffnessKernel,
        UniformStiffnessKernel, GeneralStiffnessKernel, MultiGrid, CG, SPSOLVE,
        FiniteElement, StructuredFilter3D, GeneralFilter,
        MinimumCompliance, PGD, MMA, OC,
    )
    # CUDA-only: pyFANTOM.CPU has no per-element LocalFilter. Callers must fall
    # back to StructuredFilter3D with a scalar r_min.
    LocalFilter = None
    HAS_LOCAL_FILTER = False


def to_numpy(x):
    """Device array -> numpy. Safe no-op for numpy input and plain scalars."""
    return x.get() if hasattr(x, "get") else x


def free_vram_mb() -> float | None:
    """Free VRAM in MiB, or None on CPU / if it can't be read."""
    if BACKEND != "cuda":
        return None
    try:
        import cupy as cp
        return cp.cuda.runtime.memGetInfo()[0] / 2**20
    except Exception:
        return None


# Mesh budgets keyed off actual hardware rather than a blanket CPU downgrade.
# Sized by VRAM ratio against the known-good 128x64x64 on a 40 GB A100.
# NOTE: only the A100 row is measured; the others are estimates and should be
# confirmed on real hardware before being promised to anyone.
_MESH_LADDER = [
    (24_000, (128, 64, 64)),   # >=24 GB  — A100/A6000 class: unchanged
    (10_000, (96, 48, 48)),    # >=10 GB  — T4/L4/3080 class  (ESTIMATE)
    (6_000,  (64, 32, 32)),    # >=6 GB   — small GPU         (ESTIMATE)
]
_CPU_MESH = (48, 24, 24)


def suggest_mesh(default=(128, 64, 64)) -> tuple[tuple[int, int, int], str]:
    """(nx, ny, nz) this machine can plausibly handle, plus why.

    Override with TO_MESH="nx,ny,nz" to pin it explicitly.
    """
    pin = os.environ.get("TO_MESH", "").strip()
    if pin:
        try:
            nx, ny, nz = (int(v) for v in pin.split(","))
            return (nx, ny, nz), f"pinned by TO_MESH={pin}"
        except Exception:
            raise ValueError(f'TO_MESH must look like "96,48,48", got {pin!r}')
    if BACKEND != "cuda":
        return _CPU_MESH, "CPU backend — reduced mesh so a run finishes"
    free = free_vram_mb()
    if free is None:
        return default, "CUDA, free VRAM unknown — using default"
    for floor, mesh in _MESH_LADDER:
        if free >= floor:
            note = "measured" if mesh == (128, 64, 64) else "ESTIMATE, unverified"
            return mesh, f"{free:.0f} MiB free ({note})"
    return _CPU_MESH, f"only {free:.0f} MiB free — using the CPU-sized mesh"


def describe() -> dict:
    mesh, why = suggest_mesh()
    return {"backend": BACKEND, "requested": _requested, "detection": _REASON,
            "array_module": xp.__name__, "local_filter": HAS_LOCAL_FILTER,
            "free_vram_mb": free_vram_mb(),
            "suggested_mesh": {"nx": mesh[0], "ny": mesh[1], "nz": mesh[2],
                               "elements": mesh[0] * mesh[1] * mesh[2], "why": why},
            "pyfantom_path": _PYFANTOM}


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
