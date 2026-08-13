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
           "HAS_SKSPARSE", "COARSE_SOLVER", "THREADS",
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


def _cap_threads() -> str:
    """Stop numba/OpenMP oversubscribing on many-core machines.

    pyFANTOM's CPU kernels are numba ``prange`` loops. numba defaults to one
    thread per core, which is catastrophic for the small-to-medium meshes this
    build runs. Measured on a 244-core node, 128 elements, per iteration:

        244 threads   ~12-14 s
          4 threads     0.01 s      <- ~1400x faster

    The work per element is tiny, so thread launch and barrier costs dominate
    completely. A laptop or a Colab runtime (2-4 cores) never notices; a
    workstation or an HPC login node grinds to a halt.

    Must run BEFORE numba is first imported — numba reads this at import time.
    An explicit NUMBA_NUM_THREADS from the caller is always respected.
    """
    if os.environ.get("NUMBA_NUM_THREADS"):
        return f"NUMBA_NUM_THREADS={os.environ['NUMBA_NUM_THREADS']} (from environment)"

    # numba reads NUMBA_NUM_THREADS when it launches its thread pool and then
    # refuses to see a different value:
    #   RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once
    #                 the threads have been launched
    # Anything that imported numba before us (doctor.py probes it in its package
    # check) has already launched them, so setting the variable here would turn
    # a tuning nicety into an ImportError for the whole backend. Report what is
    # already in force instead.
    if "numba" in sys.modules:
        try:
            import numba
            n = numba.get_num_threads()
            os.environ.setdefault("OMP_NUM_THREADS", str(n))
            return (f"{n} threads (numba already imported; its pool was launched "
                    f"before backend.py, so the cap could not be applied)")
        except Exception:
            return "unknown (numba already imported before backend.py)"

    try:
        cores = len(os.sched_getaffinity(0))       # respects cgroup/slurm limits
    except AttributeError:
        cores = os.cpu_count() or 1
    cap = int(os.environ.get("TO_MAX_THREADS", "8"))
    n = max(1, min(cores, cap))
    os.environ["NUMBA_NUM_THREADS"] = str(n)
    # OpenMP is numba's threading layer here; leaving it uncapped re-introduces
    # the same oversubscription one layer down.
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    return f"{n} threads (of {cores} cores; cap TO_MAX_THREADS={cap})"


THREADS = _cap_threads()


def _ensure_sksparse() -> bool:
    """Make pyFANTOM importable even when scikit-sparse is missing.

    ``pyFANTOM/solvers/CPU/_solvers.py`` does a MODULE-LEVEL
    ``from sksparse.cholmod import cholesky``, re-exported by
    ``solvers/CPU/__init__.py``. So without scikit-sparse, ``import pyFANTOM.CPU``
    fails outright — the entire CPU backend is unreachable.

    scikit-sparse is source-only on PyPI (no wheels, any Python) and compiles
    against SuiteSparse; ``cholmod.h`` missing is the single most common install
    failure. But CHOLMOD is only ever used as a MultiGrid *coarse* solver, and
    ``splu``/``spsolve`` are pure-scipy alternatives.

    So when the real package is absent we register a stub that satisfies the
    import and raises only if CHOLMOD is genuinely invoked — and
    ``to_agent_lite`` then selects a scipy coarse solver instead.

    Returns True if the real scikit-sparse is present.
    """
    try:
        import sksparse.cholmod  # noqa: F401
        return True
    except Exception:
        pass

    import types

    def _cholesky(*_a, **_k):
        raise RuntimeError(
            "CHOLMOD was requested but scikit-sparse is not installed. "
            "Either install it (needs SuiteSparse headers: "
            "`apt-get install -y libsuitesparse-dev && pip install scikit-sparse`) "
            "or use a scipy coarse solver, e.g. MultiGrid(..., coarse_solver='splu')."
        )

    pkg = types.ModuleType("sksparse")
    pkg.__path__ = []                      # mark as a package
    chol = types.ModuleType("sksparse.cholmod")
    chol.cholesky = _cholesky
    chol.analyze = _cholesky
    pkg.cholmod = chol
    sys.modules.setdefault("sksparse", pkg)
    sys.modules.setdefault("sksparse.cholmod", chol)
    return False


HAS_SKSPARSE = _ensure_sksparse()
# Coarse solver for MultiGrid. pyFANTOM defaults to 'cholmod', which is exactly
# the thing that may not exist; fall back to scipy's sparse LU when it doesn't.
COARSE_SOLVER = os.environ.get(
    "TO_COARSE_SOLVER", "cholmod" if HAS_SKSPARSE else "splu").strip()


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
        FiniteElement, StructuredFilter3D, GeneralFilter,
        MinimumCompliance, PGD, MMA, OC,
    )
    # LocalFilter is OPTIONAL and must be imported separately.
    #
    # Not every pyFANTOM build ships it, even on CUDA -- the version pip
    # installs from the repo currently does not, and importing it alongside the
    # required symbols took the whole process down with
    #   ImportError: cannot import name 'LocalFilter' from 'pyFANTOM.CUDA'
    # before a single agent was built. There is already a working fallback for
    # exactly this case (see HAS_LOCAL_FILTER below and '# LITE 5' in
    # to_agent_lite.py, which drops to StructuredFilter3D with a mean r_min), so
    # a missing optional feature must degrade, not abort.
    try:
        from pyFANTOM.CUDA import LocalFilter                # noqa: F401
        HAS_LOCAL_FILTER = True
    except ImportError:
        LocalFilter = None
        HAS_LOCAL_FILTER = False
else:
    xp = _np                                                # noqa: F401
    from pyFANTOM.CPU import (                              # noqa: F401
        StructuredMesh3D, GeneralMesh, StructuredStiffnessKernel,
        UniformStiffnessKernel, GeneralStiffnessKernel, MultiGrid, CG, SPSOLVE,
        FiniteElement, StructuredFilter3D, GeneralFilter,
        MinimumCompliance, PGD, MMA, OC,
    )
    # pyFANTOM.CPU has never shipped a per-element LocalFilter. Callers fall
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
# These sized for "what fits in VRAM", which is the wrong question. A demo is
# bounded by patience, not capacity. Measured, from the recorded time_per_iter
# in website/runs plus this build's own runs:
#
#   A100  131,072 elements   0.34 s/iter        (measured, website/runs)
#   A100  8,388,608          10.0 s/iter        (measured, website/runs)
#   CPU   128,000            8.99 s/iter        (measured here)
#
# A T4 has roughly a fifth of an A100's memory bandwidth, so the old >=10 GB row
# (96x48x48 = 221,184 elements) works out near 5 MINUTES per optimization -- and
# a run does up to four of them. Reported from an actual T4 session as "still
# optimizing the initial design" long after it should have finished.
#
# So the ladder now targets roughly a minute per optimization on the class of
# GPU each row describes, which puts a full four-optimization run in the low
# single-digit minutes. Anyone who wants resolution over speed sets TO_MESH; the
# A100 row is unchanged because it was fast enough already.
_MESH_LADDER = [
    (24_000, (128, 64, 64)),   # >=24 GB — A100/A6000: measured at 0.34 s/iter
    (10_000, (64, 32, 32)),    # >=10 GB — T4/L4/3080: was 96x48x48 (~5 min/opt)
    (6_000,  (48, 24, 24)),    # >=6 GB  — small GPU
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
            "has_sksparse": HAS_SKSPARSE, "coarse_solver": COARSE_SOLVER,
            "threads": THREADS,
            "array_module": xp.__name__, "local_filter": HAS_LOCAL_FILTER,
            "free_vram_mb": free_vram_mb(),
            "suggested_mesh": {"nx": mesh[0], "ny": mesh[1], "nz": mesh[2],
                               "elements": mesh[0] * mesh[1] * mesh[2], "why": why},
            "pyfantom_path": _PYFANTOM}


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
