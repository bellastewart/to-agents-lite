"""
doctor.py — "what can this machine actually run?"
=================================================

Run this FIRST, before touching the server:

    python doctor.py            # probe everything
    python doctor.py --no-net   # skip live model calls (config check only)

The full-fat demo needs four A100s.  Most people who find this project have
zero.  This script probes the three things that independently decide whether
the pipeline can run at all, and prints which deployment tier you qualify for:

    1. the TO solver   -- pyFANTOM CUDA (needs a GPU) or pyFANTOM CPU (numba)
    2. the renderer    -- k3d + Playwright/Chromium (CPU; no GPU required)
    3. the models      -- text / vision / judge, local or hosted

Exit code is 0 if at least one tier is runnable, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Nested (TO-Agents/website_lite) vs flat (standalone lite repo) layout.
_env_root = os.environ.get("TO_AGENTS_ROOT", "").strip()
if _env_root:
    REPO_ROOT = Path(_env_root).resolve()
elif (HERE / "agents").is_dir():
    REPO_ROOT = HERE
else:
    REPO_ROOT = HERE.parent
REPO_ROOT = REPO_ROOT.resolve()

def _probe_png(side: int = 64) -> bytes:
    """A valid greyscale-gradient PNG, built with stdlib only.

    Do NOT shrink this to 1x1: Gemini rejects degenerate images with
    HTTP 400 "Unable to process input image", which looks exactly like a
    broken key or model and sends you debugging the wrong thing. 64x64 is
    small enough to be free and large enough to be accepted everywhere.
    """
    import struct, zlib

    raw = b"".join(
        b"\x00" + bytes(((x * 4 + y * 2) % 256) for x in range(side))
        for y in range(side)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


TINY_PNG = _probe_png()

G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; B = "\033[1m"; X = "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    G = Y = R = B = X = ""

OK, WARN, BAD = f"{G}OK{X}", f"{Y}WARN{X}", f"{R}FAIL{X}"


def head(t):
    print(f"\n{B}{t}{X}\n{'-' * len(t)}")


def row(status, label, detail=""):
    print(f"  [{status}] {label}" + (f"\n         {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. python deps
# --------------------------------------------------------------------------- #
def check_deps():
    head("1. Python packages")
    required = {
        "numpy": "arrays", "scipy": "sparse solvers", "numba": "CPU kernels (JIT)",
        "fastapi": "web server", "uvicorn": "ASGI server", "openai": "OpenAI-compatible clients",
        "requests": "REST providers", "matplotlib": "density frames", "plotly": "setup diagram",
    }
    optional = {
        "cupy": "pyFANTOM CUDA backend (GPU only)",
        "k3d": "3D scene construction", "vtk": "mesh surfaces",
        "playwright": "headless screenshot capture",
        "autogen": "multi-agent group chat", "together": "Together SDK",
        "dotenv": "reads .env",
    }
    missing = []
    for mod, why in required.items():
        try:
            importlib.import_module(mod); row(OK, f"{mod:<12} {why}")
        except Exception as e:
            row(BAD, f"{mod:<12} {why}", f"{type(e).__name__}: {str(e)[:70]}")
            missing.append(mod)
    for mod, why in optional.items():
        try:
            importlib.import_module(mod); row(OK, f"{mod:<12} {why}")
        except Exception:
            row(WARN, f"{mod:<12} {why}", "not installed")
    return missing


# --------------------------------------------------------------------------- #
# 2. TO solver backends
# --------------------------------------------------------------------------- #
def check_solver():
    head("2. Topology-optimization solver (pyFANTOM)")
    # A vendored checkout is optional: pip-installed pyFANTOM lives in
    # site-packages with no such directory. Only add the path if it exists —
    # then decide by IMPORT, never by directory existence.
    pyf = Path(os.environ.get("PYFANTOM_PATH", REPO_ROOT / "pyFANTOM"))
    if pyf.is_dir() and str(pyf) not in sys.path:
        sys.path.insert(0, str(pyf))

    try:
        import pyFANTOM
        where = getattr(pyFANTOM, "__file__", "?")
        row(OK, "pyFANTOM importable", where)
    except Exception as e:
        hint = f"vendored copy at {pyf}" if pyf.is_dir() else "no vendored copy on disk"
        extra = ""
        if "sksparse" in str(e):
            extra = ("\n         sksparse is a HARD dependency of pyFANTOM.CPU "
                     "(solvers/CPU/_solvers.py imports it at module level).\n"
                     "         fix: apt-get install -y libsuitesparse-dev && "
                     "pip install scikit-sparse")
        row(BAD, "pyFANTOM not importable",
            f"{type(e).__name__}: {str(e)[:90]}\n         ({hint}; "
            f"pip install 'pyFANTOM @ git+https://github.com/bellastewart/"
            f"pyFANTOM_TO-Agents' or set PYFANTOM_PATH){extra}")
        return {"cuda": False, "cpu": False}

    cuda = cpu = False
    try:
        import cupy as cp
        cp.ones(1).sum()
        n = cp.cuda.runtime.getDeviceCount()
        row(OK, "CUDA backend", f"cupy works, {n} GPU(s) visible")
        cuda = True
    except Exception as e:
        row(WARN, "CUDA backend", f"unavailable ({type(e).__name__}) — CPU fallback required")

    # Import backend FIRST: it installs the sksparse stub that makes
    # pyFANTOM.CPU importable without scikit-sparse. Testing pyFANTOM.CPU
    # directly bypasses that and reports a failure the real app would not hit.
    # This is FATAL, not a warning. app.py and pipeline_lite.py both do a
    # module-level `import backend`, so if it raises here the server cannot
    # start at all -- no agent is ever constructed. Reporting it as a WARN
    # once let a build pass preflight and then die on launch with
    #   ImportError: cannot import name 'LocalFilter' from 'pyFANTOM.CUDA'
    # so the check now clears the backend flags it invalidates.
    backend_ok = False
    try:
        import backend as _b
        backend_ok = True
        row(OK, "backend.py", f"imports — selected '{_b.BACKEND}' backend")
        if _b.HAS_SKSPARSE:
            row(OK, "scikit-sparse", "present — CHOLMOD coarse solver available")
        else:
            row(WARN, "scikit-sparse",
                f"absent — stubbed; MultiGrid coarse solver falls back to "
                f"'{_b.COARSE_SOLVER}' (scipy). Fine for this pipeline; install "
                f"libsuitesparse-dev + scikit-sparse if you specifically want CHOLMOD.")
    except Exception as e:
        row(BAD, "backend.py",
            f"{type(e).__name__}: {str(e)[:80]}\n"
            f"         app.py imports this at module scope, so the server "
            f"CANNOT start until it is fixed.")

    try:
        CPU = importlib.import_module("pyFANTOM.CPU")
        needed = ["StructuredMesh3D", "GeneralMesh", "StructuredStiffnessKernel",
                  "UniformStiffnessKernel", "GeneralStiffnessKernel", "MultiGrid", "CG",
                  "SPSOLVE", "FiniteElement", "StructuredFilter3D", "GeneralFilter",
                  "MinimumCompliance", "PGD", "MMA", "OC"]
        miss = [n for n in needed if not hasattr(CPU, n)]
        if miss:
            row(WARN, "CPU backend", f"imported, but missing: {', '.join(miss)}")
        else:
            row(OK, "CPU backend", f"all {len(needed)} required symbols present")
        cpu = True
        # LocalFilter is CUDA-only; the lite build must not request it
        if not hasattr(CPU, "LocalFilter"):
            row(WARN, "LocalFilter", "CUDA-only — lite build must use StructuredFilter3D")
    except Exception as e:
        row(BAD, "CPU backend", f"{type(e).__name__}: {str(e)[:90]}")

    # Both paths run *through* backend.py, so neither is usable without it --
    # even when cupy and pyFANTOM.CPU import perfectly on their own, which is
    # exactly the case that slipped through before.
    if not backend_ok:
        row(BAD, "verdict", "backend.py did not import — no solver path is usable")
        cuda = cpu = False

    return {"cuda": cuda, "cpu": cpu}


# --------------------------------------------------------------------------- #
# 3. renderer
# --------------------------------------------------------------------------- #
def check_renderer():
    head("3. Screenshot renderer (k3d -> HTML -> headless Chromium)")
    ok = True
    for mod in ("k3d", "vtk"):
        try:
            importlib.import_module(mod); row(OK, mod)
        except Exception as e:
            row(BAD, mod, f"{type(e).__name__}"); ok = False
    try:
        from playwright.sync_api import sync_playwright
        row(OK, "playwright", "python package present")
        try:
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True); b.close()
            row(OK, "chromium", "launches headless — rendering needs NO GPU")
        except Exception as e:
            row(BAD, "chromium", f"{str(e)[:90]}\n         fix: playwright install chromium")
            ok = False
    except Exception:
        row(BAD, "playwright", "not installed — fix: pip install playwright && playwright install chromium")
        ok = False
    return ok


# --------------------------------------------------------------------------- #
# 4. models
# --------------------------------------------------------------------------- #
def check_models(live=True):
    head("4. Model backends (text / vision / judge)")
    try:
        import providers
    except Exception as e:
        row(BAD, "providers.py", f"{type(e).__name__}: {e}")
        return {}

    cfg = providers.describe_config()
    results = {}
    for role in ("text", "vision", "judge"):
        c = cfg.get(role, {})
        if "error" in c:
            row(BAD, f"{role:<7} config", c["error"]); results[role] = False; continue
        label = f"{role:<7} {c['provider']}/{c['model']}"
        if not live:
            row(WARN, label, f"{c['base_url']} — not probed (--no-net)")
            results[role] = None
            continue

        getter = {"text": providers.get_text_backend,
                  "vision": providers.get_vision_backend,
                  "judge": providers.get_judge_backend}[role]
        try:
            be = getter()
        except Exception as e:
            row(BAD, label, str(e)[:120]); results[role] = False; continue

        t0 = time.time()
        try:
            if role == "text":
                out = be.complete("Reply with the single word: ready",
                                  temperature=0, max_tokens=8)
            else:
                out = be.look("Reply with the single word: ready",
                              images=[TINY_PNG], temperature=0, max_tokens=8)
            row(OK, label, f"{time.time()-t0:.1f}s  reply={out[:40]!r}")
            results[role] = True
        except Exception as e:
            msg = str(e)
            hint = ""
            if "non-serverless" in msg:
                hint = "  <- model exists but needs a DEDICATED endpoint, not serverless"
            elif "429" in msg or "quota" in msg.lower():
                hint = ("  <- Gemini free tier = 20 requests/DAY per MODEL "
                        "(not per account). Switch THIS role to another model — each "
                        "has its own budget. Verified vision-capable: "
                        "gemini-3-flash-preview, gemini-3.1-flash-lite, "
                        "gemini-3.5-flash-lite, gemini-3.6-flash, gemini-flash-latest.")
            elif "leaked" in msg.lower():
                hint = "  <- key was revoked as leaked; rotate it"
            elif "Connection" in msg or "connect" in msg.lower():
                hint = "  <- nothing listening; is the local server up?"
            row(BAD, label, msg[:150] + hint)
            results[role] = False
    return results


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def verdict(solver, renderer, models):
    head("Verdict")
    have_text = models.get("text") is not False
    have_vision = models.get("vision") is not False
    have_judge = models.get("judge") is not False

    tiers = [
        ("Tier 2  full local (4 GPUs)",
         solver["cuda"] and renderer and have_text and have_vision and have_judge,
         "pyFANTOM CUDA + local vLLM for vision and judge"),
        ("Tier 1  one GPU + hosted models",
         solver["cuda"] and renderer and have_text and have_vision,
         "pyFANTOM CUDA on a single GPU (Colab/Kaggle T4), models over API"),
        ("Tier 0  ZERO GPU",
         solver["cpu"] and renderer and have_text and have_vision,
         "pyFANTOM CPU (numba) at reduced mesh, everything else over API"),
        ("Tier -1 solver only, no vision loop",
         solver["cpu"] and renderer and have_text,
         "runs the optimizer + renders, but no VLM critique/revision loop"),
    ]
    runnable = [t for t in tiers if t[1]]
    for name, ok_, why in tiers:
        print(f"  [{OK if ok_ else BAD}] {name:<34} {why}")

    print()
    if runnable:
        print(f"  {G}Best available: {runnable[0][0].strip()}{X}")
        return 0
    print(f"  {R}Nothing runnable.{X}  Most likely fix, in order:")
    if not solver["cpu"] and not solver["cuda"]:
        print("    - no usable pyFANTOM backend. If pyFANTOM imported but the CPU")
        print("      backend did not, the usual cause is scikit-sparse: either")
        print("      `apt-get install -y libsuitesparse-dev && pip install scikit-sparse`,")
        print("      or make sure backend.py is importable so its stub can load.")
    if not renderer:
        print("    - install the renderer: pip install playwright && playwright install chromium")
    if not have_vision:
        print("    - no working VISION model: set TO_VISION_PROVIDER / TO_VISION_MODEL")
        print("      (a vision model is required for the critique->revise loop)")
    return 1


def main():
    ap = argparse.ArgumentParser(description="Check what this machine can run.")
    ap.add_argument("--no-net", action="store_true", help="skip live model calls")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        # Beside the code first, then one level up — covers both layouts.
        for _cand in (HERE / ".env", REPO_ROOT / ".env"):
            if _cand.is_file():
                load_dotenv(_cand)
                break
    except ImportError:
        pass

    print(f"{B}TO-Agents lite — environment check{X}")
    print(f"repo root : {REPO_ROOT}")
    print(f"python    : {sys.version.split()[0]}  ({sys.executable})")

    check_deps()
    solver = check_solver()
    renderer = check_renderer()
    models = check_models(live=not args.no_net)
    sys.exit(verdict(solver, renderer, models))


if __name__ == "__main__":
    main()
