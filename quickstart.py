"""
quickstart.py — set up and launch the demo, with almost no output.

The full notebook shows every step because it is also a debugging tool. This is
the other audience: someone who wants the website and nothing else. It does the
same work behind four progress lines and prints one URL.

    python quickstart.py

Everything is a SUBPROCESS on purpose. Installing packages and then importing
them in the same interpreter is what forces the "restart your runtime" step in
the full notebook — numpy in particular is already imported by Colab before any
cell runs, so a version change cannot take effect in-process. Shelling out to a
fresh interpreter for each stage sidesteps that entirely, which is what lets the
whole thing be a single cell.

Reads its configuration from the environment (the notebook cell sets the keys):
    GEMINI_API_KEY / OPENROUTER_API_KEY   whichever tier is wanted
    TO_MESH, TO_MAX_RUNS, ...             optional overrides
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("TO_WEB_PORT", "8765"))
CF_BIN = "/usr/local/bin/cloudflared"
CF_LOG = "/content/cloudflared.log" if Path("/content").is_dir() else str(HERE / "cloudflared.log")

_STEPS = 4
_t0 = time.time()


def step(n: int, label: str):
    print(f"  [{n}/{_STEPS}] {label:<34}", end="", flush=True)


def ok(note: str = ""):
    print(f"done{'  ' + note if note else ''}", flush=True)


def fail(msg: str, detail: str = ""):
    print("FAILED", flush=True)
    print(f"\n  {msg}", flush=True)
    if detail:
        print("\n" + "\n".join("    " + l for l in detail.strip().splitlines()[-25:]),
              flush=True)
    sys.exit(1)


def run(cmd, **kw):
    """Run a command, capturing everything. Returns CompletedProcess."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=HERE, **kw)


# --------------------------------------------------------------------------- #
# 1. packages
# --------------------------------------------------------------------------- #
def install():
    step(1, "installing packages")
    need = run([sys.executable, "-c",
                "import importlib.util as u,sys;"
                "sys.exit(0 if all(u.find_spec(m) for m in "
                "('pyFANTOM','k3d','vtk','playwright','autogen','instructor')) else 1)"])
    if need.returncode == 0:
        ok("(already present)")
        return

    p = run([sys.executable, "-m", "pip", "install", "-q",
             "-r", str(HERE / "requirements-lite.txt")])
    if p.returncode != 0:
        fail("Could not install the Python packages.", p.stdout + p.stderr)

    p = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "pyFANTOM @ git+https://github.com/bellastewart/pyFANTOM_TO-Agents"])
    if p.returncode != 0:
        fail("Could not install pyFANTOM (the topology-optimization solver).",
             p.stdout + p.stderr)

    # cupy only when a GPU is actually present. <14 because 14.x wants numpy>=2,
    # which breaks autogen-agentchat 0.2.40 and numba.
    if run(["nvidia-smi"]).returncode == 0:
        run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x<14"])

    run([sys.executable, "-m", "playwright", "install", "chromium"])
    ok()


# --------------------------------------------------------------------------- #
# 2. configuration
# --------------------------------------------------------------------------- #
def configure():
    step(2, "choosing models")
    sys.path.insert(0, str(HERE))
    try:
        import tiers
    except Exception as e:                                        # pragma: no cover
        fail(f"Could not load the tier definitions ({type(e).__name__}: {e}).")

    tiers.normalise_keys()
    tier = tiers.detect_tier()
    if tier is None:
        fail("No API key found.\n"
             "  Add ONE of these to the notebook's Secrets (the key icon on the left):\n"
             "    GEMINI_API_KEY      free, no card    https://aistudio.google.com/apikey\n"
             "    OPENROUTER_API_KEY  needs a card     https://openrouter.ai/keys")
    tiers.apply_tier(tier)
    os.environ.setdefault("TO_BACKEND", "auto")
    os.environ.setdefault("TO_WEB_PORT", str(PORT))
    ok(f"({tier} tier)")
    return tier


# --------------------------------------------------------------------------- #
# 3. server
# --------------------------------------------------------------------------- #
def serve():
    step(3, "starting the pipeline")
    subprocess.run(["fuser", "-k", "-n", "tcp", str(PORT)], capture_output=True)
    time.sleep(1)

    log = HERE / "server.log"
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=HERE,
                            env=os.environ.copy(),
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)
    import socket
    deadline = time.time() + 420
    while time.time() < deadline:
        if proc.poll() is not None:
            fail("The server stopped while starting up.", log.read_text())
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                ok()
                return
        time.sleep(2)
    fail("The server did not start within 7 minutes.", log.read_text())


# --------------------------------------------------------------------------- #
# 4. public link
# --------------------------------------------------------------------------- #
def tunnel() -> str | None:
    step(4, "opening a public link")

    # /usr/local/bin is writable on Colab (root) but not everywhere, so fall
    # back to somewhere we certainly own rather than failing at the last step.
    binary = next((p for p in (CF_BIN, str(HERE / "cloudflared")) if Path(p).exists()), None)
    if binary is None:
        url = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
               "cloudflared-linux-amd64")
        last = None
        for cand in (CF_BIN, str(HERE / "cloudflared")):
            try:
                urllib.request.urlretrieve(url, cand)
                os.chmod(cand, 0o755)
                binary = cand
                break
            except Exception as e:
                last = e
        if binary is None:
            fail(f"Could not download the tunnel helper "
                 f"({type(last).__name__}: {last}).")

    subprocess.Popen([binary, "tunnel", "--url", f"http://localhost:{PORT}"],
                     stdout=open(CF_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(45):
        time.sleep(1)
        try:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                          Path(CF_LOG).read_text())
        except FileNotFoundError:
            continue
        if m:
            ok()
            return m.group(0)
    fail("The public link did not come up.", Path(CF_LOG).read_text())


def main():
    print(f"\nPreparing your session — this takes about four minutes.\n", flush=True)
    install()
    tier = configure()
    serve()
    url = tunnel()

    mins = (time.time() - _t0) / 60
    bar = "─" * 58
    print(f"\n  {bar}")
    print(f"   Ready in {mins:.1f} minutes. Open this:\n")
    print(f"     {url}\n")
    print(f"  {bar}")
    print("   The link is public while this notebook is running, and stops")
    print("   working when the runtime disconnects. Keep this tab open.\n")


if __name__ == "__main__":
    main()
