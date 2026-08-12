"""
app.py — TO-Agents demo web server
==================================

A FastAPI layer over ``pipeline.py``.  The user pastes one verbose technical
description; the server runs the same multi-agent pipeline the notebook runs and
streams a *narrated* view of it to the browser:

  * who is speaking + a plain-English summary of what they're actually doing
    (parsed from the agents' stdout),
  * the intended problem setup (box + BCs + loads) drawn before the TO begins,
  * the optimization "movie" — density-projection frames + a live compliance
    curve, streamed as the optimizer iterates,
  * the final 3D depth/stress result.

Endpoints
  GET  /                      -> single-page UI
  GET  /examples              -> built-in example descriptions
  POST /run                   -> Server-Sent Events stream of the run
  POST /stop                  -> cooperatively terminate the running pipeline
  GET  /runs/{run_id}/{path}  -> generated artifacts (diagrams, frames, shots)

Only one run happens at a time (the pipeline uses a process-global cwd + stdout
capture), guarded by ``_RUN_LOCK``.
"""

import io
import json
import os
import queue
import re
import threading
import time
import contextlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Turn on the optimizer's live-frame hook for website runs (notebook stays off).
os.environ.setdefault("TO_FRAME_EVERY", os.environ.get("TO_FRAME_EVERY", "5"))

print("[app] importing pipeline (building agents)...")
# LITE: pipeline_lite exposes the same surface (manager, reset_chat, run_chat,
# EXAMPLES, AGENT_LABELS) but resolves every model and the TO backend from env
# instead of pinning them to local GPUs. Set TO_LITE=0 for the original module.
if os.environ.get("TO_LITE", "1") != "0":
    import pipeline_lite as pipeline  # noqa: E402
else:
    import pipeline  # noqa: E402
import viz        # noqa: E402
print("[app] pipeline ready.")

app = FastAPI(title="TO-Agents Demo")

_RUN_LOCK = threading.Lock()
_CURRENT = {"run_dir": None}   # the active run's working directory (for /stop)


# --------------------------------------------------------------------------- #
# SSE + content helpers
# --------------------------------------------------------------------------- #
def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") in ("image_url", "image"):
                    parts.append("[image]")
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


class _QueueWriter(io.TextIOBase):
    """stdout replacement: emits 'log' events AND parsed 'action' events."""

    def __init__(self, q):
        self._q = q
        self._buf = ""

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def _emit(self, line):
        self._q.put(("log", line))
        act = _parse_action(line)
        if act:
            self._q.put(("action", act))

    def flush(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


# --------------------------------------------------------------------------- #
# stdout -> plain-English "what are they doing" parser
# --------------------------------------------------------------------------- #
# Each rule: (compiled regex, agent, kind, template).  Template uses match groups.
_ACTION_RULES = [
    # pydantic
    (r"^Time:\s", "pydantic_agent", "success", "Structured configuration generated."),
    # to_agent — setup
    (r"Detected revision #(\d+)", "to_agent", "info", "Building revision #{0}."),
    (r"Applied BC '(.*?)':\s*(\d+) nodes", "to_agent", "info", "Fixed boundary '{0}' — {1} nodes."),
    (r"Applied diagonal force:\s*(\d+) nodes, total Fy=([-\d.eE]+)", "to_agent", "info",
     "Applied diagonal load — {0} nodes, Fy={1}."),
    (r"Applied Force '(.*?)':\s*(\d+) nodes", "to_agent", "info", "Applied load '{0}' — {1} nodes."),
    (r"Filter r_min:\s*(.+)", "to_agent", "info", "Density filter radius {0}."),
    (r"Problem:\s*(.+)", "to_agent", "info", "Problem type: {0}."),
    (r"Volume fraction:\s*(.+)", "to_agent", "info", "Target volume fraction {0}."),
    (r"Optimizer:\s*(.+)", "to_agent", "info", "Optimizer: {0}."),
    (r"Starting optimization \(Revision #(\d+)\)", "to_agent", "progress", "Optimizing — revision #{0}…"),
    (r"Starting optimization \(Original\)", "to_agent", "progress", "Optimizing — initial design…"),
    (r"mid-run snapshot at iter (\d+)/(\d+).*obj=([\d.]+)", "to_agent", "progress",
     "Iteration {0}/{1} — objective {2}."),
    (r"Optimization complete", "to_agent", "success", "Optimization complete."),
    (r"Total iterations:\s*(\d+)", "to_agent", "info", "Ran {0} iterations."),
    (r"Time per iteration:\s*([\d.]+)s", "to_agent", "info", "{0}s per iteration."),
    (r"Stop requested — halting optimization at iter (\d+)/(\d+)", "to_agent", "warn",
     "Stop requested — halted at iteration {0}/{1}."),
    (r"Exported TO state", "to_agent", "success", "Saved optimization result."),
    # vllm
    (r"initialized for problem type:\s*(.+)", "vllm_agent", "info", "Problem type: {0}."),
    (r"Attached (\d+) image\(s\) for revision (\d+)", "vllm_agent", "progress",
     "Examining {0} rendered views of revision {1}…"),
    (r"Problem Type Identification", "vllm_agent", "progress", "Identifying the problem type…"),
    (r"Best iteration: revision (\d+) \(score: (\d+)", "vllm_agent", "success",
     "Best design so far: revision {0} (score {1}/5)."),
    # revise
    (r"Starting JSON revision", "revise_agent", "progress", "Revising the design parameters…"),
    (r"Revised JSON Configuration", "revise_agent", "info", "Proposed revised parameters."),
    (r"Pydantic validation passed", "revise_agent", "success", "Revised parameters validated."),
    # ai_judge
    (r"Found (\d+) screenshot directories", "ai_judge", "progress", "Comparing {0} candidate designs…"),
    (r"Comparative Analysis", "ai_judge", "progress", "Writing comparative verdict…"),
    (r"Saved judge scores.*\((\d+) total", "ai_judge", "success", "Scored the designs."),
]
_ACTION_RULES = [(re.compile(p), a, k, t) for (p, a, k, t) in _ACTION_RULES]


def _parse_action(line):
    s = line.strip()
    if not s:
        return None
    for rx, agent, kind, tmpl in _ACTION_RULES:
        m = rx.search(s)
        if m:
            try:
                text = tmpl.format(*[g.strip() if isinstance(g, str) else g for g in m.groups()])
            except Exception:
                text = tmpl
            return {"agent": agent, "label": pipeline.AGENT_LABELS.get(agent, agent),
                    "kind": kind, "text": text}
    return None


# --------------------------------------------------------------------------- #
# config summary (for the "intended setup" panel)
# --------------------------------------------------------------------------- #
def _latest_config(run_dir):
    files = sorted((run_dir / "original_TO_JSON").glob("version_*.json"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not files:
        return None, None
    p = files[-1]
    try:
        return json.loads(p.read_text()), p
    except Exception:
        return None, p


def _summarize_config(cfg):
    m = (cfg.get("mesh") or [{}])[0]
    lines = [f"Domain {m.get('lx')}×{m.get('ly')}×{m.get('lz')} · grid {m.get('nx')}×{m.get('ny')}×{m.get('nz')}"]
    for i, bc in enumerate(cfg.get("bc", []) or []):
        dofs = bc.get("dofs", {}) or {}
        fixed = [d for d in ("ux", "uy", "uz") if dofs.get(d) is not None]
        rules = "; ".join(f"{r.get('axis')}{_op(r.get('operator'))}{r.get('value')}"
                          for r in (bc.get("selection", {}) or {}).get("rules", []))
        lines.append(f"BC {i+1}: fix [{','.join(fixed) or 'none'}] where {rules}")
    for i, fr in enumerate(cfg.get("forces", []) or []):
        comps = fr.get("forces", {}) or {}
        cs = ", ".join(f"{k}={v}" for k, v in comps.items() if v)
        rules = "; ".join(f"{r.get('axis')}{_op(r.get('operator'))}{r.get('value')}"
                          for r in (fr.get("selection", {}) or {}).get("rules", []))
        lines.append(f"Load {i+1}: [{cs}] where {rules}")
    prob = cfg.get("problem", {}) or {}
    vf = prob.get("volume_fraction")
    if vf is not None:
        lines.append(f"Objective: {prob.get('type', 'MinimumCompliance')} · volume fraction {vf}")
    return lines


def _op(o):
    return {"equals": "=", "greater_than": ">", "less_than": "<", "between": "∈"}.get(o, "=")


# --------------------------------------------------------------------------- #
# Worker: runs the pipeline and pushes narrated events onto a queue
# --------------------------------------------------------------------------- #
def _run_worker(description, run_dir, q):
    orig_cwd = os.getcwd()
    stop = threading.Event()

    def push_msg(m, idx):
        name = m.get("name") or m.get("role") or "agent"
        q.put(("message", {"index": idx, "name": name,
                           "label": pipeline.AGENT_LABELS.get(name, name),
                           "content": _content_to_text(m.get("content"))}))

    def monitor():
        seen_msgs = 0
        seen_setup = set()
        seen_frames = set()
        seen_imgs = set()
        last_prog = {}
        cfg_cache = {"cfg": None}

        def do_setup():
            cfg, path = _latest_config(run_dir)
            if not cfg or path is None:
                return
            key = path.name
            cfg_cache["cfg"] = cfg
            if key in seen_setup:
                return
            seen_setup.add(key)
            out = run_dir / f"setup_{path.stem}.html"
            try:
                viz.render_setup_plotly(cfg, str(out))
                q.put(("setup", {"url": f"/runs/{run_dir.name}/{out.name}",
                                 "summary": _summarize_config(cfg)}))
            except Exception as e:  # noqa: BLE001
                q.put(("log", f"[setup diagram failed: {e}]"))

        def do_frames():
            cfg = cfg_cache["cfg"]
            if cfg is None:
                cfg, _ = _latest_config(run_dir)
                cfg_cache["cfg"] = cfg
            if cfg is None:
                return
            for npy in sorted(run_dir.glob("to_state_revision_*/frames/rho_*.npy")):
                rel = npy.relative_to(run_dir).as_posix()
                if rel in seen_frames:
                    continue
                seen_frames.add(rel)
                png = npy.with_suffix(".png")
                try:
                    viz.render_density_frame(str(npy), cfg, str(png))
                except Exception as e:  # noqa: BLE001
                    q.put(("log", f"[frame render failed: {e}]"))
                    continue
                it = int(re.sub(r"\D", "", npy.stem) or 0)
                rev = re.search(r"revision_(\d+)", npy.as_posix())
                q.put(("frame", {"iter": it,
                                 "revision": int(rev.group(1)) if rev else 0,
                                 "url": f"/runs/{run_dir.name}/{png.relative_to(run_dir).as_posix()}"}))
            # live compliance curve from progress.json (per revision)
            for prog in run_dir.glob("to_state_revision_*/frames/progress.json"):
                try:
                    mt = prog.stat().st_mtime
                    if last_prog.get(str(prog)) == mt:
                        continue
                    last_prog[str(prog)] = mt
                    data = json.loads(prog.read_text())
                    q.put(("progress", {"iter": data.get("iter"),
                                        "total": data.get("num_iterations"),
                                        "revision": data.get("revision"),
                                        "objective": data.get("objective_history", [])}))
                except Exception:
                    pass

        def do_images():
            for png in sorted(run_dir.glob("screenshots*/**/*.png")):
                rel = png.relative_to(run_dir).as_posix()
                if rel in seen_imgs:
                    continue
                seen_imgs.add(rel)
                parts = png.relative_to(run_dir).parts
                top = parts[0]
                rev = re.sub(r"\D", "", top) or "0"
                kind = parts[1] if len(parts) >= 3 else "view"
                q.put(("result", {"revision": rev, "kind": kind, "view": png.stem,
                                  "url": f"/runs/{run_dir.name}/{rel}"}))

        while not stop.is_set():
            msgs = pipeline.manager.groupchat.messages
            while seen_msgs < len(msgs):
                push_msg(msgs[seen_msgs], seen_msgs + 1); seen_msgs += 1
            do_setup(); do_frames(); do_images()
            time.sleep(0.3)
        # final sweep
        msgs = pipeline.manager.groupchat.messages
        while seen_msgs < len(msgs):
            push_msg(msgs[seen_msgs], seen_msgs + 1); seen_msgs += 1
        do_setup(); do_frames(); do_images()

    try:
        os.chdir(run_dir)
        pipeline.reset_chat()
        q.put(("status", {"state": "running", "message": "Pipeline started."}))
        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()

        writer = _QueueWriter(q)
        try:
            with contextlib.redirect_stdout(writer):
                pipeline.run_chat(description)
        finally:
            writer.flush()
            stop.set()
            mon.join(timeout=15)

        transcript = pipeline.manager.groupchat.messages
        (run_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, default=str))
        with (run_dir / "transcript.txt").open("w") as f:
            for m in transcript:
                name = m.get("name") or m.get("role") or "agent"
                f.write(f"\n{'='*70}\n{name}\n{'='*70}\n{_content_to_text(m.get('content'))}\n")

        stopped = (run_dir / "STOP").exists()
        q.put(("status", {"state": "done",
                          "message": ("Run stopped." if stopped else
                                      f"Run complete — {len(transcript)} messages."),
                          "transcript": f"/runs/{run_dir.name}/transcript.txt"}))
    except Exception as e:  # noqa: BLE001
        stop.set()
        import traceback
        q.put(("error", {"message": str(e), "trace": traceback.format_exc()}))
    finally:
        os.chdir(orig_cwd)
        q.put(("__end__", None))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "templates" / "index.html").read_text()


@app.get("/examples")
def examples():
    return JSONResponse(pipeline.EXAMPLES)


@app.get("/runs/{run_id}/{path:path}")
def run_file(run_id: str, path: str):
    base = (RUNS_DIR / run_id).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target)


@app.post("/stop")
def stop_run():
    rd = _CURRENT.get("run_dir")
    if not rd:
        return JSONResponse({"error": "no run in progress"}, status_code=409)
    try:
        (Path(rd) / "STOP").write_text("stop")
        return JSONResponse({"ok": True, "message": "Stop requested."})
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/run")
async def run(request: Request):
    body = await request.json()
    description = (body.get("description") or "").strip()
    if not description:
        return JSONResponse({"error": "description is required"}, status_code=400)
    if not _RUN_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "A run is already in progress."}, status_code=409)

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    i = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{run_id}_{i}"; i += 1
    run_dir.mkdir(parents=True)
    _CURRENT["run_dir"] = str(run_dir)

    q: "queue.Queue" = queue.Queue()
    worker = threading.Thread(target=_run_worker, args=(description, run_dir, q), daemon=True)

    def event_stream():
        try:
            worker.start()
            yield _sse("status", {"state": "starting", "run_id": run_dir.name})
            while True:
                event, data = q.get()
                if event == "__end__":
                    break
                yield _sse(event, data)
            yield _sse("close", {})
        finally:
            _CURRENT["run_dir"] = None
            _RUN_LOCK.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TO_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("TO_WEB_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")
