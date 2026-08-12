"""
claude_shim.py — expose the local `claude` CLI as an OpenAI-compatible endpoint.
===============================================================================

The vision agent (`agents/vllm_agent_both.py`) builds its own `OpenAI(...)`
client and posts the standard chat-completions schema with base64 `image_url`
parts.  Claude Code has no such surface — it is a CLI, not an HTTP API.  Rather
than modify the agent, this serves the schema it already speaks and translates
each request into a `claude -p` subprocess call.

    python claude_shim.py --port 8799
    TO_VISION_PROVIDER=vllm TO_VISION_BASE_URL=http://127.0.0.1:8799/v1 \
    TO_VISION_MODEL=opus python app.py

Images arrive as `data:image/png;base64,...` URIs.  Claude Code reads images
from disk with its Read tool, so each one is written to a scratch file and
referenced by absolute path in the prompt.

WHAT THIS COSTS YOU
-------------------
Claude Code loads its own system prompt, tool definitions and project context on
every invocation.  Measured here: ~205k cached input tokens and ~19s wall clock
for a two-image comparison, i.e. roughly $0.15/call at Sonnet API rates, and it
counts against a subscription's usage limits the same way interactive use does.
A direct vision API call is ~3s and a small fraction of a cent.

So this is a good way to *evaluate* the pipeline with a strong model you already
pay for.  It is not a sensible production backend, and it cannot serve the
public demo at all — a visitor has no Claude subscription of their own.

`--bare` would cut most of that overhead, but it forces ANTHROPIC_API_KEY auth
and never reads the OAuth token, so it defeats the purpose of using the
subscription.  Hence the full context load is unavoidable here.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODEL = os.environ.get("TO_CLAUDE_CLI_MODEL", "opus")
TIMEOUT = int(os.environ.get("TO_CLAUDE_CLI_TIMEOUT", "600"))
CLAUDE_BIN = os.environ.get("TO_CLAUDE_BIN", "claude")

_DATA_URI = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.S)


# --------------------------------------------------------------------------- #
# request -> prompt + image files
# --------------------------------------------------------------------------- #
def _extract(messages, scratch: str):
    """Flatten OpenAI messages into (prompt_text, [image_paths]).

    System messages are inlined rather than passed via --system-prompt, which
    would replace Claude Code's own system prompt and break its Read tool.
    """
    chunks, images = [], []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = content or []
        texts = []
        for p in parts:
            if p.get("type") == "text":
                texts.append(p.get("text", ""))
            elif p.get("type") == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                m = _DATA_URI.match(url)
                if not m:
                    # A plain http(s) URL — Claude Code can't fetch it with Read.
                    texts.append(f"[unsupported image URL: {url[:60]}]")
                    continue
                ext = {"image/png": ".png", "image/jpeg": ".jpg",
                       "image/webp": ".webp"}.get(m.group("mime"), ".png")
                path = os.path.join(scratch, f"img_{len(images):02d}{ext}")
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(m.group("data")))
                images.append(path)
                texts.append(f"[Image {len(images)}: {path}]")
        if texts:
            prefix = {"system": "SYSTEM INSTRUCTIONS", "assistant": "ASSISTANT"}.get(role)
            body = "\n".join(t for t in texts if t)
            chunks.append(f"{prefix}:\n{body}" if prefix else body)

    prompt = "\n\n".join(chunks)
    if images:
        listing = "\n".join(f"  Image {i + 1}: {p}" for i, p in enumerate(images))
        prompt = (
            f"Read these {len(images)} image file(s) with the Read tool, in this "
            f"order, then answer the question below about them.\n{listing}\n\n"
            f"Answer with the requested content only — no preamble, no offer to "
            f"do more work, no questions back.\n\n{prompt}"
        )
    return prompt, images


def _run_claude(prompt: str, model: str, scratch: str) -> tuple[str, dict]:
    cmd = [CLAUDE_BIN, "-p", prompt,
           "--model", model,
           "--output-format", "json",
           "--allowedTools", "Read",
           "--add-dir", scratch]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                          cwd=scratch)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude returned non-JSON: {proc.stdout[:300]}") from e
    if payload.get("is_error"):
        raise RuntimeError(f"claude reported an error: {str(payload)[:400]}")

    usage = payload.get("usage", {}) or {}
    meta = {
        "prompt_tokens": (usage.get("input_tokens", 0)
                          + usage.get("cache_read_input_tokens", 0)
                          + usage.get("cache_creation_input_tokens", 0)),
        "completion_tokens": usage.get("output_tokens", 0),
        "cost_usd": payload.get("total_cost_usd"),
        "num_turns": payload.get("num_turns"),
        "seconds": round(dt, 1),
    }
    meta["total_tokens"] = meta["prompt_tokens"] + meta["completion_tokens"]
    return payload.get("result", ""), meta


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quieter default logging
        sys.stderr.write("[claude_shim] " + (fmt % a) + "\n")

    def _json(self, code: int, obj: dict):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "anthropic"}
                for m in ("opus", "sonnet", "haiku", DEFAULT_MODEL)]})
        self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": f"no route {self.path}"}})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad request: {e}"}})

        model = req.get("model") or DEFAULT_MODEL
        scratch = tempfile.mkdtemp(prefix="claude_shim_")
        try:
            prompt, images = _extract(req.get("messages", []), scratch)
            self.log_message("model=%s images=%d prompt=%dch", model, len(images), len(prompt))
            text, meta = _run_claude(prompt, model, scratch)
            self.log_message("-> %ds, %s turns, %s tok, cost=%s",
                             meta["seconds"], meta["num_turns"],
                             meta["total_tokens"], meta["cost_usd"])
        except subprocess.TimeoutExpired:
            return self._json(504, {"error": {"message": f"claude timed out after {TIMEOUT}s"}})
        except Exception as e:
            return self._json(502, {"error": {"message": str(e)[:600]}})
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        self._json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": meta["prompt_tokens"],
                      "completion_tokens": meta["completion_tokens"],
                      "total_tokens": meta["total_tokens"]},
            "x_claude_cli": {k: meta[k] for k in ("cost_usd", "num_turns", "seconds")},
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--port", type=int, default=int(os.environ.get("TO_CLAUDE_SHIM_PORT", "8799")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not shutil.which(CLAUDE_BIN):
        sys.exit(f"`{CLAUDE_BIN}` not found on PATH — is Claude Code installed?")

    globals()["DEFAULT_MODEL"] = args.model
    print(f"[claude_shim] {CLAUDE_BIN} as OpenAI endpoint on "
          f"http://{args.host}:{args.port}/v1  (default model: {args.model})")
    print(f"[claude_shim] point a role at it with:")
    print(f"    TO_VISION_PROVIDER=vllm")
    print(f"    TO_VISION_BASE_URL=http://{args.host}:{args.port}/v1")
    print(f"    TO_VISION_MODEL={args.model}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
