# TO-Agents Lite

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bellastewart/to-agents-lite/blob/main/notebooks/TO_Agents_Lite_Colab.ipynb)

A multi-agent topology-optimization pipeline that runs **without a GPU of your own**.

**Fastest path: click the badge.** It sets everything up on a free Colab T4 in
about four minutes — you supply two free API keys and nothing else.

You paste a verbose technical description of a structural problem. A group of
agents turns it into a validated config, runs a 3D topology optimization,
*looks at the rendered result with a vision model*, proposes revisions, re-runs,
and scores the candidates against each other — streaming the whole conversation
to your browser as it happens.

The full-fat version of this project needs **four A100s**. This one needs none.

---

## Where the compute comes from

| Piece | Full version | Here |
|---|---|---|
| Vision agent | Qwen2.5-VL-7B on local vLLM (1 GPU) | Gemini API |
| AI judge | Gemma-3-27B on local vLLM (2 GPUs, TP=2) | Gemini API |
| Structured output | Together Llama-3.3-70B | unchanged — already hosted |
| TO solver | pyFANTOM CUDA (1 GPU) | **pyFANTOM CPU**, or a free Colab/Kaggle T4 |
| 3D rendering | headless Chromium | unchanged — never needed a GPU |

Nothing here assumes hardware you own. The solver is the only piece that
*benefits* from a GPU, and pyFANTOM ships a complete numba-JIT CPU backend.

---

## Quickstart

```bash
git clone <this repo> && cd to-agents-lite

sudo apt-get install -y libsuitesparse-dev      # scikit-sparse needs this first
pip install -r requirements-lite.txt
playwright install chromium

cp .env.example .env && chmod 600 .env          # then paste your keys in

python doctor.py                                # <- ALWAYS run this first
python app.py                                   # http://localhost:8080
```

### `doctor.py`

Checks the three things that independently decide whether anything can run —
solver backend, renderer, and the three model roles — then tells you which tier
you qualify for. It exits non-zero if nothing is runnable, and its error
messages name the fix. Run it before opening an issue.

```
[OK] text    together/meta-llama/Llama-3.3-70B-Instruct-Turbo   0.8s
[OK] vision  gemini/gemini-3.5-flash                            3.2s
[OK] judge   gemini/gemini-3.5-flash                            2.1s
[OK] CPU backend — all 15 required symbols present
[OK] chromium — launches headless
-> Best available: Tier 0  ZERO GPU
```

---

## Configuration

Every model role is independently pointable at any provider. See `.env.example`
for the full list; the essentials:

| Variable | Meaning |
|---|---|
| `TO_{TEXT,VISION,JUDGE}_PROVIDER` | `vllm` \| `ollama` \| `together` \| `openai` \| `hf` \| `gemini` \| `anthropic` |
| `TO_{TEXT,VISION,JUDGE}_MODEL` | model id for that provider |
| `TO_BACKEND` | `auto` (default) \| `cuda` \| `cpu` |
| `TO_MESH` | `nx,ny,nz` — pin the mesh instead of deriving it from VRAM |

Unset everything and it defaults to local vLLM on `:8000`/`:8001`, i.e. the
original layout.

### Mesh sizing

Chosen from detected free VRAM, because a resolution that is fine on an A100 is
not fine on a T4 and is hopeless on CPU:

| Hardware | Mesh | Elements |
|---|---|---|
| ≥24 GB VRAM | 128×64×64 | 524,288 |
| ≥10 GB (T4/L4) | 96×48×48 | 221,184 *(estimate)* |
| CPU only | 48×24×24 | 27,648 |

---

## Known limitations

Stated plainly, because silent degradation is worse than a missing feature:

- **`LocalFilter` is CUDA-only.** On CPU, a per-element `r_min` request collapses
  to the mean radius with `StructuredFilter3D` and prints a warning. Length-scale
  control becomes uniform — it is *not* the same problem you asked for.
- **`MinimumCompliance` differs by backend.** CUDA accepts `E_local`,
  `local_volume_constraint`, `passive_solid`, `passive_void`; CPU accepts none of
  them. They are dropped, loudly, when non-null.
- **Together's serverless tier has no vision models.** They exist in the catalog
  but need a paid dedicated endpoint. Use Gemini for anything with images.
- **Gemini 3.x thinking tokens are charged against `maxOutputTokens`.** At a low
  cap you get a truncated fragment with no error. `providers.py` enforces a
  2048-token floor and raises on `finishReason=MAX_TOKENS`.
- **CPU wall-clock is unmeasured.** The CPU path is verified to import, build,
  and apply BCs/loads. A full optimization to convergence has not been timed.

See `LITE_STRATEGY.md` for the measurements behind each of these.

---

## Layout

```
app.py             FastAPI server; streams the run over SSE
pipeline_lite.py   agent construction + group chat (provider-driven)
providers.py       one interface over 7 model backends
backend.py         picks pyFANTOM CUDA or CPU; mesh sizing
to_agent_lite.py   the TO agent, backend-switchable (grep '# LITE')
doctor.py          preflight capability probe
viz.py             setup diagram + density frames
agents/            unmodified agent classes
```
