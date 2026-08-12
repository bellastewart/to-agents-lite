# TO-Agents Lite

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bellastewart/to-agents-lite/blob/main/notebooks/TO_Agents_Lite_Colab.ipynb)

A multi-agent topology-optimization pipeline that runs **without a GPU of your own**.

**Fastest path: click the badge.** It sets everything up on a free Colab T4 in
about four minutes — you supply one free API key and nothing else.

You paste a verbose technical description of a structural problem. A group of
agents turns it into a validated config, runs a 3D topology optimization,
*looks at the rendered result with a vision model*, proposes revisions, re-runs,
and scores the candidates against each other — streaming the whole conversation
to your browser as it happens.

The full-fat version of this project needs **four A100s**. This one needs none.

---

## Where the compute comes from

| Role | Full version | Here |
|---|---|---|
| **Vision agent** — looks at the renders | Qwen2.5-VL-7B on local vLLM (1 GPU) | Gemini |
| **AI judge** — scores candidates | Gemma-3-27B on local vLLM (2 GPUs, TP=2) | Gemini |
| **Structured output** — prose → validated JSON, and revisions | Together Llama-3.3-70B | Gemini *(or Together)* |
| **TO solver** | pyFANTOM CUDA (1 GPU) | pyFANTOM **CPU**, or a free Colab/Kaggle T4 |
| **3D rendering** | headless Chromium | unchanged — never needed a GPU |

Nothing here assumes hardware you own. The solver is the only piece that
*benefits* from a GPU, and pyFANTOM ships a complete numba-JIT CPU backend.

**You need exactly one API key** (Gemini). All three model roles can run on it.
Together remains supported if you prefer to split them.

> A note on the original wiring, because the names mislead. `llm_config`
> (Qwen2.5-VL-7B) is consumed *only* by the vision agent — it is not a text
> model. `llm_config_TO` (Llama-3.3-70B) is what `pydantic_agent` and
> `revise_agent` use via `instructor` to produce all the JSON. And `generate`,
> the plain-text client both agents are constructed with, is **never called** —
> dead in the original too. `TO_TEXT_*` here drives the structured-output
> client, i.e. the role that actually does the work.

---

## Quickstart

```bash
git clone <this repo> && cd to-agents-lite

sudo apt-get install -y libsuitesparse-dev      # scikit-sparse needs this first
pip install -r requirements-lite.txt
playwright install chromium

cp .env.example .env && chmod 600 .env          # then paste your Gemini key in

python doctor.py                                # <- ALWAYS run this first
python app.py                                   # http://localhost:8765
```

### Two checks, in order

```bash
python doctor.py     # is everything installed and reachable?
python selftest.py   # does a real optimization actually run here? (no API needed)
```

`selftest.py` builds and solves a real cantilever through the same
`build_optimization` the agents call, and reports seconds per iteration. Measured
on a clean CPU-only install:

| Mesh | Elements | s/iteration | 200 iterations |
|---|---|---|---|
| 8×24×24 | 128 | 0.01 | seconds |
| 48×24×24 | 27,648 | **1.36** | **~4.5 min** |

### `doctor.py`

Checks the three things that independently decide whether anything can run —
solver backend, renderer, and the three model roles — then tells you which tier
you qualify for. It exits non-zero if nothing is runnable, and its error
messages name the fix. Run it before opening an issue.

```
[OK] text    gemini/gemini-3.5-flash                            1.4s
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
| `TO_INSTRUCTOR_MODE` | structured-output strategy. Auto-set to `TOOLS` for Gemini, `JSON_SCHEMA` otherwise |

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
- **instructor's `JSON_SCHEMA` mode does not work on Gemini.** Google's
  OpenAI-compat endpoint rejects the `response_format.schema` field with
  `Unknown name "schema"`. `TOOLS`, `JSON`, `MD_JSON` and `JSON_O1` all work and
  `TOOLS` is selected automatically for Gemini.
- **A stale `TO_AGENTS_ROOT` will silently import a different `agents/`.** If you
  copy a `.env` from another checkout, delete that line — otherwise you debug
  code you did not edit. This repo warns when it detects the situation.
- **Gemini 3.x thinking tokens are charged against `maxOutputTokens`.** At a low
  cap you get a truncated fragment with no error. `providers.py` enforces a
  2048-token floor and raises on `finishReason=MAX_TOKENS`.
- **Gemini's free tier allows 20 requests/day PER MODEL** (quota id
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`). This is the limit you will
  actually hit. Because it is per *model*, the default config gives each role a
  different one — `gemini-3.5-flash-lite` for structured output,
  `gemini-3-flash-preview` for vision, `gemini-3.6-flash` for the judge — which
  triples the effective budget. If one role 429s, switch just that role to
  another model; verified vision-capable and independently quota'd:
  `gemini-3-flash-preview`, `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`,
  `gemini-3.6-flash`, `gemini-flash-latest`.
- **Thread oversubscription is catastrophic on many-core machines.** numba
  defaults to one thread per core; at 128 elements on a 244-core node that is
  ~12 s/iteration versus 0.01 s with 8 threads — a ~1400× difference, because
  thread launch and barrier costs dwarf the per-element work. `backend.py` caps
  it at 8 (`TO_MAX_THREADS`, or set `NUMBA_NUM_THREADS` yourself). Laptops and
  Colab runtimes never hit this; workstations and HPC nodes do.
- **CPU runs are slower than GPU but usable.** 1.36 s/iteration at 27,648
  elements. A full run to convergence has still not been done end-to-end through
  the agent loop.

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
agents/            agent classes; only the two instructor call sites are
                   changed (grep 'LITE') so the structured-output mode is
                   selectable per provider
```
