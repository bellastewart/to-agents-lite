# Running TO-Agents without four A100s

**Status:** scaffolding built and verified; one blocker needs a decision (see §5).
**Nothing in `website/`, `agents/`, `pyFANTOM/`, or the conda env was modified.**

---

## 1. What the full build actually costs

Measured from `bash_scripts/` and `pipeline.py`, not estimated:

| Component | Model | GPUs | Source |
|---|---|---|---|
| Vision agent | `Qwen/Qwen2.5-VL-7B-Instruct` @ `:8000` | 1 (`CUDA_VISIBLE_DEVICES=3`) | `run_Qwen2.5-VL-7B.sh` |
| AI judge | `google/gemma-3-27b-it` @ `:8001`, TP=2 | 2 (`CUDA_VISIBLE_DEVICES=1,2`) | `Gemma3_27B.sh` |
| TO solver | pyFANTOM (CuPy) | 1 (`CUDA_VISIBLE_DEVICES=0`) | `pipeline.py:59` |
| Structured output | `Llama-3.3-70B-Instruct-Turbo` | 0 — already hosted | `pipeline.py:170` |

**Four A100s.** The goal is to get that to zero without changing the science.

---

## 2. What I verified (not assumed)

Three independent findings, each tested on this machine:

**a. pyFANTOM has a complete CPU backend — this is the big one.**
`pyFANTOM/CPU/`, `core/CPU/`, `geom/CPU/`, `solvers/CPU/`, `stiffness/CPU/`,
`Optimizers/CPU/`, `Problem/CPU/`, `mma/CPU/` all exist and mirror the CUDA tree.
With `CUDA_VISIBLE_DEVICES=""` (cupy correctly dies with `cudaErrorNoDevice`),
`import pyFANTOM.CPU` succeeds and exposes **15 of the 16** symbols
`agents/to_agent_both.py` imports from `pyFANTOM.CUDA`. A `StructuredMesh3D`
and `LinearElasticity` construct fine. Kernels are numba-JIT
(`core/CPU/_ops.py`).

> The one gap: **`LocalFilter` is CUDA-only.** A CPU build must use
> `StructuredFilter3D` instead. Both example problems in `pipeline.py` already
> use a density filter with a radius, so this looks non-blocking — but it is a
> real code change, not a config flag.

> Caveat I did **not** test: wall-clock for a full CPU optimization at
> production mesh. Import and construction are verified; throughput is not.
> Expect to need a smaller mesh (§4).

**b. Rendering needs no GPU.** `capture_solution_screenshots` goes
k3d → HTML → **Playwright headless Chromium** (`visualizers/_3d_original.py:761-766`).
That is CPU/software-WebGL. `doctor.py` launches Chromium headless here
successfully. The screenshot pipeline is fully portable.

**c. The base `FiniteElement` carries `capture_solution_screenshots`**, so the
CPU `FiniteElement` inherits it — the vision loop does not depend on the CUDA
Problem class.

---

## 3a. RESOLVED — the zero-GPU path is green

After rotating the Gemini key, the full Tier 0 chain passes:

```
[OK] text    together/meta-llama/Llama-3.3-70B-Instruct-Turbo   0.8s
[OK] vision  gemini/gemini-3.5-flash                            3.2s
[OK] judge   gemini/gemini-3.5-flash                            2.1s
[OK] CPU backend — all 15 required symbols present
[OK] chromium — launches headless
-> Best available: Tier 0  ZERO GPU
```

**Model choice.** Probed on a real depth render from `run_20260812_075140`:

| Model | Latency | Verdict |
|---|---|---|
| `gemini-3.5-flash` | **2.5 s** | best — fast, clean prose. **Use this.** |
| `gemini-3-flash-preview` | 4.1 s | fine (what `pipeline.py` already names) |
| `gemini-3.6-flash` | 9.0 s | works, slower |
| `gemma-4-31b-it` | 48 s | works but leaks its scaffold (`* Input: … * Task: …`) into the answer — poor judge discipline |
| `gemini-2.5-flash`, `gemini-2.5-pro` | — | **404, "no longer available to new users"** — do not use |

### Trap: Gemini 3.x thinking tokens eat `maxOutputTokens`

Measured on the same image:

| `maxOutputTokens` | `finishReason` | thinking | **visible output** |
|---|---|---|---|
| 220 | `MAX_TOKENS` | 207 | **9 tokens** — truncated mid-word, no error raised |
| 2000 | `STOP` | 810–1004 | 56–76 tokens — complete |

A judge that silently returns a sentence fragment is worse than one that fails,
so `GeminiBackend` now (a) enforces a `MIN_OUTPUT_TOKENS = 2048` floor, and
(b) **raises** on `finishReason == MAX_TOKENS` with the token accounting in the
message. `TO_GEMINI_THINKING_BUDGET=0` disables reasoning entirely (~2.4 s,
full answer) if you want tighter latency.

> Also note: `doctor.py`'s probe image is 64×64, not 1×1. Gemini rejects
> degenerate images with HTTP 400 *"Unable to process input image"*, which
> mimics a bad key and sends you debugging the wrong thing.

---

## 3. The blocker as originally found (historical — now resolved by §3a)

The plan was "move the models to Together." I probed all four credential sets
in `.env` with real API calls. Results:

| Provider | Status |
|---|---|
| **Together** | Only `meta-llama/Llama-3.3-70B-Instruct-Turbo` is callable. **Every** vision model (`Qwen2.5-VL-72B`, `Qwen3-VL-*`, `Llama-4-Scout/Maverick`, `llama-3.2-*-vision`) and **`google/gemma-3-27b-it` itself** returns *"Unable to access non-serverless model"* — they need a paid dedicated endpoint. The `/v1/models` catalog lists 279 models but is **not** an availability signal: the model you use in production today also reports `running=false`. |
| **Gemini** | Key **revoked**: *"Your API key was reported as leaked."* Also `gemini-2.5-flash` / `gemini-2.0-flash` now 404 for new users. |
| **OpenAI** | Key valid (lists 118 models) but **no credits remaining** (HTTP 429). |
| **Anthropic** | Key invalid (HTTP 401). |

So: solver ✅ CPU-capable, renderer ✅ portable, text ✅ works via Together —
**vision is the single thing standing between you and a zero-GPU demo.**
Both the `vllm_agent` and the `ai_judge` need image input.

> The Gemini key is in your `.env`. I confirmed `.env` is gitignored and was
> **never committed**, and no key is hardcoded in any tracked file — so the leak
> did not come from this repo's history. It still needs rotating.

---

## 4. Deployment tiers

`doctor.py` reports which of these a given machine qualifies for.

| Tier | GPUs | Solver | Vision + judge | Who it's for |
|---|---|---|---|---|
| **2** | 4 | pyFANTOM CUDA | local vLLM | you, today — unchanged |
| **1** | 1 | pyFANTOM CUDA | hosted API | Colab / Kaggle free T4 |
| **0** | **0** | pyFANTOM CPU, reduced mesh | hosted API | a visitor to the website |
| **-1** | 0 | pyFANTOM CPU | *none* | solver + renders, no critique loop |

**Mesh budget for Tier 0.** `cantilever` is currently 128×64×64 = **524,288**
elements. Realistic CPU targets: 64×32×32 = 65,536 (8× less) or
48×24×24 = 27,648 (19× less). Resolution drops, but the agent loop —
which is what the demo is showing off — is unchanged.

**Cheap wins independent of tier:**
- The judge does not need 27B. `gemma-3-4b-it` is multimodal and ~7× smaller;
  that alone removes the 2-GPU tensor-parallel requirement.
- `--limit-mm-per-prompt '{"image": 4}'` already caps images per call, so the
  vision context stays small and hosted-API cost stays low.

---

## 5. Decisions taken

**Vision + judge → a fresh Gemini API key.** Free tier, multimodal, and
`pipeline.py` already carries a `gemini` branch for `ai_judge`, so it is the
least new code. It also retires the revoked key, which had to happen anyway.
Set `TO_VISION_PROVIDER=gemini` and `TO_JUDGE_PROVIDER=gemini`.

**Solver → keep a GPU wherever one exists.** CPU is the last-resort fallback,
not the default (see §5b).

---

## 5b. "Can we not still use the GPU somehow?" — yes

The visitor has no GPU. That does **not** mean the *pipeline* has no GPU. What
makes this practical:

- pyFANTOM is **pip-installable** (`pyproject.toml`, setuptools backend) and its
  own tagline is *"GPU and CPU ready"* — the CPU path is a designed feature.
- The submodule is a **public repo**: `github.com/bellastewart/pyFANTOM_TO-Agents`,
  so any machine can `pip install git+https://…` it directly.
- Deps are ordinary (numpy/scipy/numba/vtk/k3d/pygmsh/sympy) and the GPU extra is
  `cupy-cuda12x`. This env runs **cupy 13.6 / CUDA runtime 12.9**, and Colab is
  also CUDA 12.x — a direct match, no version gymnastics.
- One install gotcha: `scikit-sparse` needs SuiteSparse headers
  (`apt-get install -y libsuitesparse-dev` before pip on Debian/Colab).

### Three routes to a GPU that isn't yours

| Route | GPU | Cost | Trade-off |
|---|---|---|---|
| **A. Colab / Kaggle notebook** | free T4 (Colab), 2×T4 30 h/wk (Kaggle) | $0 | Best fit for "someone found the website." Ship an *Open in Colab* notebook that pip-installs pyFANTOM + runs `app.py`, exposed via Colab's port forwarding. Session limits apply. |
| **B. Split app** | serverless GPU (Modal / RunPod / Replicate) | ~$0–cheap | Web frontend on a free CPU host; only `run_optimization` is a remote GPU call. Cleanest for a real public site; most infra work. |
| **C. Your NERSC box serves everyone** | your A100s | $0 | Simplest technically, but NERSC does not permit exposing a public web service from login/compute nodes. Policy blocker, not a technical one. |

**Route A is the recommendation** — it is exactly the "use Jupyter's GPU" idea,
and it needs no billing relationship from either side.

### Mesh sizing follows the hardware, not the tier label

Because a GPU is usually available *somewhere*, mesh size should key off
detected VRAM rather than a blanket CPU downgrade:

| Detected | Mesh | Elements |
|---|---|---|
| A100 40GB (your box) | 128×64×64 | 524,288 — unchanged |
| T4 16GB (Colab/Kaggle) | 96×48×48 | 221,184 |
| CPU only (last resort) | 48×24×24 | 27,648 |

`doctor.py` already reports the backend; adding a VRAM read via `pynvml`
(already a pyFANTOM dependency) is a few lines.

> Unverified: the T4 row is sized by VRAM ratio, not measured. The multigrid
> solver's peak memory at 96×48×48 should be checked on an actual T4 before
> that number is promised to anyone.

---

## 6. What's in this folder

| File | State |
|---|---|
| `app.py`, `pipeline.py`, `viz.py`, `templates/index.html`, `TO_stand.ipynb` | **byte-identical copies** of `website/` (md5-verified), untouched so far |
| `providers.py` | **new** — role-based backend abstraction (`text` / `vision` / `judge`), 7 providers. Defaults reproduce the current local setup exactly, so it is a no-op until configured. Live-verified against Together. |
| `doctor.py` | **new** — preflight probe; prints the capability matrix above and exits non-zero if nothing is runnable. |
| `backend.py` | **new** — picks `pyFANTOM.CUDA` or `pyFANTOM.CPU` at import. `TO_BACKEND=auto\|cuda\|cpu`. Also exposes `suggest_mesh()` (the VRAM ladder above) and `to_numpy()`. Verified on all four paths: GPU present, GPU full, no GPU, forced-CUDA-without-GPU. |
| `to_agent_lite.py` | **new** — `agents/to_agent_both.py` copied byte-identical, then changed in exactly **5 marked places** (`grep '# LITE'`). Imports and exposes the full agent on both backends. |

### `to_agent_lite.py` — the five changes

1. `from pyFANTOM.CUDA import (…)` → `from backend import (…)`
2. `import cupy as cp` in `apply_boundary_conditions` → `cp = _xp`
3. same in `apply_point_forces`
4. two `cp.where(mask)[0].get()` → `_to_numpy(cp.where(mask)[0])`
5. the `LocalFilter` branch — on CPU, collapses per-element `r_min` to its mean
   and uses `StructuredFilter3D`, printing a loud warning. **This changes the
   length-scale semantics from per-element to uniform.** It degrades instead of
   crashing, but it is not equivalent; a per-element `r_min` problem run on CPU
   is not the same problem.

Verified: imports clean with `TO_BACKEND=cpu`, with `CUDA_VISIBLE_DEVICES=""`,
and on the CUDA branch (`backend: cuda | xp: cupy | LocalFilter: True`).
**Not** verified: a full optimization to convergence on either backend.

### Heads-up on the Gemini route

`autogen/oai/gemini.py` emits *"All support for the `google.generativeai`
package has ended."* — so `pipeline.py`'s existing `AI_JUDGE_MODEL=gemini`
branch runs on a dead SDK. `providers.py`'s `GeminiBackend` talks to the REST
endpoint directly and does not depend on that package, so route the judge
through `providers.get_judge_backend()` rather than AutoGen's gemini client.

### Configuring

```bash
# example: text on Together, vision+judge on Gemini
export TO_TEXT_PROVIDER=together   TO_TEXT_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
export TO_VISION_PROVIDER=gemini   TO_VISION_MODEL=<a-multimodal-gemini-model>
export TO_JUDGE_PROVIDER=gemini    TO_JUDGE_MODEL=<a-multimodal-gemini-model>
python doctor.py
```

Unset = local vLLM defaults = current behaviour.

### Still to build (blocked on §5)

- `to_agent_lite.py` — CPU/CUDA-switchable copy of `to_agent_both.py`
  (swap the `pyFANTOM.CUDA` import, replace `LocalFilter`, and route the
  ~8 direct `cupy` calls at lines 87–225 through a `xp = numpy|cupy` alias).
- Rewire `vllm_agent` / `ai_judge` to call `providers.get_*_backend()`.
- A `requirements-lite.txt` with the CPU-only dependency set.
