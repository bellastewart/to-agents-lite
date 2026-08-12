"""
pipeline_lite.py — the TO-Agents pipeline with nothing pinned to local hardware.
===============================================================================

Same agents, same deterministic speaker order, same science as
`website/pipeline.py`. The differences are all about *where the compute lives*:

    role            full build                     lite build
    ------------    ---------------------------    -----------------------------
    TO solver       pyFANTOM CUDA, GPU 0           backend.py picks CUDA or CPU
    vision agent    vLLM Qwen2.5-VL-7B, GPU 3      any OpenAI-compatible endpoint
    AI judge        vLLM gemma-3-27b, GPUs 1,2     Gemini (or any of the above)
    structured out  Together Llama-3.3-70B         unchanged (already hosted)

Nothing in `agents/` is modified. Two facts make that possible:

  * `agents/vllm_agent_both.py` builds a plain `OpenAI(base_url=...)` client and
    posts standard chat-completions with base64 `image_url` parts — so pointing
    it at Google's OpenAI-compat shim is enough, no code change.
  * `agents/ai_judge_both.py` already has a `backend="gemini"` path on the
    current `google-genai` SDK; only its model name needs overriding.

Configure with the same env vars `providers.py` and `doctor.py` use:

    TO_TEXT_PROVIDER / TO_TEXT_MODEL
    TO_VISION_PROVIDER / TO_VISION_MODEL
    TO_JUDGE_PROVIDER / TO_JUDGE_MODEL
    TO_BACKEND=auto|cuda|cpu       TO_MESH=nx,ny,nz

Run `python doctor.py` first — it validates every one of these.
"""

import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    _dotenv_loaded = True
except ImportError:
    _dotenv_loaded = False

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
if _dotenv_loaded:
    # Look beside the code first, then one level up, so the standalone repo and
    # the in-tree copy both find a .env without extra configuration.
    for _cand in (HERE / ".env", REPO_ROOT / ".env"):
        if _cand.is_file():
            load_dotenv(_cand)
            break

os.environ.setdefault("AUTOGEN_CACHE_DIR", str(REPO_ROOT / ".cache"))
os.environ["AUTOGEN_USE_DOCKER"] = "False"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAX_TO_RUNS = int(os.environ.get("TO_MAX_RUNS", "5"))

import nest_asyncio
nest_asyncio.apply()

import providers                      # noqa: E402
import backend as _backend            # noqa: E402  (also fixes sys.path for pyFANTOM)

_MESH, _MESH_WHY = _backend.suggest_mesh()

print(f"[pipeline_lite] TO backend  : {_backend.BACKEND} ({_backend.describe()['detection']})")
print(f"[pipeline_lite] mesh        : {_MESH[0]}x{_MESH[1]}x{_MESH[2]} — {_MESH_WHY}")
for _role, _cfg in providers.describe_config().items():
    print(f"[pipeline_lite] {_role:<11} : {_cfg.get('provider')}/{_cfg.get('model')}"
          if "error" not in _cfg else f"[pipeline_lite] {_role:<11} : ERROR {_cfg['error']}")

# --------------------------------------------------------------------------- #
# Model configs, derived from the role config rather than hardcoded hosts
# --------------------------------------------------------------------------- #
_text = providers.openai_compat("text")
_vision = providers.openai_compat("vision")

# Used by to_agent (nominally) and by vllm_agent (actually — it reads
# config_list[0] and builds its own client from base_url/api_key/model).
llm_config = {
    "cache_seed": 9527,
    "config_list": [{
        "model": _vision["model"],
        "base_url": _vision["base_url"],
        "api_key": _vision["api_key"],
        "max_tokens": int(os.environ.get("TO_VISION_MAX_TOKENS", "8192")),
    }],
    "temperature": 0,
    # Gemini 3.x spends "thinking" tokens out of this same budget — a small
    # value yields silently truncated critiques. See LITE_STRATEGY.md §3a.
    "max_tokens": int(os.environ.get("TO_VISION_MAX_TOKENS", "8192")),
}

from openai import OpenAI                                        # noqa: E402


class _LLM:
    """Plain text completion for the pydantic / revise agents' `generate`."""

    def __init__(self, cfg):
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model = cfg["model"]
        self.max_tokens = int(os.environ.get("TO_TEXT_MAX_TOKENS", "8192"))

    def generate_cli(self, system_prompt="You are an expert in this field. Try your best "
                     "to give a clear and concise answer.",
                     prompt="Hello world! I am", temperature=0):
        try:
            messages = ([{"role": "user", "content": prompt}] if system_prompt is None
                        else [{"role": "system", "content": system_prompt},
                              {"role": "user", "content": prompt}])
            result = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=self.max_tokens)
            return result.choices[0].message.content
        except Exception as e:                                    # noqa: BLE001
            # The original swallowed this silently; a visible failure is far
            # easier to debug than an agent that mysteriously emits nothing.
            print(f"⚠️  [pipeline_lite] text generate failed: {type(e).__name__}: {e}")
            return ""


generate = _LLM(_text).generate_cli

# --------------------------------------------------------------------------- #
# Together — structured output (already hosted; unchanged from the full build)
# --------------------------------------------------------------------------- #
from together import Together                                    # noqa: E402

_TOGETHER_KEY = os.environ.get("Llama4_together") or os.environ.get("TOGETHER_API_KEY", "")
config_list_TO = [{
    "model": os.environ.get("TO_STRUCTURED_MODEL",
                            "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "api_key_TO": _TOGETHER_KEY,
    "max_tokens": 20000,
}]
client_TO = Together(api_key=config_list_TO[0]["api_key_TO"])
llm_config_TO = {
    "client_TO": client_TO,
    "model_TO": config_list_TO[0]["model"],
    "max_tokens_TO": config_list_TO[0]["max_tokens"],
}

# --------------------------------------------------------------------------- #
# Agents — unmodified classes from agents/, except the backend-switchable TO one
# --------------------------------------------------------------------------- #
import autogen                                                    # noqa: E402

from agents.pydantic_agent import PydanticAgent                   # noqa: E402
from to_agent_lite import TOAgentBoth as TOAgent                  # noqa: E402  (LITE)
from agents.vllm_agent_both import VLLMAgentBoth as VLLMAgent     # noqa: E402
from agents.ai_judge_both import AI_JudgeBoth as AI_Judge         # noqa: E402
from agents.revise_agent import ReviseAgent                       # noqa: E402

pydantic_agent = PydanticAgent(
    name="pydantic_agent", system_message="Structured output agent.",
    generate=generate, llm_config=llm_config, llm_config_TO=llm_config_TO,
    human_input_mode="NEVER", code_execution_config={"use_docker": False})

to_agent = TOAgent(
    name="to_agent", system_message="TO construction agent",
    human_input_mode="NEVER", code_execution_config={"use_docker": False},
    llm_config=llm_config)

vllm_agent = VLLMAgent(
    name="vllm_agent", system_message="Interprets images",
    human_input_mode="NEVER", code_execution_config={"use_docker": False},
    llm_config=llm_config)

revise_agent = ReviseAgent(
    name="revise_agent", system_message="Revises structured output agent.",
    generate=generate, llm_config=llm_config, llm_config_TO=llm_config_TO,
    human_input_mode="NEVER", code_execution_config={"use_docker": False})

# --- judge ---------------------------------------------------------------- #
_judge_cfg = providers.describe_config()["judge"]
_judge_provider = _judge_cfg.get("provider")

if _judge_provider == "gemini":
    ai_judge = AI_Judge(
        name="ai_judge", system_message="Judges structured output agent.",
        human_input_mode="NEVER", code_execution_config={"use_docker": False},
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        temperature=0.0, backend="gemini")
    # The class hardcodes gemini-3-flash-preview; honour TO_JUDGE_MODEL instead.
    ai_judge.model = _judge_cfg["model"]
else:
    _judge = providers.openai_compat("judge")
    ai_judge = AI_Judge(
        name="ai_judge", system_message="Judges structured output agent.",
        human_input_mode="NEVER", code_execution_config={"use_docker": False},
        llm_config={"cache_seed": 9527, "temperature": 0,
                    "config_list": [{"model": _judge["model"],
                                     "base_url": _judge["base_url"],
                                     "api_key": _judge["api_key"],
                                     "max_tokens": 8192}]},
        backend="local")
print(f"[pipeline_lite] judge wired : {_judge_provider}/{ai_judge.model}")

user_proxy = autogen.UserProxyAgent(
    name="Admin", system_message="Human admin.", human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").replace("*", "").rstrip())

# --------------------------------------------------------------------------- #
# GroupChat — identical routing to the full build
# --------------------------------------------------------------------------- #
agents = [user_proxy, pydantic_agent, to_agent, vllm_agent, revise_agent, ai_judge]
CHECKPOINT_MODE = False


def speaker_selection_func(last_speaker, groupchat):
    if os.path.exists("STOP"):
        return None
    if last_speaker is user_proxy:
        return pydantic_agent
    elif last_speaker is pydantic_agent:
        return to_agent
    elif last_speaker is to_agent:
        n = sum(1 for m in groupchat.messages if m.get("name") == "to_agent")
        if n >= MAX_TO_RUNS:
            return ai_judge
        elif n == 1:
            return None if CHECKPOINT_MODE else vllm_agent
        else:
            return ai_judge
    elif last_speaker is ai_judge:
        n = sum(1 for m in groupchat.messages if m.get("name") == "to_agent")
        return None if n >= MAX_TO_RUNS else vllm_agent
    elif last_speaker is vllm_agent:
        return revise_agent
    elif last_speaker is revise_agent:
        return to_agent
    return None


groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=100,
                              speaker_selection_method=speaker_selection_func,
                              enable_clear_history=True)
manager = autogen.GroupChatManager(groupchat)

AGENT_LABELS = {
    "Admin": "User", "pydantic_agent": "Structured Output",
    "to_agent": "Topology Optimizer", "vllm_agent": "Vision Interpreter",
    "revise_agent": "Reviser", "ai_judge": "AI Judge",
}

# --------------------------------------------------------------------------- #
# Examples, rescaled to what this machine can actually finish
# --------------------------------------------------------------------------- #
_BASE_EXAMPLES = {
    "cantilever": (
        "In this setup, the finite element analysis (FEA) uses linear elasticity physics with "
        "a Young's modulus E = 1.0 and a Poisson's ratio ν = 0.3. A three-dimensional structured "
        "mesh is defined with nx = 128, ny = 64, nz = 64 elements and physical dimensions lx = 1.0, "
        "ly = 0.5, lz = 0.5. The mesh is linked to the linear elastic physics so the material "
        "properties apply uniformly throughout the domain. A multigrid solver is configured with "
        "tol = 1e-4, maxiter = 200, and n_level = 5. Two Dirichlet boundary conditions are imposed: "
        "(1) Left face: nodes where x = 0 are fixed in the ux and uz directions while uy remains free, "
        "with zero displacement. (2) Right-bottom corner: nodes where x = 1.0 and y = 0 are fixed in "
        "the uy and uz directions while ux remains free, with zero displacement. One point force is "
        "applied: nodes where x = 0 and y = 0.5 have a force of -1.0 in the y-direction distributed "
        "among all nodes. Use a density filter with radius 1.5. Set up a minimum compliance topology "
        "optimization problem with a volume fraction of 40%, SIMP penalty of 3.0, void material "
        "stiffness of 1e-9, and enable Heaviside projection. Use the PGD optimizer with a function "
        "tolerance of 1e-4 and no change tolerance."
    ),
    "phone_stand": (
        "In this setup, the finite element analysis (FEA) uses linear elasticity physics with a "
        "Young's modulus E = 1.0 and a Poisson's ratio ν = 0.3. A three-dimensional structured mesh "
        "is defined with nx = 128, ny = 64, nz = 16 elements and physical dimensions lx = 1.0, "
        "ly = 0.5, lz = 0.125. The mesh is linked to the linear elastic physics so the material "
        "properties apply uniformly throughout the domain. A multigrid solver is configured with "
        "tol = 1e-4, maxiter = 50, and n_level = 5. One Dirichlet boundary condition is imposed: the "
        "bottom face (y = 0), where all nodes are fixed in the ux, uy, and uz directions. One force is "
        "applied along a diagonal: with a force of -1.0 in the y-direction distributed among all nodes. "
        "Use a density filter with radius 2.0. Set up a minimum compliance topology optimization "
        "problem with a volume fraction of 15%, SIMP penalty of 3.0, void material stiffness of 1e-9, "
        "and enable Heaviside projection. Use the PGD optimizer with a function tolerance of 1e-4 and "
        "no change tolerance."
    ),
}


def _rescale(text: str) -> str:
    """Swap the example's nx/ny/nz for what this hardware can finish.

    Only the grid counts change — physical dimensions, BCs, loads and the
    objective are untouched, so it stays the same problem at lower resolution.
    """
    nx, ny, nz = _MESH
    if (nx, ny, nz) == (128, 64, 64):
        return text
    m = re.search(r"nx = (\d+), ny = (\d+), nz = (\d+)", text)
    if not m:
        return text
    # Preserve the example's own aspect ratio (phone_stand is thin in z).
    onx, ony, onz = (int(g) for g in m.groups())
    sx = nx / onx
    new = (max(8, round(onx * sx)), max(4, round(ony * sx)), max(4, round(onz * sx)))
    return text.replace(m.group(0), f"nx = {new[0]}, ny = {new[1]}, nz = {new[2]}")


EXAMPLES = {k: _rescale(v) for k, v in _BASE_EXAMPLES.items()}


# --------------------------------------------------------------------------- #
# Run helpers — same surface app.py expects
# --------------------------------------------------------------------------- #
def reset_chat():
    manager.groupchat.messages = []
    try:
        manager.reset()
    except Exception:
        pass
    for a in agents:
        try:
            a.reset()
        except Exception:
            pass


def run_chat(description):
    """BLOCKING. Runs the full group chat for one description."""
    return user_proxy.initiate_chat(manager, message=description)
