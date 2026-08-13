import instructor


def _instructor_mode():
    """Structured-output strategy, selectable per provider.

    LITE: the original hardcoded ``instructor.Mode.JSON_SCHEMA``. That works on
    Together but Google's OpenAI-compat endpoint rejects it — it does not accept
    the ``response_format.schema`` field instructor sends, returning
    ``Unknown name "schema" at 'response_format'``. Verified working on Gemini:
    TOOLS, JSON, MD_JSON, JSON_O1.

    Default stays JSON_SCHEMA so behaviour is unchanged unless asked; set
    TO_INSTRUCTOR_MODE=TOOLS to run structured output on Gemini and drop the
    second API key entirely.
    """
    import os
    name = os.environ.get("TO_INSTRUCTOR_MODE", "JSON_SCHEMA").strip().upper()
    mode = getattr(instructor.Mode, name, None)
    if mode is None:
        available = [m for m in dir(instructor.Mode) if m.isupper()]
        raise ValueError(
            f"TO_INSTRUCTOR_MODE={name!r} is not a valid instructor mode. "
            f"Available: {', '.join(sorted(available))}")
    return mode


def _extra_body():
    """Provider-routing options passed through to the OpenAI-compatible call.

    LITE. OpenRouter picks a provider per request, and they do not all support
    the same features. Observed on one run: generation 1 went to Cloudflare and
    came back finish_reason='error' with a 500; generation 2 went to AkashML,
    which answered but ignored the schema and returned `physics` as an object
    where the model expects a list, with optimization_settings missing entirely.

    OpenRouter's fix for this is `provider.require_parameters`: only route to
    providers that actually support the parameters in the request, so asking for
    a JSON schema means only providers that honour one are eligible.

    Set TO_OPENAI_EXTRA_BODY to a JSON object to override; pipeline_lite sets a
    sensible default for OpenRouter.
    """
    import json as _json
    import os as _os
    raw = _os.environ.get("TO_OPENAI_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        body = _json.loads(raw)
    except ValueError as e:
        raise ValueError(f"TO_OPENAI_EXTRA_BODY is not valid JSON: {e}") from e
    return {"extra_body": body} if body else {}
# from Reasoning import make_structure_from_text
from typing import List, Dict, Optional, Literal, Union, Any
import numpy as np
from datetime import datetime
import time
try:
    import torch
except ImportError:
    # LITE: torch is used exactly once in this file — `with torch.no_grad():`
    # around make_structure_from_text, which is a remote API call. There is no
    # local model and no tensor, so the context manager is a no-op. Colab ships
    # torch preinstalled and never hits this, but requiring a ~2 GB download for
    # a no-op would make the CPU install absurd. Substitute a null context.
    import contextlib as _contextlib
    import types as _types
    torch = _types.SimpleNamespace(no_grad=_contextlib.nullcontext)
import json
import os
from pydantic import BaseModel, Field
from IPython import get_ipython
from agents.Reasoning.structure_generation import make_structure_from_text
from autogen import UserProxyAgent, Agent


##############################################################
# Pydantic Set Up 
##############################################################
class Physics(BaseModel):
    # LITE: units made explicit, as for Filter.r_min.
    #
    # pyFANTOM is unitless and this pipeline runs on NORMALISED units, the
    # standard convention in topology optimisation: only the ratio of stiffness
    # to load sets the design, never their absolute magnitudes. All 11 working
    # runs in website/runs use exactly E=1.0 with fy=-1.0.
    #
    # Nothing said so, so asked for "a steel cantilever" the model emitted the
    # physically-correct E=2.1e11 with fy=-1000. Measured result: the
    # preconditioned CG diverges, the objective is NaN for 49 of 50 iterations,
    # the design never leaves its uniform initial density, and the run still
    # reports success and is scored by the judge at 100% confidence.
    E: float = Field(
        ...,
        description=(
            "Young's modulus in NORMALISED units — use 1.0. This solver is "
            "unitless and only the stiffness-to-load ratio matters, so do NOT "
            "convert a named material to Pa. 'Steel' is E=1.0 here, not 2.1e11; "
            "large values make the linear solve diverge to NaN."
        ),
    )
    nu: float = Field(
        ..., description="Poisson's ratio, dimensionless. Typically 0.3.")


class Mesh(BaseModel):
    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float


class Multigrid(BaseModel):
    tol: float
    maxiter: int
    n_level: int

# class AxisRule(BaseModel):
#     axis: Literal["x", "y", "z"]
#     operator: Literal["equals", "greater_than", "less_than", "between"] = "equals"
#     value: float
#     value_max: Optional[float] = None  # For "between" operator

class AxisRule(BaseModel):
    axis: Literal["x", "y", "z", "diag"]  # added 'diag'
    operator: Literal["equals", "greater_than", "less_than", "between"] = "equals"
    value: float
    value_max: Optional[float] = None  # For "between" operator

class BCSelection(BaseModel):
    rules: List[AxisRule]
    tolerance: float = 1e-6

class BoundaryCondition(BaseModel):
    name: Optional[str] = None
    selection: BCSelection
    dofs: Dict[Literal["ux", "uy", "uz"], Optional[float]]

class ForceSelection(BaseModel):
    rules: List[AxisRule]
    tolerance: float = 1e-6

class PointForce(BaseModel):
    name: Optional[str] = None
    selection: ForceSelection
    forces: Dict[Literal["fx", "fy", "fz"], Optional[float]]
    divide_by_num_nodes: bool = False  # If True, divide force by number of selected nodes

class LocalVolumeConstraint(BaseModel):
    """Configuration for Wu et al.-style local volume-fraction penalty
    (max-solid feature size control). Translated into a soft penalty term
    added to the compliance objective inside MinimumCompliance._compute.
    """
    r_local: float                         # neighborhood radius
    p_local: float = 0.6                   # max local density allowed
    lambda_: float = 100.0                 # penalty weight
    q: int = 2                             # penalty exponent on violations


class Filter(BaseModel):
    # `r_min` is a scalar by default. A list (length = n_elements) switches
    # the agent to construct LocalFilter for per-element radius control
    # (HiTop-style local minimum length scale). Practically the LLM will
    # almost always emit a scalar — per-element arrays are intended for
    # programmatic setup (notebook cell, helper script).
    # LITE: units made explicit. r_min is measured in ELEMENTS, not in the
    # mesh's physical units. Nothing previously said so -- neither this schema
    # nor the prompt, whose only example was "filter radius of 2.0 -> 2.0" --
    # so given a problem stated in metres ("a 1 m long beam") the model
    # reasonably emitted r_min: 0.05 as a physical length. On a 20-element span
    # that is 1/20th of one element, i.e. no filtering at all, which produces
    # checkerboarding, diverges the multigrid solver, and yields
    # objective = nan on the first iteration with the run reported as a success.
    r_min: Union[float, List[float]] = Field(
        ...,
        description=(
            "Density filter radius in ELEMENT WIDTHS, not physical units. "
            "Typical values are 1.5 to 3.0. It must be at least 1.0 or the "
            "filter does nothing and the optimisation goes unstable. Do NOT "
            "convert from metres: on a 1 m beam meshed into 20 elements, a "
            "filter spanning 2 elements is r_min=2.0, never 0.1."
        ),
    )


class Problem(BaseModel):
    type: Literal["MinimumCompliance"]
    penalty_schedule: Optional[str] = None  # Keep this as None since it's always None
    void: float
    penalty: float
    E_mul: List[float]
    volume_fraction: List[float]
    heavyside: bool
    # Optional: list of element indices to lock as solid / void. The LLM
    # will typically leave these None; a programmatic/notebook setup can
    # populate them from a bbox or mask.
    passive_solid: Optional[List[int]] = None
    passive_void: Optional[List[int]] = None
    # Optional: max-solid feature size control (Wu-style local-VF penalty).
    local_volume_constraint: Optional[LocalVolumeConstraint] = None


class Optimizer(BaseModel):
    type: Literal["PGD", "MMA", "OC"]
    change_tol: Optional[float] = None  # Keep this as None (represents np.inf)
    fun_tol: float

class Optimization_settings(BaseModel):
    # LITE: defaulted. The example descriptions do not state an iteration count
    # -- they end at the optimizer's tolerances -- so the model has nothing to
    # base this on and omits the whole object. Under strict schema enforcement
    # (Together) it was filled in anyway; providers reached through OpenRouter
    # honour the schema more loosely, and the run then died with
    #     1 validation error for PydanticStructure
    #     optimization_settings  Field required
    # after producing a completely correct config otherwise.
    #
    # 100 is what every run in website/runs uses.
    num_iterations: int = Field(
        100, description="Number of optimization iterations. Default 100.")


class PydanticStructure(BaseModel):
    physics: List[Physics]
    mesh: List[Mesh]
    multigrid: Multigrid
    bc: List[BoundaryCondition]
    forces: Optional[List[PointForce]] = []  
    filter: Filter  
    problem: Problem  
    optimizer: Optimizer
    optimization_settings: Optimization_settings = Optimization_settings()


response_model = PydanticStructure

##############################################################
# System Prompt 
##############################################################

system_prompt = """
You are an expert in extracting parameters for topology optimization.
Always return valid JSON with the following structure:

{
  "physics": [...],
  "mesh": [...],
  "multigrid": {...},
  "bc": [...],
  "forces": [...],
  "filter": {
    "r_min": float  // filter radius in ELEMENT WIDTHS (>=1.0, typically 1.5-3.0), NOT metres
  },
  "problem": {
    "type": "MinimumCompliance",
    "penalty_schedule": null,  
    "void": float,  // Void material stiffness
    "penalty": float,  // SIMP penalty parameter 
    "E_mul": [float],  // Young's modulus multiplier(s)
    "volume_fraction": [float],  // Target volume fraction(s)
    "heavyside": bool  // Use Heaviside projection (default: true)
  },
  "optimizer": {
    "type": "PGD" | "MMA" | "OC",  // Optimizer type 
    "change_tol": float or null,  // null means infinity 
    "fun_tol": float  // Function tolerance 
  },
  "optimization_settings": {
    "num_iterations": int  // Number of optimization iterations to run
    }
}

Rules:
- ... (existing rules) ...

- The axis rule should be diag if its a diagonal instead of x,y,z directions. 

- For filter:
    - UNITS: this solver is unitless and runs on NORMALISED values. Use E=1.0
      and force magnitudes of order 1.0 (e.g. fy=-1.0). Never convert a named
      material to SI (steel is E=1.0 here, NOT 2.1e11) and never convert a
      stated load to newtons. Only the stiffness-to-load ratio affects the
      design; SI magnitudes make the solver diverge to NaN and the run silently
      returns a uniform block.
    - If a total load is spread over several selected nodes, set
      divide_by_num_nodes true, or each node receives the full value and the
      structure is loaded many times harder than asked.
    - Extract r_min (filter radius, in ELEMENT WIDTHS) if mentioned; if not
      mentioned use 1.5. Never derive it from a physical length — a value
      below 1.0 disables the filter and the optimisation diverges to NaN.
- For problem:
    - Extract void material stiffness if mentioned 
    - Extract SIMP penalty if mentioned 
    - Extract volume fraction if mentioned 
    - E_mul is typically [1.0] for single material
    - heavyside should be true unless explicitly stated otherwise
- For optimizer:
    - Extract optimizer type if mentioned (PGD, MMA, or OC)
    - change_tol should be null (infinity) unless explicitly specified
    - fun_tol (function tolerance) default is 1e-4
- Example interpretations:
    - "filter radius of 2.0" → r_min: 2.0   (element widths, not metres)
    - "20% volume fraction" → volume_fraction: [0.2]
    - "volume fraction of 0.3" → volume_fraction: [0.3]
    - "SIMP penalty of 3" → penalty: 3.0
    - "use PGD optimizer" → type: "PGD"
    - "convergence tolerance of 1e-5" → fun_tol: 1e-5
- For optimization_settings:
    - Extract num_iterations if mentioned (e.g., "run for 100 iterations")

"""


##############################################################
# Set up Pydanitc Agent 
##############################################################

class PydanticAgent(UserProxyAgent):
    def __init__(
        self,
        name="PydanticAgent",
        system_message=None,  # ✅ Accept system_message explicitly
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}, 
        generate=None,
        generate_pydantic=None,
        llm_config=None,
        llm_config_TO=None,
        **kwargs
    ):
        self.generate = generate
        self.llm_config = llm_config
        self.llm_config_TO = llm_config_TO

        super().__init__(
            name=name,
            system_message=system_message,  # ✅ Pass directly
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs
        )

        self._ipython = get_ipython()
        self.register_reply(Agent, PydanticAgent._generate_retrieve_user_reply, position=2)
    
    def generate_pydantic(self, system_prompt=system_prompt, prompt="", temperature=0.333,
                      max_tokens=4096, response_model=PydanticStructure):
        if system_prompt is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
    
        client = self.llm_config_TO["client_TO"]
        model = self.llm_config_TO["model_TO"]
        max_tokens = self.llm_config_TO["max_tokens_TO"]
    
        create = instructor.patch(
            create=client.chat.completions.create,
            mode=_instructor_mode(),  # LITE: was hardcoded JSON_SCHEMA
        )
    
        return create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            **_extra_body(),          # LITE: provider routing, see above
        )

    def make_pydantic(self, message):
        with torch.no_grad():
            try:
                now = datetime.now()
                result = make_structure_from_text(message, self.generate_pydantic)
                print("Time: ", datetime.now() - now)
                return result
            except Exception as e:
                # LITE: the original swallowed every failure as "Error or rate
                # limit", printed str(e), and slept 60s. That hides the cause of
                # the most common real failure. "'NoneType' object is not
                # subscriptable" is not a rate limit -- it is the provider
                # returning an error body with no `choices`, so the client does
                # choices[0] on None. Out of credits, an invalid key and an
                # unsupported response_format all look identical this way, and
                # the 60s sleep is paid on every one of them.
                import traceback
                print(f"Structured output failed: {type(e).__name__}: {e}")
                if "not subscriptable" in str(e) or "NoneType" in str(e):
                    print("   This usually means the API returned an error "
                          "instead of a completion — no `choices` in the "
                          "response. Most likely, in order:")
                    print("     1. the provider account is out of credit, or "
                          "the key is wrong/expired")
                    print("     2. rate limited")
                    print("     3. the model rejected the structured-output "
                          "mode — try TO_INSTRUCTOR_MODE=JSON (or MD_JSON)")
                    print("   Check the key and balance before assuming it is "
                          "the model.")
                traceback.print_exc()
                # Only a genuine rate limit is worth waiting out; sleeping a
                # minute on a bad key just makes the failure slower.
                if any(s in str(e).lower() for s in
                       ("rate", "429", "quota", "too many requests")):
                    print("   rate limited — waiting 60s")
                    time.sleep(60)
                return None

    def TOsolve(self, params: PydanticStructure):
        params_dict = params.model_dump()
        return {
            "physics": params_dict["physics"],
            "mesh": params_dict["mesh"],
            "multigrid": params_dict["multigrid"],
            "bc": params_dict["bc"],
            "forces": params_dict.get("forces", []),
            "filter": params_dict.get("filter", {"r_min": 1.5}),  # ✅ Add filter
            "problem": params_dict.get("problem", {  # ✅ Add problem with defaults
                "type": "MinimumCompliance",
                "penalty_schedule": None,
                "void": 1e-9,
                "penalty": 3.0,
                "E_mul": [1.0],
                "volume_fraction": [0.2],
                "heavyside": True
            }),
            "optimizer": params_dict.get("optimizer", {  # ✅ Add optimizer with defaults
                "type": "PGD",
                "change_tol": None,
                "fun_tol": 1e-4
            }),
            "optimization_settings": params_dict.get("optimization_settings", {
                "num_iterations": 100
            }),
        }

    def _generate_retrieve_user_reply(self, messages=None, sender=None, config=None):
        last_msg = self._oai_messages[sender][-1]
        params = self.make_pydantic(last_msg)

        if params is None:
            return False, {
                "role": "assistant",
                "content": json.dumps({"error": "Could not process input."})
            }

        TO_results = self.TOsolve(params)

        #Convert numpy types to JSON-safe types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        TO_results = convert_numpy(TO_results)

        ###############################################################
        # 💾 SAVE JSON TO DISK 
        ###############################################################
        save_dir = "original_TO_JSON"
        os.makedirs(save_dir, exist_ok=True)
        
        # Count existing files named version_X.json
        existing = [
            f for f in os.listdir(save_dir)
            if f.startswith("version_") and f.endswith(".json")
        ]
        
        # Determine next version number
        if not existing:
            next_version = 1
        else:
            nums = [
                int(f.replace("version_", "").replace(".json", ""))
                for f in existing
                if f.replace("version_", "").replace(".json", "").isdigit()
            ]
            next_version = max(nums) + 1
        
        save_path = os.path.join(save_dir, f"version_{next_version}.json")
        
        # Write JSON to disk
        with open(save_path, "w") as f:
            json.dump(TO_results, f, indent=2)
        
        ###############################################################
        
        # Return JSON output to the user
        return True, {
            "role": "assistant",
            "content": json.dumps(TO_results)
        }

