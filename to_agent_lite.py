# LITE 0: path bootstrap, must precede every other import in this file.
# `agents.*` lives at the repo root and `backend` lives beside this file, so a
# bare `python -c "import to_agent_lite"` would otherwise fail depending on cwd.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
# Works in both layouts: nested under TO-Agents/website_lite (agents/ one level
# up) and flat at the root of the standalone lite repo (agents/ right here).
_env_root = _os.environ.get("TO_AGENTS_ROOT", "").strip()
if _env_root:
    _REPO = _Path(_env_root).resolve()
elif (_HERE / "agents").is_dir():
    _REPO = _HERE
else:
    _REPO = _HERE.parent
for _p in (str(_REPO.resolve()), str(_HERE)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import numpy as np
from typing import List, Dict
from IPython import get_ipython
from autogen import UserProxyAgent, Agent
from agents.pydantic_agent import PydanticStructure
import asyncio
import nest_asyncio
import time
import os


#to_agent with multiscreenshots

# ---------------------------------------------------------------------------
# LITE BUILD — byte-for-byte copy of agents/to_agent_both.py except that the
# pyFANTOM backend is chosen at runtime instead of hard-bound to CUDA.
#
# Diff vs the original is exactly five places, all marked "# LITE:":
#   1. this import block            (pyFANTOM.CUDA  -> backend selector)
#   2. apply_boundary_conditions    (import cupy    -> backend.xp)
#   3. apply_point_forces           (import cupy    -> backend.xp)
#   4. two `.get()` calls           (-> backend.to_numpy, works on numpy too)
#   5. the LocalFilter branch       (CUDA-only; falls back on CPU)
#
# Set TO_BACKEND=cuda|cpu|auto (default auto). See backend.py.
# ---------------------------------------------------------------------------

# LITE 1: import backend BEFORE pyFANTOM.Physics — it puts PYFANTOM_PATH on
# sys.path and strips the shadowing `pyFANTOM/pyFANTOM` entry. Importing
# pyFANTOM.Physics first resolves `pyFANTOM` to the outer submodule directory,
# which has no Physics submodule, and dies with a confusing ModuleNotFoundError.
# backend.py re-exports these from pyFANTOM.CUDA or pyFANTOM.CPU.
from backend import (
    StructuredMesh3D,
    GeneralMesh,
    StructuredStiffnessKernel,
    UniformStiffnessKernel,
    GeneralStiffnessKernel,
    MultiGrid,
    CG,
    SPSOLVE,
    FiniteElement,
    StructuredFilter3D,
    GeneralFilter,
    LocalFilter,          # None on the CPU backend — see LITE 5
    MinimumCompliance,
    PGD,
    MMA,
    OC,
    BACKEND,
    HAS_LOCAL_FILTER,
    HAS_SKSPARSE,
    COARSE_SOLVER,
    to_numpy as _to_numpy,
    xp as _xp,
)

# Safe now: backend has repaired sys.path.
from pyFANTOM.Physics import LinearElasticity

print(f"[to_agent_lite] pyFANTOM backend: {BACKEND}")


# LITE 6: the CPU and CUDA classes share names but NOT signatures. Measured
# delta on MinimumCompliance — CUDA accepts four parameters CPU does not:
#     E_local, local_volume_constraint, passive_solid, passive_void
# Passing any of them to the CPU class raises TypeError and kills the run at
# the first optimization. Drop what the selected backend cannot accept, but say
# so loudly when a *meaningful* value is discarded, because silently ignoring
# a passive region or a local volume constraint solves a different problem.
def _supported_kwargs(cls, kwargs, _warned=set()):
    import inspect
    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in inspect.signature(cls.__init__).parameters.values()):
        return kwargs
    out, dropped = {}, []
    for k, v in kwargs.items():
        if k in accepted:
            out[k] = v
        elif v is not None:
            dropped.append(k)
    if dropped:
        key = (cls.__name__, tuple(sorted(dropped)))
        if key not in _warned:
            _warned.add(key)
            print(f"   ⚠️  {cls.__name__}: the {BACKEND} backend does not support "
                  f"{', '.join(sorted(dropped))} — these values were DROPPED. "
                  f"The problem being solved is not the one requested.")
    return out


class TOAgentBoth(UserProxyAgent):
    """
    Duplicate of TOAgent for use with the dual-pipeline _3d.py
    (depth + stress screenshots saved side-by-side).

    Behavior is identical to TOAgent — the visualization layer is what's
    different. With _3d.py = _3d_both.py active, each call to
    `problem.capture_solution_screenshots(...)` writes two subdirs:

        <screenshot_dir>/depth/{view}.png
        <screenshot_dir>/stress/{view}.png

    Downstream agents (VLLMAgentBoth, AI_JudgeBoth) read both subdirs.
    """

    # Toggle the final-state von Mises export (von_mises.npy in to_state_revision_N/).
    # Set TOAgentBoth.save_stress = False (class-wide) or agent.save_stress = False
    # (per-instance) to disable.
    save_stress = True

    # Mid-run snapshot: save the design (get_desvars) at this iteration to
    # to_state_revision_N/rho_snapshot.npy so the best-scoring design can later be
    # SOFT-RESTARTED from a still-malleable (not-yet-converged) state and have local
    # interventions applied before finishing convergence. If num_iterations <= this,
    # the snapshot falls back to the run's midpoint (num_iterations // 2). Set to None
    # to disable.
    snapshot_iter = 50

    def __init__(
        self,
        name="TOAgentBoth",
        system_message=None,
        llm_config=None,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )

        self._ipython = get_ipython()
        self.register_reply(Agent, TOAgentBoth._generate_retrieve_user_reply, position=2)
        
        # Track revision number
        self.revision_number = 0  # 0 = original, 1+ = revisions

    def apply_boundary_conditions(self, FE, mesh, bc_list: List[Dict]):
        """Apply Dirichlet boundary conditions to the finite element model"""
        cp = _xp   # LITE 2: cupy on CUDA, numpy on CPU
        
        axis_map = {"x": 0, "y": 1, "z": 2}
        dof_map = {"ux": 0, "uy": 1, "uz": 2}
        
        for bc in bc_list:
            selection = bc["selection"]
            dofs = bc["dofs"]
            tol = selection.get("tolerance", 1e-6)
            
            rules = selection.get("rules", [])
            mask = cp.ones(mesh.nodes.shape[0], dtype=bool)
            
            # Apply all selection rules with AND logic
            for rule in rules:
                idx = axis_map[rule["axis"]]
                operator = rule.get("operator", "equals")
                value = rule["value"]
                
                if operator == "equals":
                    mask &= cp.abs(mesh.nodes[:, idx] - value) < tol
                elif operator == "greater_than":
                    mask &= mesh.nodes[:, idx] > value
                elif operator == "less_than":
                    mask &= mesh.nodes[:, idx] < value
                elif operator == "between":
                    value_max = rule.get("value_max")
                    if value_max is None:
                        raise ValueError(f"'between' operator requires 'value_max' in rule: {rule}")
                    mask &= (mesh.nodes[:, idx] > value) & (mesh.nodes[:, idx] < value_max)
                else:
                    raise ValueError(f"Unknown operator: {operator}")
            
            node_ids = _to_numpy(cp.where(mask)[0]).astype(np.int32)  # LITE 4
            
            if node_ids.size == 0:
                print(f"⚠️ BC '{bc.get('name')}' selects no nodes.")
                continue
            
            # Apply boundary condition for each constrained DOF separately
            for dof_name, dof_value in dofs.items():
                if dof_value is not None:
                    dof_idx = dof_map[dof_name]
                    
                    # Create DOF mask with only this DOF constrained
                    dof_mask = np.zeros((1, 3), dtype=np.int32)
                    dof_mask[0, dof_idx] = 1
                    
                    FE.add_dirichlet_boundary_condition(
                        node_ids=node_ids,
                        dofs=dof_mask,
                        rhs=float(dof_value)
                    )
            
            print(f"✅ Applied BC '{bc.get('name')}': {len(node_ids)} nodes, DOFs={dofs}")

    def apply_point_forces(self, FE, mesh, force_list: List[Dict]):
        """Apply point forces to the finite element model"""
        cp = _xp   # LITE 3: cupy on CUDA, numpy on CPU
    
        # axis_map = {"x": 0, "y": 1, "z": 2, "diag": 3}
        axis_map = {"x": 0, "y": 1, "z": 2}
    
        # Precompute diagonal coordinate once (y - (0.5 - 0.5*x))
        if "diag" in [rule["axis"] for f in force_list for rule in f["selection"].get("rules", [])]:
            diag_coord = mesh.nodes[:, 1] - (0.5 - 0.5*mesh.nodes[:, 0])
            mesh.diag_coord = diag_coord  # store separately
    
            # Filter out first and last 5% along x
            tol = 1e-3
            diag_nodes = np.where(np.abs(mesh.diag_coord) < tol)[0]
            x_coords = mesh.nodes[diag_nodes, 0]
            filtered_nodes = diag_nodes[(x_coords > 0.05) & (x_coords < 0.95)]
            mesh.diag_nodes_filtered = filtered_nodes  # store for later use
    
        for force_spec in force_list:
            selection = force_spec["selection"]
            forces = force_spec["forces"]
            tol = selection.get("tolerance", 1e-3)
            divide_by_num_nodes = force_spec.get("divide_by_num_nodes", False)
    
            rules = selection.get("rules", [])
            mask = cp.ones(mesh.nodes.shape[0], dtype=bool)
    

            for rule in rules:
       
                if rule["axis"] == "diag":
        
                    # diag_mask = cp.zeros(mesh.nodes.shape[0], dtype=bool)
                    # diag_mask[cp.asarray(mesh.diag_nodes_filtered)] = True
        
                    total_force = forces.get("fy", 0.0)
                    node_ids = mesh.diag_nodes_filtered.astype(np.int32)
        
                    n = len(node_ids)
                    
                    if n == 0:
                        print("⚠️ No diagonal nodes found, skipping diagonal force.")
                        mask = None
                        break
        
                    force_vec = np.tile([0.0, total_force / n, 0.0], (n, 1))
        
                    FE.add_point_forces(
                        node_ids=node_ids,
                        forces=force_vec
                    )
        
                    print(f"✅ Applied diagonal force: {n} nodes, total Fy={total_force}")
        
                    mask = None
                    break
        
                # normal axis logic continues here...

                idx = axis_map[rule["axis"]]
                operator = rule.get("operator", "equals")
                value = rule["value"]
    
                if operator == "equals":
                    mask &= cp.abs(mesh.nodes[:, idx] - value) < tol
                elif operator == "greater_than":
                    mask &= mesh.nodes[:, idx] > value
                elif operator == "less_than":
                    mask &= mesh.nodes[:, idx] < value
                elif operator == "between":
                    value_max = rule.get("value_max")
                    if value_max is None:
                        raise ValueError(f"'between' operator requires 'value_max' in rule: {rule}")
                    mask &= (mesh.nodes[:, idx] > value) & (mesh.nodes[:, idx] < value_max)
                else:
                    raise ValueError(f"Unknown operator: {operator}")
        
            # CRITICAL: skip normal processing if diagonal already handled
            if mask is None:
                continue
    
            node_ids = _to_numpy(cp.where(mask)[0]).astype(np.int32)  # LITE 4
    
            if node_ids.size == 0:
                print(f"⚠️ Force '{force_spec.get('name')}' selects no nodes.")
                continue
    
            # Build force vector
            force_vec = np.array([[
                forces.get("fx") or 0.0,
                forces.get("fy") or 0.0,
                forces.get("fz") or 0.0
            ]], dtype=np.float64)
    
            # Divide by number of nodes if requested
            if divide_by_num_nodes:
                force_vec = force_vec / len(node_ids)
    
            FE.add_point_forces(
                node_ids=node_ids,
                forces=force_vec
            )
    
            print(f"✅ Applied Force '{force_spec.get('name')}': {len(node_ids)} nodes, "
                  f"Force={force_vec[0]}, divide_by_nodes={divide_by_num_nodes}")

    
    # def apply_point_forces(self, FE, mesh, force_list: List[Dict]):
    #     """Apply point forces to the finite element model"""
    #     import cupy as cp
        
    #     axis_map = {"x": 0, "y": 1, "z": 2}
        
    #     for force_spec in force_list:
    #         selection = force_spec["selection"]
    #         forces = force_spec["forces"]
    #         tol = selection.get("tolerance", 1e-6)
    #         divide_by_num_nodes = force_spec.get("divide_by_num_nodes", False)
            
    #         rules = selection.get("rules", [])
    #         mask = cp.ones(mesh.nodes.shape[0], dtype=bool)
            
    #         # Apply all selection rules with AND logic
    #         for rule in rules:
    #             idx = axis_map[rule["axis"]]
    #             operator = rule.get("operator", "equals")
    #             value = rule["value"]
                
    #             if operator == "equals":
    #                 mask &= cp.abs(mesh.nodes[:, idx] - value) < tol
    #             elif operator == "greater_than":
    #                 mask &= mesh.nodes[:, idx] > value
    #             elif operator == "less_than":
    #                 mask &= mesh.nodes[:, idx] < value
    #             elif operator == "between":
    #                 value_max = rule.get("value_max")
    #                 if value_max is None:
    #                     raise ValueError(f"'between' operator requires 'value_max' in rule: {rule}")
    #                 mask &= (mesh.nodes[:, idx] > value) & (mesh.nodes[:, idx] < value_max)
    #             else:
    #                 raise ValueError(f"Unknown operator: {operator}")
            
    #         node_ids = cp.where(mask)[0].get().astype(np.int32)
            
    #         if node_ids.size == 0:
    #             print(f"⚠️ Force '{force_spec.get('name')}' selects no nodes.")
    #             continue
            
    #         # Build force vector
    #         force_vec = np.array([[
    #             forces.get("fx") or 0.0,
    #             forces.get("fy") or 0.0,
    #             forces.get("fz") or 0.0
    #         ]], dtype=np.float64)
            
    #         # Divide by number of nodes if requested
    #         if divide_by_num_nodes:
    #             force_vec = force_vec / len(node_ids)
            
    #         FE.add_point_forces(
    #             node_ids=node_ids,
    #             forces=force_vec
    #         )
            
    #         print(f"✅ Applied Force '{force_spec.get('name')}': {len(node_ids)} nodes, "
    #               f"Force={force_vec[0]}, divide_by_nodes={divide_by_num_nodes}")
        
    def build_optimization(self, TO_results: Dict):
        """Build the complete optimization setup from configuration"""
        # Build FEA components
        physics_params = TO_results["physics"]
        mesh_params = TO_results["mesh"].copy()
        multigrid_params = TO_results["multigrid"]
        bc_list = TO_results.get("bc", [])
        force_list = TO_results.get("forces", [])
    
        physics = LinearElasticity(**physics_params)
        mesh_params.pop("physics", None)
        mesh = StructuredMesh3D(**mesh_params, physics=physics)
        kernel = StructuredStiffnessKernel(mesh=mesh)
        # LITE 7: pyFANTOM's MultiGrid defaults to coarse_solver='cholmod',
        # which needs scikit-sparse. That package is source-only and frequently
        # fails to build, so fall back to scipy's sparse LU when it is absent.
        # An explicit coarse_solver in the config always wins.
        multigrid_params.setdefault("coarse_solver", COARSE_SOLVER)
        if not HAS_SKSPARSE and multigrid_params["coarse_solver"] == "cholmod":
            print("   \u26a0\ufe0f  coarse_solver='cholmod' requested but scikit-sparse is "
                  "missing; using 'splu' instead.")
            multigrid_params["coarse_solver"] = "splu"
        solver = MultiGrid(kernel=kernel, mesh=mesh, **multigrid_params)
        FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)
    
        FE.reset_dirichlet_boundary_conditions()
        FE.reset_forces()
    
        if bc_list:
            self.apply_boundary_conditions(FE, mesh, bc_list)
        
        if force_list:
            self.apply_point_forces(FE, mesh, force_list)
    
        # Build filter — scalar r_min => fast convolution filter;
        # list r_min => LocalFilter for per-element minimum length scale.
        filter_params = TO_results.get("filter", {"r_min": 1.5})
        r_min_val = filter_params["r_min"]
        if isinstance(r_min_val, (list, tuple)):
            # LITE 5: LocalFilter is CUDA-only (pyFANTOM.CPU does not define it).
            # On CPU, collapse the per-element radii to their mean and use the
            # structured filter, so a per-element r_min request degrades instead
            # of crashing. Loud, because it changes the length-scale semantics.
            if HAS_LOCAL_FILTER:
                filter_obj = LocalFilter(mesh, np.asarray(r_min_val, dtype=np.float32))
                print(f"   Filter: LocalFilter (per-element r_min, "
                      f"min={float(min(r_min_val)):.2f}, max={float(max(r_min_val)):.2f})")
            else:
                r_mean = float(np.mean(np.asarray(r_min_val, dtype=np.float64)))
                filter_obj = StructuredFilter3D(mesh=mesh, r_min=r_mean)
                print(f"   ⚠️  Filter: LocalFilter unavailable on the {BACKEND} backend — "
                      f"using StructuredFilter3D with mean r_min={r_mean:.2f} "
                      f"(requested per-element min={float(min(r_min_val)):.2f}, "
                      f"max={float(max(r_min_val)):.2f}). Length-scale control is "
                      f"uniform, not per-element.")
        else:
            filter_obj = StructuredFilter3D(mesh=mesh, r_min=r_min_val)

        # Build problem — propagate optional passive-region and local-VF knobs.
        problem_params = TO_results.get("problem", {})
        lvc = problem_params.get("local_volume_constraint")
        if lvc is not None and hasattr(lvc, "model_dump"):
            lvc = lvc.model_dump()
        # LITE 6: filtered through _supported_kwargs so the CUDA-only knobs are
        # dropped (with a warning) instead of raising TypeError on the CPU class.
        problem = MinimumCompliance(**_supported_kwargs(MinimumCompliance, dict(
            FE=FE,
            filter=filter_obj,
            penalty_schedule=None,  # Always None for now
            void=problem_params.get("void", 1e-9),
            penalty=problem_params.get("penalty", 3.0),
            E_mul=problem_params.get("E_mul", [1.0]),
            volume_fraction=problem_params.get("volume_fraction", [0.2]),
            heavyside=problem_params.get("heavyside", True),
            passive_solid=problem_params.get("passive_solid"),
            passive_void=problem_params.get("passive_void"),
            local_volume_constraint=lvc,
        )))
        
        # Build optimizer
        optimizer_params = TO_results.get("optimizer", {})
        optimizer_type = optimizer_params.get("type", "PGD")
        
        change_tol = optimizer_params.get("change_tol")
        if change_tol is None:
            change_tol = np.inf
        
        if optimizer_type == "PGD":
            optimizer = PGD(
                problem=problem,
                change_tol=change_tol,
                fun_tol=optimizer_params.get("fun_tol", 1e-4),
            )
        elif optimizer_type == "MMA":
            optimizer = MMA(
                problem=problem,
                change_tol=change_tol,
                fun_tol=optimizer_params.get("fun_tol", 1e-4),
            )
        elif optimizer_type == "OC":
            optimizer = OC(
                problem=problem,
                change_tol=change_tol,
                fun_tol=optimizer_params.get("fun_tol", 1e-4),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        print(f"✅ Built optimization setup:")
        print(f"   Filter r_min: {filter_params['r_min']}")
        print(f"   Problem: {problem_params.get('type', 'MinimumCompliance')}")
        print(f"   Volume fraction: {problem_params.get('volume_fraction', [0.2])}")
        print(f"   Optimizer: {optimizer_type}")
        
        return physics, mesh, FE, filter_obj, problem, optimizer

    # LITE 8: screenshot capture on the CPU backend.
    #
    # pyFANTOM implements the capture chain ONLY on the CUDA side:
    #   Problem/CUDA/MinimumCompliance.capture_solution_screenshots
    #     -> FiniteElement/CUDA/FiniteElement.visualize_screenshot_density
    #        -> visualizers/_3d.capture_solution_screenshots_3D
    # Neither of the first two exists for CPU (FiniteElement's base declaration
    # is an abstract `raise NotImplementedError`), so on the zero-GPU path the
    # original call died with
    #   'MinimumCompliance' object has no attribute 'capture_solution_screenshots'
    # and the run continued to the judge, which then failed with
    #   "Need at least 2 screenshot directories, found 0".
    #
    # Both CUDA links are pure plumbing: they `.get()` cupy arrays and forward
    # to the third function, which takes plain arrays and renders through
    # k3d -> HTML -> headless Chromium with no GPU involved. So CPU can call it
    # directly. It is not re-exported from `visualizers/__init__`, hence the
    # private-module import.
    def _capture_screenshots(self, screenshot_dir):
        """Render depth + stress screenshots on whichever backend is active."""
        native = getattr(self.problem, "capture_solution_screenshots", None)
        if callable(native):
            return native(output_dir=screenshot_dir)

        from pyFANTOM.visualizers._3d import capture_solution_screenshots_3D
        import numpy as _np

        FE = self.FE
        if not getattr(FE, "is_3D", True):
            raise RuntimeError(
                "LITE: CPU screenshot fallback supports 3D meshes only; this "
                "problem is 2D. Use the CUDA backend for 2D capture.")

        rho = _to_numpy(self.problem.get_desvars())
        n_mat = getattr(self.problem, "n_material", 1)
        if n_mat > 1:
            rho = rho.reshape(n_mat, -1).T

        # Stress colouring is a nice-to-have: a failed FEA must not cost us the
        # depth renders, which are what the vision agent actually needs.
        stress, kw = None, {}
        try:
            stress = _to_numpy(self.problem.FEA(thresshold=True)["von_mises"])
            nz = stress[stress > 1e-10]
            if nz.size:
                kw["max_value"] = float(_np.percentile(nz, 99.0))
        except Exception as e:
            print(f"   ⚠️  von Mises unavailable ({type(e).__name__}: {e}); "
                  f"rendering depth only.")

        dof = FE.mesh.dof
        call = dict(
            nodes=_to_numpy(FE.mesh.nodes),
            elements=_to_numpy(FE.mesh.elements),
            f=_to_numpy(FE.rhs).reshape(-1, dof),
            c=_to_numpy(FE.kernel.constraints).reshape(-1, dof),
            rho=rho,
            output_dir=screenshot_dir,
            stress=stress,
            colormap="jet",
            stress_label="von Mises",
            **kw,
        )

        # The published pyFANTOM's renderer takes only
        #   nodes, elements, f, c, rho, output_dir, delay, *_color
        # while the stress arguments exist only in a locally modified copy. Drop
        # whatever this build cannot accept instead of raising TypeError, and say
        # so, because losing stress colouring silently would leave the vision
        # agent comparing depth-only renders while the prompts still say stress.
        import inspect as _inspect
        accepted = set(_inspect.signature(capture_solution_screenshots_3D).parameters)
        dropped = sorted(k for k in call if k not in accepted and call[k] is not None)
        if dropped:
            # vllm_agent_both reads `<dir>/depth/{view}.png` and
            # `<dir>/stress/{view}.png`. This renderer writes flat and cannot
            # colour by stress, so put the renders where the depth half is
            # expected and leave `stress/` genuinely absent. Duplicating depth
            # images into stress/ would satisfy the path check while feeding the
            # vision agent two identical pictures it was told differ -- a worse
            # failure than a missing directory, because nothing would report it.
            print(f"   ⚠️  this pyFANTOM's capture_solution_screenshots_3D does not "
                  f"accept {', '.join(dropped)} — rendering DEPTH ONLY into "
                  f"'{screenshot_dir}/depth'. No stress renders will exist; the "
                  f"vision agent sees half the imagery the prompts describe.")
        call["output_dir"] = os.path.join(screenshot_dir, "depth")
        os.makedirs(call["output_dir"], exist_ok=True)
        return capture_solution_screenshots_3D(
            **{k: v for k, v in call.items() if k in accepted})

    def run_optimization(self, num_iterations=None, screenshot_dir=None):
        """Run the optimization loop and capture screenshots"""    
        if not hasattr(self, 'optimizer'):
            raise RuntimeError("Optimizer not initialized. Process setup message first.")
        
        # Use stored num_iterations if not explicitly provided
        if num_iterations is None:
            num_iterations = self.num_iterations
            if num_iterations is None:
                raise RuntimeError("num_iterations not specified. Provide it in optimization_settings or pass it to run_optimization().")
        
        # CuPy → NumPy helper (also used for the final export below).
        def to_numpy(x):
            return x.get() if hasattr(x, "get") else x

        # Resolve the mid-run snapshot iteration: the configured value if it lands
        # strictly inside the run, else the run's midpoint (so short runs still get one).
        state_dir = f"to_state_revision_{self.revision_number}"
        snap_at = None
        if self.snapshot_iter:
            snap_at = self.snapshot_iter if self.snapshot_iter < num_iterations else max(1, num_iterations // 2)

        start = time.time()
        objective_history = []

        # --- Optional live-progress hooks (opt-in; off unless TO_FRAME_EVERY is set) ---
        # Used by the demo website to stream a "watch it optimize" movie + live curve
        # and to honour a Stop button. A no-op for the notebook (env var unset).
        _frame_every = int(os.environ.get("TO_FRAME_EVERY", "0") or "0")
        _frames_dir = os.path.join(state_dir, "frames") if _frame_every else None
        if _frames_dir:
            os.makedirs(_frames_dir, exist_ok=True)

        def _write_progress(cur_iter):
            """Dump the running density field + objective curve for the UI."""
            np.save(os.path.join(_frames_dir, f"rho_{cur_iter:05d}.npy"),
                    to_numpy(self.problem.get_desvars()))
            with open(os.path.join(_frames_dir, "progress.json"), "w") as _pf:
                json.dump({"iter": int(cur_iter),
                           "num_iterations": int(num_iterations),
                           "revision": int(self.revision_number),
                           "objective_history": [float(o) for o in objective_history]}, _pf)

        for i in range(num_iterations):
            self.optimizer.iter()
            objective_history.append(self.optimizer.logs()['objective'])
            if snap_at is not None and (i + 1) == snap_at:
                os.makedirs(state_dir, exist_ok=True)
                np.save(os.path.join(state_dir, "rho_snapshot.npy"),
                        to_numpy(self.problem.get_desvars()))
                with open(os.path.join(state_dir, "snapshot_meta.json"), "w") as _f:
                    json.dump({"snapshot_iteration": int(snap_at),
                               "num_iterations": int(num_iterations),
                               "objective_at_snapshot": float(objective_history[-1])}, _f)
                print(f"   💾 mid-run snapshot at iter {snap_at}/{num_iterations} "
                      f"→ {state_dir}/rho_snapshot.npy (obj={objective_history[-1]:.4f})")

            # Stream a movie frame every N iterations (and always on the last one).
            if _frames_dir and ((i + 1) % _frame_every == 0 or (i + 1) == num_iterations):
                _write_progress(i + 1)

            # Cooperative Stop: the website drops a STOP file in the run cwd.
            if os.path.exists("STOP"):
                print(f"\n🛑 Stop requested — halting optimization at iter {i + 1}/{num_iterations}.")
                if _frames_dir:
                    _write_progress(i + 1)
                break

        end = time.time()
        time_per_iter = (end - start) / len(objective_history)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Total iterations: {num_iterations}")
        print(f"   Time per iteration: {time_per_iter:.4f}s")
        
        # Capture screenshots AFTER optimization
        if screenshot_dir is None:
            if self.revision_number == 0:
                screenshot_dir = "screenshots"
            else:
                screenshot_dir = f"screenshots_revision_{self.revision_number}"
        
        # Create directory if it doesn't exist
        os.makedirs(screenshot_dir, exist_ok=True)

        # CHANGE TO ENABLE AUTOMATED RUN TIME 
        try:
            asyncio.get_running_loop()
            nest_asyncio.apply()
        except RuntimeError:
            pass
        
        # ===== ORIGINAL (commented out for trial) =====
        # func = self.problem.capture_solution_screenshots
        # if asyncio.iscoroutinefunction(func):
        #     asyncio.run(func(output_dir=screenshot_dir))
        # else:
        #     func(output_dir=screenshot_dir)
        # ===== END ORIGINAL =====

        # ===== TRIAL: await the screenshot Task so PNGs actually get written =====
        # Previously the returned Task from Jupyter's running-loop branch was
        # discarded, so Playwright never finished before the agent moved on.
        #
        # If a Stop was requested, skip the (expensive, browser-based) capture so
        # the interrupt feels immediate. The lightweight movie frames and the
        # exported rho.npy below still record where the run stopped.
        if os.path.exists("STOP"):
            print("🛑 Stop requested — skipping final screenshot capture.")
        else:
            result = self._capture_screenshots(screenshot_dir)   # LITE 8

            if asyncio.iscoroutine(result):
                asyncio.run(result)
            elif isinstance(result, asyncio.Task):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(result)
        # ===== END TRIAL =====

        
    
        # print(f"\n📸 Capturing solution screenshots to {screenshot_dir}/...")
        # nest_asyncio.apply()
        # asyncio.run(self.problem.capture_solution_screenshots(output_dir=screenshot_dir))


        ###############################################################
        # 💾 EXPORT TOPOLOGY OPTIMIZATION STATE (for other agents)
        ###############################################################
        # state_dir and to_numpy() are defined above (before the loop, for the snapshot).
        os.makedirs(state_dir, exist_ok=True)

        np.save(os.path.join(state_dir, "nodes.npy"),
                to_numpy(self.mesh.nodes))

        np.save(os.path.join(state_dir, "elements.npy"),
                to_numpy(self.mesh.elements))

        np.save(os.path.join(state_dir, "rho.npy"),
                to_numpy(self.problem.get_desvars())
               )

        # Persist the final per-element von Mises stress alongside rho.npy so
        # downstream agents/analysis can reload it without re-solving the FE.
        # Reuses the cache populated by capture_solution_screenshots when
        # available (avoids a redundant FE solve on large meshes).
        if self.save_stress:
            try:
                von_mises = to_numpy(self.problem.get_last_von_mises())
                np.save(os.path.join(state_dir, "von_mises.npy"), von_mises)
            except Exception as e:
                print(f"⚠️  Skipped von_mises.npy export ({e})")

        print(f"\n💾 Exported TO state to '{state_dir}/'")

        ###############################################################

        return objective_history, time_per_iter, screenshot_dir
    
    def _generate_retrieve_user_reply(self, messages=None, sender=None, config=None): 
        """Handle incoming setup messages and run optimization"""
        last_msg = self._oai_messages[sender][-1]
        
        try:
            data = json.loads(last_msg["content"])
            
            # Detect if this is a revision by checking for 'revised_json' field
            is_revision = "revised_json" in data
            
            if is_revision:
                self.revision_number += 1
                print(f"\n🔄 Detected revision #{self.revision_number}")
            
            # Check if this is from ReviseAgent (has 'revised_json' field)
            if "revised_json" in data:
                print("✅ Using revised_json from ReviseAgent")
                data = data["revised_json"]
            
            # Wrap physics and mesh in lists if they're dicts
            if isinstance(data.get("physics"), dict):
                data["physics"] = [data["physics"]]
            if isinstance(data.get("mesh"), dict):
                data["mesh"] = [data["mesh"]]
            
            # Convert to Pydantic which enforces correct types
            struct = PydanticStructure(**data)
            TO_results = {
                "physics": struct.physics[0].model_dump(),
                "mesh": struct.mesh[0].model_dump(),
                "multigrid": struct.multigrid.model_dump(),
                "bc": [bc.model_dump() for bc in struct.bc],
                "forces": [f.model_dump() for f in struct.forces] if struct.forces else [],
                "filter": struct.filter.model_dump() if struct.filter else {"r_min": 1.5},
                "problem": struct.problem.model_dump() if struct.problem else {},
                "optimizer": struct.optimizer.model_dump() if struct.optimizer else {},
                "optimization_settings": struct.optimization_settings.model_dump() if struct.optimization_settings else {},
            }
            
            print("✅ Structured and validated:", TO_results)
            
            physics, mesh, FE, filter_obj, problem, optimizer = self.build_optimization(TO_results)
    
            # Store components
            self.physics = physics
            self.mesh = mesh
            self.FE = FE
            self.filter_obj = filter_obj
            self.problem = problem
            self.optimizer = optimizer
            self.num_iterations = TO_results["optimization_settings"].get("num_iterations")

            # Run optimization automatically
            if is_revision:
                print(f"\n🚀 Starting optimization (Revision #{self.revision_number})...")
            else:
                print("\n🚀 Starting optimization (Original)...")
            
            objective_history, time_per_iter, screenshot_dir = self.run_optimization()

            ###############################################################
            # 💾 SAVE OBJECTIVE HISTORY AS version_X.json
            ###############################################################
            save_dir = "objective_history"
            os.makedirs(save_dir, exist_ok=True)
            
            # File numbering: version_1.json corresponds to revision_number 0
            version_number = self.revision_number + 1
            save_path = os.path.join(save_dir, f"version_{version_number}.json")
            
            objective_data = {
                "revision_number": self.revision_number,
                "version_number": version_number,
                "objective_history": objective_history,
                "time_per_iter": time_per_iter,
                "num_iterations": len(objective_history)
            }
            
            with open(save_path, "w") as f:
                json.dump(objective_data, f, indent=2)
            
            ###############################################################

            # ----- Physics metrics for downstream agent feedback -----
            try:
                _desvars = self.problem.get_desvars()
                if hasattr(_desvars, "get"):
                    _desvars = _desvars.get()
                achieved_vf = float((_desvars > 0.5).mean())
            except Exception as e:
                print(f"⚠️  Achieved VF computation failed: {e}")
                achieved_vf = None

            try:
                solver_residual = float(self.problem.get_last_residual())
            except Exception as e:
                print(f"⚠️  Residual retrieval failed: {e}")
                solver_residual = None

            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "status": "Optimization complete",
                    # "revision_number": self.revision_number,
                    "state": {
                        "revision": self.revision_number,
                        "state_dir": f"to_state_revision_{self.revision_number}",
                        "artifacts": (
                            ["nodes.npy", "elements.npy", "rho.npy", "von_mises.npy"]
                            if self.save_stress
                            else ["nodes.npy", "elements.npy", "rho.npy"]
                        )
                    },
                    "screenshot_dir": screenshot_dir,
                    "mesh": {
                        "nx": TO_results["mesh"]["nx"],
                        "ny": TO_results["mesh"]["ny"],
                        "nz": TO_results["mesh"]["nz"]
                    },
                    "num_bcs": len(TO_results["bc"]),
                    "num_forces": len(TO_results["forces"]),
                    "filter_r_min": TO_results["filter"]["r_min"],
                    "volume_fraction": TO_results["problem"]["volume_fraction"],
                    "achieved_volume_fraction": achieved_vf,
                    "solver_residual": solver_residual,
                    "optimizer": TO_results["optimizer"]["type"],
                    "num_iterations": TO_results["optimization_settings"].get("num_iterations"),
                    "time_per_iter": time_per_iter,
                    "objective_history": objective_history
                })
            }
        
        except Exception as e:
            import traceback
            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "error": f"TOAgent failed: {str(e)}",
                    "traceback": traceback.format_exc()
                })
            }
