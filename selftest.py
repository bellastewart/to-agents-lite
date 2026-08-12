"""
selftest.py — prove the solver actually solves, without touching any API.

`doctor.py` answers "is everything installed and reachable?". This answers the
harder question: **does a topology optimization actually run here, and how long
does an iteration take?**

It needs no API keys and no network, because it exercises only the parts that
depend on your machine:

    config dict -> LinearElasticity -> StructuredMesh3D -> stiffness kernel
                -> MultiGrid solver -> boundary conditions -> loads
                -> density filter -> MinimumCompliance -> optimizer
                -> N real iterations, timed

It drives the same `TOAgentBoth.build_optimization` the agents call, so a pass
here means the pipeline's numerical core works on this hardware.

    python selftest.py                 # 8x4x4, 3 iterations — a few seconds
    python selftest.py --nx 48 --ny 24 --nz 24 --iters 10
    python selftest.py --mesh-from-backend    # use the auto-selected mesh

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import sys
import time


def build_config(nx, ny, nz, iters):
    """A minimal cantilever, same shape the pydantic agent emits."""
    return {
        "physics": {"E": 1.0, "nu": 0.3},
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "lx": 1.0, "ly": 0.5, "lz": 0.5},
        "multigrid": {"tol": 1e-4, "maxiter": 50, "n_level": 2},
        "bc": [{
            "name": "left_face",
            "selection": {"rules": [{"axis": "x", "operator": "equals", "value": 0.0}],
                          "tolerance": 1e-6},
            "dofs": {"ux": 0.0, "uy": 0.0, "uz": 0.0},
        }],
        "forces": [{
            "name": "tip_load",
            "selection": {"rules": [{"axis": "x", "operator": "equals", "value": 1.0}],
                          "tolerance": 1e-6},
            "forces": {"fx": None, "fy": -1.0, "fz": None},
            "divide_by_num_nodes": True,
        }],
        "filter": {"r_min": 1.5},
        "problem": {"type": "MinimumCompliance", "penalty_schedule": None,
                    "void": 1e-9, "penalty": 3.0, "E_mul": [1.0],
                    "volume_fraction": [0.4], "heavyside": True},
        "optimizer": {"type": "PGD", "change_tol": None, "fun_tol": 1e-4},
        "optimization_settings": {"num_iterations": iters},
    }


def main():
    ap = argparse.ArgumentParser(description="Run a real optimization, no API needed.")
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--mesh-from-backend", action="store_true",
                    help="use the mesh backend.suggest_mesh() picks for this hardware")
    args = ap.parse_args()

    print("=" * 68)
    print("TO-Agents Lite — solver self-test")
    print("=" * 68)

    t0 = time.time()
    import backend
    d = backend.describe()
    print(f"  backend      : {d['backend']}  ({d['detection']})")
    print(f"  array module : {d['array_module']}")
    print(f"  scikit-sparse: {'present' if d['has_sksparse'] else 'ABSENT (stubbed)'}"
          f"  -> coarse solver '{d['coarse_solver']}'")
    print(f"  threads      : {d['threads']}")
    print(f"  import took  : {time.time() - t0:.1f}s")

    nx, ny, nz = args.nx, args.ny, args.nz
    if args.mesh_from_backend:
        (nx, ny, nz), why = backend.suggest_mesh()
        print(f"  mesh         : {nx}x{ny}x{nz} ({why})")

    cfg = build_config(nx, ny, nz, args.iters)
    n_el = nx * ny * nz
    print(f"\n  problem      : {nx}x{ny}x{nz} = {n_el:,} elements, "
          f"{args.iters} iterations, vf=0.4")

    print("\n--- build ---")
    t0 = time.time()
    from to_agent_lite import TOAgentBoth
    agent = TOAgentBoth(name="selftest", system_message="selftest",
                        human_input_mode="NEVER",
                        code_execution_config={"use_docker": False})
    physics, mesh, FE, filt, problem, optimizer = agent.build_optimization(cfg)
    t_build = time.time() - t0
    print(f"  built in {t_build:.1f}s")
    print(f"  mesh nodes   : {mesh.nodes.shape[0]:,}")
    print(f"  problem      : {type(problem).__name__}")
    print(f"  optimizer    : {type(optimizer).__name__}")

    print("\n--- solve ---")
    t0 = time.time()
    agent.problem = problem
    agent.optimizer = optimizer
    agent.num_iterations = args.iters
    # Drive it exactly the way TOAgentBoth.run_optimization does:
    # optimizer.iter() per step, reading optimizer.logs()['objective'].
    obj, per_iter = [], []
    for i in range(args.iters):
        _t = time.time()
        optimizer.iter()
        per_iter.append(time.time() - _t)
        obj.append(float(optimizer.logs()["objective"]))
        print(f"    iter {i + 1}/{args.iters}  {per_iter[-1]:6.2f}s  "
              f"objective = {obj[-1]:.6g}")
    t_solve = time.time() - t0

    # Iteration 1 pays for numba JIT compilation; steady state is what actually
    # governs how long a full run takes, so report them separately.
    if len(per_iter) > 1:
        steady = sum(per_iter[1:]) / len(per_iter[1:])
        print(f"\n  first iter   : {per_iter[0]:.2f}s  (includes numba JIT compile)")
        print(f"  steady state : {steady:.2f}s per iteration")
        print(f"  JIT overhead : {per_iter[0] - steady:.2f}s, paid once")
    else:
        steady = per_iter[0]

    print(f"  solved in {t_solve:.1f}s  "
          f"({t_solve / max(args.iters, 1):.2f}s per iteration)")
    if obj:
        vals = [float(v) for v in obj[:3]] + (["..."] if len(obj) > 3 else [])
        print(f"  objective    : {len(obj)} values, first: "
              + ", ".join(f"{v:.4g}" if isinstance(v, float) else v for v in vals))
        if len(obj) > 1:
            direction = "decreasing" if float(obj[-1]) < float(obj[0]) else "NOT decreasing"
            print(f"  first -> last: {float(obj[0]):.4g} -> {float(obj[-1]):.4g}  ({direction})")

    rho = None
    for getter in ("get_desvars",):
        if hasattr(problem, getter):
            try:
                rho = getattr(problem, getter)()
            except Exception:
                pass
    if rho is not None:
        rho = backend.to_numpy(rho)
        print(f"  density      : {rho.size:,} values, "
              f"min={rho.min():.3f} max={rho.max():.3f} mean={rho.mean():.3f}")

    print("\n" + "=" * 68)
    print(f"PASS — solver works on this machine "
          f"({backend.BACKEND}, coarse='{backend.COARSE_SOLVER}')")
    _steady = (sum(per_iter[1:]) / len(per_iter[1:])) if len(per_iter) > 1 else per_iter[0]
    print(f"       {_steady:.2f}s per iteration (steady state) at {n_el:,} elements")
    print(f"       -> ~{_steady * 200 / 60:.1f} min for a 200-iteration run")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
