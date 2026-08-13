"""
viz.py — headless renderers for the demo UI.

Two jobs, both pure/offline (no GPU, no Chromium):

1. render_setup_diagram(config, out_path)
   A schematic of the problem BEFORE optimization: the domain box, the
   Dirichlet BC regions, and the load arrows — drawn straight from the parsed
   config JSON's geometric predicates (mesh[0].l*, bc[].selection.rules,
   forces[].forces).  Mirrors the node-selection semantics in
   agents/to_agent_both.py.

2. render_density_frame(npy_path, config, out_path)
   One movie frame: reshape a saved rho density array to the mesh grid and
   project it to a 2D silhouette image with a colormap.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as mpimg
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

_AXIS = {"x": 0, "y": 1, "z": 2}


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def _mesh_dims(config):
    m = (config.get("mesh") or [{}])[0]
    return (
        int(m.get("nx", 0)), int(m.get("ny", 0)), int(m.get("nz", 0)),
        float(m.get("lx", 1.0)), float(m.get("ly", 1.0)), float(m.get("lz", 1.0)),
    )


def _sample_grid(lx, ly, lz, n=24):
    """A coarse node grid over the domain, spacing roughly uniform per axis."""
    L = max(lx, ly, lz)
    def na(l):
        return max(2, int(round(n * (l / L)))) if l > 0 else 2
    xs = np.linspace(0, lx, na(lx))
    ys = np.linspace(0, ly, na(ly))
    zs = np.linspace(0, lz, na(lz))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    spacing = max((xs[1] - xs[0]) if len(xs) > 1 else lx,
                  (ys[1] - ys[0]) if len(ys) > 1 else ly,
                  (zs[1] - zs[0]) if len(zs) > 1 else lz)
    return pts, spacing


def _rule_mask(pts, rule, vis_tol):
    """Boolean mask for one AxisRule, matching to_agent_both semantics."""
    axis = rule.get("axis", "x")
    op = rule.get("operator", "equals")
    val = float(rule.get("value", 0.0))
    if axis == "diag":
        # Special diagonal used for loads: y - (0.5 - 0.5 x), restricted band in x.
        x, y = pts[:, 0], pts[:, 1]
        d = y - (0.5 - 0.5 * x)
        return (np.abs(d) < max(vis_tol, 0.03)) & (x > 0.05) & (x < 0.95)
    c = pts[:, _AXIS.get(axis, 0)]
    if op == "equals":
        return np.abs(c - val) <= vis_tol
    if op == "greater_than":
        return c > val
    if op == "less_than":
        return c < val
    if op == "between":
        vmax = float(rule.get("value_max") if rule.get("value_max") is not None else val)
        return (c > val) & (c < vmax)
    return np.abs(c - val) <= vis_tol


def _selection_mask(pts, selection, spacing):
    rules = (selection or {}).get("rules", []) or []
    tol = float((selection or {}).get("tolerance", 1e-6) or 1e-6)
    vis_tol = max(tol, 0.6 * spacing)
    mask = np.ones(len(pts), dtype=bool)
    for r in rules:
        mask &= _rule_mask(pts, r, vis_tol)
    return mask


def _box_edges(lx, ly, lz):
    c = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
                  [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]], float)
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return c, e


# --------------------------------------------------------------------------- #
# 1. setup diagram
# --------------------------------------------------------------------------- #
def render_setup_plotly(config, out_html):
    """Interactive (drag-to-rotate) 3D setup diagram, written as a self-contained
    HTML file. Z is the vertical axis; proportions are true (aspectmode='data')."""
    import plotly.graph_objects as go

    nx, ny, nz, lx, ly, lz = _mesh_dims(config)
    pts, spacing = _sample_grid(lx, ly, lz, n=46)   # denser sampling → fuller faces/lines
    traces = []

    # domain box as a single line trace (None separators between edges)
    corners, edges = _box_edges(lx, ly, lz)
    bx, by, bz = [], [], []
    for a, b in edges:
        bx += [corners[a][0], corners[b][0], None]
        by += [corners[a][1], corners[b][1], None]
        bz += [corners[a][2], corners[b][2], None]
    traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                               line=dict(color="#c7c5bf", width=2, dash="dash"),
                               name="domain", hoverinfo="skip", showlegend=False))

    # boundary conditions
    for i, bc in enumerate(config.get("bc", []) or []):
        mask = _selection_mask(pts, bc.get("selection"), spacing)
        if not mask.any():
            continue
        dofs = bc.get("dofs", {}) or {}
        fixed = [d for d in ("ux", "uy", "uz") if dofs.get(d) is not None]
        name = bc.get("name") or f"BC {i+1}"
        traces.append(go.Scatter3d(
            x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2], mode="markers",
            marker=dict(size=4.5, color="#f778ba", symbol="square"),
            name=f"{name}: fix {','.join(fixed) or 'none'} ({int(mask.sum())})",
            hovertemplate="fixed %{text}<extra></extra>",
            text=[",".join(fixed)] * int(mask.sum())))

    # loads: sample points + a cone arrowhead + a shaft line
    diag = float(np.sqrt(lx**2 + ly**2 + lz**2))
    for i, frc in enumerate(config.get("forces", []) or []):
        mask = _selection_mask(pts, frc.get("selection"), spacing)
        if not mask.any():
            continue
        comps = frc.get("forces", {}) or {}
        vec = np.array([comps.get("fx") or 0.0, comps.get("fy") or 0.0, comps.get("fz") or 0.0], float)
        if np.linalg.norm(vec) == 0:
            continue
        loc = pts[mask].mean(axis=0)
        u = vec / np.linalg.norm(vec)
        L = 0.30 * diag
        cs = ",".join(f"{k}={v}" for k, v in comps.items() if v)
        name = frc.get("name") or f"Load {i+1}"
        traces.append(go.Scatter3d(
            x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2], mode="markers",
            marker=dict(size=4, color="#ffa657", opacity=0.7), name=f"{name}: {cs}",
            hoverinfo="skip"))
        tail = loc - u * L
        traces.append(go.Scatter3d(x=[tail[0], loc[0]], y=[tail[1], loc[1]], z=[tail[2], loc[2]],
                                   mode="lines", line=dict(color="#ffa657", width=6),
                                   showlegend=False, hoverinfo="skip"))
        traces.append(go.Cone(x=[loc[0]], y=[loc[1]], z=[loc[2]],
                              u=[u[0]*L*0.5], v=[u[1]*L*0.5], w=[u[2]*L*0.5],
                              anchor="tip", colorscale=[[0, "#ffa657"], [1, "#ffa657"]],
                              showscale=False, sizemode="absolute", sizeref=0.4*L,
                              hoverinfo="skip"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#141414"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(size=11, color="#141414"),
                    x=0, y=1, orientation="v"),
        scene=dict(
            xaxis=dict(title="x", backgroundcolor="#ffffff", gridcolor="#e6e4df", color="#6b6b6b"),
            yaxis=dict(title="y", backgroundcolor="#ffffff", gridcolor="#e6e4df", color="#6b6b6b"),
            zaxis=dict(title="z", backgroundcolor="#ffffff", gridcolor="#e6e4df", color="#6b6b6b"),
            aspectmode="data",
            # Default to a front elevation: x horizontal, y vertical (up), looking
            # along z with a slight tilt so depth reads. Fully draggable afterward.
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=0.2, y=0.25, z=1.9)),
        ),
    )
    fig.write_html(out_html, include_plotlyjs="directory", full_html=True,
                   config={"displaylogo": False, "responsive": True})
    return out_html


def render_setup_diagram(config, out_path):
    nx, ny, nz, lx, ly, lz = _mesh_dims(config)
    pts, spacing = _sample_grid(lx, ly, lz)

    fig = Figure(figsize=(6.4, 5.2), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    # domain box
    corners, edges = _box_edges(lx, ly, lz)
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color="#7d8590", lw=1.0, alpha=0.7)

    legend = []

    # boundary conditions
    for i, bc in enumerate(config.get("bc", []) or []):
        mask = _selection_mask(pts, bc.get("selection"), spacing)
        if not mask.any():
            continue
        dofs = bc.get("dofs", {}) or {}
        fixed = [d for d in ("ux", "uy", "uz") if dofs.get(d) is not None]
        name = bc.get("name") or f"BC {i+1}"
        color = "#f778ba"
        ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                   s=16, c=color, marker="s", depthshade=False, alpha=0.9)
        legend.append((color, f"{name}: fix {','.join(fixed) or 'none'} ({int(mask.sum())} nodes)", "s"))

    # loads
    diag = np.sqrt(lx**2 + ly**2 + lz**2)
    for i, frc in enumerate(config.get("forces", []) or []):
        mask = _selection_mask(pts, frc.get("selection"), spacing)
        if not mask.any():
            continue
        comps = frc.get("forces", {}) or {}
        vec = np.array([comps.get("fx") or 0.0, comps.get("fy") or 0.0, comps.get("fz") or 0.0], float)
        if np.linalg.norm(vec) == 0:
            continue
        loc = pts[mask].mean(axis=0)
        u = vec / np.linalg.norm(vec)
        L = 0.28 * diag
        name = frc.get("name") or f"Load {i+1}"
        color = "#ffa657"
        ax.scatter(pts[mask, 0], pts[mask, 1], pts[mask, 2],
                   s=14, c=color, marker="o", depthshade=False, alpha=0.6)
        # draw arrow pointing INTO the load location along the force direction
        ax.quiver(loc[0] - u[0]*L, loc[1] - u[1]*L, loc[2] - u[2]*L,
                  u[0]*L, u[1]*L, u[2]*L, color=color, lw=2.2, arrow_length_ratio=0.28)
        mag = np.linalg.norm(vec)
        comp_str = ",".join(f"{k}={v}" for k, v in comps.items() if v)
        legend.append((color, f"{name}: {comp_str} (|F|={mag:g})", "^"))

    # cosmetics
    ax.set_xlabel("x", color="#9198a1"); ax.set_ylabel("y", color="#9198a1"); ax.set_zlabel("z", color="#9198a1")
    ax.set_title(f"Problem setup — {nx}×{ny}×{nz} grid, domain {lx:g}×{ly:g}×{lz:g}",
                 color="#e6edf3", fontsize=10)
    try:
        ax.set_box_aspect((lx, ly, lz))
    except Exception:
        pass
    ax.tick_params(colors="#6e7681", labelsize=7)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
        pane.pane.set_edgecolor("#30363d")
    ax.view_init(elev=22, azim=-58)

    if legend:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker=mk, color="none", markerfacecolor=col,
                          markersize=9, label=lab) for col, lab, mk in legend]
        leg = ax.legend(handles=handles, loc="upper left", fontsize=7.5,
                        framealpha=0.15, labelcolor="#e6edf3")
        leg.get_frame().set_edgecolor("#30363d")

    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    return out_path


# --------------------------------------------------------------------------- #
# 2. density movie frame
# --------------------------------------------------------------------------- #
def _reshape_rho(rho, grid, config):
    """Reshape a flat density array to a 3D grid.

    Prefers the actual element grid recorded at dump time (``grid`` = [gx,gy,gz]);
    falls back to the config's nx/ny/nz, then to any matching permutation. Returns
    None if nothing matches (pyFANTOM bumps the requested dims, so the config
    grid alone is unreliable)."""
    n = rho.size
    candidates = []
    if grid and all(grid) and int(np.prod(grid)) == n:
        candidates.append(tuple(int(g) for g in grid))
    nx, ny, nz, *_ = _mesh_dims(config)
    candidates += [(nx, ny, nz), (nz, ny, nx), (nx, nz, ny)]
    for shape in candidates:
        if all(shape) and int(np.prod(shape)) == n:
            try:
                return rho.reshape(shape)
            except Exception:
                continue
    return None


def render_density_frame(npy_path, config, out_path, grid=None, proj_axis=None, cmap="magma"):
    """Reshape a saved rho array to the mesh grid, project along its thinnest axis
    to a 2D silhouette, and save a PNG movie frame."""
    rho = np.asarray(np.load(npy_path), dtype=float).ravel()
    vol = _reshape_rho(rho, grid, config)
    if vol is None:
        img = rho.reshape(1, -1)          # last-resort: 1D strip
    else:
        axis = proj_axis if proj_axis is not None else int(np.argmin(vol.shape))
        img = vol.max(axis=axis)          # densest material along the thickness
        img = np.flipud(img.T)            # first remaining axis horizontal, origin lower-left
    img = np.clip(img, 0.0, 1.0)
    # Magma material with transparent voids, so the structure floats on the dark
    # viewport (no black box). Bright magma reads clearly against the dark stage.
    try:
        cmap_fn = cm.get_cmap(cmap)
    except Exception:
        import matplotlib
        cmap_fn = matplotlib.colormaps[cmap]
    rgba = cmap_fn(img)                                    # (H, W, 4) float
    rgba[..., 3] = np.clip((img - 0.10) / 0.4, 0.0, 1.0)  # alpha ∝ density (voids clear)
    mpimg.imsave(out_path, rgba)
    return out_path
