"""
Generate hierarchy construction visualization (main-text Figure 1).

3-panel figure: (a) raw transport network, (b) first non-trivial hierarchy
layer with group regions and stored cost edges, (c) coarsest layer. Reads
left-to-right as fine-to-coarse.

Chosen seed=34 with spaced node placement (min_dist=800) gives a clean
layout: 40 nodes, 108 edges → 12 reps → 4 reps.

Usage:
    python -m benchmarks_final.plot_hierarchy_construction
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import ConvexHull, Delaunay

from hierx import Hierarchy

from benchmarks_final.common import FIGURES_DIR, STYLE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 34
N_ZONES = 40
AREA_SIZE = 10000.0
MIN_DIST = 800.0
BASE_RADIUS = 1500
INCREASE_FACTOR = 2.8
OVERLAP_FACTOR = 2.0

PANEL_LABELS = ["(a)", "(b)", "(c)"]

# Group region styling
HULL_BUFFER = 300       # outward expansion of convex hulls (meters)
HULL_FILL_ALPHA = 0.12
HULL_EDGE_LW = 0.6
CIRCLE_MIN_RADIUS = 550  # fallback for 1–2 member groups
CIRCLE_FILL_ALPHA = 0.15
CIRCLE_EDGE_LW = 0.8


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

def generate_spaced_transport_network(
    n_zones: int = N_ZONES,
    area_size: float = AREA_SIZE,
    min_dist: float = MIN_DIST,
    seed: int = SEED,
) -> nx.Graph:
    """Delaunay network with minimum inter-node distance via rejection sampling."""
    rng = np.random.RandomState(seed)
    points: list[np.ndarray] = []
    for _ in range(n_zones * 200):
        if len(points) >= n_zones:
            break
        candidate = rng.uniform(0, area_size, size=2)
        if all(np.linalg.norm(candidate - p) >= min_dist for p in points):
            points.append(candidate)
    positions = np.array(points[:n_zones])

    tri = Delaunay(positions)
    G = nx.Graph()
    for i in range(len(positions)):
        G.add_node(i, x=float(positions[i, 0]), y=float(positions[i, 1]))
    for simplex in tri.simplices:
        for k in range(3):
            u, v = int(simplex[k]), int(simplex[(k + 1) % 3])
            if not G.has_edge(u, v):
                dx = positions[u, 0] - positions[v, 0]
                dy = positions[u, 1] - positions[v, 1]
                G.add_edge(u, v, cost=float(np.sqrt(dx * dx + dy * dy)))
    return G


# ---------------------------------------------------------------------------
# Color assignment
# ---------------------------------------------------------------------------

def build_layer_colors(
    hierarchy: Hierarchy,
    radii: list[float],
) -> dict[float, dict[int, np.ndarray]]:
    """Assign each rep in the finest shown layer a unique HSV color; propagate.

    At coarser layers the surviving rep keeps its color and all group
    members inherit it, so colors track which rep "won" at each level.
    """
    finest = radii[0]
    finest_reps = sorted(hierarchy.repr_zones[finest])
    hues = plt.cm.hsv(np.linspace(0, 0.92, len(finest_reps)))
    rep_base_color = {rep: hues[i] for i, rep in enumerate(finest_reps)}

    layer_colors: dict[float, dict[int, np.ndarray]] = {}
    for radius in radii:
        groups = hierarchy.groups[radius]
        layer_colors[radius] = {zone: rep_base_color[rep] for zone, rep in groups.items()}
    return layer_colors


# ---------------------------------------------------------------------------
# Group region drawing
# ---------------------------------------------------------------------------

def _buffered_hull_polygon(pts: np.ndarray, buffer: float = HULL_BUFFER) -> np.ndarray:
    """Convex hull of *pts*, expanded outward by *buffer* and spline-smoothed."""
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    n = len(verts)
    centroid = verts.mean(axis=0)

    # Push each vertex outward from centroid
    expanded = np.empty_like(verts)
    for i, v in enumerate(verts):
        d = v - centroid
        norm = np.linalg.norm(d)
        expanded[i] = v + buffer * d / norm if norm > 0 else v + buffer

    # Interpolate between vertices, then spline-smooth
    arc_pts = max(2, 32 // n)
    outline = []
    for i in range(n):
        p0, p1 = expanded[i], expanded[(i + 1) % n]
        for t in np.linspace(0, 1, arc_pts, endpoint=False):
            outline.append(p0 * (1 - t) + p1 * t)
    outline = np.array(outline)

    outline_closed = np.vstack([outline, outline[0]])
    tck, _ = splprep([outline_closed[:, 0], outline_closed[:, 1]], s=0, per=True)
    xs, ys = splev(np.linspace(0, 1, 200), tck)
    return np.column_stack([xs, ys])


def draw_group_regions(
    ax: plt.Axes,
    group_members: dict[int, list[int]],
    pos: dict[int, tuple[float, float]],
    node_colors: dict[int, np.ndarray],
) -> None:
    """Draw buffered smoothed convex hulls, or circles for 1-2 member groups."""
    for rep, members in group_members.items():
        color = node_colors[rep]
        pts = np.array([pos[m] for m in members])
        cx, cy = pts.mean(axis=0)

        if len(members) >= 3:
            try:
                smooth = _buffered_hull_polygon(pts)
                ax.fill(
                    smooth[:, 0], smooth[:, 1],
                    fc=color, alpha=HULL_FILL_ALPHA, ec=color,
                    linewidth=HULL_EDGE_LW, zorder=1,
                )
                continue
            except Exception:
                pass

        # Fallback: enclosing circle
        if len(members) == 1:
            r = CIRCLE_MIN_RADIUS
        else:
            r = max(
                CIRCLE_MIN_RADIUS,
                np.max(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)) + HULL_BUFFER,
            )
        ax.add_patch(plt.Circle(
            (cx, cy), radius=r, fc=color, alpha=CIRCLE_FILL_ALPHA,
            ec=color, linewidth=CIRCLE_EDGE_LW, zorder=1,
        ))


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _panel_label(ax: plt.Axes, idx: int) -> None:
    ax.text(
        0.0, 1.06, PANEL_LABELS[idx],
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="bottom", ha="left",
    )


def draw_network_panel(ax: plt.Axes, G: nx.Graph, pos: dict) -> None:
    """Panel (a): raw network with uniform grey nodes."""
    for u, v in G.edges():
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color="grey", alpha=0.30, linewidth=0.5, zorder=2,
        )
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    ax.scatter(xs, ys, s=15, c="0.35", edgecolors="white", linewidths=0.3, zorder=4)

    ax.set_title(f"Network\n{len(G.nodes())} zones, {len(G.edges())} edges", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    _panel_label(ax, 0)


def draw_layer_panel(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict,
    hierarchy: Hierarchy,
    radius: float,
    panel_idx: int,
    layer_idx: int,
    node_colors: dict[int, np.ndarray],
) -> None:
    """Draw one hierarchy-layer panel showing groups, stored edges, and reps."""
    groups = hierarchy.groups[radius]
    representatives = hierarchy.repr_zones[radius]
    costs = hierarchy.costs[radius]
    rep_set = set(representatives)

    # Group->members mapping
    group_members: dict[int, list[int]] = {}
    for zone, rep in groups.items():
        group_members.setdefault(rep, []).append(zone)

    # Group regions
    draw_group_regions(ax, group_members, pos, node_colors)

    # Non-representative nodes
    for zone in G.nodes():
        if zone not in rep_set:
            ax.scatter(
                pos[zone][0], pos[zone][1],
                s=15, c=[node_colors[zone]], alpha=0.6, edgecolors="none", zorder=4,
            )

    # Stored cost edges between representatives
    n_cost_edges = 0
    alpha_edge = max(0.08, 0.20 - (panel_idx - 1) * 0.04)
    width_edge = 0.5 + (panel_idx - 1) * 0.3
    drawn: set[tuple[int, int]] = set()
    for src in costs:
        for dst in costs[src]:
            if src != dst:
                pair = (min(src, dst), max(src, dst))
                if pair not in drawn:
                    drawn.add(pair)
                    n_cost_edges += 1
                    ax.plot(
                        [pos[src][0], pos[dst][0]], [pos[src][1], pos[dst][1]],
                        color="steelblue", alpha=alpha_edge, linewidth=width_edge, zorder=3,
                    )

    # Representative nodes (squares)
    for rep in representatives:
        ax.scatter(
            pos[rep][0], pos[rep][1],
            s=60, c=[node_colors[rep]], marker="s",
            edgecolors="white", linewidths=0.8, zorder=5,
        )

    ax.set_title(
        f"Layer {layer_idx} ($\\rho$ = {radius:.0f})\n"
        f"{len(representatives)} reps, {n_cost_edges} stored edges",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    _panel_label(ax, panel_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    plt.rcParams.update(STYLE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    G = generate_spaced_transport_network()
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    print(f"Network (seed={SEED}): {len(G.nodes())} nodes, {len(G.edges())} edges")

    hierarchy = Hierarchy(
        G,
        base_radius=BASE_RADIUS,
        increase_factor=INCREASE_FACTOR,
        overlap_factor=OVERLAP_FACTOR,
    )
    print(f"Hierarchy: {len(hierarchy.radii)} layers")
    for info in hierarchy.get_layer_info():
        print(f"  r={info['radius']:.0f}: {info['n_representatives']} reps, {info['n_costs']} costs")

    # Show only layers with non-trivial grouping (skip finest where all nodes are reps)
    radii_to_show = [r for r in hierarchy.radii if len(hierarchy.repr_zones[r]) < len(G.nodes())][:2]
    layer_colors = build_layer_colors(hierarchy, radii_to_show)

    fig, axes = plt.subplots(1, 1 + len(radii_to_show), figsize=(11, 4))
    draw_network_panel(axes[0], G, pos)

    for idx, radius in enumerate(radii_to_show):
        true_layer_idx = hierarchy.radii.index(radius)
        draw_layer_panel(
            axes[idx + 1], G, pos, hierarchy, radius,
            panel_idx=idx + 1, layer_idx=true_layer_idx,
            node_colors=layer_colors[radius],
        )

    fig.subplots_adjust(wspace=0.05)

    stem = "paper_fig1_hierarchy_construction"
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    fig.savefig(FIGURES_DIR / f"{stem}.png")
    plt.close(fig)
    print(f"\nSaved: {FIGURES_DIR / stem}.pdf")
    print(f"Saved: {FIGURES_DIR / stem}.png")


if __name__ == "__main__":
    main()
