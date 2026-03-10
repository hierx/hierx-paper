#!/usr/bin/env python3
"""Combined accessibility-map figure for the paper.

Three panels in a GridSpec(2, 2) layout:
  (a) GB drive — population accessibility  (left, spanning both rows)
  (b) London walk — population accessibility (right top)
  (c) London walk — employment accessibility (right bottom)

Edges are coloured by the mean endpoint accessibility using inferno + LogNorm.

Usage:
    python -m benchmarks_final.plot_accessibility_maps
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import LogLocator, FuncFormatter  # noqa: E402

from benchmarks_final.common import DATA_DIR, FIGURES_DIR, STYLE  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_npz(name: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Accessibility data not found: {path}\n"
            "  Generate it first with:\n"
            "    ./reproduce.sh accessibility\n"
            "  Or fetch pre-computed results:  ./reproduce.sh data"
        )
    data = np.load(path)
    return dict(data)


def _edge_segments(node_xy: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return (N_edges, 2, 2) array of line segments."""
    return np.stack([node_xy[edges[:, 0]], node_xy[edges[:, 1]]], axis=1)


def _draw_panel(
    ax: plt.Axes,
    segments: np.ndarray,
    edge_vals: np.ndarray,
    *,
    lat_centre: float,
    linewidth: float = 0.2,
    cbar_label: str = "Accessibility",
    title: str = "",
) -> None:
    """Draw sorted edge collection with colorbar."""
    # Percentile-based LogNorm (2nd–98th of positive values)
    pos = edge_vals[edge_vals > 0]
    vmin = float(np.percentile(pos, 2))
    vmax = float(np.percentile(pos, 98))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    order = np.argsort(edge_vals)
    lc = LineCollection(
        segments[order],
        array=edge_vals[order],
        cmap="inferno",
        norm=norm,
        linewidths=linewidth,
        rasterized=True,
    )
    ax.add_collection(lc)

    # Limits from data
    all_x = segments[:, :, 0].ravel()
    all_y = segments[:, :, 1].ravel()
    pad_x = (all_x.max() - all_x.min()) * 0.01
    pad_y = (all_y.max() - all_y.min()) * 0.01
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    ax.set_aspect(1.0 / math.cos(math.radians(lat_centre)))
    ax.set_title(title, fontsize=10, pad=4)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)

    cb = plt.colorbar(lc, ax=ax, shrink=0.8, pad=0.02, aspect=30)
    cb.set_label(cbar_label, fontsize=8)
    # Place labelled ticks at 1, 2, 5 × each power of 10 so narrow ranges still get ticks
    cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 5], numticks=12))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{x:g}" if 0.01 <= x < 100 else f"{x:.0e}"
    ))
    cb.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    plt.rcParams.update(STYLE)

    # ---- Load data ----
    gb = _load_npz("gb_drive_pop_accessibility_nodes.npz")
    ln = _load_npz("london_walk_1M_pop_accessibility_nodes.npz")

    # ---- Precompute edge segments and values ----
    gb_seg = _edge_segments(gb["node_xy"], gb["edges"])
    gb_vals = 0.5 * (
        gb["acc_population"][gb["edges"][:, 0]]
        + gb["acc_population"][gb["edges"][:, 1]]
    )

    ln_seg = _edge_segments(ln["node_xy"], ln["edges"])
    ln_pop_vals = 0.5 * (
        ln["acc_population"][ln["edges"][:, 0]]
        + ln["acc_population"][ln["edges"][:, 1]]
    )
    ln_wp_vals = 0.5 * (
        ln["acc_workplace"][ln["edges"][:, 0]]
        + ln["acc_workplace"][ln["edges"][:, 1]]
    )

    # ---- Figure layout ----
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.12,
                  width_ratios=[1, 1.15])

    ax_gb = fig.add_subplot(gs[:, 0])       # left column, spans both rows
    ax_ln_pop = fig.add_subplot(gs[0, 1])   # right top
    ax_ln_wp = fig.add_subplot(gs[1, 1])    # right bottom

    # ---- (a) GB drive — population ----
    _draw_panel(
        ax_gb,
        gb_seg,
        gb_vals,
        lat_centre=55.0,
        linewidth=0.12,
        title="(a) GB drive — population accessibility",
    )

    # ---- (b) London walk — population ----
    _draw_panel(
        ax_ln_pop,
        ln_seg,
        ln_pop_vals,
        lat_centre=51.5,
        linewidth=0.20,
        title="(b) London walk — population accessibility",
    )

    # ---- (c) London walk — employment ----
    _draw_panel(
        ax_ln_wp,
        ln_seg,
        ln_wp_vals,
        lat_centre=51.5,
        linewidth=0.20,
        title="(c) London walk — employment accessibility",
    )

    # ---- Save ----
    stem = "paper_fig6_accessibility_maps"
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / stem}.pdf  and  .png")


if __name__ == "__main__":
    main()
