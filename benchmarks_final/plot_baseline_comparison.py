"""
Generate publication-quality baseline comparison plots.

Reads JSON output from ``baseline_comparison_25k.py`` and produces:
- Figure 7: Error-vs-time Pareto frontier (2x2 by interaction function)

Usage:
    python -m benchmarks_final.plot_baseline_comparison
    python -m benchmarks_final.plot_baseline_comparison --input results.json --output-dir figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks_final.common import DATA_DIR, FIGURES_DIR, STYLE

# Consistent colours
METHOD_COLORS = {
    "hierarchical": "C3",
    "cutoff": "C0",
    "nystrom": "C2",
}

FN_ORDER = ["steep", "moderate", "shallow", "exponential"]
FN_LABELS = {
    "steep": r"$(c+500)^{-2}$",
    "moderate": r"$(c+2000)^{-1.5}$",
    "shallow": r"$(c+5000)^{-1}$",
    "exponential": r"$e^{-c/5000}$",
}


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def _get_compute_time(r: dict) -> float:
    """Get comparable computation time from a result dict.

    For 25k results: uses dijkstra_time + matvec_time (parallelization-independent).
    For legacy results: uses build_time + matvec_time.
    """
    if "dijkstra_time" in r:
        return r["dijkstra_time"] + r["matvec_time"]
    return r["build_time"] + r["matvec_time"]


def _get_nodes_explored(r: dict) -> float | None:
    """Get total_nodes_explored, returning None if missing."""
    return r.get("total_nodes_explored")


def _get_error(r: dict) -> float:
    """Get relative_rmse, falling back to mean_relative_error."""
    return r.get("relative_rmse", r["mean_relative_error"])


def _x_axis_label(results: list[dict]) -> str:
    """Return appropriate x-axis label based on available data."""
    if any("total_nodes_explored" in r for r in results):
        return "Total nodes explored (Dijkstra work)"
    if any("dijkstra_time" in r for r in results):
        return "Cumulative Dijkstra time + matvec [s]"
    return "Total time (build + matvec) [s]"


def _get_x(r: dict) -> float:
    """Get x-axis value: prefer total_nodes_explored, fall back to time."""
    if "total_nodes_explored" in r:
        return r["total_nodes_explored"]
    return _get_compute_time(r)


# ---------------------------------------------------------------------------
# Figure 7 — Error-vs-time Pareto frontier
# ---------------------------------------------------------------------------


def plot_pareto(results: list[dict], output_dir: Path) -> None:
    """Error-vs-time Pareto frontier, one panel per interaction function."""
    # Use the largest network with all methods present
    max_nz = max(r["n_zones"] for r in results)
    subset = [r for r in results if r["n_zones"] == max_nz]
    if not subset:
        print("  Skipping Pareto plot (no results)")
        return

    fns_present = [fn for fn in FN_ORDER if fn in {r["interaction_fn"] for r in subset}]
    n_fns = len(fns_present)
    if n_fns == 0:
        return

    x_label = _x_axis_label(subset)

    ncols = 2
    nrows = (n_fns + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 4 * nrows), constrained_layout=True, squeeze=False
    )

    for idx, fn_name in enumerate(fns_present):
        ax = axes[idx // ncols, idx % ncols]
        fn_results = [r for r in subset if r["interaction_fn"] == fn_name]

        # Cutoff points
        cutoff_pts = sorted(
            [r for r in fn_results if r["method"] == "cutoff"],
            key=lambda r: r.get("cutoff_fraction", 0),
        )
        if cutoff_pts:
            ct_xs = [_get_x(r) for r in cutoff_pts]
            ct_errors = [_get_error(r) for r in cutoff_pts]
            ct_fracs = [r.get("cutoff_fraction", 0) for r in cutoff_pts]
            ax.plot(
                ct_xs, ct_errors, "o-", color=METHOD_COLORS["cutoff"], label="Cutoff", zorder=3
            )
            for x, e, f in zip(ct_xs, ct_errors, ct_fracs):
                ax.annotate(
                    f"{f:.0%}", (x, e), fontsize=6, textcoords="offset points", xytext=(4, 4)
                )

        # Nystrom points
        nystrom_pts = sorted(
            [r for r in fn_results if r["method"] == "nystrom"],
            key=lambda r: r.get("n_landmarks", 0),
        )
        if nystrom_pts:
            ny_xs = [_get_x(r) for r in nystrom_pts]
            ny_errors = [_get_error(r) for r in nystrom_pts]
            ny_lms = [r.get("n_landmarks", 0) for r in nystrom_pts]
            ax.plot(
                ny_xs, ny_errors, "^-", color=METHOD_COLORS["nystrom"], label="Nystr\u00f6m", zorder=3
            )
            for x, e, m in zip(ny_xs, ny_errors, ny_lms):
                ax.annotate(f"m={m}", (x, e), fontsize=6, textcoords="offset points", xytext=(4, 4))

        # Hierarchical point (star)
        hier_pts = [r for r in fn_results if r["method"] == "hierarchical"]
        if hier_pts:
            h = hier_pts[0]
            h_x = _get_x(h)
            h_err = _get_error(h)
            ax.plot(
                h_x,
                h_err,
                "*",
                color=METHOD_COLORS["hierarchical"],
                markersize=14,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label="Hierarchical",
                zorder=5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Relative RMSE")
        ax.set_title(FN_LABELS.get(fn_name, fn_name))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused panels
    for idx in range(n_fns, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Error vs. Computation Time (n = {max_nz:,})", fontsize=12, y=1.01)
    fig.savefig(output_dir / "paper_fig4_pareto.pdf")
    fig.savefig(output_dir / "paper_fig4_pareto.png")
    plt.close(fig)
    print("  Pareto plot saved (Fig 4)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline comparison plots.")
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "baseline_comparison_25k_results.json"),
        help="Path to baseline comparison results JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_DIR),
        help="Directory for output figures",
    )
    args = parser.parse_args()

    plt.rcParams.update(STYLE)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating baseline comparison plots...")

    if input_path.exists():
        results = load_results(input_path)
        plot_pareto(results, output_dir)
    else:
        print(f"  Skipping comparison plots ({input_path} not found)")

    print("Done.")


if __name__ == "__main__":
    main()
