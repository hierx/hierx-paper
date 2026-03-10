"""
Generate publication-quality London benchmark plots.

Reads JSON output from London benchmark scripts and produces:
- Figure 8: Error-vs-time Pareto comparison on London 60k road network
- Figure 9: Error-vs-time Pareto comparison on London walk 60k network

Usage:
    python -m benchmarks_final.plot_london_results
    python -m benchmarks_final.plot_london_results --input-60k path/to/london_60k_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks_final.common import DATA_DIR, FIGURES_DIR, STYLE

METHOD_COLORS = {
    "hierarchical": "C3",
    "cutoff": "C0",
    "nystrom": "C2",
}

FN_ORDER = ["very_steep", "steep", "moderate", "exponential"]
FN_ORDER_WALK = ["walk_steep"]
FN_LABELS = {
    "very_steep": r"$(c+60)^{-2}$",
    "steep": r"$(c+300)^{-2}$",
    "moderate": r"$(c+600)^{-1}$",
    "exponential": r"$e^{-c/600}$",
    "walk_steep": r"$(c+60)^{-2}$",
}


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def _get_x(r: dict) -> float:
    """Get x-axis value: prefer total_nodes_explored, fall back to time."""
    if "total_nodes_explored" in r:
        return r["total_nodes_explored"]
    return r["build_time"] + r["matvec_time"]


def _get_error(r: dict) -> float:
    """Get relative_rmse, falling back to mean_relative_error."""
    return r.get("relative_rmse", r["mean_relative_error"])


def _x_label(results: list[dict]) -> str:
    """Return appropriate x-axis label."""
    if any("total_nodes_explored" in r for r in results):
        return "Total nodes explored (Dijkstra work)"
    return "Total time (build + matvec) [s]"


# ---------------------------------------------------------------------------
# Figure 8 — London 60k road Pareto comparison
# ---------------------------------------------------------------------------


def plot_london_pareto_60k(results: list[dict], output_dir: Path) -> None:
    """Error-vs-time Pareto on the 60k London network (one panel per fn)."""
    subset = [
        r for r in results if r.get("network") == "london_60k" and "mean_relative_error" in r
    ]
    if not subset:
        print("  Skipping London 60k Pareto plot (no london_60k results)")
        return

    n_zones = subset[0]["n_zones"]
    fns_present = [fn for fn in FN_ORDER if fn in {r["interaction_fn"] for r in subset}]
    n_fns = len(fns_present)
    if n_fns == 0:
        return

    n_cols = 2
    n_rows = (n_fns + 1) // 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(9, 4.5 * n_rows), constrained_layout=True, squeeze=False
    )

    for idx, fn_name in enumerate(fns_present):
        ax = axes[idx // n_cols, idx % n_cols]
        fn_results = [r for r in subset if r["interaction_fn"] == fn_name]
        xl = _x_label(fn_results)

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
                ny_xs,
                ny_errors,
                "^-",
                color=METHOD_COLORS["nystrom"],
                label="Nystr\u00f6m",
                zorder=3,
            )
            for x, e, m in zip(ny_xs, ny_errors, ny_lms):
                ax.annotate(f"m={m}", (x, e), fontsize=6, textcoords="offset points", xytext=(4, 4))

        # Hierarchical points (envelope over overlap_factor configs)
        hier_pts = sorted(
            [r for r in fn_results if r["method"] == "hierarchical"],
            key=lambda r: _get_x(r),
        )
        if hier_pts:
            h_xs = [_get_x(r) for r in hier_pts]
            h_errors = [_get_error(r) for r in hier_pts]
            if len(hier_pts) > 1:
                ax.plot(
                    h_xs,
                    h_errors,
                    "*-",
                    color=METHOD_COLORS["hierarchical"],
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label="Hierarchical",
                    zorder=5,
                )
                for r, x, e in zip(hier_pts, h_xs, h_errors):
                    alpha = r["method_param"].split("alpha=")[1].split(",")[0]
                    ax.annotate(
                        f"\u03b1={alpha}",
                        (x, e),
                        fontsize=6,
                        textcoords="offset points",
                        xytext=(4, 4),
                    )
            else:
                ax.plot(
                    h_xs[0],
                    h_errors[0],
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
        ax.set_xlabel(xl)
        if idx % n_cols == 0:
            ax.set_ylabel("Relative RMSE")
        ax.set_title(FN_LABELS.get(fn_name, fn_name))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused axes if odd number of panels
    for idx in range(n_fns, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        f"London road network (n = {n_zones:,})",
        fontsize=12,
    )
    fig.savefig(output_dir / "si_fig_s5_london_pareto_60k.pdf")
    fig.savefig(output_dir / "si_fig_s5_london_pareto_60k.png")
    plt.close(fig)
    print("  London 60k Pareto plot saved (SI Fig S5)")


# ---------------------------------------------------------------------------
# London walking network Pareto comparison
# ---------------------------------------------------------------------------


def plot_london_walk_pareto_60k(results: list[dict], output_dir: Path) -> None:
    """Error-vs-time Pareto on the London walking network (single panel)."""
    subset = [
        r for r in results if r.get("network") == "london_walk_70k" and "mean_relative_error" in r
    ]
    if not subset:
        print("  Skipping London walk Pareto plot (no london_walk_70k results)")
        return

    n_zones = subset[0]["n_zones"]
    fns_present = [fn for fn in FN_ORDER_WALK if fn in {r["interaction_fn"] for r in subset}]
    n_fns = len(fns_present)
    if n_fns == 0:
        return

    fig, axes = plt.subplots(
        1, n_fns, figsize=(5.5 * n_fns, 4.5), constrained_layout=True, squeeze=False
    )

    for idx, fn_name in enumerate(fns_present):
        ax = axes[0, idx]
        fn_results = [r for r in subset if r["interaction_fn"] == fn_name]
        xl = _x_label(fn_results)

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
                ny_xs,
                ny_errors,
                "^-",
                color=METHOD_COLORS["nystrom"],
                label="Nystr\u00f6m",
                zorder=3,
            )
            for x, e, m in zip(ny_xs, ny_errors, ny_lms):
                ax.annotate(f"m={m}", (x, e), fontsize=6, textcoords="offset points", xytext=(4, 4))

        # Hierarchical points
        hier_pts = sorted(
            [r for r in fn_results if r["method"] == "hierarchical"],
            key=lambda r: _get_x(r),
        )
        if hier_pts:
            h_xs = [_get_x(r) for r in hier_pts]
            h_errors = [_get_error(r) for r in hier_pts]
            if len(hier_pts) > 1:
                ax.plot(
                    h_xs,
                    h_errors,
                    "*-",
                    color=METHOD_COLORS["hierarchical"],
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label="Hierarchical",
                    zorder=5,
                )
                for r, x, e in zip(hier_pts, h_xs, h_errors):
                    alpha = r["method_param"].split("alpha=")[1].split(",")[0]
                    ax.annotate(
                        f"\u03b1={alpha}",
                        (x, e),
                        fontsize=6,
                        textcoords="offset points",
                        xytext=(4, 4),
                    )
            else:
                ax.plot(
                    h_xs[0],
                    h_errors[0],
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
        ax.set_xlabel(xl)
        ax.set_ylabel("Relative RMSE")
        ax.set_title(FN_LABELS.get(fn_name, fn_name))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"London walking network (n = {n_zones:,})",
        fontsize=12,
        y=1.02,
    )
    fig.savefig(output_dir / "si_fig_s6_london_walk_pareto.pdf")
    fig.savefig(output_dir / "si_fig_s6_london_walk_pareto.png")
    plt.close(fig)
    print("  London walk Pareto plot saved (SI Fig S6)")


# ---------------------------------------------------------------------------
# Figure 4 — Combined London Pareto (3-panel: drive steep, drive very steep, walk)
# ---------------------------------------------------------------------------


def _plot_pareto_panel(
    ax: plt.Axes, fn_results: list[dict], fn_label: str, *, show_ylabel: bool = True
) -> None:
    """Draw a single Pareto panel (shared logic for the combined figure)."""
    xl = _x_label(fn_results)

    # Cutoff points
    cutoff_pts = sorted(
        [r for r in fn_results if r["method"] == "cutoff"],
        key=lambda r: r.get("cutoff_fraction", 0),
    )
    if cutoff_pts:
        ct_xs = [_get_x(r) for r in cutoff_pts]
        ct_errors = [_get_error(r) for r in cutoff_pts]
        ct_fracs = [r.get("cutoff_fraction", 0) for r in cutoff_pts]
        ax.plot(ct_xs, ct_errors, "o-", color=METHOD_COLORS["cutoff"], label="Cutoff", zorder=3)
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
            ny_xs, ny_errors, "^-", color=METHOD_COLORS["nystrom"],
            label="Nystr\u00f6m", zorder=3,
        )
        for x, e, m in zip(ny_xs, ny_errors, ny_lms):
            ax.annotate(f"m={m}", (x, e), fontsize=6, textcoords="offset points", xytext=(4, 4))

    # Hierarchical points
    hier_pts = sorted(
        [r for r in fn_results if r["method"] == "hierarchical"],
        key=lambda r: _get_x(r),
    )
    if hier_pts:
        h_xs = [_get_x(r) for r in hier_pts]
        h_errors = [_get_error(r) for r in hier_pts]
        if len(hier_pts) > 1:
            ax.plot(
                h_xs, h_errors, "*-",
                color=METHOD_COLORS["hierarchical"], markersize=12,
                markeredgecolor="black", markeredgewidth=0.5,
                label="Hierarchical", zorder=5,
            )
            for r, x, e in zip(hier_pts, h_xs, h_errors):
                alpha = r["method_param"].split("alpha=")[1].split(",")[0]
                ax.annotate(
                    f"\u03b1={alpha}", (x, e), fontsize=6,
                    textcoords="offset points", xytext=(4, 4),
                )
        else:
            ax.plot(
                h_xs[0], h_errors[0], "*",
                color=METHOD_COLORS["hierarchical"], markersize=14,
                markeredgecolor="black", markeredgewidth=0.5,
                label="Hierarchical", zorder=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    if show_ylabel:
        ax.set_ylabel("Relative RMSE")
    ax.set_title(fn_label)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3, which="both")


def plot_london_combined(
    results_60k: list[dict], results_walk: list[dict], output_dir: Path
) -> None:
    """Combined 3-panel London Pareto figure (main-text Fig 4).

    Panel (a): London drive, steep kernel (c+300)^{-2}
    Panel (b): London drive, very steep kernel (c+60)^{-2}
    Panel (c): London walk, steep kernel (c+60)^{-2}
    """
    drive_subset = [
        r for r in results_60k
        if r.get("network") == "london_60k" and "mean_relative_error" in r
    ]
    walk_subset = [
        r for r in results_walk
        if r.get("network") == "london_walk_70k" and "mean_relative_error" in r
    ]

    if not drive_subset and not walk_subset:
        print("  Skipping combined London plot (no results)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # (a) Drive — steep (c+300)^{-2}
    steep_drive = [r for r in drive_subset if r["interaction_fn"] == "steep"]
    n_drive = steep_drive[0]["n_zones"] if steep_drive else 0
    _plot_pareto_panel(
        axes[0], steep_drive,
        f"(a) Drive $(c+300)^{{-2}}$ (n={n_drive:,})",
        show_ylabel=True,
    )

    # (b) Drive — very steep (c+60)^{-2}
    very_steep_drive = [r for r in drive_subset if r["interaction_fn"] == "very_steep"]
    _plot_pareto_panel(
        axes[1], very_steep_drive,
        f"(b) Drive $(c+60)^{{-2}}$ (n={n_drive:,})",
        show_ylabel=False,
    )

    # (c) Walk — steep (c+60)^{-2}
    walk_steep = [r for r in walk_subset if r["interaction_fn"] == "walk_steep"]
    n_walk = walk_steep[0]["n_zones"] if walk_steep else 0
    _plot_pareto_panel(
        axes[2], walk_steep,
        f"(c) Walk $(c+60)^{{-2}}$ (n={n_walk:,})",
        show_ylabel=False,
    )

    fig.savefig(output_dir / "paper_fig5_london_combined.pdf")
    fig.savefig(output_dir / "paper_fig5_london_combined.png")
    plt.close(fig)
    print("  Combined London Pareto plot saved (Fig 5)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate London benchmark plots.")
    parser.add_argument(
        "--input-60k",
        default=str(DATA_DIR / "london_60k_results.json"),
        help="Path to london_60k_results.json",
    )
    parser.add_argument(
        "--input-walk-60k",
        default=str(DATA_DIR / "london_walk_60k_results.json"),
        help="Path to london_walk_60k_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_DIR),
        help="Directory for output figures",
    )
    args = parser.parse_args()

    plt.rcParams.update(STYLE)

    input_60k_path = Path(args.input_60k)
    input_walk_60k_path = Path(args.input_walk_60k)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating London benchmark plots...")

    results_60k = None
    results_walk = None

    if input_60k_path.exists():
        results_60k = load_results(input_60k_path)
        plot_london_pareto_60k(results_60k, output_dir)
    else:
        print(f"  Skipping 60k Pareto plot ({input_60k_path} not found)")

    if input_walk_60k_path.exists():
        results_walk = load_results(input_walk_60k_path)
    else:
        print(f"  Skipping walk data ({input_walk_60k_path} not found)")

    # Combined figure (main-text Fig 4)
    if results_60k and results_walk:
        plot_london_combined(results_60k, results_walk, output_dir)
    else:
        print("  Skipping combined London plot (need both 60k + walk data)")

    print("Done.")


if __name__ == "__main__":
    main()
