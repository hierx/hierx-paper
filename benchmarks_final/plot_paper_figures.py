"""
Generate paper-ready figures and tables.

Composite plots and LaTeX tables suitable for direct inclusion
in the accompanying paper. Each output is generated from the
JSON data files in paper_figs_final/data/ for reproducibility.

Usage:
    python -m benchmarks_final.plot_paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from benchmarks_final.common import DATA_DIR, FIGURES_DIR, TABLES_DIR, STYLE


def _latex_int(n: int) -> str:
    """Format integer with LaTeX thousand separators: 1402413 -> 1{,}402{,}413."""
    s = f"{n:,}"
    return s.replace(",", "{,}")


def _nlogn(n, a):
    return a * n * np.log(n)


def _nsquared(n, a, b):
    return a * n**2 + b


def _median(lst):
    return float(np.median(lst))


def load(path):
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  {name}")


# ===================================================================
# Figure 1: Scaling overview (2-panel: build time + matvec)
# ===================================================================


def figure_scaling_overview():
    """Two-panel figure showing O(n log n) scaling."""
    data = load(DATA_DIR / "scaling_results.json")
    results = data["results"]

    sizes = np.array([r["n_zones"] for r in results])
    hier_build = np.array([_median(r["t_hierarchy_build"]) for r in results])
    inter_build = np.array([_median(r["t_interaction_build"]) for r in results])
    total_build = hier_build + inter_build
    matvec = np.array([_median(r["t_matvec"]) for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Build time
    ax1.loglog(sizes, hier_build, "o-", label="Hierarchy build", zorder=3)
    ax1.loglog(sizes, inter_build, "s-", label="Interaction build", zorder=3)
    ax1.loglog(sizes, total_build, "^-", label="Total build", zorder=3, color="C2")

    # Fit O(n log n) to total
    popt, _ = curve_fit(_nlogn, sizes, total_build, p0=[1e-5], maxfev=10000)
    ss_res = np.sum((total_build - _nlogn(sizes, *popt)) ** 2)
    ss_tot = np.sum((total_build - np.mean(total_build)) ** 2)
    r2 = 1 - ss_res / ss_tot
    n_fit = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 200)
    ax1.loglog(
        n_fit,
        _nlogn(n_fit, *popt),
        "--",
        color="gray",
        label=f"$O(n \\log n)$ fit ($R^2$={r2:.3f})",
    )

    # Dense baseline
    dense_results = [r for r in results if r.get("t_dense_build") is not None]
    if dense_results:
        dn = [r["n_zones"] for r in dense_results]
        dt = [r["t_dense_build"] for r in dense_results]
        ax1.loglog(dn, dt, "D--", color="C3", label="Dense baseline", alpha=0.7)

    ax1.set_xlabel("Number of zones $n$")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("(a) Build time")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    # Panel B: Matvec time
    ax2.loglog(sizes, matvec * 1000, "o-", color="C4", label="Hierarchical matvec", zorder=3)

    popt_mv, _ = curve_fit(_nlogn, sizes, matvec, p0=[1e-9], maxfev=10000)
    ss_res = np.sum((matvec - _nlogn(sizes, *popt_mv)) ** 2)
    ss_tot = np.sum((matvec - np.mean(matvec)) ** 2)
    r2_mv = 1 - ss_res / ss_tot
    ax2.loglog(
        n_fit,
        _nlogn(n_fit, *popt_mv) * 1000,
        "--",
        color="gray",
        label=f"$O(n \\log n)$ fit ($R^2$={r2_mv:.3f})",
    )

    # Dense matvec
    if dense_results:
        dn = [r["n_zones"] for r in dense_results]
        dm = [r["t_dense_matvec"] * 1000 for r in dense_results]
        ax2.loglog(dn, dm, "D--", color="C3", label="Dense matvec", alpha=0.7)

    ax2.set_xlabel("Number of zones $n$")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("(b) Matrix-vector product")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    save(fig, "paper_fig2_scaling")


# ===================================================================
# Figure 2: Error vs overlap factor (multi-function)
# ===================================================================


def figure_error_vs_overlap():
    """Error as function of overlap factor for all interaction functions."""
    data = load(DATA_DIR / "error_sensitivity_results.json")
    results = data["results"]

    # 20x20 grid, increase_factor=2, base_radius=4000
    fig, ax = plt.subplots(figsize=(5.5, 4))

    fn_styles = {
        "steep": ("o-", "C0", "$(c+500)^{-2}$"),
        "moderate": ("s-", "C1", "$(c+2000)^{-1.5}$"),
        "shallow": ("^-", "C2", "$(c+5000)^{-1}$"),
        "exponential": ("D-", "C3", "$e^{-c/5000}$"),
    }

    overlaps = [1.0, 1.25, 1.5, 2.0, 3.0]

    for fn_name, (marker, color, label) in fn_styles.items():
        errs = []
        for of in overlaps:
            sub = [
                r
                for r in results
                if r["interaction_fn"] == fn_name
                and r["network_type"] == "grid"
                and r["network_size"] == "20x20"
                and r["increase_factor"] == 2.0
                and r["overlap_factor"] == of
                and r["base_radius"] == 4000
            ]
            if sub:
                errs.append(sub[0]["relative_rmse"])
            else:
                errs.append(np.nan)
        ax.plot(overlaps, errs, marker, color=color, label=label)

    ax.set_xlabel("Overlap factor $\\alpha$")
    ax.set_ylabel("Relative RMSE $\\varepsilon$")
    ax.set_title("Approximation error vs. overlap factor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-3)

    save(fig, "paper_fig3_error_vs_overlap")


# ===================================================================
# Figure 3: Error heatmap (overlap x base_radius) for moderate fn
# ===================================================================


def figure_error_heatmap():
    """Heatmap of error as function of base_radius and overlap_factor."""
    data = load(DATA_DIR / "error_sensitivity_results.json")
    results = data["results"]

    subset = [
        r
        for r in results
        if r["interaction_fn"] == "moderate"
        and r["network_type"] == "grid"
        and r["network_size"] == "20x20"
        and r["increase_factor"] == 2.0
    ]

    radii = sorted(set(r["base_radius"] for r in subset))
    overlaps = sorted(set(r["overlap_factor"] for r in subset))

    grid = np.full((len(overlaps), len(radii)), np.nan)
    for r in subset:
        i = overlaps.index(r["overlap_factor"])
        j = radii.index(r["base_radius"])
        grid[i, j] = r["relative_rmse"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(
        range(len(radii) + 1),
        range(len(overlaps) + 1),
        grid,
        cmap="viridis",
        shading="flat",
        vmin=0,
        vmax=min(0.3, np.nanmax(grid)),
    )
    ax.set_xticks(np.arange(len(radii)) + 0.5)
    ax.set_xticklabels([f"{r / 1000:.0f}k" for r in radii])
    ax.set_yticks(np.arange(len(overlaps)) + 0.5)
    ax.set_yticklabels([f"{o:.2f}" for o in overlaps])
    ax.set_xlabel("Base radius $r_0$ (m)")
    ax.set_ylabel("Overlap factor $\\alpha$")
    ax.set_title("Relative RMSE — $(c+2000)^{-1.5}$ on $20\\times 20$ grid")
    plt.colorbar(im, ax=ax, label="$\\varepsilon$")

    # Annotations
    for i in range(len(overlaps)):
        for j in range(len(radii)):
            val = grid[i, j]
            if not np.isnan(val):
                vmax_eff = min(0.3, np.nanmax(grid))
                color = "white" if val < 0.4 * vmax_eff else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

    save(fig, "si_fig_s1_error_heatmap")


# ===================================================================
# Figure 4: Accuracy with properly tuned parameters
# ===================================================================


def figure_tuned_scaling():
    """Show that error remains bounded when parameters are tuned to network size."""
    data = load(DATA_DIR / "scaling_tuned_results.json")
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Error vs n for different overlap factors (br=3x)
    markers = ["o", "s", "D"]
    for i, of in enumerate([1.5, 2.0, 3.0]):
        subset = [
            r
            for r in results
            if r["overlap_factor"] == of and r["br_multiplier"] == 3 and "relative_rmse" in r
        ]
        if not subset:
            continue
        ns = sorted(set(r["n_zones"] for r in subset))
        errs = [
            next((r["relative_rmse"] for r in subset if r["n_zones"] == n), np.nan)
            for n in ns
        ]
        ax1.plot(ns, errs, f"{markers[i]}-", label=f"$\\alpha={of}$")

    ax1.set_xlabel("Number of zones $n$")
    ax1.set_ylabel("Relative RMSE $\\varepsilon$")
    ax1.set_title("(a) Error with tuned $r_0 = 3\\times$ edge length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axhline(0.01, color="red", ls=":", alpha=0.5, label="1% threshold")

    # Panel B: Error vs br_multiplier for n=2000
    markers = ["o", "s", "D"]
    for i, of in enumerate([1.5, 2.0, 3.0]):
        subset = [
            r
            for r in results
            if r["overlap_factor"] == of and r["n_zones"] == 2000 and "relative_rmse" in r
        ]
        if not subset:
            continue
        brs = sorted(set(r["br_multiplier"] for r in subset))
        errs = [
            next((r["relative_rmse"] for r in subset if r["br_multiplier"] == b), np.nan)
            for b in brs
        ]
        ax2.plot(brs, errs, f"{markers[i]}-", label=f"$\\alpha={of}$")

    ax2.set_xlabel("Base radius multiplier ($r_0 / \\bar{d}$)")
    ax2.set_ylabel("Relative RMSE $\\varepsilon$")
    ax2.set_title("(b) Error vs. base radius (n=2000)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.tight_layout()
    save(fig, "si_fig_s2_tuned_scaling")


# ===================================================================
# Figure 6: Layer structure and compression
# ===================================================================


def figure_layer_structure():
    """Compression ratio and layer count."""
    data = load(DATA_DIR / "layer_structure_results.json")
    results = data["results"]

    subset = [
        r for r in results if r["network_label"] == "grid_20x20" and r["overlap_factor"] == 1.5
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Compression ratio
    markers = ["o", "s", "D"]
    for i, if_val in enumerate([1.5, 2.0, 3.0]):
        sub = [r for r in subset if r["increase_factor"] == if_val]
        if not sub:
            continue
        radii = sorted(set(r["base_radius"] for r in sub))
        ratios = [
            next((r["compression_ratio"] for r in sub if r["base_radius"] == br), np.nan)
            for br in radii
        ]
        ax1.plot([r / 1000 for r in radii], ratios, f"{markers[i]}-", label=f"$\\gamma={if_val}$")

    ax1.set_xlabel("Base radius $r_0$ (km)")
    ax1.set_ylabel("Compression ratio (dense / sparse)")
    ax1.set_title("(a) Storage compression")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Layer count
    markers = ["o", "s", "D"]
    for i, if_val in enumerate([1.5, 2.0, 3.0]):
        sub = [r for r in subset if r["increase_factor"] == if_val]
        if not sub:
            continue
        radii = sorted(set(r["base_radius"] for r in sub))
        layers = [
            next((r["n_layers"] for r in sub if r["base_radius"] == br), np.nan) for br in radii
        ]
        ax2.plot([r / 1000 for r in radii], layers, f"{markers[i]}-", label=f"$\\gamma={if_val}$")

    ax2.set_xlabel("Base radius $r_0$ (km)")
    ax2.set_ylabel("Number of layers $L$")
    ax2.set_title("(b) Layer count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    save(fig, "si_fig_s4_layer_structure")


# ===================================================================
# Figure 5: Interaction function families
# ===================================================================


def figure_interaction_families():
    """Error across interaction function families."""
    path = DATA_DIR / "interaction_fn_results.json"
    if not path.exists():
        print("  [SKIP] interaction_fn_results.json not found")
        return

    data = load(path)
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Power law — error vs beta
    power_results = [
        r
        for r in results
        if r["fn_family"] == "power_law" and r["base_radius"] == 4000 and r["overlap_factor"] == 1.5
    ]
    offsets_to_show = [200, 500, 1000, 2000, 5000]
    markers = ["o", "s", "D", "^", "v"]
    for i, offset in enumerate(offsets_to_show):
        sub = sorted(
            [r for r in power_results if r.get("offset") == offset], key=lambda x: x["beta"]
        )
        if sub:
            ax1.plot(
                [r["beta"] for r in sub],
                [r["relative_rmse"] for r in sub],
                f"{markers[i]}-",
                label=f"$\\delta={offset}$",
            )

    ax1.set_xlabel("Decay exponent $\\beta$")
    ax1.set_ylabel("Relative RMSE $\\varepsilon$")
    ax1.set_title("(a) Power law $(c+\\delta)^{-\\beta}$")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Panel B: Exp vs Gaussian
    exp_results = [
        r
        for r in results
        if r["fn_family"] == "exponential"
        and r["base_radius"] == 4000
        and r["overlap_factor"] == 1.5
    ]
    gauss_results = [
        r
        for r in results
        if r["fn_family"] == "gaussian" and r["base_radius"] == 4000 and r["overlap_factor"] == 1.5
    ]

    if exp_results:
        sub = sorted(exp_results, key=lambda x: x["scale"])
        ax2.plot(
            [r["scale"] / 1000 for r in sub],
            [r["relative_rmse"] for r in sub],
            "o-",
            label="Exponential $e^{-c/s}$",
        )
    if gauss_results:
        sub = sorted(gauss_results, key=lambda x: x["scale"])
        ax2.plot(
            [r["scale"] / 1000 for r in sub],
            [r["relative_rmse"] for r in sub],
            "s-",
            label="Gaussian $e^{-(c/s)^2}$",
        )

    ax2.set_xlabel("Scale parameter $s$ (km)")
    ax2.set_ylabel("Relative RMSE $\\varepsilon$")
    ax2.set_title("(b) Exponential vs. Gaussian decay")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    fig.tight_layout()
    save(fig, "si_fig_s3_interaction_families")


# ===================================================================
# Table 1: Scaling performance
# ===================================================================


def table_scaling():
    """Generate scaling.tex from scaling_results.json."""
    data = load(DATA_DIR / "scaling_results.json")
    results = sorted(data["results"], key=lambda r: r["n_zones"])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Scaling performance of the hierarchical operator on random",
        r"  spatial networks (mean edge cost $\bar{c} \approx 10{,}000$\,m).",
        r"  Parameters: $\rho_0 = 10{,}000\text{\,m} \approx 1\bar{c}$,",
        r"  $\gamma = 2$, $\alpha = 1.5$, $f(c) = (c+1000)^{-2}$.",
        r"  Dense baseline measured only for $n \le 10{,}000$.",
        r"  Times are median of 5 trials; matvec is median of $5 \times 10$",
        r"  repetitions. Hierarchy build uses 32 parallel Dijkstra workers",
        r"  (SciPy backend).}",
        r"\label{tab:scaling}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$n$ & edges & $K$ & $t_{\mathrm{hier}}$\,(s)",
        r"    & $t_{\mathrm{inter}}$\,(s) & $t_{\mathrm{mv}}$\,(ms)",
        r"    & $t_{\mathrm{dense}}$\,(s) & speedup \\",
        r"\midrule",
    ]

    for r in results:
        n = r["n_zones"]
        edges = r["n_edges"]
        K = r["n_layers"]
        hier = _median(r["t_hierarchy_build"])
        inter = _median(r["t_interaction_build"])
        mv_ms = _median(r["t_matvec"]) * 1000

        dense_t = r["t_dense_build"]
        if dense_t is not None:
            dense_str = f"{dense_t:.2f}"
            speedup = dense_t / (hier + inter)
            speedup_str = f"${speedup:.0f}\\times$"
        else:
            dense_str = "---"
            speedup_str = "---"

        lines.append(
            f"{_latex_int(n):>10s} & {_latex_int(edges):>12s} & {K} "
            f"& {hier:>6.2f} & {inter:>6.2f} & {mv_ms:>5.2f} "
            f"& {dense_str:>10s} & {speedup_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    out = TABLES_DIR / "scaling.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {out.name}")


# ===================================================================
# Table 2: Overlap factor vs error
# ===================================================================


def table_overlap():
    """Generate overlap.tex from error_sensitivity_results.json."""
    data = load(DATA_DIR / "error_sensitivity_results.json")
    results = data["results"]

    overlaps = sorted(set(r["overlap_factor"] for r in results))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Approximation error (relative RMSE $\varepsilon$) vs.\ overlap",
        r"  factor $\alpha$. Statistics computed across all "
        + str(len(results) // len(overlaps))
        + r" parameter",
        r"  configurations (5 base radii $\times$ 3 increase factors $\times$",
        r"  4 interaction functions $\times$ 6 networks with edge costs in",
        r"  metres) for each overlap factor.}",
        r"\label{tab:overlap}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\alpha$ & median $\varepsilon$ & mean $\varepsilon$",
        r"         & 10th pctile & 90th pctile \\",
        r"\midrule",
    ]

    for of in overlaps:
        errs = np.array([r["relative_rmse"] for r in results if r["overlap_factor"] == of])
        med = np.median(errs)
        mean = np.mean(errs)
        p10 = np.percentile(errs, 10)
        p90 = np.percentile(errs, 90)
        lines.append(f"{of:.2f} & {med:.3f} & {mean:.3f} & {p10:.3f} & {p90:.3f} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "overlap.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {out.name}")


# ===================================================================
# Table 3: Tuned scaling
# ===================================================================


def table_scaling_tuned():
    """Generate scaling_tuned.tex from scaling_tuned_results.json."""
    data = load(DATA_DIR / "scaling_tuned_results.json")
    results = data["results"]

    subset = sorted(
        [r for r in results if r["br_multiplier"] == 3 and r["overlap_factor"] == 2.0],
        key=lambda r: r["n_zones"],
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scaling with properly tuned parameters.",
        r"  $\rho_0 = 3\bar{c}$ ($\bar{c} \approx 10{,}000$\,m),",
        r"  $\alpha = 2.0$, $\gamma = 2$.",
        r"  Error is relative RMSE $\varepsilon$.}",
        r"\label{tab:scaling-tuned}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$n$ & $\varepsilon$ & $t_{\mathrm{hier}}$\,(s) & $t_{\mathrm{mv}}$\,(ms) \\",
        r"\midrule",
    ]

    for r in subset:
        n = r["n_zones"]
        rmse = r["relative_rmse"]
        hier = r["t_hierarchy_build"]
        mv_ms = r["t_matvec_median"] * 1000

        if rmse < 0.001:
            rmse_str = "$< 0.001$"
        else:
            rmse_str = f"   {rmse:.3f} "

        lines.append(
            f"{_latex_int(n):>7s} & {rmse_str} & {hier:.3f} & {mv_ms:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = TABLES_DIR / "scaling_tuned.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {out.name}")


# ===================================================================
# Main
# ===================================================================


def main():
    plt.rcParams.update(STYLE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")
    figure_scaling_overview()
    figure_error_vs_overlap()
    figure_error_heatmap()
    figure_tuned_scaling()
    figure_layer_structure()
    figure_interaction_families()

    print("Generating paper tables...")
    table_scaling()
    table_overlap()
    table_scaling_tuned()
    print("Done.")


if __name__ == "__main__":
    main()
