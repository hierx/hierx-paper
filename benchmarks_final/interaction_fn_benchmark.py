"""
Interaction function benchmark: Measure accuracy across function families.

Tests 34 interaction functions (power law, exponential, Gaussian) on a
20x20 grid network with multiple base_radius and overlap_factor values.
Produces a JSON file matching the schema of interaction_fn_results.json
used by plot_paper_figures.py (Figure 5).

Usage:
    python -m benchmarks_final.interaction_fn_benchmark
    python -m benchmarks_final.interaction_fn_benchmark --output results/fn.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hierx import (
    Hierarchy,
    InteractionHierarchy,
    compute_dense_cost_matrix,
    compute_dense_interaction_matrix,
    generate_grid_network,
)

from benchmarks_final.common import DATA_DIR, N_WORKERS, compute_errors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_ROWS: int = 20
GRID_COLS: int = 20
BASE_RADII: list[float] = [2000.0, 4000.0, 6000.0, 8000.0]
OVERLAP_FACTORS: list[float] = [1.0, 1.5, 2.0, 3.0]
INCREASE_FACTOR: float = 2.0

# Power law: (c + offset)^{-beta}
POWER_OFFSETS: list[int] = [200, 500, 1000, 2000, 5000]
POWER_BETAS: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0]

# Exponential: exp(-c / scale)
EXP_SCALES: list[int] = [1000, 2000, 5000, 10000, 20000]

# Gaussian: exp(-(c / scale)^2)
GAUSS_SCALES: list[int] = [2000, 5000, 10000, 20000]


def _build_function_list() -> list[dict]:
    """Build the list of 34 interaction functions to test."""
    fns: list[dict] = []

    # 25 power law functions
    for offset in POWER_OFFSETS:
        for beta in POWER_BETAS:
            fns.append({
                "fn_name": f"power_off{offset}_b{beta}",
                "fn_family": "power_law",
                "fn": lambda c, off=offset, b=beta: (c + off) ** (-b),
                "offset": offset,
                "beta": beta,
            })

    # 5 exponential functions
    for scale in EXP_SCALES:
        fns.append({
            "fn_name": f"exp_s{scale}",
            "fn_family": "exponential",
            "fn": lambda c, s=scale: np.exp(-c / s),
            "scale": scale,
        })

    # 4 Gaussian functions
    for scale in GAUSS_SCALES:
        fns.append({
            "fn_name": f"gauss_s{scale}",
            "fn_family": "gaussian",
            "fn": lambda c, s=scale: np.exp(-(c / s) ** 2),
            "scale": scale,
        })

    return fns


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_fn_config(
    G,
    zones: list,
    cost_matrix: np.ndarray,
    fn_info: dict,
    base_radius: float,
    overlap_factor: float,
    n_zones: int,
) -> dict:
    """Run benchmark for one (function, base_radius, overlap_factor) config."""
    fn = fn_info["fn"]

    hierarchy = Hierarchy(
        G,
        base_radius=base_radius,
        increase_factor=INCREASE_FACTOR,
        overlap_factor=overlap_factor,
        n_workers=N_WORKERS,
    )

    ih = InteractionHierarchy(hierarchy, fn)
    activity = np.ones(n_zones, dtype=np.float64)
    result_hier = ih.matvec(activity)

    # Dense ground truth
    interaction_matrix = compute_dense_interaction_matrix(cost_matrix, fn)
    result_dense = interaction_matrix @ activity

    errors = compute_errors(result_hier, result_dense)
    corr = float(np.corrcoef(result_hier, result_dense)[0, 1])

    rec = {
        "experiment": "interaction_functions",
        "fn_name": fn_info["fn_name"],
        "fn_family": fn_info["fn_family"],
        "n_zones": n_zones,
        "base_radius": base_radius,
        "overlap_factor": overlap_factor,
        "n_layers": len(hierarchy.radii),
        "mean_relative_error": errors["mean_relative_error"],
        "max_relative_error": errors["max_relative_error"],
        "relative_rmse": errors["relative_rmse"],
        "correlation": corr,
    }

    # Add family-specific parameters
    if fn_info["fn_family"] == "power_law":
        rec["offset"] = fn_info["offset"]
        rec["beta"] = fn_info["beta"]
    else:
        rec["scale"] = fn_info["scale"]

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    default_output = str(DATA_DIR / "interaction_fn_results.json")

    parser = argparse.ArgumentParser(
        description="Interaction function benchmark (Fig 5)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Path for JSON output (default: {default_output})",
    )
    args = parser.parse_args()

    fn_list = _build_function_list()
    n_fns = len(fn_list)
    total = n_fns * len(BASE_RADII) * len(OVERLAP_FACTORS)

    print("=" * 80)
    print("Interaction Function Benchmark")
    print("=" * 80)
    print(f"Network: {GRID_ROWS}x{GRID_COLS} grid")
    print(f"Functions: {n_fns}")
    print(f"Base radii: {BASE_RADII}")
    print(f"Overlap factors: {OVERLAP_FACTORS}")
    print(f"Total configurations: {total}")
    print()

    # Build network and dense cost matrix once
    G = generate_grid_network(GRID_ROWS, GRID_COLS)
    zones = sorted(G.nodes())
    n_zones = len(zones)
    print(f"Network: {n_zones} zones, {G.number_of_edges()} edges")
    print("Computing dense cost matrix...")
    cost_matrix = compute_dense_cost_matrix(G, zones)
    print(f"Cost matrix: {cost_matrix.shape}")
    print()

    results: list[dict] = []
    done = 0
    for fn_info in fn_list:
        for br in BASE_RADII:
            for of in OVERLAP_FACTORS:
                rec = benchmark_fn_config(
                    G, zones, cost_matrix, fn_info, br, of, n_zones
                )
                results.append(rec)
                done += 1

                print(
                    f"  [{done}/{total}] {fn_info['fn_name']}, "
                    f"br={br:.0f}, of={of}, "
                    f"rmse={rec['relative_rmse']:.4f}",
                    flush=True,
                )

    # Write JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "network": f"grid_{GRID_ROWS}x{GRID_COLS}",
            "n_zones": n_zones,
            "n_functions": n_fns,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to {output_path.resolve()} ({len(results)} entries)")


if __name__ == "__main__":
    main()
