"""
Error sensitivity study: systematic parameter sweep for hierx.

Measures how approximation error varies with:
- base_radius (5 values, as fractions of network extent)
- overlap_factor (1.0, 1.25, 1.5, 2.0, 3.0)
- increase_factor (1.5, 2.0, 3.0)
- interaction function (steep, moderate, shallow, exponential)

Networks tested: grid (10x10, 20x20, 30x30) and random spatial (100, 400, 900).

Error metrics:
- Mean relative error of matvec: mean(|h_i - d_i| / max(|d_i|, eps))
- Max relative error: max(|h_i - d_i| / max(|d_i|, eps))
- RMSE relative to dense norm: ||h - d||_2 / ||d||_2

Results written to JSON for downstream analysis and paper figures.

Usage:
    python benchmarks_final/error_sensitivity.py
    python benchmarks_final/error_sensitivity.py --output results.json
"""

import argparse
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from hierx import (
    Hierarchy,
    InteractionHierarchy,
    compute_dense_cost_matrix,
    compute_dense_interaction_matrix,
    generate_grid_network,
    generate_random_spatial_network,
)

from benchmarks_final.common import N_ACTIVITY_TRIALS, DATA_DIR, compute_errors, EPS

# ---------------------------------------------------------------------------
# Interaction functions
# ---------------------------------------------------------------------------

INTERACTION_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "steep": lambda c: (c + 500) ** (-2),
    "moderate": lambda c: (c + 2000) ** (-1.5),
    "shallow": lambda c: (c + 5000) ** (-1),
    "exponential": lambda c: np.exp(-c / 5000),
}

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

OVERLAP_FACTORS: list[float] = [1.0, 1.25, 1.5, 2.0, 3.0]
INCREASE_FACTORS: list[float] = [1.5, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------


def build_networks() -> list[dict]:
    """
    Return a list of network descriptors.

    Each descriptor is a dict with keys:
        network_type, network_size, n_zones, graph, spacing (grid only).
    """
    networks: list[dict] = []

    # Grid networks
    for n_side in [10, 20, 30]:
        spacing = 1000.0
        G = generate_grid_network(n_side, n_side, spacing=spacing)
        networks.append(
            {
                "network_type": "grid",
                "network_size": f"{n_side}x{n_side}",
                "n_zones": n_side * n_side,
                "graph": G,
                "spacing": spacing,
            }
        )

    # Random spatial networks
    for n_zones in [100, 400, 900]:
        G = generate_random_spatial_network(n_zones, area_size=100_000.0)
        networks.append(
            {
                "network_type": "random",
                "network_size": str(n_zones),
                "n_zones": n_zones,
                "graph": G,
                "spacing": None,
            }
        )

    return networks


def base_radii_for_network(net_desc: dict) -> list[float]:
    """
    Compute 5 base_radius values scaled to the network's spatial extent.

    For grid networks the extent is well-defined (spacing * (n_side - 1)),
    so we use multiples of the spacing.
    For random networks we use the coordinate bounding box diagonal and
    divide into 5 fractions.
    """
    G = net_desc["graph"]

    if net_desc["network_type"] == "grid" and net_desc["spacing"] is not None:
        spacing = net_desc["spacing"]
        return [spacing * k for k in [2, 4, 6, 8, 10]]

    # Random network: derive from bounding box
    xs = [G.nodes[n]["x"] for n in G.nodes()]
    ys = [G.nodes[n]["y"] for n in G.nodes()]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    # 5 fractions spanning ~5 % to ~25 % of extent
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25]
    return [extent * f for f in fractions]


# ---------------------------------------------------------------------------
# Single configuration benchmark
# ---------------------------------------------------------------------------


def run_single_config(
    net_desc: dict,
    dense_cost_matrix: np.ndarray,
    base_radius: float,
    overlap_factor: float,
    increase_factor: float,
    fn_name: str,
    fn_callable: Callable[[float], float],
    rng: np.random.RandomState,
) -> dict | None:
    """
    Run one parameter configuration and return a result dict, or None on failure.
    """
    G = net_desc["graph"]
    n_zones = net_desc["n_zones"]

    # Build hierarchy
    try:
        hierarchy = Hierarchy(
            G,
            base_radius=base_radius,
            increase_factor=increase_factor,
            overlap_factor=overlap_factor,
        )
        ih = InteractionHierarchy(hierarchy, fn_callable)
    except Exception as exc:
        print(f"    [SKIP] Hierarchy construction failed: {exc}")
        return None

    # Dense interaction matrix for this function
    dense_interaction = compute_dense_interaction_matrix(dense_cost_matrix, fn_callable)

    # Collect errors over multiple random activity vectors
    trial_errors: list[dict[str, float]] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n_zones) + 0.1  # avoid all-zero
        hier_result = ih.matvec(activity)
        dense_result = dense_interaction @ activity
        trial_errors.append(compute_errors(hier_result, dense_result))

    # Take median across trials for each metric
    median_mean_rel = float(np.median([e["mean_relative_error"] for e in trial_errors]))
    median_max_rel = float(np.median([e["max_relative_error"] for e in trial_errors]))
    median_rmse = float(np.median([e["relative_rmse"] for e in trial_errors]))

    return {
        "network_type": net_desc["network_type"],
        "network_size": net_desc["network_size"],
        "n_zones": n_zones,
        "base_radius": base_radius,
        "overlap_factor": overlap_factor,
        "increase_factor": increase_factor,
        "interaction_fn": fn_name,
        "mean_relative_error": median_mean_rel,
        "max_relative_error": median_max_rel,
        "relative_rmse": median_rmse,
        "n_layers": len(hierarchy.radii),
        "density": ih.get_density(),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_sensitivity_study(output_path: str) -> None:
    """
    Execute the full parameter sensitivity grid search and write results to JSON.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    print("=" * 72)
    print("Error Sensitivity Study — hierx")
    print("=" * 72)

    networks = build_networks()

    # Count total configurations for progress reporting
    total_configs = 0
    for net_desc in networks:
        n_radii = len(base_radii_for_network(net_desc))
        total_configs += (
            n_radii * len(OVERLAP_FACTORS) * len(INCREASE_FACTORS) * len(INTERACTION_FUNCTIONS)
        )
    print(f"Total configurations to evaluate: {total_configs}")
    print()

    all_results: list[dict] = []
    config_idx = 0
    t_start = time.time()

    for net_desc in networks:
        net_label = f"{net_desc['network_type']} {net_desc['network_size']}"
        print(f"--- Network: {net_label} ({net_desc['n_zones']} zones) ---")

        # Pre-compute dense cost matrix once per network
        G = net_desc["graph"]
        zones = sorted(G.nodes())
        print("  Computing dense cost matrix ...", end="", flush=True)
        t0 = time.time()
        dense_cost_matrix = compute_dense_cost_matrix(G, zones)
        print(f" {time.time() - t0:.1f}s")

        radii = base_radii_for_network(net_desc)

        for base_radius, overlap_factor, increase_factor in itertools.product(
            radii, OVERLAP_FACTORS, INCREASE_FACTORS
        ):
            for fn_name, fn_callable in INTERACTION_FUNCTIONS.items():
                config_idx += 1
                if config_idx % 20 == 0 or config_idx == 1:
                    elapsed = time.time() - t_start
                    rate = config_idx / elapsed if elapsed > 0 else 0
                    eta = (total_configs - config_idx) / rate if rate > 0 else 0
                    print(
                        f"  [{config_idx}/{total_configs}] "
                        f"br={base_radius:.0f} of={overlap_factor} "
                        f"if={increase_factor} fn={fn_name}  "
                        f"(elapsed {elapsed:.0f}s, ETA ~{eta:.0f}s)"
                    )

                result = run_single_config(
                    net_desc,
                    dense_cost_matrix,
                    base_radius,
                    overlap_factor,
                    increase_factor,
                    fn_name,
                    fn_callable,
                    rng,
                )
                if result is not None:
                    all_results.append(result)

        print()

    elapsed_total = time.time() - t_start

    # Assemble output JSON
    output_data = {
        "metadata": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_activity_trials": N_ACTIVITY_TRIALS,
            "overlap_factors": OVERLAP_FACTORS,
            "increase_factors": INCREASE_FACTORS,
            "interaction_functions": list(INTERACTION_FUNCTIONS.keys()),
            "n_configurations_attempted": total_configs,
            "n_configurations_succeeded": len(all_results),
            "total_runtime_seconds": round(elapsed_total, 1),
        },
        "results": all_results,
    }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    print("=" * 72)
    print(f"Done. {len(all_results)} results written to {output}")
    print(f"Total runtime: {elapsed_total:.1f}s")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity study for hierx approximation error."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "error_sensitivity_results.json"),
        help="Path for output JSON file (default: paper_figs_final/data/error_sensitivity_results.json)",
    )
    args = parser.parse_args()
    run_sensitivity_study(args.output)


if __name__ == "__main__":
    main()
