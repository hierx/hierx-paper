"""
Tuned scaling benchmark: Measure accuracy with properly-tuned parameters.

Sweeps network sizes with base_radius proportional to mean edge cost
and multiple overlap_factor values. Produces a JSON file matching the
schema of scaling_tuned_results.json used by plot_paper_figures.py
(Figure 4 and Table 3).

Usage:
    python -m benchmarks_final.tuned_scaling_benchmark
    python -m benchmarks_final.tuned_scaling_benchmark --max-zones 2000
    python -m benchmarks_final.tuned_scaling_benchmark --output results/tuned.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import median

import numpy as np

from hierx import (
    Hierarchy,
    InteractionHierarchy,
    compute_dense_cost_matrix,
    compute_dense_interaction_matrix,
    generate_large_spatial_network,
)

from benchmarks_final.common import DATA_DIR, N_WORKERS, compute_errors

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_SIZES: list[int] = [100, 200, 500, 1000, 2000, 5000]
DENSE_THRESHOLD: int = 5_000
BR_MULTIPLIERS: list[int] = [2, 3, 5, 8, 12]
OVERLAP_FACTORS: list[float] = [1.5, 2.0, 3.0]
INCREASE_FACTOR: float = 2.0
INTERACTION_FN = lambda c: (c + 1000) ** (-2)  # noqa: E731


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_config(
    n_zones: int,
    br_multiplier: int,
    overlap_factor: float,
) -> dict:
    """Run benchmark for a single (n_zones, br_multiplier, overlap_factor) config."""
    area_size = math.sqrt(n_zones) * 5000
    G = generate_large_spatial_network(n_zones, area_size=area_size, seed=42)

    n_edges = G.number_of_edges()
    zones = sorted(G.nodes())
    activity = np.ones(len(zones), dtype=np.float64)

    # Mean edge length
    mel = sum(d["cost"] for _, _, d in G.edges(data=True)) / G.number_of_edges()
    base_radius = br_multiplier * mel

    # Build hierarchy
    t0 = time.perf_counter()
    hierarchy = Hierarchy(
        G,
        base_radius=base_radius,
        increase_factor=INCREASE_FACTOR,
        overlap_factor=overlap_factor,
        n_workers=N_WORKERS,
    )
    t_hierarchy_build = time.perf_counter() - t0

    # Build interaction
    t0 = time.perf_counter()
    ih = InteractionHierarchy(hierarchy, INTERACTION_FN)
    t_interaction_build = time.perf_counter() - t0

    # Matvec (take median of 10 runs)
    mv_times: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        result_hier = ih.matvec(activity)
        mv_times.append(time.perf_counter() - t0)
    t_matvec_median = median(mv_times)

    n_layers = len(hierarchy.radii)

    # Dense baseline for error
    t_dense_build: float | None = None
    mean_rel = max_rel = p95_rel = p99_rel = rel_rmse = corr = None

    if n_zones <= DENSE_THRESHOLD:
        t0 = time.perf_counter()
        cost_matrix = compute_dense_cost_matrix(G, zones)
        interaction_matrix = compute_dense_interaction_matrix(cost_matrix, INTERACTION_FN)
        t_dense_build = time.perf_counter() - t0

        result_dense = interaction_matrix @ activity
        errors = compute_errors(result_hier, result_dense)

        # Compute additional percentile errors
        abs_diff = np.abs(result_hier - result_dense)
        safe_denom = np.maximum(np.abs(result_dense), 1e-10)
        rel_errors = abs_diff / safe_denom

        mean_rel = errors["mean_relative_error"]
        max_rel = errors["max_relative_error"]
        p95_rel = float(np.percentile(rel_errors, 95))
        p99_rel = float(np.percentile(rel_errors, 99))
        rel_rmse = errors["relative_rmse"]
        corr = float(np.corrcoef(result_hier, result_dense)[0, 1])

    rec = {
        "experiment": "scaling_tuned",
        "n_zones": n_zones,
        "n_edges": n_edges,
        "area_size": area_size,
        "mean_edge_length": mel,
        "base_radius": base_radius,
        "br_multiplier": br_multiplier,
        "overlap_factor": overlap_factor,
        "increase_factor": INCREASE_FACTOR,
        "n_layers": n_layers,
        "t_hierarchy_build": t_hierarchy_build,
        "t_interaction_build": t_interaction_build,
        "t_matvec_median": t_matvec_median,
        "t_dense_build": t_dense_build,
    }

    if mean_rel is not None:
        rec["mean_relative_error"] = mean_rel
        rec["max_relative_error"] = max_rel
        rec["p95_relative_error"] = p95_rel
        rec["p99_relative_error"] = p99_rel
        rec["relative_rmse"] = rel_rmse
        rec["correlation"] = corr

    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    default_output = str(DATA_DIR / "scaling_tuned_results.json")

    parser = argparse.ArgumentParser(
        description="Tuned scaling benchmark (Fig 4, Table 3)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Path for JSON output (default: {default_output})",
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=5000,
        help="Largest network size to include (default: 5000)",
    )
    args = parser.parse_args()

    sizes = [n for n in ALL_SIZES if n <= args.max_zones]

    print("=" * 80)
    print("Tuned Scaling Benchmark")
    print("=" * 80)
    print(f"Sizes: {sizes}")
    print(f"BR multipliers: {BR_MULTIPLIERS}")
    print(f"Overlap factors: {OVERLAP_FACTORS}")
    total = len(sizes) * len(BR_MULTIPLIERS) * len(OVERLAP_FACTORS)
    print(f"Total configurations: {total}")
    print()

    results: list[dict] = []
    done = 0
    for n_zones in sizes:
        for br_mult in BR_MULTIPLIERS:
            for of in OVERLAP_FACTORS:
                rec = benchmark_config(n_zones, br_mult, of)
                results.append(rec)
                done += 1

                rmse_str = f"{rec['relative_rmse']:.4f}" if "relative_rmse" in rec else "--"
                print(
                    f"  [{done}/{total}] n={n_zones:>6}, br_mult={br_mult}, "
                    f"of={of}, layers={rec['n_layers']}, "
                    f"rmse={rmse_str}",
                    flush=True,
                )

    # Write JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"results": results}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to {output_path.resolve()} ({len(results)} entries)")


if __name__ == "__main__":
    main()
