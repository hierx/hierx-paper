"""
Scaling benchmark: Measure O(n log n) scaling of hierarchical operator phases.

Separately times network generation, hierarchy build, interaction build, and
matvec across a wide range of network sizes. Produces a JSON file with
trial-level data suitable for paper figures, and prints a formatted table to
stdout while running.

Usage:
    python benchmarks_final/scaling_benchmark.py
    python benchmarks_final/scaling_benchmark.py --max-zones 10000
    python benchmarks_final/scaling_benchmark.py --output results/scaling.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
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

from benchmarks_final.common import N_WORKERS, N_TRIALS, N_MATVEC_REPS, DATA_DIR, compute_errors, track_memory, estimate_shm_bytes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_SIZES: list[int] = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
DENSE_THRESHOLD: int = 10_000
INTERACTION_FN = lambda c: (c + 1000) ** (-2)  # noqa: E731


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params_for_size(n_zones: int) -> dict:
    """Derive area_size and base_radius that keep the layer count stable."""
    area_size = math.sqrt(n_zones) * 5000
    base_radius = area_size / math.sqrt(n_zones) * 2  # == 10_000 always, but expressed per spec
    return {"area_size": area_size, "base_radius": base_radius}


def _print_header() -> None:
    """Print a formatted table header to stdout."""
    cols = (
        f"{'n':>8}",
        f"{'edges':>8}",
        f"{'layers':>6}",
        f"{'t_net(s)':>9}",
        f"{'t_hier(s)':>10}",
        f"{'t_inter(s)':>10}",
        f"{'t_mv(ms)':>9}",
        f"{'t_d_bld':>8}",
        f"{'t_d_mv':>8}",
        f"{'err':>8}",
        f"{'rmse':>8}",
    )
    header = " | ".join(cols)
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("Scaling Benchmark: hierarchy build / interaction build / matvec")
    print("=" * len(header))
    print(header)
    print(sep)


def _print_row(rec: dict) -> None:
    """Print a single result row (uses median of trials)."""
    t_hier = median(rec["t_hierarchy_build"])
    t_inter = median(rec["t_interaction_build"])
    t_mv = median(rec["t_matvec"]) * 1000  # convert to ms

    t_dense = rec["t_dense_build"]
    t_dense_mv = rec["t_dense_matvec"]
    err = rec["approx_error"]
    rmse = rec["approx_rmse"]

    dense_bld_str = f"{t_dense:>8.2f}" if t_dense is not None else f"{'--':>8}"
    dense_mv_str = f"{t_dense_mv * 1000:>8.2f}" if t_dense_mv is not None else f"{'--':>8}"
    err_str = f"{err:>8.4f}" if err is not None else f"{'--':>8}"
    rmse_str = f"{rmse:>8.4f}" if rmse is not None else f"{'--':>8}"

    cols = (
        f"{rec['n_zones']:>8}",
        f"{rec['n_edges']:>8}",
        f"{rec['n_layers']:>6}",
        f"{rec['t_network_gen']:>9.3f}",
        f"{t_hier:>10.3f}",
        f"{t_inter:>10.3f}",
        f"{t_mv:>9.2f}",
        dense_bld_str,
        dense_mv_str,
        err_str,
        rmse_str,
    )
    print(" | ".join(cols), flush=True)


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_size(n_zones: int, n_trials: int = N_TRIALS) -> dict:
    """Run the full benchmark for a single network size.

    Parameters
    ----------
    n_zones : int
        Number of zones in the network.
    n_trials : int
        Number of independent timing trials.

    Returns
    -------
    dict
        Result record matching the JSON schema described in the docstring.
    """
    params = _params_for_size(n_zones)
    area_size = params["area_size"]
    base_radius = params["base_radius"]

    # ------------------------------------------------------------------
    # 1. Network generation (timed once, reused across trials)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    G = generate_large_spatial_network(n_zones, area_size=area_size, seed=42)
    t_network_gen = time.perf_counter() - t0

    n_edges = G.number_of_edges()
    zones = sorted(G.nodes())
    activity = np.ones(len(zones), dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Timed trials for hierarchical phases
    # ------------------------------------------------------------------
    t_hierarchy_build: list[float] = []
    t_interaction_build: list[float] = []
    t_matvec: list[float] = []
    total_nodes_explored: int = 0
    peak_memory_bytes: int = 0

    for trial in range(n_trials):
        # --- hierarchy + interaction build with memory tracking ---
        with track_memory() as mem:
            # --- hierarchy build ---
            t0 = time.perf_counter()
            hierarchy = Hierarchy(
                G,
                base_radius=base_radius,
                increase_factor=2,
                overlap_factor=1.5,
                n_workers=N_WORKERS,
            )
            t_hierarchy_build.append(time.perf_counter() - t0)

            # --- interaction build ---
            t0 = time.perf_counter()
            ih = InteractionHierarchy(hierarchy, INTERACTION_FN)
            t_interaction_build.append(time.perf_counter() - t0)

        mem["peak_memory_bytes"] += estimate_shm_bytes(hierarchy)
        if mem["peak_memory_bytes"] > peak_memory_bytes:
            peak_memory_bytes = mem["peak_memory_bytes"]

        # --- matvec (median of N_MATVEC_REPS per trial) ---
        mv_times: list[float] = []
        for _ in range(N_MATVEC_REPS):
            t0 = time.perf_counter()
            result_hier = ih.matvec(activity)
            mv_times.append(time.perf_counter() - t0)
        t_matvec.append(median(mv_times))

    n_layers = len(hierarchy.radii)
    total_nodes_explored = hierarchy.total_nodes_explored

    # ------------------------------------------------------------------
    # 3. Dense baseline (only for small networks)
    # ------------------------------------------------------------------
    t_dense_build: float | None = None
    t_dense_matvec: float | None = None
    approx_error: float | None = None
    approx_rmse: float | None = None

    if n_zones <= DENSE_THRESHOLD:
        # Build dense matrices
        t0 = time.perf_counter()
        cost_matrix = compute_dense_cost_matrix(G, zones)
        interaction_matrix = compute_dense_interaction_matrix(cost_matrix, INTERACTION_FN)
        t_dense_build = time.perf_counter() - t0

        # Dense matvec (median of N_MATVEC_REPS)
        dense_mv_times: list[float] = []
        for _ in range(N_MATVEC_REPS):
            t0 = time.perf_counter()
            result_dense = interaction_matrix @ activity
            dense_mv_times.append(time.perf_counter() - t0)
        t_dense_matvec = median(dense_mv_times)

        # Approximation error via shared compute_errors
        errors = compute_errors(result_hier, result_dense)
        approx_error = errors["mean_relative_error"]
        approx_rmse = errors["relative_rmse"]

    # ------------------------------------------------------------------
    # 4. Assemble result record
    # ------------------------------------------------------------------
    return {
        "n_zones": n_zones,
        "area_size": area_size,
        "base_radius": base_radius,
        "n_layers": n_layers,
        "n_edges": n_edges,
        "n_workers": N_WORKERS,
        "total_nodes_explored": total_nodes_explored,
        "peak_memory_bytes": peak_memory_bytes,
        "t_network_gen": t_network_gen,
        "t_hierarchy_build": t_hierarchy_build,
        "t_interaction_build": t_interaction_build,
        "t_matvec": t_matvec,
        "t_dense_build": t_dense_build,
        "t_dense_matvec": t_dense_matvec,
        "approx_error": approx_error,
        "approx_rmse": approx_rmse,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    default_output = str(DATA_DIR / "scaling_results.json")

    parser = argparse.ArgumentParser(description="Scaling benchmark for hierx (O(n log n) claim).")
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Path for JSON output (default: {default_output})",
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=100000,
        help="Largest network size to include (default: 100000)",
    )
    args = parser.parse_args()

    sizes = [n for n in ALL_SIZES if n <= args.max_zones]

    _print_header()

    results: list[dict] = []
    for n_zones in sizes:
        rec = benchmark_size(n_zones)
        results.append(rec)
        _print_row(rec)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Summary (median times)")
    print("=" * 80)
    for rec in results:
        n = rec["n_zones"]
        t_h = median(rec["t_hierarchy_build"])
        t_i = median(rec["t_interaction_build"])
        t_m = median(rec["t_matvec"]) * 1000
        ratio_label = ""
        if len(results) > 1 and rec is not results[0]:
            prev = results[results.index(rec) - 1]
            n_prev = prev["n_zones"]
            t_h_prev = median(prev["t_hierarchy_build"])
            # Empirical scaling exponent between consecutive sizes
            if t_h_prev > 0 and n_prev > 0:
                ratio = (t_h / t_h_prev) / (n * math.log2(n) / (n_prev * math.log2(n_prev)))
                ratio_label = f"  (scaling ratio: {ratio:.2f})"
        print(f"  n={n:>7}: hier={t_h:.3f}s  inter={t_i:.3f}s  matvec={t_m:.2f}ms{ratio_label}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "n_trials": N_TRIALS,
            "n_matvec_reps": N_MATVEC_REPS,
            "n_workers": N_WORKERS,
            "sizes": sizes,
            "interaction_fn": "(c + 1000)**(-2)",
            "increase_factor": 2,
            "overlap_factor": 1.5,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
