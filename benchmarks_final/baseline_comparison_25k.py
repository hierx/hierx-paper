"""
Baseline comparison at ~25k zones -- hierarchical vs cutoff vs Nystrom.

Key design decisions:
- Build cost measured by total_nodes_explored (cumulative node visits
  across all Dijkstra calls), which is the x-axis in Figure 7. This is
  hardware-independent: it counts algorithmic work rather than wall-clock
  time, giving a fair comparison across machines. All methods use the
  SciPy sparse Dijkstra backend.
- Dense ground truth computed via hierx.compute_dense_cost_matrix (SciPy
  all-pairs Dijkstra) and stored as a numpy array for fast matvec.
- Matvec time measured separately (no Dijkstra involved).
- Cutoff fracs limited to [0.1, 0.25, 0.5] -- at 25k, higher fracs are
  obviously slower than hierarchical (>780s Dijkstra time vs ~8s).

Usage:
    python benchmarks_final/baseline_comparison_25k.py
    python benchmarks_final/baseline_comparison_25k.py --output results_25k.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from benchmarks_final.baselines import CutoffBaseline, NystromBaseline
from benchmarks_final.common import (
    DATA_DIR,
    EPS,
    N_ACTIVITY_TRIALS,
    N_MATVEC_REPS,
    compute_errors,
)
from hierx import (
    Hierarchy,
    InteractionHierarchy,
    compute_dense_cost_matrix,
    generate_grid_network,
)

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
# Configuration
# ---------------------------------------------------------------------------

# Skip frac >= 0.75: at 25k, those take >780s Dijkstra time (obviously
# slower than hierarchical at ~8s). The Pareto frontier only needs
# the competitive region.
CUTOFF_FRACTIONS: list[float] = [0.1, 0.25, 0.5]

NYSTROM_LANDMARKS: list[int] = [50, 100, 200, 500, 1000, 2000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mean_edge_length(G: nx.Graph) -> float:
    costs = [d["cost"] for _, _, d in G.edges(data=True)]
    return float(np.mean(costs)) if costs else 1000.0


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def benchmark_hierarchical(
    G: nx.Graph,
    zones: list[int],
    interaction_fn: Callable[[float], float],
    rng: np.random.RandomState,
) -> dict:
    n = len(zones)
    mel = mean_edge_length(G)

    t0 = time.perf_counter()
    hierarchy = Hierarchy(
        G, base_radius=mel * 3, increase_factor=2, overlap_factor=2.0, backend="scipy"
    )
    ih = InteractionHierarchy(hierarchy, interaction_fn)
    build_wall = time.perf_counter() - t0

    matvec_times: list[float] = []
    activities: list[np.ndarray] = []
    results: list[np.ndarray] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1
        activities.append(activity)
        t0 = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            hier_result = ih.matvec(activity)
        matvec_times.append((time.perf_counter() - t0) / N_MATVEC_REPS)
        results.append(hier_result)

    n_stored = sum(D.nnz for D in ih.D.values())

    return {
        "method": "hierarchical",
        "method_param": "alpha=2.0,r0=3xmel",
        "build_time_wall": build_wall,
        "matvec_time": float(np.median(matvec_times)),
        "n_stored_entries": n_stored,
        "total_nodes_explored": hierarchy.total_nodes_explored,
        "_activities": activities,
        "_results": results,
    }


def benchmark_cutoff(
    G: nx.Graph,
    zones: list[int],
    interaction_fn: Callable[[float], float],
    cutoff_radius: float,
    cutoff_fraction: float,
    rng: np.random.RandomState,
) -> dict:
    n = len(zones)

    t0 = time.perf_counter()
    cutoff = CutoffBaseline(G, zones, interaction_fn, cutoff_radius=cutoff_radius)
    build_wall = time.perf_counter() - t0

    matvec_times: list[float] = []
    activities: list[np.ndarray] = []
    results: list[np.ndarray] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1
        activities.append(activity)
        t0 = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            cutoff_result = cutoff.matvec(activity)
        matvec_times.append((time.perf_counter() - t0) / N_MATVEC_REPS)
        results.append(cutoff_result)

    return {
        "method": "cutoff",
        "method_param": f"frac={cutoff_fraction:.2f}",
        "cutoff_fraction": cutoff_fraction,
        "cutoff_radius": cutoff_radius,
        "build_time_wall": build_wall,
        "matvec_time": float(np.median(matvec_times)),
        "n_stored_entries": cutoff.n_entries,
        "total_nodes_explored": cutoff.total_nodes_explored,
        "_activities": activities,
        "_results": results,
    }


def benchmark_nystrom(
    G: nx.Graph,
    zones: list[int],
    interaction_fn: Callable[[float], float],
    n_landmarks: int,
    rng: np.random.RandomState,
) -> dict:
    n = len(zones)

    t0 = time.perf_counter()
    nystrom = NystromBaseline(G, zones, interaction_fn, n_landmarks=n_landmarks)
    build_wall = time.perf_counter() - t0

    matvec_times: list[float] = []
    activities: list[np.ndarray] = []
    results: list[np.ndarray] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1
        activities.append(activity)
        t0 = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            nystrom_result = nystrom.matvec(activity)
        matvec_times.append((time.perf_counter() - t0) / N_MATVEC_REPS)
        results.append(nystrom_result)

    return {
        "method": "nystrom",
        "method_param": f"m={n_landmarks}",
        "n_landmarks": n_landmarks,
        "build_time_wall": build_wall,
        "matvec_time": float(np.median(matvec_times)),
        "n_stored_entries": n * n_landmarks,
        "total_nodes_explored": nystrom.total_nodes_explored,
        "_activities": activities,
        "_results": results,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_comparison(output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 158 x 158 grid = 24,964 zones ~ 25k
    n_side = 158
    spacing = 1000.0
    n_zones = n_side * n_side

    print("=" * 72)
    print(f"Baseline Comparison at ~25k zones ({n_side}x{n_side} = {n_zones})")
    print("=" * 72)

    print("Generating grid network ...", end="", flush=True)
    t0 = time.time()
    G = generate_grid_network(n_side, n_side, spacing=spacing)
    zones = sorted(G.nodes())
    print(f" {time.time() - t0:.1f}s  ({len(zones)} zones, {G.number_of_edges()} edges)")

    mel = mean_edge_length(G)
    print(f"Mean edge length: {mel:.0f}")

    # --- Compute dense cost matrix ---
    print("\nComputing dense cost matrix ...")
    t0 = time.time()
    cost_matrix = compute_dense_cost_matrix(G, zones)
    print(f"  Done in {time.time() - t0:.0f}s")

    # Compute diameter from cost matrix
    finite_costs = cost_matrix[np.isfinite(cost_matrix)]
    diameter = float(np.max(finite_costs)) if len(finite_costs) > 0 else 1.0
    print(f"  Network diameter: {diameter:.0f}")
    print(f"  Cost matrix shape: {cost_matrix.shape}, memory: {cost_matrix.nbytes / 1e9:.1f} GB")

    all_results: list[dict] = []
    t_total = time.time()

    for fn_name, fn_callable in INTERACTION_FUNCTIONS.items():
        print(f"\n{'=' * 72}")
        print(f"Interaction: {fn_name}")
        print(f"{'=' * 72}")

        base_result = {
            "network_type": "grid",
            "network_size": f"{n_side}x{n_side}",
            "n_zones": n_zones,
            "interaction_fn": fn_name,
            "diameter": diameter,
        }

        # Precompute dense interaction matrix and ground truth matvecs
        print("  Building dense interaction matrix ...", end="", flush=True)
        t0 = time.time()
        vectorized_fn = np.vectorize(fn_callable)
        dense_interaction = vectorized_fn(cost_matrix)
        print(f" {time.time() - t0:.1f}s")

        # Generate shared activity vectors
        trial_rng = np.random.RandomState(42)
        activities = [trial_rng.rand(n_zones) + 0.1 for _ in range(N_ACTIVITY_TRIALS)]

        print("  Computing dense ground truth matvecs ...", end="", flush=True)
        t0 = time.time()
        dense_results = [dense_interaction @ a for a in activities]
        print(f" {time.time() - t0:.1f}s")

        # --- Hierarchical (rebuilt per fn to keep node-exploration tracking clean) ---
        print("\n  [Hierarchical] building ", end="", flush=True)
        fn_rng = np.random.RandomState(42)
        hier = benchmark_hierarchical(G, zones, fn_callable, fn_rng)
        print(
            f"\n    nodes_explored={hier['total_nodes_explored']}  "
            f"wall={hier['build_time_wall']:.1f}s  "
            f"matvec={hier['matvec_time']:.4f}s"
        )

        # --- Cutoff baselines ---
        cutoff_results: list[dict] = []
        for frac in CUTOFF_FRACTIONS:
            cutoff_radius = frac * diameter
            print(f"  [Cutoff frac={frac:.2f}] building ", end="", flush=True)
            fn_rng_c = np.random.RandomState(42)
            cr = benchmark_cutoff(G, zones, fn_callable, cutoff_radius, frac, fn_rng_c)
            cutoff_results.append(cr)
            print(
                f"\n    nodes_explored={cr['total_nodes_explored']}  "
                f"wall={cr['build_time_wall']:.1f}s  "
                f"matvec={cr['matvec_time']:.4f}s"
            )

        # --- Nystrom baselines ---
        nystrom_results: list[dict] = []
        for n_lm in NYSTROM_LANDMARKS:
            if n_lm > n_zones:
                continue
            print(f"  [Nystrom m={n_lm}] building ", end="", flush=True)
            fn_rng_n = np.random.RandomState(42)
            nr = benchmark_nystrom(G, zones, fn_callable, n_lm, fn_rng_n)
            nystrom_results.append(nr)
            print(
                f"\n    nodes_explored={nr['total_nodes_explored']}  "
                f"wall={nr['build_time_wall']:.1f}s  "
                f"matvec={nr['matvec_time']:.4f}s"
            )

        # --- Compute errors ---
        print("\n  Computing errors ...")
        all_methods = [hier] + cutoff_results + nystrom_results
        for rec in all_methods:
            errs = []
            for trial_idx in range(N_ACTIVITY_TRIALS):
                errs.append(compute_errors(rec["_results"][trial_idx], dense_results[trial_idx]))
            rec["mean_relative_error"] = float(np.median([e["mean_relative_error"] for e in errs]))
            rec["max_relative_error"] = float(np.median([e["max_relative_error"] for e in errs]))
            rec["relative_rmse"] = float(np.median([e["relative_rmse"] for e in errs]))

        # Finalize
        for rec in all_methods:
            rec.pop("_activities", None)
            rec.pop("_results", None)
            rec.update(base_result)
            all_results.append(rec)

        # Print summary table
        print(
            f"\n  {'Method':<12s} {'Param':<20s} {'Error':>8s} "
            f"{'Nodes expl.':>12s} {'Matvec':>8s}"
        )
        print(f"  {'-' * 12} {'-' * 20} {'-' * 8} {'-' * 12} {'-' * 8}")
        for r in all_methods:
            print(
                f"  {r['method']:<12s} {r['method_param']:<20s} "
                f"{r['mean_relative_error']:8.4f} "
                f"{r['total_nodes_explored']:12d} "
                f"{r['matvec_time']:7.4f}s"
            )

    elapsed_total = time.time() - t_total

    output_data = {
        "metadata": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid_size": f"{n_side}x{n_side}",
            "n_zones": n_zones,
            "spacing": spacing,
            "diameter": diameter,
            "mean_edge_length": mel,
            "n_matvec_reps": N_MATVEC_REPS,
            "n_activity_trials": N_ACTIVITY_TRIALS,
            "cutoff_fractions": CUTOFF_FRACTIONS,
            "nystrom_landmarks": NYSTROM_LANDMARKS,
            "interaction_functions": list(INTERACTION_FUNCTIONS.keys()),
            "n_results": len(all_results),
            "total_runtime_seconds": round(elapsed_total, 1),
            "note": "total_nodes_explored = cumulative node visits across all "
            "Dijkstra calls (hardware-independent measure of build cost). "
            "All methods use the SciPy sparse Dijkstra backend.",
        },
        "results": all_results,
    }

    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"Done. {len(all_results)} results written to {output}")
    print(f"Total runtime: {elapsed_total:.0f}s")
    print(f"{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline comparison at ~25k zones.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "baseline_comparison_25k_results.json"),
        help="Path for output JSON file",
    )
    args = parser.parse_args()
    run_comparison(args.output)


if __name__ == "__main__":
    main()
