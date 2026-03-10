"""
London 60k benchmark: hierarchical operator scaling comparison.

Extracts a ~60k-node subgraph from london_large.graphml (13 km radius around
Charing Cross), computes dense ground truth via parallel scipy Dijkstra, and
benchmarks hierarchical (overlap_factor sweep) against scipy-based Nystrom
and cutoff baselines.

All baselines use scipy.sparse.csgraph.dijkstra for fairness -- both the
hierarchical method and baselines run on the same C backend.

Usage:
    python -m benchmarks_final.london_benchmark_60k
    python -m benchmarks_final.london_benchmark_60k --kernels steep
    python -m benchmarks_final.london_benchmark_60k --resume
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from hierx import Hierarchy, InteractionHierarchy
from hierx.backends import convert_nx_to_csr

from benchmarks_final.baselines import NystromBaseline
from benchmarks_final.common import (
    N_WORKERS,
    N_MATVEC_REPS,
    N_ACTIVITY_TRIALS,
    EPS,
    DATA_DIR,
    compute_errors,
)

# ---------------------------------------------------------------------------
# Interaction functions (travel-time scale: seconds)
# ---------------------------------------------------------------------------

INTERACTION_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "very_steep": lambda c: (c + 60) ** (-2),
    "steep": lambda c: (c + 300) ** (-2),
    "moderate": lambda c: (c + 600) ** (-1),
    "exponential": lambda c: np.exp(-c / 600),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_DIJKSTRA_WORKERS: int = N_WORKERS

# Charing Cross coordinates
CHARING_CROSS_LAT: float = 51.5074
CHARING_CROSS_LON: float = -0.1278
EXTRACTION_RADIUS_KM: float = 13.0

# Hierarchical sweep
OVERLAP_FACTORS: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Baseline configs
NYSTROM_LANDMARKS: list[int] = [10, 25, 50, 100, 200, 500, 1000]
CUTOFF_FRACTIONS: list[float] = [0.1, 0.25, 0.5]

NETWORK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCH_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Stage 1: Network extraction
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_60k_network(cache_path: Path | None = None) -> nx.Graph:
    """Extract ~60k-node subgraph from london_large.graphml.

    Filters nodes within 13 km of Charing Cross, takes largest connected
    component, relabels to 0..n-1.
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached network from {cache_path}")
        G = nx.read_graphml(cache_path, node_type=int)
        for _, _, d in G.edges(data=True):
            d["cost"] = float(d["cost"])
        for _, d in G.nodes(data=True):
            d["x"] = float(d["x"])
            d["y"] = float(d["y"])
        print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
        return G

    source_path = NETWORK_DATA_DIR / "london_large.graphml"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Network not found: {source_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_london_networks\n"
            "  Or:  ./reproduce.sh data-fetch"
        )

    print(f"  Loading {source_path} ...")
    G_full = nx.read_graphml(source_path)

    # Ensure float attributes
    for _, d in G_full.nodes(data=True):
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])
    for _, _, d in G_full.edges(data=True):
        d["cost"] = float(d["cost"])

    # Filter by distance
    nodes_in_radius = []
    for node, d in G_full.nodes(data=True):
        dist_km = haversine_km(CHARING_CROSS_LAT, CHARING_CROSS_LON, d["y"], d["x"])
        if dist_km <= EXTRACTION_RADIUS_KM:
            nodes_in_radius.append(node)

    print(f"  Nodes within {EXTRACTION_RADIUS_KM} km: {len(nodes_in_radius):,}")
    G_sub = G_full.subgraph(nodes_in_radius).copy()
    del G_full

    # Largest connected component
    lcc_nodes = max(nx.connected_components(G_sub), key=len)
    G_lcc = G_sub.subgraph(lcc_nodes).copy()
    del G_sub

    # Relabel to 0..n-1
    G_out = nx.convert_node_labels_to_integers(G_lcc, ordering="sorted")
    del G_lcc

    print(f"  LCC: {G_out.number_of_nodes():,} nodes, {G_out.number_of_edges():,} edges")

    if cache_path:
        print(f"  Caching to {cache_path}")
        nx.write_graphml(G_out, cache_path)

    return G_out


# ---------------------------------------------------------------------------
# Stage 2: Dense cost matrix via parallel scipy Dijkstra
# ---------------------------------------------------------------------------


def graph_to_csr(G: nx.Graph) -> sp.csr_matrix:
    """Convert NetworkX graph to scipy CSR adjacency matrix (cost weights)."""
    csr, _idx_to_node, _node_to_idx = convert_nx_to_csr(G)
    return csr


def compute_dense_cost_parallel(
    csr: sp.csr_matrix,
    n_workers: int = 32,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Compute full APSP via parallel scipy Dijkstra.

    scipy.sparse.csgraph.dijkstra releases the GIL, so ThreadPoolExecutor
    gives true parallelism without process serialization overhead.

    Results are stored as float32 (sufficient for travel times 0-2000s).
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached dense cost matrix from {cache_path}")
        return np.load(cache_path)

    n = csr.shape[0]
    # Adaptive chunk size: keep concurrent memory ~16 GB
    chunk_size = min(1024, max(1, 500_000_000 // (n * 4)))
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((start, end))

    print(f"  APSP: n={n:,}, {len(chunks)} chunks of ~{chunk_size}, {n_workers} workers")
    cost_matrix = np.empty((n, n), dtype=np.float32)

    def process_chunk(chunk_range: tuple[int, int]) -> None:
        start, end = chunk_range
        indices = list(range(start, end))
        result = csg.dijkstra(csr, directed=False, indices=indices)
        cost_matrix[start:end, :] = result.astype(np.float32)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for i, future in enumerate(futures):
            future.result()
            if (i + 1) % 10 == 0 or i + 1 == len(futures):
                elapsed = time.time() - t0
                print(f"    Chunk {i + 1}/{len(futures)} done ({elapsed:.0f}s)")

    dt = time.time() - t0
    print(f"  APSP completed in {dt:.1f}s")

    # Verify symmetry
    max_asym = float(np.max(np.abs(cost_matrix - cost_matrix.T)))
    print(f"  Symmetry check: max|C - C.T| = {max_asym:.2e}")

    if cache_path:
        print(f"  Caching to {cache_path} ({cost_matrix.nbytes / 1e9:.1f} GB)")
        np.save(cache_path, cost_matrix)

    return cost_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mean_edge_cost(G: nx.Graph) -> float:
    """Compute mean edge cost in the network."""
    costs = [d["cost"] for _, _, d in G.edges(data=True)]
    return float(np.mean(costs)) if costs else 1.0


def compute_network_diameter(cost_matrix: np.ndarray) -> float:
    """Compute the network diameter (max finite shortest-path distance)."""
    finite_costs = cost_matrix[np.isfinite(cost_matrix)]
    return float(np.max(finite_costs)) if len(finite_costs) > 0 else 1.0


def compute_dense_interaction(
    cost_matrix: np.ndarray,
    fn_name: str,
) -> np.ndarray:
    """Apply interaction function element-wise to cost matrix.

    Uses direct numpy operations instead of np.vectorize for performance
    on large matrices (~3.8 billion elements at n=60k). Operates in float32
    to save memory (~15 GB vs 30 GB at n=60k).
    """
    if fn_name == "very_steep":
        return np.float32(1.0) / (cost_matrix + np.float32(60.0)) ** 2
    elif fn_name == "steep":
        return np.float32(1.0) / (cost_matrix + np.float32(300.0)) ** 2
    elif fn_name == "moderate":
        return np.float32(1.0) / (cost_matrix + np.float32(600.0))
    elif fn_name == "exponential":
        return np.exp(-cost_matrix / np.float32(600.0))
    else:
        raise ValueError(f"Unknown interaction function: {fn_name}")


# ---------------------------------------------------------------------------
# Stage 3: Benchmark runners
# ---------------------------------------------------------------------------


def benchmark_hierarchical(
    G: nx.Graph,
    n: int,
    interaction_fn: Callable[[float], float],
    dense_interaction: np.ndarray,
    dense_mass: float,
    mel: float,
    overlap_factor: float,
    rng: np.random.RandomState,
) -> dict:
    """Benchmark the hierarchical operator at a given overlap_factor."""
    t0 = time.perf_counter()
    hierarchy = Hierarchy(
        G,
        base_radius=mel * 3,
        increase_factor=2,
        overlap_factor=overlap_factor,
        n_workers=N_DIJKSTRA_WORKERS,
    )
    ih = InteractionHierarchy(hierarchy, interaction_fn)
    build_time = time.perf_counter() - t0

    errors_list: list[dict[str, float]] = []
    matvec_times: list[float] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1

        t0 = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            hier_result = ih.matvec(activity)
        matvec_times.append((time.perf_counter() - t0) / N_MATVEC_REPS)

        dense_result = dense_interaction @ activity
        errors_list.append(compute_errors(hier_result, dense_result))

    ones_result = ih.matvec(np.ones(n))
    method_mass = float(np.sum(ones_result))

    return {
        "method": "hierarchical",
        "method_param": f"alpha={overlap_factor},r0=3xmel",
        "overlap_factor": overlap_factor,
        "build_time": build_time,
        "matvec_time": float(np.median(matvec_times)),
        "mean_relative_error": float(np.median([e["mean_relative_error"] for e in errors_list])),
        "max_relative_error": float(np.median([e["max_relative_error"] for e in errors_list])),
        "relative_rmse": float(np.median([e["relative_rmse"] for e in errors_list])),
        "n_stored_entries": sum(D.nnz for D in ih.D.values()),
        "interaction_mass_fraction": method_mass / dense_mass if dense_mass > EPS else 0.0,
        "total_nodes_explored": hierarchy.total_nodes_explored,
    }


def benchmark_nystrom_scipy(
    csr: sp.csr_matrix,
    n: int,
    interaction_fn: Callable[[float], float],
    dense_interaction: np.ndarray,
    dense_mass: float,
    n_landmarks: int,
    rng: np.random.RandomState,
    G: nx.Graph | None = None,
) -> dict:
    """Benchmark Nystrom baseline using scipy Dijkstra via NystromBaseline."""
    if G is None:
        raise ValueError("G (NetworkX graph) is required for NystromBaseline")

    zones = sorted(G.nodes())
    # Use same seed as rng's current state for reproducibility
    seed = rng.randint(0, 2**31)

    t0 = time.perf_counter()
    nystrom = NystromBaseline(G, zones, interaction_fn, n_landmarks=n_landmarks, seed=seed)
    build_time = time.perf_counter() - t0

    # Matvec timing and errors
    errors_list: list[dict[str, float]] = []
    matvec_times: list[float] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1
        dense_result = dense_interaction @ activity

        t0 = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            nystrom_result = nystrom.matvec(activity)
        matvec_times.append((time.perf_counter() - t0) / N_MATVEC_REPS)

        errors_list.append(compute_errors(nystrom_result, dense_result))

    return {
        "method": "nystrom",
        "method_param": f"m={n_landmarks}",
        "n_landmarks": n_landmarks,
        "build_time": build_time,
        "matvec_time": float(np.median(matvec_times)),
        "mean_relative_error": float(np.median([e["mean_relative_error"] for e in errors_list])),
        "max_relative_error": float(np.median([e["max_relative_error"] for e in errors_list])),
        "relative_rmse": float(np.median([e["relative_rmse"] for e in errors_list])),
        "n_stored_entries": n * min(n_landmarks, n),
        "interaction_mass_fraction": nystrom.interaction_mass / dense_mass if dense_mass > EPS else 0.0,
        "total_nodes_explored": nystrom.total_nodes_explored,
    }


def _apply_interaction_to_dists(
    dists: np.ndarray,
    fn_name: str,
) -> np.ndarray:
    """Apply interaction function to a distance array using numpy ops."""
    if fn_name == "very_steep":
        return (dists + 60.0) ** (-2)
    elif fn_name == "steep":
        return (dists + 300.0) ** (-2)
    elif fn_name == "moderate":
        return (dists + 600.0) ** (-1)
    elif fn_name == "exponential":
        return np.exp(-dists / 600.0)
    else:
        raise ValueError(f"Unknown interaction function: {fn_name}")


def benchmark_cutoff_dense(
    cost_matrix: np.ndarray,
    n: int,
    fn_name: str,
    dense_interaction: np.ndarray,
    dense_mass: float,
    cutoff_radius: float,
    cutoff_fraction: float,
    rng: np.random.RandomState,
    interaction_fn: Callable[[float], float] | None = None,
) -> dict:
    """Benchmark cutoff baseline using the pre-computed dense cost matrix.

    Builds the cutoff interaction matrix by masking the dense cost matrix
    at cutoff_radius and applying the interaction function. Avoids a
    redundant Dijkstra computation that would duplicate the dense matrix
    in memory.
    """
    if interaction_fn is None:
        interaction_fn = INTERACTION_FUNCTIONS[fn_name]

    t0 = time.perf_counter()
    mask = cost_matrix <= cutoff_radius
    total_nodes_explored = int(np.sum(mask))

    # Mask the dense interaction matrix (cheaper than sparse for high fill ratios)
    cutoff_interaction = np.where(mask, dense_interaction, 0.0)
    n_entries = int(np.sum(mask))
    del mask

    interaction_mass = float(cutoff_interaction.sum())
    build_time = time.perf_counter() - t0

    # Matvec timing and errors
    errors_list: list[dict[str, float]] = []
    matvec_times: list[float] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1
        dense_result = dense_interaction @ activity

        t0p = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            cutoff_result = cutoff_interaction @ activity
        matvec_times.append((time.perf_counter() - t0p) / N_MATVEC_REPS)

        errors_list.append(compute_errors(cutoff_result, dense_result))

    del cutoff_interaction

    return {
        "method": "cutoff",
        "method_param": f"frac={cutoff_fraction:.2f}",
        "cutoff_fraction": cutoff_fraction,
        "cutoff_radius": cutoff_radius,
        "build_time": build_time,
        "matvec_time": float(np.median(matvec_times)),
        "mean_relative_error": float(np.median([e["mean_relative_error"] for e in errors_list])),
        "max_relative_error": float(np.median([e["max_relative_error"] for e in errors_list])),
        "relative_rmse": float(np.median([e["relative_rmse"] for e in errors_list])),
        "n_stored_entries": n_entries,
        "interaction_mass_fraction": interaction_mass / dense_mass if dense_mass > EPS else 0.0,
        "total_nodes_explored": total_nodes_explored,
    }


# ---------------------------------------------------------------------------
# Incremental JSON I/O
# ---------------------------------------------------------------------------


def _write_results(
    output: Path,
    all_results: list[dict],
    n: int,
    diameter: float,
    mel: float,
    elapsed: float,
) -> None:
    """Write (or overwrite) the results JSON file."""
    output_data = {
        "metadata": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": "London 60k road network benchmark (13 km radius from Charing Cross)",
            "n_zones": n,
            "diameter": diameter,
            "mean_edge_cost": mel,
            "extraction_radius_km": EXTRACTION_RADIUS_KM,
            "n_matvec_reps": N_MATVEC_REPS,
            "n_activity_trials": N_ACTIVITY_TRIALS,
            "overlap_factors": OVERLAP_FACTORS,
            "cutoff_fractions": CUTOFF_FRACTIONS,
            "nystrom_landmarks": NYSTROM_LANDMARKS,
            "interaction_functions": list(INTERACTION_FUNCTIONS.keys()),
            "n_results": len(all_results),
            "total_runtime_seconds": round(elapsed, 1),
        },
        "results": all_results,
    }
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)


def _load_existing_results(output: Path) -> list[dict]:
    """Load previously saved results for resume support."""
    if not output.exists():
        return []
    with open(output) as f:
        data = json.load(f)
    return data.get("results", [])


def _completed_kernels(results: list[dict]) -> set[str]:
    """Return set of interaction function names that have all methods completed."""
    fns = {r["interaction_fn"] for r in results}
    complete = set()
    for fn in fns:
        fn_results = [r for r in results if r["interaction_fn"] == fn]
        methods = {r["method"] for r in fn_results}
        # Consider complete if we have at least hierarchical + nystrom + cutoff
        if {"hierarchical", "nystrom", "cutoff"}.issubset(methods):
            complete.add(fn)
    return complete


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_benchmark(output_path: str | None = None, kernels: list[str] | None = None) -> None:
    """Execute the London 60k benchmark and write results to JSON.

    By default, recomputes all results. Pass --resume to skip kernels that
    already have results in the output file. Use --kernels to run only
    specific kernels.
    """
    output = Path(output_path) if output_path else DATA_DIR / "london_60k_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    all_results = _load_existing_results(output)
    done_kernels = _completed_kernels(all_results)
    if all_results:
        print(f"  Resuming: {len(all_results)} existing results, "
              f"completed kernels: {done_kernels or 'none'}")

    t_start = time.time()

    print("=" * 72)
    print("London 60k Road Network Benchmark")
    print("=" * 72)

    # --- Stage 1: Network extraction ---
    print("\n--- Stage 1: Network extraction ---")
    cache_graphml = NETWORK_DATA_DIR / "london_60k.graphml"
    G = extract_60k_network(cache_path=cache_graphml)
    n = G.number_of_nodes()
    mel = mean_edge_cost(G)
    print(f"  Mean edge cost: {mel:.1f}s")

    # --- Stage 2: Dense cost matrix ---
    print("\n--- Stage 2: Dense cost matrix (parallel scipy Dijkstra) ---")
    csr = graph_to_csr(G)
    cache_npy = BENCH_DIR / "london_60k_dense_cost.npy"
    cost_matrix = compute_dense_cost_parallel(
        csr, n_workers=N_DIJKSTRA_WORKERS, cache_path=cache_npy
    )
    diameter = compute_network_diameter(cost_matrix)
    print(f"  Network diameter: {diameter:.0f}s")

    # Determine which kernels to run
    run_kernels = kernels if kernels else list(INTERACTION_FUNCTIONS.keys())

    # --- Stage 3: Run methods ---
    for fn_name in run_kernels:
        if fn_name not in INTERACTION_FUNCTIONS:
            print(f"\n  Unknown kernel '{fn_name}', skipping")
            continue

        if fn_name in done_kernels:
            print(f"\n  Kernel '{fn_name}' already complete, skipping (--resume mode)")
            continue

        fn_callable = INTERACTION_FUNCTIONS[fn_name]
        rng = np.random.RandomState(42)  # Reset per-kernel for reproducibility

        # Remove any partial results for this kernel
        all_results = [r for r in all_results if r.get("interaction_fn") != fn_name]

        print(f"\n{'=' * 72}")
        print(f"Interaction: {fn_name}")
        print(f"{'=' * 72}")

        print("  Computing dense interaction matrix ...", end="", flush=True)
        t0 = time.time()
        dense_interaction = compute_dense_interaction(cost_matrix, fn_name)
        dense_mass = float(np.sum(dense_interaction))
        dt = time.time() - t0
        print(f" {dt:.1f}s  mass={dense_mass:.4e}")

        base_result = {
            "network": "london_60k",
            "n_zones": n,
            "interaction_fn": fn_name,
            "mean_edge_cost": mel,
            "diameter": diameter,
        }

        # --- Hierarchical sweep ---
        for alpha in OVERLAP_FACTORS:
            print(f"  Hierarchical alpha={alpha} ...", end="", flush=True)
            try:
                result = benchmark_hierarchical(
                    G, n, fn_callable, dense_interaction, dense_mass,
                    mel, alpha, rng,
                )
                result.update(base_result)
                all_results.append(result)
                print(
                    f" err={result['mean_relative_error']:.4f}"
                    f" t_build={result['build_time']:.1f}s"
                    f" t_mv={result['matvec_time']:.4f}s"
                )
            except Exception as exc:
                print(f" FAILED: {exc}")

        # --- Nystrom baselines (scipy) ---
        for n_lm in NYSTROM_LANDMARKS:
            if n_lm > n:
                continue
            print(f"  Nystrom m={n_lm} ...", end="", flush=True)
            try:
                result = benchmark_nystrom_scipy(
                    csr, n, fn_callable, dense_interaction, dense_mass,
                    n_lm, rng, G=G,
                )
                result.update(base_result)
                all_results.append(result)
                print(
                    f" err={result['mean_relative_error']:.4f}"
                    f" t_build={result['build_time']:.1f}s"
                )
            except Exception as exc:
                print(f" FAILED: {exc}")

        # --- Cutoff baselines (scipy) ---
        for frac in CUTOFF_FRACTIONS:
            cutoff_radius = frac * diameter
            print(
                f"  Cutoff frac={frac:.2f} (r={cutoff_radius:.0f}s) ...",
                end="",
                flush=True,
            )
            try:
                result = benchmark_cutoff_dense(
                    cost_matrix, n, fn_name, dense_interaction, dense_mass,
                    cutoff_radius, frac, rng, interaction_fn=fn_callable,
                )
                result.update(base_result)
                all_results.append(result)
                print(
                    f" err={result['mean_relative_error']:.4f}"
                    f" mass={result['interaction_mass_fraction']:.3f}"
                    f" t_build={result['build_time']:.1f}s"
                )
            except Exception as exc:
                print(f" FAILED: {exc}")

        # Free dense interaction to save memory between kernels
        del dense_interaction

        # --- Incremental save after each kernel ---
        elapsed = time.time() - t_start
        _write_results(output, all_results, n, diameter, mel, elapsed)
        n_kernel = len([r for r in all_results if r["interaction_fn"] == fn_name])
        print(f"\n  Saved {n_kernel} results for '{fn_name}' -> {output}")

    elapsed = time.time() - t_start

    # --- Final save ---
    _write_results(output, all_results, n, diameter, mel, elapsed)
    print(f"\n  Results: {output} ({len(all_results)} entries)")

    print(f"\n{'=' * 72}")
    print(f"Done. Total runtime: {elapsed / 60:.1f} min")
    print(f"{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="London 60k benchmark: hierarchical scaling advantage."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output JSON file (default: DATA_DIR/london_60k_results.json)",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=list(INTERACTION_FUNCTIONS.keys()),
        default=None,
        help="Run only specific kernels (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs that already have results in the output file",
    )
    args = parser.parse_args()

    if not args.resume:
        output = Path(args.output) if args.output else DATA_DIR / "london_60k_results.json"
        if output.exists():
            output.unlink()

    run_benchmark(args.output, kernels=args.kernels)


if __name__ == "__main__":
    main()
