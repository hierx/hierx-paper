"""
London walking network ~70k benchmark: hierarchical operator vs baselines.

Uses london_walk_large.graphml (70,008 nodes, 5 km radius around Charing Cross).
Edge costs are walking time in seconds at 5 km/h (pre-computed during network fetch).

Interaction kernel: (c + 60)^{-2}  -- the 60s offset (~83m walking) sets a short
characteristic interaction scale, heavily concentrating interaction mass in the
near field and making this a hard case for sparse landmark sampling (Nystrom).

Usage:
    python benchmarks_final/london_walk_benchmark_60k.py
    python benchmarks_final/london_walk_benchmark_60k.py --output path/to/results.json
    python benchmarks_final/london_walk_benchmark_60k.py --resume
"""

import argparse
import json
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

from benchmarks_final.common import N_WORKERS, DATA_DIR, compute_errors

# ---------------------------------------------------------------------------
# Interaction function (walking-time scale: seconds, 60s offset ~ 83m)
# ---------------------------------------------------------------------------

INTERACTION_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "walk_steep": lambda c: (c + 60) ** (-2),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_MATVEC_REPS: int = 10
N_ACTIVITY_TRIALS: int = 5
EPS: float = 1e-10

# Hierarchical sweep
OVERLAP_FACTORS: list[float] = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Baseline configs
NYSTROM_LANDMARKS: list[int] = [10, 25, 50, 100, 200, 500, 1000]
CUTOFF_FRACTIONS: list[float] = [0.1, 0.25, 0.5]

NETWORK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCH_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Stage 1: Network loading
# ---------------------------------------------------------------------------


def load_walk_network() -> nx.Graph:
    """Load the London walking network from london_walk_large.graphml.

    The file already contains 70,008 nodes with edge costs in walking
    seconds (length_meters / 1.389 m/s). No extraction or filtering needed.
    """
    source_path = NETWORK_DATA_DIR / "london_walk_large.graphml"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Network not found: {source_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_london_walking\n"
            "  Or:  ./reproduce.sh data-fetch"
        )

    print(f"  Loading {source_path} ...")
    G = nx.read_graphml(source_path)

    # Ensure float attributes
    for _, d in G.nodes(data=True):
        if "x" in d:
            d["x"] = float(d["x"])
        if "y" in d:
            d["y"] = float(d["y"])
    for _, _, d in G.edges(data=True):
        d["cost"] = float(d["cost"])

    # Relabel to 0..n-1 integers for consistent indexing
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    # Ensure connected -- take LCC if needed
    if not nx.is_connected(G):
        lcc_nodes = max(nx.connected_components(G), key=len)
        print(f"  Taking LCC: {len(lcc_nodes):,} of {G.number_of_nodes():,} nodes")
        G = G.subgraph(lcc_nodes).copy()
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    print(f"  Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
    return G


# ---------------------------------------------------------------------------
# Stage 2: Dense cost matrix via parallel scipy Dijkstra
# ---------------------------------------------------------------------------


def graph_to_csr(G: nx.Graph) -> sp.csr_matrix:
    """Convert NetworkX graph to scipy CSR adjacency matrix (cost weights)."""
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for u, v, d in G.edges(data=True):
        cost = float(d["cost"])
        ui, vi = node_to_idx[u], node_to_idx[v]
        rows.extend([ui, vi])
        cols.extend([vi, ui])
        data.extend([cost, cost])

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def compute_dense_cost_parallel(
    csr: sp.csr_matrix,
    n_workers: int = 32,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Compute full APSP via parallel scipy Dijkstra.

    scipy.sparse.csgraph.dijkstra releases the GIL, so ThreadPoolExecutor
    gives true parallelism without process serialization overhead.

    Results are stored as float32 (sufficient for walking times).
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

    Uses direct numpy operations for performance on large matrices
    (~4.9 billion elements at n=70k). Operates in float32.
    """
    if fn_name == "walk_steep":
        return np.float32(1.0) / (cost_matrix + np.float32(60.0)) ** 2
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
        n_workers=N_WORKERS,
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
    total_nodes_explored = hierarchy.total_nodes_explored

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
        "total_nodes_explored": total_nodes_explored,
    }


def benchmark_nystrom_scipy(
    csr: sp.csr_matrix,
    n: int,
    interaction_fn: Callable[[float], float],
    dense_interaction: np.ndarray,
    dense_mass: float,
    n_landmarks: int,
    rng: np.random.RandomState,
) -> dict:
    """Benchmark Nystrom baseline using scipy Dijkstra."""
    t0 = time.perf_counter()

    landmark_indices = rng.choice(n, size=min(n_landmarks, n), replace=False)
    landmark_indices.sort()

    vectorized_fn = np.vectorize(interaction_fn)

    # Compute cross-similarity matrix C (n x m) using scipy Dijkstra
    C = np.zeros((n, len(landmark_indices)))
    for k, lm_idx in enumerate(landmark_indices):
        dists = csg.dijkstra(csr, directed=False, indices=[lm_idx])
        C[:, k] = vectorized_fn(dists.ravel())

    # Landmark submatrix W (m x m) and pseudo-inverse
    W = C[landmark_indices, :]
    W_inv = np.linalg.pinv(W)

    build_time = time.perf_counter() - t0

    total_nodes_explored = len(landmark_indices) * n

    # Matvec: C @ (W_inv @ (C.T @ x))
    errors_list: list[dict[str, float]] = []
    matvec_times: list[float] = []

    for _ in range(N_ACTIVITY_TRIALS):
        activity = rng.rand(n) + 0.1
        dense_result = dense_interaction @ activity

        t0 = time.perf_counter()
        for _ in range(N_MATVEC_REPS):
            step1 = C.T @ activity
            step2 = W_inv @ step1
            nystrom_result = C @ step2
        matvec_times.append((time.perf_counter() - t0) / N_MATVEC_REPS)

        errors_list.append(compute_errors(nystrom_result, dense_result))

    ones = np.ones(n)
    method_mass = float(ones @ (C @ (W_inv @ (C.T @ ones))))

    return {
        "method": "nystrom",
        "method_param": f"m={n_landmarks}",
        "n_landmarks": n_landmarks,
        "build_time": build_time,
        "matvec_time": float(np.median(matvec_times)),
        "mean_relative_error": float(np.median([e["mean_relative_error"] for e in errors_list])),
        "max_relative_error": float(np.median([e["max_relative_error"] for e in errors_list])),
        "relative_rmse": float(np.median([e["relative_rmse"] for e in errors_list])),
        "n_stored_entries": n * len(landmark_indices),
        "interaction_mass_fraction": method_mass / dense_mass if dense_mass > EPS else 0.0,
        "total_nodes_explored": total_nodes_explored,
    }


def _apply_interaction_to_dists(
    dists: np.ndarray,
    fn_name: str,
) -> np.ndarray:
    """Apply interaction function to a distance array using numpy ops."""
    if fn_name == "walk_steep":
        return (dists + 60.0) ** (-2)
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
) -> dict:
    """Benchmark cutoff baseline using the pre-computed dense cost matrix.

    Builds the cutoff interaction matrix by masking the dense cost matrix
    at cutoff_radius. Avoids a redundant Dijkstra computation that would
    duplicate the dense matrix in memory.
    """
    t0 = time.perf_counter()
    mask = cost_matrix <= cutoff_radius
    total_nodes_explored = int(np.sum(mask))

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
            "description": "London walking network benchmark (70k nodes, 5 km radius)",
            "n_zones": n,
            "diameter": diameter,
            "mean_edge_cost": mel,
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


def _completed_configs(results: list[dict]) -> set[str]:
    """Return set of (fn_name, method, param) keys already completed."""
    return {
        f"{r['interaction_fn']}|{r['method']}|{r['method_param']}"
        for r in results
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_benchmark(output_path: str) -> None:
    """Execute the London walking 70k benchmark and write results to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    all_results = _load_existing_results(output)
    done_configs = _completed_configs(all_results)
    if all_results:
        print(f"  Resuming: {len(all_results)} existing results, "
              f"{len(done_configs)} configs done")

    t_start = time.time()

    print("=" * 72)
    print("London Walking Network ~70k Benchmark")
    print("=" * 72)

    # --- Stage 1: Network loading ---
    print("\n--- Stage 1: Network loading ---")
    G = load_walk_network()
    n = G.number_of_nodes()
    mel = mean_edge_cost(G)
    print(f"  Mean edge cost: {mel:.1f}s")
    print(f"  base_radius (mel*3): {mel * 3:.1f}s")

    # --- Stage 2: Dense cost matrix ---
    print("\n--- Stage 2: Dense cost matrix (parallel scipy Dijkstra) ---")
    csr = graph_to_csr(G)
    cache_npy = BENCH_DIR / "london_walk_60k_dense_cost.npy"
    cost_matrix = compute_dense_cost_parallel(
        csr, n_workers=N_WORKERS, cache_path=cache_npy
    )
    diameter = compute_network_diameter(cost_matrix)
    print(f"  Network diameter: {diameter:.0f}s")

    # Determine which kernels to run
    run_kernels = list(INTERACTION_FUNCTIONS.keys())

    def _save_incremental() -> None:
        elapsed = time.time() - t_start
        _write_results(output, all_results, n, diameter, mel, elapsed)

    # --- Stage 3: Run methods ---
    for fn_name in run_kernels:
        fn_callable = INTERACTION_FUNCTIONS[fn_name]
        rng = np.random.RandomState(42)

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
            "network": "london_walk_70k",
            "n_zones": n,
            "interaction_fn": fn_name,
            "mean_edge_cost": mel,
            "diameter": diameter,
        }

        # --- Hierarchical sweep ---
        for alpha in OVERLAP_FACTORS:
            config_key = f"{fn_name}|hierarchical|alpha={alpha},r0=3xmel"
            if config_key in done_configs:
                print(f"  Hierarchical alpha={alpha} ... SKIP (done)")
                continue
            print(f"  Hierarchical alpha={alpha} ...", end="", flush=True)
            try:
                result = benchmark_hierarchical(
                    G, n, fn_callable, dense_interaction, dense_mass,
                    mel, alpha, rng,
                )
                result.update(base_result)
                all_results.append(result)
                _save_incremental()
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
            config_key = f"{fn_name}|nystrom|m={n_lm}"
            if config_key in done_configs:
                print(f"  Nystrom m={n_lm} ... SKIP (done)")
                continue
            print(f"  Nystrom m={n_lm} ...", end="", flush=True)
            try:
                result = benchmark_nystrom_scipy(
                    csr, n, fn_callable, dense_interaction, dense_mass,
                    n_lm, rng,
                )
                result.update(base_result)
                all_results.append(result)
                _save_incremental()
                print(
                    f" err={result['mean_relative_error']:.4f}"
                    f" t_build={result['build_time']:.1f}s"
                )
            except Exception as exc:
                print(f" FAILED: {exc}")

        # --- Cutoff baselines (scipy) ---
        for frac in CUTOFF_FRACTIONS:
            cutoff_radius = frac * diameter
            config_key = f"{fn_name}|cutoff|frac={frac:.2f}"
            if config_key in done_configs:
                print(f"  Cutoff frac={frac:.2f} ... SKIP (done)")
                continue
            print(
                f"  Cutoff frac={frac:.2f} (r={cutoff_radius:.0f}s) ...",
                end="",
                flush=True,
            )
            try:
                result = benchmark_cutoff_dense(
                    cost_matrix, n, fn_name, dense_interaction, dense_mass,
                    cutoff_radius, frac, rng,
                )
                result.update(base_result)
                all_results.append(result)
                _save_incremental()
                print(
                    f" err={result['mean_relative_error']:.4f}"
                    f" mass={result['interaction_mass_fraction']:.3f}"
                    f" t_build={result['build_time']:.1f}s"
                )
            except Exception as exc:
                print(f" FAILED: {exc}")

        # Free dense interaction to save memory between kernels
        del dense_interaction

        n_kernel = len([r for r in all_results if r["interaction_fn"] == fn_name])
        print(f"\n  Total {n_kernel} results for '{fn_name}'")

    elapsed = time.time() - t_start

    # --- Final save ---
    _write_results(output, all_results, n, diameter, mel, elapsed)
    print(f"\n  Results: {output} ({len(all_results)} entries)")

    print(f"\n{'=' * 72}")
    print(f"Done. Total runtime: {elapsed / 60:.1f} min")
    print(f"{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="London walking network ~70k benchmark."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "london_walk_60k_results.json"),
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs that already have results in the output file",
    )
    args = parser.parse_args()

    if not args.resume:
        output = Path(args.output)
        if output.exists():
            output.unlink()

    run_benchmark(args.output)


if __name__ == "__main__":
    main()
