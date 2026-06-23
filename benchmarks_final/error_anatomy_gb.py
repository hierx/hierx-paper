#!/usr/bin/env python3
"""Error and bias anatomy of the hierarchical operator on the GB driving network.

Deeper error and bias quantification for the hierarchical operator on the
national-scale (Great Britain) case study, all against *exact* Dijkstra ground
truth. Four analyses:

1. Distance error vs. distance: sample origins uniformly, compute exact
   shortest-path costs to all nodes, compare against the hierarchical
   representative cost (Hierarchy.get_cost) for distance-stratified
   destination samples. Reports signed (bias) and absolute error per
   distance bin.
2. Interaction error vs. distance: the same OD pairs pushed through several
   distance-decay kernels, quantifying how distance error propagates into
   interaction error as a function of kernel steepness.
3. Accessibility error at sampled nodes: exact accessibility h_i for the
   census population/workplace activity vectors requires one full Dijkstra
   per sampled node i, so the error of the published national accessibility
   map can be measured directly at scale (signed, per kernel).
4. Sparse-activity error: when activity is concentrated on k support nodes,
   exact accessibility for the *entire* network needs only k Dijkstra runs.
   Sweeps support size k to quantify how error grows as activity becomes
   sparser (the "airports vs. population" effect).

Ground truth is computed fresh with scipy Dijkstra (GIL-released, threaded);
no precomputed dense matrices are required.

Usage:
    python -m benchmarks_final.error_anatomy_gb                 # full GB run
    python -m benchmarks_final.error_anatomy_gb --quick         # London 60k smoke test
    python -m benchmarks_final.error_anatomy_gb --self-test     # grid-network validation
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse.csgraph as csg

from benchmarks_final.common import DATA_DIR as RESULTS_DIR
from benchmarks_final.common import EPS, N_WORKERS
from hierx import Hierarchy, InteractionHierarchy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

UK_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uk_drive"
LONDON_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED = 42

# Distance-decay kernels evaluated on the sampled OD pairs and at the
# accessibility level. Offsets/scales are in seconds of driving time.
# "steep_paper" is the kernel used for the published GB map.
KERNELS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "steep_paper": ("(c+300)^{-2}", lambda c: (c + 300.0) ** (-2.0)),
    "very_steep": ("(c+60)^{-2}", lambda c: (c + 60.0) ** (-2.0)),
    "moderate": ("(c+1800)^{-1.5}", lambda c: (c + 1800.0) ** (-1.5)),
    "shallow": ("(c+3600)^{-1}", lambda c: (c + 3600.0) ** (-1.0)),
}
PAPER_KERNEL = "steep_paper"

# Global distance-bin edges (seconds): one near-field bin [0, 60) followed by
# log-spaced bins up to 10^5 s, then +inf so no finite distance is ever sent
# out of range by np.digitize. Empty top bins are pruned during aggregation.
BIN_EDGES = np.concatenate(
    [[0.0], np.logspace(np.log10(60.0), 5.0, 31), [np.inf]]
)


# ---------------------------------------------------------------------------
# Ground-truth Dijkstra helpers
# ---------------------------------------------------------------------------


def exact_distances(
    csr, source_csr_indices: list[int], n_workers: int
) -> np.ndarray:
    """Exact shortest-path distances from each source to all CSR nodes.

    Returns array of shape (len(sources), n_csr_nodes), float64. scipy's
    Dijkstra releases the GIL, so threads give true parallelism on the
    shared CSR without copying the graph.
    """
    out = np.empty((len(source_csr_indices), csr.shape[0]), dtype=np.float64)

    def run_one(pos: int) -> None:
        out[pos, :] = csg.dijkstra(
            csr, directed=False, indices=source_csr_indices[pos]
        )

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_one, p) for p in range(len(source_csr_indices))]
        for f in futures:
            f.result()
    return out


def kernel_values(fn: Callable[[np.ndarray], np.ndarray], d: np.ndarray) -> np.ndarray:
    """Apply a decay kernel, mapping unreachable (inf) distances to zero."""
    finite = np.isfinite(d)
    vals = np.zeros_like(d, dtype=np.float64)
    vals[finite] = fn(d[finite])
    return vals


# ---------------------------------------------------------------------------
# Stage A: OD-pair sampling and accessibility ground truth at origins
# ---------------------------------------------------------------------------


def stage_origins(
    csr,
    zone_csr_idx: np.ndarray,
    zones: np.ndarray,
    hierarchy: Hierarchy,
    activities: dict[str, np.ndarray],
    origins: np.ndarray,
    dest_per_bin: int,
    n_workers: int,
    rng: np.random.RandomState,
) -> dict:
    """Sample OD pairs stratified by distance and compute exact accessibility.

    For each origin (zone-array position): one exact Dijkstra, then
    (i) exact accessibility h_i = sum_j f(c_ij) x_j for every kernel and
        activity vector,
    (ii) destination samples stratified into global distance bins, recording
        the exact distance and the hierarchical cost from get_cost.
    """
    n_origins = len(origins)
    log.info(f"Stage A: {n_origins} origins, {dest_per_bin} destinations/bin")

    h_exact = {
        (act_name, k_name): np.zeros(n_origins)
        for act_name in activities
        for k_name in KERNELS
    }
    finite_fraction = np.zeros(n_origins)
    pair_origin_pos: list[np.ndarray] = []
    pair_dest_pos: list[np.ndarray] = []
    pair_d_exact: list[np.ndarray] = []

    # Per-origin destination sampling uses independent child seeds so results
    # don't depend on thread completion order.
    child_seeds = rng.randint(0, 2**31, size=n_origins)
    origin_results: list[tuple[int, np.ndarray, np.ndarray]] = []

    def process_origin(pos: int) -> tuple[int, np.ndarray, np.ndarray]:
        origin = origins[pos]
        d_csr = csg.dijkstra(
            csr, directed=False, indices=int(zone_csr_idx[origin])
        )
        d = d_csr[zone_csr_idx]  # aligned to zone-array order
        finite = np.isfinite(d)
        finite_fraction[pos] = finite.mean()

        for k_name, (_, fn) in KERNELS.items():
            f_vals = kernel_values(fn, d)
            for act_name, x in activities.items():
                h_exact[(act_name, k_name)][pos] = f_vals @ x

        # Stratified destination sampling
        local_rng = np.random.RandomState(child_seeds[pos])
        bin_idx = np.digitize(d[finite], BIN_EDGES) - 1
        finite_pos = np.flatnonzero(finite)
        sampled = []
        for b in range(len(BIN_EDGES) - 1):
            members = finite_pos[bin_idx == b]
            members = members[members != origin]
            if len(members) == 0:
                continue
            take = min(dest_per_bin, len(members))
            sampled.append(local_rng.choice(members, size=take, replace=False))
        dest_pos = (
            np.concatenate(sampled) if sampled else np.empty(0, dtype=np.int64)
        )
        return pos, dest_pos, d[dest_pos]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_origin, p) for p in range(n_origins)]
        for i, f in enumerate(as_completed(futures)):
            origin_results.append(f.result())
            if (i + 1) % 100 == 0 or i + 1 == n_origins:
                log.info(
                    f"  origin {i + 1}/{n_origins} "
                    f"({time.perf_counter() - t0:.0f}s)"
                )

    for pos, dest_pos, d_vals in sorted(origin_results, key=lambda r: r[0]):
        pair_origin_pos.append(np.full(len(dest_pos), pos, dtype=np.int64))
        pair_dest_pos.append(dest_pos)
        pair_d_exact.append(d_vals)

    pair_origin_pos = np.concatenate(pair_origin_pos)
    pair_dest_pos = np.concatenate(pair_dest_pos)
    pair_d_exact = np.concatenate(pair_d_exact)

    # Hierarchical cost lookup (pure-Python dict walk; single-threaded)
    log.info(f"  get_cost lookup for {len(pair_d_exact):,} sampled pairs ...")
    t0 = time.perf_counter()
    pair_c_hier = np.array(
        [
            hierarchy.get_cost(int(zones[origins[op]]), int(zones[dp]))
            for op, dp in zip(pair_origin_pos, pair_dest_pos)
        ]
    )
    log.info(f"  lookups done in {time.perf_counter() - t0:.1f}s")

    return {
        "h_exact": h_exact,
        "finite_fraction": finite_fraction,
        "pair_origin_pos": pair_origin_pos,
        "pair_dest_pos": pair_dest_pos,
        "pair_d_exact": pair_d_exact,
        "pair_c_hier": pair_c_hier,
    }


# ---------------------------------------------------------------------------
# Stage B: sparse-activity ground truth
# ---------------------------------------------------------------------------


def stage_sparse_supports(
    csr,
    zone_csr_idx: np.ndarray,
    support_pos: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    n_workers: int,
) -> np.ndarray:
    """Exact full-network accessibility for unit activity on the support.

    h_exact = sum over support nodes j of f(d(j, .)) — one Dijkstra per
    support node; partial sums combined in the main thread.
    """
    n_zones = len(zone_csr_idx)
    chunk_size = 8
    chunks = [
        support_pos[s : s + chunk_size]
        for s in range(0, len(support_pos), chunk_size)
    ]

    def process_chunk(chunk: np.ndarray) -> np.ndarray:
        partial = np.zeros(n_zones)
        for p in chunk:
            d = csg.dijkstra(csr, directed=False, indices=int(zone_csr_idx[p]))
            partial += kernel_values(fn, d[zone_csr_idx])
        return partial

    h_exact = np.zeros(n_zones)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_chunk, c) for c in chunks]
        for f in as_completed(futures):
            h_exact += f.result()
    return h_exact


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def binned_pair_stats(
    d_exact: np.ndarray, c_hier: np.ndarray
) -> list[dict]:
    """Per-distance-bin error statistics for sampled OD pairs.

    Distance error is signed (c_hier - d_exact); interaction error is
    signed (f(c_hier) - f(d_exact)) / f(d_exact) for each kernel.
    """
    finite = np.isfinite(c_hier)
    bins = np.digitize(d_exact, BIN_EDGES) - 1
    out = []
    for b in range(len(BIN_EDGES) - 1):
        mask = (bins == b) & finite
        n_pairs = int(mask.sum())
        if n_pairs == 0:
            continue
        de, ch = d_exact[mask], c_hier[mask]
        dist_err = ch - de
        rel_dist_err = dist_err / np.maximum(de, EPS)
        row = {
            "bin_lo": float(BIN_EDGES[b]),
            "bin_hi": float(BIN_EDGES[b + 1]),
            "n_pairs": n_pairs,
            "n_unreachable_hier": int(((bins == b) & ~finite).sum()),
            "dist_bias_abs": float(dist_err.mean()),
            "dist_bias_rel": float(rel_dist_err.mean()),
            "dist_rmse_rel": float(np.sqrt((rel_dist_err**2).mean())),
            "dist_err_rel_p5": float(np.percentile(rel_dist_err, 5)),
            "dist_err_rel_p50": float(np.percentile(rel_dist_err, 50)),
            "dist_err_rel_p95": float(np.percentile(rel_dist_err, 95)),
            "kernels": {},
        }
        for k_name, (label, fn) in KERNELS.items():
            f_exact = fn(de)
            f_hier = fn(ch)
            rel = (f_hier - f_exact) / np.maximum(f_exact, EPS)
            row["kernels"][k_name] = {
                "label": label,
                "bias_rel": float(rel.mean()),
                "rmse_rel": float(np.sqrt((rel**2).mean())),
                "p5": float(np.percentile(rel, 5)),
                "p50": float(np.percentile(rel, 50)),
                "p95": float(np.percentile(rel, 95)),
            }
        out.append(row)
    return out


def accessibility_error_stats(
    h_hier_at_origins: np.ndarray, h_exact_at_origins: np.ndarray
) -> dict:
    """Signed and absolute relative accessibility error at sampled origins."""
    rel = (h_hier_at_origins - h_exact_at_origins) / np.maximum(
        np.abs(h_exact_at_origins), EPS
    )
    return {
        "n_samples": int(len(rel)),
        "bias_rel": float(rel.mean()),
        "mean_abs_rel": float(np.abs(rel).mean()),
        "rmse_rel": float(np.sqrt((rel**2).mean())),
        "p5": float(np.percentile(rel, 5)),
        "p50": float(np.percentile(rel, 50)),
        "p95": float(np.percentile(rel, 95)),
        "max_abs_rel": float(np.abs(rel).max()),
    }


# ---------------------------------------------------------------------------
# Self-test on a small grid network
# ---------------------------------------------------------------------------


def self_test() -> None:
    """Validate the ground-truth plumbing against a brute-force dense baseline."""
    from hierx.utils import (
        compute_dense_cost_matrix,
        generate_grid_network,
    )

    log.info("Self-test on a 15x15 grid network ...")
    G = generate_grid_network(15, 15, spacing=1000.0)
    zones = np.array(sorted(G.nodes()))
    n = len(zones)

    hierarchy = Hierarchy(
        G, base_radius=3000.0, increase_factor=2, overlap_factor=2.0, n_workers=4
    )
    csr = hierarchy._csr
    zone_csr_idx = np.array([hierarchy._node_to_idx[z] for z in zones])

    dense_cost = compute_dense_cost_matrix(G, zones=list(zones))
    rng = np.random.RandomState(SEED)
    x = rng.uniform(0.1, 1.1, size=n)

    fn = KERNELS[PAPER_KERNEL][1]
    h_dense = kernel_values(fn, dense_cost) @ x

    # Exact accessibility via the threaded Dijkstra path must match dense
    d_all = exact_distances(csr, [int(i) for i in zone_csr_idx], 4)
    h_stage = np.array(
        [kernel_values(fn, d_all[i][zone_csr_idx]) @ x for i in range(n)]
    )
    assert np.allclose(h_stage, h_dense, rtol=1e-10), "Dijkstra accumulation mismatch"

    # Sparse-support ground truth must match dense restricted to the support
    support = rng.choice(n, size=10, replace=False)
    h_sparse = stage_sparse_supports(csr, zone_csr_idx, support, fn, 4)
    x_sparse = np.zeros(n)
    x_sparse[support] = 1.0
    h_sparse_dense = kernel_values(fn, dense_cost) @ x_sparse
    assert np.allclose(h_sparse, h_sparse_dense, rtol=1e-10), (
        "Sparse-support accumulation mismatch"
    )

    # get_cost must reproduce the operator's effective per-pair value:
    # matvec(e_j)[i] == f(get_cost(i, j)) wherever the pair is stored
    ih = InteractionHierarchy(hierarchy, lambda c: fn(np.float64(c)))
    j = int(rng.choice(n))
    e_j = np.zeros(n)
    e_j[j] = 1.0
    col = ih.matvec(e_j)
    for i in rng.choice(n, size=50, replace=False):
        c = hierarchy.get_cost(int(zones[i]), int(zones[j]))
        if np.isfinite(c):
            assert np.isclose(col[i], fn(np.float64(c)), rtol=1e-9), (
                f"get_cost inconsistent with operator at pair ({i}, {j}): "
                f"matvec={col[i]:.6e} f(get_cost)={fn(np.float64(c)):.6e}"
            )
    log.info("Self-test passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error and bias anatomy of the hierarchical operator"
    )
    parser.add_argument("--network", type=str, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test on London 60k with reduced samples")
    parser.add_argument("--self-test", action="store_true",
                        help="Validate plumbing on a small grid and exit")
    parser.add_argument("--n-origins", type=int, default=None)
    parser.add_argument("--dest-per-bin", type=int, default=20)
    parser.add_argument("--support-sizes", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    parser.add_argument("--skip-extra-kernels", action="store_true",
                        help="Only run the paper kernel at accessibility level")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    t_start = time.perf_counter()

    # ---- Resolve configuration ----
    if args.quick:
        net_path = Path(args.network) if args.network else (
            LONDON_DATA_DIR / "london_60k.graphml"
        )
        n_origins = args.n_origins or 24
        support_sizes = [int(s) for s in (args.support_sizes or "5,20").split(",")]
        n_trials = min(args.n_trials, 1)
        network_label = "london_60k_quick"
    else:
        net_path = Path(args.network) if args.network else (
            UK_DATA_DIR / "gb_drive.pkl"
        )
        n_origins = args.n_origins or 1000
        support_sizes = [
            int(s) for s in (args.support_sizes or "10,100,1000").split(",")
        ]
        n_trials = args.n_trials
        network_label = "gb_drive"

    # ---- Load network ----
    log.info(f"Loading {net_path} ...")
    t0 = time.perf_counter()
    if net_path.suffix == ".pkl":
        with open(net_path, "rb") as f:
            G = pickle.load(f)
    else:
        G = nx.read_graphml(net_path, node_type=int)
        for _, _, d in G.edges(data=True):
            d["cost"] = float(d["cost"])
    log.info(
        f"  Loaded in {time.perf_counter() - t0:.1f}s: "
        f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
    )

    zones = np.array(sorted(G.nodes()), dtype=np.int64)
    n = len(zones)
    mel = sum(d["cost"] for _, _, d in G.edges(data=True)) / G.number_of_edges()
    log.info(f"Network: {n:,} zones, mean edge cost {mel:.1f}s")

    # ---- Load or synthesize activity vectors ----
    activities: dict[str, np.ndarray] = {}
    npz_path = UK_DATA_DIR / "gb_node_activity.npz"
    if not args.quick and npz_path.exists():
        activity_data = np.load(npz_path)
        assert np.array_equal(zones, activity_data["zones"]), (
            "Zone order mismatch between network and activity NPZ"
        )
        activities["population"] = activity_data["node_pop"].astype(np.float64)
        activities["workplace"] = activity_data["node_wp"].astype(np.float64)
        log.info(
            f"Census activity: population sum={activities['population'].sum():,.0f}, "
            f"workplace sum={activities['workplace'].sum():,.0f}"
        )
    else:
        rng_act = np.random.RandomState(SEED)
        activities["population"] = rng_act.lognormal(0.0, 1.0, size=n)
        log.info("Using synthetic lognormal activity (quick mode)")

    # ---- Build hierarchy ----
    base_radius = max(3 * mel, 60.0)
    log.info(
        f"Building hierarchy: base_radius={base_radius:.1f}s, "
        f"increase_factor=2, overlap_factor=2.0, n_workers={args.n_workers}"
    )
    t0 = time.perf_counter()
    hierarchy = Hierarchy(
        G,
        base_radius=base_radius,
        increase_factor=2,
        overlap_factor=2.0,
        n_workers=args.n_workers,
    )
    hierarchy_time = time.perf_counter() - t0
    log.info(f"  Hierarchy built in {hierarchy_time:.1f}s "
             f"({len(hierarchy.radii)} layers)")

    csr = hierarchy._csr
    zone_csr_idx = np.array([hierarchy._node_to_idx[z] for z in zones])

    # ---- Sample origins and supports (before heavy stages, fixed seed) ----
    rng = np.random.RandomState(SEED)
    origins = rng.choice(n, size=min(n_origins, n), replace=False)

    pop = activities["population"]
    p_pop = pop / pop.sum()
    # Per-(support size, trial) seed. A guard rejects the rare collision the
    # additive formula can produce if support sizes near a trial multiple are
    # added later; the published support sizes (10/100/1000) are collision-free.
    supports: dict[tuple[int, int], np.ndarray] = {}
    seen_seeds: set[int] = set()
    for k in support_sizes:
        for trial in range(n_trials):
            seed = SEED + 1000 * trial + k
            assert seed not in seen_seeds, (
                f"support seed collision at (k={k}, trial={trial}); "
                "adjust the seed formula for these support sizes"
            )
            seen_seeds.add(seed)
            trial_rng = np.random.RandomState(seed)
            supports[(k, trial)] = trial_rng.choice(
                n, size=k, replace=False, p=p_pop
            )

    # ---- Accessibility-level: extra kernels (build, matvec, release) ----
    h_hier: dict[str, dict[str, np.ndarray]] = {k: {} for k in KERNELS}
    # Paper kernel last: it is the only operator kept alive (for Stage B),
    # so this keeps at most one transient operator in memory at a time.
    kernel_names = (
        [PAPER_KERNEL]
        if args.skip_extra_kernels
        else [k for k in KERNELS if k != PAPER_KERNEL] + [PAPER_KERNEL]
    )
    ih_paper = None
    interaction_times = {}
    for k_name in kernel_names:
        label, fn = KERNELS[k_name]
        log.info(f"Building interaction operator for {k_name} = {label} ...")
        t0 = time.perf_counter()
        ih = InteractionHierarchy(hierarchy, lambda c, _fn=fn: float(_fn(np.float64(c))))
        interaction_times[k_name] = time.perf_counter() - t0
        n_entries = sum(D.nnz for D in ih.D.values())
        log.info(
            f"  Built in {interaction_times[k_name]:.1f}s, {n_entries:,} entries"
        )
        for act_name, x in activities.items():
            t0 = time.perf_counter()
            h_hier[k_name][act_name] = ih.matvec(x)
            log.info(
                f"  matvec[{act_name}]: {(time.perf_counter() - t0) * 1000:.0f}ms"
            )
        if k_name == PAPER_KERNEL:
            ih_paper = ih  # keep alive for sparse-support matvecs
        else:
            del ih
            gc.collect()

    # ---- Confirm the rebuilt hierarchy reproduces this paper's own GB
    #      accessibility field (Figure 6a, computed earlier by
    #      run_uk_drive_accessibility.py). A ~zero diff shows the build is
    #      deterministic, so the error-anatomy statistics below characterize
    #      the exact field shown in Figure 6a. ----
    map_repro_check = None
    paper_map_npz = RESULTS_DIR / "gb_drive_pop_accessibility_nodes.npz"
    if not args.quick and paper_map_npz.exists() and PAPER_KERNEL in h_hier:
        paper_map = np.load(paper_map_npz)
        if np.array_equal(paper_map["zones"], zones):
            rebuilt = h_hier[PAPER_KERNEL]["population"]
            reference = paper_map["acc_population"]
            rel = np.abs(rebuilt - reference) / np.maximum(np.abs(reference), EPS)
            map_repro_check = {
                "max_rel_diff": float(rel.max()),
                "mean_rel_diff": float(rel.mean()),
            }
            log.info(
                f"Rebuild reproduces our Figure 6a GB map: "
                f"max rel diff {rel.max():.2e}, mean {rel.mean():.2e}"
            )

    # ---- Stage A: origins ----
    stage_a = stage_origins(
        csr, zone_csr_idx, zones, hierarchy, activities, origins,
        args.dest_per_bin, args.n_workers, rng,
    )

    # ---- Stage B: sparse supports (paper kernel) ----
    fn_paper = KERNELS[PAPER_KERNEL][1]
    sparse_results = []
    for (k, trial), support_pos in sorted(supports.items()):
        log.info(f"Stage B: support size {k}, trial {trial} ...")
        t0 = time.perf_counter()
        h_exact_sparse = stage_sparse_supports(
            csr, zone_csr_idx, support_pos, fn_paper, args.n_workers
        )
        x_sparse = np.zeros(n)
        x_sparse[support_pos] = 1.0
        h_hier_sparse = ih_paper.matvec(x_sparse)
        reachable = h_exact_sparse > EPS
        rel = (
            h_hier_sparse[reachable] - h_exact_sparse[reachable]
        ) / h_exact_sparse[reachable]
        sparse_results.append({
            "support_size": k,
            "trial": trial,
            "n_reachable": int(reachable.sum()),
            "bias_rel": float(rel.mean()),
            "mean_abs_rel": float(np.abs(rel).mean()),
            "relative_rmse": float(
                np.linalg.norm(h_hier_sparse[reachable] - h_exact_sparse[reachable])
                / np.linalg.norm(h_exact_sparse[reachable])
            ),
            "p5": float(np.percentile(rel, 5)),
            "p50": float(np.percentile(rel, 50)),
            "p95": float(np.percentile(rel, 95)),
            "mass_ratio": float(h_hier_sparse.sum() / h_exact_sparse.sum()),
            "elapsed_s": time.perf_counter() - t0,
        })
        log.info(
            f"  RMSE={sparse_results[-1]['relative_rmse']:.4f}, "
            f"bias={sparse_results[-1]['bias_rel']:+.4f} "
            f"({sparse_results[-1]['elapsed_s']:.0f}s)"
        )

    # ---- Aggregate ----
    pair_stats = binned_pair_stats(
        stage_a["pair_d_exact"], stage_a["pair_c_hier"]
    )
    accessibility_errors = {}
    for (act_name, k_name), h_ex in stage_a["h_exact"].items():
        if k_name not in h_hier or act_name not in h_hier[k_name]:
            continue
        accessibility_errors[f"{act_name}__{k_name}"] = accessibility_error_stats(
            h_hier[k_name][act_name][origins], h_ex
        )

    # ---- Save ----
    results = {
        "metadata": {
            "network": str(net_path),
            "network_label": network_label,
            "n_zones": int(n),
            "mean_edge_cost": float(mel),
            "base_radius": float(base_radius),
            "overlap_factor": 2.0,
            "increase_factor": 2,
            "n_layers": len(hierarchy.radii),
            "hierarchy_build_time": hierarchy_time,
            "interaction_build_times": interaction_times,
            "seed": SEED,
            "n_origins": int(len(origins)),
            "dest_per_bin": args.dest_per_bin,
            "n_sampled_pairs": int(len(stage_a["pair_d_exact"])),
            "support_sizes": support_sizes,
            "n_trials": n_trials,
            "kernels": {k: v[0] for k, v in KERNELS.items()},
            "paper_kernel": PAPER_KERNEL,
            "activities": list(activities.keys()),
            "n_workers": args.n_workers,
            "total_runtime_s": time.perf_counter() - t_start,
        },
        "figure6a_reproduction": map_repro_check,
        "pair_stats_by_distance_bin": pair_stats,
        "accessibility_errors_at_origins": accessibility_errors,
        "sparse_activity_results": sparse_results,
        "origin_finite_fraction": {
            "min": float(stage_a["finite_fraction"].min()),
            "mean": float(stage_a["finite_fraction"].mean()),
        },
    }

    suffix = "_quick" if args.quick else ""
    results_path = RESULTS_DIR / f"error_anatomy_gb_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    npz_payload = {
        "origins": origins,
        "zones_at_origins": zones[origins],
        "pair_origin_pos": stage_a["pair_origin_pos"],
        "pair_dest_pos": stage_a["pair_dest_pos"],
        "pair_d_exact": stage_a["pair_d_exact"],
        "pair_c_hier": stage_a["pair_c_hier"],
        "bin_edges": BIN_EDGES,
    }
    for (act_name, k_name), h_ex in stage_a["h_exact"].items():
        npz_payload[f"h_exact__{act_name}__{k_name}"] = h_ex
    for k_name, by_act in h_hier.items():
        for act_name, h in by_act.items():
            npz_payload[f"h_hier_at_origins__{act_name}__{k_name}"] = h[origins]
    npz_path_out = RESULTS_DIR / f"error_anatomy_gb_pairs{suffix}.npz"
    np.savez_compressed(npz_path_out, **npz_payload)
    log.info(f"Raw sampled data saved to {npz_path_out}")

    log.info("=" * 60)
    log.info(f"DONE in {(time.perf_counter() - t_start) / 60:.1f} min")
    for key, st in accessibility_errors.items():
        log.info(
            f"  accessibility {key}: bias={st['bias_rel']:+.4f}, "
            f"rmse={st['rmse_rel']:.4f}"
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
