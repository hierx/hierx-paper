#!/usr/bin/env python3
"""Compute population- and employment-weighted walking accessibility.

Uses Census population and workplace counts disaggregated to network
nodes as the activity vectors.

Outputs JSON results and NPZ node-level data used by plot_accessibility_maps.py (Fig 10).

Usage:
    python -m benchmarks_final.run_population_accessibility
    python -m benchmarks_final.run_population_accessibility --n-workers 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from hierx import Hierarchy, InteractionHierarchy

from benchmarks_final.common import (
    DATA_DIR,
    N_WORKERS,
    estimate_shm_bytes,
    track_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NETWORK_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CENSUS_DIR = NETWORK_DATA_DIR / "london_census"

CATEGORIES = {
    "population": {
        "key": "node_pop",
        "label": "Population",
        "cmap": "inferno",
    },
    "workplace": {
        "key": "node_wp",
        "label": "Employment",
        "cmap": "inferno",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute population/employment walking accessibility"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=N_WORKERS,
        help=f"Parallel workers for Dijkstra (default: {N_WORKERS})",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="london_walk_1M.graphml",
        help="Network file in data/ (default: london_walk_1M.graphml)",
    )
    args = parser.parse_args()

    import networkx as nx

    # ---- Load node activity vectors ----
    npz_path = CENSUS_DIR / "london_node_activity.npz"
    if not npz_path.exists():
        log.error(
            f"Node activity not found: {npz_path}\n"
            "  Build it first with:\n"
            "    python -m benchmarks_final.build_population_activity\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
        sys.exit(1)

    activity_data = np.load(npz_path)
    log.info(f"Loaded node activity from {npz_path}")
    for key in ["node_pop", "node_wp"]:
        arr = activity_data[key]
        log.info(f"  {key}: sum={arr.sum():,.0f}, nonzero={np.count_nonzero(arr):,}")

    # ---- Load network ----
    net_path = NETWORK_DATA_DIR / args.network
    if not net_path.exists():
        log.error(
            f"Network not found: {net_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_london_walking\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
        sys.exit(1)

    log.info(f"Loading {net_path} ...")
    t0 = time.perf_counter()
    G = nx.read_graphml(net_path, node_type=int)
    for _, _, d in G.edges(data=True):
        d["cost"] = float(d["cost"])
    for _, d in G.nodes(data=True):
        d["x"] = float(d["x"])
        d["y"] = float(d["y"])
    dt = time.perf_counter() - t0
    log.info(f"  Loaded in {dt:.1f}s: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    zones = sorted(G.nodes())
    n = len(zones)
    mel = sum(d["cost"] for _, _, d in G.edges(data=True)) / G.number_of_edges()
    node_xy = np.array([[G.nodes[z]["x"], G.nodes[z]["y"]] for z in zones])

    log.info(f"Network: {n:,} nodes, mean edge cost: {mel:.1f}s")

    # Verify zone order matches NPZ
    npz_zones = activity_data["zones"]
    assert np.array_equal(np.array(zones, dtype=np.int64), npz_zones), (
        "Zone order mismatch between network and NPZ"
    )

    # ---- Build hierarchy + interaction (with memory tracking) ----
    base_radius = mel * 3
    overlap_factor = 2.0

    log.info("Building hierarchy ...")
    with track_memory() as mem_info:
        t0 = time.perf_counter()
        hierarchy = Hierarchy(
            G,
            base_radius=base_radius,
            increase_factor=2,
            overlap_factor=overlap_factor,
            n_workers=args.n_workers,
        )
        hierarchy_time = time.perf_counter() - t0
        log.info(f"  Hierarchy built in {hierarchy_time:.1f}s ({len(hierarchy.radii)} layers)")

        # ---- Build interaction ----
        fn = lambda c: (c + 60) ** (-2)  # noqa: E731

        log.info("Building interaction matrices ...")
        t0 = time.perf_counter()
        ih = InteractionHierarchy(hierarchy, fn)
        ih_time = time.perf_counter() - t0
        n_entries = sum(D.nnz for D in ih.D.values())
        log.info(f"  Built in {ih_time:.1f}s, {n_entries:,} stored entries")

    peak_memory_bytes = mem_info["peak_memory_bytes"] + estimate_shm_bytes(hierarchy)
    log.info(f"  Peak memory (hierarchy+interaction): {peak_memory_bytes / 1e6:.1f} MB")

    # Record total_nodes_explored if available
    total_nodes_explored = getattr(hierarchy, "total_nodes_explored", None)
    if total_nodes_explored is not None:
        log.info(f"  Total nodes explored (Dijkstra): {total_nodes_explored:,}")

    # ---- Compute accessibility for each activity vector ----
    results = {}
    for cat_name, cat_info in CATEGORIES.items():
        activity = activity_data[cat_info["key"]]
        total = activity.sum()
        n_active = int(np.sum(activity > 0))
        log.info(f"\nCategory: {cat_name}")
        log.info(f"  Total activity: {total:,.0f}, active nodes: {n_active:,}")

        t0 = time.perf_counter()
        accessibility = ih.matvec(activity)
        matvec_time = time.perf_counter() - t0
        log.info(f"  Matvec: {matvec_time * 1000:.1f}ms")
        log.info(
            f"  Accessibility: min={accessibility.min():.4e}, "
            f"mean={accessibility.mean():.4e}, max={accessibility.max():.4e}"
        )

        results[cat_name] = {
            "total_activity": float(total),
            "n_active_nodes": n_active,
            "matvec_time": matvec_time,
            "accessibility": accessibility,
            "stats": {
                "min": float(accessibility.min()),
                "mean": float(accessibility.mean()),
                "max": float(accessibility.max()),
                "std": float(accessibility.std()),
                "p5": float(np.percentile(accessibility, 5)),
                "p50": float(np.percentile(accessibility, 50)),
                "p95": float(np.percentile(accessibility, 95)),
            },
        }

    # ---- Save results JSON ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_out = {
        "metadata": {
            "network": args.network,
            "n_zones": n,
            "n_layers": len(hierarchy.radii),
            "hierarchy_build_time": hierarchy_time,
            "interaction_build_time": ih_time,
            "n_stored_entries": n_entries,
            "interaction_fn": "steep: (c+60)^{-2}",
            "total_nodes_explored": total_nodes_explored,
            "peak_memory_bytes": peak_memory_bytes,
        },
        "results": {
            cat: {k: v for k, v in res.items() if k != "accessibility"}
            for cat, res in results.items()
        },
    }
    results_path = DATA_DIR / "population_accessibility_results.json"
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)
    log.info(f"\nResults saved to {results_path}")

    # ---- Save node-level NPZ ----
    net_stem = Path(args.network).stem
    npz_out = {
        "node_xy": node_xy,
        "zones": np.array(zones, dtype=np.int64),
        "edges": np.array(list(G.edges()), dtype=np.int64),
    }
    for cat_name, res in results.items():
        npz_out[f"acc_{cat_name}"] = res["accessibility"]
        npz_out[f"activity_{cat_name}"] = activity_data[CATEGORIES[cat_name]["key"]]
    npz_out_path = DATA_DIR / f"{net_stem}_pop_accessibility_nodes.npz"
    np.savez_compressed(npz_out_path, **npz_out)
    log.info(f"Node-level data saved to {npz_out_path}")

    # ---- Summary ----
    log.info("\n" + "=" * 60)
    log.info(f"SUMMARY: Population/employment accessibility on {n:,}-node network")
    log.info(f"  Hierarchy: {hierarchy_time:.1f}s, Interaction: {ih_time:.1f}s")
    log.info(f"  Peak memory: {peak_memory_bytes / 1e6:.1f} MB")
    if total_nodes_explored is not None:
        log.info(f"  Total nodes explored: {total_nodes_explored:,}")
    for cat_name, res in results.items():
        log.info(
            f"  {cat_name}: {res['total_activity']:,.0f} total, "
            f"matvec={res['matvec_time'] * 1000:.1f}ms"
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
