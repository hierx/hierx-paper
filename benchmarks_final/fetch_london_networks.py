#!/usr/bin/env python3
"""Download concentric London road networks from OpenStreetMap via OSMnx.

By default, fetches only the network needed by the benchmarks:
  - london_large.graphml   (25 km radius, ~50-100k nodes)

Pass --all to also fetch exploratory networks:
  - london_small.graphml   (1 km radius, ~500-2k nodes)
  - london_medium.graphml  (5 km radius, ~5-10k nodes)

Edge attribute "cost" is free-flow travel time in seconds, imputed from
OSM speed limits.  Node attributes "x" (longitude) and "y" (latitude)
are preserved.

Requirements:
    pip install osmnx
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx

try:
    import osmnx as ox
except ImportError:
    print("ERROR: osmnx is required.  Install with:\n  pip install --user osmnx\n")
    sys.exit(1)


# Charing Cross, London
CENTER = (51.5074, -0.1278)

# Only london_large is needed by the benchmarks (london_benchmark_60k.py).
# The smaller networks are available for experimentation; pass --all to fetch them.
NETWORKS_DEFAULT = [
    {"label": "london_large", "dist": 25_000},
]

NETWORKS_ALL = [
    {"label": "london_small", "dist": 1_000},
    {"label": "london_medium", "dist": 5_000},
    {"label": "london_large", "dist": 25_000},
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_and_process(label: str, dist: int) -> nx.Graph:
    """Download, clean, and return a simple undirected graph.

    Steps:
      1. Download drive network from OSMnx
      2. Add free-flow speeds and travel times
      3. Convert to undirected
      4. Extract largest connected component
      5. Simplify MultiGraph → Graph (keep shortest-time parallel edge)
      6. Set "cost" = travel_time (seconds), keep "length" (meters)
      7. Relabel nodes to 0..n-1, preserving x/y coordinates
    """
    print(f"\n{'=' * 60}")
    print(f"  Fetching {label} (radius={dist}m)")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()

    # 1. Download
    G_directed = ox.graph.graph_from_point(CENTER, dist=dist, network_type="drive")
    print(
        f"  Downloaded: {G_directed.number_of_nodes()} nodes, {G_directed.number_of_edges()} edges"
    )

    # 2. Add speeds and travel times
    G_directed = ox.routing.add_edge_speeds(G_directed)
    G_directed = ox.routing.add_edge_travel_times(G_directed)

    # 3. Convert to undirected
    G_multi = ox.convert.to_undirected(G_directed)
    print(f"  Undirected: {G_multi.number_of_nodes()} nodes, {G_multi.number_of_edges()} edges")

    # 4. Largest connected component
    largest_cc = max(nx.connected_components(G_multi), key=len)
    G_multi = G_multi.subgraph(largest_cc).copy()
    print(f"  Largest CC: {G_multi.number_of_nodes()} nodes, {G_multi.number_of_edges()} edges")

    # 5. MultiGraph → simple Graph (keep shortest travel_time parallel edge)
    G = nx.Graph()
    for node, data in G_multi.nodes(data=True):
        G.add_node(node, **data)

    for u, v, key, data in G_multi.edges(keys=True, data=True):
        tt = data.get("travel_time", float("inf"))
        if G.has_edge(u, v):
            if tt < G[u][v].get("travel_time", float("inf")):
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)

    # 6. Set "cost" = travel_time; strip non-serializable attributes
    #    Keep only: nodes(x, y), edges(cost, length)
    for node, data in G.nodes(data=True):
        keep = {"x": float(data["x"]), "y": float(data["y"])}
        data.clear()
        data.update(keep)

    for u, v, data in G.edges(data=True):
        keep = {
            "cost": float(data.get("travel_time", data.get("length", 1.0))),
            "length": float(data.get("length", 0.0)),
        }
        data.clear()
        data.update(keep)

    # 7. Relabel to 0..n-1
    old_to_new = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, old_to_new)

    elapsed = time.perf_counter() - t0
    print(f"  Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Processed in {elapsed:.1f}s")

    return G


def verify_graph(G: nx.Graph, label: str) -> None:
    """Print summary stats and verify required attributes."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Check connectivity
    assert nx.is_connected(G), f"{label}: graph is not connected"

    # Check node attributes
    sample_node = next(iter(G.nodes()))
    node_data = G.nodes[sample_node]
    assert "x" in node_data, f"{label}: missing 'x' node attribute"
    assert "y" in node_data, f"{label}: missing 'y' node attribute"

    # Check edge attributes
    sample_edge = next(iter(G.edges()))
    edge_data = G.edges[sample_edge]
    assert "cost" in edge_data, f"{label}: missing 'cost' edge attribute"

    # Edge cost statistics
    costs = [d["cost"] for _, _, d in G.edges(data=True)]
    mean_cost = sum(costs) / len(costs)
    min_cost = min(costs)
    max_cost = max(costs)

    print(f"\n  {label} verification:")
    print(f"    Nodes: {n:,}")
    print(f"    Edges: {m:,}")
    print("    Connected: True")
    print("    Edge cost (travel_time seconds):")
    print(f"      min={min_cost:.1f}  mean={mean_cost:.1f}  max={max_cost:.1f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download London drive networks")
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch all network sizes (small, medium, large). Default: large only.",
    )
    args = parser.parse_args()

    networks = NETWORKS_ALL if args.all else NETWORKS_DEFAULT

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for spec in networks:
        label = spec["label"]
        dist = spec["dist"]
        out_path = DATA_DIR / f"{label}.graphml"

        # Skip if already downloaded (saves time when iterating)
        if out_path.exists():
            print(f"\n  {label}: already exists at {out_path}, skipping download.")
            print("  (Delete the file to force re-download.)")
            try:
                G = nx.read_graphml(out_path, node_type=int)
                for _, _, d in G.edges(data=True):
                    d["cost"] = float(d["cost"])
                for _, d in G.nodes(data=True):
                    d["x"] = float(d["x"])
                    d["y"] = float(d["y"])
                verify_graph(G, label)
            except Exception as e:
                print(f"  WARNING: existing file failed verification: {e}")
            continue

        try:
            G = fetch_and_process(label, dist)
            verify_graph(G, label)
            nx.write_graphml(G, out_path)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"\n  ERROR fetching {label}: {e}")
            print("  If the Overpass API is blocked, download the network")
            print("  externally and place the GraphML file at:")
            print(f"    {out_path}")
            continue

    print(f"\n{'=' * 60}")
    print("  Done. GraphML files in:", DATA_DIR)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
