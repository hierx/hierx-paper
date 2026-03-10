#!/usr/bin/env python3
"""Download concentric London WALKING networks from OpenStreetMap via OSMnx.

By default, fetches only the networks needed by the benchmarks:
  - london_walk_large.graphml   (5 km radius, ~30-100k nodes)
  - london_walk_1M.graphml      (22 km radius, ~1.5M nodes, unsimplified — for accessibility)

Pass --all to also fetch exploratory networks:
  - london_walk_tiny.graphml    (500 m radius, ~hundreds of nodes)
  - london_walk_small.graphml   (1 km radius, ~1-5k nodes)
  - london_walk_medium.graphml  (2 km radius, ~5-20k nodes)
  - london_walk_city.graphml    (10 km radius, ~100-400k nodes)

Edge attribute "cost" is walking time in seconds at 5 km/h (~1.389 m/s).
Node attributes "x" (longitude) and "y" (latitude) are preserved.
Edge attribute "length" (meters) is also preserved.

Walking networks are much denser than drive networks (includes footpaths,
pedestrian ways, service roads, etc.), so we use smaller radii.

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

# Walking speed: 5 km/h = 5000/3600 m/s ≈ 1.389 m/s
WALKING_SPEED_MS = 5000.0 / 3600.0

# Only london_walk_large (benchmark) and london_walk_1M (accessibility) are
# needed by the paper scripts. The others are available for experimentation;
# pass --all to fetch them.
NETWORKS_DEFAULT = [
    {"label": "london_walk_large", "dist": 5_000},
    {"label": "london_walk_1M", "dist": 22_000, "simplify": False},
]

NETWORKS_ALL = [
    {"label": "london_walk_tiny", "dist": 500},
    {"label": "london_walk_small", "dist": 1_000},
    {"label": "london_walk_medium", "dist": 2_000},
    {"label": "london_walk_large", "dist": 5_000},
    {"label": "london_walk_city", "dist": 10_000},
    {"label": "london_walk_1M", "dist": 22_000, "simplify": False},
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_and_process(label: str, dist: int, simplify: bool = True) -> nx.Graph:
    """Download, clean, and return a simple undirected walking graph.

    Steps:
      1. Download walk network from OSMnx
      2. Convert to undirected
      3. Extract largest connected component
      4. Simplify MultiGraph → Graph (keep shortest-length parallel edge)
      5. Set "cost" = length / walking_speed (seconds), keep "length" (meters)
      6. Relabel nodes to 0..n-1, preserving x/y coordinates
    """
    print(f"\n{'=' * 60}")
    print(f"  Fetching {label} (radius={dist}m, walk network)")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()

    # 1. Download walking network
    G_directed = ox.graph.graph_from_point(CENTER, dist=dist, network_type="walk", simplify=simplify)
    print(
        f"  Downloaded: {G_directed.number_of_nodes()} nodes, {G_directed.number_of_edges()} edges"
    )

    # 2. Convert to undirected
    G_multi = ox.convert.to_undirected(G_directed)
    print(f"  Undirected: {G_multi.number_of_nodes()} nodes, {G_multi.number_of_edges()} edges")

    # 3. Largest connected component
    largest_cc = max(nx.connected_components(G_multi), key=len)
    G_multi = G_multi.subgraph(largest_cc).copy()
    print(f"  Largest CC: {G_multi.number_of_nodes()} nodes, {G_multi.number_of_edges()} edges")

    # 4. MultiGraph → simple Graph (keep shortest-length parallel edge)
    G = nx.Graph()
    for node, data in G_multi.nodes(data=True):
        G.add_node(node, **data)

    for u, v, key, data in G_multi.edges(keys=True, data=True):
        length = data.get("length", float("inf"))
        if G.has_edge(u, v):
            if length < G[u][v].get("length", float("inf")):
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)

    # 5. Set "cost" = walking time in seconds; strip non-serializable attributes
    #    Keep only: nodes(x, y), edges(cost, length)
    for node, data in G.nodes(data=True):
        keep = {"x": float(data["x"]), "y": float(data["y"])}
        data.clear()
        data.update(keep)

    for u, v, data in G.edges(data=True):
        length_m = float(data.get("length", 0.0))
        walking_time_s = length_m / WALKING_SPEED_MS
        keep = {
            "cost": walking_time_s,
            "length": length_m,
        }
        data.clear()
        data.update(keep)

    # 6. Relabel to 0..n-1
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
    assert "length" in edge_data, f"{label}: missing 'length' edge attribute"

    # Edge cost statistics (walking time in seconds)
    costs = [d["cost"] for _, _, d in G.edges(data=True)]
    lengths = [d["length"] for _, _, d in G.edges(data=True)]
    mean_cost = sum(costs) / len(costs)
    min_cost = min(costs)
    max_cost = max(costs)

    print(f"\n  {label} verification:")
    print(f"    Nodes: {n:,}")
    print(f"    Edges: {m:,}")
    print("    Connected: True")
    print("    Edge cost (walking time, seconds):")
    print(f"      min={min_cost:.1f}  mean={mean_cost:.1f}  max={max_cost:.1f}")
    print("    Edge length (meters):")
    mean_len = sum(lengths) / len(lengths)
    print(f"      min={min(lengths):.1f}  mean={mean_len:.1f}  max={max(lengths):.1f}")
    print(f"    Walking speed: {WALKING_SPEED_MS:.3f} m/s = {WALKING_SPEED_MS * 3.6:.1f} km/h")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download London walking networks")
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch all network sizes (tiny through 1M). Default: large + 1M only.",
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
            # Still verify the existing file
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
            G = fetch_and_process(label, dist, simplify=spec.get("simplify", True))
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
