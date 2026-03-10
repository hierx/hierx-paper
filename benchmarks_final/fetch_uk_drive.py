#!/usr/bin/env python3
"""Download the Great Britain driving network from a Geofabrik PBF via pyrosm.

Extracts the driving network from great-britain-latest.osm.pbf, simplifies it
with OSMnx, converts to an undirected graph with travel-time edge costs, and
saves as pickle.

Edge attribute "cost" is free-flow travel time in seconds.
Node attributes "x" (longitude) and "y" (latitude) are preserved.

Requirements (all available in the pyrosm micromamba env):
    pyrosm, osmnx, networkx, scipy

Usage:
    # Run in pyrosm env:
    MAMBA_ROOT_PREFIX=$HOME/.mamba /tmp/bin/micromamba run -n pyrosm \
        python -m benchmarks_final.fetch_uk_drive

    # Custom output path:
    ... python -m benchmarks_final.fetch_uk_drive --output /path/to/graph.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import networkx as nx

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uk_drive"
LOG_FILE = DATA_DIR / "fetch_uk_drive.log"

GEOFABRIK_URL = "https://download.geofabrik.de/europe/great-britain-latest.osm.pbf"

# Highway types that form the driveable network (matches pyrosm driving filter)
DRIVING_HIGHWAY_TYPES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "unclassified",
    "living_street",
]


def ensure_pbf(data_dir: Path) -> Path:
    """Ensure the Geofabrik PBF is on disk, downloading if needed."""
    pbf_path = data_dir / "great-britain-latest.osm.pbf"
    if pbf_path.exists():
        size_gb = pbf_path.stat().st_size / 1e9
        log.info(f"PBF already on disk: {pbf_path} ({size_gb:.1f} GB)")
        return pbf_path

    log.info(f"Downloading {GEOFABRIK_URL} ...")
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(GEOFABRIK_URL, pbf_path)
    size_gb = pbf_path.stat().st_size / 1e9
    log.info(f"Downloaded to {pbf_path} ({size_gb:.1f} GB)")
    return pbf_path


def filter_pbf_driving(full_pbf: Path, data_dir: Path) -> Path:
    """Use osmium to pre-filter the PBF to driving-only highway types.

    This reduces the file from ~2 GB to ~220 MB, making pyrosm extraction
    feasible in memory at national scale.
    """
    import subprocess

    filtered = data_dir / "gb_driving_only.osm.pbf"
    if filtered.exists():
        size_mb = filtered.stat().st_size / 1e6
        log.info(f"Filtered PBF already on disk: {filtered} ({size_mb:.0f} MB)")
        return filtered

    highway_filter = ",".join(DRIVING_HIGHWAY_TYPES)
    cmd = [
        "osmium", "tags-filter", str(full_pbf),
        f"w/highway={highway_filter}",
        "-o", str(filtered), "--overwrite",
    ]
    log.info(f"Filtering PBF to driving highways with osmium ...")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    size_mb = filtered.stat().st_size / 1e6
    log.info(f"  Filtered in {dt:.0f}s: {size_mb:.0f} MB")
    return filtered


def extract_network_from_pbf(pbf_path: Path) -> nx.MultiDiGraph:
    """Extract driving network from a (pre-filtered) PBF via pyrosm.

    Builds an OSMnx-compatible MultiDiGraph manually to control memory usage.
    Only the attributes needed for OSMnx simplification are kept.
    """
    import warnings

    import pyrosm

    log.info("Extracting driving network with pyrosm ...")
    t0 = time.perf_counter()
    osm = pyrosm.OSM(str(pbf_path))
    # Suppress pandas Copy-on-Write warnings from pyrosm internals
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chained assignment.*")
        nodes_gdf, edges_gdf = osm.get_network(network_type="driving", nodes=True)
    dt = time.perf_counter() - t0
    log.info(f"  Extracted in {dt:.0f}s: {len(nodes_gdf):,} nodes, {len(edges_gdf):,} edges")

    # Build graph manually with minimal attributes to save memory.
    # OSMnx simplify_graph needs: node x/y/osmid, edge geometry/highway/length,
    # and the graph must be a MultiDiGraph with crs.
    log.info("Building NetworkX graph (lean) ...")
    t0 = time.perf_counter()

    G = nx.MultiDiGraph(crs="EPSG:4326")

    # Add nodes — only x, y, osmid
    node_ids = nodes_gdf["id"].values
    node_lons = nodes_gdf["lon"].values
    node_lats = nodes_gdf["lat"].values
    for i in range(len(node_ids)):
        nid = int(node_ids[i])
        G.add_node(nid, x=float(node_lons[i]), y=float(node_lats[i]), osmid=nid)

    log.info(f"  Added {G.number_of_nodes():,} nodes in {time.perf_counter() - t0:.0f}s")

    # Add edges — keep highway, maxspeed, oneway, length, geometry, osmid
    t1 = time.perf_counter()
    us = edges_gdf["u"].values
    vs = edges_gdf["v"].values
    lengths = edges_gdf["length"].values
    highways = edges_gdf["highway"].values
    maxspeeds = edges_gdf["maxspeed"].values if "maxspeed" in edges_gdf.columns else [None] * len(edges_gdf)
    oneways = edges_gdf["oneway"].values if "oneway" in edges_gdf.columns else [None] * len(edges_gdf)
    osmids = edges_gdf["id"].values
    geometries = edges_gdf["geometry"].values

    node_set = set(G.nodes())
    n_edges = 0
    n_skipped = 0
    for i in range(len(us)):
        u, v = int(us[i]), int(vs[i])
        if u not in node_set or v not in node_set:
            n_skipped += 1
            continue

        edata = {
            "osmid": int(osmids[i]),
            "length": float(lengths[i]),
            "highway": highways[i],
            "geometry": geometries[i],
        }
        ms = maxspeeds[i] if not isinstance(maxspeeds, list) else maxspeeds[i]
        if ms is not None and str(ms) != "nan" and str(ms) != "None":
            edata["maxspeed"] = ms
        ow = oneways[i] if not isinstance(oneways, list) else oneways[i]
        if ow is not None and str(ow) != "nan" and str(ow) != "None":
            edata["oneway"] = ow

        G.add_edge(u, v, **edata)
        n_edges += 1

        # For driving: add reverse edge unless oneway
        is_oneway = str(ow).lower() in ("yes", "true", "1", "-1")
        is_reverse = str(ow).strip() == "-1"
        if is_reverse:
            # -1 means oneway in reverse direction: remove forward, add reverse
            G.remove_edge(u, v, list(G[u][v].keys())[-1])
            G.add_edge(v, u, **edata)
        elif not is_oneway:
            G.add_edge(v, u, **edata)
            n_edges += 1

    dt = time.perf_counter() - t1
    log.info(f"  Added {n_edges:,} directed edges in {dt:.0f}s (skipped {n_skipped:,} orphans)")

    del nodes_gdf, edges_gdf, us, vs, lengths, highways, maxspeeds, oneways, osmids, geometries
    dt_total = time.perf_counter() - t0
    log.info(
        f"  Graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges in {dt_total:.0f}s"
    )
    return G


def process_graph(G_raw: nx.MultiDiGraph) -> nx.Graph:
    """Process raw graph: simplify, add speeds, undirect, LCC, clean, relabel.

    Steps:
      1. Topological simplification (OSMnx consolidation of unsimplified graph)
      2. Add free-flow speeds and travel times
      3. Strip 'key' attr from edge data (pyrosm/osmnx compat fix)
      4. Convert to undirected
      5. Extract largest connected component
      6. MultiGraph -> simple Graph (keep shortest travel_time)
      7. Set "cost" = travel_time, keep "length"
      8. Relabel nodes 0..n-1
    """
    import osmnx as ox

    # Step 1: Topological simplification (merge intermediate nodes)
    log.info("Step 1/8: Topological simplification ...")
    t0 = time.perf_counter()
    n_before = G_raw.number_of_nodes()
    G_raw = ox.simplification.simplify_graph(G_raw)
    dt = time.perf_counter() - t0
    log.info(
        f"  Simplified: {n_before:,} -> {G_raw.number_of_nodes():,} nodes, "
        f"{G_raw.number_of_edges():,} edges in {dt:.0f}s"
    )

    # Step 2: Add speeds and travel times
    log.info("Step 2/8: Adding speeds and travel times ...")
    t0 = time.perf_counter()
    G_raw = ox.routing.add_edge_speeds(G_raw)
    G_raw = ox.routing.add_edge_travel_times(G_raw)
    dt = time.perf_counter() - t0
    log.info(f"  Done in {dt:.0f}s")

    # Step 3: Strip 'key' from edge data (pyrosm adds it, conflicts with ox.convert)
    log.info("Step 3/8: Stripping 'key' attr from edge data ...")
    for _u, _v, _k, data in G_raw.edges(keys=True, data=True):
        data.pop("key", None)

    # Step 4: Convert to undirected
    log.info("Step 4/8: Converting to undirected ...")
    t0 = time.perf_counter()
    G_multi = ox.convert.to_undirected(G_raw)
    del G_raw
    dt = time.perf_counter() - t0
    log.info(
        f"  Undirected: {G_multi.number_of_nodes():,} nodes, "
        f"{G_multi.number_of_edges():,} edges in {dt:.0f}s"
    )

    # Step 5: Largest connected component
    log.info("Step 5/8: Extracting largest connected component ...")
    t0 = time.perf_counter()
    largest_cc = max(nx.connected_components(G_multi), key=len)
    n_before = G_multi.number_of_nodes()
    G_multi = G_multi.subgraph(largest_cc).copy()
    dt = time.perf_counter() - t0
    n_dropped = n_before - G_multi.number_of_nodes()
    pct = 100 * G_multi.number_of_nodes() / n_before
    log.info(
        f"  LCC: {G_multi.number_of_nodes():,} nodes ({pct:.1f}%), "
        f"dropped {n_dropped:,} in {dt:.0f}s"
    )

    # Step 6: MultiGraph -> simple Graph (keep shortest travel_time)
    log.info("Step 6/8: Simplifying to simple graph ...")
    t0 = time.perf_counter()
    G = nx.Graph()
    for node, data in G_multi.nodes(data=True):
        G.add_node(node, **data)

    for u, v, _key, data in G_multi.edges(keys=True, data=True):
        tt = data.get("travel_time", float("inf"))
        if G.has_edge(u, v):
            if tt < G[u][v].get("travel_time", float("inf")):
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)
    del G_multi
    dt = time.perf_counter() - t0
    log.info(
        f"  Simple graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges in {dt:.0f}s"
    )

    # Step 6b: Remove self-loops
    n_selfloops = nx.number_of_selfloops(G)
    if n_selfloops:
        G.remove_edges_from(nx.selfloop_edges(G))
        log.info(f"  Removed {n_selfloops:,} self-loops")

    # Step 7: Clean attributes
    log.info("Step 7/8: Cleaning attributes ...")
    t0 = time.perf_counter()
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
    dt = time.perf_counter() - t0
    log.info(f"  Done in {dt:.0f}s")

    # Step 8: Relabel to 0..n-1
    log.info("Step 8/8: Relabeling nodes ...")
    t0 = time.perf_counter()
    old_to_new = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, old_to_new)
    dt = time.perf_counter() - t0
    log.info(f"  Done in {dt:.0f}s")

    return G


def verify_graph(G: nx.Graph) -> None:
    """Verify the graph meets expected properties for GB driving network."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    assert nx.is_connected(G), "Graph is not connected"

    sample_node = next(iter(G.nodes()))
    assert "x" in G.nodes[sample_node], "Missing 'x' node attribute"
    assert "y" in G.nodes[sample_node], "Missing 'y' node attribute"

    sample_edge = next(iter(G.edges()))
    assert "cost" in G.edges[sample_edge], "Missing 'cost' edge attribute"

    costs = [d["cost"] for _, _, d in G.edges(data=True)]
    mean_cost = sum(costs) / len(costs)

    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    lon_min, lon_max = min(xs), max(xs)
    lat_min, lat_max = min(ys), max(ys)

    log.info("Verification:")
    log.info(f"  Nodes: {n:,}")
    log.info(f"  Edges: {m:,}")
    log.info(f"  Connected: True")
    log.info(f"  Edge cost: min={min(costs):.1f} mean={mean_cost:.1f} max={max(costs):.1f}")
    log.info(f"  Longitude: [{lon_min:.2f}, {lon_max:.2f}]")
    log.info(f"  Latitude:  [{lat_min:.2f}, {lat_max:.2f}]")

    assert mean_cost > 1.0, f"Mean edge cost too low ({mean_cost:.1f}s), expected driving times"
    assert mean_cost < 200.0, f"Mean edge cost too high ({mean_cost:.1f}s)"
    assert lon_min < -1.0, f"Longitude min too high ({lon_min:.2f}), expected western GB"
    assert lat_max > 55.0, f"Latitude max too low ({lat_max:.2f}), expected Scotland"
    assert lat_min < 51.5, f"Latitude min too high ({lat_min:.2f}), expected south England"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract GB driving network from Geofabrik PBF via pyrosm"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output pickle path (default: data/uk_drive/gb_drive.pkl)",
    )
    parser.add_argument(
        "--pbf", type=str, default=None,
        help="Use an existing PBF file instead of downloading from Geofabrik",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip GB-specific assertions in verify_graph()",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="w"),
        ],
    )

    out_path = Path(args.output) if args.output else DATA_DIR / "gb_drive.pkl"

    # Fail fast if pyrosm is not installed (before downloading ~2 GB PBF)
    try:
        import pyrosm  # noqa: F401
    except ImportError:
        log.error(
            "pyrosm is required but not installed.\n"
            "  Install in a conda environment:\n"
            "    conda create -n pyrosm python=3.12 pyrosm osmium-tool -c conda-forge\n"
            "    conda activate pyrosm\n"
            "    pip install -r requirements.txt\n"
            "    python -m benchmarks_final.fetch_uk_drive"
        )
        sys.exit(1)

    t_total = time.perf_counter()

    # 1. Ensure PBF on disk
    if args.pbf:
        filtered_pbf = Path(args.pbf)
        if not filtered_pbf.exists():
            log.error(f"PBF not found: {filtered_pbf}")
            sys.exit(1)
        log.info(f"Using provided PBF: {filtered_pbf}")
    else:
        pbf_path = ensure_pbf(DATA_DIR)
        # 2. Pre-filter to driving highways (reduces ~2GB → ~220MB)
        filtered_pbf = filter_pbf_driving(pbf_path, DATA_DIR)

    # 3. Extract network
    G_raw = extract_network_from_pbf(filtered_pbf)

    # 3. Process: simplify, add speeds, undirect, LCC, clean
    log.info("\nProcessing graph ...")
    G = process_graph(G_raw)
    del G_raw

    # 4. Verify
    if args.skip_verify:
        log.info("\nSkipping verification (--skip-verify)")
    else:
        log.info("\nVerifying graph ...")
        verify_graph(G)

    # 5. Save as pickle
    log.info(f"\nSaving to {out_path} ...")
    t0 = time.perf_counter()
    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    dt = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    log.info(f"  Saved in {dt:.1f}s ({size_mb:.0f} MB)")

    total = time.perf_counter() - t_total
    log.info(f"\nTotal time: {total:.0f}s ({total / 60:.1f} min)")
    log.info(f"Output: {out_path}")
    log.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")


if __name__ == "__main__":
    main()
