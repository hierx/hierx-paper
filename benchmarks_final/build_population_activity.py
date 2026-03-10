#!/usr/bin/env python3
"""Disaggregate Census population & workplace from OAs to OSM buildings to network nodes.

Pipeline:
  1. Download OA boundary polygons from ONS Open Geography Portal
  2. Download OSM building footprints (~22km radius from central London)
  3. Compute floor area per building (footprint × levels)
  4. Spatial join — assign buildings to Output Areas
  5. Distribute OA population & workplace proportionally by floor area
  6. Snap buildings to nearest network edge, then split to endpoint nodes
  7. Save node-level activity arrays + maps

Usage:
    python -m benchmarks_final.build_population_activity
    python -m benchmarks_final.build_population_activity --no-plot         # skip map generation
    python -m benchmarks_final.build_population_activity --no-plot         # skip maps
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---- Paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CENSUS_DIR = Path(__file__).resolve().parent.parent / "data" / "london_census"
PAPER_FIG_DIR = SCRIPT_DIR.parent / "paper_figs_final" / "figures"

LONDON_CENTER = (51.5074, -0.1278)  # lat, lon
BNG_CRS = "EPSG:27700"  # British National Grid

# Default levels by building type when tag is missing
DEFAULT_LEVELS = {
    "residential": 2,
    "apartments": 4,
    "house": 2,
    "detached": 2,
    "semi": 2,
    "semidetached_house": 2,
    "terrace": 2,
    "commercial": 2,
    "office": 3,
    "retail": 1,
    "industrial": 1,
    "warehouse": 1,
    "garage": 1,
    "garages": 1,
    "shed": 1,
    "roof": 1,
    "church": 1,
    "cathedral": 1,
    "school": 2,
    "university": 3,
    "hospital": 3,
    "hotel": 4,
}
FALLBACK_LEVELS = 2  # for 'yes' and unrecognised types


# =============================================================================
# Step 1: Download OA boundaries
# =============================================================================


def download_oa_boundaries(out_path: Path) -> gpd.GeoDataFrame:
    """Download Census 2021 OA boundaries from ONS Open Geography Portal.

    Uses the ArcGIS FeatureServer API to page through all London OAs.
    """
    import requests

    # ONS OA 2021 boundaries (generalised, clipped to coastline)
    BASE_URL = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Output_Areas_2021_EW_BGC_V2/FeatureServer/0/query"
    )

    # Load our OA list to filter
    oa_csv = CENSUS_DIR / "london_oa_combined.csv"
    if not oa_csv.exists():
        raise FileNotFoundError(
            f"Census data not found: {oa_csv}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_census_data\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
    oa_df = pd.read_csv(oa_csv, usecols=["oa21cd"])
    oa_codes = set(oa_df["oa21cd"].tolist())
    log.info(f"  Need boundaries for {len(oa_codes):,} London OAs")

    all_features: list[dict] = []
    batch_size = 2000
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "outFields": "OA21CD",
            "outSR": 4326,
            "f": "geojson",
            "resultRecordCount": batch_size,
            "resultOffset": offset,
        }
        log.info(f"  Fetching OA boundaries batch at offset {offset} ...")
        resp = requests.get(BASE_URL, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        # Filter to London OAs only
        london_features = [f for f in features if f["properties"].get("OA21CD") in oa_codes]
        all_features.extend(london_features)
        log.info(
            f"    Got {len(features)} features, {len(london_features)} London "
            f"(total: {len(all_features)})"
        )

        if len(features) < batch_size:
            break
        offset += batch_size

    log.info(f"  Downloaded {len(all_features):,} OA boundaries")

    # Build GeoDataFrame
    geojson = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
    gdf = gdf.rename(columns={"OA21CD": "oa21cd"})

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    log.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return gdf


# =============================================================================
# Step 2: Download OSM building footprints
# =============================================================================


def download_buildings(out_path: Path, radius: int = 22000) -> gpd.GeoDataFrame:
    """Download OSM building footprints using OSMnx."""
    import osmnx as ox

    ox.settings.requests_timeout = 600
    ox.settings.overpass_memory = 4_000_000_000
    ox.settings.use_cache = True
    ox.settings.log_console = True

    log.info(f"  Downloading buildings within {radius / 1000:.0f}km of central London ...")
    t0 = time.perf_counter()
    gdf = ox.features_from_point(LONDON_CENTER, tags={"building": True}, dist=radius)
    dt = time.perf_counter() - t0
    log.info(f"  Downloaded {len(gdf):,} features in {dt:.1f}s")

    # Keep only polygon/multipolygon geometries (skip nodes tagged as building)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    log.info(f"  {len(gdf):,} polygon buildings after filtering")

    # Keep useful columns
    keep = ["geometry"]
    for col in ["building", "building:levels", "height", "name"]:
        if col in gdf.columns:
            keep.append(col)
    gdf = gdf[keep].reset_index(drop=True)

    # Save as GeoPackage (fast, compact)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    log.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return gdf


# =============================================================================
# Step 3: Compute floor area
# =============================================================================


def compute_floor_area(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add footprint_area, levels, and floor_area columns."""
    log.info("  Computing floor areas ...")

    # Reproject to BNG for accurate area computation
    buildings_bng = buildings.to_crs(BNG_CRS)
    buildings["footprint_area_m2"] = buildings_bng.geometry.area

    # Parse levels
    level_col = "building:levels"
    if level_col in buildings.columns:
        buildings["levels"] = pd.to_numeric(buildings[level_col], errors="coerce")
    else:
        buildings["levels"] = np.nan

    # Impute missing levels from building type
    btype_col = "building" if "building" in buildings.columns else None
    mask_missing = buildings["levels"].isna()

    if btype_col is not None:
        buildings.loc[mask_missing, "levels"] = buildings.loc[mask_missing, btype_col].map(
            DEFAULT_LEVELS
        )
    # Fill remaining NaN with fallback
    buildings["levels"] = buildings["levels"].fillna(FALLBACK_LEVELS).clip(lower=1)

    buildings["floor_area_m2"] = buildings["footprint_area_m2"] * buildings["levels"]

    log.info(f"  Footprint area: median={buildings['footprint_area_m2'].median():.0f} m²")
    log.info(
        f"  Levels: median={buildings['levels'].median():.0f}, "
        f"mean={buildings['levels'].mean():.1f}"
    )
    log.info(f"  Floor area: total={buildings['floor_area_m2'].sum() / 1e6:.1f} M m²")

    return buildings


# =============================================================================
# Step 4: Spatial join — buildings to OAs
# =============================================================================


def assign_buildings_to_oas(
    buildings: gpd.GeoDataFrame,
    oa_boundaries: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatial join: assign each building to an OA by centroid containment."""
    log.info("  Spatial join: buildings → OAs ...")

    # Use building centroids for the join (robust for boundary cases)
    # Project to BNG before computing centroids to avoid geographic CRS errors
    buildings = buildings.copy()
    buildings["_centroid"] = buildings.to_crs(BNG_CRS).geometry.centroid.to_crs(buildings.crs)
    buildings_pts = buildings.set_geometry("_centroid")

    joined = gpd.sjoin(
        buildings_pts,
        oa_boundaries[["oa21cd", "geometry"]],
        how="left",
        predicate="within",
    )
    # Drop duplicates from multi-boundary overlap (keep first match)
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.set_geometry("geometry").drop(columns=["_centroid", "index_right"])

    n_matched = joined["oa21cd"].notna().sum()
    n_total = len(joined)
    log.info(
        f"  Matched {n_matched:,}/{n_total:,} buildings to OAs ({100 * n_matched / n_total:.1f}%)"
    )

    return joined


# =============================================================================
# Step 5: Distribute population & workplace by floor area
# =============================================================================


def distribute_population(
    buildings: gpd.GeoDataFrame,
    census: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Distribute OA-level population & workplace to buildings by floor area share."""
    log.info("  Distributing population & workplace to buildings ...")

    buildings = buildings.copy()

    # Total floor area per OA
    oa_floor_area = (
        buildings[buildings["oa21cd"].notna()].groupby("oa21cd")["floor_area_m2"].transform("sum")
    )

    # Fraction of OA floor area in each building
    mask = buildings["oa21cd"].notna()
    buildings["floor_area_share"] = 0.0
    buildings.loc[mask, "floor_area_share"] = buildings.loc[mask, "floor_area_m2"] / oa_floor_area

    # Merge census data
    buildings = buildings.merge(
        census[["oa21cd", "population", "workplace_population"]],
        on="oa21cd",
        how="left",
    )

    # Distribute
    buildings["pop"] = buildings["floor_area_share"] * buildings["population"].fillna(0)
    buildings["wp"] = buildings["floor_area_share"] * buildings["workplace_population"].fillna(0)

    # Clean up
    buildings = buildings.drop(columns=["population", "workplace_population"], errors="ignore")

    total_pop = buildings["pop"].sum()
    total_wp = buildings["wp"].sum()
    log.info(f"  Distributed pop: {total_pop:,.0f}, workplace: {total_wp:,.0f}")

    return buildings


# =============================================================================
# Step 6: Snap buildings to network edges → node-level activity
# =============================================================================


def point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Squared distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def snap_buildings_to_edges(
    buildings: gpd.GeoDataFrame,
    G,  # networkx graph
    zones: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Snap building centroids to nearest network edge, return node-level pop & wp.

    For each building, finds the nearest edge and splits the building's pop/wp
    equally between the two endpoint nodes.

    Returns (node_pop, node_wp) arrays indexed by zone order.
    """

    log.info("  Snapping buildings to network edges ...")

    n_zones = len(zones)
    zone_to_idx = {z: i for i, z in enumerate(zones)}

    # Node coordinates in BNG for distance computation
    node_xy_bng = np.array([[G.nodes[z]["x_bng"], G.nodes[z]["y_bng"]] for z in zones])

    # Edge data: midpoints for KDTree, endpoints for precise distance
    edges = list(G.edges())
    edge_midpoints = np.empty((len(edges), 2))
    edge_endpoints = np.empty((len(edges), 4))  # ax, ay, bx, by

    for i, (u, v) in enumerate(edges):
        ui, vi = zone_to_idx[u], zone_to_idx[v]
        ax, ay = node_xy_bng[ui]
        bx, by = node_xy_bng[vi]
        edge_midpoints[i] = [(ax + bx) / 2, (ay + by) / 2]
        edge_endpoints[i] = [ax, ay, bx, by]

    midpoint_tree = cKDTree(edge_midpoints)

    # Building centroids in BNG (project before centroid to avoid geographic CRS error)
    bld_bng = buildings.to_crs(BNG_CRS).geometry.centroid
    bld_xy = np.column_stack([bld_bng.x, bld_bng.y])

    bld_pop = buildings["pop"].values
    bld_wp = buildings["wp"].values

    node_pop = np.zeros(n_zones, dtype=np.float64)
    node_wp = np.zeros(n_zones, dtype=np.float64)

    # For each building, find nearest edge using KDTree candidate then refine
    K_CANDIDATES = 8  # check top-K midpoint neighbours
    _, candidate_idxs = midpoint_tree.query(bld_xy, k=K_CANDIDATES)

    n_buildings = len(bld_xy)
    log.info(f"  Processing {n_buildings:,} buildings ...")
    t0 = time.perf_counter()

    for i in range(n_buildings):
        px, py = bld_xy[i]
        p = bld_pop[i]
        w = bld_wp[i]

        if p == 0.0 and w == 0.0:
            continue

        # Find nearest edge among candidates
        best_dist = np.inf
        best_edge_idx = candidate_idxs[i, 0]
        for j in range(K_CANDIDATES):
            eidx = candidate_idxs[i, j]
            ax, ay, bx, by = edge_endpoints[eidx]
            d = point_to_segment_distance(px, py, ax, ay, bx, by)
            if d < best_dist:
                best_dist = d
                best_edge_idx = eidx

        # Split equally between the two endpoint nodes
        u, v = edges[best_edge_idx]
        ui, vi = zone_to_idx[u], zone_to_idx[v]
        node_pop[ui] += p / 2
        node_pop[vi] += p / 2
        node_wp[ui] += w / 2
        node_wp[vi] += w / 2

    dt = time.perf_counter() - t0
    log.info(f"  Snapping done in {dt:.1f}s")
    log.info(f"  Node pop total: {node_pop.sum():,.0f}, node wp total: {node_wp.sum():,.0f}")

    return node_pop, node_wp


# =============================================================================
# Step 7: Maps
# =============================================================================


def plot_distribution_maps(
    buildings: gpd.GeoDataFrame,
    node_xy: np.ndarray,
    node_pop: np.ndarray,
    node_wp: np.ndarray,
    out_dir: Path,
) -> None:
    """2×2 panel: (a) building pop density, (b) building wp density,
    (c) pop per node, (d) wp per node."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.dpi": 150,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    aspect = 1.0 / math.cos(math.radians(51.5))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # Building-level panels: use centroids (geographic CRS is fine for plotting)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*centroid.*")
        bld_x = buildings.geometry.centroid.x
        bld_y = buildings.geometry.centroid.y

    # (a) Building population
    ax = axes[0, 0]
    mask = buildings["pop"] > 0
    sc = ax.scatter(
        bld_x[mask],
        bld_y[mask],
        c=buildings.loc[mask, "pop"],
        s=0.1,
        cmap="YlOrRd",
        norm=LogNorm(vmin=0.01, vmax=buildings["pop"].max()),
        rasterized=True,
    )
    ax.set_aspect(aspect)
    ax.set_title(f"(a) Building population (n={mask.sum():,})")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Population")

    # (b) Building workplace
    ax = axes[0, 1]
    mask = buildings["wp"] > 0
    sc = ax.scatter(
        bld_x[mask],
        bld_y[mask],
        c=buildings.loc[mask, "wp"],
        s=0.1,
        cmap="YlGnBu",
        norm=LogNorm(vmin=0.01, vmax=max(buildings["wp"].max(), 0.02)),
        rasterized=True,
    )
    ax.set_aspect(aspect)
    ax.set_title(f"(b) Building workplace pop (n={mask.sum():,})")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Workplace pop")

    # (c) Node population
    ax = axes[1, 0]
    mask_n = node_pop > 0
    sc = ax.scatter(
        node_xy[mask_n, 0],
        node_xy[mask_n, 1],
        c=node_pop[mask_n],
        s=0.2,
        cmap="YlOrRd",
        norm=LogNorm(vmin=node_pop[mask_n].min(), vmax=node_pop[mask_n].max()),
        rasterized=True,
    )
    ax.set_aspect(aspect)
    ax.set_title(f"(c) Network node population (n={mask_n.sum():,})")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Population")

    # (d) Node workplace
    ax = axes[1, 1]
    mask_n = node_wp > 0
    sc = ax.scatter(
        node_xy[mask_n, 0],
        node_xy[mask_n, 1],
        c=node_wp[mask_n],
        s=0.2,
        cmap="YlGnBu",
        norm=LogNorm(vmin=node_wp[mask_n].min(), vmax=node_wp[mask_n].max()),
        rasterized=True,
    )
    ax.set_aspect(aspect)
    ax.set_title(f"(d) Network node workplace pop (n={mask_n.sum():,})")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Workplace pop")

    fig.suptitle(
        "Population & workplace disaggregation: Census OA → OSM buildings → network nodes",
        fontsize=12,
    )

    fig.savefig(out_dir / "population_disaggregation.png", dpi=200)
    fig.savefig(out_dir / "population_disaggregation.pdf")
    plt.close(fig)
    log.info(f"  Maps saved to {out_dir / 'population_disaggregation.png'}")


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disaggregate Census population to buildings and network nodes"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="london_walk_1M.graphml",
        help="Network file in data/ (default: london_walk_1M.graphml)",
    )
    parser.add_argument(
        "--building-radius",
        type=int,
        default=22000,
        help="OSM building download radius in meters (default: 22000)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip map generation",
    )
    args = parser.parse_args()

    import networkx as nx

    oa_boundary_path = CENSUS_DIR / "london_oa_boundaries.geojson"
    buildings_path = CENSUS_DIR / "london_buildings.gpkg"
    census_path = CENSUS_DIR / "london_oa_combined.csv"

    if not census_path.exists():
        log.error(
            f"Census data not found: {census_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_census_data\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
        sys.exit(1)

    # ---- Step 1: OA boundaries ----
    log.info("=" * 60)
    log.info("STEP 1: OA boundaries")
    if oa_boundary_path.exists():
        log.info(f"  Loading cached {oa_boundary_path}")
        oa_boundaries = gpd.read_file(oa_boundary_path)
    else:
        oa_boundaries = download_oa_boundaries(oa_boundary_path)
    log.info(f"  {len(oa_boundaries):,} OA boundaries loaded")

    # ---- Step 2: OSM buildings ----
    log.info("=" * 60)
    log.info("STEP 2: OSM buildings")
    if buildings_path.exists():
        log.info(f"  Loading cached {buildings_path}")
        buildings = gpd.read_file(buildings_path)
    else:
        buildings = download_buildings(buildings_path, radius=args.building_radius)
    log.info(f"  {len(buildings):,} buildings loaded")

    # ---- Step 3: Floor area ----
    log.info("=" * 60)
    log.info("STEP 3: Floor area computation")
    buildings = compute_floor_area(buildings)

    # ---- Step 4: Spatial join ----
    log.info("=" * 60)
    log.info("STEP 4: Spatial join (buildings → OAs)")
    buildings = assign_buildings_to_oas(buildings, oa_boundaries)

    # ---- Step 5: Distribute population ----
    log.info("=" * 60)
    log.info("STEP 5: Distribute population & workplace to buildings")
    census = pd.read_csv(census_path)
    buildings = distribute_population(buildings, census)

    # Verification
    census_pop_total = census["population"].sum()
    census_wp_total = census["workplace_population"].sum()
    bld_pop_total = buildings["pop"].sum()
    bld_wp_total = buildings["wp"].sum()
    log.info(f"  Census totals:   pop={census_pop_total:,}, wp={census_wp_total:,}")
    log.info(f"  Building totals: pop={bld_pop_total:,.0f}, wp={bld_wp_total:,.0f}")
    log.info(
        f"  Coverage: pop={100 * bld_pop_total / census_pop_total:.1f}%, "
        f"wp={100 * bld_wp_total / census_wp_total:.1f}%"
    )
    assert buildings["pop"].min() >= 0, "Negative population found!"
    assert buildings["wp"].min() >= 0, "Negative workplace found!"

    # Save attributed buildings
    bld_out_path = CENSUS_DIR / "london_buildings_attributed.gpkg"
    buildings.to_file(bld_out_path, driver="GPKG")
    log.info(f"  Saved attributed buildings to {bld_out_path}")

    # ---- Step 6: Snap to network ----
    log.info("=" * 60)
    log.info("STEP 6: Load network & snap buildings to edges")
    net_path = DATA_DIR / args.network
    if not net_path.exists():
        log.error(
            f"Network not found: {net_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_london_walking\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
        sys.exit(1)

    log.info(f"  Loading {net_path} ...")
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
    node_xy = np.array([[G.nodes[z]["x"], G.nodes[z]["y"]] for z in zones])

    # Add BNG coordinates to graph for distance computation
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", BNG_CRS, always_xy=True)
    for z in zones:
        x_bng, y_bng = transformer.transform(G.nodes[z]["x"], G.nodes[z]["y"])
        G.nodes[z]["x_bng"] = x_bng
        G.nodes[z]["y_bng"] = y_bng

    # Filter buildings to those with pop/wp > 0 for snapping efficiency
    bld_active = buildings[(buildings["pop"] > 0) | (buildings["wp"] > 0)].copy()
    log.info(f"  {len(bld_active):,} buildings with pop/wp > 0 to snap")

    node_pop, node_wp = snap_buildings_to_edges(bld_active, G, zones)

    # Verification
    log.info(f"  Node pop total: {node_pop.sum():,.0f} (building total: {bld_pop_total:,.0f})")
    log.info(f"  Node wp total:  {node_wp.sum():,.0f} (building total: {bld_wp_total:,.0f})")

    # ---- Step 7: Save outputs ----
    log.info("=" * 60)
    log.info("STEP 7: Save outputs")

    # Node-level activity arrays
    npz_path = CENSUS_DIR / "london_node_activity.npz"
    np.savez_compressed(
        npz_path,
        zones=np.array(zones, dtype=np.int64),
        node_xy=node_xy,
        node_pop=node_pop,
        node_wp=node_wp,
    )
    log.info(f"  Saved node activity to {npz_path}")

    # Maps
    if not args.no_plot:
        log.info("  Generating distribution maps ...")
        plot_distribution_maps(buildings, node_xy, node_pop, node_wp, PAPER_FIG_DIR)

    # ---- Summary ----
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info(f"  OA boundaries:  {len(oa_boundaries):,}")
    log.info(f"  OSM buildings:  {len(buildings):,}")
    log.info(f"  Buildings with OA match: {buildings['oa21cd'].notna().sum():,}")
    log.info(f"  Census pop total:      {census_pop_total:,}")
    log.info(f"  Building pop total:    {bld_pop_total:,.0f}")
    log.info(f"  Node pop total:        {node_pop.sum():,.0f}")
    log.info(f"  Census wp total:       {census_wp_total:,}")
    log.info(f"  Building wp total:     {bld_wp_total:,.0f}")
    log.info(f"  Node wp total:         {node_wp.sum():,.0f}")
    log.info(f"  Active nodes (pop>0):  {(node_pop > 0).sum():,}")
    log.info(f"  Active nodes (wp>0):   {(node_wp > 0).sum():,}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
