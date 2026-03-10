#!/usr/bin/env python3
"""Disaggregate Census population & workplace from OAs/Data Zones to OSM buildings,
then snap to GB driving network nodes.

Pipeline:
  1. Prepare Census data (England & Wales from Nomis, Scotland from NRS)
  2. Download OA/Data Zone boundaries from ONS and NRS
  3. Download OSM building footprints region-by-region
  4. Compute floor areas, spatial join, distribute Census data
  5. Snap buildings to nearest network edge → node-level activity
  6. Save outputs + maps

Usage:
    python -m benchmarks_final.build_uk_population_activity
    python -m benchmarks_final.build_uk_population_activity --skip-buildings
    python -m benchmarks_final.build_uk_population_activity --skip-buildings
    python -m benchmarks_final.build_uk_population_activity --no-plot
"""

from __future__ import annotations

import argparse
import logging
import pickle
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
DATA_DIR = SCRIPT_DIR.parent / "data" / "uk_drive"
CENSUS_DIR = SCRIPT_DIR.parent / "data" / "london_census"  # E&W census files
PAPER_FIG_DIR = SCRIPT_DIR.parent / "paper_figs_final" / "figures"

BNG_CRS = "EPSG:27700"  # British National Grid

GB_REGIONS = [
    "North East England",
    "North West England",
    "Yorkshire and the Humber",
    "East Midlands, England",
    "West Midlands, England",
    "East of England",
    "London",
    "South East England",
    "South West England",
    "Wales",
    "Scotland",
]

# Region centroids (lat, lon) for OSMnx building downloads
REGION_CENTERS: dict[str, tuple[float, float]] = {
    "North East England": (55.0, -1.6),
    "North West England": (53.8, -2.6),
    "Yorkshire and the Humber": (53.8, -1.3),
    "East Midlands, England": (52.8, -1.1),
    "West Midlands, England": (52.5, -1.9),
    "East of England": (52.2, 0.5),
    "London": (51.51, -0.13),
    "South East England": (51.3, -0.5),
    "South West England": (50.9, -3.0),
    "Wales": (52.1, -3.5),
    "Scotland": (56.5, -4.0),
}

# Default building levels by type
DEFAULT_LEVELS = {
    "residential": 2, "apartments": 4, "house": 2, "detached": 2,
    "semi": 2, "semidetached_house": 2, "terrace": 2,
    "commercial": 2, "office": 3, "retail": 1,
    "industrial": 1, "warehouse": 1, "garage": 1, "garages": 1,
    "shed": 1, "roof": 1, "church": 1, "cathedral": 1,
    "school": 2, "university": 3, "hospital": 3, "hotel": 4,
}
FALLBACK_LEVELS = 2


# =============================================================================
# Step 1: Prepare Census data
# =============================================================================


def prepare_ew_census() -> pd.DataFrame:
    """Load England & Wales Census 2021 population and workplace data.

    Uses the full E&W OA-level files from Nomis (already on disk).
    """
    log.info("  Loading England & Wales census data ...")

    # Population: census2021-ts001-oa.csv
    pop_path = CENSUS_DIR / "census2021-ts001-oa.csv"
    if not pop_path.exists():
        raise FileNotFoundError(
            f"Census population data not found: {pop_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_census_data\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
    pop_df = pd.read_csv(pop_path)
    pop_df = pop_df.rename(columns={
        "geography code": "oa_code",
        "Residence type: Total; measures: Value": "population",
    })[["oa_code", "population"]]

    # Workplace: WP001_oa.csv
    wp_path = CENSUS_DIR / "WP001_oa.csv"
    if not wp_path.exists():
        raise FileNotFoundError(
            f"Workplace population data not found: {wp_path}\n"
            "  Fetch it first with:\n"
            "    python -m benchmarks_final.fetch_census_data\n"
            "  Or run the full pipeline:  ./reproduce.sh data-fetch"
        )
    wp_df = pd.read_csv(wp_path)
    wp_df = wp_df.rename(columns={
        "Output Areas Code": "oa_code",
        "Count": "workplace_population",
    })[["oa_code", "workplace_population"]]

    # Merge
    census = pop_df.merge(wp_df, on="oa_code", how="outer")
    census["population"] = census["population"].fillna(0).astype(int)
    census["workplace_population"] = census["workplace_population"].fillna(0).astype(int)

    # Filter to E&W codes only (E/W prefix)
    census = census[census["oa_code"].str.startswith(("E", "W"))].copy()

    log.info(
        f"  E&W: {len(census):,} OAs, "
        f"pop={census['population'].sum():,}, "
        f"wp={census['workplace_population'].sum():,}"
    )
    return census


def prepare_scotland_census() -> pd.DataFrame:
    """Download and prepare Scotland Census 2022 OA-level data.

    Scotland uses Output Areas with S-prefixed codes (~46,000 OAs).
    Downloads bulk OA topic tables from scotlandscensus.gov.uk and
    extracts UV101b (usual resident population).

    If fetch_census_data.py has already been run, the cached CSV at
    data/uk_drive/scotland_census.csv will be used directly.
    """
    scot_cache = DATA_DIR / "scotland_census.csv"
    if scot_cache.exists():
        log.info(f"  Loading cached Scotland census: {scot_cache}")
        return pd.read_csv(scot_cache)

    import io
    import zipfile

    import requests

    log.info("  Downloading Scotland Census 2022 OA data ...")

    # Bulk OA topic tables from scotlandscensus.gov.uk (~74 MB zip)
    zip_url = (
        "https://www.scotlandscensus.gov.uk/media/"
        "zz85kfinmf97whklasd98gfkadft5hj4f_Topic2H_20241120_1747/"
        "Census-2022-Output-Area-v1.zip"
    )

    try:
        log.info(f"  Downloading {zip_url} ...")
        resp = requests.get(zip_url, timeout=300)
        resp.raise_for_status()
        log.info(f"    Downloaded {len(resp.content) / 1e6:.1f} MB")
    except Exception as e:
        raise RuntimeError(
            f"Could not download Scotland census data: {e}\n"
            "Download manually from:\n"
            f"  {zip_url}\n"
            "Or run: python -m benchmarks_final.fetch_census_data"
        ) from e

    # Extract UV101b (usual resident population) from the zip
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        uv101_name = None
        for name in zf.namelist():
            if "UV101" in name and name.endswith(".csv"):
                uv101_name = name
                break
        if uv101_name is None:
            raise RuntimeError(
                f"UV101 population CSV not found in zip. Contents: {zf.namelist()[:10]}"
            )

        log.info(f"    Extracting {uv101_name}")
        with zf.open(uv101_name) as f:
            # Multi-header format: 6 header lines, then data rows
            # Col 0 = OA code, Col 1 = total population
            df = pd.read_csv(
                f, skiprows=6, header=None, na_values=["-"], low_memory=False,
            )

    df = df.iloc[:, [0, 1]]
    df.columns = ["oa_code", "population"]
    df = df.dropna(subset=["oa_code"])
    df["oa_code"] = df["oa_code"].astype(str).str.strip()

    oa_mask = df["oa_code"].str.match(r"^S\d{8}$")
    df = df[oa_mask].copy()
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)

    # Scotland workplace population is not available at OA level;
    # estimate as fraction of residential population (Scotland average ~45%)
    df["workplace_population"] = (df["population"] * 0.45).astype(int)

    scot_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(scot_cache, index=False)
    log.info(
        f"  Scotland: {len(df):,} OAs, "
        f"pop={df['population'].sum():,}, "
        f"wp={df['workplace_population'].sum():,}"
    )
    return df


def prepare_combined_census() -> pd.DataFrame:
    """Combine E&W and Scotland census into unified DataFrame."""
    out_path = DATA_DIR / "gb_census_combined.csv"
    if out_path.exists():
        log.info(f"  Loading cached combined census: {out_path}")
        return pd.read_csv(out_path)

    ew = prepare_ew_census()
    scot = prepare_scotland_census()
    combined = pd.concat([ew, scot], ignore_index=True)

    combined.to_csv(out_path, index=False)
    log.info(
        f"  Combined: {len(combined):,} areas, "
        f"pop={combined['population'].sum():,}, "
        f"wp={combined['workplace_population'].sum():,}"
    )
    return combined


# =============================================================================
# Step 2: Download OA/Data Zone boundaries
# =============================================================================


def download_ew_oa_boundaries(out_path: Path) -> gpd.GeoDataFrame:
    """Download all E&W OA boundaries from ONS ArcGIS FeatureServer."""
    import requests

    BASE_URL = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Output_Areas_2021_EW_BGC_V2/FeatureServer/0/query"
    )

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
        log.info(f"  Fetching E&W OA boundaries batch at offset {offset} ...")
        resp = requests.get(BASE_URL, params=params, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        log.info(f"    Got {len(features)} features (total: {len(all_features)})")

        if len(features) < batch_size:
            break
        offset += batch_size

    log.info(f"  Downloaded {len(all_features):,} E&W OA boundaries")

    geojson = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
    gdf = gdf.rename(columns={"OA21CD": "oa_code"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="ew_oa")
    log.info(f"  Saved E&W boundaries to {out_path}")
    return gdf


def download_scotland_boundaries(out_path: Path) -> gpd.GeoDataFrame:
    """Download Scotland Census 2022 OA boundaries from NRS WFS.

    Uses the maps.gov.scot WFS endpoint (NRS/Census2022 service).
    Downloads in pages of 5000 features to handle the 46,363 OAs.
    """
    import requests
    from io import BytesIO

    log.info("  Downloading Scotland OA boundaries (WFS) ...")

    wfs_url = (
        "https://maps.gov.scot/server/services/"
        "NRS/Census2022/MapServer/WFSServer"
    )

    all_features: list[dict] = []
    page_size = 1000  # WFS server caps at 1000 per request
    offset = 0

    while True:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "CEN2022:OutputArea2022",
            "outputFormat": "GEOJSON",
            "srsName": "EPSG:4326",
            "count": str(page_size),
            "startIndex": str(offset),
        }
        resp = requests.get(wfs_url, params=params, timeout=120)
        resp.raise_for_status()

        fc = resp.json()
        features = fc.get("features", [])
        all_features.extend(features)
        log.info(f"    Fetched {len(features)} features (total: {len(all_features)})")

        if len(features) < page_size:
            break
        offset += page_size

    if not all_features:
        raise RuntimeError("WFS returned no Scotland OA boundaries")

    # Build GeoDataFrame from collected features
    import json
    geojson = {"type": "FeatureCollection", "features": all_features}
    gdf = gpd.read_file(BytesIO(json.dumps(geojson).encode()))
    log.info(f"  Downloaded {len(gdf)} Scotland OA boundaries")

    # WFS 2.0.0 with EPSG:4326 returns coordinates in (lat, lon) axis order
    # per the GML/WFS spec, but GeoJSON and geopandas expect (lon, lat).
    # Detect and fix the swap by checking whether x-coords look like latitudes.
    sample_bounds = gdf.geometry.iloc[0].bounds  # (minx, miny, maxx, maxy)
    if sample_bounds[0] > 50:  # x looks like latitude (50-61 for Scotland)
        from shapely.ops import transform as shp_transform

        log.info("  Fixing WFS axis order: swapping (lat,lon) → (lon,lat)")
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: shp_transform(lambda x, y: (y, x), geom)
        )
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    # Standardise column name to oa_code
    for col in gdf.columns:
        cl = col.upper()
        if ("OA" in cl and "CD" in cl) or col.lower() in ("code", "cenoacd"):
            gdf = gdf.rename(columns={col: "oa_code"})
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="scotland_oa")
    log.info(f"  Saved to {out_path}")
    return gdf


def load_or_download_boundaries() -> gpd.GeoDataFrame:
    """Load or download all GB OA boundaries (cached automatically)."""
    ew_path = DATA_DIR / "ew_oa_boundaries.gpkg"
    scot_path = DATA_DIR / "scotland_oa_boundaries.gpkg"
    combined_path = DATA_DIR / "gb_oa_boundaries.gpkg"

    if combined_path.exists():
        log.info(f"  Loading cached boundaries: {combined_path}")
        return gpd.read_file(combined_path)

    # England & Wales
    if ew_path.exists():
        log.info(f"  Loading cached E&W boundaries: {ew_path}")
        ew = gpd.read_file(ew_path)
    else:
        ew = download_ew_oa_boundaries(ew_path)

    # Scotland
    if scot_path.exists():
        log.info(f"  Loading cached Scotland boundaries: {scot_path}")
        scot = gpd.read_file(scot_path)
    else:
        scot = download_scotland_boundaries(scot_path)

    # Ensure consistent column names
    if "oa_code" not in scot.columns:
        # Try to find the OA code column
        for col in scot.columns:
            if col.lower() not in ("geometry",) and scot[col].dtype == "object":
                s = scot[col].iloc[0] if len(scot) > 0 else ""
                if isinstance(s, str) and s.startswith("S"):
                    scot = scot.rename(columns={col: "oa_code"})
                    break

    # Combine
    combined = pd.concat(
        [ew[["oa_code", "geometry"]], scot[["oa_code", "geometry"]]],
        ignore_index=True,
    )
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    combined.to_file(combined_path, driver="GPKG")
    log.info(f"  Combined boundaries: {len(combined):,} areas → {combined_path}")
    return combined


# =============================================================================
# Step 3: Download OSM buildings from Geofabrik PBF via pyrosm
# =============================================================================

GEOFABRIK_URL = (
    "https://download.geofabrik.de/europe/great-britain-latest.osm.pbf"
)


def download_geofabrik_pbf(out_path: Path) -> Path:
    """Download the Great Britain PBF file from Geofabrik (~1.2 GB)."""
    if out_path.exists():
        log.info(f"  PBF already cached: {out_path} ({out_path.stat().st_size / 1e9:.1f} GB)")
        return out_path

    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"  Downloading {GEOFABRIK_URL} ...")
    log.info("  (This is ~1.2 GB, may take a few minutes)")
    t0 = time.perf_counter()
    urllib.request.urlretrieve(GEOFABRIK_URL, out_path)
    dt = time.perf_counter() - t0
    size_gb = out_path.stat().st_size / 1e9
    log.info(f"  Downloaded in {dt:.0f}s ({size_gb:.1f} GB)")
    return out_path


def extract_buildings_from_pbf(pbf_path: Path, out_path: Path) -> gpd.GeoDataFrame:
    """Extract building footprints from PBF using pyrosm.

    Returns GeoDataFrame with geometry, building type, and building:levels.
    """
    if out_path.exists():
        log.info(f"  Loading cached buildings: {out_path}")
        return gpd.read_file(out_path)

    import warnings

    import pyrosm

    log.info(f"  Extracting buildings from {pbf_path.name} with pyrosm ...")
    t0 = time.perf_counter()
    osm = pyrosm.OSM(str(pbf_path))
    # Suppress pandas Copy-on-Write warnings from pyrosm internals
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*chained assignment.*")
        buildings = osm.get_buildings()
    dt = time.perf_counter() - t0
    log.info(f"  Extracted {len(buildings):,} buildings in {dt:.0f}s")

    # Keep only polygon/multipolygon geometries
    buildings = buildings[
        buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    log.info(f"  {len(buildings):,} polygon buildings after filtering")

    # Keep useful columns
    keep_cols = ["geometry"]
    for col in ["building", "building:levels", "height", "name"]:
        if col in buildings.columns:
            keep_cols.append(col)
    buildings = buildings[keep_cols].reset_index(drop=True)

    # Ensure CRS is WGS84
    if buildings.crs is None:
        buildings = buildings.set_crs("EPSG:4326")
    elif buildings.crs.to_epsg() != 4326:
        buildings = buildings.to_crs("EPSG:4326")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"  Saving buildings to {out_path} ...")
    buildings.to_file(out_path, driver="GPKG")
    size_gb = out_path.stat().st_size / 1e9
    log.info(f"  Saved ({size_gb:.1f} GB)")

    return buildings


# =============================================================================
# Step 4-5: Floor area, spatial join, distribute population
# =============================================================================


def compute_floor_area(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add footprint_area, levels, and floor_area columns."""
    buildings_bng = buildings.to_crs(BNG_CRS)
    buildings["footprint_area_m2"] = buildings_bng.geometry.area

    level_col = "building:levels"
    if level_col in buildings.columns:
        buildings["levels"] = pd.to_numeric(buildings[level_col], errors="coerce")
    else:
        buildings["levels"] = np.nan

    btype_col = "building" if "building" in buildings.columns else None
    mask_missing = buildings["levels"].isna()
    if btype_col is not None:
        buildings.loc[mask_missing, "levels"] = buildings.loc[
            mask_missing, btype_col
        ].map(DEFAULT_LEVELS)
    buildings["levels"] = buildings["levels"].fillna(FALLBACK_LEVELS).clip(lower=1)
    buildings["floor_area_m2"] = buildings["footprint_area_m2"] * buildings["levels"]
    return buildings


def assign_buildings_to_oas(
    buildings: gpd.GeoDataFrame,
    oa_boundaries: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Spatial join: assign each building to an OA by centroid containment."""
    buildings = buildings.copy()
    buildings["_centroid"] = buildings.geometry.centroid
    buildings_pts = buildings.set_geometry("_centroid")

    joined = gpd.sjoin(
        buildings_pts,
        oa_boundaries[["oa_code", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.set_geometry("geometry").drop(
        columns=["_centroid", "index_right"], errors="ignore"
    )
    return joined


def distribute_population(
    buildings: gpd.GeoDataFrame,
    census: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Distribute OA-level population & workplace to buildings by floor area share."""
    buildings = buildings.copy()

    oa_floor_area = (
        buildings[buildings["oa_code"].notna()]
        .groupby("oa_code")["floor_area_m2"]
        .transform("sum")
    )

    mask = buildings["oa_code"].notna()
    buildings["floor_area_share"] = 0.0
    buildings.loc[mask, "floor_area_share"] = (
        buildings.loc[mask, "floor_area_m2"] / oa_floor_area
    )

    buildings = buildings.merge(
        census[["oa_code", "population", "workplace_population"]],
        on="oa_code",
        how="left",
    )

    buildings["pop"] = buildings["floor_area_share"] * buildings["population"].fillna(0)
    buildings["wp"] = (
        buildings["floor_area_share"] * buildings["workplace_population"].fillna(0)
    )
    buildings = buildings.drop(
        columns=["population", "workplace_population"], errors="ignore"
    )
    return buildings


# =============================================================================
# Step 6: Snap buildings to network edges → node-level activity
# =============================================================================


def snap_buildings_to_nodes(
    buildings: gpd.GeoDataFrame,
    node_xy_bng: np.ndarray,
    zones: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Snap building centroids to nearest network node using KDTree.

    For a national-scale analysis, snapping to nearest node (rather than
    nearest edge) is acceptably accurate and much faster.

    Returns (node_pop, node_wp) arrays indexed by zone order.
    """
    log.info("  Snapping buildings to nearest network nodes ...")
    t0 = time.perf_counter()

    n_zones = len(zones)
    node_tree = cKDTree(node_xy_bng)

    # Building centroids in BNG
    bld_centroids = buildings.geometry.centroid
    bld_bng = gpd.GeoSeries(bld_centroids, crs=buildings.crs).to_crs(BNG_CRS)
    bld_xy = np.column_stack([bld_bng.x, bld_bng.y])

    bld_pop = buildings["pop"].values.astype(np.float64)
    bld_wp = buildings["wp"].values.astype(np.float64)

    # Query nearest node for each building
    _, nearest_idx = node_tree.query(bld_xy)

    # Accumulate
    node_pop = np.zeros(n_zones, dtype=np.float64)
    node_wp = np.zeros(n_zones, dtype=np.float64)
    np.add.at(node_pop, nearest_idx, bld_pop)
    np.add.at(node_wp, nearest_idx, bld_wp)

    dt = time.perf_counter() - t0
    log.info(f"  Snapping done in {dt:.1f}s")
    log.info(f"  Node pop total: {node_pop.sum():,.0f}, node wp total: {node_wp.sum():,.0f}")
    return node_pop, node_wp


# =============================================================================
# Step 7: Maps
# =============================================================================


def plot_distribution_maps(
    node_xy: np.ndarray,
    node_pop: np.ndarray,
    node_wp: np.ndarray,
    out_dir: Path,
) -> None:
    """GB-wide scatter plots of node-level population and workplace."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "figure.dpi": 150,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    # Average aspect for GB (~55°N)
    aspect = 1.0 / math.cos(math.radians(55.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 16), constrained_layout=True)

    # (a) Node population
    ax = axes[0]
    mask = node_pop > 0
    sc = ax.scatter(
        node_xy[mask, 0], node_xy[mask, 1],
        c=node_pop[mask], s=0.05, cmap="YlOrRd",
        norm=LogNorm(vmin=max(node_pop[mask].min(), 0.01), vmax=node_pop[mask].max()),
        rasterized=True,
    )
    ax.set_aspect(aspect)
    ax.set_title(f"(a) Node population (n={mask.sum():,})")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Population")

    # (b) Node workplace
    ax = axes[1]
    mask = node_wp > 0
    sc = ax.scatter(
        node_xy[mask, 0], node_xy[mask, 1],
        c=node_wp[mask], s=0.05, cmap="YlGnBu",
        norm=LogNorm(vmin=max(node_wp[mask].min(), 0.01), vmax=node_wp[mask].max()),
        rasterized=True,
    )
    ax.set_aspect(aspect)
    ax.set_title(f"(b) Node workplace pop (n={mask.sum():,})")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Workplace pop")

    fig.suptitle(
        "GB driving network: population & workplace disaggregation",
        fontsize=12,
    )
    fig.savefig(out_dir / "gb_drive_population_disaggregation.png", dpi=200)
    fig.savefig(out_dir / "gb_drive_population_disaggregation.pdf")
    plt.close(fig)
    log.info(f"  Maps saved to {out_dir / 'gb_drive_population_disaggregation.png'}")


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disaggregate Census population to buildings and GB driving network nodes"
    )
    parser.add_argument(
        "--skip-buildings", action="store_true",
        help="Skip building download, reuse cached building files",
    )
    parser.add_argument(
        "--network", type=str, default=None,
        help="Network pickle path (default: data/uk_drive/gb_drive.pkl)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip map generation",
    )
    parser.add_argument(
        "--access-mask", type=str, default=None,
        help="Path to node access mask NPZ (gb_node_access_mask.npz). "
             "If provided, builds a second activity vector snapping only "
             "to eligible (non-motorway/tunnel/bridge) nodes.",
    )
    args = parser.parse_args()

    import networkx as nx
    from pyproj import Transformer

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    net_path = Path(args.network) if args.network else DATA_DIR / "gb_drive.pkl"

    # ---- Step 1: Census data ----
    log.info("=" * 60)
    log.info("STEP 1: Prepare Census data")
    census = prepare_combined_census()
    log.info(f"  {len(census):,} areas loaded")

    # ---- Step 2: OA boundaries ----
    log.info("=" * 60)
    log.info("STEP 2: OA / Data Zone boundaries")
    oa_boundaries = load_or_download_boundaries()
    log.info(f"  {len(oa_boundaries):,} boundaries loaded")

    # ---- Step 3: Buildings from Geofabrik PBF ----
    log.info("=" * 60)
    log.info("STEP 3: OSM buildings (Geofabrik PBF via pyrosm)")
    pbf_path = DATA_DIR / "great-britain-latest.osm.pbf"
    buildings_gpkg = DATA_DIR / "gb_buildings.gpkg"

    if args.skip_buildings:
        if buildings_gpkg.exists():
            log.info(f"  Loading cached buildings: {buildings_gpkg}")
            buildings = gpd.read_file(buildings_gpkg)
        else:
            log.error(f"  No cached buildings at {buildings_gpkg}")
            sys.exit(1)
    else:
        download_geofabrik_pbf(pbf_path)
        buildings = extract_buildings_from_pbf(pbf_path, buildings_gpkg)

    log.info(f"  Total buildings: {len(buildings):,}")

    # ---- Step 4: Floor area ----
    log.info("=" * 60)
    log.info("STEP 4: Floor area computation")
    buildings = compute_floor_area(buildings)
    log.info(
        f"  Floor area: total={buildings['floor_area_m2'].sum() / 1e6:.1f} M m², "
        f"median footprint={buildings['footprint_area_m2'].median():.0f} m²"
    )

    # ---- Step 5: Spatial join + distribute ----
    log.info("=" * 60)
    log.info("STEP 5: Spatial join & distribute population")
    buildings = assign_buildings_to_oas(buildings, oa_boundaries)

    n_matched = buildings["oa_code"].notna().sum()
    log.info(f"  Matched {n_matched:,}/{len(buildings):,} buildings to OAs")

    buildings = distribute_population(buildings, census)
    total_pop = buildings["pop"].sum()
    total_wp = buildings["wp"].sum()
    log.info(f"  Distributed pop: {total_pop:,.0f}, workplace: {total_wp:,.0f}")

    census_pop = census["population"].sum()
    census_wp = census["workplace_population"].sum()
    log.info(f"  Census totals:   pop={census_pop:,}, wp={census_wp:,}")
    log.info(
        f"  Coverage: pop={100 * total_pop / census_pop:.1f}%, "
        f"wp={100 * total_wp / census_wp:.1f}%"
    )

    # ---- Step 6: Load network & snap ----
    log.info("=" * 60)
    log.info("STEP 6: Load network & snap buildings to nodes")

    if not net_path.exists():
        log.error(f"Network not found: {net_path}")
        log.error("  Run: python -m benchmarks_final.fetch_uk_drive")
        sys.exit(1)

    log.info(f"  Loading {net_path} ...")
    t0 = time.perf_counter()
    with open(net_path, "rb") as f:
        G = pickle.load(f)
    dt = time.perf_counter() - t0
    log.info(
        f"  Loaded in {dt:.1f}s: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    zones = sorted(G.nodes())
    node_xy = np.array([[G.nodes[z]["x"], G.nodes[z]["y"]] for z in zones])

    # Project node coordinates to BNG for spatial snapping
    log.info("  Projecting node coordinates to BNG ...")
    transformer = Transformer.from_crs("EPSG:4326", BNG_CRS, always_xy=True)
    xs_bng, ys_bng = transformer.transform(node_xy[:, 0], node_xy[:, 1])
    node_xy_bng = np.column_stack([xs_bng, ys_bng])

    # Filter to buildings with pop/wp > 0
    bld_active = buildings[(buildings["pop"] > 0) | (buildings["wp"] > 0)].copy()
    log.info(f"  {len(bld_active):,} buildings with pop/wp > 0 to snap")

    node_pop, node_wp = snap_buildings_to_nodes(bld_active, node_xy_bng, zones)

    # ---- Step 7: Save outputs ----
    log.info("=" * 60)
    log.info("STEP 7: Save outputs")

    npz_path = DATA_DIR / "gb_node_activity.npz"
    np.savez_compressed(
        npz_path,
        zones=np.array(zones, dtype=np.int64),
        node_xy=node_xy,
        node_pop=node_pop,
        node_wp=node_wp,
    )
    log.info(f"  Saved node activity to {npz_path}")

    # ---- Step 7b: Filtered snap (if access mask provided) ----
    if args.access_mask:
        log.info("=" * 60)
        log.info("STEP 7b: Filtered snap (access-eligible nodes only)")

        mask_path = Path(args.access_mask)
        if not mask_path.exists():
            log.error(f"  Access mask not found: {mask_path}")
            sys.exit(1)

        mask_data = np.load(mask_path)
        access_mask = mask_data["access_eligible"]
        log.info(f"  Loaded mask: {access_mask.sum():,}/{len(access_mask):,} eligible "
                 f"({100 * access_mask.sum() / len(access_mask):.1f}%)")

        if len(access_mask) != len(zones):
            log.error(
                f"  Mask size {len(access_mask)} != network size {len(zones)}"
            )
            sys.exit(1)

        # Build KDTree on eligible nodes only
        eligible_idx = np.where(access_mask)[0]
        eligible_xy_bng = node_xy_bng[eligible_idx]
        eligible_tree = cKDTree(eligible_xy_bng)

        # Snap buildings to nearest eligible node
        log.info(f"  Snapping {len(bld_active):,} buildings to nearest eligible node ...")
        t0 = time.perf_counter()
        bld_centroids = bld_active.geometry.centroid
        bld_bng = gpd.GeoSeries(bld_centroids, crs=bld_active.crs).to_crs(BNG_CRS)
        bld_xy = np.column_stack([bld_bng.x, bld_bng.y])
        bld_pop_vals = bld_active["pop"].values.astype(np.float64)
        bld_wp_vals = bld_active["wp"].values.astype(np.float64)

        dists_filtered, nearest_in_eligible = eligible_tree.query(bld_xy)
        nearest_global = eligible_idx[nearest_in_eligible]  # map back to global indices

        filt_node_pop = np.zeros(len(zones), dtype=np.float64)
        filt_node_wp = np.zeros(len(zones), dtype=np.float64)
        np.add.at(filt_node_pop, nearest_global, bld_pop_vals)
        np.add.at(filt_node_wp, nearest_global, bld_wp_vals)
        dt = time.perf_counter() - t0
        log.info(f"  Filtered snap done in {dt:.1f}s")

        # Also compute unfiltered snap distances for comparison
        all_tree = cKDTree(node_xy_bng)
        dists_unfiltered, _ = all_tree.query(bld_xy)

        log.info(f"  Snap distances (unfiltered): "
                 f"mean={dists_unfiltered.mean():.0f}m, max={dists_unfiltered.max():.0f}m")
        log.info(f"  Snap distances (filtered):   "
                 f"mean={dists_filtered.mean():.0f}m, max={dists_filtered.max():.0f}m")

        # Population conservation check
        log.info(f"  Population conservation: "
                 f"original={node_pop.sum():,.0f}, filtered={filt_node_pop.sum():,.0f}")
        log.info(f"  Workplace conservation:  "
                 f"original={node_wp.sum():,.0f}, filtered={filt_node_wp.sum():,.0f}")

        # Save filtered activity
        filt_npz_path = DATA_DIR / "gb_node_activity_filtered.npz"
        np.savez_compressed(
            filt_npz_path,
            zones=np.array(zones, dtype=np.int64),
            node_xy=node_xy,
            node_pop=filt_node_pop,
            node_wp=filt_node_wp,
        )
        log.info(f"  Saved filtered activity to {filt_npz_path}")

    # Maps
    if not args.no_plot:
        log.info("  Generating distribution maps ...")
        plot_distribution_maps(node_xy, node_pop, node_wp, PAPER_FIG_DIR)

    # ---- Summary ----
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info(f"  Census areas:          {len(census):,}")
    log.info(f"  OA boundaries:         {len(oa_boundaries):,}")
    log.info(f"  OSM buildings:         {len(buildings):,}")
    log.info(f"  Buildings matched:     {n_matched:,}")
    log.info(f"  Census pop total:      {census_pop:,}")
    log.info(f"  Building pop total:    {total_pop:,.0f}")
    log.info(f"  Node pop total:        {node_pop.sum():,.0f}")
    log.info(f"  Census wp total:       {census_wp:,}")
    log.info(f"  Building wp total:     {total_wp:,.0f}")
    log.info(f"  Node wp total:         {node_wp.sum():,.0f}")
    log.info(f"  Network nodes:         {len(zones):,}")
    log.info(f"  Active nodes (pop>0):  {(node_pop > 0).sum():,}")
    log.info(f"  Active nodes (wp>0):   {(node_wp > 0).sum():,}")
    if args.access_mask:
        log.info(f"  Filtered pop total:    {filt_node_pop.sum():,.0f}")
        log.info(f"  Filtered wp total:     {filt_node_wp.sum():,.0f}")
        log.info(f"  Filtered active (pop): {(filt_node_pop > 0).sum():,}")
        log.info(f"  Filtered active (wp):  {(filt_node_wp > 0).sum():,}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
