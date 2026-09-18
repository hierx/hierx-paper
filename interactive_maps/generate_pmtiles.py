#!/usr/bin/env python3
"""Generate PMTiles maps from accessibility results.

Pipeline: NPZ → GeoJSONSeq → PMTiles (via tippecanoe).

Two tile profiles are produced (see PROFILES):

* ``full`` keeps every edge at every zoom. Overview tiles reach ~25 MB
  decompressed, which desktop browsers handle but phones do not.
* ``lite`` caps each tile at 2 MB by dropping the shortest edges first at
  coarse zooms (``--drop-smallest-as-needed``), which keeps the long-distance
  road skeleton and the urban colour pattern while cutting the overview
  payload by an order of magnitude. Written to ``*_lite.pmtiles``.

Usage:
    python interactive_maps/generate_pmtiles.py all
    python interactive_maps/generate_pmtiles.py gb_drive
    python interactive_maps/generate_pmtiles.py london_walk --profile lite
    python interactive_maps/generate_pmtiles.py london_walk --keep-geojson
    python interactive_maps/generate_pmtiles.py gb_drive --npz path/to/file.npz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROFILES: dict[str, dict] = {
    "full": {
        "suffix": "",
        "tippecanoe": ["--no-feature-limit", "--no-tile-size-limit", "--drop-densest-as-needed"],
    },
    "lite": {
        "suffix": "_lite",
        "tippecanoe": ["--maximum-tile-bytes", "2000000", "--drop-smallest-as-needed"],
    },
}

DATASETS: dict[str, dict] = {
    "gb_drive": {
        "npz": "paper_figs_final/data/gb_drive_pop_accessibility_nodes.npz",
        "output": "interactive_maps/gb_drive_accessibility.pmtiles",
        "min_zoom": 5,
        "max_zoom": 13,
    },
    "london_walk": {
        "npz": "paper_figs_final/data/london_walk_1M_pop_accessibility_nodes.npz",
        "output": "interactive_maps/london_walk_accessibility.pmtiles",
        "min_zoom": 8,
        "max_zoom": 14,
    },
}


def write_geojsonseq(
    npz_path: Path,
    output_path: Path,
) -> dict[str, float]:
    """Write edge features as newline-delimited GeoJSON.

    Returns dict with dataset statistics for reference.
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    node_xy = data["node_xy"]  # (N, 2) lon/lat
    edges = data["edges"]  # (E, 2) node index pairs
    acc_pop = data["acc_population"]  # (N,)
    acc_emp = data["acc_workplace"]  # (N,)

    n_nodes = len(node_xy)
    n_edges = len(edges)
    print(f"  {n_nodes:,} nodes, {n_edges:,} edges")

    # Per-edge accessibility: average of endpoints
    src, dst = edges[:, 0], edges[:, 1]
    edge_pop = (acc_pop[src] + acc_pop[dst]) / 2.0
    edge_emp = (acc_emp[src] + acc_emp[dst]) / 2.0

    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "pop_min": float(edge_pop.min()),
        "pop_max": float(edge_pop.max()),
        "emp_min": float(edge_emp.min()),
        "emp_max": float(edge_emp.max()),
    }

    print(f"  pop_reached range: [{stats['pop_min']:.4f}, {stats['pop_max']:.4f}]")
    print(f"  emp_reached range: [{stats['emp_min']:.4f}, {stats['emp_max']:.4f}]")

    print(f"Writing {n_edges:,} features to {output_path}...")
    t0 = time.time()

    with open(output_path, "w") as f:
        for i in range(n_edges):
            s, d = int(src[i]), int(dst[i])
            lon_s, lat_s = node_xy[s]
            lon_d, lat_d = node_xy[d]
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(lon_s, 6), round(lat_s, 6)],
                        [round(lon_d, 6), round(lat_d, 6)],
                    ],
                },
                "properties": {
                    "pop_reached": round(float(edge_pop[i]), 4),
                    "emp_reached": round(float(edge_emp[i]), 4),
                },
            }
            f.write(json.dumps(feature, separators=(",", ":")) + "\n")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return stats


def run_tippecanoe(
    geojson_path: Path,
    pmtiles_path: Path,
    min_zoom: int = 8,
    max_zoom: int = 14,
    profile: str = "full",
) -> None:
    """Convert GeoJSONSeq to PMTiles using tippecanoe.

    Args:
        geojson_path: Newline-delimited GeoJSON input.
        pmtiles_path: Output archive.
        min_zoom: Coarsest zoom level to generate.
        max_zoom: Finest zoom level to generate.
        profile: Key into PROFILES selecting the tile-size policy.
    """
    import shutil

    tippecanoe = shutil.which("tippecanoe") or str(Path.home() / "bin" / "tippecanoe")
    cmd = [
        tippecanoe,
        "-o", str(pmtiles_path),
        "-l", "accessibility",
        "-Z", str(min_zoom),
        "-z", str(max_zoom),
        *PROFILES[profile]["tippecanoe"],
        "--no-line-simplification",
        "--force",
        "--extend-zooms-if-still-dropping",
        "-P",  # parallel read
        str(geojson_path),
    ]

    print("Running tippecanoe...")
    print(f"  {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"tippecanoe stderr:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    size_mb = pmtiles_path.stat().st_size / (1024 * 1024)
    print(f"  Done in {elapsed:.1f}s → {pmtiles_path} ({size_mb:.1f} MB)")


def generate_dataset(
    name: str,
    npz_override: Path | None = None,
    output_override: Path | None = None,
    keep_geojson: bool = False,
    profiles: tuple[str, ...] = ("full", "lite"),
) -> None:
    """Generate PMTiles for a single dataset, one archive per profile."""
    cfg = DATASETS[name]
    npz_path = npz_override or Path(cfg["npz"])
    base_path = output_override or Path(cfg["output"])

    print(f"\n{'='*60}")
    print(f"Generating: {name} ({', '.join(profiles)})")
    print(f"{'='*60}")

    geojson_path = base_path.with_suffix(".geojsonl")

    stats = write_geojsonseq(npz_path, geojson_path)
    for profile in profiles:
        suffix = PROFILES[profile]["suffix"]
        pmtiles_path = base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")
        run_tippecanoe(geojson_path, pmtiles_path, cfg["min_zoom"], cfg["max_zoom"], profile)

    if not keep_geojson:
        geojson_path.unlink()
        print(f"Removed intermediate {geojson_path}")

    print(f"\nDataset statistics ({name}):")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["all", *DATASETS.keys()],
        help="Which dataset to generate (default: all)",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Override NPZ input path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override PMTiles output path",
    )
    parser.add_argument(
        "--keep-geojson",
        action="store_true",
        help="Keep intermediate GeoJSONSeq file",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "lite", "both"],
        default="both",
        help="Tile-size profile to build (default: both)",
    )
    args = parser.parse_args()
    profiles = ("full", "lite") if args.profile == "both" else (args.profile,)

    if args.dataset == "all":
        if args.npz or args.output:
            print("Error: --npz and --output cannot be used with 'all'", file=sys.stderr)
            sys.exit(1)
        for name in DATASETS:
            generate_dataset(name, keep_geojson=args.keep_geojson, profiles=profiles)
    else:
        generate_dataset(
            args.dataset,
            npz_override=args.npz,
            output_override=args.output,
            keep_geojson=args.keep_geojson,
            profiles=profiles,
        )

    print("\nServe locally: python interactive_maps/serve.py")


if __name__ == "__main__":
    main()
