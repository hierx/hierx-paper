"""
Layer structure benchmark: Analyse compression ratio and layer counts.

Sweeps base_radius, increase_factor, and overlap_factor across grid networks
to record hierarchical structure details. Produces a JSON file matching the
schema of layer_structure_results.json used by plot_paper_figures.py (Figure 6).

Usage:
    python -m benchmarks_final.layer_structure_benchmark
    python -m benchmarks_final.layer_structure_benchmark --output results/layers.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierx import Hierarchy, generate_grid_network

from benchmarks_final.common import DATA_DIR, N_WORKERS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_CONFIGS: list[tuple[str, int, int]] = [
    ("grid_10x10", 10, 10),
    ("grid_20x20", 20, 20),
    ("grid_30x30", 30, 30),
]
BASE_RADII: list[float] = [2000.0, 3000.0, 5000.0, 8000.0]
INCREASE_FACTORS: list[float] = [1.5, 2.0, 2.5, 3.0]
OVERLAP_FACTORS: list[float] = [1.0, 1.5, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_config(
    label: str,
    G,
    n_zones: int,
    base_radius: float,
    increase_factor: float,
    overlap_factor: float,
) -> dict:
    """Run layer structure analysis for a single configuration."""
    hierarchy = Hierarchy(
        G,
        base_radius=base_radius,
        increase_factor=increase_factor,
        overlap_factor=overlap_factor,
        n_workers=N_WORKERS,
    )

    # Collect per-layer statistics
    layers_info: list[dict] = []
    total_cost_entries = 0
    for i, radius in enumerate(hierarchy.radii):
        n_reps = len(set(hierarchy.groups[radius].values()))
        n_entries = sum(
            len(dests) for dests in hierarchy.costs[radius].values()
        )
        total_cost_entries += n_entries
        layers_info.append({
            "layer": i,
            "radius": radius,
            "n_representatives": n_reps,
            "n_cost_entries": n_entries,
        })

    dense_entries = n_zones * n_zones
    compression_ratio = dense_entries / total_cost_entries if total_cost_entries > 0 else float("inf")

    return {
        "experiment": "layer_structure",
        "network_label": label,
        "n_zones": n_zones,
        "base_radius": base_radius,
        "increase_factor": increase_factor,
        "overlap_factor": overlap_factor,
        "n_layers": len(hierarchy.radii),
        "layers": layers_info,
        "total_cost_entries": total_cost_entries,
        "dense_entries": dense_entries,
        "compression_ratio": compression_ratio,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    default_output = str(DATA_DIR / "layer_structure_results.json")

    parser = argparse.ArgumentParser(
        description="Layer structure benchmark (Fig 6)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Path for JSON output (default: {default_output})",
    )
    args = parser.parse_args()

    total = len(GRID_CONFIGS) * len(BASE_RADII) * len(INCREASE_FACTORS) * len(OVERLAP_FACTORS)

    print("=" * 80)
    print("Layer Structure Benchmark")
    print("=" * 80)
    print(f"Networks: {[c[0] for c in GRID_CONFIGS]}")
    print(f"Base radii: {BASE_RADII}")
    print(f"Increase factors: {INCREASE_FACTORS}")
    print(f"Overlap factors: {OVERLAP_FACTORS}")
    print(f"Total configurations: {total}")
    print()

    # Pre-build networks
    networks: dict[str, tuple] = {}
    for label, rows, cols in GRID_CONFIGS:
        G = generate_grid_network(rows, cols)
        n_zones = G.number_of_nodes()
        networks[label] = (G, n_zones)
        print(f"  Built {label}: {n_zones} zones, {G.number_of_edges()} edges")

    print()

    results: list[dict] = []
    done = 0
    for label, rows, cols in GRID_CONFIGS:
        G, n_zones = networks[label]
        for br in BASE_RADII:
            for if_val in INCREASE_FACTORS:
                for of in OVERLAP_FACTORS:
                    rec = benchmark_config(label, G, n_zones, br, if_val, of)
                    results.append(rec)
                    done += 1

                    print(
                        f"  [{done}/{total}] {label}, br={br:.0f}, "
                        f"if={if_val}, of={of}, "
                        f"layers={rec['n_layers']}, "
                        f"compress={rec['compression_ratio']:.1f}x",
                        flush=True,
                    )

    # Write JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"results": results}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to {output_path.resolve()} ({len(results)} entries)")


if __name__ == "__main__":
    main()
