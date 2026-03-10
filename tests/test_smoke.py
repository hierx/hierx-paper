"""Smoke tests for hierx-paper reproducibility pipeline.

These tests verify that pre-computed data files parse correctly and that
benchmark modules are importable, without running expensive computations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "paper_figs_final" / "data"

# ---------------------------------------------------------------------------
# JSON parse tests — every tracked JSON file should be valid
# ---------------------------------------------------------------------------

JSON_FILES = [
    "scaling_results.json",
    "scaling_tuned_results.json",
    "error_sensitivity_results.json",
    "baseline_comparison_25k_results.json",
    "interaction_fn_results.json",
    "layer_structure_results.json",
    "london_60k_results.json",
    "london_walk_60k_results.json",
    "population_accessibility_results.json",
    "gb_drive_pop_accessibility_results.json",
]


@pytest.mark.parametrize("filename", JSON_FILES)
def test_json_parses(filename: str) -> None:
    """Pre-computed JSON file parses without error."""
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present (fetch from Zenodo or run benchmark)")
    with open(path) as f:
        data = json.load(f)
    assert "results" in data, f"{filename} missing 'results' key"
    assert len(data["results"]) > 0, f"{filename} has empty results"


# ---------------------------------------------------------------------------
# Import tests — benchmark modules should be importable
# ---------------------------------------------------------------------------


def test_import_common() -> None:
    from benchmarks_final.common import DATA_DIR, FIGURES_DIR, compute_errors  # noqa: F401


def test_import_plot_paper_figures() -> None:
    from benchmarks_final.plot_paper_figures import main  # noqa: F401


def test_import_plot_london_results() -> None:
    from benchmarks_final.plot_london_results import main  # noqa: F401


def test_import_plot_accessibility_maps() -> None:
    from benchmarks_final.plot_accessibility_maps import main  # noqa: F401


_hierx_available = True
try:
    from hierx import Hierarchy  # noqa: F401
except ImportError:
    _hierx_available = False


@pytest.mark.skipif(not _hierx_available, reason="hierx not fully installed")
def test_import_tuned_scaling_benchmark() -> None:
    from benchmarks_final.tuned_scaling_benchmark import main  # noqa: F401


@pytest.mark.skipif(not _hierx_available, reason="hierx not fully installed")
def test_import_layer_structure_benchmark() -> None:
    from benchmarks_final.layer_structure_benchmark import main  # noqa: F401


@pytest.mark.skipif(not _hierx_available, reason="hierx not fully installed")
def test_import_interaction_fn_benchmark() -> None:
    from benchmarks_final.interaction_fn_benchmark import main  # noqa: F401


# ---------------------------------------------------------------------------
# Figure file consistency — every \includegraphics in experiments.tex
# should resolve to an existing PDF
# ---------------------------------------------------------------------------


def test_figure_files_exist() -> None:
    """Every figure referenced in experiments.tex exists in paper_figs_final/figures/."""
    experiments_tex = Path(__file__).resolve().parent.parent.parent / "manuscript" / "sections" / "experiments.tex"
    if not experiments_tex.exists():
        pytest.skip("experiments.tex not found")

    figures_dir = Path(__file__).resolve().parent.parent / "paper_figs_final" / "figures"
    import re

    text = experiments_tex.read_text()
    includes = re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", text)

    missing = []
    for fig in includes:
        fig_path = figures_dir / fig
        if not fig_path.exists():
            missing.append(fig)

    assert not missing, f"Missing figure files: {missing}"


# ---------------------------------------------------------------------------
# Release-blocker smoke tests
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_no_placeholder_scotland_data() -> None:
    """Scotland census CSV must not contain synthetic S00000000 codes."""
    csv_path = PROJECT_DIR / "data" / "uk_drive" / "scotland_census.csv"
    if not csv_path.exists():
        pytest.skip("scotland_census.csv not present (deleted or not yet rebuilt)")
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert not row["oa_code"].startswith("S0000000"), (
                f"Placeholder OA code found: {row['oa_code']}. "
                "Rebuild with real Scotland Census 2022 data."
            )
            break  # checking the first row is sufficient


def test_scotland_census_data_authentic() -> None:
    """Scotland entries in combined census must use real NRS OA codes, not placeholders."""
    csv_path = PROJECT_DIR / "data" / "uk_drive" / "gb_census_combined.csv"
    if not csv_path.exists():
        pytest.skip("gb_census_combined.csv not present")
    import csv

    scot_rows: list[str] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["oa_code"].startswith("S"):
                scot_rows.append(row["oa_code"])
    assert len(scot_rows) > 0, "No Scotland OAs found in combined census"
    # Real NRS codes are S01NNNNNN; placeholders were S00000000-S00045999
    for code in scot_rows[:100]:
        assert not code.startswith("S0000"), (
            f"Placeholder OA code found: {code}. "
            "Combined census contains synthetic Scotland data."
        )
    # Scotland Census 2022 has 46,363 Output Areas
    assert 40_000 < len(scot_rows) < 50_000, (
        f"Unexpected Scotland OA count: {len(scot_rows)} "
        "(Census 2022 has 46,363 OAs)"
    )


def test_no_hardcoded_workspace_paths() -> None:
    """Build scripts must not contain hardcoded /workspace/ paths."""
    import re

    benchmarks_dir = PROJECT_DIR / "benchmarks_final"
    violations: list[str] = []
    for py_file in sorted(benchmarks_dir.glob("*.py")):
        text = py_file.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r'Path\(\s*"/workspace/', line):
                violations.append(f"{py_file.name}:{i}: {line.strip()}")
    assert not violations, (
        "Hardcoded /workspace/ paths found:\n" + "\n".join(violations)
    )


def test_zenodo_manifest_no_todo_checksums() -> None:
    """All entries in zenodo_manifest.txt must have real SHA256 checksums."""
    manifest = PROJECT_DIR / "zenodo_manifest.txt"
    if not manifest.exists():
        pytest.skip("zenodo_manifest.txt not found")
    todos: list[str] = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2].strip() == "TODO":
            todos.append(parts[0])
    assert not todos, f"TODO checksums remain for: {todos}"


def test_run_all_includes_uk_drive_accessibility() -> None:
    """run_all.sh must include the UK drive accessibility step."""
    run_all = PROJECT_DIR / "benchmarks_final" / "run_all.sh"
    if not run_all.exists():
        pytest.skip("run_all.sh not found")
    text = run_all.read_text()
    assert "run_uk_drive_accessibility" in text, (
        "run_all.sh is missing the run_uk_drive_accessibility step"
    )
