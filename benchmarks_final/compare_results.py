#!/usr/bin/env python3
"""Compare reproduced benchmark JSON results against manuscript reference.

Field-aware comparison that distinguishes structural fields (must match),
accuracy metrics (must be close), and timings (informational only).

Usage:
    python -m benchmarks_final.compare_results --reproduced /path/to/reproduced/
    python -m benchmarks_final.compare_results --reproduced /path/to/reproduced/ --verbose
    python -m benchmarks_final.compare_results --reference paper_figs_final/data/ --reproduced /tmp/repro/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------

TIMING_FIELDS = frozenset({
    "t_hierarchy_build", "t_interaction_build", "t_matvec", "t_matvec_median",
    "t_network_gen", "t_dense_build", "t_dense_matvec", "build_time",
    "build_time_wall", "matvec_time", "hierarchy_build_time",
    "interaction_build_time", "total_runtime_seconds", "peak_memory_bytes",
    "date", "n_workers",
})

MAY_DIFFER_FIELDS = frozenset({
    "total_nodes_explored",
})

CLOSE_MATCH_FIELDS = frozenset({
    "mean_relative_error", "max_relative_error", "relative_rmse",
    "correlation", "p95_relative_error", "p99_relative_error",
    "approx_error", "approx_rmse", "density", "compression_ratio",
    "interaction_mass_fraction", "total_activity", "total_cost_entries",
    # accessibility stats (used when traversing stats sub-dict)
    "min", "mean", "max", "std", "p5", "p50", "p95",
})


def classify(field: str) -> str:
    if field in TIMING_FIELDS:
        return "timing"
    if field in MAY_DIFFER_FIELDS:
        return "may_differ"
    if field in CLOSE_MATCH_FIELDS:
        return "close"
    return "exact"


# ---------------------------------------------------------------------------
# Row-matching composite keys per file
# ---------------------------------------------------------------------------

ROW_KEYS: dict[str, list[str]] = {
    "scaling_results.json": ["n_zones"],
    "scaling_tuned_results.json": ["n_zones", "overlap_factor", "br_multiplier"],
    "error_sensitivity_results.json": [
        "network_size", "base_radius", "overlap_factor",
        "increase_factor", "interaction_fn",
    ],
    "baseline_comparison_25k_results.json": ["method", "method_param", "interaction_fn"],
    "interaction_fn_results.json": ["fn_name", "base_radius", "overlap_factor"],
    "layer_structure_results.json": [
        "network_label", "base_radius", "increase_factor", "overlap_factor",
    ],
    "london_60k_results.json": ["method", "method_param", "interaction_fn"],
    "london_walk_60k_results.json": ["method", "method_param"],
}

ACCESSIBILITY_FILES = frozenset({
    "population_accessibility_results.json",
    "gb_drive_pop_accessibility_results.json",
})

ALL_FILES = list(ROW_KEYS.keys()) + sorted(ACCESSIBILITY_FILES)

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _reldiff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


class FileReport:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.n_rows_ref = 0
        self.n_rows_matched = 0
        self.exact_pass = 0
        self.exact_fail = 0
        self.close_pass = 0
        self.close_fail = 0
        self.timing_skipped = 0
        self.may_differ_warn = 0
        self.max_reldiff = 0.0
        self.max_reldiff_field = ""
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.missing_rows: list[str] = []
        self.extra_rows: list[str] = []
        # Timing: list of (field_name, ref_value, repro_value) for summary
        self.timing_pairs: list[tuple[str, float, float]] = []

    @property
    def passed(self) -> bool:
        return self.exact_fail == 0 and self.close_fail == 0 and not self.missing_rows

    def _track_reldiff(self, rd: float, field: str) -> None:
        if rd > self.max_reldiff:
            self.max_reldiff = rd
            self.max_reldiff_field = field


def compare_scalar(
    ref: Any,
    repro: Any,
    field: str,
    classification: str,
    rtol: float,
    report: FileReport,
    path: str,
    verbose: bool,
) -> None:
    """Compare a single scalar value and update the report."""
    full = f"{path}.{field}" if path else field

    if classification == "timing":
        report.timing_skipped += 1
        if ref is not None and repro is not None:
            # Scalar timing
            if isinstance(ref, (int, float)) and isinstance(repro, (int, float)):
                if isinstance(ref, (int, float)) and ref != 0:
                    report.timing_pairs.append((full, float(ref), float(repro)))
            # List of trials — use median
            elif isinstance(ref, list) and isinstance(repro, list):
                ref_nums = [x for x in ref if isinstance(x, (int, float))]
                repro_nums = [x for x in repro if isinstance(x, (int, float))]
                if ref_nums and repro_nums:
                    report.timing_pairs.append((full, _median(ref_nums), _median(repro_nums)))
        return

    # Both None
    if ref is None and repro is None:
        if classification == "exact":
            report.exact_pass += 1
        else:
            report.close_pass += 1
        return

    # One None
    if ref is None or repro is None:
        msg = f"  FAIL {full}: ref={ref}, repro={repro} (null mismatch)"
        report.failures.append(msg)
        if classification == "exact":
            report.exact_fail += 1
        else:
            report.close_fail += 1
        if verbose:
            print(msg)
        return

    # Lists (e.g. layers array) — compare element-wise
    if isinstance(ref, list) and isinstance(repro, list):
        if len(ref) != len(repro):
            msg = f"  FAIL {full}: list length {len(ref)} vs {len(repro)}"
            report.failures.append(msg)
            report.exact_fail += 1
            if verbose:
                print(msg)
            return
        for i, (rv, pv) in enumerate(zip(ref, repro)):
            if isinstance(rv, dict) and isinstance(pv, dict):
                compare_dict(rv, pv, f"{full}[{i}]", rtol, report, verbose)
            else:
                compare_scalar(rv, pv, str(i), classification, rtol, report, full, verbose)
        return

    # Dicts (nested)
    if isinstance(ref, dict) and isinstance(repro, dict):
        compare_dict(ref, repro, full, rtol, report, verbose)
        return

    # Strings
    if isinstance(ref, str) or isinstance(repro, str):
        if ref == repro:
            report.exact_pass += 1
        else:
            msg = f"  FAIL {full}: '{ref}' != '{repro}'"
            report.failures.append(msg)
            report.exact_fail += 1
            if verbose:
                print(msg)
        return

    # Numeric
    if classification == "exact":
        if ref == repro:
            report.exact_pass += 1
            if verbose:
                print(f"    [exact]  {full}: {ref} OK")
        else:
            msg = f"  FAIL {full}: {ref} != {repro}"
            report.failures.append(msg)
            report.exact_fail += 1
            if verbose:
                print(msg)
    elif classification == "close":
        rd = _reldiff(float(ref), float(repro))
        report._track_reldiff(rd, full)
        if rd <= rtol:
            report.close_pass += 1
            if verbose:
                print(f"    [close]  {full}: reldiff={rd:.2e} OK")
        else:
            msg = f"  FAIL {full}: ref={ref}, repro={repro}, reldiff={rd:.2e} > rtol={rtol}"
            report.failures.append(msg)
            report.close_fail += 1
            if verbose:
                print(msg)
    elif classification == "may_differ":
        rd = _reldiff(float(ref), float(repro))
        report._track_reldiff(rd, full)
        if rd > 0.01:
            msg = f"  WARN {full}: ref={ref}, repro={repro}, reldiff={rd:.2e} (>1%)"
            report.warnings.append(msg)
            report.may_differ_warn += 1
            if verbose:
                print(msg)
        else:
            report.exact_pass += 1
            if verbose:
                print(f"    [~ok]    {full}: reldiff={rd:.2e}")


def compare_dict(
    ref: dict,
    repro: dict,
    path: str,
    rtol: float,
    report: FileReport,
    verbose: bool,
) -> None:
    """Compare two dicts field-by-field."""
    all_keys = sorted(set(ref.keys()) | set(repro.keys()))
    for key in all_keys:
        if key not in ref:
            if verbose:
                print(f"    [extra]  {path}.{key} (only in reproduced)")
            continue
        if key not in repro:
            if verbose:
                print(f"    [missing] {path}.{key} (only in reference)")
            report.failures.append(f"  FAIL {path}.{key}: missing in reproduced")
            report.exact_fail += 1
            continue
        cls = classify(key)
        compare_scalar(ref[key], repro[key], key, cls, rtol, report, path, verbose)


# ---------------------------------------------------------------------------
# File-level comparison
# ---------------------------------------------------------------------------


def compare_list_file(
    ref_data: dict,
    repro_data: dict,
    filename: str,
    rtol: float,
    verbose: bool,
) -> FileReport:
    """Compare a list-of-dicts results file."""
    report = FileReport(filename)
    keys = ROW_KEYS[filename]

    ref_rows = ref_data.get("results", [])
    repro_rows = repro_data.get("results", [])
    report.n_rows_ref = len(ref_rows)

    def make_key(row: dict) -> tuple:
        return tuple(row.get(k) for k in keys)

    ref_index: dict[tuple, dict] = {}
    for row in ref_rows:
        ref_index[make_key(row)] = row

    repro_index: dict[tuple, dict] = {}
    for row in repro_rows:
        repro_index[make_key(row)] = row

    # Check for missing/extra rows
    ref_keys_set = set(ref_index.keys())
    repro_keys_set = set(repro_index.keys())

    for k in sorted(ref_keys_set - repro_keys_set):
        label = ", ".join(f"{kn}={kv}" for kn, kv in zip(keys, k))
        report.missing_rows.append(label)

    for k in sorted(repro_keys_set - ref_keys_set):
        label = ", ".join(f"{kn}={kv}" for kn, kv in zip(keys, k))
        report.extra_rows.append(label)

    # Compare matched rows
    for k in sorted(ref_keys_set & repro_keys_set):
        report.n_rows_matched += 1
        label = ", ".join(f"{kn}={kv}" for kn, kv in zip(keys, k))
        if verbose:
            print(f"  Row: {label}")
        compare_dict(ref_index[k], repro_index[k], label, rtol, report, verbose)

    # Compare metadata (informational)
    if "metadata" in ref_data and "metadata" in repro_data:
        if verbose:
            print("  Metadata:")
        compare_dict(ref_data["metadata"], repro_data["metadata"], "metadata", rtol, report, verbose)

    return report


def compare_accessibility_file(
    ref_data: dict,
    repro_data: dict,
    filename: str,
    rtol: float,
    verbose: bool,
) -> FileReport:
    """Compare an accessibility results file."""
    report = FileReport(filename)
    report.n_rows_ref = 1
    report.n_rows_matched = 1

    # Compare metadata
    if "metadata" in ref_data and "metadata" in repro_data:
        if verbose:
            print("  Metadata:")
        compare_dict(ref_data["metadata"], repro_data["metadata"], "metadata", rtol, report, verbose)

    # Compare results
    ref_results = ref_data.get("results", {})
    repro_results = repro_data.get("results", {})

    for cat in ["population", "workplace"]:
        if cat not in ref_results or cat not in repro_results:
            if cat not in ref_results and cat not in repro_results:
                continue
            report.failures.append(f"  FAIL results.{cat}: missing in {'reproduced' if cat in ref_results else 'reference'}")
            report.exact_fail += 1
            continue
        if verbose:
            print(f"  results.{cat}:")
        compare_dict(ref_results[cat], repro_results[cat], f"results.{cat}", rtol, report, verbose)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_report(report: FileReport) -> None:
    """Print a summary for one file."""
    status = "PASS" if report.passed else "FAIL"
    if report.n_rows_ref > 1:
        print(f"\n{report.filename} ({report.n_rows_ref} rows) [{status}]")
    else:
        print(f"\n{report.filename} [{status}]")

    if report.n_rows_ref > 1:
        print(f"  Rows matched: {report.n_rows_matched}/{report.n_rows_ref}")

    if report.missing_rows:
        print(f"  Missing rows: {len(report.missing_rows)}")
        for r in report.missing_rows[:5]:
            print(f"    - {r}")

    if report.extra_rows:
        print(f"  Extra rows: {len(report.extra_rows)}")
        for r in report.extra_rows[:5]:
            print(f"    - {r}")

    n_exact = report.exact_pass + report.exact_fail
    n_close = report.close_pass + report.close_fail

    if n_exact > 0:
        label = "OK" if report.exact_fail == 0 else f"{report.exact_fail} FAILED"
        print(f"  Exact:   {report.exact_pass}/{n_exact} {label}")

    if n_close > 0:
        label = "OK" if report.close_fail == 0 else f"{report.close_fail} FAILED"
        detail = ""
        if report.max_reldiff > 0 and report.close_fail == 0:
            detail = f"  (max reldiff: {report.max_reldiff:.2e} in {report.max_reldiff_field})"
        print(f"  Close:   {report.close_pass}/{n_close} {label}{detail}")

    if report.timing_pairs:
        ratios = [r / ref for _, ref, r in report.timing_pairs if ref > 0]
        if ratios:
            med_ratio = _median(ratios)
            print(f"  Timing:  {len(report.timing_pairs)} fields (median ratio: {med_ratio:.2f}x reproduced/reference)")
            # Show the key timing fields
            # Group by base field name (strip row prefix) and show most interesting
            # Show one line per distinct field name
            _TIME_SUFFIXES = {"_build", "_time", "_matvec", "_gen", "runtime"}
            shown = set()
            for full, ref_val, repro_val in report.timing_pairs:
                base = full.rsplit(".", 1)[-1]
                if base in shown:
                    continue
                shown.add(base)
                ratio = repro_val / ref_val if ref_val > 0 else float("inf")
                is_time = any(s in base for s in _TIME_SUFFIXES)
                if is_time:
                    if ref_val < 1.0:
                        print(f"           {base}: {ref_val*1000:.1f}ms -> {repro_val*1000:.1f}ms ({ratio:.2f}x)")
                    else:
                        print(f"           {base}: {ref_val:.1f}s -> {repro_val:.1f}s ({ratio:.2f}x)")
                else:
                    print(f"           {base}: {ref_val:,.0f} -> {repro_val:,.0f} ({ratio:.2f}x)")
    elif report.timing_skipped > 0:
        print(f"  Timing:  {report.timing_skipped} fields (no numeric pairs to compare)")

    if report.may_differ_warn > 0:
        print(f"  Warnings: {report.may_differ_warn} (total_nodes_explored divergence)")

    for msg in report.failures:
        print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reproduced benchmark JSON results against reference"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "paper_figs_final" / "data",
        help="Reference JSON directory (default: paper_figs_final/data/)",
    )
    parser.add_argument(
        "--reproduced",
        type=Path,
        required=True,
        help="Reproduced JSON directory",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for close-match fields (default: 1e-6)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all field comparisons, not just failures",
    )
    args = parser.parse_args()

    if not args.reference.is_dir():
        print(f"ERROR: Reference directory not found: {args.reference}", file=sys.stderr)
        sys.exit(2)
    if not args.reproduced.is_dir():
        print(f"ERROR: Reproduced directory not found: {args.reproduced}", file=sys.stderr)
        sys.exit(2)

    print(f"Reference:  {args.reference}")
    print(f"Reproduced: {args.reproduced}")
    print(f"Tolerance:  rtol={args.rtol}")

    reports: list[FileReport] = []
    missing_files: list[str] = []

    for filename in ALL_FILES:
        ref_path = args.reference / filename
        repro_path = args.reproduced / filename

        if not ref_path.exists():
            print(f"\nSKIP {filename}: not in reference directory")
            continue
        if not repro_path.exists():
            missing_files.append(filename)
            print(f"\nMISSING {filename}: not in reproduced directory")
            continue

        with open(ref_path) as f:
            ref_data = json.load(f)
        with open(repro_path) as f:
            repro_data = json.load(f)

        if filename in ACCESSIBILITY_FILES:
            report = compare_accessibility_file(ref_data, repro_data, filename, args.rtol, args.verbose)
        else:
            report = compare_list_file(ref_data, repro_data, filename, args.rtol, args.verbose)

        reports.append(report)
        print_report(report)

    # Summary
    n_pass = sum(1 for r in reports if r.passed)
    n_fail = sum(1 for r in reports if not r.passed)
    n_warn = sum(1 for r in reports if r.may_differ_warn > 0)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(reports)} files compared")
    print(f"  PASS: {n_pass}")
    if n_fail:
        print(f"  FAIL: {n_fail}")
    if n_warn:
        print(f"  WARN: {n_warn} (total_nodes_explored divergence)")
    if missing_files:
        print(f"  MISSING: {len(missing_files)} ({', '.join(missing_files)})")
    print("=" * 60)

    if n_fail > 0 or missing_files:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
