"""
Shared constants, metrics, and style for final paper benchmarks.

All benchmark scripts import from this module for consistency.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# ru_maxrss units differ by platform; Windows lacks the resource module entirely
if sys.platform == "linux":
    import resource
    _RSS_UNIT = 1024       # KB
elif sys.platform == "darwin":
    import resource
    _RSS_UNIT = 1          # bytes
else:
    resource = None        # type: ignore[assignment]
    _RSS_UNIT = 1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

import os as _os
N_WORKERS: int = min(32, _os.cpu_count() or 1)
N_TRIALS: int = 5
N_MATVEC_REPS: int = 10
N_ACTIVITY_TRIALS: int = 5
EPS: float = 1e-10

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent / "paper_figs_final"
DATA_DIR: Path = _ROOT / "data"
FIGURES_DIR: Path = _ROOT / "figures"
TABLES_DIR: Path = _ROOT / "tables"

for _d in (DATA_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def compute_errors(method_vec: np.ndarray, dense_vec: np.ndarray) -> dict[str, float]:
    """Compute error metrics between method result and dense ground truth.

    Returns dict with keys: relative_rmse, mean_relative_error, max_relative_error.
    """
    abs_diff = np.abs(method_vec - dense_vec)
    safe_denom = np.maximum(np.abs(dense_vec), EPS)
    rel_errors = abs_diff / safe_denom

    mean_rel = float(np.mean(rel_errors))
    max_rel = float(np.max(rel_errors))

    dense_norm = float(np.linalg.norm(dense_vec))
    if dense_norm < EPS:
        relative_rmse = 0.0
    else:
        relative_rmse = float(np.linalg.norm(abs_diff) / dense_norm)

    return {
        "relative_rmse": relative_rmse,
        "mean_relative_error": mean_rel,
        "max_relative_error": max_rel,
    }


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------


@contextmanager
def track_memory():
    """Context manager that yields a dict; on exit, stores peak_memory_bytes.

    Uses resource.getrusage(RUSAGE_SELF) peak RSS instead of tracemalloc.
    This captures C-level allocations (NumPy/SciPy buffers) that tracemalloc
    misses, but does NOT capture memory in ProcessPoolExecutor workers (they
    are separate OS processes). Use estimate_shm_bytes() to account for the
    shared-memory CSR graph segment used by parallel scipy workers.

    Limitations:
    - ru_maxrss is a monotonic high-water mark, so the delta is only
      meaningful when track_memory() wraps the dominant allocation (true
      for the first trial in benchmarks; later trials may report 0).
    - Per-worker scratch memory (distance arrays, heaps, result dicts) is
      not captured — typically small relative to the graph and operator.
    """
    result = {"peak_memory_bytes": 0}
    if resource is None:
        yield result
        return
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        yield result
    finally:
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_memory_bytes"] = (rss_after - rss_before) * _RSS_UNIT


def estimate_shm_bytes(hierarchy) -> int:
    """Estimate the shared-memory footprint of the CSR graph during build.

    The scipy backend copies the CSR arrays (data, indices, indptr) into
    /dev/shm for worker processes. These segments are allocated and freed
    inside Hierarchy.__init__, so they don't appear in post-build RSS.
    This function reconstructs their size from the CSR matrix that the
    Hierarchy retains.

    Returns 0 if the hierarchy has no CSR matrix (e.g. networkx backend).
    """
    csr = getattr(hierarchy, "_csr", None)
    if csr is None:
        return 0
    return csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

STYLE = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "font.family": "serif",
}
