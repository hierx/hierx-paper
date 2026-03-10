"""
Baseline methods for comparison with the hierarchical operator.

Implements two alternatives to the hierarchical sparse-plus-correction approach:
1. CutoffBaseline — Distance cutoff truncation (sparse exact within radius)
2. NystromBaseline — Nyström low-rank approximation

Both classes expose a ``.matvec(activity)`` method matching ``InteractionHierarchy``.

Usage:
    from benchmarks.baselines import CutoffBaseline, NystromBaseline

    cutoff = CutoffBaseline(network, zones, interaction_fn, cutoff_radius=5000)
    result = cutoff.matvec(activity)

    nystrom = NystromBaseline(network, zones, interaction_fn, n_landmarks=50)
    result = nystrom.matvec(activity)
"""

import time
from typing import Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from hierx.backends import convert_nx_to_csr


class CutoffBaseline:
    """Distance cutoff truncation baseline.

    Computes exact shortest-path distances within a cutoff radius using
    scipy's C-level Dijkstra, applies the interaction function, and stores
    the result as a sparse CSR matrix. Interactions beyond the cutoff
    are assumed to be zero.

    Parameters
    ----------
    network : nx.Graph
        Network with 'cost' edge attribute.
    zones : list[int]
        Ordered list of zone IDs.
    interaction_fn : Callable[[float], float]
        Distance-decay function mapping cost to interaction value.
    cutoff_radius : float
        Maximum shortest-path distance. Interactions beyond this are zero.
    """

    def __init__(
        self,
        network: nx.Graph,
        zones: list[int],
        interaction_fn: Callable[[float], float],
        cutoff_radius: float,
    ) -> None:
        t0 = time.perf_counter()

        self.n = len(zones)
        self.zones = zones
        self.zone_to_idx = {z: i for i, z in enumerate(zones)}

        csr, _idx_to_node, node_to_idx = convert_nx_to_csr(network)
        zone_csr_indices = np.array([node_to_idx[z] for z in zones])

        dist_block = csg.dijkstra(
            csr, directed=False, indices=zone_csr_indices, limit=cutoff_radius,
        )
        # Slice to zone-to-zone distances
        dist_block = dist_block[:, zone_csr_indices]

        # Build sparse matrix directly from finite entries (no dense intermediate)
        rows_idx, cols_idx = np.where(np.isfinite(dist_block))
        self.total_nodes_explored = len(rows_idx)
        finite_dists = dist_block[rows_idx, cols_idx]

        vectorized_fn = np.vectorize(interaction_fn)
        vals = vectorized_fn(finite_dists)

        # Keep only positive interaction values
        pos_mask = vals > 0
        self.interaction_matrix = sp.csr_matrix(
            (vals[pos_mask], (rows_idx[pos_mask], cols_idx[pos_mask])),
            shape=(self.n, self.n),
        )

        self.build_time = time.perf_counter() - t0
        self.n_entries = self.interaction_matrix.nnz
        self.interaction_mass = float(self.interaction_matrix.sum())

    def matvec(self, activity: np.ndarray) -> np.ndarray:
        """Compute sparse matrix-vector product.

        Parameters
        ----------
        activity : np.ndarray
            Activity vector of length n.

        Returns
        -------
        np.ndarray
            Interaction result of length n.
        """
        return np.asarray(self.interaction_matrix @ activity).ravel()


class NystromBaseline:
    """Nyström low-rank approximation baseline.

    Samples a set of landmark zones, computes exact distances from all
    zones to landmarks, and builds a low-rank approximation of the
    interaction matrix: K ≈ C @ pinv(W) @ C.T, where C is the
    zone-to-landmark interaction matrix and W is the landmark-to-landmark
    submatrix.

    Parameters
    ----------
    network : nx.Graph
        Network with 'cost' edge attribute.
    zones : list[int]
        Ordered list of zone IDs.
    interaction_fn : Callable[[float], float]
        Distance-decay function mapping cost to interaction value.
    n_landmarks : int
        Number of landmark zones to sample.
    seed : int
        Random seed for landmark selection.
    """

    def __init__(
        self,
        network: nx.Graph,
        zones: list[int],
        interaction_fn: Callable[[float], float],
        n_landmarks: int,
        seed: int = 42,
    ) -> None:
        t0 = time.perf_counter()

        self.n = len(zones)
        self.n_landmarks = min(n_landmarks, self.n)
        self.zones = zones
        self.zone_to_idx = {z: i for i, z in enumerate(zones)}

        csr, _idx_to_node, node_to_idx = convert_nx_to_csr(network)
        zone_csr_indices = np.array([node_to_idx[z] for z in zones])

        rng = np.random.RandomState(seed)
        landmark_indices = rng.choice(self.n, size=self.n_landmarks, replace=False)
        landmark_indices.sort()
        landmark_csr_indices = zone_csr_indices[landmark_indices]

        # Batch Dijkstra for all landmarks at once
        dist_all = csg.dijkstra(
            csr, directed=False, indices=landmark_csr_indices,
        )
        # Slice to zone columns: dist_all shape (m, N) -> (m, n)
        dist_zones = dist_all[:, zone_csr_indices]  # (m, n)

        # Compute cross-similarity matrix C (n × m)
        vectorized_fn = np.vectorize(interaction_fn)
        C = vectorized_fn(dist_zones).T  # (n, m)

        # Extract landmark submatrix W (m × m)
        W = C[landmark_indices, :]

        # Compute pseudo-inverse of W
        self.W_inv = np.linalg.pinv(W)
        self.C = C

        self.build_time = time.perf_counter() - t0
        # Each landmark's Dijkstra explores the full graph without cutoff
        n_graph = csr.shape[0]
        self.total_nodes_explored = self.n_landmarks * n_graph
        # Compute interaction mass without materialising n×n matrix:
        # sum(C @ W_inv @ C.T) = ones.T @ C @ W_inv @ C.T @ ones
        ones = np.ones(self.n)
        self.interaction_mass = float(ones @ (C @ (self.W_inv @ (C.T @ ones))))

    def matvec(self, activity: np.ndarray) -> np.ndarray:
        """Compute low-rank matrix-vector product: C @ (W_inv @ (C.T @ x)).

        Parameters
        ----------
        activity : np.ndarray
            Activity vector of length n.

        Returns
        -------
        np.ndarray
            Interaction result of length n.
        """
        # O(nm) computation via sequential matrix-vector products
        step1 = self.C.T @ activity  # (m,)
        step2 = self.W_inv @ step1  # (m,)
        result = self.C @ step2  # (n,)
        return result
