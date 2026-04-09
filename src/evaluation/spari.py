"""
Python implementation of spARI (Spatially Aware Adjusted Rand Index).

Based on the R package: https://github.com/yinqiaoyan/spARI

spARI extends ARI by weighting disagreement pairs according to spatial distance:
  - SG pairs (same cluster, different ground truth): weighted by f(d) = alpha * exp(-d^2)
  - GS pairs (same ground truth, different cluster): weighted by h(d) = alpha * (1 - exp(-d^2))

Coordinates are normalized to [0,1] before distance computation.

Reference:
  Yan et al. "spARI: a spatially aware Rand index for clustering evaluation 
  in spatial transcriptomics"
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from typing import Optional, Callable, Tuple
from math import comb


def _default_f(t: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Default f function: alpha * exp(-t^2). Applied to SG pair distances."""
    return alpha * np.exp(-t**2)


def _default_h(t: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Default h function: alpha * (1 - exp(-t^2)). Applied to GS pair distances."""
    return alpha * (1.0 - np.exp(-t**2))


def _generate_sg_pairs(c_labels: np.ndarray, r_labels: np.ndarray) -> np.ndarray:
    """
    Generate SG pairs: same cluster in c_labels, different group in r_labels.
    
    Args:
        c_labels: Clustering labels (n,), integer
        r_labels: Reference labels (n,), integer
    
    Returns:
        pairs: (M, 2) array of index pairs
    """
    i_vec = []
    j_vec = []
    # Group by c_labels
    unique_c = np.unique(c_labels)
    for c in unique_c:
        indices = np.where(c_labels == c)[0]
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                idx_i = indices[ii]
                idx_j = indices[jj]
                if r_labels[idx_i] != r_labels[idx_j]:
                    i_vec.append(idx_i)
                    j_vec.append(idx_j)
    if len(i_vec) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([i_vec, j_vec])


def _generate_gs_pairs(c_labels: np.ndarray, r_labels: np.ndarray) -> np.ndarray:
    """
    Generate GS pairs: same group in r_labels, different cluster in c_labels.
    
    Args:
        c_labels: Clustering labels (n,), integer
        r_labels: Reference labels (n,), integer
    
    Returns:
        pairs: (M, 2) array of index pairs
    """
    i_vec = []
    j_vec = []
    # Group by r_labels
    unique_r = np.unique(r_labels)
    for r in unique_r:
        indices = np.where(r_labels == r)[0]
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                idx_i = indices[ii]
                idx_j = indices[jj]
                if c_labels[idx_i] != c_labels[idx_j]:
                    i_vec.append(idx_i)
                    j_vec.append(idx_j)
    if len(i_vec) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([i_vec, j_vec])


def compute_spari(r_labels: np.ndarray,
                  c_labels: np.ndarray,
                  coords: Optional[np.ndarray] = None,
                  dist_mat: Optional[np.ndarray] = None,
                  f_func: Optional[Callable] = None,
                  h_func: Optional[Callable] = None,
                  alpha: float = 0.8) -> Tuple[float, float]:
    """
    Compute spRI and spARI (spatially aware Rand Index and its adjusted version).
    
    Args:
        r_labels: Reference (ground truth) labels (n,)
        c_labels: Clustering labels (n,)
        coords: Spatial coordinates (n, 2). If provided and dist_mat is None,
                coordinates are normalized to [0,1] and Euclidean distance is computed.
        dist_mat: Precomputed distance matrix (n, n). If provided, used directly.
        f_func: Weight function for SG pairs. Default: alpha * exp(-t^2)
        h_func: Weight function for GS pairs. Default: alpha * (1 - exp(-t^2))
        alpha: Coefficient in (0, 1) for default f and h functions. Default: 0.8.
    
    Returns:
        (spRI, spARI) tuple
    """
    n = len(r_labels)
    assert len(c_labels) == n, "r_labels and c_labels must have same length"
    
    # Trivial case: all same label
    if len(np.unique(r_labels)) == 1 and len(np.unique(c_labels)) == 1:
        return 1.0, 1.0
    
    # Distance matrix
    if dist_mat is None:
        assert coords is not None, "Either coords or dist_mat must be provided"
        assert coords.shape[0] == n
        # Normalize coordinates to [0, 1]
        coords_norm = coords.copy().astype(np.float64)
        for d in range(coords_norm.shape[1]):
            cmin = coords_norm[:, d].min()
            cmax = coords_norm[:, d].max()
            if cmax > cmin:
                coords_norm[:, d] = (coords_norm[:, d] - cmin) / (cmax - cmin)
            else:
                coords_norm[:, d] = 0.0
        dist_mat = squareform(pdist(coords_norm, metric='euclidean'))
    
    # Convert labels to integer encoding
    r_unique = np.unique(r_labels)
    c_unique = np.unique(c_labels)
    r_map = {v: i for i, v in enumerate(r_unique)}
    c_map = {v: i for i, v in enumerate(c_unique)}
    r_int = np.array([r_map[v] for v in r_labels])
    c_int = np.array([c_map[v] for v in c_labels])
    
    # Weight functions
    if f_func is None:
        f_func = lambda t: _default_f(t, alpha)
    if h_func is None:
        h_func = lambda t: _default_h(t, alpha)
    
    n_obj_choose = comb(n, 2)
    
    # Generate SG and GS pairs
    sg_pairs = _generate_sg_pairs(c_int, r_int)
    gs_pairs = _generate_gs_pairs(c_int, r_int)
    
    # Compute weighted sums for SG and GS pairs
    sums_f_and_h = 0.0
    if sg_pairs.shape[0] > 0:
        dist_sg = dist_mat[sg_pairs[:, 0], sg_pairs[:, 1]]
        nonzero_sg = dist_sg[dist_sg != 0]
        if len(nonzero_sg) > 0:
            sums_f_and_h += np.sum(f_func(nonzero_sg))
    
    if gs_pairs.shape[0] > 0:
        dist_gs = dist_mat[gs_pairs[:, 0], gs_pairs[:, 1]]
        nonzero_gs = dist_gs[dist_gs != 0]
        if len(nonzero_gs) > 0:
            sums_f_and_h += np.sum(h_func(nonzero_gs))
    
    # Contingency table
    K_R = len(r_unique)
    K_C = len(c_unique)
    nij = np.zeros((K_R, K_C), dtype=np.int64)
    for i in range(n):
        nij[r_int[i], c_int[i]] += 1
    
    parSum_r = nij.sum(axis=1)  # row sums
    parSum_c = nij.sum(axis=0)  # col sums
    parSum_r_sqsum = np.sum(parSum_r.astype(np.float64)**2)
    parSum_c_sqsum = np.sum(parSum_c.astype(np.float64)**2)
    
    # Number of agreement pairs
    num_A = n_obj_choose + np.sum(nij.astype(np.float64)**2) - 0.5 * (parSum_r_sqsum + parSum_c_sqsum)
    
    # spRI
    spRI = (num_A + sums_f_and_h) / n_obj_choose
    
    # spARI
    p = 0.5 * (parSum_r_sqsum - n) / n_obj_choose
    q = 0.5 * (parSum_c_sqsum - n) / n_obj_choose
    
    # Total f and h sums over all pairs
    dist_lower = dist_mat[np.tril_indices(n, k=-1)]
    nonzero_lower = dist_lower[dist_lower != 0]
    sum_f_total = np.sum(f_func(nonzero_lower)) if len(nonzero_lower) > 0 else 0.0
    sum_h_total = np.sum(h_func(nonzero_lower)) if len(nonzero_lower) > 0 else 0.0
    
    E_spRI = (p * q + (1 - p) * (1 - q) +
              (1 - p) * q * sum_f_total / n_obj_choose +
              p * (1 - q) * sum_h_total / n_obj_choose)
    
    if E_spRI == 1.0:
        return 1.0, 1.0
    
    spARI = (spRI - E_spRI) / (1.0 - E_spRI)
    
    return float(spRI), float(spARI)


__all__ = ['compute_spari']
