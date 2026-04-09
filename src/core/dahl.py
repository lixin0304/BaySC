"""
Dahl's method for selecting a representative clustering from the MCMC posterior.

Given the co-clustering (membership) matrix averaged over post-burn-in iterations,
selects the single iteration whose membership matrix is closest (least squared error)
to the posterior mean membership matrix.

Reference:
    Dahl, D.B. (2006). Model-Based Clustering for Expression Data via a
    Dirichlet Process Mixture Model.
"""

import numpy as np
from typing import Dict, Optional


def get_dahl(history: Dict[str, np.ndarray],
             burn_in: int = 0,
             K_trace: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Dahl's method: select the iteration whose membership matrix is closest
    to the posterior mean membership matrix.
    
    Args:
        history: dict with keys:
            'z': (n_iterations, n) int32 cluster assignments
            'mu': (n_iterations, M, max_K, max_K) float64 means
            'tau': (n_iterations, M, max_K, max_K) float64 precisions
        burn_in: number of initial iterations to discard
        K_trace: (n_iterations,) array of K per iteration (optional, for logging)
    
    Returns:
        dict with:
            'cluster_assign': (n,) selected clustering
            'mu': (M, K, K) parameters at selected iteration
            'tau': (M, K, K) parameters at selected iteration
            'iter_index': index of selected iteration (0-indexed, relative to full chain)
            'burn_in': burn-in used
    """
    z_history = history['z']
    mu_history = history['mu']
    tau_history = history['tau']
    
    n_iterations, n = z_history.shape
    
    if burn_in >= n_iterations:
        raise ValueError(
            f"burn_in ({burn_in}) >= n_iterations ({n_iterations}). "
            f"No post-burn-in samples available."
        )
    
    # Post-burn-in iterations
    z_post = z_history[burn_in:]
    n_post = z_post.shape[0]
    
    # Compute membership matrices and their average
    # membership[i,j] = 1 if z[i] == z[j], else 0
    membership_avg = np.zeros((n, n), dtype=np.float64)
    for t in range(n_post):
        z_t = z_post[t]
        mem_t = np.equal.outer(z_t, z_t).astype(np.float64)
        membership_avg += mem_t
    membership_avg /= n_post
    
    # Find the iteration with minimum squared error to the average
    min_error = np.inf
    dahl_idx = 0
    for t in range(n_post):
        z_t = z_post[t]
        mem_t = np.equal.outer(z_t, z_t).astype(np.float64)
        error = np.sum((mem_t - membership_avg) ** 2)
        if error < min_error:
            min_error = error
            dahl_idx = t
    
    # Map back to full-chain index
    full_idx = burn_in + dahl_idx
    
    # Extract the selected iteration's state
    z_selected = z_history[full_idx].copy()
    K_selected = int(z_selected.max())
    mu_selected = mu_history[full_idx, :, :K_selected, :K_selected].copy()
    tau_selected = tau_history[full_idx, :, :K_selected, :K_selected].copy()
    
    return {
        'cluster_assign': z_selected,
        'mu': mu_selected,
        'tau': tau_selected,
        'iter_index': full_idx,
        'burn_in': burn_in,
    }


__all__ = ['get_dahl']
