"""
Similarity matrix computation for multi-modal MRFC-MFM.

Supports:
- logit transform: A = log((x-min+eps)/(max-x+eps))
- fisher_z transform: A = arctanh(clip(S, -0.9999, 0.9999))

Configurable diagonal handling (keep_diag).
"""

import numpy as np


def compute_similarity_matrix(embedding, transform='fisher_z', keep_diag=True, eps=1e-10, method='inner_product_scaled'):
    """
    Compute similarity matrix from embedding.
    
    Args:
        embedding: Embedding matrix (n_samples, n_dims)
        transform: 'logit' or 'fisher_z'
        keep_diag: Whether to keep diagonal values (True) or set to 0 (False)
        eps: Small value for numerical stability
        method: 
            - 'inner_product_scaled': R = S / d (for L1-normalized embeddings)
            - 'cosine': R = S directly (for L2-normalized embeddings, inner product = cosine similarity)
            - 'pearson': Pearson correlation coefficient
    
    Returns:
        A: Similarity matrix (n_samples, n_samples)
    """
    n_samples, n_dims = embedding.shape
    
    # Step 1: Compute inner product
    inner_prod = embedding @ embedding.T
    
    print(f"Inner product:")
    print(f"  Shape: {inner_prod.shape}")
    print(f"  Range: [{inner_prod.min():.4f}, {inner_prod.max():.4f}]")
    print(f"  Diagonal mean: {np.diag(inner_prod).mean():.4f}")
    
    if method == 'pearson':
        # Pearson correlation coefficient
        X_centered = embedding - embedding.mean(axis=1, keepdims=True)
        X_std = embedding.std(axis=1, keepdims=True)
        X_std[X_std < eps] = 1.0
        X_norm = X_centered / X_std
        R = (X_norm @ X_norm.T) / n_dims
        print(f"Pearson correlation: R range [{R.min():.4f}, {R.max():.4f}]")
    elif method == 'cosine':
        # For L2-normalized embeddings, inner product IS cosine similarity
        # Already in range [-1, 1], no scaling needed
        R = inner_prod
        print(f"Cosine similarity (no scaling): R range [{R.min():.4f}, {R.max():.4f}]")
    else:
        # Default: scale inner product by dimension
        # R = S / d (for L1-normalized or unnormalized embeddings)
        R = inner_prod / n_dims
        print(f"Scaled by dimension (d={n_dims}): R range [{R.min():.4f}, {R.max():.4f}]")
    
    if transform == 'fisher_z':
        # Numerical stability - clip to valid range for arctanh
        R_clipped = np.clip(R, -0.9999, 0.9999)
        
        # Fisher's Z-transform: arctanh(R) = 0.5 * log((1+R)/(1-R))
        A = 0.5 * np.log((1 + R_clipped) / (1 - R_clipped))
        
        print(f"Fisher Z transform:")
        print(f"  Clipped R range: [{R_clipped.min():.4f}, {R_clipped.max():.4f}]")
        
    elif transform == 'logit':
        # Logit transformation
        # Exclude diagonal for min/max computation
        mask = ~np.eye(n_samples, dtype=bool)
        x_min = inner_prod[mask].min()
        x_max = inner_prod[mask].max()
        
        # Logit transform: log((x-min+eps)/(max-x+eps))
        A = np.log((inner_prod - x_min + eps) / (x_max - inner_prod + eps))
        
        # Handle infinities
        finite_vals = A[np.isfinite(A)]
        if len(finite_vals) > 0:
            A[np.isposinf(A)] = finite_vals.max()
            A[np.isneginf(A)] = finite_vals.min()
        
        print(f"Logit transform:")
        print(f"  x_min (off-diag): {x_min:.4f}")
        print(f"  x_max (off-diag): {x_max:.4f}")
    else:
        raise ValueError(f"Unknown transform: {transform}")
    
    # Handle diagonal
    if not keep_diag:
        np.fill_diagonal(A, 0)
        print(f"  Diagonal set to 0")
    
    # Handle any remaining NaN/Inf
    A = np.nan_to_num(A, nan=0.0, posinf=A[np.isfinite(A)].max(), neginf=A[np.isfinite(A)].min())
    
    print(f"Final A:")
    print(f"  Range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"  Mean (off-diag): {A[~np.eye(n_samples, dtype=bool)].mean():.4f}")
    print(f"  Std (off-diag): {A[~np.eye(n_samples, dtype=bool)].std():.4f}")
    
    return A


def compute_neighbor_matrix(pos, threshold=3.0):
    """
    Compute spatial neighbor matrix.
    
    Args:
        pos: Spatial coordinates (n_samples, 2)
        threshold: Distance threshold for neighbors
    
    Returns:
        neighbor_matrix: Binary neighbor matrix (n_samples, n_samples)
    """
    from scipy.spatial.distance import cdist
    
    distance = cdist(pos, pos, metric='euclidean')
    neighbor_matrix = (distance <= threshold).astype(np.int32)
    np.fill_diagonal(neighbor_matrix, 0)
    
    avg_neighbors = neighbor_matrix.sum(axis=1).mean()
    print(f"Neighbor matrix (threshold={threshold}):")
    print(f"  Average neighbors: {avg_neighbors:.1f}")
    print(f"  Range: [{neighbor_matrix.sum(axis=1).min()}, {neighbor_matrix.sum(axis=1).max()}]")
    
    return neighbor_matrix


__all__ = ['compute_similarity_matrix', 'compute_neighbor_matrix']
