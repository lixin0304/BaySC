"""
Evaluation metrics for spatial clustering.

Metrics:
    1. ARI (Adjusted Rand Index)
    2. NMI (Normalized Mutual Information)
    3. AMI (Adjusted Mutual Information)
    4. Moran's I (Spatial Autocorrelation)
    5. LISI (Local Inverse Simpson's Index)
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.spatial.distance import cdist
from typing import Optional, Dict, Tuple


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    
    ARI measures the similarity between two clusterings, adjusted for chance.
    Range: [-1, 1], where 1 = perfect match, 0 = random, <0 = worse than random.
    
    Args:
        labels_true: Ground truth labels (n,)
        labels_pred: Predicted labels (n,)
    
    Returns:
        ARI score
    """
    return adjusted_rand_score(labels_true, labels_pred)


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    
    NMI measures the mutual information between two clusterings, normalized.
    Range: [0, 1], where 1 = perfect match.
    
    Args:
        labels_true: Ground truth labels (n,)
        labels_pred: Predicted labels (n,)
    
    Returns:
        NMI score
    """
    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Mutual Information (AMI).
    
    AMI is mutual information adjusted for chance, similar to ARI but information-theoretic.
    Range: [-1, 1], where 1 = perfect match, 0 = random.
    
    Args:
        labels_true: Ground truth labels (n,)
        labels_pred: Predicted labels (n,)
    
    Returns:
        AMI score
    """
    return adjusted_mutual_info_score(labels_true, labels_pred)


def compute_morans_i(labels: np.ndarray, 
                     coords: np.ndarray,
                     weight_matrix: Optional[np.ndarray] = None,
                     threshold: float = 4.0) -> Tuple[float, float, float]:
    """
    Compute Moran's I for spatial autocorrelation of cluster labels.
    
    Moran's I measures spatial autocorrelation - whether nearby cells tend to 
    have similar cluster labels.
    Range: [-1, 1], where:
        - 1 = perfect positive spatial autocorrelation (clustered)
        - 0 = random spatial distribution
        - -1 = perfect negative spatial autocorrelation (dispersed)
    
    Args:
        labels: Cluster labels (n,)
        coords: Spatial coordinates (n, 2)
        weight_matrix: Optional precomputed spatial weight matrix (n, n)
        threshold: Distance threshold for neighbors if weight_matrix not provided
    
    Returns:
        Tuple of (Moran's I, expected I, z-score)
    """
    n = len(labels)
    
    # Convert labels to numeric if needed
    if labels.dtype.kind not in ['i', 'u', 'f']:
        unique_labels = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
    
    # Create weight matrix if not provided
    if weight_matrix is None:
        dist_matrix = cdist(coords, coords, metric='euclidean')
        weight_matrix = (dist_matrix <= threshold).astype(float)
        np.fill_diagonal(weight_matrix, 0)
    
    # Row-standardize weights
    row_sums = weight_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = weight_matrix / row_sums
    
    # Compute Moran's I
    y = labels.astype(float)
    y_mean = y.mean()
    y_dev = y - y_mean
    
    # Numerator: sum of weighted cross-products
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            numerator += W[i, j] * y_dev[i] * y_dev[j]
    
    # Denominator: sum of squared deviations
    denominator = np.sum(y_dev ** 2)
    
    if denominator == 0:
        return 0.0, 0.0, 0.0
    
    # Moran's I
    S0 = W.sum()
    if S0 == 0:
        return 0.0, 0.0, 0.0
    
    I = (n / S0) * (numerator / denominator)
    
    # Expected value under null hypothesis
    E_I = -1.0 / (n - 1)
    
    # Variance (under normality assumption)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    
    k = (np.sum(y_dev ** 4) / n) / ((np.sum(y_dev ** 2) / n) ** 2)
    
    var_I = (n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2) - 
             k * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0**2)) / \
            ((n - 1) * (n - 2) * (n - 3) * S0**2) - E_I**2
    
    if var_I <= 0:
        z_score = 0.0
    else:
        z_score = (I - E_I) / np.sqrt(var_I)
    
    return I, E_I, z_score


def compute_lisi(coords: np.ndarray,
                 labels: np.ndarray,
                 perplexity: int = 30) -> Tuple[float, np.ndarray]:
    """
    Compute Local Inverse Simpson's Index (LISI).
    
    LISI measures the effective number of clusters in the local neighborhood.
    Higher LISI = more mixing of clusters locally.
    Lower LISI = better spatial separation of clusters.
    
    For spatial clustering, we want LOW LISI (clusters are spatially coherent).
    
    Args:
        coords: Spatial coordinates (n, 2)
        labels: Cluster labels (n,)
        perplexity: Number of neighbors to consider (default=30)
    
    Returns:
        Tuple of (mean LISI, per-cell LISI array)
    """
    n = len(labels)
    k = min(perplexity * 3, n - 1)  # Number of neighbors to consider
    
    # Compute pairwise distances
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    # Get k nearest neighbors for each cell
    lisi_scores = np.zeros(n)
    
    for i in range(n):
        # Get distances to all other cells
        distances = dist_matrix[i, :]
        distances[i] = np.inf  # Exclude self
        
        # Get k nearest neighbors
        neighbor_idx = np.argsort(distances)[:k]
        neighbor_labels = labels[neighbor_idx]
        neighbor_dists = distances[neighbor_idx]
        
        # Compute Gaussian kernel weights (similar to t-SNE)
        # Find sigma using binary search to match perplexity
        sigma = _find_sigma(neighbor_dists, perplexity)
        
        # Compute weights
        weights = np.exp(-neighbor_dists**2 / (2 * sigma**2))
        weights = weights / weights.sum()
        
        # Compute Simpson's Index: sum of squared proportions per cluster
        unique_labels = np.unique(neighbor_labels)
        simpson = 0.0
        for label in unique_labels:
            p = weights[neighbor_labels == label].sum()
            simpson += p ** 2
        
        # Inverse Simpson's Index
        if simpson > 0:
            lisi_scores[i] = 1.0 / simpson
        else:
            lisi_scores[i] = 1.0
    
    return lisi_scores.mean(), lisi_scores


def _find_sigma(distances: np.ndarray, perplexity: float, 
                tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    Find sigma for Gaussian kernel to match target perplexity.
    
    Args:
        distances: Distances to neighbors
        perplexity: Target perplexity
        tol: Tolerance for convergence
        max_iter: Maximum iterations
    
    Returns:
        Optimal sigma value
    """
    target_entropy = np.log(perplexity)
    
    sigma_min = 1e-10
    sigma_max = 1e10
    sigma = 1.0
    
    for _ in range(max_iter):
        # Compute weights with current sigma
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights_sum = weights.sum()
        
        if weights_sum == 0:
            sigma *= 2
            continue
        
        weights = weights / weights_sum
        
        # Compute entropy
        weights_nonzero = weights[weights > 0]
        entropy = -np.sum(weights_nonzero * np.log(weights_nonzero))
        
        # Binary search
        if np.abs(entropy - target_entropy) < tol:
            break
        
        if entropy < target_entropy:
            sigma_min = sigma
            sigma = (sigma + sigma_max) / 2
        else:
            sigma_max = sigma
            sigma = (sigma + sigma_min) / 2
    
    return sigma


def compute_all_metrics(labels_true: np.ndarray,
                        labels_pred: np.ndarray,
                        coords: np.ndarray,
                        neighbor_matrix: Optional[np.ndarray] = None,
                        threshold: float = 4.0,
                        perplexity: int = 30) -> Dict[str, float]:
    """
    Compute all 5 evaluation metrics.
    
    Args:
        labels_true: Ground truth labels (n,)
        labels_pred: Predicted labels (n,)
        coords: Spatial coordinates (n, 2)
        neighbor_matrix: Optional precomputed neighbor matrix (n, n)
        threshold: Distance threshold for Moran's I
        perplexity: Perplexity for LISI
    
    Returns:
        Dictionary with all metrics:
            - ARI: Adjusted Rand Index
            - NMI: Normalized Mutual Information
            - AMI: Adjusted Mutual Information
            - MoransI: Moran's I statistic
            - MoransI_zscore: Moran's I z-score
            - LISI: Mean Local Inverse Simpson's Index
    """
    metrics = {}
    
    # Clustering quality metrics (require ground truth)
    if labels_true is not None:
        metrics['ARI'] = compute_ari(labels_true, labels_pred)
        metrics['NMI'] = compute_nmi(labels_true, labels_pred)
        metrics['AMI'] = compute_ami(labels_true, labels_pred)
    else:
        metrics['ARI'] = -1.0
        metrics['NMI'] = -1.0
        metrics['AMI'] = -1.0
    
    # Spatial metrics (do not require ground truth)
    morans_i, morans_e, morans_z = compute_morans_i(
        labels_pred, coords, neighbor_matrix, threshold
    )
    metrics['MoransI'] = morans_i
    metrics['MoransI_expected'] = morans_e
    metrics['MoransI_zscore'] = morans_z
    
    lisi_mean, _ = compute_lisi(coords, labels_pred, perplexity)
    metrics['LISI'] = lisi_mean
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the output
    """
    print(f"\n{title}")
    print("-" * 50)
    
    # Clustering quality
    print("Clustering Quality (vs Ground Truth):")
    print(f"  ARI (Adjusted Rand Index):     {metrics.get('ARI', -1):.4f}")
    print(f"  NMI (Normalized Mutual Info):  {metrics.get('NMI', -1):.4f}")
    print(f"  AMI (Adjusted Mutual Info):    {metrics.get('AMI', -1):.4f}")
    
    # Spatial quality
    print("\nSpatial Quality:")
    print(f"  Moran's I:                     {metrics.get('MoransI', 0):.4f}")
    print(f"  Moran's I z-score:             {metrics.get('MoransI_zscore', 0):.4f}")
    print(f"  LISI (lower = better spatial): {metrics.get('LISI', 0):.4f}")
    print("-" * 50)


__all__ = [
    'compute_ari',
    'compute_nmi', 
    'compute_ami',
    'compute_morans_i',
    'compute_lisi',
    'compute_all_metrics',
    'print_metrics',
]
