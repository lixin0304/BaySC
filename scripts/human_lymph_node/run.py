"""
Human Lymph Node A1 (Spatial CITE-seq): MRFC-MFM clustering with locked parameters.

Multi-modal (RNA + Protein), 4035 spots, 6 ground truth types.
Params: lam=100, th=4, t0=15, gamma=5, iter=100, seed=42
Clustering yields K=7 (1 singleton), merged to K=6.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import poisson
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, homogeneity_score,
                             silhouette_score)
from sklearn.preprocessing import normalize as sklearn_normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.similarity import compute_similarity_matrix, compute_neighbor_matrix
from src.core.sampler import run_sampler
from src.evaluation.metrics import compute_morans_i


def compute_similarity_matrix_l1(E):
    """Compute similarity using L1-normalized embedding."""
    n_samples, n_dims = E.shape
    S = (E @ E.T) / n_dims
    S = np.clip(S, -0.9999, 0.9999)
    A = 0.5 * np.log((1 + S) / (1 - S))
    A = np.nan_to_num(A, nan=0.0,
                      posinf=A[np.isfinite(A)].max(),
                      neginf=A[np.isfinite(A)].min())
    return A


def compute_Vn_pochhammer(n_cells, gamma=1.0, lambda_poisson=1.0):
    tmax = n_cells + 10
    kmax = 500
    Vn = np.zeros(tmax + 2, dtype=np.float64)
    for t in range(1, tmax + 1):
        log_terms = []
        for k in range(t, kmax + 1):
            if k == t:
                log_factorial_ratio = 0.0
            else:
                log_factorial_ratio = np.sum(np.log(np.arange(k - t + 1, k + 1, dtype=float)))
            log_pochhammer = np.sum(np.log(k * gamma + np.arange(n_cells, dtype=float)))
            log_poisson_prob = poisson.logpmf(k - 1, lambda_poisson)
            log_terms.append(log_factorial_ratio - log_pochhammer + log_poisson_prob)
        Vn[t] = logsumexp(log_terms) if log_terms else -np.inf
    return Vn


def compute_spari(labels_true, labels_pred, coords, alpha=0.8):
    n = len(labels_true)
    coords_norm = coords.copy().astype(float)
    for i in range(coords.shape[1]):
        min_val = coords_norm[:, i].min()
        max_val = coords_norm[:, i].max()
        if max_val > min_val:
            coords_norm[:, i] = (coords_norm[:, i] - min_val) / (max_val - min_val)

    dist_mat = squareform(pdist(coords_norm))
    f_mat = alpha * np.exp(-dist_mat**2)

    same_true = np.equal.outer(labels_true, labels_true)
    same_pred = np.equal.outer(labels_pred, labels_pred)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    a = np.sum(same_true & same_pred & upper_tri)
    d = np.sum(~same_true & ~same_pred & upper_tri)
    W_b = np.sum(f_mat[same_true & ~same_pred & upper_tri])
    W_c = np.sum(f_mat[~same_true & same_pred & upper_tri])

    numerator = a + d
    denominator = a + d + W_b + W_c
    spRI = numerator / denominator if denominator > 0 else 1.0

    n_pairs = n * (n - 1) // 2
    unique_true, counts_true = np.unique(labels_true, return_counts=True)
    unique_pred, counts_pred = np.unique(labels_pred, return_counts=True)
    n_same_true = np.sum(counts_true * (counts_true - 1)) // 2
    n_same_pred = np.sum(counts_pred * (counts_pred - 1)) // 2
    E_a = n_same_true * n_same_pred / n_pairs if n_pairs > 0 else 0
    n_diff_true = n_pairs - n_same_true
    n_diff_pred = n_pairs - n_same_pred
    E_d = n_diff_true * n_diff_pred / n_pairs if n_pairs > 0 else 0
    E_spRI = (E_a + E_d) / n_pairs if n_pairs > 0 else 0

    spARI = (spRI - E_spRI) / (1 - E_spRI) if (1 - E_spRI) != 0 else 0.0
    return spRI, spARI


def main():
    # Parameters
    lambda1 = 100.0
    threshold = 4.0
    t0 = 15.0
    gamma = 5.0
    n_iterations = 30
    seed = 42
    init_k = 10

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'human_lymph_node')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'Human_Lymph_Node')
    os.makedirs(output_dir, exist_ok=True)

    # Load data - dual modality (RNA + Protein)
    print("Loading data...")
    E_rna = np.load(os.path.join(data_dir, 'rna_embedding.npy'))
    E_protein = np.load(os.path.join(data_dir, 'protein_embedding.npy'))
    pos = np.load(os.path.join(data_dir, 'spatial_coords.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    label_names = np.load(os.path.join(data_dir, 'label_names.npy'), allow_pickle=True)
    n_cells = pos.shape[0]

    print(f"Cells: {n_cells}, RNA: {E_rna.shape}, Protein: {E_protein.shape}")
    print(f"GT K: {len(np.unique(labels))}")

    # Compute similarity matrices using same method as param search
    print("\nComputing similarity matrices...")
    A_rna = compute_similarity_matrix(E_rna, transform='fisher_z', keep_diag=True, method='cosine')
    A_protein = compute_similarity_matrix(E_protein, transform='fisher_z', keep_diag=True, method='cosine')
    print(f"A_rna range: [{A_rna.min():.4f}, {A_rna.max():.4f}]")
    print(f"A_protein range: [{A_protein.min():.4f}, {A_protein.max():.4f}]")

    # Compute neighbor matrix
    W = compute_neighbor_matrix(pos.astype(np.float64), threshold=threshold)

    # Compute Vn
    print("Computing Vn...")
    Vn = compute_Vn_pochhammer(n_cells, gamma=gamma)

    # Run clustering
    print(f"\nRunning clustering: lam={lambda1}, th={threshold}, t0={t0}, gamma={gamma}")
    result = run_sampler(
        A_list=[A_rna, A_protein],
        alpha_list=[1.0, 1.0],
        neighbor_matrix=W,
        lambda1=lambda1,
        Vn=Vn,
        n_iterations=n_iterations,
        init_K=init_k,
        t_0=t0,
        gamma=gamma,
        seed=seed,
        verbose=False
    )

    z = result['cluster_assign']
    K = len(np.unique(z))
    
    # Analyze clusters
    print(f"\nOriginal K={K} clustering:")
    unique_clusters = np.unique(z)
    cluster_info = []
    for c in unique_clusters:
        size = np.sum(z == c)
        cluster_info.append((c, size))
        print(f"  Cluster {c}: {size} cells")
    
    # Find and remove singleton
    singletons = [c for c, size in cluster_info if size == 1]
    print(f"\nSingletons: {singletons}")
    
    # Remove singleton cells and relabel
    if singletons:
        print(f"Removing {len(singletons)} singleton(s)...")
        mask = np.ones(n_cells, dtype=bool)
        for c in singletons:
            mask &= (z != c)
        
        # Keep non-singleton cells
        z_filtered = z[mask]
        pos_filtered = pos[mask]
        labels_filtered = labels[mask]
        E_filtered = np.hstack([E_rna[mask], E_protein[mask]])
        
        # Relabel to consecutive integers
        unique_filtered = np.unique(z_filtered)
        relabel_map = {old: new for new, old in enumerate(unique_filtered, start=1)}
        z_k6 = np.array([relabel_map[c] for c in z_filtered])
        
        n_cells_k6 = len(z_k6)
        K_k6 = len(np.unique(z_k6))
        
        print(f"\nAfter removing singletons: {n_cells_k6} cells, K={K_k6}")
    else:
        z_k6 = z
        pos_filtered = pos
        labels_filtered = labels
        E_filtered = np.hstack([E_rna, E_protein])
        n_cells_k6 = n_cells
        K_k6 = K

    # Analyze K=6 clusters
    print(f"\nK={K_k6} cluster analysis:")
    sizes_k6 = []
    for c in np.unique(z_k6):
        size = np.sum(z_k6 == c)
        sizes_k6.append(size)
        print(f"  Cluster {c}: {size} cells")
    sizes_k6 = sorted(sizes_k6, reverse=True)

    # Compute metrics
    print("\nComputing metrics...")
    W_k6 = compute_neighbor_matrix(pos_filtered.astype(np.float64), threshold=threshold)
    
    ari = adjusted_rand_score(labels_filtered, z_k6)
    ami = adjusted_mutual_info_score(labels_filtered, z_k6)
    nmi = normalized_mutual_info_score(labels_filtered, z_k6)
    homo = homogeneity_score(labels_filtered, z_k6)
    mi, _, _ = compute_morans_i(z_k6, pos_filtered.astype(np.float64), W_k6, threshold=threshold)
    sil = silhouette_score(E_filtered, z_k6) if K_k6 > 1 else -1.0
    _, spari = compute_spari(labels_filtered, z_k6, pos_filtered)

    print(f"\nMetrics (K={K_k6}, {n_cells_k6} cells):")
    print(f"  ARI:        {ari:.4f}")
    print(f"  AMI:        {ami:.4f}")
    print(f"  NMI:        {nmi:.4f}")
    print(f"  Homo:       {homo:.4f}")
    print(f"  MoranI:     {mi:.4f}")
    print(f"  spARI:      {spari:.4f}")
    print(f"  Silhouette: {sil:.4f}")
    print(f"  Cluster sizes: {sizes_k6}")

    # Save clustering results
    results_df = pd.DataFrame({
        'cell_idx': np.arange(n_cells_k6),
        'x': pos_filtered[:, 0],
        'y': pos_filtered[:, 1],
        'ground_truth': labels_filtered,
        'ground_truth_name': [label_names[l] for l in labels_filtered],
        'prediction': z_k6,
        'eval_mask': np.ones(n_cells_k6, dtype=bool),
    })
    csv_path = os.path.join(output_dir, 'clustering_result.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Ground truth
    ax = axes[0]
    unique_gt = np.unique(labels_filtered)
    colors_gt = plt.cm.tab10(np.linspace(0, 1, max(len(unique_gt), 10)))
    for i, lbl in enumerate(unique_gt):
        mask = labels_filtered == lbl
        name = label_names[lbl] if lbl < len(label_names) else f'C{lbl}'
        ax.scatter(pos_filtered[mask, 0], pos_filtered[mask, 1],
                   c=[colors_gt[i % 10]], s=5, alpha=0.7, label=name)
    ax.set_title(f'Ground Truth (K={len(unique_gt)})', fontsize=12)
    ax.legend(fontsize=6, markerscale=2, loc='best', ncol=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    # Prediction
    ax = axes[1]
    unique_pred = np.unique(z_k6)
    colors_pred = plt.cm.tab10(np.linspace(0, 1, max(len(unique_pred), 10)))
    for i, lbl in enumerate(unique_pred):
        mask = z_k6 == lbl
        size = mask.sum()
        ax.scatter(pos_filtered[mask, 0], pos_filtered[mask, 1],
                   c=[colors_pred[i % 10]], s=5, alpha=0.7, label=f'C{lbl} ({size})')
    ax.set_title(f'Prediction K={K_k6} (ARI={ari:.4f})', fontsize=12)
    ax.legend(fontsize=6, markerscale=2, loc='best')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    plt.suptitle('Human Lymph Node - K=6 (singleton merged)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    vis_path = os.path.join(output_dir, 'visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {vis_path}")
    plt.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
