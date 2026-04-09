"""
Breast Cancer (10x Visium): MRFC-MFM clustering with locked parameters.

Single modality (RNA only), 3798 spots, 20 ground truth types.
Params: lam=28.0, th=700.0, t0=8.0, gamma=7.0, iter=100, burn_in=50, seed=42

Pipeline: Gibbs sampling -> Dahl's method (posterior representative) -> mDIC
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
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.similarity import compute_similarity_matrix, compute_neighbor_matrix
from src.core.sampler import run_sampler
from src.core.dahl import get_dahl
from src.evaluation.metrics import compute_morans_i
from src.evaluation.model_selection import compute_mdic


def compute_spari(labels_true, labels_pred, coords, alpha=0.8):
    """Compute spRI and spARI."""
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


def main():
    # Parameters
    lambda1 = 28.0
    threshold = 700.0
    t0 = 8.0
    gamma = 7.0
    n_iterations = 100
    burn_in = 50
    seed = 42
    init_k = 20

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'breast_cancer')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'Breast_Cancer')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    E = np.load(os.path.join(data_dir, 'rna_embedding_pca50_zscore.npy'))
    pos = np.load(os.path.join(data_dir, 'spatial_coords.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    label_names = np.load(os.path.join(data_dir, 'label_names.npy'), allow_pickle=True)
    n_cells = pos.shape[0]

    print(f"Cells: {n_cells}, Embedding: {E.shape}")
    print(f"GT K: {len(np.unique(labels))}")
    print(f"Label names: {list(label_names)}")

    # Compute similarity matrix
    print("\nComputing similarity matrix...")
    E_l2 = sklearn_normalize(E, norm='l2', axis=1)
    A = compute_similarity_matrix(E_l2, transform='fisher_z', keep_diag=True, method='cosine')

    # Compute neighbor matrix
    W = compute_neighbor_matrix(pos.astype(np.float64), threshold=threshold)

    # Compute Vn
    print("Computing Vn...")
    Vn = compute_Vn_pochhammer(n_cells, gamma=gamma)

    # Run clustering
    print(f"\nRunning clustering: lam={lambda1}, th={threshold}, t0={t0}, gamma={gamma}")
    print(f"  iterations={n_iterations}, burn_in={burn_in}, seed={seed}")
    result = run_sampler(
        A_list=[A],
        alpha_list=[1.0],
        neighbor_matrix=W,
        lambda1=lambda1,
        Vn=Vn,
        n_iterations=n_iterations,
        init_K=init_k,
        t_0=t0,
        gamma=gamma,
        seed=seed,
        verbose=True
    )

    print(f"\nK trace: {list(result['K_trace'])}")
    print(f"Elapsed: {result['elapsed_time']:.2f}s")

    # --- Dahl's method: select representative clustering from posterior ---
    print("\n" + "=" * 60)
    print("Dahl's method (posterior representative selection)")
    print("=" * 60)
    dahl_result = get_dahl(result['history'], burn_in=burn_in, K_trace=result['K_trace'])
    z_dahl = dahl_result['cluster_assign']
    K_dahl = len(np.unique(z_dahl))
    sizes_dahl = sorted([int(np.sum(z_dahl == c)) for c in np.unique(z_dahl)], reverse=True)
    print(f"Selected iteration: {dahl_result['iter_index']} (burn_in={burn_in})")
    print(f"Dahl K={K_dahl}, Sizes: {sizes_dahl}")

    # --- mDIC: model quality ---
    print("\nComputing mDIC...")
    mdic_result = compute_mdic(result['history'], A, burn_in=burn_in)
    print(f"  mDIC:       {mdic_result['mDIC']:.2f}")
    print(f"  Dev(theta): {mdic_result['Dev_theta']:.2f}")
    print(f"  D_bar:      {mdic_result['D_bar']:.2f}")
    print(f"  p_D:        {mdic_result['p_D']:.2f}")
    print(f"  K_mean:     {mdic_result['K_mean']:.2f}")
    print(f"  K_median:   {mdic_result['K_median']:.1f}")

    # --- Compare last-iteration vs Dahl ---
    z_last = result['cluster_assign']
    z = z_dahl  # Use Dahl as the final result

    K = K_dahl
    sizes = sizes_dahl
    singletons = sum(1 for s in sizes if s == 1)

    # Evaluate: last iteration
    ari_last = adjusted_rand_score(labels, z_last)
    # Evaluate: Dahl
    ari = adjusted_rand_score(labels, z)
    ami = adjusted_mutual_info_score(labels, z)
    nmi = normalized_mutual_info_score(labels, z)
    homo = homogeneity_score(labels, z)
    mi, _, _ = compute_morans_i(z, pos.astype(np.float64), W, threshold=threshold)
    sil = silhouette_score(E_l2, z) if K > 1 else -1.0

    print("Computing spARI...")
    _, spari = compute_spari(labels, z, pos)

    print(f"\n{'='*60}")
    print(f"Last iteration: ARI={ari_last:.4f} (K={len(np.unique(z_last))})")
    print(f"Dahl's method:  ARI={ari:.4f} (K={K})")
    print(f"{'='*60}")
    print(f"\nMetrics (Dahl, all {n_cells} spots):")
    print(f"  ARI:        {ari:.4f}")
    print(f"  AMI:        {ami:.4f}")
    print(f"  NMI:        {nmi:.4f}")
    print(f"  Homo:       {homo:.4f}")
    print(f"  MoranI:     {mi:.4f}")
    print(f"  spARI:      {spari:.4f}")
    print(f"  Silhouette: {sil:.4f}")
    print(f"  Singletons: {singletons}")
    print(f"  mDIC:       {mdic_result['mDIC']:.2f}")

    # Save clustering results (standard format)
    results_df = pd.DataFrame({
        'cell_idx': np.arange(n_cells),
        'x': pos[:, 0],
        'y': pos[:, 1],
        'ground_truth': labels,
        'ground_truth_name': [label_names[l] for l in labels],
        'prediction': z,
        'eval_mask': np.ones(n_cells, dtype=bool),  # All spots have labels
    })
    results_path = os.path.join(output_dir, 'clustering_result.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved clustering results: {results_path}")

    # Visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Ground truth
    ax = axes[0]
    unique_gt = np.unique(labels)
    colors_gt = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, lbl in enumerate(unique_gt):
        mask = labels == lbl
        name = label_names[lbl] if lbl < len(label_names) else f'C{lbl}'
        ax.scatter(pos[mask, 0], pos[mask, 1],
                   c=[colors_gt[i]], s=8, alpha=0.7, label=name)
    ax.set_title(f'Ground Truth (K={len(unique_gt)})')
    ax.legend(fontsize=5, markerscale=2, loc='best', ncol=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    # Prediction
    ax = axes[1]
    unique_pred = np.unique(z)
    n_pred = len(unique_pred)
    colors_pred = plt.cm.tab20(np.linspace(0, 1, max(n_pred, 20)))
    for i, lbl in enumerate(unique_pred):
        mask = z == lbl
        size = mask.sum()
        ax.scatter(pos[mask, 0], pos[mask, 1],
                   c=[colors_pred[i % len(colors_pred)]], s=8, alpha=0.7, label=f'C{lbl} ({size})')
    ax.set_title(f'Prediction K={K} (ARI={ari:.4f})')
    ax.legend(fontsize=5, markerscale=2, loc='best', ncol=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    plt.tight_layout()
    vis_path = os.path.join(output_dir, 'visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {vis_path}")
    plt.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
