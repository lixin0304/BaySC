"""
MISAR-seq Mouse E15 Brain: MRFC-MFM clustering with locked parameters.

Multi-modal (ATAC + RNA), 4018 spots, 7 ground truth types.
Params: lam=80, th=4, t0=10, gamma=5, iter=100, seed=100, init_K=9
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from scipy.special import logsumexp
from scipy.stats import poisson
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, homogeneity_score,
                             silhouette_score)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.similarity import compute_neighbor_matrix
from src.core.sampler import run_sampler
from src.evaluation.metrics import compute_morans_i


def compute_similarity_matrix_innerprod(E, eps=1e-10):
    """
    Similarity: inner_prod / n_dims + Fisher Z (matches expected result).
    """
    n_samples, n_dims = E.shape
    S = (E @ E.T) / n_dims
    S = np.clip(S, -0.9999, 0.9999)
    A = 0.5 * np.log((1 + S) / (1 - S))
    A = np.nan_to_num(A, nan=0.0,
                      posinf=A[np.isfinite(A)].max(),
                      neginf=A[np.isfinite(A)].min())
    return A


def compute_Vn_pochhammer(n_cells, gamma=1.0, lambda_poisson=1.0):
    """
    Pochhammer Vn formula (the formula that used for this result).
    """
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
            log_term = log_factorial_ratio - log_pochhammer + log_poisson_prob
            log_terms.append(log_term)
        Vn[t] = logsumexp(log_terms) if log_terms else -np.inf

    return Vn


def main():
    # ======================================================================
    # Parameters 
    # ======================================================================
    lambda1 = 80.0
    threshold = 4.0
    gamma = 5.0
    t0 = 10.0
    n_iterations = 100
    init_k = 9
    seed = 100
    alpha_rna = 1.0
    alpha_atac = 1.0

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'misar_brain')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'MISAR_mouse_E15_brain')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("MISAR Brain: Multi-modal MRFC-MFM Clustering (Pochhammer Vn)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: lambda1={lambda1}, threshold={threshold}, gamma={gamma}, "
          f"t0={t0}, seed={seed}, init_K={init_k}, iterations={n_iterations}")

    # ======================================================================
    # Load data
    # ======================================================================
    print("\n[1/5] Loading data...")
    E_rna = np.load(os.path.join(data_dir, 'rna_embedding.npy'))
    E_atac = np.load(os.path.join(data_dir, 'atac_embedding.npy'))
    pos = np.load(os.path.join(data_dir, 'spatial_coords.npy'))
    labels_true = np.load(os.path.join(data_dir, 'labels.npy'))
    label_names = np.load(os.path.join(data_dir, 'label_names.npy'), allow_pickle=True)

    n_cells = pos.shape[0]
    n_gt = len(label_names)
    print(f"  Cells: {n_cells}, GT K: {n_gt}")
    print(f"  RNA: {E_rna.shape}, L2 norm: {np.sqrt((E_rna**2).sum(axis=1)).mean():.2f}")
    print(f"  ATAC: {E_atac.shape}, L2 norm: {np.sqrt((E_atac**2).sum(axis=1)).mean():.2f}")

    # ======================================================================
    # Similarity matrices
    # ======================================================================
    print("\n[2/5] Computing similarity matrices...")
    A_rna = compute_similarity_matrix_innerprod(E_rna)
    A_atac = compute_similarity_matrix_innerprod(E_atac)
    print(f"  A_rna: range=[{A_rna.min():.4f}, {A_rna.max():.4f}]")
    print(f"  A_atac: range=[{A_atac.min():.4f}, {A_atac.max():.4f}]")

    # ======================================================================
    # Neighbor matrix
    # ======================================================================
    print("\n[3/5] Computing neighbor matrix...")
    W = compute_neighbor_matrix(pos.astype(np.float64), threshold=threshold)

    # ======================================================================
    # Pochhammer Vn
    # ======================================================================
    print("\n[4/5] Computing Pochhammer Vn...")
    t_vn = time.time()
    Vn = compute_Vn_pochhammer(n_cells, gamma=gamma)
    print(f"  Done in {time.time() - t_vn:.1f}s")

    # ======================================================================
    # MCMC
    # ======================================================================
    print("\n[5/5] Running MCMC...")
    t_start = time.time()
    result = run_sampler(
        A_list=[A_atac, A_rna],
        alpha_list=[alpha_atac, alpha_rna],
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
    elapsed = time.time() - t_start

    z = result['cluster_assign']
    K_trace = result['K_trace']
    K = len(np.unique(z))
    singletons = sum(1 for c in np.unique(z) if np.sum(z == c) == 1)

    # ======================================================================
    # Evaluation
    # ======================================================================
    ari = adjusted_rand_score(labels_true, z)
    nmi = normalized_mutual_info_score(labels_true, z)
    ami = adjusted_mutual_info_score(labels_true, z)
    homo = homogeneity_score(labels_true, z)
    mi, _, _ = compute_morans_i(z, pos.astype(np.float64), W, threshold=threshold)
    sil = silhouette_score(np.hstack([E_rna, E_atac]), z) if K > 1 else -1.0

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"  K={K}, ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}")
    print(f"  Homo={homo:.4f}, MoransI={mi:.4f}, Silhouette={sil:.4f}")
    print(f"  Singletons={singletons}, Time={elapsed:.1f}s")

    cluster_sizes = sorted([np.sum(z == c) for c in np.unique(z)], reverse=True)
    print(f"  Cluster sizes: {cluster_sizes}")

    # ======================================================================
    # Save
    # ======================================================================
    np.savez(os.path.join(output_dir, 'result.npz'),
             z=z, K_trace=K_trace, pos=pos,
             labels_true=labels_true, label_names=label_names,
             params={'lambda1': lambda1, 'threshold': threshold,
                     'gamma': gamma, 't0': t0, 'seed': seed})
    print(f"\nSaved: {output_dir}/result.npz")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
