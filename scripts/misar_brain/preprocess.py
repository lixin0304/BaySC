"""
Preprocessing script for MISAR-seq Mouse E15 Brain dataset.

Verified pipeline (corr=1.0 with embedding that produced K=7, ARI=0.4550):

RNA pipeline (matches python_src/preprocessing/rna_embedding.py):
  1. Gene filter (min_cells=3)
  2. Library size normalization (target_sum=1e4)
  3. Log1p
  4. HVG selection (n_top_genes=2000)
  5. Per-gene z-score (StandardScaler, clip=10)
  6. PCA (n_components=50, random_state=42)
  7. Per-cell z-score (row-wise, ddof=1)

ATAC pipeline (matches python_src/preprocessing/lsi_embedding.py):
  1. Binarize (X > 0 -> 1)
  2. Filter low-activity peaks (min 0.5% cells)
  3. TF-IDF
  4. L1 normalization (row-wise)
  5. Log1p(X * 1e4)
  6. Randomized SVD -> U matrix (n_components=51)
  7. Per-cell z-score (row-wise, ddof=1)
  8. Drop first component -> 50 dims

Clustering parameters:
  lambda1=80, threshold=4, t0=10, gamma=5, seed=100, init_K=9, iterations=100
  Vn formula: Pochhammer
  Similarity: inner_prod / n_dims + Fisher Z
  Result: K=7, ARI=0.4550, NMI=0.5186

Data source:
  data/raw/MISAR_seq_mouse_E15_brain_mRNA_data.h5
  data/raw/MISAR_seq_mouse_E15_brain_ATAC_data.h5

Output:
  data/processed/misar_brain/
"""

import os
import sys
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.utils.extmath import randomized_svd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def preprocess_rna(rna_count, n_components=50, n_top_genes=2000, min_cells=3,
                   clip_value=10.0, random_state=42):
    """
    RNA preprocessing pipeline.

    Steps: gene filter -> normalize 1e4 -> log1p -> HVG -> per-gene zscore
           -> PCA -> per-cell zscore
    """
    n_cells, n_genes = rna_count.shape
    print(f"  Input: {n_cells} cells x {n_genes} genes")

    # Step 1: Gene filter
    gene_counts = np.sum(rna_count > 0, axis=0)
    gene_mask = gene_counts >= min_cells
    X = rna_count[:, gene_mask]
    print(f"  Step 1: Gene filter (min_cells={min_cells}): {n_genes} -> {X.shape[1]} genes")

    # Step 2: Library size normalization
    X = X / (X.sum(axis=1, keepdims=True) + 1e-10) * 10000
    print(f"  Step 2: Library size normalization (target=1e4)")

    # Step 3: Log1p
    X = np.log1p(X)
    print(f"  Step 3: Log1p, range=[{X.min():.4f}, {X.max():.4f}]")

    # Step 4: HVG selection
    if n_top_genes and n_top_genes < X.shape[1]:
        gene_vars = np.var(X, axis=0)
        top_idx = np.argsort(gene_vars)[-n_top_genes:]
        X = X[:, top_idx]
    print(f"  Step 4: HVG selection (n_top_genes={n_top_genes}): {X.shape[1]} genes")

    # Step 5: Per-gene z-score with clip
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, -clip_value, clip_value)
    print(f"  Step 5: Per-gene z-score (clip={clip_value}), mean={X.mean():.6f}")

    # Step 6: PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    E = pca.fit_transform(X)
    print(f"  Step 6: PCA ({n_components} components), explained_var={pca.explained_variance_ratio_.sum():.4f}")

    # Step 7: Per-cell z-score (row-wise, ddof=1)
    E = E - E.mean(axis=1, keepdims=True)
    E = E / E.std(axis=1, ddof=1, keepdims=True)
    l2_norm = np.sqrt((E ** 2).sum(axis=1)).mean()
    print(f"  Step 7: Per-cell z-score (ddof=1), L2 norm={l2_norm:.4f}")

    print(f"  Final: shape={E.shape}, range=[{E.min():.4f}, {E.max():.4f}]")
    return E, pca


def preprocess_atac(atac_count, n_components=51, random_state=42):
    """
    ATAC preprocessing pipeline.

    Steps: binarize -> filter peaks -> TF-IDF -> L1 norm -> log1p(*1e4)
           -> SVD(U) -> per-cell zscore -> drop first component
    """
    n_cells, n_peaks = atac_count.shape
    print(f"  Input: {n_cells} cells x {n_peaks} peaks")

    # Step 1: Binarize
    X = atac_count.copy()
    n_changed = np.sum(X > 1)
    X[X > 0] = 1
    print(f"  Step 1: Binarize, {n_changed} values > 1 set to 1")

    # Step 2: Filter low-activity peaks (< 0.5% cells)
    min_cells = int(0.005 * n_cells)
    peak_counts = np.array((X > 0).sum(axis=0)).flatten()
    valid_peaks = peak_counts >= min_cells
    X = X[:, valid_peaks]
    print(f"  Step 2: Filter peaks (min={min_cells} cells): {n_peaks} -> {X.shape[1]} peaks")

    # Step 3: TF-IDF
    row_sums = np.array(X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    X_tf = X / row_sums[:, np.newaxis]
    n_cells_count = X.shape[0]
    doc_freq = np.array((X > 0).sum(axis=0)).flatten()
    doc_freq[doc_freq == 0] = 1
    idf = np.log(1 + n_cells_count / doc_freq)
    X_tfidf = X_tf * idf
    print(f"  Step 3: TF-IDF, range=[{X_tfidf.min():.6f}, {X_tfidf.max():.6f}]")

    # Step 4: L1 normalization
    X_norm = Normalizer(norm='l1').fit_transform(X_tfidf)
    print(f"  Step 4: L1 normalization, row sums={X_norm[:3].sum(axis=1)}")

    # Step 5: Log1p(X * 1e4)
    X_log = np.log1p(X_norm * 1e4)
    print(f"  Step 5: Log1p(*1e4), range=[{X_log.min():.4f}, {X_log.max():.4f}]")

    # Step 6: Randomized SVD -> U matrix
    U, s, Vt = randomized_svd(X_log, n_components=n_components, random_state=random_state)
    E = U
    print(f"  Step 6: SVD ({n_components} components), singular values[:5]={s[:5]}")

    # Step 7: Per-cell z-score (row-wise, ddof=1)
    E = E - E.mean(axis=1, keepdims=True)
    E = E / E.std(axis=1, ddof=1, keepdims=True)
    print(f"  Step 7: Per-cell z-score (ddof=1)")

    # Step 8: Drop first component
    E = E[:, 1:]
    l2_norm = np.sqrt((E ** 2).sum(axis=1)).mean()
    print(f"  Step 8: Drop first component -> {E.shape[1]} dims, L2 norm={l2_norm:.4f}")

    print(f"  Final: shape={E.shape}, range=[{E.min():.4f}, {E.max():.4f}]")
    return E


def main():
    print("=" * 80)
    print("Preprocessing MISAR-seq Mouse E15 Brain Dataset")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'misar_brain')
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # Load data
    # ========================================================================
    print("\n[1/4] Loading RNA data...")
    with h5py.File(os.path.join(data_dir, 'MISAR_seq_mouse_E15_brain_mRNA_data.h5'), 'r') as f:
        rna_count = f['X'][:]
        pos = f['pos'][:]
        labels_raw = f['Y'][:]
    labels_str = np.array([y.decode() if isinstance(y, bytes) else y for y in labels_raw])
    unique_types = np.unique(labels_str)
    gt_map = {v: i for i, v in enumerate(unique_types)}
    labels = np.array([gt_map[v] for v in labels_str])
    print(f"  RNA shape: {rna_count.shape}")
    print(f"  Ground truth: {len(unique_types)} types")

    print("\n[2/4] Loading ATAC data...")
    with h5py.File(os.path.join(data_dir, 'MISAR_seq_mouse_E15_brain_ATAC_data.h5'), 'r') as f:
        atac_count = f['X'][:]
    print(f"  ATAC shape: {atac_count.shape}")

    n_cells = rna_count.shape[0]
    print(f"\n  Cells: {n_cells}")
    print(f"  Spatial range: X=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}], Y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]")

    # ========================================================================
    # RNA preprocessing
    # ========================================================================
    print("\n[3/4] Processing RNA data...")
    E_rna, pca_rna = preprocess_rna(rna_count, n_components=50, n_top_genes=2000,
                                     min_cells=3, clip_value=10.0, random_state=42)

    # ========================================================================
    # ATAC preprocessing
    # ========================================================================
    print("\n[4/4] Processing ATAC data...")
    E_atac = preprocess_atac(atac_count, n_components=51, random_state=42)

    # ========================================================================
    # Save
    # ========================================================================
    print("\nSaving processed data...")
    np.save(os.path.join(output_dir, 'rna_embedding.npy'), E_rna.astype(np.float32))
    np.save(os.path.join(output_dir, 'atac_embedding.npy'), E_atac.astype(np.float32))
    np.save(os.path.join(output_dir, 'spatial_coords.npy'), pos.astype(np.float32))
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    np.save(os.path.join(output_dir, 'label_names.npy'), unique_types)

    np.savez(os.path.join(output_dir, 'metadata.npz'),
             n_cells=n_cells,
             n_genes=rna_count.shape[1],
             n_peaks=atac_count.shape[1],
             n_types=len(unique_types),
             rna_explained_variance=pca_rna.explained_variance_ratio_.sum(),
             spatial_range_x=[float(pos[:,0].min()), float(pos[:,0].max())],
             spatial_range_y=[float(pos[:,1].min()), float(pos[:,1].max())])

    print(f"\nSaved to {output_dir}:")
    for fname in ['rna_embedding.npy', 'atac_embedding.npy', 'spatial_coords.npy',
                   'labels.npy', 'label_names.npy', 'metadata.npz']:
        print(f"  - {fname}")

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
