"""
Preprocessing script for Breast Cancer (10x Visium) dataset.

Pipeline (single modality RNA, same as Human Lymph Node RNA):
    Step 1: Filter genes (min_cells=10)
    Step 2: Select 3000 HVG (seurat)
    Step 3: Normalize per spot (target_sum=1e4)
    Step 4: Log1p transform
    Step 5: Per-gene scaling (Z-score)
    Step 6: PCA 30
    Step 7: L1 normalization

Data source: data/raw/Breast cancer/
    - V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5
    - spatial/tissue_positions_list.csv
    - metadata.tsv (ground truth: 20 fine_annot_type classes)
    - 3,798 in-tissue spots x 36,601 genes

Output: data/processed/breast_cancer/
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 80)
    print("Preprocessing Breast Cancer Dataset")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Breast cancer')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'breast_cancer')
    os.makedirs(output_dir, exist_ok=True)

    # Parameters (same as Human Lymph Node RNA pipeline)
    n_hvg = 3000
    min_cells_gene = 10
    n_components_max = 50  # PCA with max dims, then slice for 30/50

    print(f"\nParameters:")
    print(f"  RNA: filter genes (min {min_cells_gene} cells), HVG {n_hvg}, PCA {n_components_max}")
    print(f"  Output directory: {output_dir}")

    # ========================================================================
    # [1/3] Load data
    # ========================================================================
    print("\n[1/3] Loading data...")

    # Read 10x h5 expression matrix
    h5_path = os.path.join(data_dir, 'V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5')
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    print(f"  Expression matrix: {adata.shape[0]} spots x {adata.shape[1]} genes")

    # Read spatial positions
    pos_df = pd.read_csv(
        os.path.join(data_dir, 'spatial', 'tissue_positions_list.csv'),
        header=None,
        names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
    )
    pos_df = pos_df.set_index('barcode')

    # Filter to in-tissue spots that match expression barcodes
    common_barcodes = adata.obs.index.intersection(pos_df.index)
    adata = adata[common_barcodes].copy()
    pos_df = pos_df.loc[common_barcodes]
    print(f"  Matched spots: {adata.shape[0]}")

    # Spatial coordinates (array coordinates)
    pos = pos_df[['pxl_col', 'pxl_row']].values.astype(np.float32)
    print(f"  Spatial coords shape: {pos.shape}")
    print(f"  Spatial range: X=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}], "
          f"Y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]")

    # Ground truth labels (fine_annot_type, 20 classes)
    meta_df = pd.read_csv(os.path.join(data_dir, 'metadata.tsv'), sep='\t')
    meta_df = meta_df.set_index('ID')

    # Match to expression barcodes
    labels_raw = meta_df.loc[adata.obs.index, 'fine_annot_type'].values.astype(str)

    # Filter out NaN annotations if any
    nan_mask = (labels_raw == 'nan') | (labels_raw == 'NaN') | (labels_raw == '')
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  Filtering {n_nan} spots with NaN annotations")
        adata = adata[~nan_mask].copy()
        pos = pos[~nan_mask]
        labels_raw = labels_raw[~nan_mask]

    unique_labels = np.unique(labels_raw)
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[l] for l in labels_raw])
    n_true_types = len(unique_labels)

    print(f"  Annotations: {n_true_types} types (fine_annot_type)")
    for label in unique_labels:
        cnt = np.sum(labels_raw == label)
        print(f"    {label}: {cnt}")

    n_cells = adata.shape[0]
    print(f"  Final spots: {n_cells}")

    # ========================================================================
    # [2/3] RNA preprocessing (same pipeline as Human Lymph Node)
    # ========================================================================
    print("\n[2/3] Processing RNA data...")
    adata_proc = adata.copy()

    # Step 1: Filter genes (min_cells)
    print(f"  Step 1: Filter genes (min_cells={min_cells_gene})")
    n_genes_before = adata_proc.n_vars
    sc.pp.filter_genes(adata_proc, min_cells=min_cells_gene)
    print(f"    Genes: {n_genes_before} -> {adata_proc.n_vars}")

    # Step 1b: Filter cells (min_counts=1)
    print(f"  Step 1b: Filter cells (min_counts=1)")
    n_cells_before = adata_proc.n_obs
    sc.pp.filter_cells(adata_proc, min_counts=1)
    print(f"    Cells: {n_cells_before} -> {adata_proc.n_obs}")

    # Step 2: Select HVG (seurat)
    # Normalize first for HVG selection, then go back to raw counts
    print(f"  Step 2: Select {n_hvg} highly variable genes (seurat)")
    adata_temp = adata_proc.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_hvg, flavor='seurat')
    hvg_mask = adata_temp.var.highly_variable
    print(f"    HVG selected: {hvg_mask.sum()}")

    # Apply HVG to original (raw count) data
    adata_hvg = adata_proc[:, hvg_mask].copy()

    # Step 3: Normalize per spot (target_sum=1e4)
    print("  Step 3: Normalize per spot (target_sum=1e4)")
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)

    # Step 4: Log1p transform
    print("  Step 4: Log1p transform")
    sc.pp.log1p(adata_hvg)

    # Step 5: Per-gene scaling (Z-score)
    print("  Step 5: Per-gene scaling (Z-score)")
    sc.pp.scale(adata_hvg)

    X_scaled = adata_hvg.X
    if issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    print(f"    Scaled matrix shape: {X_scaled.shape}")
    print(f"    Range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
    print(f"    Mean per gene (should be ~0): {X_scaled.mean(axis=0).mean():.6f}")
    print(f"    Std per gene (should be ~1): {X_scaled.std(axis=0).mean():.6f}")

    # Step 6: PCA (compute max dims, then slice)
    print(f"  Step 6: PCA (n_components={n_components_max})")
    sc.tl.pca(adata_hvg, n_comps=n_components_max)
    rna_pca_full = adata_hvg.obsm['X_pca']
    print(f"    Explained variance (50): {adata_hvg.uns['pca']['variance_ratio'].sum():.4f}")
    print(f"    PCA range: [{rna_pca_full.min():.4f}, {rna_pca_full.max():.4f}]")

    # Update pos/labels to match filtered cells
    kept_barcodes = adata_hvg.obs.index
    original_barcodes = adata.obs.index
    keep_mask = original_barcodes.isin(kept_barcodes)
    pos_final = pos[keep_mask]
    labels_int_final = labels_int[keep_mask]
    labels_raw_final = labels_raw[keep_mask]
    n_cells_final = rna_pca_full.shape[0]

    # Step 7: Save multiple normalization variants (PCA30/PCA50 x L1/zscore)
    from sklearn.preprocessing import normalize, StandardScaler
    for n_pc in [30, 50]:
        rna_pca = rna_pca_full[:, :n_pc]
        ev = adata_hvg.uns['pca']['variance_ratio'][:n_pc].sum()
        print(f"\n  PCA{n_pc}: explained variance = {ev:.4f}")

        # L1 normalization
        E_l1 = normalize(rna_pca, norm='l1', axis=1).astype(np.float32)
        print(f"    L1: shape={E_l1.shape}, range=[{E_l1.min():.4f}, {E_l1.max():.4f}]")
        np.save(os.path.join(output_dir, f'rna_embedding_pca{n_pc}_l1.npy'), E_l1)

        # Z-score normalization (per-component standardization)
        scaler = StandardScaler()
        E_zscore = scaler.fit_transform(rna_pca).astype(np.float32)
        print(f"    Zscore: shape={E_zscore.shape}, range=[{E_zscore.min():.4f}, {E_zscore.max():.4f}]")
        np.save(os.path.join(output_dir, f'rna_embedding_pca{n_pc}_zscore.npy'), E_zscore)

    # Also save the default (PCA30 L1) as rna_embedding.npy for backward compat
    E_rna = normalize(rna_pca_full[:, :30], norm='l1', axis=1).astype(np.float32)
    print(f"\n  Final dataset: {n_cells_final} spots")

    # ========================================================================
    # [3/3] Save processed data
    # ========================================================================
    print("\n[3/3] Saving processed data...")
    np.save(os.path.join(output_dir, 'rna_embedding.npy'), E_rna)
    np.save(os.path.join(output_dir, 'spatial_coords.npy'), pos_final)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_int_final)
    np.save(os.path.join(output_dir, 'label_names.npy'), unique_labels)

    # Save label mapping
    with open(os.path.join(output_dir, 'label_mapping.txt'), 'w') as f:
        for label, idx in label_to_int.items():
            cnt = np.sum(labels_raw_final == label)
            f.write(f"{idx}: {label} ({cnt} spots)\n")

    # Save metadata
    metadata = {
        'n_cells': n_cells_final,
        'n_genes_raw': adata.shape[1],
        'n_hvg': n_hvg,
        'min_cells_gene': min_cells_gene,
        'n_components': n_components_max,
        'n_labels': n_true_types,
        'spatial_range_x': [float(pos_final[:, 0].min()), float(pos_final[:, 0].max())],
        'spatial_range_y': [float(pos_final[:, 1].min()), float(pos_final[:, 1].max())],
    }
    np.savez(os.path.join(output_dir, 'metadata.npz'), **metadata)

    print(f"\nSaved files:")
    print(f"  - {output_dir}/rna_embedding.npy ({E_rna.shape})")
    print(f"  - {output_dir}/rna_embedding_pca30_l1.npy")
    print(f"  - {output_dir}/rna_embedding_pca30_zscore.npy")
    print(f"  - {output_dir}/rna_embedding_pca50_l1.npy")
    print(f"  - {output_dir}/rna_embedding_pca50_zscore.npy")
    print(f"  - {output_dir}/spatial_coords.npy ({pos_final.shape})")
    print(f"  - {output_dir}/labels.npy ({n_true_types} types, {n_cells_final} spots)")
    print(f"  - {output_dir}/label_names.npy")
    print(f"  - {output_dir}/label_mapping.txt")
    print(f"  - {output_dir}/metadata.npz")

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
