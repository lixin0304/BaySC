"""
Preprocessing script for HER2 Breast Tumor H1 sample.

Data: ST (Spatial Transcriptomics) single-modality RNA
    - 613 spots x 15,029 genes
    - 7 labels (6 tissue types + undetermined)
    - Clustering uses all 613 spots; evaluation excludes undetermined (530 spots)

Pipeline (single modality RNA, same as Human Lymph Node / Stereo-seq):
    Step 1: Filter genes (min_cells=10)
    Step 2: Select 3000 HVG (seurat)
    Step 3: Normalize per spot (target_sum=1e4)
    Step 4: Log1p transform
    Step 5: Per-gene scaling (Z-score)
    Step 6: PCA 30
    Step 7: L1 normalization

Data source: data/raw/H1_sample/
    - H1.tsv.gz: count matrix (613 spots x 15,029 genes), index = array_x x array_y
    - H1_labeled_coordinates.tsv: spot labels + pixel coordinates
    - H1_selection.tsv: spot selection with pixel coordinates

Output: data/processed/her2_breast_h1/
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from datetime import datetime

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 80)
    print("Preprocessing HER2 Breast Tumor H1 Sample")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    raw_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'H1_sample')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'her2_breast_h1')
    os.makedirs(output_dir, exist_ok=True)

    n_hvg = 3000
    min_cells_gene = 10
    n_components = 30

    print(f"\nParameters:")
    print(f"  RNA: filter genes (min {min_cells_gene} cells), HVG {n_hvg}, PCA {n_components}")
    print(f"  Output directory: {output_dir}")

    # ========================================================================
    # [1/4] Load count matrix
    # ========================================================================
    print("\n[1/4] Loading count matrix...")
    counts_df = pd.read_csv(os.path.join(raw_dir, 'H1.tsv.gz'), sep='\t', index_col=0)
    print(f"  Count matrix: {counts_df.shape[0]} spots x {counts_df.shape[1]} genes")
    print(f"  Spot index format: {counts_df.index[0]}")
    print(f"  Non-zero fraction: {(counts_df.values > 0).mean():.4f}")

    # ========================================================================
    # [2/4] Load spatial coordinates and labels
    # ========================================================================
    print("\n[2/4] Loading spatial coordinates and labels...")
    labels_df = pd.read_csv(os.path.join(raw_dir, 'H1_labeled_coordinates.tsv'), sep='\t')
    print(f"  Labels file: {labels_df.shape[0]} spots")

    # H1_selection.tsv: array coords (x, y) -> adjusted coords (new_x, new_y) -> pixel coords
    # H1_labeled_coordinates.tsv: (x, y) match selection's (new_x, new_y)
    # Matching key: round(new_x, 3) == round(label_x, 3) AND round(new_y, 3) == round(label_y, 3)
    selection_df = pd.read_csv(os.path.join(raw_dir, 'H1_selection.tsv'), sep='\t')
    print(f"  Selection file: {selection_df.shape[0]} spots")

    # Build mapping: count_matrix_index ("array_x x array_y") -> (new_x, new_y, sel_pixel_x, sel_pixel_y)
    sel_map = {}
    for _, row in selection_df.iterrows():
        key = f"{int(row['x'])}x{int(row['y'])}"
        sel_map[key] = (round(row['new_x'], 3), round(row['new_y'], 3),
                        row['pixel_x'], row['pixel_y'])

    # Build mapping: (rounded_x, rounded_y) -> (label_pixel_x, label_pixel_y, label)
    label_map = {}
    for _, row in labels_df.iterrows():
        rkey = (round(row['x'], 3), round(row['y'], 3))
        label_map[rkey] = (row['pixel_x'], row['pixel_y'], row['label'])

    # Match: count_index -> sel (new_x, new_y) -> label
    spot_ids = list(counts_df.index)
    pixel_coords = np.zeros((len(spot_ids), 2), dtype=np.float64)
    labels_all = []
    matched = 0

    for i, sid in enumerate(spot_ids):
        if sid in sel_map:
            nx, ny, spx, spy = sel_map[sid]
            rkey = (nx, ny)
            if rkey in label_map:
                lpx, lpy, lbl = label_map[rkey]
                # Use label file's pixel coords (original ST pixel positions)
                pixel_coords[i] = [lpx, lpy]
                labels_all.append(lbl)
                matched += 1
            else:
                # No label entry -> use selection pixel coords, mark undetermined
                pixel_coords[i] = [spx, spy]
                labels_all.append('undetermined')
        else:
            pixel_coords[i] = [0, 0]
            labels_all.append('undetermined')

    labels_all = np.array(labels_all)
    print(f"  Matched spots: {matched}/{len(spot_ids)}")

    # Label distribution (all 613)
    unique_all, counts_all = np.unique(labels_all, return_counts=True)
    print(f"\n  Label distribution (all {len(spot_ids)} spots):")
    for lbl, cnt in zip(unique_all, counts_all):
        print(f"    {lbl}: {cnt}")

    # Separate: all spots for clustering, non-undetermined for evaluation
    undet_mask = labels_all == 'undetermined'
    eval_mask = ~undet_mask
    n_eval = eval_mask.sum()
    print(f"\n  Evaluation spots (excluding undetermined): {n_eval}")

    # Integer labels for all spots (undetermined gets its own label)
    unique_labels = np.unique(labels_all)
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    labels_int_all = np.array([label_to_int[l] for l in labels_all])

    # Integer labels for evaluation only (no undetermined)
    eval_labels_raw = labels_all[eval_mask]
    unique_eval = np.unique(eval_labels_raw)
    eval_label_to_int = {l: i for i, l in enumerate(unique_eval)}
    labels_int_eval = np.array([eval_label_to_int[l] for l in eval_labels_raw])

    print(f"  Evaluation labels: {len(unique_eval)} types")
    for lbl in unique_eval:
        cnt = np.sum(eval_labels_raw == lbl)
        print(f"    {lbl}: {cnt}")

    # ========================================================================
    # [3/4] RNA preprocessing
    # ========================================================================
    print("\n[3/4] Processing RNA data...")
    adata = sc.AnnData(X=counts_df.values.astype(np.float32))
    adata.obs_names = counts_df.index
    adata.var_names = counts_df.columns
    print(f"  AnnData: {adata.shape}")

    # Step 1: Filter genes
    print(f"  Step 1: Filter genes (min_cells={min_cells_gene})")
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells_gene)
    print(f"    Genes: {n_before} -> {adata.n_vars}")

    # Step 2: HVG
    print(f"  Step 2: Select {n_hvg} HVG (seurat)")
    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_hvg, flavor='seurat')
    hvg_mask = adata_temp.var.highly_variable
    n_hvg_actual = hvg_mask.sum()
    print(f"    HVG selected: {n_hvg_actual}")
    adata_hvg = adata[:, hvg_mask].copy()

    # Step 3: Normalize
    print("  Step 3: Normalize per spot (target_sum=1e4)")
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)

    # Step 4: Log1p
    print("  Step 4: Log1p transform")
    sc.pp.log1p(adata_hvg)

    # Step 5: Z-score
    print("  Step 5: Per-gene scaling (Z-score)")
    sc.pp.scale(adata_hvg)
    X_scaled = adata_hvg.X
    if issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    print(f"    Scaled shape: {X_scaled.shape}")
    print(f"    Mean per gene: {X_scaled.mean(axis=0).mean():.6f}")
    print(f"    Std per gene: {X_scaled.std(axis=0).mean():.6f}")

    # Step 6: PCA
    print(f"  Step 6: PCA (n_components={n_components})")
    sc.tl.pca(adata_hvg, n_comps=n_components)
    rna_pca = adata_hvg.obsm['X_pca']
    print(f"    Explained variance: {adata_hvg.uns['pca']['variance_ratio'].sum():.4f}")
    print(f"    PCA range: [{rna_pca.min():.4f}, {rna_pca.max():.4f}]")

    # Step 7: L1 normalization
    print("  Step 7: L1 normalization")
    E_rna = normalize(rna_pca, norm='l1', axis=1).astype(np.float32)
    print(f"    Final shape: {E_rna.shape}")
    print(f"    Range: [{E_rna.min():.4f}, {E_rna.max():.4f}]")
    print(f"    L1 norm check (row sum): {np.abs(E_rna[0]).sum():.4f}")

    # ========================================================================
    # [4/4] Save
    # ========================================================================
    print("\n[4/4] Saving processed data...")
    np.save(os.path.join(output_dir, 'rna_embedding.npy'), E_rna)
    np.save(os.path.join(output_dir, 'spatial_coords.npy'), pixel_coords.astype(np.float64))
    np.save(os.path.join(output_dir, 'labels_all.npy'), labels_int_all)
    np.save(os.path.join(output_dir, 'labels_eval.npy'), labels_int_eval)
    np.save(os.path.join(output_dir, 'eval_mask.npy'), eval_mask)
    np.save(os.path.join(output_dir, 'label_names_all.npy'), unique_labels)
    np.save(os.path.join(output_dir, 'label_names_eval.npy'), unique_eval)

    with open(os.path.join(output_dir, 'label_mapping.txt'), 'w') as f:
        f.write("# All labels (including undetermined)\n")
        for lbl, idx in label_to_int.items():
            cnt = np.sum(labels_all == lbl)
            f.write(f"{idx}: {lbl} ({cnt} spots)\n")
        f.write(f"\n# Evaluation labels (excluding undetermined)\n")
        for lbl, idx in eval_label_to_int.items():
            cnt = np.sum(eval_labels_raw == lbl)
            f.write(f"{idx}: {lbl} ({cnt} spots)\n")

    print(f"\nSaved files:")
    print(f"  - rna_embedding.npy ({E_rna.shape})")
    print(f"  - spatial_coords.npy ({pixel_coords.shape})")
    print(f"  - labels_all.npy ({len(unique_labels)} types, {len(labels_int_all)} spots)")
    print(f"  - labels_eval.npy ({len(unique_eval)} types, {n_eval} spots)")
    print(f"  - eval_mask.npy ({eval_mask.sum()} True / {len(eval_mask)} total)")

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
