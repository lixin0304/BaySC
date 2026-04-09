"""
Preprocessing script for Human Lymph Node A1 Spatial CITE-seq dataset.

Pipeline:
- RNA: filter genes (min 10 spots) -> HVG 3000 -> normalize 1e4 -> log1p -> scale -> PCA
- ADT: CLR per spot -> scale -> PCA

Data source: Dataset11_Human_Lymph_Node_A1/
Output: Preprocessed embeddings saved to data/processed/human_lymph_node/
"""

import os
import sys
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def clr_normalize_each_cell(adata, inplace=True):
    """
    Seurat-style CLR (Centered Log-Ratio) normalization for protein data.
    Per-spot CLR: log(x+1) - mean(log(x+1)) for each spot
    """
    if not inplace:
        adata = adata.copy()
    
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    X_log = np.log1p(X)
    row_means = X_log.mean(axis=1, keepdims=True)
    X_clr = X_log - row_means
    
    adata.X = X_clr
    return adata


def main():
    print("=" * 80)
    print("Preprocessing Human Lymph Node A1 Dataset")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Dataset11_Human_Lymph_Node_A1')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'human_lymph_node')
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    # ADT has 31 proteins, so use n_proteins - 1 = 30 for PCA
    # RNA uses same dimension to align
    n_hvg = 3000  # HVG selection (Seurat v3)
    min_cells_gene = 10  # Filter genes appearing in < 10 spots
    n_components_rna = 30  # Align with ADT dimension
    n_components_adt = 30  # n_proteins - 1
    
    print(f"\nParameters:")
    print(f"  RNA: filter genes (min {min_cells_gene} spots), HVG {n_hvg}, PCA {n_components_rna}")
    print(f"  ADT: CLR -> scale -> PCA {n_components_adt}")
    print(f"  Output directory: {output_dir}")
    
    # Load data
    print("\n[1/4] Loading data...")
    adata_rna = sc.read_h5ad(os.path.join(data_dir, 'adata_RNA.h5ad'))
    adata_adt = sc.read_h5ad(os.path.join(data_dir, 'adata_ADT.h5ad'))
    
    print(f"  RNA: {adata_rna.shape}")
    print(f"  ADT: {adata_adt.shape}")
    
    # Load annotation (ground truth labels)
    import pandas as pd
    anno_path = os.path.join(data_dir, 'annotation.csv')
    if os.path.exists(anno_path):
        anno_df = pd.read_csv(anno_path)
        anno_df = anno_df.set_index('Barcode')
        # Match to RNA barcodes
        labels = anno_df.loc[adata_rna.obs_names, 'manual-anno'].values
        unique_labels = np.unique(labels)
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        labels_int = np.array([label_to_int[l] for l in labels])
        print(f"  Labels: {len(unique_labels)} types")
        print(f"  Label counts: {dict(zip(*np.unique(labels, return_counts=True)))}")
    else:
        labels = None
        labels_int = None
        unique_labels = None
        print("  Labels: Not found")
    
    # Get spatial coordinates
    if 'spatial' in adata_rna.obsm:
        pos = adata_rna.obsm['spatial']
    elif 'X_spatial' in adata_rna.obsm:
        pos = adata_rna.obsm['X_spatial']
    else:
        pos = np.column_stack([adata_rna.obs['x'], adata_rna.obs['y']])
    
    pos = pos.astype(np.float32)
    n_cells = pos.shape[0]
    print(f"  Cells: {n_cells}")
    print(f"  Spatial range: X=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}], Y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]")
    
    # RNA preprocessing pipeline (user specified):
    # Step 1: Filter genes appearing in < 10 spots
    # Step 2: Select HVG (Seurat v3, 3000 genes)
    # Step 3: Normalize per spot (target_sum=1e4)
    # Step 4: Log1p transform
    # Step 5: Per-gene scaling (Z-score)
    # Step 6: PCA on HVG -> feat
    print("\n[2/4] Processing RNA data...")
    adata_rna_proc = adata_rna.copy()
    
    print(f"  Step 1: Filter genes (min_cells={min_cells_gene})")
    sc.pp.filter_genes(adata_rna_proc, min_cells=min_cells_gene)
    print(f"    Genes after filter: {adata_rna_proc.n_vars}")
    
    print(f"  Step 2: Select {n_hvg} highly variable genes (Seurat)")
    # Need to normalize first for HVG selection
    adata_rna_temp = adata_rna_proc.copy()
    sc.pp.normalize_total(adata_rna_temp, target_sum=1e4)
    sc.pp.log1p(adata_rna_temp)
    sc.pp.highly_variable_genes(adata_rna_temp, n_top_genes=n_hvg, flavor='seurat')
    hvg_mask = adata_rna_temp.var.highly_variable
    print(f"    HVG selected: {hvg_mask.sum()}")
    
    # Apply HVG to original data
    adata_rna_hvg = adata_rna_proc[:, hvg_mask].copy()
    
    print("  Step 3: Normalize per spot (target_sum=1e4)")
    sc.pp.normalize_total(adata_rna_hvg, target_sum=1e4)
    
    print("  Step 4: Log1p transform")
    sc.pp.log1p(adata_rna_hvg)
    
    print("  Step 5: Per-gene scaling (Z-score)")
    sc.pp.scale(adata_rna_hvg)
    
    print(f"  Step 6: PCA (n_components={n_components_rna})")
    sc.tl.pca(adata_rna_hvg, n_comps=n_components_rna)
    rna_pca = adata_rna_hvg.obsm['X_pca']
    print(f"    Explained variance: {adata_rna_hvg.uns['pca']['variance_ratio'].sum():.4f}")
    print(f"    PCA range: [{rna_pca.min():.4f}, {rna_pca.max():.4f}]")
    
    print("  Step 7: L1 normalization (same as MISAR brain)")
    from sklearn.preprocessing import normalize
    E_rna = normalize(rna_pca, norm='l1', axis=1).astype(np.float32)
    print(f"    Final shape: {E_rna.shape}")
    print(f"    Range: [{E_rna.min():.4f}, {E_rna.max():.4f}]")
    print(f"    L1 norm check (row sum): {np.abs(E_rna[0]).sum():.4f}")
    
    print("  Step 7b: Z-score normalization (per-component)")
    scaler_rna = StandardScaler()
    E_rna_zscore = scaler_rna.fit_transform(rna_pca).astype(np.float32)
    print(f"    Final shape: {E_rna_zscore.shape}")
    print(f"    Range: [{E_rna_zscore.min():.4f}, {E_rna_zscore.max():.4f}]")
    
    # ADT preprocessing pipeline (user specified):
    # Step 1: CLR normalization per spot
    # Step 2: Per-feature scaling (Z-score)
    # Step 3: PCA (n_proteins - 1 = 30)
    print("\n[3/4] Processing Protein (ADT) data...")
    adata_adt_proc = adata_adt.copy()
    
    print("  Step 1: CLR normalization (per spot)")
    clr_normalize_each_cell(adata_adt_proc)
    
    X_adt = adata_adt_proc.X
    if hasattr(X_adt, 'toarray'):
        X_adt = X_adt.toarray()
    print(f"    After CLR: range [{X_adt.min():.4f}, {X_adt.max():.4f}]")
    
    print("  Step 2: Per-feature scaling (Z-score)")
    scaler = StandardScaler()
    X_adt_scaled = scaler.fit_transform(X_adt)
    print(f"    After scale: range [{X_adt_scaled.min():.4f}, {X_adt_scaled.max():.4f}]")
    
    print(f"  Step 3: PCA (n_components={n_components_adt})")
    pca = PCA(n_components=n_components_adt)
    adt_pca = pca.fit_transform(X_adt_scaled)
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"    PCA range: [{adt_pca.min():.4f}, {adt_pca.max():.4f}]")
    
    print("  Step 4: L1 normalization (same as MISAR brain)")
    from sklearn.preprocessing import normalize
    E_protein = normalize(adt_pca, norm='l1', axis=1).astype(np.float32)
    print(f"    Final shape: {E_protein.shape}")
    print(f"    Range: [{E_protein.min():.4f}, {E_protein.max():.4f}]")
    print(f"    L1 norm check (row sum): {np.abs(E_protein[0]).sum():.4f}")
    
    print("  Step 4b: Z-score normalization (per-component)")
    scaler_adt = StandardScaler()
    E_protein_zscore = scaler_adt.fit_transform(adt_pca).astype(np.float32)
    print(f"    Final shape: {E_protein_zscore.shape}")
    print(f"    Range: [{E_protein_zscore.min():.4f}, {E_protein_zscore.max():.4f}]")
    
    # Save processed data
    print("\n[4/4] Saving processed data...")
    np.save(os.path.join(output_dir, 'rna_embedding.npy'), E_rna.astype(np.float32))
    np.save(os.path.join(output_dir, 'protein_embedding.npy'), E_protein.astype(np.float32))
    np.save(os.path.join(output_dir, 'rna_embedding_zscore.npy'), E_rna_zscore.astype(np.float32))
    np.save(os.path.join(output_dir, 'protein_embedding_zscore.npy'), E_protein_zscore.astype(np.float32))
    np.save(os.path.join(output_dir, 'spatial_coords.npy'), pos)
    
    # Save labels if available
    if labels_int is not None:
        np.save(os.path.join(output_dir, 'labels.npy'), labels_int)
        np.save(os.path.join(output_dir, 'label_names.npy'), unique_labels)
    
    # Save metadata
    metadata = {
        'n_cells': n_cells,
        'n_genes': adata_rna.shape[1],
        'n_hvg': n_hvg,
        'min_cells_gene': min_cells_gene,
        'n_proteins': adata_adt.shape[1],
        'n_components_rna': n_components_rna,
        'n_components_adt': n_components_adt,
        'n_labels': len(unique_labels) if unique_labels is not None else 0,
        'spatial_range_x': [float(pos[:,0].min()), float(pos[:,0].max())],
        'spatial_range_y': [float(pos[:,1].min()), float(pos[:,1].max())],
    }
    np.savez(os.path.join(output_dir, 'metadata.npz'), **metadata)
    
    print(f"\nSaved files:")
    print(f"  - {output_dir}/rna_embedding.npy")
    print(f"  - {output_dir}/protein_embedding.npy")
    print(f"  - {output_dir}/spatial_coords.npy")
    if labels_int is not None:
        print(f"  - {output_dir}/labels.npy ({len(unique_labels)} types)")
        print(f"  - {output_dir}/label_names.npy")
    print(f"  - {output_dir}/metadata.npz")
    
    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
