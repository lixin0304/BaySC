# BaySC: Bayesian Spatial Clustering via MRFC-MFM

BaySC is a Bayesian nonparametric spatial clustering method for spatially resolved transcriptomics (SRT) data. It combines a **Markov Random Field Constrained Mixture of Finite Mixtures (MRFC-MFM)** model with Gibbs sampling to identify spatial domains while automatically determining the number of clusters.

## Method Overview

BaySC models pairwise cell-cell similarity matrices (derived from gene expression or multi-modal embeddings) using a block-structured Gaussian likelihood. Spatial coherence is enforced through a MRF prior on cluster labels, and the number of clusters is inferred via a MFM prior with Pochhammer-symbol-based $V_n$ coefficients.

**Key features:**
- Nonparametric: automatically determines K (number of clusters)
- Spatial-aware: MRF prior encourages spatially coherent domains
- Multi-modal: supports RNA-only, ATAC-only, and RNA+ATAC/Protein fusion
- Numba-accelerated Gibbs sampler for scalability

**Pipeline:**
1. Preprocessing: HVG selection, normalization, PCA/SVD embedding
2. Similarity: L2-normalize embedding, compute cosine similarity, apply Fisher's Z transform
3. Spatial neighbor matrix: distance-thresholded binary adjacency
4. MFM prior: precompute $V_n$ coefficients (Pochhammer formula)
5. Gibbs sampling: iteratively update cluster assignments, block-wise $\mu/\tau$ parameters
6. Posterior summary: Dahl's method for representative clustering selection
7. Model assessment: mDIC / WAIC for model quality
8. Evaluation: ARI, AMI, NMI, Homogeneity, Moran's I, spARI, Silhouette

## Project Structure

```
BaySC/
  requirements.txt          # Python dependencies
  src/                      # Core library
    core/
      sampler.py            #   Numba-accelerated multi-modal Gibbs sampler
      dahl.py               #   Dahl's method (posterior representative selection)
    preprocessing/
      similarity.py         #   Cosine + Fisher Z similarity & neighbor matrix
    evaluation/
      metrics.py            #   ARI, NMI, AMI, Moran's I, LISI, etc.
      model_selection.py    #   mDIC, WAIC model selection criteria
      spari.py              #   Spatially-aware ARI (spARI)
  scripts/                  # Per-dataset: preprocess.py + run.py
    breast_cancer/          #   10x Visium Breast Cancer (K=20)
    her2_breast_{a1..h1}/   #   HER2+ Breast Cancer, 8 patients
    starmap/                #   STARmap Mouse Visual Cortex (K=7)
    misar_brain/            #   MISAR-seq Mouse E15 Brain, ATAC+RNA (K=7)
    human_lymph_node/       #   Spatial CITE-seq Human Lymph Node A1 (K=6)
  data/
    raw/                    # Original data files
    processed/              # Preprocessed embeddings (.npy)
  outputs/                  # Clustering results, metrics, visualizations
```

Each dataset directory under `scripts/` contains exactly two files:
- **`preprocess.py`** -- raw data to embeddings
- **`run.py`** -- clustering with locked parameters, evaluation, and visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Each dataset is self-contained. Run directly:

```bash
# Step 1: Preprocess (raw -> embeddings)
python scripts/breast_cancer/preprocess.py

# Step 2: Cluster + evaluate + visualize
python scripts/breast_cancer/run.py
```

All 12 datasets follow the same pattern:

```bash
python scripts/<dataset>/preprocess.py
python scripts/<dataset>/run.py
```

Where `<dataset>` is one of: `breast_cancer`, `her2_breast_a1`, `her2_breast_b1`, `her2_breast_c1`, `her2_breast_d1`, `her2_breast_e1`, `her2_breast_f1`, `her2_breast_g2`, `her2_breast_h1`, `starmap`, `misar_brain`, `human_lymph_node`.

## Datasets

| Dataset | Modality | Spots | True K | ARI | spARI |
|---------|----------|-------|--------|-----|-------|
| Breast Cancer | RNA | 3,798 | 20 | 0.5541 | 0.6643 |
| HER2 A1 | RNA | 346 | 6 | - | - |
| HER2 B1 | RNA | 372 | 5 | - | - |
| HER2 C1 | RNA | 351 | 4 | - | - |
| HER2 D1 | RNA | 310 | 6 | - | - |
| HER2 E1 | RNA | 307 | 6 | - | - |
| HER2 F1 | RNA | 364 | 6 | - | - |
| HER2 G2 | RNA | 356 | 7 | - | - |
| HER2 H1 | RNA | 613 | 6 | - | - |
| STARmap | RNA | 1,207 | 7 | - | - |
| MISAR Brain | ATAC+RNA | 4,018 | 7 | - | - |
| Human Lymph Node | RNA+Protein | 4,035 | 6 | - | - |

## Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `lambda1` | Spatial penalty strength (MRF weight) | 10 ~ 100 |
| `threshold` | Spatial neighbor distance threshold | Dataset-dependent |
| `t0` | Prior precision scaling for $\mu/\tau$ updates | 1.0 ~ 10.0 |
| `gamma` | MFM concentration parameter in $V_n$ | 1.0 ~ 10.0 |
| `init_k` | Initial number of clusters | Set near expected K |
| `n_iterations` | MCMC iterations | 50 ~ 200 |
| `burn_in` | Burn-in iterations to discard | ~50% of total |
| `seed` | Random seed for reproducibility | 42 |

## Evaluation Metrics

- **ARI** (Adjusted Rand Index): overall clustering agreement with ground truth
- **AMI** (Adjusted Mutual Information): information-theoretic agreement
- **NMI** (Normalized Mutual Information): normalized variant of MI
- **Homogeneity**: each cluster contains only members of a single class
- **Moran's I**: spatial autocorrelation of cluster labels
- **spARI**: spatially-aware ARI (penalizes spatially incoherent misassignments less)
- **Silhouette**: embedding-space cluster compactness and separation
- **mDIC**: modified Deviance Information Criterion for model quality

## Reproducibility

All results are deterministic given the same seed. Verified on Breast Cancer (Dahl's method):

```
Params: lam=28.0, th=700.0, t0=8.0, gamma=7.0, iter=100, burn_in=50, seed=42
K=20, ARI=0.5541, AMI=0.6663, NMI=0.6722, Homo=0.6928,
MoranI=0.7327, spARI=0.6643, Sil=0.0443
```
