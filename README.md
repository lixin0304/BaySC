# BaySC: Bayesian Spatial Clustering for Spatial Transcriptomics

BaySC is a fully Bayesian framework for spatial domain identification in spatially resolved transcriptomics (SRT) and spatial multi-omics data. It jointly infers the number of spatial domains, enforces spatial continuity at the label level, and supports unified multi-modal fusion of RNA, ATAC, and protein (ADT) modalities within a single collapsed Gibbs sampler.

## Features

- **Automatic K inference** — Mixture of Finite Mixtures (MFM) prior suppresses spurious clusters without pre-specifying the number of domains.
- **Spatial MRF regularization** — MRF constraint acts on discrete labels during Gibbs sampling, not on expression features, avoiding over-smoothing at domain boundaries.
- **Multi-modal fusion** — Weighted log-likelihood fusion of RNA, ATAC, and protein modalities. Single-modality is a natural special case.
- **Uncertainty quantification** — Posterior co-membership matrix provides cell-level uncertainty for identifying boundary and transitional regions.
- **Automatic hyperparameter selection** — Spatial smoothing strength λ and distance threshold δ are selected via the modified deviance information criterion (mDIC).
- **Scalable sampler** — Numba-accelerated collapsed Gibbs sampler with analytic marginalization of continuous parameters.

## Method Overview

BaySC models pairwise cell–cell similarity matrices derived from low-dimensional modality embeddings using a stochastic block model with Normal-Gamma conjugate priors. The clustering prior combines:

- **MFM prior**: Pólya urn scheme with a correction factor `V_n(K*+1)/V_n(K*)` that systematically penalizes new domain creation, enabling consistent estimation of K.
- **MRF prior**: Pairwise energy function that rewards neighboring cells sharing the same domain label, controlled by smoothing parameter λ.

The joint **MRFC-MFM** prior is sampled via collapsed Gibbs, and posterior point estimates are extracted using Dahl's method. Spatial hyperparameters are selected by minimizing mDIC over a candidate grid (including λ=0 as a no-smoothing baseline).

**Pipeline:**

```
Raw counts → Embedding (PCA / LSI / CLR) → Cosine similarity + Fisher-Z transform
    → MRFC-MFM collapsed Gibbs sampler → Dahl posterior point estimate → mDIC hyperparameter selection
```

## Project Structure

```
BaySC/
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── sampler.py          # Numba-accelerated multi-modal Gibbs sampler
│   │   └── dahl.py             # Dahl's method for posterior point estimation
│   ├── preprocessing/
│   │   └── similarity.py       # Cosine similarity + Fisher-Z + spatial neighbor matrix
│   └── evaluation/
│       ├── metrics.py          # ARI, AMI, NMI, Homogeneity, Moran's I
│       ├── model_selection.py  # mDIC, WAIC
│       └── spari.py            # Spatially-aware ARI (spARI)
├── scripts/
│   ├── breast_cancer/          # 10x Visium Human Breast Cancer 
│   ├── her2_breast_{a1..h1}/   # HER2+ Breast Cancer
│   ├── starmap/                # STARmap Mouse Visual Cortex 
│   ├── misar_brain/            # MISAR-seq Mouse E15.5 Brain, RNA+ATAC
│   └── human_lymph_node/       # Spatial CITE-seq Human Lymph Node A1, RNA+Protein 
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
```

Each dataset directory contains:
- `preprocess.py` — raw data to embeddings
- `run.py` — clustering, evaluation, and visualization with locked parameters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Each dataset is self-contained:

```bash
# Step 1: Preprocess (raw data → embeddings)
python scripts/breast_cancer/preprocess.py

# Step 2: Cluster + evaluate + visualize
python scripts/breast_cancer/run.py
```

Available datasets: `breast_cancer`, `her2_breast_a1`, `her2_breast_b1`, `her2_breast_c1`, `her2_breast_d1`, `her2_breast_e1`, `her2_breast_f1`, `her2_breast_g2`, `her2_breast_h1`, `starmap`, `misar_brain`, `human_lymph_node`.

## Datasets

| Dataset | Modality | Spots | True K |
|---|---|---|---|
| Human Breast Cancer | RNA | 3,798 | 20 |
| HER2+ Breast Cancer A1–H1 | RNA | 167–659 | 3–6 |
| STARmap Mouse Visual Cortex | RNA | 1,207 | 7 |
| MISAR-seq Mouse E15.5 Brain | RNA + ATAC | 1,949 | 7 |
| Human Lymph Node A1 | RNA + Protein | 3,484 | 6 |

## Key Parameters

| Parameter | Description | 
|---|---|
| `lambda` | MRF spatial smoothing strength |
| `threshold` | Spatial neighbor distance threshold δ | 
| `gamma` | MFM Dirichlet concentration parameter | 
| `init_k` | Initial number of clusters | 
| `n_iterations` | Total Gibbs iterations | 
| `burn_in` | Burn-in iterations to discard | 
| `alpha_m` | Per-modality fusion weights | 

## Evaluation Metrics

| Metric | Description |
|---|---|
| ARI | Adjusted Rand Index |
| AMI | Adjusted Mutual Information |
| NMI | Normalized Mutual Information |
| Homogeneity | Each cluster contains only one ground-truth class |
| Moran's I | Global spatial autocorrelation of predicted labels |
| spARI | Spatially-aware ARI, penalizes spatially incoherent errors |
| mDIC | Modified Deviance Information Criterion for model selection |

## Reproducibility

All results are deterministic given a fixed seed. Verified result on Human Breast Cancer:

```
Parameters: lam=28.0, threshold=700.0, gamma=7.0, n_iter=100, burn_in=50, seed=42
K=20, ARI=0.5541, AMI=0.6663, NMI=0.6722, Homogeneity=0.6928
Moran's I=0.7327, spARI=0.6643
```

## Citation

If you use BaySC in your research, please cite:

```bibtex
@article{baysc2025,
  title   = {BaySC: Bayesian Spatial Clustering for Spatial Transcriptomics},
  author  = {},
  journal = {Bioinformatics},
  year    = {2025}
}
```
