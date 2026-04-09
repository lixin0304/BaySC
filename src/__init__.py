"""
BaySC: Bayesian Spatial Clustering via MRFC-MFM

Package structure:
- core/: Gibbs sampler (sampler.py) + Dahl's method (dahl.py)
- preprocessing/: Similarity matrix computation (cosine + Fisher Z)
- evaluation/: Clustering metrics, mDIC, WAIC, spARI
"""

from . import core, preprocessing, evaluation

__all__ = [
    "core",
    "preprocessing",
    "evaluation",
]
