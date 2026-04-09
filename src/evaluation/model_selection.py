"""
Model selection criteria for MRFC-MFM clustering.

Implements:
- mDIC (modified Deviance Information Criterion), formula (4.3) from
  "Bayesian Spatial Homogeneity Pursuit of Functional Data"
- WAIC (Watanabe-Akaike Information Criterion)

Both work with the numpy-array MCMC history returned by the sampler.
"""

import numpy as np
from numba import njit
from typing import Dict, Optional


@njit
def _compute_deviance(cluster_assign, mu, tau, A):
    """
    Compute deviance Dev(theta) = -2 * log-likelihood.
    
    Sums over upper triangle (i < j) of the similarity matrix:
        log L = sum_{i<j} log Normal(S_ij | U_{z_i,z_j}, T_{z_i,z_j}^{-1})
    
    Args:
        cluster_assign: (n,) int array, 1-indexed
        mu: (K, K) mean parameters
        tau: (K, K) precision parameters
        A: (n, n) similarity matrix
    
    Returns:
        deviance: scalar (-2 * log L)
    """
    n = len(cluster_assign)
    log_lik = 0.0
    log_2pi = 1.8378770664093453
    
    for i in range(n):
        for j in range(i + 1, n):
            zi = cluster_assign[i] - 1
            zj = cluster_assign[j] - 1
            
            U_rs = mu[zi, zj]
            T_rs = tau[zi, zj]
            S_ij = A[i, j]
            
            if T_rs > 0:
                log_lik += -0.5 * log_2pi + 0.5 * np.log(T_rs) \
                           - 0.5 * T_rs * (S_ij - U_rs) ** 2
            else:
                return 1e10
    
    return -2.0 * log_lik


def compute_mdic(history: Dict[str, np.ndarray],
                   A: np.ndarray,
                   burn_in: int = 0,
                   modality_index: int = 0) -> Dict[str, float]:
    """
    Calculate modified DIC.
    
    mDIC = Dev(theta_bar) + log(n*(n+1)/2) * p_D
    where p_D = D_bar - Dev(theta_bar)
    
    Args:
        history: dict with 'z', 'mu', 'tau' arrays from sampler
        A: (n, n) similarity matrix (single modality)
        burn_in: iterations to discard
        modality_index: which modality to use (default 0)
    
    Returns:
        dict with mDIC, Dev_theta, D_bar, p_D, K statistics
    """
    z_hist = history['z']
    mu_hist = history['mu']
    tau_hist = history['tau']
    
    n_iterations = z_hist.shape[0]
    n = A.shape[0]
    
    if burn_in >= n_iterations:
        return {'mDIC': np.inf, 'Dev_theta': np.inf, 'D_bar': np.inf,
                'p_D': 0.0, 'K_mean': 0.0, 'K_median': 0.0}
    
    z_post = z_hist[burn_in:]
    mu_post = mu_hist[burn_in:]
    tau_post = tau_hist[burn_in:]
    n_samples = z_post.shape[0]
    m = modality_index
    
    # Compute deviance for each posterior sample and accumulate mean params
    deviances = np.zeros(n_samples)
    K_values = np.zeros(n_samples, dtype=int)
    max_K = 0
    
    for t in range(n_samples):
        z_t = z_post[t]
        K_t = int(z_t.max())
        K_values[t] = K_t
        max_K = max(max_K, K_t)
        
        mu_t = mu_post[t, m, :K_t, :K_t]
        tau_t = tau_post[t, m, :K_t, :K_t]
        
        deviances[t] = _compute_deviance(
            z_t.astype(np.int64), mu_t, tau_t, A
        )
    
    D_bar = float(np.mean(deviances))
    
    # Posterior mean parameters (accumulated then averaged)
    sum_mu = np.zeros((max_K, max_K), dtype=np.float64)
    sum_tau = np.zeros((max_K, max_K), dtype=np.float64)
    for t in range(n_samples):
        K_t = K_values[t]
        sum_mu[:K_t, :K_t] += mu_post[t, m, :K_t, :K_t]
        sum_tau[:K_t, :K_t] += tau_post[t, m, :K_t, :K_t]
    mu_bar = sum_mu / n_samples
    tau_bar = sum_tau / n_samples
    
    # Use last iteration's z for Dev(theta_bar)
    z_bar = z_post[-1]
    K_bar = int(z_bar.max())
    
    Dev_theta = _compute_deviance(
        z_bar.astype(np.int64),
        mu_bar[:K_bar, :K_bar],
        tau_bar[:K_bar, :K_bar],
        A
    )
    
    p_D = max(D_bar - Dev_theta, 0.0)
    penalty = np.log(n * (n + 1) / 2.0)
    mDIC = Dev_theta + penalty * p_D
    
    return {
        'mDIC': float(mDIC),
        'Dev_theta': float(Dev_theta),
        'D_bar': float(D_bar),
        'p_D': float(p_D),
        'penalty_factor': float(penalty),
        'K_mean': float(np.mean(K_values)),
        'K_median': float(np.median(K_values)),
    }


def compute_waic(history: Dict[str, np.ndarray],
                   A: np.ndarray,
                   burn_in: int = 0,
                   modality_index: int = 0) -> Dict[str, float]:
    """
    Calculate WAIC = -2 * (lppd - pWAIC).
    
    Args:
        history: dict with 'z', 'mu', 'tau' arrays
        A: (n, n) similarity matrix
        burn_in: iterations to discard
        modality_index: which modality
    
    Returns:
        dict with WAIC, lppd, pWAIC
    """
    z_hist = history['z']
    mu_hist = history['mu']
    tau_hist = history['tau']
    
    n_iterations = z_hist.shape[0]
    n = A.shape[0]
    
    if burn_in >= n_iterations:
        return {'WAIC': np.inf, 'lppd': -np.inf, 'pWAIC': 0.0}
    
    z_post = z_hist[burn_in:]
    mu_post = mu_hist[burn_in:]
    tau_post = tau_hist[burn_in:]
    n_samples = z_post.shape[0]
    m = modality_index
    
    log_2pi = 1.8378770664093453
    n_obs = n * (n - 1) // 2
    log_liks = np.zeros((n_obs, n_samples))
    
    for s_idx in range(n_samples):
        z = z_post[s_idx]
        K = int(z.max())
        mu = mu_post[s_idx, m, :K, :K]
        tau = tau_post[s_idx, m, :K, :K]
        
        obs_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                zi = int(z[i]) - 1
                zj = int(z[j]) - 1
                mu_ij = mu[zi, zj]
                tau_ij = tau[zi, zj]
                
                if tau_ij > 0:
                    log_liks[obs_idx, s_idx] = (
                        -0.5 * log_2pi + 0.5 * np.log(tau_ij)
                        - 0.5 * tau_ij * (A[i, j] - mu_ij) ** 2
                    )
                else:
                    log_liks[obs_idx, s_idx] = -np.inf
                obs_idx += 1
    
    # lppd = sum_i log(mean(exp(log_lik_i)))
    lppd = 0.0
    for obs_idx in range(n_obs):
        max_ll = np.max(log_liks[obs_idx, :])
        lppd += max_ll + np.log(np.mean(np.exp(log_liks[obs_idx, :] - max_ll)))
    
    # pWAIC = sum of variances of log-likelihoods
    pWAIC = float(np.sum(np.var(log_liks, axis=1)))
    
    WAIC = -2.0 * (lppd - pWAIC)
    
    return {
        'WAIC': float(WAIC),
        'lppd': float(lppd),
        'pWAIC': pWAIC,
    }


__all__ = ['compute_mdic', 'compute_waic']
