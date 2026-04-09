"""Multi-modal MRFC-MFM Gibbs sampler (Numba accelerated).

Implements the collapsed Gibbs sampler for the Markov Random Field Constrained
Mixture of Finite Mixtures (MRFC-MFM) model with multi-modal support.

Block-structured Gaussian likelihood with separate mu_0 (diagonal) and
mu_0off (off-diagonal) priors.  Singleton handling uses the V_n ratio
trick: V_n[K] - V_n[K-1] for singletons, V_n[K+1] - V_n[K] otherwise.

Public API:
    run_sampler(...)  -- Python entry point (wraps the Numba core)
"""

import numpy as np
from numba import njit
import time

try:
    from numba_scipy.special import gammaln
    HAS_NUMBA_SCIPY = True
except ImportError:
    HAS_NUMBA_SCIPY = False
    print("Warning: numba-scipy not available, using Stirling approximation")


@njit
def gammaln_stirling(x):
    """Stirling approximation for log(Gamma(x))."""
    if x <= 1:
        return 0.0
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)


if HAS_NUMBA_SCIPY:
    gammaln_numba = gammaln
else:
    gammaln_numba = gammaln_stirling


@njit
def _loglike(i, cluster_assign, mu_list, tau_list, A_stack, alpha_weights):
    """Compute weighted log-likelihood for observation i across all modalities."""
    M = A_stack.shape[0]
    n = len(cluster_assign)
    c_i = cluster_assign[i] - 1
    
    total_log_lik = 0.0
    log_2pi = 1.8378770664093453
    
    for m in range(M):
        mu = mu_list[m]
        tau = tau_list[m]
        A = A_stack[m]
        alpha_m = alpha_weights[m]
        
        log_lik_m = 0.0
        
        for j in range(n):
            c_j = cluster_assign[j] - 1
            
            if c_i >= mu.shape[0] or c_j >= mu.shape[1]:
                return -np.inf
            
            mean = mu[c_i, c_j]
            precision = tau[c_i, c_j]
            if precision <= 0:
                precision = 1e-10
            residual = A[i, j] - mean
            
            log_lik_m += -0.5 * (log_2pi - np.log(precision) + precision * residual * residual)
        
        total_log_lik += alpha_m * log_lik_m
    
    return total_log_lik


@njit
def _logmargs(i, cluster_assign, A_stack, alpha_weights, mu_0off_list, t_0, alpha_prior, beta):
    """Compute marginal log-likelihood for proposing a new cluster (multi-modal)."""
    M = A_stack.shape[0]
    n = len(cluster_assign)
    K = cluster_assign.max() - 1
    
    total_log_marg = 0.0
    log_2pi = 1.8378770664093453
    
    for m in range(M):
        A = A_stack[m]
        alpha_m = alpha_weights[m]
        mu_0 = mu_0off_list[m]
        
        log_marg_m = 0.0
        
        for c in range(1, K + 2):  # 1 to K+1 (since K = max-1)
            count = 0
            for j in range(n):
                if cluster_assign[j] == c:
                    count += 1
            
            if count == 0:
                continue
            
            # Collect data[i, j] for j in cluster c
            data_ic = np.empty(count, dtype=np.float64)
            idx = 0
            for j in range(n):
                if cluster_assign[j] == c:
                    data_ic[idx] = A[i, j]
                    idx += 1
            
            S = len(data_ic)
            if S == 0:
                continue
            
            meanA = np.mean(data_ic)
            
            # Compute variance * (S-1) = SSE
            if S > 1:
                sse = 0.0
                for val in data_ic:
                    sse += (val - meanA) ** 2
            else:
                sse = 0.0
            
            # Normal-Gamma posterior update
            alpha_n = alpha_prior + S / 2.0
            t_n = t_0 + S
            beta_n = beta + (meanA - mu_0) ** 2 * t_0 * S / (2.0 * t_n) + 0.5 * sse
            
            # Log marginal likelihood under Normal-Gamma conjugacy
            log_marg_m += (
                gammaln_numba(alpha_n) - gammaln_numba(alpha_prior)
                + alpha_prior * np.log(beta) - alpha_n * np.log(beta_n)
                + 0.5 * (np.log(t_0) - np.log(t_n))
                - S / 2.0 * log_2pi
            )
        
        total_log_marg += alpha_m * log_marg_m
    
    return total_log_marg


@njit
def _update_params(cluster_assign, A_stack, K, mu_0_list, mu_0off_list, t_0, alpha_prior, beta):
    """Update mu and tau for all cluster-pair blocks (Normal-Gamma posterior)."""
    M = A_stack.shape[0]
    n = len(cluster_assign)
    
    mu_list = np.zeros((M, K, K), dtype=np.float64)
    tau_list = np.zeros((M, K, K), dtype=np.float64)
    
    for m in range(M):
        A = A_stack[m]
        mu_0 = mu_0_list[m]
        mu_0off = mu_0off_list[m]
        
        for r in range(1, K + 1):
            for s in range(r, K + 1):
                # Collect datapoints
                if r != s:
                    # Off-diagonal: all pairs (i,j) where i in r, j in s
                    count = 0
                    for ii in range(n):
                        if cluster_assign[ii] == r:
                            for jj in range(n):
                                if cluster_assign[jj] == s:
                                    count += 1
                    
                    data_rs = np.empty(count, dtype=np.float64)
                    idx = 0
                    for ii in range(n):
                        if cluster_assign[ii] == r:
                            for jj in range(n):
                                if cluster_assign[jj] == s:
                                    data_rs[idx] = A[ii, jj]
                                    idx += 1
                else:
                    # Diagonal block: lower triangle including diagonal
                    count = 0
                    for ii in range(n):
                        if cluster_assign[ii] == r:
                            for jj in range(ii + 1):  # jj <= ii (lower triangle with diag)
                                if cluster_assign[jj] == r:
                                    count += 1
                    
                    data_rs = np.empty(count, dtype=np.float64)
                    idx = 0
                    for ii in range(n):
                        if cluster_assign[ii] == r:
                            for jj in range(ii + 1):
                                if cluster_assign[jj] == r:
                                    data_rs[idx] = A[ii, jj]
                                    idx += 1
                
                # Choose prior mean based on diagonal vs off-diagonal
                if r == s:
                    mu_prior = mu_0
                else:
                    mu_prior = mu_0off
                
                if len(data_rs) == 0:
                    # Empty: sample from prior
                    tau_val = np.random.gamma(alpha_prior, 1.0 / beta)
                    tau_list[m, r-1, s-1] = tau_list[m, s-1, r-1] = tau_val
                    std = np.sqrt(1.0 / (t_0 * tau_val))
                    mu_val = np.random.normal(mu_prior, std)
                    mu_list[m, r-1, s-1] = mu_list[m, s-1, r-1] = mu_val
                    continue
                
                nr = len(data_rs)
                thetah = np.mean(data_rs)
                
                # Compute SSE
                if nr > 1:
                    sse = 0.0
                    for val in data_rs:
                        sse += (val - thetah) ** 2
                else:
                    sse = 0.0
                
                # Normal-Gamma posterior update
                kr = 1.0 / (t_0 + nr)
                alpha_n = alpha_prior + nr / 2.0
                beta_n = beta + 0.5 * (nr * t_0 / (t_0 + nr) * (thetah - mu_prior) ** 2 + sse)
                
                tau_val = np.random.gamma(alpha_n, 1.0 / beta_n)
                if tau_val <= 0:
                    tau_val = 1e-10
                tau_list[m, r-1, s-1] = tau_list[m, s-1, r-1] = tau_val
                
                # mu posterior: mean = kr*nr*thetah + t_0*kr*mu_prior
                mu_post_mean = kr * nr * thetah + t_0 * kr * mu_prior
                mu_post_std = np.sqrt(kr / tau_val)
                mu_val = np.random.normal(mu_post_mean, mu_post_std)
                mu_list[m, r-1, s-1] = mu_list[m, s-1, r-1] = mu_val
    
    return mu_list, tau_list


@njit
def _compute_spatial_penalty(i, c, cluster_assign, neighbor_matrix):
    """Count neighbors of cell i in cluster c."""
    n = len(cluster_assign)
    n_neighbors = 0
    for j in range(n):
        if cluster_assign[j] == c and neighbor_matrix[i, j] == 1:
            n_neighbors += 1
    return n_neighbors


@njit
def _remove_empty_clusters(cluster_assign, mu_list, tau_list):
    """Remove empty clusters and relabel."""
    n = len(cluster_assign)
    M = mu_list.shape[0]
    K = cluster_assign.max()
    
    cluster_sizes = np.zeros(K, dtype=np.int32)
    for i in range(n):
        cluster_sizes[cluster_assign[i] - 1] += 1
    
    old_to_new = np.zeros(K, dtype=np.int32)
    new_label = 1
    for old_label in range(1, K + 1):
        if cluster_sizes[old_label - 1] > 0:
            old_to_new[old_label - 1] = new_label
            new_label += 1
    
    K_new = new_label - 1
    
    for i in range(n):
        cluster_assign[i] = old_to_new[cluster_assign[i] - 1]
    
    if K_new < K:
        mu_list_new = np.zeros((M, K_new, K_new), dtype=np.float64)
        tau_list_new = np.zeros((M, K_new, K_new), dtype=np.float64)
        
        for m in range(M):
            for old_r in range(1, K + 1):
                if cluster_sizes[old_r - 1] == 0:
                    continue
                new_r = old_to_new[old_r - 1]
                
                for old_s in range(1, K + 1):
                    if cluster_sizes[old_s - 1] == 0:
                        continue
                    new_s = old_to_new[old_s - 1]
                    
                    mu_list_new[m, new_r - 1, new_s - 1] = mu_list[m, old_r - 1, old_s - 1]
                    tau_list_new[m, new_r - 1, new_s - 1] = tau_list[m, old_r - 1, old_s - 1]
        
        return cluster_assign, mu_list_new, tau_list_new
    
    return cluster_assign, mu_list, tau_list


@njit
def _gibbs_sampler_core(A_stack, alpha_weights, neighbor_matrix, lambda1, Vn, 
                        n_iterations, init_K, mu_0_list, mu_0off_list, t_0, 
                        alpha_prior, beta, gamma, seed=100, max_K=100):
    """Numba-accelerated Gibbs sampler core. Saves full MCMC history per iteration."""
    np.random.seed(seed)
    
    M = A_stack.shape[0]
    n = A_stack.shape[1]
    
    # Initialize cluster assignments (ensure all init_K clusters appear)
    cluster_assign = np.zeros(n, dtype=np.int32)
    for i in range(init_K):
        cluster_assign[i] = i + 1
    for i in range(init_K, n):
        cluster_assign[i] = np.random.randint(1, init_K + 1)
    
    # Initialize parameters for each modality
    mu_list = np.zeros((M, init_K, init_K), dtype=np.float64)
    tau_list = np.zeros((M, init_K, init_K), dtype=np.float64)
    
    for m in range(M):
        for r in range(init_K):
            for s in range(r, init_K):
                tau_val = np.random.gamma(alpha_prior, 1.0 / beta)
                tau_list[m, r, s] = tau_list[m, s, r] = tau_val
                std = np.sqrt(1.0 / (t_0 * tau_val))
                # Use mu_0 for diagonal, mu_0off for off-diagonal
                if r == s:
                    mu_val = np.random.normal(mu_0_list[m], std)
                else:
                    mu_val = np.random.normal(mu_0off_list[m], std)
                mu_list[m, r, s] = mu_list[m, s, r] = mu_val
    
    K_trace = np.zeros(n_iterations, dtype=np.int32)
    
    # Pre-allocate MCMC history arrays
    z_history = np.zeros((n_iterations, n), dtype=np.int32)
    mu_history = np.zeros((n_iterations, M, max_K, max_K), dtype=np.float64)
    tau_history = np.zeros((n_iterations, M, max_K, max_K), dtype=np.float64)
    
    for iter_idx in range(n_iterations):
        K = cluster_assign.max()
        
        cluster_sizes = np.zeros(K, dtype=np.int32)
        for i in range(n):
            cluster_sizes[cluster_assign[i] - 1] += 1
        
        for i in range(n):
            cur_cluster_i = cluster_assign[i]
            is_singleton = (cluster_sizes[cur_cluster_i - 1] == 1)
            
            # Both singleton and non-singleton have K+1 candidates
            # But singleton uses different VN ratio
            n_candidates = K + 1
            log_probs = np.full(n_candidates, -np.inf, dtype=np.float64)
            
            # Compute c_counts_noi (cluster sizes without i)
            c_counts_noi = cluster_sizes.copy()
            c_counts_noi[cur_cluster_i - 1] -= 1
            
            # For singleton, offset gamma
            if is_singleton:
                c_counts_noi[cur_cluster_i - 1] -= gamma
            
            # Existing clusters
            for c_idx in range(K):
                c = c_idx + 1
                cluster_assign[i] = c
                
                log_lik = _loglike(i, cluster_assign, mu_list, tau_list, A_stack, alpha_weights)
                n_neighbors = _compute_spatial_penalty(i, c, cluster_assign, neighbor_matrix)
                spatial_term = lambda1 * n_neighbors
                
                size_prior = gamma + c_counts_noi[c - 1]
                
                if size_prior > 0:
                    log_probs[c_idx] = np.log(size_prior) + spatial_term + log_lik
            
            cluster_assign[i] = cur_cluster_i
            
            # New cluster option
            cluster_assign[i] = K + 1
            log_marg = _logmargs(i, cluster_assign, A_stack, alpha_weights, 
                                                  mu_0off_list, t_0, alpha_prior, beta)
            
            # VN ratio depends on singleton status
            if is_singleton:
                # Singleton: VN[K] - VN[K-1]
                if K >= 2:
                    vn_term = Vn[K] - Vn[K - 1]
                else:
                    vn_term = 0.0
            else:
                # Non-singleton: VN[K+1] - VN[K]
                vn_term = Vn[K + 1] - Vn[K]
            
            log_probs[K] = np.log(gamma) + log_marg + vn_term
            cluster_assign[i] = cur_cluster_i
            
            # Sample new cluster
            max_log_prob = -np.inf
            for val in log_probs:
                if np.isfinite(val) and val > max_log_prob:
                    max_log_prob = val
            
            if not np.isfinite(max_log_prob):
                continue
            
            probs = np.exp(log_probs - max_log_prob)
            probs = probs / np.sum(probs)
            
            u = np.random.random()
            cumsum = 0.0
            new_c = 1
            for j in range(len(probs)):
                cumsum += probs[j]
                if u <= cumsum:
                    new_c = j + 1
                    break
            
            # Update cluster assignment
            if new_c > K:
                # Chose new cluster
                if is_singleton:
                    # Singleton choosing new cluster: stay in place
                    cluster_assign[i] = cur_cluster_i
                else:
                    # Non-singleton: create new cluster
                    cluster_assign[i] = K + 1
                    K_new = K + 1
                    
                    # Expand mu, tau matrices
                    mu_list_new = np.zeros((M, K_new, K_new), dtype=np.float64)
                    tau_list_new = np.zeros((M, K_new, K_new), dtype=np.float64)
                    
                    mu_list_new[:, :K, :K] = mu_list
                    tau_list_new[:, :K, :K] = tau_list
                    
                    for m in range(M):
                        for j in range(K_new):
                            tau_val = np.random.gamma(alpha_prior, 1.0 / beta)
                            tau_list_new[m, K, j] = tau_list_new[m, j, K] = tau_val
                            std = np.sqrt(1.0 / (t_0 * tau_val))
                            # New cluster row/col: use mu_0off_list
                            mu_val = np.random.normal(mu_0off_list[m], std)
                            mu_list_new[m, K, j] = mu_list_new[m, j, K] = mu_val
                    
                    mu_list = mu_list_new
                    tau_list = tau_list_new
                    
                    cluster_sizes = np.zeros(K_new, dtype=np.int32)
                    for ii in range(n):
                        cluster_sizes[cluster_assign[ii] - 1] += 1
                    K = K_new
            else:
                # Chose existing cluster
                if is_singleton and new_c != cur_cluster_i:
                    # Singleton moving to different cluster: remove empty cluster
                    cluster_assign[i] = new_c
                    
                    # Relabel clusters to remove gap
                    for ii in range(n):
                        if cluster_assign[ii] > cur_cluster_i:
                            cluster_assign[ii] -= 1
                    if new_c > cur_cluster_i:
                        cluster_assign[i] = new_c - 1
                    
                    # Shrink mu, tau matrices
                    K_new = K - 1
                    if K_new >= 1:
                        mu_list_new = np.zeros((M, K_new, K_new), dtype=np.float64)
                        tau_list_new = np.zeros((M, K_new, K_new), dtype=np.float64)
                        
                        for m in range(M):
                            new_r = 0
                            for old_r in range(K):
                                if old_r == cur_cluster_i - 1:
                                    continue
                                new_s = 0
                                for old_s in range(K):
                                    if old_s == cur_cluster_i - 1:
                                        continue
                                    mu_list_new[m, new_r, new_s] = mu_list[m, old_r, old_s]
                                    tau_list_new[m, new_r, new_s] = tau_list[m, old_r, old_s]
                                    new_s += 1
                                new_r += 1
                        
                        mu_list = mu_list_new
                        tau_list = tau_list_new
                    
                    cluster_sizes = np.zeros(K_new, dtype=np.int32)
                    for ii in range(n):
                        cluster_sizes[cluster_assign[ii] - 1] += 1
                    K = K_new
                else:
                    # Non-singleton or singleton staying: just update
                    cluster_assign[i] = new_c
                    cluster_sizes = np.zeros(K, dtype=np.int32)
                    for ii in range(n):
                        cluster_sizes[cluster_assign[ii] - 1] += 1
        
        # Remove any remaining empty clusters
        cluster_assign, mu_list, tau_list = _remove_empty_clusters(
            cluster_assign, mu_list, tau_list)
        
        # Update parameters
        K = cluster_assign.max()
        mu_list, tau_list = _update_params(
            cluster_assign, A_stack, K, mu_0_list, mu_0off_list, t_0, alpha_prior, beta)
        
        K_trace[iter_idx] = K
        
        # Save MCMC history for this iteration
        z_history[iter_idx, :] = cluster_assign
        K_cur = min(K, max_K)
        for m in range(M):
            mu_history[iter_idx, m, :K_cur, :K_cur] = mu_list[m, :K_cur, :K_cur]
            tau_history[iter_idx, m, :K_cur, :K_cur] = tau_list[m, :K_cur, :K_cur]
    
    return cluster_assign, K_trace, z_history, mu_history, tau_history


def run_sampler(A_list, alpha_list, neighbor_matrix, lambda1, Vn, n_iterations,
                init_K=9, mu_0_list=None, mu_0off_list=None, t_0=2.0, 
                alpha=1.0, beta=1.0, gamma=1.0, seed=100, verbose=True,
                max_K=100):
    """
    Run multi-modal MRFC-MFM Gibbs sampler.
    
    Args:
        A_list: List of similarity matrices, each shape (n, n)
        alpha_list: List of modality weights
        neighbor_matrix: Spatial neighbor matrix (n, n)
        lambda1: Spatial penalty strength
        Vn: Precomputed VN coefficients
        n_iterations: Number of MCMC iterations
        init_K: Initial number of clusters
        mu_0_list: Prior means for diagonal blocks (if None, computed from data)
        mu_0off_list: Prior means for off-diagonal blocks (if None, computed from data)
        t_0, alpha, beta, gamma: Prior parameters
        seed: Random seed
        verbose: Print progress
        max_K: Maximum K for history pre-allocation
    
    Returns:
        result: dict with cluster_assign, K_trace, elapsed_time, history
                history contains per-iteration z, mu, tau arrays for Dahl/mDIC
    """
    M = len(A_list)
    n = A_list[0].shape[0]
    
    # Stack similarity matrices
    A_stack = np.stack(A_list, axis=0).astype(np.float64)
    alpha_weights = np.array(alpha_list, dtype=np.float64)
    
    # Compute mu_0 and mu_0off for each modality if not provided
    if mu_0_list is None:
        mu_0_list = []
        for A in A_list:
            # Diagonal mean
            mu_0_list.append(np.diag(A).mean())
    
    if mu_0off_list is None:
        mu_0off_list = []
        for A in A_list:
            # Off-diagonal mean
            mask = ~np.eye(n, dtype=bool)
            mu_0off_list.append(A[mask].mean())
    
    mu_0_arr = np.array(mu_0_list, dtype=np.float64)
    mu_0off_arr = np.array(mu_0off_list, dtype=np.float64)
    
    if verbose:
        print("=" * 80)
        print("MRFC-MFM Gibbs Sampler (Numba)")
        print("=" * 80)
        print(f"Modalities: {M}, Cells: {n}, Iterations: {n_iterations}")
        print(f"Lambda1: {lambda1}, Init_K: {init_K}, Gamma: {gamma}")
        print(f"Alpha weights: {alpha_weights}")
        print(f"mu_0 (diagonal) per modality: {mu_0_arr}")
        print(f"mu_0off (off-diagonal) per modality: {mu_0off_arr}")
        print("=" * 80)
    
    start_time = time.time()
    
    cluster_assign, K_trace, z_history, mu_history, tau_history = \
        _gibbs_sampler_core(
            A_stack, alpha_weights, neighbor_matrix.astype(np.int32), 
            lambda1, Vn, n_iterations, init_K, mu_0_arr, mu_0off_arr,
            t_0, alpha, beta, gamma, seed=seed, max_K=max_K
        )
    
    elapsed = time.time() - start_time
    
    if verbose:
        K_final = int(cluster_assign.max())
        cluster_sizes = np.bincount(cluster_assign)[1:]
        size_str = ' '.join([f'C{j+1}:{s}' for j, s in enumerate(cluster_sizes)])
        print(f"Final: K={K_final}, Sizes: {size_str}")
        print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        print("=" * 80)
    
    return {
        'cluster_assign': cluster_assign,
        'K_trace': K_trace,
        'elapsed_time': elapsed,
        'history': {
            'z': z_history,          # (n_iterations, n)
            'mu': mu_history,        # (n_iterations, M, max_K, max_K)
            'tau': tau_history,       # (n_iterations, M, max_K, max_K)
        }
    }


__all__ = ['run_sampler']
