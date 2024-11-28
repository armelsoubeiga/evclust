# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Evidential Gaussian Mixture Model  (EGMM).  :
    Lianmeng Jiao, Thierry Denœux, Zhun-ga Liu, Quan Pan, EGMM: An evidential version of the Gaussian mixture model for clustering,
    Applied Soft Computing, Volume 129, 2022, 109619, ISSN 1568-4946.
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
from sklearn.cluster import KMeans





#---------------------- EGMM------------------------------------------------
def egmm(X, c, type='simple', pairs=None, Omega=True, max_iter=20, epsi=1e-3, init='kmeans', disp=True):
    """
    Evidential Gaussian Mixture Model (EGMM) clustering algorithm. 
    Model parameters are estimated by Expectation-Maximization algorithm and by 
    extending the classical GMM in the belief function framework directly.

    Parameters:
    ------------
    X (ndarray):
        Input data of shape (n_samples, n_features).
    c (int):
        Number of clusters.
    type (str, optional):
        Type of focal sets ('simple', 'full', 'pairs'). Default is 'simple'.
    pairs (ndarray, None): 
        Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
    Omega (bool): 
        If True (default), the whole frame is included (for types 'simple' and 'pairs').
    max_iter ( int, optional):
        Maximum number of iterations. Default is 100.
    epsi (float, optional): 
        Convergence tolerance for the algorithm. Default is 1e-6.
    init (str, optional):
        Initialization method ('random' or 'kmeans'). Default is 'random'.

    Returns:
    ---------
    The credal partition (an object of class "credpart").

    
    Example:
    --------
    .. highlight:: python
    .. code-block:: python
    
        from evclust.egmm import egmm
        import numpy as np
        import matplotlib.pyplot as plt

        np.random.seed(42); n = 200 
        X1 = np.random.normal(loc=[1, 3], scale=0.5, size=(n//3, 2))
        X2 = np.random.normal(loc=[7, 5], scale=0.3, size=(n//3, 2))
        X3 = np.random.normal(loc=[10, 2], scale=0.7, size=(n//3, 2))
        X = np.vstack([X1, X2, X3])

        clus = egmm(X, c=3, type='full', max_iter=20, epsi=1e-3, init='kmeans')

        clus['F']  # Focal sets
        clus['g']  # Cluster centroids
        clus['mass']  # Mass functions
        clus['y_pl']  # Maximum plausibility clusters
        
    References:
    -----------
        Lianmeng Jiao, Thierry Denœux, Zhun-ga Liu, Quan Pan, EGMM: An evidential version of the Gaussian mixture model for clustering,
        Applied Soft Computing, Volume 129, 2022, 109619, ISSN 1568-4946.


    .. seealso::
        :func:`~extractMass`, :func:`~makeF`, 
        :func:`~init_params_random_egmm`,  :func:`~init_params_kmeans_egmm`
    
    .. note::
        Keywords : Belief function theory, Evidential partition, Gaussian mixture model, Model-based clustering, Expectation–Maximization
        The parameters in EGMM are estimated by a specially designed Expectation–Maximization (EM) algorithm. 
        A validity index allowing automatic determination of the proper number of clusters is also provided. 
        The proposed EGMM is as simple as the classical GMM, but can generate a more informative evidential partition for the considered dataset.

    """
    n, p = X.shape
    F = makeF(c, type, pairs, Omega) 
    f = F.shape[0]
    
    # Initialize parameters
    if init == 'random':
        g, S, alpha, mass = init_params_random_egmm(X, c, f)
    elif init == 'kmeans':
        g, S, alpha, mass = init_params_kmeans_egmm(X, c, f)
    else:
        raise ValueError("Invalid initialization method. Choose 'kmeans' or 'random'.")
    
    # Iterative optimization
    L = []
    for it in range(max_iter):
        # E-step: Update the mass functions
        D = np.zeros((n, f))
        for j in range(f):
            indices = np.where(F[j, :] == 1)[0]
            if len(indices) == 0:  # Empty set
                D[:, j] = np.inf
            else:
                d = np.zeros((n, len(indices)))
                for k, cluster in enumerate(indices):
                    delta = X - g[cluster]
                    d[:, k] = np.sum(delta @ S[cluster] * delta, axis=1)
                D[:, j] = np.sum(d, axis=1) / len(indices)

        D = np.exp(-D)
        mass[:, 1:] = (D[:, 1:] * alpha[1:]) / (np.sum(D[:, 1:] * alpha[1:], axis=1, keepdims=True) + np.finfo(float).eps)
        mass[:, 0] = 1 - np.sum(mass[:, 1:], axis=1)

        # M-step: Update parameters
        g_old = g.copy()
        for k in range(c):
            weights = np.sum(mass[:, F[:, k] == 1], axis=1)
            total_weight = np.sum(weights)

            if total_weight > 1e-6:
                g[k] = np.sum((weights[:, np.newaxis] * X), axis=0) / total_weight
                delta = X - g[k]
                regularization_term = 1e-6
                cov_matrix = (delta.T @ (delta * weights[:, np.newaxis])) / total_weight
                cov_matrix += regularization_term * np.eye(p)
                condition_number = np.linalg.cond(cov_matrix)
                if condition_number > 1e12:
                    S[k] = np.eye(p)  
                else:
                    S[k] = np.linalg.inv(cov_matrix)
            else:
                S[k] = np.eye(p)
        
        alpha = np.sum(mass, axis=0) / n

        # Compute log-likelihood
        L.append(np.sum(np.log(np.sum(D[:, 1:] * alpha[1:], axis=1) + np.finfo(float).eps)))
        if disp:
            print(it, round(float(L[-1]), 3))
        # Check for convergence
        if it > 0 and np.abs(L[-1] - L[-2]) < epsi:
            break
        
    # Compute EBIC (Evidential Bayesian Information Criterion)
    params_count = f-1 + c*p + p*(p+1)/2
    EBIC = float(L[-1]) - params_count * np.log(n)/2
    
    clus = extractMass(mass, F, g=g, S=S, method='EGMM', crit=L[-1], param={'alpha': alpha, 'L': L, 'EBIC': EBIC})
    return clus






#---------------------- Utils for EGMM------------------------------------------------  
def init_params_random_egmm(X, c, f):
    """
    Random initialization of EGMM parameters.
    """
    n, p = X.shape
    g = np.random.rand(c, p) * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
    S = [np.eye(p) for _ in range(c)]
    alpha = np.random.rand(f)
    alpha /= np.sum(alpha)
    mass = np.random.rand(n, f)
    mass /= np.sum(mass, axis=1, keepdims=True)
    return g, S, alpha, mass


def init_params_kmeans_egmm(X, c, f):
    """
    KMeans-based initialization of EGMM parameters.
    """
    kmeans = KMeans(n_clusters=c, random_state=42).fit(X)
    g = kmeans.cluster_centers_
    labels = kmeans.labels_
    n, p = X.shape
    S = [np.cov(X[labels == k].T) if np.sum(labels == k) > 1 else np.eye(p) for k in range(c)]
    alpha = np.random.rand(f)
    alpha /= np.sum(alpha)
    mass = np.random.rand(n, f)
    mass /= np.sum(mass, axis=1, keepdims=True)
    return g, S, alpha, mass

