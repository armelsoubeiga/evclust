# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Credal c-means (CCM) clustering method based on belief functions.

    Zhun-ga Liu, Quan Pan, Jean Dezert, Grégoire Mercier, Credal c-means clustering method based on belief functions,
    Knowledge-Based Systems, Volume 74, 2015, Pages 119-132, ISSN 0950-7051
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
from scipy.cluster.vq import kmeans




#----------------------ccm------------------------------------------------
def ccm(x, c, type="full", gamma=1, beta=2, delta=10, epsi=1e-3, 
        maxit=50, init="kmeans", disp=True):
    """
    Credal c-means (CCM) is a clustering method designed to handle uncertain and imprecise data. 
    It addresses this challenge by assigning data points to meta-clusters, which are essentially unions of closely related clusters. 
    This approach allows for more flexible and nuanced cluster assignments, reducing misclassification errors. 
    CCM is particularly robust to noisy data due to the inclusion of an outlier cluster.

    Parameters:
    -----------
    x (Matrix): 
        Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
    c (int): 
        Number of clusters.
    type (str): 
        Type of focal sets ("simple": empty set, singletons, and Omega; "full": all 2^c subsets of Omega;
            "pairs": empty set, singletons, Omega, and all or selected pairs).
    beta (float): 
        Exponent of masses in the cost function.
    delta (float): 
        > 0. Distance to the empty set. 
    epsi (float): 
        Minimum amount of improvement.
    maxit (int): 
        Maximum number of iterations.
    disp (bool): 
        If True (default), intermediate results are displayed.
    gamma (float): 
        > 0 weight of the distance (by default take 1).

    Returns:
    --------
    The credal partition (an object of class "credpart").

    
    Example:
    --------
    .. highlight:: python
    .. code-block:: python

        # Test data
        from sklearn.metrics.pairwise import euclidean_distances
        from evclust.datasets import load_iris
        df = load_iris()
        df = df.drop(['species'], axis = 1)

        # CCM clustering
        from evclust.ccm import ccm
        clus = ccm(x=df, c=3, type="full", gamma=1, beta=2, delta=10, epsi=1e-3, 
                maxit=50, init="kmeans", disp=True)

    References:
    -----------
        Zhun-ga Liu, Quan Pan, Jean Dezert, Grégoire Mercier, Credal c-means clustering method based on belief functions,
        Knowledge-Based Systems, Volume 74, 2015, Pages 119-132, ISSN 0950-7051.

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`

    .. note::
        Keywords : Belief functions, Credal partition, Data clustering, Uncertain data
        A meta-cluster threshold is introduced in CCM to eliminate the meta-clusters with big cardinality. 
        CCM can still provide good clustering results with admissible computation complexity. 
        The credal partition can be easily approximated to a fuzzy (probabilistic) partition if necessary, 
        thanks to the plausibility transformation, pignistic transformation or any other available transformation that may be preferred by the user. 
        The output of CCM is not necessarily used for making the final classification, but it can serve as an interesting source of information to combine with additional complementary information sources if one wants to get more precise results before taking decision. 
        The effectiveness of CCM has been proved through different experiments using artificial and real data sets. 
    """
    
    #---------------------- initialisations -----------------------------------
    x = np.array(x)
    n, p = x.shape
    delta2 = delta ** 2

    # Construction des ensembles focaux
    F = makeF(c, type=type, pairs=None, Omega=True)
    f = F.shape[0]
    cardinalities = np.sum(F[1:, :], axis=1)

    # Initialisation des prototypes
    if init == "kmeans":
        g, _ = kmeans(x, c)
    else:
        g = x[np.random.choice(n, c), :] + 0.1 * np.random.randn(c * p).reshape(c, p)

    pasfini = True
    Jold = np.inf
    g_plus = np.zeros((f - 1, p))
    iter = 0

    while pasfini and iter < maxit:
        iter += 1

        # Step 1 : Calcul des prototypes étendus 
        for i in range(1, f):
            subset = F[i, :]
            mask = np.tile(subset, (p, 1)).T
            denominator = np.sum(subset)
            if denominator > 0:
                g_plus[i - 1, :] = np.sum(g * mask, axis=0) / denominator

        # Step 2 : Calcul des distances généralisées D_ij^2
        D = np.zeros((n, f - 1))
        for j in range(f - 1):
            subset = F[j + 1, :]
            active_centroids = g[subset == 1, :]
            if active_centroids.shape[0] == 1:
                D[:, j] = np.linalg.norm(x - active_centroids[0], axis=1) ** 2
            elif active_centroids.shape[0] > 1:
                distances = np.linalg.norm(x[:, None, :] - active_centroids, axis=2) ** 2
                numerator = np.sum(distances, axis=1) + gamma * np.min(distances, axis=1)
                D[:, j] = numerator / (np.sum(subset) + gamma)

        D[:, 0] = delta2  # Ensemble vide

        # Step 3 : Calcul des masses m_{ij}
        m = np.zeros((n, f - 1))
        for i in range(n):
            denom = 0
            for j in range(f - 1):
                if np.sum(F[j + 1, :]) == 0:
                    m[i, j] = delta ** (-2 / (beta - 1))
                elif np.sum(F[j + 1, :]) == 1:
                    m[i, j] = D[i, j] ** (-1 / (beta - 1))
                else:
                    numerator = D[i, j]
                    m[i, j] = (numerator ** (-1 / (beta - 1)))
                denom += m[i, j]
            m[i, :] /= denom

        # Step 4 : Matrice B et H pour recalculer les prototypes g
        B = np.zeros((c, p))
        for l in range(c):
            subset_l = (F[:, l] == 1)
            indices_l = np.where(subset_l)[0] - 1  # Correspondance des indices avec m
            if indices_l.size > 0:
                mi_beta = m[:, indices_l] ** beta
                weights = (1 + gamma / cardinalities[indices_l]) / (cardinalities[indices_l] + gamma)
                weighted_masses = (mi_beta * weights).T
                B[l, :] = np.sum(weighted_masses[:, :, None] * x[None, :, :], axis=(0, 1))

        H = np.zeros((c, c))
        for l in range(c):
            for q in range(c):
                if l == q:
                    subset_l = (F[:, l] == 1)
                    indices_l = np.where(subset_l)[0] - 1
                    if indices_l.size > 0:
                        weights_diag = (1 + gamma / (cardinalities[indices_l] ** 2)) / (cardinalities[indices_l] + gamma)
                        H[l, q] = np.sum(m[:, indices_l] ** beta * weights_diag)
                else:
                    subset_joint = (F[:, l] == 1) & (F[:, q] == 1)
                    indices_joint = np.where(subset_joint)[0] - 1
                    if indices_joint.size > 0:
                        weights_offdiag = gamma / (cardinalities[indices_joint] ** 2 * (cardinalities[indices_joint] + gamma))
                        H[l, q] = np.sum(m[:, indices_joint] ** beta * weights_offdiag)

        g = np.linalg.solve(H, B)

        # Calcul de la fonction objective
        J = np.sum((m ** beta) * D) + delta2 * np.sum((1 - np.sum(m, axis=1)) ** beta)
        if disp:
            print(f"{iter}, {J}")

        pasfini = np.abs(J - Jold) > epsi
        Jold = J

    # Extraction des masses finales
    m_final = np.concatenate((1 - np.sum(m, axis=1).reshape(-1, 1), m), axis=1)
    clus = extractMass(m_final, F, g=g, method="ccm", crit=J, 
                       param={'gamma': gamma, 'beta': beta, 'delta': delta})
    return clus


