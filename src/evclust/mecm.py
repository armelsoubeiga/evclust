# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Median evidential c-means algorithm (MECM).

    Kuang Zhou, Arnaud Martin, Quan Pan, Zhun-ga Liu,
    Median evidential c-means algorithm and its application to community detection,
    Knowledge-Based Systems, Volume 74, 2015, Pages 69-88, ISSN 0950-7051.
"""
#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np



#----------------------mecm---------------------------------------------------
def mecm(x, c, type='full', alpha=1, beta=2, delta=10, epsi=1e-4, maxit=20,
         disp=True, gamma=0.5, eta=1):
    """
    Median evidential c-means algorithm (MECM) is using to partitioning relational data.
    MECM is an extension of median c-means and median fuzzy c-means on the theoretical framework of belief function.
    The median variant relaxes the restriction of a metric space embedding for the objects but constrains the prototypes to be in the original data set. 
    
    Parameters:
    -----------
    x (Matrix): 
        A distance or dissimilarity matrix of size n x n, where n is the number of objects.
    c (int): 
        Number of clusters.
    type (str): 
        Type of focal sets ("simple": empty set, singletons, and Omega; "full": all 2^c subsets of Omega;
            "pairs": empty set, singletons, Omega, and all or selected pairs).
    alpha (float): 
        Exponent of the cardinality in the cost function.
    beta (float): 
        Exponent of masses in the cost function.
    delta (float): 
        Distance to the empty set. If None, it is set to the 95th percentile of the upper triangle of x.
    epsi (float): 
        Minimum amount of improvement.
    maxit (int): 
        Maximum number of iterations.
    disp (bool): 
        If True (default), intermediate results are displayed.
    gamma (float): 
        [0,1] (0.5 default) weighs the contribution of uncertainty to the dissimilarity between objects and imprecise clusters.
    eta (float): 
        > 0 (1 default) use to distinguish the outliers from the possible medoids (default 1).

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
        D = euclidean_distances(df)

        # MECM clustering
        from evclust.mecm import mecm
        clus = mecm(x=D, c=2, type='simple', alpha=1, beta=2, delta=10, epsi=1e-4, maxit=20,
                        disp=True, gamma=0.5, eta=1)

    References:
    -----------
        Kuang Zhou, Arnaud Martin, Quan Pan, Zhun-ga Liu,
        Median evidential c-means algorithm and its application to community detection,
        Knowledge-Based Systems, Volume 74, 2015, Pages 69-88, ISSN 0950-7051.

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`

    .. note::
        Keywords : Credal partition, Belief function theory, Median clustering, Community detection, Imprecise communities
        Median Evidential C-Means (MECM), which is an extension of median c-means and median fuzzy c-means on the theoretical framework of belief functions is proposed.  
        Due to these properties, MECM could be applied to graph clustering problems. 
        A community detection scheme for social networks based on MECM is investigated and the obtained credal partitions of graphs, which are more refined than crisp and fuzzy ones, enable us to have a better understanding of the graph structures.
    """
    
    #---------------------- initialisations -----------------------------------
    n, d = x.shape
    
    if delta is None:
        delta2 = np.quantile(x[np.triu_indices(n, k=1)], 0.95)
    else:
        delta2 = delta**2

    F = makeF(c=c, type=type, pairs=None, Omega=True)
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)
    g = x[np.random.choice(n, c, replace=False), :]

    pasfini = True
    Jold = np.inf
    gplus = np.zeros((f-1, d))
    iter_count = 0

    #------------------------ iterations---------------------------------------
    while pasfini and iter_count < maxit:
        iter_count += 1

        # Update of imprecise (medoid - prototype)
        for i in range(1, f):
            fi = F[i, :]
            truc = np.tile(fi, (d, 1)).T * g
            truc_non_zero = np.any(truc != 0, axis=1)
            if np.sum(fi) == 1:
                gplus[i-1, :] = truc[truc_non_zero, :]
            elif np.sum(fi) > 1:
                truc = truc[truc_non_zero, :]
                p_values = np.zeros(n)
                for p in range(n):
                    p_values[p] = (1 / card[i - 1]) * np.sum([(truc[j, p] - ((1 / card[i - 1]) * np.sum(truc[:, p]))) ** 2 for j in range(len(truc[:, p]))]) + eta * (1 / card[i - 1]) * np.sum(truc[:, p])
                p = np.argmin(p_values)
                gplus[i-1, :] = x[p, :]

        ## Calculation of distances to centers
        D = np.zeros((n, f-1))
        for j in range(f-1):
            fj = F[j+1, :]
            if np.sum(fj) == 1:
                D[:, j] = np.sum((x - gplus[j, :]) ** 2, axis=1)
            elif np.sum(fj) > 1:
                for i in range(n):
                    d_ij = np.sum((x[i, :] - gplus[j, :]) ** 2)
                    min_d = np.min([np.sum((x[i, :] - g[k, :]) ** 2) for k in range(c) if fj[k] == 1])
                    rho_ij = np.sum([np.sqrt((np.sum((x[i, :] - g[k, :]) ** 2) - np.sum((x[i, :] - g[l, :]) ** 2)) ** 2) for k in range(c) for l in range(c) if fj[k] == 1 and fj[l] == 1]) / (eta * np.sum([np.sum((g[k, :] - g[l, :]) ** 2) for k in range(c) for l in range(c) if fj[k] == 1 and fj[l] == 1]))
                    D[i, j] = (gamma * d_ij + rho_ij * min_d) / (gamma + 1)

        # Calcul des masses
        m = np.zeros((n, f-1))
        for i in range(n):
            if (D[i, :] == 0).any():
                for j in range(f-1):
                    if D[i, j] == 0:  
                        m[i, j] = 1
                    else:  
                        m[i, j] = 0
            else:  
                vect0 = D[i, :]
                for j in range(f-1):
                    vect1 = (np.tile(D[i, j], f-1) / vect0) ** (1 / (beta - 1))
                    vect2 = np.tile(card[j] ** (alpha / (beta - 1)), f-1) / (card ** (alpha / (beta - 1)))
                    vect3 = vect1 * vect2
                    m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / delta2) ** (1 / (beta - 1)))
                    if np.isnan(m[i, j]):
                        m[i, j] = 1  

        # Calcul des centres
        V = np.zeros((n, f-1))
        for k in range(f-1):
            fk = F[k+1, :]
            if np.sum(fk) == 1:
                mk_beta = m[:, k]**beta
                for i in range(n):
                    V[i, k] = np.sum(x[i, :] * mk_beta)

        V_filtered = V[:, np.sum(V, axis=0) > 0]
        g_indices = np.argmin(V_filtered, axis=0)
        g = x[g_indices, :]

        mvide = 1 - np.sum(m, axis=1)
        J = np.sum((m ** beta) * D * np.tile(card ** alpha, (n, 1))) + delta2 * np.sum(mvide ** beta)
        if disp:
            print(iter_count, J)
        pasfini = np.abs(J - Jold) > epsi
        Jold = J

    m = np.hstack((1 - np.sum(m, axis=1).reshape(n, 1), m))
    clus = extractMass(m, F, g=g, method="mecm", crit=J, 
                       param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus
