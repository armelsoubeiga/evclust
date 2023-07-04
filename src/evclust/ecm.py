# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains the main function for ecm
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
from scipy.cluster.vq import kmeans




#---------------------- ecm------------------------------------------------
def ecm(x, c, g0=None, type='full', pairs=None, Omega=True, ntrials=1, alpha=1, beta=2, delta=10,
        epsi=1e-3, init="kmeans", disp=True):
    """
    Evidential c-means algorithm. `ecm` Computes a credal partition from a matrix of attribute data using the Evidential c-means (ECM) algorithm.

    ECM is an evidential version algorithm of the Hard c-Means (HCM) and Fuzzy c-Means (FCM) algorithms.
    As in HCM and FCM, each cluster is represented by a prototype. However, in ECM, some sets of clusters
    are also represented by a prototype, which is defined as the center of mass of the prototypes in each
    individual cluster. The algorithm iteratively optimizes a cost function, with respect to the prototypes
    and to the credal partition. By default, each mass function in the credal partition has 2^c focal sets,
    where c is the supplied number of clusters. We can also limit the number of focal sets to subsets of
    clusters with cardinalities 0, 1 and c (recommended if c>=10), or to all or some selected pairs of clusters.
    If initial prototypes g0 are provided, the number of trials is automatically set to 1.

    Parameters:
    ----------
    x: 
        input matrix of size n x d, where n is the number of objects and d is the number of attributes.
    c: 
        Number of clusters.
    g0: 
        Initial prototypes, matrix of size c x d. If not supplied, the prototypes are initialized randomly.
    type: 
        Type of focal sets ("simple": empty set, singletons and Omega; "full": all 2^c subsets of Omega;
            "pairs": empty set, singletons, Omega, and all or selected pairs).
    pairs: 
        Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
    Omega: 
        Logical. If True (default), the whole frame is included (for types 'simple' and 'pairs').
    ntrials (int): 
        Number of runs of the optimization algorithm (set to 1 if g0 is supplied).
    alpha (float): 
        Exponent of the cardinality in the cost function.
    beta (float): 
        Exponent of masses in the cost function.
    delta (float): 
        Distance to the empty set.
    epsi (float): 
        Minimum amount of improvement.
    init (str): 
        Initialization: "kmeans" (default) or "rand" (random).
    disp (bool): 
        If True (default), intermediate results are displayed.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    References:
    ----------
    M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm.
      Pattern Recognition, Vol. 41, Issue 4, pages 1384--1397, 2008.

    Examples:
    --------
    """
    

    #---------------------- initialisations --------------------------------------

    x = np.array(x)
    n = x.shape[0]
    d = x.shape[1]
    delta2 = delta ** 2

    if (ntrials > 1) and (g0 is not None):
        print('WARNING: ntrials>1 and g0 provided. Parameter ntrials set to 1.')
        ntrials = 1

    F = makeF(c, type, pairs, Omega)
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)

    #------------------------ iterations--------------------------------
    Jbest = np.inf
    for itrial in range(ntrials):
        if g0 is None:
            if init == "kmeans":
                centroids, distortion = kmeans(x, c)
                g = centroids
            else:
                g = x[np.random.choice(n, c), :] + 0.1 * np.random.randn(c * d).reshape(c, d)
        else:
            g = g0
        pasfini = True
        Jold = np.inf
        gplus = np.zeros((f-1, d))
        iter = 0
        while pasfini:
            iter += 1
            for i in range(1, f):
                fi = F[i, :]
                truc = np.tile(fi, (d, 1)).T
                gplus[i-1, :] = np.sum(g * truc, axis=0) / np.sum(fi)

            # calculation of distances to centers
            D = np.zeros((n, f-1))
            for j in range(f-1):
                D[:, j] = np.nansum((x - np.tile(gplus[j, :], (n, 1))) ** 2, axis=1)

            # Calculation of masses
            m = np.zeros((n, f-1))
            for i in range(n):
                vect0 = D[i, :]
                for j in range(f-1):
                    vect1 = (np.tile(D[i, j], f-1) / vect0) ** (1 / (beta-1))
                    vect2 = np.tile(card[j] ** (alpha / (beta-1)), f-1) / (card ** (alpha / (beta-1)))
                    vect3 = vect1 * vect2
                    m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / delta2) ** (1 / (beta-1)))
                    if np.isnan(m[i, j]):
                        m[i, j] = 1  # in case the initial prototypes are training vectors

            # Calculation of centers
            A = np.zeros((c, c))
            for k in range(c):
                for l in range(c):
                    truc = np.zeros(c)
                    truc[[k, l]] = 1
                    t = np.tile(truc, (f, 1))
                    indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]    # indices of all Aj including wk and wl
                    indices = indices - 1
                    if len(indices) == 0:
                        A[l, k] = 0
                    else:
                        for jj in range(len(indices)):
                            j = indices[jj]
                            mj = m[:, j] ** beta
                            A[l, k] += np.sum(mj) * card[j] ** (alpha - 2)

            # Construction of the B matrix
            B = np.zeros((c, d))
            for l in range(c):
                truc = np.zeros(c)
                truc[l] = 1
                t = np.tile(truc, (f, 1))
                indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]   # indices of all Aj including wl
                indices = indices - 1
                mi = np.tile(card[indices] ** (alpha - 1), (n, 1)) * m[:, indices] ** beta
                s = np.sum(mi, axis=1)
                mats = np.tile(s.reshape(n, 1), (1, d))
                xim = x * mats
                B[l, :] = np.sum(xim, axis=0)

            g = np.linalg.solve(A, B)

            mvide = 1 - np.sum(m, axis=1)
            J = np.nansum((m ** beta) * D * np.tile(card.reshape(1, f-1), (n, 1))) + delta2 * np.nansum(mvide ** beta)
            if disp:
                print([iter, J])
            pasfini = (np.abs(J - Jold) > epsi)
            Jold = J

        if J < Jbest:
            Jbest = J
            mbest = m
            gbest = g
        res = np.array([itrial, J, Jbest])
        res = np.squeeze(res)
        if ntrials > 1:
            print(res)

    m = np.concatenate((1 - np.sum(mbest, axis=1).reshape(n, 1), mbest), axis=1)
    clus = extractMass(m, F, g=gbest, method="ecm", crit=Jbest, param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus