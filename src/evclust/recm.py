# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains the main function for cecm : Relational ecm
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np


def recm(D, c, type='full', pairs=None, Omega=True, m0=None, ntrials=1, alpha=1, beta=1.5,
         delta=None, epsi=1e-4, maxit=5000, disp=True):
    
    """
    Relational Evidential c-means algorithm. `recm` computes a credal partition from a dissimilarity matrix using the Relational Evidential c-means (RECM) algorithm.

    RECM is a relational version of the Evidential c-Means (ECM) algorithm. Convergence is guaranteed only if elements of matrix D are squared Euclidean distances.
    However, the algorithm is quite robust and generally provides sensible results even if the dissimilarities are not metric.
    By default, each mass function in the credal partition has 2^c focal sets, where c is the supplied number of clusters.
    We can also limit the number of focal sets to subsets of clusters with cardinalities 0, 1 and c (recommended if c >= 10), or to all or some selected pairs of clusters.
    If an initial credal partition m0 is provided, the number of trials is automatically set to 1.

    Parameters:
    ----------
        D (ndarray): 
            Dissimilarity matrix of size (n,n), where n is the number of objects. Dissimilarities must be squared Euclidean distances to ensure convergence.
        c (int): 
            Number of clusters.
        type (str): 
            Type of focal sets ("simple": empty set, singletons and Omega; "full": all 2^c subsets of Omega; "pairs": empty set, singletons, Omega, and all or selected pairs).
        pairs (ndarray or None): 
            Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
        Omega (bool): 
            If True (default), the whole frame is included (for types 'simple' and 'pairs').
        m0 (ndarray or None): 
            Initial credal partition. Should be a matrix with n rows and a number of columns equal to the number f of focal sets specified by 'type' and 'pairs'.
        ntrials (int): 
            Number of runs of the optimization algorithm (set to 1 if m0 is supplied).
        alpha (float): 
            Exponent of the cardinality in the cost function.
        beta (float): 
            Exponent of masses in the cost function.
        delta (float): 
            Distance to the empty set.
        epsi (float): 
            Minimum amount of improvement.
        maxit (int): 
            Maximum number of iterations.
        disp (bool): 
            If True (default), intermediate results are displayed.

    Returns:
    -------
        clus (object): 
            The credal partition (an object of class "credpart").

    References:
    ----------
        M.-H. Masson and T. Denoeux. RECM: Relational Evidential c-means algorithm. Pattern Recognition Letters, Vol. 30, pages 1015--1026, 2009.

    Author:
   -------
        Armel Soubeiga (from Thierry Denoeux code in R and from a MATLAB code written by Marie-Helene Masson).
    """
    if delta is None:
        delta = np.quantile(D[np.triu_indices(D.shape[0], k=1)], q=0.95)
        
    if ntrials > 1 and m0 is not None:
        print('WARNING: ntrials>1 and m0 provided. Parameter ntrials set to 1.')
        ntrials = 1

    D = np.asmatrix(D)

    # Scalar products computation from distances
    delta2 = delta ** 2
    delta2 = delta2.item()

    n = D.shape[1]
    e = np.ones(n)
    Q = np.diag(e) - np.outer(e, e) / n
    XX = -0.5 * np.dot(np.dot(Q, D), Q)

    # Initializations
    F = makeF(c, type, pairs, Omega)
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)

    if m0 is not None:
        if m0.shape[0] != n or m0.shape[1] != f:
            raise ValueError("ERROR: dimension of m0 is not compatible with specified focal sets")

    # Optimization
    Jbest = np.inf
    for itrial in range(1, ntrials + 1):
        if m0 is None:
            m = np.random.uniform(size=(n, f - 1))
            m = m / np.sum(m, axis=1)[:, np.newaxis]
        else:
            m = m0[:, 1:f]

        pasfini = True
        Mold = np.full((n, f), 1e9)
        it = 0
        while pasfini and it < maxit:
            it += 1
            # Construction of the H matrix
            H = np.zeros((c, c))
            for k in range(1, c + 1):
                for l in range(1, c + 1):
                    truc = np.zeros(c)
                    truc[k-1] = 1
                    truc[l-1] = 1
                    t = np.tile(truc, (f, 1))
                    indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]
                    indices = indices - 1
                    if len(indices) == 0:
                        H[l-1, k-1] = 0
                    else:
                        for jj in range(len(indices)):
                            j = indices[jj]
                            mj = m[:, j] ** beta
                            H[l-1, k-1] += np.sum(mj) * card[j] ** (alpha - 2)

            # Construction of the U matrix
            U = np.zeros((c, n))
            for l in range(1, c + 1):
                truc = np.zeros(c)
                truc[l-1] = 1
                t = np.tile(truc, (f, 1))
                indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]
                indices = indices - 1
                mi = np.tile(card[indices] ** (alpha - 1), (n, 1)) * m[:, indices] ** beta
                U[l-1, :] = np.sum(mi, axis=1)

            B = np.dot(U, XX)
            VX = np.linalg.solve(H, B)
            B = np.dot(U, VX.T)
            VV = np.linalg.solve(H, B.T)

            # distances to focal elements
            D = np.zeros((n, f - 1))
            for i in range(n):
                for j in range(f - 1):
                    ff = F[j+1, :]
                    truc = np.dot(ff, ff.T)  # indices of pairs (wk, wl) in Aj
                    indices = np.where(ff == 1)[0]  # indices of classes in Aj
                    D[i, j] = XX[i, i] - 2 * np.sum(VX[indices, i]) / card[j] + np.sum(truc * VV) / (card[j] ** 2)

            # masses
            m = np.zeros((n, f - 1))
            for i in range(n):
                vect0 = D[i, :]
                for j in range(f - 1):
                    vect1 = (np.repeat(D[i, j], f - 1) / vect0) ** (1 / (beta - 1))
                    vect2 = np.repeat(card[j] ** (alpha / (beta - 1)), f - 1) / (card ** (alpha / (beta - 1)))
                    vect3 = vect1 * vect2
                    m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / delta2) ** (1 / (beta - 1)))
            
            mvide = 1 - np.sum(m, axis=1)
            M = np.column_stack((m, mvide))
            J = np.nansum((m ** beta) * D[:, :f - 1] * np.tile(card[:f - 1] ** alpha, (n, 1))) + delta2 * np.nansum(mvide[:f - 1] ** beta)
            DeltaM = np.linalg.norm(M - Mold, ord='fro') / (n * f)
            if disp:
                print([J, DeltaM])
            pasfini = (DeltaM > epsi)
            Mold = M

        if J < Jbest:
            Jbest = J
            mbest = m
            
        res = [itrial, J, Jbest]
        if ntrials > 1:
            print(res)

    # add mass to the empty set
    m = np.column_stack((1 - np.sum(mbest, axis=1), mbest))
    clus = extractMass(m, F, method="recm", crit=Jbest)
    return clus