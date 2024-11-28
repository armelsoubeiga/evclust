# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for kevclus :

    T. Denoeux and M.-H. Masson. EVCLUS: Evidential Clustering of Proximity Data.
    IEEE Transactions on Systems, Man and Cybernetics B, Vol. 34, Issue 1, 95--109, 2004.

    T. Denoeux, S. Sriboonchitta and O. Kanjanatarakul. Evidential clustering of large
    dissimilarity data. Knowledge-Based Systems, vol. 106, pages 179-195, 2016.
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import lsq_linear




#---------------------- kevclus------------------------------------------------
def kevclus(x=None, k=None, D=None, J=None, c=None, type='simple', pairs=None, m0=None, ntrials=1,
            disp=True, maxit=20, epsi=1e-3, d0=None, tr=False, change_order=False, norm=1):
    """
    k-EVCLUS algorithm for evidential clustering computes a credal partition from a dissimilarity matrix.
    This version of the EVCLUS algorithm uses the Iterative Row-wise Quadratic Programming (IRQP) algorithm (see ter Braak et al., 2009). 
    It also makes it possible to use only a random sample of the dissimilarities, reducing the time and space complexity from quadratic to roughly linear (Denoeux et al., 2016).

    Parameters:
    -----------
    x (Matrix, optional): 
        A matrix of size (n, p) containing the values of p attributes for n objects.
    k (int, optional): 
        Number of distances to compute for each object. Default is n-1.
    D (Matrix, optional): 
        An nxn or nxk dissimilarity matrix. Used only if x is not supplied.
    J (Matrix, optional): 
        An nxk matrix of indices. D[i, j] is the distance between objects i and J[i, j].
        Used only if D is supplied and ncol(D) < n; then k is set to ncol(D).
    c (int): 
        Number of clusters.
    type (str): 
        Type of focal sets ("simple": empty set, singletons, and Omega; 
            "full": all 2^c subsets of Omega; "pairs": empty set, singletons, Omega, and all or selected pairs).
    pairs (list, optional): 
        Set of pairs to be included in the focal sets. If None, all pairs are included. 
        Used only if type="pairs".
    m0 (Matrix, optional): 
        Initial credal partition. Should be a matrix with n rows and a number of columns equal 
        to the number of focal sets specified by `type` and `pairs`.
    ntrials (int): 
        Number of runs of the optimization algorithm. Default is 1. 
        Set to 1 if m0 is supplied and change_order=False.
    disp (bool): 
        If True (default), intermediate results are displayed.
    maxit (int): 
        Maximum number of iterations.
    epsi (float): 
        Minimum amount of improvement. Default is 1e-5.
    d0 (float, optional): 
        Parameter used for matrix normalization. The normalized distance corresponding 
        to d0 is 0.95. Default is set to the 90th percentile of D.
    tr (bool): 
        If True, a trace of the stress function is returned.
    change_order (bool): 
        If True, the order of objects is changed at each iteration of the IRQP algorithm.
    norm (int): 
        Normalization of distances. 
        1: division by mean(D^2) (default); 
        2: division by n*p.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    Example:
    --------
    .. highlight:: python
    .. code-block:: python
    
        # Clustering
        from evclust.datasets import load_protein
        from evclust.kevclus import kevclus

        D = load_protein()
        clus = kevclus(D=D, k=30, c=4, type='simple', d0=D.max().max())
        clus['mass"]

    References:
    -----------
        Denoeux, T., & Masson, M. H. (2004). EVCLUS: Evidential Clustering of Proximity Data. 
        IEEE Transactions on Systems, Man, and Cybernetics, B, 34(1), 95-109.

        Denoeux, T., Sriboonchitta, S., & Kanjanatarakul, O. (2016). Evidential clustering of large dissimilarity data. 
        Knowledge-Based Systems, 106, 179-195.

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`, 
        :func:`~createD`

    .. note::
        Keywords : Relational Data, Clustering, Unsupervised Learning, Dempster-Shafer theory, Evidence theory, Belief Functions, Multi-dimensional Scaling
        Given a matrix of dissimilarities between n objects, EVCLUS assigns a basic belief assignment (or mass function) to each object, in such a way
        that the degree of conflict between the masses given to any two objects, reflects their dissimilarity.
        The method was shown to be capable of discovering meaningful clusters in several non-Euclidean data sets, and its performances compared favorably with those of stateof-the-art techniques.
    """
    # Validate inputs and prepare data
    if x is not None:
        x = np.array(x)
        n = x.shape[0]
        if k is None or k == (n - 1):
            D_full = squareform(pdist(x))
            J = np.zeros((n, n - 1), dtype=int)
            D = np.zeros_like(J, dtype=float)
            for i in range(n):
                J[i, :] = np.delete(np.arange(n), i)
                D[i, :] = D_full[i, J[i, :]]
            p = n - 1
        else:
            dist_result = createD(x, k) 
            D = dist_result['D']
            J = dist_result['J']
            p = k
    elif D is not None:
        D = np.array(D)
        n, p = D.shape
        if n == p and (k is None or k == (n - 1)):
            J = np.zeros((n, n - 1), dtype=int)
            D_new = np.zeros_like(J, dtype=float)
            for i in range(n):
                J[i, :] = np.delete(np.arange(n), i)
                D_new[i, :] = D[i, J[i, :]]
            D = D_new
            p = n - 1
        elif n == p and k < n:
            D_new = np.zeros((n, k), dtype=float)
            J = np.zeros_like(D_new, dtype=int)
            for i in range(n):
                ii = np.random.choice(np.delete(np.arange(n), i), k, replace=False)
                J[i, :] = ii
                D_new[i, :] = D[i, ii]
            D = D_new
            p = k
        else:
            k = p
    else:
        raise ValueError("Either x or D must be supplied.")

    # Normalization
    if d0 is None:
        d0 = np.quantile(D, 0.9)
    g = -np.log(0.05) / d0**2
    D = 1 - np.exp(-g * D**2)
    C = 1 / np.sum(D**2) if norm == 1 else 1 / (n * p)

    # Compute focal sets
    F = makeF(c=c, type=type, pairs=pairs)  
    f = F.shape[0]
    xi = np.zeros((f, f))
    for i in range(f):
        for j in range(f):
            xi[i, j] = 1 - np.max(np.minimum(F[i, :], F[j, :]))

    # Initialize variables for the optimization
    Sbest = float('inf')
    Tracebest = None
    for N in range(ntrials):
        if m0 is None:
            mass = np.random.rand(n, f)
            mass /= mass.sum(axis=1, keepdims=True)
        else:
            mass = m0

        # Compute conflict matrix K
        K = np.zeros_like(D)
        for i in range(n):
            K[i, :] = np.dot(np.dot(mass[i, :], xi), mass[J[i, :], :].T)

        Spred = C * np.sum((K - D)**2)
        gain = 1
        k = 0

        if tr:
            Trace = {"temps": np.zeros((maxit + 1, 3)), "fct": np.zeros(maxit + 1)}
            Trace["fct"][0] = Spred
        else:
            Trace = None

        while gain > epsi and k <= maxit:
            k += 1
            S = 0
            order = np.random.permutation(n) if change_order else np.arange(n)
            for i in order:
                A = np.dot(mass[J[i, :], :], xi.T)
                B = D[i, :]
                # res = linprog(np.zeros(f), A_eq=np.ones((1, f)), b_eq=np.array([1]),
                #               method='highs', options={"disp": False})
                res = lsq_linear(A, B, method='trf', lsq_solver='exact')
                mass[i, :] = res.x
                S += np.linalg.norm(np.dot(A, res.x) - B)**2
            S = C * S

            if tr:
                Trace["fct"][k] = S

            gain = 0.5 * gain + 0.5 * abs(Spred - S) / (1e-9 + abs(Spred))
            Spred = S

            if disp:
                print(f" {k}, {S:.3f}")

        if S < Sbest:
            Sbest = S
            mass_best = mass.copy()
            Tracebest = Trace

    # Final conflict matrix K
    for i in range(n):
        K[i, :] = np.dot(np.dot(mass_best[i, :], xi), mass_best[J[i, :], :].T)

    clus = extractMass(mass_best, F, method="kevclus", crit=Sbest, Kmat=K, D=D,
                        trace=Tracebest, J=J)
    return clus





#---------------------- Utils for kevclus------------------------------------------------  
def createD(x, k=None):
    """
    Compute a Euclidean distance matrix.
    """
    x = np.array(x)  # Ensure x is a numpy array
    n, p = x.shape

    if k is None:
        # Compute the full n x n distance matrix
        D = squareform(pdist(x, metric='euclidean'))
        J = None
    else:
        # Compute n x k distance matrix with randomly selected neighbors
        D = np.zeros((n, k))
        J = np.zeros((n, k), dtype=int)

        for i in range(n):
            # Randomly sample k indices from all other rows (excluding the current row i)
            ii = np.random.choice([idx for idx in range(n) if idx != i], k, replace=False)
            J[i, :] = ii
            # Compute the Euclidean distances to these sampled rows
            D[i, :] = np.sqrt(np.sum((x[ii, :] - x[i, :])**2, axis=1))

    return {'D': D, 'J': J}