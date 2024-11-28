# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Multi-View Evidential C-Medoid clustering (MECMdd) with adaptive weightings with Relevance Weight for each dissimilarity matrix
estimated globally (MECMdd-RWG) using weight Sum (MECMdd-RWG-S) constraint or weight Product (MECMdd-RWG-P) constraint.

    Armel Soubeiga, Violaine Antoine and Sylvain Moreno. "Multi-View Relational Evidential C-Medoid Clustering with Adaptive Weighted" 
    2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA)
"""


#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np




#---------------------- MECMdd-RWG------------------------------------------------
def mecmdd_rwg(Xlist, c, type='full', alpha=1, beta=1.5, delta=9, epsi=1e-4, 
               disp=True, gamma=0.5, eta=1, weight='sum', s=None):
    """
    Multi-View Evidential C-Medoid clustering (MECMdd) with adaptive weightings with Relevance Weight for each dissimilarity matrix
    estimated globally (MECMdd-RWG) using weight Sum (MECMdd-RWG-S) constraint or weight Product (MECMdd-RWG-P) constraint.

    Parameters:
    -----------
    Xlist (list of np.array): 
        A list of square symmetric dissimilarity matrices.
    c (int): 
        Number of clusters to create.
    type (str): 
        Type of structure ('full' by default).
    alpha (float): 
        Weighting parameter for cardinalities.
    beta (float): 
        uncertainty exponent for membership degrees.
    delta (float or None): 
        Precision parameter (used to calculate delta2). If None, distribution of matrix are considere.
    epsi (float): 
        Convergence threshold.
    disp (bool): 
        Whether to display the objective function value during iteration.
    gamma (float): 
        Coefficient to adjust the imprecise contribution.
    eta (float): 
        Coefficient for outlier identification.
    weight (str): 
        Type of constraint function using to learn weight ('sum' or 'prod').
    s (float or None): 
        Additional weight parameter.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    Example:
    --------
    .. highlight:: python
    .. code-block:: python

        from evclust.mecmdd_rwg import mecmdd_rwg

        matrix1 = np.random.rand(5, 5)
        matrix1 = (matrix1 + matrix1.T) / 2  
        matrix2 = np.random.rand(5, 5)
        matrix2 = (matrix2 + matrix2.T) / 2 
        Xlist = [matrix1, matrix2]

        clus = mecmdd_rwg(Xlist, c=2, type='simple', alpha=1, beta=1.5, delta=9, epsi=1e-4, 
                        disp=True, gamma=0.5, eta=1, weight='sum', s=None)

        clus['param']['lambda'] # View weight

    References:
    -----------
        Armel Soubeiga, Violaine Antoine and Sylvain Moreno. "Multi-View Relational Evidential C-Medoid Clustering with Adaptive Weighted" 
        2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA).

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`, 
        :func:`~lambdaInit_global`

    .. note::
        Keywords : Evidential clustering, credal partition, relational multi-view, belief function
        MECMdd-RWG is able to address the uncertainty and imprecision of multi-view relational data clustering, and provides a credible partition, which extends fuzzy, possibilistic and rough partitions. 
        It can automatically lern the importance of each views by estimated a weight globally for each cluster in a collaborative learning framework.
    """
    if not Xlist or not isinstance(Xlist, list) or len(Xlist) <= 1:
        raise ValueError("The data set (Xlist) must be given, must not be empty, must be a list, and must contain more than one matrix")

    tailles = [x.shape for x in Xlist]
    if not all(t == tailles[0] for t in tailles):
        raise ValueError("All matrices in Xlist must have the same size")

    # Determine s based on weight type
    if weight == 'prod':
        s = 1
    elif weight == 'sum':
        if s is None:
            s = 2

    # Initialize delta
    pmatrix = len(Xlist)
    if delta is None:
        delta2 = sum(np.quantile(mat[np.triu_indices_from(mat, k=1)], 0.95) for mat in Xlist)
    elif isinstance(delta, (int, float)):
        delta2 = sum([delta ** 2] * pmatrix)
    elif len(delta) == pmatrix:
        delta2 = sum([d ** 2 for d in delta])
    else:
        raise ValueError("Delta must be None, a scalar, or a list of length equal to the number of matrices.")

    # Generate focal set matrix F and initialize parameters
    F = makeF(c=c, type=type)
    nF = F.shape[0]
    card = np.sum(F[1:], axis=1)

    n = Xlist[0].shape[0]
    medoids = np.random.choice(n, c, replace=False)
    prototype = np.zeros(nF-1, dtype=int)
    lambda_ = lambdaInit_global(weight, pmatrix)

    pasfini = True
    Jold = float('inf')
    iter = 0

    while pasfini:
        iter += 1

        # Update singleton and imprecise (medoid)
        for j in range(nF-1):
            fj = F[j+1]
            medoidsj = medoids[fj != 0]
            l_i = np.zeros(n)

            if np.sum(fj) == 1:
                prototype[j] = medoidsj[0]
            else:
                for i in range(n):
                    var_ij = (1/card[j]) * np.sum([(np.sum([Xlist[l][i, medoidsj[k]] for l in range(pmatrix)]) -
                                                    (1/card[j]) * np.sum([np.sum([Xlist[l][i, medoidsj[kk]] for l in range(pmatrix)]) for kk in range(len(medoidsj))]))**2 for k in range(len(medoidsj))])
                    l_i[i] = var_ij + eta * (1/card[j]) * np.sum([np.sum([Xlist[l][i, medoidsj[kk]] for l in range(pmatrix)]) for kk in range(len(medoidsj))])
                prototype[j] = np.argmin(l_i)

        # Calculation of distances to centers
        D = [np.zeros((n, nF-1)) for _ in range(pmatrix)]
        for l in range(pmatrix):
            for j in range(nF-1):
                fj = F[j+1]
                medoidsj = medoids[fj != 0]
                if np.sum(fj) == 1:
                    D[l][:, j] = Xlist[l][:, prototype[j]]
                elif np.sum(fj) > 1:
                    D[l][:, j] = (Xlist[l][:, prototype[j]] + (gamma / card[j]) * np.sum(Xlist[l][:, medoidsj], axis=1)) / (1 + gamma)

        # Calculation of masses
        m = np.zeros((n, nF-1))
        for j in range(nF-1):
            fj = F[j+1]
            medoidsj = medoids[fj != 0]

            for i in range(n):
                num = (card[j]**alpha * np.sum([D[l][i, j] * (lambda_[l])**s for l in range(pmatrix)]))**(-1/(beta-1))
                den = (np.sum([(card[k]**alpha * np.sum([D[l][i, k] * (lambda_[l])**s for l in range(pmatrix)]))**(-1/(beta-1)) for k in range(nF-1)]) + delta2**(-1/(beta-1)))
                m[i, j] = num / den
                if np.isnan(m[i, j]) or np.isinf(m[i, j]):
                    m[i, j] = 1

        # Weight
        if weight == 'sum':
            for l in range(pmatrix):
                num = np.sum([s * (card[j]**alpha) * m[i, j]**beta * D[l][i, j] for i in range(n) for j in range(nF-1)])**(-1/(s-1))
                den = np.sum([np.sum([s * (card[j]**alpha) * m[i, j]**beta * D[h][i, j] for i in range(n) for j in range(nF-1)])**(-1/(s-1)) for h in range(pmatrix)])
                lambda_[l] = num / den

        if weight == 'prod':
            for l in range(pmatrix):
                num = np.prod([np.sum([(card[j]**alpha) * m[i, j]**beta * D[h][i, j] for i in range(n) for j in range(nF-1)])**(1/pmatrix) for h in range(pmatrix)], axis=0)
                den = np.sum([(card[j]**alpha) * m[i, j]**beta * D[l][i, j] for i in range(n) for j in range(nF-1)])
                lambda_[l] = num / den

        # Medoids
        V = np.zeros((n, nF-1))
        for k in range(nF-1):
            fk = F[k+1]
            if np.sum(fk) == 1:
                mk_beta = m[:, k]**beta
                for i in range(n):
                    V[i, k] = np.sum([np.sum([Xlist[l][i, :] * (lambda_[l])**s for l in range(pmatrix)]) * mk_beta[i]])

        V_filtered = V[:, np.sum(V, axis=0) > 0]
        medoids = np.array([np.argmin(V_filtered[:, i]) for i in range(V_filtered.shape[1])])

        mvide = 1 - np.sum(m, axis=1)
        J = np.sum([np.sum([card[j]**alpha * m[i, j]**beta * np.sum([Xlist[l][i, prototype[j]] * (lambda_[l])**s for l in range(pmatrix)]) for j in range(nF-1)]) for i in range(n)]) + delta2 * np.sum(mvide**beta)

        if disp:
            print(iter, J)
        pasfini = abs(J - Jold) > epsi
        Jold = J

    g = medoids
    m = np.hstack((1 - np.sum(m, axis=1).reshape(-1, 1), m))
    clus = extractMass(m, F, g=g, method="mecmdd", crit=J, param={'alpha': alpha, 'beta': beta, 'delta': delta, 'lambda': lambda_})
    return clus


#---------------------- Utils of MECMdd-RWG------------------------------------------------
def lambdaInit_global(weight, p):
    """
    Initializes the weight vector lambda based on the weighting scheme.
    """
    if weight == 'sum':
        return np.full(p, 1 / p)
    elif weight == 'prod':
        return np.ones(p)
    else:
        raise ValueError("Unknown weight type. Choose 'sum' or 'prod'.")
