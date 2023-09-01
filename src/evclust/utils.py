# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains the utils function 
"""

#---------------------- Packges------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import ConvexHull
from itertools import combinations
import seaborn as sns
from sklearn.decomposition import PCA








#---------------------- makeF--------------------------------------------------
def makeF(c, type=['simple', 'full', 'pairs'], pairs=None, Omega=True):
    """
    Creation of a matrix of focal sets. `makeF` creates a matrix of focal sets.

    Parameters:
    ----------
    c (int): 
        Number of clusters.
    type (str): 
        Type of focal sets ("simple": {}, singletons, and Ω; "full": all 2^c subsets of Ω;
                "pairs": {}, singletons, Ω, and all or selected pairs).
    pairs (ndarray or None): 
        Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
    Omega (bool): 
        If True (default), Ω is a focal set (for types 'simple' and 'pairs').


    Returns:
    --------
    ndarray: A matrix (f, c) of focal sets.


    Examples:
    ---------

    """
    if type == 'full':  # All the 2^c focal sets
        ii = np.arange(2**c)
        N = len(ii)
        F = np.zeros((N, c))
        CC = np.array([np.binary_repr(i, width=c) for i in range(N)])
        for i in range(N):
            F[i, :] = np.array([int(s) for s in CC[i]])
        F = F[:, ::-1]
    else:  # type = 'simple' or 'pairs'
        F = np.vstack((np.zeros(c), np.eye(c)))  # the empty set and the singletons
        if type == 'pairs':  # type = 'pairs'
            if pairs is None:  # pairs not specified: we take them all
                for i in range(c - 1):
                    for j in range(i + 1, c):
                        f = np.zeros(c)
                        f[[i, j]] = 1
                        F = np.vstack((F, f))
            else:  # pairs specified
                n = pairs.shape[0]
                for i in range(n):
                    f = np.zeros(c)
                    f[pairs[i, :]] = 1
                    F = np.vstack((F, f))
        if Omega and not ((type == "pairs") and (c == 2)) and not ((type == "simple") and (c == 1)):
            F = np.vstack((F, np.ones(c)))  # the whole frame
    return F



#---------------------- get_ensembles------------------------------------------
def get_ensembles(table):
    result = []
    for row in table:
        row_str = 'Cl_' + '_'.join([str(i + 1) if elem == 1 else str(int(elem)) for i, elem in enumerate(row) if elem != 0])
        result.append(row_str)

    result[0] = 'Cl_atypique'
    result[-1] = 'Cl_incertains'

    cleaned_result = [''.join(ch for i, ch in enumerate(row_str) if ch != '_' or (i > 0 and row_str[i-1] != '_')) for row_str in result]

    return cleaned_result





#---------------------- extractMass--------------------------------------------
def extractMass(mass, F, g=None, S=None, method=None, crit=None, Kmat=None, trace=None, D=None, W=None, J=None, param=None):
    """
    Creates an object of class "credpart". `extractMass` computes different outputs (hard, fuzzy, rough partitions, etc.)
    from a credal partition and creates an object of class "credpart".

    This function collects varied information on a credal partition and stores it in
    an object of class "credpart". The lower and upper
    approximations of clusters define rough partitions. They can be computed in two ways:
    either from the set of clusters with maximum mass, or from the set of non-dominated clusters.
    A cluster ω_k is non-dominated if pl(ω_k) ≥ bel(ω_l) for all l different from k.
    Once a set of cluster Y_i has been computed for each object,
    object i belongs to the lower approximation of cluster k if Y_i = ω_k.
    It belongs to the upper approximation of cluster k if ω_k ∈ Y_i.
    See Masson and Denoeux (2008) for more details, and Denoeux and Kanjanatarakul (2016) for
    the interval dominance rule. The function creates an object of class "credpart".
    There are three methods for this class: plot.credpart, summary.credpart, and predict.credpart.


    Parameters:
    ----------
    mass (ndarray): 
        Matrix of mass functions. The first column corresponds to the degree of conflict.
    F (ndarray): 
        Matrix of focal sets. The first row always corresponds to the empty set.
    g (ndarray, optional): 
        The prototypes (if defined). Defaults to None.
    S (ndarray, optional): 
        The matrices S_j defining the metrics for each cluster and each group of clusters (if defined). Defaults to None.
    method (str): 
        The method used to construct the credal partition.
    crit (float, optional): 
        The value of the optimized criterion (depends on the method used). Defaults to None.
    Kmat (ndarray, optional): 
        The matrix of degrees of conflict. Same size as D (for method "kevclus"). Defaults to None.
    trace (ndarray, optional): 
        The trace of criterion values (for methods "kevclus" and "EkNNclus"). Defaults to None.
    D (ndarray, optional): 
        The normalized dissimilarity matrix (for method "kevclus"). Defaults to None.
    W (ndarray, optional): 
        The weight matrix (for method "EkNNclus"). Defaults to None.
    J (ndarray, optional): 
        The matrix of indices (for method "kevclus"). Defaults to None.
    param (list, optional): 
        A method-dependent list of parameters. Defaults to None.

    Returns:
    -------
    method (str): 
        The method used to construct the credal partition.
    F (ndarray): 
        Matrix of focal sets. The first row always corresponds to the empty set.

    mass (ndarray): 
        Mass functions.
    g (ndarray, optional): 
        The prototypes (if defined).
    S (ndarray, optional): 
        The matrices S_j defining the metrics for each cluster and each group of clusters (if defined).
    pl (ndarray): 
        Unnormalized plausibilities of the singletons.
    pl_n (ndarray): 
        Normalized plausibilities of the singletons.
    p (ndarray): 
        Probabilities derived from pl by the plausibility transformation.
    bel (ndarray): 
        Unnormalized beliefs of the singletons.
    bel_n (ndarray): 
        Normalized beliefs of the singletons.
    y_pl (ndarray): 
        Maximum plausibility clusters.
    y_bel (ndarray): 
        Maximum belief clusters.
    betp (ndarray): 
        Unnormalized pignistic probabilities of the singletons.
    betp_n (ndarray):
        Normalized pignistic probabilities of the singletons.
    Y (ndarray): 
        Sets of clusters with maximum mass.
    outlier (ndarray): 
        Array of 0's and 1's, indicating which objects are outliers.
    lower_approx (list): 
        Lower approximations of clusters, a list of length c.
    upper_approx (list): 
        Upper approximations of clusters, a list of length c.
    Ynd (ndarray): 
        Sets of clusters selected by the interval dominance rule.
    lower_approx_nd (list):
        Lower approximations of clusters using the interval dominance rule, a list of length c.
    upper_approx_nd (list): 
        Upper approximations of clusters using the interval dominance rule, a list of length c.
    N (float): 
        Average nonspecificity.
    crit (float, optional): 
        The value of the optimized criterion (depends on the method used).
    Kmat (ndarray, optional): 
        The matrix of degrees of conflict. Same size as D (for method "kevclus").
    D (ndarray, optional): 
        The normalized dissimilarity matrix (for method "kevclus").
    trace (ndarray, optional): 
        The trace of criterion values (for methods "kevclus" and "EkNNclus").
    W (ndarray, optional): 
        The weight matrix (for method "EkNNclus").
    J (ndarray, optional): 
        The matrix of indices (for method "kevclus").
    param (list, optional): 
        A method-dependent list of parameters.

    References:
    ---------
        - T. Denoeux and O. Kanjanatarakul. Beyond Fuzzy, Possibilistic and Rough: An Investigation of Belief Functions in Clustering. 8th International conference on soft methods in probability and statistics, Rome, 12-14 September, 2016.
        - M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm. Pattern Recognition, Vol. 41, Issue 4, pages 1384-1397, 2008.

    Examples:
    ---------

    """
    n = mass.shape[0]
    c = F.shape[1]
    if any(F[0, :] == 1):
        F = np.vstack((np.zeros(c), F))  # add the empty set
        mass = np.hstack((np.zeros((n, 1)), mass))
    
    f = F.shape[0]
    card = np.sum(F, axis=1)
    
    conf = mass[:, 0]             # degree of conflict
    C = 1 / (1 - conf)
    mass_n = C[:, np.newaxis] * mass[:, 1:f]   # normalized mass function
    pl = np.matmul(mass, F)          # unnormalized plausibility
    pl_n = C[:, np.newaxis] * pl             # normalized plausibility
    p = pl / np.sum(pl, axis=1, keepdims=True)      # plausibility-derived probability
    bel = mass[:, card == 1]    # unnormalized belief
    bel_n = C[:, np.newaxis] * bel            # normalized belief
    y_pl = np.argmax(pl, axis=1)       # maximum plausibility cluster
    y_bel = np.argmax(bel, axis=1)     # maximum belief cluster
    Y = F[np.argmax(mass, axis=1), :]    # maximum mass set of clusters

    # non dominated elements
    Ynd = np.zeros((n, c))
    for i in range(n):
        ii = np.where(pl[i, :] >= bel[i, y_bel[i]])[0]
        Ynd[i, ii] = 1

    #P = F / card[:, np.newaxis]
    nonzero_card = np.where(card != 0)  
    P = np.zeros_like(F)
    P[nonzero_card] = F[nonzero_card] / card[nonzero_card, np.newaxis]
    P[0, :] = 0
    betp = np.matmul(mass, P)       # unnormalized pignistic probability
    betp_n = C[:, np.newaxis] * betp        # normalized pignistic probability

    lower_approx, upper_approx = [], []
    lower_approx_nd, upper_approx_nd = [], []
    nclus = np.sum(Y, axis=1)
    outlier = np.where(nclus == 0)[0]  # outliers
    nclus_nd = np.sum(Ynd, axis=1)
    for i in range(c):
        upper_approx.append(np.where(Y[:, i] == 1)[0])  # upper approximation
        lower_approx.append(np.where((Y[:, i] == 1) & (nclus == 1))[0])  # upper approximation
        upper_approx_nd.append(np.where(Ynd[:, i] == 1)[0])  # upper approximation
        lower_approx_nd.append(np.where((Ynd[:, i] == 1) & (nclus_nd == 1))[0])  # upper approximation
    
    # Nonspecificity
    card = np.concatenate(([c], card[1:f]))
    Card = np.tile(card, (n, 1))
    N = np.sum(np.log(Card) * mass) / np.log(c) / n

    clus = {'conf': conf, 'F': F, 'mass': mass, 'mass_n': mass_n, 'pl': pl, 'pl_n': pl_n, 'bel': bel, 'bel_n': bel_n,
            'y_pl': y_pl, 'y_bel': y_bel, 'Y': Y, 'betp': betp, 'betp_n': betp_n, 'p': p,
            'upper_approx': upper_approx, 'lower_approx': lower_approx, 'Ynd': Ynd,
            'upper_approx_nd': upper_approx_nd, 'lower_approx_nd': lower_approx_nd,
            'N': N, 'outlier': outlier , 'g': g, 'S': S,
            'crit': crit, 'Kmat': Kmat, 'trace': trace, 'D': D, 'method': method, 'W': W, 'J': J, 'param': param}

    return clus









#---------------------- setCentersECM--------------------------------------------
def setCentersECM(x, m, F, Smean, alpha, beta):
    
    """
    Computation of centers in CECM. Function called by cecm.

    Parameters:
    ----------
    - x: 
        The data matrix.
    - m: 
        The mass matrix.
    - F: 
        The focal sets matrix.
    - Smean: 
        A list of matrices representing the centers of the focal sets.
    - alpha: 
        The alpha parameter.
    - beta: 
        The beta parameter.

    Returns:
    -------
    - g: 
        The computed centers matrix.
    """

    nbFoc = F.shape[0]
    K = F.shape[1]
    n = x.shape[0]
    nbAtt = x.shape[1]

    card = np.sum(F[1:nbFoc, :], axis=1)
    indSingleton = np.where(card == 1)[0] + 1

    R = None
    B = None
    for l in range(K):
        indl = indSingleton[l]
        Rl = None
        for i in range(n):
            Ril = np.zeros((nbAtt, nbAtt))
            Fl = np.tile(F[indl, :], (nbFoc, K))
            indAj = np.where(np.sum(np.minimum(Fl, F), axis=1) == 1)[0] - 1
            for j in range(len(indAj)):
                Ril += card[indAj[j]] ** (alpha - 1) * m[i, indAj[j]] ** beta * Smean[indAj[j]]
            Rl = np.vstack((Rl, Ril))
        R = np.column_stack((R, Rl))

        Bl = None
        for k in range(K):
            Bkl = np.zeros((nbAtt, nbAtt))
            indk = indSingleton[k]
            for i in range(n):
                Fl = np.tile(np.sign(F[indl, :] + F[indk, :]), (nbFoc, K))
                indAj = np.where(np.sum(np.minimum(Fl, F), axis=1) == np.sum(Fl[0, :]))[0] - 1
                for j in range(len(indAj)):
                    Bkl += card[indAj[j]] ** (alpha - 2) * m[i, indAj[j]] ** beta * Smean[indAj[j]]
            Bl = np.vstack((Bl, Bkl))
        B = np.column_stack((B, Bl))

    X = x.flatten()
    g = np.linalg.solve(B.T, R.T @ X)
    g = g.reshape((K, nbAtt))
    return g









#---------------------- setCentersECM------------------------------------------
def createMLCL(y, nbConst):
    """
    Random generation of Must-Link (ML) and Cannot-Link (CL) constraints.

    Parameters:
    ----------
    - y: 
        Vector of class labels.
    - nbConst: 
        Number of constraints.

    Returns:
    -------
    A dictionary with two keys:
    - ML: 
        Matrix of ML constraints. Each row corresponds to a constraint.
    - CL: 
        Matrix of CL constraints. Each row corresponds to a constraint.
    """

    n = len(y)
    pairs = list(combinations(range(n), 2))
    N = len(pairs)
    selected_pairs = np.random.choice(N, nbConst, replace=False)
    const = np.array(pairs)[selected_pairs].T
    ML = const[:, y[const[0]] == y[const[1]]]
    CL = const[:, y[const[0]] != y[const[1]]]
    
    return {'ML': ML, 'CL': CL}







#---------------------- setDistances------------------------------------------
def setDistances(x, F, g, m, alpha, distance):
    """
    Computation of distances to centers and variance matrices in each cluster.
    Function called by cecm.

    Parameters:
    ----------
    - x: 
        Data matrix.
    - F: 
        Focal matrix.
    - g: 
        Centers matrix.
    - m: 
        Membership matrix.
    - alpha: 
        Alpha parameter.
    - distance: 
        Distance type (0 for Euclidean, 1 for Mahalanobis).

    Returns:
    --------
    A dictionary with two keys:
    - D: 
        Matrix of distances to centers. Each column corresponds to a center.
    - Smean: 
        List of variance matrices in each cluster.
    """

    nbFoc = F.shape[0]
    K = F.shape[1]
    n = x.shape[0]
    nbAtt = x.shape[1]
    beta = 2

    gplus = np.zeros((nbFoc-1, nbAtt))
    for i in range(1, nbFoc):
        fi = F[i, :]
        truc = np.tile(fi, (nbAtt, 1)).T
        gplus[i-1, :] = np.sum(g * truc, axis=0) / np.sum(fi)

    if distance == 0:
        S = [np.eye(nbAtt)] * K  # Euclidean distance
    else:
        ind = np.where(np.sum(F, axis=1) == 1)[0]
        S = []
        for i in ind:
            Sigmai = np.zeros((nbAtt, nbAtt))
            for k in range(n):
                omegai = np.tile(F[i, :], (K, 1))
                indAj = np.where(np.sum(np.minimum(omegai, F), axis=1) > 0)[0]
                for j in indAj:
                    aux = x[k, :] - gplus[j-1, :]
                    Sigmai += np.sum(F[j, :]) ** (alpha - 1) * m[k, j-1] ** beta * np.outer(aux, aux)
            Si = np.linalg.det(Sigmai) ** (1/nbAtt) * np.linalg.inv(Sigmai)
            S.append(Si)

    Smean = []
    for i in range(nbFoc-1):
        aux = np.zeros((nbAtt, nbAtt))
        for j in range(K):
            aux += F[i+1, j] * S[j]
        Smean.append(aux / max(np.sum(F[i+1, :]), 1))

    D = np.zeros((n, nbFoc-1))
    for j in range(nbFoc-1):
        aux = x - np.tile(gplus[j, :], (n, 1))
        if distance == 0:
            D[:, j] = np.diag(np.dot(aux, aux.T))
        else:
            D[:, j] = np.diag(np.dot(np.dot(aux, Smean[j]), aux.T))

    return {'D': D, 'Smean': Smean}











#--------------------------- solqp---------------------------------------------
def solqp(Q, A, b, c, x, verbose=False, toler=1e-5, beta=0.8):
    """
    Solve the quadratic program in standard form:
        minimize    0.5 * (x'Qx) + c'x
        subject to  Ax = b, x >= 0

    Parameters:
    ----------
    Q (ndarray):
        Sparse symmetric objective matrix.
    A (ndarray): 
        Sparse constraint left-hand matrix.
    b (ndarray): 
        Constraint right-hand column vector.
    c (ndarray): 
        Objective column vector.
    x (ndarray): 
        Initial solution vector.
    verbose (bool): 
        If True, print the message when the optimal solution is found.
    toler (float): 
        Relative stopping tolerance. The optimization stops when the objective value 
        is close to the local optimal value within the range of the tolerance.
    beta (float): 
        Step size for the algorithm. 0 < beta < 1.

    Returns:
    ----------
    dict: A dictionary containing the optimal solution and additional information.
    'x': 
        Optimal solution vector.
    'y': 
        Optimal dual solution (Lagrange multiplier).
    'obhis': 
        Objective value history vs iterations.
    """
    m = A.shape[0]
    n = A.shape[1]
    eps = np.finfo(float).eps

    #ob = 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c.T, x)
    ob = 0.5 * x.T @ Q @ x + c@ x

    alpha = 0.9
    comp = np.random.uniform(size=n)
    #comp = np.linalg.solve(np.block([[np.diag(comp), A], [A.T, np.zeros((m, m))]]),np.concatenate([comp, np.zeros(m)]))[:n]
    #comp = np.linalg.pinv(np.block([[np.diag(comp), A.T], [A, np.zeros((m, m))]])) @ np.concatenate([comp, np.zeros(m)])
    
    comp = np.linalg.solve(np.vstack((np.hstack((np.diag(comp), A.T)), np.hstack((A, np.zeros((m, m)))))),
                          np.vstack((comp.reshape(-1, 1), np.zeros((m, 1)))))

    comp = comp[:n]
    comp = comp / x
    nora = np.min(comp)
    if nora < 0:
        nora = -0.01 / nora
    else:
        nora = np.max(comp)
        if nora == 0:
            print('The problem has a unique feasible point')
            #return
        nora = 0.01 / nora
        
    x = x + nora * comp

    obvalue = np.dot(x.T, np.dot(Q, x)) / 2 + np.dot(c, x)
    obvalue = obvalue[0, 0]
    
    obvalue = np.sum(x.T @ (Q @ x)/2 + c @ x)
    
    obhis = [obvalue]
    lower = -np.inf
    zhis = [lower]
    gap = 1
    lamda = max(1, np.abs(obvalue) / np.sqrt(np.sqrt(n)))
    iter = 0

    while gap >= toler:
        iter += 1

        # spphase2
        lamda = (1 - beta) * lamda
        go = 0
        #gg = np.dot(Q, x) + c
        gg = np.dot(Q, x.reshape(-1, 1)) + c.reshape(-1, 1)
        XX = np.diag(x)
        AA = np.dot(A, XX)
        XX = np.dot(XX, np.dot(Q, XX))

        # Repeatly solve an ellipsoid constrained QP problem by solving a linear system equation until find a positive solution.
        while go <= 0:
            #u = np.linalg.solve(np.block([[XX + lamda * np.diag(np.ones(n)), AA.T], [AA, np.zeros((m, m))]]), np.concatenate([- np.multiply(x, gg.T.flatten()).reshape(-1, 1), np.zeros((m, 1))], axis=0))
            #u = np.linalg.solve(np.block([[XX + lamda * np.diag(np.ones(n)), AA.T], [AA, np.zeros((m, m))]]), np.vstack([- np.multiply(x, gg.T).T, np.zeros((m, 1))]))
            
            a = np.hstack((XX + lamda * np.diag(np.ones(n)), AA.T))
            b = np.hstack((AA, np.zeros((m, m))))
            ree = np.vstack((a,b))
            res = np.vstack((- np.multiply(x, gg.T).T, np.zeros((m, 1))))
            u = np.linalg.solve(ree,res)

            xx = x + np.multiply(x, u[:n].flatten())
            xx = xx.T
            go = np.min(xx)
            if go > 0:
                ob = float(np.dot(np.dot(xx, Q), xx.T)) / 2 + np.dot(c, xx.T)
                go = min(go, obvalue - ob + eps)[0, 0]
            lamda = 2 * lamda
            if lamda >= (1 + np.abs(obvalue)) / toler:
                print('The problem seems unbounded.')
                y = -u[n:n+m]


        y = -u[n:n+m]
        u = u[:n]
        nora = min(u)
        if nora < 0:
            nora = -alpha / nora
        else:
            if nora == 0:
                nora = alpha
            else:
                nora = np.inf

        u = np.multiply(x, u.flatten())
        w1 = np.dot(np.dot(u, Q), u.T)[0, 0]
        w2 = np.dot(-u, gg)[0, 0]

        if w1 > 0:
            nora = min(w2 / w1, nora)
        if nora == np.inf:
            ob = -np.inf
        else:
            x = x + nora * u
            ob = np.dot(np.dot(x, Q), x.T) / 2 + np.dot(c, x.T)
            ob = ob[0, 0]

        # This is the Phase 2 procedure called by SPSOLQP.
        if ob == -np.inf:
            gap = 0
            print('The problem is unbounded.')
        else:
            obhis.append(ob)
            comp = np.dot(Q, x.T) + c - np.dot(A.T, y)
            if np.min(comp) >= 0:
                zhis.append(ob - np.dot(x, comp))
                lower = zhis[iter]
                gap = (ob - lower) / (1 + np.abs(ob))
                obvalue = ob
            else:
                zhis.append(zhis[-1])
                lower = zhis[iter]
                gap = (obvalue - ob) / (1 + np.abs(ob))
                obvalue = ob

        if iter > 200:
            print([gap, toler])

    if verbose:
        print('A (local) optimal solution is found.')

    return {'x': x, 'y': y, 'obhis': obhis}









#---------------------- summary------------------------------------------------
def ev_summary(clus):
    """
    Summary of a credal partition. `summary_credpart` is the summary method for "credpart" objects.
    
    This function extracts basic information from "credpart" objects, such as created by
    ecm, recm, cecm, EkNNclus, or kevclus.
    
    Parameters:
    -----------
    clus : object
        An object of class "credpart", encoding a credal partition.
    
    Returns:
    --------
    None
        Prints basic information on the credal partition.
    
    
    References:
    -----------
    T. Denoeux and O. Kanjanatarakul. Beyond Fuzzy, Possibilistic and Rough: An
    Investigation of Belief Functions in Clustering. 8th International conference on soft
    methods in probability and statistics, Rome, 12-14 September, 2016.

    M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm.
    Pattern Recognition, Vol. 41, Issue 4, pages 1384--1397, 2008.

    T. Denoeux, S. Sriboonchitta and O. Kanjanatarakul. Evidential clustering of large
    dissimilarity data. Knowledge-Based Systems, vol. 106, pages 179-195, 2016.
    
    Examples:
    ---------
    """
    c = clus['F'].shape[1]
    n = clus['mass'].shape[0]
    print("------ Credal partition ------")
    print(f"{c} classes,")
    print(f"{n} objects")
    print(f"Generated by {clus['method']}")
    print("Focal sets:")
    print(clus['F'])
    print(f"Value of the criterion = {clus['crit']:.2f}")
    print(f"Nonspecificity = {clus['N']:.2f}")
    if clus['g'] is not None:
        print("Prototypes:")
        print(clus['g'])
    print(f"Number of outliers = {len(clus['outlier']):.2f}")
    
    
    
    
    



#---------------------- plot------------------------------------------------
def ev_plot(x, X=None, ytrue=None, Outliers=True, Approx=1, cex=1,
                  cexvar='pl', cex_outliers=5, cex_protos=5, lwd=1,
                  ask=False, plot_Shepard=False, plot_approx=True,
                  plot_protos=True, xlab='$x_1$' , ylab='$x_2$'):
    """
    Plotting a credal partition. Generates plots of a credal partition.     
    This function plots different views of a credal partition in a two-dimensional attribute space.
    
    
    Parameters:
    ----------
    x : object
        An object of class "credpart", encoding a credal partition.
    X : array-like, optional
        A data matrix. If it has more than two columns (attributes), only the first two columns are used.
    ytrue : array-like, optional
        The vector of true class labels. If supplied, a different color is used for each true cluster.
        Otherwise, the maximum-plausibility clusters are used instead.
    Outliers : bool, optional
        If True, the outliers are plotted, and they are not included in the lower and upper approximations of the clusters.
    Approx : int, optional
        If Approx==1 (default), the lower and upper cluster approximations are computed using the interval dominance rule.
        Otherwise, the maximum mass rule is used.
    cex : float, optional
        Maximum size of data points.
    cexvar : str, optional
        Parameter determining if the size of the data points is proportional to the plausibilities ('pl', the default),
        the plausibilities of the normalized credal partition ('pl.n'), the degrees of belief ('bel'),
        the degrees of belief of the normalized credal partition ('bel.n'), or if it is constant ('cst', default).
    cex_outliers : float, optional
        Size of data points for outliers.
    cex_protos : float, optional
        Size of data points for prototypes (if applicable).
    lwd : int, optional
        Line width for drawing the lower and upper approximations.
    ask : bool, optional
        Logical; if True, the user is asked before each plot.
    plot_Shepard : bool, optional
        Logical; if True and if the credal partition was generated by kevclus, the Shepard diagram is plotted.
    plot_approx : bool, optional
        Logical; if True (default) the convex hulls of the lower and upper approximations are plotted.
    plot_protos : bool, optional
        Logical; if True (default) the prototypes are plotted (for methods generating prototypes, like ECM).
    xlab : str, optional
        Label of horizontal axis.
    ylab : str, optional
        Label of vertical axis.
    
    Returns:
    ----------
    None
    
    The maximum plausibility hard partition, as well as the lower and upper approximations of each cluster
    are drawn in the two-dimensional space specified by matrix X. If prototypes are defined (for methods "ecm"
    and "cecm"), they are also represented on the plot. For methods "kevclus", "kcevclus" or "nnevclus",
    a second plot with Shepard's diagram (degrees of conflict vs. transformed dissimilarities) is drawn.
    If input X is not supplied and the Shepard diagram exists, then only the Shepard diagram is drawn.
    """
  
    clus = x
    if X is not None:
        x = X
        y = ytrue
        plt.rcParams['interactive'] = ask
        
        if y is None:
            y = clus['y_pl']
        c = len(np.unique(clus['y_pl']))
        
        if Approx == 1:
            lower_approx = clus['lower_approx_nd']
            upper_approx = clus['upper_approx_nd']
        else:
            lower_approx = clus['lower_approx']
            upper_approx = clus['upper_approx']
        
        if Outliers:
            for i in range(c):
                lower_approx[i] = np.setdiff1d(lower_approx[i], clus['outlier'])
                upper_approx[i] = np.setdiff1d(upper_approx[i], clus['outlier'])
        
        if cexvar == 'pl':
            cex = cex * np.apply_along_axis(np.max, 1, clus['pl'])
        elif cexvar == 'pl_n':
            cex = cex * np.apply_along_axis(np.max, 1, clus['pl_n'])
        elif cexvar == 'bel':
            cex = cex * np.apply_along_axis(np.max, 1, clus['bel'])
        elif cexvar == 'bel_n':
            cex = cex * np.apply_along_axis(np.max, 1, clus['bel_n'])
        
        colors = [mcolors.to_rgba('C{}'.format(i)) for i in y]
        color = [mcolors.to_rgba('C{}'.format(i)) for i in np.unique(y)]
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=colors,  s=cex)
        if Outliers:
            plt.scatter(x.iloc[clus['outlier'], 0], x.iloc[clus['outlier'], 1], c='black', marker='x', s=cex_outliers)
        if 'g' in clus and plot_protos and clus['g'] is not None:
            plt.scatter(clus['g'][:, 0], clus['g'][:, 1], c=color, marker='s', s=cex_protos)
        
        if plot_approx:
            for i in range(1, c + 1):
                xx = x.iloc[lower_approx[i - 1]]
                if xx.shape[0] >= 3:
                    hull = ConvexHull(xx.iloc[:, :2])
                    for simplex in hull.simplices:
                        plt.plot(xx.iloc[simplex, 0], xx.iloc[simplex, 1], linewidth=lwd, color='C{}'.format(i-1))
                xx = x.iloc[upper_approx[i - 1]]
                if xx.shape[0] >= 3:
                    hull = ConvexHull(xx.iloc[:, :2])
                    for simplex in hull.simplices:
                        plt.plot(xx.iloc[simplex, 0], xx.iloc[simplex, 1],  linestyle='dashed', linewidth=lwd, color='C{}'.format(i-1))
        
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.tight_layout()
        plt.show()
        
        
        







#---------------------- plot------------------------------------------------
def ev_pcaplot(data, x, normalize=False, splite=False, cex=8, cex_protos=5):
    """
    Plot PCA results with cluster colors. 
    
    This function performs PCA on the input data and plots the resulting PCA scores,
    using the specified cluster information in 'x'.

    Parameters:
    ----------
    data : DataFrame
        The input data containing the attributes (columns) and samples (rows).
    x : object
        An object of class "credpart", encoding a credal partition.
    normalize : bool, optional
        If True, the data will be normalized before performing PCA. Default is False.
    splite : bool, optional
        If True, provides access to several different axes-level functions that show the views of clusters. 

    Returns:
    --------
    None

    The function plots the PCA scores in a scatter plot with cluster colors.
    """
    if normalize:
        data = (data - data.mean()) / data.std()  # Normalize the data

    mas = pd.DataFrame(x["mass"])
    c = len(np.unique(x['y_pl']))
    cols = get_ensembles(x['F'])
    mas.columns = cols
    mas["Cluster"] = mas.apply(lambda row: row.idxmax(), axis=1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    variance_percent = np.round(pca.explained_variance_ratio_ * 100, 1)

    ind_coord = pd.DataFrame(pca_result, columns=["Dim.1", "Dim.2"])
    ind_coord["Cluster"] = pd.Categorical(mas["Cluster"])
    mean_coords = ind_coord.groupby('Cluster').mean()

    pcolor = sns.color_palette("Dark2", n_colors=len(ind_coord["Cluster"].unique()))
    plt.figure(figsize=(8, 6))

    if splite:
        sns.relplot(data=ind_coord, x="Dim.1", y="Dim.2", hue="Cluster", col="Cluster", 
                    style="Cluster", palette=pcolor, s=cex, col_wrap=int((c**2)/2)) 
    else:
        sns.scatterplot(data=ind_coord, x="Dim.1", y="Dim.2", hue="Cluster", palette=pcolor, 
                        style="Cluster", s=cex)
        sns.scatterplot(data=mean_coords, x="Dim.1", y="Dim.2", s=(cex+25), hue="Cluster", 
                        palette=pcolor, style="Cluster",legend=False)

        
    sns.despine()
    legend = plt.legend(title="Cluster", loc='lower right', markerscale=0.3)
    plt.setp(legend.get_title(), fontsize=7) 
    plt.setp(legend.get_texts(), fontsize=7)
    plt.tick_params(axis='both', labelsize=7)
    plt.xlabel("X Label", fontsize=7)
    plt.ylabel("Y Label", fontsize=7)
    plt.xlabel(f"Dim 1 ({variance_percent[0]}%)")
    plt.ylabel(f"Dim 2 ({variance_percent[1]}%)")
    plt.show()