# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains the utils function 
"""

#---------------------- Packges------------------------------------------------
import numpy as np
#import matplotlib.pyplot as plt



#---------------------- makeF--------------------------------------------------
def makeF(c, type=['simple', 'full', 'pairs'], pairs=None, Omega=True):
    """
    Creation of a matrix of focal sets. `makeF` creates a matrix of focal sets.

    Parameters:
    ----------
    c (int): Number of clusters.
    type (str): Type of focal sets ("simple": {}, singletons, and Ω; "full": all 2^c subsets of Ω;
                "pairs": {}, singletons, Ω, and all or selected pairs).
    pairs (ndarray or None): Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
    Omega (bool): If True (default), Ω is a focal set (for types 'simple' and 'pairs').


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
        mass (ndarray): Matrix of mass functions. The first column corresponds to the degree of conflict.
        F (ndarray): Matrix of focal sets. The first row always corresponds to the empty set.
        g (ndarray, optional): The prototypes (if defined). Defaults to None.
        S (ndarray, optional): The matrices S_j defining the metrics for each cluster and each group of clusters (if defined). Defaults to None.
        method (str): The method used to construct the credal partition.
        crit (float, optional): The value of the optimized criterion (depends on the method used). Defaults to None.
        Kmat (ndarray, optional): The matrix of degrees of conflict. Same size as D (for method "kevclus"). Defaults to None.
        trace (ndarray, optional): The trace of criterion values (for methods "kevclus" and "EkNNclus"). Defaults to None.
        D (ndarray, optional): The normalized dissimilarity matrix (for method "kevclus"). Defaults to None.
        W (ndarray, optional): The weight matrix (for method "EkNNclus"). Defaults to None.
        J (ndarray, optional): The matrix of indices (for method "kevclus"). Defaults to None.
        param (list, optional): A method-dependent list of parameters. Defaults to None.

    Returns:
    -------
        dict: An object of class "credpart" with the following components:
            - method (str): The method used to construct the credal partition.
            - F (ndarray): Matrix of focal sets. The first row always corresponds to the empty set.
            - conf (ndarray): Masses assigned to the empty set.
            - mass (ndarray): Mass functions.
            - mass_n (ndarray): Normalized mass functions.
            - g (ndarray, optional): The prototypes (if defined).
            - S (ndarray, optional): The matrices S_j defining the metrics for each cluster and each group of clusters (if defined).
            - pl (ndarray): Unnormalized plausibilities of the singletons.
            - pl_n (ndarray): Normalized plausibilities of the singletons.
            - p (ndarray): Probabilities derived from pl by the plausibility transformation.
            - bel (ndarray): Unnormalized beliefs of the singletons.
            - bel_n (ndarray): Normalized beliefs of the singletons.
            - y_pl (ndarray): Maximum plausibility clusters.
            - y_bel (ndarray): Maximum belief clusters.
            - betp (ndarray): Unnormalized pignistic probabilities of the singletons.
            - betp_n (ndarray): Normalized pignistic probabilities of the singletons.
            - Y (ndarray): Sets of clusters with maximum mass.
            - outlier (ndarray): Array of 0's and 1's, indicating which objects are outliers.
            - lower_approx (list): Lower approximations of clusters, a list of length c.
            - upper_approx (list): Upper approximations of clusters, a list of length c.
            - Ynd (ndarray): Sets of clusters selected by the interval dominance rule.
            - lower_approx_nd (list): Lower approximations of clusters using the interval dominance rule, a list of length c.
            - upper_approx_nd (list): Upper approximations of clusters using the interval dominance rule, a list of length c.
            - N (float): Average nonspecificity.
            - crit (float, optional): The value of the optimized criterion (depends on the method used).
            - Kmat (ndarray, optional): The matrix of degrees of conflict. Same size as D (for method "kevclus").
            - D (ndarray, optional): The normalized dissimilarity matrix (for method "kevclus").
            - trace (ndarray, optional): The trace of criterion values (for methods "kevclus" and "EkNNclus").
            - W (ndarray, optional): The weight matrix (for method "EkNNclus").
            - J (ndarray, optional): The matrix of indices (for method "kevclus").
            - param (list, optional): A method-dependent list of parameters.

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

    P = F / card[:, np.newaxis]
    P[0, :] = 0
    betp = np.matmul(mass, P)       # unnormalized pignistic probability
    betp_n = C[:, np.newaxis] * betp        # normalized pignistic probability

    lower_approx = [np.where((Y[:, i] == 1) & (np.sum(Y, axis=1) == 1))[0] for i in range(c)]  # lower approximation
    upper_approx = [np.where(Y[:, i] == 1)[0] for i in range(c)]  # upper approximation
    lower_approx_nd = [np.where((Ynd[:, i] == 1) & (np.sum(Ynd, axis=1) == 1))[0] for i in range(c)]  # lower approximation
    upper_approx_nd = [np.where(Ynd[:, i] == 1)[0] for i in range(c)]  # upper approximation

    # Nonspecificity
    card = np.concatenate(([c], card[1:f]))
    Card = np.tile(card, (n, 1))
    N = np.sum(np.log(Card) * mass) / np.log(c) / n

    clus = {'conf': conf, 'F': F, 'mass': mass, 'mass_n': mass_n, 'pl': pl, 'pl_n': pl_n, 'bel': bel, 'bel_n': bel_n,
            'y_pl': y_pl, 'y_bel': y_bel, 'Y': Y, 'betp': betp, 'betp_n': betp_n, 'p': p,
            'upper_approx': upper_approx, 'lower_approx': lower_approx, 'Ynd': Ynd,
            'upper_approx_nd': upper_approx_nd, 'lower_approx_nd': lower_approx_nd,
            'N': N, 'outlier': np.where(np.sum(Y, axis=1) == 0)[0], 'g': g, 'S': S,
            'crit': crit, 'Kmat': Kmat, 'trace': trace, 'D': D, 'method': method, 'W': W, 'J': J, 'param': param}

    return clus







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
