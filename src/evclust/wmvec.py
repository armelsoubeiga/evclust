# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Adaptive Weighted Multi-View Evidential Clustering (WMVEC).

    Zhe Liu, Haojian Huang, Sukumar Letchmunan, Muhammet Deveci, Adaptive weighted multi-view evidential clustering with feature preference,
    Knowledge-Based Systems, Volume 294, 2024, 111770, ISSN 0950-7051
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np



#---------------------- WMVEC------------------------------------------------
def wmvec(X, c, alpha=2, delta=5, maxit=20, epsi=1e-3, beta=1.1, lmbda=403.4288, 
                type="simple", disp=True):
    
    """
    Weighted Multi-View Evidential Clustering (WMVEC) Algorithm. WMVEC can be viewed as a multi-view version of conventional evidential c-means clustering.
    Specifically, the view weight can measure the contribution of each view in clustering. WMVEC is based on objets row-data.

    Parameters:
    -----------
    X (list of np.ndarray): 
        List of datasets from different views.
    c (int): 
        Number of clusters.
    alpha (float): 
        Parameter for distance weighting.
    beta (float): 
        Exponent for mass function calculation.
    lmbda (float): 
        Parameter for R update.
    delta (list): 
        List of penalties for empty sets for each view.
    epsi (float): 
        Convergence threshold.
    maxit (int): 
        Maximum number of iterations.
    type (str): 
        Type of focal set matrix to generate ('simple', 'full', 'pairs').
    disp (bool): 
        If True (default), intermediate results are displayed.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    Example:
    --------
    .. highlight:: python
    .. code-block:: python

        from evclust.wmvec import wmvec
        from evclust.datasets import load_prop

        df = load_prop()
        clus = wmvec(X=df, c=4, alpha=2, delta=5, maxit=20, epsi=1e-3, 
                    beta=1.1, lmbda=403.4288, type="simple", disp=True)

        # View weight
        clus['param]['R']

    References:
    -----------
        Zhe Liu, Haojian Huang, Sukumar Letchmunan, Muhammet Deveci, Adaptive weighted multi-view evidential clustering with feature preference,
        Knowledge-Based Systems, Volume 294, 2024, 111770, ISSN 0950-7051

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`, 
        :func:`~Centroids_Initialization`, :func:`~get_distance_wmvec`, :func:`~update_Aj_wmvec`,
        :func:`~update_M_wmvec`, :func:`~update_R_wmvec`, :func:`~update_V_wmvec`, :func:`~update_jaccard_wmvec`
    
    .. note::
        Keywords : Evidential clustering, Multi-view learning, Theory of belief functions, Credal partition
        WMVEC can be viewed as a multi-view version of conventional evidential c-means clustering. 
        The objec-tive function of WMVEC integrating the learning of view weightsand credal partition into a unified framework, and design an optimiza-tion scheme to obtain the optimal results of WMVEC. 
        Specifically, the view weight can measure the contribution of each view in clustering. Thecredal partition can provide a deeper understanding of the data structureby allowing samples to belong not only to singleton clusters, but also toa union of different singleton clusters, called meta-cluster.  
    """
    
    # Initialize variables
    cluster = c
    view = len(X)
    features = [x.shape[1] for x in X]
    delta = [delta] * view
    center = Aj = F_update = []
    for i in range(view):
        center.append(Centroids_Initialization(X[i], cluster))

    R = np.ones(view) / view
    
    
    # Generate focal set matrix
    F = makeF(c, type, pairs=None, Omega=True)
    nbFoc = F.shape[0]

    iter = 0
    pasfini = True
    Jold = np.inf

    while pasfini and iter < maxit:
        iter += 1
        if iter > 1:
            m = M
            r = R

        Aj = update_Aj_wmvec(view, cluster, features, Aj, F, center, nbFoc)
        F_update = [F] * view

        dis = get_distance_wmvec(1, view, X, nbFoc, Aj, F_update, alpha, beta, delta, features, R)
        M = update_M_wmvec(view, X, dis, beta, nbFoc)

        dis = get_distance_wmvec(2, view, X, nbFoc, Aj, F_update, alpha, beta, delta, features, M)
        R = update_R_wmvec(view, dis, lmbda)

        center = update_V_wmvec(view, cluster, alpha, beta, X, M, R, F_update, features)
        dis = get_distance_wmvec(2, view, X, nbFoc, Aj, F_update, alpha, beta, delta, features, M)

        J = update_jaccard_wmvec(view, lmbda, R, dis)
        if disp:
            print(iter, J)
        pasfini = np.abs(J - Jold) > epsi
        Jold = J

    clus = extractMass(m, F, g=center, method="wmvec", crit=J, 
                       param={'alpha': alpha, 'beta': beta, 'delta': delta, 'R' : r})
    return clus






#---------------------- Utils of WMVEC------------------------------------------------
def Centroids_Initialization(X, K):
    centroids = np.empty((0, X.shape[1]))  # Initialize centroids
    for i in range(K):
        if i == 1:
            d = []
            for j in range(X.shape[0]):
                center = X[j, :]
                dis = 0
                for z in range(X.shape[0]):
                    if z != j:
                        dis += np.linalg.norm(center - X[z, :])
                d.append(dis)
            idx = np.argmin(d)
            centroids = np.vstack([centroids, X[idx, :]])
            X = np.delete(X, idx, axis=0)
        else:
            if centroids.shape[0] == 1:
                for j in range(centroids.shape[0]):
                    center = centroids[j, :]
                    dis = np.sum((X - center) ** 2, axis=1)
                    idx = np.argmax(dis)
                    centroids = np.vstack([centroids, X[idx, :]])
                    X = np.delete(X, idx, axis=0)
            else:
                Dis = []
                Idx = []
                for j in range(centroids.shape[0]):
                    center = centroids[j, :]
                    dis = np.sum((X - center) ** 2, axis=1)
                    Dis.append(np.sum(dis))
                    idx = np.argmax(dis)
                    Idx.append(idx)
                if len(Dis) > 0:
                    idx = np.argmin(Dis)
                    centroids = np.vstack([centroids, X[Idx[idx], :]])
                    X = np.delete(X, Idx[idx], axis=0)
    return centroids


def get_distance_wmvec(mode, view, data, nbFoc, Aj, F_update, alpha, beta, delta, features, R_or_M):
    Dis = [None] * view
    if mode == 1:
        R = R_or_M
    if mode == 2:
        M = R_or_M
    for i in range(view):
        Row = data[i].shape[0]
        ROW = F_update[i].shape[0]
        dis_temp = np.zeros((Row, ROW))
        dis_cell = [np.zeros((Row, nbFoc)) for _ in range(features[i])]
        row = Aj[i].shape[0]
        for j in range(1, row):
            temp = (data[i] - Aj[i][j, :]) ** 2
            card = np.sum(F_update[i][j, :]) if np.sum(F_update[i][j, :]) > 0 else 0
            if mode == 1:
                temp = np.sum(temp, axis=1)
                temp[temp == 0] = 1e-10
                dis_temp[:, j] = temp * R[i] * card ** alpha
            if mode == 2:
                temp = np.sum(temp, axis=1)
                temp[temp == 0] = 1e-10
                dis_temp[:, j] = temp * (M[:, j] ** beta) * card ** alpha
        if mode == 1:
            dis_temp[:, 0] = delta[i] ** 2 * R[i]
            Dis[i] = dis_temp
        if mode == 2:
            dis_temp[:, 0] = delta[i] ** 2 * (M[:, 0] ** beta)
            Dis[i] = dis_temp
    return Dis


def update_Aj_wmvec(view, cluster, features, Aj, F, center, nbFoc):
    for i in range(view):
        new_center = np.zeros((nbFoc, features[i]))
        for j in range(F.shape[0]):
            if np.sum(F[j, :]) != 0:
                temp1 = 0
                for k in range(min(cluster, center[i].shape[0])):
                    temp1 += F[j, k] * center[i][k, :]
                new_center[j, :] = temp1 / np.sum(F[j, :])
        Aj[i] = new_center
    return Aj


def update_M_wmvec(view, data, dis, beta, nbFoc):
    row = data[0].shape[0]
    M = np.zeros((row, nbFoc))
    index = 1 / (beta - 1)
    D = np.zeros((row, nbFoc))
    for i in range(view):
        D += dis[i]
    m = np.zeros((row, nbFoc - 1))
    for i in range(row):
        vect0 = D[i, 1:]
        for j in range(1, nbFoc):
            if np.sum(D[:, j]) != 0:
                vect1 = ((D[i, j] * np.ones(nbFoc - 1)) / vect0) ** index
                vect1[np.isinf(vect1)] = 0
                vect3 = vect1
                m[i, j - 1] = 1 / (np.sum(vect3) + (D[i, j] / D[i, 0]) ** index)
    m = np.hstack([1 - np.sum(m, axis=1).reshape(-1, 1), m])
    M = m
    M[M[:, 0] < 0] = 0
    return M


def update_R_wmvec(view, dis, lmbda):
    R = np.ones(view) / view
    temp = []
    for i in range(view):
        F_i = np.sum(dis[i])
        temp.append(np.exp(-F_i / lmbda))
    for i in range(view):
        R[i] = temp[i] / np.sum(temp)
    return R


def update_V_wmvec(view, cluster, alpha, beta, data, M, R, F_update, features):
    V = []
    for i in range(view):
        v_i = np.zeros((cluster, features[i]))
        for p in range(features[i]):
            B = np.zeros((cluster, 1))
            for j in range(cluster):
                pos = [k for k in range(F_update[i].shape[0]) if F_update[i][k, j] == 1]
                B_X = 0
                for n in pos:
                    card = np.sum(F_update[i][n, :])
                    r = R[i]
                    aj_dis = card ** (alpha - 1) * r * data[i][:, p] * (M[:, n] ** beta)
                    B_X += np.sum(aj_dis)
                B[j] = B_X
            H = np.zeros((cluster, cluster))
            for c in range(cluster):
                for k in range(cluster):
                    loc = [n for n in range(F_update[i].shape[0]) if F_update[i][n, c] == 1 and F_update[i][n, k] == 1]
                    H_ck = 0
                    for n in loc:
                        card = np.sum(F_update[i][n, :])
                        r = R[i]
                        aj_all_dis = card ** (alpha - 2) * r * M[:, n] ** beta
                        H_ck += np.sum(aj_all_dis)
                    H[c, k] = H_ck
            v = np.linalg.solve(H, B)
            v_i[:, p] = v.flatten()
        V.append(v_i)
    return V

def update_jaccard_wmvec(view, lmbda, R, dis):
    part_1 = sum(R[i] * np.sum(dis[i]) for i in range(view))
    part_2 = sum(lmbda * (R[i] + 1e-4) * np.log(R[i] + 1e-3) for i in range(view))
    jaccard = part_1 + part_2
    return jaccard