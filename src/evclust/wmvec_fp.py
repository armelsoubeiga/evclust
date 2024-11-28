# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Adaptive Weighted Multi-View Evidential Clustering With Feature Preference (WMVEC-FP).

    Zhe Liu, Haojian Huang, Sukumar Letchmunan, Muhammet Deveci, Adaptive weighted multi-view evidential clustering with feature preference,
    Knowledge-Based Systems, Volume 294, 2024, 111770, ISSN 0950-7051
"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np



#---------------------- WMVEC------------------------------------------------
def wmvec_fp(X, c, alpha=2, delta=5, maxit=20, epsi=1e-3, beta=1.1, lmbda=403.4288, gamma = 1, 
                type="simple", disp=True):

    """
    Adaptive Weighted Multi-View Evidential Clustering With Feature Preference (WMVEC-FP) Algorithm.
    WMVEC-FP learn the importance of each view and he importance of each feature under different views.
    WMVEC-FP is based on objets row-data.
    
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

        from evclust.wmvec_fp import wmvec_fp
        from evclust.datasets import load_prop

        df = load_prop()
        clus = wmvec_fp(X=df, c=4, alpha=2, delta=5, maxit=20, epsi=1e-3, beta=1.1, lmbda=403.4288, gamma = 1, 
                        type="simple", disp=True)
        
        # View weight
        clus['param]['R']

        # Features relevences
        clus['vectorW]['R']

    References:
    -----------
        Zhe Liu, Haojian Huang, Sukumar Letchmunan, Muhammet Deveci, Adaptive weighted multi-view evidential clustering with feature preference,
        Knowledge-Based Systems, Volume 294, 2024, 111770, ISSN 0950-7051

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`, 
        :func:`~update_Aj_fp`, :func:`~get_distance_fp`, :func:`~update_M_fp`, 
        :func:`~update_R_fp`, :func:`~update_V_fp`,
        :func:`~update_W_fp`, :func:`~update_jaccard_fp`, :func:`~Centroids_Initialization_fp`

    .. note::
        Keywords : Evidential clustering, Multi-view learning, Theory of belief functions, Credal partition
        WMVEC-FP solve the problems faced by high-dimensional multi-view data clustering. 
        By integrating feature weight learning, view weight learning, and evidential clustering into a unified framework, WMVEC-FP identifies the contributions of different features in each view.
    """
    
    # Initialize variables
    cluster = c
    view = len(X)
    features = [x.shape[1] for x in X]
    delta = [delta] * view
    center = [None] * view
    Aj = [None] * view
    U = [None] * view
    F_update = [None] * view
    W = [None] * view
    
    for i in range(view):
        center[i] = Centroids_Initialization_fp(X[i], cluster)
        
    R = np.ones(view) / view
    
    for i in range(view):
        W[i] = np.eye(features[i]) / features[i]
    
    
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
            w = W

        Aj = update_Aj_fp(view, cluster, features, Aj, F, center, nbFoc)
        F_update = [F] * view
        dis = get_distance_fp(1, view, X, nbFoc, Aj, F_update, alpha, beta, delta, features, R, W)
        M = update_M_fp(view, X, dis, beta, nbFoc)
        dis = get_distance_fp(2, view, X, nbFoc, Aj, F_update, alpha, beta, delta, features, M, W)
        R = update_R_fp(view, dis, lmbda)
        center = update_V_fp(view, cluster, alpha, beta, X, M, R, F_update, features)
        dis = get_distance_fp(3, view, X, cluster, Aj, F_update, alpha, beta, delta, features, M, R)
        W = update_W_fp(view, dis, features, gamma)
        dis = get_distance_fp(4, view, X, nbFoc, Aj, F_update, alpha, beta, delta, features, W, M, R, gamma)
        J = update_jaccard_fp(view, lmbda, gamma, R, W, dis)
        
        if disp:
            print(iter, J)
        pasfini = np.abs(J - Jold) > epsi
        Jold = J

    vectorW = [arr.ravel()[arr.ravel() != 0] for arr in w]
    clus = extractMass(m, F, g=center, method="wmvec-fp", crit=J, 
                       param={'alpha': alpha, 'beta': beta, 'delta': delta, 'R': r, 'W': w,
                              'vectorW': vectorW})
    return clus



#---------------------- Utils of WMVEC------------------------------------------------
def Centroids_Initialization_fp(X, K):
    centroids = np.empty((0, X.shape[1]))
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

def get_distance_fp(mode, view, data, nbFoc, Aj, F_update, alpha, beta, delta, features, *args):

    Dis = [None] * view
    if mode == 1:
        R, W = args
    elif mode == 2:
        M, W = args
    elif mode == 3:
        M, R = args
    elif mode == 4:
        W, M, R, gamma = args

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
                temp = temp @ W[i]
                temp = np.sum(temp.T, axis=0)
                temp[temp == 0] = 1e-10
                dis_temp[:, j] = temp * R[i] * card ** alpha
            elif mode == 2:
                temp = temp @ W[i]
                temp = np.sum(temp.T, axis=0)
                temp[temp == 0] = 1e-10
                dis_temp[:, j] = temp * (M[:, j] ** beta) * card ** alpha
            elif mode == 3:
                for p in range(features[i]):
                    if j < dis_cell[p].shape[1]:  # Check if index is within bounds
                        dis_cell[p][:, j] = temp[:, p] * R[i] * (M[:, j] ** beta) * card ** alpha
            elif mode == 4:
                temp = temp @ W[i]
                temp = np.sum(temp.T, axis=0)
                temp[temp == 0] = 1e-10
                dis_temp[:, j] = temp * (M[:, j] ** beta) * card ** alpha * R[i]
        if mode in [1, 2, 4]:
            dis_temp[:, 0] = delta[i] ** 2 * (R[i] if mode == 1 else M[:, 0] ** beta)
            Dis[i] = dis_temp
        elif mode == 3:
            Dis[i] = dis_cell
    return Dis



def update_Aj_fp(view, cluster, features, Aj, F, center, nbFoc):
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

def update_M_fp(view, data, dis, beta, nbFoc):
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

def update_R_fp(view, dis, lmbda):
    R = np.ones(view) / view
    temp = []
    for i in range(view):
        F_i = np.sum(dis[i])
        temp.append(np.exp(-F_i / lmbda))
    for i in range(view):
        R[i] = temp[i] / np.sum(temp)
    return R

def update_V_fp(view, cluster, alpha, beta, data, M, R, F_update, features):
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
                    loc = [n for n in range(len(F_update[i])) if F_update[i][n, c] == 1 and F_update[i][n, k] == 1]
                    H_ck = 0
                    for n in loc:
                        card = np.sum(F_update[i][n, :])
                        r = R[i]
                        aj_all_dis = card ** (alpha - 2) * r * M[:, n] ** beta
                        H_ck += np.sum(aj_all_dis)
                    H[c, k] = H_ck
            v = np.linalg.pinv(H) @ B
            v_i[:, p] = v.flatten()
        V.append(v_i)
    return V



def update_W_fp(view, dis, features, gamma):
    W = []
    for i in range(view):
        W_i = np.eye(features[i]) / features[i]
        temp = []
        for j in range(features[i]):
            D = np.sum(dis[i][j])
            temp.append(np.exp(-D / gamma))
        for j in range(features[i]):
            W_i[j, j] = temp[j] / np.sum(temp)
        W.append(W_i)
    return W

def update_jaccard_fp(view, lmbda, gamma, R, W, dis):
    part_1 = sum(R[i] * np.sum(dis[i]) for i in range(view))
    part_2 = sum(lmbda * (R[i] + 1e-4) * np.log(R[i] + 1e-4) for i in range(view))
    part_3 = sum(gamma * (W[i][j, j] + 1e-4) * np.log(W[i][j, j] + 1e-4) for i in range(view) for j in range(W[i].shape[0]))
    jaccard = part_1 + part_2 + part_3
    return jaccard


