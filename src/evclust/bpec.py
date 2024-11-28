# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for Belief Peak Evidential Clustering (BPEC).

    Z.-G. Su and T. Denoeux. BPEC: Belief-Peaks Evidential Clustering. IEEE Transactions on Fuzzy Systems, 27(1):111-123, 2019.

"""

#---------------------- Packges------------------------------------------------
from evclust.utils import makeF, extractMass
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt




#---------------------- bpec------------------------------------------------
def bpec(x, g, type='full', pairs=None, Omega=True, alpha=1, beta=2, delta=10, epsi=1e-3, disp=True, m0=None):

    """
    Belief Peak Evidential Clustering (BPEC) computes a credal partition from a matrix of attribute data.
    BPEC is identical to ECM, except that the prototypes are computed from delta-Bel graph using function delta_Bel. 
    The ECM algorithm is then run keeping the prototypes fixed. The distance to the prototypes can be the Euclidean disatnce.

    Parameters:
    ------------
    x (numpy.ndarray): 
        Input matrix of size n x d, where n is the number of objects and d the number of attributes.
    g (numpy.ndarray, or None): 
        Matrix of size c x d of prototypes (the belief peaks).
    type (str): 
        Type of focal sets ("simple", "full", or "pairs").
    Omega (bool): 
        Whether to include the whole frame in focal sets (default: True).
    pairs (list or None): 
        Pairs to include in focal sets, used only if type="pairs".
    alpha (float): 
        Exponent of the cardinality in the cost function.
    beta (float): 
        Exponent of masses in the cost function.
    delta (float): 
        Distance to the empty set.
    epsi (float): 
        Minimum amount of improvement.
    disp (bool): 
        Whether to display intermediate results (default: True).
    m0 (numpy.ndarray or None): 
        Initial credal partition. Should be a matrix with n rows and a number of columns equal to the number 
        of focal sets specified by 'type' and 'pairs'.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    Example:
    --------
    .. highlight:: python
    .. code-block:: python

        # Clustering
        from evclust.datasets import load_fourclass
        from evclust.bpec import bpec, delta_Bel

        df = load_fourclass()
        x = df.iloc[:, 0:2]
        DB = delta_Bel(x,100,0.9)
        clus = bpec(x, DB['g0'], type='pairs', delta=3, distance=1)
    
    References:
    ------------
        Z.-G. Su and T. Denoeux. BPEC: Belief-Peaks Evidential Clustering. IEEE Transactions on Fuzzy Systems, 27(1):111-123, 2019.


    .. seealso::
        :func:`~extractMass`, :func:`~makeF`, 
        :func:`~delta_Bel`,  :func:`~setDistances`
    
    .. note::
        Keywords : Dempster-Shafer theory, belief functions, unsupervised learning, soft clustering, density peaks clustering.
        BPEC can find the true number of clusters and create a credal partition for some datasets with good performances.
        When the number of clusters is small (usually less than ten), the performances of BPEC and its informative variant with a limited number of composite clusters are approximately equal. 
        In contrast, BPEC can be enhanced if less informative composite clusters (i.e., focal sets) when the number of clusters is large. 
        Furthermore, BPEC can provide hard, fuzzy, possibilistic and even rough partitions. 
    """
    
    # ---------------------- Delta-Bel --------------------------------------
    if g is None:
        DB = delta_Bel(x,100,0.9)
        g = DB['g0']
    else:
        g = g
    
    # ---------------------- Initializations --------------------------------------
    x = np.array(x)
    n, d = x.shape  
    delta2 = delta ** 2
    c = g.shape[0]  

    F = makeF(c, type, pairs, Omega)  
    f = F.shape[0]
    card = np.sum(F[1:], axis=1)  

    # ---------------------- Iterations -------------------------------------------
    pasfini = True  
    gplus = np.zeros((f - 1, d))
    iter_ = 0

    # Compute prototypes for focal sets
    for i in range(1, f):
        fi = F[i, :]
        truc = np.tile(fi, (c, 1))
        gplus[i - 1, :] = np.sum(g * truc, axis=0) / np.sum(fi)

    # Compute initial Euclidean distances to prototypes
    D = np.zeros((n, f - 1))
    for j in range(f - 1):
        D[:, j] = np.sum((x - np.tile(gplus[j, :], (n, 1))) ** 2, axis=1)

    # Compute initial masses
    if m0 is None:
        m = np.zeros((n, f - 1))
        for i in range(n):
            vect0 = D[i, :]
            for j in range(f - 1):
                vect1 = (np.tile(D[i, j], f - 1) / vect0) ** (1 / (beta - 1))
                vect2 = (np.tile(card[j] ** (alpha / (beta - 1)), f - 1) / 
                         (card ** (alpha / (beta - 1))))
                vect3 = vect1 * vect2
                m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / delta2) ** (1 / (beta - 1)))
                if np.isnan(m[i, j]):
                    m[i, j] = 1
    else:
        m = m0[:, 1:f]

    mvide = 1 - np.sum(m, axis=1)  
    Jold = np.sum((m ** beta) * D * np.tile(card ** alpha, (n, 1))) + delta2 * np.sum(mvide ** beta)

    if disp:
        print(iter_, Jold)

    # Main iteration loop
    while pasfini:
        iter_ += 1
        dist = setDistances(x, F, g, m, alpha, distance=0)
        D = dist['D']
        Smeans = dist['Smean']

        # Update masses
        for i in range(n):
            vect0 = D[i, :]
            for j in range(f - 1):
                vect1 = (np.tile(D[i, j], f - 1) / vect0) ** (1 / (beta - 1))
                vect2 = (np.tile(card[j] ** (alpha / (beta - 1)), f - 1) / 
                         (card ** (alpha / (beta - 1))))
                vect3 = vect1 * vect2
                m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / delta2) ** (1 / (beta - 1)))
                if np.isnan(m[i, j]):
                    m[i, j] = 1

        mvide = 1 - np.sum(m, axis=1)
        J = np.sum((m ** beta) * D * np.tile(card ** alpha, (n, 1))) + delta2 * np.sum(mvide ** beta)

        if disp:
            print(iter_, J)

        pasfini = np.abs(J - Jold) > epsi
        Jold = J

    # Finalize masses and return results
    m = np.hstack((mvide[:, np.newaxis], m))
    clus = extractMass(m, F, g=g, method="bpec", crit=J, param={'alpha': alpha,
                                        'beta': beta, 'delta': delta, 'S': Smeans})
    return clus



#---------------------- Utils for BPEC------------------------------------------------  
def delta_Bel(x, K, q=0.9):
    """
    Delta-Bel graph for Belief Peak Evidential Clustering (BPEC).
    This function computes the delta-Bel graph used to determine the prototypes in the Belief Peak Evidential Clustering (BPEC) algorithm. The user must manually specify 
    the rectangles containing the prototypes (typically in the upper-right corner of the graph if the clusters are well-separated). These prototypes are then used 
    by the bpec function to compute a credal partition.

    Parameters:
    -----------
    x (numpy.ndarray): 
        Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
    K (int): 
        Number of neighbors used to determine belief values.
    q (float): 
        Parameter of the algorithm, between 0 and 1 (default: 0.9).

    Returns:
    --------
    dict: A dictionary containing:
        BelC (numpy.ndarray): 
            The belief values.
        delta (numpy.ndarray): 
            The delta values.
        g0 (numpy.ndarray): 
            A c x d matrix containing the prototypes.
        ii (list):
            List of indices of the belief peaks.
    """

    n = x.shape[0]
    # Compute K-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=K).fit(x)
    distances, indices = nbrs.kneighbors(x)

    # Calculate beliefs that possibly support itself as a center
    alpha = 1 / K
    if q is None:
        g = np.ones(n)
    else:
        g = 1 / np.apply_along_axis(lambda row: np.quantile(row, q), axis=1, arr=distances)

    BelC = np.zeros(n)
    for i in range(n):
        BelC[i] = 1 - np.prod(1 - alpha * np.exp(-g[indices[i]] ** 2 * distances[i] ** 2))

    # Calculate delta values
    Dist = distance.cdist(x, x, metric='euclidean')
    maxdist = np.max(Dist)
    bel_sorted_indices = np.argsort(-BelC)  
    #bel_sorted  = BelC[bel_sorted_indices]

    delta = np.zeros(n)
    delta[bel_sorted_indices[0]] = -1
    for i in range(1, n):
        delta[bel_sorted_indices[i]] = maxdist
        for j in range(i):
            dist_ij = Dist[bel_sorted_indices[i], bel_sorted_indices[j]]
            if dist_ij < delta[bel_sorted_indices[i]]:
                delta[bel_sorted_indices[i]] = dist_ij
    delta[bel_sorted_indices[0]] = np.max(delta)

    # Plot the delta-Bel graph
    plt.scatter(delta, BelC)
    plt.ylim([np.min(BelC) - 0.1, np.max(BelC) + 0.1])
    plt.xlabel("Delta")
    plt.ylabel("Belief")
    plt.title("Delta-Bel Graph")
    plt.show()
    print("With the mouse, select the lower-left corner of a rectangle containing the prototypes.")
    
    # Wait for the user to click on the plot to select the rectangle
    selected_points = plt.ginput(1)
    if len(selected_points) == 0:
        print("No points were selected. Automatically selecting prototypes...")

        # Automatically select prototypes in the upper-right corner of the Delta-Bel graph
        delta_threshold = np.percentile(delta, 95)  # Top 5% delta values
        belief_threshold = np.percentile(BelC, 95)  # Top 5% belief values
        ii = np.where((delta > delta_threshold) & (BelC > belief_threshold))[0]

        print(f"Automatically selected prototype indices: {ii}")
    else:
        t = selected_points[0]  # Extract the selected point (x, y)
        plt.axvline(x=t[0], linestyle='--')
        plt.axhline(y=t[1], linestyle='--')

        # Find the indices of the prototypes based on the user's selection
        ii = np.where((delta > t[0]) & (BelC > t[1]))[0]
        print(f"Manually selected prototype indices: {ii}")

    # Highlight selected prototypes on the plot
    plt.scatter(delta[ii], BelC[ii], c='red', marker='x')
    plt.show()


    # Extract prototypes
    if isinstance(x, np.ndarray):
        g0 = x[ii, :]
    elif isinstance(x, pd.DataFrame):
        g0 = x.iloc[ii, :].values  # Convertir en tableau NumPy si nécessaire
    else:
        raise TypeError("Input matrix x must be a numpy array or pandas DataFrame.")
    return { 'BelC': BelC, 'delta': delta, 'g0': g0, 'ii': ii.tolist() }



#---------------------- setDistances------------------------------------------
def setDistances(x, F, g, m, alpha, distance):
    """
    Computation of distances to centers and variance matrices in each cluster.
    Function called by cecm.

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
        S = [np.eye(nbAtt)] * K 
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
