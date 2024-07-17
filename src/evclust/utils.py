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
    -----------
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
    """Creates an object of class "credpart". `extractMass` computes different outputs (hard, fuzzy, rough partitions, etc.)
        from a credal partition and creates an object of class "credpart".

    Parameters:
    ------------
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
    ---------
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
    ------------
        T. Denoeux and O. Kanjanatarakul. Beyond Fuzzy, 
        Possibilistic and Rough: An Investigation of Belief Functions in Clustering. 
        8th International conference on soft methods in probability and statistics, Rome, 12-14 September, 2016.
        
        M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm. 
        Pattern Recognition, Vol. 41, Issue 4, pages 1384-1397, 2008.
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
        lower_approx_nd.append(np.where((Ynd[:, i] == 1) & (nclus_nd == 1))[0])

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
        
        
        




#---------------------- plot with pca------------------------------------------------
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
    
    
    
    
    

#---------------------- Utils for cat ecm------------------------------------------------  
def catecm_get_dom_vals_and_size(X):
    """Get the feature domains and size.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training instances to cluster.

    Returns
    -------
    dom_vals : array of shape n_unique_vals
        The domains of the features.

    n_attr_doms : int
        The length of the number of categories of X.
    """
    dom_vals = []
    n_attr_doms = []
    n_features = X.shape[1]
    for k in range(n_features):
        unique = list(np.unique(X[:, k]))
        dom_vals += unique
        n_attr_doms += [len(unique)]
    return dom_vals, n_attr_doms

def catecm_check_params(X):
    """Check the correcteness of input parameters.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input intances to be clustered.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        If X contains features with one unique category the feature is dropped.
    """
    attr_with_one_uniq_val = list()
    for l in range(X.shape[1]):
        _, uniq_vals = np.unique(X[:, l], return_counts=True)
        n_l = len(uniq_vals)
        if n_l == 1:
            attr_with_one_uniq_val.append(l)
    if attr_with_one_uniq_val:
        message = f"Attributes {attr_with_one_uniq_val} contain one unique\
            value,they will be dropped before training."
        X = np.delete(X, attr_with_one_uniq_val, axis=1)
    return X

def catecm_init_centers_singletons(n_attr_doms, f, c, size_attr_doms):
    """Initialize the centers of clusters."""
    w0 = np.zeros((n_attr_doms, f), dtype='float')
    for j in range(1, c + 1):
        k = 0
        l = 0
        for n_l in size_attr_doms:
            l += n_l
            rand_num = np.abs(np.random.randn(n_l))
            rand_num /= np.sum(rand_num)
            w0[k:l, j] = rand_num
            k = l
    return w0

def catecm_update_centers_focalsets_gt_2(c, f, F, w):
    """Update the centers of focal sets with size greater than two."""
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    for i in range(c + 1, f):
        idx = list(focalsets[i])
        w[:, i] = w[:, idx].mean(axis=1)
    return w

def catecm_distance_objects_to_centers(F, f, n, size_attr_doms, _dom_vals, X, w):
    """Compute the distance between objects and clusters.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training instances to cluster.
    w : ndarray of shape (n_attr_doms, n_clusters)
        The centers of clusters.

    Returns
    -------
    dist : np.array
        The distances between objects and clusters.
    """
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    dim_dist = f - 1
    dist = np.zeros((n, dim_dist), dtype='float')
    for i in range(n):
        xi = X[i]
        for j in range(dim_dist):
            sum_ = 0.0
            k = 0
            l = 0
            for x_l, n_l in zip(xi, size_attr_doms):
                l += n_l
                dom_val = np.array(_dom_vals[k:l])
                w_ = np.array(w[k:l, j])
                sum_ += 1 - np.sum(w_[dom_val == x_l])
                k += n_l
            dist[i, j] = sum_ / len(focalsets[j + 1])
    return dist

def catecm_get_credal_partition(alpha, beta, delta, n, f, F,  dist):
    """Compute the credal partition from the distances between objects and cluster centers."""
    power_alpha = -alpha / (beta - 1)
    power_beta = -2.0 / (beta - 1)
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    credal_p = np.zeros((n, f), dtype='float')
    for i in range(n):
        if 0 in dist[i, :]:
            credal_p[i, 1:] = 0
            idx_0 = dist[i, :].tolist().index(0)
            #  If the index in dist is i, the index in m is i + 1 as dim(m) = dim(dist) + 1
            idx_0 += 1
            credal_p[i, idx_0] = 1
        else:
            sum_dij = np.sum([
                len(focalsets[k + 1])**power_alpha *
                dist[i, k]**power_beta for k in range(f - 1)
            ])
            for j in range(1, f):
                len_fs = len(focalsets[j])
                credal_p[i, j] = (len_fs**power_alpha *
                                    dist[i, j - 1]**power_beta) / (
                                        sum_dij + delta**power_beta)
    credal_p[:, 0] = 1 - np.sum(credal_p[:, 1:], axis=1)
    credal_p = np.where(credal_p < np.finfo("float").eps, 0, credal_p)
    return credal_p

def catecm_update_centers_singletons(alpha, beta, f, F, c, size_attr_doms, n_attr_doms, _dom_vals, X, credal_p):
    """Update the centers of singletons.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Training instances to cluster.

    credal_p : ndarray (n, f)
        The credal partition.

    Returns
    -------
    w : ndarray of shape (n_attr_doms, f)
        The updated centers of singletons.
    """
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    try:
        mbeta = credal_p**beta
        w = np.zeros((n_attr_doms, f), dtype='float')
        for j in range(1, c + 1):
            s = 0
            z = 0
            for l, n_l in enumerate(size_attr_doms):
                s += n_l
                w_jl = w[z:s, j]
                a_l = _dom_vals[z:s]
                
                attr_values_freq = np.zeros((n_l), dtype="float")
                for t in range(n_l):
                    len_fs = len(focalsets[j])
                    freq = np.sum(mbeta[np.array(X[:, l]) == a_l[t], j])
                    attr_values_freq[t] = len_fs**(alpha - 1) * freq
                idx_max_freq = np.argmax(attr_values_freq)
                w_jl[idx_max_freq] = 1
                w[z:s,j] = w_jl
                z = s
    except RuntimeWarning:
        exit()
    return w

def catecm_cost(F, dist, beta, alpha, delta, credal_p):
    """Compute the cost (intertia) from an iteration.

    Parameters
    ----------
    dist : ndarray of shape (n_samples, n_clusters)
        The distance between objects and clusters.

    credal_p : ndarray of shape (n_samples, n_focalsets)
        The credal partition matrix.

    Returns
    -------
    cost : float
        The cost of the current iteration.
    """
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    len_fs = np.array([len(fs) for fs in focalsets[1:]])
    bba = np.copy(credal_p)
    bba_power = np.where(bba > 0, bba**beta, bba)
    cost = np.sum(len_fs**alpha * bba_power[:, 1:] * dist**2.) + np.sum(delta**2. * bba_power[:, 0])
    return cost