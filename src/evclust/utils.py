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
    """
    Get cluster name using mass
    
    Parameters:
    ------------
    table (ndarray):
        Matrix of mass functions. The first column corresponds to the degree of conflict.

    Returns:
    --------
    Returns a list of cluster names in order according to the max mass approach
    """
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
    Creates an object of class credpart. extractMass computes different outputs (hard, fuzzy, rough partitions, etc.)
    from a credal partition and creates an object of class credpart.

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
        The matrix of degrees of conflict. Same size as D. Defaults to None.
    trace (ndarray, optional): 
        The trace of criterion values. Defaults to None.
    D (ndarray, optional): 
        The normalized dissimilarity matrix. Defaults to None.
    W (ndarray, optional): 
        The weight matrix. Defaults to None.
    J (ndarray, optional): 
        The matrix of indices. Defaults to None.
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
        The matrix of degrees of conflict. Same size as D .
    D (ndarray, optional): 
        The normalized dissimilarity matrix .
    trace (ndarray, optional): 
        The trace of criterion values .
    W (ndarray, optional): 
        The weight matrix .
    J (ndarray, optional): 
        The matrix of indices.
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
    Summary of a credal partition. summary_credpart is the summary method for credpart objects.
    This function extracts basic information from credpart objects.
    
    Parameters:
    -----------
    clus : object
        An object of class "credpart", encoding a credal partition.
    
    Returns:
    --------
    None
        Prints basic information on the credal partition.
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
    -----------
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
        Logical.
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
    -----------
    data : DataFrame
        The input data containing the attributes (columns) and samples (rows).
    x : object
        An object of class "credpart", encoding a credal partition.
    normalize : bool, optional
        If True, the data will be normalized before performing PCA. Default is False.
    splite : bool, optional
        If True, provides access to several different axes-level functions that show the views of clusters. 

    Returns:
    ---------
        None
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
    
    
    
    
    
#---------------------- plot for time series------------------------------------------------
def ev_tsplot(X, V, clus, plot_centers=True):
    """
    Plot the results of evidential clustering algorithm for time series.

    Parameters
    -----------
    X : array-like
        The time series data.
    V : array-like
        The medoid (center) time series for each cluster.
    clus : dict
        The clustering results, with 'mass' and 'F' keys.
    plot_centers : bool, optional
        If True, plot the cluster centers in color. If False, plot the individual series and the cluster centers in black. 
        Default is True.
    """
    # Get the cluster labels from 'clus'
    mas = pd.DataFrame(clus['mass'])
    mas.columns = get_ensembles(clus['F'])
    cluster = pd.Categorical(mas.apply(lambda row: row.idxmax(), axis=1))

    # Number of clusters
    unique_clusters = np.unique(cluster)
    k = len(unique_clusters)

    # Number grid
    grid_cols = int(np.ceil(np.sqrt(k)))
    grid_rows = int(np.ceil(k / grid_cols))

    fig, axes = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=(10, 6))
    plt.rcParams["figure.dpi"] = 100

    colors = plt.cm.viridis(np.linspace(0, 1, k))  

    for i in range(grid_rows):
        for j in range(grid_cols):
            idx = i * grid_cols + j
            if idx < k:
                ax = axes[i, j]
                if plot_centers:
                    ax.plot(V[idx], color=colors[idx], linewidth=2)
                else:
                    cluster_series = X[cluster == unique_clusters[idx]]
                    for series in cluster_series:
                        ax.plot(series, color=colors[idx], alpha=0.5)
                        ax.plot(V[idx], color='black', linewidth=2)
                ax.set_title(f'Cluster {unique_clusters[idx]}')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()
    return fig, axes
