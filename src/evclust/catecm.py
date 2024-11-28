# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for catecm.

    A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data," 
    2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.
"""
#---------------------- Packages------------------------------------------------
import numpy as np
from evclust.utils import makeF, extractMass



#---------------------- catecm------------------------------------------------
def catecm(X, c, type='full', alpha=1, beta=2, delta=10,  epsi=1e-3, maxit=20, disp=True):
    """
    Categorical Evidential c-means algorithm. `catecm` Evidential clustering for categorical data.
    The proposed algorithm, referred to as catECM, considers a new dissimilarity measure and introduces an alternating minimization scheme in order to obtain a credal partition.
    
    Parameters:
    ------------
    X (DataFrame):
        Data containing only categorical variables with each variable having more than 1 modality
    c (int):
        The number of desired clusters.
    alpha (float):
        Weighting exponent to penalize focal sets with high elements. The value of alpha should be > 1.
    beta (float):
        The fuzziness weigthing exponent. The default value.
    delta (float):
        The distance to the empty set i.e. if the distance between an object and a cluster is greater than delta, the object is considered as an outlier.
    type (str): 
        Type of focal sets ("simple": empty set, singletons, and Omega; "full": all 2^c subsets of Omega;
        "pairs": empty set, singletons, Omega, and all or selected pairs).
    epsi (float):
        The stop criteria i.e., if the absolute difference between two consecutive inertia is less than epsillon, then the algorithm will stop.
    maxit (int): 
        Maximum number of iterations.
    disp (bool): 
        If True (default), intermediate results are displayed.

    Returns:
    --------
    The credal partition (an object of class "credpart").

    Example:
    ---------
    .. highlight:: python
    .. code-block:: python

        # CATECM clustering
        import numpy as np
        from evclust.catecm import catecm

        df = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
        delimiter=",", dtype="O")
        soybean = np.delete(df,  df.shape[1] - 1, axis=1)
        clus = catecm(soybean, c=4, type='full', alpha=1, beta=2, delta=10,  epsi=1e-3, disp=True)
        clus['mass"]


    References:
    -----------
        A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data," 
        2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.

    .. seealso::
        :func:`~extractMass`, :func:`~makeF`,
        :func:`~catecm_get_dom_vals_and_size`, :func:`~catecm_check_params`,
        :func:`~catecm_init_centers_singletons`, :func:`~catecm_update_centers_focalsets_gt_2`,
        :func:`~catecm_distance_objects_to_centers`, :func:`~catecm_get_credal_partition`,
        :func:`~catecm_update_centers_singletons`, :func:`~catecm_cost`

    .. note::
        Keywords : clustering, categorical data, credal partition, evidential c-means, belief functions
        Preliminary results on three data sets show that cat-ECM is efficient for the analysis of data sets containing outliers and overlapping clusters. 
        Additional validation work needs to be performed to understand how changes to the various parameters of cat-ECM affects the clustering solution, how these results vary with the
        number of objects in a data set, and how the performance of cat-ECM compares to closed categorical clustering methods.
        Nevertheless, the ability of cat-ECM to handle categorical data  makes it highly useful for the analysis of survey data, which
        are common in for e.g. health research and which often contain     categorical, discrete and continuous data types.
    """
    
    #---------------------- initialisations --------------------------------------
    n = X.shape[0] 
    X = catecm_check_params(X)
    _dom_vals, size_attr_doms = catecm_get_dom_vals_and_size(X)
    n_attr_doms = np.sum(size_attr_doms)
    
    F = makeF(c, type) 
    f = F.shape[0] 
    
    # Initialize the centers of clusters.
    w0 = catecm_init_centers_singletons(n_attr_doms, f, c, size_attr_doms)
    w = catecm_update_centers_focalsets_gt_2(c, f, F, w0)

    #------------------------ iterations--------------------------------
    Jold = np.inf
    is_finished = True
    n_iter = 0
    history = []
    while is_finished and n_iter < maxit:
        n_iter += 1
        dist = catecm_distance_objects_to_centers(F, f, n, size_attr_doms, _dom_vals, X, w[:, 1:])
        m = catecm_get_credal_partition(alpha, beta, delta, n, f, F,  dist)
        
        w = catecm_update_centers_singletons(alpha, beta, f, F, c, size_attr_doms, n_attr_doms, _dom_vals, X, m)
        w = catecm_update_centers_focalsets_gt_2(c, f, F, w)
        
        J = catecm_cost(F, dist, beta, alpha, delta, m.copy())
        
        history.append(J)
        if disp:
            print([n_iter, J])
        is_finished = np.abs(Jold - J) > epsi
        Jold = J
        if J > Jold:
            break
  
    clus = extractMass(m, F, g=w, method="catecm", crit=J, param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus





#---------------------- Utils for cat ecm------------------------------------------------  
def catecm_get_dom_vals_and_size(X):
    """Get the feature domains and size.
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
    """
    attr_with_one_uniq_val = list()
    for l in range(X.shape[1]):
        _, uniq_vals = np.unique(X[:, l], return_counts=True)
        n_l = len(uniq_vals)
        if n_l == 1:
            attr_with_one_uniq_val.append(l)
    if attr_with_one_uniq_val:
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
    """Update the centers of focal sets with size greater than two.
    """
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    for i in range(c + 1, f):
        idx = list(focalsets[i])
        w[:, i] = w[:, idx].mean(axis=1)
    return w

def catecm_distance_objects_to_centers(F, f, n, size_attr_doms, _dom_vals, X, w):
    """Compute the distance between objects and clusters.
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
    """Compute the credal partition from the distances between objects and cluster centers.
    """
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
    """
    focalsets = [tuple(index + 1 for index in row.nonzero()[0]) for row in F]
    len_fs = np.array([len(fs) for fs in focalsets[1:]])
    bba = np.copy(credal_p)
    bba_power = np.where(bba > 0, bba**beta, bba)
    cost = np.sum(len_fs**alpha * bba_power[:, 1:] * dist**2.) + np.sum(delta**2. * bba_power[:, 0])
    return cost