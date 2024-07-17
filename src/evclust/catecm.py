# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024

"""
This module contains the main function for catecm :
    A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data," 
    2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.
"""
#---------------------- Packages------------------------------------------------


import numpy as np
from evclust.utils import makeF, extractMass
from evclust.utils import catecm_get_dom_vals_and_size, catecm_check_params, catecm_init_centers_singletons
from evclust.utils import catecm_init_centers_singletons, catecm_update_centers_focalsets_gt_2, catecm_cost
from evclust.utils import catecm_distance_objects_to_centers, catecm_get_credal_partition, catecm_update_centers_singletons


#---------------------- catecm------------------------------------------------
def catecm(X, c, type='full', alpha=1, beta=2, delta=10,  epsi=1e-3, maxit=20, disp=True):
    """Categorical Evidential c-emans algorithm.. `catecm` Evidential clustering for categorical data.

    Parameters:
    -----------
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

    References:
    -----------
        A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data," 
        2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.
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