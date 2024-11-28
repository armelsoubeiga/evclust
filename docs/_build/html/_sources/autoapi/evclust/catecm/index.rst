evclust.catecm
==============

.. py:module:: evclust.catecm

.. autoapi-nested-parse::

   This module contains the main function for catecm.

       A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data,"
       2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.





Module Contents
---------------

.. py:function:: catecm(X, c, type='full', alpha=1, beta=2, delta=10, epsi=0.001, maxit=20, disp=True)

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


.. py:function:: catecm_get_dom_vals_and_size(X)

   Get the feature domains and size.



.. py:function:: catecm_check_params(X)

   Check the correcteness of input parameters.



.. py:function:: catecm_init_centers_singletons(n_attr_doms, f, c, size_attr_doms)

   Initialize the centers of clusters.


.. py:function:: catecm_update_centers_focalsets_gt_2(c, f, F, w)

   Update the centers of focal sets with size greater than two.



.. py:function:: catecm_distance_objects_to_centers(F, f, n, size_attr_doms, _dom_vals, X, w)

   Compute the distance between objects and clusters.



.. py:function:: catecm_get_credal_partition(alpha, beta, delta, n, f, F, dist)

   Compute the credal partition from the distances between objects and cluster centers.



.. py:function:: catecm_update_centers_singletons(alpha, beta, f, F, c, size_attr_doms, n_attr_doms, _dom_vals, X, credal_p)

   Update the centers of singletons.



.. py:function:: catecm_cost(F, dist, beta, alpha, delta, credal_p)

   Compute the cost (intertia) from an iteration.



