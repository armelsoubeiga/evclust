evclust.wmvec_fp
================

.. py:module:: evclust.wmvec_fp

.. autoapi-nested-parse::

   This module contains the main function for Adaptive Weighted Multi-View Evidential Clustering With Feature Preference (WMVEC-FP).

       Zhe Liu, Haojian Huang, Sukumar Letchmunan, Muhammet Deveci, Adaptive weighted multi-view evidential clustering with feature preference,
       Knowledge-Based Systems, Volume 294, 2024, 111770, ISSN 0950-7051





Module Contents
---------------

.. py:function:: wmvec_fp(X, c, alpha=2, delta=5, maxit=20, epsi=0.001, beta=1.1, lmbda=403.4288, gamma=1, type='simple', disp=True)

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


.. py:function:: Centroids_Initialization_fp(X, K)

.. py:function:: get_distance_fp(mode, view, data, nbFoc, Aj, F_update, alpha, beta, delta, features, *args)

.. py:function:: update_Aj_fp(view, cluster, features, Aj, F, center, nbFoc)

.. py:function:: update_M_fp(view, data, dis, beta, nbFoc)

.. py:function:: update_R_fp(view, dis, lmbda)

.. py:function:: update_V_fp(view, cluster, alpha, beta, data, M, R, F_update, features)

.. py:function:: update_W_fp(view, dis, features, gamma)

.. py:function:: update_jaccard_fp(view, lmbda, gamma, R, W, dis)

