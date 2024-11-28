evclust.wmvec
=============

.. py:module:: evclust.wmvec

.. autoapi-nested-parse::

   This module contains the main function for Adaptive Weighted Multi-View Evidential Clustering (WMVEC).

       Zhe Liu, Haojian Huang, Sukumar Letchmunan, Muhammet Deveci, Adaptive weighted multi-view evidential clustering with feature preference,
       Knowledge-Based Systems, Volume 294, 2024, 111770, ISSN 0950-7051





Module Contents
---------------

.. py:function:: wmvec(X, c, alpha=2, delta=5, maxit=20, epsi=0.001, beta=1.1, lmbda=403.4288, type='simple', disp=True)

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


.. py:function:: Centroids_Initialization(X, K)

.. py:function:: get_distance_wmvec(mode, view, data, nbFoc, Aj, F_update, alpha, beta, delta, features, R_or_M)

.. py:function:: update_Aj_wmvec(view, cluster, features, Aj, F, center, nbFoc)

.. py:function:: update_M_wmvec(view, data, dis, beta, nbFoc)

.. py:function:: update_R_wmvec(view, dis, lmbda)

.. py:function:: update_V_wmvec(view, cluster, alpha, beta, data, M, R, F_update, features)

.. py:function:: update_jaccard_wmvec(view, lmbda, R, dis)

