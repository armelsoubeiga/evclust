evclust.recm
============

.. py:module:: evclust.recm

.. autoapi-nested-parse::

   This module contains the main function for Relational Evidential C-Means (RECM).

       M.-H. Masson and T. Denoeux. RECM: Relational Evidential c-means algorithm.
       Pattern Recognition Letters, Vol. 30, pages 1015--1026, 2009.





Module Contents
---------------

.. py:function:: recm(D, c, type='full', pairs=None, Omega=True, m0=None, ntrials=1, alpha=1, beta=1.5, delta=None, epsi=0.0001, maxit=50, disp=True)

   Relational Evidential c-means algorithm.
   `recm` computes a credal partition from a dissimilarity matrix using the Relational Evidential c-means (RECM) algorithm.

   Parameters:
   -----------
   D (Matric):
       Dissimilarity matrix of size (n,n), where n is the number of objects. Dissimilarities must be squared Euclidean distances to ensure convergence.
   c (int):
       Number of clusters.
   type (str):
       Type of focal sets ("simple": empty set, singletons and Omega; "full": all 2^c subsets of Omega; "pairs": empty set, singletons, Omega, and all or selected pairs).
   pairs (ndarray or None):
       Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
   Omega (bool):
       If True (default), the whole frame is included (for types 'simple' and 'pairs').
   m0 (ndarray or None):
       Initial credal partition. Should be a matrix with n rows and a number of columns equal to the number f of focal sets specified by 'type' and 'pairs'.
   ntrials (int):
       Number of runs of the optimization algorithm (set to 1 if m0 is supplied).
   alpha (float):
       Exponent of the cardinality in the cost function.
   beta (float):
       Exponent of masses in the cost function.
   delta (float):
       Distance to the empty set.
   epsi (float):
       Minimum amount of improvement.
   maxit (int):
       Maximum number of iterations.
   disp (bool):
       If True (default), intermediate results are displayed.

   Returns:
   --------
   The credal partition (an object of class "credpart").

   Example:
   --------
   .. highlight:: python
   .. code-block:: python

       # Test data
       from sklearn.metrics.pairwise import euclidean_distances
       from evclust.datasets import load_iris
       df = load_iris()
       df = df.drop(['species'], axis = 1)
       distance_matrix = euclidean_distances(df)

       # RECM clustering
       from evclust.recm import recm
       clus = recm(D=distance_matrix, c=3)
       ev_summary(model)

   References:
   -----------
       M.-H. Masson and T. Denoeux. RECM: Relational Evidential c-means algorithm.
       Pattern Recognition Letters, Vol. 30, pages 1015--1026, 2009.

   .. seealso::
       :func:`~extractMass`, :func:`~makeF`

   .. note::
       Keywords : Clustering, Proximity data, Unsupervised learning, Dempster–Shafer theory, Belief functions
       RECM algorithm can be seen as an evidential counterpart of relational fuzzy clustering algorithm such as RFCM. Although based on the assumption that the input dissimilarities are squared Euclidean distances.
       The advantages of RECM are twofold: first, RECM is faster and more stable; secondly, it allows the construction of general credal partition in which belief masses are assigned to focal sets of any cardinality, thus exploiting the full expressive power of belief functions.



