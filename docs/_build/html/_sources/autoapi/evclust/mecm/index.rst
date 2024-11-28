evclust.mecm
============

.. py:module:: evclust.mecm

.. autoapi-nested-parse::

   This module contains the main function for Median evidential c-means algorithm (MECM).

       Kuang Zhou, Arnaud Martin, Quan Pan, Zhun-ga Liu,
       Median evidential c-means algorithm and its application to community detection,
       Knowledge-Based Systems, Volume 74, 2015, Pages 69-88, ISSN 0950-7051.





Module Contents
---------------

.. py:function:: mecm(x, c, type='full', alpha=1, beta=2, delta=10, epsi=0.0001, maxit=20, disp=True, gamma=0.5, eta=1)

   Median evidential c-means algorithm (MECM) is using to partitioning relational data.
   MECM is an extension of median c-means and median fuzzy c-means on the theoretical framework of belief function.
   The median variant relaxes the restriction of a metric space embedding for the objects but constrains the prototypes to be in the original data set.

   Parameters:
   -----------
   x (Matrix):
       A distance or dissimilarity matrix of size n x n, where n is the number of objects.
   c (int):
       Number of clusters.
   type (str):
       Type of focal sets ("simple": empty set, singletons, and Omega; "full": all 2^c subsets of Omega;
           "pairs": empty set, singletons, Omega, and all or selected pairs).
   alpha (float):
       Exponent of the cardinality in the cost function.
   beta (float):
       Exponent of masses in the cost function.
   delta (float):
       Distance to the empty set. If None, it is set to the 95th percentile of the upper triangle of x.
   epsi (float):
       Minimum amount of improvement.
   maxit (int):
       Maximum number of iterations.
   disp (bool):
       If True (default), intermediate results are displayed.
   gamma (float):
       [0,1] (0.5 default) weighs the contribution of uncertainty to the dissimilarity between objects and imprecise clusters.
   eta (float):
       > 0 (1 default) use to distinguish the outliers from the possible medoids (default 1).

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
       D = euclidean_distances(df)

       # MECM clustering
       from evclust.mecm import mecm
       clus = mecm(x=D, c=2, type='simple', alpha=1, beta=2, delta=10, epsi=1e-4, maxit=20,
                       disp=True, gamma=0.5, eta=1)

   References:
   -----------
       Kuang Zhou, Arnaud Martin, Quan Pan, Zhun-ga Liu,
       Median evidential c-means algorithm and its application to community detection,
       Knowledge-Based Systems, Volume 74, 2015, Pages 69-88, ISSN 0950-7051.

   .. seealso::
       :func:`~extractMass`, :func:`~makeF`

   .. note::
       Keywords : Credal partition, Belief function theory, Median clustering, Community detection, Imprecise communities
       Median Evidential C-Means (MECM), which is an extension of median c-means and median fuzzy c-means on the theoretical framework of belief functions is proposed.
       Due to these properties, MECM could be applied to graph clustering problems.
       A community detection scheme for social networks based on MECM is investigated and the obtained credal partitions of graphs, which are more refined than crisp and fuzzy ones, enable us to have a better understanding of the graph structures.


