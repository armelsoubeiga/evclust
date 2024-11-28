evclust.ecmdd
=============

.. py:module:: evclust.ecmdd

.. autoapi-nested-parse::

   This module contains the main function for Evidential relational clustering using medoids (ECMdd).

       Kuang Zhou, Arnaud Martin, Quan Pan, and Zhun-ga Liu,
       ECMdd: Evidential c-medoids clustering with multiple prototypes. Pattern Recognition,
       Vol. 60, ISSN 0031-3203, Pages 239-257, 2016.





Module Contents
---------------

.. py:function:: ecmdd(x, c, type='full', alpha=1, beta=2, delta=10, epsi=0.0001, maxit=20, disp=True, gamma=0.5, eta=1)

   Evidential relational clustering using medoids (ecmdd) algorithm with distance and dissimilarity measures.
   In ECMdd, medoids are utilized as the prototypes to represent the detected classes, including specific classes and imprecise classes.

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
       To distinguish the outliers from the possible medoids (default 1).

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

       # ECMdd clustering
       from evclust.ecmdd import ecmdd
       clus = ecmdd(D, c=2, type='full', alpha=1, beta=2, delta=10)

       # Read the output
       from evclust.utils import ev_summary, ev_pcaplot
       ev_pcaplot(data=df, x=clus, normalize=False)

   References:
   -----------
       Kuang Zhou, Arnaud Martin, Quan Pan, and Zhun-ga Liu,
       ECMdd: Evidential c-medoids clustering with multiple prototypes. Pattern Recognition,
       Vol. 60, ISSN 0031-3203, Pages 239-257, 2016.

   .. seealso::
       :func:`~extractMass`

   .. note::
       Keywords : Credal partitions, Relational clustering, Multiple prototypes, Imprecise classes
       Extend medoid-based clustering algorithm on the framework of belief functions.
       Introduce imprecise clusters which enable us to make soft decisions for uncertain data.



