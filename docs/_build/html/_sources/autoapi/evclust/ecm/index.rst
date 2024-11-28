evclust.ecm
===========

.. py:module:: evclust.ecm

.. autoapi-nested-parse::

   This module contains the main function for ecm :

       M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm.
       Pattern Recognition, Vol. 41, Issue 4, pages 1384--1397, 2008.





Module Contents
---------------

.. py:function:: ecm(x, c, g0=None, type='full', pairs=None, Omega=True, ntrials=1, alpha=1, beta=2, delta=10, epsi=0.001, init='kmeans', disp=True)

   Evidential c-means algorithm. `ecm` Computes a credal partition from a matrix of attribute data using the Evidential c-means (ECM) algorithm.

   ECM is an evidential version algorithm of the Hard c-Means (HCM) and Fuzzy c-Means (FCM) algorithms.
   As in HCM and FCM, each cluster is represented by a prototype. However, in ECM, some sets of clusters
   are also represented by a prototype, which is defined as the center of mass of the prototypes in each
   individual cluster. The algorithm iteratively optimizes a cost function, with respect to the prototypes
   and to the credal partition. By default, each mass function in the credal partition has 2^c focal sets,
   where c is the supplied number of clusters. We can also limit the number of focal sets to subsets of
   clusters with cardinalities 0, 1 and c (recommended if c>=10), or to all or some selected pairs of clusters.
   If initial prototypes g0 are provided, the number of trials is automatically set to 1.

   Parameters:
   -----------
   x (DataFrame):
       input matrix of size n x d, where n is the number of objects and d is the number of attributes.
   c (int):
       Number of clusters.
   g0:
       Initial prototypes, matrix of size c x d. If not supplied, the prototypes are initialized randomly.
   type (str):
       Type of focal sets ("simple": empty set, singletons and Omega; "full": all 2^c subsets of Omega;
           "pairs": empty set, singletons, Omega, and all or selected pairs).
   pairs:
       Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
   Omega:
       Logical. If True (default), the whole frame is included (for types 'simple' and 'pairs').
   ntrials (int):
       Number of runs of the optimization algorithm (set to 1 if g0 is supplied).
   alpha (float):
       Exponent of the cardinality in the cost function.
   beta (float):
       Exponent of masses in the cost function.
   delta (float):
       Distance to the empty set.
   epsi (float):
       Minimum amount of improvement.
   init (str):
       Initialization: "kmeans" (default) or "rand" (random).
   disp (bool):
       If True (default), intermediate results are displayed.


   Returns:
   --------
   The credal partition (an object of class "credpart").


   Example:
   --------
   .. highlight:: python
   .. code-block:: python

       # Import test data
       from evclust.datasets import load_iris
       df = load_iris()
       df=df.drop(['species'], axis = 1)

       # Evidential clustering with c=3
       from evclust.ecm import ecm
       model = ecm(x=df, c=3,beta = 1.1,  alpha=0.1, delta=9)

       # Read the output
       from evclust.utils import ev_summary, ev_pcaplot
       ev_summary(model)
       ev_pcaplot(data=df, x=model, normalize=False)

   References:
   -----------
       M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm.
       Pattern Recognition, Vol. 41, Issue 4, pages 1384--1397, 2008.

   .. seealso::
       :func:`~extractMass`

   .. note::
       Keywords : Clustering, Unsupervised learning, Dempster–Shafer theory, Evidence theory, Belief functions, Cluster validity, Robustness



