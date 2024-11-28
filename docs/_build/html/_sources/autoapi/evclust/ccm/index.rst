evclust.ccm
===========

.. py:module:: evclust.ccm

.. autoapi-nested-parse::

   This module contains the main function for Credal c-means (CCM) clustering method based on belief functions.

       Zhun-ga Liu, Quan Pan, Jean Dezert, Grégoire Mercier, Credal c-means clustering method based on belief functions,
       Knowledge-Based Systems, Volume 74, 2015, Pages 119-132, ISSN 0950-7051





Module Contents
---------------

.. py:function:: ccm(x, c, type='full', gamma=1, beta=2, delta=10, epsi=0.001, maxit=50, init='kmeans', disp=True)

   Credal c-means (CCM) is a clustering method designed to handle uncertain and imprecise data.
   It addresses this challenge by assigning data points to meta-clusters, which are essentially unions of closely related clusters.
   This approach allows for more flexible and nuanced cluster assignments, reducing misclassification errors.
   CCM is particularly robust to noisy data due to the inclusion of an outlier cluster.

   Parameters:
   -----------
   x (Matrix):
       Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
   c (int):
       Number of clusters.
   type (str):
       Type of focal sets ("simple": empty set, singletons, and Omega; "full": all 2^c subsets of Omega;
           "pairs": empty set, singletons, Omega, and all or selected pairs).
   beta (float):
       Exponent of masses in the cost function.
   delta (float):
       > 0. Distance to the empty set.
   epsi (float):
       Minimum amount of improvement.
   maxit (int):
       Maximum number of iterations.
   disp (bool):
       If True (default), intermediate results are displayed.
   gamma (float):
       > 0 weight of the distance (by default take 1).

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

       # CCM clustering
       from evclust.ccm import ccm
       clus = ccm(x=df, c=3, type="full", gamma=1, beta=2, delta=10, epsi=1e-3,
               maxit=50, init="kmeans", disp=True)

   References:
   -----------
       Zhun-ga Liu, Quan Pan, Jean Dezert, Grégoire Mercier, Credal c-means clustering method based on belief functions,
       Knowledge-Based Systems, Volume 74, 2015, Pages 119-132, ISSN 0950-7051.

   .. seealso::
       :func:`~extractMass`, :func:`~makeF`

   .. note::
       Keywords : Belief functions, Credal partition, Data clustering, Uncertain data
       A meta-cluster threshold is introduced in CCM to eliminate the meta-clusters with big cardinality.
       CCM can still provide good clustering results with admissible computation complexity.
       The credal partition can be easily approximated to a fuzzy (probabilistic) partition if necessary,
       thanks to the plausibility transformation, pignistic transformation or any other available transformation that may be preferred by the user.
       The output of CCM is not necessarily used for making the final classification, but it can serve as an interesting source of information to combine with additional complementary information sources if one wants to get more precise results before taking decision.
       The effectiveness of CCM has been proved through different experiments using artificial and real data sets.


