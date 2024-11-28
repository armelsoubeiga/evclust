evclust.bpec
============

.. py:module:: evclust.bpec

.. autoapi-nested-parse::

   This module contains the main function for Belief Peak Evidential Clustering (BPEC).

       Z.-G. Su and T. Denoeux. BPEC: Belief-Peaks Evidential Clustering. IEEE Transactions on Fuzzy Systems, 27(1):111-123, 2019.





Module Contents
---------------

.. py:function:: bpec(x, g, type='full', pairs=None, Omega=True, alpha=1, beta=2, delta=10, epsi=0.001, disp=True, m0=None)

   Belief Peak Evidential Clustering (BPEC) computes a credal partition from a matrix of attribute data.
   BPEC is identical to ECM, except that the prototypes are computed from delta-Bel graph using function delta_Bel.
   The ECM algorithm is then run keeping the prototypes fixed. The distance to the prototypes can be the Euclidean disatnce.

   Parameters:
   ------------
   x (numpy.ndarray):
       Input matrix of size n x d, where n is the number of objects and d the number of attributes.
   g (numpy.ndarray, or None):
       Matrix of size c x d of prototypes (the belief peaks).
   type (str):
       Type of focal sets ("simple", "full", or "pairs").
   Omega (bool):
       Whether to include the whole frame in focal sets (default: True).
   pairs (list or None):
       Pairs to include in focal sets, used only if type="pairs".
   alpha (float):
       Exponent of the cardinality in the cost function.
   beta (float):
       Exponent of masses in the cost function.
   delta (float):
       Distance to the empty set.
   epsi (float):
       Minimum amount of improvement.
   disp (bool):
       Whether to display intermediate results (default: True).
   m0 (numpy.ndarray or None):
       Initial credal partition. Should be a matrix with n rows and a number of columns equal to the number
       of focal sets specified by 'type' and 'pairs'.

   Returns:
   --------
   The credal partition (an object of class "credpart").

   Example:
   --------
   .. highlight:: python
   .. code-block:: python

       # Clustering
       from evclust.datasets import load_fourclass
       from evclust.bpec import bpec, delta_Bel

       df = load_fourclass()
       x = df.iloc[:, 0:2]
       DB = delta_Bel(x,100,0.9)
       clus = bpec(x, DB['g0'], type='pairs', delta=3, distance=1)

   References:
   ------------
       Z.-G. Su and T. Denoeux. BPEC: Belief-Peaks Evidential Clustering. IEEE Transactions on Fuzzy Systems, 27(1):111-123, 2019.


   .. seealso::
       :func:`~extractMass`, :func:`~makeF`,
       :func:`~delta_Bel`,  :func:`~setDistances`

   .. note::
       Keywords : Dempster-Shafer theory, belief functions, unsupervised learning, soft clustering, density peaks clustering.
       BPEC can find the true number of clusters and create a credal partition for some datasets with good performances.
       When the number of clusters is small (usually less than ten), the performances of BPEC and its informative variant with a limited number of composite clusters are approximately equal.
       In contrast, BPEC can be enhanced if less informative composite clusters (i.e., focal sets) when the number of clusters is large.
       Furthermore, BPEC can provide hard, fuzzy, possibilistic and even rough partitions.


.. py:function:: delta_Bel(x, K, q=0.9)

   Delta-Bel graph for Belief Peak Evidential Clustering (BPEC).
   This function computes the delta-Bel graph used to determine the prototypes in the Belief Peak Evidential Clustering (BPEC) algorithm. The user must manually specify
   the rectangles containing the prototypes (typically in the upper-right corner of the graph if the clusters are well-separated). These prototypes are then used
   by the bpec function to compute a credal partition.

   Parameters:
   -----------
   x (numpy.ndarray):
       Input matrix of size n x d, where n is the number of objects and d is the number of attributes.
   K (int):
       Number of neighbors used to determine belief values.
   q (float):
       Parameter of the algorithm, between 0 and 1 (default: 0.9).

   Returns:
   --------
   dict: A dictionary containing:
       BelC (numpy.ndarray):
           The belief values.
       delta (numpy.ndarray):
           The delta values.
       g0 (numpy.ndarray):
           A c x d matrix containing the prototypes.
       ii (list):
           List of indices of the belief peaks.


.. py:function:: setDistances(x, F, g, m, alpha, distance)

   Computation of distances to centers and variance matrices in each cluster.
   Function called by cecm.



