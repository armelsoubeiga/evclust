evclust.kevclus
===============

.. py:module:: evclust.kevclus

.. autoapi-nested-parse::

   This module contains the main function for kevclus :

       T. Denoeux and M.-H. Masson. EVCLUS: Evidential Clustering of Proximity Data.
       IEEE Transactions on Systems, Man and Cybernetics B, Vol. 34, Issue 1, 95--109, 2004.

       T. Denoeux, S. Sriboonchitta and O. Kanjanatarakul. Evidential clustering of large
       dissimilarity data. Knowledge-Based Systems, vol. 106, pages 179-195, 2016.





Module Contents
---------------

.. py:function:: kevclus(x=None, k=None, D=None, J=None, c=None, type='simple', pairs=None, m0=None, ntrials=1, disp=True, maxit=20, epsi=0.001, d0=None, tr=False, change_order=False, norm=1)

   k-EVCLUS algorithm for evidential clustering computes a credal partition from a dissimilarity matrix.
   This version of the EVCLUS algorithm uses the Iterative Row-wise Quadratic Programming (IRQP) algorithm (see ter Braak et al., 2009).
   It also makes it possible to use only a random sample of the dissimilarities, reducing the time and space complexity from quadratic to roughly linear (Denoeux et al., 2016).

   Parameters:
   -----------
   x (Matrix, optional):
       A matrix of size (n, p) containing the values of p attributes for n objects.
   k (int, optional):
       Number of distances to compute for each object. Default is n-1.
   D (Matrix, optional):
       An nxn or nxk dissimilarity matrix. Used only if x is not supplied.
   J (Matrix, optional):
       An nxk matrix of indices. D[i, j] is the distance between objects i and J[i, j].
       Used only if D is supplied and ncol(D) < n; then k is set to ncol(D).
   c (int):
       Number of clusters.
   type (str):
       Type of focal sets ("simple": empty set, singletons, and Omega;
           "full": all 2^c subsets of Omega; "pairs": empty set, singletons, Omega, and all or selected pairs).
   pairs (list, optional):
       Set of pairs to be included in the focal sets. If None, all pairs are included.
       Used only if type="pairs".
   m0 (Matrix, optional):
       Initial credal partition. Should be a matrix with n rows and a number of columns equal
       to the number of focal sets specified by `type` and `pairs`.
   ntrials (int):
       Number of runs of the optimization algorithm. Default is 1.
       Set to 1 if m0 is supplied and change_order=False.
   disp (bool):
       If True (default), intermediate results are displayed.
   maxit (int):
       Maximum number of iterations.
   epsi (float):
       Minimum amount of improvement. Default is 1e-5.
   d0 (float, optional):
       Parameter used for matrix normalization. The normalized distance corresponding
       to d0 is 0.95. Default is set to the 90th percentile of D.
   tr (bool):
       If True, a trace of the stress function is returned.
   change_order (bool):
       If True, the order of objects is changed at each iteration of the IRQP algorithm.
   norm (int):
       Normalization of distances.
       1: division by mean(D^2) (default);
       2: division by n*p.

   Returns:
   --------
   The credal partition (an object of class "credpart").

   Example:
   --------
   .. highlight:: python
   .. code-block:: python

       # Clustering
       from evclust.datasets import load_protein
       from evclust.kevclus import kevclus

       D = load_protein()
       clus = kevclus(D=D, k=30, c=4, type='simple', d0=D.max().max())
       clus['mass"]

   References:
   -----------
       Denoeux, T., & Masson, M. H. (2004). EVCLUS: Evidential Clustering of Proximity Data.
       IEEE Transactions on Systems, Man, and Cybernetics, B, 34(1), 95-109.

       Denoeux, T., Sriboonchitta, S., & Kanjanatarakul, O. (2016). Evidential clustering of large dissimilarity data.
       Knowledge-Based Systems, 106, 179-195.

   .. seealso::
       :func:`~extractMass`, :func:`~makeF`,
       :func:`~createD`

   .. note::
       Keywords : Relational Data, Clustering, Unsupervised Learning, Dempster-Shafer theory, Evidence theory, Belief Functions, Multi-dimensional Scaling
       Given a matrix of dissimilarities between n objects, EVCLUS assigns a basic belief assignment (or mass function) to each object, in such a way
       that the degree of conflict between the masses given to any two objects, reflects their dissimilarity.
       The method was shown to be capable of discovering meaningful clusters in several non-Euclidean data sets, and its performances compared favorably with those of stateof-the-art techniques.


.. py:function:: createD(x, k=None)

   Compute a Euclidean distance matrix.


