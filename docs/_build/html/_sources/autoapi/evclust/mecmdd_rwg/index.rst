evclust.mecmdd_rwg
==================

.. py:module:: evclust.mecmdd_rwg

.. autoapi-nested-parse::

   This module contains the main function for Multi-View Evidential C-Medoid clustering (MECMdd) with adaptive weightings with Relevance Weight for each dissimilarity matrix
   estimated globally (MECMdd-RWG) using weight Sum (MECMdd-RWG-S) constraint or weight Product (MECMdd-RWG-P) constraint.

       Armel Soubeiga, Violaine Antoine and Sylvain Moreno. "Multi-View Relational Evidential C-Medoid Clustering with Adaptive Weighted"
       2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA)





Module Contents
---------------

.. py:function:: mecmdd_rwg(Xlist, c, type='full', alpha=1, beta=1.5, delta=9, epsi=0.0001, disp=True, gamma=0.5, eta=1, weight='sum', s=None)

   Multi-View Evidential C-Medoid clustering (MECMdd) with adaptive weightings with Relevance Weight for each dissimilarity matrix
   estimated globally (MECMdd-RWG) using weight Sum (MECMdd-RWG-S) constraint or weight Product (MECMdd-RWG-P) constraint.

   Parameters:
   -----------
   Xlist (list of np.array):
       A list of square symmetric dissimilarity matrices.
   c (int):
       Number of clusters to create.
   type (str):
       Type of structure ('full' by default).
   alpha (float):
       Weighting parameter for cardinalities.
   beta (float):
       uncertainty exponent for membership degrees.
   delta (float or None):
       Precision parameter (used to calculate delta2). If None, distribution of matrix are considere.
   epsi (float):
       Convergence threshold.
   disp (bool):
       Whether to display the objective function value during iteration.
   gamma (float):
       Coefficient to adjust the imprecise contribution.
   eta (float):
       Coefficient for outlier identification.
   weight (str):
       Type of constraint function using to learn weight ('sum' or 'prod').
   s (float or None):
       Additional weight parameter.

   Returns:
   --------
   The credal partition (an object of class "credpart").

   Example:
   --------
   .. highlight:: python
   .. code-block:: python

       from evclust.mecmdd_rwg import mecmdd_rwg

       matrix1 = np.random.rand(5, 5)
       matrix1 = (matrix1 + matrix1.T) / 2
       matrix2 = np.random.rand(5, 5)
       matrix2 = (matrix2 + matrix2.T) / 2
       Xlist = [matrix1, matrix2]

       clus = mecmdd_rwg(Xlist, c=2, type='simple', alpha=1, beta=1.5, delta=9, epsi=1e-4,
                       disp=True, gamma=0.5, eta=1, weight='sum', s=None)

       clus['param']['lambda'] # View weight

   References:
   -----------
       Armel Soubeiga, Violaine Antoine and Sylvain Moreno. "Multi-View Relational Evidential C-Medoid Clustering with Adaptive Weighted"
       2024 IEEE 11th International Conference on Data Science and Advanced Analytics (DSAA).

   .. seealso::
       :func:`~extractMass`, :func:`~makeF`,
       :func:`~lambdaInit_global`

   .. note::
       Keywords : Evidential clustering, credal partition, relational multi-view, belief function
       MECMdd-RWG is able to address the uncertainty and imprecision of multi-view relational data clustering, and provides a credible partition, which extends fuzzy, possibilistic and rough partitions.
       It can automatically lern the importance of each views by estimated a weight globally for each cluster in a collaborative learning framework.


.. py:function:: lambdaInit_global(weight, p)

   Initializes the weight vector lambda based on the weighting scheme.


