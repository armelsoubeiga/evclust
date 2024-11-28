evclust.egmm
============

.. py:module:: evclust.egmm

.. autoapi-nested-parse::

   This module contains the main function for Evidential Gaussian Mixture Model  (EGMM).  :
       Lianmeng Jiao, Thierry Denœux, Zhun-ga Liu, Quan Pan, EGMM: An evidential version of the Gaussian mixture model for clustering,
       Applied Soft Computing, Volume 129, 2022, 109619, ISSN 1568-4946.





Module Contents
---------------

.. py:function:: egmm(X, c, type='simple', pairs=None, Omega=True, max_iter=20, epsi=0.001, init='kmeans', disp=True)

   Evidential Gaussian Mixture Model (EGMM) clustering algorithm.
   Model parameters are estimated by Expectation-Maximization algorithm and by
   extending the classical GMM in the belief function framework directly.

   Parameters:
   ------------
   X (ndarray):
       Input data of shape (n_samples, n_features).
   c (int):
       Number of clusters.
   type (str, optional):
       Type of focal sets ('simple', 'full', 'pairs'). Default is 'simple'.
   pairs (ndarray, None):
       Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
   Omega (bool):
       If True (default), the whole frame is included (for types 'simple' and 'pairs').
   max_iter ( int, optional):
       Maximum number of iterations. Default is 100.
   epsi (float, optional):
       Convergence tolerance for the algorithm. Default is 1e-6.
   init (str, optional):
       Initialization method ('random' or 'kmeans'). Default is 'random'.

   Returns:
   ---------
   The credal partition (an object of class "credpart").


   Example:
   --------
   .. highlight:: python
   .. code-block:: python

       from evclust.egmm import egmm
       import numpy as np
       import matplotlib.pyplot as plt

       np.random.seed(42); n = 200
       X1 = np.random.normal(loc=[1, 3], scale=0.5, size=(n//3, 2))
       X2 = np.random.normal(loc=[7, 5], scale=0.3, size=(n//3, 2))
       X3 = np.random.normal(loc=[10, 2], scale=0.7, size=(n//3, 2))
       X = np.vstack([X1, X2, X3])

       clus = egmm(X, c=3, type='full', max_iter=20, epsi=1e-3, init='kmeans')

       clus['F']  # Focal sets
       clus['g']  # Cluster centroids
       clus['mass']  # Mass functions
       clus['y_pl']  # Maximum plausibility clusters

   References:
   -----------
       Lianmeng Jiao, Thierry Denœux, Zhun-ga Liu, Quan Pan, EGMM: An evidential version of the Gaussian mixture model for clustering,
       Applied Soft Computing, Volume 129, 2022, 109619, ISSN 1568-4946.


   .. seealso::
       :func:`~extractMass`, :func:`~makeF`,
       :func:`~init_params_random_egmm`,  :func:`~init_params_kmeans_egmm`

   .. note::
       Keywords : Belief function theory, Evidential partition, Gaussian mixture model, Model-based clustering, Expectation–Maximization
       The parameters in EGMM are estimated by a specially designed Expectation–Maximization (EM) algorithm.
       A validity index allowing automatic determination of the proper number of clusters is also provided.
       The proposed EGMM is as simple as the classical GMM, but can generate a more informative evidential partition for the considered dataset.



.. py:function:: init_params_random_egmm(X, c, f)

   Random initialization of EGMM parameters.


.. py:function:: init_params_kmeans_egmm(X, c, f)

   KMeans-based initialization of EGMM parameters.


