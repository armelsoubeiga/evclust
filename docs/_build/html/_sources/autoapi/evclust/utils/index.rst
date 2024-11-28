evclust.utils
=============

.. py:module:: evclust.utils

.. autoapi-nested-parse::

   This module contains the utils function





Module Contents
---------------

.. py:function:: makeF(c, type=['simple', 'full', 'pairs'], pairs=None, Omega=True)

   Creation of a matrix of focal sets. `makeF` creates a matrix of focal sets.

   Parameters:
   -----------
   c (int):
       Number of clusters.
   type (str):
       Type of focal sets ("simple": {}, singletons, and Ω; "full": all 2^c subsets of Ω;
               "pairs": {}, singletons, Ω, and all or selected pairs).
   pairs (ndarray or None):
       Set of pairs to be included in the focal sets; if None, all pairs are included. Used only if type="pairs".
   Omega (bool):
       If True (default), Ω is a focal set (for types 'simple' and 'pairs').


   Returns:
   --------
   ndarray: A matrix (f, c) of focal sets.


.. py:function:: get_ensembles(table)

   Get cluster name using mass

   Parameters:
   ------------
   table (ndarray):
       Matrix of mass functions. The first column corresponds to the degree of conflict.

   Returns:
   --------
   Returns a list of cluster names in order according to the max mass approach


.. py:function:: extractMass(mass, F, g=None, S=None, method=None, crit=None, Kmat=None, trace=None, D=None, W=None, J=None, param=None)

   Creates an object of class credpart. extractMass computes different outputs (hard, fuzzy, rough partitions, etc.)
   from a credal partition and creates an object of class credpart.

   Parameters:
   ------------
   mass (ndarray):
       Matrix of mass functions. The first column corresponds to the degree of conflict.
   F (ndarray):
       Matrix of focal sets. The first row always corresponds to the empty set.
   g (ndarray, optional):
       The prototypes (if defined). Defaults to None.
   S (ndarray, optional):
       The matrices S_j defining the metrics for each cluster and each group of clusters (if defined). Defaults to None.
   method (str):
       The method used to construct the credal partition.
   crit (float, optional):
       The value of the optimized criterion (depends on the method used). Defaults to None.
   Kmat (ndarray, optional):
       The matrix of degrees of conflict. Same size as D. Defaults to None.
   trace (ndarray, optional):
       The trace of criterion values. Defaults to None.
   D (ndarray, optional):
       The normalized dissimilarity matrix. Defaults to None.
   W (ndarray, optional):
       The weight matrix. Defaults to None.
   J (ndarray, optional):
       The matrix of indices. Defaults to None.
   param (list, optional):
       A method-dependent list of parameters. Defaults to None.

   Returns:
   ---------
   method (str):
       The method used to construct the credal partition.
   F (ndarray):
       Matrix of focal sets. The first row always corresponds to the empty set.
   mass (ndarray):
       Mass functions.
   g (ndarray, optional):
       The prototypes (if defined).
   S (ndarray, optional):
       The matrices S_j defining the metrics for each cluster and each group of clusters (if defined).
   pl (ndarray):
       Unnormalized plausibilities of the singletons.
   pl_n (ndarray):
       Normalized plausibilities of the singletons.
   p (ndarray):
       Probabilities derived from pl by the plausibility transformation.
   bel (ndarray):
       Unnormalized beliefs of the singletons.
   bel_n (ndarray):
       Normalized beliefs of the singletons.
   y_pl (ndarray):
       Maximum plausibility clusters.
   y_bel (ndarray):
       Maximum belief clusters.
   betp (ndarray):
       Unnormalized pignistic probabilities of the singletons.
   betp_n (ndarray):
       Normalized pignistic probabilities of the singletons.
   Y (ndarray):
       Sets of clusters with maximum mass.
   outlier (ndarray):
       Array of 0's and 1's, indicating which objects are outliers.
   lower_approx (list):
       Lower approximations of clusters, a list of length c.
   upper_approx (list):
       Upper approximations of clusters, a list of length c.
   Ynd (ndarray):
       Sets of clusters selected by the interval dominance rule.
   lower_approx_nd (list):
       Lower approximations of clusters using the interval dominance rule, a list of length c.
   upper_approx_nd (list):
       Upper approximations of clusters using the interval dominance rule, a list of length c.
   N (float):
       Average nonspecificity.
   crit (float, optional):
       The value of the optimized criterion (depends on the method used).
   Kmat (ndarray, optional):
       The matrix of degrees of conflict. Same size as D .
   D (ndarray, optional):
       The normalized dissimilarity matrix .
   trace (ndarray, optional):
       The trace of criterion values .
   W (ndarray, optional):
       The weight matrix .
   J (ndarray, optional):
       The matrix of indices.
   param (list, optional):
       A method-dependent list of parameters.

   References:
   ------------
       T. Denoeux and O. Kanjanatarakul. Beyond Fuzzy,
       Possibilistic and Rough: An Investigation of Belief Functions in Clustering.
       8th International conference on soft methods in probability and statistics, Rome, 12-14 September, 2016.

       M.-H. Masson and T. Denoeux. ECM: An evidential version of the fuzzy c-means algorithm.
       Pattern Recognition, Vol. 41, Issue 4, pages 1384-1397, 2008.


.. py:function:: ev_summary(clus)

   Summary of a credal partition. summary_credpart is the summary method for credpart objects.
   This function extracts basic information from credpart objects.

   Parameters:
   -----------
   clus : object
       An object of class "credpart", encoding a credal partition.

   Returns:
   --------
   None
       Prints basic information on the credal partition.


.. py:function:: ev_plot(x, X=None, ytrue=None, Outliers=True, Approx=1, cex=1, cexvar='pl', cex_outliers=5, cex_protos=5, lwd=1, ask=False, plot_Shepard=False, plot_approx=True, plot_protos=True, xlab='$x_1$', ylab='$x_2$')

   Plotting a credal partition. Generates plots of a credal partition.
   This function plots different views of a credal partition in a two-dimensional attribute space.

   Parameters:
   -----------
   x : object
       An object of class "credpart", encoding a credal partition.
   X : array-like, optional
       A data matrix. If it has more than two columns (attributes), only the first two columns are used.
   ytrue : array-like, optional
       The vector of true class labels. If supplied, a different color is used for each true cluster.
       Otherwise, the maximum-plausibility clusters are used instead.
   Outliers : bool, optional
       If True, the outliers are plotted, and they are not included in the lower and upper approximations of the clusters.
   Approx : int, optional
       If Approx==1 (default), the lower and upper cluster approximations are computed using the interval dominance rule.
       Otherwise, the maximum mass rule is used.
   cex : float, optional
       Maximum size of data points.
   cexvar : str, optional
       Parameter determining if the size of the data points is proportional to the plausibilities ('pl', the default),
       the plausibilities of the normalized credal partition ('pl.n'), the degrees of belief ('bel'),
       the degrees of belief of the normalized credal partition ('bel.n'), or if it is constant ('cst', default).
   cex_outliers : float, optional
       Size of data points for outliers.
   cex_protos : float, optional
       Size of data points for prototypes (if applicable).
   lwd : int, optional
       Line width for drawing the lower and upper approximations.
   ask : bool, optional
       Logical; if True, the user is asked before each plot.
   plot_Shepard : bool, optional
       Logical.
   plot_approx : bool, optional
       Logical; if True (default) the convex hulls of the lower and upper approximations are plotted.
   plot_protos : bool, optional
       Logical; if True (default) the prototypes are plotted (for methods generating prototypes, like ECM).
   xlab : str, optional
       Label of horizontal axis.
   ylab : str, optional
       Label of vertical axis.

   Returns:
   ----------
       None


.. py:function:: ev_pcaplot(data, x, normalize=False, splite=False, cex=8, cex_protos=5)

   Plot PCA results with cluster colors.
   This function performs PCA on the input data and plots the resulting PCA scores,
   using the specified cluster information in 'x'.

   Parameters:
   -----------
   data : DataFrame
       The input data containing the attributes (columns) and samples (rows).
   x : object
       An object of class "credpart", encoding a credal partition.
   normalize : bool, optional
       If True, the data will be normalized before performing PCA. Default is False.
   splite : bool, optional
       If True, provides access to several different axes-level functions that show the views of clusters.

   Returns:
   ---------
       None


.. py:function:: ev_tsplot(X, V, clus, plot_centers=True)

   Plot the results of evidential clustering algorithm for time series.

   :param X: The time series data.
   :type X: array-like
   :param V: The medoid (center) time series for each cluster.
   :type V: array-like
   :param clus: The clustering results, with 'mass' and 'F' keys.
   :type clus: dict
   :param plot_centers: If True, plot the cluster centers in color. If False, plot the individual series and the cluster centers in black.
                        Default is True.
   :type plot_centers: bool, optional


