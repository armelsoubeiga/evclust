evclust.catecm
==============

.. py:module:: evclust.catecm

.. autoapi-nested-parse::

   This module contains the main function for catecm :
       A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data,"
       2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.





Module Contents
---------------

.. py:function:: catecm(X, c, type='full', alpha=1, beta=2, delta=10, epsi=0.001, maxit=20, disp=True)

   Categorical Evidential c-emans algorithm.. `catecm` Evidential clustering for categorical data.

   Parameters:
   -----------
       X (DataFrame):
           Data containing only categorical variables with each variable having more than 1 modality
       c (int):
           The number of desired clusters.
       alpha (float):
           Weighting exponent to penalize focal sets with high elements. The value of alpha should be > 1.
       beta (float):
           The fuzziness weigthing exponent. The default value.
       delta (float):
           The distance to the empty set i.e. if the distance between an object and a cluster is greater than delta, the object is considered as an outlier.
       type (str):
           Type of focal sets ("simple": empty set, singletons, and Omega; "full": all 2^c subsets of Omega;
           "pairs": empty set, singletons, Omega, and all or selected pairs).
       epsi (float):
           The stop criteria i.e., if the absolute difference between two consecutive inertia is less than epsillon, then the algorithm will stop.
       maxit (int):
           Maximum number of iterations.
       disp (bool):
           If True (default), intermediate results are displayed.

   Returns:
   --------
       The credal partition (an object of class "credpart").

   References:
   -----------
       A. J. Djiberou Mahamadou, V. Antoine, G. J. Christie and S. Moreno, "Evidential clustering for categorical data,"
       2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE), New Orleans, LA, USA.


