# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024


""" 
Documentation for the package.

A recent developing trend in clustering is the advancement of algorithms that not only identify clusters within data, but also express and capture the uncertainty of cluster membership. 
Evidential clustering addresses this by using the Dempster-Shafer theory of belief functions, a framework designed to manage and represent uncertainty. 
This approach results in a credal partition, a structured set of mass functions that quantify the uncertain assignment of each object to potential groups. 
The Python package evclust, presented here, offers a suite of efficient evidence clustering algorithms as well as tools for visualizing, evaluating and analyzing credal partitions.


Utils Functions in evclust
----------------------------

.. csv-table::
    :header: Functions, Description

    :func:`~makeF()` , Creation of a matrix of focal sets
    :func:`~get_ensembles()` , Labelled focal sets
    :func:`~ev_summary()` , Summary of a credal partition by extracts basic information
    :func:`~ev_plot()` , Generates plots of a credal partition
    :func:`~ev_pcaplot()` , Plot PCA results with cluster colors of a credal partition
    :func:`~extractMass()` , Computes different outputs from a credal partition 



Evidential Clustering Algorithms in evclust
--------------------------------------------

.. csv-table::
    :header: Methods, Description, Functions

    ECM,  Evidential C-Means, :func:`~ecm()`
    RECM, Relational Evidential C-Means, :func:`~recm()`
    k-EVCLUS, K Evidential Clustering, :func:`~kevclus()`
    CatECM, Categorical Evidential C-Means, :func:`~catecm()`
    EGMM,  Evidential Gaussian Mixture Model, :func:`~egmm()`
    BPEC, Belief Peak Evidential Clustering, :func:`~bpec()`
    ECMdd, Evidential C-Medoids, :func:`~ecmdd()`
    MECM, Median Evidential C-Means, :func:`~mecm()`
    WMVEC, Weighted Multi-View Evidential Clustering, :func:`~wmvec()`
    WMVEC-FP, Weighted Multi-View Evidential Clustering With Feature Preference, :func:`~wmvec_fp()`
    MECMdd-RWG, Multi-View Evidential C-Medoid with Relevance Weight estimated Globally, :func:`~mecmdd_rwg()`
    MECMdd-RWL, Multi-View Evidential C-Medoid with Relevance Weight estimated Locally, :func:`~mecmdd_rwl()`
    CCM, Credal C-Means, :func:`~ccm()`




Metrics Functions in evclust
----------------------------

.. csv-table::
    :header: Functions, Description

    :func:`~nonspecificity()`, Non specificity of a credal partition
    :func:`~credalRI()`,  Rand index to compare two credal partitions

    

Coming soon:
------------
* Notebook examples 
* Belief Shift Clustering (BSC), 
* Dynamic evidential clustering, 
* Deep Evidential Clustering (DEC),
* Decision tree-based evidential clustering (DTEC),
* Transfer learning-based evidential c-means clustering (TECM),
* Etc ..

"""