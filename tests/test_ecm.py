# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pytest
import numpy as np
from evclust.ecm import ecm

# Test cases for the ecm function
@pytest.mark.parametrize(
    "x, c, g0, type, pairs, Omega, ntrials, alpha, beta, delta, epsi, init, disp, expected_clusters",
    [
        # Test case 1 - Provide basic inputs and check the number of clusters
        (
            np.random.rand(100, 2),  # x: random 100x2 array
            3,  # c: number of clusters
            None,  # g0: initial cluster centers (None for random initialization)
            "full",  # type: type of clustering
            None,  # pairs: cluster pairs
            True,  # Omega: True or False
            1,  # ntrials: number of trials
            1,  # alpha: alpha value
            2,  # beta: beta value
            10,  # delta: delta value
            1e-3,  # epsi: epsilon value
            "kmeans",  # init: initialization method
            False,  # disp: display output
            3,  # expected number of clusters
        ),
        # Add more test cases as needed
        # ...
    ],
)
def test_ecm(x, c, g0, type, pairs, Omega, ntrials, alpha, beta, delta, epsi, init, disp, expected_clusters):
    # Call the ecm function with the provided inputs
    ecm_model = ecm(x, c, g0=g0, type=type, pairs=pairs, Omega=Omega, ntrials=ntrials, alpha=alpha, beta=beta, delta=delta, epsi=epsi, init=init, disp=disp)

    # Check the number of clusters in the output
    assert len(np.unique(ecm_model['y_pl'])) == expected_clusters
    
    # Check the output elements
    expected_params=['F', 'mass','pl', 'y_pl', 'Y', 'N', 'g', 'D', 'method','W', 'J', 'param']
    for param in expected_params:
        assert param in ecm_model
        
    # Check the output type and shape
    assert isinstance(ecm_model, dict)

    

