# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023


import pytest
import numpy as np
from evclust.recm import recm

# Test cases for the ecm function
@pytest.mark.parametrize(
    "x, c, type, pairs, Omega, m0, ntrials, alpha, beta, delta, epsi, maxit, disp, expected_clusters",
    [
        # Test case 1 - Provide basic inputs and check the number of clusters
        (
            np.random.rand(25, 25),  # x: random 100x2 array
            3,  # c: number of clusters
            "full",  # type: type of clustering
            None,  # pairs: cluster pairs
            True,  # Omega: True or False
            None, # m0 : Initial credal partition
            1,  # ntrials: number of trials
            1,  # alpha: alpha value
            2,  # beta: beta value
            None,  # delta: delta value
            1e-3,  # epsi: epsilon value
            50,  # maxit: Maximum number of iterations
            True,  # disp: display output
            3,  # expected number of clusters
        ),
        # Add more test cases as needed
        # ...
    ],
)
def test_recm(x, c, type, pairs, Omega, m0, ntrials, alpha, beta, delta, epsi, maxit, disp, expected_clusters):
    # Call the ecm function with the provided inputs
    recm_model = recm(x, c, type=type, pairs=pairs, Omega=Omega, m0=m0, ntrials=ntrials, alpha=alpha, beta=beta, delta=delta, epsi=epsi, maxit=maxit, disp=disp)

   
    # Check the output elements
    expected_params=['F', 'mass','pl', 'y_pl', 'Y', 'N', 'g', 'D', 'method','W', 'J', 'param']
    for param in expected_params:
        assert param in recm_model
        
    # Check the output type and shape
    assert isinstance(recm_model, dict)

    

