# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2024


import pytest
import numpy as np
from evclust.catecm import catecm

# Test cases for the ecm function
@pytest.mark.parametrize(
    "X, c, type, alpha, beta, delta, epsi, maxit, disp, expected_clusters",
    [
        # Test case 1 - Provide basic inputs and check the number of clusters
        (
            np.array([np.random.choice(['A', 'B', 'C'], 25), np.random.choice(['X', 'Y', 'Z'], 25)]).T,  # x: random 25x2 array of categorical data
            3,  # c: number of clusters
            "full",  # type: type of clustering
            1,  # alpha: alpha value
            2,  # beta: beta value
            10,  # delta: delta value
            1e-3,  # epsi: epsilon value
            20,  # maxit: Maximum number of iterations
            True,  # disp: display output
            3,  # expected number of clusters
        ),
        # Add more test cases as needed
        # ...
    ],
)
def test_catecm(X, c, type, alpha, beta, delta, epsi, maxit, disp, expected_clusters):
    # Call the ecm function with the provided inputs
    catecm_model = catecm(X, c, type=type, alpha=alpha, beta=beta, delta=delta, epsi=epsi, maxit=maxit, disp=disp)

   
    # Check the output elements
    expected_params=['F', 'mass','pl', 'y_pl', 'Y', 'N', 'g', 'D', 'method','W', 'J', 'param']
    for param in expected_params:
        assert param in catecm_model
        
    # Check the output type and shape
    assert isinstance(catecm_model, dict)

    

