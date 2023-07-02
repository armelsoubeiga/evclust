# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import unittest
import numpy as np
from evclust.ecm import ecm

class TestECM(unittest.TestCase):

    def test_ecm_with_kmeans_init(self):
        # Test ecm with kmeans initialization
        x = np.random.rand(100, 2)  # Example data
        c = 3  # Number of clusters
        result = ecm(x, c, init="kmeans")
        self.assertEqual(len(result), c)  # Check if the number of clusters in the result matches c

    def test_ecm_with_custom_init(self):
        # Test ecm with custom initialization
        x = np.random.rand(100, 2)  # Example data
        c = 4  # Number of clusters
        g0 = np.random.rand(c, 2)  # Custom initial prototypes
        result = ecm(x, c, g0=g0, init="rand")
        self.assertEqual(len(result), c)  # Check if the number of clusters in the result matches c

    def test_ecm_with_invalid_input(self):
        # Test ecm with invalid input (e.g., negative number of clusters)
        x = np.random.rand(100, 2)  # Example data
        c = -1  # Invalid number of clusters
        with self.assertRaises(ValueError):
            ecm(x, c)

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestECM))
