# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains all tests datasets
"""

#---------------------- Packges------------------------------------------------
import pathlib
import pandas as pd
DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"



#---------------------- Data 1-------------------------------------------------
def load_decathlon():
    """The Decathlon dataset from FactoMineR."""
    
    decathlon = pd.read_csv(DATASETS_DIR / "decathlon.csv")
    decathlon.columns = ["athlete", *map(str.lower, decathlon.columns[1:])]
    decathlon.athlete = decathlon.athlete.apply(str.title)
    decathlon = decathlon.set_index(["competition", "athlete"])
    return decathlon


#---------------------- Data 2-------------------------------------------------
def load_iris():
    """Iris data."""
    return pd.read_csv(DATASETS_DIR / "iris.csv")

