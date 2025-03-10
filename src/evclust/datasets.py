# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains all tests datasets
"""

#---------------------- Packges------------------------------------------------
import pathlib
import pandas as pd
from scipy.io import loadmat
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


#---------------------- Data 3-------------------------------------------------

def load_protein():
  """Protein data. """
  return pd.read_csv(DATASETS_DIR  / "protein.csv", index_col=False)



#---------------------- Data 4-------------------------------------------------

def load_fourclass():
  """fourclass data. """
  return pd.read_csv(DATASETS_DIR  / "fourclass.csv", index_col=False)


#---------------------- Data 5-------------------------------------------------
def load_prop():
    """Load ProP.mat data and return it as a list of views."""
    prop_data = loadmat(str(DATASETS_DIR / 'ProP.mat'))
    return [prop_data['gene_repert'], prop_data['proteome_comp'], prop_data['text']]
