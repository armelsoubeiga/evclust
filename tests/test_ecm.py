# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from evclust.ecm import ecm
from evclust.datasets import load_decathlon
from evclust.utils import ev_summary


df = load_decathlon()
model = ecm(x=df, c=2)
ev_summary(model)

    


    
    
