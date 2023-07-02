# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from evclust.ecm import ecm
from evclust.datasets import load_decathlon
from evclust.utils import ev_summary


df = pd.read_csv("F:/package/wpy3/scripts/evclust/src/evclust/datasets/edol.csv")

model = ecm(x=df, c=2,beta = 1.1,  alpha=0.1, delta=9)
model = ecm(x=df, c=2)
ev_summary(model)
ev_plot(x=model,X=df)        
ev_pcaplot(data=df, x=model, normalize=False)    
ev_pcaplot(data=df, x=model, normalize=False, splite=True)  

mean_coords = ind_coord.groupby('Cluster').mean()