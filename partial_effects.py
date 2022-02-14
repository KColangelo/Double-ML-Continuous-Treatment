# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:33:46 2020
Last update Monday Jan 10 1:13 pm 2022

This file is to be run after empirical_application.py. It's purpose is to take
the main estimates from the empirical estimation and compute partial effects. 
The function "shift" is used to create a vector of betas that is shifted by
the specified number of places so that an easy difference can be computed to 
approximate the slope near a given beta. 

"""

import numpy as np
import pandas as pd
import os

# This function shifts the vector of estimates so we can take differences
# between estimates at two different t's.
def shift(xs, gap):
    e = np.empty_like(xs)
    if gap >= 0:
        e[:gap] = np.nan
        e[gap:] = xs[:-gap]
    else:
        e[gap:] = np.nan
        e[:gap] = xs[-gap:]
    return e

# the first loop computes estimates under optimal bandwidth. The second loop
# computes estimates under rule of thumb bandwidth.
ml_list = ['lasso','rf','nn']
gap = 4
eta = 40*gap
for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5_hstar.xlsx'
    file = path + name
    dat = pd.read_excel(file)
    h = dat['h'][0]
    dat['partial effect'] = (dat['beta']-shift(dat['beta'],gap))/eta
    dat['se partial effect'] = ((np.sqrt(15/6)/h)*dat['se'])

    dat.to_excel(file,index=False)


for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5.xlsx'
    file = path + name
    dat = pd.read_excel(file)
    h = dat['h'][0]
    dat['partial effect'] = (dat['beta']-shift(dat['beta'],gap))/eta
    dat['se partial effect'] = ((np.sqrt(15/6)/h)*dat['se'])

    dat.to_excel(file,index=False)











