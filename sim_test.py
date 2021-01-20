# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:40:33 2020

@author: Kyle
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:12:26 2020

This file is used for testing the simulations on a smaller scale. This file
is NOT used for the main results in Colangelo and Lee (2020).

@author: Kyle Colangelo
"""

import CL2020
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor
import os
from itertools import product


args_lasso1 = {
        'alpha':0.00418519,
        'max_iter':5000,
        'normalize':True,
        'tol':0.001
        }

args_lasso2 = {
        'alpha':0.00281957,
        'max_iter':5000,
        'normalize':True,
        'tol':0.001
        }

args_rf1 = {
        'n_estimators':1000,
        'max_depth':None,
        'min_samples_leaf':40,
        'min_samples_split':40
        }

args_rf2 = {
        'n_estimators':1000,
        'max_depth':None,
        'min_samples_leaf':40,
        'min_samples_split':40
        }

model_lasso1 = linear_model.Lasso(**args_lasso1)
model_lasso2 = linear_model.Lasso(**args_lasso2)

model_rf1 = ExtraTreesRegressor(**args_rf1)
model_rf2 = ExtraTreesRegressor(**args_rf2)

model_nn1 = CL2020.NeuralNet1(101)
model_nn2 = CL2020.NeuralNet2(100)


models = {
        'lasso': [model_lasso1, model_lasso2],
        'rf': [model_rf1, model_rf2],
        'nn': [model_nn1, model_nn2]
        }

basis = {
    'lasso':True,
    'rf':False,
    'nn':False
    }



J = 100
N = 500
L = 5
c = 1
t = 0
beta_hat= np.zeros(J) #initialize the numpy array that will store the parameter estimate
std_error = np.zeros(J) #initialize the numpy array that will store the standard error
ml='nn'

for i in range(J):
    print(1)
    X, T, Y = CL2020.DGP(N)
    model = CL2020.DDMLCT(models[ml][0],models[ml][1])
    model.fit(X,T,Y,t,L,h=h,basis=basis[ml],standardize=False)
    
    beta_hat[i] = model.beta
    std_error[i] = model.std_errors


E_Y_t = 1.2*t + (t**2)

t_stat = np.abs((beta_hat-E_Y_t)/std_error)
t_stat_new = t_stat[~np.isnan(t_stat)]
coverage_rate = np.nanmean((t_stat_new<1.96))
bias = (np.nanmean(beta_hat-E_Y_t)) #can't compute percent when truth is 0
rmse = np.sqrt(np.nanmean((beta_hat-E_Y_t)**2))


print(bias)
print(rmse)  
print(coverage_rate)
























