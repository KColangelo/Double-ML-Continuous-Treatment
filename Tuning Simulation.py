# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:17:37 2020

@author: Kyle
"""
#Import necessary packages
import numpy as np
from sklearn.datasets import make_spd_matrix
import math
from sklearn.ensemble import RandomForestRegressor # Our ML algorithm
from scipy.optimize import minimize 
from scipy.stats import gaussian_kde
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import norm
from sklearn import model_selection
from scipy.stats import logistic
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from scipy.stats import logistic
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

#import torch for creating and fitting neural networks.
import torch
import torch.nn as nn
torch.set_default_tensor_type('torch.DoubleTensor')
from scipy.sparse import diags

#sys is used to give arguments via the command line
import sys


###############################################################################
######################### Tuning Parameter Selection ##########################
###############################################################################

N=500
k=1
ml='lasso'

c=1
J=20
alphas = []
alphas1 = []
alphas2 = []
for i in range(1,(J+1)):
    t=0
    X, T, Y,e= DGP2(N)
    
    kept = []
    N=len(Y)
    a=np.zeros(N)
    kernel=np.zeros(N)
    conditional=np.zeros(N)
    gamma_hat=np.zeros(N)
    psi = np.zeros(N)
    var = np.zeros(N)
    e = np.zeros(N)
    mean = np.zeros(N)
    random.shuffle(a)
    #split the data into k parts
    I_split = np.split(np.array(range(N)),k)
    beta_hat = []
    #repeat the value t to length n for computation
    t = np.repeat(t,len(X))
    #Define variable as diffence between T and t.
    T_t = T-t
    t_T = t-T
    
    
    X_aug = gen_basis(T,X)
    X_t = plug_t(t,X)
    X_first_stage = gen_basis_first_stage(X)
    
    for i in list(range(0,k)):
        if k==1:
            I = np.array(range(N))
            I_C = np.array(range(N))
        else:
            #pick the i'th subset
            I=I_split[i]
            n = len(I)
            #Define the complement as the union of all other sets
            I_C = [x for x in np.arange(N) if x not in I]
            
            #Use random forest to estimate g. We use the complement to estimate the
            #function and then evaluate it on the i'th subset of the data.
            #D = np.column_stack((T,X_aug))
        sigma_T = .87
        h = c*sigma_T*(N**(-.2))
        epsilon=h
        g = gaussian_kernel(T_t,h)
        model = linear_model.LassoCV(alphas=None,cv=10,max_iter=10000,normalize=True,tol=0.001)
        model.fit(X_first_stage[I_C],g[I_C])
        alphas2.append(model.alpha_)

        DMLg = linear_model.LassoCV(alphas=None, cv=10, max_iter=5000, normalize=True)
        design = np.column_stack((T[I_C],X_aug[I_C]))
        # Compute ghat
        DMLg.fit(design,Y[I_C])
        alphas.append(DMLg.alpha_)
       
        
        # cdf_plus_h = norm.cdf(((t_T/h)+(epsilon/h)))
        # model1 = linear_model.LassoCV(alphas=None, cv=10, max_iter=10000, normalize=True,tol=0.001)
        # model1.fit(X_first_stage,cdf_plus_h)
        # #Second, -h
        # cdf_minus_h = norm.cdf(((t_T/h)-(epsilon/h)))
        # model2 = linear_model.LassoCV(alphas=None, cv=10, max_iter=10000, normalize=True,tol=0.001)
        # model2.fit(X_first_stage,cdf_minus_h)
        

        # alphas1.append(model1.alpha_)
        # alphas2.append(model2.alpha_)


print(np.mean(alphas))
print(np.mean(alphas2))




n500k5_1 = np.mean(alphas)
n500k5_2 = np.mean(alphas1)
n500k5_3 = np.mean(alphas2)
print(n500k5_1)
print(n500k5_2)
print(n500k5_3)

n500k1_1 = np.mean(alphas)
n500k1_2 = np.mean(alphas1)
n500k1_3 = np.mean(alphas2)

print(n500k1_1)
print(n500k1_2)
print(n500k1_3)
         