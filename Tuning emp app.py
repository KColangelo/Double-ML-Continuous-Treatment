# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:26:01 2020

@author: Kyle
"""

# Import Necessary packages.
import numpy as np
import pandas as pd
from sklearn.datasets import make_spd_matrix
import math
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import minimize 
from scipy.stats import gaussian_kde
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import norm
from sklearn import model_selection
from scipy.stats import logistic
from sklearn import preprocessing
import os
import torch
import torch.nn as nn
import sys
import sklearn
from numpy import genfromtxt
from matplotlib import pyplot as plt
import os
torch.set_default_tensor_type('torch.DoubleTensor')

# Set directory to where the data is located
os.chdir('C:/Users/kyle/OneDrive/Double ML')

#Read the data from file
data = pd.read_csv('emp_app.csv',index_col=0)

# This randomizes the rows, for the purpose of sample splitting later during estimation
data = data.sample(frac=1)

a = data.select_dtypes('int64')
b = data.select_dtypes('float64')

#Convert discrete varaibles to dummies
a = a.astype('category')
a = pd.get_dummies(a)

# Redefine data with new dummies replacing the old discrete variables
data = pd.concat([b,a],axis=1)

# Define treatment T, outcome Y, covariates X.
T = np.array(data['d'])
Y = np.array(data['y'])
mwearn = data.iloc[:,2]
mwearn_scale = preprocessing.scale(mwearn).reshape(len(mwearn),1)
X = np.append(mwearn_scale,np.array(data.iloc[:,3:]),1)

X = X[Y!=0]
T = T[Y!=0]
Y = Y[Y!=0]



#Store this for the purpose of rescaling at the end
a = np.mean(Y) 
b = np.std(Y)

#Standardize the data
scaler_T = preprocessing.StandardScaler().fit(T.reshape(1,-1))
scaler_Y = preprocessing.StandardScaler().fit(Y.reshape(1,-1))
T_scale = preprocessing.scale(T)
Y_scale = preprocessing.scale(Y)
X_scale = X

# Store mean and standard deviation of T for later scaling
sd_T = np.std(T)
mean_T = np.mean(T)



def e_kernel(x,h):
    k = (1/h)*(3/4)*(1-((x/h)**2))
    k = k*(abs(x/h)<=1)
    return k


def gen_basis(T,X):
    T = T.reshape(len(T),1)
    #x = X[:,0].reshape(500,1)
    new = np.column_stack(((T**2),(T**3),X,T*X,X**2,X**3))
    return new
    
def gen_basis_first_stage(X):
    new = np.column_stack((X,(X**2),(X**3)))
    return new



def plug_t(t,X):
    t = t.reshape(len(t),1)
    #x = X[:,0].reshape(500,1)
    new =np.column_stack(((t**2),(t**3),X,t*X,X**2,X**3))
    return new

def gaussian_kernel(x,h):
    k = (1/h)*norm.pdf((x/h),0,1)
    return k

X = X_scale
T = T_scale
Y = Y_scale

X_aug = gen_basis(T,X)
X_first_stage = gen_basis_first_stage(X)


X_aug, ind = np.unique(X_aug,axis=1,return_index=True)
X_first_stage = np.unique(X_first_stage, axis=1)

DMLg = linear_model.LassoCV(alphas=None,cv=5, max_iter=10000, normalize=True,tol=0.001)
DMLg.fit(np.column_stack((T,X_aug)),Y)
DMLg.alpha_

# t=0
h = 3*(len(Y)**(-0.2))
g = gaussian_kernel(T,h)
model = linear_model.LassoCV(alphas=None, cv=5, max_iter=10000, normalize=True,tol=0.001)
model.fit(X_first_stage,g)
model.alpha_































