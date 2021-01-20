# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:12:26 2020

This file is for running the main simulation results. Estimates and standard
errors for each simulation are saved in multiple files. We begin by defining the 
parameters used for lasso and random forest, and then create all of the machine
learning models we use. Models ending in '1' are used for the estimation of gamma. 
Models ending in "2" are used for the estimation of the generalized propensity 
score. Random forest is typically abbreviated with "rf" and neural network is 
typicall abbreviated with "nn" in the code.

List of algorithms: ml_list = ['lasso','rf','nn']
Sample sizez: n_list = [500,1000]
number of sub samples for cross-fitting: L_list = [1,5]
number of repetitions: J=1000
coefficient for rule of thumb bandwidth: c_list = [0.5,0.75,1.0,1.25,1.5]

Files are created for each combination of machine learning method, n, L, and c.
Models for random forest and lasso are from sklearn. Neural networks use pytorch.
The neural networks used are defined in /CL2020/models.py. The data generating
process used is defined in /CL2020/DGP.py.Estimation is carried out using the
estimator defined in /CL2020/estimation.py as the class DDMLCT. An instance
of the class is initialized with 2 models, the first for the estimation of 
gamma and the second for the estimation of the generalized propensity score.
The .fit method takes arguments for covariates X, treatment T, outcome Y,
bandwidth choice h, choice of t to estimate the dose response function at, 
choice of the number of sub-samples for cross-fitting (L), whether or not to 
use the added basis functions, and whether to standardize the data.

If the code is terminated while running, the simulations up until that point 
should be saved, and can be continued by starting the code again.

comments on packages used:
    -CL2020 is the package defined for this paper that creates the DGP, main
    estimator, and neural network models.
    -numpy is used to create arrays to store the beta hats and standard errors
    -sklearn.linear_model is used to create the lasso models 
    -sklearn.ensemble.ExtraTreesRegressor is used to create the random forest models
    -os is used to obtain the current working directory for the purpose of file
    organization.

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

# I collect the models into a dictionary so that they can be easily iterated over
models = {
        'lasso': [model_lasso1, model_lasso2],
        'rf': [model_rf1, model_rf2],
        'nn': [model_nn1, model_nn2]
        }

# This dictionary defines which ml methods uses added basis functions and which
# do not. An option is included in the fit method of DDMLCT to generate the 
# basis functions. 
basis = {
    'lasso':True,
    'rf':False,
    'nn':False
    }


ml_set = ['lasso','rf','nn']
n_set = [500,1000]
c_set = [0.5,0.75,1.0,1.25,1.5]
L_set = [1,5]
J = 1000
t=0

# Establish what directory the files are going to be saved in, and if it doesn't
# exist, then the directory is to be created by the make_dirs function which
# we define in CL2020/file_management.py. 
path = os.getcwd() + "\\Simulations\\"
CL2020.make_dirs(path)


# The outer loop iterates over every combination of sample size n and replication
# number. We only generate the data with respect to n and replication number,
# and for this reason we had to use 2 loops. The inner loop iterates over ml
# method, choice of c and choice of L. Files are saved with a particular naming
# convention to describe what n,L,c and ml method they correspond to.
# Example: "dgp_c0.5_lasso_L1_N500.csv" means this is a file for lasso, with
# c=0.5, L=1, and n=500
for sim in list(product(range(J),n_set)):
    n = sim[1]
    X, T, Y = CL2020.DGP(n)
    for group in list(product(ml_set,product(L_set,c_set))):
        c = group[1][1]
        ml = group[0]
        L = group[1][0]
        h = 0.87*c*(n**-0.2)

        model = CL2020.DDMLCT(models[ml][0],models[ml][1])
        model.fit(X,T,Y,t,L=L,h=h,basis=basis[ml],standardize=False)
        print(model.beta)
        out = np.column_stack((model.beta,model.std_errors))
        name = "dgp_c" + str(c) + "_" + str(ml) + "_L" + str(L) + "_N" +str(n)+ ".csv"
        file = path + name
        with open(file, 'ab') as f:
            np.savetxt(f,out,delimiter=',', fmt='%f')





















