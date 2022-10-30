"""
Created on Fri Mar 27 22:12:26 2020
Last update Monday Jan 10 12:26 pm 2022

This file is for running the main simulation results. Estimates and standard
errors for each simulation are saved in multiple files. We begin by defining the 
parameters used for lasso and generalized random forest, and then create all of the machine
learning models we use. Models ending in '1' are used for the estimation of gamma. 
Models ending in "2" are used for the estimation of the generalized propensity 
score. Random forest is typically abbreviated with "rf",neural network is 
typicall abbreviated with "nn", generalized random forest is abbreviated as 
"grf", and the neural network proposed in Colangelo and Lee (2021) is 
abbreviated as "knn", short for "K Neural Network", in the code.

List of algorithms: ml_list = ['lasso','grf','rf','nn','knn']
Sample sizez: n_list = [500,1000]
number of sub samples for cross-fitting: L_list = [1,5]
number of repetitions: J=1000
coefficient for rule of thumb bandwidth: c_list = [0.5,0.75,1.0,1.25,1.5]

The numerical results of the paper focus on lasso, grf, and knn. Previous
versions of the paper focused on regular random forests and neural networks. 
We left these in the code, and the results are saved to a separate folder 
"Extra_Simulations" for our own reference.

Files are created for each combination of machine learning method, n, L, and c.
Models for random forest and lasso are from sklearn. Neural networks use pytorch.
The neural networks used are defined in /Supplement/models.py, including the 
K Neural Network. The data generating
process used is defined in /Supplement/DGP.py.Estimation is carried out using the
estimator defined in /Supplement/estimation.py as the class DDMLCT. An instance
of the class is initialized with 2 models, the first for the estimation of 
gamma and the second for the estimation of the generalized propensity score.
The .fit method takes arguments for covariates X, treatment T, outcome Y,
bandwidth choice h, choice of t to estimate the dose response function at, 
choice of the number of sub-samples for cross-fitting (L), whether or not to 
use the added basis functions, and whether to standardize the data.

The generalized random forest implementation uses the R package, and utilizes
Rpy2 to use the R package in Python. There is a known issue in our code with
this implementation in that there is a memory leak. After a large number of
simulations which estimate the generalized random forest, the memory usage
increases until the program eventually crashes. We have not found a fix for 
this memory leak issue yet, so please contact us if you figure out how to fix
it. 

If the code is terminated while running, the simulations up until that point 
should be saved, and can be continued by starting the code again.

comments on packages used:
    -Supplement is the package defined for this paper that creates the DGP, main
    estimator, and neural network models.
    -numpy is used to create arrays to store the beta hats and standard errors
    -sklearn.linear_model is used to create the lasso models 
    -sklearn.ensemble.ExtraTreesRegressor is used to create the random forest models
    -os is used to obtain the current working directory for the purpose of file
    organization.
    -Rpy2 is used to call the R random forest package. Note that this code was
    written for windows and the verions for Rpy2 are different for mac, resulting
    in some differences with the functions that may need to be accounted for

"""
# %% Import Necessary packages.
import Supplement
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor
import os
from itertools import product
import gc

# %% Define necessary parameters for ML algorithms. 
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

# %% Define the models we will be using with the specified parameters
model_lasso1 = linear_model.Lasso(**args_lasso1)
model_lasso2 = linear_model.Lasso(**args_lasso2)

model_grf1 = Supplement.regression_forest()
model_grf2 = Supplement.regression_forest2()

model_rf1 = ExtraTreesRegressor(**args_rf1)
model_rf2 = ExtraTreesRegressor(**args_rf2)

# For the neural networks we need to specify the number of covariates.
# we have 100 X variables and one treatment. The Second stage of the regular
# neural network takes treatment as an input, hence "101". But the knn does
# not take treatment as an input, hence why its second stage has "100". The
# First stage regresses T on X, hence we do not do the K neural network.
model_nn1 = Supplement.NeuralNet1(101) 
model_nn2 = Supplement.NeuralNet2(100)

model_knn1 = Supplement.NeuralNet1k(100)
model_knn2 = Supplement.NeuralNet2(100) 

# I collect the models into a dictionary so that they can be easily iterated over
models = {
        'lasso': [model_lasso1, model_lasso2],
        'grf': [model_grf1, model_grf2],
        'rf': [model_rf1,model_rf2],
        'nn': [model_nn1, model_nn2],
        'knn': [model_knn1, model_knn2]
        }

# This dictionary defines which ml methods uses added basis functions and which
# do not. An option is included in the fit method of DDMLCT to generate the 
# basis functions. 
basis = {
    'lasso':True,
    'grf':False,
    'rf':False,
    'nn':False,
    'knn':False
    }

# %% Iterate over all t and ml algorithms for estimation
#ml_set = ['lasso','grf','rf','nn','knn'] # All machine learning algorithms
ml_set = ['lasso','grf','knn']
n_set = [500,1000] # All sample sizes used
c_set = [0.5,0.75,1.0,1.25,1.5] # All c's used for bandwidth choice
L_set = [2] # All numbers of folds used for cross-fitting
J = 1000# Number of replications
t=0 # Choice of t to estimate at.

# Establish what directory the files are going to be saved in, and if it doesn't
# exist, then the directory is to be created by the make_dirs function which
# we define in Supplement/file_management.py. 
path = os.getcwd() + "\\Simulations\\"
Supplement.make_dirs(path)

path = os.getcwd() + "\\Simulations\\Extra_Simulations\\"
Supplement.make_dirs(path)


# The outer loop iterates over every combination of sample size n and replication
# number. We only generate the data with respect to n and replication number,
# and for this reason we had to use 2 loops. The inner loop iterates over ml
# method, choice of c and choice of L. Files are saved with a particular naming
# convention to describe what n,L,c and ml method they correspond to.
# Example: "dgp_c0.5_lasso_L1_N500.csv" means this is a file for lasso, with
# c=0.5, L=1, and n=500
for sim in list(product(range(J),n_set)):
    n = sim[1]
    X, T, Y = Supplement.DGP(n)
    for group in list(product(ml_set,product(L_set,c_set))):
        c = group[1][1]
        ml = group[0]
        L = group[1][0]
        h = 0.87*c*(n**-0.2)
        if ml=='knn':
            model = Supplement.NN_DDMLCT(models[ml][0],models[ml][1])
            model.fit(X,T,Y,t,L=L,h=h,basis=basis[ml],standardize=False)
        else:
            model = Supplement.DDMLCT(models[ml][0],models[ml][1])
            model.fit(X,T,Y,t,L=L,h=h,basis=basis[ml],standardize=False)
        out = np.column_stack((model.beta,model.std_errors))
        name = "dgp_c" + str(c) + "_" + str(ml) + "_L" + str(L) + "_N" +str(n)+ ".csv"
        
        # Baseline rf and nn will not be saved in main folder as they are not
        # included in the main results.
        if ml=='rf' or ml=='nn':
            path = os.getcwd() + "\\Simulations\\Extra_Simulations\\"
        else:
            path = os.getcwd() + "\\Simulations\\"
            
            
        file = path + name
        with open(file, 'ab') as f:
            np.savetxt(f,out,delimiter=',', fmt='%f')
    # This is an attempt to partially fix the memory leak problem but it
    # is not sufficient.
    gc.collect() 



# %% If we wish to use the simulated DML version of our estimator, we can run
# the following simulation, the only difference is that we add the argument
# 'sdml=True'
path = os.getcwd() + "\\Simulations_SDML\\"
Supplement.make_dirs(path)

path = os.getcwd() + "\\Simulations_SDML\\Extra_Simulations\\"
Supplement.make_dirs(path) 

for sim in list(product(range(J),n_set)):
    n = sim[1]
    X, T, Y = Supplement.DGP(n)
    for group in list(product(ml_set,product(L_set,c_set))):
        c = group[1][1]
        ml = group[0]
        L = group[1][0]
        h = 0.87*c*(n**-0.2)
        if ml=='knn':
            model = Supplement.NN_DDMLCT(models[ml][0],models[ml][1])
            model.fit(X,T,Y,t,L=L,h=h,basis=basis[ml],standardize=False,sdml=True)
        else:
            model = Supplement.DDMLCT(models[ml][0],models[ml][1],sdml=True)
            model.fit(X,T,Y,t,L=L,h=h,basis=basis[ml],standardize=False)
        out = np.column_stack((model.beta,model.std_errors))
        name = "dgp_c" + str(c) + "_" + str(ml) + "_L" + str(L) + "_N" +str(n)+ ".csv"
        
        # Baseline rf and nn will not be saved in main folder as they are not
        # included in the main results.
        if ml=='rf' or ml=='nn':
            path = os.getcwd() + "\\Simulations_SDML\\Extra_Simulations\\"
        else:
            path = os.getcwd() + "\\Simulations_SDML\\"
            
            
        file = path + name
        with open(file, 'ab') as f:
            np.savetxt(f,out,delimiter=',', fmt='%f')
    # This is an attempt to partially fix the memory leak problem but it
    # is not sufficient.
    gc.collect() 















