# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:40:50 2022

@author: Kyle
"""

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

# %% Define the models we will be using with the specified parameters
model_lasso1 = linear_model.Lasso(**args_lasso1)
model_lasso2 = linear_model.Lasso(**args_lasso2)

# This dictionary defines which ml methods uses added basis functions and which
# do not. An option is included in the fit method of DDMLCT to generate the 
# basis functions. 
basis = {
    'lasso':True,
    }

models = {
        'lasso': [model_lasso1, model_lasso2],
        }
# %% Iterate over all t and ml algorithms for estimation
ml_set = ['lasso'] # All machine learning algorithms
n_set = [500,1000] # All sample sizes used
c_set = [0.5,0.75,1.0,1.25,1.5] # All c's used for bandwidth choice
L_set = [2] # All numbers of folds used for cross-fitting
J = 700 # Number of replications
t=0 # Choice of t to estimate at.

# Establish what directory the files are going to be saved in, and if it doesn't
# exist, then the directory is to be created by the make_dirs function which
# we define in Supplement/file_management.py. 

path = os.getcwd() + "\\Simulations\\Test_Simulations\\"
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
    #X, T, Y = Supplement.DGP(n)
    X, T, Y = Supplement.DGP(n)
    for group in list(product(ml_set,product(L_set,c_set))):
        c = group[1][1]
        ml = group[0]
        L = group[1][0]
        h = 0.87*c*(n**-0.2)
        model = Supplement.DDMLCT_gps2(models[ml][0],models[ml][1])
        model.fit(X,T,Y,t,L=L,h=h,basis=basis[ml],standardize=False)
        out = np.column_stack((model.beta,model.std_errors))
        #print(model.beta)
        name = "dgp_c" + str(c) + "_" + str(ml) + "_L" + str(L) + "_N" +str(n)+ ".csv"
        
        # Baseline rf and nn will not be saved in main folder as they are not
        # included in the main results.
        path = os.getcwd() + "\\Simulations\\Test_Simulations\\"

        
        file = path + name
        with open(file, 'ab') as f:
            np.savetxt(f,out,delimiter=',', fmt='%f')
    # This is an attempt to partially fix the memory leak problem but it
    # is not sufficient.
    gc.collect() 








import numpy as np 
import os
import pandas as pd
from itertools import product


c_set = [0.5,0.75,1.0,1.25,1.5]
n_set = [500,1000]
L_set = [1,2,5]
ml_set = ['lasso','grf','knn']
stats = ['Bias','RMSE','Coverage']

# We initialize the pandas dataframe that will store information in the same
# strucutre as in the paper. We first define the hierarchical row index, and 
# the hierarchical column labels, and then we initialize an empty data frame.
index = [n_set,L_set,c_set]
index = pd.MultiIndex.from_product(index, names=['n', 'L', 'c'])

columns = [ml_set,stats]
columns = pd.MultiIndex.from_product(columns)

output = pd.DataFrame(index=index, columns=columns)


# E_Y_t denotes the true value of the object of interest, in the case of these
# simulations we structured it so that it is exactly equal to 0.
E_Y_t = 0

# Change to the directory where all the results were stored from running
# simulation.py
path = os.getcwd() + "\\Simulations\\Test_Simulations\\"

# Iterate over all combinations of ml method, n, L, and c. Recall that each file
# is named based on these 4 values, so given a set (ml,n,L,c) uniquely identifies
# one of the files. 
ml_set=['lasso']
for group in list(product(ml_set,product(n_set,L_set,c_set))):
    name = "dgp_c" + str(group[1][2]) + "_" + group[0] + \
    "_L" + str(group[1][1]) + "_N" + str(group[1][0]) + ".csv"
    print(name)
    file = path + name
    estimates = np.genfromtxt(file, delimiter=',')
    beta_hat = estimates[:,0]
    std_error = estimates[:,1]
    beta_hat = beta_hat[~np.isnan(beta_hat)]
    std_error = std_error[~np.isnan(beta_hat)]
    t_stat = np.abs((beta_hat-E_Y_t)/std_error)
    
    coverage_rate = np.nanmean((t_stat<1.96),axis=0)
    bias = np.nanmean((beta_hat-E_Y_t),axis=0)
    rmse = np.sqrt(np.nanmean((beta_hat-E_Y_t)**2,axis=0))


    output.loc[group[1],(group[0],'Bias')] = np.round(bias,3)
    output.loc[group[1],(group[0],'RMSE')] = np.round(rmse,3)
    output.loc[group[1],(group[0],'Coverage')] = np.round(coverage_rate,3)

# Define the name of the file to save the results to. The path is unchanged as
# we are saving in the same folder as the other raw results. 
name = 'table_raw.xlsx'
file = path + name 
output.to_excel(file,index=True) 





# %% Import Necessary packages.
import Supplement
import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import matplotlib.pyplot as plt

# %% Read and Initialize Data
path = os.getcwd() + "\\Empirical Application\\"
directories = [path + "\\Testing\\Estimates",path+"\\Testing\\Estimates\\GPS"]
Supplement.make_dirs(directories)

name = 'emp_app.csv'
file = path + name
data = pd.read_csv(file,index_col=0)
data = data.sample(frac=1,random_state=20)

data = pd.concat([data.select_dtypes(exclude='int64'),
                  pd.get_dummies(data.select_dtypes('int64').astype('category'),
                                 drop_first=True)
                  ],
                 axis=1)

X = data.drop(['d','y'], axis=1)
T = data['d']
Y = data['y']

# %% Create the table of summary statistics
file = path + "\\Estimates\\Summary.xlsx"
summary_table = pd.DataFrame(index = pd.Index(['Share of Weeks Unemployed in Second Year (Y)',
                                      'Total Hours Spent in First Year Training (T)'],
                                              name = 'Variable'
                                              ),
                             columns = ['Mean','Median','StdDev','Min','Max']
                             )
summary_table.iloc[0] = [np.mean(Y),np.median(Y),np.std(Y),np.min(Y),np.max(Y)]
summary_table.iloc[1] = [np.mean(T),np.median(T),np.std(T),np.min(T),np.max(T)]
summary_table.to_excel(file,index=True)

# %% Create the histogram of the treatment 
plt.title('Histogram of Hours of Training',fontsize=16)
plt.xlabel('Hours of Training',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.hist(T, bins = 15, histtype='bar',ec='black', color='w')
plt.savefig(path + '\\Figures\\histogram.png')

# %% Define models and their parameters
# Proceed very similarly to the simulation section. Define the models and then
# use the DDMLCT class to estimate the dose-response function. 
args_lasso1 = {
        'alpha':0.00069944,
        'max_iter':10000,
        'tol':0.0001,
        'normalize':True
        }

args_lasso2 = {
        'alpha':0.000160472,
        'max_iter':10000,
        'tol':0.0001,
        'normalize':True
        }

model_lasso1 = linear_model.Lasso(**args_lasso1)
model_lasso2 = linear_model.Lasso(**args_lasso2)



model_rf1 = Supplement.regression_forest()
model_rf2 = Supplement.regression_forest2()

model_nn1 = Supplement.NeuralNet3k((138))
model_nn2 = Supplement.NeuralNet4((138))

models = {
        'lasso': [model_lasso1, model_lasso2],
        'rf': [model_rf1, model_rf2],
        'nn': [model_nn1, model_nn2]
        }

models = {
        'rf': [model_rf1, model_rf2]
        }

basis = {
    'lasso':True,
    'rf':False,
    'nn':False
    }

# %% Iterate over all t and ml algorithms for estimation
import time
start = time.time()
t_list = np.arange(160,2001,40) # Set of all t we will evaluate at.

t_list= np.array([500])
h = np.std(T)*3*(len(Y)**(-0.2)) # Initial rule of thumb bandwidth choice
h=2*h
L=5 # Number of sub-samples for cross-fitting
ml_list = ['nn'] # ml methods to be used.
col_names = ['t','beta','se','h_star','h'] # names for everything we store from the estimation
u  = 0.5


# We first iterate over every method, estimating using an initial rule of 
# thumb bandwidth choice. The second loop performs the estimation a second time
# using the estimated optimal bandwidth from the estimates from the initial 
# bandwidth choice.
for ml in ml_list:
    model = Supplement.DDMLCT_gps2(models[ml][0],models[ml][1])
    model.fit(X,T,Y,t_list,L,h=h,basis=basis[ml],standardize=True)
    
    model2 = Supplement.DDMLCT_gps2(models[ml][0],models[ml][1])
    model2.fit(X,T,Y,t_list,L,h=h*u,basis=basis[ml],standardize=True)
    
    
    print(1,model.h_star)
    Bt = (model.beta-model2.beta)/((model.h**2)*(1-(u**2)))
    h_star = ((model.Vt/(4*(Bt**2)))**0.2)*(model.n**-0.2)
    
    
    print(2,h_star)

end = time.time()

    output = np.column_stack((np.array(t_list),model.beta,model.std_errors,
                                 np.repeat(h_star,len(t_list)),
                                 np.repeat(model.h,len(t_list))))
    output = pd.DataFrame(output,columns=col_names)
    
    path = os.getcwd() + "\\Empirical Application\\Testing\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5.xlsx'
    file = path + name
    output.to_excel(file)

    path = os.getcwd() + "\\Empirical Application\\Testing\\Estimates\\GPS\\"
    name = 'GPS_' + str(ml) +'.xlsx'
    file = path +  name
    model.gps.to_excel(file,index=True)

for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Testing\\Estimates\\"
    name ='emp_app_' + str(ml) + '_c3_L5.xlsx'
    file = path+name
    dat = pd.read_excel(file)
    h = dat['h_star'][0]  
    h = (0.8*h)
    
    if ml=='nn':
        model = Supplement.NN_DDMLCT(models[ml][0],models[ml][1])
        model.fit(X,T,Y,t_list,L,h=h,basis=basis[ml],standardize=True)
    else:
        model = Supplement.DDMLCT(models[ml][0],models[ml][1])
        model.fit(X,T,Y,t_list,L,h=h,basis=basis[ml],standardize=True)
    
    output = np.column_stack((np.array(t_list),model.beta,model.std_errors,
                                 np.repeat(model.h_star,len(t_list)),
                                 np.repeat(model.h,len(t_list))))
    output = pd.DataFrame(output,columns=col_names)
    
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5_hstar.xlsx'
    file = path + name
    output.to_excel(file)

    path = os.getcwd() + "\\Empirical Application\\Estimates\\GPS\\"
    name = 'GPS_' + str(ml) +'_hstar.xlsx'
    file = path +  name
    model.gps.to_excel(file,index=True)
    





