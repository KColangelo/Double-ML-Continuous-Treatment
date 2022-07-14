# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:45:54 2020
Last update Monday Jan 10 1:30 pm 2022

In this file we perform the main computation for the empirical application section
of Colangelo and Lee (2021). We start by reading in the data which we put in the 
sub folder "Empirical Application". Categorical variables are converted to dummies 
and we then define our outcome Y, treatment T, and covariates X. The machine learning
models to be used are then defined, as well as the values of t to evaluate the 
estimator at, the initial bandwidth choice, and the choice of L for the number
of sub-samples used for cross-fitting. We then loop over all 3 machine learning
methods and create a model using DDMLCT and fit the model on the data. basis functions
are only used for lasso, so this parameter is set to True only for lasso. After
an initial estimation with the initial bandwidth, the DDMLCT model object stores
and estimate of h_star. This h_star is then used in a successive estimation with
the new estimated optimal bandiwidth. The main results are stored in the 
"/empirical application/estimates" folder. GPS estimates are stored in 
"/empirical application/GPS". Although the GPS estimates are not referenced in the
paper, we still stored them for further evaluation of the estimator. 

Comments on packages used:
    -Supplement is the package designed specifically for this project which defines
    the neural network models used, and main estimator DDMLCT, and a tool we use
    for managing the file structure of the output.
    -numpy is primarily used just to stack the estimates we want into a matrix
    which is then converted to a pandas dataframe and stored in excel
    -pandas is used to read and manipulate the data before it is used to fit the models.
    It is also used to store the estimates in a nice format to then be saved to excel
    -sklearn.linear_model is used to create the lasso models
    -sklearn.ensemble.ExtraTreesRegressor is used to create the random forest models
    -os is used to get the current directory. We use os.getcwd() to create and
    use the file strucutre so that the code is robust to being run on different machines.

"""

# %% Import Necessary packages.
import Supplement
import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import matplotlib.pyplot as plt

# %% Read and Initialize Data
path = os.getcwd() + "\\Empirical Application\\"
directories = [path + "\\Estimates",path+"\\Estimates\\GPS"]
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
plt.hist(data[data['d']<2000]['d'], bins = 15, histtype='bar',ec='black', color='w')
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
        'nn': [model_nn1, model_nn2]
        }

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

# %% Iterate over all t and ml algorithms for estimation
t_list = np.arange(160,2001,40) # Set of all t we will evaluate at.
h = np.std(T)*3*(len(Y)**(-0.2)) # Initial rule of thumb bandwidth choice
h=2*h
L=5 # Number of sub-samples for cross-fitting
ml_list = ['lasso','rf','nn'] # ml methods to be used.
col_names = ['t','beta','se','h_star','h'] # names for everything we store from the estimation
u=0.5


# We first iterate over every method, estimating for bandwidth 2*h
# where h is chosen as a rule of thumb bandwidth 3*std(T)*N^(-0.2).
# We use the estimates from these two bandwidth choices and apply
# the optimal bandwidth estimator in Colangelo and Lee (2022) to obtain
# the final optimal bandwidth. The second loop then iterates over all ML
# methods and uses the optimal bandwidth. 
for ml in ml_list:
    if ml=='nn':
        model = Supplement.NN_DDMLCT(models[ml][0],models[ml][1])
        model.fit(X,T,Y,t_list,L,h=h,basis=basis[ml],standardize=True)

        model2 = Supplement.NN_DDMLCT(models[ml][0],models[ml][1])
        model2.fit(X,T,Y,t_list,L,h=h*u,basis=basis[ml],standardize=True)
    else:
        model = Supplement.DDMLCT(models[ml][0],models[ml][1])
        model.fit(X,T,Y,t_list,L,h=h,basis=basis[ml],standardize=True)

        model2 = Supplement.DDMLCT(models[ml][0],models[ml][1])
        model2.fit(X,T,Y,t_list,L,h=h*u,basis=basis[ml],standardize=True)
        

    Bt = (model.beta-model2.beta)/((model.h**2)*(1-(u**2)))
    h_star = np.mean(((model2.Vt/(4*(Bt**2)))**0.2)*(model.n**-0.2))
    print(h_star)
    
    output = np.column_stack((np.array(t_list),model.beta,model.std_errors,
                                 np.repeat(h_star,len(t_list)),
                                 np.repeat(model.h,len(t_list))))
    output = pd.DataFrame(output,columns=col_names)
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5.xlsx'
    file = path + name
    output.to_excel(file)

    path = os.getcwd() + "\\Empirical Application\\Estimates\\GPS\\"
    name = 'GPS_' + str(ml) +'.xlsx'
    file = path +  name
    model.gps.to_excel(file,index=True)
    

for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
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
    

































