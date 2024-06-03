# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:21:06 2019
Last update Sunday Oct 27 11:21 am 2023

This file is for taking the raw data files, which contain point estimates
and standard errors for each simulation, and turning them into a nice
table which summarizes the stats we are interested in (bias, RMSE, Coverage rate).
We iterate over all the files from the simulations, compute the coverage rate,
bias and rmse and then input that information into the pandas dataframe
called "output". Output is defined with hierarchical row labels for n,L and c,
and hierarchical column names for the machine learning method, and statistic of
interest (bias,rmse,coverage). The new table is saved in the same folder
as the raw simulation data as "table_raw.xlsx". Additional formatting in excel is 
necessary to reproduce the exact formatting in Colangelo and Lee (2023). The file 
"dgp2_table_beta.xlsx" contains the exact formatting used in the paper for beta. 
While "dgp2_table_theta.xlsx" containts the exact formatting use in the paper
for theta.  After getting the results in "dgp2_table.xlsx" we use the 
Excel2Latex add-in to convert it into a latex table.

Comments on packages used:
    -Numpy is used for reading the raw csv files, mathematical operations, and rounding
    -os is used to obtain current working directory for use in locating files. Used 
    for the Purpose of generalizing the code to easily reproduce results on any machine.
    -pandas is used for creating the dataframe which will store the bias, rmse and 
    coverage information
    -itertools.product allows us to create a cartesian product between python lists.
    I used this to easily iterate over the different choices of c,L,n, and ml
    method very easily.

"""
# %% Import necessary packages
import numpy as np 
import os
import pandas as pd
from itertools import product


dgps = ['dgp2']
c_set = [1.0,1.25,1.5]
#c_set = [1.25]
n_set = [1000,10000]
L_set = [1,5]
#ml_set = ['lasso','nn']
ml_set = ['lasso']
stats = ['Bias','RMSE','Coverage']
# We initialize the pandas dataframe that will store information in the same
# strucutre as in the paper. We first define the hierarchical row index, and 
# the hierarchical column labels, and then we initialize an empty data frame.

for dgp in dgps:
        
    index = [n_set,L_set,c_set]
    index = pd.MultiIndex.from_product(index, names=['n', 'L', 'c'])
    
    columns = [ml_set,stats]
    columns = pd.MultiIndex.from_product(columns)
    
    output = pd.DataFrame(index=index, columns=columns)
    output_theta = pd.DataFrame(index=index, columns=columns)
    
    # E_Y_t denotes the true value of the object of interest, in the case of these
    # simulations we structured it so that it is exactly equal to 0. We also
    # define theta_0 as the true value of theta. 
    E_Y_t = 0
    theta_0 = 1.2
    # Change to the directory where all the results were stored from running
    # simulation.py
    path = os.getcwd() + "\\Simulations\\"

    # Iterate over all combinations of ml method, n, L, and c. Recall that each file
    # is named based on these 4 values, so given a set (ml,n,L,c) uniquely identifies
    # one of the files. 
    
    for group in list(product(ml_set,product(n_set,L_set,c_set))):
        name = dgp+"_multigps_c" + str(group[1][2]) + "_" + group[0] + \
        "_L" + str(group[1][1]) + "_N" + str(group[1][0]) + ".csv"
        print(name)
        file = path + name
        estimates = np.genfromtxt(file, delimiter=',')
        beta_hat = estimates[:,0]
        
        std_error = estimates[:,1]
        theta_hat = estimates[:,2]
        theta_hat_std = estimates[:,3]
        
        beta_hat = beta_hat[~np.isnan(beta_hat)]
        std_error = std_error[~np.isnan(beta_hat)]
        theta_hat = theta_hat[~np.isnan(theta_hat)]
        theta_hat_std = theta_hat_std[~np.isnan(theta_hat_std)]
        
        t_stat = np.abs((beta_hat-E_Y_t)/std_error)
        
        coverage_rate = np.nanmean((t_stat<1.96),axis=0)
        bias = np.nanmean((beta_hat-E_Y_t),axis=0)
        rmse = np.sqrt(np.nanmean((beta_hat-E_Y_t)**2,axis=0))
        
        t_stat_theta = np.abs((theta_hat-theta_0)/theta_hat_std)
        coverage_rate_theta = np.nanmean((t_stat_theta<1.96),axis=0)
        bias_theta = np.nanmean((theta_hat-theta_0),axis=0)
        rmse_theta = np.sqrt(np.nanmean((theta_hat-theta_0)**2,axis=0))
    
        output.loc[group[1],(group[0],'Bias')] = np.round(bias,3)
        output.loc[group[1],(group[0],'RMSE')] = np.round(rmse,3)
        output.loc[group[1],(group[0],'Coverage')] = np.round(coverage_rate,3)
        
        output_theta.loc[group[1],(group[0],'Bias')] = np.round(bias_theta,3)
        output_theta.loc[group[1],(group[0],'RMSE')] = np.round(rmse_theta,3)
        output_theta.loc[group[1],(group[0],'Coverage')] = np.round(coverage_rate_theta,3)

# Define the name of the file to save the results to. The path is unchanged as
# we are saving in the same folder as the other raw results. 
    name = 'table_raw_'+str(dgp)+'.xlsx'
    file = path + name 
    output.to_excel(file,index=True)


    name = 'table_raw_'+str(dgp) + '_theta.xlsx'
    file = path + name 
    output_theta.to_excel(file,index=True) 



# %% We use this section to generate results for sdml and regps for lasso.
dgps = ['dgp2']
c_set = [1.0,1.25,1.5]
#c_set = [1.25]
n_set = [1000,10000]
L_set = [1,5]
#ml_set = ['lasso','grf','knn']
ml_set = ['lasso']
stats = ['Bias','RMSE','Coverage']
#estimators = ['sdml','regps']
#estimators = ['regps','sdml']
estimators = ['regps']

# We initialize the pandas dataframe that will store information in the same
# strucutre as in the paper. We first define the hierarchical row index, and 
# the hierarchical column labels, and then we initialize an empty data frame.
for estimator in estimators:
    for dgp in dgps:
            
        index = [n_set,L_set,c_set]
        index = pd.MultiIndex.from_product(index, names=['n', 'L', 'c'])
        
        columns = [ml_set,stats]
        columns = pd.MultiIndex.from_product(columns)
        
        output = pd.DataFrame(index=index, columns=columns)
        output_theta = pd.DataFrame(index=index, columns=columns)
        
        # E_Y_t denotes the true value of the object of interest, in the case of these
        # simulations we structured it so that it is exactly equal to 0.
        E_Y_t = 0
        theta_0 = 1.2
        # Change to the directory where all the results were stored from running
        # simulation.py
        path = os.getcwd() + "\\Simulations\\"
        #path = os.getcwd() + "\\Simulations\\Extra_Simulations\\"
        # Iterate over all combinations of ml method, n, L, and c. Recall that each file
        # is named based on these 4 values, so given a set (ml,n,L,c) uniquely identifies
        # one of the files. 
        
        for group in list(product(ml_set,product(n_set,L_set,c_set))):
            name = dgp+"_" + str(estimator) + "_c" + str(group[1][2]) + "_" + group[0] + \
            "_L" + str(group[1][1]) + "_N" + str(group[1][0]) + ".csv"
            print(name)
            file = path + name
            estimates = np.genfromtxt(file, delimiter=',')
            beta_hat = estimates[:,0]
            
            std_error = estimates[:,1]
            theta_hat = estimates[:,2]
            theta_hat_std = estimates[:,3]
            
            beta_hat = beta_hat[~np.isnan(beta_hat)]
            std_error = std_error[~np.isnan(beta_hat)]
            theta_hat = theta_hat[~np.isnan(theta_hat)]
            theta_hat_std = theta_hat_std[~np.isnan(theta_hat_std)]
            
            t_stat = np.abs((beta_hat-E_Y_t)/std_error)
            
            coverage_rate = np.nanmean((t_stat<1.96),axis=0)
            bias = np.nanmean((beta_hat-E_Y_t),axis=0)
            rmse = np.sqrt(np.nanmean((beta_hat-E_Y_t)**2,axis=0))
            
            t_stat_theta = np.abs((theta_hat-theta_0)/theta_hat_std)
            coverage_rate_theta = np.nanmean((t_stat_theta<1.96),axis=0)
            bias_theta = np.nanmean((theta_hat-theta_0),axis=0)
            rmse_theta = np.sqrt(np.nanmean((theta_hat-theta_0)**2,axis=0))
        
            output.loc[group[1],(group[0],'Bias')] = np.round(bias,3)
            output.loc[group[1],(group[0],'RMSE')] = np.round(rmse,3)
            output.loc[group[1],(group[0],'Coverage')] = np.round(coverage_rate,3)
            
            output_theta.loc[group[1],(group[0],'Bias')] = np.round(bias_theta,3)
            output_theta.loc[group[1],(group[0],'RMSE')] = np.round(rmse_theta,3)
            output_theta.loc[group[1],(group[0],'Coverage')] = np.round(coverage_rate_theta,3)
    
    # Define the name of the file to save the results to. The path is unchanged as
    # we are saving in the same folder as the other raw results. 
        name = 'table_raw_' + str(estimator)+'_'+str(dgp)+'.xlsx'
        file = path + name 
        output.to_excel(file,index=True) 
    
    
        name = 'table_raw_'+str(estimator)+'_'+str(dgp) + '_theta.xlsx'
        file = path + name 
        output_theta.to_excel(file,index=True) 
        
        
        
        
# %%
# Below was used to trace memory allocation. Under previous versions of the code
# we had memory errors that needed to be fixed. We have left the code here commented
# in case we need to use again in future iterations. 
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')


# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)
import pandas as pd
import matplotlib.pyplot as plt
path = os.getcwd() + "\\Simulations\\" 
name = "dgp2" + "_regps" + "_c" + str(1.0) + "_" + str('lasso') + "_L" + str(5) + "_N" +str(10000)+ ".csv"    
file = path +"GPS\\"+ 'gps_'+name         
df = pd.read_csv(file)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
plt.hist(df.values.ravel(), bins='auto', range=(-1, 1)) 
plt.hist(df.iloc[0].values.ravel(), bins='auto') 

path = os.getcwd() + "\\Simulations\\" 
name = "dgp2" + "_multigps_c" + str(1.0) + "_" + str('lasso') + "_L" + str(5) + "_N" +str(1000)+ ".csv"    
file = path +"GPS\\"+ 'gps_'+name         
df = pd.read_csv(file)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
plt.hist(df.values.ravel(), bins='auto')
plt.hist(df.iloc[0].values.ravel(), bins='auto') 


