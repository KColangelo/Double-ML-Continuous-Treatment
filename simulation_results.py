# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:21:06 2019
Last update Monday Jan 31 11:21 am 2025

This file is for taking the raw data files, which contain point estimates
and standard errors for each simulation, and turning them into a nice
table which summarizes the stats we are interested in (bias, RMSE, Coverage rate).
We iterate over all the files from the simulations, compute the coverage rate,
bias and rmse and then input that information into the pandas dataframe
called "output". Output is defined with hierarchical row labels for n,L and c,
and hierarchical column names for the machine learning method, and statistic of
interest (bias,rmse,coverage). The new table is saved in the same folder
as the raw simulation data as "table_raw.xlsx". Additional formatting in excel is 
necessary to reproduce the exact formatting in Colangelo and Lee (2025). 


"""
# %% Import necessary packages
import numpy as np 
import os
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import time

# %% 

c_set = [0.25,0.5,0.75,1.0,1.25,1.5] #values of c to use
n_set = [250,500,1000]#set of n's to use
L_set = [1,5]#set of L's to use
#ml_set = ['lasso','nn','knn'] #ML algorithms to loop over
ml_set = ['lasso'] #ML algorithms to loop over
#the set of ml algorithms needs to be changed depending on 
#specifics, as not all algorithms were used for every single parameter set.
# some simulations used only lasso. others used neural networks as well. 
stats = ['Bias','RMSE','Coverage'] #stats to compute

# We initialize the pandas dataframe that will store information in the same
# strucutre as in the paper. We first define the hierarchical row index, and 
# the hierarchical column labels, and then we initialize an empty data frame.
def gen_results(c_set,n_set,L_set,ml_set,stats, estimator,dgp):
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
        #print(name)
        file = path + name
        estimates = np.genfromtxt(file, delimiter=',')
        beta_hat = estimates[:,0]
        
        std_error = estimates[:,1]
        theta_hat = estimates[:,2]
        theta_hat_std = estimates[:,3]

        
        std_error = std_error[~np.isnan(beta_hat)]
        beta_hat = beta_hat[~np.isnan(beta_hat)]
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
        
    if dgp!='dgp2':
        for group in list(product(ml_set,n_set,L_set)):
            df = pd.DataFrame()
            for c in c_set:
                name = dgp+"_" + str(estimator) + "_c" + str(c) + "_" + group[0] + \
                        "_L" + str(group[2]) + "_N" + str(group[1]) + ".csv"
    
                file = path +"GPS\\"+ 'gps_'+name   
                df2 = pd.read_csv(file, engine='python',header=None)
                df2.replace([np.inf, -np.inf], np.nan, inplace=True)
                df = pd.concat([df,df2])
            df=df.apply(lambda x: pd.to_numeric(x,errors='coerce'))
            df = df[~df.isna()]
            if dgp=='dgp2a' or dgp=='dgp4a':
                my_range = (-0.25, 0.5)
            elif dgp=='dgp2b' or dgp=='dgp4b':
                my_range = (-0.4,1)
            plt.hist(df.values.ravel(), bins=50, range=my_range, histtype='bar',ec='black', color='w') 
            plt.title('n='+str(group[1]))
            #plt.ylim(0,group[1]*len(L_set)*len(c_set)*1000)
            name = "./figures/gps_" +dgp+"_" + str(estimator) + "_" + group[0] + \
            "_L" + str(group[2]) + "_N" + str(group[1]) + ".png"
    
            plt.savefig(name)
            plt.close()



estimators = ['multigps','regps'] #estimation method to use.
dgps = ['dgp2a','dgp2b','dgp4a','dgp4b'] # all DGPs to generate result tables for
for estimator in estimators:
    for dgp in dgps:
        gen_results(c_set,n_set,L_set,ml_set,stats, estimator,dgp)


dgp = 'dgp2'
estimator='multigps'
c_set = [0.5,0.75,1.0,1.25,1.5] #values of c to use
n_set = [1000,10000]#set of n's to use
L_set = [1,5]#set of L's to use
ml_set = ['lasso','nn','knn'] #ML algorithms to loop over
stats = ['Bias','RMSE','Coverage'] #stats to compute
gen_results(c_set,n_set,L_set,ml_set,stats, estimator,dgp)



