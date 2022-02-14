# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:21:06 2019
Last update Monday Jan 10 12:11 pm 2022

This file is for taking the raw data files, which contain point estimates
and standard deviations for each simulation, and turning them into a nice
table which summarizes the stats we are interested in (bias, RMSE, Coverage rate).
We iterate over all the files from the simulations, compute the coverage rate,
bias and rmse and then input that information into the pandas dataframe
called "output". Output is defined with hierarchical row labels for n,L and c,
and hierarchical column names for the machine learning method, and statistic of
interest (bias,rmse,coverage). The new table is saved in the same folder
as the raw simulation data as "table_raw.xlsx". Additional formatting in excel is 
necessary to reproduce the exact formatting in Colangelo and Lee (2021). the file 
"dgp_table.xlsx" contains the exact formatting used in the paper. Simply copy 
values from "table_raw.xlsx" into "dgp_table.xslx" to get the exact excel formatted 
table we use in the text. After getting the results in "dgp_table.xlsx" we use the 
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

import numpy as np 
import os
import pandas as pd
from itertools import product


c_set = [0.5,0.75,1.0,1.25,1.5]
n_set = [500,1000]
L_set = [1,5]
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
path = os.getcwd() + "\\Simulations\\"

# Iterate over all combinations of ml method, n, L, and c. Recall that each file
# is named based on these 4 values, so given a set (ml,n,L,c) uniquely identifies
# one of the files. 
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