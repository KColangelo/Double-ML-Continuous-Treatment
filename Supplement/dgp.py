# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:05:48 2020
Last update Monday Jan 10 12:19 pm 2022

This file just defines the DGP used for the simulations in Colangelo and Lee (2021).
DGP is the function which generates the data used in the simulations in the paper,
DGP2 was used for our personal experimentation to investigate what happens when
the data is less sparse. 


comments on packages used:
    -numpy is used primarily. to store the data, generate random numbers, and the coefficients
    -scipy.sparse.diags is used to generate the tridiagonal covariance matrix of the X's
    -scipy.stats.norm is used for the cdf function we used to generate T.
@author: Kyle Colangelo
"""

import numpy as np
from scipy.sparse import diags
from scipy.stats import norm

def DGP(N):

    rho=0.5 #correlation between adjacent Xs
    size=100 #number of covariates
    k = np.array([rho*np.ones(size-1),np.ones(size),rho*np.ones(size-1)])
    offset = [-1,0,1]
    sigma = diags(k,offset).toarray()

    d=1
    a=3
    b=0.75

    theta = np.array([(1/(l**2)) for l in list(range(1,(size+1)))])
    epsilon = np.random.normal(0,1,N)
    nu = np.random.normal(0,1,N)
    X = np.random.multivariate_normal(np.zeros(size),sigma,size=[N,])

    T = d*norm.cdf((a*X@theta)) + b*nu - 0.5
    Y = 1.2*T + (T**2) + (T*X[:,0]) + 1.2*(X@theta) + epsilon

    return X, T, Y

def DGP2(N):
    rho=0.5
    size=100
    k = np.array([rho*np.ones(size-1),np.ones(size),rho*np.ones(size-1)])
    offset = [-1,0,1]
    sigma = diags(k,offset).toarray()

    d=1
    a=3
    b=0.75

    theta = np.array([(1/(l**2)) for l in list(range(1,(size+1)))])

    nu = np.random.normal(0,1,N)
    X = np.random.multivariate_normal(np.zeros(size),sigma,size=[N,])
    epsilon = np.random.normal(0,0.5+norm.cdf(X[:,0]),N)

    T = d*norm.cdf((a*X@theta)) + b*nu - 0.5
    Y = 1.2*T + (T**2) + (T*X[:,0]) + 1.2*(X@theta) + epsilon

    return X, T, Y

def DGP2a(N):
    rho=0.5
    size=100
    # k = np.array([rho*np.ones(size-1),np.ones(size),rho*np.ones(size-1)])
    # offset = [-1,0,1]
    # sigma = diags(k,offset).toarray()
    sigma = np.zeros((size,size))
    sigma[np.eye(len(sigma), k=1, dtype=bool)] = rho
    sigma[np.eye(len(sigma), k=0, dtype=bool)] = 1
    sigma[np.eye(len(sigma), k=-1, dtype=bool)] = rho
    #d=4
    d=1
    a=3
    b=.75

    theta = np.array([(1/(l**2)) for l in list(range(1,(size+1)))])

    nu = np.random.normal(0,1,N)
    X = np.random.multivariate_normal(np.zeros(size),sigma,size=[N,])
    epsilon = np.random.normal(0,0.5+norm.cdf(X[:,0]),N)

    #T = d*norm.cdf((a*X@theta)) + b*nu + 1
    T = d*norm.cdf((a*X@theta)) + b*nu + 0.5
    Y = 1.2*T + (T**2) + (T*X[:,0]) + 1.2*(X@theta) + epsilon

    return X, T, Y

def DGP3(N):
    rho=0.5
    size=100
    k = np.array([rho*np.ones(size-1),np.ones(size),rho*np.ones(size-1)])
    offset = [-1,0,1]
    sigma = diags(k,offset).toarray()

    d=1
    a=3
    b=0.75

    theta = np.array([(1/(l**2)) for l in list(range(1,(size+1)))])

    nu = np.random.normal(0,1,N)
    X = np.random.multivariate_normal(np.zeros(size),sigma,size=[N,])

    T = d*norm.cdf((a*X@theta)) + b*nu - 0.5
    Y = np.random.normal(1.2*T + (T**2) + (T*X[:,0]) + 1.2*(X@theta),1,N)

    return X, T, Y
















