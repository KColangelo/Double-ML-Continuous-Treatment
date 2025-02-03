# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 00:13:07 2024
This file stores all of the ML model parameters we use for each DGP. Depending
on DGP, sample size, and algorithm the tuning parameters are allowed to vary. 
This is then imported into simulation.py. Tuning parameters were chosen roughly
using cross validation.

@author: Kyle
"""
from sklearn import linear_model
import Supplement
tuned_models = {
    'dgp2':{
        'lasso': {
                      '1000':{'gamma':linear_model.Lasso(alpha=0.00534454178,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.00281957,max_iter=5000,tol=0.001) }, 
                      '10000':{'gamma':linear_model.Lasso(alpha=0.00022478146902344708,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.00019551412581822255,max_iter=5000,tol=0.001)}},
        'nn': {
                      '1000':{'gamma':Supplement.NeuralNet1_n1000(k=101,lr=0.01,momentum=0.9,epochs=100,weight_decay=0.05),
                              'ipw':Supplement.NeuralNet2_n1000(k=100,lr=0.01,momentum=0.9,epochs=100,weight_decay=0.3) }, 
                      '10000':{'gamma':Supplement.NeuralNet1_n10000(k=101,lr=0.05,momentum=0.95,epochs=100,weight_decay=0.05),
                              'ipw':Supplement.NeuralNet2_n10000(k=100,lr=0.4,momentum=0.0,epochs=100,weight_decay=0.075) }},
        'knn':{
                      '1000':{'gamma':Supplement.NeuralNet1k_n1000(k=100,lr=0.05,momentum=0.9,epochs=100,weight_decay=0.1),
                              'ipw':Supplement.NeuralNet2_n1000(k=100,lr=0.01,momentum=0.9,epochs=100,weight_decay=0.3) }, 
                      '10000':{'gamma':Supplement.NeuralNet1k_n10000(k=100,lr=0.05,momentum=0.9,epochs=100,weight_decay=0.1),
                              'ipw':Supplement.NeuralNet2_n10000(k=100,lr=0.4,momentum=0.0,epochs=100,weight_decay=0.075) }},
        },
    'dgp2a':{
            'lasso': {'250':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) },
                      '500':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) },
                      '1000':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) }}
            },
    'dgp2b':{
            'lasso': {'250':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) },
                      '500':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) },
                      '1000':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) }}
            },
    'dgp4a':{
            'lasso': {'250':{'gamma':linear_model.Lasso(alpha=0.03648429616241389,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.11573130145261468,max_iter=5000,tol=0.001) },
                      '500':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) },
                      '1000':{'gamma':linear_model.Lasso(alpha=0.05799388335524879,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09594364277243858,max_iter=5000,tol=0.001) }}
            },
    'dgp4b':{
            'lasso': {'250':{'gamma':linear_model.Lasso(alpha=0.06392092587650156,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.1342610685218138,max_iter=5000,tol=0.001) },
                      '500':{'gamma':linear_model.Lasso(alpha=0.05326651811078838,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.09582013134650663,max_iter=5000,tol=0.001) },
                      '1000':{'gamma':linear_model.Lasso(alpha=0.04013478128781652,max_iter=5000,tol=0.001),
                              'ipw':linear_model.Lasso(alpha=0.06448428161354879,max_iter=5000,tol=0.001) }}
            },
    }

basis = {
    'lasso':True,
    'grf':False,
    'rf':False,
    'nn':False,
    'knn':False
    }