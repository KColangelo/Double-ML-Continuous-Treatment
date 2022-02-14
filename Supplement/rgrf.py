# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:16:53 2021
Last update Monday Jan 10 1:15 pm 2022

This file uses the rpy2 package to call the generalizes random forest R 
package.

@author: Kyle
"""

import numpy as np
import rpy2
from rpy2.robjects.packages import importr
import gc
# Install packages for later use.
# import rpy2's package module
import rpy2.robjects.packages as rpackages
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('grf')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

import rpy2.robjects as robjects
grf = importr('grf')

from rpy2.robjects import numpy2ri
numpy2ri.activate()

class regression_forest():
    def __init__(self):
        self.f = None
        # self.Rpredict = robjects.r['predict']
        # self.mat = robjects.r['as.matrix']
    def fit(self,X,Y):
        gc.collect()
        Y = Y.reshape(len(Y),1)
        robjects.r.assign('rX',X)
        robjects.r.assign('rY',Y)
        self.f = robjects.r('''
                      f = regression_forest(rX,rY, tune.parameters="all")
                      ''')
        self.tuning = robjects.r('''
                      tuning = f$tunable.params
                      ''')
        #print(self.tuning)
        return self
    def predict(self,X):
        #print(type(self.Rpredict(self.f)))
        #print(1)
        robjects.r.assign('rXp',X)
        yhat = robjects.r('''
                          pred = predict(f,rXp)$predictions
                          ''')
        yhat=np.array(yhat)
        robjects.r('''
                      rm(f)
                      rm(rXp2)
                      rm(rX2)
                      rm(rY2)
                      rm(pred)
                      gc()
                      ''')
        gc.collect()
        #yhat = np.array(self.mat((self.Rpredict(self.f,X))))
        #print(yhat)
        return yhat

class regression_forest2():
    def __init__(self):
        self.f = None
        # self.Rpredict = robjects.r['predict']
        # self.mat = robjects.r['as.matrix']
    def fit(self,X,Y):
        gc.collect()
        Y = Y.reshape(len(Y),1)
        robjects.r.assign('rX2',X)
        robjects.r.assign('rY2',Y)
        self.f = robjects.r('''
                      g = regression_forest(rX2,rY2, tune.parameters="all")
                      ''')
        
        return self
    def predict(self,X):
        #print(type(self.Rpredict(self.f)))
        #print(1)
        robjects.r.assign('rXp2',X)
        yhat = robjects.r('''
                          pred = predict(g,rXp2)$predictions
                          ''')
        yhat=np.array(yhat)
        robjects.r('''
                      rm(f)
                      rm(rXp2)
                      rm(rX2)
                      rm(rY2)
                      rm(pred)
                      gc()
                      ''')
        gc.collect()
        #yhat = np.array(self.mat((self.Rpredict(self.f,X))))
        #print(yhat)
        return yhat       