"""
Created on Fri Mar 27 22:12:26 2020
Last update Sunday Oct 27 11:02 am 2023

This file is for running the main simulation results. Estimates and standard
errors for each simulation are saved in multiple files. We begin by defining the 
parameters used for lasso and generalized random forest, and then create all of the machine
learning models we use. Models ending in '1' are used for the estimation of gamma. 
Models ending in "2" are used for the estimation of the generalized propensity 
score.Random forest is typically abbreviated with "rf",neural network is 
typicall abbreviated with "nn", generalized random forest is abbreviated as 
"grf", and the neural network proposed in Colangelo and Lee (2023) is 
abbreviated as "knn", short for "Kernel Neural Network", in the code.

List of algorithms: ml_list = ['lasso','grf','rf','nn','knn']
Sample sizez: n_list = [1000,10000]
number of sub samples for cross-fitting: L_list = [1,5]
number of repetitions: J=1000
coefficient for rule of thumb bandwidth: c_list = [0.75,1.0,1.25,1.5]

The numerical results of the paper focus on lasso, and nn. Previous
versions of the paper focused on rf, knn, and grf. 
We left these in the code, and the results are saved to a separate folder 
"Extra_Simulations" for our own reference.

Files are created for each combination of machine learning method, n, L, and c.
Models for random forest and lasso are from sklearn. Neural networks use pytorch.
The neural networks used are defined in /Supplement/models.py, including the 
knn. The data generating process used is defined in /Supplement/DGP.py.
Estimation is carried out using the estimator defined in /Supplement/estimation.py 
as the class DDMLCT. An instance of the class is initialized with 2 models, 
the first for the estimation of gamma and the second for the estimation of 
the generalized propensity score. The .fit method takes arguments for covariates X, 
treatment T, outcome Y, bandwidth choice h, choice of t to estimate the dose 
response function at, choice of the number of sub-samples for cross-fitting (L), 
whether or not to use the added basis functions, and whether to standardize the data.

The generalized random forest implementation uses the R package, and utilizes
Rpy2 to use the R package in Python. There is a known issue in our code with
this implementation in that there is a memory leak. After a large number of
simulations which estimate the generalized random forest, the memory usage
increases until the program eventually crashes. We have not found a fix for 
this memory leak issue yet, so please contact us if you figure out how to fix
it. There is now a grf implementation in Python, however we have not updated
our code to use it. At the time we originally tried grf, this package was not
available.

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

from sklearn import linear_model
#from sklearn.ensemble import ExtraTreesRegressor
import os
from itertools import product, repeat
import gc
#import tracemalloc
import multiprocessing
from filelock import FileLock
import time
import cProfile, pstats, io
from pstats import SortKey
import numpy as np

# %%
# The outer loop iterates over every combination of sample size n and replication
# number. We only generate the data with respect to n and replication number,
# and for this reason we had to use 2 loops. The inner loop iterates over ml
# method, choice of c and choice of L. Files are saved with a particular naming
# convention to describe what n,L,c and ml method they correspond to.
# Example: "dgp2_c0.5_lasso_L1_N500.csv" means this is a file for lasso, with
# c=0.5, L=1, and n=500, and for DGP2. The numbering of the DGP is not present
# in the paper, but is left in the code as we had been experimenting with different
# DGP definitions before settling on what we termed "DGP2."
def simulate(J=1, t=0, L_set=[5], c_set=[1.25],n_set=[1000], method='multigps',ml_set=['lasso']):
    # %% Define necessary parameters for ML algorithms. These were tuned via cross
    # fitting. Code related to this is in the file "Tuning Simulation.py". The code
    # in the tuning file is however not "streamlined" and does take some manual
    # tinkering to do the tuning. We use different hyperparameters for each 
    # sample size. 
    args_lasso1_n1000 = {
            'alpha':0.04962103409792727,
            'max_iter':5000,
            'tol':0.001
            }
    
    args_lasso2_n1000 = {
            'alpha':0.07754708123579417,
            'max_iter':5000,
            'tol':0.001
            }
    #0.022303389538363024
    #0.016899722478163216
    args_lasso1_n10000 = {
            'alpha':0.022303389538363024,
            'max_iter':5000,
            'tol':0.001
            }
    
    args_lasso2_n10000 = {
            'alpha':0.016899722478163216,
            'max_iter':5000,
            'tol':0.001
            }
    # Define parameters for random forest. Currently this is commented out as the
    # present version of the paper does not use random forest in the simulation
    # results. 
    # args_rf1 = {
    #         'n_estimators':1000,
    #         'max_depth':None,
    #         'min_samples_leaf':40,
    #         'min_samples_split':40
    #         }
    
    # args_rf2 = {
    #         'n_estimators':1000,
    #         'max_depth':None,
    #         'min_samples_leaf':40,
    #         'min_samples_split':40
    #         }
    # %% Define the models we will be using with the specified parameters
    model_lasso1_n10000 = linear_model.Lasso(**args_lasso1_n10000)
    model_lasso2_n10000 = linear_model.Lasso(**args_lasso2_n10000)
    model_lasso1_n1000 = linear_model.Lasso(**args_lasso1_n1000)
    model_lasso2_n1000 = linear_model.Lasso(**args_lasso2_n1000)
    
    # Define the generalized random forest models. Not currently used in paper so
    # commented out.
    # model_grf1 = Supplement.regression_forest()
    # model_grf2 = Supplement.regression_forest2()
    
    # Define random forest models. Not currently used in paper so commented out.
    # model_rf1 = ExtraTreesRegressor(**args_rf1)
    # model_rf2 = ExtraTreesRegressor(**args_rf2)
    
    # For the neural networks we need to specify the number of covariates.
    # we have 100 X variables and one treatment. The Second stage of the regular
    # neural network takes treatment as an input, hence "101". But the knn does
    # not take treatment as an input, hence why its second stage has "100". The
    # First stage regresses T on X, hence we do not do the knn. Learning rate
    # and related parameters are tuned in "Tuning Simulation.py". 
    model_nn1_n10000 = Supplement.NeuralNet1_n10000(k=101, 
                                      lr=0.05,
                                      momentum=0.95,
                                      epochs=100,
                                      weight_decay=0.05) 
    model_nn2_n10000 = Supplement.NeuralNet2_n10000(k=100, 
                                      lr=0.4,
                                      momentum = 0.0, 
                                      epochs=100,
                                      weight_decay=0.075)
    
    model_nn1_n1000 = Supplement.NeuralNet1_n1000(k=101, 
                                      lr=0.01,
                                      momentum=0.9,
                                      epochs=100,
                                      weight_decay=0.05) 
    
    model_nn2_n1000 = Supplement.NeuralNet2_n1000(k=100, 
                                      lr=0.01,
                                      momentum = 0.9, 
                                      epochs=100,
                                      weight_decay=0.3)
    
    model_knn1_n10000 = Supplement.NeuralNet1k_n10000(k=100, 
                                      lr=0.05,
                                      momentum = 0.9,
                                      epochs=100,
                                      weight_decay=0.1)
    
    model_knn2_n10000 = Supplement.NeuralNet2_n10000(k=100, 
                                      lr=0.4,
                                      momentum = 0.0, 
                                      epochs=100,
                                      weight_decay=0.075)
    
    model_knn1_n1000 = Supplement.NeuralNet1k_n1000(k=100, 
                                      lr=0.01,
                                      momentum=0.9,
                                      epochs=100,
                                      weight_decay=0.05) 
    
    model_knn2_n1000 = Supplement.NeuralNet2_n1000(k=100, 
                                      lr=0.01,
                                      momentum = 0.9, 
                                      epochs=100,
                                      weight_decay=0.3)
    
    
    # I collect the models into a dictionary so that they can be easily iterated over
    # for estimation.
    models = {
            'lasso': {'1000':[model_lasso1_n1000, model_lasso2_n1000], 
                      '10000':[model_lasso1_n10000, model_lasso2_n10000]},
            'nn': {'1000':[model_nn1_n1000, model_nn2_n1000], 
                      '10000':[model_nn1_n10000, model_nn2_n10000]},
            'knn': {'1000':[model_knn1_n1000, model_knn2_n1000], 
                      '10000':[model_knn1_n10000, model_knn2_n10000]}
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




    # These are the directories we store results to. If the directories don't exist
    # when this code is run, then they are created. 
    path = os.getcwd() + "/Simulations/"
    Supplement.make_dirs(path)
    
    path = os.getcwd() + "/Simulations/Extra_Simulations/"
    Supplement.make_dirs(path)
    
    path = os.getcwd() + "/Simulations/GPS/"
    Supplement.make_dirs(path)
    # pr = cProfile.Profile()
    # pr.enable()
    # tracemalloc.start() This was used previously to diagnose memory leak issues. 
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    for sim in list(product(range(J),n_set)):
        n = sim[1]
        # We experimented with multiple DGPs before aligning on using "DGP2". It
        # is not referred to by number in the paper, but we had considered 3 different
        # DGPs, and you can view their definitions in the Supplment.dgp.py file.
        # DPG2 was chosen as it incorporated a more non-linear relationship in the 
        # data while not looking too "pathological."
        X, T, Y = Supplement.DGP2a(n) 
        for group in list(product(ml_set,product(L_set,c_set))):
            c = group[1][1]
            ml = group[0]
            L = group[1][0]
            h = np.std(T)*c*(n**-0.2) # rule of thumb bandwidth. 
            # eta defines how far around t=0 we go to estimate theta.
            eta =  h*(n**(-(1/6)))
            # We need to compute the estimator at three values of t in order to compute
            # both beta and theta at t=0. This is because theta needs to take the difference
            # between betas at either side of t=0. 
            t_list = np.array([0,-(eta/2),(eta/2)]) 
            #start_time = time.time()
            if method=='multigps':
                if ml=='knn':
                    model = Supplement.NN_DDMLCT(models[ml][str(n)][0],models[ml][str(n)][1])
                    model.fit(X,T,Y,t_list,L=L,h=h,basis=basis[ml],standardize=True)
                    beta = model.beta
                    std_error = model.std_errors
        
                    partial_effect = (beta[2]-beta[1])/eta
                    partial_effect_std = ((np.sqrt(15/6)/h)*std_error[0])
        
                else:
                    model = Supplement.DDMLCT(models[ml][str(n)][0],models[ml][str(n)][1])
                    model.fit(X,T,Y,t_list,L=L,h=h,basis=basis[ml],standardize=True)
                    beta = model.beta
                    std_error = model.std_errors
        
                    partial_effect = (beta[2]-beta[1])/eta
                    partial_effect_std = ((np.sqrt(5/2)/h)*std_error[0])
            else:
                t_list = np.array([0,-(eta/2),(eta/2)])
                model = Supplement.DDMLCT_gps2(models[ml][str(n)][0],models[ml][str(n)][1])
                model.fit(X,T,Y,t_list,L=L,h=h,basis=basis[ml],standardize=True,sdml=False)
                beta = model.beta
                std_error = model.std_errors
    
                partial_effect = (beta[2]-beta[1])/eta
                partial_effect_std = ((np.sqrt(5/2)/h)*std_error[0])
        
            out = np.column_stack((beta[0],std_error[0],partial_effect,partial_effect_std))
            name = "dgp2_"+method + "_c" + str(c) + "_" + str(ml) + "_L" + str(L) + "_N" +str(n)+ ".csv"
            
            # Baseline rf and nn will not be saved in main folder as they are not
            # included in the main results.
            if ml=='rf' or ml=='knn' or ml=='grf':
                path = os.getcwd() + "/Simulations/Extra_Simulations/" 
            else:
                path = os.getcwd() + "/Simulations/"
    
            file = path + name
            lock_path = file + ".lock"
            lock = FileLock(lock_path, timeout=1)
            with lock:
                with open(file, 'ab') as f:
                    np.savetxt(f,out,delimiter=',', fmt='%f')
            
            file = path +"GPS/"+ 'gps_'+name
            lock_path = file + ".lock"
            lock = FileLock(lock_path, timeout=1)
            with lock:
                with open(file, 'ab') as f:
                    np.savetxt(f,model.gps[[0]].T,delimiter=',', fmt='%f')
        # This is an attempt to partially fix the memory leak problem but it
        # is not sufficient.
        gc.collect() # This was used as there was a tendency for memory leakage.

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats(30)
    # lock_path = "test.lock"
    # lock = FileLock(lock_path)
    # with lock:
    #     with open('test.txt', 'w+') as f:
    #         f.write(s.getvalue())

# %%
if __name__=='__main__':
    # Iterate over all t and ml algorithms for estimation
    # Define all the sets of parameters that we will create estimates for. That is,
    # define which sets of N, c, L and ML algorithms will be used for our results.
    ml_set = ['lasso']
    #n_set = [1000,10000] # All sample sizes used
    n_set = [1000,10000] # All sample sizes used
    c_set = [1.0,1.25,1.5] # All c's used for bandwidth choice
    #c_set = [1.25] # All c's used for bandwidth choice
    L_set = [1,5] # All numbers of folds used for cross-fitting
    total_replications = 1000 # Number of replications
    t=0 # Choice of t to estimate at.
    
    #start = time.time()
    n_processes = multiprocessing.cpu_count()-1
    
    # params = tuple(repeat((J, 0, L_set, c_set, n_set, 'multigps',ml_set),n_processes))
    # with multiprocessing.Pool(n_processes) as pool:
    #     pool.starmap(simulate, params)
    
    J_list = np.repeat(np.floor(total_replications/(n_processes-1)),(n_processes-1))
    J_list = np.append(J_list,total_replications-np.sum(J_list))

    params = tuple((x, 0, L_set, c_set, n_set, 'regps',ml_set) for x in J_list)
    
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap(simulate, params)
        
    # simulate(J=J,t=t,L_set = L_set, c_set = c_set, n_set = n_set,method='multigps',ml_set = ml_set)

    #end = time.time()
    #print(end-start)
    # Process stuff to be profiled

    
    #print(s.getvalue())
    # simulate(J=J,t=t,L_set = L_set, c_set = c_set, n_set = n_set,method='regps',ml_set = ml_set)
    
    # cProfile.run("""simulate(J=J,t=t,L_set = L_set, 
    #                       c_set = c_set, n_set = n_set,
    #                       method=\'regps\',ml_set = ml_set)
    #             """, 'stats')
    # p = pstats.Stats("stats")
    # p.sort_stats("tottime").print_stats(10)






