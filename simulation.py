"""
Created on Fri Mar 27 22:12:26 2020
Last update Sunday Jan 27 11:02 am 2025

This file is for running the main simulation results. Estimates and standard
errors for each simulation are saved in multiple files. We define  a function
"simulate" which does the heavy lifting, simulating for the specified data 
generating processes, ml algorithms, values of N, c, L, and specified number
of replications. "simulate_all" is a function which passes a sequence of parameter
sets to simulate. This was to simplify using different DGPs and estimation procedures
(some simulations we use multigps, some regps, and a number of different DGPSs).
Random forest is typically abbreviated with "rf",neural network is 
typicall abbreviated with "nn", generalized random forest is abbreviated as 
"grf", and the neural network proposed in Colangelo and Lee (2025) is 
abbreviated as "knn", short for "Kernel Neural Network", in the code. The
__main__ function is designed to use the multiprocessing package to parallelize
the simulations. This was done to drastically improve the computation time. It
is coded to automatically use 1 fewer thread as there exist cores on your computer.
If you wish this can be manually changed to however many threads you wish depending
on your specs. The code is designed so that in the event of a crash, you should 
be able to restart without issue and have it continue to append to the files
until the specified number of replications is reached. 

Note that some of the names in the code may not agree with the paper due to the
many iterations we have made. As such, you may see "DGP2" in the code referred
to by a different name in the paper. 

The numerical results of the paper focus on lasso primarily, with some results
using nn and knn. Previous versions of the paper also focused on rf, and grf. 
We left these in the code, but comment out the parts that would have run the
rf and grf simulations. If you choose to run with these algorithms the results
will be saved to the folder "Extra_Simulations" for reference.

Files are created for each combination of machine learning method, DGP, n, L, and c.
Models for random forest and lasso are from sklearn. Neural networks use pytorch.
The neural networks used are defined in /Supplement/models.py, including the 
knn. The data generating processes used is defined in /Supplement/DGP.py.
Estimation is carried out using the estimator defined in /Supplement/estimation.py 
as the class DDMLCT, or the class DDMLCT_NN for knn. An instance of the class 
is initialized with 2 models, the first for the estimation of gamma and the second 
for the estimation of the generalized propensity score. The .fit method takes 
arguments for covariates X, treatment T, outcome Y, bandwidth choice h, 
choice of t to estimate the dose response function at, choice of the number 
of sub-samples for cross-fitting (L), whether or not to use the added basis 
functions, and whether to standardize the data.

While not used in the paper, the generalized random forest implementation uses 
the R package, and utilizes Rpy2 to use the R package in Python. There is a 
known issue in our code withthis implementation in that there is a memory leak. 
After a large number of simulations which estimate the generalized random forest, 
the memory usage increases until the program eventually crashes. We have not 
found a fix for this memory leak issue yet, so please contact us if you figure 
out how to fix it. There is now a grf implementation in Python in econml which was
not available when we first worked on this paper which is likely a better option to
utilize.

If the code is terminated while running, the simulations up until that point 
will be saved, and can be continued by starting the code again.

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
#from sklearn.ensemble import ExtraTreesRegressor
import os
from itertools import product
import gc
#import tracemalloc
import multiprocessing
from filelock import FileLock
import time
#import cProfile, pstats, io
#from pstats import SortKey
import numpy as np
# %%
# This function does the main heavy lifting of the simulations, running
# the specified number of replications (J) for value of t and sets of other
# parameters. 
# Files are saved with a particular naming convention to describe what n,L,c 
# and ml method they correspond to. Example: "dgp2_c0.5_lasso_L1_N500.csv" 
# means this is a file for lasso, with c=0.5, L=1, and n=500, and for DGP2. 
#The numbering of the DGP is not the same as in in the paper, but is left in 
#the code as we had been experimenting with different DGP definitions.
def simulate(J=1, t=0, L_set=[5], c_set=[1.25],n_set=[1000],
             method='multigps',ml_set=['lasso'], dgp=Supplement.dgp2, B=1000):
    # %% The file tuning_parameters.py in Supplement contains all the models and
    # parameters. 
    models = Supplement.tuned_models
    basis = Supplement.basis
    
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
    groups = list(product(ml_set,product(L_set,c_set)))
    for sim in list(product(range(J),n_set)):
        n = sim[1]
        # depending on what dgp is passed as an argument, determines which 
        # dgp we generate data with. DGPs defined in dgp.py
        X, T, Y = dgp(n) 
        for group in groups:
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
                    model = Supplement.NN_DDMLCT(models[dgp.__name__][ml][str(n)]['gamma'],models[dgp.__name__][ml][str(n)]['ipw'])
                    model.fit(X,T,Y,t_list,L=L,h=h,basis=basis[ml],standardize=True)
                    beta = model.beta
                    std_error = model.std_errors
        
                    partial_effect = (beta[2]-beta[1])/eta
                    partial_effect_std = ((np.sqrt(15/6)/h)*std_error[0])
        
                else:
                    model = Supplement.DDMLCT(models[dgp.__name__][ml][str(n)]['gamma'],models[dgp.__name__][ml][str(n)]['ipw'])
                    model.fit(X,T,Y,t_list,L=L,h=h,basis=basis[ml],standardize=True)
                    beta = model.beta
                    std_error = model.std_errors
        
                    partial_effect = (beta[2]-beta[1])/eta
                    partial_effect_std = ((np.sqrt(5/2)/h)*std_error[0])
            else:
                model = Supplement.DDMLCT_gps2(models[dgp.__name__][ml][str(n)]['gamma'],models[dgp.__name__][ml][str(n)]['ipw'])
                model.fit(X,T,Y,t_list,L=L,h=h,basis=basis[ml],standardize=True,sdml=False)
                beta = model.beta
                std_error = model.std_errors
    
                partial_effect = (beta[2]-beta[1])/eta
                partial_effect_std = ((np.sqrt(5/2)/h)*std_error[0])
        
            out = np.column_stack((beta[0],std_error[0],partial_effect,partial_effect_std))
            name = dgp.__name__ + "_"+method + "_c" + str(c) + "_" + str(ml) + "_L" + str(L) + "_N" +str(n)+ ".csv"
            
            # Baseline rf, and grf will not be saved in main folder as they 
            # are considered auxiliary results not used in the paper.
            if ml=='rf' or ml=='grf':
                path = os.getcwd() + "/Simulations/Extra_Simulations/" 
            else:
                path = os.getcwd() + "/Simulations/"
    
            file = path + name
            lock_path = file.replace(".csv","") + ".lock"
            lock = FileLock(lock_path, timeout=1) # Make sure there's no conflict between threads
            with lock:
                with open(file, 'ab') as f, open(file,'rb') as f2:
                    n_rows = sum(1 for _ in f2)
                    # check if we have met the number of replications. This is
                    # only necessary if there is a crash and simulations needed
                    # to be restarted. It ensures we don't run more replications
                    # than we want for any given parameter set. 
                    if n_rows<B: 
                
                        np.savetxt(f,out,delimiter=',', fmt='%f')
                    else:
                        groups.remove(group)
                        continue
            
            file = path +"GPS/"+ 'gps_'+name
            lock_path = file.replace(".csv","") + ".lock"
            lock = FileLock(lock_path, timeout=1)
            with lock:
                with open(file, 'ab') as f, open(file,'rb') as f2:
                    n_rows = sum(1 for _ in f2)
                    if n_rows<B:
                        np.savetxt(f,model.gps[[0]].T,delimiter=',', fmt='%f')
                    else:
                        continue

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
    
    
#This function passes a parameter grid to the simulate function in order to 
#allow us to run simulations for all the DGPs without manually calling the function
#a bunch of times.   
def simulate_all(J):
    sim_set = {
            'dgp2': {'dgp':Supplement.dgp2,
                      'n_set':[1000,10000],
                      'c_set':[0.5,0.75,1.0,1.25,1.5],
                      'L_set':[1,5], 
                      'method_set':['multigps'],
                      'ml_set':['knn']},
            'dgp2a': {'dgp':Supplement.dgp2a,
                      'n_set':[250,500,1000],
                      'c_set':[0.25,0.5,0.75,1.0,1.25,1.5],
                      'L_set':[1,5], 
                      'method_set':['multigps','regps'],
                      'ml_set':['lasso']},
            'dgp2b': {'dgp':Supplement.dgp2b,
                      'n_set':[250,500,1000],
                      'c_set':[0.25,0.5,0.75,1.0,1.25,1.5],
                      'L_set':[1,5], 
                      'method_set':['multigps'],
                      'ml_set':['lasso']},
            'dgp4a': {'dgp':Supplement.dgp4a,
                      'n_set':[250,500,1000],
                      'c_set':[0.25,0.5,0.75,1.0,1.25,1.5],
                      'L_set':[1,5], 
                      'method_set':['multigps','regps'],
                      'ml_set':['lasso']},
            'dgp4b': {'dgp':Supplement.dgp4b,
                      'n_set':[250,500,1000],
                      'c_set':[0.25,0.5,0.75,1.0,1.25,1.5],
                      'L_set':[1,5], 
                      'method_set':['multigps','regps'],
                      'ml_set':['lasso']}
            }
    for sim in sim_set:
        for method in sim_set[sim]['method_set']:
            simulate(J=J, L_set = sim_set[sim]['L_set'],c_set=sim_set[sim]['c_set'],
                     n_set=sim_set[sim]['n_set'],method=method,
                     dgp=sim_set[sim]['dgp'])


# %%
if __name__=='__main__':
    # Here we use multiprocessing to divide up the simulations into parallel
    # processes and increase speed. 
    total_replications = 1000# Number of replications 
    t=0 # Choice of t to estimate at.
    

    n_processes = multiprocessing.cpu_count()-1


    
    # J_list is the list of the number of replications that each thread will run.
    # we make sure that the sum of replications across threads equals the exact
    #amount we desire, in this case 1000. 
    if n_processes>1:
        J_list = np.repeat(np.floor(total_replications/(n_processes-1)),(n_processes-1))
        J_list = np.append(J_list,int(total_replications-np.sum(J_list)))
        J_list = J_list.astype(int)\

    else:
        J_list = [total_replications]


    # These are necessary because otherwise numpy insists on using all cores within
    # each thread which causes everything to freeze up due to trying to use
    # too many computational resources. 
    os.system("taskset -p 0xff %d" % os.getpid())
    os.environ["OMP_NUM_THREADS"] = "1"
    
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    with multiprocessing.Pool(n_processes) as pool:
        pool.map(simulate_all, J_list)
        
    
    # This was used to profile the code and find computational bottlenecks
    # simulate(J=J,t=t,L_set = L_set, c_set = c_set, n_set = n_set,method='regps',ml_set = ml_set)
    
    # cProfile.run("""simulate(J=J,t=t,L_set = L_set, 
    #                       c_set = c_set, n_set = n_set,
    #                       method=\'regps\',ml_set = ml_set)
    #             """, 'stats')
    # p = pstats.Stats("stats")
    # p.sort_stats("tottime").print_stats(10)






