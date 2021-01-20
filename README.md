# Double Debiased Machine Learning for Continuous Treatments

# Table of Contents
1. [Introduction](#introduction)
2. [How to Replicate the Results](#replication)
3. [Additional Code Files](#additional-files)
4. [Simulation Results Files](#sim-names)
5. [Empirical Application Files](#emp-files)
6. [Packages Used](#packages)
7. [Additional Notes](#notes)

## Introduction <a name="introduction"></a>
This code was used for the simulation and empirical application
results in Colangelo and Lee (2020). Please cite the authors appropriately if you 
use any of this code.

## How to Replicate the Results <a name="replication"></a>

To replicate the results as in Colangelo and Lee (2020):

To replicate the simulation results:<br />
(1) Run simulation.py<br />
(2) Run simulation_results.py

simulation.py runs the simulations and saves all estimates and corresponding standard errors to 
various files. simulation_results.py takes the files and converts them into the statistics we need
and puts them in a table of the form similar to the form used in the paper and saves this to excel.
simulation.py will append to any existing simulation data in the simulations folder. If you want
to start from scratch you must delete the excel files in the simulation folder.

To replicate the empirical application:<br />
(1) run empirical_application.py<br />
(2) run partial_effects.py<br />
(3) run graphs.py

empirical_application.py reads the data and runs the estimation, and the estimates, standard errors,
bandwidth information and gps estimates are saved for each ml method (for both the initial choice of
bandwidth and the estimated optimal bandwidth). partial_effects.py then takes the saved estimates and
computes the partial effects as we did in the paper and resaves each file with the additional information.
graphs.py then generates graphs for all the estimates of beta and the partial effects. The graphs
used in the paper are saved as "beta.png" and "theta.png". 

## Additional Code Files <a name="additional-files"></a>
Additional files are in the folder "CL2020". CL2020 was created as a module to be imported into 
the simulation and empirical application files. Within the folder are 4 files:

-dgp.py defines the data generating process used in the simulations<br />
-estimation.py is where we define the class which describes the main estimator, defined as DDMLCT. 
 After importing CL2020, we can initialize a DDMLCT object by calling CL2020.DDMLCT(model1,model2).<br />
-file_management.py defines a function which we use to help organize the file structure of the output<br />
-models.py defines the neural network models we use in both the simulations and empirical application

## Simulation Results Files <a name"sim-names"></a>
In the simulation folder, we save files with names that denote choice of c,n,L, and ml method.
For example: dgp_c0.5_lasso_L1_N500.csv means this is a file corresponding the DGP in the simulations,
with c=0.5, ml=lasso, L=1, and n=500. After these are compiled into the concise results using 
simulation_results.py, the results are save in table_raw.xlsx. To get the exact formatting as in the
paper, copy the values into dgp_table.xlsx.

## Empirical Application Files <a name="emp-files"></a>
The data used is denoted "emp_app.csv"

In the estimates subfolder there are 7 files. "Summary.xlsx" stores the summary statistcs we dispay
in the paper. Every other file stores the estimates for the empirical application. File names denote
which machine learning method was used, choice of the number of sub-samples for cross fitting (L),
choice of c for the initial bandwidth computation, and whether the estimated optimal bandwidth was used
for the computation. 
For example: emp_app_lasso_c3_L5_hstar.xlsx means this is the estimates for the empirical application
(emp_app) for lasso, with c=3 for the initial rule of thumb bandwidth chocie, and L=5, where the estimates
use the estimated hstar. The estimates for the initial rule of thumb bandwidth choice are not used except
for the purpose of estimating h_star. But the estimates are stored nonetheless.

The sub-folder "GPS" stores the GPS estimates for each method, with and without the use of the estimated
h-star. These estimates are not used in the paper but were used for our own investigations. Again, files
ending in "h_star" denote where the estimated optimal bandwidth was used. Files not ending in "h_star"
are from when the rule of thumb bandwidth is used. 

In the figures sub-folder: Each figure for each ml method is saved separately. 
For example: "beta_lasso.png" stores the figure for the estiamted dose-response function for lasso.
Files ending in "hstar" are generated using the optimal computed bandwidth. Files not ending in 
"hstar" corresond to the initial bandwidth. The combined figures used in the paper are named
"beta.png" for the dose-response function and "theta.png" for the partial effects. We also save the
histogram we show in the paper as histogram.png.

## Packages Used <a name="packages"></a>
Packages and exact versions used when we generated our results:<br />
-Numpy 1.18.1 <br />
-pandas 1.0.3<br />
-Pytorch 1.3.1<br />
-scikit-learn 0.22.1<br />
-pathlib2 2.3.5<br />
-scipy 1.4.1<br />
-matplotlib 3.1.3<br />
-pillow 7.0.0

## Additional Notes <a name="notes"></a>
Additional details are included as comments within each respective file.

