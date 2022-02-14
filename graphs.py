# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:25:17 2020
Last update Monday Jan 10 1:13 pm 2022

This file is to be run after both "empirical_application.py" and "partial_effects.py".
The only purpose of this file is to generate all figures used in Colangelo and Lee (2020).
The function "plot_ci" was defined to easily plot the confidence interval for a 
range of t. 

comments on packages used:
    -matplotlib.pyplot is used for all of the plots.
    -PIL.Image is used to combine the individual plots after they have been 
    saved into combined plots for both beta and theta. This could also have been
    accomplished by creating a 1x3 plot with matplotlib.pyplot but we found
    this to be simpler given our desire to also have the individual plots.
    -pandas is used to read the esetimates data and then pass it to the plot_ci function.
    -os is used as in the other files to obtain the current working directory 
    for the purpose of file organization.

"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# We define this function to plot the confidence interval
def plot_ci(x,y,c,title,ylab,xlab,f,ylim=[50,70],size=[4,4]):
    plt.figure(figsize=(size[0],size[1]))
    plt.plot(x, y, 'k-')
    plt.plot(x, y-c, '--')
    plt.plot(x, y+c, '--')
    plt.title(title)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.ylim(ylim[0],ylim[1])
    plt.tight_layout()
    plt.savefig(f)
    
    
    
path = os.getcwd() + "\\Empirical Application\\Figures\\"
if not os.path.exists(path):
    os.makedirs(path)
    

ml_list = ['lasso','rf','nn']
titles = {'lasso':'Lasso',
          'rf':'Generalized Random Forest',
          'nn':'K Neural Network'}


# The first loop iterates over all the estimates for each ml algorithm, where
# the estimates used the estimated optimal bandwidth. The second loop iterates
# over all the files for corresponding the the estimates from the rule-of-thumb
# bandwidth choice. Graphs for the dose-response function and partial effects
# with corresponding confidence intervals are saved individually. The graphs 
# for the rule-of-thumb bandwidth choice are not shown in the paper but are 
# for additional reference.
ylim_beta = [30,60]
ylim_theta = [-0.025,0.045]
for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5_hstar.xlsx'
    file = path + name
    dat = pd.read_excel(file)

    
    # Filenames to save figures for both beta and theta(partial effect)
    path = os.getcwd() + "\\Empirical Application\\Figures\\"
    name = 'beta_' + str(ml) + '_hstar.png'
    f = path + name
    c = 1.96*dat['se']
    title = titles[str(ml)]
    ylab = '% of Employment'
    xlab = 'Hours in Training'
    size=[4,4]
    plot_ci(dat['t'],dat['beta'],c,title,ylab,xlab,f,ylim=ylim_beta,size=size)
    
    
    name = 'theta_' + str(ml) + '_hstar.png'
    f = path + name
    c = dat['se partial effect']
    title = titles[str(ml)]
    ylab = 'Partial Effect'
    xlab = 'Hours in Training'
    size=[4,4]
    plot_ci(dat['t'],dat['partial effect'],c,title,ylab,xlab,f,ylim=ylim_theta,size=size)
    

for ml in ml_list:
    path = os.getcwd() + "\\Empirical Application\\Estimates\\"
    name = 'emp_app_' + str(ml) + '_c3_L5.xlsx'
    file = path + name
    dat = pd.read_excel(file)

    
    # Filenames to save figures for both beta and theta(partial effect)
    path = os.getcwd() + "\\Empirical Application\\Figures\\"
    name = 'beta_' + str(ml) + '.png'
    f = path +  name
    c = 1.96*dat['se']
    title = titles[str(ml)]
    ylab = '% of Employment'
    xlab = 'Hours in Training'
    size=[4,4]
    plot_ci(dat['t'],dat['beta'],c,title,ylab,xlab,f,ylim=ylim_beta,size=size)
    
    
    
    path = os.getcwd() + "\\Empirical Application\\Figures\\"
    name = 'theta_' + str(ml) + '.png'
    f = path + name
    c = dat['se partial effect']
    title = titles[str(ml)]
    ylab = 'Partial Effect'
    xlab = 'Hours in Training'
    size=[4,4]
    plot_ci(dat['t'],dat['partial effect'],c,title,ylab,xlab,f,ylim=ylim_theta,size=size)
    


# After the individual figures are saved, we used Pillow to combine them together
# so that we have one figure for the 3 methods and their average dose-response function
# and one figure for the the 3 methods and their partial effects estiamtes. 
path = os.getcwd() + '\\Empirical Application\\Figures\\'
image_list = ['beta','theta']
for name in image_list:
    
    images = [Image.open(x) for x in [(path + name + '_lasso_hstar.png'), 
                                      (path + name + '_rf_hstar.png'), 
                                      (path + name + '_nn_hstar.png')]]
    
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    
    new_im.save((path + name + '.png'))

path = os.getcwd() + '\\Empirical Application\\Figures\\'
image_list = ['beta','theta']
for name in image_list:
    
    images = [Image.open(x) for x in [(path + name + '_lasso.png'), 
                                      (path + name + '_rf.png'), 
                                      (path + name + '_nn.png')]]
    
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    
    new_im.save((path + name + '_rot.png'))
    
    
# Here we generate a histogram of the GPS for neural network for t=400. This is
# not included in the paper but was used for our own investigations.
path = os.getcwd() + '\\Empirical Application\\Estimates\\GPS\\'
name = 'GPS_nn_hstar.xlsx'
file = path + name
gps = pd.read_excel(file)
plt.figure()
plt.title('Histogram of GPS for Neural Network (t=400)',fontsize=16)
plt.xlabel('GPS',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.hist(gps[400], bins = 25, histtype='bar',ec='black', color='w')

path = os.getcwd() + '\\Empirical Application\\Figures\\'
name = 'nn_gps_histogram.png'
file = path + name
plt.savefig(file)























