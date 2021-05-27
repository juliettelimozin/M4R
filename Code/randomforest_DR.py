#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:33:28 2021

@author: juliette
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
random.seed(1234)
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from DR_estimation import DR_estimator, RF_crossval
import time
W1 = np.random.uniform(-2,2,1000000)
W2 = np.random.binomial(1,0.5,1000000)
A = np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), 1000000)
Y =np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), 1000000)

B_true = np.mean(expit(0.2-W1+ 2*np.multiply(W1,W2)))

iters = 2500
N = [200*i for i in range(1,6)]

estimates = np.zeros((9,iters,len(N)))
CI = 1.0*estimates
SEs = 1.0*CI

for i in range(5):
    estimates[:,:,i] = np.load('estimates_RF'+ str(N[i]) + '.npy')
    CI[:,:,i] = np.load('CI_RF'+ str(N[i]) + '.npy')
    SEs[:,:,i] = np.load('SEs_RF'+ str(N[i]) + '.npy')
   
bias = np.zeros((9,len(N)))       
for i in range(9):
    bias[i,:] = np.mean(estimates[i,:,:], axis = 0) - B_true*np.ones(len(N))


B_true_Mat = B_true*np.ones((iters,len(N)))
se_ratio = np.zeros((9,len(N)))
for i in range(9):
    se_ratio[i,:] = (np.sqrt(np.sum(np.square(estimates[i,:,:]-B_true_Mat),
                               axis = 0)/(iters-1)))/np.mean(SEs[i,:,:],
                                                             axis = 0)


labels = ['DR logistic ps, forest om',
          'DR logistic om, forest ps',
          'DR forest',
          'DR wrong logistic om, forest ps',
          'DR wrong loigstic ps, forest om',
          'DR wrong forest om, forest ps',
          'DR wrong forest ps, forest om',
          'DR wrong forest ps & om',
          'DR exact']

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, bias[i,:], label = labels[i], marker = 'o')
plt.legend()
plt.hlines(0, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel('Bias', fontsize = 15)
plt.title('Emperial bias convergence of various DR estimators', fontsize = 15)
plt.savefig('../figures/biasRF.png')
plt.show()

plt.figure(figsize = (10,6.6))
plt.title('Coverage of 95% confidence intervals', fontsize = 15)
for i in range(9):
    plt.plot(N,np.mean(CI[i,:,:], axis = 0), label = labels[i], marker = 'o')
plt.hlines(0.95, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize =15)
plt.ylabel('Coverage', fontsize = 15)
plt.savefig('../figures/CIRF.png')
plt.legend()
plt.show()

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, se_ratio[i,:], label = labels[i], marker = 'o')
plt.legend()
plt.hlines(1, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel(r'SD($ \hat \beta )/ \widebar{\widehat{se}_{n}}$', fontsize = 15)
plt.title('Ratio of standard deviation to mean standard error estimates', fontsize = 15)
plt.savefig('../figures/SERF.png')
plt.show()

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, np.multiply(np.sqrt(N),bias[i,:]), label = labels[i], marker = 'o')
plt.legend()
plt.hlines(0, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel('$\sqrt{N}$Bias', fontsize = 15)
plt.title('$\sqrt{N}$bias convergence of various DR estimators', fontsize = 15)
plt.savefig('../figures/sqrtnRF.png')
plt.show()


fig, ax = plt.subplots(2,2,figsize = (10,6.6), sharex= True, sharey= True)
ax[0,0].hist(estimates[2,:,-1],bins = np.linspace(min(estimates[2,:,-1]),
                                           max(estimates[2,:,-1]),
                                           75), alpha = 0.5 )
ax[0,0].set_title(labels[2])
ax[0,0].set_xlabel('Estimate value', fontsize = 15)
ax[0,0].set_ylabel('Count', fontsize = 15)
ax[0,0].vlines(B_true, 0, 120)

ax[0,1].hist(estimates[7,:,-1],bins = np.linspace(min(estimates[7,:,-1]),
                                           max(estimates[7,:,-1]),
                                           75), alpha = 0.5 )
ax[0,1].set_title(labels[7])
ax[0,1].set_xlabel('Estimate value', fontsize = 15)
ax[0,1].set_ylabel('Count', fontsize = 15)
ax[0,1].vlines(B_true, 0, 120)

ax[1,1].hist(estimates[6,:,-1],bins = np.linspace(min(estimates[6,:,-1]),
                                           max(estimates[6,:,-1]),
                                           75), alpha = 0.5 )
ax[1,1].set_title(labels[6])
ax[1,1].set_xlabel('Estimate value', fontsize = 15)
ax[1,1].set_ylabel('Count', fontsize = 15)
ax[1,1].vlines(B_true, 0, 120)

ax[1,0].hist(estimates[5,:,-1],bins = np.linspace(min(estimates[5,:,-1]),
                                           max(estimates[5,:,-1]),
                                           75), alpha = 0.5 )
ax[1,0].set_title(labels[5])
ax[1,0].set_xlabel('Estimate value', fontsize = 15)
ax[1,0].set_ylabel('Count', fontsize = 15)
ax[1,0].vlines(B_true, 0, 120,label = 'True mean')

fig.legend()
fig.suptitle(r'Histograms of estimates for sample size $N = 1000$', fontsize = 15)
plt.savefig('../figures/histRF.png')
plt.show()