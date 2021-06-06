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
W3 = np.random.normal(0,1,1000000)
W4 = np.random.exponential(1,1000000)
A = np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2) - W3 + 2*np.multiply(W3,W4)), 1000000)
Y = np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2) - W3 + 2*np.multiply(W3,W4)), 1000000)

B_true = np.mean(expit(0.2-W1 + 2*np.multiply(W1,W2) - W3 + 2*np.multiply(W3,W4)))
iters = 2500
N = [200,400,1000]

estimates = np.zeros((9,iters,len(N)))
CI = 1.0*estimates
SEs = 1.0*CI

for i in range(len(N)):
    estimates[:,:,i] = np.load('estimates_MLP_moreW_'+ str(N[i]) + '.npy')
    CI[:,:,i] = np.load('CI_MLP_moreW_'+ str(N[i]) + '.npy')
    SEs[:,:,i] = np.load('SEs_MLP_moreW_'+ str(N[i]) + '.npy')
   
bias = np.zeros((9,len(N)))       
for i in range(9):
    bias[i,:] = np.mean(estimates[i,:,:], axis = 0) - B_true*np.ones(len(N))


B_true_Mat = B_true*np.ones((iters,len(N)))
se_ratio = np.zeros((9,len(N)))
for i in range(9):
    se_ratio[i,:] = (np.sqrt(np.sum(np.square(estimates[i,:,:]-B_true_Mat),
                               axis = 0)/(iters-1)))/np.mean(SEs[i,:,:],
                                                             axis = 0)


labels = ['DR logistic ps, MLP om',
          'DR logistic om, MLP ps',
          'DR MLP',
          'DR mis. logistic om, MLP ps',
          'DR mis. logistic ps, MLP om',
          'DR mis. MLP om, MLP ps',
          'DR mis. MLP ps, MLP om',
          'DR mis. MLP ps & om',
          'DR exact spec.']

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, bias[i,:], label = labels[i], marker = 'o')
plt.legend()
plt.hlines(0, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel('Bias', fontsize = 15)
plt.title('Emperial bias convergence of various DR estimators', fontsize = 15)
plt.savefig('../figures/biasMLP_moreW.png')
plt.show()

plt.figure(figsize = (10,6.6))
plt.title('Coverage of 95% confidence intervals', fontsize = 15)
for i in range(9):
    plt.plot(N,np.mean(CI[i,:,:], axis = 0), label = labels[i], marker = 'o')
plt.hlines(0.95, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize =15)
plt.ylabel('Coverage', fontsize = 15)
plt.legend()
plt.savefig('../figures/CIMLP_moreW.png')
plt.show()

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, se_ratio[i,:], label = labels[i], marker = 'o')
plt.legend()
plt.hlines(1, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel(r'SD($ \hat \beta )/ \widebar{\widehat{se}_{n}}$', fontsize = 15)
plt.title('Ratio of standard deviation to mean standard error estimates', fontsize = 15)
plt.savefig('../figures/SEMLP_moreW.png')
plt.show()

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, np.multiply(np.sqrt(N),bias[i,:]), label = labels[i], marker = 'o')
plt.legend()
plt.hlines(0, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel('$\sqrt{N}$Bias', fontsize = 15)
plt.title('$\sqrt{N}$bias convergence of various DR estimators', fontsize = 15)
plt.savefig('../figures/sqrtnMLP_moreW.png')
plt.show()


fig, ax = plt.subplots(2,2,figsize = (10,6.6), sharex= True, sharey= True)
ax[0,0].hist(estimates[2,:,-1],bins = np.linspace(min(estimates[2,:,-1]),
                                           max(estimates[2,:,-1]),
                                           75), alpha = 0.5 )
ax[0,0].set_title(labels[2])
ax[0,0].set_xlabel('Estimate value', fontsize = 15)
ax[0,0].set_ylabel('Count', fontsize = 15)
ax[0,0].vlines(B_true, 0, 300)

ax[0,1].hist(estimates[7,:,-1],bins = np.linspace(min(estimates[7,:,-1]),
                                           max(estimates[7,:,-1]),
                                           75), alpha = 0.5 )
ax[0,1].set_title(labels[7])
ax[0,1].set_xlabel('Estimate value', fontsize = 15)
ax[0,1].set_ylabel('Count', fontsize = 15)
ax[0,1].vlines(B_true, 0, 300)

ax[1,1].hist(estimates[6,:,-1],bins = np.linspace(min(estimates[6,:,-1]),
                                           max(estimates[6,:,-1]),
                                           75), alpha = 0.5 )
ax[1,1].set_title(labels[6])
ax[1,1].set_xlabel('Estimate value', fontsize = 15)
ax[1,1].set_ylabel('Count', fontsize = 15)
ax[1,1].vlines(B_true, 0, 300)

ax[1,0].hist(estimates[5,:,-1],bins = np.linspace(min(estimates[5,:,-1]),
                                           max(estimates[5,:,-1]),
                                           75), alpha = 0.5 )
ax[1,0].set_title(labels[5])
ax[1,0].set_xlabel('Estimate value', fontsize = 15)
ax[1,0].set_ylabel('Count', fontsize = 15)
ax[1,0].vlines(B_true, 0, 300,label = 'True mean')

fig.legend()
fig.suptitle(r'Histograms of estimates for sample size $N = $'+str(N[-1]), fontsize = 15)
plt.savefig('../figures/histMLP_moreW.png')
plt.show()