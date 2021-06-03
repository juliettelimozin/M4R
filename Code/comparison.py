#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:56:39 2021

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

estimates_MLP = np.zeros((9,iters,len(N)))
CI_MLP = 1.0*estimates_MLP
SEs_MLP = 1.0*CI_MLP

for i in range(len(N)):
    estimates_MLP[:,:,i] = np.load('estimates_MLP_'+ str(N[i]) + '.npy')
    CI_MLP[:,:,i] = np.load('CI_MLP_'+ str(N[i]) + '.npy')
    SEs_MLP[:,:,i] = np.load('SEs_MLP_'+ str(N[i]) + '.npy')
   
bias_MLP = np.zeros((9,len(N)))       
for i in range(9):
    bias_MLP[i,:] = np.mean(estimates_MLP[i,:,:], axis = 0) - B_true*np.ones(len(N))


B_true_Mat = B_true*np.ones((iters,len(N)))
se_ratio_MLP = np.zeros((9,len(N)))
for i in range(9):
    se_ratio_MLP[i,:] = (np.sqrt(np.sum(np.square(estimates_MLP[i,:,:]-B_true_Mat),
                               axis = 0)/(iters-1)))/np.mean(SEs_MLP[i,:,:],
                                                             axis = 0)
estimates_RF = np.zeros((9,iters,len(N)))
CI_RF = 1.0*estimates_RF
SEs_RF = 1.0*CI_RF

for i in range(5):
    estimates_RF[:,:,i] = np.load('estimates_RF'+ str(N[i]) + '.npy')
    CI_RF[:,:,i] = np.load('CI_RF'+ str(N[i]) + '.npy')
    SEs_RF[:,:,i] = np.load('SEs_RF'+ str(N[i]) + '.npy')
   
bias_RF = np.zeros((9,len(N)))       
for i in range(9):
    bias_RF[i,:] = np.mean(estimates_RF[i,:,:], axis = 0) - B_true*np.ones(len(N))


se_ratio_RF = np.zeros((9,len(N)))
for i in range(9):
    se_ratio_RF[i,:] = (np.sqrt(np.sum(np.square(estimates_RF[i,:,:]-B_true_Mat),
                               axis = 0)/(iters-1)))/np.mean(SEs_RF[i,:,:],
                                                             axis = 0)

#kldehg;qeuhrg
labels_RF = ['DR logistic ps, forest om',
          'DR logistic om, forest ps',
          'DR forest',
          'DR wrong logistic om, forest ps',
          'DR wrong loigstic ps, forest om',
          'DR wrong forest om, forest ps',
          'DR wrong forest ps, forest om',
          'DR wrong forest ps & om',
          'DR exact']

labels_MLP = ['DR logistic ps, MLP om',
          'DR logistic om, MLP ps',
          'DR MLP',
          'DR wrong logistic om, MLP ps',
          'DR wrong loigstic ps, MLP om',
          'DR wrong MLP om, MLP ps',
          'DR wrong MLP ps, MLP om',
          'DR wrong MLP ps & om',
          'DR exact']

numbers = [2,5,6,7]
fig, (ax1,ax2) = plt.subplots(1,2,sharey = True, figsize = (15,6.6))

for i in numbers:
    ax1.plot(N, bias_RF[i,:], label = labels_RF[i], marker = i)
for i in numbers:
    ax2.plot(N, bias_MLP[i,:], label = labels_MLP[i], marker = 'o')
ax1.legend(fontsize = 15)
ax2.legend(fontsize = 15)
ax1.hlines(0, N[0], N[-1], linestyles = 'dashed')
ax2.hlines(0, N[0], N[-1], linestyles = 'dashed')
ax1.set_xlabel('Sample size', fontsize = 20)
ax2.set_xlabel('Sample size', fontsize = 20)
ax1.set_ylabel('Bias', fontsize = 20)
ax2.set_ylabel('Bias', fontsize = 20)
fig.suptitle('Emperial bias convergence of RF vs MLP DR estimators', fontsize = 20)
plt.savefig('../figures/biascompare.png')
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2,sharey = True, figsize = (15,6.6))

for i in numbers:
    ax1.plot(N, np.multiply(np.sqrt(N),bias_RF[i,:]), label = labels_RF[i], marker = i)
for i in numbers:
    ax2.plot(N, np.multiply(np.sqrt(N),bias_MLP[i,:]), label = labels_MLP[i], marker = 'o')
ax1.legend(fontsize = 15)
ax2.legend(fontsize = 15)
ax1.hlines(0, N[0], N[-1], linestyles = 'dashed')
ax2.hlines(0, N[0], N[-1], linestyles = 'dashed')
ax1.set_xlabel('Sample size', fontsize = 20)
ax2.set_xlabel('Sample size', fontsize = 20)
ax1.set_ylabel(r'$\sqrt{N}$bias', fontsize = 20)
ax2.set_ylabel(r'$\sqrt{N}$bias', fontsize = 20)
fig.suptitle(r'$\sqrt{N}$bias convergence of RF vs MLP DR estimators', fontsize = 20)
plt.savefig('../figures/sqrtncompare.png')
plt.show()
