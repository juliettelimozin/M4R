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
N = [200*i for i in range(1,11)]

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
plt.xlabel('Sample size')
plt.ylabel('Bias')
plt.title('Emperial bias convergence of various DR estimators')
plt.show()

plt.figure(figsize = (10,6.6))
plt.title('Coverage of 95% confidence intervals')
for i in range(9):
    plt.plot(N,np.mean(CI[i,:,:], axis = 0), label = labels[i], marker = 'o')
plt.hlines(0.95, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size')
plt.legend()
plt.show()

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, se_ratio[i,:], label = labels[i], marker = 'o')
plt.legend()
plt.hlines(1, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size')
plt.ylabel('SD(estimates of mean)/mean(estimates of standard error)')
plt.title('Ratio of standard deviation to mean standard error estimates')
plt.show()

plt.figure(figsize = (10,6.6))
for i in range(9):
    plt.plot(N, np.multiply(np.sqrt(N),bias[i,:]), label = labels[i], marker = 'o')
plt.legend()
plt.hlines(0, N[0], N[-1], linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size')
plt.ylabel('Bias')
plt.title('sqrt(n)*bias convergence of various DR estimators')
plt.show()