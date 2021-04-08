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

W1 = np.random.uniform(-2,2,1000000)
W2 = np.random.binomial(1,0.5,1000000)
A = np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), 1000000)
Y =np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), 1000000)

B_true = np.mean(expit(0.2-W1+ 2*np.multiply(W1,W2)))
N = int(os.getenv('PBS_ARRAYID'))
random.seed(N+1234)
iters = 2500

estimates = np.zeros((9,iters))
CI = np.zeros((9,iters))
SEs = np.zeros((9,iters))
for i in range(iters):
    W1 = np.random.uniform(-2,2,N)
    W2 = np.random.binomial(1,0.5,N)
    A = 1.0*np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), N)
    Y = 1.0*np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), N)
    W = np.array([W1,W2]).T
    
    W_r = 1.0*W
    W_r[:,1] = W1*W2
    
    ps_log = LogisticRegression().fit(W_r,A).predict_proba(W_r)[:,1]
    ps_f = RF_crossval(W_r, A, W_r)
    ps_w_log = LogisticRegression().fit(W,A).predict_proba(W)[:,1]
    ps_w_f = RF_crossval(W, A, W)
    ps_r = expit(-W1 + 2*np.multiply(W1,W2))
    
    om_log = LogisticRegression().fit(W_r[A==1],Y[A==1]).predict_proba(W_r)[:,1]
    om_f = RF_crossval(W_r[A==1], Y[A==1], W_r)
    om_w_log = LogisticRegression().fit(W[A==1],Y[A==1]).predict_proba(W)[:,1]
    om_w_f = RF_crossval(W[A==1],Y[A==1], W)
    om_r = expit(0.2-W1 + 2*np.multiply(W1,W2))
    
    
    DR_l_f_om, se_l_f_om = DR_estimator(Y, W, A).fit_regression(ps = ps_log,om = om_f).estimate()
    DR_l_f_ps, se_l_f_ps =  DR_estimator(Y, W, A).fit_regression(ps = ps_f,om = om_log).estimate()
    DR_f_r, se_f_r = DR_estimator(Y, W, A).fit_regression(ps = ps_f,om = om_f).estimate()
    DR_om, se_om = DR_estimator(Y, W, A).fit_regression(ps = ps_f,om = om_w_log).estimate()
    DR_ps, se_ps = DR_estimator(Y, W, A).fit_regression(ps = ps_w_log,om = om_f).estimate()
    DR_f_om, se_f_om = DR_estimator(Y, W, A).fit_regression(ps = ps_f,om = om_w_f).estimate()
    DR_f_ps, se_f_ps = DR_estimator(Y, W, A).fit_regression(ps = ps_w_f,om = om_f).estimate()
    DR_f_w, se_f_w = DR_estimator(Y, W, A).fit_regression(ps = ps_w_f,om = om_w_f).estimate()
    DR_r, se_r = DR_estimator(Y, W, A).fit_regression(ps = ps_r,om = om_r).estimate()
    
    estimates[0,i] = DR_l_f_om
    estimates[1,i] = DR_l_f_ps
    estimates[2,i] = DR_f_r
    estimates[3,i] = DR_om
    estimates[4,i] = DR_ps
    estimates[5,i] = DR_f_om
    estimates[6,i] = DR_f_ps
    estimates[7,i] = DR_f_w
    estimates[8,i]  = DR_r

    SEs[0,i] = se_l_f_om
    SEs[1,i] = se_l_f_ps
    SEs[2,i] = se_f_r
    SEs[3,i] = se_om
    SEs[4,i] = se_ps
    SEs[5,i] = se_f_om
    SEs[6,i] = se_f_ps
    SEs[7,i] = se_f_w
    SEs[8,i] = se_r
    
    tval = 1.96
    
    for k in range(9):
        if estimates[k,i] - tval*SEs[k,i] <= B_true <= estimates[k,i] + tval*SEs[k,i]:
            CI[k,i] = 1
        else:
            CI[k,i] = 0
       
np.save('estimates_RF'+str(N)+'.npy',estimates)
np.save('CI_RF'+str(N)+'.npy', CI)
np.save('SEs_RF'+str(N)+'.npy', SEs)