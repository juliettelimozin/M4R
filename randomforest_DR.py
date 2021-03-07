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

'''
def RF_crossval(Xtrain, ytrain, Xtest):
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 999)
    
    #Fit it to the train set
    cv=skf.split(Xtrain, ytrain)
    
    # Choose the values of the number of decision trees, their depth and 
    # the maximum number of descriptors (features) randomly chosen at each split that I want to test out with GridSearchCV
    n_estimators = np.arange(100,1000,100)
    max_depth = np.arange(10,30,5)
    max_features = [2]
    
    # I summarise these values into a grid
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
                  max_features = max_features)
    
    # Set GridSearch CV with model to tune as RFC, grid of hyperparameters, cross validation method as
    # stratified K fold, and the scoring method as the  the accuracy of your validation predictions
    gridF = GridSearchCV(RandomForestClassifier(), hyperF, cv = cv, scoring = 'accuracy', verbose = 1, 
                          n_jobs = -1)
    gridF.fit(Xtrain, ytrain)
    output =  gridF.predict_proba(Xtest)[:,1]
    output[np.abs(output) < 0.000001] = 0.000001
    return output


def DR_logit_forest_ps(Y, W, A, n):
    W0 = 1.0*W
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    

    X_train_ps, _, y_train_ps, _ = train_test_split(W0,A)

    
    ps = RF_crossval(X_train_ps,y_train_ps,W0)
    
    o_r = LogisticRegression().fit(W0[A==1,:],Y[A==1]).predict_proba(W0)[:, 1]
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_logit_forest_om(Y, W, A, n):
    W0 = 1.0*W
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    
    X_train_om, _, y_train_om, _ = train_test_split(W0[A==1],Y[A==1])

    
    ps = LogisticRegression().fit(W0, A).predict_proba(W0)[:,1]
    
    o_r = RF_crossval(X_train_om,y_train_om,W0)
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_right(Y,W,A, n):
    ps = expit(-W[:,0] + 2*np.multiply(W[:,0],W[:,1]))
    o_r = expit(0.2-W[:,0]+ 2*np.multiply(W[:,0],W[:,1]))
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_forest_right(Y, W, A, n):
    W0 = 1.0*W
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    

    X_train_ps, _, y_train_ps, _ = train_test_split(W0,A)

    ps = RF_crossval(X_train_ps,y_train_ps,W0)
    
    X_train_om, _, y_train_om, _ = train_test_split(W0[A==1],Y[A==1])
    o_r = RF_crossval(X_train_om,y_train_om,W0)
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_wrong_om(Y, W, A, n):
    W0 = 1.0*W
    o_r = LogisticRegression().fit(W0[A==1,:],Y[A==1]).predict_proba(W0)[:, 1]
    
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    
    X_train_ps, _, y_train_ps, _ = train_test_split(W0,A)

    ps = RF_crossval(X_train_ps,y_train_ps,W0)
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_wrong_ps(Y, W, A, n):
    W0 = 1.0*W
    ps = LogisticRegression().fit(W0, A).predict_proba(W0)[:, 1]
    
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    X_train_om, _, y_train_om, _ = train_test_split(W0[A==1],Y[A==1])
    o_r = RF_crossval(X_train_om,y_train_om,W0)
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_forest_wrong_om(Y, W, A, n):
    W0 = 1.0*W
    X_train_om, _, y_train_om, _ = train_test_split(W0[A==1],Y[A==1])
    o_r = RF_crossval(X_train_om,y_train_om,W0)
    
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    X_train_ps, _, y_train_ps, _ = train_test_split(W0,A)
    
    ps = RF_crossval(X_train_ps,y_train_ps,W0)
    
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_forest_wrong_ps(Y, W, A, n):
    W0 = 1.0*W
    

    X_train_ps, _, y_train_ps, _ = train_test_split(W0,A)

    ps = RF_crossval(X_train_ps,y_train_ps,W0)
    
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
   
    X_train_om, _, y_train_om, _ = train_test_split(W0[A==1],Y[A==1])
    
    o_r = RF_crossval(X_train_om,y_train_om,W0)
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_forest_wrong(Y, W, A, n):
    W0 = 1.0*W
    

    X_train_ps, _, y_train_ps, _ = train_test_split(W0,A)

    ps = RF_crossval(X_train_ps,y_train_ps,W0)
    
   
    X_train_om, _, y_train_om, _ = train_test_split(W0[A==1],Y[A==1])

    o_r = RF_crossval(X_train_om,y_train_om,W0)

    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_estimator(Y,W,A,ps,om,n):
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*om)
    se = np.sqrt(np.sum(np.square(A*(Y-om)*(1/ps)+om - DR*np.ones(n))))/n
    return DR, se
'''

N = list(i*200 for i in range(1,6))
iters = 500

estimates = np.zeros((9,iters,len(N)))
CI = np.zeros((9,iters,len(N)))
SEs = np.zeros((9,iters,len(N)))
t = time.time()
for i in range(iters): 
    print('Iteration ' + str(i))
    for j in range(len(N)):
        W1 = np.random.uniform(-2,2,N[j])
        W2 = np.random.binomial(1,0.5,N[j])
        A = 1.0*np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), N[j])
        Y = 1.0*np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), N[j])
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
        
        estimates[0,i,j] = DR_l_f_om
        estimates[1,i,j] = DR_l_f_ps
        estimates[2,i,j] = DR_f_r
        estimates[3,i,j] = DR_om
        estimates[4,i,j] = DR_ps
        estimates[5,i,j] = DR_f_om
        estimates[6,i,j] = DR_f_ps
        estimates[7,i,j] = DR_f_w
        estimates[8,i,j] = DR_r
        
        SEs[0,i,j] = se_l_f_om
        SEs[1,i,j] = se_l_f_ps
        SEs[2,i,j] = se_f_r 
        SEs[3,i,j] = se_om
        SEs[4,i,j] = se_ps
        SEs[5,i,j] = se_f_om
        SEs[6,i,j] = se_f_ps
        SEs[7,i,j] = se_f_w
        SEs[8,i,j] = se_r
        
        tval = 1.96
        
        for k in range(9):
            if estimates[k,i,j] - tval*SEs[k,i,j] <= B_true <= estimates[k,i,j] + tval*SEs[k,i,j]:
                CI[k,i,j] = 1
            else: 
                CI[k,i,j] = 0
print('Computation timne: ' + str(time.time() - t))
       
np.save('estimates_RF.npy',estimates)
np.save('CI_RF.npy', CI)
np.save('SEs_RF.npy', SEs)

estimates = np.load('estimates_RF.npy')
CI = np.load('CI_RF.npy')
SEs = np.load ('SEs_RF.npy')
   
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

