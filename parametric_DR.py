import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
random.seed(1234)
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import t

def DR_parametric(Y, W, A, n):
    W0 = 1.0*W
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    ps = LogisticRegression().fit(W0, A).predict_proba(W0)[:, 1]
    o_r = LogisticRegression().fit(W0[A==1,:],Y[A==1]).predict_proba(W0)[:, 1]
    
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

def DR_wrong_om(Y, W, A, n):
    W0 = 1.0*W
    o_r = LogisticRegression().fit(W0[A==1,:],Y[A==1]).predict_proba(W0)[:, 1]
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    ps = LogisticRegression().fit(W0, A).predict_proba(W0)[:, 1]
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

def DR_wrong_ps(Y, W, A, n):
    W0 = 1.0*W
    ps = LogisticRegression().fit(W0, A).predict_proba(W0)[:, 1]
    W0[:,1] = np.multiply(W0[:,0],W0[:,1])
    o_r = LogisticRegression().fit(W0[A==1,:],Y[A==1]).predict_proba(W0)[:, 1]
    
    DR = np.mean(A*Y*(1/ps)
                   + (np.ones(n)-A*(1/ps))*o_r)
    se = np.sqrt(np.sum(np.square(A*(Y-o_r)*(1/ps)+o_r - DR*np.ones(n))))/n
    return DR, se

W1 = np.random.uniform(-2,2,1000000)
W2 = np.random.binomial(1,0.5,1000000)
A = np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), 1000000)
Y =np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), 1000000)

B_true = np.mean(expit(0.2-W1+ 2*np.multiply(W1,W2)))

N = list(i*200 for i in range(1,11))

estimates = np.zeros((4,2500,len(N)))
CI = np.zeros((4,2500,len(N)))

for i in range(2500): 
    for j in range(len(N)):
        W1 = np.random.uniform(-2,2,N[j])
        W2 = np.random.binomial(1,0.5,N[j])
        A = np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), N[j])
        Y = np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), N[j])
        W = np.array([W1,W2]).T
        
        DR_para, se_para =  DR_parametric(Y, W, A, N[j])
        DR_r, se_r = DR_right(Y, W, A, N[j])
        DR_om, se_om = DR_wrong_om(Y, W, A, N[j])
        DR_ps, se_ps = DR_wrong_ps(Y, W, A, N[j])
        
        estimates[0,i,j] = DR_para
        estimates[1,i,j] = DR_r
        estimates[2,i,j] = DR_om
        estimates[3,i,j] = DR_ps
        
        tval = 1.96
        
        if DR_para - tval*se_para <= B_true <= DR_para + tval*se_para:
            CI[0,i,j] = 1
        else: 
            CI[0,i,j] = 0
            
        if DR_r - tval*se_r <= B_true <= DR_r + tval*se_r:
            CI[1,i,j] = 1
        else: 
            CI[1,i,j] = 0
            
        if DR_om - tval*se_om <= B_true <= DR_om + tval*se_om:
            CI[2,i,j] = 1
        else: 
            CI[2,i,j] = 0
            
        if DR_ps - tval*se_ps <= B_true <= DR_ps + tval*se_ps:
            CI[3,i,j] = 1
        else: 
            CI[3,i,j] = 0
       
np.save('estimates.npy',estimates)
np.save('CI.npy', CI)

estimates = np.load('estimates.npy')

bias_para = np.mean(estimates[0,:,:], axis = 0) - B_true*np.ones(len(N))
bias_right = np.mean(estimates[1,:,:], axis = 0) - B_true*np.ones(len(N))
bias_wrong_om = np.mean(estimates[2,:,:], axis = 0) - B_true*np.ones(len(N))
bias_wrong_ps = np.mean(estimates[3,:,:], axis = 0) - B_true*np.ones(len(N))


plt.figure()
plt.plot(N, bias_para, label = 'DRE logistic')
plt.title('Bias of DR with logistic regressions')
plt.show()

plt.figure()
plt.plot(N,bias_right, label = 'DR right models')
plt.title('Bias of DR correct specification')
plt.show()

plt.figure()
plt.plot(N, bias_wrong_om, label = 'DR wrong outcome model')
plt.title('Bias of DR with wrong outcome model')
plt.show()

plt.figure()
plt.plot(N,bias_wrong_ps, label =  'DR wrong propensity score')
plt.title('Bias of DR with wrong propensity score')
plt.show()

plt.figure()
plt.plot(N, bias_para, label = 'DRE logistic')
plt.plot(N,bias_right, label = 'DR right models')
plt.plot(N, bias_wrong_om, label = 'DR wrong outcome model')
plt.plot(N,bias_wrong_ps, label =  'DR wrong propensity score')
plt.legend()
plt.xlabel('Sample size')
plt.ylabel('Bias')
plt.title('Emperial bias convergence of various DR estimators')
plt.show()

plt.hist(estimates[0,:,-1], bins = 200)
plt.vlines(B_true, 0, 100)
plt.hist(estimates[1,:,-1], bins = 200)


plt.figure()
plt.title('Coverage of 95% confidence intervals')
plt.plot(N,np.mean(CI[0,:,:], axis = 0), label = 'DR logistic')
plt.plot(N,np.mean(CI[1,:,:], axis = 0), label = 'DR right')
plt.plot(N,np.mean(CI[2,:,:], axis = 0), label = 'DR wrong outcome model')
plt.plot(N,np.mean(CI[3,:,:], axis = 0), label = 'DR wrong propensity score')
plt.hlines(0.95, 200, 2000, linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size')
plt.legend()
plt.show()
