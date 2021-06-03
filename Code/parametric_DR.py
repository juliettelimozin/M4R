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

def DR_exact(Y,W,A, n):
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

def DR_wrong_spe(Y, W, A, n):
    W0 = 1.0*W
    ps = LogisticRegression().fit(W0, A).predict_proba(W0)[:, 1]
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

N = [i*200 for i in range(1,11)]

estimates = np.zeros((5,2500,len(N)))
CI = np.zeros((5,2500,len(N)))
SEs = np.zeros((5,2500,len(N)))
for i in range(2500): 
    for j in range(len(N)):
        W1 = np.random.uniform(-2,2,N[j])
        W2 = np.random.binomial(1,0.5,N[j])
        A = np.random.binomial(1, expit(-W1 + 2*np.multiply(W1,W2)), N[j])
        Y = np.random.binomial(1, expit(0.2*A-W1 + 2*np.multiply(W1,W2)), N[j])
        W = np.array([W1,W2]).T
        
        DR_para, se_para =  DR_parametric(Y, W, A, N[j])
        DR_r, se_r = DR_exact(Y, W, A, N[j])
        DR_om, se_om = DR_wrong_om(Y, W, A, N[j])
        DR_ps, se_ps = DR_wrong_ps(Y, W, A, N[j])
        DR_wrong, se_wrong = DR_wrong_spe(Y, W, A, N[j])
        
        estimates[0,i,j] = DR_para
        estimates[1,i,j] = DR_r
        estimates[2,i,j] = DR_om
        estimates[3,i,j] = DR_ps
        estimates[4,i,j] = DR_wrong
        SEs[0,i,j] = se_para
        SEs[1,i,j] = se_r
        SEs[2,i,j] = se_om
        SEs[3,i,j] = se_ps
        SEs[4,i,j] = se_wrong
        
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
            
        if DR_wrong - tval*se_wrong <= B_true <= DR_wrong + tval*se_wrong:
            CI[4,i,j] = 1
        else: 
            CI[4,i,j] = 0
       
np.save('estimates.npy',estimates)
np.save('CI.npy', CI)
np.save('SEs.npy', SEs)

estimates = np.load('estimates.npy')
CI = np.load('CI.npy')
SEs = np.load ('SEs.npy')
          
bias_para = np.mean(estimates[0,:,:], axis = 0) - B_true*np.ones(len(N))
bias_right = np.mean(estimates[1,:,:], axis = 0) - B_true*np.ones(len(N))
bias_wrong_om = np.mean(estimates[2,:,:], axis = 0) - B_true*np.ones(len(N))
bias_wrong_ps = np.mean(estimates[3,:,:], axis = 0) - B_true*np.ones(len(N))
bias_wrong = np.mean(estimates[4,:,:], axis = 0) - B_true*np.ones(len(N))

B_true_Mat = B_true*np.ones((2500,len(N)))
sd_para = np.sqrt(np.sum(np.square(estimates[0,:,:]-B_true_Mat), axis = 0)/2499)
sd_r = np.sqrt(np.sum(np.square(estimates[1,:,:]-B_true_Mat), axis = 0)/2499)
sd_wrong_om = np.sqrt(np.sum(np.square(estimates[2,:,:]-B_true_Mat), axis = 0)/2499)
sd_wrong_ps = np.sqrt(np.sum(np.square(estimates[3,:,:]-B_true_Mat), axis = 0)/2499)
sd_wrong = np.sqrt(np.sum(np.square(estimates[4,:,:]-B_true_Mat), axis = 0)/2499)

ratio_para = sd_para/np.mean(SEs[0,:,:], axis = 0)
ratio_r = sd_r/np.mean(SEs[1,:,:], axis = 0)
ratio_wrong_om = sd_wrong_om/np.mean(SEs[2,:,:], axis = 0)
ratio_wrong_ps = sd_wrong_ps/np.mean(SEs[3,:,:], axis = 0)
ratio_wrong = sd_wrong/np.mean(SEs[4,:,:], axis = 0)


plt.figure(figsize = (10,6.6))
plt.plot(N, bias_para, label = 'DR logistic models', marker = 'o')
plt.plot(N,bias_right, label = 'DR exact specification', marker = 'o')
plt.plot(N, bias_wrong_om, label = 'DR mis. outcome model', marker = 'o')
plt.plot(N,bias_wrong_ps, label =  'DR mis. propensity score', marker = 'o')
plt.plot(N,bias_wrong, label =  'DR misspecified', marker = 'o')
plt.legend()
plt.hlines(0, 200, 2000, linestyles = 'dashed')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel('Bias', fontsize = 15)
plt.title('Emperial bias convergence of various DR estimators', fontsize = 15)
plt.savefig('../figures/biaspara.png')
plt.show()

plt.figure(figsize = (10,6.6))
plt.title('Coverage of 95% confidence intervals', fontsize = 15)
plt.plot(N,np.mean(CI[0,:,:], axis = 0), label = 'DR logistic models', marker = 'o')
plt.plot(N,np.mean(CI[1,:,:], axis = 0), label = 'DR exact specification', marker = 'o')
plt.plot(N,np.mean(CI[2,:,:], axis = 0), label = 'DR mis. outcome model',
         marker = 'o')
plt.plot(N,np.mean(CI[3,:,:], axis = 0), label = 'DR mis. propensity score',
         marker = 'o')
plt.plot(N,np.mean(CI[4,:,:], axis = 0), label = 'DR misspecified',
         marker = 'o')
plt.hlines(0.95, 200, 2000, linestyles = 'dashed', label = '95% goal')
plt.xlabel('Sample size', fontsize = 15)
plt.legend()
plt.savefig('../figures/CIpara.png')
plt.show()

plt.figure(figsize = (10,6.6))
plt.plot(N, ratio_para, label = 'DR logistic models', marker = 'o')
plt.plot(N, ratio_r, label = 'DR exact specification', marker = 'o')
plt.plot(N, ratio_wrong_om, label = 'DR mis. outcome model', marker = 'o')
plt.plot(N, ratio_wrong_ps, label = 'DR mis. propensity score', marker = 'o')
plt.plot(N, ratio_wrong, label = 'DR misspecified', marker = 'o')
plt.legend()
plt.hlines(1, 200, 2000, linestyles = 'dashed')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel(r'SD($ \hat \beta )/ \widebar{\widehat{se}_{n}}$', fontsize = 15)
plt.title('Ratio of standard deviation to mean standard error estimates', fontsize = 15)
plt.savefig('../figures/SEpara.png')
plt.show()

plt.figure(figsize = (10,6.6))
plt.plot(N, np.multiply(np.sqrt(N),bias_para), label = 'DR logistic models', marker = 'o')
plt.plot(N,np.multiply(np.sqrt(N),bias_right), label = 'DR exact specification', marker = 'o')
plt.plot(N, np.multiply(np.sqrt(N),bias_wrong_om), label = 'DR mis. outcome model', marker = 'o')
plt.plot(N,np.multiply(np.sqrt(N),bias_wrong_ps), label =  'DR mis. propensity score', marker = 'o')
plt.plot(N,np.multiply(np.sqrt(N),bias_wrong), label =  'DR misspecified', marker = 'o')
plt.legend()
plt.hlines(0, 200, 2000, linestyles = 'dashed')
plt.xlabel('Sample size', fontsize = 15)
plt.ylabel('$\sqrt{N}$Bias', fontsize = 15)
plt.title('$\sqrt{N}$bias convergence of various DR estimators', fontsize = 15)
plt.savefig('../figures/sqrtnpara.png')
plt.show()

labels = ['DR logistic models', 'DR exact specification', 'DR mis. outcome model',
          'DR mis. propensity score', 'DR misspecified']
fig, ax = plt.subplots(2,2,figsize = (10,6.6), sharex= True, sharey= True)
for i in range(2):
    ax[i,0].hist(estimates[i,:,-1],bins = np.linspace(min(estimates[i,:,-1]),
                                               max(estimates[i,:,-1]),
                                               75), alpha = 0.5 )
    ax[i,0].set_title(labels[i])
    ax[i,0].set_xlabel('Estimate value', fontsize = 15)
    ax[i,0].set_ylabel('Count', fontsize = 15)
    ax[i,0].vlines(B_true, 0, 120)
for i in range(2,4):
    ax[i-2,1].hist(estimates[i,:,-1],bins = np.linspace(min(estimates[i,:,-1]),
                                               max(estimates[i,:,-1]),
                                               75), alpha = 0.5)
    ax[i-2,1].set_title(labels[i])
    ax[i-2,1].set_xlabel('Estimate value', fontsize = 15)
    ax[i-2,1].set_ylabel('Count', fontsize = 15)
    if i == 3:
        ax[i-2,1].vlines(B_true, 0, 120, label = 'True mean')
    else:
        ax[i-2,1].vlines(B_true, 0, 120)
fig.legend()
fig.suptitle(r'Histograms of estimates for sample size $N = 2000$', fontsize = 15)
plt.savefig('../figures/histpara.png')
plt.show()
    