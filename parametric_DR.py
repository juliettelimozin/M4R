import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LinearRegression
random.seed(1234)

n = 200
W1 = np.random.uniform(-2,2,n)
W2 = np.random.binomial(1,0.5,n)
A = np.random.binomial(1, expit(-W1 + 2*W1*W2), n)
Y = np.random.binomial(1, expit(0.2*A-W1 + 2*W1*W2), n)
Y_true = np.random.binomial(1, expit(0.2 - W1 + 2*W1*W2), n)
B_true = np.mean(Y_true)

W = np.array([W1, 
              W2]).T

def DR_parametric(Y, W, A, n):
    ps = LogisticRegression().fit(W, A).predict_proba(W)[:, 1]
    o_r = LinearRegression().fit(W,Y).predict(W)
    return(np.mean(A*Y/ps)-np.mean((1-A/ps)*o_r))

def DR_right(Y,W,A, n):
    ps = expit(-W[:,0] + 2*np.multiply(W[:,0],W[:,1]))
    o_r = np.multiply(expit(-W[:,0] + 2*np.multiply(W[:,0],W[:,1])),
                      (expit(0.2-W[:,0]+ 2*np.multiply(W[:,0],W[:,1]))
                       + np.ones(A.shape[0]) 
                       - expit(-W[:,0] + 2*np.multiply(W[:,0] ,W[:,1]))))
    return(np.mean(A*Y/ps)-np.mean((1-A/ps)*o_r))

def DR_wrong_om(Y, W, A, n):
    ps = expit(-W[:,0] + 2*np.multiply(W[:,0],W[:,1]))
    o_r = np.multiply(expit(-W[:,0] + 2*W[:,1]),
                      (expit(0.2-W[:,0]+ 2*W[:,1])
                       + np.ones(A.shape[0]) 
                       - expit(-W[:,0] + 2*W[:,1])))
    return(np.mean(A*Y/ps)-np.mean((1-A/ps)*o_r))

def DR_wrong_ps(Y, W, A, n):
    ps = expit(-W[:,0] + 2*W[:,1])
    o_r = np.multiply(expit(-W[:,0] + 2*np.multiply(W[:,0],W[:,1])),
                      (expit(0.2-W[:,0]+ 2*np.multiply(W[:,0],W[:,1]))
                       + np.ones(A.shape[0]) 
                       - expit(-W[:,0] + 2*np.multiply(W[:,0] ,W[:,1]))))
    return(np.mean(A*Y/ps)-np.mean((1-A/ps)*o_r))

W1 = np.random.uniform(-2,2,10000)
W2 = np.random.binomial(1,0.5,10000)
A = np.random.binomial(1, expit(-W1 + 2*W1*W2), 10000)
Y_true = np.random.binomial(1, expit(0.2 - W1 + 2*W1*W2),10000)
B_true = np.mean(Y_true)

N = [250, 500, 750, 1000,2000, 3000, 4000, 5000, 6000, 7000, 8000]
bias = np.zeros((4, len(N)))
CI = np.zeros((4, len(N)))
for i in range(len(N)):
    for j in range(5000):
        W1 = np.random.uniform(-2,2,N[i])
        W2 = np.random.binomial(1,0.5,N[i])
        A = np.random.binomial(1, expit(-W1 + 2*W1*W2), N[i])
        Y = np.random.binomial(1, expit(0.2*A-W1 + 2*W1*W2), N[i])
        W = np.array([W1, 
                  W2]).T
        bias[0,i] = bias[0,i] + DR_parametric(Y, W, A, N[i])/5000
        bias[1,i] = bias[1,i] + DR_right(Y, W, A, N[i])/5000
        bias[2,i] = bias[2,i] + DR_wrong_om(Y, W, A, N[i])/5000
        bias[3,i] = bias[3,i] + DR_wrong_ps(Y, W, A, N[i])/5000
        CI[0,i] = CI[0,i] + 2*1.96*np.sqrt(DR_parametric(Y, W, A, N[i])*(1-DR_parametric(Y, W, A, N[i]))/n)/5000
        CI[1,i] = CI[1,i] + 2*1.96*np.sqrt(DR_right(Y, W, A, N[i])*(1-DR_right(Y, W, A, N[i]))/n)/5000
        CI[2,i] = CI[2,i] + 2*1.96*np.sqrt(DR_wrong_om(Y, W, A, N[i])*(1-DR_wrong_om(Y, W, A, N[i]))/n)/5000
        CI[3,i] = CI[3,i] + 2*1.96*np.sqrt(DR_wrong_ps(Y, W, A, N[i])*(1-DR_wrong_ps(Y, W, A, N[i]))/n)/5000
        
    bias[:,i] += -B_true*np.ones(4).T

np.save('bias.npy',bias)

bias = np.load('bias.npy')

plt.figure()
plt.plot(N, bias[0,:], label = 'DRE')
plt.title('Dr parametric')
plt.figure()

plt.plot(N, bias[1,:], label = 'DR wrong propensity score')
plt.figure()

plt.plot(N, bias[2,:], label = 'DR wrong outcome model')
plt.legend()
plt.show()












