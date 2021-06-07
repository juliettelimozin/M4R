#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
random.seed(1234)
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def RF_crossval(Xtrain, ytrain, Xtest,):
    '''
    Cross-validation and fitting for Random Forest.
    Input: 
        - Xtrain: training feature
        - ytrain: training outcome 
        - Xtest: testing/fitting feature
    Output:
        output: trained RF with hyperparameter chosen from cross-valdiation 
                output
    '''
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 999)
    
    #Fit it to the train set
    cv=skf.split(Xtrain, ytrain)
    
    # Choose the values of the number of decision trees, their depth and 
    # the maximum number of descriptors (features) randomly chosen at each 
    #split that I want to test out with GridSearchCV
    n_estimators = np.arange(600,1001,200)
    max_depth = np.arange(10,31,5)
    max_features = np.arange(1,int(Xtrain.shape[1])+1,1)
    
    # I summarise these values into a grid
    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
                  max_features = max_features)
    
    # Set GridSearch CV with model to tune as RF, grid of hyperparameters, 
    #cross validation method as stratified K fold, and the scoring method as 
    #the  the accuracy of your validation predictions
    gridF = GridSearchCV(RandomForestClassifier(), hyperF, cv = cv, 
                         scoring = 'accuracy', verbose = 1, 
                          n_jobs = -1)
    gridF.fit(Xtrain, ytrain)
    print(gridF.best_params_)
    output =  gridF.predict_proba(Xtest)[:,1]
    output[np.logical_and(0<= output, output<0.000001)] = 0.000001
    output[np.logical_and(0> output, output>-0.000001)] = -0.000001
    return output


def MLP_crossval(Xtrain, ytrain, Xtest,):
    '''
    Cross-validation and fitting for MLP.
    Input: 
        - Xtrain: training feature
        - ytrain: training outcome 
        - Xtest: testing/fitting feature
    Output:
        output: trained MLP with hyperparameter chosen from cross-valdiation
                output
    '''
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 999)
    
    #Fit it to the train set
    cv=skf.split(Xtrain, ytrain)
    tuned_params = {'hidden_layer_sizes': [(10,20,30), (50,50), (100,)],
                    'learning_rate':['constant','invscaling', 'adaptive'],
                    'activation': ["logistic", "relu", "Tanh"],
                    'alpha': [10.0 **(-i) for i in range(1,5)]
                    }
    # Set GridSearch CV with model to tune as MLP, grid of hyperparameters, 
    #cross validation method as stratified K fold, and the scoring method as 
    #the  the accuracy of your validation predictions
    gridMLP = GridSearchCV(MLPClassifier(solver = 'sgd', max_iter = 500), 
                           tuned_params, cv = cv, 
                           scoring='accuracy', verbose = 1, n_jobs = -1)
    gridMLP.fit(Xtrain, ytrain)
    output =  gridMLP.predict_proba(Xtest)[:,1]
    output[np.logical_and(0<= output, output<0.000001)] = 0.000001
    output[np.logical_and(0> output, output>-0.000001)] = -0.000001
    return output

def SVM_crossval(Xtrain, ytrain, Xtest,):
    '''
    Cross-validation and fitting for SVM.
    Input: 
        - Xtrain: training feature
        - ytrain: training outcome 
        - Xtest: testing/fitting feature
    Output:
        output: trained SVM with hyperparameter chosen from cross-valdiation 
                output
    '''
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 999)
    
    #Fit it to the train set
    cv=skf.split(Xtrain, ytrain)
    tuned_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                     'C': [ 10, 100, 1000]},
                    {'kernel': ['linear'], 
                     'C': [10, 100, 1000]},
                    {'kernel': ['poly'], 
                     'gamma': [1e-3, 1e-4, 1e-5], 
                     'degree': np.arange(1,int(Xtrain.shape[1])+1,1), 
                     'C': [10, 100, 1000]}]
    # Set GridSearch CV with model to tune as SVM, grid of hyperparameters, 
    #cross validation method as stratified K fold, and the scoring method as 
    #the  the accuracy of your validation predictions
    gridSVM = GridSearchCV(SVC(probability = True), tuned_params, cv = cv, 
                           scoring='accuracy', verbose = 1, n_jobs = -1)
    gridSVM.fit(Xtrain, ytrain)
    output =  gridSVM.predict_proba(Xtest)[:,1]
    output[np.logical_and(0<= output, output<0.000001)] = 0.000001
    output[np.logical_and(0> output, output>-0.000001)] = -0.000001
    return output

class DR_estimator:
    ''''
    Class for DR estimation procedure
    '''
    def __init__(self, Y, W, A):
        self.Y = Y
        self.W = W
        self.A = A
        self.size = Y.shape[0]
    
    def fit_regression(self, ps = None, om = None, premade = True,
                       method_ps = 'Logistic',
                       method_om = 'Logistic'):
        '''
        Fitting nuisance parameter models; option to add pre-made models, 
        or construct Logistic, RF or MLP models.
        
        Output: self with nuisance parameter models specified
        '''
        if premade:
            try:
                self.ps = ps
                self.om = om
            except ValueError:
                print('Please define regression models')
        else:
            if method_ps == 'Logistic':
                self.ps = LogisticRegression().fit(self.W,self.A) \
                            .predict_proba(self.W)[:,1]
            elif method_ps == 'Random Forest':
                X_train, _, y_train, _ = train_test_split(self.W,self.A)
                self.ps = RF_crossval(X_train,y_train,self.W)
            elif method_ps == 'MLP':
                X_train, _, y_train, _ = train_test_split(self.W,self.A)
                self.ps = MLP_crossval(X_train,y_train,self.W)
            else: 
                raise ValueError("No regression method")

            if method_om == 'Logistic':
                self.om = LogisticRegression().fit(self.W[self.A==1],
                                                   self.Y[self.A==1])\
                                        .predict_proba(self.W)[:,1]
            elif method_om == 'Random Forest':
                X_train, _, y_train, _ = train_test_split(self.W[self.A==1],
                                                          self.Y[self.A==1])
                self.om = RF_crossval(X_train,y_train,self.W)
            elif method_om == 'MLP':
                X_train, _, y_train, _ = train_test_split(self.W[self.A==1],
                                                          self.Y[self.A==1])
                self.om = MLP_crossval(X_train,y_train,self.W)
            else: 
                raise ValueError("No regression method")
        return self
        
    
    def estimate(self):
        '''
        Construc DR and Lunceford-Davidian SE estimate based on class 
        nuisance parameter models
        
        Output: tuple of DR and SE estimate based on self's nuisance parameter
                models
        '''
        DR = np.mean(self.A*self.Y*(1/self.ps)
                   + (np.ones(self.size)-self.A*(1/self.ps))*self.om)
        se = np.sqrt(np.sum(np.square(self.A*(self.Y-self.om)*(1/self.ps)
                                      +self.om 
                                      - DR*np.ones(self.size))))/self.size
        return DR, se 
