# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:37:30 2018

@author: Evan

@title: Higgs Particle Detection with Adaboost
"""

# library imports
import csv
import math
import os
import random

import numpy as np
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.grid_search import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.svm import *
from sklearn.tree import *

# change local directory
home = 'c://Users/Evan/.spyder/ML_Project'
os.chdir(home)

# Metrics 

def ams(s, b):
    return math.sqrt(2 * ((s + b + 10) * math.log(1.0 + s/(b + 10)) - s))

def get_ams_score(W, Y, Y_pred):
    s = W * (Y == 1) * (Y_pred == 1)
    b = W * (Y == 0) * (Y_pred == 1)
    s = np.sum(s)
    b = np.sum(b)
    return ams(s, b)

# Analysis

# random seed selection (for reproductibility)    
    seed = 77
    random.seed(seed)
    
# Load training data
    print 'Loading training data...'
    data = np.loadtxt('training.csv', \
            delimiter=',', \
            skiprows=1, \
            converters={32: lambda x:int(x=='s'.encode('utf-8'))})

    X = data[:,1:31]
    Y = data[:,32]
    W = data[:,31]
    
# Load testing data
    print 'Loading testing data...'
    test_data = np.loadtxt('test.csv', \
        delimiter=',', \
        skiprows=1)

    ids_test = test_data[:,0]
    X_test = test_data[:,1:31]
    W = data[:,31]
    
# EDA
    print 'Imputing data...'
    imputer = Imputer(missing_values = -999.0, strategy = 'most_frequent')
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)
    
    print 'Scaling data...'
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
# Ada Boost Model
    cls = AdaBoostClassifier(
            n_estimators = 20,
            learning_rate = 0.70,
            base_estimator = ExtraTreesClassifier(
                n_estimators = 350,
                max_features = 30,
                max_depth = 12,
                min_samples_leaf = 100,
                min_samples_split = 100,
                verbose = 1,
                n_jobs = -1))

# Model fit    
    print 'Modeling data...'
    cls.fit(X, Y, sample_weight = W)
 
# Model prediction
    print 'Predicting data...'
    Y_pred = cls.predict_proba(X)[:,1]
    Y_test_pred = cls.predict_proba(X_test)[:,1]

# Model score    
    print 'Scoring model...'
    cls.score(X, Y, sample_weight = W)

# AMS score    
    print 'AMS Score:',get_ams_score(W, Y, Y_pred)

# Threshold cut    
    signal_threshold = 80
    cut = np.percentile(Y_test_pred, signal_threshold)
    thresholded_Y_pred = Y_pred > cut
    thresholded_Y_test_pred = Y_test_pred > cut
    
# Output Kaggle file
    print 'Generating submission file...'
    ids_probs = np.transpose(np.vstack((ids_test, Y_test_pred)))
    ids_probs = np.array(sorted(ids_probs, key = lambda x: -x[1]))
    ids_probs_ranks = np.hstack((
        ids_probs,
        np.arange(1, ids_probs.shape[0]+1).reshape((ids_probs.shape[0], 1))))

    test_ids_map = {}
    for test_id, prob, rank in ids_probs_ranks:
        test_id = int(test_id)
        rank = int(rank)
        test_ids_map[test_id] = rank

    f = open('ada.submission.out.csv' , 'wb')
    writer = csv.writer(f)
    writer.writerow(['EventId', 'RankOrder', 'Class'])
    for i, pred in enumerate(thresholded_Y_test_pred):
        event_id = int(ids_test[i])
        rank = test_ids_map[ids_test[i]]
        klass = pred and 's' or 'b'
        writer.writerow([event_id, rank, klass])
    f.close()
    
    
