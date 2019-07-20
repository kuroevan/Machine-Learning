# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:37:30 2018

@author: Evangelos.Giakoumaki
"""

# imports
import os
import pandas
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from IPython.display import display

# set working directory
home = 'c:\\Users\Evan\.spyder/ML_Project'
work = 'C://Users/Evangelos.Giakoumaki/.spyder/ML_Project'
os.chdir(work)
print os.getcwd()

# Load the data set
pdf = pandas.read_csv('training.csv')

# EDA
print pdf.info()
print pdf.head()

# check for empty values
pdf.isnull().sum()

# -999 imouted missing value
eda = pdf == -999 
print eda.sum()

# replace with mean value
pdf['DER_mass_MMC'].mean()

pdf['DER_mass_MMC'].plot()

pdf['PRI_jet_leading_pt'].plot()

# set train and test data sets
train = pdf
test = pandas.read_csv('test.csv')

# turn string into categorical data
train['catLabel'] = train.Label.astype('category').cat.codes 
train = train.drop(["Label"], axis=1) 

#train = train.drop(["Weight"], axis=1) 

#test = test.drop(["Weight"], axis=1)
# add missing collumns on test set
test['Weight'] = ""
test['Weight'].astype(np.float)
test['catLabel'] = ""
test['catLabel'].astype('category')

clf = RandomForestClassifier(n_estimators=30)
clf.fit(train, train["catLabel"])

# Make predictions
predictions = clf.predict(test)
probs = clf.predict_proba(test)
display(predictions)

score = clf.score(test, test["Label"])
print("Accuracy: ", score)

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()