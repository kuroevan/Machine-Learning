# MSDS 7335 - Black Box Machine Learning
# Homework 5
# Evangelos Giakoumakis

# imports
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

# set working directory
home = 'c:\\Users\Evan\.spyder'
#work = 'C://Users/Evangelos.Giakoumaki/.spyder'
os.chdir(home)
print os.getcwd()

# import csv file
pdf = pandas.read_csv('claim.sample.csv', index_col='V1')
print pdf.head()


# display basic info
list(pdf)
pdf.info()

# 1. J-codes are procedure codes that start with the letter 'J'.

# A. Find the number of claim lines that have J-codes.
count = 0
for index, row in pdf.iterrows():
    if row['Procedure.Code'].startswith("J"):
        count += 1
print 'Number of claims with J codes:'
print count
tot_jcodes = count

# B. How much was paid for J-codes to providers for 'in network' claims?
pdfsum = 0
for index, row in pdf.iterrows():
    if row['Procedure.Code'].startswith("J"):
        if row['In.Out.Of.Network'] == "I":
            pdfsum = pdfsum + row['Provider.Payment.Amount']
print 'Sum of money paid for all in network J codes:'
print int(pdfsum)

# C. What are the top five J-codes based on the payment to providers?
top5 = pdf.sort_values('Provider.Payment.Amount', ascending=False)

count = 0
for index, row in top5.iterrows():
    if row['Procedure.Code'].startswith("J"):
        print row['Provider.Payment.Amount']
        count += 1
        if count == 4:
            break

print 'Top 5 J-code claims:'  
top5['Procedure.Code'].head()
        
#2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims 
# for these providers to complete the following exercises.
#
# A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) 
# for each provider versus the number of paid claims.

unique_providers = pdf['Provider.ID'].unique()
print unique_providers

unpaid = np.zeros(26)
paid = np.zeros(26)
for index, row in pdf.iterrows():
    for c in range(26):
        if unique_providers[c] == row['Provider.ID']:
            if row['Provider.Payment.Amount'] == 0:
                unpaid[c] += 1
            else: 
                paid[c] += 1

jid =np.arange(26)
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(jid, unpaid, alpha=0.8, c='red', label='unpaid')
ax.scatter(jid, paid, alpha=0.8, c='green', label='paid')
 
plt.title('Paid-Unpaid Claims scatter plot')
plt.legend(loc=1)
plt.show()
# B. What insights can you suggest from the graph?
res_p = np.zeros(26)
res_up = np.zeros(26)
for c in range(26):
    res_p[c] = (100 * paid[c]) / (unpaid[c] + paid[c])
    res_up[c] = (100 * unpaid[c]) / (unpaid[c] + paid[c])
# FA0001389001, FA0001387001, FA0001411001 (are high outliers)  with too many unpaid claims
# PG0024271001, PG0024271003 suspicious with too few 0 paid or unpaid claims
    
# C. Based on the graph, is the behavior of any of the providers concerning? Explain.
# Providers PG0024271001, PG0024271003 have 0 paid claims and 1 unpaid claim. These numbers seem suspicious for a health provider. 
# Also provider FA0001387001 has a hugely disproportionate number of unpaid/paid claims 97%

# 3. Consider all claim lines with a J-code.

# A. What percentage of J-code claim lines were unpaid?

jcode_pct = (unpaid.sum() *100) / (unpaid.sum() + paid.sum())
print('Unpaid J-code claims percentage:') 
print jcode_pct
# B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
# The modeling approach chosen was Gradient Boosting for regression. GB builds an additive model in a forward stage-wise fashion
# it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative 
# gradient of the given loss function.

from sklearn import ensemble
from sklearn.cross_validation import train_test_split

labels = pdf['Provider.Payment.Amount']
# remoce variables we dont want to use in our model
train = pdf.drop(['Provider.Payment.Amount','Claim.Pre.Prince.Index','Agreement.ID','Denial.Reason.Code','Diagnosis.Code','Procedure.Code','Service.Code','Revenue.Code','Network.ID','Revenue.Code','Place.Of.Service.Code','Line.Of.Business.ID'], axis=1)
# change string variables to numerical categorical
cst = {'H': 1,'M': 2}
train['Claim.Subscriber.Type'] = [cst[item] for item in train['Claim.Subscriber.Type']] 

ct = {'E': 1,'M': 2}
train['Claim.Type'] = [ct[item] for item in train['Claim.Type']] 

ci = {'N': 1,'R': 2, ' ': 0}
train['Capitation.Index'] = [ci[item] for item in train['Capitation.Index']] 

pi = {'F': 1,'N': 2, 'V': 3, 'W': 4, ' ': 0}
train['Pricing.Index'] = [pi[item] for item in train['Pricing.Index']] 

ri = {'F': 1,'N': 2, 'V': 3, ' ': 0}
train['Reference.Index'] = [ri[item] for item in train['Reference.Index']] 

ion = {'I': 1,'O': 2, ' ': 0}
train['In.Out.Of.Network'] = [ion[item] for item in train['In.Out.Of.Network']] 

pd = {'A': 1,'E': 2, ' ': 0}
train['Price.Index'] = [pd[item] for item in train['Price.Index']] 

pid = {'FA0001411002':1, 'FA0001422001':2, 'FA0001389001':3, 'PG0024278003':4, 'FA0001387002':5, 'PG0024278005':6, 'FA0004551001':7, 'PG0024370001':8,
 'PG0024370002':9, 'PG0024370006':10, 'FA0001774002':11, 'FA0001774001':12, 'FA0001389003':13, 'FA0001411003':14, 'PG0043644001':15, 'PG0024271001':16,
 'PG0024271003':16, 'FA0001387001':17, 'FA1000014001':18, 'FA1000014002':19, 'FA1000015001':20, 'FA1000015002':21, 'FA1000016001':22, 'FA1000016002':23,
 'FA0001389004':24, 'FA0001411001':25}
train['Provider.ID'] = [pid[item] for item in train['Provider.ID']] 

#train.dtypes
# split dataset to train 80% and test 20%
x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size = 0.2, random_state=2)
# apply gradient boosting trees
clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=7, min_samples_split=2, learning_rate=0.1, loss='ls')
clf.fit(x_train, y_train)
#clf.score(x_test, y_test)
clf.score(x_train, y_train)
#     C. How accurate is your model at predicting unpaid claims?
# This model yields a score of around 94% which is very good. However it can be improved since alot of parameters were left out, 
# so if added to our model and dataset was properly cleaned we would definitely get better results. 
# D. What data attributes are predominately influencing the rate of non-payment?
# it appears that Provider.ID and In.Out.Of.Network play a major role influencing the rate on non-payment. 
# However as always these results should be taken with a grain of salt.