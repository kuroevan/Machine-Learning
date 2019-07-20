# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:17:10 2018

@author: Evan

@title: Higgs Particle Detection with Gradientboost
"""

# library imports
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
import math
import os
from matplotlib import pyplot as plt

# change local directory
home = 'c://Users/Evan/.spyder/ML_Project'
os.chdir(home)

# Metrics

def AMSScore(s,b): 
    return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))

# Analysis

# random seed selection (for reproductibility) 
np.random.seed(77)
 
# Load training data
print 'Loading training data...'
data_train = np.loadtxt( 'training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
 
# Random number for training/validation splitting
r =np.random.rand(data_train.shape[0])
 
# Place Y(truth), X(data), W(weight), and I(index) into their arrays
print 'Assigning data to numpy arrays...'
# First 80% are training
Y_train = data_train[:,32][r<0.8]
X_train = data_train[:,1:31][r<0.8]
W_train = data_train[:,31][r<0.8]
# Last 20% are validation
Y_valid = data_train[:,32][r>=0.8]
X_valid = data_train[:,1:31][r>=0.8]
W_valid = data_train[:,31][r>=0.8]

# Load the testing data
print 'Loading testing data...'
data_test = np.loadtxt( 'test.csv', delimiter=',', skiprows=1 )
X_test = data_test[:,1:31]
I_test = data_test[:,0]
I_test = list(I_test)
 
# Modeling the GradientBoostingClassifier
print 'Training classifier...'
gbc = GBC(n_estimators=60, max_depth=5,min_samples_leaf=200,max_features=9,verbose=1)

gbc.fit(X_train,Y_train) 

# score model
print 'Scoring model...'
gbc.score(X_train,Y_train) 
 
# Get output from the trained method
print 'Predicting data...'
prob_predict_train = gbc.predict_proba(X_train)[:,1]
prob_predict_valid = gbc.predict_proba(X_valid)[:,1]
 
# Grab top 20%
pcut = np.percentile(prob_predict_train,80)
 
# final signal and background predictions
print 'Signal & Background predictions...'
Yhat_train = prob_predict_train > pcut 
Yhat_valid = prob_predict_valid > pcut
 
# Get the true positives and true negatives & scale weights according to cutoff.
TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.8)
TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.8)
TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.2)
TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.2)
 
# create s and b for the training 
s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )
 
# Calculate the AMS scores
print 'AMS Score: ',AMSScore(s_valid,b_valid)
 
# Create vector of predictions for ranking
print 'Building predictions...'
Predictions_test = gbc.predict_proba(X_test)[:,1]

# Assign labels based on pcut
print 'Assigning labels...'
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)

# Visualize Results

print 'Generating plot...'
Classifier_training_S = gbc.predict_proba(X_train[Y_train>0.5])[:,1].ravel()
Classifier_training_B = gbc.predict_proba(X_train[Y_train<0.5])[:,1].ravel()
Classifier_testing_A = gbc.predict_proba(X_test)[:,1].ravel()
  
c_max = max([Classifier_training_S.max(),Classifier_training_B.max(),Classifier_testing_A.max()])
c_min = min([Classifier_training_S.min(),Classifier_training_B.min(),Classifier_testing_A.min()])
  
# Get histograms of classifiers
Histo_training_S = np.histogram(Classifier_training_S,bins=50,range=(c_min,c_max))
Histo_training_B = np.histogram(Classifier_training_B,bins=50,range=(c_min,c_max))
Histo_testing_A = np.histogram(Classifier_testing_A,bins=50,range=(c_min,c_max))
  
# Get the min/max of histograms
AllHistos= [Histo_training_S,Histo_training_B]
h_max = max([histo[0].max() for histo in AllHistos])*1.2
h_min = 1.0
  
# Get histogram properties (binning, widths, centers)
bin_edges = Histo_training_S[1]
bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
bin_widths = (bin_edges[1:] - bin_edges[:-1])
  
# Create error bars for data (Poisson uncertainty sqrt(N))
ErrorBar_testing_A = np.sqrt(Histo_testing_A[0])
  
# Draw objects
ax1 = plt.subplot(111)
  
# Draw histograms for training data
ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)
ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],bottom=Histo_training_B[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
 
ff = (1.0*(sum(Histo_training_S[0])+sum(Histo_training_B[0])))/(1.0*sum(Histo_testing_A[0]))
 
# # Draw error-bar histograms for testing data
ax1.errorbar(bin_centers, ff*Histo_testing_A[0], yerr=ff*ErrorBar_testing_A, xerr=None, ecolor='black',c='black',fmt='.',label='Test (reweighted)')

# Seperate clasification regions in red and blue
ax1.axvspan(pcut, c_max, color='blue',alpha=0.08)
ax1.axvspan(c_min,pcut, color='red',alpha=0.08)
  
# Adjust the axis boundaries
ax1.axis([c_min, c_max, h_min, h_max])
  
# Add labels and title
plt.title("Higgs Kaggle Signal-Background Separation")
plt.xlabel("Probability Output (Gradient Boosting)")
plt.ylabel("Counts/Bin")
 
# Add legend 
legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
for alabel in legend.get_texts():
            alabel.set_fontsize('small')
  
# Save result to png
plt.savefig("Sklearn_gbc.png")

# Generate Kaggle output file

# Append results to list
results = []
results.append([I_test, Predictions_test, 's'*(Label_test==1.0)+'b'*(Label_test==0.0)])

print 'Organizing prediction results...'
resultlist = []
for x in range(len(I_test)):
    resultlist.append([I_test[x], Predictions_test[x], 's'*(Label_test[x]==1.0)+'b'*(Label_test[x]==0.0)])
 
# Sort result list by probability prediction
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[1]) 
 
# Loop over result list and replace probability prediction with integer ranking
for y in range(len(resultlist)):
    resultlist[y][1]=y+1
 
# Re-sort result list according to index
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[0])
 
# Write the result list data to a csv file
print 'Generating submission file...'
fcsv = open('gbc_submission_out.csv','w')
fcsv.write('EventId,RankOrder,Class\n')
for line in resultlist:
    theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
    fcsv.write(theline) 
fcsv.close()