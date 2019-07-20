# MSDS 7335 - Black Box Machine Learning
# Homework 1
# Evangelos Giakoumakis

# Load Train and Test datasets from population_accidents.m
import os
os.chdir("C:\Users\Evan\.spyder")
print os.getcwd()

# some stuff to make it easier
import h5py
f = h5py.File('population_accidents.mat','r')
f.keys()
list(f.get('x_train'))
print(list(f.keys()))

with h5py.File('population_accidents.mat', 'r') as file:
    xtrain = list(file['x_train'])
    xtest = list(file['x_test'])
    ytrain = list(file['y_train'])
    ytest = list(file['y_test'])
    
print "xtrain set"
print xtrain
print "ytrain set"
print ytrain
print "xtest set"
print xtest
print "ytest set"
print ytest

# Create linear regression object called f using the  function called fit
# specify a first order polynomial (i.e. linear fit)
from sklearn import linear_model
import numpy as np

xtr = np.array(xtrain, dtype=int)
print xtr.shape
print xtr
ytr = np.array(ytrain, dtype=int)
print ytr.shape
print ytr
xts = np.array(xtest, dtype=int)
print xts.shape
print xts
yts = np.array(ytest, dtype=int)
print yts.shape
print yts

f = linear_model.LinearRegression()

f.fit(xtr, ytr)

print('Coefficients: ', f.coef_) 

# A more comprensive output can be obtained using
print('Variance score: {}'.format(f.score(xtr, ytr))) 

# Create a variable called y_predicted_train that predicts
# the output for the training dataset
y_predicted_train = f.predict(xtr)
print ('Predicted train: ', y_predicted_train)

# Create a variable called y_predicted_test that predicts
# the output for the test dataset
xts2 = xts
xts2 = np.append(xts2, np.zeros(39))
xts2 = np.array(xts2, dtype=int)
print xts
xts2.reshape(1,-1)
xts2 = np.array([xts2])
print xts2 
y_predicted_test = f.predict(xts2)
print ('Predicted test: ' ,y_predicted_test)

# Plot a diagram of the data.
# 1. The x-axis should be the input x-data (the population)
# 2. the y-axis should be the output y-data (the number of accidents)
# 3. add a grid (graph paper)
# 4. add a legend
# 5. label the x and y-axis
# 6. add a title
# 7. Save the result to a png file called linearregression.png
import matplotlib.pyplot as plt

plt.scatter(xtr, ytr,  color='black')
#plt.plot(xtr, ytr, color='blue')
plt.title('Population - Number of Accidents Plot')
plt.xlabel('Population')
plt.ylabel('Number of Accidents')
plt.legend(['observation'])
plt.grid()
fig1 = plt.gcf()
plt.show()

# save the plot to a png file
fig1.savefig('linearregression.png')

# calculate the mean square error (MSE) for the prediction of the training and test points
# create variables called mse_train and mse_test
from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(xtr, y_predicted_test)
print ('MSE train: ' ,mse_train)

mse_test = mean_squared_error(ytr, y_predicted_train)
print ('MSE test: ', mse_test)

# calculate the correlation coefficients for the training called r_train and the test data called r_test
# with the associated linear fits
r_train = np.corrcoef(xtr,ytr)
print ('Corr Coef train: ' , r_train)
r_test = np.corrcoef(xts,yts)
print ('Corr Coef test: ' , r_test)

# Draw a scatter diagram to characterize how good the linear estimates of
# the number of accidents are. This scatter diagram should include:
# 1. The 1:1 perfect fit line (use the plot function)
# to add curves to the same plot check out 'hold on'
# 2. The training data and its associated linear fit (use the scatter function)
# 3. The test data and its associated linear fit (use the scatter function)
# 4. A legend that contains the R value in the label (check out the legend function and num2str)
# 5. Save the result to a png file called linearregression-scatter-m.png (check out the print function)
plt.scatter(xtr, ytr,  color='blue', label='train data')
plt.scatter(xts, yts,  color='green', label='test data')
plt.plot(xts, f.predict(xtr) , color='red' ,linewidth=3)
plt.plot([100,30000000], [1, 3850] , color='red' ,linewidth=3, label='perfect fit')
plt.scatter(r_train, r_test, color='yellow', label='R-val: '+ str(r_test[1]))
plt.title('Population - Number of Accidents Plot')
plt.xlabel('Population')
plt.ylabel('Number of Accidents')
plt.legend(loc='lower right')
plt.hold(True)
plt.grid(True)
#plt.xticks(())
#plt.yticks(())
fig2 = plt.gcf()
plt.show()

fig2.savefig('linearregression-scatter-m.png')
