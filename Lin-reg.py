## Linear Regression

import numpy as np
import Tkinter

## External file input
def fileinp():
    while True:
        try:
            filename = raw_input("Enter the file name:")
            filedata = np.loadtxt(filename, delimiter = ',')
            break
        except IOError:
            print ("Please enter a valid file name")
    print ("Loading data...")
    total_dataset = np.array(filedata)
    return total_dataset

## Feature Normalization
def featurenormalize():
    mu = np.ones((num_rows, 1))
    mu = np.mean(X, axis = 0)
    sigma = np.zeros((num_rows, 1))
    sigma = np.std(X, axis = 0)
    X_nor = np.zeros((num_rows, 1))
    X_nor = (X - mu)/sigma
    return X_nor

## Cost Computation
def computecost(X1, y, theta):
    h = np.dot(X1, theta)
    error = np.square(h - y)
    error_sqr_sum = np.sum(error)
    su = float(error_sqr_sum)
    J = ((0.5/num_rows)* su)
    return J

## Gradient Descent
def gradientdescent(X1, y, alpha, theta):
    count = 0
    theta_ran = np.zeros((num_cols, 1), float)
    while True:
        theta_ran = theta
        h = np.dot(X1, theta)
        error = h - y
        theta = theta - alpha * (X1.T.dot(X1.dot(theta)-y)/num_rows)
        count += 1
        if count > 100000:
            print ("Training data took too long to converge!")
            break
        if np.all(theta_ran - theta) < 0.001:
            break
    return theta, count    

total_dataset = fileinp()
num_rows, num_cols = total_dataset.shape
print ("Done!")
print ("Size of the training data: "), num_rows, ("X"), num_cols
X = total_dataset[:,0:(num_cols - 1)]
y = total_dataset[:,(num_cols - 1)].reshape(num_rows, 1)
x0 = np.ones((num_rows, 1))
X1 = np.concatenate((x0, featurenormalize()), axis = 1)
X2 = np.concatenate((x0, X), axis = 1)
theta = np.zeros((num_cols, 1))
alpha = 0.005
print computecost(X1, y, theta)
theta, count = gradientdescent(X1, y, alpha, theta)
if count == 100001:
    print ("Check into advanced options to change learning parameters")
print ("Number of iterations needed to converge:"), count
print ("Intercept and Coefficients:")
print theta


