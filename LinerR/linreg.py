import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (x-mu)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def loss_gradient(X, y, B, lmbda):
    """
    This is the gradient of loss for linear regression.
    It can be calculated directly because of convex feature.

    Loss(y, 𝐗, 𝛃) = (y - 𝐗𝛃)ᵀ(y - 𝐗𝛃)
    Gradient of Loss = -𝐗ᵀ(y - 𝐗𝛃), so the func will return this.
    """
    
    return -np.dot(np.transpose(X), y - np.dot(X, B))

def loss_ridge(X, y, B, lmbda):
    # here B has no column of 1s
    yxbeta = y - np.dot(X, B)
    return np.dot(np.transpose(yxbeta), yxbeta) + lmbda*np.dot(np.transpose(B), B)

def loss_gradient_ridge(X, y, B, lmbda):
    return -np.dot(np.transpose(X), y - np.dot(X, B))+lmbda*B


def sigmoid(z):
    return np.exp(z)/(1+np.exp(z))

def log_likelihood(X, y, B,lmbda):
    log_like = 0
    for i in range(len(X)):
        yi = y[i]
        yhat = np.dot(X[i], B)
        log_like -= yi*yhat - np.log(1+np.e**yhat)
    return log_like

def log_likelihood_gradient(X, y, B, lmbda):
    """
    This is the gradient of log loss for logistic regression.
    The log loss itself does not have a closed form min solution, thus, 
    we need to do gradient descent. However, we can calculate the solution
    for the gradient, which is:
    
    Gradient of Loss = -𝐗ᵀ(y - σ(𝐗𝛃)), so the func will return this.
    """
    log_like_g = 0
    yhat = np.dot(X, B)
    log_like_g -= np.transpose(X)@(y-sigmoid(yhat))
    return log_like_g

def minimize(X, y_, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    """
    This function add 1s to 𝛃 (if necessary), and initialize 𝛃 randomly,
    Then do a while loop to perform gradient descent.
    """
    # check X and y dimensions and set n and p to X dimensions
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y_.shape == (n,):
        y = y_.reshape(n, 1)
        print(f"Your y is of shape ({n},), we create a new one of ({n}, 1) for your model.")
    else:
        y = y_
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    # if we are doing linear or logistic regression, we want to estimate B0 by adding a column of 1s and increase p by 1
    # for Ridge regression we will set addB0 to False and estimate B0 as part of the RidgeRegression621 fit method
    if addB0:
        X0 = np.ones((n,1))
        X = np.hstack((X0, X))
        p += 1

    # initiate a random vector of Bs of size p
    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)

    # start the minimization procedure here
    prev_B = B
    eps = 1e-5 # prevent division by 0
    
    # remember we need to retain the history of the gradients to use as part of our iteration procedure
    # remember to check stopping condition L2-norm of the gradient <= precision
    
    #### WRITE YOUR CODE HERE, MY SOLUTION HAS 8 LINES OF CODE ####
    h = np.zeros((p,1))
    while np.linalg.norm(grade := loss_gradient(X,y,B,lmbda)) > precision and max_iter>0:
        # h += loss_gradient(X,y,B,lmbda)**2
        # h += np.linalg.norm(grade)**2
        h += grade**2
        B = B - eta/((h+eps)**0.5)*grade

        max_iter-=1
    return B

    

class LinearRegression621: # NO MODIFICATION NECESSARY
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression621: # MODIFY THIS ONE
    "Use the above class as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        This can return soft predictions.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.e**(np.dot(X, self.B))/(1+np.e**(np.dot(X, self.B)))

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        proba = self.predict_proba(X)
        pred = []
        for i in range(len(proba)):
            if proba[i] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return np.array(pred)


    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621: # MODIFY THIS ONE
    "Use the above classes as a guide."

    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
    def predict(self, X):
        print(X)
        bbb = self.B
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)
    def fit(self, X, y):
        # Remember here that you need to estimate B0 separately
        self.B = minimize(X, y,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter, addB0=False)
        self.B = np.vstack([np.array([[np.average(y)]]), self.B])