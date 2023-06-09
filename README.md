> **Warning**
>
> For the minimization, we are using Adagradient descent without mini-batch, meaning that **our loss**
> **function will increase** with the increase of data volumn. So **please select the comparable learning**
> **rate eta**, I recommend to try from eta=1. Do not use the default cause it's barely training.

> **Warning**
>
> For the ridge regression, you must center the prediter variables for the package will calculate coefficients
> without intercept and then add y_mean as intercept. You can find more about this in **GetStarted.ipynb**


# Linear Regression
|            |                                                             |
|------------|-------------------------------------------------------------|
| Attributes | eta: the learning rate η of Adagradient descent default=0.00001            |
|            | lambda: the regularization rate of L2(Ridge) regularization, here it's not used for we didn't use regularization here |
|            | max_iter: max iter rounds for Adagradient descent, default=1000           |
|            | B: 𝛃, coefs of model, β₀, β₁, ..., βₚ                       |
| Methods    | fit(X, y) Fit linear model                                  |
|            | predict(X) Predict linear model                             |



## Example
```python
In [1]: from LinerR.linreg import *
   ...: from sklearn.datasets import load_wine, load_iris
   ...: from sklearn.model_selection import train_test_split
   ...: import numpy as np
   ...: import pandas as pd
   ...: from sklearn.metrics import r2_score

In [2]: X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
   ...: y = np.dot(X, np.array([1, 2])) + 3

In [3]: # create model
   ...: model = LinearRegression621(eta=1, max_iter=60000)
   ...: # fit model
   ...: model.fit(X, y)
Your y is of shape (4,), we create a new one of (4, 1) for your model.

In [4]: # get predictions
   ...: y_pred = model.predict(X)

In [5]: y_pred
Out[5]: 
array([[ 6.],
       [ 8.],
       [ 9.],
       [11.]])

In [6]: model.B
Out[6]: 
array([[3.],
       [1.],
       [2.]])
```
# Logistic Regression
|            |                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------|
| Attributes | eta: the learning rate η of Adagradient descent, default=0.00001                                                      |
|            | lambda: the regularization rate of L2(Ridge) regularization, here it's not used for we didn't use regularization here |
|            | max_iter: max iter rounds for Adagradient descent, default=1000                                                       |
|            | B: 𝛃, coefs of model, β₀, β₁, ..., βₚ                                                                                 |
| Methods    | fit(X, y) Fit logistic model                                                                                          |
|            | predict_proba(X) Predict logistic model, return soft predictions                                                      |
|            | predict(X) Predict logistic model                                                                                     |


## Example
```python
from LinerR.linreg import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y=True)
X, y = X[y<=1], y[y<=1] # here we only use 2 labels

"""
Here bacuase we are not using mini-batch, so it doesn't matter whether
or not we shuffle the data, all the data will take part in the gradient
calculation!!!"""
# from sklearn.utils import shuffle
# X, y = shuffle(X, y, random_state=13)

"""
It's recommended to normalize the data, and it's recommended to only normalize
according to the μ and σ of the training set. However, for small dataset, it's
okay if you don't."""
# normalize(X)

model = LogisticRegression621(max_iter=10_000, eta=1)
model.fit(X, y)
ans = (np.sum(model.predict(X)==y))
print("correct rate: {}".format(ans/len(y)))
del model

Out:
Your y is of shape (100,), we create a new one of (100, 1) for your model.
correct rate: 1.0
```

# Ridge Linear Regression
|            |                                                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------------------------|
| Attributes | eta: the learning rate η of Adagradient descent, default=0.00001                                                      |
|            | lambda: the regularization rate of L2(Ridge) regularization |
|            | max_iter: max iter rounds for Adagradient descent, default=1000                                                       |
|            | B: 𝛃, coefs of model, β₀, β₁, ..., βₚ                                                                                 |
| Methods    | fit(X, y) Fit logistic model   |
|            | predict(X) Predict logistic model  |

## Example
```python
def synthetic_data():
    """
    X:[0, 10]
    y = X + standard normal distribution, [0, 10]
    """
    n = 1000
    df = pd.DataFrame()
    df['x'] = np.linspace(0, 10, num=n)
    df['y'] = df['x'] + np.random.normal(0, 1, size=n)
    X = df['x'].values
    y = df['y'].values
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return X, y
X, y = synthetic_data()

"""
You can still choose not to normalize the data, 
BUT YOU WILL NEED TO AT LEAST CENTER THE DATA!
This is because the package is estimating coefficient without
intercept, thus you must center the predictor variables!
"""
X = X-np.mean(X)

# normalize(X)
model = RidgeRegression621(max_iter=10000_000, eta=1, lmbda=200)
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
r2

Out:
0.8960437764010043
```
