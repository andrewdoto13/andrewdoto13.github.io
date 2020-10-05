---
title: "Linear Regression Gradient Descent from Scratch"
classes: wide
tags: [python, machine learning, linear-regression, numpy]
desc: "Exploring linear regression by building from scratch"
---

In the past few weeks of my Applied Machine Learning course, we've been focusing on linear regression and its extensions. I wanted to continue my exploration of the algorithms by trying to build an implementation from scratch. A main part of this section of the course focused on gradient descent, so this implementation uses gradient descent as the main driver of the algorithm, as opposed to the closed-form solution to linear regression. As I did with K Nearest Neighbors, I built an object-oriented implementation modeled after the SKLearn "fit-predict" paradigm. Here are a few other notes about this project:

* I am using full gradient descent here. I tried to implement a stochastic version but I was having trouble with the indexing/slicing inside the _gd function. Any tips would be very helpful!

* The objective function that I used in this is mean squared error. Throughout this section of the course, we became familiar with the gradient of this objective function.

* I added L2 regularization, which is also known as ridge regression. By setting the "l2_reg" hyperparameter to 0, you can "turn it off".

* The learning rate (eta) and regularization hyperparameter are passed at the initial instantiation of the class, opposed to at the "fit" method. Are there pros/cons to each?

* I didn't realize exactly how important scaling the data is for this to work. I spent a lot of time trying to figure out why gradient descent wasn't working. Actually gradient descent was going in the **opposite** direction! I'll do a little demonstration at the end to show this. It was a hard lesson to learn, but very valuable at the end of the day.

First, I will import numpy, pandas, matplotlib, and seaborn. Numpy is the foundation for this project, and I've used pandas to build a nicer-looking dataframe that holds the history of gradient descent. Matplotlib and seaborn are imported so that I could show some visualizations.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
```

Now that I have the necessary packages, I can go ahead and create the class.


```python
class AndyLinRegGD(object):

    def __init__(self, eta = 0.001, l2_reg = 0.0):
        self.coef_ = None       # weight vector
        self.bias_ = None       # bias term
        self._theta = None      # augmented weight vector, i.e., bias + weights
        self._eta = eta         # step size for gradient descent
        self.l2_reg = l2_reg    # control parameter for regularization penalty term

        #gradient descent history, will build as DataFrame in later function
        self.history = {"MSE": [],
                       "Updated Weights":[]}

    def _gradient(self, X, y):
        """
        Calculate the gradient of the MSE objective function

        Args:
            X(ndarray):        training data
            y(ndarray):        target data
        Return:
            gradient(ndarray): vector of partial derivatives wrt each weight
        """
        # calc predictions
        predictions = np.dot(X, self._theta)
        # get error for each example
        errors = predictions - y
        #calculate gradient
        gradient = 2 * np.dot(errors, X) / X.shape[0]
        # penalties only for weights
        gradient[1:] += 2 * self.l2_reg * self._theta[1:]
        return gradient

    def score(self, X, y):
        """
        Calculate the Mean Squared Error

        Args:
            X(ndarray):        training data
            y(ndarray):        target data
        Return:
            score(float): mean squared error
        """
        predictions = np.dot(X, self._theta)
        errors = predictions - y
        mse = np.sum(errors**2) / X.shape[0]

        return mse

    def _gd(self, X, y, max_iter):
        """
        Performs full gradient descent

        Args:
            X(ndarray):        training data
            y(ndarray):        target data
            max_iter:          number of times gradient descent happens, passed into fit method
        Return:
            None, but updates self._theta
        """
        # for each interation
        for epoch in range(max_iter):
            # calc mse
            mse = self.score(X, y)
            # store in history dict
            self.history["MSE"].append(mse)

            # calculate gradient
            gradient = self._gradient(X, y)

            # do gradient step, update theta
            self._theta -= self._eta * gradient
            # store in history dict
            self.history['Updated Weights'].append(self._theta)

            if mse < 1e-6:
                break

    def fit(self, X, y, max_iter=1000):
        """
        Utilizes other functions to train model and update theta, coef, and intercept

        Args:
            X(ndarray):        training data
            y(ndarray):        target data
            max_iter:          number of times gradient descent happens
        Return:
            self(object)
        """
        # create array of ones for bias term
        bias_term = np.ones((X.shape[0],1))
        # add the bias term to the data
        X = np.concatenate((bias_term, X), axis = 1)
        # initialize weights
        self._theta = np.random.rand(X.shape[1])
        # perform gradient descent
        self._gd(X, y, max_iter)
        # set intercept from theta
        self.intercept_ = self._theta[0]
        # set coefs from theta
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X):
        """
        Utilizes the fitted model to generate predictions

        Args:
            X(ndarray):        training data
        Return:
            predictions(array)
        """
        # create array of ones for bias term
        bias_term = np.ones((X.shape[0],1))
        # add the bias term to the data
        X = np.concatenate((bias_term, X), axis = 1)
        # calc predictions
        predictions = np.dot(X, self._theta)

        return predictions

    def get_history(self):
        """
        Generate DataFrame of gradient descent history

        Args:
            None
        Return:
            history(DataFrame)
        """
        # pass self.history dict into pandas df
        df = pd.DataFrame(self.history)
        # create iterations column from index
        df['Iterations'] = df.index + 1
        # reorder dataframe
        return df[['Iterations','MSE','Updated Weights']]
```

In order to demonstrate this implementation, I will use the prototypical Boston dataset. I will load the data, split it into train and test, use the standard scaler, and then perform the fitting. I will also output a lineplot so that you can see the progress of gradient descent in action.


```python
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
boston = sklearn.datasets.load_boston()
from sklearn.model_selection import train_test_split

# load the data
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

# scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# instantiate the model and fit to the scaled training data
lr = AndyLinRegGD(eta = 0.1, l2_reg = 0.01)
lr.fit(X_train_scaled, y_train, max_iter = 100)

# plot the gradient descent history
hist = lr.get_history()
plt.figure(figsize=(8,8))
sns.lineplot(x = 'Iterations', y = "MSE", data = hist)
```




    <AxesSubplot:xlabel='Iterations', ylabel='MSE'>




<img src="{{ site.url }}{{ site.baseurl }}/images/lr/lr_fig1.png" alt="">


Let's also take a look at the history dataframe for gradient descent.


```python
hist.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iterations</th>
      <th>MSE</th>
      <th>Updated Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>593.982458</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>358.076426</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>234.355188</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>157.144036</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>108.103451</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
  </tbody>
</table>
</div>




```python
hist.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iterations</th>
      <th>MSE</th>
      <th>Updated Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>96</td>
      <td>19.646877</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>97</td>
      <td>19.645203</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>98</td>
      <td>19.643577</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>99</td>
      <td>19.641997</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
    <tr>
      <th>99</th>
      <td>100</td>
      <td>19.640460</td>
      <td>[22.38926553220308, -0.870699471124182, 0.9237...</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the model, let's go ahead and create predictions for the test set and see how it performs


```python
predictions = lr.predict(X_test_scaled)

mse = np.sum((predictions - y_test)**2)/len(y_test)

print("The MSE for the test set is", mse)
```

    The MSE for the test set is 27.919555078006432


There it is! The model is at least functioning how I would expect. Before I end this post, I want to demonstrate how important scaling the data is before you fit the model to it. It took me so long to figure out why gradient descent wasn't working, only to find out that scaling the data single-handedly fixed the issue. I never realized just how important it was, but for me, gradient descent just was not working at all. Here's a demo so that you can see what I mean.


```python
#Use the pre-scaled dataset and a new lr object
lr2 = AndyLinRegGD(eta = 0.1, l2_reg = 0.01)
lr2.fit(X_train, y_train)

# plot the gradient descent history, notice the scale on the y-axis
hist2 = lr2.get_history()
plt.figure(figsize=(8,8))
sns.lineplot(x = 'Iterations', y = "MSE", data = hist2)
```

    /usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:46: RuntimeWarning: overflow encountered in square
    /usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:29: RuntimeWarning: overflow encountered in multiply





    <AxesSubplot:xlabel='Iterations', ylabel='MSE'>




<img src="{{ site.url }}{{ site.baseurl }}/images/lr/lr_fig2.png" alt="">



```python
hist2.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iterations</th>
      <th>MSE</th>
      <th>Updated Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.178545e+05</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.437804e+14</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.707692e+24</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>6.571943e+33</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2.529172e+43</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>9.733367e+52</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>3.745827e+62</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1.441559e+72</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>5.547752e+81</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>2.135019e+91</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
hist2.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iterations</th>
      <th>MSE</th>
      <th>Updated Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>995</th>
      <td>996</td>
      <td>NaN</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>996</th>
      <td>997</td>
      <td>NaN</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>997</th>
      <td>998</td>
      <td>NaN</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>998</th>
      <td>999</td>
      <td>NaN</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1000</td>
      <td>NaN</td>
      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>
    </tr>
  </tbody>
</table>
</div>



Isn't that wild? I don't really know what's happening with the weights but you can see at least in the first ten iterations of gradient descent how the MSE just skyrockets instead of steadily decreasing. I can't tell you much time I spent pouring over the gradient descent function in this project! Ultimately, now I know how important scaling can be to a linear regression model.

Thank you for taking the time to read through this post. If you notice any errors, inconsistencies, or opportunities to improve the code, please let me know! I want to eventually extend this to an elastic net/stochastic version, so any advice or tips would be very appreciated! Take care!
