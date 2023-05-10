---
title: "Comparing Probability Distributions with Kullback-Leibler (KL) Divergence"
classes: wide
tags: [python, statistics, probability]
desc: "A simple tour of KL Divergence and why it is relevant"
---

Probability distributions are ubiquitous in data science, and you will come across and use them in some form or fashion in almost everything you do in terms of data science. An interesting operation that you can do with distributions is to quantify how different or how much "information is lost" between two distributions. One way to do this and what this post will explore is Kullback-Leibler (KL) Divergence. 

One modern example of an application of KL Divergence is inside the loss function for Variational Autoencoders (VAEs). VAEs are a type of generative deep learning model, and KL Divergence is used to assist the model in creating distributions that are close to a standard normal distribution. VAEs won't be covered in this post directly, but they are certainly a fascinating topic that I would highly recommend as future reading.

To begin exploring KL Divergence, let's set up a toy dataset that is intuitive enough to allow you see the big picture. First, let's import some necessary libraries.


```python
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
```

With the libraries imported, I'll create a simple dataset. The dataset is synthetic and is meant to simply store a count of goals scored by some imaginary set of players. Each of the other columns stores the probabilities of the associated goals value, and we have three different distributions represented: Observed, uniform, and binomial. 


```python
# create dataset with some toy data
samples = pd.DataFrame({"goals":[2]*3 + [3]*3 + [4]*5 + [5]*6 + [6]*3 + [7]*2 + [8]*3})
# calculate probabilities associated with each observed value
emp_probs = (samples.groupby("goals").goals.count() / len(samples)).rename("emp_probs").reset_index()
samples = samples.merge(emp_probs, on = ["goals"])
# calculate values under a uniform distribution for each observed value
samples['uniform_probs'] = 1 / samples.goals.nunique()
# calculate values under a binomial distribution constructed from observed data
samples['binomial_probs'] = samples.goals.apply(lambda x: binom.pmf(x, n = len(samples), p = samples.goals.mean()/len(samples)))
```


```python
samples
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
      <th>goals</th>
      <th>emp_probs</th>
      <th>uniform_probs</th>
      <th>binomial_probs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.079725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.079725</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.079725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.146742</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.146742</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.146742</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>0.20</td>
      <td>0.142857</td>
      <td>0.193764</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>0.20</td>
      <td>0.142857</td>
      <td>0.193764</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>0.20</td>
      <td>0.142857</td>
      <td>0.193764</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>0.20</td>
      <td>0.142857</td>
      <td>0.193764</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>0.20</td>
      <td>0.142857</td>
      <td>0.193764</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5</td>
      <td>0.24</td>
      <td>0.142857</td>
      <td>0.195379</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5</td>
      <td>0.24</td>
      <td>0.142857</td>
      <td>0.195379</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5</td>
      <td>0.24</td>
      <td>0.142857</td>
      <td>0.195379</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>0.24</td>
      <td>0.142857</td>
      <td>0.195379</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>0.24</td>
      <td>0.142857</td>
      <td>0.195379</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>0.24</td>
      <td>0.142857</td>
      <td>0.195379</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.156355</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.156355</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.156355</td>
    </tr>
    <tr>
      <th>20</th>
      <td>7</td>
      <td>0.08</td>
      <td>0.142857</td>
      <td>0.101888</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>0.08</td>
      <td>0.142857</td>
      <td>0.101888</td>
    </tr>
    <tr>
      <th>22</th>
      <td>8</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.055037</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.055037</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8</td>
      <td>0.12</td>
      <td>0.142857</td>
      <td>0.055037</td>
    </tr>
  </tbody>
</table>
</div>



First, let's inspect the "observed" data. The visualization below displays the distribution of the goals data. 


```python
sns.barplot(x = samples.goals, 
            y = samples.emp_probs, 
            color = "red", 
            alpha = 0.75)

plt.title("Distribution of Goals (Observed)")
_ = plt.xlabel("Goals")
_ = plt.ylabel("Probability")
sns.despine()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kl_divergence/output_6_0.png" alt="">
    


Next, let's look at a uniform constructed to represent the goals data. In this uniform distribution, all seven unique values have the same probability, 1/7.


```python
sns.barplot(x = samples.drop_duplicates().goals, 
            y = samples.drop_duplicates().uniform_probs, 
            color = "red", 
            alpha = 0.75)
_ = plt.title("Distribution of Goals (Uniform)")
_ = plt.xlabel("Goals")
_ = plt.ylabel("Probability")
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kl_divergence/output_8_0.png" alt="">


Finally, we have constructed a binomial distribution. If you are familiar with the normal distribution, you will see that it looks very similar. You can consider the binomial distribution the "normal" distribution for discrete data. The binomial distribution can be represented by two values, n and p, where n is the number of trials and p is the probability. We can estimate the probability by taking the mean of the data divided by the number of trials.


```python
sns.barplot(x = samples.drop_duplicates().goals, 
            y = samples.drop_duplicates().binomial_probs, 
            color = "red", 
            alpha = 0.75)
_ = plt.title("Distribution of Goals (Binomial)")
_ = plt.xlabel("Goals")
_ = plt.ylabel("Probability")
```


<img src="{{ site.url }}{{ site.baseurl }}/images/kl_divergence/output_10_0.png" alt="">
    


Now, let's take a look at all distributions together.


```python
grouped_samples = samples.melt(id_vars=["goals"]).drop_duplicates().replace({"emp_probs":"Observed", "uniform_probs":"Uniform", "binomial_probs":"Binomial"})
sns.barplot(data = grouped_samples, 
            x = "goals", 
            y = "value", 
            hue = "variable", 
            alpha = 0.75, 
            palette = ["red", "blue", "green"])

plt.title("All Distributions")
plt.xlabel("Goals")
plt.ylabel("Probability")
```




    Text(0, 0.5, 'Probability')




<img src="{{ site.url }}{{ site.baseurl }}/images/kl_divergence/output_12_1.png" alt="">
    


Now that we have our three different distributions for the data, let's get into the implementation of KL Divergence. Below is the equation for KL Divergence, where p is the probability of the sample and q is the probability of the sample from the approximating distribution.

<img src="{{ site.url }}{{ site.baseurl }}/images/kl_divergence/kl_div.png" alt="">

After taking this equation and implementing it in code, we can calculate the KL Divergence between the observed distribution and the uniform and binomial distributions. 


```python
def kl_divergence(p, q):
    return np.sum(p * (np.log2(p) - np.log2(q)))
```


```python
print(f"KL Divergence between Observed and Uniform Distribution: {kl_divergence(samples.emp_probs, samples.uniform_probs)}")
```

    KL Divergence between Observed and Uniform Distribution: 1.067154975387125



```python
print(f"KL Divergence between Observed and Binomial Distribution: {kl_divergence(samples.emp_probs, samples.binomial_probs)}")
```

    KL Divergence between Observed and Binomial Distribution: 0.792492668625009


As you can see from the results, you can see that the binomial distribution has a lower KL Divergence from the observed distribution than the uniform, so it's a better approximation. Finally, let's check our function to make sure it's working how we expect. If you calculate the KL Divergence of the distribution from itself, you should get a value of 0.


```python
print("KL Divergence Check - all results should display 0:\n"
      f"Observed - Observed: {kl_divergence(samples.emp_probs, samples.emp_probs)}\n"
      f"Uniform - Uniform: {kl_divergence(samples.uniform_probs, samples.uniform_probs)}\n"
      f"Binomial - Binomial: {kl_divergence(samples.binomial_probs, samples.binomial_probs)}\n")
```

    KL Divergence Check - all results should display 0:
    Observed - Observed: 0.0
    Uniform - Uniform: 0.0
    Binomial - Binomial: 0.0
    


And it looks like it is working well! Thank you for following this post, and I hope you found it informative.
