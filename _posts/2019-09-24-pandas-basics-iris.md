---
layout: post
title:  "Back to basics with Pandas"
date:   2019-09-25 20:58:00 +0100
categories: machine_learning
excerpt: A simple end-to-end example using the scientific python stack.
---

In the spirit of our [last post](/machine_learning/2019/09/25/mastering-basics), let's take things back to basics with a simple end-to-end data analysis example.

<!-- This markdown created via `jupyter nbconvert`, with some extra edits. -->

<p style="text-align: center;">
<a href="https://colab.research.google.com/github/aslanides/aslanides.github.io/blob/master/colabs/2019-09-24-pandas-basics-iris.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>

# Back to basics with `pandas`, `plotnine`, and `sklearn`

In this colab we do a quick review of the 'data science basics':
- Download a (small) dataset from an online repository.
- Use `pandas` to store and manipulate the data.
- Use `plotnine` to generate plots/visualizations.
- Use `scikit-learn` to fit a simple logistic regression model.


```
# @title Imports

import numpy as np
import pandas as pd
import plotnine as gg
import requests
import sklearn
```

## Let there be data

Let's fetch the famous [Iris dataset] from the UCI [machine learning datasets] repository.

[Iris dataset]: https://archive.ics.uci.edu/ml/datasets/iris
[machine learning datasets]: https://archive.ics.uci.edu/ml


```python
# The URL for the raw data in text form.
data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Column names/metadata.
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_columns = ['target']
```


```python
# Fetch the raw data from the website.
response = requests.get(url=data_url)
rows = response.text.split('\n')
data = [r.split(',') for r in rows]

# Turn it into a dataframe and drop any null values.
df = pd.DataFrame(data, columns=feature_columns + target_columns)
df.dropna(inplace=True)

# Fix up the data types: features are numeric, targets are categorical.
df[feature_columns] = df[feature_columns].astype('float')
df[target_column] = df[target_column].astype('category')
```

## Take a quick look at the data


```
# Look at the raw data to check it looks sensible.
df.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```
# Look at some summary statistics.
df.describe()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 6.1 KB ... now that's some #bigdata ;)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 150 entries, 0 to 149
    Data columns (total 5 columns):
    sepal_length    150 non-null float64
    sepal_width     150 non-null float64
    petal_length    150 non-null float64
    petal_width     150 non-null float64
    target          150 non-null category
    dtypes: category(1), float64(4)
    memory usage: 6.1 KB



```python
# Let's take a visual look at the data now, projecting down to 2 dimensions.
gg.theme_set(gg.theme_bw())

p = (gg.ggplot(df)
     + gg.aes(x='sepal_length', y='petal_length', color='target')
     + gg.geom_point(size=2)
     + gg.ggtitle('Iris dataset is not linearly separable')
    )
p
```


![png](/assets/pandas-basics/2019-09-24-pandas-basics-iris_10_0.png)



## Fitting a simple linear model

Let's get 'old school' and fit a simple logistic regression classifier using the raw features.

In the binary case, this is a linear probabilistic model with Bernoulli likelihood given by

$$
\mathcal{L}\left(\pmb{x}, y \lvert \pmb{x}\right) = f(\pmb{x})^y (1 - f(\pmb{x}))^{1-y},
$$

where $f(\pmb{x}) = \sigma\left(\pmb{w}^T\pmb{x}\right)$.

Assuming the dataset $\mathcal{D}=\left\{\pmb{x}^{(i)}, y^{(i)}\right\}_{i=1}^{n}$ is iid, the negative log-likelihood yields the familiar cross-entropy loss:

$$
\begin{aligned}
\log\mathcal{L}\left(\mathcal{D} \lvert \pmb{w}\right) &= \log\prod_{i=1}^{n} \sigma\left(\pmb{w}^T\pmb{x}^{(i)}\right)^{y^{(i)}} \left(1 - \sigma\left(\pmb{w}^T\pmb{x}^{(i)}\right)\right)^{1-y^{(i)}} \\
&= \sum_{i=1}^n y^{(i)}\log f\left(\pmb{x}^{(i)}\right) + (1-y^{(i)})\log\left(1 - f\left(\pmb{x}^{(i)}\right)\right)
\end{aligned}
$$

This is a convex and trivially differnetiable objective that we can optimize with gradient-based methods. We'll defer this to `scikit-learn` (which also does some other tricks + regularization).

```python
# Create a logistic regression model.
model = sklearn.linear_model.LogisticRegression(solver='lbfgs',
                                                multi_class='multinomial')

# Extract the features and targets as numpy arrays.
X = df[feature_columns].values
y = df[target_column].cat.codes  # Note that we 'code' the categories as integers.

# Fit the model.
model.fit(X, y)
```

```python
# Compute the model accuracy over the training set (we don't have a test set).
pred_df = df.copy()
pred_df['prediction'] = model.predict(X).astype('int8')
pred_df['correct'] = pred_df.prediction == pred_df.target.cat.codes
accuracy = pred_df.correct.sum() / len(pred_df)
print('Accuracy: {:.3f}'.format(accuracy))
```

    Accuracy: 0.973

```python
# Let's look again at the data, highlighting the examples which we mis-classified.
p = (gg.ggplot(pred_df)
     + gg.aes(x='sepal_length', y='petal_length', shape='target', color='correct')
     + gg.geom_point(size=2)
    )
p
```


![png](/assets/pandas-basics/2019-09-24-pandas-basics-iris_14_0.png)



