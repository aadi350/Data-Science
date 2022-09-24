```python
import sklearn
from category_encoders.helmert import HelmertEncoder
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd

data = pd.read_csv('german_credit_data.csv',index_col='Unnamed: 0')
data['Risk'] = np.where(data['Risk'] == 'bad', 1, 0) # 1 bad, 0 good
data.head()
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



 # Helmert

 ```python
HelmertEncoder(verbose=0, cols=None, mapping=None, drop_invariant=False, return_df=True, handle_unknown='value', handle_missing='value')
 
 ```

 Returns a new column for each level


```python
data['Checking account'].value_counts()
```




    little      274
    moderate    269
    rich         63
    Name: Checking account, dtype: int64




```python
enc = HelmertEncoder(
    verbose=3, 
    cols=None,#['Checking account'], # None sets all strings to be encoded
    drop_invariant=False, 
    return_df=True, 
)

enc.fit_transform(data.drop('Risk', axis=1), data[['Risk']])
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
      <th>intercept</th>
      <th>Age</th>
      <th>Sex_0</th>
      <th>Job</th>
      <th>Housing_0</th>
      <th>Housing_1</th>
      <th>Saving accounts_0</th>
      <th>Saving accounts_1</th>
      <th>Saving accounts_2</th>
      <th>Saving accounts_3</th>
      <th>Checking account_0</th>
      <th>Checking account_1</th>
      <th>Checking account_2</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose_0</th>
      <th>Purpose_1</th>
      <th>Purpose_2</th>
      <th>Purpose_3</th>
      <th>Purpose_4</th>
      <th>Purpose_5</th>
      <th>Purpose_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>67</td>
      <td>-1.0</td>
      <td>2</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1169</td>
      <td>6</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>1.0</td>
      <td>2</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>5951</td>
      <td>48</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>49</td>
      <td>-1.0</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>2096</td>
      <td>12</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>45</td>
      <td>-1.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>7882</td>
      <td>42</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>53</td>
      <td>-1.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4870</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1</td>
      <td>31</td>
      <td>1.0</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>1736</td>
      <td>12</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>40</td>
      <td>-1.0</td>
      <td>3</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>3857</td>
      <td>30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>1</td>
      <td>38</td>
      <td>-1.0</td>
      <td>2</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>804</td>
      <td>12</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1</td>
      <td>23</td>
      <td>-1.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1845</td>
      <td>45</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>1</td>
      <td>27</td>
      <td>-1.0</td>
      <td>2</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4576</td>
      <td>45</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 22 columns</p>
</div>




```python
Pipeline([
    enc
])
```

# Gamma-Poisson
`fastText` library is a pain to install

# Frequency/Cardinal  

Both from `pycaret` 

Requires `sklearn==1.1.0` or later  

Not sure if this can encoding alone, it seems it wants to always do the entire training peocess

Three most import input parameters:

| param | used |  
| --- | --- |   
| ` categorical_features` | `list` of categoricsl |   
| ` high_cardinality_features` | `bool` which attempts to compress features with multiple options |  
|  `high_cardinality_method` | `str`, default = 'frequency'.Categorical features with high cardinality are replaced with the frequency of values in each levelo ccurring in the training dataset. Other available method is 'clustering' which trains the K-Means clustering algorithm on the statistical attribute of the training data and replaces the original value of feature with the cluster label. The number of clusters is determined by optimizing Calinski-Harabasz and Silhouette criterion. |  


```python
from pycaret.classification import setup
from pycaret.classification import * # I DO NOT LIKE THIS, but the official documentation recommends it 

help(setup)
```


```python
from random import shuffle

# without fold_shuffle=True it throws an error
clf1 = setup(data = data, target = 'Risk', categorical_features=['Checking account'],  fold_shuffle=True)
```


    IntProgress(value=0, description='Processing: ', max=3)



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
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Initiated</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>13:27:32</td>
    </tr>
    <tr>
      <th>Status</th>
      <td>. . . . . . . . . . . . . . . . . .</td>
      <td>Preprocessing Data</td>
    </tr>
  </tbody>
</table>
</div>



    Text(value="Following data types have been inferred automatically, if they are correct press enter to continue…



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
      <th>Data Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>Numeric</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>Categorical</td>
    </tr>
    <tr>
      <th>Job</th>
      <td>Categorical</td>
    </tr>
    <tr>
      <th>Housing</th>
      <td>Categorical</td>
    </tr>
    <tr>
      <th>Saving accounts</th>
      <td>Categorical</td>
    </tr>
    <tr>
      <th>Checking account</th>
      <td>Categorical</td>
    </tr>
    <tr>
      <th>Credit amount</th>
      <td>Numeric</td>
    </tr>
    <tr>
      <th>Duration</th>
      <td>Numeric</td>
    </tr>
    <tr>
      <th>Purpose</th>
      <td>Categorical</td>
    </tr>
    <tr>
      <th>Risk</th>
      <td>Label</td>
    </tr>
  </tbody>
</table>
</div>



```python
import pycaret.classification as c 
```


```python
help(c)
```

# Hash Encoding

```python 
HashingEncoder(max_process=0, max_sample=0, verbose=0, n_components=8, cols=None, drop_invariant=False, return_df=True, hash_method='md5')
```


```python

from category_encoders.hashing import HashingEncoder

he = HashingEncoder(n_components=8)

he.fit_transform(X=data.drop('Risk', axis=1), y=data[['Risk']])
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
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
      <th>col_3</th>
      <th>col_4</th>
      <th>col_5</th>
      <th>col_6</th>
      <th>col_7</th>
      <th>Age</th>
      <th>Job</th>
      <th>Credit amount</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>2</td>
      <td>1169</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>2</td>
      <td>5951</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>1</td>
      <td>2096</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>2</td>
      <td>7882</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>2</td>
      <td>4870</td>
      <td>24</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>31</td>
      <td>1</td>
      <td>1736</td>
      <td>12</td>
    </tr>
    <tr>
      <th>996</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>3</td>
      <td>3857</td>
      <td>30</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>38</td>
      <td>2</td>
      <td>804</td>
      <td>12</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>2</td>
      <td>1845</td>
      <td>45</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>2</td>
      <td>4576</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 12 columns</p>
</div>



# Count Encoder

```python 
CountEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value', handle_missing='value', min_group_size=None, combine_min_nan_groups=None, min_group_name=None, normalize=False)
````


```python
from category_encoders.count import CountEncoder

ce = CountEncoder(cols=['Housing'])

ce.fit_transform(data)
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>713</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>713</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>713</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>108</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>108</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>31</td>
      <td>female</td>
      <td>1</td>
      <td>713</td>
      <td>little</td>
      <td>NaN</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>713</td>
      <td>little</td>
      <td>little</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>38</td>
      <td>male</td>
      <td>2</td>
      <td>713</td>
      <td>little</td>
      <td>NaN</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>108</td>
      <td>little</td>
      <td>little</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
      <td>1</td>
    </tr>
    <tr>
      <th>999</th>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>713</td>
      <td>moderate</td>
      <td>moderate</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 10 columns</p>
</div>



# Quantile Encoder
```python
QuantileEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, handle_missing='value', handle_unknown='value', quantile=0.5, m=1.0)
```


```python
from category_encoders.quantile_encoder import QuantileEncoder

qe = QuantileEncoder(verbose=3,cols=['Checking account'])

(qe_transform := qe.fit_transform(data.drop('Risk', axis=1), data[['Risk']]))
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>0.0</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>0.0</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.0</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.0</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>31</td>
      <td>female</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>0.0</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>996</th>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>0.0</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
    </tr>
    <tr>
      <th>997</th>
      <td>38</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>0.0</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>998</th>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.0</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>999</th>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>0.0</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 9 columns</p>
</div>




```python
qe_transform[['Credit amount']].value_counts()
```




    Credit amount
    1275             3
    1393             3
    1262             3
    1258             3
    1478             3
                    ..
    1867             1
    1872             1
    1880             1
    1881             1
    18424            1
    Length: 921, dtype: int64



# Weight of Evidence
```python
WOEEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, regularization=1.0)
```


```python
from category_encoders.woe import WOEEncoder

we = WOEEncoder(cols=['Checking account'])

we.fit_transform(data.drop('Risk', axis=1), data[['Risk']])
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>0.814519</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>0.401000</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>-1.161418</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.814519</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.814519</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>31</td>
      <td>female</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>-1.161418</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>996</th>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>0.814519</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
    </tr>
    <tr>
      <th>997</th>
      <td>38</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>-1.161418</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>998</th>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.814519</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>999</th>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>0.401000</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 9 columns</p>
</div>



## Binary Encoding
Need to figure out how to squeeze into single column


```python
from category_encoders.binary import BinaryEncoder

be =BinaryEncoder(cols=['Checking account'])
be_trans = be.fit_transform(data)
be_trans['Checking_account_BE'] = str(be_trans['Checking account_0']) + str(be_trans['Checking account_1'])

be_trans
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account_0</th>
      <th>Checking account_1</th>
      <th>Checking account_2</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
      <th>Checking_account_BE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>1</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>1</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>31</td>
      <td>female</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>996</th>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>997</th>
      <td>38</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>998</th>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
      <td>1</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
    <tr>
      <th>999</th>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
      <td>0</td>
      <td>0      0\n1      0\n2      0\n3      0\n4     ...</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 13 columns</p>
</div>




```python
from feature_engine.encoding import PRatioEncoder

pre = PRatioEncoder(encoding_method='log_ratio', variables=['Checking account']) # default is WoE
data = data.dropna(subset=['Checking account'])

pre.fit_transform(data.drop('Risk', axis=1), data[['Risk']])
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>-0.029199</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>-0.445906</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>-0.029199</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>-0.029199</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35</td>
      <td>male</td>
      <td>3</td>
      <td>rent</td>
      <td>little</td>
      <td>-0.445906</td>
      <td>6948</td>
      <td>36</td>
      <td>car</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>992</th>
      <td>23</td>
      <td>male</td>
      <td>1</td>
      <td>rent</td>
      <td>NaN</td>
      <td>-0.029199</td>
      <td>1936</td>
      <td>18</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>993</th>
      <td>30</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>-0.029199</td>
      <td>3959</td>
      <td>36</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>996</th>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>-0.029199</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
    </tr>
    <tr>
      <th>998</th>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>-0.029199</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>999</th>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>-0.445906</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
    </tr>
  </tbody>
</table>
<p>606 rows × 9 columns</p>
</div>




```python
pre = PRatioEncoder(encoding_method='ratio', variables=['Checking account']) # default is WoE
pre.fit_transform(data.drop('Risk', axis=1), data[['Risk']])
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
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>0.971223</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>0.640244</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.971223</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.971223</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35</td>
      <td>male</td>
      <td>3</td>
      <td>rent</td>
      <td>little</td>
      <td>0.640244</td>
      <td>6948</td>
      <td>36</td>
      <td>car</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>992</th>
      <td>23</td>
      <td>male</td>
      <td>1</td>
      <td>rent</td>
      <td>NaN</td>
      <td>0.971223</td>
      <td>1936</td>
      <td>18</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>993</th>
      <td>30</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>0.971223</td>
      <td>3959</td>
      <td>36</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>996</th>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>0.971223</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
    </tr>
    <tr>
      <th>998</th>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>0.971223</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>999</th>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>0.640244</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
    </tr>
  </tbody>
</table>
<p>606 rows × 9 columns</p>
</div>



# Using an Autoencoder 


```python

```

# Testing Static Method


```python
class Foo: 
    def __init__(self, *args, **kwargs):
        pass
    @staticmethod
    def s_method(*args, **kwargs):
        return args, kwargs

f = Foo()
f.s_method('a', {'test_in': 0})
```




    (('a', {'test_in': 0}), {})




```python
import datetime 
import pandas as pd
# pandas dataframe
from numpy.random import randint


base = datetime.datetime.today()
date_list = [base - datetime.timedelta(days=randint(0, 200)) for x in range(16)]
date_list2 = [base - datetime.timedelta(days=randint(0, 200)) for x in range(16)]

df = pd.DataFrame({
    'start_date': date_list,
    'other_date': date_list2
})

df.dtypes
```




    start_date    datetime64[ns]
    other_date    datetime64[ns]
    dtype: object




```python
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
      <th>start_date</th>
      <th>other_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-04-30 15:22:53.008753</td>
      <td>2022-07-24 15:22:53.008753</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-04-12 15:22:53.008753</td>
      <td>2022-05-26 15:22:53.008753</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-07-30 15:22:53.008753</td>
      <td>2022-05-22 15:22:53.008753</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-05-22 15:22:53.008753</td>
      <td>2022-07-15 15:22:53.008753</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-09-04 15:22:53.008753</td>
      <td>2022-06-14 15:22:53.008753</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
((df.start_date - df.other_date)/np.timedelta64(1, 'M'))
```




    0    -2.792665
    1    -1.445615
    2     2.266987
    3    -1.774164
    4     2.694100
    5     2.661246
    6     0.558533
    7    -3.745457
    8    -0.755662
    9     4.172570
    10   -1.807019
    11    0.525678
    12   -3.482618
    13   -1.741309
    14    3.088359
    15    0.854227
    dtype: float64




```python
help(np.timedelta64)
```

    Help on class timedelta64 in module numpy:
    
    class timedelta64(signedinteger)
     |  A timedelta stored as a 64-bit integer.
     |  
     |  See :ref:`arrays.datetime` for more information.
     |  
     |  :Character code: ``'m'``
     |  
     |  Method resolution order:
     |      timedelta64
     |      signedinteger
     |      integer
     |      number
     |      generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __eq__(self, value, /)
     |      Return self==value.
     |  
     |  __ge__(self, value, /)
     |      Return self>=value.
     |  
     |  __gt__(self, value, /)
     |      Return self>value.
     |  
     |  __hash__(self, /)
     |      Return hash(self).
     |  
     |  __le__(self, value, /)
     |      Return self<=value.
     |  
     |  __lt__(self, value, /)
     |      Return self<value.
     |  
     |  __ne__(self, value, /)
     |      Return self!=value.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __str__(self, /)
     |      Return str(self).
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from integer:
     |  
     |  __round__(...)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from integer:
     |  
     |  denominator
     |      denominator of value (1)
     |  
     |  numerator
     |      numerator of value (the value itself)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from generic:
     |  
     |  __abs__(self, /)
     |      abs(self)
     |  
     |  __add__(self, value, /)
     |      Return self+value.
     |  
     |  __and__(self, value, /)
     |      Return self&value.
     |  
     |  __array__(...)
     |      sc.__array__(dtype) return 0-dim array from scalar with specified dtype
     |  
     |  __array_wrap__(...)
     |      sc.__array_wrap__(obj) return scalar from array
     |  
     |  __bool__(self, /)
     |      self != 0
     |  
     |  __copy__(...)
     |  
     |  __deepcopy__(...)
     |  
     |  __divmod__(self, value, /)
     |      Return divmod(self, value).
     |  
     |  __float__(self, /)
     |      float(self)
     |  
     |  __floordiv__(self, value, /)
     |      Return self//value.
     |  
     |  __format__(...)
     |      NumPy array scalar formatter
     |  
     |  __getitem__(self, key, /)
     |      Return self[key].
     |  
     |  __int__(self, /)
     |      int(self)
     |  
     |  __invert__(self, /)
     |      ~self
     |  
     |  __lshift__(self, value, /)
     |      Return self<<value.
     |  
     |  __mod__(self, value, /)
     |      Return self%value.
     |  
     |  __mul__(self, value, /)
     |      Return self*value.
     |  
     |  __neg__(self, /)
     |      -self
     |  
     |  __or__(self, value, /)
     |      Return self|value.
     |  
     |  __pos__(self, /)
     |      +self
     |  
     |  __pow__(self, value, mod=None, /)
     |      Return pow(self, value, mod).
     |  
     |  __radd__(self, value, /)
     |      Return value+self.
     |  
     |  __rand__(self, value, /)
     |      Return value&self.
     |  
     |  __rdivmod__(self, value, /)
     |      Return divmod(value, self).
     |  
     |  __reduce__(...)
     |      Helper for pickle.
     |  
     |  __rfloordiv__(self, value, /)
     |      Return value//self.
     |  
     |  __rlshift__(self, value, /)
     |      Return value<<self.
     |  
     |  __rmod__(self, value, /)
     |      Return value%self.
     |  
     |  __rmul__(self, value, /)
     |      Return value*self.
     |  
     |  __ror__(self, value, /)
     |      Return value|self.
     |  
     |  __rpow__(self, value, mod=None, /)
     |      Return pow(value, self, mod).
     |  
     |  __rrshift__(self, value, /)
     |      Return value>>self.
     |  
     |  __rshift__(self, value, /)
     |      Return self>>value.
     |  
     |  __rsub__(self, value, /)
     |      Return value-self.
     |  
     |  __rtruediv__(self, value, /)
     |      Return value/self.
     |  
     |  __rxor__(self, value, /)
     |      Return value^self.
     |  
     |  __setstate__(...)
     |  
     |  __sizeof__(...)
     |      Size of object in memory, in bytes.
     |  
     |  __sub__(self, value, /)
     |      Return self-value.
     |  
     |  __truediv__(self, value, /)
     |      Return self/value.
     |  
     |  __xor__(self, value, /)
     |      Return self^value.
     |  
     |  all(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.all`.
     |  
     |  any(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.any`.
     |  
     |  argmax(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.argmax`.
     |  
     |  argmin(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.argmin`.
     |  
     |  argsort(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.argsort`.
     |  
     |  astype(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.astype`.
     |  
     |  byteswap(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.byteswap`.
     |  
     |  choose(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.choose`.
     |  
     |  clip(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.clip`.
     |  
     |  compress(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.compress`.
     |  
     |  conj(...)
     |  
     |  conjugate(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.conjugate`.
     |  
     |  copy(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.copy`.
     |  
     |  cumprod(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.cumprod`.
     |  
     |  cumsum(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.cumsum`.
     |  
     |  diagonal(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.diagonal`.
     |  
     |  dump(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.dump`.
     |  
     |  dumps(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.dumps`.
     |  
     |  fill(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.fill`.
     |  
     |  flatten(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.flatten`.
     |  
     |  getfield(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.getfield`.
     |  
     |  item(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.item`.
     |  
     |  itemset(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.itemset`.
     |  
     |  max(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.max`.
     |  
     |  mean(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.mean`.
     |  
     |  min(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.min`.
     |  
     |  newbyteorder(...)
     |      newbyteorder(new_order='S', /)
     |      
     |      Return a new `dtype` with a different byte order.
     |      
     |      Changes are also made in all fields and sub-arrays of the data type.
     |      
     |      The `new_order` code can be any from the following:
     |      
     |      * 'S' - swap dtype from current to opposite endian
     |      * {'<', 'little'} - little endian
     |      * {'>', 'big'} - big endian
     |      * '=' - native order
     |      * {'|', 'I'} - ignore (no change to byte order)
     |      
     |      Parameters
     |      ----------
     |      new_order : str, optional
     |          Byte order to force; a value from the byte order specifications
     |          above.  The default value ('S') results in swapping the current
     |          byte order.
     |      
     |      
     |      Returns
     |      -------
     |      new_dtype : dtype
     |          New `dtype` object with the given change to the byte order.
     |  
     |  nonzero(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.nonzero`.
     |  
     |  prod(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.prod`.
     |  
     |  ptp(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.ptp`.
     |  
     |  put(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.put`.
     |  
     |  ravel(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.ravel`.
     |  
     |  repeat(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.repeat`.
     |  
     |  reshape(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.reshape`.
     |  
     |  resize(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.resize`.
     |  
     |  round(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.round`.
     |  
     |  searchsorted(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.searchsorted`.
     |  
     |  setfield(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.setfield`.
     |  
     |  setflags(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.setflags`.
     |  
     |  sort(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.sort`.
     |  
     |  squeeze(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.squeeze`.
     |  
     |  std(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.std`.
     |  
     |  sum(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.sum`.
     |  
     |  swapaxes(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.swapaxes`.
     |  
     |  take(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.take`.
     |  
     |  tobytes(...)
     |  
     |  tofile(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.tofile`.
     |  
     |  tolist(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.tolist`.
     |  
     |  tostring(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.tostring`.
     |  
     |  trace(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.trace`.
     |  
     |  transpose(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.transpose`.
     |  
     |  var(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.var`.
     |  
     |  view(...)
     |      Scalar method identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.view`.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from generic:
     |  
     |  T
     |      Scalar attribute identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.T`.
     |  
     |  __array_interface__
     |      Array protocol: Python side
     |  
     |  __array_priority__
     |      Array priority.
     |  
     |  __array_struct__
     |      Array protocol: struct
     |  
     |  base
     |      Scalar attribute identical to the corresponding array attribute.
     |      
     |      Please see `ndarray.base`.
     |  
     |  data
     |      Pointer to start of data.
     |  
     |  dtype
     |      Get array data-descriptor.
     |  
     |  flags
     |      The integer value of flags.
     |  
     |  flat
     |      A 1-D view of the scalar.
     |  
     |  imag
     |      The imaginary part of the scalar.
     |  
     |  itemsize
     |      The length of one element in bytes.
     |  
     |  nbytes
     |      The length of the scalar in bytes.
     |  
     |  ndim
     |      The number of array dimensions.
     |  
     |  real
     |      The real part of the scalar.
     |  
     |  shape
     |      Tuple of array dimensions.
     |  
     |  size
     |      The number of elements in the gentype.
     |  
     |  strides
     |      Tuple of bytes steps in each dimension.
    



```python

```
