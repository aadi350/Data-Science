This entire post is dedicated to coming to terms with `tensorflow`'s metrics, and the varying input formats associated with it. It is a direct result of me not using the correct version of a metric and leaving a model to train for 4 days (only realising afterwards that I should have used the non-sparse version of Accuracy). 

This is not meant to expose the underpinnings and statistical wizardry of the intentions of these metrics (information theory, physics, etc), but is rather meant to be my notes on how to correctly use these metrics in deep-learning applications

# Accuracy-Derived

## Accuracy

Probably not the one you should use, this expects a list, where each item is a prediction label The only place I've seen it used is in the [CropNet example](https://www.tensorflow.org/hub/tutorials/cropnet_cassava), where individual examples were evaluated separately. Most architectures for classification typically output a vector of class probabilities, as opposed to a hard prediction, so this might be useful after-the-fact OR if you threshold the vector




```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import metrics
```


```python
# binary example
y_actual = [1, 0, 0, 1]
y_pred = [1, 1, 0, 1] # we expect 0.75% accuracy

m = metrics.Accuracy()
m(y_actual, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.75>




```python
# multinomial example
y_actual = [1, 2, 3, 4]
y_pred = [1, 2, 2, 4] # we expect 0.75% accuracy

m = metrics.Accuracy()
m(y_actual, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.75>




```python
# OHE? Apparently not
y_actual = [
    [0, 1, 0],
    [1, 0, 0]
]

y_pred = [
    [0.2, 0.5, 0.3],
    [0.2, 0.5, 0.3],
]

m = metrics.Accuracy()
m(y_actual, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.0>



## Binary Accuracy

This seems to be controlled via a threshold parameter, and is a specific version of the above. This might be useful if (e.g.) your network has a single output cell `Dense(1)`, which represents a positive/negative class.

This is a simple:
$$
\frac{\text{Number True Predictions}}{\text{Number Predictions}}
$$ 
(exactly the same as Accuracy above, except each entry is expected to be some probability of class) 


```python
m = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
m([1, 0, 0, 1], [0.501, 0, 1, 1]) # expect 0.75 when default threshold used
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.75>



## Categorical Accuracy
This is one of my most-used accuracy measures. It calcualtes how often predictions match one-hot-labels. `y_true` and `y_pred` are both one-hot encode


```python
y_true = [
    [1, 0, 0],
    [0, 0, 1],
]
y_pred = [
    [0.6, 0.15, 0.25],
    [0.5, 0.3, 0.2], # expect 50% accuracy
]

m = metrics.CategoricalAccuracy()
m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>



## SparseCategorical Accuracy

This expects a vector of class probabilities as `y_pred` and a list of actual class-labels as `y_true`. It is taken as the ratio of the correct predictions (argmax of the vector) over the net number of predictions


```python
y_true = [0, 2] # exactly the same as prior, now just as actual labels

m = metrics.SparseCategoricalAccuracy()
m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>



# Crossentropy-Type

As opposed to raw true-vs-false predictions, # TODO

## Binary Crossentropy

If `from_logits` is true, the output is not assumed to be bounded between 0 and 1. (The negative sign is to counteract the fact that log of a number less than 1 is negative)

$$
\frac{1}{N}\sum_{i=1}^N - \left[y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]
$$


```python
y_true = [0, 1] # shape is (batch_size, d0, .., dN)
y_pred = [0.2, 0.9]
m = metrics.BinaryCrossentropy() 
m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.1642519>



Doing this manually


```python
# manually, N=1 so I ignore that
log_loss = 0
for y_i, p_i in zip(y_true, y_pred):
    log_loss += -(y_i * np.log(p_i) + (1-y_i)*np.log(1-p_i))

# so it makes sense, same as above!
log_loss /= 2 

log_loss
```




    0.164252033486018




## Categorical Crossentropy

Essentially binary cross-entropy with an added dimension:

$$
-\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^M  y_{ij} \log(p_{ij}) 
$$ 


```python
y_true = [[0, 0, 1]] # here, our sample belongs to class 2 (index of position is 2)
y_pred = [[0.1, 0.1, 0.8]] # our predicted implies class 2 has the highest probability
m = metrics.CategoricalCrossentropy()

m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.22314353>




```python
N = 1
M = 3
log_loss = 0
for i in range(N):
    for j in range(M):
        log_loss -= y_true[i][j]*np.log(y_pred[i][j])
            

log_loss /= N
log_loss
```




    0.2231435513142097



## Sparse Categorical Crossentropy

Exactly the same as above, except accepts `y_true` as single labels, instead of vectors


```python
y_true = [2]
m = metrics.SparseCategoricalCrossentropy()

m(y_true, y_pred) # the answer should be EXACTLY the same as above
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.22314355>



# Checkpoint - What does "sparse" mean?
So the difference between accuracy/crossentropy and their respective *sparse* versions are the format of the labels. The sparse versions expect that the labels are defined as-is (class 2 implies the label is [2]), whilst the non-sparse versions expect one-hot encoded labels (so the same class 2 looks like [0, 0, 1, 0... N] if we have N classes)  

# Other Common Metrics
## Mean Absolute Error

This is the typical difference between predicted and actual scaled by the number of samples (also taken as the absolute sum of errors)

$$
\frac{\sum_{i=1}^{N}|\hat{y}-y|}{N}
$$


```python
y_true = [1, 1, 2]
y_pred = [1, 2, 2] # this expects 0.33 error
m = metrics.MeanAbsoluteError()
m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.33333334>




```python
# does it also work with OHE?
y_pred = [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
]
m(y_true, y_pred) # apprently not...
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.8333333>




```python
# if we define y_true as OHE
y_true = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
]
m(y_true, y_pred) # also no
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.5714286>



## Mean Absolute Percentage Error

This appears to not only consider how many predictions are wrong, but appears to be scaled by the label (following the usual MAPE formula)

$$
\frac{100}{N}\sum_{i=1}^N \left|\frac{y-\hat{y}}{y} \right|
$$

where $y$ is the actual value, and $\hat{y}$ is the forecast value


```python
y_true = [1, 2, 4]
y_pred = [1, 2, 3]

m = metrics.MeanAbsolutePercentageError()
m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=8.333334>




```python
y_true = [1, 2, 4]
y_pred = [1, 2, 2]

m = metrics.MeanAbsolutePercentageError()
m(y_true, y_pred) # although the same class is wrong, the percentage is different
```




    <tf.Tensor: shape=(), dtype=float32, numpy=16.666668>




```python
# doing it in numpy
error = 0
N = 3
for true, pred in zip(y_true, y_pred):
    error += np.abs((true-pred)/true)

error *= (100/N)

error # seems to line up with the above
```




    16.666666666666668




```python
# one-hot-encoding?
y_true = [[0, 1, 0]]
y_pred = [[0, 1, 0]]

m(y_true, y_pred) # guess not, because this error is supposed to be zero
```




    <tf.Tensor: shape=(), dtype=float32, numpy=8.333334>



## Mean Squared Error

This is given as the sum of errors squared:

$$
\frac{1}{N}\sum_{i=1}^N\left(y_i-\hat{y}_i\right)^2
$$


```python
y_true = [1, 2, 4]
y_pred = [1, 2, 3]

m = metrics.MeanSquaredError()

m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.33333334>




```python
N = 3
error = 0
for true, pred in zip(y_true, y_pred):
    error += np.power(true-pred, 2)

error /= N

error # seems about right
```




    0.3333333333333333




```python
# OHE?
y_true = [
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
]

y_pred = [
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
]

m(y_true, y_pred) # nope, I'm still not even sure how this is calculated....
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.18333334>




```python
# simpler example
y_true = [
    [0, 1],
]

y_pred = [
    [1, 0],
]

m(y_true, y_pred) # THIS SHOULD BE ZERO
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.21666667>



In short: the MSE, MAPE and MAE metrics are not suitable for one-hot-encoded labels/predictions 

## Precision

This is taken as:
$$
\frac{\text{Number of true positives}}{\text{Number of true+false positives}}
$$ 

This implementation only supports binary targets (which makes sense in the context of true-vs-false samples)


```python
# binary example
y_true = [0, 1, 0]
y_pred = [0, 0.6, 1]

m = metrics.Precision()
m(y_true, y_pred) # should be 1/2
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>




```python
# multinomial 
y_true = [1, 2, 3, 3]
y_pred = [1, 2, 3, 4]

m(y_true, y_pred) # doesn't work here
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-53-7c4421497ddf> in <module>
          3 y_pred = [1, 2, 3, 4]
          4 
    ----> 5 m(y_true, y_pred)
    

    /usr/local/lib/python3.7/dist-packages/keras/metrics/base_metric.py in __call__(self, *args, **kwargs)
        199     from keras.distribute import distributed_training_utils  # pylint:disable=g-import-not-at-top
        200     return distributed_training_utils.call_replica_local_fn(
    --> 201         replica_local_fn, *args, **kwargs)
        202 
        203   def __str__(self):


    /usr/local/lib/python3.7/dist-packages/keras/distribute/distributed_training_utils.py in call_replica_local_fn(fn, *args, **kwargs)
         58     with strategy.scope():
         59       return strategy.extended.call_for_each_replica(fn, args, kwargs)
    ---> 60   return fn(*args, **kwargs)
         61 
         62 


    /usr/local/lib/python3.7/dist-packages/keras/metrics/base_metric.py in replica_local_fn(*args, **kwargs)
        179         update_op = None
        180       else:
    --> 181         update_op = self.update_state(*args, **kwargs)  # pylint: disable=not-callable
        182       update_ops = []
        183       if update_op is not None:


    /usr/local/lib/python3.7/dist-packages/keras/utils/metrics_utils.py in decorated(metric_obj, *args, **kwargs)
         68 
         69     with tf_utils.graph_context_for_symbolic_tensors(*args, **kwargs):
    ---> 70       update_op = update_state_fn(*args, **kwargs)
         71     if update_op is not None:  # update_op will be None in eager execution.
         72       metric_obj.add_update(update_op)


    /usr/local/lib/python3.7/dist-packages/keras/metrics/base_metric.py in update_state_fn(*args, **kwargs)
        138         ag_update_state = tf.__internal__.autograph.tf_convert(
        139             obj_update_state, control_status)
    --> 140         return ag_update_state(*args, **kwargs)
        141     else:
        142       if isinstance(obj.update_state, tf.__internal__.function.Function):


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
        687       try:
        688         with conversion_ctx:
    --> 689           return converted_call(f, args, kwargs, options=options)
        690       except Exception as e:  # pylint:disable=broad-except
        691         if hasattr(e, 'ag_error_metadata'):


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
        329   if conversion.is_in_allowlist_cache(f, options):
        330     logging.log(2, 'Allowlisted %s: from cache', f)
    --> 331     return _call_unconverted(f, args, kwargs, options, False)
        332 
        333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
        456 
        457   if kwargs is not None:
    --> 458     return f(*args, **kwargs)
        459   return f(*args)
        460 


    /usr/local/lib/python3.7/dist-packages/keras/metrics/metrics.py in update_state(self, y_true, y_pred, sample_weight)
        827         top_k=self.top_k,
        828         class_id=self.class_id,
    --> 829         sample_weight=sample_weight)
        830 
        831   def result(self):


    /usr/local/lib/python3.7/dist-packages/keras/utils/metrics_utils.py in update_confusion_matrix_variables(variables_to_update, y_true, y_pred, thresholds, top_k, class_id, sample_weight, multi_label, label_weights, thresholds_distributed_evenly)
        607           y_pred,
        608           tf.cast(1.0, dtype=y_pred.dtype),
    --> 609           message='predictions must be <= 1')
        610   ]):
        611     if sample_weight is None:


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
        151     except Exception as e:
        152       filtered_tb = _process_traceback_frames(e.__traceback__)
    --> 153       raise e.with_traceback(filtered_tb) from None
        154     finally:
        155       del filtered_tb


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/check_ops.py in _binary_assert(sym, opname, op_func, static_func, x, y, data, summarize, message, name)
        408           node_def=None,
        409           op=None,
    --> 410           message=('\n'.join(_pretty_print(d, summarize) for d in data)))
        411 
        412     else:  # not context.executing_eagerly()


    InvalidArgumentError: predictions must be <= 1
    Condition x <= y did not hold.
    First 3 elements of x:
    [1. 2. 3.]
    First 1 elements of y:
    [1.]



```python
# OHE?
y_true = [
    [0, 1, 0],
    [1, 0, 0]
]

y_pred = [
    [1, 0, 0],
    [1, 0, 0],
]

m(y_true, y_pred) # seems like it's able to handle OHE labels
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>



## Recall
This is taken as:
$$
\frac{\text{Number of true positives}}{\text{Number of true positives + false negatives}}
$$ 

This implementation only supports binary targets (which makes sense in the context of true-vs-false samples


```python
# binary example
y_true = [0, 1, 0]
y_pred = [0, 1, 1]

m = metrics.Recall()
m(y_true, y_pred) # should be 1
```




    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>




```python
# OHE?
y_true = [
    [1, 0, 0],
    [1, 0, 0]
]

y_pred = [
    [1, 0, 0],
    [1, 0, 0],
]

m(y_true, y_pred) # seems like it's not able to handle OHE labels....
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.8888889>



## KL Divergence
This is taken as a product of the true-class probability multiplied by the log-ratio of predicted to true class probability per-sample
$$
\sum_{i=1}^K p_k\log{\frac{p_k}{q_k}}
$$


```python
y_true = [[0, 1], [0, 0]] 
y_pred = [[0.6, 0.4], [0.4, 0.6]]

m = metrics.KLDivergence()
m(y_true, y_pred)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=0.45814306>




```python
def KL(P, Q):
    epsilon = 1e-4

    P = np.array(P) + epsilon
    Q = np.array(Q) + epsilon

    return np.sum(P*np.log(P/Q))

KL(y_true, y_pred) # hmm
```




    0.9136630059540092




```python
t = np.clip(y_true, 1e-4, 1)
p = np.clip(y_pred, 1e-4, 1)

np.sum(t * np.log(t/p), axis=-1) # so then what exactly is Tensorflow's KL divergence doing?!
```




    array([ 0.91542078, -0.00169936])



# Summary

This summarizes the metric name and input formats for the metris listed:

| Metric | True Format | Predicted Format |   
| --- | --- | --- |
| Accuracy | List of classes | List of classes |  
| Binary Accuracy | List of classes | List of class probabilities | 
| Categorical Accuracy | OHE vector of classes | Vector of class probabilites per-sample |  
| Sparse Categorical Accuracy | List of classes (not OHE) | Vector of class probabilites per-sample |  
| Binary Cross Entropy | List of classes | List of class probabilities |  
| Categorical Cross Entropy | OHE vector of classes | Vector of class probabilities |  
| Sparse Categorical Cross Entropy | List of classes | Vector of class probabilities |  
| MAE | List of classes | List of classes |  
| MSE | List of classes | List of classes |  
| MAPE | List of classes | List of classes |  
| Precision | List of binary labels | List of probabilities |  
| Recall | List of binary labels | List of probabilities |  
| KL Divergence | ??? | ??? |  

The KL divergence in `keras` still eludes me, hopefully I should be able to make more sense of it and update it in the future.  

