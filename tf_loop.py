'''
This overviews the TensorFlow custom training loop in its (what I think is) most general sense. Four steps:
    1. Define Model
    2. Define Metrics, Optimizer and Losses
    3. Define train and test (validation) functions
    4. Write training loop
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# STEP 0 - Set up datasets

'''
---------------------------------------------------------
STEP 1 - Define Model 

Define inputs, outputs and wrap using keras
'''
inputs = ...
outputs = ...
model = keras.Model(inputs=inputs, outputs=outputs)

'''
---------------------------------------------------------
STEP 2 - Define Metrics, Optimizer and Losses

Use keras.metrics.Metric and keras.optimizers
Can subclass if necessary

'''
train_metric = keras.metrics...
val_metric = keras.metrics...

optimizer = keras.optimizers...

loss_fn = keras.losses...


'''
---------------------------------------------------------
STEP 3 - Define training and test functions 

both take inputs and labels
both return a loss value

training invoke tape and applies loss gradient to weights
test just finds loss value

'''

@tf.function
def train_step(input, labels):
    # invoke GradientTape()
    with tf.GradientTape() as tape:
        # find predicted
        pred = model(input, training=True) 
        # calculate loss
        loss_value = loss_fn(labels, pred)
        loss_value += sum(model.losses)

    # find gradient loss and weights
    grads = tape.gradient(loss_value, model.trainable_weights)
    # apply gradients to update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metric
    train_metric.update_state(labels, pred)

    return loss_value

@tf.function
def test_step(input, labels):
    # find predicted
    pred = model(input, training=False)
    # update metric
    val_metric.update_state(labels, pred)


'''
---------------------------------------------------------
STEP 4 - Training/Validation Loop

Remember to reset metric states

'''

for epoch in range(epochs):
    # training loop 
    for step, (in_batch_train, label_batch_train) in enumerate(train_dataset):
        loss_value = train_step(in_batch_train, label_batch_train)

    # print metrics
    print(train_metric.result())
    train_metric.reset_states()

    # validation loop 
    for step, (in_batch_val, label_batch_val) in enumerate(val_dataset):
        test_step(in_batch_val, label_batch_val)

    # print metrics
    print(val_metric.result())
    val_metric.reset_states()
