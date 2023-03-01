import math

import tensorflow as tf


# GELU https://arxiv.org/pdf/1606.08415.pdf
# pytorch approximation
def GELU(x):
    pi = tf.constant(math.pi)
    return (
        0.5
        * x
        * (1 + tf.math.tanh(tf.math.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x, 3))))
    )
