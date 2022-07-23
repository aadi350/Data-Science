import pdb
import numpy as np  
# import tensorflow as tf


def function(in_1, in_2):
    in_1_transformed = nested_function(in_1) 
    pdb.set_trace()
    return in_1 + in_2

def nested_function(in_1):
    pdb.set_trace()
    return in_1**2

a = 2 
b = 5
pdb.set_trace()

c = function(a, b)