import numpy as np
import tensorflow as tf


def indToXY(id, width, height):
    """Index to (x,y) location"""
    # Observation map is y-major coordinate
    x, y = id % width, id // width
    return [x, y]


def XYToInd(location, width, height):
    """Location (x,y) to index"""
    return location[1] * width + location[0]


def count_vars(trainable_var):
    """Count trainable variables"""
    return sum([np.prod(var.shape.as_list()) for var in trainable_var])


def entropy(logp):
    p = tf.exp(logp)
    return -tf.reduce_sum(p * logp)
