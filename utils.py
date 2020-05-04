import numpy as np
import tensorflow as tf

from constants import EPS
from pysc2.lib import actions


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


def entropy(prob, is_masked=False):
    log_p = tf.math.log(tf.maximum(prob, EPS))
    ent = -tf.reduce_sum(prob * log_p, axis=-1)
    # normalize by actions available
    if is_masked:
        normalized_ent = ent / tf.math.log(
            tf.cast(tf.math.count_nonzero(prob), tf.float32)
        )
    else:
        normalized_ent = ent / tf.math.log(tf.cast(prob.shape[-1], tf.float32))

    return normalized_ent


def compute_over_actions(func, out_logits, available_act_mask, act_arg_mask):
    ent = tf.zeros((available_act_mask.shape[0], 1))
    # action_id
    p = tf.nn.softmax(out_logits["action_id"], axis=-1)
    p *= available_act_mask
    p /= tf.reduce_sum(p, axis=-1, keepdims=True)
    ent += func(p, is_masked=True)

    for arg in actions.TYPES:
        if arg.name in out_logits.keys():
            p = tf.nn.softmax(out_logits[arg.name], axis=-1)
            # tf.print(out_logits[arg.name].shape, arg.sizes, arg.name)
            ent += func(p) * act_arg_mask[:, arg.id]
        if arg.name in ["screen", "screen2", "minimap"]:
            p = tf.nn.softmax(out_logits["target_location"], axis=-1)
            ent += func(p) * act_arg_mask[:, arg.id]
    return ent
