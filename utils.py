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


def entropy(policy_logits, is_masked=False):
    policy = tf.nn.softmax(policy_logits)
    log_policy = tf.nn.log_softmax(policy_logits)
    ent = -tf.reduce_sum(policy * log_policy, axis=-1)
    tf.debugging.check_numerics(ent, "bad orig ent")
    # normalize by actions available
    normalized_ent = ent / tf.math.log(tf.cast(policy_logits.shape[-1], tf.float32))

    tf.debugging.check_numerics(
        normalized_ent, "bad entropy {0} {1}".format(log_policy, ent)
    )

    return normalized_ent


def compute_over_actions(func, out_logits, available_act_mask, act_arg_mask):
    # action_id
    policy_logits = out_logits["action_id"] * available_act_mask

    ent = func(policy_logits, is_masked=True)

    for arg in actions.TYPES:
        if arg.name in out_logits.keys():
            # tf.print(out_logits[arg.name].shape, arg.sizes, arg.name)
            ent += func(out_logits[arg.name]) * act_arg_mask[:, arg.id]
        if arg.name in ["screen", "screen2", "minimap"]:
            ent += func(out_logits["target_location"]) * act_arg_mask[:, arg.id]
    return ent


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    
    From OpenAI baseline code.
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def log_prob(label, logits):
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(label, logits)
