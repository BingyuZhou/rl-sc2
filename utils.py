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


def entropy_naive(logits, mask=None):
    p = tf.nn.softmax(logits, axis=-1)
    if mask is not None:
        p *= mask
        p /= tf.reduce_sum(p, axis=-1, keepdims=True)

    logp = tf.math.log(p + EPS)
    ent = -tf.reduce_sum(logp * p, axis=-1)
    # normalize by actions available
    if mask is None:
        ent = ent / tf.math.log(tf.cast(logits.shape[-1], tf.float32))
    else:
        mask = tf.stop_gradient(mask)
        ent = ent / tf.math.log(tf.cast(tf.math.count_nonzero(mask), tf.float32))

    tf.debugging.check_numerics(ent, "bad entropy {}".format(ent))

    return ent


def entropy(policy_logits, mask=None):
    a0 = policy_logits - tf.reduce_max(policy_logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p = ea0 / z0
    if mask is not None:
        p *= mask
    ent = tf.reduce_sum(p * (tf.math.log(z0) - a0), axis=1)
    # normalize by actions available
    # if mask is None:
    #     ent = ent / tf.math.log(tf.cast(policy_logits.shape[-1], tf.float32))
    # else:
    #     mask = tf.stop_gradient(mask)
    #     ent = ent / tf.math.log(
    #         tf.cast(tf.math.count_nonzero(mask), tf.float32)
    #     )
    ent = ent / tf.math.log(tf.cast(policy_logits.shape[-1], tf.float32))
    tf.debugging.check_numerics(ent, "bad entropy {}".format(ent))

    return ent


def compute_over_actions(func, out_logits, available_act_mask, act_arg_mask):
    # action_id
    # available_action_logits = apply_action_mask(
    #     out_logits["action_id"], available_act_mask
    # )

    # tf.debugging.assert_equal(
    #     available_action_logits.shape, out_logits["action_id"].shape
    # )

    ent = func(out_logits["action_id"], mask=available_act_mask)

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


def gumbel_sample(logits, mask=None):
    """Gumbel max sample

    Args:
        mask: boolen mask. 1 means valid, 0 means invalid.
    """

    noise = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
    prob = logits - tf.math.log(-tf.math.log(noise))
    if mask is not None:
        mask_2 = tf.where(mask > 0, 0.0, -np.inf)
        prob += mask_2  # FIXME
    return tf.argmax(prob, axis=-1)


def categorical_sample(logits, mask=None, temp=None):
    if mask is not None:
        logits = tf.where(mask > 0, logits, -1e5)

    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
