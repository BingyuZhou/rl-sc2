import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from utils import XYToInd
import scipy
from constants import EPS


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    """Replay buffer"""

    def __init__(
        self,
        batch_size,
        minibatch_size,
        width,
        height,
        action_spec,
        gamma=0.99,
        lam=0.95,
    ):
        # obs
        self.batch_player = []
        self.batch_home_away_race = []
        self.batch_upgrades = []
        self.batch_available_act = []
        self.batch_minimap = []
        self.batch_screen = []

        self.batch_act_id = []  # batch action
        self.batch_act_args = []
        # This mask is useful to compute logp(a|s) for action args.
        # It makes the loss derivable for all variables!
        self.batch_act_masks = []
        self.batch_len = []  # batch trajectory length
        self.batch_logp = []  # batch logp(a|s)
        self.batch_ret = []  # rewards to go, used for value function
        self.batch_adv = []
        self.batch_vals = []
        self.ep_rew = []  # episode rewards (trajectory rewards)
        self.ep_vals = []  # episode estimated values
        self.ep_len = 0  # length of trajectory
        self.width = width
        self.height = height

        self.gamma = gamma  # discount
        self.lam = lam  # GAE-lambda

        self.count = 0
        self.batch_size = batch_size
        assert batch_size % minibatch_size == 0
        self.minibatch_size = minibatch_size
        self.action_spec = action_spec

    def add(
        self,
        player,
        home_away_race,
        upgrades,
        available_act,
        minimap,
        # screen,
        act_id,
        act_args,
        act_mask,
        logp_a,
        val,
    ):
        """Add one entry"""
        self.batch_player.append(player)
        self.batch_home_away_race.append(home_away_race)
        self.batch_upgrades.append(upgrades)
        self.batch_available_act.append(available_act)
        self.batch_minimap.append(minimap)
        # self.batch_screen.append(screen)

        self.batch_act_id.append(act_id)
        # reorder act_args so that its arguemnts order in increasing order
        arg_order = np.asarray([a.id for a in self.action_spec.functions[act_id].args])
        index_order = np.argsort(arg_order)
        act_args = np.asarray(act_args)[index_order]
        for arg in act_args:
            if len(arg) > 1:
                # tansfer spatial args to scalar
                self.batch_act_args.append(XYToInd(arg, self.width, self.height))
            else:
                # flatten list
                self.batch_act_args.append(arg[0])
        self.batch_act_masks.append(act_mask)
        self.batch_logp.append(logp_a)
        self.ep_len += 1
        self.ep_vals.append(val)
        self.batch_vals.append(val)

        self.count += 1

    def add_rew(self, rew):
        self.ep_rew.append(rew)

    def finalize(self, last_val):
        """Finalize one trajectory"""
        self.ep_vals.append(last_val)  # bootstrap
        # GAE
        # A(s,a) = r(s,a) + \gamma * v(s') - v(s)
        self.ep_rew = np.asarray(self.ep_rew, dtype="float32")
        self.ep_vals = np.asarray(self.ep_vals, dtype="float32")
        deltas = self.ep_rew + self.gamma * self.ep_vals[1:] - self.ep_vals[:-1]
        self.batch_adv.append(discount_cumsum(deltas, self.gamma * self.lam))

        self.batch_ret.append(discount_cumsum(self.ep_rew, self.gamma))

        self.batch_len.append(self.ep_len)

        # reset
        self.ep_len = 0
        self.ep_rew = []
        self.ep_vals = []

    def size(self):
        return self.count

    def is_full(self):
        return self.count == self.batch_size

    def shuffle(self):
        self.ind = np.arange(self.batch_size, dtype="int32")
        np.random.shuffle(self.ind)

    def post_process(self):
        """Return buffer elements"""
        # fill args vector for better computation of logp
        args = np.zeros((self.size(), len(actions.TYPES)), dtype="float32")

        args[
            np.nonzero(np.array(self.batch_act_masks, dtype=np.int8))
        ] = self.batch_act_args
        self.batch_act_args = args

        # concatenate adv and ret
        self.batch_adv = np.concatenate(self.batch_adv, axis=0)
        self.batch_ret = np.concatenate(self.batch_ret, axis=0)
        # adv normalization trick
        adv_mean, adv_std = np.mean(self.batch_adv), np.std(self.batch_adv)
        self.batch_adv = (self.batch_adv - adv_mean) / (adv_std + EPS)

    def minibatch(self, index):
        start, end = index * self.minibatch_size, (index + 1) * self.minibatch_size
        slices = self.ind[start:end]
        return (
            tf.constant(np.asarray(self.batch_player)[slices]),
            # tf.constant(np.asarray(self.batch_home_away_race)[slices]),
            # tf.constant(np.asarray(self.batch_upgrades)[slices]),
            tf.constant(np.asarray(self.batch_available_act)[slices]),
            tf.constant(np.asarray(self.batch_minimap)[slices]),
            # tf.constant(np.asarray(self.batch_screen)[slices]),
            tf.constant(np.asarray(self.batch_act_id)[slices], dtype=tf.int32),
            tf.constant(self.batch_act_args[slices], dtype=tf.int32),
            tf.constant(np.asarray(self.batch_act_masks)[slices]),
            tf.constant(np.asarray(self.batch_logp)[slices], dtype=tf.float32),
            tf.constant(np.asarray(self.batch_vals)[slices], dtype=tf.float32),
            tf.constant(self.batch_ret[slices], dtype=tf.float32),
            tf.constant(self.batch_adv[slices], dtype=tf.float32),
        )
