import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from utils import XYToInd


class Buffer:
    """Replay buffer"""

    def __init__(self, width, height):
        """
        Obs
        """
        self.batch_player = []
        self.batch_home_away_race = []
        self.batch_upgrades = []
        self.batch_available_act = []
        self.batch_minimap = []

        self.batch_act_id = []  # batch action
        self.batch_act_args = []
        # This mask is useful to compute logp(a|s) for action args.
        # It makes the loss derivable for all variables!
        self.batch_act_masks = []
        self.batch_ret = []  # batch return
        self.batch_len = []  # batch trajectory length
        self.batch_logp = []  # batch logp(a|s)
        self.ep_rew = []  # episode rewards (trajectory rewards)
        self.ep_len = 0  # length of trajectory
        self.width = width
        self.height = height

    def add(
        self,
        player,
        home_away_race,
        upgrades,
        available_act,
        minimap,
        act_id,
        act_args,
        act_mask,
        logp_a,
        reward,
    ):
        """Add one entry"""
        self.batch_player.append(player)
        self.batch_home_away_race.append(home_away_race)
        self.batch_upgrades.append(upgrades)
        self.batch_available_act.append(available_act)
        self.batch_minimap.append(minimap)

        self.batch_act_id.append(act_id)
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
        self.ep_rew.append(reward)

    def finalize(self, reward):
        """Finalize one trajectory"""
        self.ep_rew.append(reward)
        ret = np.sum(self.ep_rew, dtype="float32")
        self.batch_ret += [ret] * self.ep_len
        self.batch_len.append(self.ep_len)

        # reset
        self.ep_len = 0
        self.ep_rew.clear()

    def size(self):
        return len(self.batch_ret)

    def sample(self):
        """Return buffer elements"""
        # fill args vector for better computation of logp
        args = np.zeros((self.size(), len(actions.TYPES)))

        args[
            np.nonzero(np.array(self.batch_act_masks, dtype=np.int8))
        ] = self.batch_act_args

        # TODO: In order to fulfill tf.function requirements for autograph,
        # we need to use tf.Tensor object as interface.
        # Thus obs needs to be seperated for each feature entity
        return (
            tf.constant(self.batch_player),
            tf.constant(self.batch_home_away_race),
            tf.constant(self.batch_upgrades),
            tf.constant(self.batch_available_act),
            tf.constant(self.batch_minimap),
            tf.constant(self.batch_act_id),
            tf.constant(args, dtype=tf.int32),
            tf.constant(self.batch_act_masks),
            tf.constant(self.batch_ret),
        )
