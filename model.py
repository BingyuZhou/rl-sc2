import tensorflow as tf
from tensorflow import keras
import numpy as np

from pysc2.lib import features, actions

""" constants"""
NUM_ACTION_FUNCTIONS = 573
EPS = 1e-8


class GLU(keras.Model):
    """Gated linear unit"""

    def __init__(self, input_size, out_size):
        super(GLU, self).__init__(name="GLU")
        self.input_size = input_size
        self.out_size = out_size
        self.layer1 = keras.layers.Dense(input_size, activation="sigmoid")
        self.layer2 = keras.layers.Dense(out_size)

    def call(self, input, context):
        x = self.layer1(context)  # gate
        x = x * input  # gated input
        x = self.layer2(x)
        return x


class Actor_Critic(keras.Model):
    def __init__(self):
        super(Actor_Critic, self).__init__(name="ActorCritic")

        # upgrades
        self.embed_upgrads = keras.layers.Dense(64, activation="tanh")
        # player (agent statistics)
        self.embed_player = keras.layers.Dense(64, activation="relu")
        # available_actions
        self.embed_available_act = keras.layers.Dense(64, activation="relu")
        # race_requested
        self.embed_race = keras.layers.Dense(64, activation="relu")
        # minimap feature
        self.embed_minimap = keras.layers.Conv2D(
            32, 1, padding="same", activation="relu"
        )
        self.embed_minimap_2 = keras.layers.Conv2D(
            64, 3, padding="same", activation="relu"
        )
        self.embed_minimap_3 = keras.layers.Conv2D(
            128, 3, padding="same", activation="relu"
        )
        # screen feature
        # self.embed_screen = keras.layers.Conv2D(32,
        #                                         1,
        #                                         padding='same',
        #                                         activation=tf.nn.relu)
        # self.embed_screen_2 = keras.layers.Conv2D(64,
        #                                           3,
        #                                           padding='same',
        #                                           activation=tf.nn.relu)
        # self.embed_screen_3 = keras.layers.Conv2D(128,
        #                                           3,
        #                                           padding='same',
        #                                           activation=tf.nn.relu)
        # core
        self.flat = keras.layers.Flatten(name="core_flatten")
        """
        Output
        """
        # TODO: autoregressive embedding
        self.action_id_layer = keras.layers.Dense(256, name="action_id_out")
        self.action_id_gate = GLU(input_size=256, out_size=NUM_ACTION_FUNCTIONS)
        # self.delay_logits = keras.layers.Dense(128, name="delay_out")
        self.queued_logits = keras.layers.Dense(2, name="queued_out")
        self.select_point_logits = keras.layers.Dense(4, name="select_point_out")
        self.select_add_logits = keras.layers.Dense(2, name="select_add_out")
        # self.select_unit_act=keras.layers.Dense(4)
        # self.selec_unit_id_logits=keras.layers.Dense(64)
        self.select_worker_logits = keras.layers.Dense(4, name="select_worker_out")
        # self.target_unit_logits = keras.layers.Dense(32, name="target_unit_out")
        self.target_location_flat = keras.layers.Flatten(name="target_location_flatten")
        self.target_location_logits = keras.layers.Conv2D(
            1, 1, padding="same", name="target_location_out"
        )

        # self.value = keras.layers.Dense(1, name="value_out")

    def set_act_spec(self, action_spec):
        self.action_spec = action_spec

    def call(self, player, home_away_race, upgrades, available_act, minimap):
        """
        Embedding of inputs
        """
        """ 
        Scalar features
        
        These are embedding of scalar features
        """
        embed_player = self.embed_player(tf.math.log(player + 1))

        embed_race = self.embed_race(
            tf.reshape(tf.one_hot(home_away_race, depth=4), shape=[-1, 8])
        )

        embed_upgrades = self.embed_upgrads(upgrades)
        embed_available_act = self.embed_available_act(available_act)

        scalar_out = tf.concat(
            [embed_player, embed_race, embed_upgrades, embed_available_act], axis=1
        )
        # print("scalar_out: {}".format(scalar_out.shape))
        """ 
        Map features 
        
        These are embedding of map features
        """

        def one_hot_map(obs, screen_on=False):
            assert len(obs.shape) == 4

            if screen_on:
                Features = features.SCREEN_FEATURES
            else:
                Features = features.MINIMAP_FEATURES
            out = []
            for ind, feature in enumerate(Features):
                if feature.type is features.FeatureType.CATEGORICAL:
                    one_hot = tf.one_hot(obs[:, :, :, ind], depth=feature.scale)
                else:  # features.FeatureType.SCALAR
                    one_hot = tf.cast(obs[:, :, :, ind:], dtype=tf.float32) / 255.0

                out.append(one_hot)
            out = tf.concat(out, axis=-1)
            return out

        one_hot_minimap = one_hot_map(minimap)
        embed_minimap = self.embed_minimap(one_hot_minimap)
        # embed_minimap = self.embed_minimap_2(embed_minimap)
        # embed_minimap = self.embed_minimap_3(embed_minimap)

        # one_hot_screen = one_hot_map(obs.feature_screen, screen_on=True)
        # embed_screen = self.embed_screen(one_hot_screen)
        # embed_screen = self.embed_screen_2(embed_screen)
        # embed_screen = self.embed_screen_3(embed_screen)
        # map_out = tf.concat([embed_minimap, embed_screen], axis=-1)

        # TODO: entities feature
        """
        State representation
        """
        # core
        scalar_out_2d = tf.tile(
            tf.expand_dims(tf.expand_dims(scalar_out, 1), 2),
            [1, embed_minimap.shape[1], embed_minimap.shape[2], 1],
        )
        core_out = tf.concat([scalar_out_2d, embed_minimap], axis=3, name="core")
        core_out_flat = self.flat(core_out)
        """
        Decision output
        """
        # value
        # value_out = self.value(core_out_flat)
        # action id
        action_id_out = self.action_id_layer(core_out_flat)
        action_id_out = self.action_id_gate(action_id_out, embed_available_act)
        # delay
        # delay_out = self.delay_logits(core_out_flat)

        # queued
        queued_out = self.queued_logits(core_out_flat)
        # selected units
        select_point_out = self.select_point_logits(core_out_flat)

        select_add_out = self.select_add_logits(core_out_flat)

        select_worker_out = self.select_worker_logits(core_out_flat)
        # target unit
        # target_unit_out = self.target_unit_logits(core_out_flat)
        # target location
        target_location_out = self.target_location_logits(core_out)
        (
            _,
            self.location_out_width,
            self.location_out_height,
            _,
        ) = target_location_out.shape

        target_location_out = self.target_location_flat(target_location_out)

        out = {
            # "value": value_out,
            "action_id": action_id_out,
            "queued": queued_out,
            "select_point_act": select_point_out,
            "select_add": select_add_out,
            "select_worker": select_worker_out,
            "target_location": target_location_out,
        }

        return out

    def step(self, player, home_away_race, upgrades, available_act, minimap):
        """Sample actions and compute logp(a|s)"""
        out = self.call(player, home_away_race, upgrades, available_act, minimap)

        # EPS is used to avoid log(0) =-inf
        available_act_mask = (
            tf.ones(NUM_ACTION_FUNCTIONS, dtype=np.float32) * EPS + available_act
        )
        out["action_id"] = tf.math.softmax(out["action_id"]) * available_act_mask
        # renormalize
        out["action_id"] /= tf.reduce_sum(out["action_id"], axis=-1, keepdims=True)
        out["action_id"] = tf.math.log(out["action_id"])

        action_id = tf.random.categorical(out["action_id"], 1)

        # Fill out args based on sampled action type
        arg_spatial = []
        arg_nonspatial = []
        logp_a = tf.reduce_sum(
            out["action_id"] * tf.one_hot(action_id, depth=NUM_ACTION_FUNCTIONS),
            axis=-1,
        )

        # FIXME: how to get value in tf.function

        for arg_type in self.action_spec.functions[action_id.numpy().item()].args:
            if arg_type.name in ["screen", "screen2", "minimap"]:
                location_id = tf.random.categorical(out["target_location"], 1)
                arg_spatial.append(location_id)

                logp_a += tf.reduce_sum(
                    out["target_location"]
                    * tf.one_hot(
                        location_id,
                        depth=self.location_out_width * self.location_out_height,
                    ),
                    axis=-1,
                )
            else:
                # non-spatial args
                sample = tf.random.categorical(out[arg_type.name], 1)
                arg_nonspatial.append(sample)
                logp_a += tf.reduce_sum(
                    out[arg_type.name] * tf.one_hot(sample, depth=arg_type.sizes[0]),
                    axis=-1,
                )

        return (
            action_id,
            arg_spatial,
            arg_nonspatial,
            logp_a,
        )

    def logp_a(self, action_ids, action_args, action_mask, pi, action_spec):
        """logp(a|s)"""
        # from logits to logp
        logp_pi = {}
        for key in pi:
            if key != "value":
                logp_pi[key] = tf.nn.log_softmax(pi[key], axis=-1)

        # action function id prob
        assert len(pi["action_id"].shape) == 2
        logp = tf.reduce_sum(
            logp_pi["action_id"] * tf.one_hot(action_ids, depth=NUM_ACTION_FUNCTIONS),
            axis=-1,
        )
        # args
        assert len(action_args.shape) == 2
        logp_args = []
        for ind, arg_type in enumerate(actions.TYPES):
            if arg_type.name in ["screen", "screen2", "minimap"]:
                logp_args.append(
                    tf.reduce_sum(
                        logp_pi["target_location"]
                        * tf.one_hot(action_args[:, ind], depth=32 * 32),
                        axis=-1,
                    )
                )
            else:
                if arg_type.name in logp_pi.keys():
                    logp_args.append(
                        tf.reduce_sum(
                            logp_pi[arg_type.name]
                            * tf.one_hot(
                                action_args[:, ind], depth=np.prod(arg_type.sizes)
                            ),
                            axis=-1,
                        )
                    )
                else:
                    logp_args.append(tf.constant([0] * logp.shape[0], dtype=tf.float32))
        # mask out unused args for each sampled action id
        logp += tf.reduce_sum(action_mask * tf.stack(logp_args, axis=-1), axis=-1)

        return logp
        # logp_args = []
        # for batch_ind, action_id in enumerate(action_ids):
        #     logp_args_ind = [0]
        #     for ind, arg_type in enumerate(
        #             action_spec.functions[action_id].args):
        #         if arg_type.name in ['screen', 'screen2', 'minimap']:
        #             location_id = action_args[batch_ind][ind][
        #                 1] * self.location_out_width + action_args[batch_ind][
        #                     ind][0]
        #             logp_args_ind += tf.reduce_sum(
        #                 logp_pi['target_location'][batch_ind] *
        #                 tf.one_hot(location_id,
        #                            depth=self.location_out_width *
        #                            self.location_out_height),
        #                 axis=-1)
        #         else:
        #             # non-spatial args
        #             logp_args_ind += tf.reduce_sum(
        #                 logp_pi[arg_type.name][batch_ind] *
        #                 tf.one_hot(action_args[batch_ind][ind],
        #                            depth=arg_type.sizes[0]),
        #                 axis=-1)
        #     logp_args.append(logp_args_ind)

        # return logp + tf.squeeze(tf.stack(logp_args))

    def loss(
        self,
        player,
        home_away_race,
        upgrades,
        available_act,
        minimap,
        batch_size,
        act_id,
        act_args,
        act_mask,
        ret,
        action_spec,
    ):
        # expection grad log
        out = self.call(player, home_away_race, upgrades, available_act, minimap)

        logp = self.logp_a(act_id, act_args, act_mask, out, action_spec)

        return -tf.reduce_mean(logp * ret)
