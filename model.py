import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import indToXY, XYToInd, compute_over_actions, entropy
from constants import *
from log import train_summary_writer

from pysc2.lib import features, actions


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
        self.optimizer = keras.optimizers.RMSprop(
            learning_rate=3e-4, rho=0.99, epsilon=1e-5
        )
        # self.optimizer = keras.optimizers.Adam(
        #     learning_rate=3e-5, beta_1=0, beta_2=0.99, epsilon=1e-5
        # )
        self.clip_range = 0.3
        self.v_coef = 0.5
        self.entropy_coef = 1e-3
        self.max_grad_norm = 1.0

        # upgrades
        self.embed_upgrads = keras.layers.Dense(64, activation="relu")
        # player (agent statistics)
        self.embed_player = keras.layers.Dense(64, activation="relu")
        # available_actions
        self.embed_available_act = keras.layers.Dense(64, activation="relu")
        # race_requested
        self.embed_race = keras.layers.Dense(64, activation="relu")
        # minimap feature
        self.embed_minimap = keras.layers.Conv2D(
            32, 1, padding="valid", activation="relu"
        )
        self.embed_minimap_2 = keras.layers.Conv2D(
            64, 4, 2, padding="same", activation="relu"
        )
        self.embed_minimap_3 = keras.layers.Conv2D(
            128, 3, 2, padding="same", activation="relu"
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
        self.core_fc = keras.layers.Dense(256, activation="relu", name="core_fc")
        # self.layer_norm = keras.layers.LayerNormalization()
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

        self.value = keras.layers.Dense(1, name="value_out")

    def set_act_spec(self, action_spec):
        self.action_spec = action_spec

    def call(
        self,
        player,
        home_away_race,
        upgrades,
        available_act,
        minimap,
        step=None,
        training=True,
    ):
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
            [embed_player, embed_race, embed_upgrades, embed_available_act], axis=-1
        )
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
                    one_hot = (
                        tf.cast(obs[:, :, :, ind : ind + 1], dtype=tf.float32) / 255.0
                    )

                out.append(one_hot)
            out = tf.concat(out, axis=-1)
            return out

        one_hot_minimap = one_hot_map(minimap)
        embed_minimap = self.embed_minimap(one_hot_minimap)
        if step is not None:
            with train_summary_writer.as_default():
                tf.summary.image(
                    "embed_minimap",
                    tf.transpose(embed_minimap[2:3, :, :, :], (3, 1, 2, 0)),
                    step=step,
                    max_outputs=5,
                )
                tf.summary.image(
                    "input_minimap",
                    one_hot_minimap[:, :, :, 29:30],
                    step=step,
                    max_outputs=5,
                )
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
        core_out = tf.concat([scalar_out_2d, embed_minimap], axis=-1, name="core")
        core_out_flat = self.flat(core_out)
        core_out_flat = self.core_fc(core_out_flat)
        # core_out_flat = self.layer_norm(core_out_flat)
        # core_out_flat = tf.nn.relu(core_out_flat)

        """
        Decision output
        """
        # value
        value_out = self.value(core_out_flat)
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
            "value": value_out,
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
        out = self.call(
            player, home_away_race, upgrades, available_act, minimap, training=False
        )

        for key in out:
            if key != "value" and key != "action_id":
                out[key] = tf.nn.log_softmax(out[key], axis=-1)

        # EPS is used to avoid log(0) =-inf
        out["action_id"] = tf.math.softmax(out["action_id"]) * available_act
        # renormalize
        out["action_id"] /= EPS + tf.reduce_sum(
            out["action_id"], axis=-1, keepdims=True
        )
        out["action_id"] = tf.math.log(tf.maximum(out["action_id"], EPS))

        action_id = tf.random.categorical(out["action_id"], 1)
        while tf.less_equal(available_act[:, action_id.numpy().item()], 0.9):
            action_id = tf.random.categorical(out["action_id"], 1)

        # Fill out args based on sampled action type
        arg_spatial = []
        arg_nonspatial = []
        logp_a = tf.reduce_sum(
            out["action_id"] * tf.one_hot(action_id, depth=NUM_ACTION_FUNCTIONS),
            axis=-1,
        )
        tf.debugging.check_numerics(
            logp_a,
            "Bad logp(a|s) {0}\n {1}\n {2}".format(
                action_id, available_act[:, action_id.numpy().item()], out["action_id"]
            ),
        )

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
        tf.debugging.check_numerics(logp_a, "Bad logp(a|s)")

        return (
            out["value"],
            action_id,
            arg_spatial,
            arg_nonspatial,
            logp_a,
        )

    def logp_a(self, action_ids, action_args, action_mask, available_act, pi):
        """logp(a|s)"""
        # from logits to logp
        logp_pi = {}
        for key in pi:
            if key != "value":
                logp_pi[key] = tf.nn.log_softmax(pi[key], axis=-1)

        # action function id prob
        assert len(pi["action_id"].shape) == 2
        pi_act = tf.nn.softmax(pi["action_id"]) * available_act
        pi_act /= tf.reduce_sum(pi_act, axis=-1, keepdims=True)
        logpi_act = tf.math.log(tf.maximum(pi_act, EPS))
        logp = tf.reduce_sum(
            logpi_act * tf.one_hot(action_ids, depth=NUM_ACTION_FUNCTIONS), axis=-1,
        )
        # args
        assert len(action_args.shape) == 2
        for arg_type in actions.TYPES:
            if arg_type.name in ["screen", "screen2", "minimap"]:
                logp_pi = tf.nn.log_softmax(pi["target_location"], axis=-1)
                tf.debugging.check_numerics(logp_pi, "Bad logp(a|s)")

                logp += (
                    tf.reduce_sum(
                        logp_pi
                        * tf.one_hot(
                            action_args[:, arg_type.id], depth=MINIMAP_RES * MINIMAP_RES
                        ),
                        axis=-1,
                    )
                    * action_mask[:, arg_type.id]
                )

            if arg_type.name in pi.keys():
                logp_pi = tf.nn.log_softmax(pi[arg_type.name], axis=-1)
                tf.debugging.check_numerics(logp_pi, "Bad logp(a|s)")

                logp += (
                    tf.reduce_sum(
                        logp_pi
                        * tf.one_hot(
                            action_args[:, arg_type.id], depth=np.prod(arg_type.sizes),
                        ),
                        axis=-1,
                    )
                    * action_mask[:, arg_type.id]
                )

        return logp

    def loss(
        self,
        step,
        player,
        home_away_race,
        upgrades,
        available_act,
        minimap,
        act_id,
        act_args,
        act_mask,
        old_logp,
        old_v,
        ret,
        adv,
    ):
        # expection grad log
        out = self.call(
            player, home_away_race, upgrades, available_act, minimap, step=step
        )

        # new pi(a|s)
        logp = self.logp_a(act_id, act_args, act_mask, available_act, out)

        delta_pi = tf.exp(logp - old_logp)

        pg_loss_1 = delta_pi * adv
        pg_loss_2 = (
            tf.clip_by_value(delta_pi, 1 - self.clip_range, 1 + self.clip_range) * adv
        )

        pg_loss = -tf.reduce_mean(tf.minimum(pg_loss_1, pg_loss_2))

        v_clip = old_v + tf.clip_by_value(
            out["value"] - old_v, -self.clip_range, self.clip_range
        )
        v_clip_loss = tf.square(v_clip - ret)

        v_loss = tf.square(out["value"] - ret)
        v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_clip_loss, v_loss))

        approx_entropy = tf.reduce_mean(
            compute_over_actions(entropy, out, available_act, act_mask)
        )
        approx_kl = tf.reduce_mean(tf.square(old_logp - logp))
        clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(delta_pi - 1.0), self.clip_range), tf.float32)
        )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss/pg_loss", pg_loss, step)
            tf.summary.scalar("loss/v_loss", v_loss, step)
            tf.summary.scalar("stat/approx_entropy", approx_entropy, step)
            tf.summary.scalar("stat/approx_kl", approx_kl, step)
            tf.summary.scalar("stat/clip_frac", clip_frac, step)

        return pg_loss + self.v_coef * v_loss - self.entropy_coef * approx_entropy

    @tf.function
    def train_step(
        self,
        step,
        player,
        home_away_race,
        upgrades,
        available_act,
        minimap,
        act_id,
        act_args,
        act_mask,
        old_logp,
        old_v,
        ret,
        adv,
    ):
        tf.debugging.check_numerics(old_logp, "Bad old_logp")
        tf.debugging.check_numerics(old_v, "Bad old_v")
        tf.debugging.check_numerics(adv, "Bad adv")

        with tf.GradientTape() as tape:
            ls = self.loss(
                step,
                player,
                home_away_race,
                upgrades,
                available_act,
                minimap,
                act_id,
                act_args,
                act_mask,
                old_logp,
                old_v,
                ret,
                adv,
            )
        grad = tape.gradient(ls, self.trainable_variables)
        for g in grad:
            tf.debugging.check_numerics(g, "Bad grad")
        # clip grad (https://arxiv.org/pdf/1211.5063.pdf)
        grad, _ = tf.clip_by_global_norm(grad, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return ls
