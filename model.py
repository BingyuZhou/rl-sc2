import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import (
    indToXY,
    XYToInd,
    compute_over_actions,
    entropy,
    log_prob,
    gumbel_sample,
    categorical_sample,
)
from constants import *
from log import train_summary_writer
from hparams import *

from pysc2.lib import features, actions

"""
Tow ways to handle available action：
1. Manually apply available action mask at the end of action sampling step. Agent itself doesn't learn from the available action info.
2. [DeepMind] Encoding available action into the model. Using GLU to learn which actions are available. No mask in the action sampling step.
"""


class GLU(keras.Model):
    """Gated linear unit
    [DeepMind] method to learn which actions types are availble
    """

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
    def __init__(self, hparam):
        super(Actor_Critic, self).__init__(name="ActorCritic")
        # self.optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.95)
        print(hparam)
        self.optimizer = keras.optimizers.Adam(learning_rate=hparam[HP_LR])
        self.clip_ratio = hparam[HP_CLIP]
        self.clip_value = hparam[HP_CLIP_VALUE]
        self.v_coef = 0.1
        self.entropy_coef = hparam[HP_ENTROPY_COEF]
        self.max_grad_norm = hparam[HP_GRADIENT_NORM]

        # upgrades
        # self.embed_upgrads = keras.layers.Dense(
        #     64, activation="relu", name="embed_upgrads"
        # )
        # player (agent statistics)
        self.embed_player = keras.layers.Dense(
            64, activation="relu", name="embed_player"
        )
        # available_actions
        self.embed_available_act = keras.layers.Dense(
            64, activation="relu", name="embed_available_act"
        )
        # race_requested
        # self.embed_race = keras.layers.Dense(64, activation="relu", name="embed_race")
        # minimap feature
        self.embed_minimap = keras.layers.Conv2D(
            32, 1, padding="same", activation="relu", name="embed_minimap"
        )
        self.embed_minimap_2 = keras.layers.Conv2D(
            64, 2, padding="same", activation="relu"
        )
        # self.embed_minimap_3 = keras.layers.Conv2D(
        #     64, 3, padding="same", activation="relu"
        # )
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
        self.action_id_layer = keras.layers.Dense(
            NUM_ACTION_FUNCTIONS, name="action_id_out"
        )
        # self.action_id_gate = GLU(input_size=256, out_size=NUM_ACTION_FUNCTIONS)
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
        # home_away_race,
        # upgrades,
        available_act,
        minimap,
        step=None,
    ):
        """
        Embedding of inputs
        """
        """ 
        Scalar features
        
        These are embedding of scalar features
        """
        player = tf.stop_gradient(player)
        available_act = tf.stop_gradient(available_act)
        minimap = tf.stop_gradient(minimap)

        embed_player = self.embed_player(tf.stop_gradient(tf.math.log(player + 1.0)))

        # embed_race = self.embed_race(
        #     tf.reshape(tf.one_hot(home_away_race, depth=4), shape=[-1, 8])
        # )

        # embed_upgrades = self.embed_upgrads(upgrades)
        embed_available_act = self.embed_available_act(available_act)

        scalar_out = tf.concat([embed_player, embed_available_act], axis=-1)
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

        one_hot_minimap = tf.stop_gradient(one_hot_map(minimap))
        embed_minimap = self.embed_minimap(one_hot_minimap)
        embed_minimap = self.embed_minimap_2(embed_minimap)
        # embed_minimap = self.embed_minimap_3(embed_minimap)

        if step is not None:
            with train_summary_writer.as_default():
                tf.summary.image(
                    "embed_minimap",
                    tf.transpose(embed_minimap[2:3, :, :, :], (3, 1, 2, 0)),
                    step=step,
                    max_outputs=5,
                )
                # tf.summary.image(
                #     "input_minimap",
                #     one_hot_minimap[:, :, :, 29:30],
                #     step=step,
                #     max_outputs=5,
                # )
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

        """
        Decision output
        """
        # value
        value_out = self.value(core_out_flat)
        # action id
        action_id_out = self.action_id_layer(core_out_flat)
        # action_id_out = self.action_id_gate(action_id_out, embed_available_act)
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

    def step(self, player, home_away_race, upgrades, available_act_mask, minimap):
        """Sample actions and compute logp(a|s)"""
        out = self.call(player, available_act_mask, minimap)

        # Gumbel-max sampling
        action_id = categorical_sample(out["action_id"], available_act_mask)

        tf.assert_greater(available_act_mask[:, action_id.numpy().item()], 0.0)

        # Fill out args based on sampled action type
        arg_spatial = []
        arg_nonspatial = []

        logp_a = log_prob(action_id, out["action_id"])

        for arg_type in self.action_spec.functions[action_id.numpy().item()].args:
            if arg_type.name in ["screen", "screen2", "minimap"]:
                location_id = gumbel_sample(out["target_location"])
                arg_spatial.append(location_id)
                logp_a += log_prob(location_id, out["target_location"])
            else:
                # non-spatial args
                sample = gumbel_sample(out[arg_type.name])
                arg_nonspatial.append(sample)
                logp_a += log_prob(sample, out[arg_type.name])
        # tf.debugging.check_numerics(logp_a, "Bad logp(a|s)")

        return (
            out["value"],
            action_id,
            arg_spatial,
            arg_nonspatial,
            logp_a,
        )

    def logp_a(self, action_ids, action_args, action_mask, available_act_mask, pi):
        """logp(a|s)
        logp(a|s) = log (p1*p2*p3) = logp1 + logp2 + logp3

        """
        # action function id prob
        # available_action_logits = apply_action_mask(pi["action_id"], available_act_mask)
        logp = log_prob(action_ids, pi["action_id"])

        # tf.debugging.assert_shapes([(logp, (128,))])

        for arg_type in actions.TYPES:
            if arg_type.name in ["screen", "screen2", "minimap"]:
                action_log_prob = log_prob(
                    action_args[:, arg_type.id], pi["target_location"]
                )
                logp += action_log_prob * action_mask[:, arg_type.id]

            if arg_type.name in pi.keys():
                action_log_prob = log_prob(
                    action_args[:, arg_type.id], pi[arg_type.name]
                )
                logp += action_log_prob * action_mask[:, arg_type.id]

        return logp

    def loss(
        self,
        step,
        player,
        # home_away_race,
        # upgrades,
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
        out = self.call(player, available_act, minimap, step=step)

        # new pi(a|s)
        logp = self.logp_a(act_id, act_args, act_mask, available_act, out)

        delta_pi = tf.exp(logp - old_logp)

        pg_loss_1 = delta_pi * adv
        pg_loss_2 = (
            tf.clip_by_value(delta_pi, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        )
        # expection grad log
        pg_loss = -tf.reduce_mean(tf.minimum(pg_loss_1, pg_loss_2))

        if self.clip_value > 0:
            v_clip = old_v + tf.clip_by_value(
                out["value"] - old_v, -self.clip_value, self.clip_value
            )
            v_clip_loss = tf.square(v_clip - ret)

            v_loss = tf.square(out["value"] - ret)
            v_loss = tf.reduce_mean(tf.maximum(v_clip_loss, v_loss))
        else:
            v_loss = tf.reduce_mean(tf.square(out["value"] - ret))

        approx_entropy = tf.reduce_mean(
            compute_over_actions(entropy, out, available_act, act_mask), name="entropy"
        )
        tf.debugging.check_numerics(approx_entropy, "bad entropy")
        approx_kl = tf.reduce_mean(tf.square(old_logp - logp), name="kl")
        clip_frac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(delta_pi - 1.0), self.clip_ratio), tf.float32),
            name="clip_frac",
        )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss/pg_loss", pg_loss, step)
            tf.summary.scalar("loss/v_loss", v_loss, step)
            tf.summary.scalar("loss/approx_entropy", approx_entropy, step)
            tf.summary.scalar("stat/approx_kl", approx_kl, step)
            tf.summary.scalar("stat/clip_frac", clip_frac, step)

        return pg_loss + self.v_coef * v_loss - self.entropy_coef * approx_entropy

    @tf.function
    def train_step(
        self,
        step,
        player,
        # home_away_race,
        # upgrades,
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
        with tf.GradientTape() as tape:
            ls = self.loss(
                step,
                player,
                # home_away_race,
                # upgrades,
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
        with train_summary_writer.as_default():
            norm_tmp = [tf.norm(g) for g in grad]
            tf.summary.scalar("batch/gradient_norm", tf.reduce_mean(norm_tmp), step)
            tf.summary.scalar("batch/gradient_norm_max", tf.reduce_max(norm_tmp), step)

        for g in grad:
            tf.debugging.check_numerics(g, "Bad grad {}".format(g))
        # clip grad (https://arxiv.org/pdf/1211.5063.pdf)
        if self.max_grad_norm > 0.0:
            grad, _ = tf.clip_by_global_norm(grad, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return ls
