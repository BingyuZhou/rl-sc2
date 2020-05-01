# Boilerplate Code for DRL with Tensorboard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from buffer import Buffer
from model import Actor_Critic
from utils import indToXY, XYToInd
from constants import *

from log import train_summary_writer

import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import numpy as np
from absl import app, flags
import sys

from pysc2.env import sc2_env
from pysc2.lib import actions, features
from sc2env_wrapper import SC2EnvWrapper
from pysc2.lib.named_array import NamedNumpyArray

tf.keras.backend.set_floatx("float32")


# args
FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "CollectMineralShards", "Select which map to play.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum(
    "difficulty",
    "very_easy",
    sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
    "If agent2 is a built-in Bot, it's strength.",
)
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")


def preprocess(obs):
    """Seperate raw obs(NamedDict) to numpy array objects for sake of Autograph"""
    # Currently, not every obs is used!!

    # combine both races
    home_away_race = np.concatenate(
        [obs.home_race_requested, obs.away_race_requested], axis=0
    )

    # FIXME: boolen vector of upgrades, size is unknown, assume 20
    upgrades_bool_vec = np.zeros(20, dtype="float32")
    upgrades_bool_vec[obs.upgrades] = 1

    available_act_bool_vec = np.zeros(NUM_ACTION_FUNCTIONS, dtype="float32")
    available_act_bool_vec[obs.available_actions] = 1
    # HACK: intentionally turn off control group, select_unit !!
    available_act_bool_vec[[4, 5]] = 0

    minimap_feature = np.moveaxis(obs.feature_minimap, 0, -1)  # NCWH -> NWHC

    return (
        obs.player.astype("float32"),
        home_away_race,
        upgrades_bool_vec,
        available_act_bool_vec,
        minimap_feature,
    )


def get_mask(action_id, action_spec):
    """Get action arguments mask"""
    mask = np.zeros(len(actions.TYPES), dtype="float32")
    for arg_type in action_spec.functions[action_id].args:
        mask[arg_type.id] = 1
    return mask


def translateActionToSC2(arg_spatial, arg_nonspatial, width, height):
    """Translate Tensor action ouputs to pysc2 readable objects"""
    args_all = [[arg.numpy().item()] for arg in arg_nonspatial]
    args_spatial = [indToXY(arg.numpy().item(), width, height) for arg in arg_spatial]

    args_all.extend(args_spatial)
    return args_all


# run one policy update
def train(env_name, batch_size, minibatch_size, epochs):
    """
    Main learning function

    Args:
        batch_size: size of the buffer, may have multiple trajecties inside
        minibatch_size: one batch is seperated into several minibatches. Each has this size.
        epochs: in one epoch, buffer is fully filled, and trained multiple times with minibatches.
    """
    actor_critic = Actor_Critic()

    # set env
    with SC2EnvWrapper(
        map_name=env_name,
        players=[sc2_env.Agent(sc2_env.Race.random)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_minimap=MINIMAP_RES, feature_screen=1
        ),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=FLAGS.game_steps_per_episode,
        disable_fog=FLAGS.disable_fog,
    ) as env:
        actor_critic.set_act_spec(env.action_spec()[0])  # assume one agent

        def train_one_epoch(step, tracing_on):
            # initialize replay buffer
            buffer = Buffer(batch_size, minibatch_size, MINIMAP_RES, MINIMAP_RES)

            # initial observation
            timestep = env.reset()
            step_type, reward, _, obs = timestep[0]
            obs = preprocess(obs)

            # fill in recorded trajectories
            while True:
                tf_obs = (
                    tf.constant(each_obs, shape=(1, *each_obs.shape))
                    for each_obs in obs
                )

                # print("computing action ...")
                val, act_id, arg_spatial, arg_nonspatial, logp_a = actor_critic.step(
                    *tf_obs
                )

                sc2act_args = translateActionToSC2(
                    arg_spatial, arg_nonspatial, MINIMAP_RES, MINIMAP_RES
                )

                act_mask = get_mask(act_id.numpy().item(), actor_critic.action_spec)
                buffer.add(
                    *obs,
                    act_id.numpy().item(),
                    sc2act_args,
                    act_mask,
                    logp_a.numpy().item(),
                    val.numpy().item()
                )
                step_type, reward, _, obs = env.step(
                    [actions.FunctionCall(act_id.numpy().item(), sc2act_args)]
                )[0]
                buffer.add_rew(reward)
                obs = preprocess(obs)

                if step_type == step_type.LAST or buffer.is_full():
                    if step_type == step_type.LAST:
                        buffer.finalize(reward)
                    else:
                        # trajectory is cut off, bootstrap last state with estimated value
                        tf_obs = (
                            tf.constant(each_obs, shape=(1, *each_obs.shape))
                            for each_obs in obs
                        )
                        val, _, _, _, _ = actor_critic.step(*tf_obs)
                        buffer.finalize(val)

                    if buffer.is_full():
                        break

                    # respawn env
                    env.render(True)
                    timestep = env.reset()
                    _, _, _, obs = timestep[0]
                    obs = preprocess(obs)

            # train in minibatches
            buffer.post_process()
            buffer.shuffle()

            mb_loss = []
            for ind in range(batch_size // minibatch_size):
                (
                    player,
                    home_away_race,
                    upgrades,
                    available_act,
                    minimap,
                    act_id,
                    act_args,
                    act_mask,
                    logp,
                    val,
                    ret,
                    adv,
                ) = buffer.minibatch(ind)
                if tracing_on:
                    tf.summary.trace_on(graph=True, profiler=False)

                mb_loss.append(
                    actor_critic.train_step(
                        tf.constant(step, dtype=tf.int64),
                        player,
                        home_away_race,
                        upgrades,
                        available_act,
                        minimap,
                        act_id,
                        act_args,
                        act_mask,
                        logp,
                        val,
                        ret,
                        adv,
                    )
                )
                step += 1

                if tracing_on:
                    tracing_on = False
                    with train_summary_writer.as_default():
                        tf.summary.trace_export(name="train_step", step=0)

            batch_loss = np.mean(mb_loss)

            return batch_loss, buffer.batch_ret, buffer.batch_len

        num_train_per_epoch = batch_size // minibatch_size
        for i in range(epochs):
            if i == 0:
                tracing_on = True
            else:
                tracing_on = False
            batch_loss, batch_ret, batch_len = train_one_epoch(
                i * num_train_per_epoch, tracing_on
            )
            with train_summary_writer.as_default():
                tf.summary.scalar("batch/batch_ret", np.mean(batch_ret), step=i)
                tf.summary.scalar("batch/batch_len", np.mean(batch_len), step=i)
                tf.summary.scalar("loss/batch_loss", batch_loss, step=i)
            print("----------------------------")
            print(
                "epoch {0:2d} loss {1:.3f} batch_ret {2:.3f} batch_len {3:.3f}".format(
                    i, batch_loss, np.mean(batch_ret), np.mean(batch_len)
                )
            )
            print("----------------------------")


def main(argv):
    epochs = 10
    batch_size = 320
    minibatch_size = 32
    train(FLAGS.env_name, batch_size, minibatch_size, epochs)


if __name__ == "__main__":
    app.run(main)
