# Boilerplate Code for DRL with Tensorboard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from buffer import Buffer
from model import Actor_Critic

import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import numpy as np
import datetime
from absl import app, flags
import sys

from pysc2.env import sc2_env
from pysc2.lib import actions, features
from sc2env_wrapper import SC2EnvWrapper
from pysc2.lib.named_array import NamedNumpyArray

tf.keras.backend.set_floatx("float32")
""" log info"""
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/" + current_time + "/train"
test_log_dir = "logs/" + current_time + "/test"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
""" constants"""
NUM_ACTION_FUNCTIONS = 573
EPS = 1e-8
MINIMAP_RES = 32

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


def indToXY(id, width, height):
    """Index to (x,y) location"""
    # Observation map is y-major coordinate
    y, x = id % width, id // width
    return [x, y]


def XYToInd(location, width, height):
    """Location (x,y) to index"""
    return location[0] * width + location[1]


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
def train(env_name, batch_size, epochs):
    actor_critic = Actor_Critic()
    optimizer = keras.optimizers.Adam(learning_rate=0.0003)

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

        action_spec = env.action_spec()[0]  # assume one agent
        actor_critic.set_act_spec(action_spec)

        def train_one_epoch(tracing_on):
            # initialize replay buffer
            buffer = Buffer(MINIMAP_RES, MINIMAP_RES)

            # initial observation
            timeStepTuple = env.reset()
            step_type, reward, discount, obs = timeStepTuple[0]
            obs = preprocess(obs)
            # render first episode of each epoch
            render_env = True

            # fill in recorded trajectories
            while True:
                env.render(render_env)
                tf_obs = (
                    tf.constant(each_obs, shape=(1, *each_obs.shape))
                    for each_obs in obs
                )
                # print("computing action ...")
                act_id, arg_spatial, arg_nonspatial, logp_a = actor_critic.step(*tf_obs)

                sc2act_args = translateActionToSC2(
                    arg_spatial, arg_nonspatial, MINIMAP_RES, MINIMAP_RES
                )

                # print("buffer logging ...")
                act_mask = get_mask(act_id.numpy().item(), action_spec)
                buffer.add(
                    *obs, act_id.numpy().item(), sc2act_args, act_mask, logp_a, reward
                )
                # print("apply action in env ...")
                timeStepTuple = env.step(
                    [actions.FunctionCall(act_id.numpy().item(), sc2act_args)]
                )
                step_type, reward, discount, obs = timeStepTuple[0]
                obs = preprocess(obs)

                if step_type == step_type.LAST:
                    buffer.finalize(reward)

                    # respawn env
                    _, _, _, obs = env.reset()[0]
                    obs = preprocess(obs)

                    # stop render
                    render_env = True

                    if buffer.size() > batch_size:
                        break

            @tf.function
            def train_step(
                player,
                home_away_race,
                upgrades,
                available_act,
                minimap,
                act_id,
                act_args,
                act_mask,
                ret,
                batch_size,
            ):
                # FIXME: some variables don't have gradient due to multihead action layers
                with tf.GradientTape() as tape:
                    ls = actor_critic.loss(
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
                    )
                grad = tape.gradient(ls, actor_critic.trainable_variables)
                optimizer.apply_gradients(zip(grad, actor_critic.trainable_variables))
                return ls

            # update policy
            (
                player,
                home_away_race,
                upgrades,
                available_act,
                minimap,
                act_id,
                act_args,
                act_mask,
                ret,
            ) = buffer.sample()

            if tracing_on:
                tf.summary.trace_on(graph=True, profiler=False)

            batch_loss = train_step(
                player,
                home_away_race,
                upgrades,
                available_act,
                minimap,
                act_id,
                act_args,
                act_mask,
                ret,
                buffer.size(),
            )

            if tracing_on:
                with train_summary_writer.as_default():
                    tf.summary.trace_export(name="train_step", step=0)

            return batch_loss, buffer.batch_ret, buffer.batch_len

        for i in range(epochs):
            if i == 0:
                tracing_on = True
            else:
                tracing_on = False
            batch_loss, batch_ret, batch_len = train_one_epoch(tracing_on)
            with train_summary_writer.as_default():
                tf.summary.scalar("batch_ret", np.mean(batch_ret), step=i)
                tf.summary.scalar("batch_len", np.mean(batch_len), step=i)
                tf.summary.scalar("batch_loss", batch_loss.numpy(), step=i)
            print("----------------------------")
            print(
                "epoch {0:2d} loss {1:.3f} batch_ret {2:.3f} batch_len {3:.3f}".format(
                    i, batch_loss.numpy(), np.mean(batch_ret), np.mean(batch_len)
                )
            )
            print("----------------------------")


def main(argv):
    epochs = 100
    batch_size = 128
    train(FLAGS.env_name, batch_size, epochs)


if __name__ == "__main__":
    app.run(main)
