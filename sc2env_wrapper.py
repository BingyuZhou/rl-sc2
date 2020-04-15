from pysc2.env import sc2_env, environment
import collections


class SC2EnvWrapper(sc2_env.SC2Env):
    def __init__(self, map_name, players, agent_interface_format, step_mul,
                 game_steps_per_episode, disable_fog):
        super(SC2EnvWrapper,
              self).__init__(map_name=map_name,
                             players=players,
                             agent_interface_format=agent_interface_format,
                             step_mul=step_mul,
                             game_steps_per_episode=game_steps_per_episode,
                             disable_fog=disable_fog,
                             visualize=True)

    def render(self, render_on):
        if not render_on:
            self._renderer_human = None

    @staticmethod
    def preprocess_obs(obs):
        """
         preprocess raw obs from SC2Env to structured observations for better interface to model
         Following the observation clusters from DeepMind "Grandmaster level in StarCraft II using multi-agent reinforcement learning", the raw observations are classified into:
         - entities
         - map
         - player_info
         - game_statistics
        """
        assert isinstance(
            obs, environment.TimeStep), "observation must be `TimeStep` type"

        new_obs = collections.namedtuple(
            'Obs_processed',
            ['Entities', 'Maps', 'Player_info', 'Game_statistics'])

        new_obs.Maps = [
            obs['alerts'], obs['feature_minimap'], obs['feature_screen']
        ]

        # :param home_race_requested: Agent requested race
        # :param away_race_requested: Opponent requested race
        new_obs.Player_info = [
            obs['player'], obs['upgrades'], obs['available_actions'],
            obs['home_race_requested'], obs['away_race_requested']
        ]
        new_obs.Entities = [
            obs['production_queue'], obs['build_queue'], obs['cargo'],
            obs['cargo_slots_available']
        ]
        new_obs.Game_statistics = [
            obs['score_cumulative'], obs['score_by_vital']
        ]

        return new_obs
