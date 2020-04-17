from pysc2.env import sc2_env
from pysc2.lib import renderer_human


class SC2EnvWrapper(sc2_env.SC2Env):
    """SC2Env wrapper to control rendering"""

    def __init__(
        self,
        map_name,
        players,
        agent_interface_format,
        step_mul,
        game_steps_per_episode,
        disable_fog,
    ):
        super(SC2EnvWrapper, self).__init__(
            map_name=map_name,
            players=players,
            agent_interface_format=agent_interface_format,
            step_mul=step_mul,
            game_steps_per_episode=game_steps_per_episode,
            disable_fog=disable_fog,
            visualize=True,
        )

    def render(self, render_on):
        if render_on:
            self._renderer_human = renderer_human.RendererHuman()
            self._renderer_human.init(
                self._controllers[0].game_info(), self._controllers[0].data()
            )
        else:
            self._renderer_human = None
