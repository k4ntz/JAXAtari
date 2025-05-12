from jaxatari.environment import JaxEnvironment, EnvState, EnvObs, EnvInfo, JAXAtariAction
from jaxatari.games.tennis_from_scratch.tennis_main import TennisState
from typing import NamedTuple, Tuple
import jax.numpy as jnp

from jaxatari.renderers import AtraJaxisRenderer
from tennis_main import tennis_step, GAME_WIDTH
from tennis_renderer import TennisRenderer as tr

renderer = tr()

class TennisObs(NamedTuple):
    some_shit: jnp.ndarray = 0

class TennisInfo(NamedTuple):
    pass

class AtraJaxisTennisRenderer(AtraJaxisRenderer):

    def render(self, state: TennisState):
        return renderer.render(state)

class TennisJaxEnv(JaxEnvironment[TennisState, TennisObs, TennisInfo]):

    def reset(self, key) -> Tuple[TennisObs, TennisState]:
        # center player by subtracting half player width, hardcode for now todo fix
        reset_state = TennisState(GAME_WIDTH / 2.0 - 2.5, 0.0)
        reset_obs = self._get_observation(reset_state)

        return reset_obs, reset_state

    def step( self, state: TennisState, action) -> Tuple[TennisObs, TennisState, float, bool, TennisInfo]:
        new_state = tennis_step(state, action)
        new_obs = self._get_observation(new_state)
        reward = self._get_reward(state, new_state)
        done = self._get_done(new_state)
        info = self._get_info(new_state)

        return new_obs, new_state, reward, done, info

    def render(self, state: TennisState) -> Tuple[jnp.ndarray]:
        return renderer.render(state)

    def get_action_space(self) -> jnp.ndarray:
        return [JAXAtariAction.RIGHT, JAXAtariAction.LEFT, JAXAtariAction.UP, JAXAtariAction.DOWN, JAXAtariAction.FIRE]

    def get_observation_space(self) -> Tuple:
        pass

    def _get_observation(self, state: TennisState) -> TennisObs:
        return TennisObs()

    def _get_info(self, state: TennisState) -> TennisInfo:
        return TennisInfo()

    def _get_reward(self, previous_state: TennisState, state: TennisState) -> float:
        return 0.0

    def _get_done(self, state: TennisState) -> bool:
        return False