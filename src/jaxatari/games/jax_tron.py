from dataclasses import dataclass
from jaxatari.renderers import JAXGameRenderer
from typing import NamedTuple, Tuple
from jax import Array, jit, random, numpy as jnp
from functools import partial
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, EnvObs, EnvState
import jaxatari.spaces as spaces


class TronConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    scaling_factor: int = 3


class TronState(NamedTuple):
    score: Array


class EntityPosition(NamedTuple):
    x: Array
    y: Array
    width: Array
    height: Array


class TronObservation(NamedTuple):
    pass


class TronInfo(NamedTuple):
    pass


class TronRenderer(JAXGameRenderer):
    def __init__(self, consts: TronConstants = None) -> None:
        super().__init__()
        self.consts = consts or TronConstants()

    @partial(jit, static_argnums=(0,))
    def render(self, state) -> Array:
        raster = jr.create_initial_frame(
            width=self.consts.screen_width, height=self.consts.screen_height
        )
        return raster


class JaxTron(
    JaxEnvironment[TronState, TronObservation, TronInfo, TronConstants]
):
    def __init__(
        self, consts: TronConstants = None, reward_funcs: list[callable] = None
    ) -> None:
        consts = consts or TronConstants()
        super().__init__(consts)
        self.renderer = TronRenderer

    def reset(
        self, key: random.PRNGKey = None
    ) -> Tuple[TronObservation, TronState]:
        new_state: TronState = TronState(
            score=jnp.array(0, dtype=jnp.int32),
        )
        obs = self._get_observation(new_state)
        return obs, new_state

    @partial(jit, static_argnums=(0,))
    def step(
        self, state: TronState, action: Array
    ) -> Tuple[TronObservation, TronState, float, bool, TronInfo]:
        new_state: TronState = state
        obs: TronObservation = self._get_observation(new_state)
        env_reward: float = self._get_reward(state, new_state)
        done: bool = self._get_done(new_state)
        info: TronInfo = self._get_info(state)

        return obs, new_state, env_reward, done, info

    @partial(jit, static_argnums=(0,))
    def _get_observation(self, state: TronState) -> TronObservation:
        return TronObservation()

    @partial(jit, static_argnums=(0,))
    def _get_reward(self, previous_state: TronState, state: TronState) -> float:
        return 0.0

    @partial(jit, static_argnums=(0,))
    def _get_done(self, state: TronState) -> bool:
        return False

    @partial(jit, static_argnums=(0,))
    def _get_info(self, state: TronState) -> TronInfo:
        return TronInfo()

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(18)
