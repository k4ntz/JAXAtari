from functools import partial
from typing import NamedTuple, Tuple
import os

import jax
import jax.numpy as jnp
import chex
from jax import lax
from jax import random as jrandom

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.rendering import jax_rendering_utils as render_utils
import numpy as np
from jaxatari.renderers import JAXGameRenderer

class BasicMathConstants(NamedTuple):
    COLOR_CODES = [
        [(89, 90, 10), (140, 151, 62)],
        [(143, 80, 21), (37, 89, 127)],
        [(37, 89, 127), (142, 107, 39)],
        [(147, 146, 32), (26, 46, 129)],
        [(18, 46, 137), (113, 115, 25)],
        [(143, 114, 41), (63, 1, 106)],
        [(110, 110, 15), (145, 120, 43)],
        [(161, 104, 35), (65, 144, 58)]
    ]

    SCALINGFACTOR: int = 3
    SCREEN_WIDTH: int = 160 * SCALINGFACTOR
    SCREEN_HEIGHT: int = 210 * SCALINGFACTOR

class BasicMathState(NamedTuple):
    numArr: chex.Array        # (H,W) int32 {0,1}
    arrPos: chex.Array          # (2,) int32 [y,x]
    score: chex.Array        # () int32
    problemNum: chex.Array
    screen: chex.Array      # () int32
    chosenNum: chex.Array
    gameMode: chex.Array

class BasicMathObservation(NamedTuple):
    score: chex.Array

class BasicMathInfo(NamedTuple):
    score: chex.Array

class JaxBasicMath(JaxEnvironment[BasicMathState, BasicMathObservation, BasicMathInfo, BasicMathConstants]):
    def __init__(self, consts: BasicMathConstants = None):
        consts = consts or BasicMathConstants
        super().__init__(consts)
        self.renderer = BasicMathRenderer(self.consts)
        self.action_set = [
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        ]

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(5)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BasicMathState, ) -> BasicMathState:
        return BasicMathInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BasicMathState, state: BasicMathState):
        return (state.score) - (
            previous_state.score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BasicMathState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.problemNum, 10),
        )

class BasicMathRenderer(JAXGameRenderer):
    def __init__(self, consts: BasicMathConstants = None):
        super().__init__(consts)
        self.consts = consts or BasicMathConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )