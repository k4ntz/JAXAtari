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

def _get_default_asset_config() -> tuple:
    num_files = [f'num{i}.npy' for i in range(16)]
    sym_files = [f'sym{i}.npy' for i in range(4)]
    underscore_files = [f'underscore{i}.npy' for i in range(2)]

    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'nums', 'type': 'background', 'digits': num_files},
        {'name': 'symbols', 'type': 'background', 'file': sym_files},
        {'name': 'underscore', 'type': 'background', 'file': underscore_files}
    )

class BasicMathConstants(NamedTuple):
    SCALINGFACTOR: int = 3
    SCREEN_WIDTH: int = 160 * SCALINGFACTOR
    SCREEN_HEIGHT: int = 210 * SCALINGFACTOR

    COLOR_CODES = [
        [(18, 46, 137), (113, 115, 25)],
        [(143, 114, 41), (63, 1, 106)],
        [(110, 110, 15), (145, 120, 43)],
        [(161, 104, 35), (65, 144, 58)]
    ]

    OPERATIONS = [
        "+",
        "-",
        "*",
        "/"
    ]

    ASSET_CONFIG: tuple = _get_default_asset_config()

class BasicMathState(NamedTuple):
    numArr: chex.Array
    remainderArr: chex.Array
    arrPos: chex.Array
    score: chex.Array
    numberProb: chex.Array
    problemNum1: chex.Array
    problemNum2: chex.Array
    key: chex.PRNGKey

class BasicMathObservation(NamedTuple):
    numArr: chex.Array
    remainderArr: chex.Array
    arrPos: chex.Array
    score: chex.Array
    problemNum1: chex.Array
    problemNum2: chex.Array

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
    
    def _generate_problem(state: BasicMathState) -> BasicMathState:
        key, k1 = jax.random.split(state.key)
        key, k2 = jax.random.split(state.key)

        x = jax.random.randint(k1, shape=(), minval=1, maxval=10)
        y = jax.random.randint(k2, shape=(), minval=1, maxval=10)

        return BasicMathState(
            state.numArr,
            state.remainderArr,
            state.arrPos,
            state.score,
            state.numberProb,
            x,
            y,
            key
        )
    
    def _evaluate_issue(state: BasicMathState, gameMode) -> BasicMathState:
        ops = [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
            lambda a, b: (a / b, a % b)
        ]

        result = ops[gameMode](state.problemNum1, state.problemNum2)

        eval = True

        score = jax.lax.cond(
            eval,
            lambda s: s + 1,
            lambda s: s,
            operand=state.score,
        )

        return BasicMathState(
            state.numArr,
            state.remainderArr,
            state.arrPos,
            score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.key
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BasicMathState, action: chex.Array) -> Tuple[BasicMathObservation, BasicMathState, float, bool, BasicMathInfo]:
        left = action == Action.LEFT
        right = action == Action.RIGHT
        up = action == Action.UP
        down = action == Action.DOWN

        player_pos = state.arrPos

        on_left = jnp.logical_and(left, state.arrPos == 0)
        player_pos = jax.lax.cond(
            on_left,
            lambda _: 5,
            lambda s: s,
            operand= player_pos
        )

        on_right = jnp.logical_and(right, state.arrPos == 5)
        player_pos = jax.lax.cond(
            on_right,
            lambda _: 0,
            lambda s: s,
            operand= player_pos
        )

        move_left = jnp.logical_and(left, jnp.logical_not(on_left))
        player_pos = jax.lax.cond(
            move_left,
            lambda s: s - 1,
            lambda s: s,
            operand= player_pos
        )

        move_right = jnp.logical_and(right, jnp.logical_not(on_right))
        player_pos = jax.lax.cond(
            move_right,
            lambda s: s + 1,
            lambda s: s,
            operand= player_pos
        )

        arr = state.numArr
        value = arr[player_pos]

        new_arr = arr.at[player_pos].set(value)


        done = self._get_done
        return done

class BasicMathRenderer(JAXGameRenderer):
    def __init__(self, consts: BasicMathConstants = None):
        super().__init__(consts)
        self.consts = consts or BasicMathConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        final_asset_config = list(self.consts.ASSET_CONFIG)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/basicmath"
        (
            self.NUMS,
            self.SYMS,
            self.BACKGROUND,
            self.UNDERSCORE,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)