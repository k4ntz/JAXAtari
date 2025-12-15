import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

from dataclasses import dataclass

def _create_background_sprite(consts: "BasicMathConstants", gameMode: int) -> jnp.ndarray:
    bg_color_rgba = (*consts.COLOR_CODES[gameMode][0], 255)
    return jnp.tile(
        jnp.array(bg_color_rgba, dtype=jnp.uint8),
        (consts.SCREEN_HEIGHT, consts.SCREEN_WIDTH, 1),
    )

def _get_default_asset_config() -> tuple:
    sym_files = [f'sym{i}.npy' for i in range(4)]
    underscore_files = [f'underscore{i}.npy' for i in range(2)]

    return (
        {'name': 'nums', 'type': 'digits', 'pattern': "num{}.npy"},
        {'name': 'symbols', 'type': 'group', 'files': sym_files},
        {'name': 'underscore', 'type': 'group', 'files': underscore_files}
    )

class BasicMathConstants(NamedTuple):
    SCALINGFACTOR: int = 1
    SCREEN_WIDTH: int = 160 * SCALINGFACTOR
    SCREEN_HEIGHT: int = 210 * SCALINGFACTOR

    COLOR_CODES = [
        [(18, 46, 137), (113, 115, 25)],
        [(143, 114, 41), (63, 1, 106)],
        [(110, 110, 15), (145, 120, 43)],
        [(161, 104, 35), (65, 144, 58)]
    ]

    X_OFFSET: int = 47 * SCALINGFACTOR
    Y_OFFSET: int = 55 * SCALINGFACTOR
    num0 = (X_OFFSET + 20 * SCALINGFACTOR, Y_OFFSET + 20 * SCALINGFACTOR)
    num1 = (num0[0], num0[1] + 40 * SCALINGFACTOR)
    num2 = (num1[0], num1[1] + 40 * SCALINGFACTOR)
    bar0 = (num2[0], num2[1] + 29 * SCALINGFACTOR)
    bar1 = (X_OFFSET, num1[1] + 20 * SCALINGFACTOR)
    symbol = (X_OFFSET + 5 * SCALINGFACTOR, num1[1])


    INITIAL_NUMARR = chex.Array = jnp.array([-1, -1, -1, -1, -1, -1], dtype=jnp.int32)

    ASSET_CONFIG: tuple = _get_default_asset_config()

class BasicMathState(NamedTuple):
    numArr: chex.Array
    arrPos: chex.Array
    score: chex.Array
    numberProb: chex.Array
    problemNum1: chex.Array
    problemNum2: chex.Array
    key: chex.PRNGKey
    step_counter: chex.PRNGKey

class BasicMathObservation(NamedTuple):
    numArr: chex.Array
    arrPos: chex.Array
    problemNum1: chex.Array
    problemNum2: chex.Array

class BasicMathInfo(NamedTuple):
    score: chex.Array
    round: chex.Array

class JaxBasicMath(JaxEnvironment[BasicMathState, BasicMathObservation, BasicMathInfo, BasicMathConstants]):
    def __init__(self, consts: BasicMathConstants = None):
        consts = consts or BasicMathConstants()
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

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BasicMathState, ) -> BasicMathState:
        return BasicMathInfo(score=state.score, round=state.numberProb)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BasicMathState, state: BasicMathState):
        return (state.score) - (
            previous_state.score
        )
    
    def _get_observation(self, state: BasicMathState):
        return BasicMathObservation(
            state.numArr, 
            state.arrPos, 
            state.problemNum1, 
            state.problemNum2
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BasicMathState) -> bool:
        return jnp.greater_equal(state.numberProb, 10)

    
    def _generate_problem(self, state: BasicMathState, gameMode) -> BasicMathState:
        key, k1 = jax.random.split(state.key)
        key, k2 = jax.random.split(key)

        x = jax.random.randint(k1, shape=(), minval=1, maxval=10)

        y = jax.lax.cond(
            gameMode != 3,
            lambda _: jax.random.randint(k2, shape=(), minval=1, maxval=10),
            lambda s: jax.random.randint(k2, shape=(), minval=1, maxval=s),
            operand=x
        )

        return x, y, key
        
    
    def _evaluate_arr(self, arr: chex.Array):
        arr = jax.numpy.nan_to_num(arr, nan=0).astype(jnp.int32)

        a, b = arr[:3], arr[3:]

        weights = jnp.array([100, 10, 1], dtype=jnp.int32)

        val_a = jnp.dot(a, weights)
        val_b = jnp.dot(b, weights)

        return val_a, val_b
    
    def _evaluate_issue(self, state: BasicMathState, gameMode) -> BasicMathState:
        ops = [
            lambda a, b: (a + b, 0),
            lambda a, b: (a - b, 0),
            lambda a, b: (a * b, 0),
            lambda a, b: (a / b, a % b)
        ]

        result = ops[gameMode](state.problemNum1, state.problemNum2)

        a, b = self._evaluate_arr(state.numArr)

        eval = jnp.logical_and(gameMode != 3, a == result[0])
        evalDiv = jnp.logical_and(
            gameMode == 3,
            jnp.logical_and(a == result[0], b == result[1])
        )

        score = jax.lax.cond(
            eval,
            lambda s: s + 1,
            lambda s: s,
            operand=state.score,
        )

        score = jax.lax.cond(
            evalDiv,
            lambda s: s + 1,
            lambda s: s,
            operand=state.score,
        )

        x, y, key = self._generate_problem(state, gameMode)

        return BasicMathState(
            self.consts.INITIAL_NUMARR,
            state.arrPos,
            score,
            state.numberProb + 1,
            x,
            y,
            key,
            state.step_counter
        )
    
    def _change_value(self, state: BasicMathState, action: chex.Array) -> BasicMathState:
        up = action == Action.UP
        down = action == Action.DOWN

        arr = state.numArr
        value = arr[state.arrPos]

        non_val_up = jnp.logical_and(up, value == -1)
        up_edge = jnp.logical_and(up, value == 9)
        up_add = jnp.logical_and(up, jnp.logical_and(jnp.logical_not(up_edge), jnp.logical_not(non_val_up)))

        value = jax.lax.cond(
            non_val_up,
            lambda: 0,
            lambda: value,
        )

        value = jax.lax.cond(
            up_edge,
            lambda: -1,
            lambda: value,
        )

        value = jax.lax.cond(
            up_add,
            lambda: value + 1,
            lambda: value,
        )

        non_val_down = jnp.logical_and(down, value == -1)
        down_edge = jnp.logical_and(down, value == 1)
        down_add = jnp.logical_and(down, jnp.logical_and(jnp.logical_not(down_edge), jnp.logical_not(non_val_down)))

        value = jax.lax.cond(
            non_val_down,
            lambda: 9,
            lambda: value,
        )

        value = jax.lax.cond(
            down_edge,
            lambda: -1,
            lambda: value,
        )

        value = jax.lax.cond(
            down_add,
            lambda: value - 1,
            lambda: value,
        )

        new_arr = arr.at[state.arrPos].set(value)

        return BasicMathState(
            new_arr,
            state.arrPos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.key,
            state.step_counter
        )
    
    def _change_pos(self, state: BasicMathState, action: chex.Array) -> BasicMathState:
        left = action == Action.LEFT
        right = action == Action.RIGHT

        player_pos = state.arrPos

        on_left = jnp.logical_and(left, state.arrPos == 0)
        on_right = jnp.logical_and(right, state.arrPos == 5)
        move_left = jnp.logical_and(left, jnp.logical_not(on_left))
        move_right = jnp.logical_and(right, jnp.logical_not(on_right))

        player_pos = jax.lax.cond(
            on_left,
            lambda: 5,
            lambda: player_pos,
        )

        player_pos = jax.lax.cond(
            on_right,
            lambda: 0,
            lambda: player_pos,
        )

        player_pos = jax.lax.cond(
            move_left,
            lambda: player_pos - 1,
            lambda: player_pos,
        )

        player_pos = jax.lax.cond(
            move_right,
            lambda: player_pos + 1,
            lambda: player_pos,
        )

        return BasicMathState(
            state.numArr,
            player_pos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.key,
            state.step_counter
        )
    
    def render(self, state: BasicMathState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BasicMathObservation, BasicMathState]:
        state_key, _step_key = jax.random.split(key)

        state = BasicMathState(
            self.consts.INITIAL_NUMARR,
            arrPos= jnp.array(2).astype(jnp.int32),
            score= jnp.array(0).astype(jnp.int32),
            numberProb= jnp.array(0).astype(jnp.int32),
            problemNum1=jnp.array(1).astype(jnp.int32),
            problemNum2=jnp.array(1).astype(jnp.int32),
            key=state_key,
            step_counter=jnp.array(0).astype(jnp.int32)
        )

        obs = self._get_observation(state)

        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BasicMathState, action: chex.Array, gameMode: int = 5) -> Tuple[BasicMathObservation, BasicMathState, float, bool, BasicMathInfo]:
        previous_state = state

        state = BasicMathState(
            state.numArr,
            state.arrPos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.key,
            state.step_counter
        )

        chosenGameMode = (gameMode - 1) % 4

        is_fire = action == Action.FIRE

        state = self._change_pos(state, action)
        state = self._change_value(state, action)
        state = jax.lax.cond(
            is_fire, lambda s: self._evaluate_issue(s, chosenGameMode), lambda s: s, operand=state
        )

        done = self._get_done(state)
        reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        obs = self._get_observation(state)

        return obs, state, reward, done, info

class BasicMathRenderer(JAXGameRenderer):
    def __init__(self, consts: BasicMathConstants = None):
        super().__init__(consts)
        self.consts = consts or BasicMathConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        final_asset_config = list(self.consts.ASSET_CONFIG)

        wall_sprite = _create_background_sprite(self.consts, 0)

        final_asset_config.append({'name': 'background', 'type': 'background', 'data': wall_sprite},)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/basicmath"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

        self.NUM_MASKS_STACKED = self._stack_num_masks

    def _stack_num_masks(self) -> jnp.ndarray:
        """Helper to get all player-related masks from the main padded group."""
        # The first 16 are rotation, the next 3 are death animations
        return self.SHAPE_MASKS['nums']

    def render(self, state: BasicMathState):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        underscore_mask = self.SHAPE_MASKS["underscore"]
        raster = self.jr.render_at(raster, 35 * self.consts.SCALINGFACTOR + state.arrPos * 15 * self.consts.SCALINGFACTOR, self.consts.bar0[1], underscore_mask[0])    
        raster = self.jr.render_at(raster, *self.consts.bar1, underscore_mask[1])
        symbol = self.SHAPE_MASKS["symbols"]
        raster = self.jr.render_at(raster, *self.consts.symbol, symbol[0])
        digit_masks = self._stack_num_masks()

        digit0 = self.jr.int_to_digits(state.problemNum1, max_digits=1)
        
        raster = self.jr.render_label_selective(raster, *self.consts.num0, digit0, digit_masks, 0, state.problemNum1, spacing=16)

        digit1 = self.jr.int_to_digits(state.problemNum2, max_digits=1)
        raster = self.jr.render_label_selective(raster, *self.consts.num1, digit1, digit_masks, 0, state.problemNum2, spacing=16)

        # --- Render Asteroids ---
        def render_nums(i, r):
            num = state.numArr[i]
            digit = self.jr.int_to_digits(num, max_digits=1)

            is_active = num != -1
            
            return jax.lax.cond(is_active, 
                                lambda ras: self.jr.render_label_selective(ras, 35 * self.consts.SCALINGFACTOR + i * 15 * self.consts.SCALINGFACTOR, self.consts.num2[1], digit, digit_masks, state.arrPos, num, spacing=16), 
                                lambda ras: ras, 
                                r)
        raster = jax.lax.fori_loop(0, 6, render_nums, raster)

        return self.jr.render_from_palette(raster, self.PALETTE)