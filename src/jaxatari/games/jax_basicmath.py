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

    INITIAL_NUMARR = chex.Array = jnp.array([
        jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan
    ])

    ASSET_CONFIG: tuple = _get_default_asset_config()

class BasicMathState(NamedTuple):
    numArr: chex.Array
    arrPos: chex.Array
    score: chex.Array
    numberProb: chex.Array
    problemNum1: chex.Array
    problemNum2: chex.Array
    key: chex.PRNGKey

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

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BasicMathState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.numberProb, 10),
        )
    
    def _generate_problem(state: BasicMathState, action: chex.Array, gameMode) -> BasicMathState:
        fire = action == Action.FIRE

        key, k1 = jax.random.split(state.key)
        key, k2 = jax.random.split(state.key)

        x = jax.random.randint(k1, shape=(), minval=1, maxval=10)

        y = jax.lax.cond(
            gameMode != 3,
            lambda: jax.random.randint(k2, shape=(), minval=1, maxval=10),
            lambda: jax.random.randint(k2, shape=(), minval=1, maxval=x),
        )

        return BasicMathState(
            state.numArr,
            state.arrPos,
            state.score,
            state.numberProb,
            x,
            y,
            key
        )
    
    def _evaluate_arr(arr: chex.Array):
        arr = jax.numpy.nan_to_num(arr).astype(jnp.int32)

        a, b = arr[:3], arr[3:]

        weights = jnp.array([100, 10, 1], dtype=jnp.int32)

        val_a = jnp.dot(a, weights)
        val_b = jnp.dot(b, weights)

        return val_a, val_b
    
    def _evaluate_issue(self, state: BasicMathState, action: chex.Array, gameMode) -> BasicMathState:
        ops = [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
            lambda a, b: (a / b, a % b)
        ]

        result = ops[gameMode](state.problemNum1, state.problemNum2)

        a, b = self._evaluate_arr(state.numArr)

        eval = jnp.logical_and(gameMode != 3, a == result)
        evalDiv = jnp.logical_and(gameMode == 3, (a, b) == result)

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


        return BasicMathState(
            state.numArr,
            state.arrPos,
            score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.key
        )
    
    def _change_value(self, state: BasicMathState, action: chex.Array) -> BasicMathState:
        up = action == Action.UP
        down = action == Action.DOWN

        arr = state.numArr
        value = arr[state.arrPos]

        is_nan = jnp.isnan(value)
        non_val_up = jnp.logical_and(up, is_nan)
        up_edge = jnp.logical_and(up, value == 9)
        up_add = jnp.logical_and(up, jnp.logical_and(jnp.logical_not(up_edge), jnp.logical_not(non_val_up)))

        value = jax.lax.cond(
            non_val_up,
            lambda: 0,
            lambda: value,
        )

        value = jax.lax.cond(
            up_edge,
            lambda: jnp.nan,
            lambda: value,
        )

        value = jax.lax.cond(
            up_add,
            lambda: value + 1,
            lambda: value,
        )

        non_val_down = jnp.logical_and(down, is_nan)
        down_edge = jnp.logical_and(down, value == 1)
        down_add = jnp.logical_and(down, jnp.logical_and(jnp.logical_not(down_edge), jnp.logical_not(non_val_down)))

        value = jax.lax.cond(
            non_val_down,
            lambda: 9,
            lambda: value,
        )

        value = jax.lax.cond(
            down_edge,
            lambda: jnp.nan,
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
            state.key            
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
            state.key
        )
    
    def _process_fire(self, state, action, gameMode):
        is_fire = action == Action.FIRE

        def do_fire(_):
            s = self._evaluate_issue(state, action, gameMode)
            s = self._generate_problem(s, action, gameMode)
            return s

        def no_fire(_):
            return state

        return jax.lax.cond(is_fire, do_fire, no_fire, operand=None)
    
    def render(self, state: BasicMathState):
        return self.renderer.render(state)
    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BasicMathObservation, BasicMathState]:
        state_key, _step_key = jax.random.split(key)

        state = BasicMathState(
            numArr=jnp.array(self.consts.INITIAL_NUMARR).astype(jnp.int32),
            arrPos= jnp.array(2).astype(jnp.int32),
            score= jnp.array(0).astype(jnp.int32),
            numberProb= jnp.array(0).astype(jnp.int32),
            problemNum1= jnp.array(1).astype(jnp.int32),
            problemNum2= jnp.array(1).astype(jnp.int32),
            key= state_key           
        )

        return None, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BasicMathState, action: chex.Array, gameMode: int = 5) -> Tuple[BasicMathObservation, BasicMathState, float, bool, BasicMathInfo]:
        state = BasicMathState(
            state.numArr,
            state.arrPos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.key
        )

        chosenGameMode = (gameMode - 1) % 4

        state = self._change_pos(state, action)
        state = self._change_value(state, action)
        state = self._process_fire(state, action, chosenGameMode)

        done = self._get_done
        reward = self._get_reward
        info = self._get_info

        state = self._evaluate_issue(state, gameMode)

        return None, state, reward, done, info

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