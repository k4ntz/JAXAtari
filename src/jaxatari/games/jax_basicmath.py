import os
import numpy as np
from functools import partial
from typing import Any, Tuple, NamedTuple
import jax
import jax.lax
import jax.numpy as jnp
import chex
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _create_background_sprite(consts: "BasicMathConstants", dim: Tuple, gameMode: int) -> jnp.ndarray:
    bg_color_rgba = (*consts.COLOR_CODES[gameMode][0], 255)
    return jnp.tile(
        jnp.array(bg_color_rgba, dtype=jnp.uint8),
        (dim[0], dim[1], 1),
    )

def _get_default_asset_config() -> tuple:
    sym_files = [f'sym{i}.npy' for i in range(4)]
    underscore_files = [f'underscore{i}.npy' for i in range(2)]

    return (
        {'name': 'nums', 'type': 'digits', 'pattern': "num{}.npy"},
        {'name': 'symbols', 'type': 'group', 'files': sym_files},
        {'name': 'underscore', 'type': 'group', 'files': underscore_files}
    )

class BasicMathConstants(struct.PyTreeNode):
    SCREEN_WIDTH: int = struct.field(pytree_node=False, default=160)
    SCREEN_HEIGHT: int = struct.field(pytree_node=False, default=210)

    GAMEMODE: int = struct.field(pytree_node=False, default=5)
    DIFFICULTY: int = struct.field(pytree_node=False, default=0)

    COLOR_CODES: Tuple[Tuple[int, int, int], ...] = struct.field(pytree_node=False, default_factory=lambda: tuple([
        [(18, 46, 137), (113, 115, 25)],
        [(143, 114, 41), (63, 1, 106)],
        [(110, 110, 15), (145, 120, 43)],
        [(161, 104, 35), (65, 144, 58)]
    ]))

    X_OFFSET: int = struct.field(pytree_node=False, default=47)
    Y_OFFSET: int = struct.field(pytree_node=False, default=35)
    BAR_OFFSET: int = struct.field(pytree_node=False, default= 31)
    num0: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: tuple((47 + 20, 35 + 20)))
    num1: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: tuple((47 + 20, 35 + 60)))
    num2: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: tuple((47 + 20, 35 + 100)))
    bar0: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: tuple((47 + 20, 35 + 129)))
    bar1: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: tuple((47, 35 + 90)))
    symbol: Tuple[int, int] = struct.field(pytree_node=False, default_factory=lambda: tuple((47 + 5, 35 + 60)))

    problemNumLen: int = struct.field(pytree_node=False, default=1)
    numArrLen: int = struct.field(pytree_node=False, default=6)
    initialArrPos: int = struct.field(pytree_node=False, default=2)
    spacing: int = struct.field(pytree_node=False, default=0)

    DIFFICULTY_TIMES = jnp.array([-1, 180, 360, 720], dtype=jnp.int32)
    PAD_OBS: int = struct.field(pytree_node=False, default=2)

    ASSET_CONFIG: tuple = _get_default_asset_config()

@struct.dataclass
class BasicMathState:
    numArr: chex.Array
    arrPos: chex.Array
    score: chex.Array
    numberProb: chex.Array
    problemNum1: chex.Array
    problemNum2: chex.Array
    inactive: chex.Array
    difficultyTime: chex.Array
    key: chex.PRNGKey
    step_counter: chex.Array

@struct.dataclass
class BasicMathObservation(struct.PyTreeNode):
    problem_num1: jnp.ndarray
    problem_num2: jnp.ndarray
    digits: jnp.ndarray
    arrPos: jnp.ndarray

@struct.dataclass
class BasicMathInfo:
    score: chex.Array
    round: chex.Array

class JaxBasicMath(JaxEnvironment[BasicMathState, BasicMathObservation, BasicMathInfo, BasicMathConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
       [Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN],
       dtype=jnp.int32,
   )

    def __init__(self, consts: BasicMathConstants = None):
        consts = consts or BasicMathConstants()
        super().__init__(consts)
        self.renderer = BasicMathRenderer(consts)
        self.INITARR = jnp.full((self.consts.numArrLen,), -1, dtype=jnp.int32)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTION_SET))

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "problem_num1": spaces.Box(low=0, high=1000, shape=(), dtype=jnp.int32),
            "problem_num2": spaces.Box(low=0, high=1000, shape=(), dtype=jnp.int32),
            "digits": spaces.Box(low=-1, high=9, shape=(8,), dtype=jnp.int32),
            "arrPos": spaces.Box(low=0, high=8, shape=(), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BasicMathState, ) -> BasicMathInfo:
        return BasicMathInfo(score=state.score, round=state.numberProb)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BasicMathState, state: BasicMathState):
        return (state.score) - (
            previous_state.score
        )
    
    def _get_observation(self, state: BasicMathState):
        midNumArr = self.consts.numArrLen // 2
        halfPadObs = self.consts.PAD_OBS // 2
        a, b = state.numArr[:midNumArr], state.numArr[midNumArr:]

        padded_digits_a = jnp.pad(
                a, 
                (halfPadObs, 0), 
                mode='constant', 
                constant_values=-1
            )
        
        padded_digits_b = jnp.pad(
                b, 
                (halfPadObs, 0), 
                mode='constant', 
                constant_values=-1
            )

        return BasicMathObservation(
            problem_num1=state.problemNum1,
            problem_num2=state.problemNum2,
            digits=jnp.concatenate([padded_digits_a, padded_digits_b]),
            arrPos=state.arrPos,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BasicMathState) -> bool:
        return state.numberProb >= 10


    def _int_to_fixed_digits(self, x, length=3):
        powers = 10 ** jnp.arange(length - 1, -1, -1)
        digits = (x // powers) % 10

        started = jnp.cumsum(digits != 0) > 0
        digits = jnp.where(started, digits, -1)

        digits = jnp.where(jnp.equal(x, 0), jnp.full((length,), -1), digits)

        return digits
    
    def _generate_problem(self, key, gameMode: int) -> BasicMathState:
        key, k1 = jax.random.split(key)
        key, k2 = jax.random.split(key)

        x = jax.random.randint(k1, shape=(), minval=1, maxval=10**self.consts.problemNumLen)

        y = jax.lax.cond(
            jnp.logical_or(gameMode == 1, gameMode == 3),
            lambda s: jax.random.randint(k2, shape=(), minval=1, maxval=s),
            lambda _: jax.random.randint(k2, shape=(), minval=1, maxval=10),
            operand=x
        )

        return x, y, key
    
    def _evaluate_issue(self, state: BasicMathState, gameMode: int) -> BasicMathState:
        midNumArr = self.consts.numArrLen // 2

        ops = [
            lambda a, b: (a + b, 0),
            lambda a, b: (a - b, 0),
            lambda a, b: (a * b, 0),
            lambda a, b: (a // b, a % b)
        ]

        result = ops[gameMode](state.problemNum1, state.problemNum2)

        a, b = state.numArr[:midNumArr], state.numArr[midNumArr:]
        res_a, res_b = self._int_to_fixed_digits(result[0], midNumArr), self._int_to_fixed_digits(result[1], midNumArr)
        is_correct = jax.lax.cond(
            gameMode == 3,
            lambda: jnp.logical_and(
                jnp.all(a == res_a),
                jnp.all(b == res_b)
            ),
            lambda: jnp.all(a == res_a)
        )

        score = jax.lax.cond(
            is_correct,
            lambda: state.score + 1,
            lambda: state.score
        )

        inactive_time = jax.lax.cond(
            is_correct,
            lambda: jnp.array(60).astype(jnp.int32),
            lambda: jnp.array(120).astype(jnp.int32)
        )

        return BasicMathState(
            jnp.concatenate([res_a, res_b]),
            state.arrPos,
            score,
            state.numberProb + 1,
            state.problemNum1,
            state.problemNum2,
            inactive_time,
            state.difficultyTime,
            state.key,
            state.step_counter
        )
    
    def _change_value(self, state: BasicMathState, action: chex.Array) -> BasicMathState:
        arr = state.numArr
        value = arr[state.arrPos]

        up = jnp.equal(action, Action.UP)
        down = jnp.equal(action, Action.DOWN)

        case = jnp.select(
            [
                jnp.logical_and(up, value == 9),
                jnp.logical_and(up, value != 9),
                jnp.logical_and(down, value == -1),
                jnp.logical_and(down, value != -1),
            ],
            [
                0,
                1,
                2,
                3,
            ],
            default=4,
        )

        def up_edge(v): return -1
        def up_add(v): return v + 1
        def down_edge(v): return 9
        def down_add(v): return v - 1
        def noop(v): return v

        value = jax.lax.switch(
            case,
            (up_edge, up_add, down_edge, down_add, noop),
            value
        )

        new_arr = arr.at[state.arrPos].set(value)

        return BasicMathState(
            new_arr,
            state.arrPos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.inactive,
            state.difficultyTime,
            state.key,
            state.step_counter
        )
    
    def _change_pos(self, state: BasicMathState, action: chex.Array) -> BasicMathState:
        end = self.consts.numArrLen - 1
        pos = state.arrPos

        left = jnp.equal(action, Action.LEFT)
        right = jnp.equal(action, Action.RIGHT)

        case = jnp.select(
            [
                jnp.logical_and(left, pos == 0),
                jnp.logical_and(right, pos == end),
                jnp.logical_and(left, pos != 0),
                jnp.logical_and(right, pos != end),
            ],
            [
                0,
                1,
                2,
                3,
            ],
            default=4,
        )

        def wrap_left(p): return end
        def wrap_right(p): return 0
        def move_left(p): return p - 1
        def move_right(p): return p + 1
        def noop(p): return p

        pos = jax.lax.switch(
            case,
            (wrap_left, wrap_right, move_left, move_right, noop),
            pos,
        )

        return BasicMathState(
            state.numArr,
            pos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.inactive,
            state.difficultyTime,
            state.key,
            state.step_counter
        )
    
    def render(self, state: BasicMathState) -> jnp.ndarray:
        return self.renderer.render(state)
    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(np.random.randint(0, 2**32))) -> Tuple[BasicMathObservation, BasicMathState]:
        probNum1, probNum2, key = self._generate_problem(key, (self.consts.GAMEMODE - 1) % 4)

        state = BasicMathState(
            self.INITARR,
            arrPos= jnp.array(self.consts.initialArrPos).astype(jnp.int32),
            score= jnp.array(0).astype(jnp.int32),
            numberProb= jnp.array(0).astype(jnp.int32),
            problemNum1=probNum1,
            problemNum2=probNum2,
            inactive=jnp.array(0).astype(jnp.int32),
            difficultyTime=self.consts.DIFFICULTY_TIMES[self.consts.DIFFICULTY],
            key=key,
            step_counter=jnp.array(0).astype(jnp.int32)
        )

        obs = self._get_observation(state)

        return obs, state
    
    def _active(self, state: BasicMathState, action: chex.Array, gameMode: int) -> BasicMathState:
        act = jnp.equal(state.step_counter % 2, 0)
        countDown = jnp.equal(state.difficultyTime, 0)
        is_fire = jnp.equal(action, Action.FIRE)
        eval_issue = jnp.logical_or(jnp.logical_and(is_fire, act), countDown)

        state = jax.lax.cond(
            act, 
            lambda s: self._change_pos(s, action), 
            lambda s: s, 
            operand=state
        )

        state = jax.lax.cond(
            act, 
            lambda s: self._change_value(s, action),
            lambda s: s, 
            operand=state
        )

        state = jax.lax.cond(
            eval_issue, 
            lambda s: self._evaluate_issue(s, gameMode), 
            lambda s: s, 
            operand=state
        )

        new_state = BasicMathState(
            state.numArr,
            state.arrPos,
            state.score,
            state.numberProb,
            state.problemNum1,
            state.problemNum2,
            state.inactive,
            state.difficultyTime,
            state.key,
            state.step_counter
        )

        return new_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BasicMathState, action: chex.Array) -> Tuple[BasicMathObservation, BasicMathState, float, bool, BasicMathInfo]:
        chosenGameMode = (self.consts.GAMEMODE - 1) % 4
        difficulty_time = self.consts.DIFFICULTY_TIMES[self.consts.DIFFICULTY]
        
        previous_state = state

        active = jnp.equal(state.inactive, 0)

        state = jax.lax.cond(
            active, 
            lambda s: self._active(s, action, chosenGameMode), 
            lambda s: s, 
            operand=state
        )

        timer = jax.lax.cond(
            jnp.logical_not(active),
            lambda: state.inactive - 1,
            lambda: state.inactive,
        )

        reset = jnp.logical_and(jnp.equal(state.inactive, 1), jnp.equal(timer, 0))

        x, y, key = jax.lax.cond(
            reset, 
            lambda: self._generate_problem(state.key, chosenGameMode), 
            lambda: (state.problemNum1, state.problemNum2, state.key)
        )

        arr = jax.lax.cond(
            reset, 
            lambda: self.INITARR, 
            lambda: state.numArr
        )

        new_difficulty_time = jax.lax.cond(
            active,
            lambda: state.difficultyTime - 1, 
            lambda: state.difficultyTime
        )

        new_difficulty_time = jax.lax.cond(
            reset, 
            lambda: difficulty_time, 
            lambda: new_difficulty_time
        )

        new_state = BasicMathState(
            arr,
            state.arrPos,
            state.score,
            state.numberProb,
            x,
            y,
            timer,
            new_difficulty_time,
            key,
            state.step_counter + 1
        )

        done = self._get_done(state)
        reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        obs = self._get_observation(state)

        return obs, new_state, reward, done, info

class BasicMathRenderer(JAXGameRenderer):
    def __init__(self, consts: BasicMathConstants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or BasicMathConstants()

        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160)
            )
        else:
            self.config = config
        
        self.jr = render_utils.JaxRenderingUtils(self.config)
        self.chosenGamemode = (self.consts.GAMEMODE - 1) % 4

        final_asset_config = list(self.consts.ASSET_CONFIG)

        wall_sprite = _create_background_sprite(self.consts, self.config.game_dimensions, self.chosenGamemode)

        final_asset_config.append({'name': 'background', 'type': 'background', 'data': wall_sprite},)

        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/basicmath"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)


        if self.PALETTE.shape[1] != 1:
            for i in range(1, len(self.PALETTE)):
                self.PALETTE = self.PALETTE.at[i].set(jnp.array(self.consts.COLOR_CODES[0][1], dtype=jnp.uint8))

    def _stack_num_masks(self) -> jnp.ndarray:
        """Helper to get all player-related masks from the main padded group."""
        # The first 16 are rotation, the next 3 are death animations
        return self.SHAPE_MASKS['nums']

    def render(self, state: BasicMathState):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        underscore_mask = self.SHAPE_MASKS["underscore"]
        symbol = self.SHAPE_MASKS["symbols"]


        raster = jax.lax.cond(
            jnp.logical_and(jnp.less(state.step_counter % 150, 120), jnp.equal(state.inactive, 0)),
            lambda: self.jr.render_at(raster, (self.consts.BAR_OFFSET  + state.arrPos * 15), self.consts.bar0[1], underscore_mask[0]),
            lambda: raster
        )
        raster = self.jr.render_at(raster, *self.consts.bar1, underscore_mask[1])
        raster = self.jr.render_at(raster, *self.consts.symbol, symbol[self.chosenGamemode])
        digit_masks = self._stack_num_masks()

        digit0 = self.jr.int_to_digits(state.problemNum1, max_digits=self.consts.problemNumLen)
        digit1 = self.jr.int_to_digits(state.problemNum2, max_digits=self.consts.problemNumLen)

        def render_problemNum1(i, r):
            num = digit0[i]
            digit = self.jr.int_to_digits(num, max_digits=1)

            return self.jr.render_label_selective(r, self.consts.num0[0] + i * self.consts.spacing, self.consts.num0[1], digit, digit_masks, 0, 1, spacing=0)
        
        def render_problemNum2(i, r):
            num = digit1[i]
            digit = self.jr.int_to_digits(num, max_digits=1)

            return self.jr.render_label_selective(r, self.consts.num1[0] + i * self.consts.spacing, self.consts.num1[1], digit, digit_masks, 0, 1, spacing=0)
        
        def render_nums(i, r):
            num = state.numArr[i]
            digit = self.jr.int_to_digits(num, max_digits=1)

            is_active = num != -1
            
            return jax.lax.cond(is_active, 
                                lambda ras: self.jr.render_label_selective(ras, self.consts.Y_OFFSET + i * 15, self.consts.num2[1], digit, digit_masks, 0, 1, spacing=0),
                                lambda ras: ras, 
                                r)
        
        raster = jax.lax.fori_loop(0, self.consts.problemNumLen, render_problemNum1, raster)
        raster = jax.lax.fori_loop(0, self.consts.problemNumLen, render_problemNum2, raster)
        raster = jax.lax.cond(
            jnp.less(state.inactive % 90, 60),
            lambda: jax.lax.fori_loop(0, self.consts.numArrLen, render_nums, raster),
            lambda: raster
        )

        return self.jr.render_from_palette(raster, self.PALETTE)