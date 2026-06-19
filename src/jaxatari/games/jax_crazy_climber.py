from encodings.punycode import digits
from functools import partial
import os
from enum import IntEnum

import chex
from flax import struct

from flax.nnx import state
import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import _create_wall_sprite
from jaxatari.renderers import JAXGameRenderer
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.rendering import jax_rendering_utils as render_utils
from typing import Tuple
import jaxatari.spaces as spaces

# Player movement states

class PlayerStableStates(IntEnum):
    Neutral = 0
    HalfPullUp = 1
    PullUp = 2
    
@chex.dataclass
class PlayerMoveState:
    main_state: PlayerStableStates 
    sub_step: int 
    side_step: int 
    hand_dir: int
    pos_x: float

class CrazyClimberState(struct.PyTreeNode):
    key: chex.PRNGKey
    score: chex.Array
    step_counter: chex.Array
    player_move_state: PlayerMoveState

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass

def _get_default_asset_config() -> tuple:
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        {'name': 'player_left_first_up_state_group', 'type': 'group', 'files': [
            'player_neutral_0.npy',
            'player_neutral_2_l.npy',
            'player_neutral_3_l.npy',
            'player_neutral_4_l.npy',
            'player_half_pull_up_0_l.npy',
            'player_half_pull_up_2_l.npy',
            'player_half_pull_up_3_l.npy',
            'player_half_pull_up_4_l.npy',
            'player_pull_up_0.npy',
            'player_pull_up_1.npy',
            'player_pull_up_4.npy',
            'player_pull_up_7.npy',
            ]},
        {'name': 'player_right_first_up_state_group', 'type': 'group', 'files': [
            'player_neutral_0.npy',
            'player_neutral_2_r.npy',
            'player_neutral_3_r.npy',
            'player_neutral_4_r.npy',
            'player_half_pull_up_0_r.npy',
            'player_half_pull_up_2_r.npy',
            'player_half_pull_up_3_r.npy',
            'player_half_pull_up_4_r.npy',
            'player_pull_up_0.npy',
            'player_pull_up_1.npy',
            'player_pull_up_4.npy',
            'player_pull_up_7.npy',
            ]},
        {'name': 'player_right_side_neutral_state_group', 'type': 'group', 'files': [
            'player_neutral_0.npy',
            'player_neutral_5_sideways_r.npy',
            'player_neutral_9_sideways_r.npy',
            ]},
        {'name': 'player_left_side_neutral_state_group', 'type': 'group', 'files': [
            'player_neutral_0.npy',
            'player_neutral_5_sideways_l.npy',
            'player_neutral_9_sideways_l.npy',
            ]},

        {'name': 'player_left_hand_right_side_half_pull_up_state_group', 'type': 'group', 'files': [
            'player_half_pull_up_0_l.npy',
            'player_half_pull_up_5_l_sideways_r.npy',
            'player_half_pull_up_9_l_sideways_r.npy',
            ]},
        {'name': 'player_left_hand_left_side_half_pull_up_state_group', 'type': 'group', 'files': [
            'player_half_pull_up_0_l.npy',
            'player_half_pull_up_5_l_sideways_l.npy',
            'player_half_pull_up_9_l_sideways_l.npy',
            ]},

        {'name': 'player_right_hand_right_side_half_pull_up_state_group', 'type': 'group', 'files': [
            'player_half_pull_up_0_r.npy',
            'player_half_pull_up_5_r_sideways_r.npy',
            'player_half_pull_up_9_r_sideways_r.npy',
            ]},
        {'name': 'player_right_hand_left_side_half_pull_up_state_group', 'type': 'group', 'files': [
            'player_half_pull_up_0_r.npy',
            'player_half_pull_up_5_r_sideways_l.npy',
            'player_half_pull_up_9_r_sideways_l.npy',
            ]},

        {'name': 'player_right_side_pull_up_state_group', 'type': 'group', 'files': [
            'player_pull_up_0.npy',
            'player_pull_up_5_sideways_r.npy',
            'player_pull_up_9_sideways_r.npy',
            ]},
        {'name': 'player_left_side_pull_up_state_group', 'type': 'group', 'files': [
            'player_pull_up_0.npy',
            'player_pull_up_5_sideways_l.npy',
            'player_pull_up_9_sideways_l.npy',
            ]},
    )

class CrazyClimberConstants(struct.PyTreeNode):
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0)),
    ASSET_CONFIG: tuple = _get_default_asset_config()

    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    PLAYER_Y: int = struct.field(pytree_node=False, default=140)
    PLAYER_DELTA_X: float = struct.field(pytree_node=False, default=5)

    SCORE_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(236, 236, 236))

class JaxCrazyClimber(JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]):
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT],
        dtype=jnp.int32,
    )

    def __init__(self, consts: CrazyClimberConstants = None):
        self.consts = consts or CrazyClimberConstants()
        super().__init__(self.consts)
        self.renderer = self.CrazyClimberRenderer(consts)

    def _score_and_reset(self, state: CrazyClimberState) -> CrazyClimberState:
        score_condition = jnp.array(True)

        score = jax.lax.cond(
            score_condition, # cond for score
            lambda s: s + jnp.array(1),
            lambda s: s,
            operand=state.score,
        )

        return state.replace(
            score=score,
        )

        initial_obs = self._get_observation(state)

        return initial_obs, state

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> (CrazyClimberObservation, CrazyClimberState):
        state_key, _step_key = jax.random.split(key)
        state = CrazyClimberState(
            key=state_key,
            score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            player_move_state=PlayerMoveState(main_state=PlayerStableStates.Neutral, sub_step=0, side_step=0, hand_dir=1, pos_x=96),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrazyClimberState, action: chex.Array) -> (CrazyClimberObservation, CrazyClimberState, float, bool, CrazyClimberInfo):
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        previous_state = state
        
        state = self._player_step(state, atari_action)
        state = self._score_and_reset(state)

        _, next_rng = jax.random.split(state.key)
        state = state.replace(key=next_rng)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info
    
    def _player_step(self, state: CrazyClimberState, action: chex.Array) -> CrazyClimberState:
        def move_upwards(s: PlayerMoveState): 
            is_up_move_possible = (jax.lax.abs(s.side_step) <= 3)
            transitioning_states = (((s.main_state != PlayerStableStates.PullUp) & (s.sub_step == 4)) |
                                    ((s.main_state == PlayerStableStates.PullUp) & (s.sub_step == 9)))
            next_state_on_transition = jnp.array([PlayerStableStates.Neutral, PlayerStableStates.HalfPullUp, PlayerStableStates.PullUp])[(s.main_state + 1) % 3] 
            next_hand_dir = jax.lax.select(
                transitioning_states & (next_state_on_transition == PlayerStableStates.Neutral),
                s.hand_dir * -1,
                s.hand_dir
            )
            return jax.lax.cond(
                is_up_move_possible,
                lambda s: jax.lax.cond(
                    transitioning_states,
                    lambda _: s.replace(main_state=next_state_on_transition, sub_step=0, hand_dir=next_hand_dir),
                    lambda s: s.replace(sub_step=s.sub_step + 1),
                    operand=s,
                ),
                lambda s: s,
                operand=s
            )

        def move_downwards(s: PlayerMoveState):
            is_down_move_possible = (jax.lax.abs(s.side_step) <= 3) & (s.sub_step > 0) & (s.main_state != PlayerStableStates.PullUp)
            next_hand_dir = jax.lax.select(
                (s.main_state == PlayerStableStates.Neutral) & (s.sub_step == 1),
                s.hand_dir * -1,
                s.hand_dir
            )
            return jax.lax.cond(
                is_down_move_possible,
                lambda s: jax.lax.cond(
                    (s.main_state == PlayerStableStates.HalfPullUp) & (s.sub_step == 1),
                    lambda _: s.replace(main_state=PlayerStableStates.Neutral, sub_step=0, hand_dir=s.hand_dir * -1),
                    lambda s: s.replace(sub_step=s.sub_step - 1, hand_dir=next_hand_dir),
                    operand=s
                ),
                lambda s: s,
                operand=s
            )
        
        def move_horizontal(s: PlayerMoveState, dir: int):
            is_right_move_possible = s.sub_step <= 1
            return jax.lax.cond(
                is_right_move_possible,
                lambda s: jax.lax.cond(
                    (jax.lax.abs(s.side_step) >= 12) & (jax.lax.sign(s.side_step) == jax.lax.sign(dir)),
                    lambda s: s.replace(side_step=0, pos_x=s.pos_x + dir * self.consts.PLAYER_DELTA_X),
                    lambda s: s.replace(side_step=s.side_step + dir),
                    operand=s,
                ),
                lambda s: s,
                operand=s,
            )

        up = action == Action.UP
        down = action == Action.DOWN
        left = action == Action.LEFT
        right = action == Action.RIGHT

        player_move_state = state.player_move_state
        action_state_cases = [
            up & (player_move_state.main_state != PlayerStableStates.PullUp),
            down & (player_move_state.main_state == PlayerStableStates.PullUp),
            down & (player_move_state.main_state != PlayerStableStates.PullUp),
            left,
            right,
        ]
        
        branch_idx = jnp.select(
            action_state_cases, 
            [0, 1, 2, 3, 4], 
            default=5
        )
        
        next_player_move_state = jax.lax.switch(
            branch_idx,
            [
                lambda s: move_upwards(s),
                lambda s: move_upwards(s),
                lambda s: move_downwards(s),
                lambda s: move_horizontal(s, -1),
                lambda s: move_horizontal(s, 1),
                lambda s: s
            ],
            operand=player_move_state
        )
        
        jax.debug.print(
            "main state: {x}, sub step: {y}, side step {z}, hand dir: {w}", 
            x=next_player_move_state.main_state, 
            y=next_player_move_state.sub_step,
            z=next_player_move_state.side_step,
            w=next_player_move_state.hand_dir,
        )

        return state.replace(
            player_move_state=next_player_move_state,
        )

    def render(self, state: CrazyClimberState) -> jnp.ndarray:
        return self.renderer.render(state)

    # TODO: Returntype needs to be altered to match actual implementation
    def action_space(self) -> spaces.Discrete:
        pass

    # TODO: Returntype needs to be altered to match actual implementation
    def observation_space(self) -> spaces.Dict:
        object_space = spaces.get_object_space(n=None, screen_size=(self.consts.HEIGHT, self.consts.WIDTH))

        return spaces.Dict({
            "player": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
        })

    # TODO: Returntype needs to be altered to match actual implementation
    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        ) 

    def _get_observation(self, state: CrazyClimberState) -> CrazyClimberObservation:
        pass
    
    def obs_to_flat_array(self, obs: CrazyClimberObservation) -> jnp.ndarray:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: CrazyClimberState) -> CrazyClimberInfo:
        pass

    def _get_reward(self, previous_state: CrazyClimberState, state: CrazyClimberState) -> float:
        return 0

    def _get_done(self, state: CrazyClimberState) -> bool:
        pass

    class CrazyClimberRenderer(JAXGameRenderer):
        def __init__(self, consts: CrazyClimberConstants = None, config: render_utils.RendererConfig = None):
            self.consts = consts or CrazyClimberConstants()
            super().__init__(consts)
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
            self.jr = render_utils.JaxRenderingUtils(self.config)
            
            final_asset_config = list(self.consts.ASSET_CONFIG)

            sprite_path = os.path.join(os.path.dirname(__file__), "sprites", "crazy_climber")

            (
                self.PALETTE,
                self.SHAPE_MASKS,
                self.BACKGROUND,
                self.COLOR_TO_ID,
                self.FLIP_OFFSETS
            ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

            self.PLAYER_UPWARDS_SPRITES = jnp.array([
                self.SHAPE_MASKS["player_left_first_up_state_group"],
                self.SHAPE_MASKS["player_right_first_up_state_group"],
            ])

            self.PLAYER_SIDEWAYS_SPRITES = jnp.array([
                jnp.array([
                    jnp.array([
                        self.SHAPE_MASKS["player_left_side_neutral_state_group"],
                        self.SHAPE_MASKS["player_right_side_neutral_state_group"],
                    ]),
                    # TODO: Placeholder, to fix dim mismatch. Just a workaround, will be reworked
                    jnp.array([
                        self.SHAPE_MASKS["player_left_side_neutral_state_group"],
                        self.SHAPE_MASKS["player_right_side_neutral_state_group"],
                    ]),
                    
                ]),
                jnp.array([
                    jnp.array([
                        self.SHAPE_MASKS["player_left_hand_left_side_half_pull_up_state_group"],
                        self.SHAPE_MASKS["player_left_hand_right_side_half_pull_up_state_group"],
                    ]),
                    jnp.array([
                        self.SHAPE_MASKS["player_right_hand_left_side_half_pull_up_state_group"],
                        self.SHAPE_MASKS["player_right_hand_right_side_half_pull_up_state_group"],
                    ]),
                ]),
                jnp.array([
                    jnp.array([
                        self.SHAPE_MASKS["player_left_side_pull_up_state_group"],
                        self.SHAPE_MASKS["player_right_side_pull_up_state_group"],
                    ]),
                    # TODO: Placeholder, to fix dim mismatch. Just a workaround, will be reworked
                    jnp.array([
                        self.SHAPE_MASKS["player_left_side_pull_up_state_group"],
                        self.SHAPE_MASKS["player_right_side_pull_up_state_group"],
                    ]),
                ]),
            ])

            self.PLAYER_UPWARDS_SPRITE_SEQUENCE = jnp.array([0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11, 11, 11])
            self.PLAYER_SIDEWAYS_SPRITE_SEQUENCE = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3])

        @partial(jax.jit, static_argnums=(0,))
        def render(self, state: CrazyClimberState) -> jnp.ndarray:
            raster = self.jr.create_object_raster(self.BACKGROUND)

            move_state = state.player_move_state

            sprite_index_up = self.PLAYER_UPWARDS_SPRITE_SEQUENCE[5 * move_state.main_state + move_state.sub_step]
            hand_index = jax.lax.switch(move_state.hand_dir, [
                lambda: 0,
                lambda: 1
            ])
            sprite_index_side = self.PLAYER_SIDEWAYS_SPRITE_SEQUENCE[jnp.abs(move_state.side_step)] 
            side_index = jnp.where(move_state.side_step > 0, 1, 0)

            def map_player_to_sprite(sprite_index_up, sprite_index_side, hand_index, side_index):
                return jax.lax.cond(
                    jnp.logical_and(move_state.sub_step <= 1, jnp.abs(move_state.side_step) > 3),
                    lambda _: self.PLAYER_SIDEWAYS_SPRITES[move_state.main_state][hand_index][side_index][sprite_index_side],
                    lambda _: self.PLAYER_UPWARDS_SPRITES[hand_index][sprite_index_up],
                    operand=None
                )
            
            player_sprite = map_player_to_sprite(sprite_index_up, sprite_index_side, hand_index, side_index)

            raster = self.jr.render_at(
                raster,
                jnp.round(state.player_move_state.pos_x).astype(jnp.int32),
                self.consts.PLAYER_Y,
                player_sprite,
            )

            digits = self.jr.int_to_digits(state.score, max_digits=6)
            digit_masks = self.SHAPE_MASKS["digits"]

            start_index = 1
            num_to_render = 6

            raster = self.jr.render_label_selective(raster, 55, 20, digits, digit_masks, start_index, num_to_render, spacing=8, max_digits_to_render=6)

            return self.jr.render_from_palette(raster, self.PALETTE)