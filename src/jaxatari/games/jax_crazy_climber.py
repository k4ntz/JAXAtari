import os

from functools import partial

from enum import IntEnum
import chex
from flax import struct
from typing import Tuple

import jax
import jax.numpy as jnp

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _create_block_sprite(color: tuple[int, int, int, int], shape: tuple[int, int]) -> jnp.ndarray:
    return jnp.tile(jnp.array(color, dtype=jnp.uint8), (*shape[:2], 1))

def _get_default_asset_config() -> tuple:
    wall_sprite = _create_block_sprite((0, 0, 148, 255), (169, 4))
    ceiling_sprite = _create_block_sprite((0, 48, 100, 255), (5, 80))
    return (
        {'name': 'background', 'type': 'background', 'file': 'misc/background.npy'},
        {'name': 'digits', 'type': 'digits', 'pattern': 'numbers/score_{}.npy'},
        {'name': 'life', 'type': 'single', 'file': 'misc/life.npy'},
        {'name': 'player_upwards_left_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/upwards/left_first/neutral_2.npy',
            'player/upwards/left_first/neutral_3.npy',
            'player/upwards/left_first/neutral_4.npy',
            'player/upwards/left_first/half_pull_up_0.npy',
            'player/upwards/left_first/half_pull_up_2.npy',
            'player/upwards/left_first/half_pull_up_3.npy',
            'player/upwards/left_first/half_pull_up_4.npy',
            'player/upwards/pull_up_0.npy',
            'player/upwards/pull_up_1.npy',
            'player/upwards/pull_up_4.npy',
            'player/upwards/pull_up_7.npy',
            ]},
        {'name': 'player_upwards_right_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/upwards/right_first/neutral_2.npy',
            'player/upwards/right_first/neutral_3.npy',
            'player/upwards/right_first/neutral_4.npy',
            'player/upwards/right_first/half_pull_up_0.npy',
            'player/upwards/right_first/half_pull_up_2.npy',
            'player/upwards/right_first/half_pull_up_3.npy',
            'player/upwards/right_first/half_pull_up_4.npy',
            'player/upwards/pull_up_0.npy',
            'player/upwards/pull_up_1.npy',
            'player/upwards/pull_up_4.npy',
            'player/upwards/pull_up_7.npy',
            ]},

        {'name': 'player_neutral_left_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/sideways/left/neutral_5.npy',
            'player/sideways/left/neutral_9.npy',
            ]},
        {'name': 'player_neutral_right_group', 'type': 'group', 'files': [
            'player/neutral_0.npy',
            'player/sideways/right/neutral_5.npy',
            'player/sideways/right/neutral_9.npy',
            ]},

        {'name': 'player_left_pull_up_left_group', 'type': 'group', 'files': [
            'player/upwards/left_first/half_pull_up_0.npy',
            'player/sideways/left/left_up/half_pull_up_5.npy',
            'player/sideways/left/left_up/half_pull_up_9.npy',
            ]},
        {'name': 'player_left_pull_up_right_group', 'type': 'group', 'files': [
            'player/upwards/left_first/half_pull_up_0.npy',
            'player/sideways/right/left_up/half_pull_up_5.npy',
            'player/sideways/right/left_up/half_pull_up_9.npy',
            ]},

        {'name': 'player_pull_up_left_group', 'type': 'group', 'files': [
            'player/upwards/pull_up_0.npy',
            'player/sideways/left/pull_up_5.npy',
            'player/sideways/left/pull_up_9.npy',
            ]},
        {'name': 'player_pull_up_right_group', 'type': 'group', 'files': [
            'player/upwards/pull_up_0.npy',
            'player/sideways/right/pull_up_5.npy',
            'player/sideways/right/pull_up_9.npy',
            ]},

        {'name': 'player_right_pull_up_left_group', 'type': 'group', 'files': [
            'player/upwards/right_first/half_pull_up_0.npy',
            'player/sideways/left/right_up/half_pull_up_5.npy',
            'player/sideways/left/right_up/half_pull_up_9.npy',
            ]},
        {'name': 'player_right_pull_up_right_group', 'type': 'group', 'files': [
            'player/upwards/right_first/half_pull_up_0.npy',
            'player/sideways/right/right_up/half_pull_up_5.npy',
            'player/sideways/right/right_up/half_pull_up_9.npy',
            ]},
        {'name': 'wall', 'type': 'procedural', 'data': wall_sprite},
        {'name': 'ceiling', 'type': 'procedural', 'data': ceiling_sprite},
        {'name': 'bird_left', 'type': 'group', 'files': [
            'bird/left/0.npy',
            'bird/left/4.npy',
            'bird/left/8.npy',
            'bird/left/12.npy',
            'bird/left/16.npy',
        ]},
        {'name': 'bird_right', 'type': 'group', 'files': [
            'bird/right/0.npy',
            'bird/right/4.npy',
            'bird/right/8.npy',
            'bird/right/12.npy',
            'bird/right/16.npy',
        ]},
    )

# Player movement states
class PlayerStableStates(IntEnum):
    NEUTRAL = 0
    HALF_PULL_UP = 1
    PULL_UP = 2
    
@chex.dataclass
class PlayerMoveState:
    main_state: PlayerStableStates 
    sub_step: int 
    side_step: int 
    hand_dir: int
    pos_x: int

@chex.dataclass
class BirdState:
    drop_coin: chex.Array
    pos_x: int
    pos_y: int
    dir: int
    
class CrazyClimberState(struct.PyTreeNode):
    key: chex.PRNGKey
    step_counter: chex.Array

    score: chex.Array
    was_at_apex: chex.Array
    bonus: chex.Array
    
    player_move_state: PlayerMoveState
    tower_step: chex.Array

    bird_state: BirdState

class CrazyClimberObservation(struct.PyTreeNode):
    pass

class CrazyClimberInfo(struct.PyTreeNode):
    pass


class CrazyClimberConstants(struct.PyTreeNode):
    BACKGROUND_COLOR: Tuple[int, int, int] = struct.field(pytree_node=False, default=(0, 0, 0)),
    ASSET_CONFIG: tuple = _get_default_asset_config()

    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)

    PLAYER_Y: int = struct.field(pytree_node=False, default=160)
    PLAYER_POSSIBLE_X: jnp.ndarray = struct.field(pytree_node=False, default=jnp.array([40, 46, 52, 58, 64, 72, 80, 86, 92, 98, 104]))
    PLAYER_WIDTH: int = 16
    PLAYER_HEIGHT: int = 23
    TOWER_POSSIBLE_SPRITE_CLIP: jnp.ndarray = struct.field(pytree_node=False, default=jnp.array([0, 4, 7, 10])) 

    BIRD_WIDTH: int = 15
    BIRD_HEIGHT: int = 10

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

    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> (CrazyClimberObservation, CrazyClimberState):
        state_key, _step_key = jax.random.split(key)
        state = CrazyClimberState(
            key=state_key,
            step_counter=jnp.array(0).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            was_at_apex=jnp.array(False),
            bonus=jnp.array(10000).astype(jnp.int32),
            player_move_state=PlayerMoveState(
                main_state=PlayerStableStates.NEUTRAL, 
                sub_step=0, side_step=0, 
                hand_dir=1, 
                pos_x=0),
            tower_step=0,
            bird_state=BirdState(
                drop_coin=jnp.array(False),
                pos_x=self.consts.BIRD_WIDTH,
                pos_y=40,
                dir=1),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CrazyClimberState, action: chex.Array) -> (CrazyClimberObservation, CrazyClimberState, float, bool, CrazyClimberInfo):
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))
        previous_state = state
        
        state = self._step_counter(state)
        state = self._player_step(state, atari_action)
        state = self._tower_step(state)
        state = self._bird_step(state)
        state = self._score_step(state)
        state = self._bonus_step(state)

        _, next_rng = jax.random.split(state.key)
        state = state.replace(key=next_rng)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def _step_counter(self, state: CrazyClimberState) -> CrazyClimberState:
        return state.replace(step_counter=state.step_counter + 1)

    @partial(jax.jit, static_argnums=(0,))
    def _tower_step(self, state: CrazyClimberState) -> CrazyClimberState:
        possible_tower_steps = jnp.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
            
        tower_step = jax.lax.cond(
            state.player_move_state.main_state == PlayerStableStates.PULL_UP,
            lambda: possible_tower_steps[state.player_move_state.sub_step],
            lambda: 0,
        )

        return state.replace(tower_step=tower_step) 
        
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: CrazyClimberState, action: chex.Array) -> CrazyClimberState:
        @partial(jax.jit)
        def move_upwards(s: PlayerMoveState) -> CrazyClimberState: 
            is_up_move_possible = (jax.lax.abs(s.side_step) <= 3)
            transitioning_states = (((s.main_state != PlayerStableStates.PULL_UP) & (s.sub_step == 4)) |
                                    ((s.main_state == PlayerStableStates.PULL_UP) & (s.sub_step == 9)))
            next_state_on_transition = jnp.array([PlayerStableStates.NEUTRAL, PlayerStableStates.HALF_PULL_UP, PlayerStableStates.PULL_UP])[(s.main_state + 1) % 3] 
            next_hand_dir = jax.lax.select(
                transitioning_states & (next_state_on_transition == PlayerStableStates.NEUTRAL),
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

        @partial(jax.jit)
        def move_downwards(s: PlayerMoveState) -> CrazyClimberState:
            is_down_move_possible = (jax.lax.abs(s.side_step) <= 3) & (s.sub_step > 0) & (s.main_state != PlayerStableStates.PULL_UP)
            next_hand_dir = jax.lax.select(
                (s.main_state == PlayerStableStates.NEUTRAL) & (s.sub_step == 1),
                s.hand_dir * -1,
                s.hand_dir
            )
            return jax.lax.cond(
                is_down_move_possible,
                lambda s: jax.lax.cond(
                    (s.main_state == PlayerStableStates.HALF_PULL_UP) & (s.sub_step == 1),
                    lambda _: s.replace(main_state=PlayerStableStates.NEUTRAL, sub_step=0, hand_dir=s.hand_dir * -1),
                    lambda s: s.replace(sub_step=s.sub_step - 1, hand_dir=next_hand_dir),
                    operand=s
                ),
                lambda s: s,
                operand=s
            )
        
        @partial(jax.jit)
        def move_horizontal(s: PlayerMoveState, dir: int) -> CrazyClimberState:
            is_right_move_possible = s.sub_step <= 1
            return jax.lax.cond(
                is_right_move_possible,
                lambda s: jax.lax.cond(
                    (jax.lax.abs(s.side_step) >= 12) & (jax.lax.sign(s.side_step) == jax.lax.sign(dir)),
                    lambda s: s.replace(side_step=0, pos_x=jnp.clip(s.pos_x + dir, 0, 10)),
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
            up & (player_move_state.main_state != PlayerStableStates.PULL_UP),
            down & (player_move_state.main_state == PlayerStableStates.PULL_UP),
            down & (player_move_state.main_state != PlayerStableStates.PULL_UP),
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
    
    @partial(jax.jit, static_argnums=(0,))
    def _bird_step(self, state: CrazyClimberState) -> CrazyClimberState:
        bird_state = state.bird_state
        
        hit_right_wall = bird_state.pos_x > self.consts.WIDTH - (self.consts.BIRD_WIDTH * 2)
        hit_left_wall  = bird_state.pos_x < self.consts.BIRD_WIDTH - 1

        should_move = (state.step_counter % 2 == 0)

        bird_state.dir = jnp.where(
            should_move & (hit_right_wall | hit_left_wall),
            bird_state.dir * -1,
            bird_state.dir
        )
        bird_state.pos_x = jnp.where(
            should_move, 
            bird_state.pos_x + bird_state.dir, 
            bird_state.pos_x
        )
        return state.replace(bird_state = bird_state)

    @partial(jax.jit, static_argnums=(0,))
    def _score_step(self, state: CrazyClimberState) -> CrazyClimberState:
        current_at_apex = (state.player_move_state.sub_step == 9) & (state.player_move_state.main_state == PlayerStableStates.PULL_UP)

        score_triggered = (state.player_move_state.main_state == PlayerStableStates.NEUTRAL) & state.was_at_apex

        new_score = jax.lax.select(score_triggered, state.score + 100, state.score)

        return state.replace(
            score=new_score,
            was_at_apex=current_at_apex
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _bonus_step(self, state: CrazyClimberState) -> CrazyClimberState: 
        bonus_condition = (state.step_counter > 1228) & ((state.step_counter - 1229) % 600 == 0)

        bonus = jnp.where(bonus_condition, state.bonus - 100, state.bonus)

        return state.replace(bonus=bonus)

    
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
                self.SHAPE_MASKS["player_upwards_left_group"],
                self.SHAPE_MASKS["player_upwards_right_group"],
            ])

            self.PLAYER_SIDEWAYS_SPRITES = jnp.array([
                jnp.array([
                    self.SHAPE_MASKS["player_neutral_left_group"],
                    self.SHAPE_MASKS["player_neutral_right_group"],
                ]),
                jnp.array([
                    self.SHAPE_MASKS["player_pull_up_left_group"],
                    self.SHAPE_MASKS["player_pull_up_right_group"],
                ]),
            ])

            self.PLAYER_SIDEWAYS_SPRITES_ARM_SPECIFIC = jnp.array([
                jnp.array([
                    self.SHAPE_MASKS["player_left_pull_up_left_group"],
                    self.SHAPE_MASKS["player_left_pull_up_right_group"],
                ]),
                jnp.array([
                    self.SHAPE_MASKS["player_right_pull_up_left_group"],
                    self.SHAPE_MASKS["player_right_pull_up_right_group"],
                ]),
            ])

            self.BIRD_SPRITES = jnp.array([
                self.SHAPE_MASKS["bird_left"],
                self.SHAPE_MASKS["bird_right"],
            ])

            self.PLAYER_UPWARDS_SPRITE_SEQUENCE = jnp.array([0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11])
            self.PLAYER_SIDEWAYS_SPRITE_SEQUENCE = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3])

            self.BIRD_SEQUENCE = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0])
            
            self.TOWER_SPRITE = self._render_tower_sprite()
            
        def _render_tower_sprite(self) -> jnp.ndarray:
            tower_raster = self._create_raster((169, 80))

            wall_offset_x = jnp.array([0, 12, 24, 36, 40, 52, 64, 76])
            wall_offset_y = jnp.array([0,  0,  0,  0,  0,  0,  0,  0])
            wall_sprite = self.SHAPE_MASKS["wall"]
            wall_sprite_masks = jnp.repeat(wall_sprite[jnp.newaxis, :, :], len(wall_offset_x), axis=0)
            tower_raster = self.jr.render_at_batch(
                tower_raster,
                wall_offset_x,
                wall_offset_y,
                wall_sprite_masks,
            )
            
            ceiling_offset_x = jnp.array([0,  0,  0,  0,  0,  0,  0,  0,  0,   0,   0,   0,   0,   0])
            ceiling_offset_y = jnp.array([0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169])
            ceiling_sprite = self.SHAPE_MASKS["ceiling"]
            ceiling_sprite_masks = jnp.repeat(ceiling_sprite[jnp.newaxis, :, :], len(ceiling_offset_x), axis=0)
            tower_raster = self.jr.render_at_batch(
                tower_raster,
                ceiling_offset_x,
                ceiling_offset_y,
                ceiling_sprite_masks,
            )
            
            return tower_raster

        @partial(jax.jit, static_argnums=(0,1))
        def _create_raster(self, shape: tuple[int, int]) -> jnp.ndarray:
            return jnp.zeros(shape=shape, dtype=jnp.uint8)
        
        @partial(jax.jit, static_argnums=(0,))
        def _clip_raster(self, base: jnp.ndarray, overlay: jnp.ndarray, offset_x: int, offset_y: int) -> jnp.ndarray:
            base_slice = jax.lax.dynamic_slice(
                base,
                (offset_y, offset_x),
                overlay.shape
            )
            
            merged_slice = jnp.where(overlay != 0, overlay, base_slice)
            base = self.jr.render_at_clipped(
                base,
                offset_x, offset_y,
                merged_slice
            )
            
            return base
        
        @partial(jax.jit, static_argnums=(0,))
        def _render_player(self, state: CrazyClimberState) -> jnp.ndarray:
            player_raster = self._create_raster((self.consts.PLAYER_HEIGHT, self.consts.PLAYER_WIDTH))

            move_state = state.player_move_state

            sprite_index_up = self.PLAYER_UPWARDS_SPRITE_SEQUENCE[5 * move_state.main_state + move_state.sub_step]
            hand_index = (move_state.hand_dir < 0).astype(int) # maps -1 -> 1 and 1 -> 0
            sprite_index_side = self.PLAYER_SIDEWAYS_SPRITE_SEQUENCE[jnp.abs(move_state.side_step)] 
            side_index = (move_state.side_step > 0).astype(int) # maps -1 -> 1 and 1 -> 0

            @partial(jax.jit)
            def map_player_to_sprite(sprite_index_up: int, sprite_index_side: int, hand_index: int, side_index: int) -> jnp.ndarray:
                return jax.lax.cond(
                    jnp.logical_and(move_state.sub_step <= 1, jnp.abs(move_state.side_step) > 3),
                    lambda _: jax.lax.cond(
                        move_state.main_state == PlayerStableStates.HALF_PULL_UP,
                        lambda _: self.PLAYER_SIDEWAYS_SPRITES_ARM_SPECIFIC[hand_index][side_index][sprite_index_side],
                        lambda _: self.PLAYER_SIDEWAYS_SPRITES[move_state.main_state][side_index][sprite_index_side],
                        operand=None),
                    lambda _: self.PLAYER_UPWARDS_SPRITES[hand_index][sprite_index_up],
                    operand=None
                )
            
            player_sprite = map_player_to_sprite(sprite_index_up, sprite_index_side, hand_index, side_index)
            player_raster = self.jr.render_at(
                player_raster, 
                0, 0,
                player_sprite,
            )
            
            return player_raster
        

        @partial(jax.jit, static_argnums=(0,))
        def _render_tower(self, state: CrazyClimberState) -> jnp.ndarray:
            tower_sprite = self.TOWER_SPRITE

            top_clip = 13 - self.consts.TOWER_POSSIBLE_SPRITE_CLIP[state.tower_step]
            tower_raster = jax.lax.dynamic_slice_in_dim(
                tower_sprite,
                top_clip,
                156,
                axis=0
            )
            
            return tower_raster
        
        @partial(jax.jit, static_argnums=(0,))
        def _render_bird(self, state: CrazyClimberState) -> jnp.ndarray:
            bird_raster = self._create_raster((self.consts.BIRD_HEIGHT, self.consts.BIRD_WIDTH))

            bird_state = state.bird_state

            dir_index = (bird_state.dir > 0).astype(int)
            bird_index = self.BIRD_SEQUENCE[bird_state.pos_x % 40]

            bird_sprite = self.BIRD_SPRITES[dir_index][bird_index]

            bird_raster = self.jr.render_at(
                bird_raster, 
                0, 0,
                bird_sprite,
            )
            
            return bird_raster

        @partial(jax.jit, static_argnums=(0,))
        def render(self, state: CrazyClimberState) -> jnp.ndarray:
            raster = self.jr.create_object_raster(self.BACKGROUND) # can be substituted with self._create_raster((210, 160))

            player_raster = self._render_player(state)
            tower_raster = self._render_tower(state)
            bird_raster = self._render_bird(state)
            raster = self._clip_raster(raster, tower_raster, 40, 44) # self.jr.render_at_clipped(raster, 0, 0, tower_raster)
            raster = self._clip_raster(raster, player_raster, self.consts.PLAYER_POSSIBLE_X[state.player_move_state.pos_x], self.consts.PLAYER_Y) # self.jr.render_at_clipped(raster, state.player_move_state.pos_x, self.consts.PLAYER_Y, player_raster)
            raster = self._clip_raster(raster, bird_raster, state.bird_state.pos_x, state.bird_state.pos_y)

            score_digits = self.jr.int_to_digits(state.score, max_digits=6)
            bonus_digits = self.jr.int_to_digits(state.bonus, max_digits=5)
            digit_masks = self.SHAPE_MASKS["digits"]

            life_mask = self.SHAPE_MASKS["life"]
            raster = self.jr.render_at(raster, 58, 12, life_mask)
            raster = self.jr.render_at(raster, 74, 12, life_mask)
            raster = self.jr.render_at(raster, 90, 12, life_mask)
            
            raster = self.jr.render_label_selective(raster, 57, 20, bonus_digits, digit_masks, start_index=0, num_to_render=5, spacing=8, max_digits_to_render=6)
            raster = self.jr.render_label_selective(raster, 49, 30, score_digits, digit_masks, start_index=0, num_to_render=6, spacing=8, max_digits_to_render=6)

            return self.jr.render_from_palette(raster, self.PALETTE)