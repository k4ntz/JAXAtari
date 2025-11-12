import random
from turtle import left
from jax import random as jrandom
from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import (
    JaxEnvironment,
    JAXAtariAction as Action,
    EnvObs,
    EnvState,
    EnvInfo,
)
from jaxatari.spaces import Space
from abc import ABC, abstractmethod
from typing import Tuple, Any
import chex


class DefenderConstants(NamedTuple):
    MAX_SPEED: int = 4
    ENEMY_STEP_SIZE: float = 0.35
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_ACCELERATION: int = 0.15
    PLAYER_BREAK: int = 0.1
    BACKGROUND_COLOR: Tuple[int, int, int] = (144, 72, 17)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    SCORE_COLOR: Tuple[int, int, int] = (236, 236, 236)
    SPACE_SHIP_COLOR: Tuple[int, int, int] = (0, 0, 200)
    PLAYER_X: int = 140
    PLAYER_Y: int = 80
    PLAYER_SIZE: Tuple[int, int] = (16, 16)
    CAMERA_OFFSET_MAX: int = 40
    CAMERA_WIDTH: int = 160
    CAMERA_TOP_BOUND: int = 37
    CAMERA_X: int = 80
    CAMERA_Y: int = 80
    BOMBER_AMOUNT: Tuple[int, int, int, int, int] = (1, 2, 2, 2, 2)
    MAX_BOMBER_AMOUNT: int = 1
    LANDER_AMOUNT: Tuple[int, int, int, int, int] = (18, 18, 19, 20, 20)
    MAX_LANDER_AMOUNT: int = 5
    POD_AMOUNT: Tuple[int, int, int, int, int] = (2, 2, 3, 3, 3)
    BAITER_TIME_SEC: int = 20
    SWARM_SPAWN_MIN: int = 1
    SWARM_SPAWN_MAX: int = 2
    BULLET_SPEED: int = 2
    BULLET_COLOR: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]] = (
        (200, 0, 0),      # Red
        (255, 255, 0),    # Yellow
        (0, 0, 255)       # Blue
    )
    ENEMY_SPEED: float = 0.06

    ACTIVE: int = 1
    INACTIVE: int = 0                            # active, x_pos, y_pos
    ENEMY_SPEED: float = 0.24
    LANDER_Y_SPEED: float = 0.08
    BOMBER_Y_SPEED: float = -0.2
    INIT_LANDER_STATES: chex.Array = jnp.array([
        # active, pos_x, pos_y, speed,
        [1, random.randint(130, 480), random.randint(39, 160)]
        for _ in range(MAX_LANDER_AMOUNT)
    ]).reshape(MAX_LANDER_AMOUNT, 3)
    
    INIT_BOMBER_STATES: chex.Array = jnp.array([
        # active, pos_x, pos_y, direction_right
        [1, 120, 70, 0]
        for _ in range(MAX_BOMBER_AMOUNT)
    ]).reshape(MAX_BOMBER_AMOUNT, 4)


# immutable state container
class DefenderState(NamedTuple):
    space_ship_speed: chex.Array
    space_ship_x: chex.Array
    space_ship_y: chex.Array
    space_ship_facing_right: chex.Array
    camera_offset: chex.Array
    step_counter: chex.Array
    lander_states: chex.Array
    bomber_states: chex.Array
    bullet_state: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_dir_x: chex.Array
    bullet_dir_y: chex.Array
    bullet_color_pos: chex.Array
    # pod_array: chex.Array
    # bomber_array: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class DefenderObservation(NamedTuple):
    player: EntityPosition
    score: jnp.ndarray


class DefenderInfo(NamedTuple):
    time: jnp.ndarray


class DefenderRenderer(JAXGameRenderer):
    def __init__(self, consts: DefenderConstants = None):
        super().__init__()
        self.consts = consts or DefenderConst
    def __init__(self, consts: DefenderConstants = None):
        super().__init__()
        self.consts = consts or DefenderConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            # downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        # 1. Create procedural assets for both walls

        # 2. Update asset config to include both walls
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/defender"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [
            {"name": "background", "type": "background", "file": "ui_overlay.npy"},
            {"name": "space_ship", "type": "single", "file": "space_ship.npy"},
            {"name": "baiter", "type": "single", "file": "baiter.npy"},
            {"name": "bomber", "type": "single", "file": "bomber.npy"},
            {"name": "lander", "type": "single", "file": "lander.npy"},
            {"name": "mutant", "type": "single", "file": "mutant.npy"},
            {"name": "pod", "type": "single", "file": "pod.npy"},
            {"name": "swarmers", "type": "single", "file": "swarmers.npy"},
            {"name": "ui_overlay", "type": "single", "file": "ui_overlay.npy"},
            {"name": "city", "type": "single", "file": "city.npy"},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DefenderState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)


        city_start = -(state.space_ship_x + state.camera_offset) % 80
        city_mask = self.SHAPE_MASKS["city"]
        raster = self.jr.render_at_clipped(
            raster,
            city_start,
            160,
            city_mask,
        )
        raster = self.jr.render_at_clipped(
            raster,
            city_start-80,
            160,
            city_mask,
        )
        raster = self.jr.render_at_clipped(
            raster,
            city_start+80,
            160,
            city_mask,
        )

        # render all Landers
        for i in range(state.lander_states.shape[0]):
            lander = state.lander_states[i]
            is_active = lander[0] == self.consts.ACTIVE
            pos_x = lander[1]
            pos_y = lander[2]
            raster = jax.lax.cond(
                is_active,
                lambda raster: self.jr.render_at_clipped(
                    raster,
                    self.consts.CAMERA_X - state.camera_offset + pos_x - state.space_ship_x,
                    pos_y,
                    self.SHAPE_MASKS["lander"],
                ),
                lambda raster: raster,
                operand=raster,
            )

        # render all Bombers
        for i in range(state.bomber_states.shape[0]):
            bomber = state.bomber_states[i]
            is_active = bomber[0] == self.consts.ACTIVE
            pos_x = bomber[1]
            pos_y = bomber[2]
            direction_right = bomber[3]
            raster = jax.lax.cond(
                is_active,
                lambda raster: self.jr.render_at_clipped(
                    raster,
                    self.consts.CAMERA_X - state.camera_offset + pos_x - state.space_ship_x,
                    pos_y,
                    self.SHAPE_MASKS["bomber"],
                    flip_horizontal= jnp.logical_not(direction_right),
                ),
                lambda raster: raster,
                operand=raster,
            )
            
        
        space_ship_mask = self.SHAPE_MASKS["space_ship"]

        space_ship_facing_left = jnp.where(state.space_ship_facing_right, False, True)

        raster = self.jr.render_at(
            raster,
            self.consts.CAMERA_X - state.camera_offset,
            state.space_ship_y,
            space_ship_mask,
            flip_horizontal=space_ship_facing_left,
        )
        
        return self.jr.render_from_palette(raster, self.PALETTE)


class JaxDefender(
    JaxEnvironment[DefenderState, DefenderObservation, DefenderInfo, DefenderConstants]
):

    def __init__(self, consts: DefenderConstants = None):
        consts = consts or DefenderConstants()
        super().__init__(consts)
        self.renderer = DefenderRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ]

    def _is_onscreen(self, state: DefenderState, pos_x: int) -> bool:
        camera_left = state.space_ship_x + state.camera_offset - self.consts.CAMERA_WIDTH/2
        camera_right = camera_left + self.consts.CAMERA_WIDTH
        return jnp.logical_and(pos_x >= camera_left, pos_x < camera_right)

    def _get_onscreen_enemies(self, state: DefenderState) -> chex.Array:
        
        pass

    def _player_step(self, state: DefenderState, action: chex.Array) -> DefenderState:
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )

        direction_x = jnp.where(left, -1, 0) + jnp.where(right, 1, 0)
        direction_y = jnp.where(up, -1, 0) + jnp.where(down, 1, 0)

        space_ship_facing_right = state.space_ship_facing_right
        space_ship_facing_right = jax.lax.cond(
            direction_x != 0,
            lambda _: direction_x > 0,
            lambda _: state.space_ship_facing_right,
            operand=None,
        )

        space_ship_speed = jax.lax.cond(direction_x != 0, 
            lambda _ : state.space_ship_speed + direction_x * self.consts.PLAYER_ACCELERATION,
            lambda _ : state.space_ship_speed * (1 - self.consts.PLAYER_BREAK),
            operand=None
        )

        space_ship_speed = jnp.clip(
            space_ship_speed, -self.consts.MAX_SPEED, self.consts.MAX_SPEED
        )

        space_ship_x = state.space_ship_x + space_ship_speed
        space_ship_y = state.space_ship_y + direction_y

        return DefenderState(
            space_ship_speed=space_ship_speed,
            space_ship_x=space_ship_x,
            space_ship_y=space_ship_y,
            space_ship_facing_right=space_ship_facing_right,
            camera_offset=state.camera_offset,
            step_counter=state.step_counter + 1,
            lander_states=state.lander_states,
            bomber_states=state.bomber_states,
            bullet_state = state.bullet_state, 
            bullet_x = state.bullet_x,
            bullet_y = state.bullet_y,
            bullet_dir_x = state.bullet_dir_x,
            bullet_dir_y = state.bullet_dir_y,
            bullet_color_pos = state.bullet_color_pos, 
        )

    def _bullet_step(self, state: DefenderState) -> chex.Array: 
        bullet_x = jax.lax.cond(state.bullet_state, 
        lambda _: state.bullet_x + state.bullet_dir_x * self.consts.BULLET_SPEED, 
        lambda _: state.bullet_x, 
        operand = None)
        bullet_y = jax.lax.cond(state.bullet_state, 
        lambda _: state.bullet_y + state.bullet_dir_y * self.consts.BULLET_SPEED, 
        lambda _: state.bullet_y, 
        operand = None)

        bullet_state = self._is_onscreen(state, bullet_x)
        bullet_color_pos = (state.bullet_color_pos + 1) % len(self.consts.BULLET_COLOR)

        return DefenderState(
            space_ship_speed=state.space_ship_speed,
            space_ship_x=state.space_ship_x,
            space_ship_y=state.space_ship_y,
            space_ship_facing_right=state.space_ship_facing_right,
            camera_offset=state.camera_offset,
            step_counter=state.step_counter + 1,
            lander_states=state.lander_states,
            bomber_states=state.bomber_states,
            bullet_state = bullet_state, 
            bullet_x = bullet_x,
            bullet_y = bullet_y,
            bullet_dir_x = state.bullet_dir_x,
            bullet_dir_y = state.bullet_dir_y,
            bullet_color_pos = bullet_color_pos, 
        )
    
    def _shoot(self, state: DefenderState, pos_x: float, pos_y: float, dir_x: int, dir_y:int) -> chex.Array: 
        return DefenderState(
            space_ship_speed=state.space_ship_speed,
            space_ship_x=state.space_ship_x,
            space_ship_y=state.space_ship_y,
            space_ship_facing_right=state.space_ship_facing_right,
            camera_offset=state.camera_offset,
            step_counter=state.step_counter + 1,
            lander_states=state.lander_states,
            bomber_states=state.bomber_states,
            bullet_state = True, 
            bullet_x = pos_x,
            bullet_y = pos_y,
            bullet_dir_x= dir_x,
            bullet_dir_y= dir_y,
            bullet_color_pos = state.bullet_color_pos, 
        )
        

    def _lander_step(self, state: DefenderState, lander: chex.Array) -> chex.Array:   
        x_pos = lander[1]
        y_pos = lander[2]
    
        speed = self.consts.ENEMY_SPEED 
        # acceleration in x direction
        speed = jax.lax.cond(
            state.space_ship_facing_right,
            lambda s: -s,
            lambda s: s,
            operand=speed,
        )
        
        x_pos = x_pos + speed
        
        # movement in y direction
        y_pos += self.consts.LANDER_Y_SPEED
        y_pos = 39 + (y_pos - 39) % (160 - 39 + 1)
        
        lander = lander.at[1].set(x_pos)
        lander = lander.at[2].set(y_pos)
        lander = lander.at[3].set(speed)
        return lander

    def _lander_step_all(self, state: DefenderState) -> DefenderState:
        new_lander_states = []
        for i in range(state.lander_states.shape[0]):
            lander = state.lander_states[i]
            lander = jax.lax.cond(
                lander[0] == self.consts.INACTIVE,
                lambda _: lander,
                lambda _: self._lander_step(state, lander),
                operand=None,
            )
            new_lander_states.append(lander)
          
        return DefenderState(
            space_ship_speed=state.space_ship_speed,
            space_ship_x=state.space_ship_x,
            space_ship_y=state.space_ship_y,
            space_ship_facing_right=state.space_ship_facing_right,
            camera_offset=state.camera_offset,
            step_counter=state.step_counter,
            lander_states=jnp.array(new_lander_states),
            bomber_states=state.bomber_states,
            bullet_state = state.bullet_state, 
            bullet_x = state.bullet_x,
            bullet_y = state.bullet_y,
            bullet_dir_x = state.bullet_dir_x,
            bullet_dir_y = state.bullet_dir_y,
            bullet_color_pos = state.bullet_color_pos, 
        )
    
    def _bomber_step(self, state: DefenderState, bomber: chex.Array) -> chex.Array:
        # The bomber has a direction and when the spaceship is crossed and has passed it by 30 the direction changes
        x_pos = bomber[1]
        y_pos = bomber[2]
        direction_right = bomber[3]
        speed = self.consts.ENEMY_SPEED
        # acceleration in x direction
        speed = jax.lax.cond(
            direction_right,
            lambda s: s,
            lambda s: -s,
            operand=speed,
        )
        x_pos = x_pos + speed
        # movement in y direction
        y_pos += self.consts.BOMBER_Y_SPEED
        y_pos = 39 + (y_pos - 39) % (160 - 39 + 1)
        # change direction if spaceship is crossed and passed by 30
        direction_right = jax.lax.cond(
            jnp.logical_and(
                direction_right,
                x_pos > state.space_ship_x + 30
            ),
            lambda _: 0.0,
            lambda _: jax.lax.cond(
                jnp.logical_and(
                    jnp.logical_not(direction_right),
                    x_pos < state.space_ship_x - 30
                ),
                lambda _: 1.0,
                lambda _: direction_right,
                operand=None,
            ),
            operand=None,
        )
        jax.debug.print("Bomber pos_x: {}, pos_y: {}, dir_right: {} X-pos - 30 {}", x_pos, y_pos, direction_right,(state.space_ship_x - 30) )
                
        bomber = bomber.at[1].set(x_pos)
        bomber = bomber.at[2].set(y_pos)
        bomber = bomber.at[3].set(direction_right)
        return bomber


    def _bomber_step_all(self, state: DefenderState) -> DefenderState:
        new_bomber_states = []
        for i in range(state.bomber_states.shape[0]):
            bomber = state.bomber_states[i]
            bomber = jax.lax.cond(
                bomber[0] == self.consts.INACTIVE,
                lambda _: bomber,
                lambda _: self._bomber_step(state, bomber),
                operand=None,
            )
            new_bomber_states.append(bomber)
          
        return DefenderState(
            space_ship_speed=state.space_ship_speed,
            space_ship_x=state.space_ship_x,
            space_ship_y=state.space_ship_y,
            space_ship_facing_right=state.space_ship_facing_right,
            camera_offset=state.camera_offset,
            step_counter=state.step_counter,
            lander_states=state.lander_states,
            bomber_states=jnp.array(new_bomber_states),
            bullet_state = state.bullet_state, 
            bullet_x = state.bullet_x,
            bullet_y = state.bullet_y,
            bullet_dir_x= state.bullet_dir_x,
            bullet_dir_y= state.bullet_dir_y,
            bullet_color_pos = state.bullet_color_pos, 
        )

    def _shooting_step(self, state: DefenderState) -> DefenderState:

        return DefenderState(
            space_ship_speed=state.space_ship_speed,
            space_ship_x=state.space_ship_x,
            space_ship_y=state.space_ship_y,
            space_ship_facing_right=state.space_ship_facing_right,
            camera_offset=state.camera_offset,
            step_counter=state.step_counter,
            lander_states=state.lander_states,
            bomber_states=state.bomber_states,
            bullet_state = state.bullet_state, 
            bullet_x = state.bullet_x,
            bullet_y = state.bullet_y,
            bullet_dir_x= state.bullet_dir_x,
            bullet_dir_y= state.bullet_dir_y,
            bullet_color_pos = state.bullet_color_pos, 
        )


    def _camera_step(self, state: DefenderState) -> DefenderState:
        offset_gain = 2
        camera_offset = state.camera_offset
        camera_offset += jnp.where(state.space_ship_facing_right, 1, -1) * offset_gain

        camera_offset = jnp.clip(
            camera_offset,
            -self.consts.CAMERA_OFFSET_MAX,
            self.consts.CAMERA_OFFSET_MAX,
        )

        return DefenderState(
            space_ship_speed=state.space_ship_speed,
            space_ship_x=state.space_ship_x,
            space_ship_y=state.space_ship_y,
            space_ship_facing_right=state.space_ship_facing_right,
            camera_offset=camera_offset,
            step_counter=state.step_counter,
            lander_states=state.lander_states,
            bomber_states=state.bomber_states,
            bullet_state = state.bullet_state, 
            bullet_x = state.bullet_x,
            bullet_y = state.bullet_y,
            bullet_dir_x= state.bullet_dir_x,
            bullet_dir_y= state.bullet_dir_y,
            bullet_color_pos = state.bullet_color_pos, 
        )

    def reset(self, key=None) -> Tuple[DefenderObservation, DefenderState]:
        initial_state = DefenderState(
            space_ship_speed=jnp.array(0).astype(jnp.float32),
            space_ship_x=jnp.array(20).astype(jnp.int32),
            space_ship_y=jnp.array(100).astype(jnp.int32),
            space_ship_facing_right=jnp.array(True, dtype=jnp.bool),
            step_counter=jnp.array(0).astype(jnp.int32),
            camera_offset=jnp.array(0).astype(jnp.int32),
            lander_states=jnp.array(self.consts.INIT_LANDER_STATES).astype(jnp.float32),
            bomber_states=jnp.array(self.consts.INIT_BOMBER_STATES).astype(jnp.float32),
            bullet_state=jnp.array(True, dtype=jnp.bool),
            bullet_x=jnp.array(0).astype(jnp.float32),
            bullet_y=jnp.array(0).astype(jnp.float32),
            bullet_dir_x=jnp.array(0).astype(jnp.int32),
            bullet_dir_y=jnp.array(0).astype(jnp.int32),
            bullet_color_pos=jnp.array(0).astype(jnp.int32),
        )
        observation = self._get_observation(initial_state)
        return observation, initial_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: DefenderState, action: chex.Array
    ) -> Tuple[DefenderObservation, DefenderState, float, bool, DefenderInfo]:
        state = self._player_step(state, action)
        state = self._lander_step_all(state)
        state = self._bomber_step_all(state)
        state = self._camera_step(state)
        state = self._bullet_step(state)
        state = self._shooting_step(state)
        # state = self._collision_step(state)
        observation = self._get_observation(state)
        env_reward = self._get_reward(state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        return observation, state, env_reward, done, info

    def render(self, state: DefenderState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Space:
        pass

    def observation_space(self) -> spaces.Space:
        pass

    def image_space(self) -> Space:
        pass

    def _get_observation(self, state: DefenderState) -> DefenderObservation:
        return DefenderObservation(
            player=EntityPosition(
                x=state.space_ship_x,
                y=state.space_ship_y,
                width=jnp.array(self.consts.PLAYER_SIZE[0]),
                height=jnp.array(self.consts.PLAYER_SIZE[1]),
        ),
        score=0
        )

    def observation_spaces(self) -> spaces.Space:
        pass

    def _get_info(
        self, state: DefenderState, all_rewards: jnp.array = None
    ) -> DefenderInfo:
        return DefenderInfo(time=state.step_counter)

    def _get_reward(self, previous_state: DefenderState, state: DefenderState) -> float:
        return 0.0

    def _get_done(self, state: DefenderState) -> bool:
        return False
