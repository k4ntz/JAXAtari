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
    # Game Window
    WINDOW_WIDTH: int = 160
    WINDOW_HEIGHT: int = 210
    GAME_WIDTH: int = 480
    GAME_HEIGHT: int = 135
    GAME_AREA_TOP: int = 38
    GAME_AREA_BOTTOM: int = GAME_AREA_TOP + GAME_HEIGHT

    # Camera
    CAMERA_WINDOW_X: int = 80
    CAMERA_OFFSET_MAX: int = 40
    CAMERA_OFFSET_GAIN: int = 2
    INITIAL_CAMERA_GAME_X: int = 240
    INITIAL_CAMERA_OFFSET: int = 40

    # UI
    CITY_WIDTH: int = 80
    CITY_HEIGHT: int = 13

    # Space Ship
    SPACE_SHIP_ACCELERATION: float = 0.15
    SPACE_SHIP_BREAK: float = 0.1
    SPACE_SHIP_MAX_SPEED: float = 4.0
    SPACE_SHIP_COLOR: Tuple[int, int, int] = (0, 0, 200)
    SPACE_SHIP_WIDTH: int = 13
    SPACE_SHIP_HEIGHT: int = 5

    # Enemy
    ACTIVE: int = 1
    INACTIVE: int = 0  # active, x_pos, y_pos
    ENEMY_SPEED: float = 2
    ENEMY_WIDTH: int = 13
    ENEMY_HEIGHT: int = 7
    ENEMY_MAX: int = 20

    # Types
    INACTIVE: int = 0
    LANDER: int = 1
    POD: int = 2
    BOMBER: int = 3
    SWARMERS: int = 4
    MUTANT: int = 5
    BAITER: int = 6

    # Bomber
    BOMBER_AMOUNT: Tuple[int, int, int, int, int] = (1, 2, 2, 2, 2)
    MAX_BOMBER_AMOUNT: int = 1
    BOMBER_Y_SPEED: float = -0.2

    # Lander
    LANDER_AMOUNT: Tuple[int, int, int, int, int] = (18, 18, 19, 20, 20)
    MAX_LANDER_AMOUNT: int = 5
    LANDER_Y_SPEED: float = 0.08
    LANDER_STATE_PATROL: int = 0
    LANDER_STATE_DESCEND: int = 1
    LANDER_STATE_ASCEND: int = 2

    # Pod
    POD_AMOUNT: Tuple[int, int, int, int, int] = (2, 2, 3, 3, 3)
    MAX_POD_AMOUNT: int = 3

    # Baiter
    BAITER_TIME_SEC: int = 20
    SWARM_SPAWN_MIN: int = 1
    SWARM_SPAWN_MAX: int = 2

    # Bullet
    BULLET_SPEED: int = 2
    BULLET_COLOR: Tuple[
        Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]
    ] = (
        (200, 0, 0),  # Red
        (255, 255, 0),  # Yellow
        (0, 0, 255),  # Blue
    )

    # Initial Wave State
    # Positions are in game world positions
    INITIAL_SPACE_SHIP_X: int = INITIAL_CAMERA_GAME_X - INITIAL_CAMERA_OFFSET
    INITIAL_SPACE_SHIP_Y: int = 80
    INITIAL_SPACE_SHIP_FACING_RIGHT: bool = True
    INITIAL_ENEMY_STATES: chex.Array = jnp.array(
        [
            # MAX 20 ENEMYS ON FIELD, see ENEMY_MAX
            # x, y, type, arg1, arg2
            # Landers
            # x, y, type, state, human_num
            [360, 30, LANDER, LANDER_STATE_PATROL, 0],
            [20, 100, LANDER, LANDER_STATE_PATROL, 0],
            [80, 80, LANDER, LANDER_STATE_PATROL, 0],
            [30, 60, LANDER, LANDER_STATE_PATROL, 0],
            # Pods
            # x, y, type, .., ..
            [340, 20, POD, 0, 0],
            [350, 30, POD, 0, 0],
            # Bomber
            # x, y, type, .., ..
            [300, 50, BOMBER, 0, 0],
            # Inactives
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
            [0, 0, INACTIVE, 0, 0],
        ]
    )


# immutable state container
class DefenderState(NamedTuple):
    # Game
    step_counter: chex.Array
    # Camera
    camera_offset: chex.Array
    # Space Ship
    space_ship_speed: chex.Array
    space_ship_x: chex.Array
    space_ship_y: chex.Array
    space_ship_facing_right: chex.Array
    # Bullet
    bullet_state: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_dir_x: chex.Array
    bullet_dir_y: chex.Array
    bullet_color_pos: chex.Array
    # Enemies
    enemy_states: (
        chex.Array
    )  # (20, 5) array with (x, y, type, arg1, arg2) for each enemy


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
            game_dimensions=(self.consts.WINDOW_HEIGHT, self.consts.WINDOW_WIDTH),
            channels=3,
            # downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Update asset config
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/defender"

        # Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Stack the enemy sprites to use with index
        enemy_mask_before_pad = [
            self.SHAPE_MASKS["lander"],
            self.SHAPE_MASKS["pod"],
            self.SHAPE_MASKS["bomber"],
            self.SHAPE_MASKS["swarmers"],
            self.SHAPE_MASKS["mutant"],
            self.SHAPE_MASKS["baiter"],
        ]

        max_h = max(m.shape[0] for m in enemy_mask_before_pad)
        max_w = max(m.shape[1] for m in enemy_mask_before_pad)

        padded_masks = []
        padded_masks.append(jnp.zeros((max_h, max_w)))
        for mask in enemy_mask_before_pad:
            h, w = mask.shape
            padded_mask = jnp.pad(
                mask,
                ((0, max_h - h), (0, max_w - w)),
                mode="constant",
                constant_values=self.jr.TRANSPARENT_ID,
            )
            padded_masks.append(padded_mask)

        self.ENEMY_MASKS = jnp.stack(padded_masks)

    def _get_asset_config(self) -> list:
        # Returns the declarative manifest of all assets for the game, including both wall sprites
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

    # Use together with onscreen_pos, as it returns even ones that are slightly offscreen, for clip
    def is_onscreen(
        self, screen_x: int, screen_y: int, width: int, height: int
    ) -> bool:
        x_onscreen = jnp.logical_and(
            screen_x + width > 0, screen_x < self.consts.WINDOW_WIDTH
        )
        y_onscreen = jnp.logical_and(
            screen_y + height > self.consts.GAME_AREA_TOP,
            screen_y < self.consts.GAME_AREA_BOTTOM,
        )
        return jnp.logical_and(x_onscreen, y_onscreen)

    # Camera offset calculation function, does return coordinates that are not on screen!
    def onscreen_pos(self, state, game_x, game_y):
        camera_window_x = self.consts.CAMERA_WINDOW_X
        camera_game_x = state.space_ship_x + state.camera_offset

        camera_left_border = jnp.mod(
            camera_game_x - self.consts.WINDOW_WIDTH / 2, self.consts.GAME_WIDTH
        )
        camera_right_border = jnp.mod(
            camera_game_x + self.consts.WINDOW_WIDTH / 2, self.consts.GAME_WIDTH
        )

        is_in_left_wrap = jnp.logical_and(
            game_x >= camera_left_border, camera_left_border > camera_game_x
        )
        is_in_right_wrap = jnp.logical_and(
            game_x < camera_right_border, camera_right_border < camera_game_x
        )

        screen_x = (game_x - camera_game_x + camera_window_x).astype(jnp.int32)

        screen_x = jax.lax.cond(
            is_in_left_wrap,
            lambda: jnp.mod(game_x, camera_left_border).astype(jnp.int32),
            lambda: screen_x,
        )

        screen_x = jax.lax.cond(
            is_in_right_wrap,
            lambda: (
                self.consts.GAME_WIDTH - camera_game_x + game_x + camera_window_x
            ).astype(jnp.int32),
            lambda: screen_x,
        )

        screen_y = game_y + self.consts.GAME_AREA_TOP
        return screen_x, screen_y

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DefenderState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render City
        city_start = -(state.space_ship_x + state.camera_offset) % 80
        city_y_pos = self.consts.GAME_AREA_BOTTOM - self.consts.CITY_HEIGHT
        city_mask = self.SHAPE_MASKS["city"]
        raster = self.jr.render_at_clipped(
            raster,
            city_start,
            city_y_pos,
            city_mask,
        )
        raster = self.jr.render_at_clipped(
            raster,
            city_start - 80,
            city_y_pos,
            city_mask,
        )
        raster = self.jr.render_at_clipped(
            raster,
            city_start + 80,
            city_y_pos,
            city_mask,
        )

        # TODO Render Score

        # Render Enemy custom function to use in for loop, renders if it is on screen
        def render_enemy(i: int, r):
            enemy = state.enemy_states[i]
            screen_x, screen_y = self.onscreen_pos(state, enemy[0], enemy[1])

            enemy_type = enemy[2]

            mask = self.ENEMY_MASKS[enemy_type]

            onscreen = self.is_onscreen(
                screen_x, screen_y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT
            )

            is_active_and_onscreen = jnp.logical_and(
                enemy_type != self.consts.INACTIVE, onscreen
            )
            return jax.lax.cond(
                is_active_and_onscreen,
                lambda ras: self.jr.render_at_clipped(ras, screen_x, screen_y, mask),
                lambda ras: ras,
                r,
            )

        raster = jax.lax.fori_loop(0, self.consts.ENEMY_MAX, render_enemy, raster)

        # Render Space Ship
        space_ship_mask = self.SHAPE_MASKS["space_ship"]
        space_ship_facing_right = jnp.where(state.space_ship_facing_right, False, True)
        space_ship_window_x, space_ship_window_y = self.onscreen_pos(
            state, state.space_ship_x, state.space_ship_y
        )

        raster = self.jr.render_at(
            raster,
            space_ship_window_x,
            space_ship_window_y,
            space_ship_mask,
            flip_horizontal=space_ship_facing_right,
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

    def is_colliding(self, e1: EntityPosition, e2: EntityPosition) -> chex.Array:
        e1max_x = e1.x + e1.width
        e2max_x = e2.x + e2.width
        e1max_y = e1.y + e1.height
        e2max_y = e2.y + e2.height

        check_1 = e1.x <= e2max_x
        check_2 = e1max_x >= e2.x

        check_3 = e1.y <= e2max_y
        check_4 = e1max_y >= e2.y

        check_x = jnp.logical_and(check_1, check_2)
        check_y = jnp.logical_and(check_3, check_4)

        return jnp.logical_and(check_x, check_y)

    # Wrap function, returns wrapped position
    def wrap_pos(self, game_x: int, game_y: int):
        return game_x % self.consts.GAME_WIDTH, game_y % self.consts.GAME_HEIGHT

    def _move_and_wrap(
        self, x_pos: float, y_pos: float, x_speed: float, y_speed: float
    ) -> Tuple[float, float]:
        x = x_pos + x_speed
        y = y_pos + y_speed
        x, y = self.wrap_pos(int(x), int(y))
        return x, y

    def _space_ship_step(
        self, state: DefenderState, action: chex.Array
    ) -> DefenderState:
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

        space_ship_facing_right = jax.lax.cond(
            direction_x != 0,
            lambda _: direction_x > 0,
            lambda _: state.space_ship_facing_right,
            operand=None,
        )

        space_ship_speed = jax.lax.cond(
            direction_x != 0,
            lambda _: state.space_ship_speed
            + direction_x * self.consts.SPACE_SHIP_ACCELERATION,
            lambda _: state.space_ship_speed * (1 - self.consts.SPACE_SHIP_BREAK),
            operand=None,
        )

        space_ship_speed = jnp.clip(
            space_ship_speed,
            -self.consts.SPACE_SHIP_MAX_SPEED,
            self.consts.SPACE_SHIP_MAX_SPEED,
        )

        space_ship_x = state.space_ship_x + space_ship_speed
        space_ship_y = state.space_ship_y + direction_y
        space_ship_x, space_ship_y = self.wrap_pos(space_ship_x, space_ship_y)

        return state._replace(
            space_ship_speed=space_ship_speed,
            space_ship_x=space_ship_x,
            space_ship_y=space_ship_y,
            space_ship_facing_right=space_ship_facing_right,
        )

    def _lander_movement(self, index: int, enemy_states):
        lander = enemy_states[index]
        x = lander[0] + self.consts.ENEMY_SPEED
        y = lander[1] + self.consts.LANDER_Y_SPEED
        new_lander = [x, y, lander[2], lander[3], lander[4]]
        enemy_states.at[index].set(new_lander)
        return enemy_states

    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(index: int, enemy_states):
            enemy = enemy_states[index]
            enemy_type = enemy[2]
            enemy_states = jax.lax.switch(
                enemy_type,
                [
                    lambda: enemy_states,
                    lambda: self._lander_movement(index, enemy_states),
                ],
            )

            return enemy_states

        enemy_states = state.enemy_states
        enemy_states = jax.lax.fori_loop(
            0, self.consts.ENEMY_MAX, _enemy_move_switch, enemy_states
        )

        return state._replace(enemy_states=enemy_states)

    def _camera_step(self, state: DefenderState) -> DefenderState:
        # Returns: camera_offset
        offset_gain = self.consts.CAMERA_OFFSET_GAIN
        camera_offset = state.camera_offset
        camera_offset += jnp.where(state.space_ship_facing_right, 1, -1) * offset_gain

        camera_offset = jnp.clip(
            camera_offset,
            -self.consts.CAMERA_OFFSET_MAX,
            self.consts.CAMERA_OFFSET_MAX,
        )

        return state._replace(camera_offset=camera_offset)

    def reset(self, key=None) -> Tuple[DefenderObservation, DefenderState]:
        initial_state = DefenderState(
            # Game
            step_counter=jnp.array(0).astype(jnp.int32),
            # Camera
            camera_offset=jnp.array(self.consts.INITIAL_CAMERA_OFFSET).astype(
                jnp.int32
            ),
            # Space Ship
            space_ship_speed=jnp.array(0).astype(jnp.float32),
            space_ship_x=jnp.array(self.consts.INITIAL_SPACE_SHIP_X).astype(jnp.int32),
            space_ship_y=jnp.array(self.consts.INITIAL_SPACE_SHIP_Y).astype(jnp.int32),
            space_ship_facing_right=jnp.array(
                self.consts.INITIAL_SPACE_SHIP_FACING_RIGHT, dtype=jnp.bool
            ),
            # Bullet
            bullet_state=jnp.array(False, dtype=jnp.bool),
            bullet_x=jnp.array(0).astype(jnp.float32),
            bullet_y=jnp.array(0).astype(jnp.float32),
            bullet_dir_x=jnp.array(0).astype(jnp.int32),
            bullet_dir_y=jnp.array(0).astype(jnp.int32),
            bullet_color_pos=jnp.array(0).astype(jnp.int32),
            # Enemies
            enemy_states=jnp.array(self.consts.INITIAL_ENEMY_STATES).astype(jnp.int32),
        )
        observation = self._get_observation(initial_state)
        return observation, initial_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: DefenderState, action: chex.Array
    ) -> Tuple[DefenderObservation, DefenderState, float, bool, DefenderInfo]:
        # Get all updated values from individual step functions
        previous_state = state

        state = self._space_ship_step(state, action)
        state = self._camera_step(state)
        state = self._enemy_step(state)

        # state = self._collision_step(new_state)
        observation = self._get_observation(state)
        env_reward = self._get_reward(previous_state, state)
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
                width=jnp.array(self.consts.SPACE_SHIP_WIDTH),
                height=jnp.array(self.consts.SPACE_SHIP_HEIGHT),
            ),
            score=0,
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
