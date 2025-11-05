# Simplified jax_roadrunner.py

import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


# --- Constants ---
class RoadRunnerConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    PLAYER_MOVE_SPEED: int = 4
    PLAYER_ANIMATION_SPEED: int = 2
    ENEMY_MOVE_SPEED: int = 1
    PLAYER_START_X: int = 140
    PLAYER_START_Y: int = 96
    ENEMY_X: int = 16
    ENEMY_Y: int = 96
    PLAYER_SIZE: Tuple[int, int] = (8, 32)
    ENEMY_SIZE: Tuple[int, int] = (4, 4)
    WALL_TOP_Y: int = 24
    WALL_TOP_HEIGHT: int = 10
    WALL_BOTTOM_Y: int = 194
    WALL_BOTTOM_HEIGHT: int = 16
    ROAD_HEIGHT: int = 90
    ROAD_TOP_Y: int = 110
    BACKGROUND_COLOR: Tuple[int, int, int] = (255, 204, 102)
    ROAD_COLOR: Tuple[int, int, int] = (0, 0, 0)
    PLAYER_COLOR: Tuple[int, int, int] = (92, 186, 92)
    ENEMY_COLOR: Tuple[int, int, int] = (213, 130, 74)
    WALL_COLOR: Tuple[int, int, int] = (236, 236, 236)


# --- State and Observation ---
class RoadRunnerState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    step_counter: chex.Array
    player_is_moving: chex.Array
    player_looks_right: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class RoadRunnerObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition


# --- Main Environment Class ---
class JaxRoadRunner(
    JaxEnvironment[RoadRunnerState, RoadRunnerObservation, None, RoadRunnerConstants]
):
    def __init__(self, consts: RoadRunnerConstants = None):
        consts = consts or RoadRunnerConstants()
        super().__init__(consts)
        self.renderer = RoadRunnerRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
        ]
        self.obs_size = 2 * 4  # Simplified

    def _handle_input(self, action: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Handles user input to determine player velocity, ensuring constant speed
        for both cardinal and diagonal movements.
        """
        # Define movement components based on the selected action
        is_up = (
            (action == Action.UP)
            | (action == Action.UPRIGHT)
            | (action == Action.UPLEFT)
        )
        is_down = (
            (action == Action.DOWN)
            | (action == Action.DOWNRIGHT)
            | (action == Action.DOWNLEFT)
        )
        is_left = (
            (action == Action.LEFT)
            | (action == Action.UPLEFT)
            | (action == Action.DOWNLEFT)
        )
        is_right = (
            (action == Action.RIGHT)
            | (action == Action.UPRIGHT)
            | (action == Action.DOWNRIGHT)
        )

        # Create a raw direction vector (dx, dy)
        dx = is_right.astype(jnp.float32) - is_left.astype(jnp.float32)
        dy = is_down.astype(jnp.float32) - is_up.astype(jnp.float32)
        vel_vec = jnp.array([dx, dy])

        # Check if there is any movement
        is_moving = jnp.any(vel_vec != 0)

        # Normalize the vector to have a magnitude of 1 if moving.
        # This is done conditionally to avoid division by zero and is JIT-friendly.
        normalized_vel = jax.lax.cond(
            is_moving, lambda v: v / jnp.linalg.norm(v), lambda v: v, vel_vec
        )

        # Scale the normalized vector by the player's move speed
        scaled_vel = normalized_vel * self.consts.PLAYER_MOVE_SPEED
        x_vel, y_vel = scaled_vel[0], scaled_vel[1]

        return x_vel, y_vel

    def _player_step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> RoadRunnerState:

        # --- Update Player Position ---
        vel_x, vel_y = self._handle_input(action)
        player_x = state.player_x + vel_x
        player_y = state.player_y + vel_y

        # --- Boundary Checks ---
        player_y = jnp.clip(
            player_y,
            self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT,
            self.consts.WALL_BOTTOM_Y - self.consts.PLAYER_SIZE[1],
        )
        player_x = jnp.clip(
            player_x,
            0,
            self.consts.WIDTH - self.consts.PLAYER_SIZE[0],
        )

        is_moving = (vel_x != 0) | (vel_y != 0)

        # Update player orientation based on horizontal movement
        player_looks_right = jax.lax.cond(
            vel_x > 0,
            lambda: True,
            lambda: jax.lax.cond(
                vel_x < 0, lambda: False, lambda: state.player_looks_right
            ),
        )

        return state._replace(
            player_x=player_x,
            player_y=player_y,
            player_is_moving=is_moving,
            player_looks_right=player_looks_right,
        )

    def _enemy_step(self, state: RoadRunnerState) -> RoadRunnerState:
        # Get the direction vector towards the player
        dir_x = jnp.sign(state.player_x - state.enemy_x)
        dir_y = jnp.sign(state.player_y - state.enemy_y)

        # Update enemy position
        new_enemy_x = state.enemy_x + dir_x * self.consts.ENEMY_MOVE_SPEED
        new_enemy_y = state.enemy_y + dir_y * self.consts.ENEMY_MOVE_SPEED

        # Boundary Checks
        new_enemy_x = jnp.clip(
            new_enemy_x,
            0,
            self.consts.WIDTH - self.consts.ENEMY_SIZE[0],
        )
        new_enemy_y = jnp.clip(
            new_enemy_y,
            self.consts.WALL_TOP_Y + self.consts.WALL_TOP_HEIGHT,
            self.consts.WALL_BOTTOM_Y + self.consts.WALL_TOP_HEIGHT,
        )

        return state._replace(enemy_x=new_enemy_x, enemy_y=new_enemy_y)

    def reset(self, key=None) -> Tuple[RoadRunnerObservation, RoadRunnerState]:
        state = RoadRunnerState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> Tuple[RoadRunnerObservation, RoadRunnerState, float, bool, None]:
        state = self._player_step(state, action)
        state = self._enemy_step(state)
        state = state._replace(step_counter=state.step_counter + 1)

        done = False  # Game never ends
        reward = 0.0  # No reward
        info = None  # No info
        observation = self._get_observation(state)

        return observation, state, reward, done, info

    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: RoadRunnerState) -> RoadRunnerObservation:
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )
        enemy = EntityPosition(
            x=state.enemy_x,
            y=state.enemy_y,
            width=jnp.array(self.consts.ENEMY_SIZE[0]),
            height=jnp.array(self.consts.ENEMY_SIZE[1]),
        )
        return RoadRunnerObservation(player=player, enemy=enemy)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        # Simplified observation space
        return spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "enemy": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(
                            low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32
                        ),
                        "height": spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8,
        )


# --- Renderer Class (Simplified) ---
class RoadRunnerRenderer(JAXGameRenderer):
    def __init__(self, consts: RoadRunnerConstants = None):
        super().__init__()
        self.consts = consts or RoadRunnerConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        background_sprite = self._create_background_sprite()
        wall_sprite_top = self._create_wall_sprite(self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)
        road_sprite = self._create_road_sprite()
        asset_config = self._get_asset_config(
            background_sprite, road_sprite, wall_sprite_bottom
        )
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/roadrunner"

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _create_background_sprite(self) -> jnp.ndarray:
        background_color_rgba = (*self.consts.BACKGROUND_COLOR, 255)
        background_shape = (self.consts.HEIGHT, self.consts.WIDTH, 4)
        return jnp.tile(
            jnp.array(background_color_rgba, dtype=jnp.uint8),
            (*background_shape[:2], 1),
        )

    def _create_road_sprite(self) -> jnp.ndarray:
        ROAD_WIDTH = self.consts.WIDTH
        road_color_rgba = (*self.consts.ROAD_COLOR, 255)
        road_shape = (self.consts.ROAD_HEIGHT, ROAD_WIDTH, 4)
        # TODO procedurally add road markings

        return jnp.tile(
            jnp.array(road_color_rgba, dtype=jnp.uint8), (*road_shape[:2], 1)
        )

    def _create_wall_sprite(self, height: int) -> jnp.ndarray:
        wall_color_rgba = (*self.consts.WALL_COLOR, 255)
        wall_shape = (height, self.consts.WIDTH, 4)
        return jnp.tile(
            jnp.array(wall_color_rgba, dtype=jnp.uint8), (*wall_shape[:2], 1)
        )

    def _get_asset_config(
        self,
        background_sprite: jnp.ndarray,
        road_sprite: jnp.ndarray,
        wall_sprite_bottom: jnp.ndarray,
    ) -> list:
        return [
            {"name": "background", "type": "background", "data": background_sprite},
            {"name": "player", "type": "single", "file": "roadrunner_stand.npy"},
            {"name": "player_run1", "type": "single", "file": "roadrunner_run1.npy"},
            {"name": "player_run2", "type": "single", "file": "roadrunner_run2.npy"},
            {"name": "enemy", "type": "single", "file": "enemy.npy"},
            {"name": "road", "type": "procedural", "data": road_sprite},
            {"name": "wall_bottom", "type": "procedural", "data": wall_sprite_bottom},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render Road
        raster = self.jr.render_at(
            raster, 0, self.consts.ROAD_TOP_Y, self.SHAPE_MASKS["road"]
        )

        # Group player sprites for selection
        player_sprites = (
            self.SHAPE_MASKS["player"],
            self.SHAPE_MASKS["player_run1"],
            self.SHAPE_MASKS["player_run2"],
        )

        # Select animation frame based on movement
        # If moving, cycle between run1 and run2.
        run_frame_idx = (
            state.step_counter // self.consts.PLAYER_ANIMATION_SPEED % 2
        ) + 1

        # Use lax.cond to select sprite index: 0 for stand, 1 or 2 for run
        sprite_idx = jax.lax.cond(
            state.player_is_moving,
            lambda: run_frame_idx,
            lambda: 0,
        )

        # Select the sprite mask. jax.lax.switch is efficient for this.
        player_mask = jax.lax.switch(
            sprite_idx,
            [
                lambda: player_sprites[0],
                lambda: player_sprites[1],
                lambda: player_sprites[2],
            ],
        )

        # Flip the sprite if player_looks_right is True
        player_mask = jax.lax.cond(
            state.player_looks_right,
            lambda: jnp.fliplr(player_mask),
            lambda: player_mask,
        )
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        # Render Enemy
        enemy_mask = self.SHAPE_MASKS["enemy"]
        raster = self.jr.render_at(raster, state.enemy_x, state.enemy_y, enemy_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
