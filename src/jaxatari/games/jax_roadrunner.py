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
    BACKGROUND_COLOR: Tuple[int, int, int] = (144, 72, 17)
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
        ]
        self.obs_size = 2 * 4  # Simplified

    def _player_step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> RoadRunnerState:
        # --- Simplified Input Logic ---
        up = action == Action.UP
        down = action == Action.DOWN
        left = action == Action.LEFT
        right = action == Action.RIGHT

        # --- Update Player Position ---
        player_y = state.player_y - up * self.consts.PLAYER_MOVE_SPEED
        player_y = player_y + down * self.consts.PLAYER_MOVE_SPEED

        player_x = state.player_x - left * self.consts.PLAYER_MOVE_SPEED
        player_x = player_x + right * self.consts.PLAYER_MOVE_SPEED

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

        return state._replace(player_x=player_x, player_y=player_y)

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

        wall_sprite_top = self._create_wall_sprite(self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)
        asset_config = self._get_asset_config(wall_sprite_top, wall_sprite_bottom)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/roadrunner"

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _create_wall_sprite(self, height: int) -> jnp.ndarray:
        wall_color_rgba = (*self.consts.WALL_COLOR, 255)
        wall_shape = (height, self.consts.WIDTH, 4)
        return jnp.tile(
            jnp.array(wall_color_rgba, dtype=jnp.uint8), (*wall_shape[:2], 1)
        )

    def _get_asset_config(
        self, wall_sprite_top: jnp.ndarray, wall_sprite_bottom: jnp.ndarray
    ) -> list:
        return [
            {"name": "background", "type": "background", "file": "background.npy"},
            {"name": "player", "type": "single", "file": "roadrunner_stand.npy"},
            {"name": "enemy", "type": "single", "file": "enemy.npy"},
            {"name": "wall_top", "type": "procedural", "data": wall_sprite_top},
            {"name": "wall_bottom", "type": "procedural", "data": wall_sprite_bottom},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render Player
        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        # Render Enemy
        enemy_mask = self.SHAPE_MASKS["enemy"]
        raster = self.jr.render_at(raster, state.enemy_x, state.enemy_y, enemy_mask)

        # Render Walls
        raster = self.jr.render_at(
            raster, 0, self.consts.WALL_TOP_Y, self.SHAPE_MASKS["wall_top"]
        )
        raster = self.jr.render_at(
            raster, 0, self.consts.WALL_BOTTOM_Y, self.SHAPE_MASKS["wall_bottom"]
        )

        return self.jr.render_from_palette(raster, self.PALETTE)
