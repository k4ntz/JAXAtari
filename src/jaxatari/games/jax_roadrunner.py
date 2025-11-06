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
    # If the players x coordinate would be below this value after applying movement, we move everything one to the right to simulate movement.
    X_SCROLL_THRESHOLD: int = 50
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
    ROAD_DASH_LENGTH: int = 5
    ROAD_GAP_HEIGHT: int = 17
    ROAD_PATTERN_WIDTH: int = ROAD_DASH_LENGTH * 4
    BACKGROUND_COLOR: Tuple[int, int, int] = (255, 204, 102)
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
    score: chex.Array
    is_scrolling: chex.Array
    scrolling_step_counter: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class RoadRunnerObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    score: jnp.ndarray


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

    def _check_bounds(
        self, x_pos: chex.Array, y_pos: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        # This assumes player and enemy have the same size
        checked_y = jnp.clip(
            y_pos,
            self.consts.ROAD_TOP_Y - (self.consts.PLAYER_SIZE[1] / 3),
            self.consts.ROAD_TOP_Y
            + self.consts.ROAD_HEIGHT
            - self.consts.PLAYER_SIZE[1],
        )
        checked_x = jnp.clip(
            x_pos,
            0,
            self.consts.WIDTH - self.consts.PLAYER_SIZE[0],
        )
        return (checked_x, checked_y)

    def _handle_scrolling(self, state: RoadRunnerState, player_x):
        state = state._replace(scrolling_step_counter=state.scrolling_step_counter + 1)
        player_x = player_x + self.consts.PLAYER_MOVE_SPEED
        return state, player_x

    def _player_step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> RoadRunnerState:

        # --- Update Player Position ---
        vel_x, vel_y = self._handle_input(action)
        player_x = state.player_x + vel_x
        player_y = state.player_y + vel_y

        player_x, player_y = self._check_bounds(player_x, player_y)

        is_moving = (vel_x != 0) | (vel_y != 0)

        # Update player orientation based on horizontal movement
        player_looks_right = jax.lax.cond(
            vel_x > 0,
            lambda: True,
            lambda: jax.lax.cond(
                vel_x < 0, lambda: False, lambda: state.player_looks_right
            ),
        )

        is_scrolling = player_x < self.consts.X_SCROLL_THRESHOLD
        state = state._replace(is_scrolling=is_scrolling)

        state, player_x = jax.lax.cond(
            state.is_scrolling,
            lambda: self._handle_scrolling(state, player_x),
            lambda: (state, player_x),
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

        # Handle scrolling
        new_enemy_x = jax.lax.cond(
            state.is_scrolling,
            # move the enemy backwards by the player speed to simulate moving down the road
            lambda: new_enemy_x + self.consts.PLAYER_MOVE_SPEED,
            lambda: new_enemy_x,
        )

        new_enemy_x, new_enemy_y = self._check_bounds(new_enemy_x, new_enemy_y)
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
            score=jnp.array(0, dtype=jnp.int32),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> Tuple[RoadRunnerObservation, RoadRunnerState, float, bool, None]:
        state = self._player_step(state, action)
        state = self._enemy_step(state)
        state = state._replace(
            step_counter=state.step_counter + 1, score=state.score + 1
        )

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
        return RoadRunnerObservation(player=player, enemy=enemy, score=state.score)

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
                        height: spaces.Box(
                            low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32
                        ),
                    }
                ),
                "score": spaces.Box(
                    low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32
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
        ROAD_HEIGHT = self.consts.ROAD_HEIGHT
        WIDTH = self.consts.WIDTH
        DASH_LENGTH = self.consts.ROAD_DASH_LENGTH
        GAP_HEIGHT = self.consts.ROAD_GAP_HEIGHT
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH

        # Create a wider road for scrolling
        SCROLL_WIDTH = WIDTH + PATTERN_WIDTH

        road_color_rgba = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        marking_color_rgba = jnp.array([255, 255, 255, 255], dtype=jnp.uint8)

        # Create a coordinate grid for the wider sprite
        y, x = jnp.indices((ROAD_HEIGHT, SCROLL_WIDTH))

        # Define the pattern using modular arithmetic
        is_marking_col = (x % PATTERN_WIDTH) >= (3 * DASH_LENGTH)
        is_marking_row = (y % (GAP_HEIGHT + 1)) == GAP_HEIGHT
        is_not_last_row = y < (ROAD_HEIGHT - 1)
        is_marking = is_marking_col & is_marking_row & is_not_last_row

        # Use jnp.where to create the sprite from the pattern
        road_sprite = jnp.where(
            is_marking[:, :, jnp.newaxis],
            marking_color_rgba,
            road_color_rgba,
        )

        return road_sprite

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
        asset_config = [
            {"name": "background", "type": "background", "data": background_sprite},
            {"name": "player", "type": "single", "file": "roadrunner_stand.npy"},
            {"name": "player_run1", "type": "single", "file": "roadrunner_run1.npy"},
            {"name": "player_run2", "type": "single", "file": "roadrunner_run2.npy"},
            {"name": "enemy", "type": "single", "file": "enemy.npy"},
            {"name": "road", "type": "procedural", "data": road_sprite},
            {"name": "wall_bottom", "type": "procedural", "data": wall_sprite_bottom},
            {"name": "score_digits", "type": "digits", "pattern": "score_{}.npy"},
        ]

        return asset_config

    def _render_score(self, raster: jnp.ndarray, score: jnp.ndarray) -> jnp.ndarray:
        score_digits = self.jr.int_to_digits(score, max_digits=6)
        score_digit_masks = self.SHAPE_MASKS["score_digits"]

        # Position the score at the top center
        score_x = (
            self.consts.WIDTH // 2 - (score_digits.shape[0] * 12) // 2
        )  # Assuming digit width of 12
        score_y = 16

        raster = self.jr.render_label_selective(
            raster,
            score_x,
            score_y,
            score_digits,
            score_digit_masks,
            0,
            score_digits.shape[0],
            spacing=14,
            max_digits_to_render=6,
        )
        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: RoadRunnerState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # --- Animate Road ---
        PATTERN_WIDTH = self.consts.ROAD_PATTERN_WIDTH

        # Calculate the horizontal offset for scrolling
        offset = PATTERN_WIDTH - (
            (state.scrolling_step_counter * self.consts.PLAYER_MOVE_SPEED)
            % PATTERN_WIDTH
        )

        # Slice the wide road mask to get the current frame's view
        road_mask = jax.lax.dynamic_slice(
            self.SHAPE_MASKS["road"],
            (0, offset),
            (self.consts.ROAD_HEIGHT, self.consts.WIDTH),
        )

        # Render the sliced road portion
        raster = self.jr.render_at(raster, 0, self.consts.ROAD_TOP_Y, road_mask)

        # Render score
        raster = self._render_score(raster, state.score)

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
