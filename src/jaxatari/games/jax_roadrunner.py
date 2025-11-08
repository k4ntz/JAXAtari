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
    ENEMY_MOVE_SPEED: int = 3
    ENEMY_REACTION_DELAY: int = 6
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
    player_x_history: chex.Array
    player_y_history: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    step_counter: chex.Array
    player_is_moving: chex.Array
    player_looks_right: chex.Array
    enemy_is_moving: chex.Array
    enemy_looks_right: chex.Array
    score: chex.Array
    is_scrolling: chex.Array
    scrolling_step_counter: chex.Array
    is_round_over: chex.Array


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

        # Pre-calculate normalized velocities
        sqrt2_inv = 1 / jnp.sqrt(2)
        self._velocities = (
            jnp.array(
                [
                    [0, 0],  # NOOP
                    [0, -1],  # UP
                    [0, 1],  # DOWN
                    [-1, 0],  # LEFT
                    [1, 0],  # RIGHT
                    [sqrt2_inv, -sqrt2_inv],  # UPRIGHT
                    [-sqrt2_inv, -sqrt2_inv],  # UPLEFT
                    [sqrt2_inv, sqrt2_inv],  # DOWNRIGHT
                    [-sqrt2_inv, sqrt2_inv],  # DOWNLEFT
                ]
            )
            * self.consts.PLAYER_MOVE_SPEED
        )

    def _handle_input(self, action: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Handles user input to determine player velocity."""
        # Map action to the corresponding index in the action_set
        action_idx = jnp.argmax(jnp.array(self.action_set) == action)
        vel = self._velocities[action_idx]
        return vel[0], vel[1]

    def _check_bounds(
        self, x_pos: chex.Array, y_pos: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        # This assumes player and enemy have the same size
        checked_y = jnp.clip(
            y_pos,
            self.consts.ROAD_TOP_Y - (self.consts.PLAYER_SIZE[1] // 3),
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

    def _handle_scrolling(self, state: RoadRunnerState, x_pos: chex.Array):
        return jax.lax.cond(
            state.is_scrolling,
            lambda: x_pos + self.consts.PLAYER_MOVE_SPEED,
            lambda: x_pos,
        )

    def _player_step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> RoadRunnerState:

        # --- Update Player Position ---
        input_vel_x, input_vel_y = self._handle_input(action)

        # If round is over, player is forced to move right.
        vel_x = jax.lax.cond(
            state.is_round_over,
            lambda: jnp.array(self.consts.PLAYER_MOVE_SPEED, dtype=jnp.float32),
            lambda: input_vel_x,
        )
        vel_y = jax.lax.cond(state.is_round_over, lambda: 0.0, lambda: input_vel_y)

        # Determine if scrolling should happen based on the potential next position.
        tentative_player_x = state.player_x + vel_x
        is_scrolling = tentative_player_x < self.consts.X_SCROLL_THRESHOLD

        # When scrolling, the player's horizontal velocity should counteract the scroll.
        # We use the original vel_x for non-scrolling movement.
        final_vel_x = jax.lax.cond(
            is_scrolling,
            lambda: -float(self.consts.PLAYER_MOVE_SPEED),
            lambda: vel_x,
        )

        player_x = state.player_x + final_vel_x
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

        # Update the state with the scrolling flag for other parts of the game (e.g., rendering).
        state = state._replace(is_scrolling=is_scrolling)
        state = jax.lax.cond(
            state.is_scrolling,
            lambda: state._replace(scrolling_step_counter=state.scrolling_step_counter + 1),
            lambda: state,
        )

        # Apply the scroll offset to the player's final position.
        player_x = self._handle_scrolling(state, player_x)

        # Update player position history for enemy AI
        new_x_history = jnp.roll(state.player_x_history, shift=1)
        new_x_history = new_x_history.at[0].set(state.player_x)
        new_y_history = jnp.roll(state.player_y_history, shift=1)
        new_y_history = new_y_history.at[0].set(state.player_y)

        return state._replace(
            player_x=player_x.astype(jnp.int32),
            player_y=player_y.astype(jnp.int32),
            player_is_moving=is_moving,
            player_looks_right=player_looks_right,
            player_x_history=new_x_history,
            player_y_history=new_y_history,
        )

    def _enemy_step(self, state: RoadRunnerState) -> RoadRunnerState:
        def game_over_logic(st: RoadRunnerState) -> RoadRunnerState:
            new_enemy_x = st.enemy_x + self.consts.PLAYER_MOVE_SPEED
            new_enemy_x, new_enemy_y = self._check_bounds(new_enemy_x, st.enemy_y)
            return st._replace(
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_is_moving=True,
                enemy_looks_right=True,
            )

        def normal_logic(st: RoadRunnerState) -> RoadRunnerState:
            # Get the distance to the player, with a configurable frame delay.
            delayed_player_x = st.player_x_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delayed_player_y = st.player_y_history[
                self.consts.ENEMY_REACTION_DELAY - 1
            ]
            delta_x = delayed_player_x - st.enemy_x
            delta_y = delayed_player_y - st.enemy_y

            # Determine enemy movement and orientation
            enemy_is_moving = (delta_x != 0) | (delta_y != 0)
            enemy_looks_right = jax.lax.cond(
                delta_x > 0,
                lambda: True,
                lambda: jax.lax.cond(
                    delta_x < 0, lambda: False, lambda: st.enemy_looks_right
                ),
            )

            # Update enemy position, clipping movement to ENEMY_MOVE_SPEED to prevent jittering
            new_enemy_x = st.enemy_x + jnp.clip(
                delta_x, -self.consts.ENEMY_MOVE_SPEED, self.consts.ENEMY_MOVE_SPEED
            )
            new_enemy_y = st.enemy_y + jnp.clip(
                delta_y, -self.consts.ENEMY_MOVE_SPEED, self.consts.ENEMY_MOVE_SPEED
            )

            new_enemy_x = self._handle_scrolling(st, new_enemy_x)

            new_enemy_x, new_enemy_y = self._check_bounds(new_enemy_x, new_enemy_y)
            return st._replace(
                enemy_x=new_enemy_x,
                enemy_y=new_enemy_y,
                enemy_is_moving=enemy_is_moving,
                enemy_looks_right=enemy_looks_right,
            )

        return jax.lax.cond(state.is_round_over, game_over_logic, normal_logic, state)

    def _check_game_over(self, state: RoadRunnerState) -> RoadRunnerState:
        # Here we check if the enemy and the player overlap
        player_x2 = state.player_x + self.consts.PLAYER_SIZE[0]
        player_y2 = state.player_y + self.consts.PLAYER_SIZE[1]
        enemy_x2 = state.enemy_x + self.consts.ENEMY_SIZE[0]
        enemy_y2 = state.enemy_y + self.consts.ENEMY_SIZE[1]

        # Check for overlap on both axes
        overlap_x = (state.player_x < enemy_x2) & (player_x2 > state.enemy_x)
        overlap_y = (state.player_y < enemy_y2) & (player_y2 > state.enemy_y)

        # Collision happens if there is overlap on both axes
        collision = overlap_x & overlap_y

        return jax.lax.cond(
            collision,
            lambda st: st._replace(
                is_round_over=True,
                player_x=(st.enemy_x + self.consts.ENEMY_SIZE[0] + 2).astype(jnp.int32),
                player_y=st.enemy_y.astype(jnp.int32),
            ),
            lambda st: st,
            state,
        )
    def reset(self, key=None) -> Tuple[RoadRunnerObservation, RoadRunnerState]:
        state = RoadRunnerState(
            player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
            player_x_history=jnp.array(
                [self.consts.PLAYER_START_X] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            player_y_history=jnp.array(
                [self.consts.PLAYER_START_Y] * self.consts.ENEMY_REACTION_DELAY,
                dtype=jnp.int32,
            ),
            enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
            enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            player_is_moving=jnp.array(False, dtype=jnp.bool_),
            player_looks_right=jnp.array(False, dtype=jnp.bool_),
            enemy_is_moving=jnp.array(False, dtype=jnp.bool_),
            enemy_looks_right=jnp.array(False, dtype=jnp.bool_),
            score=jnp.array(0, dtype=jnp.int32),
            is_scrolling=jnp.array(False, dtype=jnp.bool_),
            scrolling_step_counter=jnp.array(0, dtype=jnp.int32),
            is_round_over=jnp.array(False, dtype=jnp.bool_),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: RoadRunnerState, action: chex.Array
    ) -> Tuple[RoadRunnerObservation, RoadRunnerState, float, bool, None]:
        state = self._player_step(state, action)
        state = self._enemy_step(state)
        state = self._check_game_over(state)

        def reset_round(st: RoadRunnerState) -> RoadRunnerState:
            return st._replace(
                player_x=jnp.array(self.consts.PLAYER_START_X, dtype=jnp.int32),
                player_y=jnp.array(self.consts.PLAYER_START_Y, dtype=jnp.int32),
                player_x_history=jnp.array(
                    [self.consts.PLAYER_START_X] * self.consts.ENEMY_REACTION_DELAY,
                    dtype=jnp.int32,
                ),
                player_y_history=jnp.array(
                    [self.consts.PLAYER_START_Y] * self.consts.ENEMY_REACTION_DELAY,
                    dtype=jnp.int32,
                ),
                enemy_x=jnp.array(self.consts.ENEMY_X, dtype=jnp.int32),
                enemy_y=jnp.array(self.consts.ENEMY_Y, dtype=jnp.int32),
                is_round_over=jnp.array(False, dtype=jnp.bool_),
            )

        player_at_end = state.player_x >= self.consts.WIDTH - self.consts.PLAYER_SIZE[0]
        state = jax.lax.cond(
            state.is_round_over & player_at_end, reset_round, lambda st: st, state
        )

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
                        "height": spaces.Box(
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
            {"name": "enemy", "type": "single", "file": "enemy_stand.npy"},
            {"name": "enemy_run1", "type": "single", "file": "enemy_run1.npy"},
            {"name": "enemy_run2", "type": "single", "file": "enemy_run2.npy"},
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

    def _get_animated_sprite(
        self,
        is_moving: chex.Array,
        looks_right: chex.Array,
        step_counter: chex.Array,
        animation_speed: int,
        stand_sprite: jnp.ndarray,
        run1_sprite: jnp.ndarray,
        run2_sprite: jnp.ndarray,
    ) -> jnp.ndarray:
        sprites = (stand_sprite, run1_sprite, run2_sprite)
        run_frame_idx = (step_counter // animation_speed % 2) + 1
        sprite_idx = jax.lax.cond(
            is_moving,
            lambda: run_frame_idx,
            lambda: 0,
        )
        mask = jax.lax.switch(
            sprite_idx,
            [
                lambda: sprites[0],
                lambda: sprites[1],
                lambda: sprites[2],
            ],
        )
        return jax.lax.cond(
            looks_right,
            lambda: jnp.fliplr(mask),
            lambda: mask,
        )

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

        # Render Player
        player_mask = self._get_animated_sprite(
            state.player_is_moving,
            state.player_looks_right,
            state.step_counter,
            self.consts.PLAYER_ANIMATION_SPEED,
            self.SHAPE_MASKS["player"],
            self.SHAPE_MASKS["player_run1"],
            self.SHAPE_MASKS["player_run2"],
        )
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        # Render Enemy
        enemy_mask = self._get_animated_sprite(
            state.enemy_is_moving,
            state.enemy_looks_right,
            state.step_counter,
            self.consts.PLAYER_ANIMATION_SPEED,  # Assuming same speed for now
            self.SHAPE_MASKS["enemy"],
            self.SHAPE_MASKS["enemy_run1"],
            self.SHAPE_MASKS["enemy_run2"],
        )
        raster = self.jr.render_at(raster, state.enemy_x, state.enemy_y, enemy_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
