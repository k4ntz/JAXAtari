from jax import random as jrandom
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
)
from jaxatari.spaces import Space
from typing import Tuple


class DefenderConstants(NamedTuple):
    ## Game settings
    # Screen settings
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210

    # Game world
    WORLD_WIDTH: int = 640
    WORLD_HEIGHT: int = 135

    # Playing area
    GAME_AREA_TOP: int = 38
    GAME_AREA_BOTTOM: int = GAME_AREA_TOP + WORLD_HEIGHT
    GAME_STATE_GAMEOVER: int = 0
    GAME_STATE_PLAYING: int = 1
    GAME_STATE_TRANSITION: int = 2

    # Camera
    CAMERA_SCREEN_X: int = 80
    CAMERA_OFFSET_MAX: int = 40
    CAMERA_OFFSET_GAIN: int = 2
    CAMERA_INIT_OFFSET: int = 40

    ## UI
    # Scanner
    SCANNER_WIDTH: int = 64
    SCANNER_HEIGHT: int = 21
    SCANNER_SCREEN_Y: int = 13
    SCANNER_SCREEN_X: int = 48

    # Drawables
    CITY_WIDTH: int = 60
    CITY_HEIGHT: int = 13
    SCORE_SCREEN_Y: int = 177
    SCORE_SCREEN_X: int = 57
    SCORE_MAX_DIGITS: int = 6
    SCORE_MAX: int = 999999
    DIGIT_WIDTH: int = 7
    DIGIT_HEIGHT: int = 7
    DIGIT_PADDING: int = 1
    SMART_BOMB_SCREEN_X: int = 110
    SMART_BOMB_SCREEN_Y: int = 190
    SMART_BOMB_PADDING: int = 17
    LIVES_SCREEN_X: int = 5
    LIVES_SCREEN_Y: int = 190
    LIVES_SCREEN_PADDING: int = 17

    # Colors
    COLOR_SPACE_SHIP_BLUE: Tuple[int, int, int] = (132, 144, 252)
    COLOR_BOMBER_BLUE: Tuple[int, int, int] = (104, 116, 208)
    COLOR_LANDER_YELLOW: Tuple[int, int, int] = (252, 224, 140)
    COLOR_MUTANT_RED: Tuple[int, int, int] = (192, 104, 72)
    COLOR_PINK: Tuple[int, int, int] = (235, 176, 224)
    PARTICLE_FLICKER_EVERY_N_FRAMES: int = 1

    # Level change animation
    LEVEL_DIGIT_SCREEN_X: int = 80
    LEVEL_DIGIT_SCREEN_Y: int = 60
    LEVEL_TRANSITION_DURATION: int = 60

    ## ENTITY

    # Space Ship
    SPACE_SHIP_WIDTH: int = 13
    SPACE_SHIP_HEIGHT: int = 5
    SPACE_SHIP_INIT_GAME_X: float = 200
    SPACE_SHIP_INIT_GAME_Y: float = 80
    SPACE_SHIP_INIT_FACE_RIGHT: bool = True
    SPACE_SHIP_ACCELERATION: float = 0.07
    SPACE_SHIP_BREAK: float = 0.1
    SPACE_SHIP_MAX_SPEED: float = 2.5
    SPACE_SHIP_EXHAUST_WIDTH: int = 7
    SPACE_SHIP_EXHAUST_FLICKER_N_FRAMES: int = 5
    SPACE_SHIP_INIT_LIVES: int = 3
    SPACE_SHIP_INIT_BOMBS: int = 3
    SPACE_SHIP_SHOOT_CD: int = 10
    SCORE_BONUS_THRESHOLD: int = 10000
    SPACE_SHIP_DEATH_LOOP_FRAMES: int = 3
    SPACE_SHIP_DEATH_LOOP_AMOUNT: int = 12
    SPACE_SHIP_DEATH_EXPLOSION_AMOUNT: int = 8
    SPACE_SHIP_DEATH_ANIM_FRAME_AMOUNT: int = (
        SPACE_SHIP_DEATH_LOOP_AMOUNT * SPACE_SHIP_DEATH_LOOP_FRAMES
        + SPACE_SHIP_DEATH_EXPLOSION_AMOUNT
        + 40  # After explosion empty frames
    )

    SPACE_SHIP_SCANNER_WIDTH: int = 3
    SPACE_SHIP_SCANNER_HEIGHT: int = 2

    # Enemy
    ENEMY_WIDTH: int = 13
    ENEMY_HEIGHT: int = 7
    ENEMY_SPEED: float = 0.12
    SHIP_SPEED_INFLUENCE_ON_SPEED: float = 0.4

    ENEMY_MAX_IN_GAME: int = 30
    ENEMY_MAX_ON_SCREEN: int = 10
    ENEMY_SECTORS: int = 4

    ENEMY_SCANNER_WIDTH: int = 2
    ENEMY_SCANNER_HEIGHT: int = 2

    # Enemy types
    INACTIVE: int = 0
    LANDER: int = 1
    POD: int = 2
    BOMBER: int = 3
    SWARMERS: int = 4
    MUTANT: int = 5
    BAITER: int = 6
    DEAD: int = 7  # To keep position to draw animation

    # Lander
    LANDER_Y_SPEED: float = 0.08
    LANDER_PICKUP_X_THRESHOLD: float = 2.0
    LANDER_START_Y: float = 10.0
    LANDER_PICKUP_DURATION_FRAMES: int = 120  # 4 seconds at 30 fps

    LANDER_STATE_PATROL: int = 0
    LANDER_STATE_DESCEND: int = 1
    LANDER_STATE_PICKUP: int = 2
    LANDER_STATE_ASCEND: int = 3

    LANDER_DEATH_SCORE: int = 150
    LANDER_MAX_AMOUNT: int = 5
    LANDER_LEVEL_AMOUNT: chex.Array = jnp.array([16, 18, 19, 20, 20])

    # Pod
    POD_Y_SPEED: float = 0.08
    POD_DEATH_SCORE: int = 1000
    POD_MAX_AMOUNT: int = 3
    POD_LEVEL_AMOUNT: chex.Array = jnp.array([2, 2, 3, 3, 3])
    ENEMY_SPAWN_AROUND_MIN_RADIUS: float = 5.0
    ENEMY_SPAWN_AROUND_MAX_RADIUS: float = 25.0

    # Bomber
    BOMBER_Y_SPEED: float = -0.2
    BOMB_TTL_FRAMES: int = 30
    BOMBER_DEATH_SCORE: int = 250
    BOMBER_MAX_AMOUNT: int = 1
    BOMBER_LEVEL_AMOUNT: chex.Array = jnp.array([1, 2, 2, 2, 2])

    # Swarmers
    SWARM_SPAWN_MIN: int = 1
    SWARM_SPAWN_MAX: int = 2
    SWARMERS_DEATH_SCORE: int = 500
    SWARMERS_MAX_SPEED: float = 2.0
    SWARMERS_Y_SPEED: float = 0.8

    # Mutant
    MUTANT_DEATH_SCORE: int = 150

    # Baiter
    BAITER_MAX_AMOUNT: int = 2
    BAITER_TIME_SEC: int = 20
    BAITER_DEATH_SCORE: int = 200

    # Dead states, if enemy_type = dead, then arg1 decides color
    DEAD_YELLOW: int = 0
    DEAD_BLUE: int = 1
    DEAD_RED: int = 2

    # Humans
    HUMAN_WIDTH: int = 2
    HUMAN_HEIGHT: int = 4
    HUMAN_FALLING_SPEED: float = 0.5
    HUMAN_DEADLY_FALL_HEIGHT: float = 80.0
    HUMAN_INIT_GAME_Y: int = WORLD_HEIGHT - HUMAN_HEIGHT
    HUMAN_BRING_BACK_FRAMES: int = 300  # 10 sec

    # INACTIVE = 0
    HUMAN_STATE_IDLE: int = 1
    HUMAN_STATE_ABDUCTED: int = 2
    HUMAN_STATE_FALLING: int = 3
    HUMAN_STATE_FALLING_DEADLY: int = 4
    HUMAN_STATE_CAUGHT: int = 5

    HUMAN_SCANNER_HEIGHT: int = 1
    HUMAN_SCANNER_WIDTH: int = 1

    HUMAN_LIVING_FALL_SCORE: int = 250
    HUMAN_CAUGHT_BUT_FORGOTTEN_SCORE: int = 500
    HUMAN_CAUGHT_AND_RETURNED_SCORE: int = 1000

    HUMAN_MAX_AMOUNT: int = 6
    HUMAN_LEVEL_AMOUNT: chex.Array = jnp.array([5, 5, 6, 6, 6])

    ## BULLET AND LASER

    # Enemy bullet
    BULLET_WIDTH: int = 2
    BULLET_HEIGHT: int = 2
    BULLET_SPEED: float = 3.0
    BULLET_MOVE_WITH_SPACE_SHIP: float = 0.9
    BULLET_MIN_SPREAD: float = 0.3
    BULLET_MAX_SPREAD: float = 0.6

    # Space ship laser
    # Width and height are for each laser
    # as the final laser is made out of 2
    LASER_WIDTH: int = 15
    LASER_HEIGHT: int = 1
    LASER_2ND_OFFSET: int = 7
    LASER_FINAL_WIDTH: int = LASER_WIDTH + LASER_2ND_OFFSET
    LASER_FINAL_HEIGHT: int = LASER_HEIGHT * 2
    LASER_SPEED: int = 20


# immutable state container
class DefenderState(NamedTuple):
    # Game
    step_counter: chex.Array
    level: chex.Array
    score: chex.Array
    game_state: chex.Array
    # Camera
    camera_offset: chex.Array
    # Space Ship
    space_ship_speed: chex.Array
    space_ship_x: chex.Array
    space_ship_y: chex.Array
    space_ship_facing_right: chex.Array
    space_ship_lives: chex.Array
    # Bullet
    bullet_active: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_dir_x: chex.Array
    bullet_dir_y: chex.Array
    # Laser
    laser_active: chex.Array
    laser_x: chex.Array
    laser_y: chex.Array
    laser_dir_x: chex.Array
    # Smart bomb
    smart_bomb_amount: chex.Array
    # Enemies
    enemy_states: chex.Array
    enemy_killed: chex.Array
    # Human
    human_states: chex.Array
    # Cooldowns
    shooting_cooldown: chex.Array
    bring_back_human: chex.Array
    # Randomness
    key: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class DefenderObservation(NamedTuple):
    # Needs more implementation, work in progress
    player: EntityPosition
    score: jnp.ndarray
    lives: jnp.ndarray
    enemy_states: jnp.ndarray
    human_states: jnp.ndarray
    bullet_active: jnp.ndarray
    bullet_x: jnp.ndarray
    bullet_y: jnp.ndarray
    laser_active: jnp.ndarray
    laser_x: jnp.ndarray
    laser_y: jnp.ndarray


class DefenderInfo(NamedTuple):
    step_counter: jnp.ndarray
    score: jnp.ndarray


# Helper class that gets implemented by renderer and game for shared functionality
class DefenderHelper:
    def __init__(self, consts: DefenderConstants):
        self.consts = consts
        return

    def _onscreen_pos(self, state: DefenderState, game_x, game_y):
        camera_screen_x = self.consts.CAMERA_SCREEN_X
        camera_game_x = state.space_ship_x + state.camera_offset
        camera_left_border = jnp.mod(
            camera_game_x - self.consts.SCREEN_WIDTH / 2, self.consts.WORLD_WIDTH
        )

        camera_right_border = jnp.mod(
            camera_game_x + self.consts.SCREEN_WIDTH / 2, self.consts.WORLD_WIDTH
        )

        is_in_left_wrap = jnp.logical_and(
            game_x >= camera_left_border, camera_left_border > camera_game_x
        )
        is_in_right_wrap = jnp.logical_and(
            game_x < camera_right_border, camera_right_border < camera_game_x
        )

        screen_x = (game_x - camera_game_x + camera_screen_x).astype(jnp.int32)

        screen_x = jax.lax.cond(
            is_in_left_wrap,
            lambda: jnp.mod(game_x, camera_left_border).astype(jnp.int32),
            lambda: screen_x,
        )

        screen_x = jax.lax.cond(
            is_in_right_wrap,
            lambda: (
                self.consts.WORLD_WIDTH - camera_game_x + game_x + camera_screen_x
            ).astype(jnp.int32),
            lambda: screen_x,
        )

        screen_y = game_y + self.consts.GAME_AREA_TOP
        return screen_x, screen_y

    # Calculate with in game positions
    def _is_onscreen_from_game(
        self, state: DefenderState, game_x, game_y, width: int, height: int
    ):
        screen_x, screen_y = self._onscreen_pos(state, game_x, game_y)
        x_onscreen = jnp.logical_and(
            screen_x + width > 0, screen_x < self.consts.SCREEN_WIDTH
        )
        y_onscreen = jnp.logical_and(
            screen_y + height > self.consts.GAME_AREA_TOP,
            screen_y < self.consts.GAME_AREA_BOTTOM,
        )
        return jnp.logical_and(x_onscreen, y_onscreen)

    # Calculate with on screen positions
    def _is_onscreen_from_screen(self, screen_x, screen_y, width: int, height: int):
        x_onscreen = jnp.logical_and(
            screen_x + width > 0, screen_x < self.consts.SCREEN_WIDTH
        )
        y_onscreen = jnp.logical_and(
            screen_y + height > self.consts.GAME_AREA_TOP,
            screen_y < self.consts.GAME_AREA_BOTTOM,
        )
        return jnp.logical_and(x_onscreen, y_onscreen)

    def _current_sector(self, state: DefenderState):
        sector_width = self.consts.WORLD_WIDTH / self.consts.ENEMY_SECTORS
        camera_game_x = state.camera_offset + state.space_ship_x
        return jnp.floor_divide(camera_game_x, sector_width).astype(jnp.int32)

    def _sector_bounds(self, sector):
        sector_width = self.consts.WORLD_WIDTH / self.consts.ENEMY_SECTORS
        return (sector_width * sector).astype(jnp.int32), (
            sector_width * (sector + 1)
        ).astype(jnp.int32)

    def _which_sector(self, game_x):
        sector_width = self.consts.WORLD_WIDTH / self.consts.ENEMY_SECTORS
        sector = jnp.floor_divide(game_x, sector_width)
        return sector.astype(jnp.int32)

    def _print_array(self, data):
        jax.debug.print("Array: \n{}", data)


class DefenderRenderer(JAXGameRenderer):
    def __init__(self, consts: DefenderConstants = None):
        super().__init__()
        self.consts = consts or DefenderConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(self.consts.SCREEN_HEIGHT, self.consts.SCREEN_WIDTH),
            channels=3,
            # downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # To access helper functions
        self.dh = DefenderHelper(self.consts)

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

        def _create_padded_masks(self, enemy_mask_before_pad):
            """Create padded enemy masks with uniform dimensions."""
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

            return jnp.array(padded_masks)

        self.ENEMY_MASKS = _create_padded_masks(self, enemy_mask_before_pad)
        self.ENEMY_MASK_SIZE = self.ENEMY_MASKS[0].shape

        d_h, d_w = self.SHAPE_MASKS["death_yellow"].shape
        self.DEATH_MASKS = jnp.array(
            [
                jnp.pad(
                    self.SHAPE_MASKS["death_yellow"],
                    (
                        (0, self.ENEMY_MASK_SIZE[0] - d_h),
                        (0, self.ENEMY_MASK_SIZE[1] - d_w),
                    ),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
                jnp.pad(
                    self.SHAPE_MASKS["death_blue"],
                    (
                        (0, self.ENEMY_MASK_SIZE[0] - d_h),
                        (0, self.ENEMY_MASK_SIZE[1] - d_w),
                    ),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
                jnp.pad(
                    self.SHAPE_MASKS["death_red"],
                    (
                        (0, self.ENEMY_MASK_SIZE[0] - d_h),
                        (0, self.ENEMY_MASK_SIZE[1] - d_w),
                    ),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
            ],
            float,
        )

        # Enemy colors to ids
        color_s_blue = self.COLOR_TO_ID.get(self.consts.COLOR_SPACE_SHIP_BLUE, 0)
        color_b_blue = self.COLOR_TO_ID.get(self.consts.COLOR_BOMBER_BLUE, 0)
        color_m_red = self.COLOR_TO_ID.get(self.consts.COLOR_MUTANT_RED, 0)
        color_l_yellow = self.COLOR_TO_ID.get(self.consts.COLOR_LANDER_YELLOW, 0)
        color_pink = self.COLOR_TO_ID.get(self.consts.COLOR_PINK, 0)

        self.ENEMY_COLOR_IDS = jnp.array(
            [
                0,
                color_l_yellow,
                color_l_yellow,
                color_b_blue,
                color_l_yellow,
                color_m_red,
                color_b_blue,
                0,
            ]
        )

        self.PARTICLE_COLOR_IDS = jnp.array(
            [color_s_blue, color_pink, color_b_blue, color_m_red, color_l_yellow]
        )

    def _get_asset_config(self) -> list:
        # Returns the declarative manifest of all assets for the game, including both wall sprites
        return [
            {"name": "background", "type": "background", "file": "background.npy"},
            {"name": "space_ship", "type": "single", "file": "space_ship.npy"},
            {
                "name": "space_ship_death",
                "type": "group",
                "files": [
                    "space_ship_death_0.npy",
                    "space_ship_death_1.npy",
                    "space_ship_death_2.npy",
                    "space_ship_death_3.npy",
                    "space_ship_death_4.npy",
                    "space_ship_death_5.npy",
                    "space_ship_death_6.npy",
                    "space_ship_death_7.npy",
                    "space_ship_death_8.npy",
                    "space_ship_death_9.npy",
                    "space_ship_death_10.npy",
                    "space_ship_death_11.npy",
                ],
            },
            {"name": "exhaust", "type": "single", "file": "exhaust.npy"},
            {"name": "baiter", "type": "single", "file": "baiter.npy"},
            {"name": "bomber", "type": "single", "file": "bomber.npy"},
            {"name": "lander", "type": "single", "file": "lander.npy"},
            {"name": "lander_abduct", "type": "single", "file": "lander_abduct.npy"},
            {"name": "death_yellow", "type": "single", "file": "death_yellow.npy"},
            {"name": "death_blue", "type": "single", "file": "death_blue.npy"},
            {"name": "death_red", "type": "single", "file": "death_red.npy"},
            {"name": "mutant", "type": "single", "file": "mutant.npy"},
            {"name": "pod", "type": "single", "file": "pod.npy"},
            {"name": "swarmers", "type": "single", "file": "swarmers.npy"},
            {"name": "ui_overlay", "type": "single", "file": "ui_overlay.npy"},
            {"name": "city", "type": "single", "file": "city.npy"},
            {"name": "smart_bomb", "type": "single", "file": "smart_bomb.npy"},
            {"name": "score_digits", "type": "digits", "pattern": "score_{}.npy"},
        ]

    def _on_scanner_pos(self, state: DefenderState, game_x, game_y):
        camera_game_x = state.space_ship_x + state.camera_offset
        left_border = jnp.mod(
            camera_game_x - self.consts.WORLD_WIDTH / 2, self.consts.WORLD_WIDTH
        )

        # Calculate position inside scanner
        is_after_zero_wrap = game_x < left_border
        game_to_scanner_ratio_x = self.consts.SCANNER_WIDTH / self.consts.WORLD_WIDTH
        game_to_scanner_ratio_y = self.consts.SCANNER_HEIGHT / self.consts.WORLD_HEIGHT
        screen_x = jax.lax.cond(
            is_after_zero_wrap,
            lambda: (
                (self.consts.WORLD_WIDTH - left_border + game_x)
                * game_to_scanner_ratio_x
            ),
            lambda: (game_x - left_border) * game_to_scanner_ratio_x,
        )
        screen_y = game_y * game_to_scanner_ratio_y

        # Move to scanner position in screen
        screen_x += self.consts.SCANNER_SCREEN_X
        screen_y += self.consts.SCANNER_SCREEN_Y

        return screen_x, screen_y

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: DefenderState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Util function for all entities
        def render_on_scanner(game_x, game_y, width, height, color_id, r):
            scanner_x, scanner_y = self._on_scanner_pos(state, game_x, game_y)

            # To render all entities inside the scanner
            right_barrier = float(
                self.consts.SCANNER_SCREEN_X + self.consts.SCANNER_WIDTH
            )
            scanner_x = jax.lax.cond(
                scanner_x + width > right_barrier,
                lambda: right_barrier - width,
                lambda: scanner_x,
            )

            r = self.jr.draw_rects(
                r,
                jnp.asarray([[scanner_x, scanner_y]]),
                jnp.asarray([[width, height]]),
                color_id,
            )

            return r

        def render_enemy(index: int, r):
            enemy = state.enemy_states[index]
            screen_x, screen_y = self.dh._onscreen_pos(state, enemy[0], enemy[1])
            enemy_type = enemy[2].astype(int)
            enemy_arg1 = enemy[3]

            color_id = self.ENEMY_COLOR_IDS[jnp.array(enemy_type, int)]

            onscreen = self.dh._is_onscreen_from_screen(
                screen_x, screen_y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT
            )

            is_active_and_onscreen = jnp.logical_and(
                enemy_type != self.consts.INACTIVE, onscreen
            )

            scanner_width = self.consts.ENEMY_SCANNER_WIDTH
            scanner_height = self.consts.ENEMY_SCANNER_HEIGHT

            # Render on scanner
            r = jax.lax.cond(
                enemy_type != self.consts.INACTIVE,
                lambda: render_on_scanner(
                    enemy[0], enemy[1], scanner_width, scanner_height, color_id, r
                ),
                lambda: r,
            )

            # Render on screen
            def render_normal(r):
                mask = jax.lax.cond(
                    enemy_type == self.consts.DEAD,
                    lambda: self.DEATH_MASKS[enemy_arg1.astype(int)],
                    lambda: self.ENEMY_MASKS[enemy_type],
                )
                r = self.jr.render_at_clipped(r, screen_x, screen_y, mask)
                return r

            def render_lander(r):
                pickup_mask = self.SHAPE_MASKS["lander_abduct"]
                normal_mask = self.ENEMY_MASKS[enemy_type]

                r = jax.lax.cond(
                    enemy_arg1 == self.consts.LANDER_STATE_PICKUP,
                    lambda: self.jr.render_at_clipped(
                        r, screen_x, screen_y, pickup_mask
                    ),
                    lambda: self.jr.render_at_clipped(
                        r, screen_x, screen_y, normal_mask
                    ),
                )
                return r

            def render_choice(r):
                r = jax.lax.cond(
                    enemy_type == self.consts.LANDER,
                    lambda: render_lander(r),
                    lambda: render_normal(r),
                )
                return r

            # Render on screen and return new raster
            return jax.lax.cond(
                is_active_and_onscreen,
                lambda: render_choice(r),
                lambda: r,
            )

        raster = jax.lax.fori_loop(
            0, self.consts.ENEMY_MAX_IN_GAME, render_enemy, raster
        )

        # Used for bullet color and human color
        current_particle_color_id = self.PARTICLE_COLOR_IDS[
            jnp.mod(
                jnp.floor_divide(
                    state.step_counter, self.consts.PARTICLE_FLICKER_EVERY_N_FRAMES
                ),
                len(self.PARTICLE_COLOR_IDS),
            )
        ]

        def render_space_ship(r):
            game_x = state.space_ship_x
            game_y = state.space_ship_y
            flip_horizontal = jnp.logical_not(state.space_ship_facing_right)

            screen_x, screen_y = self.dh._onscreen_pos(state, game_x, game_y)

            scanner_width = self.consts.SPACE_SHIP_SCANNER_WIDTH
            scanner_height = self.consts.SPACE_SHIP_SCANNER_HEIGHT

            color_id = self.ENEMY_COLOR_IDS[0]

            # Render on scanner
            r = jax.lax.cond(
                state.game_state != self.consts.GAME_STATE_TRANSITION,
                lambda: render_on_scanner(
                    game_x, game_y, scanner_width, scanner_height, color_id, r
                ),
                lambda: r,
            )

            def render_normal(r):
                # Test for hyperspace
                in_screen = screen_y > (
                    self.consts.GAME_AREA_TOP - self.consts.SPACE_SHIP_HEIGHT
                )

                # Render exhaust
                draw_exhaust = (
                    jnp.mod(
                        state.step_counter,
                        self.consts.SPACE_SHIP_EXHAUST_FLICKER_N_FRAMES,
                    )
                    == 0
                )
                draw_exhaust = jnp.logical_and(draw_exhaust, in_screen)
                exhaust_game_x = screen_x + jnp.where(
                    flip_horizontal,
                    self.consts.SPACE_SHIP_WIDTH,
                    -self.consts.SPACE_SHIP_EXHAUST_WIDTH,
                )
                exhaust_game_y = screen_y
                r = jax.lax.cond(
                    draw_exhaust,
                    lambda: self.jr.render_at(
                        r,
                        exhaust_game_x,
                        exhaust_game_y,
                        self.SHAPE_MASKS["exhaust"],
                        flip_horizontal=flip_horizontal,
                    ),
                    lambda: r,
                )

                # Render on screen
                mask = self.SHAPE_MASKS["space_ship"]
                r = jax.lax.cond(
                    in_screen,
                    lambda: self.jr.render_at(
                        r,
                        screen_x,
                        screen_y,
                        mask,
                        flip_horizontal=flip_horizontal,
                    ),
                    lambda: r,
                )
                return r

            def render_death(r):
                # Go through animation, loop 3 times in first 3 frames then let the rest go on
                current_frame = state.shooting_cooldown
                beginning_loop = self.consts.SPACE_SHIP_DEATH_LOOP_AMOUNT
                beginning_frames = self.consts.SPACE_SHIP_DEATH_LOOP_FRAMES
                in_beginning = current_frame < (beginning_loop * beginning_frames)
                current_frame = jax.lax.cond(
                    in_beginning,
                    lambda: jnp.mod(current_frame, beginning_frames),
                    lambda: current_frame - (beginning_loop - 1) * beginning_frames,
                )
                mask = self.SHAPE_MASKS["space_ship_death"][current_frame]

                new_x = screen_x - 5
                new_y = screen_y - 5
                r = self.jr.render_at(
                    r, new_x, new_y, mask, flip_horizontal=flip_horizontal
                )
                return r

            r = jax.lax.cond(
                state.game_state == self.consts.GAME_STATE_PLAYING,
                lambda: render_normal(r),
                lambda: r,
            )
            r = jax.lax.cond(
                state.game_state == self.consts.GAME_STATE_GAMEOVER,
                lambda: render_death(r),
                lambda: r,
            )

            return r

        raster = render_space_ship(raster)

        def render_laser(r):
            game_x = state.laser_x
            game_y = state.laser_y
            screen_x, screen_y = self.dh._onscreen_pos(state, game_x, game_y)
            laser2_x = screen_x + (
                jnp.where(state.space_ship_facing_right, 1.0, -1.0)
                * self.consts.LASER_2ND_OFFSET
            )
            laser2_y = screen_y - self.consts.LASER_HEIGHT

            color_id = current_particle_color_id

            width = self.consts.LASER_WIDTH
            height = self.consts.LASER_HEIGHT

            return self.jr.draw_rects(
                r,
                jnp.asarray(
                    [
                        [screen_x, screen_y],
                        [laser2_x, laser2_y],
                    ]
                ),
                jnp.asarray([[width, height], [width, height]]),
                color_id,
            )

        raster = jax.lax.cond(
            state.laser_active,
            lambda: render_laser(raster),
            lambda: raster,
        )

        def render_city(r):
            # Get the next city starting position to the left
            city_game_x = jnp.multiply(
                jnp.floor_divide(
                    state.space_ship_x + state.camera_offset, self.consts.CITY_WIDTH
                ),
                self.consts.CITY_WIDTH,
            )

            city_game_y = self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT
            city_screen_x, city_screen_y = self.dh._onscreen_pos(
                state, city_game_x, city_game_y
            )

            city_mask = self.SHAPE_MASKS["city"]
            r = self.jr.render_at_clipped(r, city_screen_x, city_screen_y, city_mask)
            r = self.jr.render_at_clipped(
                r, city_screen_x + self.consts.CITY_WIDTH, city_screen_y, city_mask
            )
            r = self.jr.render_at_clipped(
                r, city_screen_x + 2 * self.consts.CITY_WIDTH, city_screen_y, city_mask
            )
            r = self.jr.render_at_clipped(
                r, city_screen_x - self.consts.CITY_WIDTH, city_screen_y, city_mask
            )
            r = self.jr.render_at_clipped(
                r, city_screen_x - 2 * self.consts.CITY_WIDTH, city_screen_y, city_mask
            )
            return r

        raster = render_city(raster)

        def render_ui(index: int, r):
            # Smart bombs
            padding = self.consts.SMART_BOMB_PADDING * index
            r = jax.lax.cond(
                index < state.smart_bomb_amount,
                lambda: self.jr.render_at(
                    r,
                    self.consts.SMART_BOMB_SCREEN_X + padding,
                    self.consts.SMART_BOMB_SCREEN_Y,
                    self.SHAPE_MASKS["smart_bomb"],
                ),
                lambda: r,
            )

            # Lives
            padding = self.consts.LIVES_SCREEN_PADDING * index
            r = jax.lax.cond(
                index < state.space_ship_lives,
                lambda: self.jr.render_at(
                    r,
                    self.consts.LIVES_SCREEN_X + padding,
                    self.consts.LIVES_SCREEN_Y,
                    self.SHAPE_MASKS["space_ship"],
                ),
                lambda: r,
            )
            return r

        raster = self.jr.render_at(raster, 0, 0, self.SHAPE_MASKS["ui_overlay"])

        raster = jax.lax.fori_loop(0, 3, render_ui, raster)

        def render_human(index: int, r):
            human = state.human_states[index]
            screen_x, screen_y = self.dh._onscreen_pos(state, human[0], human[1])
            human_state = human[2]
            onscreen = self.dh._is_onscreen_from_screen(screen_x, screen_y, 5, 5)
            is_active_and_onscreen = jnp.logical_and(
                human_state != self.consts.INACTIVE, onscreen
            )

            color_id = current_particle_color_id
            scanner_width = self.consts.HUMAN_SCANNER_WIDTH
            scanner_height = self.consts.HUMAN_SCANNER_HEIGHT

            # Render on scanner
            r = jax.lax.cond(
                human_state != self.consts.INACTIVE,
                lambda: render_on_scanner(
                    human[0], human[1], scanner_width, scanner_height, color_id, r
                ),
                lambda: r,
            )

            # Render on screen and return the changed raster
            return jax.lax.cond(
                is_active_and_onscreen,
                lambda: self.jr.draw_rects(
                    r,
                    jnp.asarray([[screen_x, screen_y]]),
                    jnp.asarray([[self.consts.HUMAN_WIDTH, self.consts.HUMAN_HEIGHT]]),
                    color_id,
                ),
                lambda: r,
            )

        # For each loop renders all humans on screen and scanner
        raster = jax.lax.fori_loop(
            0, self.consts.HUMAN_MAX_AMOUNT, render_human, raster
        )

        def render_bullet(r):
            game_x = state.bullet_x
            game_y = state.bullet_y
            screen_x, screen_y = self.dh._onscreen_pos(state, game_x, game_y)

            color_id = current_particle_color_id

            width = self.consts.BULLET_WIDTH
            height = self.consts.BULLET_HEIGHT

            return self.jr.draw_rects(
                r,
                jnp.asarray([[screen_x, screen_y]]),
                jnp.asarray([[width, height]]),
                color_id,
            )

        raster = jax.lax.cond(
            state.bullet_active,
            lambda: render_bullet(raster),
            lambda: raster,
        )

        def render_score(r):
            max_digits = self.consts.SCORE_MAX_DIGITS

            digit_spacing = self.consts.DIGIT_PADDING + self.consts.DIGIT_WIDTH
            score_digits = self.jr.int_to_digits(state.score, max_digits=max_digits)

            # Implement logic to get more digits
            num_digits = jnp.where(
                state.score == 0,
                2,
                jnp.floor(jnp.log10(state.score.astype(jnp.float32)) + 1),
            )

            start_index = max_digits - num_digits

            screen_x = self.consts.SCORE_SCREEN_X + start_index * digit_spacing
            screen_y = self.consts.SCORE_SCREEN_Y

            return self.jr.render_label_selective(
                r,
                screen_x,
                screen_y,
                score_digits,
                self.SHAPE_MASKS["score_digits"],
                start_index.astype(int),
                num_digits.astype(int),
                digit_spacing,
                max_digits,
            )

        raster = render_score(raster)

        def render_transition(r):
            level = jnp.asarray([state.level])
            screen_x = self.consts.LEVEL_DIGIT_SCREEN_X
            screen_y = self.consts.LEVEL_DIGIT_SCREEN_Y
            r = self.jr.render_label(
                r,
                screen_x,
                screen_y,
                level,
                self.SHAPE_MASKS["score_digits"],
                0,
                1,
            )
            return r

        raster = jax.lax.cond(
            state.game_state == self.consts.GAME_STATE_TRANSITION,
            lambda: render_transition(raster),
            lambda: raster,
        )

        return self.jr.render_from_palette(raster, self.PALETTE)


class JaxDefender(
    JaxEnvironment[DefenderState, DefenderObservation, DefenderInfo, DefenderConstants]
):
    def __init__(self, consts: DefenderConstants = None):
        consts = consts or DefenderConstants()
        super().__init__(consts)
        self.renderer = DefenderRenderer(self.consts)
        self.dh = DefenderHelper(self.consts)

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

    def render(self, state: DefenderState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: DefenderState) -> DefenderObservation:

        player = EntityPosition(
            x=jnp.asarray(state.space_ship_x, dtype=jnp.int32),
            y=jnp.asarray(state.space_ship_y, dtype=jnp.int32),
            width=jnp.asarray(self.consts.SPACE_SHIP_WIDTH, dtype=jnp.int32),
            height=jnp.asarray(self.consts.SPACE_SHIP_HEIGHT, dtype=jnp.int32),
        )

        return DefenderObservation(
            score=jnp.asarray(state.score, dtype=jnp.int32),
            player=player,
            lives=jnp.asarray(state.space_ship_lives, dtype=jnp.int32),
            enemy_states=jnp.asarray(state.enemy_states, dtype=jnp.float32),
            human_states=jnp.asarray(state.human_states, dtype=jnp.float32),
            bullet_active=jnp.asarray(state.bullet_active, dtype=jnp.float32),
            bullet_x=jnp.asarray(state.bullet_x, dtype=jnp.float32),
            bullet_y=jnp.asarray(state.bullet_y, dtype=jnp.float32),
            laser_active=jnp.asarray(state.laser_active, dtype=jnp.float32),
            laser_x=jnp.asarray(state.laser_x, dtype=jnp.float32),
            laser_y=jnp.asarray(state.laser_y, dtype=jnp.float32),
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: DefenderObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                obs.score.flatten(),
                jnp.concatenate(
                    [
                        jnp.atleast_1d(obs.player.x),
                        jnp.atleast_1d(obs.player.y),
                        jnp.atleast_1d(obs.player.width),
                        jnp.atleast_1d(obs.player.height),
                    ]
                ),
                obs.lives.flatten(),
                obs.enemy_states.flatten(),
                obs.human_states.flatten(),
                obs.bullet_active.flatten(),
                obs.bullet_x.flatten(),
                obs.bullet_y.flatten(),
                obs.laser_active.flatten(),
                obs.laser_x.flatten(),
                obs.laser_y.flatten(),
            ]
        )

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
        shoot = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.DOWNFIRE,
                    action == Action.UPFIRE,
                    action == Action.RIGHTFIRE,
                    action == Action.LEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )

        direction_x = jnp.where(left, -1, 0) + jnp.where(right, 1, 0)
        direction_y = jnp.where(up, -1, 0) + jnp.where(down, 1, 0)

        space_ship_facing_right = jax.lax.cond(
            direction_x != 0,
            lambda: direction_x > 0,
            lambda: state.space_ship_facing_right,
        )

        space_ship_speed = jax.lax.cond(
            direction_x != 0,
            lambda: (
                state.space_ship_speed
                + direction_x * self.consts.SPACE_SHIP_ACCELERATION
            ),
            lambda: (
                state.space_ship_speed
                - state.space_ship_speed * self.consts.SPACE_SHIP_BREAK
            ),
        )

        space_ship_speed = jnp.clip(
            space_ship_speed,
            -self.consts.SPACE_SHIP_MAX_SPEED,
            self.consts.SPACE_SHIP_MAX_SPEED,
        )

        space_ship_stopping_deadzone = 0.001

        space_ship_speed = jnp.where(
            jnp.abs(space_ship_speed) <= space_ship_stopping_deadzone,
            0,
            space_ship_speed,
        )

        space_ship_x = state.space_ship_x
        space_ship_y = state.space_ship_y

        x_speed = space_ship_speed
        y_speed = direction_y
        space_ship_x, space_ship_y = self._move_and_clip(
            space_ship_x, space_ship_y, x_speed, y_speed, self.consts.SPACE_SHIP_HEIGHT
        )

        # Decrease shooting cooldown
        shooting_cooldown = jax.lax.cond(
            state.shooting_cooldown > 0,
            lambda: state.shooting_cooldown - 1,
            lambda: 0,
        )

        # Shooting if the cooldown is down
        shoot = jnp.logical_and(shoot, shooting_cooldown <= 0)

        # Not be able to shoot in hyperspace
        hyperspace = space_ship_y < (2 - self.consts.SPACE_SHIP_HEIGHT)
        shoot_laser = jnp.logical_and(shoot, jnp.logical_not(hyperspace))

        # Shoot bomb if inside city
        in_city = (
            self.consts.WORLD_HEIGHT
            - self.consts.CITY_HEIGHT
            - self.consts.SPACE_SHIP_HEIGHT
        )
        shoot_smart_bomb = jnp.logical_and(
            shoot,
            space_ship_y > in_city,
        )

        # Shoot laser if not in hyperspace and city
        shoot_laser = jnp.logical_xor(shoot_laser, shoot_smart_bomb)

        # If smart bomb is the chosen shot, look up if it is available
        shoot_smart_bomb = jnp.logical_and(
            shoot_smart_bomb, state.smart_bomb_amount > 0
        )

        return (
            space_ship_x,
            space_ship_y,
            space_ship_speed,
            space_ship_facing_right,
            shooting_cooldown,
            shoot_laser,
            shoot_smart_bomb,
        )
    
    def _check_laser(self, state: DefenderState) -> DefenderState:
        laser_x = state.laser_x
        laser_y = state.laser_y
        laser_width = self.consts.LASER_WIDTH
        laser_height = self.consts.LASER_HEIGHT
        laser_active = self.dh._is_onscreen_from_game(
            state, laser_x, laser_y, laser_width, laser_height
        )

        return state._replace(laser_active=laser_active)

    def _laser_update(self, state: DefenderState) -> DefenderState:
        laser_x = state.laser_x
        laser_y = state.laser_y
        laser_speed_x = state.laser_dir_x * self.consts.LASER_SPEED
        laser_x, laser_y = self._move_and_wrap(laser_x, laser_y, laser_speed_x, 0.0)
        return state._replace(laser_x=laser_x, laser_y=laser_y)

    def _laser_step(self, state: DefenderState) -> DefenderState:
        state = jax.lax.cond(state.laser_active, lambda: self._laser_update(state), lambda: state)
        state = jax.lax.cond(state.laser_active, lambda: self._check_laser(state), lambda: state)
        return state
    
    def _camera_step(self, state: DefenderState) -> DefenderState:
        # Returns: camera_offset
        offset_gain = self.consts.CAMERA_OFFSET_GAIN
        camera_offset = state.camera_offset
        camera_offset += jnp.where(state.space_ship_facing_right, 1, -1) * offset_gain

        camera_offset = jnp.clip(
            camera_offset, -self.consts.CAMERA_OFFSET_MAX, self.consts.CAMERA_OFFSET_MAX
        )

        return state._replace(camera_offset=camera_offset)


    def _spawn_enemy_random_pos(self, state: DefenderState, e_type: int) -> DefenderState:

        def fill_sector(index):
            game_x, _, e_type, _, _ = self._get_enemy(state, index)
            sector = self.dh._which_sector(game_x)

            is_alive = jnp.logical_and(e_type != self.consts.INACTIVE, e_type != self.consts.DEAD)

            contribution = jnp.zeros(self.consts.ENEMY_SECTORS)
            val = jnp.where(is_alive, 1, 0)
            return contribution.at[sector].set(val)

        sector_amounts = jnp.sum(
            jax.vmap(fill_sector)(jnp.arange(self.consts.ENEMY_MAX_IN_GAME)), axis=0
        )

        # Determine at max 2 sectors, 1 is current one and other is left or right overlapping
        overlap_ease = 1  # Allow left and right border to be inside same sector

        camera_left_border = jnp.mod(
            state.camera_offset + state.space_ship_x - self.consts.SCREEN_WIDTH / 2 + overlap_ease,
            self.consts.WORLD_WIDTH,
        )
        camera_right_border = jnp.mod(
            state.camera_offset + state.space_ship_x + self.consts.SCREEN_WIDTH / 2 - overlap_ease,
            self.consts.WORLD_WIDTH,
        )

        left_border_sector = self.dh._which_sector(camera_left_border)
        right_border_sector = self.dh._which_sector(camera_right_border)

        # Max out both sectors
        sector_amounts = sector_amounts.at[left_border_sector].set(self.consts.ENEMY_MAX_IN_GAME)
        sector_amounts = sector_amounts.at[right_border_sector].set(self.consts.ENEMY_MAX_IN_GAME)

        smallest_sector = jnp.argmin(sector_amounts)
        left_bound, right_bound = self.dh._sector_bounds(smallest_sector)

        # Spawn randomly inside sector
        key = state.key
        key, subkey_x = jax.random.split(key)
        key, subkey_y = jax.random.split(key)

        game_x = jax.random.uniform(
            subkey_x, minval=left_bound, maxval=right_bound - self.consts.ENEMY_WIDTH
        )
        game_y = jax.random.uniform(
            subkey_y,
            minval=0,
            maxval=self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT,
        )

        state = self._spawn_enemy(state, game_x, game_y, e_type, 0.0, 0.0)

        return state._replace(key=key)

    def _update_human(
        self, state: DefenderState, index, game_x=-1.0, game_y=-1.0, h_state=-1.0
    ) -> DefenderState:
        human_state = state.human_states
        human = human_state[index]
        # Handle defaults
        game_x = jnp.where(game_x == -1.0, human[0], game_x)
        game_y = jnp.where(game_y == -1.0, human[1], game_y)
        h_state = jnp.where(h_state == -1.0, human[2], h_state)

        human_state = human_state.at[index].set([game_x, game_y, h_state])
        return state._replace(human_states=human_state)

    def _spawn_human(self, state: DefenderState, game_x) -> DefenderState:
        # Get first slot that has inactive human
        mask = jnp.array(state.human_states[:, 2] == self.consts.INACTIVE)
        match = mask.argmax()

        # Default values
        game_y = self.consts.HUMAN_INIT_GAME_Y
        h_state = self.consts.HUMAN_STATE_IDLE

        # If no open slot availabe, dismiss new human
        open_slot_available = jnp.logical_or(match != 0, mask[0])

        state = jax.lax.cond(
            open_slot_available,
            lambda: self._update_human(state, match, game_x, game_y, h_state),
            lambda: state,
        )

        return state

    def _update_enemy(
        self,
        state: DefenderState,
        index,
        game_x=-1.0,
        game_y=-1.0,
        enemy_type=-1.0,
        arg1=-1.0,
        arg2=-1.0,
    ) -> DefenderState:
        enemy_state = state.enemy_states
        enemy = state.enemy_states[index]
        # Handle defaults
        game_x = jnp.where(game_x == -1.0, enemy[0], game_x)
        game_y = jnp.where(game_y == -1.0, enemy[1], game_y)
        enemy_type = jnp.where(enemy_type == -1.0, enemy[2], enemy_type)
        arg1 = jnp.where(arg1 == -1.0, enemy[3], arg1)
        arg2 = jnp.where(arg2 == -1.0, enemy[4], arg2)

        enemy_state = enemy_state.at[index].set([game_x, game_y, enemy_type, arg1, arg2])
        return state._replace(enemy_states=enemy_state)


    def _add_score(self, state: DefenderState, score) -> DefenderState:
        old_score = jnp.floor_divide(state.score, self.consts.SCORE_BONUS_THRESHOLD)

        score += state.score

        new_score = jnp.floor_divide(score, self.consts.SCORE_BONUS_THRESHOLD)

        # Check for item threshold
        gain_items = new_score > old_score

        score = jnp.clip(score, 0, self.consts.SCORE_MAX)

        space_ship_lives = state.space_ship_lives + 1
        smart_bombs = state.smart_bomb_amount + 1
        state = jax.lax.cond(
            gain_items,
            lambda: state._replace(
                space_ship_lives=space_ship_lives, smart_bomb_amount=smart_bombs
            ),
            lambda: state,
        )
        return state._replace(score=score)

    def _end_level(self, state: DefenderState) -> DefenderState:
        # Add score for every human alive
        def alive_human_bonus(index, state):
            is_alive = state.human_states[index][2] == self.consts.HUMAN_STATE_IDLE
            state = jax.lax.cond(is_alive, lambda: self._add_score(state, 100), lambda: state)
            return state

        state = jax.lax.fori_loop(0, self.consts.HUMAN_MAX_AMOUNT, alive_human_bonus, state)

        # End level
        state = state._replace(
            game_state=self.consts.GAME_STATE_TRANSITION,
            camera_offset=self.consts.CAMERA_INIT_OFFSET,
            space_ship_speed=0.0,
            space_ship_x=jnp.asarray(self.consts.SPACE_SHIP_INIT_GAME_X).astype(jnp.float32),
            space_ship_y=jnp.asarray(self.consts.SPACE_SHIP_INIT_GAME_Y).astype(jnp.float32),
            space_ship_facing_right=self.consts.SPACE_SHIP_INIT_FACE_RIGHT,
            laser_active=False,
            bullet_active=False,
            enemy_states=jnp.zeros((self.consts.ENEMY_MAX_IN_GAME, 5)),
            human_states=jnp.zeros((self.consts.HUMAN_MAX_AMOUNT, 3)),
            shooting_cooldown=0,
            level=state.level + 1,
        )
        return state

    def _check_level_done(self, state: DefenderState) -> DefenderState:
        enemy_killed = state.enemy_killed
        needed_kills = jnp.asarray(
            [
                self.consts.LANDER_LEVEL_AMOUNT[state.level],
                self.consts.POD_LEVEL_AMOUNT[state.level],
                self.consts.BOMBER_LEVEL_AMOUNT[state.level],
            ]
        )
        is_done = jnp.array_equal(enemy_killed, needed_kills)
        state = jax.lax.cond(is_done, lambda: self._end_level(state), lambda: state)
        return state

    def _spawn_enemy(
        self, state: DefenderState, game_x, game_y, e_type, arg1, arg2
    ) -> DefenderState:
        # Find first enemy that is inactive
        mask = jnp.array(state.enemy_states[:, 2] == self.consts.INACTIVE)
        match = mask.argmax()
        # If no open slot availabe, dismiss new enemy
        open_slot_available = jnp.logical_or(match != 0, mask[0])
        state = jax.lax.cond(
            open_slot_available,
            lambda: self._update_enemy(state, match, game_x, game_y, e_type, arg1, arg2),
            lambda: state,
        )
        return state
    
    def _calculate_score(
        self, state: DefenderState, enemy_state: chex.Array
    ):
        old_enemy_counts = jnp.bincount(state.enemy_states[:, 3].astype(jnp.int32), length=8)
        new_enemy_counts = jnp.bincount(enemy_state[:, 3].astype(jnp.int32), length=8)

        enemy_diff = old_enemy_counts - new_enemy_counts
        score_multiplier = jnp.array([0, self.consts.LANDER_DEATH_SCORE, self.consts.POD_DEATH_SCORE, self.consts.BOMBER_DEATH_SCORE,
        self.consts.SWARMERS_DEATH_SCORE, self.consts.MUTANT_DEATH_SCORE, self.consts.BAITER_DEATH_SCORE, 0])

        score = sum(enemy_diff * score_multiplier)
        return score

    def _get_enemy(self, state: DefenderState, index):
        # Returns the enemy list at index
        return state.enemy_states[index]

    def _death_step(self, state: DefenderState):
        # Handles dead enemies to respawn new ones

        return

    def _is_colliding(
        self, e1_x, e1_y, e1_width, e1_height, e2_x, e2_y, e2_width, e2_height
    ) -> chex.Array:
        e1max_x = e1_x + e1_width
        e2max_x = e2_x + e2_width
        e1max_y = e1_y + e1_height
        e2max_y = e2_y + e2_height

        check_1 = e1_x <= e2max_x
        check_2 = e1max_x >= e2_x

        check_3 = e1_y <= e2max_y
        check_4 = e1max_y >= e2_y

        check_x = jnp.logical_and(check_1, check_2)
        check_y = jnp.logical_and(check_3, check_4)

        return jnp.logical_and(check_x, check_y)

    def _space_ship_collision_step(self, state: DefenderState) -> bool: 
        is_colliding = self._is_colliding(
            state.space_ship_x,
            state.space_ship_y,
            self.consts.SPACE_SHIP_WIDTH,
            self.consts.SPACE_SHIP_HEIGHT,
            state.bullet_x,
            state.bullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT
        )
        return jnp.where(state.bullet_active, is_colliding, False)

    def _space_ship_catching_humans(self, state: DefenderState):
        
        return


    def _reset_player(self, state: DefenderState) -> DefenderState:
        state = state._replace(
            game_state=self.consts.GAME_STATE_PLAYING,
            camera_offset=self.consts.CAMERA_INIT_OFFSET,
            space_ship_speed=0.0,
            space_ship_x=jnp.array(self.consts.SPACE_SHIP_INIT_GAME_X).astype(jnp.float32),
            space_ship_y=jnp.array(self.consts.SPACE_SHIP_INIT_GAME_Y).astype(jnp.float32),
            space_ship_facing_right=self.consts.SPACE_SHIP_INIT_FACE_RIGHT,
            shooting_cooldown=0,
        )
        return state
    
    def _lander_movement(
        self,
        lander: chex.Array,
        state: DefenderState,
    ) -> chex.Array:
        lander_x = lander[0]
        lander_y = lander[1]
        lander_state = lander[3]
        current_counter = lander[4]

        def check_proximity(human_state: chex.Array) -> chex.Array:

            return jnp.logical_and(
                jnp.abs(human_state[0] - lander_x - 5) < self.consts.LANDER_PICKUP_X_THRESHOLD,
                human_state[2] == self.consts.HUMAN_STATE_IDLE,
            )

        def lander_patrol(state: DefenderState) -> chex.Array:
            speed_x, speed_y = jax.lax.cond(
                state.space_ship_speed > 0,
                lambda: (-self.consts.ENEMY_SPEED, self.consts.LANDER_Y_SPEED),
                lambda: (self.consts.ENEMY_SPEED, self.consts.LANDER_Y_SPEED),
            )
            speed_x += state.space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED
            # check if on top of human to switch to descend

            proximity_checks = jax.vmap(check_proximity)(state.human_states)
            is_near_human = jnp.any(proximity_checks)
            # Check if any other lander (not this one) is already descending or picking up
            other_landers_descending = jnp.any(
                jnp.logical_or(
                    state.enemy_states[:, 3] == self.consts.LANDER_STATE_DESCEND,
                    state.enemy_states[:, 3] == self.consts.LANDER_STATE_PICKUP,
                ),
            )
            lander_state = jax.lax.cond(
                jnp.logical_and(is_near_human, jnp.logical_not(other_landers_descending)),
                lambda: self.consts.LANDER_STATE_DESCEND,
                lambda: self.consts.LANDER_STATE_PATROL,
            )

            x, y = self._move_and_wrap(lander_x, lander_y, speed_x, speed_y)
            new_lander = [x, y, lander[2], lander_state, counter_id]
            return new_lander

        def lander_descend() -> chex.Array:
            speed_x = 0.0
            speed_y = self.consts.LANDER_Y_SPEED * 5

            # Check if lander reached the bottom (human level)
            lander_state = jax.lax.cond(
                lander_y >= 115,
                lambda: self.consts.LANDER_STATE_PICKUP,
                lambda: self.consts.LANDER_STATE_DESCEND,
            )
            x, y = self._move_and_wrap(lander_x, lander_y, speed_x, speed_y)
            new_lander = [x, y, lander[2], lander_state, counter_id]
            return new_lander

        def lander_pickup(current_counter: float, state: DefenderState) -> chex.Array:
            speed_x = 0.0
            speed_y = 0.0
            current_counter += 1.0
            lander_state = self.consts.LANDER_STATE_PICKUP

            human_states = state.human_states

            near = jax.vmap(check_proximity)(human_states)
            human_index = jnp.argmax(near)

            lander_state, current_counter = jax.lax.cond(
                current_counter >= self.consts.LANDER_PICKUP_DURATION_FRAMES,
                lambda: (
                    self.consts.LANDER_STATE_ASCEND,
                    jnp.array(human_index, float),
                ),
                lambda: (lander_state, current_counter),
            )
            x, y = self._move_and_wrap(lander_x, lander_y, speed_x, speed_y)
            new_lander = [x, y, lander[2], lander_state, current_counter]
            return new_lander

        def lander_ascend(human_id: int, state: DefenderState) -> DefenderState:
            def lander_reached_top(
                human_index: int,
                state: DefenderState,
            ) -> chex.Array:
                speed_x = 0.0
                speed_y = -self.consts.LANDER_Y_SPEED * 5

                x, y = self._move_and_wrap(lander_x, lander_y, speed_x, speed_y)
                new_lander = [
                    x,
                    y,
                    lander[2],
                    self.consts.LANDER_STATE_PATROL,
                    counter_id,
                ]
                return new_lander

            def lander_ascend_continue(
                human_index: int,
                lander_y: float,
                state: DefenderState,
            ) -> chex.Array:
                speed_x = 0.0
                speed_y = -self.consts.LANDER_Y_SPEED * 5


                x, y = self._move_and_wrap(lander_x, lander_y, speed_x, speed_y)
                new_lander = [
                    x,
                    y,
                    lander[2],
                    self.consts.LANDER_STATE_ASCEND,
                    counter_id,
                ]
                return new_lander

            # Check if lander reached the top
            new_lander = jax.lax.cond(
                lander_y <= self.consts.LANDER_START_Y,
                lambda: lander_reached_top(jnp.array(human_id, int), state),
                lambda: lander_ascend_continue(jnp.array(human_id, int), lander_y, state),
            )

            return new_lander

        counter_id = lander[4]
        patrol_state = lander_patrol(state)
        descend_state = lander_descend()
        pickup_state = lander_pickup(current_counter, state)
        ascend_state = lander_ascend(counter_id, state)

        new_lander = jax.lax.switch(
            jnp.array(lander_state, int),
            [
                lambda: patrol_state,
                lambda: descend_state,
                lambda: pickup_state,
                lambda: ascend_state,
            ],
        )
        return jnp.stack(new_lander)

    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(enemy: chex.Array, state: DefenderState) -> DefenderState:
            enemy_type = enemy[2]
            inactive_state = enemy
            lander_state = self._lander_movement(enemy, state)
            # pod_state = self._pod_movement(enemy, state)
            # bomber_state = self._bomber_movement(enemy, state)
            # swarmers_state = self._swarmers_movement(enemy, state)
            # mutant_state = self._mutant_movement(enemy, state)
            baiter_state = enemy
            
            pod_state = enemy
            bomber_state = enemy
            swarmers_state = enemy
            mutant_state = enemy
            delete_state = enemy

            enemy_state = jax.lax.switch(
                jnp.array(enemy_type, int),
                [
                    lambda: inactive_state,  # Inactive
                    lambda: lander_state,  # Lander
                    lambda: pod_state,  # Pod
                    lambda: bomber_state,  # Bomber
                    lambda: swarmers_state,  # Swarmer
                    lambda: mutant_state,  # Mutant
                    lambda: baiter_state,  # Baiter
                    lambda: delete_state,  # To be deleted (used for lander when it reaches top with human)
                ],
            )

            return enemy_state

        def _enemy_move_switch_wrapped(enemy: chex.Array) -> chex.Array:
            return _enemy_move_switch(enemy, state)

        enemy_states_updated = jax.vmap(_enemy_move_switch_wrapped, in_axes=(0,))(state.enemy_states)

        return enemy_states_updated

    def _human_step(self, state: DefenderState) -> DefenderState:
        # check if human is being carried by lander and update position accordingly
        def update_human_position(human: chex.Array) -> chex.Array:
            human_x = human[0]
            human_y = human[1]
            human_state = human[2]

            def being_carried():
                # Get lander that is carrying the human
                def check_lander_carrying(h):
                    return jnp.logical_and(
                        h[3] == self.consts.LANDER_STATE_PICKUP,
                        h[4] == human_state,  # lander counter is stored in arg2
                    )

                lander_index = jnp.argmax(jax.vmap(check_lander_carrying)(state.enemy_states))
                lander = state.enemy_states[lander_index]
                lander_x = lander[0]
                lander_y = lander[1]

                new_human_x = lander_x + 5
                new_human_y = lander_y + 10

                return jnp.array([new_human_x, new_human_y, human_state])

            def not_being_carried():
                return human

            new_human = jax.lax.cond(
                human_state >= 0, being_carried, not_being_carried
            )
            return new_human
    
        human_states_updated = jax.vmap(update_human_position)(state.human_states)
        return human_states_updated

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: DefenderState, action: chex.Array
    ) -> Tuple[DefenderObservation, DefenderState, float, bool, DefenderInfo]:
        # Get all updated values from individual step functions
        previous_state = state

        # Randomness
        key, subkey = jax.random.split(state.key)
        state = state._replace(key=subkey)

        def state_playing(state) -> DefenderState:

            (
                space_ship_x,
                space_ship_y,
                space_ship_speed,
                space_ship_facing_right,
                shooting_cooldown,
                shoot_laser,
                shoot_smart_bomb,
            ) = self._space_ship_step(state, action)

            enemy_states = self._enemy_step(state)

            score = self._calculate_score(state, enemy_states)
            

            

            # human_states = self._human_step(state)

            state = self._camera_step(state)
            # state = self._enemy_step(state)
            # state = self._human_step(state)
            # state = self._bullet_step(state)
            # state = self._laser_step(state)
            # state = self._collision_step(state)


            state = jax.lax.cond(
                shoot_laser,
                lambda: state._replace(
                    laser_active=True,
                    laser_x=space_ship_x + jnp.where(space_ship_facing_right, self.consts.SPACE_SHIP_WIDTH, -self.consts.LASER_WIDTH),
                    laser_y=space_ship_y + self.consts.SPACE_SHIP_HEIGHT / 2,
                ),
                lambda: state
            )


            state = state._replace(
                space_ship_x=space_ship_x,
                space_ship_y=space_ship_y,
                space_ship_speed=space_ship_speed,
                space_ship_facing_right=space_ship_facing_right,
                shooting_cooldown=shooting_cooldown,
                enemy_states=enemy_states,
                score=score,
                # human_states=human_states
                        )
            
            return state

        state = jax.lax.cond(
            state.game_state == self.consts.GAME_STATE_PLAYING,
            lambda: state_playing(state),
            lambda: state,
        )

        def state_game_over(state) -> DefenderState:
            # For animation
            current_frame = state.shooting_cooldown + 1
            state = state._replace(shooting_cooldown=current_frame)
            # Check if game_over animation is done and space ship still has lives
            game_resume = jnp.logical_and(
                state.space_ship_lives > 0, self._get_done(state)
            )
            state = jax.lax.cond(
                game_resume, lambda: self._reset_player(state), lambda: state
            )
            return state

        state = jax.lax.cond(
            state.game_state == self.consts.GAME_STATE_GAMEOVER,
            lambda: state_game_over(state),
            lambda: state,
        )

        state = jax.lax.cond(
            state.game_state == self.consts.GAME_STATE_PLAYING,
            lambda: self._check_level_done(state),
            lambda: state,
        )

        def state_transition(state) -> DefenderState:
            # For duration
            current_frame = state.shooting_cooldown + 1
            game_resume = current_frame >= self.consts.LEVEL_TRANSITION_DURATION
            state = jax.lax.cond(
                game_resume,
                lambda: self._start_level(self._reset_player(state), state.level),
                lambda: state._replace(shooting_cooldown=current_frame),
            )
            return state

        state = jax.lax.cond(
            state.game_state == self.consts.GAME_STATE_TRANSITION,
            lambda: state_transition(state),
            lambda: state,
        )

        state = state._replace(step_counter=(state.step_counter + 1))
        observation = self._get_observation(state)
        env_reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)

        # Swap key to parent key
        state = state._replace(key=key)
        return observation, state, env_reward, done, info

    def _start_level(self, state: DefenderState, level: int) -> DefenderState:
        # Spawn Lander
        state = jax.lax.fori_loop(
            0,
            jnp.minimum(self.consts.LANDER_LEVEL_AMOUNT[level], self.consts.LANDER_MAX_AMOUNT),
            lambda _, state: self._spawn_enemy_random_pos(state, self.consts.LANDER),
            state,
        )

        # Spawn Bomber
        state = jax.lax.fori_loop(
            0,
            jnp.minimum(self.consts.BOMBER_LEVEL_AMOUNT[level], self.consts.BOMBER_MAX_AMOUNT),
            lambda _, state: self._spawn_enemy_random_pos(state, self.consts.BOMBER),
            state,
        )

        # Spawn Pod
        state = jax.lax.fori_loop(
            0,
            jnp.minimum(self.consts.POD_LEVEL_AMOUNT[level], self.consts.POD_MAX_AMOUNT),
            lambda _, state: self._spawn_enemy_random_pos(state, self.consts.POD),
            state,
        )

        # Spawn Humans
        state = jax.lax.fori_loop(
            0,
            self.consts.HUMAN_LEVEL_AMOUNT[level],
            lambda index, state: self._spawn_human(
                state,
                (10 + index * (self.consts.WORLD_WIDTH / self.consts.HUMAN_LEVEL_AMOUNT[level])),
            ),
            state,
        )
        return state

    


    def reset(self, key=None) -> Tuple[DefenderObservation, DefenderState]:
        key = jax.lax.cond(key == None, lambda: jax.random.PRNGKey(0), lambda: key)
        initial_state = DefenderState(
            # Game
            step_counter=jnp.array(0).astype(jnp.int32),
            level=jnp.array(0).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            game_state=jnp.array(self.consts.GAME_STATE_PLAYING).astype(jnp.int32),
            # Camera
            camera_offset=jnp.array(self.consts.CAMERA_INIT_OFFSET).astype(jnp.int32),
            # Space Ship
            space_ship_speed=jnp.array(0).astype(jnp.float32),
            space_ship_x=jnp.array(self.consts.SPACE_SHIP_INIT_GAME_X).astype(jnp.float32),
            space_ship_y=jnp.array(self.consts.SPACE_SHIP_INIT_GAME_Y).astype(jnp.float32),
            space_ship_facing_right=jnp.array(self.consts.SPACE_SHIP_INIT_FACE_RIGHT).astype(
                jnp.bool
            ),
            space_ship_lives=jnp.array(self.consts.SPACE_SHIP_INIT_LIVES).astype(jnp.int32),
            # Laser
            laser_active=jnp.array(False).astype(jnp.bool),
            laser_x=jnp.array(0).astype(jnp.float32),
            laser_y=jnp.array(0).astype(jnp.float32),
            laser_dir_x=jnp.array(0).astype(jnp.float32),
            # Smart Bomb
            smart_bomb_amount=jnp.array(self.consts.SPACE_SHIP_INIT_BOMBS).astype(jnp.int32),
            # Bullet
            bullet_active=jnp.array(False).astype(jnp.bool),
            bullet_x=jnp.array(0).astype(jnp.float32),
            bullet_y=jnp.array(0).astype(jnp.float32),
            bullet_dir_x=jnp.array(0).astype(jnp.float32),
            bullet_dir_y=jnp.array(0).astype(jnp.float32),
            # Enemies: x,y,type,arg1,arg2
            enemy_states=jnp.zeros((self.consts.ENEMY_MAX_IN_GAME, 5)).astype(jnp.float32),
            enemy_killed=jnp.zeros(3).astype(jnp.int32),
            # Humans
            human_states=jnp.zeros((self.consts.HUMAN_MAX_AMOUNT, 3)).astype(jnp.float32),
            # Cooldowns
            shooting_cooldown=jnp.array(0).astype(jnp.int32),
            bring_back_human=jnp.array(0).astype(jnp.int32),
            # Randomness
            key=key,
        )
        initial_state = self._start_level(initial_state, 0)
        observation = self._get_observation(initial_state)
        return observation, initial_state

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "score": spaces.Box(
                    low=0, high=self.consts.SCORE_MAX + 1, shape=(), dtype=jnp.int32
                ),
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(
                            low=0, high=160 * 256, shape=(), dtype=jnp.int32
                        ),
                        "y": spaces.Box(
                            low=0, high=210 * 256, shape=(), dtype=jnp.int32
                        ),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(
                            low=0, high=210, shape=(), dtype=jnp.int32
                        ),
                    },
                ),
                "lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
                "enemy_states": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.consts.ENEMY_MAX_IN_GAME, 5),
                    dtype=jnp.float32,
                ),
                "human_states": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.consts.HUMAN_MAX_AMOUNT, 3),
                    dtype=jnp.float32,
                ),
                "bullet_active": spaces.Discrete(2),
                "bullet_x": spaces.Box(
                    low=0, high=self.consts.WORLD_WIDTH, shape=(), dtype=jnp.float32
                ),
                "bullet_y": spaces.Box(
                    low=0, high=self.consts.WORLD_HEIGHT, shape=(), dtype=jnp.float32
                ),
                "laser_active": spaces.Discrete(2),
                "laser_x": spaces.Box(
                    low=0, high=self.consts.WORLD_WIDTH, shape=(), dtype=jnp.float32
                ),
                "laser_y": spaces.Box(
                    low=0, high=self.consts.WORLD_HEIGHT, shape=(), dtype=jnp.float32
                ),
            }
        )




    def _wrap_pos(self, game_x: float, game_y: float):
        return (
            game_x % self.consts.WORLD_WIDTH,
            game_y
            % (
                self.consts.WORLD_HEIGHT
                - self.consts.CITY_HEIGHT  # move already when the top of the city == top of entity
            ),
        )

    def _move(
        self, game_x: float, game_y: float, x_speed: float, y_speed: float
    ) -> Tuple[float, float]:
        new_game_x = game_x + x_speed
        new_game_y = game_y + y_speed
        # Wrap only around x, y not needed
        new_game_x, _ = self._wrap_pos(new_game_x, 0)
        return new_game_x, new_game_y.astype(float)

    def _move_and_clip(
        self, game_x, game_y, x_speed, y_speed, height
    ) -> Tuple[float, float]:
        new_game_x, new_game_y = self._move(game_x, game_y, x_speed, y_speed)
        new_game_y = jnp.clip(new_game_y, 0 - height, self.consts.WORLD_HEIGHT - height)

        return new_game_x, new_game_y.astype(float)

    def _move_and_wrap(
        self, game_x: float, game_y: float, x_speed: float, y_speed: float
    ) -> Tuple[float, float]:
        new_game_x, new_game_y = self._move(game_x, game_y, x_speed, y_speed)
        new_game_x, new_game_y = self._wrap_pos(new_game_x, new_game_y)
        return new_game_x, new_game_y

    def _move_with_space_ship(
        self,
        state: DefenderState,
        game_x,
        game_y,
        speed_x,
        speed_y,
        space_ship_coefficient,
    ):

        speed_x += state.space_ship_speed * space_ship_coefficient

        new_game_x, new_game_y = self._move(game_x, game_y, speed_x, speed_y)
        return new_game_x, new_game_y

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: DefenderState) -> DefenderInfo:
        return DefenderInfo(score=state.score, step_counter=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: DefenderState, state: DefenderState) -> float:
        reward = state.score - previous_state.score
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: DefenderState) -> bool:
        is_done = jnp.logical_and(
            state.game_state == self.consts.GAME_STATE_GAMEOVER,
            state.shooting_cooldown == self.consts.SPACE_SHIP_DEATH_ANIM_FRAME_AMOUNT,
        )
        return is_done
