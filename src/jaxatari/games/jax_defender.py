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
        SPACE_SHIP_DEATH_LOOP_AMOUNT * SPACE_SHIP_DEATH_LOOP_FRAMES + SPACE_SHIP_DEATH_EXPLOSION_AMOUNT + 40  # After explosion empty frames
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
    SWARMERS_MAX_SPEED: float = 1.0
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
        camera_left_border = jnp.mod(camera_game_x - self.consts.SCREEN_WIDTH / 2, self.consts.WORLD_WIDTH)

        camera_right_border = jnp.mod(camera_game_x + self.consts.SCREEN_WIDTH / 2, self.consts.WORLD_WIDTH)

        is_in_left_wrap = jnp.logical_and(game_x >= camera_left_border, camera_left_border > camera_game_x)
        is_in_right_wrap = jnp.logical_and(game_x < camera_right_border, camera_right_border < camera_game_x)

        screen_x = (game_x - camera_game_x + camera_screen_x).astype(jnp.int32)

        screen_x = jax.lax.cond(
            is_in_left_wrap,
            lambda: jnp.mod(game_x, camera_left_border).astype(jnp.int32),
            lambda: screen_x,
        )

        screen_x = jax.lax.cond(
            is_in_right_wrap,
            lambda: (self.consts.WORLD_WIDTH - camera_game_x + game_x + camera_screen_x).astype(jnp.int32),
            lambda: screen_x,
        )

        screen_y = game_y + self.consts.GAME_AREA_TOP
        return screen_x, screen_y

    # Calculate with in game positions
    def _is_onscreen_from_game(self, state: DefenderState, game_x, game_y, width: int, height: int):
        screen_x, screen_y = self._onscreen_pos(state, game_x, game_y)
        x_onscreen = jnp.logical_and(screen_x + width > 0, screen_x < self.consts.SCREEN_WIDTH)
        y_onscreen = jnp.logical_and(
            screen_y + height > self.consts.GAME_AREA_TOP,
            screen_y < self.consts.GAME_AREA_BOTTOM,
        )
        return jnp.logical_and(x_onscreen, y_onscreen)

    # Calculate with on screen positions
    def _is_onscreen_from_screen(self, screen_x, screen_y, width: int, height: int):
        x_onscreen = jnp.logical_and(screen_x + width > 0, screen_x < self.consts.SCREEN_WIDTH)
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
        return (sector_width * sector).astype(jnp.int32), (sector_width * (sector + 1)).astype(jnp.int32)

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
                    ((0, self.ENEMY_MASK_SIZE[0] - d_h), (0, self.ENEMY_MASK_SIZE[1] - d_w)),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
                jnp.pad(
                    self.SHAPE_MASKS["death_blue"],
                    ((0, self.ENEMY_MASK_SIZE[0] - d_h), (0, self.ENEMY_MASK_SIZE[1] - d_w)),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
                jnp.pad(
                    self.SHAPE_MASKS["death_red"],
                    ((0, self.ENEMY_MASK_SIZE[0] - d_h), (0, self.ENEMY_MASK_SIZE[1] - d_w)),
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

        self.PARTICLE_COLOR_IDS = jnp.array([color_s_blue, color_pink, color_b_blue, color_m_red, color_l_yellow])

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
        left_border = jnp.mod(camera_game_x - self.consts.WORLD_WIDTH / 2, self.consts.WORLD_WIDTH)

        # Calculate position inside scanner
        is_after_zero_wrap = game_x < left_border
        game_to_scanner_ratio_x = self.consts.SCANNER_WIDTH / self.consts.WORLD_WIDTH
        game_to_scanner_ratio_y = self.consts.SCANNER_HEIGHT / self.consts.WORLD_HEIGHT
        screen_x = jax.lax.cond(
            is_after_zero_wrap,
            lambda: (self.consts.WORLD_WIDTH - left_border + game_x) * game_to_scanner_ratio_x,
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
            right_barrier = float(self.consts.SCANNER_SCREEN_X + self.consts.SCANNER_WIDTH)
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

            onscreen = self.dh._is_onscreen_from_screen(screen_x, screen_y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT)

            is_active_and_onscreen = jnp.logical_and(enemy_type != self.consts.INACTIVE, onscreen)

            scanner_width = self.consts.ENEMY_SCANNER_WIDTH
            scanner_height = self.consts.ENEMY_SCANNER_HEIGHT

            # Render on scanner
            r = jax.lax.cond(
                enemy_type != self.consts.INACTIVE,
                lambda: render_on_scanner(enemy[0], enemy[1], scanner_width, scanner_height, color_id, r),
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
                    lambda: self.jr.render_at_clipped(r, screen_x, screen_y, pickup_mask),
                    lambda: self.jr.render_at_clipped(r, screen_x, screen_y, normal_mask),
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

        raster = jax.lax.fori_loop(0, self.consts.ENEMY_MAX_IN_GAME, render_enemy, raster)

        # Used for bullet color and human color
        current_particle_color_id = self.PARTICLE_COLOR_IDS[
            jnp.mod(
                jnp.floor_divide(state.step_counter, self.consts.PARTICLE_FLICKER_EVERY_N_FRAMES),
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
                lambda: render_on_scanner(game_x, game_y, scanner_width, scanner_height, color_id, r),
                lambda: r,
            )

            def render_normal(r):
                # Test for hyperspace
                in_screen = screen_y > (self.consts.GAME_AREA_TOP - self.consts.SPACE_SHIP_HEIGHT)

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
                r = self.jr.render_at(r, new_x, new_y, mask, flip_horizontal=flip_horizontal)
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
            laser2_x = screen_x + (jnp.where(state.space_ship_facing_right, 1.0, -1.0) * self.consts.LASER_2ND_OFFSET)
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
                jnp.floor_divide(state.space_ship_x + state.camera_offset, self.consts.CITY_WIDTH),
                self.consts.CITY_WIDTH,
            )

            city_game_y = self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT
            city_screen_x, city_screen_y = self.dh._onscreen_pos(state, city_game_x, city_game_y)

            city_mask = self.SHAPE_MASKS["city"]
            r = self.jr.render_at_clipped(r, city_screen_x, city_screen_y, city_mask)
            r = self.jr.render_at_clipped(r, city_screen_x + self.consts.CITY_WIDTH, city_screen_y, city_mask)
            r = self.jr.render_at_clipped(r, city_screen_x + 2 * self.consts.CITY_WIDTH, city_screen_y, city_mask)
            r = self.jr.render_at_clipped(r, city_screen_x - self.consts.CITY_WIDTH, city_screen_y, city_mask)
            r = self.jr.render_at_clipped(r, city_screen_x - 2 * self.consts.CITY_WIDTH, city_screen_y, city_mask)
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
            is_active_and_onscreen = jnp.logical_and(human_state != self.consts.INACTIVE, onscreen)

            color_id = current_particle_color_id
            scanner_width = self.consts.HUMAN_SCANNER_WIDTH
            scanner_height = self.consts.HUMAN_SCANNER_HEIGHT

            # Render on scanner
            r = jax.lax.cond(
                human_state != self.consts.INACTIVE,
                lambda: render_on_scanner(human[0], human[1], scanner_width, scanner_height, color_id, r),
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
        raster = jax.lax.fori_loop(0, self.consts.HUMAN_MAX_AMOUNT, render_human, raster)

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


class JaxDefender(JaxEnvironment[DefenderState, DefenderObservation, DefenderInfo, DefenderConstants]):

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

    ## -------- SCORE ------------------------------

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
            lambda: state._replace(space_ship_lives=space_ship_lives, smart_bomb_amount=smart_bombs),
            lambda: state,
        )
        return state._replace(score=score)

    ## -------- ENTITY UTILS ------------------------------

    def _get_enemy(self, state: DefenderState, index):
        # Returns the enemy list at index
        return state.enemy_states[index]

    def _spawn_enemy(self, state: DefenderState, game_x, game_y, e_type, arg1, arg2) -> DefenderState:
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

    def _delete_enemy(self, enemy: chex.Array) -> chex.Array:
        enemy_type = enemy[2].astype(jnp.int32)
        e_arg1 = enemy[3]
        e_arg2 = enemy[4]

        # When iterating in enemy_step, dead ones go to inactive, only one frame dead for animation
        is_dead = enemy_type == self.consts.DEAD
        new_type = jax.lax.cond(is_dead, lambda: self.consts.INACTIVE, lambda: self.consts.DEAD)
        color = jax.lax.switch(
            enemy_type,
            [
                lambda: self.consts.DEAD_YELLOW,
                lambda: self.consts.DEAD_YELLOW,
                lambda: self.consts.DEAD_YELLOW,
                lambda: self.consts.DEAD_BLUE,
                lambda: self.consts.DEAD_YELLOW,
                lambda: self.consts.DEAD_RED,
                lambda: self.consts.DEAD_BLUE,
                lambda: self.consts.DEAD_RED,
            ],
        )

        # Score gets subtracted by 50, as when an enemy dies, the score increases
        # and adds itself -50, then when going from dead to inactive, the other 50 get added, to simulate
        # number going up
        score = jax.lax.switch(
            enemy_type,
            [
                lambda: 50,
                lambda: self.consts.LANDER_DEATH_SCORE,
                lambda: self.consts.POD_DEATH_SCORE,
                lambda: self.consts.BOMBER_DEATH_SCORE,
                lambda: self.consts.SWARMERS_DEATH_SCORE,
                lambda: self.consts.MUTANT_DEATH_SCORE,
                lambda: self.consts.BAITER_DEATH_SCORE,
                lambda: 100,
            ],
        )

        def add_killed_enemy(state: DefenderState, e_type) -> DefenderState:
            enemy_killed = state.enemy_killed
            e_index = e_type - 1
            new_amount = enemy_killed[e_index] + 1
            enemy_killed = enemy_killed.at[e_index].set(new_amount)
            state = state._replace(enemy_killed=enemy_killed)
            return state

        # state = jax.lax.cond(
        #     enemy_type < self.consts.SWARMERS,
        #     lambda: add_killed_enemy(state, enemy_type),
        #     lambda: state,
        # )

        def lander_death(state: DefenderState) -> DefenderState:
            current_killed = state.enemy_killed[self.consts.LANDER - 1]

            # Spawn human if lander held one
            # is_holding = e_arg1 == self.consts.LANDER_STATE_ASCEND
            # state = jax.lax.cond(
            #     is_holding,
            #     lambda: self._set_human_from_lander_falling(state, index),
            #     lambda: state,
            # )

            # If its with collision with player, kill the human and give nothing
            # game_over = state.game_state == self.consts.GAME_STATE_GAMEOVER
            # state = jax.lax.cond(
            #     game_over,
            #     lambda: self._update_human(state, e_arg2, 0.0, 0.0, self.consts.INACTIVE),
            #     lambda: state,
            # )

            # Spawn new landers
            # spawn_more = jnp.less(current_killed, self.consts.LANDER_LEVEL_AMOUNT[state.level])

            # state = jax.lax.cond(
            #     spawn_more,
            #     lambda: self._spawn_enemy_random_pos(state, self.consts.LANDER),
            #     lambda: state,
            # )
            return state

        # state = jax.lax.cond(enemy_type == self.consts.LANDER, lambda: lander_death(state), lambda: state)

        # def pod_death(state: DefenderState, index: int) -> DefenderState:
        #     key, subkey = jax.random.split(state.key)
        #     spawn_amount = jnp.round(jax.random.uniform(subkey, minval=1, maxval=2)).astype(jnp.int32)
        #     x, y = self._get_enemy(state, index)[:2]
        #     state = jax.lax.fori_loop(
        #         0,
        #         spawn_amount,
        #         lambda _, state: self._spawn_enemy_around_pos(state, x, y, self.consts.SWARMERS),
        #         state,
        #     )
        #     return state._replace(key=key)

        # state = jax.lax.cond(enemy_type == self.consts.POD, lambda: pod_death(state, index), lambda: state)

        new_enemy = enemy.at[2].set(new_type)
        new_enemy = new_enemy.at[3].set(color)
        return new_enemy, score
    
    def _delete_enemy_index(self, state: DefenderState, index) -> DefenderState:
        enemy = state.enemy_states[index]
        new_enemy = self._delete_enemy(enemy)
        jax.debug.print("Deleted enemy at index {index} with type {type}", index=index, type=enemy[2])
        enemy_state = state.enemy_states.at[index].set(new_enemy)
        return state._replace(enemy_states=enemy_state)

    def _enemies_on_screen(self, state) -> Tuple:
        # Returns an array of indices corresponding to position in enemy_states, and a max_indice to random under

        indices = jnp.full((self.consts.ENEMY_MAX_ON_SCREEN,), -1)

        # Returns data = (enemy_states, indices)
        def add_indices(i, data):
            enemy_states = data[0]
            indices = data[1]
            enemy_count_in_indices = data[2]

            enemy = enemy_states[i]
            x = enemy[0]
            y = enemy[1]
            is_active = enemy[2] != self.consts.INACTIVE
            is_onscreen = jax.lax.cond(
                is_active,
                lambda: self.dh._is_onscreen_from_game(state, x, y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT),
                lambda: False,
            )

            # Pass -1 for non on screen enemies, array has to be static size
            indices = indices.at[enemy_count_in_indices].set(jax.lax.cond(is_onscreen, lambda: i, lambda: -1))
            enemy_count_in_indices += is_onscreen

            return (enemy_states, indices, enemy_count_in_indices)

        result = jax.lax.fori_loop(
            0,
            self.consts.ENEMY_MAX_IN_GAME,
            add_indices,
            (state.enemy_states, indices, 0),
        )

        indices = result[1]
        max_indice = result[2]

        return (indices, max_indice)

    def _spawn_enemy_random_pos(self, state: DefenderState, e_type: int) -> DefenderState:

        def fill_sector(index):
            game_x, _, e_type, _, _ = self._get_enemy(state, index)
            sector = self.dh._which_sector(game_x)

            is_alive = jnp.logical_and(e_type != self.consts.INACTIVE, e_type != self.consts.DEAD)

            contribution = jnp.zeros(self.consts.ENEMY_SECTORS)
            val = jnp.where(is_alive, 1, 0)
            return contribution.at[sector].set(val)

        sector_amounts = jnp.sum(jax.vmap(fill_sector)(jnp.arange(self.consts.ENEMY_MAX_IN_GAME)), axis=0)

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

        game_x = jax.random.uniform(subkey_x, minval=left_bound, maxval=right_bound - self.consts.ENEMY_WIDTH)
        game_y = jax.random.uniform(
            subkey_y,
            minval=0,
            maxval=self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT,
        )

        state = self._spawn_enemy(state, game_x, game_y, e_type, 0.0, 0.0)

        return state._replace(key=key)

    def _spawn_enemy_around_pos(self, state: DefenderState, center_x: float, center_y: float, e_type: int) -> DefenderState:
        # Spawn randomly around position within a radius
        key = state.key
        key, subkey_angle = jax.random.split(key)
        key, subkey_radius = jax.random.split(key)

        angle = jax.random.uniform(subkey_angle, minval=0.0, maxval=2 * jnp.pi)
        radius = jax.random.uniform(
            subkey_radius,
            minval=self.consts.ENEMY_SPAWN_AROUND_MIN_RADIUS,
            maxval=self.consts.ENEMY_SPAWN_AROUND_MAX_RADIUS,
        )

        offset_x = radius * jnp.cos(angle)
        offset_y = radius * jnp.sin(angle)

        game_x = center_x + offset_x
        game_y = center_y + offset_y

        # Wrap position
        game_x, game_y = self._wrap_pos(game_x, game_y)

        state = self._spawn_enemy(state, game_x, game_y, e_type, 0.0, 0.0)

        return state._replace(key=key)

    def _update_human(self, state: DefenderState, index, game_x=-1.0, game_y=-1.0, h_state=-1.0) -> DefenderState:
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

    ## -------- MOVEMENT ------------------------------

    # Wrap function, returns wrapped position
    def _wrap_pos(self, game_x: float, game_y: float):
        return game_x % self.consts.WORLD_WIDTH, game_y % (
            self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT  # move already when the top of the city == top of entity
        )

    def _move(self, game_x: float, game_y: float, x_speed: float, y_speed: float) -> Tuple[float, float]:
        new_game_x = game_x + x_speed
        new_game_y = game_y + y_speed
        # Wrap only around x, y not needed
        new_game_x, _ = self._wrap_pos(new_game_x, 0)
        return new_game_x, new_game_y.astype(float)

    def _move_and_clip(self, game_x, game_y, x_speed, y_speed, height) -> Tuple[float, float]:
        new_game_x, new_game_y = self._move(game_x, game_y, x_speed, y_speed)
        new_game_y = jnp.clip(new_game_y, 0 - height, self.consts.WORLD_HEIGHT - height)

        return new_game_x, new_game_y.astype(float)

    def _move_and_wrap(self, game_x: float, game_y: float, x_speed: float, y_speed: float) -> Tuple[float, float]:
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

    def _shoot_smart_bomb(self, state: DefenderState) -> DefenderState:
        # Get all enemies on screen and delete them
        enemy_indices, max_indice = self._enemies_on_screen(state)

        def delete_enemy(index, state: DefenderState) -> DefenderState:
            return self._delete_enemy_index(state, enemy_indices[index])

        state = jax.lax.fori_loop(0, max_indice, delete_enemy, state)
        smart_bomb_amount = state.smart_bomb_amount - 1

        shooting_cooldown = self.consts.SPACE_SHIP_SHOOT_CD

        state = state._replace(smart_bomb_amount=smart_bomb_amount, shooting_cooldown=shooting_cooldown)
        return state

    def _shoot_laser(self, state: DefenderState) -> DefenderState:
        # When spawning, speed gets added instantly, so frame 0 should still be at the right position
        laser_x_adjust = self.consts.LASER_SPEED - 5
        laser_x = jax.lax.cond(
            state.space_ship_facing_right,
            lambda: state.space_ship_x + self.consts.SPACE_SHIP_WIDTH - laser_x_adjust,
            lambda: state.space_ship_x - self.consts.LASER_WIDTH + laser_x_adjust,
        )
        laser_y = state.space_ship_y + 0.5 + self.consts.SPACE_SHIP_HEIGHT / 2
        laser_dir_x = jnp.where(state.space_ship_facing_right, 1.0, -1.0)

        shooting_cooldown = self.consts.SPACE_SHIP_SHOOT_CD

        state = state._replace(
            laser_x=laser_x,
            laser_y=laser_y,
            laser_dir_x=laser_dir_x,
            laser_active=True,
            shooting_cooldown=shooting_cooldown,
        )

        return state

    def _space_ship_step(self, state: DefenderState, action: chex.Array) -> DefenderState:
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
            lambda: state.space_ship_speed + direction_x * self.consts.SPACE_SHIP_ACCELERATION,
            lambda: state.space_ship_speed - state.space_ship_speed * self.consts.SPACE_SHIP_BREAK,
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
        space_ship_x, space_ship_y = self._move_and_clip(space_ship_x, space_ship_y, x_speed, y_speed, self.consts.SPACE_SHIP_HEIGHT)

        # Decrease shooting cooldown
        shooting_cooldown = jax.lax.cond(
            state.shooting_cooldown > 0,
            lambda: state.shooting_cooldown - 1,
            lambda: 0,
        )

        state = state._replace(
            space_ship_speed=space_ship_speed,
            space_ship_x=space_ship_x,
            space_ship_y=space_ship_y,
            space_ship_facing_right=space_ship_facing_right,
            shooting_cooldown=shooting_cooldown,
        )

        # Shooting if the cooldown is down
        shoot = jnp.logical_and(shoot, shooting_cooldown <= 0)

        # Not be able to shoot in hyperspace
        hyperspace = space_ship_y < (2 - self.consts.SPACE_SHIP_HEIGHT)
        shoot_laser = jnp.logical_and(shoot, jnp.logical_not(hyperspace))

        # Shoot bomb if inside city
        in_city = self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT - self.consts.SPACE_SHIP_HEIGHT
        shoot_smart_bomb = jnp.logical_and(
            shoot,
            space_ship_y > in_city,
        )

        # Shoot laser if not in hyperspace and city
        shoot_laser = jnp.logical_xor(shoot_laser, shoot_smart_bomb)

        # If smart bomb is the chosen shot, look up if it is available
        shoot_smart_bomb = jnp.logical_and(shoot_smart_bomb, state.smart_bomb_amount > 0)

        state = jax.lax.cond(shoot_laser, lambda: self._shoot_laser(state), lambda: state)
        state = jax.lax.cond(shoot_smart_bomb, lambda: self._shoot_smart_bomb(state), lambda: state)

        return state

    ## -------- CAMERA STEP ------------------------------

    def _camera_step(self, state: DefenderState) -> DefenderState:
        # Returns: camera_offset
        offset_gain = self.consts.CAMERA_OFFSET_GAIN
        camera_offset = state.camera_offset
        camera_offset += jnp.where(state.space_ship_facing_right, 1, -1) * offset_gain

        camera_offset = jnp.clip(camera_offset, -self.consts.CAMERA_OFFSET_MAX, self.consts.CAMERA_OFFSET_MAX)

        return state._replace(camera_offset=camera_offset)

    ## -------- ENEMY STEP ------------------------------

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

                human_states = state.human_states
                human = human_states[human_index]
                human_x = human[0]
                human_y = human[1]
                new_human = [
                    human_x,
                    human_y,
                    self.consts.INACTIVE,
                ]
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

                human_y = lander_y + 5  # Move human up with lander
                state = self._update_human(
                    state,
                    human_index,
                    game_y=human_y,
                    h_state=self.consts.HUMAN_STATE_ABDUCTED,
                )

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

    def _pod_movement(self, pod: chex.Array, state: DefenderState) -> chex.Array:
        pod_x = pod[0]
        pod_y = pod[1]

        speed_x, speed_y = jax.lax.cond(
            state.space_ship_speed > 0,
            lambda: (-self.consts.ENEMY_SPEED, self.consts.POD_Y_SPEED),
            lambda: (self.consts.ENEMY_SPEED, self.consts.POD_Y_SPEED),
        )

        x, y = self._move_and_wrap(
            pod_x,
            pod_y,
            speed_x + state.space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            speed_y,
        )
        new_pod = [x, y, pod[2], pod[3], pod[4]]
        return jnp.stack(new_pod)

    def _mutant_movement(self, mutant: chex.Array, state: DefenderState) -> chex.Array:
        mutant_x = mutant[0]
        mutant_y = mutant[1]

        speed_x, speed_y = jax.lax.cond(
            state.space_ship_speed > 0,
            lambda: (-self.consts.ENEMY_SPEED, self.consts.POD_Y_SPEED),
            lambda: (self.consts.ENEMY_SPEED, self.consts.POD_Y_SPEED),
        )

        x, y = self._move_and_wrap(
            mutant_x,
            mutant_y,
            speed_x + state.space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            speed_y,
        )
        new_mutant = [x, y, mutant[2], mutant[3], mutant[4]]
        return jnp.stack(new_mutant)

    def _bomber_movement(
        self,
        bomber: chex.Array,
        state: DefenderState,
    ) -> chex.Array:

        x_pos = bomber[0]
        y_pos = bomber[1]
        direction_right = bomber[4]
        speed_x = self.consts.ENEMY_SPEED
        # acceleration in x direction
        speed_x = jax.lax.cond(direction_right, lambda s: s, lambda s: -s, operand=speed_x)

        # change direction if spaceship is crossed and passed by 30
        direction_right = jax.lax.cond(
            jnp.logical_and(direction_right, x_pos > state.space_ship_x + 30),
            lambda _: 0.0,
            lambda _: jax.lax.cond(
                jnp.logical_and(jnp.logical_not(direction_right), x_pos < state.space_ship_x - 30),
                lambda _: 1.0,
                lambda _: direction_right,
                operand=None,
            ),
            operand=None,
        )
        x_pos, y_pos = self._move_and_wrap(
            x_pos,
            y_pos,
            speed_x + state.space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            self.consts.BOMBER_Y_SPEED,
        )
        new_bomber = [x_pos, y_pos, bomber[2], bomber[3], direction_right]
        return jnp.stack(new_bomber)

    def _swarmers_movement(
        self,
        swarmer: chex.Array,
        state: DefenderState,
    ) -> chex.Array:
        x_pos = swarmer[0]
        y_pos = swarmer[1]
        speed = swarmer[4]
        swarmer_direction = swarmer[3]
        speed = speed + 0.05  # acceleration over time
        # max speed
        speed = jnp.clip(speed, a_min=-self.consts.SWARMERS_MAX_SPEED, a_max=self.consts.SWARMERS_MAX_SPEED)
        speed_x = speed
        # acceleration in x direction
        speed_x = jax.lax.cond(
            swarmer_direction == 1,
            lambda s: s,
            lambda s: -s,
            operand=speed_x,
        )
        swarmer_direction = jax.lax.cond(
            swarmer_direction == 0,
            lambda _: jax.lax.cond(
                x_pos < state.space_ship_x,
                lambda _: 1.0,
                lambda _: -1.0,
                operand=None,
            ),
            lambda _: jax.lax.cond(
                swarmer_direction == 3,
                lambda _: jax.lax.cond(
                    x_pos > state.space_ship_x + 100,
                    lambda _: -1.0,
                    lambda _: jax.lax.cond(
                        x_pos < state.space_ship_x - 100,
                        lambda _: 1.0,
                        lambda _: swarmer_direction,
                        operand=None,
                    ),
                    operand=None,
                ),
                lambda _: jax.lax.cond(
                    jnp.logical_and(x_pos < state.space_ship_x + 5, x_pos > state.space_ship_x - 5),
                    lambda _: 0.0,
                    lambda _: swarmer_direction,
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

        speed_x, speed_y = jax.lax.cond(
            jnp.logical_or(swarmer_direction == 3, swarmer_direction == 0),
            lambda: (0.0, self.consts.SWARMERS_Y_SPEED),
            lambda: (speed_x, 0.0),
        )

        speed = jnp.where(swarmer_direction == 0, 0.0, speed)

        x_pos, y_pos = self._move_and_wrap(
            x_pos,
            y_pos,
            speed_x + state.space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            speed_y,
        )

        new_swarmer = [x_pos, y_pos, swarmer[2], swarmer_direction, speed]
        return jnp.stack(new_swarmer)

    def _set_human_from_lander_falling(
        self,
        state: DefenderState,
        lander_index,
    ) -> DefenderState:

        human_index = state.enemy_states[lander_index][4].astype(int)
        human = state.human_states[human_index]
        human_y = human[1]
        human_status = jnp.where(
            human_y > self.consts.HUMAN_DEADLY_FALL_HEIGHT,
            self.consts.HUMAN_STATE_FALLING,
            self.consts.HUMAN_STATE_FALLING_DEADLY,
        )

        state = self._update_human(state, human_index, h_state=human_status)
        return state

    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(enemy: chex.Array, state: DefenderState) -> DefenderState:
            score = 0
            enemy_type = enemy[2]
            inactive_state = enemy
            lander_state = self._lander_movement(enemy, state)
            pod_state = self._pod_movement(enemy, state)
            bomber_state = self._bomber_movement(enemy, state)
            swarmers_state = self._swarmers_movement(enemy, state)
            mutant_state = self._mutant_movement(enemy, state)
            baiter_state = enemy
            delete_state, score = self._delete_enemy(enemy)

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
        state = state._replace(enemy_states=enemy_states_updated)

        return state


    def _human_step(self, state: DefenderState) -> DefenderState:
        def _human_falling(state: DefenderState, index: int) -> DefenderState:
            human = state.human_states[index]
            human_y = human[1]
            human_y += self.consts.HUMAN_FALLING_SPEED

            hit_ground = human_y >= self.consts.HUMAN_INIT_GAME_Y

            state = jax.lax.cond(
                jnp.logical_and(hit_ground, human[2] == self.consts.HUMAN_STATE_FALLING),
                lambda: self._add_score(state, self.consts.HUMAN_LIVING_FALL_SCORE),
                lambda: state,
            )

            state = jax.lax.cond(
                hit_ground,
                lambda: self._update_human(
                    state,
                    index,
                    10 + index * (self.consts.WORLD_WIDTH / self.consts.HUMAN_LEVEL_AMOUNT[state.level]),
                    self.consts.HUMAN_INIT_GAME_Y,
                    self.consts.HUMAN_STATE_IDLE,
                ),
                lambda: self._update_human(state, index, game_y=human_y),
            )

            return state

        def _human_caught(state: DefenderState, index: int) -> DefenderState:
            human = state.human_states[index]

            is_in_city = human[1] >= (self.consts.WORLD_HEIGHT - self.consts.CITY_HEIGHT - self.consts.HUMAN_HEIGHT)

            bring_back_human = state.bring_back_human + 1
            is_overdue = bring_back_human >= self.consts.HUMAN_BRING_BACK_FRAMES

            # It is either in the city and then overdue is not counted, or overdue, or nothing

            state = jax.lax.cond(
                is_in_city,
                lambda: self._add_score(state, self.consts.HUMAN_CAUGHT_AND_RETURNED_SCORE),
                lambda: state,
            )
            state = jax.lax.cond(
                jnp.logical_and(is_overdue, jnp.logical_not(is_in_city)),
                lambda: self._add_score(state, self.consts.HUMAN_CAUGHT_BUT_FORGOTTEN_SCORE),
                lambda: state,
            )

            state = jax.lax.cond(
                jnp.logical_or(is_in_city, is_overdue),
                lambda: self._update_human(
                    state,
                    index,
                    10 + index * (self.consts.WORLD_WIDTH / self.consts.HUMAN_LEVEL_AMOUNT[state.level]),
                    self.consts.HUMAN_INIT_GAME_Y,
                    self.consts.HUMAN_STATE_IDLE,
                ),
                lambda: self._update_human(state, index, state.space_ship_x + 5, state.space_ship_y + 4),
            )

            bring_back_human = jnp.where(jnp.logical_or(is_overdue, is_in_city), 0, bring_back_human)
            state = state._replace(bring_back_human=bring_back_human)
            return state

        def _human_move_switch(index, state: DefenderState) -> DefenderState:
            state = jax.lax.switch(
                jnp.array(state.human_states[index][2], int),
                [
                    lambda: state,  # Inactive
                    lambda: state,  # Idle
                    lambda: state,  # Abducted
                    lambda: _human_falling(state, index),  # Falling
                    lambda: _human_falling(state, index),  # Falling Deadly
                    lambda: _human_caught(state, index),  # Caught
                ],
            )

            return state

        def human_merge(state, states):
            idx = jnp.arange(self.consts.HUMAN_MAX_AMOUNT)
            new_humans = states.human_states[idx, idx]
            return state._replace(human_states=new_humans)

        state = human_merge(state, jax.vmap(_human_move_switch, in_axes=(0, None))(jnp.arange(self.consts.HUMAN_MAX_AMOUNT), state))

        return state

    ## -------- BULLET STEP ------------------------------

    def _bullet_spawn(self, state: DefenderState) -> DefenderState:
        # Spawns a bullet, call only if bullet is inactive
        enemy_indices, max_indice = self._enemies_on_screen(state)

        # Choose a random enemy
        chosen_one = enemy_indices[jrandom.randint(state.key, (), 0, max_indice, dtype=jnp.int32)]

        def _shoot(s_state: DefenderState):
            ss_onscreen_x, ss_onscreen_y = self.dh._onscreen_pos(s_state, s_state.space_ship_x, s_state.space_ship_y)

            e_x, e_y, e_type, _, _ = self._get_enemy(s_state, chosen_one)
            e_onscreen_x, e_onscreen_y = self.dh._onscreen_pos(s_state, e_x, e_y)

            def _bomber():
                b = self._update_enemy(s_state, chosen_one, arg1=self.consts.BOMB_TTL_FRAMES)
                return b._replace(bullet_dir_x=0.0, bullet_dir_y=0.0)

            def _normal():
                # aim at player
                dir_x = jnp.where((ss_onscreen_x - e_onscreen_x) < 0, -1.0, 1.0)
                dir_y = jnp.where((ss_onscreen_y - e_onscreen_y) < 0, -1.0, 1.0)
                dir_y *= jrandom.uniform(
                    state.key,
                    (),
                    float,
                    self.consts.BULLET_MIN_SPREAD,
                    self.consts.BULLET_MAX_SPREAD,
                )

                dir_vector = jnp.array([dir_x, dir_y])
                magnitude = jnp.linalg.norm(dir_vector)
                normalized_dir = dir_vector / magnitude
                return s_state._replace(bullet_dir_x=normalized_dir[0], bullet_dir_y=normalized_dir[1])

            s_state = jax.lax.cond(e_type == self.consts.BOMBER, lambda: _bomber(), lambda: _normal())

            bullet_x = e_x + self.consts.ENEMY_WIDTH / 2
            bullet_y = e_y + self.consts.ENEMY_WIDTH / 2

            return s_state._replace(
                bullet_x=bullet_x,
                bullet_y=bullet_y,
                bullet_active=True,
            )

        # Shoot the bullet if an enemy is onscreen
        return jax.lax.cond(max_indice > 0, lambda: _shoot(state), lambda: state)

    def _bullet_update(self, state: DefenderState) -> DefenderState:
        # Updates a bullet, call only if bullet state active
        b_x = state.bullet_x
        b_y = state.bullet_y
        b_dir_x = state.bullet_dir_x
        b_dir_y = state.bullet_dir_y

        # Update position
        speed_x = b_dir_x * self.consts.BULLET_SPEED
        speed_y = b_dir_y * self.consts.BULLET_SPEED
        b_x, b_y = self._move_with_space_ship(state, b_x, b_y, speed_x, speed_y, self.consts.BULLET_MOVE_WITH_SPACE_SHIP)

        # If it is a bomber, update its ttl
        is_bomber = jnp.logical_and(b_dir_x == 0.0, b_dir_y == 0.0)

        def _bomber():
            # Find bomber, there has to be one
            mask = (state.enemy_states[:, 2] == self.consts.BOMBER) & (state.enemy_states[:, 3] > 0.0)
            match = jnp.nonzero(mask, size=self.consts.BOMBER_MAX_AMOUNT, fill_value=-1)[0]
            enemy = state.enemy_states[match[0]]
            new_ttl = enemy[3] - 1

            return self._update_enemy(state, match[0], arg1=new_ttl)

        state = jax.lax.cond(is_bomber, lambda: _bomber(), lambda: state)

        return state._replace(bullet_x=b_x, bullet_y=b_y)

    def _bullet_check(self, state: DefenderState) -> DefenderState:
        # Check if its a bomber and if there is no bomber with ttl > 0
        b_x = state.bullet_x
        b_y = state.bullet_y
        b_dir_x = state.bullet_dir_x
        b_dir_y = state.bullet_dir_y

        is_bomber = jnp.logical_and(b_dir_x == 0.0, b_dir_y == 0.0)

        def _ttl_death():
            mask = (state.enemy_states[:, 2] == self.consts.BOMBER) & (state.enemy_states[:, 3] > 0.0)
            matches = jnp.nonzero(mask, size=self.consts.BOMBER_MAX_AMOUNT, fill_value=-1)[0]
            return matches[0] == -1

        is_ttl_death = jax.lax.cond(is_bomber, lambda: _ttl_death(), lambda: False)

        # Check if it is on_screen
        is_offscreen = jnp.logical_not(self.dh._is_onscreen_from_game(state, b_x, b_y, self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT))

        def _reset_ttl() -> DefenderState:
            mask = (state.enemy_states[2] == self.consts.BOMBER) & (state.enemy_states[3] > 0.0)

            def _update_bombers(state, idx):
                return jax.lax.cond(
                    idx != -1,
                    lambda: (self._update_enemy(state, idx, arg1=0.0), None),
                    lambda: (state, None),
                )

            matches = jnp.nonzero(mask, size=self.consts.BOMBER_MAX_AMOUNT, fill_value=-1)[0]
            return jax.lax.scan(_update_bombers, state, matches)[0]

        state = jax.lax.cond(
            jnp.logical_or(is_ttl_death, is_offscreen),
            lambda: _reset_ttl()._replace(bullet_active=False),
            lambda: state,
        )

        return state

    def _bullet_step(self, state: DefenderState) -> DefenderState:
        state = jax.lax.cond(state.bullet_active, lambda: state, lambda: self._bullet_spawn(state))
        # Update bullet is active
        state = jax.lax.cond(state.bullet_active, lambda: self._bullet_update(state), lambda: state)

        # Check for destruction
        state = jax.lax.cond(state.bullet_active, lambda: self._bullet_check(state), lambda: state)

        return state

    ## -------- LASER STEP ------------------------------

    def _check_laser(self, state: DefenderState) -> DefenderState:
        laser_x = state.laser_x
        laser_y = state.laser_y
        laser_width = self.consts.LASER_WIDTH
        laser_height = self.consts.LASER_HEIGHT
        laser_active = self.dh._is_onscreen_from_game(state, laser_x, laser_y, laser_width, laser_height)

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

    ## -------- COLLISION STEP ------------------------------

    def _is_colliding(self, e1_x, e1_y, e1_width, e1_height, e2_x, e2_y, e2_width, e2_height) -> chex.Array:
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

    def _check_space_ship_collision(self, state: DefenderState) -> DefenderState:
        is_colliding = jax.lax.cond(
            state.bullet_active,
            lambda: self._is_colliding(
                state.space_ship_x,
                state.space_ship_y,
                self.consts.SPACE_SHIP_WIDTH,
                self.consts.SPACE_SHIP_HEIGHT,
                state.bullet_x,
                state.bullet_y,
                self.consts.BULLET_WIDTH,
                self.consts.BULLET_HEIGHT,
            ),
            lambda: False,
        )
        state = jax.lax.cond(is_colliding, lambda: self._game_over(state), lambda: state)

        return state

    def _space_ship_catching_human(self, state: DefenderState) -> DefenderState:
        def check_human_collision(index, state: DefenderState) -> DefenderState:
            human = state.human_states[index]
            h_x = human[0]
            h_y = human[1]
            is_colliding = self._is_colliding(
                h_x,
                h_y,
                self.consts.HUMAN_WIDTH,
                self.consts.HUMAN_HEIGHT,
                state.space_ship_x,
                state.space_ship_y,
                self.consts.SPACE_SHIP_WIDTH,
                self.consts.SPACE_SHIP_HEIGHT,
            )

            def catch_human(state: DefenderState, index: int) -> DefenderState:
                state = self._update_human(state, index, h_state=self.consts.HUMAN_STATE_CAUGHT)
                bring_back_human = 0
                return state._replace(bring_back_human=bring_back_human)

            state = jax.lax.cond(is_colliding, lambda: catch_human(state, index), lambda: state)
            return state

        def check_for_falling(index, state):
            human = state.human_states[index]
            is_falling = jnp.logical_or(
                human[2] == self.consts.HUMAN_STATE_FALLING,
                human[2] == self.consts.HUMAN_STATE_FALLING_DEADLY,
            )
            state = jax.lax.cond(is_falling, lambda: check_human_collision(index, state), lambda: state)
            return state

        def merge_states(state, states):
            idx = jnp.arange(self.consts.HUMAN_MAX_AMOUNT)

            new_humans = states.human_states[idx, idx]

            return state._replace(human_states=new_humans)

        state = merge_states(state, jax.vmap(check_for_falling, in_axes=(0, None))(jnp.arange(self.consts.HUMAN_MAX_AMOUNT), state))

        return state

    def _check_enemy_collisions(self, state: DefenderState) -> DefenderState:
        def collision(index, state: DefenderState) -> DefenderState:
            e = state.enemy_states[index]
            e_x = e[0]
            e_y = e[1]
            # First check laser
            laser_is_colliding = jax.lax.cond(
                state.laser_active,
                lambda: self._is_colliding(
                    e_x,
                    e_y,
                    self.consts.ENEMY_WIDTH,
                    self.consts.ENEMY_HEIGHT,
                    state.laser_x,
                    state.laser_y,
                    self.consts.LASER_FINAL_WIDTH,
                    self.consts.LASER_FINAL_HEIGHT,
                ),
                lambda: False,
            )

            # Now check space ship
            space_ship_is_colliding = self._is_colliding(
                e_x,
                e_y,
                self.consts.ENEMY_WIDTH,
                self.consts.ENEMY_HEIGHT,
                state.space_ship_x,
                state.space_ship_y,
                self.consts.SPACE_SHIP_WIDTH,
                self.consts.SPACE_SHIP_HEIGHT,
            )

            state = jax.lax.cond(space_ship_is_colliding, lambda: self._game_over(state), lambda: state)

            is_dead = jnp.logical_or(laser_is_colliding, space_ship_is_colliding)
            state = jax.lax.cond(is_dead, lambda: self._delete_enemy_index(state, index), lambda: state)
            state = jax.lax.cond(
                laser_is_colliding,
                lambda: state._replace(laser_active=False),
                lambda: state,
            )
            return state

        def check_for_inactive(index, state):
            enemy = self._get_enemy(state, index)
            is_active = jnp.logical_and(enemy[2] != self.consts.INACTIVE, enemy[2] != self.consts.DEAD)
            state = jax.lax.cond(is_active, lambda: collision(index, state), lambda: state)
            return state

        # Vmap here is hard as game over gets called in function, changing whole state
        state = jax.lax.fori_loop(0, self.consts.ENEMY_MAX_IN_GAME, check_for_inactive, state)
        return state

    def _collision_step(self, state) -> DefenderState:
        # check space ship and bullet
        hyperspace = state.space_ship_y < (1 - self.consts.SPACE_SHIP_HEIGHT)
        state = jax.lax.cond(hyperspace, lambda: state, lambda: self._check_space_ship_collision(state))
        # check laser and enemies
        state = self._check_enemy_collisions(state)

        # check space ship and humans
        state = self._space_ship_catching_human(state)

        return state

    ## -------- GAME STATE CHANGERS ------------------------------

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

    def _game_over(self, state: DefenderState) -> DefenderState:
        # Clears human that died with ship
        def kill_human_that_was_on_ship(index, state: DefenderState) -> DefenderState:
            is_caught = state.human_states[index][2] == self.consts.HUMAN_STATE_CAUGHT
            state = jax.lax.cond(
                is_caught,
                lambda: self._update_human(state, index, 0.0, 0.0, self.consts.INACTIVE),
                lambda: state,
            )
            return state

        state = jax.lax.fori_loop(0, self.consts.HUMAN_MAX_AMOUNT, kill_human_that_was_on_ship, state)

        # Shows game over animation, subtracts a live
        # As game is over, use shooting_cooldown for animation index, to not waste state space
        state = state._replace(
            game_state=self.consts.GAME_STATE_GAMEOVER,
            space_ship_lives=state.space_ship_lives - 1,
            shooting_cooldown=0,
            bullet_active=False,
            laser_active=False,
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

    ## -------- GAME STEP AND RESET ------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DefenderState, action: chex.Array) -> Tuple[DefenderObservation, DefenderState, float, bool, DefenderInfo]:
        # Get all updated values from individual step functions
        previous_state = state

        # Randomness
        key, subkey = jax.random.split(state.key)
        state = state._replace(key=subkey)

        def state_playing(state) -> DefenderState:
            state = self._space_ship_step(state, action)
            state = self._camera_step(state)
            state = self._enemy_step(state)
            state = self._human_step(state)
            state = self._bullet_step(state)
            state = self._laser_step(state)
            state = self._collision_step(state)
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
            game_resume = jnp.logical_and(state.space_ship_lives > 0, self._get_done(state))
            state = jax.lax.cond(game_resume, lambda: self._reset_player(state), lambda: state)
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
            space_ship_facing_right=jnp.array(self.consts.SPACE_SHIP_INIT_FACE_RIGHT).astype(jnp.bool),
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

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "score": spaces.Box(low=0, high=self.consts.SCORE_MAX + 1, shape=(), dtype=jnp.int32),
                "player": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160 * 256, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210 * 256, shape=(), dtype=jnp.int32),
                        "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                    },
                ),
                "lives": spaces.Box(low=0, high=99, shape=(), dtype=jnp.int32),
                "enemy_states": spaces.Box(low=0, high=255, shape=(self.consts.ENEMY_MAX_IN_GAME, 5), dtype=jnp.int32),
                "human_states": spaces.Box(low=0, high=255, shape=(self.consts.HUMAN_MAX_AMOUNT, 3), dtype=jnp.int32),
                "bullet_active": spaces.Discrete(2),
                "bullet_x": spaces.Box(low=0, high=self.consts.WORLD_WIDTH, shape=(), dtype=jnp.int32),
                "bullet_y": spaces.Box(low=0, high=self.consts.WORLD_HEIGHT, shape=(), dtype=jnp.int32),
                "laser_active": spaces.Discrete(2),
                "laser_x": spaces.Box(low=0, high=self.consts.WORLD_WIDTH, shape=(), dtype=jnp.int32),
                "laser_y": spaces.Box(low=0, high=self.consts.WORLD_HEIGHT, shape=(), dtype=jnp.int32),
            }
        )

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
