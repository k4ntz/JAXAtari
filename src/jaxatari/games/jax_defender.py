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
from typing import Tuple, Any


class DefenderConstants(NamedTuple):
    # Game screen
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    GAME_WIDTH: int = 640
    GAME_HEIGHT: int = 135
    GAME_AREA_TOP: int = 38
    GAME_AREA_BOTTOM: int = GAME_AREA_TOP + GAME_HEIGHT
    GAME_TICK_PER_FRAME: float = 0.030

    # Camera
    CAMERA_SCREEN_X: int = 80
    CAMERA_OFFSET_MAX: int = 40
    CAMERA_OFFSET_GAIN: int = 2
    INITIAL_CAMERA_GAME_X: int = 240
    INITIAL_CAMERA_OFFSET: int = 40

    # UI
    CITY_WIDTH: int = 80
    CITY_HEIGHT: int = 13

    COLOR_SPACE_SHIP_BLUE: Tuple[int, int, int] = (132, 144, 252)
    COLOR_BOMBER_BLUE: Tuple[int, int, int] = (104, 116, 208)
    COLOR_LANDER_YELLOW: Tuple[int, int, int] = (252, 224, 140)
    COLOR_MUTANT_RED: Tuple[int, int, int] = (192, 104, 72)

    ENEMY_COLORS: list = [
        COLOR_SPACE_SHIP_BLUE,
        COLOR_LANDER_YELLOW,
        COLOR_LANDER_YELLOW,
        COLOR_BOMBER_BLUE,
        COLOR_LANDER_YELLOW,
        COLOR_MUTANT_RED,
        COLOR_BOMBER_BLUE,
    ]

    PARTICLE_COLOR_CYCLE: list = [
        COLOR_SPACE_SHIP_BLUE,
        COLOR_LANDER_YELLOW,
        COLOR_SPACE_SHIP_BLUE,
        COLOR_MUTANT_RED,
        COLOR_BOMBER_BLUE,
    ]
    PARTICLE_FLICKER_EVERY_N_FRAMES: int = 2

    # Space Ship
    SPACE_SHIP_ACCELERATION: float = 0.15
    SPACE_SHIP_BREAK: float = 0.1
    SPACE_SHIP_MAX_SPEED: float = 4.0
    SPACE_SHIP_WIDTH: int = 13
    SPACE_SHIP_HEIGHT: int = 5
    SPACE_SHIP_LIVES: int = 3
    SPACE_SHIP_BOMBS: int = 3

    # Enemy
    ACTIVE: int = 1
    INACTIVE: int = 0  # active, x_pos, y_pos
    ENEMY_SPEED: float = 0.24
    SHIP_SPEED_INFLUENCE_ON_SPEED: float = 0.4
    ENEMY_WIDTH: int = 13
    ENEMY_HEIGHT: int = 7
    ENEMY_MAX: int = 20
    ENEMY_MAX_ON_SCREEN: int = 5

    # Types
    INACTIVE: int = 0
    LANDER: int = 1
    POD: int = 2
    BOMBER: int = 3
    SWARMERS: int = 4
    MUTANT: int = 5
    BAITER: int = 6
    DEAD: int = 7  # To keep position to draw animation

    # Dead states, if enemy_type = dead, then arg1 decides color
    DEAD_YELLOW: int = 0
    DEAD_BLUE: int = 1
    DEAD_RED: int = 2

    # Bomber
    BOMBER_AMOUNT: Tuple[int, int, int, int, int] = (1, 2, 2, 2, 2)
    MAX_BOMBER_AMOUNT: int = 1
    BOMBER_Y_SPEED: float = -0.2
    BOMB_TTL_IN_SEC: float = 1.0  # Time to live

    # Lander
    LANDER_AMOUNT: Tuple[int, int, int, int, int] = (18, 18, 19, 20, 20)
    MAX_LANDER_AMOUNT: int = 5
    LANDER_Y_SPEED: float = 0.08
    LANDER_PICKUP_X_THRESHOLD: float = 2.0
    LANDER_START_Y: float = 10.0
    LANDER_PICKUP_DURATION_FRAMES: int = 120  # 4 seconds at 30 fps
    LANDER_STATE_PATROL: int = 0
    LANDER_STATE_DESCEND: int = 1
    LANDER_STATE_PICKUP: int = 2
    LANDER_STATE_ASCEND: int = 3

    # Pod
    POD_AMOUNT: Tuple[int, int, int, int, int] = (2, 2, 3, 3, 3)
    MAX_POD_AMOUNT: int = 3
    POD_Y_SPEED: float = 0.08

    # Baiter
    BAITER_TIME_SEC: int = 20
    SWARM_SPAWN_MIN: int = 1
    SWARM_SPAWN_MAX: int = 2

    # Enemy bullet
    BULLET_WIDTH: int = 2
    BULLET_HEIGHT: int = 2
    BULLET_SPEED: float = 3.0
    BULLET_MOVE_WITH_SPACE_SHIP: float = 0.9
    BULLET_STATE_INACTIVE: int = 0
    BULLET_STATE_ACTIVE: int = 1
    BULLET_MIN_SPREAD: float = 0.6
    BULLET_MAX_SPREAD: float = 0.3

    # Space ship laser
    # Width and height are for each laser
    # as the final laser is made out of 2
    LASER_WIDTH: int = 20
    LASER_HEIGHT: int = 1
    LASER_2ND_OFFSET: int = 7
    LASER_FINAL_WIDTH: int = LASER_WIDTH + LASER_2ND_OFFSET
    LASER_FINAL_HEIGHT: int = LASER_HEIGHT * 2
    LASER_SPEED: int = 20
    LASER_STATE_INACTIVE: int = 0
    LASER_STATE_ACTIVE: int = 1

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
            # x, y, type, bomb remaining time to live, is_facing_right
            [300, 50, BOMBER, 0, 0],
            # Inactives
            [100, 50, SWARMERS, 0, 0],
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

    # Humans
    # INACTIVE = 0
    HUMAN_STATE_IDLE: int = 1
    HUMAN_STATE_ABDUCTED: int = 2
    HUMAN_STATE_FALLING: int = 3
    HUMAN_AMOUNT: Tuple[int, int, int, int, int] = (5, 5, 6, 6, 6)
    HUMAN_MAX: int = 6
    HUMAN_WIDTH: int = 3
    HUMAN_HEIGHT: int = 3
    HUMAN_Y: int = GAME_HEIGHT - HUMAN_HEIGHT
    HUMAN_X: Tuple[float, float, float, float, float] = (
        GAME_WIDTH / HUMAN_AMOUNT[0],
        GAME_WIDTH / HUMAN_AMOUNT[1],
        GAME_WIDTH / HUMAN_AMOUNT[2],
        GAME_WIDTH / HUMAN_AMOUNT[3],
        GAME_WIDTH / HUMAN_AMOUNT[4],
    )

    INITIAL_HUMAN_STATES: chex.Array = jnp.array(
        [
            # x, y, state
            [0, HUMAN_Y, HUMAN_STATE_IDLE],
            [HUMAN_X[0], HUMAN_Y, HUMAN_STATE_IDLE],
            [HUMAN_X[0] * 2, HUMAN_Y, HUMAN_STATE_IDLE],
            [HUMAN_X[0] * 3, HUMAN_Y, HUMAN_STATE_IDLE],
            [HUMAN_X[0] * 4, HUMAN_Y, HUMAN_STATE_IDLE],
            [HUMAN_X[0] * 5, HUMAN_Y, INACTIVE],
        ]
    )

    # Scanner
    SCANNER_WIDTH: int = 64
    SCANNER_HEIGHT: int = 21
    SCANNER_SCREEN_Y: int = 13
    SCANNER_SCREEN_X: int = 48
    ENEMY_SCANNER_WIDTH: int = 2
    ENEMY_SCANNER_HEIGHT: int = 2
    SPACE_SHIP_SCANNER_WIDTH: int = 3
    SPACE_SHIP_SCANNER_HEIGHT: int = 2
    HUMAN_SCANNER_WIDTH: int = 1
    HUMAN_SCANNER_HEIGHT: int = 1


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
    # Laser
    laser_state: chex.Array
    laser_x: chex.Array
    laser_y: chex.Array
    laser_dir_x: chex.Array
    # Enemies
    enemy_states: chex.Array
    # Human
    human_states: chex.Array
    # Randomness
    key: chex.Array


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


# Helper class that gets implemented by renderer and game for shared functionality
class DefenderHelper:
    def __init__(self, consts: DefenderConstants):
        self.consts = consts
        return

    def _onscreen_pos(self, state: DefenderState, game_x, game_y):
        camera_screen_x = self.consts.CAMERA_SCREEN_X
        camera_game_x = state.space_ship_x + state.camera_offset
        camera_left_border = jnp.mod(
            camera_game_x - self.consts.SCREEN_WIDTH / 2, self.consts.GAME_WIDTH
        )

        camera_right_border = jnp.mod(
            camera_game_x + self.consts.SCREEN_WIDTH / 2, self.consts.GAME_WIDTH
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
                self.consts.GAME_WIDTH - camera_game_x + game_x + camera_screen_x
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

        lander_animatin_before_pad = self.SHAPE_MASKS["lander_pickup"]

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

        self.LANDER_MASKS = _create_padded_masks(self, lander_animatin_before_pad)
        self.LANDER_MASK_SIZE = self.LANDER_MASKS[0].shape

        self.DEATH_MASKS = jnp.array(
            [
                jnp.pad(
                    self.SHAPE_MASKS["death_yellow"],
                    ((0, 10), (0, 0)),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
                jnp.pad(
                    self.SHAPE_MASKS["death_blue"],
                    ((0, 10), (0, 0)),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
                jnp.pad(
                    self.SHAPE_MASKS["death_red"],
                    ((0, 10), (0, 0)),
                    mode="constant",
                    constant_values=self.jr.TRANSPARENT_ID,
                ),
            ],
            float,
        )

        # Enemy colors to ids
        color_ids = []
        for color in self.consts.ENEMY_COLORS:
            color_ids.append(self.COLOR_TO_ID.get(color, 0))

        self.ENEMY_COLOR_IDS = jnp.array(color_ids)

        # Particle colors to ids
        particle_ids = []
        for color in self.consts.PARTICLE_COLOR_CYCLE:
            particle_ids.append(self.COLOR_TO_ID.get(color, 0))

        self.PARTICLE_COLOR_IDS = jnp.array(particle_ids)

    def _get_asset_config(self) -> list:
        # Returns the declarative manifest of all assets for the game, including both wall sprites
        return [
            {"name": "background", "type": "background", "file": "ui_overlay.npy"},
            {"name": "space_ship", "type": "single", "file": "space_ship.npy"},
            {"name": "baiter", "type": "single", "file": "baiter.npy"},
            {"name": "bomber", "type": "single", "file": "bomber.npy"},
            {"name": "lander", "type": "single", "file": "lander.npy"},
            {
                "name": "lander_pickup",
                "type": "group",
                "files": [
                    "lander_0.npy",
                    "lander_1.npy",
                    "lander_2.npy",
                    "lander_3.npy",
                    "lander_4.npy",
                    "lander_5.npy",
                    "lander_6.npy",
                    "lander_7.npy",
                    "lander_8.npy",
                    "lander_9.npy",
                    "lander_10.npy",
                    "lander_11.npy",
                    "lander_12.npy",
                ],
            },
            {"name": "death_yellow", "type": "single", "file": "death_yellow.npy"},
            {"name": "death_blue", "type": "single", "file": "death_blue.npy"},
            {"name": "death_red", "type": "single", "file": "death_red.npy"},
            {"name": "mutant", "type": "single", "file": "mutant.npy"},
            {"name": "pod", "type": "single", "file": "pod.npy"},
            {"name": "swarmers", "type": "single", "file": "swarmers.npy"},
            {"name": "ui_overlay", "type": "single", "file": "ui_overlay.npy"},
            {"name": "city", "type": "single", "file": "city.npy"},
        ]

    def on_scanner_pos(self, state: DefenderState, game_x, game_y):
        camera_game_x = state.space_ship_x + state.camera_offset
        left_border = jnp.mod(
            camera_game_x - self.consts.GAME_WIDTH / 2, self.consts.GAME_WIDTH
        )

        # Calculate position inside scanner
        is_after_zero_wrap = game_x < left_border
        game_to_scanner_ratio_x = self.consts.SCANNER_WIDTH / self.consts.GAME_WIDTH
        game_to_scanner_ratio_y = self.consts.SCANNER_HEIGHT / self.consts.GAME_HEIGHT
        screen_x = jax.lax.cond(
            is_after_zero_wrap,
            lambda: (self.consts.GAME_WIDTH - left_border + game_x)
            * game_to_scanner_ratio_x,
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

        # Render City
        # Get the next city starting position to the left
        city_game_x = jnp.multiply(
            jnp.floor_divide(
                state.space_ship_x + state.camera_offset, self.consts.CITY_WIDTH
            ),
            self.consts.CITY_WIDTH,
        )

        city_game_y = self.consts.GAME_HEIGHT - self.consts.CITY_HEIGHT
        city_screen_x, city_screen_y = self.dh._onscreen_pos(
            state, city_game_x, city_game_y
        )

        city_mask = self.SHAPE_MASKS["city"]
        raster = self.jr.render_at_clipped(
            raster, city_screen_x, city_screen_y, city_mask
        )
        raster = self.jr.render_at_clipped(
            raster, city_screen_x + self.consts.CITY_WIDTH, city_screen_y, city_mask
        )
        raster = self.jr.render_at_clipped(
            raster, city_screen_x - self.consts.CITY_WIDTH, city_screen_y, city_mask
        )

        # TODO Render Score

        def render_on_scanner(game_x, game_y, width, height, color_id, r):
            scanner_x, scanner_y = self.on_scanner_pos(state, game_x, game_y)

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
                mask_index = jnp.clip(
                    jnp.floor_divide(enemy[4].astype(jnp.int32), 10), 0, 13
                )

                pickup_mask = self.LANDER_MASKS[mask_index + 1]
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

        # For each loop renders all enemys on screen and scanner
        raster = jax.lax.fori_loop(0, self.consts.ENEMY_MAX, render_enemy, raster)

        # Used for bullet color and human color
        current_particle_color_id = self.PARTICLE_COLOR_IDS[
            jnp.mod(
                jnp.floor_divide(
                    state.step_counter, self.consts.PARTICLE_FLICKER_EVERY_N_FRAMES
                ),
                len(self.PARTICLE_COLOR_IDS),
            )
        ]

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
                lambda ras: render_on_scanner(
                    human[0], human[1], scanner_width, scanner_height, color_id, ras
                ),
                lambda ras: ras,
                r,
            )

            # Render on screen and return the changed raster
            return jax.lax.cond(
                is_active_and_onscreen,
                lambda ras: self.jr.draw_rects(
                    ras,
                    jnp.asarray([[screen_x, screen_y]]),
                    jnp.asarray([[self.consts.HUMAN_WIDTH, self.consts.HUMAN_HEIGHT]]),
                    color_id,
                ),
                lambda ras: ras,
                r,
            )

        # For each loop renders all humans on screen and scanner
        raster = jax.lax.fori_loop(0, self.consts.HUMAN_MAX, render_human, raster)

        def render_space_ship(r):
            mask = self.SHAPE_MASKS["space_ship"]
            game_x = state.space_ship_x
            game_y = state.space_ship_y

            screen_x, screen_y = self.dh._onscreen_pos(state, game_x, game_y)

            facing_right = jnp.where(state.space_ship_facing_right, False, True)

            scanner_width = self.consts.SPACE_SHIP_SCANNER_WIDTH
            scanner_height = self.consts.SPACE_SHIP_SCANNER_HEIGHT

            color_id = self.ENEMY_COLOR_IDS[0]

            # Render on scanner
            r = render_on_scanner(
                game_x, game_y, scanner_width, scanner_height, color_id, r
            )

            # Render on screen and return
            return self.jr.render_at(
                r, screen_x, screen_y, mask, flip_horizontal=facing_right
            )

        raster = render_space_ship(raster)

        # Render laser
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
            state.laser_state == self.consts.LASER_STATE_ACTIVE,
            lambda: render_laser(raster),
            lambda: raster,
        )

        # Render bullet
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
            state.bullet_state == self.consts.BULLET_STATE_ACTIVE,
            lambda: render_bullet(raster),
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

    def _spawn_enemy(
        self, state: DefenderState, game_x, game_y, e_type, arg1, arg2
    ) -> DefenderState:
        # Find first enemy that is inactive
        mask = jnp.array(state.enemy_states[:, 2] == self.consts.INACTIVE)
        match = mask.argmax()
        # If no open slot availabe, dismiss new enemy
        open_slot_available = jnp.logical_or(match != 0, mask[0] == True)
        state = jax.lax.cond(
            open_slot_available,
            lambda: self._update_enemy(
                state, match, game_x, game_y, e_type, arg1, arg2
            ),
            lambda: state,
        )
        return state

    def _delete_enemy(self, state: DefenderState, index) -> DefenderState:
        is_index = jnp.logical_and(index > 0, index < self.consts.ENEMY_MAX)
        enemy_type = self._get_enemy(state, index)[2].astype(int)
        is_dead = enemy_type == self.consts.DEAD
        new_type = jax.lax.cond(
            is_dead, lambda: self.consts.INACTIVE, lambda: self.consts.DEAD
        )
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
        state = jax.lax.cond(
            is_index,
            lambda: self._update_enemy(
                state, index, enemy_type=new_type, arg1=color, arg2=0.0
            ),
            lambda: state,
        )
        # TODO Add score
        return state

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

    # Wrap function, returns wrapped position
    def wrap_pos(self, game_x: float, game_y: float):
        return game_x % self.consts.GAME_WIDTH, game_y % (
            self.consts.GAME_HEIGHT
            - self.consts.CITY_HEIGHT  # move already when the top of the city == top of entity
        )

    def _move(
        self, game_x: float, game_y: float, x_speed: float, y_speed: float
    ) -> Tuple[float, float]:
        new_game_x = game_x + x_speed
        new_game_y = game_y + y_speed
        # Wrap only around x, y not needed
        new_game_x, _ = self.wrap_pos(new_game_x, 0)
        return new_game_x, new_game_y

    def _move_and_wrap(
        self, game_x: float, game_y: float, x_speed: float, y_speed: float
    ) -> Tuple[float, float]:
        new_game_x, new_game_y = self._move(game_x, game_y, x_speed, y_speed)
        new_game_x, new_game_y = self.wrap_pos(new_game_x, new_game_y)
        return new_game_x, new_game_y

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
        return state._replace(
            laser_x=laser_x,
            laser_y=laser_y,
            laser_dir_x=laser_dir_x,
            laser_state=self.consts.LASER_STATE_ACTIVE,
        )

    def _check_laser(self, state: DefenderState) -> DefenderState:
        laser_x = state.laser_x
        laser_y = state.laser_y
        laser_width = self.consts.LASER_WIDTH
        laser_height = self.consts.LASER_HEIGHT
        is_onscreen = self.dh._is_onscreen_from_game(
            state, laser_x, laser_y, laser_width, laser_height
        )
        laser_state = jax.lax.cond(
            is_onscreen,
            lambda: self.consts.LASER_STATE_ACTIVE,
            lambda: self.consts.LASER_STATE_INACTIVE,
        )

        return state._replace(laser_state=laser_state)

    def _laser_update(self, state: DefenderState) -> DefenderState:
        laser_x = state.laser_x
        laser_y = state.laser_y
        laser_speed_x = state.laser_dir_x * self.consts.LASER_SPEED
        laser_x, laser_y = self._move_and_wrap(laser_x, laser_y, laser_speed_x, 0.0)
        return state._replace(laser_x=laser_x, laser_y=laser_y)

    def _laser_step(self, state: DefenderState) -> DefenderState:
        laser_active = state.laser_state == self.consts.LASER_STATE_ACTIVE
        state = jax.lax.cond(
            laser_active, lambda: self._laser_update(state), lambda: state
        )
        state = jax.lax.cond(
            laser_active, lambda: self._check_laser(state), lambda: state
        )
        return state

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

        shoot = jnp.logical_and(
            shoot, state.laser_state == self.consts.LASER_STATE_INACTIVE
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
            lambda: state.space_ship_speed
            + direction_x * self.consts.SPACE_SHIP_ACCELERATION,
            lambda: state.space_ship_speed
            - state.space_ship_speed * self.consts.SPACE_SHIP_BREAK,
        )

        space_ship_speed = jnp.clip(
            space_ship_speed,
            -self.consts.SPACE_SHIP_MAX_SPEED,
            self.consts.SPACE_SHIP_MAX_SPEED,
        )

        space_ship_stopping_deadzone = 0.0001

        space_ship_speed = jnp.where(
            jnp.abs(space_ship_speed) <= space_ship_stopping_deadzone,
            0,
            space_ship_speed,
        )

        space_ship_x = state.space_ship_x
        space_ship_y = state.space_ship_y

        x_speed = space_ship_speed
        y_speed = direction_y
        space_ship_x, space_ship_y = self._move(
            space_ship_x, space_ship_y, x_speed, y_speed
        )

        state = state._replace(
            space_ship_speed=space_ship_speed,
            space_ship_x=space_ship_x,
            space_ship_y=space_ship_y,
            space_ship_facing_right=space_ship_facing_right,
        )

        state = jax.lax.cond(shoot, lambda: self._shoot_laser(state), lambda: state)

        return state

    def _lander_movement(
        self,
        index: int,
        enemy_states: chex.Array,
        space_ship_speed: float,
        human_states: chex.Array,
        state: DefenderState,
    ) -> chex.Array:
        lander = enemy_states[index]
        lander_x = lander[0]
        lander_y = lander[1]
        lander_state = lander[3]
        current_counter = lander[4]

        new_human_states = human_states

        def check_proximity(human_state: chex.Array) -> chex.Array:
            on_screen_lander_x, _ = self.dh._onscreen_pos(state, lander_x, lander_y)
            on_screen_human_x, _ = self.dh._onscreen_pos(
                state, human_state[0], human_state[1]
            )

            return jnp.logical_and(
                jnp.abs(on_screen_human_x - on_screen_lander_x - 5)
                < self.consts.LANDER_PICKUP_X_THRESHOLD,
                human_state[2] == self.consts.HUMAN_STATE_IDLE,
            )

        def lander_patrol(space_ship_speed: float) -> Tuple[float, float]:
            speed_x, speed_y = jax.lax.cond(
                space_ship_speed > 0,
                lambda: (-self.consts.ENEMY_SPEED, self.consts.LANDER_Y_SPEED),
                lambda: (self.consts.ENEMY_SPEED, self.consts.LANDER_Y_SPEED),
            )

            speed_x += space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED
            # check if on top of human to switch to descend

            proximity_checks = jax.vmap(check_proximity)(human_states)
            is_near_human = jnp.any(proximity_checks)
            # Check if any other lander (not this one) is already descending or picking up
            indices = jnp.arange(self.consts.ENEMY_MAX)
            other_landers_descending = jnp.any(
                jnp.logical_and(
                    jnp.logical_and(
                        indices != index, enemy_states[:, 2] == self.consts.LANDER
                    ),
                    jnp.logical_or(
                        enemy_states[:, 3] == self.consts.LANDER_STATE_DESCEND,
                        enemy_states[:, 3] == self.consts.LANDER_STATE_PICKUP,
                    ),
                )
            )

            lander_state = jax.lax.cond(
                jnp.logical_and(
                    is_near_human, jnp.logical_not(other_landers_descending)
                ),
                lambda: self.consts.LANDER_STATE_DESCEND,
                lambda: self.consts.LANDER_STATE_PATROL,
            )

            return speed_x, speed_y, lander_state, 0.0

        def lander_descend(_: float) -> Tuple[float, float]:
            speed_x = 0.0
            speed_y = self.consts.LANDER_Y_SPEED * 5

            # Check if lander reached the bottom (human level)
            lander_state = jax.lax.cond(
                lander_y >= 115,
                lambda: self.consts.LANDER_STATE_PICKUP,
                lambda: self.consts.LANDER_STATE_DESCEND,
            )

            return speed_x, speed_y, lander_state, 0.0

        def lander_pickup(_: float, current_counter: float) -> Tuple[float, float]:
            speed_x = 0.0
            speed_y = 0.0
            current_counter += 1.0
            lander_state = self.consts.LANDER_STATE_PICKUP

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
            return speed_x, speed_y, lander_state, current_counter

        def lander_ascend(
            human_id: float,
        ) -> Tuple[float, float]:
            speed_x = 0.0
            speed_y = -self.consts.LANDER_Y_SPEED

            def lander_reached_top(
                human_index: int,
            ) -> Tuple[float, float, float, float]:
                # Mark human as rescued
                human = human_states[human_index]
                human_x = human[0]
                human_y = human[1]
                new_human = [
                    human_x,
                    human_y,
                    self.consts.INACTIVE,
                ]
                new_human_states = human_states.at[human_index].set(new_human)
                return self.consts.LANDER_STATE_PATROL, new_human_states

            def lander_ascend_continue(
                human_index: int, lander_y: float
            ) -> Tuple[float, float, float, float]:

                human = human_states[human_index]
                # jax.debug.print("Lander ascending, lander_y: {}, human_index: {}, human state: {}", lander_y, human_index, human)
                human_x = human[0]
                human_y = lander_y + 5  # Move human up with lander
                new_human = [
                    human_x,
                    human_y,
                    self.consts.HUMAN_STATE_ABDUCTED,
                ]
                new_human_states = human_states.at[human_index].set(new_human)
                return self.consts.LANDER_STATE_ASCEND, new_human_states

            # Check if lander reached the top
            lander_state, new_human_states = jax.lax.cond(
                lander_y <= self.consts.LANDER_START_Y,
                lambda: lander_reached_top(jnp.array(human_id, int)),
                lambda: lander_ascend_continue(jnp.array(human_id, int), lander_y),
            )

            return speed_x, speed_y, lander_state, human_id, new_human_states

        counter_id = lander[4]
        speed_x, speed_y, lander_state, counter_id, new_human_states = jax.lax.switch(
            jnp.array(lander_state, int),
            [
                lambda: (*lander_patrol(space_ship_speed), new_human_states),  # Patrol
                lambda: (
                    *lander_descend(space_ship_speed),
                    new_human_states,
                ),  # Descend
                lambda: (
                    *lander_pickup(space_ship_speed, current_counter),
                    new_human_states,
                ),  # Pickup
                lambda: lander_ascend(counter_id),  # Ascend
            ],
        )

        x, y = self._move_and_wrap(lander_x, lander_y, speed_x, speed_y)
        new_lander = [x, y, lander[2], lander_state, counter_id]
        enemy_states = enemy_states.at[index].set(new_lander)
        return enemy_states, new_human_states

    def _pod_movement(
        self, index: int, enemy_states: chex.Array, space_ship_speed: float
    ) -> chex.Array:
        pod = enemy_states[index]
        pod_x = pod[0]
        pod_y = pod[1]

        speed_x, speed_y = jax.lax.cond(
            space_ship_speed > 0,
            lambda: (-self.consts.ENEMY_SPEED, self.consts.POD_Y_SPEED),
            lambda: (self.consts.ENEMY_SPEED, self.consts.POD_Y_SPEED),
        )

        x, y = self._move_and_wrap(
            pod_x,
            pod_y,
            speed_x + space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            speed_y,
        )
        new_pod = [x, y, pod[2], pod[3], pod[4]]
        enemy_states = enemy_states.at[index].set(new_pod)
        return enemy_states

    def _bomber_movement(
        self,
        index: int,
        enemy_states: chex.Array,
        space_ship_speed: float,
        space_ship_x: float,
    ) -> chex.Array:
        bomber = enemy_states[index]
        x_pos = bomber[0]
        y_pos = bomber[1]
        direction_right = bomber[4]

        speed_x = self.consts.ENEMY_SPEED
        # acceleration in x direction
        speed_x = jax.lax.cond(
            direction_right, lambda s: s, lambda s: -s, operand=speed_x
        )

        # change direction if spaceship is crossed and passed by 30
        direction_right = jax.lax.cond(
            jnp.logical_and(direction_right, x_pos > space_ship_x + 30),
            lambda _: 0.0,
            lambda _: jax.lax.cond(
                jnp.logical_and(
                    jnp.logical_not(direction_right), x_pos < space_ship_x - 30
                ),
                lambda _: 1.0,
                lambda _: direction_right,
                operand=None,
            ),
            operand=None,
        )
        x_pos, y_pos = self._move_and_wrap(
            x_pos,
            y_pos,
            speed_x + space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            self.consts.BOMBER_Y_SPEED,
        )
        new_bomber = [x_pos, y_pos, bomber[2], bomber[3], direction_right]
        enemy_states = enemy_states.at[index].set(new_bomber)
        return enemy_states

    def _swarmers_movement(
        self, index: int, enemy_states: chex.Array, space_ship_speed: float
    ) -> chex.Array:
        swarmer = enemy_states[index]
        x_pos = swarmer[0]
        y_pos = swarmer[1]
        swarmer_direction_right = swarmer[3]

        # Swarmers move opposite to spaceship direction
        speed_x = jax.lax.cond(
            space_ship_speed > 0,
            lambda: self.consts.ENEMY_SPEED,  # Spaceship going right, swarmer goes right
            lambda: -self.consts.ENEMY_SPEED,  # Spaceship going left, swarmer goes left
        )

        # Check if swarmer crossed the spaceship
        crossed = jax.lax.cond(
            swarmer_direction_right,
            lambda: x_pos > state.space_ship_x,
            lambda: x_pos < state.space_ship_x,
        )

        # If crossed, set speed_x to 0
        speed_x = jax.lax.cond(crossed, lambda: 0.0, lambda: speed_x)

        # Update direction based on current movement
        swarmer_direction_right = jax.lax.cond(
            speed_x != 0, lambda: speed_x > 0, lambda: swarmer_direction_right
        )

        # Fixed Y speed for swarmers
        speed_y = self.consts.ENEMY_SPEED

        x_pos, y_pos = self._move_and_wrap(
            x_pos,
            y_pos,
            speed_x + space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            speed_y,
        )

        new_swarmer = [x_pos, y_pos, swarmer[2], swarmer_direction_right, swarmer[4]]
        enemy_states = enemy_states.at[index].set(new_swarmer)
        return enemy_states

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

        enemy_state = enemy_state.at[index].set(
            [game_x, game_y, enemy_type, arg1, arg2]
        )
        return state._replace(enemy_states=enemy_state)

    def _get_enemy(self, state: DefenderState, index):
        # Returns the enemy list at index
        return state.enemy_states[index]

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
                lambda: self.dh._is_onscreen_from_game(
                    state, x, y, self.consts.ENEMY_WIDTH, self.consts.ENEMY_HEIGHT
                ),
                lambda: False,
            )

            # Pass -1 for non on screen enemies, array has to be static size
            indices = indices.at[enemy_count_in_indices].set(
                jax.lax.cond(is_onscreen, lambda: i, lambda: -1)
            )
            enemy_count_in_indices += is_onscreen

            return (enemy_states, indices, enemy_count_in_indices)

        result = jax.lax.fori_loop(
            0, self.consts.ENEMY_MAX, add_indices, (state.enemy_states, indices, 0)
        )

        indices = result[1]
        max_indice = result[2]

        return (indices, max_indice)

    def _bullet_spawn(self, state: DefenderState) -> DefenderState:
        # Spawns a bullet, call only if bullet is inactive
        enemy_indices, max_indice = self._enemies_on_screen(state)

        # Choose a random enemy
        chosen_one = enemy_indices[
            jrandom.randint(state.key, (), 0, max_indice, dtype=jnp.int32)
        ]

        def _shoot(s_state: DefenderState):
            ss_onscreen_x, ss_onscreen_y = self.dh._onscreen_pos(
                s_state, s_state.space_ship_x, s_state.space_ship_y
            )

            e_x, e_y, e_type, _, _ = self._get_enemy(s_state, chosen_one)
            e_onscreen_x, e_onscreen_y = self.dh._onscreen_pos(s_state, e_x, e_y)

            def _bomber():
                b = self._update_enemy(
                    s_state, chosen_one, arg1=self.consts.BOMB_TTL_IN_SEC
                )
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
                return s_state._replace(
                    bullet_dir_x=normalized_dir[0], bullet_dir_y=normalized_dir[1]
                )

            s_state = jax.lax.cond(
                e_type == self.consts.BOMBER, lambda: _bomber(), lambda: _normal()
            )

            bullet_x = e_x + self.consts.ENEMY_WIDTH / 2
            bullet_y = e_y + self.consts.ENEMY_WIDTH / 2

            return s_state._replace(
                bullet_x=bullet_x,
                bullet_y=bullet_y,
                bullet_state=self.consts.BULLET_STATE_ACTIVE,
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
        b_x, b_y = self._move_with_space_ship(
            state, b_x, b_y, speed_x, speed_y, self.consts.BULLET_MOVE_WITH_SPACE_SHIP
        )

        # If it is a bomber, update its ttl
        is_bomber = jnp.logical_and(b_dir_x == 0.0, b_dir_y == 0.0)

        def _bomber():
            # Find bomber, there has to be one
            mask = (state.enemy_states[:, 2] == self.consts.BOMBER) & (
                state.enemy_states[:, 3] > 0.0
            )
            match = jnp.nonzero(
                mask, size=self.consts.MAX_BOMBER_AMOUNT, fill_value=-1
            )[0]
            enemy = state.enemy_states[match[0]]
            new_ttl = enemy[3] - self.consts.GAME_TICK_PER_FRAME

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
            mask = (state.enemy_states[:, 2] == self.consts.BOMBER) & (
                state.enemy_states[:, 3] > 0.0
            )
            matches = jnp.nonzero(
                mask, size=self.consts.MAX_BOMBER_AMOUNT, fill_value=-1
            )[0]
            return matches[0] == -1

        is_ttl_death = jax.lax.cond(is_bomber, lambda: _ttl_death(), lambda: False)

        # Check if it is on_screen
        is_offscreen = jnp.logical_not(
            self.dh._is_onscreen_from_game(
                state, b_x, b_y, self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT
            )
        )

        def _reset_ttl() -> DefenderState:
            mask = (state.enemy_states[2] == self.consts.BOMBER) & (
                state.enemy_states[3] > 0.0
            )

            def _update_bombers(state, idx):
                return jax.lax.cond(
                    idx != -1,
                    lambda: (self._update_enemy(state, idx, arg1=0.0), None),
                    lambda: (state, None),
                )

            matches = jnp.nonzero(
                mask, size=self.consts.MAX_BOMBER_AMOUNT, fill_value=-1
            )[0]
            return jax.lax.scan(_update_bombers, state, matches)[0]

        state = jax.lax.cond(
            jnp.logical_or(is_ttl_death, is_offscreen),
            lambda: _reset_ttl()._replace(
                bullet_state=self.consts.BULLET_STATE_INACTIVE
            ),
            lambda: state,
        )

        return state

    def _bullet_step(self, state: DefenderState) -> DefenderState:
        bullet_is_active = state.bullet_state == self.consts.BULLET_STATE_ACTIVE
        state = jax.lax.cond(
            bullet_is_active, lambda: state, lambda: self._bullet_spawn(state)
        )
        # Update bullet is active
        bullet_is_active = state.bullet_state == self.consts.BULLET_STATE_ACTIVE
        state = jax.lax.cond(
            bullet_is_active, lambda: self._bullet_update(state), lambda: state
        )

        # Check for destruction
        state = jax.lax.cond(
            bullet_is_active, lambda: self._bullet_check(state), lambda: state
        )

        return state

    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(index: int, enemy_states: chex.Array):
            enemy_states, human_states = enemy_states
            enemy = enemy_states[index]
            enemy_type = enemy[2]
            enemy_states, human_states = jax.lax.switch(
                jnp.array(enemy_type, int),
                [
                    lambda: (enemy_states, human_states),
                    lambda: self._lander_movement(
                        index,
                        enemy_states,
                        state.space_ship_speed,
                        human_states,
                        state,
                    ),
                    lambda: (
                        self._pod_movement(index, enemy_states, state.space_ship_speed),
                        human_states,
                    ),
                    lambda: (
                        self._bomber_movement(
                            index,
                            enemy_states,
                            state.space_ship_speed,
                            state.space_ship_x,
                        ),
                        human_states,
                    ),
                    lambda: (enemy_states, human_states),
                    lambda: (enemy_states, human_states),
                    lambda: (enemy_states, human_states),
                    lambda: (
                        self._delete_enemy(state, index).enemy_states,
                        human_states,
                    ),
                ],
            )

            return enemy_states, human_states

        enemy_states = state.enemy_states
        human_states = state.human_states
        enemy_states, human_states = jax.lax.fori_loop(
            0, self.consts.ENEMY_MAX, _enemy_move_switch, (enemy_states, human_states)
        )

        return state._replace(enemy_states=enemy_states, human_states=human_states)

    def _camera_step(self, state: DefenderState) -> DefenderState:
        # Returns: camera_offset
        offset_gain = self.consts.CAMERA_OFFSET_GAIN
        camera_offset = state.camera_offset
        camera_offset += jnp.where(state.space_ship_facing_right, 1, -1) * offset_gain

        camera_offset = jnp.clip(
            camera_offset, -self.consts.CAMERA_OFFSET_MAX, self.consts.CAMERA_OFFSET_MAX
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
            space_ship_x=jnp.array(self.consts.INITIAL_SPACE_SHIP_X).astype(
                jnp.float32
            ),
            space_ship_y=jnp.array(self.consts.INITIAL_SPACE_SHIP_Y).astype(
                jnp.float32
            ),
            space_ship_facing_right=jnp.array(
                self.consts.INITIAL_SPACE_SHIP_FACING_RIGHT, dtype=jnp.bool
            ),
            # Laser
            laser_state=jnp.array(0).astype(jnp.int32),
            laser_x=jnp.array(0).astype(jnp.float32),
            laser_y=jnp.array(0).astype(jnp.float32),
            laser_dir_x=jnp.array(0).astype(jnp.float32),
            # Bullet
            bullet_state=jnp.array(0).astype(jnp.int32),
            bullet_x=jnp.array(0).astype(jnp.float32),
            bullet_y=jnp.array(0).astype(jnp.float32),
            bullet_dir_x=jnp.array(0).astype(jnp.float32),
            bullet_dir_y=jnp.array(0).astype(jnp.float32),
            # Enemies
            enemy_states=jnp.array(self.consts.INITIAL_ENEMY_STATES).astype(
                jnp.float32
            ),
            # Humans
            human_states=jnp.array(self.consts.INITIAL_HUMAN_STATES).astype(
                jnp.float32
            ),
            # Randomness
            key=jnp.array(jax.random.PRNGKey(0)),
        )
        observation = self._get_observation(initial_state)
        return observation, initial_state

    def _check_space_ship_collision(self, state: DefenderState) -> DefenderState:
        is_colliding = jax.lax.cond(
            state.bullet_state == self.consts.BULLET_STATE_ACTIVE,
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
        # TODO implement game over here
        state = jax.lax.cond(is_colliding, lambda: state, lambda: state)

        return state

    def _check_enemy_collisions(self, state: DefenderState) -> DefenderState:
        def collision(index, state):
            e = state.enemy_states[index]
            e_x = e[0]
            e_y = e[1]
            # First check laser
            laser_is_colliding = jax.lax.cond(
                state.laser_state == self.consts.LASER_STATE_ACTIVE,
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

            # TODO implement game over here, if space ship is colliding
            is_dead = jnp.logical_or(laser_is_colliding, space_ship_is_colliding)
            state = jax.lax.cond(
                is_dead, lambda: self._delete_enemy(state, index), lambda: state
            )
            state = jax.lax.cond(
                laser_is_colliding,
                lambda: state._replace(laser_state=self.consts.LASER_STATE_INACTIVE),
                lambda: state,
            )
            return state

        def check_for_inactive(index, state):
            enemy = self._get_enemy(state, index)
            is_active = jnp.logical_and(
                enemy[2] != self.consts.INACTIVE, enemy[2] != self.consts.DEAD
            )
            state = jax.lax.cond(
                is_active, lambda: collision(index, state), lambda: state
            )
            return state

        state = jax.lax.fori_loop(0, self.consts.ENEMY_MAX, check_for_inactive, state)
        return state

    def _collision_step(self, state) -> DefenderState:
        # check player and bullet
        state = self._check_space_ship_collision(state)
        # check laser and enemies
        state = self._check_enemy_collisions(state)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: DefenderState, action: chex.Array
    ) -> Tuple[DefenderObservation, DefenderState, float, bool, DefenderInfo]:
        # Get all updated values from individual step functions
        previous_state = state

        # Randomness
        key, subkey = jax.random.split(state.key)
        state = state._replace(key=subkey)

        state = self._space_ship_step(state, action)
        state = self._camera_step(state)
        state = self._enemy_step(state)
        state = self._bullet_step(state)
        state = self._laser_step(state)
        state = self._collision_step(state)
        state = state._replace(step_counter=(state.step_counter + 1))

        observation = self._get_observation(state)
        env_reward = self._get_reward(previous_state, state)
        done = self._get_done(state)
        info = self._get_info(state)

        # Swap key to parent key
        state = state._replace(key=key)
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
