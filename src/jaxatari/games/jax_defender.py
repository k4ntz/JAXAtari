import random
from turtle import left
from jax import random as jrandom
from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
from jax._src.source_info_util import current
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
import chex


class DefenderConstants(NamedTuple):
    # Game screen
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    GAME_WIDTH: int = 640
    GAME_HEIGHT: int = 135
    GAME_AREA_TOP: int = 38
    GAME_AREA_BOTTOM: int = GAME_AREA_TOP + GAME_HEIGHT

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

    # Bomber
    BOMBER_AMOUNT: Tuple[int, int, int, int, int] = (1, 2, 2, 2, 2)
    MAX_BOMBER_AMOUNT: int = 1
    BOMBER_Y_SPEED: float = -0.2
    BOMB_TTL_IN_SEC: float = 2  # Time to live

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

    # Space ship laser
    LASER_WIDTH: int = 30
    LASER_HEIGHT: int = 3
    LASER_SPEED: int = 2

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
            # x, y, type, bomb remaining time to live, ..
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
        self,
        state: DefenderState,
        game_x,
        game_y,
        width: int,
        height: int,
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

        self.ENEMY_MASKS = jnp.array(padded_masks)

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
                state.space_ship_x + state.camera_offset,
                self.consts.CITY_WIDTH,
            ),
            self.consts.CITY_WIDTH,
        )

        city_game_y = self.consts.GAME_HEIGHT - self.consts.CITY_HEIGHT
        city_screen_x, city_screen_y = self.dh._onscreen_pos(
            state, city_game_x, city_game_y
        )

        city_mask = self.SHAPE_MASKS["city"]
        raster = self.jr.render_at_clipped(
            raster,
            city_screen_x,
            city_screen_y,
            city_mask,
        )
        raster = self.jr.render_at_clipped(
            raster,
            city_screen_x + self.consts.CITY_WIDTH,
            city_screen_y,
            city_mask,
        )
        raster = self.jr.render_at_clipped(
            raster,
            city_screen_x - self.consts.CITY_WIDTH,
            city_screen_y,
            city_mask,
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

            enemy_type = enemy[2]

            mask = self.ENEMY_MASKS[jnp.array(enemy_type, int)]
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
                lambda ras: render_on_scanner(
                    enemy[0], enemy[1], scanner_width, scanner_height, color_id, ras
                ),
                lambda ras: ras,
                r,
            )

            # Render on screen and return new raster
            return jax.lax.cond(
                is_active_and_onscreen,
                lambda ras: self.jr.render_at_clipped(ras, screen_x, screen_y, mask),
                lambda ras: ras,
                r,
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

        raster = render_bullet(raster)

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

        return state._replace(
            space_ship_speed=space_ship_speed,
            space_ship_x=space_ship_x,
            space_ship_y=space_ship_y,
            space_ship_facing_right=space_ship_facing_right,
        )

    def _lander_movement(
        self, index: int, enemy_states: chex.Array, space_ship_speed: float
    ) -> chex.Array:
        lander = enemy_states[index]
        lander_x = lander[0]
        lander_y = lander[1]
        lander_state = lander[3]

        def lander_patrol(space_ship_speed: float) -> Tuple[float, float]:
            speed_x, speed_y = jax.lax.cond(
                space_ship_speed > 0,
                lambda: (-self.consts.ENEMY_SPEED, self.consts.LANDER_Y_SPEED),
                lambda: (self.consts.ENEMY_SPEED, self.consts.LANDER_Y_SPEED),
            )

            return speed_x, speed_y

        speed_x, speed_y = jax.lax.switch(
            jnp.array(lander_state, int),
            [
                lambda: lander_patrol(space_ship_speed),  # Patrol
                lambda: (0.0, 0.0),  # Descend
                lambda: (0.0, 0.0),  # Ascendss
            ],
        )

        x, y = self._move_and_wrap(
            lander_x,
            lander_y,
            speed_x + space_ship_speed * self.consts.SHIP_SPEED_INFLUENCE_ON_SPEED,
            speed_y,
        )
        new_lander = [x, y, lander[2], lander[3], lander[4]]
        enemy_states = enemy_states.at[index].set(new_lander)
        return enemy_states

    def _pod_movement(
        self, index: int, enemy_states: chex.Array, space_ship_speed: float
    ) -> chex.Array:
        pod = enemy_states[index]
        pod_x = pod[0]
        pod_y = pod[1]
        pod_state = pod[3]

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
        direction_right = bomber[3]

        speed_x = self.consts.ENEMY_SPEED
        # acceleration in x direction
        speed_x = jax.lax.cond(
            direction_right,
            lambda s: s,
            lambda s: -s,
            operand=speed_x,
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
        new_bomber = [x_pos, y_pos, bomber[2], direction_right, bomber[4]]
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
        speed_x = jax.lax.cond(
            crossed,
            lambda: 0.0,
            lambda: speed_x,
        )

        # Update direction based on current movement
        swarmer_direction_right = jax.lax.cond(
            speed_x != 0,
            lambda: speed_x > 0,
            lambda: swarmer_direction_right,
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
        game_x: float,
        game_y: float,
        enemy_type: float,
        arg1: float,
        arg2: float,
    ) -> DefenderState:
        enemy_state = state.enemy_states
        enemy_state = enemy_state.at[index].set(
            [game_x, game_y, enemy_type, arg1, arg2]
        )
        return state._replace(enemy_states=enemy_state)

    def _shoot_bullet(self, state: DefenderState, enemy_index) -> DefenderState:
        # get on screen pos to have wrap around functionality
        space_ship_onscreen_x, space_ship_onscreen_y = self.dh._onscreen_pos(
            state, state.space_ship_x, state.space_ship_y
        )

        enemy_states = state.enemy_states
        enemy = enemy_states[enemy_index]
        enemy_x = enemy[0]
        enemy_y = enemy[1]
        enemy_type = enemy[2]

        bullet_onscreen_x, bullet_onscreen_y = self.dh._onscreen_pos(
            state, enemy_x, enemy_y
        )

        # differentiate between bomber and others
        def bomber_shoot():
            bullet_dir_x = 0.0
            bullet_dir_y = 0.0
            return bullet_dir_x, bullet_dir_y

        def lander_shoot():
            # aim at player
            dir_x = jnp.where(
                (space_ship_onscreen_x - bullet_onscreen_x) < 0, -1.0, 1.0
            )
            dir_y = jnp.where(
                (space_ship_onscreen_y - bullet_onscreen_y) < 0, -1.0, 1.0
            )
            dir_y *= jrandom.uniform(state.key, (), float, 0.1, 0.3)

            dir_vector = jnp.array([dir_x, dir_y])
            magnitude = jnp.linalg.norm(dir_vector)

            normalized_dir = dir_vector / magnitude
            return normalized_dir[0], normalized_dir[1]

        dir_x, dir_y = jax.lax.cond(
            enemy_type == self.consts.BOMBER,
            lambda: bomber_shoot(),
            lambda: lander_shoot(),
        )

        # update if bomber has shot
        state = jax.lax.cond(
            enemy_type == self.consts.BOMBER,
            lambda: self._update_enemy(
                state,
                enemy_index,
                enemy_x,
                enemy_y,
                enemy_type,
                self.consts.BOMB_TTL_IN_SEC,
                0.0,
            ),
            lambda: state,
        )

        # spawn bullet from inside enemy, not topleft
        bullet_x = enemy_x + self.consts.ENEMY_WIDTH / 2
        bullet_y = enemy_y + self.consts.ENEMY_HEIGHT / 2

        # different types need to be added
        return state._replace(
            bullet_x=bullet_x,
            bullet_y=bullet_y,
            bullet_dir_x=dir_x,
            bullet_dir_y=dir_y,
            bullet_state=self.consts.BULLET_STATE_ACTIVE,
        )

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
            0,
            self.consts.ENEMY_MAX,
            add_indices,
            (state.enemy_states, indices, 0),
        )

        indices = result[1]
        max_indice = result[2]

        return (indices, max_indice)

    def _bullet_step(self, state: DefenderState) -> DefenderState:
        bullet_is_active = state.bullet_state == self.consts.BULLET_STATE_ACTIVE

        x = state.bullet_x
        y = state.bullet_y
        dir_x = state.bullet_dir_x
        dir_y = state.bullet_dir_y

        # Starts a shot
        def _start_shooting(enemy_indices_and_max):
            enemy_indices = enemy_indices_and_max[0]
            max_indice = enemy_indices_and_max[1]

            # Choose a random one
            p = jrandom.randint(state.key, (), 0, max_indice, dtype=jnp.int32)
            chosen = enemy_indices[p]

            return jax.lax.cond(
                max_indice > 0,
                lambda: self._shoot_bullet(state, chosen),
                lambda: state,
            )

        def _bomber_update():
            # find bomber
            enemy_states = state.enemy_states
            mask = (enemy_states[:, 2] == self.consts.BOMBER) & (enemy_states[:, 3] > 0)
            first = jnp.where(jnp.any(mask), jnp.argmax(mask), -1)
            exists = first != -1
            enemy = jax.lax.cond(
                exists,
                lambda: jnp.array(
                    [
                        enemy_states[first][0],
                        enemy_states[first][1],
                        enemy_states[first][2],
                        enemy_states[first][3] - 1,
                        enemy_states[first][4],
                    ]
                ),
                lambda: enemy_states[0],
            )

            return jax.lax.cond(
                exists,
                lambda: (
                    enemy[3] > 0,
                    self._update_enemy(
                        state, first, enemy[0], enemy[1], enemy[2], enemy[3], enemy[4]
                    ),
                ),
                lambda: (True, state),
            )

        # Updates a bullet
        def _bullet_update():
            speed_x = dir_x * self.consts.BULLET_SPEED
            speed_y = dir_y * self.consts.BULLET_SPEED
            new_x, new_y = self._move_with_space_ship(
                state, x, y, speed_x, speed_y, self.consts.BULLET_MOVE_WITH_SPACE_SHIP
            )
            is_onscreen = self.dh._is_onscreen_from_game(
                state, new_x, new_y, self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT
            )

            return new_x, new_y, is_onscreen

        # Update if active
        x, y, bullet_is_active = jax.lax.cond(
            bullet_is_active, _bullet_update, lambda: (x, y, False)
        )

        # Update bomber ttl if it is the one shooting
        bullet_is_active, state = jax.lax.cond(
            bullet_is_active,
            lambda: _bomber_update(),
            lambda: (bullet_is_active, state),
        )

        # If it is now inactive, it was inactive before or went out of screen, so spawn bullet
        return jax.lax.cond(
            bullet_is_active,
            lambda: state._replace(bullet_x=x, bullet_y=y),
            lambda: _start_shooting(self._enemies_on_screen(state)),
        )

    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(index: int, enemy_states):
            enemy = enemy_states[index]
            enemy_type = enemy[2]
            enemy_states = jax.lax.switch(
                jnp.array(enemy_type, int),
                [
                    lambda: enemy_states,
                    lambda: self._lander_movement(
                        index, enemy_states, state.space_ship_speed
                    ),
                    lambda: self._pod_movement(
                        index, enemy_states, state.space_ship_speed
                    ),
                    lambda: self._bomber_movement(
                        index, enemy_states, state.space_ship_speed, state.space_ship_x
                    ),
                    lambda: enemy_states,
                    lambda: enemy_states,
                    lambda: enemy_states,
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
            space_ship_x=jnp.array(self.consts.INITIAL_SPACE_SHIP_X).astype(
                jnp.float32
            ),
            space_ship_y=jnp.array(self.consts.INITIAL_SPACE_SHIP_Y).astype(
                jnp.float32
            ),
            space_ship_facing_right=jnp.array(
                self.consts.INITIAL_SPACE_SHIP_FACING_RIGHT, dtype=jnp.bool
            ),
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
        state = state._replace(step_counter=(state.step_counter + 1))

        # state = self._collision_step(new_state)
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
