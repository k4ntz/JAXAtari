
from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import random
import os
from sys import maxsize
import numpy as np

from jaxatari.environment import JaxEnvironment

NOOP = 0
LEFT = 1
RIGHT = 2
JUMP = 3  # New action for jumping

BOTTOM_BORDER = 176
TOP_BORDER = 23


@dataclass
class GameConfig:
    """Game configuration parameters"""
    screen_width: int = 160
    screen_height: int = 210
    skier_width: int = 10
    skier_height: int = 18
    skier_y: int = 40
    flag_width: int = 5
    flag_height: int = 14
    flag_distance: int = 20
    tree_width: int = 16
    tree_height: int = 30
    rock_width: int = 16
    rock_height: int = 7
    max_num_flags: int = 2
    max_num_trees: int = 4
    max_num_rocks: int = 3
    speed: float = 1.0
    jump_duration: int = 30  # Duration of jump in frames (adjust if needed)
    jump_scale_factor: float = 1.5  # Maximum size increase during jump


class GameState(NamedTuple):
    """Represents the current state of the game"""
    skier_x: chex.Array
    skier_pos: chex.Array  # --> --_  \   |   |   /  _-- <-- States are doubles in ALE
    skier_fell: chex.Array
    skier_x_speed: chex.Array
    skier_y_speed: chex.Array
    flags: chex.Array
    trees: chex.Array
    rocks: chex.Array
    score: chex.Array
    time: chex.Array
    direction_change_counter: chex.Array
    game_over: chex.Array
    key: chex.Array
    jumping: chex.Array  # Is the skier currently jumping?
    jump_timer: chex.Array  # Frames left in current jump
    collision_type: chex.Array  # 0 = keine, 1 = Baum, 2 = Stein, 3 = Flagge

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class SkiingObservation(NamedTuple):
    skier: EntityPosition
    flags: jnp.ndarray
    trees: jnp.ndarray
    rocks: jnp.ndarray
    score: jnp.ndarray
    jumping: jnp.ndarray
    jump_timer: jnp.ndarray


class SkiingInfo(NamedTuple):
    time: jnp.ndarray


class JaxSkiing(JaxEnvironment[GameState, SkiingObservation, SkiingInfo]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()
        
    def _npy_to_surface(self, npy_path, width, height):
        arr = np.load(npy_path)  # Erwartet (H, W, 4) RGBA
        arr = arr.astype(np.uint8)
        surf = pygame.Surface((arr.shape[1], arr.shape[0]), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surf)[:, :, :] = arr[..., :3]
        pygame.surfarray.pixels_alpha(surf)[:, :] = arr[..., 3]
        surf = pygame.transform.rotate(surf)  # <--- Kopf zeigt jetzt nach oben
        surf = pygame.transform.scale(surf, (width, height))
        return surf

    def reset(self, key: jax.random.PRNGKey = jax.random.key(1701)) -> Tuple[SkiingObservation, GameState]:
        """Initialize a new game state"""
        flags = []

        y_spacing = (
            self.config.screen_height - 4 * self.config.flag_height
        ) / self.config.max_num_flags
        for i in range(self.config.max_num_flags):
            x = random.randint(
                self.config.flag_width,
                self.config.screen_width
                - self.config.flag_width
                - self.config.flag_distance,
            )
            y = int((i + 1) * y_spacing + self.config.flag_height)
            flags.append((float(x), float(y)))

        trees = []
        for _ in range(self.config.max_num_trees):
            x = random.randint(
                self.config.tree_width,
                self.config.screen_width - self.config.tree_width,
            )
            y = random.randint(
                self.config.tree_height,
                self.config.screen_height - self.config.tree_height,
            )
            trees.append((float(x), float(y)))

        rocks = []
        for _ in range(self.config.max_num_rocks):
            x = random.randint(
                self.config.rock_width,
                self.config.screen_width - self.config.rock_width,
            )
            y = random.randint(
                self.config.rock_height,
                self.config.screen_height - self.config.rock_height,
            )
            rocks.append((float(x), float(y)))

        state = GameState(
            skier_x=jnp.array(76.0),
            skier_pos=jnp.array(4),
            skier_fell=jnp.array(0),
            skier_x_speed=jnp.array(0.0),
            skier_y_speed=jnp.array(1.0),
            flags=jnp.array(flags),
            trees=jnp.array(trees),
            rocks=jnp.array(rocks),
            score=jnp.array(20),
            time=jnp.array(0),
            direction_change_counter=jnp.array(0),
            game_over=jnp.array(False),
            key=key,
            jumping=jnp.array(False),
            jump_timer=jnp.array(0),
            collision_type=jnp.array(0)
        )
        obs = self._get_observation(state)

        return obs, state

    def _create_new_objs(self, state, new_flags, new_trees, new_rocks):
        k, k1, k2, k3, k4 = jax.random.split(state.key, num=5)

        k1 = jnp.array([k1, k2, k3, k4])

        def check_flags(i, flags):
            x_flag = jax.random.randint(
                k1.at[i].get(),
                [],
                self.config.flag_width,
                self.config.screen_width
                - self.config.flag_width
                - self.config.flag_distance,
            )
            x_flag = jnp.array(x_flag, jnp.float32)
            y = BOTTOM_BORDER + jax.random.randint(k1.at[3 - i].get(), [], 0, 100)

            new_f = jax.lax.cond(
                jnp.less(flags.at[i, 1].get(), TOP_BORDER),
                lambda _: jnp.array([x_flag, y], jnp.float32),
                lambda _: flags.at[i].get(),
                operand=None,
            )

            flags = flags.at[i].set(new_f)

            return flags

        flags = jax.lax.fori_loop(0, 2, check_flags, new_flags)

        k, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(k, 9)
        k1 = jnp.array([k1, k2, k3, k4, k5, k6, k7, k8])

        def check_trees(i, trees):
            x_tree = jax.random.randint(
                k1.at[i].get(),
                [],
                self.config.tree_width,
                self.config.screen_width - self.config.tree_width,
            )
            x_tree = jnp.array(x_tree, jnp.float32)
            y = BOTTOM_BORDER + jax.random.randint(k1.at[7 - i].get(), [], 0, 100)

            new_f = jax.lax.cond(
                jnp.less(trees.at[i, 1].get(), TOP_BORDER),
                lambda _: jnp.array([x_tree, y], jnp.float32),
                lambda _: trees.at[i].get(),
                operand=None,
            )
            trees = trees.at[i].set(new_f)
            return trees

        trees = jax.lax.fori_loop(0, 4, check_trees, new_trees)

        k, k1, k2, k3, k4, k5, k6 = jax.random.split(k, 7)
        k1 = jnp.array([k1, k2, k3, k4, k5, k6])

        def check_rocks(i, rocks):
            x_rock = jax.random.randint(
                k1.at[i].get(),
                [],
                self.config.rock_width,
                self.config.screen_width - self.config.rock_width,
            )
            x_rock = jnp.array(x_rock, jnp.float32)
            y = BOTTOM_BORDER + jax.random.randint(k1.at[5 - i].get(), [], 0, 100)

            new_f = jax.lax.cond(
                jnp.less(rocks.at[i, 1].get(), TOP_BORDER),
                lambda _: jnp.array([x_rock, y], jnp.float32),
                lambda _: rocks.at[i].get(),
                operand=None,
            )
            rocks = rocks.at[i].set(new_f)
            return rocks

        rocks = jax.lax.fori_loop(0, 3, check_rocks, new_rocks)

        return flags, trees, rocks, k

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> tuple[SkiingObservation, GameState, float, bool, SkiingInfo]:
        #                              -->  --_      \     |     |    /    _-- <--
        side_speed = jnp.array(
            [-1.0, -0.5, -0.333, 0.0, 0.0, 0.333, 0.5, 1], jnp.float32
        )

        #                              -->  --_   \     |    |     /    _--  <--
        down_speed = jnp.array(
            [0.0, 0.5, 0.875, 1.0, 1.0, 0.875, 0.5, 0.0], jnp.float32
        )

        """Take a step in the game given an action"""
        # Handle jump action
        jumping = state.jumping
        jump_timer = state.jump_timer

        # If JUMP action and not currently jumping, start a jump
        jumping, jump_timer = jax.lax.cond(
            jnp.logical_and(jnp.equal(action, JUMP), jnp.logical_not(jumping)),
            lambda _: (jnp.array(True), jnp.array(self.config.jump_duration)),
            lambda _: (state.jumping, state.jump_timer),
            operand=None,
        )
        
        # If already jumping, decrement timer
        jump_timer = jax.lax.cond(
            jumping,
            lambda t: jnp.maximum(t - 1, 0),
            lambda t: t,
            operand=jump_timer,
        )
        
        # End jump if timer reaches 0
        jumping = jax.lax.cond(
            jnp.equal(jump_timer, 0),
            lambda _: jnp.array(False),
            lambda _: jumping,
            operand=None,
        )

        # Handle left/right movement
        new_skier_pos = jax.lax.cond(
            jnp.equal(action, LEFT),
            lambda _: state.skier_pos - 1,
            lambda _: state.skier_pos,
            operand=None,
        )
        new_skier_pos = jax.lax.cond(
            jnp.equal(action, RIGHT),
            lambda _: state.skier_pos + 1,
            lambda _: new_skier_pos,
            operand=None,
        )
        skier_pos = jnp.clip(new_skier_pos, 0, 7)

        skier_pos, direction_change_counter = jax.lax.cond(
            jnp.greater(state.direction_change_counter, 0),
            lambda _: (state.skier_pos, state.direction_change_counter - 1),
            lambda _: (skier_pos, 0),
            operand=None,
        )

        direction_change_counter = jax.lax.cond(
            jnp.logical_and(
                jnp.not_equal(skier_pos, state.skier_pos),
                jnp.equal(direction_change_counter, 0),
            ),
            lambda _: jnp.array(16),
            lambda _: direction_change_counter,
            operand=None,
        )

        dy = down_speed.at[skier_pos].get()
        dx = side_speed.at[skier_pos].get()

        new_skier_x_speed = state.skier_x_speed + ((dx - state.skier_x_speed) * 0.1)

        # IMPORTANT FIX: Don't increase vertical speed during jumps
        # Instead, maintain normal vertical speed but handle the visual effect separately
        new_skier_y_speed = state.skier_y_speed + ((dy - state.skier_y_speed) * 0.05)

        new_x = jnp.clip(
            state.skier_x + new_skier_x_speed,
            self.config.skier_width / 2,
            self.config.screen_width - self.config.skier_width / 2,
        )

        # Move objects at normal speed regardless of jumping state
        new_trees = state.trees
        for i in range(len(state.trees)):
            new_trees = new_trees.at[i, 1].set(state.trees[i][1] - new_skier_y_speed)

        new_rocks = state.rocks
        for i in range(len(state.rocks)):
            new_rocks = new_rocks.at[i, 1].set(state.rocks[i][1] - new_skier_y_speed)

        new_flags = state.flags
        for i in range(len(state.flags)):
            new_flags = new_flags.at[i, 1].set(state.flags[i][1] - new_skier_y_speed)

        def check_pass_flag(flag_pos):
            fx, fy = flag_pos
            dx_0 = new_x - fx
            dy_0 = jnp.abs(self.config.skier_y - jnp.round(fy))
            return (dx_0 > 0) & (dx_0 < self.config.flag_distance) & (dy_0 < 1)

        def check_collision_flag(obj_pos, x_distance=1, y_distance=1):
            x, y = obj_pos
            dx_1 = jnp.abs(new_x - x)
            dy_1 = jnp.abs(jnp.round(self.config.skier_y) - jnp.round(y))

            dx_2 = jnp.abs(new_x - (x + self.config.flag_distance))
            dy_2 = jnp.abs(jnp.round(self.config.skier_y) - jnp.round(y))

            return jnp.logical_or(
                jnp.logical_and(dx_1 <= x_distance, dy_1 < y_distance),
                jnp.logical_and(dx_2 <= x_distance, dy_2 < y_distance),
            )

        def check_collision_tree(tree_pos, x_distance=3, y_distance=1):
            x, y = tree_pos
            dx = jnp.abs(new_x - x)
            dy = jnp.abs(jnp.round(self.config.skier_y) - jnp.round(y))

            return jnp.logical_and(dx <= x_distance, dy < y_distance)

        def check_collision_rock(rock_pos, x_distance=1, y_distance=1):
            x, y = rock_pos
            dx = jnp.abs(new_x - x)
            dy = jnp.abs(jnp.round(self.config.skier_y) - jnp.round(y))

            # No collision with rocks when jumping
            return jnp.logical_and(
                jnp.logical_and(dx < x_distance, dy < y_distance),
                jnp.logical_not(jumping)  # This ensures no collision when jumping
            )


        new_flags, new_trees, new_rocks, new_key = self._create_new_objs(
            state, new_flags, new_trees, new_rocks
        )

        passed_flags = jax.vmap(check_pass_flag)(jnp.array(new_flags))

        collisions_flag = jax.vmap(check_collision_flag)(jnp.array(new_flags))
        collisions_tree = jax.vmap(check_collision_tree)(jnp.array(new_trees))
        collisions_rocks = jax.vmap(check_collision_rock)(jnp.array(new_rocks))

        num_colls = (
            jnp.sum(collisions_tree)
            + jnp.sum(collisions_rocks)
            + jnp.sum(collisions_flag)
        )

        # Bestimme, wodurch die erste Kollision verursacht wurde
        collision_type = jax.lax.select(
            jnp.sum(collisions_tree) > 0,
            jnp.array(1),  # Baum
            jax.lax.select(
                jnp.sum(collisions_rocks) > 0,
                jnp.array(2),  # Stein
                jax.lax.select(
                    jnp.sum(collisions_flag) > 0,
                    jnp.array(3),  # Flagge
                    jnp.array(0),  # Keine
                ),
            ),
        )


        (
            new_x,
            skier_fell,
            num_colls,
            new_flags,
            new_trees,
            new_rocks,
            skier_pos,
            new_skier_x_speed,
            new_skier_y_speed,
        ) = jax.lax.cond(
            jnp.greater(state.skier_fell, 0),
            lambda _: (
                state.skier_x,
                state.skier_fell - 1,
                0,
                state.flags,
                state.trees,
                state.rocks,
                state.skier_pos,
                state.skier_x_speed,
                state.skier_y_speed,
            ),
            lambda _: (
                new_x,
                state.skier_fell,
                num_colls,
                new_flags,
                new_trees,
                new_rocks,
                skier_pos,
                new_skier_x_speed,
                new_skier_y_speed,
            ),
            operand=None,
        )

        skier_fell = jax.lax.cond(
            jnp.logical_and(jnp.greater(num_colls, 0), jnp.equal(skier_fell, 0)),
            lambda _: jnp.array(60),
            lambda _: skier_fell,
            operand=None,
        )

        new_score = jax.lax.cond(
            jnp.equal(skier_fell, 0),
            lambda _: state.score - jnp.sum(passed_flags),
            lambda _: state.score,
            operand=None,
        )
        game_over = jax.lax.cond(
            jnp.equal(new_score, 0),
            lambda _: jnp.array(True),
            lambda _: jnp.array(False),
            operand=None,
        )
        new_time = jax.lax.cond(
            jnp.greater(state.time, 9223372036854775807/2),
            lambda _: 0,
            lambda _: state.time + 1,
            operand=None,
        )

        new_state = GameState(
            skier_x=new_x,
            skier_pos=jnp.array(skier_pos),
            skier_fell=skier_fell,
            skier_x_speed=new_skier_x_speed,
            skier_y_speed=new_skier_y_speed,
            flags=jnp.array(new_flags),
            trees=jnp.array(new_trees),
            rocks=jnp.array(new_rocks),
            score=new_score,
            time=new_time,
            direction_change_counter=direction_change_counter,
            game_over=game_over,
            key=new_key,
            jumping=jumping,
            jump_timer=jump_timer,
            collision_type=collision_type
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        # create skier
        skier = EntityPosition(
            x=state.skier_x,
            y=jnp.array(self.config.skier_y),
            width=jnp.array(self.config.skier_width),
            height=jnp.array(self.config.skier_height),
        )

        # create trees
        trees = jnp.zeros((self.config.max_num_trees, 4))
        for i in range(self.config.max_num_trees):
            tree_pos = state.trees.at[i].get()
            trees = trees.at[i].set(
                jnp.array(
                    [
                        tree_pos.at[0].get(),  # x position
                        tree_pos.at[1].get(),  # y position
                        self.config.tree_width,  # width
                        self.config.tree_height,  # height
                    ]
                )
            )

        # create flags
        flags = jnp.zeros((self.config.max_num_flags, 4))
        for i in range(self.config.max_num_flags):
            flag_pos = state.flags.at[i].get()
            flags = flags.at[i].set(
                jnp.array(
                    [
                        flag_pos.at[0].get(),  # x position
                        flag_pos.at[1].get(),  # y position
                        self.config.flag_width,  # width
                        self.config.flag_height,  # height
                    ]
                )
            )

        # create rocks
        rocks = jnp.zeros((self.config.max_num_rocks, 4))
        for i in range(self.config.max_num_rocks):
            rock_pos = state.rocks.at[i].get()
            rocks = rocks.at[i].set(
                jnp.array(
                    [
                        rock_pos.at[0].get(),  # x position
                        rock_pos.at[1].get(),  # y position
                        self.config.rock_width,  # width
                        self.config.rock_height,  # height
                    ]
                )
            )

        return SkiingObservation(
            skier=skier, trees=trees, flags=flags, rocks=rocks, score=state.score, jumping=state.jumping, jump_timer=state.jump_timer
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState) -> SkiingInfo:
        return SkiingInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState):
        return previous_state.score - state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        return jnp.equal(state.score, 0)


@dataclass
class RenderConfig:
    """Configuration for rendering"""

    scale_factor: int = 4
    background_color: Tuple[int, int, int] = (255, 255, 255)
    skier_color = [
        (0, 0, 255),
        (0, 0, 255),
        (0, 0, 100),
        (0, 0, 100),
        (0, 255, 0),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 255, 0),
        (100, 0, 255),
        (100, 0, 255),
    ]
    flag_color: Tuple[int, int, int] = (255, 0, 0)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    tree_color: Tuple[int, int, int] = (0, 100, 0)
    rock_color: Tuple[int, int, int] = (128, 128, 128)
    game_over_color: Tuple[int, int, int] = (255, 0, 0)
    jump_text_color: Tuple[int, int, int] = (0, 0, 255)



class GameRenderer:
    def __init__(self, game_config: GameConfig, render_config: RenderConfig):
        
        self.game_config = game_config
        self.render_config = render_config

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (
                self.game_config.screen_width * self.render_config.scale_factor,
                self.game_config.screen_height * self.render_config.scale_factor,
            )
        )
        pygame.display.set_caption("JAX Skiing Game")

        # Create sprites
        self.skier_sprite = self._create_skier_sprite()
        self.skier_jump_sprite = self._create_skier_jump_sprite()  # Add jump sprite
        self.flag_sprite = self._create_flag_sprite()
        self.rock_sprite = self._create_rock_sprite()
        self.tree_sprite = self._create_tree_sprite()
        self.font = pygame.font.Font(None, 36)
        
    def _npy_to_surface(self, npy_path, width, height):
        arr = np.load(npy_path)  # Erwartet (H, W, 4) RGBA
        arr = arr.astype(np.uint8)
        surf = pygame.Surface((arr.shape[1], arr.shape[0]), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surf)[:, :, :] = arr[..., :3]
        pygame.surfarray.pixels_alpha(surf)[:, :] = arr[..., 3]
        surf = pygame.transform.scale(surf, (width, height))
        return surf

    def _create_skier_jump_sprite(self) -> pygame.Surface:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "games", "sprites", "skiing")
        full_path = os.path.join(sprite_dir, "skiier_jump.npy")
        return self._npy_to_surface(
            full_path,
            self.game_config.skier_width * self.render_config.scale_factor,
            self.game_config.skier_height * self.render_config.scale_factor,
        )

    def get_path_center(self, world_y: float) -> float:
        # Returns center X of the path at given world Y
        return 80 + 30 * jnp.sin(world_y / 30.0)


    def _draw_blue_path(self, skier_y_pos: float):
        path_width = 40
        scale = self.render_config.scale_factor
        screen_height = self.game_config.screen_height

        for screen_y in range(0, screen_height, 4):
            world_y = skier_y_pos + screen_y
            x_center = float(80 + 30 * jnp.sin(world_y / 30.0))  # Use environment logic
            y_pos = float(screen_y)

            left_x = int((x_center - (path_width / 2)) * scale)
            right_x = int((x_center + (path_width / 2)) * scale)
            y_scaled = int(y_pos * scale)

            pygame.draw.line(
                self.screen, (0, 0, 255), (left_x, y_scaled), (right_x, y_scaled), 1
            )




    def _create_skier_sprite(self) -> list[pygame.Surface]:
        # Base path relative to the project root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "games", "sprites", "skiing")

        filenames = {
            "left": "skiier_left.npy",
            "front": "skiier_front.npy",
            "right": "skiier_right.npy"
        }

        sprites = {}
        for direction, filename in filenames.items():
            full_path = os.path.join(sprite_dir, filename)
            sprites[direction] = self._npy_to_surface(
                full_path,
                self.game_config.skier_width * self.render_config.scale_factor,
                self.game_config.skier_height * self.render_config.scale_factor,
            )

        # Map skier_pos (0-7) to a direction
        sprite_list = []
        for i in range(8):
            if i <= 2:
                sprite_list.append(sprites["left"])
            elif i >= 5:
                sprite_list.append(sprites["right"])
            else:
                sprite_list.append(sprites["front"])
        
        return sprite_list


    def _create_flag_sprite(self) -> pygame.Surface:
        size = (
            self.game_config.flag_width * self.render_config.scale_factor,
            self.game_config.flag_height * self.render_config.scale_factor,
        )
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        scaled_width = size[0]
        scaled_height = size[1]

        pygame.draw.line(
            sprite,
            self.render_config.text_color,
            (scaled_width // 2, 0),
            (scaled_width // 2, scaled_height),
            2,
        )

        pygame.draw.polygon(
            sprite,
            self.render_config.flag_color,
            [
                (scaled_width // 2, scaled_height // 4),
                (scaled_width, scaled_height // 2),
                (scaled_width // 2, scaled_height // 4 * 3),
            ],
        )

        return sprite

    def _create_tree_sprite(self) -> pygame.Surface:
        size = (
            self.game_config.tree_width * self.render_config.scale_factor,
            self.game_config.tree_height * self.render_config.scale_factor,
        )
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        scaled_width = size[0]
        scaled_height = size[1]

        # Tree trunk
        trunk_width = scaled_width // 3
        trunk_height = scaled_height // 3
        trunk_rect = pygame.Rect(
            (scaled_width - trunk_width) // 2,
            scaled_height - trunk_height,
            trunk_width,
            trunk_height,
        )
        pygame.draw.rect(sprite, (139, 69, 19), trunk_rect)

        # Tree triangles
        for i in range(3):
            height_offset = i * (scaled_height // 3)
            width_factor = 0.8 + (i * 0.1)
            pygame.draw.polygon(
                sprite,
                self.render_config.tree_color,
                [
                    (scaled_width // 2, height_offset),
                    (
                        scaled_width * (1 - width_factor) // 2,
                        height_offset + scaled_height // 3,
                    ),
                    (
                        scaled_width * (1 + width_factor) // 2,
                        height_offset + scaled_height // 3,
                    ),
                ],
            )

        return sprite

    def _create_rock_sprite(self) -> pygame.Surface:
        size = (
            self.game_config.rock_width * self.render_config.scale_factor,
            self.game_config.rock_height * self.render_config.scale_factor,
        )
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        scaled_width = size[0]
        scaled_height = size[1]

        # Draw a polygon for the rock
        points = [
            (scaled_width * 0.2, scaled_height * 0.8),
            (0, scaled_height * 0.4),
            (scaled_width * 0.3, scaled_height * 0.2),
            (scaled_width * 0.7, scaled_height * 0.1),
            (scaled_width, scaled_height * 0.5),
            (scaled_width * 0.8, scaled_height),
        ]
        pygame.draw.polygon(sprite, self.render_config.rock_color, points)

        return sprite

    def render(self, state: GameState):
        """Render the current game state"""
        self.screen.fill(self.render_config.background_color)

        # Draw blue path
        # self._draw_blue_path(state.skier_x)

        # Draw skier
        skier_pos = (
            int(
                (state.skier_x - self.game_config.skier_width / 2)
                * self.render_config.scale_factor
            ),
            int(
                (self.game_config.skier_y - self.game_config.skier_height / 2)
                * self.render_config.scale_factor
            ),
        )

        # Calculate scale factor based on jump state
        scale_factor = 1.0
        if state.jumping:
            # Calculate scale based on jump progress
            # Use a parabolic curve for the scale: start normal, grow to max at middle, return to normal
            jump_progress = state.jump_timer / self.game_config.jump_duration
            # This creates a parabolic curve that peaks in the middle of the jump
            scale_factor = 1.0 + (self.game_config.jump_scale_factor - 1.0) * (4 * jump_progress * (1 - jump_progress))

        # Choose sprite based on jumping state
        if state.jumping:
            # Scale the jump sprite based on jump progress
            scaled_width = int(self.game_config.skier_width * self.render_config.scale_factor * scale_factor)
            scaled_height = int(self.game_config.skier_height * self.render_config.scale_factor * scale_factor)
            
            # Adjust position to keep skier centered when scaled
            adjusted_x = skier_pos[0] - (scaled_width - self.game_config.skier_width * self.render_config.scale_factor) // 2
            adjusted_y = skier_pos[1] - (scaled_height - self.game_config.skier_height * self.render_config.scale_factor) // 2
            
            # Scale the jump sprite
            scaled_sprite = pygame.transform.scale(self.skier_jump_sprite, (scaled_width, scaled_height))
            self.screen.blit(scaled_sprite, (adjusted_x, adjusted_y))
        else:
            # Use regular skier sprite based on direction
            direction_index = int(state.skier_pos)
            # Make sure the index is valid for our sprite list
            if direction_index < 0:
                direction_index = 0
            elif direction_index >= len(self.skier_sprite):
                direction_index = len(self.skier_sprite) - 1
                
            self.screen.blit(self.skier_sprite[direction_index], skier_pos)

        # Draw flags
        for fx, fy in state.flags:
            flag_pos = (
                int(
                    (fx - self.game_config.flag_width / 2)
                    * self.render_config.scale_factor
                ),
                int(
                    (fy - self.game_config.flag_height / 2)
                    * self.render_config.scale_factor
                ),
            )
            self.screen.blit(self.flag_sprite, flag_pos)
            second_flag_pos = (
                flag_pos[0]
                + self.game_config.flag_distance * self.render_config.scale_factor,
                flag_pos[1],
            )
            self.screen.blit(self.flag_sprite, second_flag_pos)

        for fx, fy in state.trees:
            tree_pos = (
                int(
                    (fx - self.game_config.tree_width / 2)
                    * self.render_config.scale_factor
                ),
                int(
                    (fy - self.game_config.tree_height / 2)
                    * self.render_config.scale_factor
                ),
            )
            self.screen.blit(self.tree_sprite, tree_pos)

        for fx, fy in state.rocks:
            rock_pos = (
                int(
                    (fx - self.game_config.rock_width / 2)
                    * self.render_config.scale_factor
                ),
                int(
                    (fy - self.game_config.rock_height / 2)
                    * self.render_config.scale_factor
                ),
            )
            self.screen.blit(self.rock_sprite, rock_pos)

        # Draw UI
        score_text = self.font.render(
            f"Score: {state.score}", True, self.render_config.text_color
        )
                # Zeit formatieren wie 00:00.00
        total_time = int(state.time)
        minutes = total_time // (60 * 60)
        seconds = (total_time // 60) % 60
        hundredths = total_time % 60
        time_str = f"{minutes:02}:{seconds:02}.{hundredths:02}"
        time_text = self.font.render(time_str, True, self.render_config.text_color)

        # Score mittig oben, Zeit darunter mittig
        screen_width_px = self.game_config.screen_width * self.render_config.scale_factor

        score_rect = score_text.get_rect(center=(screen_width_px // 2, 10 + score_text.get_height() // 2))
        time_rect = time_text.get_rect(center=(screen_width_px // 2, 10 + score_text.get_height() + time_text.get_height() // 2))
        
        self.screen.blit(score_text, score_rect)
        self.screen.blit(time_text, time_rect)

        if state.game_over:
            game_over_text = self.font.render(
                "You Won!", True, self.render_config.game_over_color
            )
            text_rect = game_over_text.get_rect(
                center=(
                    self.game_config.screen_width
                    * self.render_config.scale_factor
                    // 2,
                    self.game_config.screen_height
                    * self.render_config.scale_factor
                    // 2,
                )
            )
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()

    def close(self):
        """Clean up pygame resources"""
        pygame.quit()


def show_game_over_popup(screen, scale_factor):
    popup_width = 300
    popup_height = 150
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_surface.fill((200, 200, 200))
    pygame.draw.rect(popup_surface, (0, 0, 0), popup_surface.get_rect(), 2)

    font = pygame.font.Font(None, 36)
    restart_button = pygame.Rect(50, 90, 90, 40)
    close_button = pygame.Rect(160, 90, 90, 40)

    popup_surface.blit(font.render("Game Over!", True, (0, 0, 0)), (80, 20))
    pygame.draw.rect(popup_surface, (100, 255, 100), restart_button)
    pygame.draw.rect(popup_surface, (255, 100, 100), close_button)
    popup_surface.blit(font.render("Restart", True, (0, 0, 0)), (55, 95))
    popup_surface.blit(font.render("Close", True, (0, 0, 0)), (170, 95))

    screen.blit(
        popup_surface,
        (
            (screen.get_width() - popup_width) // 2,
            (screen.get_height() - popup_height) // 2,
        ),
    )
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                abs_restart = restart_button.move(
                    (screen.get_width() - popup_width) // 2,
                    (screen.get_height() - popup_height) // 2,
                )
                abs_close = close_button.move(
                    (screen.get_width() - popup_width) // 2,
                    (screen.get_height() - popup_height) // 2,
                )
                if abs_restart.collidepoint(mouse_pos):
                    return "restart"
                elif abs_close.collidepoint(mouse_pos):
                    return "quit"
        pygame.time.wait(100)


def main():
    # Create configurations
    game_config = GameConfig()
    render_config = RenderConfig()

    while True:
        # Initialize game and renderer
        game = JaxSkiing()
        _, state = game.reset()
        renderer = GameRenderer(game_config, render_config)

        clock = pygame.time.Clock()
        running = True
        while running and not state.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            action = NOOP
            if keys[pygame.K_SPACE]:
                action = JUMP
            elif keys[pygame.K_a]:
                action = LEFT
            elif keys[pygame.K_d]:
                action = RIGHT

            obs, state, reward, done, info = game.step(state, action)
            renderer.render(state)

            if state.skier_fell > 0 and state.collision_type in (1, 2):  # Nur Baum oder Stein
                # Trigger popup on collision
                result = show_game_over_popup(renderer.screen, render_config.scale_factor)
                if result == "restart":
                    running = False  # Exit inner loop to restart
                elif result == "quit":
                    pygame.quit()
                    return

            clock.tick(60)

        renderer.close()


if __name__ == "__main__":
    main()
