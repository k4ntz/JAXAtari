
from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import random
import os

from jaxatari.environment import JaxEnvironment

NOOP = 0
LEFT = 1
RIGHT = 2

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


class SkiingInfo(NamedTuple):
    time: jnp.ndarray


class JaxSkiing(JaxEnvironment[GameState, SkiingObservation, SkiingInfo]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()

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
            lambda _: 16,
            lambda _: direction_change_counter,
            operand=None,
        )

        dy = down_speed.at[skier_pos].get()

        dx = side_speed.at[skier_pos].get()

        new_skier_x_speed = state.skier_x_speed + ((dx - state.skier_x_speed) * 0.1)

        new_skier_y_speed = state.skier_y_speed + ((dy - state.skier_y_speed) * 0.05)

        new_x = jnp.clip(
            state.skier_x + new_skier_x_speed,
            self.config.skier_width / 2,
            self.config.screen_width - self.config.skier_width / 2,
        )

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

            return jnp.logical_and(dx < x_distance, dy < y_distance)

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
            jnp.greater(state.time, 254),
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
            skier=skier, trees=trees, flags=flags, rocks=rocks, score=state.score
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
        self.flag_sprite = self._create_flag_sprite()
        self.rock_sprite = self._create_rock_sprite()
        self.tree_sprite = self._create_tree_sprite()
        self.font = pygame.font.Font(None, 36)

    # def _create_skier_sprite(self) -> list[pygame.Surface]:
    #     repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    #     img_path = os.path.join(repo_root, "jaxatari", "assets", "skiing", "skiier_front.jpg")
    #     img = pygame.image.load(img_path).convert_alpha()
    #     img = pygame.transform.scale(
    #         img,
    #         (
    #             self.game_config.skier_width * self.render_config.scale_factor,
    #             self.game_config.skier_height * self.render_config.scale_factor,
    #         )
    #     )

    #     # 4. Gib Sprite mehrfach zurück (für `skier_pos`-Kompatibilität)
    #     return [img for _ in range(16)]

    def _create_skier_sprite(self) -> list[pygame.Surface]:
        # Basispfad relativ zum Projekt-Root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "assets", "skiing")

        filenames = {
            "left": "skiier_left.png",
            "front": "skiier_front.png",
            "right": "skiier_right.png"
        }

        sprites = {}
        for direction, filename in filenames.items():
            full_path = os.path.join(sprite_dir, filename)
            img = pygame.image.load(full_path).convert_alpha()
            img = pygame.transform.scale(
                img,
                (
                    self.game_config.skier_width * self.render_config.scale_factor,
                    self.game_config.skier_height * self.render_config.scale_factor,
                )
            )
            sprites[direction] = img

            # Map skier_pos (0–7) auf eine Richtung
            sprite_list = []
            for i in range(8):
                if i <= 2:
                    sprite_list.append(sprites["left"])
                elif i >= 5:
                    sprite_list.append(sprites["left"])
                else:
                    sprite_list.append(sprites["left"])

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

        direction = state.skier_pos
        if direction < 3:
            index = 0  # left
        elif direction > 5:
            index = 2  # right
        else:
            index = 1  # straight

        self.screen.blit(self.skier_sprite[index], skier_pos)

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
        time_text = self.font.render(
            f"Time: {state.time:.1f}", True, self.render_config.text_color
        )
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 40))

        if state.game_over:
            game_over_text = self.font.render(
                "Game Over!", True, self.render_config.game_over_color
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


# main.py
def main():
    # Create configurations
    game_config = GameConfig()
    render_config = RenderConfig()

    # Initialize game and renderer
    game = JaxSkiing()
    _, state = game.reset()
    renderer = GameRenderer(game_config, render_config)

    # Setup game loop
    clock = pygame.time.Clock()
    running = True

    while running and not state.game_over:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            action = LEFT
        elif keys[pygame.K_d]:
            action = RIGHT
        else:
            action = NOOP

        # Update game state
        obs, state, reward, done, info = game.step(state, action)

        # Render
        renderer.render(state)

        # Cap at 60 FPS
        clock.tick(60)

    # If game over, wait before closing
    if state.game_over:
        pygame.time.wait(2000)

    renderer.close()


if __name__ == "__main__":
    main()
