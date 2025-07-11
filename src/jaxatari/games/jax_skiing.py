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


def _npy_to_jax_array(npy_path: str) -> jnp.ndarray:
    """Load a sprite stored as ``.npy`` into a JAX array.

    Parameters
    ----------
    npy_path : str
        Path to the ``.npy`` file containing RGBA pixel data in
        ``(H, W, 4)`` format.

    Returns
    -------
    jnp.ndarray
        Sprite as a ``uint8`` JAX array with shape ``(H, W, 4)``.
    """

    arr = np.load(npy_path).astype(np.uint8)
    return jnp.array(arr)


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
    flags_passed: chex.Array

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
            collision_type=jnp.array(0),
            flags_passed=jnp.zeros(self.config.max_num_flags, dtype=bool)
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

        # --- NEU: Fallen-Logik ---
        def handle_fallen(state, action):
            # Nur auf LEFT oder RIGHT reagieren
            is_side = jnp.logical_or(jnp.equal(action, LEFT), jnp.equal(action, RIGHT))
            # Wenn seitliche Eingabe: skier_fell auf 0, sonst dekrementieren
            skier_fell = jax.lax.cond(
                is_side,
                lambda _: jnp.array(0),
                lambda _: jnp.maximum(state.skier_fell - 1, 0),
                operand=None,
            )
            # Keine Bewegung, keine Punkteänderung, keine Objektbewegung
            flags_passed = state.flags_passed  # Fix: Use the existing flags_passed from state
            new_state = GameState(
                skier_x=state.skier_x,
                skier_pos=state.skier_pos,
                skier_fell=skier_fell,
                skier_x_speed=jnp.array(0.0),
                skier_y_speed=jnp.array(0.0),
                flags=state.flags,
                trees=state.trees,
                rocks=state.rocks,
                score=state.score,
                time=state.time,
                direction_change_counter=state.direction_change_counter,
                game_over=state.game_over,
                key=state.key,
                jumping=state.jumping,
                jump_timer=state.jump_timer,
                collision_type=state.collision_type,
                flags_passed=flags_passed
            )
            obs = self._get_observation(new_state)
            info = self._get_info(new_state)
            reward = jnp.array(0.0, dtype=jnp.float32)  # Typ explizit setzen
            done = self._get_done(new_state)
            return obs, new_state, reward, done, info

        # --- Standardspielablauf ---
        def normal_step(state, action):
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
            # Move objects using vectorized operations instead of Python loops
            new_trees = state.trees.at[:, 1].add(-new_skier_y_speed)
            new_rocks = state.rocks.at[:, 1].add(-new_skier_y_speed)
            new_flags = state.flags.at[:, 1].add(-new_skier_y_speed)

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

            # Flags vor und nach Schritt
            old_flags = state.flags  # shape (max_num_flags, 2)
            # new_flags wird weiter unten berechnet

            passed_flags = jax.vmap(check_pass_flag)(jnp.array(new_flags))
            flags_passed = state.flags_passed | passed_flags
            # Penalty nur, wenn Flagge im letzten Frame noch sichtbar war (old_y >= TOP_BORDER),
            # jetzt despawnt (new_y < TOP_BORDER), und NICHT passiert wurde
            old_y = old_flags[:, 1]
            new_y = new_flags[:, 1]
            despawn_mask = jnp.logical_and(old_y >= TOP_BORDER, new_y < TOP_BORDER)
            missed_penalty_mask = jnp.logical_and(despawn_mask, jnp.logical_not(flags_passed))
            missed_penalty_count = jnp.sum(missed_penalty_mask)
            missed_penalty = missed_penalty_count * 300

            collisions_flag = jax.vmap(check_collision_flag)(jnp.array(new_flags))
            collisions_tree = jax.vmap(check_collision_tree)(jnp.array(new_trees))
            collisions_rocks = jax.vmap(check_collision_rock)(jnp.array(new_rocks))

            num_colls_pre = (
                jnp.sum(collisions_tree)
                + jnp.sum(collisions_rocks)
                + jnp.sum(collisions_flag)
            )

            collision_occurred = jnp.logical_and(
                jnp.greater(num_colls_pre, 0), jnp.equal(state.skier_fell, 0)
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
                    num_colls_pre,
                    new_flags,
                    new_trees,
                    new_rocks,
                    skier_pos,
                    new_skier_x_speed,
                    new_skier_y_speed,
                ),
                operand=None,
            )

            # Apply small knockback when colliding with an obstacle
            new_x = jax.lax.cond(
                collision_occurred,
                lambda _: jnp.clip(
                    new_x - 5.0,
                    self.config.skier_width / 2,
                    self.config.screen_width - self.config.skier_width / 2,
                ),
                lambda _: new_x,
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
            penalty = jax.lax.select(collision_occurred, jnp.array(1), jnp.array(0))
            new_score = new_score - penalty
            game_over = jax.lax.cond(
                jnp.equal(new_score, 0),
                lambda _: jnp.array(True),
                lambda _: jnp.array(False),
                operand=None,
            )
            new_time = jax.lax.cond(
                jnp.greater(state.time, 9223372036854775807/2),
                lambda _: 0,
                lambda _: state.time + 1 + missed_penalty,
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
                collision_type=collision_type,
                flags_passed=flags_passed  # <--- Argument ergänzt
            )

            done = self._get_done(new_state)
            reward = self._get_reward(state, new_state)
            reward = jnp.asarray(reward, dtype=jnp.float32)
            obs = self._get_observation(new_state)
            info = self._get_info(new_state)

            return obs, new_state, reward, done, info

        # Hauptlogik: Fallen oder normal
        obs, new_state, reward, done, info = jax.lax.cond(
            jnp.greater(state.skier_fell, 0),
            lambda _: handle_fallen(state, action),
            lambda _: normal_step(state, action),
            operand=None,
        )
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
        tree_static = jnp.array(
            [self.config.tree_width, self.config.tree_height], dtype=jnp.float32
        )
        trees = jnp.concatenate(
            [state.trees, jnp.tile(tree_static, (self.config.max_num_trees, 1))],
            axis=1,
        )

        # create flags
        flag_static = jnp.array(
            [self.config.flag_width, self.config.flag_height], dtype=jnp.float32
        )
        flags = jnp.concatenate(
            [state.flags, jnp.tile(flag_static, (self.config.max_num_flags, 1))],
            axis=1,
        )

        # create rocks
        rock_static = jnp.array(
            [self.config.rock_width, self.config.rock_height], dtype=jnp.float32
        )
        rocks = jnp.concatenate(
            [state.rocks, jnp.tile(rock_static, (self.config.max_num_rocks, 1))],
            axis=1,
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

        pygame.init()
        self.screen = pygame.display.set_mode(
            (
                self.game_config.screen_width * self.render_config.scale_factor,
                self.game_config.screen_height * self.render_config.scale_factor,
            )
        )
        pygame.display.set_caption("JAX Skiing Game")

        # Erstelle alle Sprites mit der neuen Hilfsmethode
        self.skier_sprite = self._create_skier_sprite()
        self.skier_jump_sprite = self._create_object_sprite(
            "skiier_jump.npy",
            int(self.game_config.skier_width * self.render_config.scale_factor * 2),
            int(self.game_config.skier_height * self.render_config.scale_factor * 2),
        )
        self.skier_fallen_sprite = self._create_object_sprite(
            "skier_fallen.npy",
            int(self.game_config.skier_width * self.render_config.scale_factor * 2),
            int(self.game_config.skier_height * self.render_config.scale_factor * 2),
        )
        # JAX arrays for vectorized operations
        self.skier_jump_array = self._load_object_array("skiier_jump.npy")
        self.skier_fallen_array = self._load_object_array("skier_fallen.npy")
        self.flag_sprite = self._create_object_sprite(
            "checkered_flag.npy",
            int(self.game_config.flag_width * self.render_config.scale_factor * 2),
            int(self.game_config.flag_height * self.render_config.scale_factor * 2),
        )
        self.flag_array = self._load_object_array("checkered_flag.npy")
        self.rock_sprite = self._create_object_sprite(
            "stone.npy",
            int(self.game_config.rock_width * self.render_config.scale_factor * 3),
            int(self.game_config.rock_height * self.render_config.scale_factor * 6),
        )
        self.rock_array = self._load_object_array("stone.npy")
        self.tree_sprite = self._create_object_sprite(
            "tree.npy",
            int(self.game_config.tree_width * self.render_config.scale_factor * 1.5),
            int(self.game_config.tree_height * self.render_config.scale_factor * 1.5),
        )
        self.tree_array = self._load_object_array("tree.npy")
        self.font = pygame.font.Font(None, 36)

    def _npy_to_surface(self, npy_path, width, height):
        arr = np.load(npy_path)  # Erwartet (H, W, 4) RGBA
        arr = arr.astype(np.uint8)
        surf = pygame.Surface((arr.shape[1], arr.shape[0]), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(surf)[:, :, :] = arr[..., :3]
        pygame.surfarray.pixels_alpha(surf)[:, :] = arr[..., 3]
        surf = pygame.transform.scale(surf, (width, height))
        return surf

    def _create_object_sprite(self, filename, width, height):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "games", "sprites", "skiing")
        full_path = os.path.join(sprite_dir, filename)
        return self._npy_to_surface(full_path, width, height)

    def _load_object_array(self, filename) -> jnp.ndarray:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "games", "sprites", "skiing")
        full_path = os.path.join(sprite_dir, filename)
        return _npy_to_jax_array(full_path)

    def _create_skier_sprite(self) -> list[pygame.Surface]:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "games", "sprites", "skiing")
        filenames = {
            "right": "skiier_left.npy",
            "front": "skiier_front.npy",
            "left": "skiier_right.npy"
        }
        width = self.game_config.skier_width * self.render_config.scale_factor
        height = self.game_config.skier_height * self.render_config.scale_factor
        sprites = {}
        for direction, filename in filenames.items():
            full_path = os.path.join(sprite_dir, filename)
            sprites[direction] = self._npy_to_surface(full_path, width, height)
        self.skier_array = {d: _npy_to_jax_array(os.path.join(sprite_dir, f)) for d, f in filenames.items()}
        sprite_list = []
        for i in range(8):
            if i <= 2:
                sprite_list.append(sprites["left"])
            elif i >= 5:
                sprite_list.append(sprites["right"])
            else:
                sprite_list.append(sprites["front"])
        return sprite_list

    @partial(jax.jit, static_argnums=(0,))
    def _calc_flag_centers(self, flags: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate pixel centers for all flags.

        Parameters
        ----------
        flags : jnp.ndarray
            Array with shape (N, 2) containing flag positions in game
            coordinates.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Pixel coordinates for the left and right flag of each gate.
        """

        scale = jnp.array(self.render_config.scale_factor, dtype=jnp.float32)
        distance = jnp.array(self.game_config.flag_distance, dtype=jnp.float32)

        left = jnp.round(flags * scale).astype(jnp.int32)
        right = jnp.round((flags + jnp.array([distance, 0.0])) * scale).astype(jnp.int32)

        return left, right

    @partial(jax.jit, static_argnums=(0,))
    def _calc_tree_centers(self, trees: jnp.ndarray) -> jnp.ndarray:
        """Calculate pixel centers for all trees."""

        scale = jnp.array(self.render_config.scale_factor, dtype=jnp.float32)
        return jnp.round(trees * scale).astype(jnp.int32)

    def render(self, state: GameState):
        """Render the current game state"""
        self.screen.fill(self.render_config.background_color)

        # Skier
        skier_img = None
        # Zeige "skier_fallen" Sprite bei Baum- oder Stein-Kollision
        if state.skier_fell > 0 and state.collision_type in (1, 2):
            skier_img = self.skier_fallen_sprite
            # Zentriere Sprite exakt auf das kollidierte Hindernis
            if state.collision_type == 1:  # Baum
                # Finde den ersten Baum, der mit dem Skifahrer kollidiert
                tree_centers = self._calc_tree_centers(state.trees)
                skier_x_px = int(state.skier_x * self.render_config.scale_factor)
                skier_y_px = int(self.game_config.skier_y * self.render_config.scale_factor)
                # Suche den Baum mit minimalem Abstand zum Skifahrer
                dists = [abs(tx - skier_x_px) + abs(ty - skier_y_px) for tx, ty in np.array(tree_centers)]
                idx = int(np.argmin(dists))
                cx, cy = np.array(tree_centers[idx])
            elif state.collision_type == 2:  # Stein
                rock_centers = np.round(np.array(state.rocks) * self.render_config.scale_factor).astype(int)
                skier_x_px = int(state.skier_x * self.render_config.scale_factor)
                skier_y_px = int(self.game_config.skier_y * self.render_config.scale_factor)
                dists = [abs(rx - skier_x_px) + abs(ry - skier_y_px) for rx, ry in np.array(rock_centers)]
                idx = int(np.argmin(dists))
                cx, cy = rock_centers[idx]
            else:
                # Fallback auf Skier-Position
                cx = int(state.skier_x * self.render_config.scale_factor)
                cy = int(self.game_config.skier_y * self.render_config.scale_factor)
            skier_rect = skier_img.get_rect(center=(cx, cy))
        else:
            if state.jumping:
                skier_img = self.skier_jump_sprite
            else:
                skier_img = self.skier_sprite[int(state.skier_pos)]
            skier_rect = skier_img.get_rect(center=(
                int(state.skier_x * self.render_config.scale_factor),
                int(self.game_config.skier_y * self.render_config.scale_factor),
            ))

        if state.jumping and not (state.skier_fell > 0 and state.collision_type in (1, 2)):
            jump_progress = state.jump_timer / self.game_config.jump_duration
            scale_factor = 1.0 + (self.config.jump_scale_factor - 1.0) * (4 * jump_progress * (1 - jump_progress))
            new_size = (
                int(self.game_config.skier_width * self.render_config.scale_factor * scale_factor),
                int(self.game_config.skier_height * self.render_config.scale_factor * scale_factor),
            )
            skier_img = pygame.transform.scale(skier_img, new_size)
            skier_rect = skier_img.get_rect(center=skier_rect.center)
        self.screen.blit(skier_img, skier_rect)

        # Flags
        left_centers, right_centers = self._calc_flag_centers(state.flags)
        for left, right in zip(np.array(left_centers), np.array(right_centers)):
            flag_rect = self.flag_sprite.get_rect()
            flag_rect.center = (int(left[0]), int(left[1]))
            self.screen.blit(self.flag_sprite, flag_rect)

            second_flag_rect = self.flag_sprite.get_rect()
            second_flag_rect.center = (int(right[0]), int(right[1]))
            self.screen.blit(self.flag_sprite, second_flag_rect)

        # Trees
        tree_centers = self._calc_tree_centers(state.trees)
        for tx, ty in np.array(tree_centers):
            tree_rect = self.tree_sprite.get_rect()
            tree_rect.center = (int(tx), int(ty))
            self.screen.blit(self.tree_sprite, tree_rect)

        # Rocks
        for fx, fy in state.rocks:
            rock_rect = self.rock_sprite.get_rect()
            rock_rect.center = (
                int(fx * self.render_config.scale_factor),
                int(fy * self.render_config.scale_factor),
            )
            self.screen.blit(self.rock_sprite, rock_rect)

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
                    running = False  # Fenster-X: Spiel beenden
                    pygame.quit()
                    return
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


            clock.tick(60)

        renderer.close()

if __name__ == "__main__":
    main()
