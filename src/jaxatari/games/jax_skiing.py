from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
import jax.image as jimage
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import random
import os
from sys import maxsize
import numpy as np

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces

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


class SkiingInfo(NamedTuple):
    time: jnp.ndarray


class JaxSkiing(JaxEnvironment[GameState, SkiingObservation, SkiingInfo, None]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()
        self.renderer = SkiingRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        # Aktionen sind bei dir: NOOP=0, LEFT=1, RIGHT=2
        return spaces.Discrete(4)

    def reset(
        self, key: jax.random.PRNGKey = jax.random.key(1701)
    ) -> Tuple[SkiingObservation, GameState]:
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
            collision_type=jnp.array(0),
            flags_passed=jnp.zeros(self.config.max_num_flags, dtype=bool),
        )
        obs = self._get_observation(state)

        return obs, state
    
    def render(self, state: GameState) -> jnp.ndarray:
        """Delegiert an den SkiingRenderer, sodass play.py ein RGB-Frame bekommt."""
        return self.renderer.render(state)

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
    def step(
        self, state: GameState, action: int
    ) -> tuple[SkiingObservation, GameState, float, bool, SkiingInfo]:
        #                              -->  --_      \     |     |    /    _-- <--
        side_speed = jnp.array(
            [-1.0, -0.5, -0.333, 0.0, 0.0, 0.333, 0.5, 1], jnp.float32
        )

        #                              -->  --_   \     |    |     /    _--  <--
        down_speed = jnp.array(
            [0.0, 0.5, 0.875, 1.0, 1.0, 0.875, 0.5, 0.0], jnp.float32
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
            flags_passed = (
                state.flags_passed
            )  # Fix: Use the existing flags_passed from state
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
                collision_type=state.collision_type,
                flags_passed=flags_passed,
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

            # Instead, maintain normal vertical speed but handle the visual effect separately
            new_skier_y_speed = state.skier_y_speed + (
                (dy - state.skier_y_speed) * 0.05
            )

            new_x = jnp.clip(
                state.skier_x + new_skier_x_speed,
                self.config.skier_width / 2,
                self.config.screen_width - self.config.skier_width / 2,
            )

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

                return jnp.logical_and(dx < x_distance, dy < y_distance)

            # Check if gates have been passed before respawn
            passed_flags = jax.vmap(check_pass_flag)(jnp.array(new_flags))
            flags_passed = state.flags_passed | passed_flags

            # Determine which flags despawn this frame (y < TOP_BORDER)
            despawn_mask = new_flags[:, 1] < TOP_BORDER
            missed_penalty_mask = jnp.logical_and(
                despawn_mask, jnp.logical_not(flags_passed)
            )
            missed_penalty_count = jnp.sum(missed_penalty_mask)
            missed_penalty = missed_penalty_count * 300

            # Reset flags_passed when a flag despawns
            flags_passed = jnp.where(despawn_mask, False, flags_passed)

            # Spawn new objects after calculating penalties
            new_flags, new_trees, new_rocks, new_key = self._create_new_objs(
                state, new_flags, new_trees, new_rocks
            )

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
                jnp.greater(state.time, 9223372036854775807 / 2),
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
                collision_type=collision_type,
                flags_passed=flags_passed,  # <--- Argument ergänzt
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
            skier=skier,
            trees=trees,
            flags=flags,
            rocks=rocks,
            score=state.score
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
    
    # Text-Overlay-Option: True = UI-Text via Pygame auf fertiges JAX-Frame
    # False = (optionale) JAX-Bitmap-Font (siehe Kommentar unten im Code)
    use_pygame_text: bool = True

# ---- Sprite-Assets als JAX-Arrays ------------------------------------------------

class RenderAssets(NamedTuple):
    skier_left: jnp.ndarray      # (Hs, Ws, 4) uint8
    skier_front: jnp.ndarray
    skier_right: jnp.ndarray
    skier_fallen: jnp.ndarray
    flag_red: jnp.ndarray
    flag_blue: jnp.ndarray
    tree: jnp.ndarray
    rock: jnp.ndarray

def _device_put_u8(arr: np.ndarray) -> jnp.ndarray:
    return jax.device_put(jnp.asarray(arr, dtype=jnp.uint8))

def _load_sprite_npy(base_dir: str, name: str) -> jnp.ndarray:
    path = os.path.join(base_dir, name)
    rgba = np.load(path).astype(np.uint8)  # (H, W, 4)
    if rgba.shape[-1] == 3:
        a = np.full(rgba.shape[:2] + (1,), 255, np.uint8)
        rgba = np.concatenate([rgba, a], axis=-1)
    return _device_put_u8(rgba)

def _recolor_rgba(sprite_rgba: jnp.ndarray, rgb: Tuple[int,int,int]) -> jnp.ndarray:
    """Ersetzt RGB an allen Pixeln mit Alpha>0 (gerade, einfache Variante)."""
    mask = (sprite_rgba[..., 3:4] > 0).astype(jnp.uint8)  # (H,W,1)
    rgb_arr = jnp.asarray(jnp.array(rgb, dtype=jnp.uint8))[None, None, :].astype(jnp.uint8)
    new_rgb = (sprite_rgba[..., :3] * (1 - mask)) + (rgb_arr * mask)
    return jnp.concatenate([new_rgb, sprite_rgba[..., 3:4]], axis=-1)

# ---- JAX Hilfsfunktionen: Skalierung & Blitting ----------------------------------

def _nn_resize_rgba(img: jnp.ndarray, new_h: int, new_w: int) -> jnp.ndarray:
    out = jimage.resize(img, (new_h, new_w, 4), method="nearest")
    return jnp.clip(out, 0, 255).astype(jnp.uint8)

def _alpha_over(dst: jnp.ndarray, src: jnp.ndarray, top: jnp.ndarray, left: jnp.ndarray) -> jnp.ndarray:
    """
    Alpha-Compositing (SrcOver) eines RGBA-Sprites in ein RGBA-Frame – JIT-sicher.
    Verwendet Padding + dynamic_slice/update, damit alle Slice-Größen statisch sind.
    """
    H, W, _ = dst.shape
    h, w, _ = src.shape

    top  = jnp.asarray(top,  jnp.int32)
    left = jnp.asarray(left, jnp.int32)

    # Puffer-Padding: groß genug, dass wir immer (h,w) aus der gepaddeten Fläche schneiden können
    ph = h
    pw = w
    pad_cfg = ((ph, ph), (pw, pw), (0, 0))
    dst_pad = jnp.pad(dst, pad_cfg, mode="constant", constant_values=0)

    # Startkoordinaten in der gepaddeten Fläche (immer gültig)
    start_y = jnp.clip(top  + ph, 0, H + 2*ph - h).astype(jnp.int32)
    start_x = jnp.clip(left + pw, 0, W + 2*pw - w).astype(jnp.int32)

    # Fixe (statische) Slice-Größen: (h, w, 4)
    dst_sub = jax.lax.dynamic_slice(dst_pad, (start_y, start_x, 0), (h, w, 4)).astype(jnp.float32)
    src_sub = src.astype(jnp.float32)

    sa = src_sub[..., 3:4] / 255.0
    da = dst_sub[..., 3:4] / 255.0
    out_a   = sa + da * (1.0 - sa)
    out_rgb = src_sub[..., :3] * sa + dst_sub[..., :3] * (1.0 - sa)
    out = jnp.concatenate([out_rgb, out_a * 255.0], axis=-1)
    out = jnp.clip(out, 0.0, 255.0).astype(jnp.uint8)

    # Patch zurückschreiben
    dst_pad = jax.lax.dynamic_update_slice(dst_pad, out, (start_y, start_x, 0))

    # Originalbereich aus der gepaddeten Fläche zurückschneiden
    dst_final = jax.lax.dynamic_slice(dst_pad, (ph, pw, 0), (H, W, 4))
    return dst_final

def _blit_center(dst: jnp.ndarray, sprite: jnp.ndarray, cx: jnp.ndarray, cy: jnp.ndarray) -> jnp.ndarray:
    """Blit Sprite so, dass seine Mitte bei (cx, cy) in Pixelkoordinaten liegt (JIT-sicher)."""
    h = sprite.shape[0]
    w = sprite.shape[1]
    # cx, cy sind JAX-Scalars; rechne alles als jnp.int32 weiter
    top  = (cy - (h // 2)).astype(jnp.int32)
    left = (cx - (w // 2)).astype(jnp.int32)
    return _alpha_over(dst, sprite, top, left)

def _scan_blit(dst: jnp.ndarray, sprites: jnp.ndarray, centers_xy: jnp.ndarray) -> jnp.ndarray:
    """Zeichnet mehrere Sprites nacheinander (Reihenfolge bleibt erhalten).
       sprites: (N, hs, ws, 4) oder (N, 1, 1, 4) wenn gleiche Größe → wir vmap’en nicht über Größe.
       centers_xy: (N, 2) int32 Pixelcenter."""
    def body(frame, inputs):
        spr, cxy = inputs
        frame = _blit_center(frame, spr, cxy[0], cxy[1])
        return frame, None
    out, _ = jax.lax.scan(body, dst, (sprites, centers_xy))
    return out

# ---- Pure JAX Renderer -----------------------------------------------------------

@partial(jax.jit,
         static_argnames=("screen_width","screen_height","scale_factor","skier_y","flag_distance","draw_ui_jax"))
def render_frame(
    state: GameState,
    assets: RenderAssets,
    *,
    screen_width: int,
    screen_height: int,
    scale_factor: int,
    skier_y: int,
    flag_distance: int,
    draw_ui_jax: bool = False,
) -> jnp.ndarray:
    """Erzeugt ein RGBA-Frame (uint8) rein in JAX – keine Seiteneffekte."""
    # 1) Leeres (upgescaltes) RGBA-Frame
    H = screen_height * scale_factor
    W = screen_width * scale_factor
    bg_rgb = jnp.array([255, 255, 255], dtype=jnp.uint8)
    frame = jnp.concatenate(
        [jnp.full((H, W, 3), bg_rgb, dtype=jnp.uint8),
         jnp.full((H, W, 1), 255, dtype=jnp.uint8)],
        axis=-1
    )

    # 2) Hilfsfunktionen für Koordinaten (Game→Pixel)
    def to_px_x(x): return jnp.round(x * scale_factor).astype(jnp.int32)
    def to_px_y(y): return jnp.round(y * scale_factor).astype(jnp.int32)

    # 3) Skier-Sprite auswählen (links/front/rechts; „fallen“ hat Vorrang)
    # mapping wie bisher: 0..2 = left, 3..4 = front, 5..7 = right
    pos = jnp.clip(state.skier_pos, 0, 7)
  
    skier_base = jax.lax.cond(
        pos <= 2,
        lambda _: assets.skier_left,
        lambda _: jax.lax.cond(
            pos >= 5,
            lambda __: assets.skier_right,
            lambda __: assets.skier_front,
            operand=None,   # <— WICHTIG: inneres cond bekommt ein operand
        ),
        operand=None,       # <— WICHTIG: äußeres cond bekommt ein operand
    )

    skier_sprite = jax.lax.cond(
        jnp.logical_and(state.skier_fell > 0, jnp.logical_or(state.collision_type == 1, state.collision_type == 2)),
        lambda _: assets.skier_fallen,
        lambda _: skier_base,
        operand=None,       # <— operand setzen
    )

    skier_cx = to_px_x(state.skier_x)
    skier_cy = to_px_y(jnp.array(skier_y))

    # Wenn gefallen: Position am nächsten Kollisionsobjekt (Tree/Rock)
    def fallen_center(_):
        # Distanz zu Trees/Rocks in Pixeln (L1), argmin
        tree_xy = jnp.round(state.trees * scale_factor).astype(jnp.int32)
        rock_xy = jnp.round(state.rocks * scale_factor).astype(jnp.int32)
        sx = skier_cx
        sy = skier_cy
        def nearest(xy):
            d = jnp.abs(xy[:,0] - sx) + jnp.abs(xy[:,1] - sy)
            idx = jnp.argmin(d)
            return xy[idx]
        cxcy = jax.lax.cond(
            state.collision_type == 1,  # tree
            lambda __: nearest(tree_xy),
            lambda __: jax.lax.cond(
                state.collision_type == 2,
                lambda ___: nearest(rock_xy),
                lambda ___: jnp.array([sx, sy], jnp.int32),
                operand=None,  # 👈 inneres cond braucht das operand
            ),
            operand=None
        )
        return cxcy[0], cxcy[1]
    skier_cx, skier_cy = jax.lax.cond(
        jnp.logical_and(state.skier_fell > 0, jnp.logical_or(state.collision_type == 1, state.collision_type == 2)),
        fallen_center,
        lambda _: (skier_cx, skier_cy),
        operand=None
    )

    # 4) Flags (links & rechts), jede 20. Gate rot, sonst blau
    #    centers sind Pixelcenter; Reihenfolge: Skier -> Flags -> Trees -> Rocks (wie zuvor)
    flags = state.flags  # (N,2) in Game-Koordinaten
    left_px  = jnp.round(flags * scale_factor).astype(jnp.int32)               # (N,2)
    right_px = jnp.round((flags + jnp.array([float(flag_distance), 0.0])) * scale_factor).astype(jnp.int32)
    # Farbe wählen: 1..N, idx%20==0 => red
    n_flags = flags.shape[0]
    idxs = jnp.arange(1, n_flags+1, dtype=jnp.int32)
    is_red = (idxs % 20) == 0
    # je Gate zwei Sprites (links & rechts)
    flag_sprites_gate = jax.vmap(lambda r: jax.lax.cond(r, lambda _: assets.flag_red, lambda _: assets.flag_blue, operand=None))(is_red)
    # tiles: (N*2, ...)
    flag_sprites = jnp.concatenate([flag_sprites_gate, flag_sprites_gate], axis=0)
    flag_centers = jnp.concatenate([left_px, right_px], axis=0)

    # 5) Trees & Rocks (Pixelcenter)
    tree_px = jnp.round(state.trees * scale_factor).astype(jnp.int32)
    rock_px = jnp.round(state.rocks * scale_factor).astype(jnp.int32)

    # 6) Skalierungen der Sprites für Ausgabeauflösung (scale_factor)
    def scale_sprite(spr: jnp.ndarray) -> jnp.ndarray:
        # Zielgröße ist immer "input * scale_factor"
        h = spr.shape[0]
        w = spr.shape[1]
        return _nn_resize_rgba(spr, h * scale_factor, w * scale_factor)

    skier_draw = scale_sprite(skier_sprite)
    flag_red_draw  = scale_sprite(assets.flag_red)
    flag_blue_draw = scale_sprite(assets.flag_blue)
    tree_draw = scale_sprite(assets.tree)
    rock_draw = scale_sprite(assets.rock)
    # Da Flags gemischt sind, bauen wir ein Array der tatsächlich verwendeten Sprites:
    # (N*2, h, w, 4) – wir mappen „is_red“ erneut:
    flag_sprites = jax.vmap(lambda r: jax.lax.cond(r, lambda _: flag_red_draw, lambda _: flag_blue_draw, operand=None))(jnp.concatenate([is_red, is_red], axis=0))

    # 7) Zeichnen in der korrekten Reihenfolge
    frame = _blit_center(frame, skier_draw, skier_cx, skier_cy)
    frame = _scan_blit(frame, flag_sprites, flag_centers)
    frame = _scan_blit(frame, jnp.repeat(tree_draw[None, ...], tree_px.shape[0], axis=0), tree_px)
    frame = _scan_blit(frame, jnp.repeat(rock_draw[None, ...], rock_px.shape[0], axis=0), rock_px)

    # 8) (Optional) UI/Text direkt in JAX – hier deaktiviert um identische Optik
    #    Du kannst hier eine 5x7-Ziffern-Bitmap hinterlegen und blitten.
    #    Für 1:1 identischen Look nutze unten den Pygame-Fallback im Display-Bridge.
    return frame

class SkiingRenderer(JAXGameRenderer):
    def __init__(self, consts=None):
        super().__init__()
        # Deine Game-Konstanten (Breite/Höhe/Abstände) – wie bei Pong: env.consts
        self.consts = consts or GameConfig()

        # Sprites einmalig als JAX-Arrays laden (RGBA)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_dir = os.path.join(base_dir, "sprites", "skiing")

        flag_red = _load_sprite_npy(sprite_dir, "checkered_flag.npy")
        self.assets = RenderAssets(
            skier_left   = _load_sprite_npy(sprite_dir, "skiier_right.npy"),  # (deine Spiegelung beibehalten)
            skier_front  = _load_sprite_npy(sprite_dir, "skiier_front.npy"),
            skier_right  = _load_sprite_npy(sprite_dir, "skiier_left.npy"),
            skier_fallen = _load_sprite_npy(sprite_dir, "skier_fallen.npy"),
            flag_red     = flag_red,
            flag_blue    = _recolor_rgba(flag_red, (0, 96, 255)),
            tree         = _load_sprite_npy(sprite_dir, "tree.npy"),
            rock         = _load_sprite_npy(sprite_dir, "stone.npy"),
        )

        # JIT’ter für deine pure Renderfunktion, **ohne Upscaling**
        self._render_fn = partial(
            render_frame,
            screen_width   = self.consts.screen_width,    # 160
            screen_height  = self.consts.screen_height,   # 210
            scale_factor   = 1,         # play.py skaliert selbst
            skier_y        = self.consts.skier_y,
            flag_distance  = self.consts.flag_distance,
            draw_ui_jax    = False,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        rgba = self._render_fn(state, self.assets)   # (210,160,4) uint8
        return rgba[..., :3]                         # -> (210,160,3) RGB

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

        # ---- Sprite-Assets als JAX Arrays laden (einmalig) ----
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sprite_dir = os.path.join(base_path, "jaxatari", "games", "sprites", "skiing")

        skier_left   = _load_sprite_npy(sprite_dir, "skiier_right.npy")  # (ALE links/rechts sind invertiert in deinem Bestand)
        skier_front  = _load_sprite_npy(sprite_dir, "skiier_front.npy")
        skier_right  = _load_sprite_npy(sprite_dir, "skiier_left.npy")
        skier_fallen = _load_sprite_npy(sprite_dir, "skier_fallen.npy")

        flag_red = _load_sprite_npy(sprite_dir, "checkered_flag.npy")
        flag_blue = _recolor_rgba(flag_red, (0, 96, 255))
        tree = _load_sprite_npy(sprite_dir, "tree.npy")
        rock = _load_sprite_npy(sprite_dir, "stone.npy")

        self.assets = RenderAssets(
            skier_left=skier_left,
            skier_front=skier_front,
            skier_right=skier_right,
            skier_fallen=skier_fallen,
            flag_red=flag_red,
            flag_blue=flag_blue,
            tree=tree,
            rock=rock,
        )

        # JIT-Compile die Renderfunktion mit statischen Parametern (Ints/Bools)
        self._render_jit = partial(
            render_frame,
            screen_width=self.game_config.screen_width,
            screen_height=self.game_config.screen_height,
            scale_factor=self.render_config.scale_factor,
            skier_y=self.game_config.skier_y,
            flag_distance=self.game_config.flag_distance,
            draw_ui_jax=False,
        )
        # Warmup (optional)
        # _ = self._render_jit(self._fake_state(), self.assets)

        # Font nur noch für optionalen UI-Fallback:
        self.font = pygame.font.Font(None, 36)

    # --- Mini-Helfer: RGBA-ndarray auf den Screen bringen (ohne Layout-Logik) ---
    def _blit_rgba_to_screen(self, rgba: np.ndarray):
        # rgba: (H, W, 4) uint8
        H, W, _ = rgba.shape
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        # Pygame erwartet (W,H,3) und (W,H) für alpha
        rgb = rgba[..., :3].transpose(1, 0, 2).copy()
        a   = rgba[..., 3].T.copy()
        pygame.surfarray.pixels3d(surf)[:] = rgb
        pygame.surfarray.pixels_alpha(surf)[:] = a
        self.screen.blit(surf, (0, 0))

    def render(self, state: GameState):
        # 1) Pure JAX Frame rendern
        frame = self._render_jit(state, self.assets)  # jnp.uint8[H,W,4]
        frame_np = np.asarray(frame)
        # 2) Optional: UI-Text via Pygame oben drauf (identische Optik)
        if self.render_config.use_pygame_text:
            self._blit_rgba_to_screen(frame_np)
            # Score mittig oben, Zeit darunter (wie vorher)
            score_text = self.font.render(f"Score: {int(state.score)}", True, self.render_config.text_color)
            total_time = int(state.time)
            minutes = total_time // (60 * 60)
            seconds = (total_time // 60) % 60
            hundredths = total_time % 60
            time_str = f"{minutes:02}:{seconds:02}.{hundredths:02}"
            time_text = self.font.render(time_str, True, self.render_config.text_color)

            screen_width_px = self.game_config.screen_width * self.render_config.scale_factor
            score_rect = score_text.get_rect(center=(screen_width_px // 2, 10 + score_text.get_height() // 2))
            time_rect = time_text.get_rect(center=(screen_width_px // 2, 10 + score_text.get_height() + time_text.get_height() // 2))
            self.screen.blit(score_text, score_rect)
            self.screen.blit(time_text, time_rect)

            if state.game_over:
                game_over_text = self.font.render("You Won!", True, self.render_config.game_over_color)
                text_rect = game_over_text.get_rect(
                    center=(self.game_config.screen_width * self.render_config.scale_factor // 2,
                            self.game_config.screen_height * self.render_config.scale_factor // 2)
                )
                self.screen.blit(game_over_text, text_rect)
        else:
            self._blit_rgba_to_screen(frame_np)

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
            if keys[pygame.K_a]:
                action = LEFT
            elif keys[pygame.K_d]:
                action = RIGHT

            obs, state, reward, done, info = game.step(state, action)
            renderer.render(state)

            clock.tick(60)

        renderer.close()


if __name__ == "__main__":
    main()
