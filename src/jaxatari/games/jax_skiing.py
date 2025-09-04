from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
import jax.image as jimage
from dataclasses import dataclass
from typing import Tuple, NamedTuple, Callable, Sequence, Optional
import os
import numpy as np
import collections

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces

class SkiingConstants(NamedTuple):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FIRE = 3
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
    collision_cooldown: chex.Array  # Frames, in denen Kollisionen ignoriert werden (Debounce nach Recovery)


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
    all_rewards: jnp.ndarray


class JaxSkiing(JaxEnvironment[GameState, SkiingObservation, SkiingInfo, SkiingConstants]):
    def __init__(self, consts: SkiingConstants | None = None, reward_funcs: Optional[Sequence[Callable[[GameState, GameState], jnp.ndarray]]] = None,):
        consts = consts or SkiingConstants()
        super().__init__(consts)
        self.config = GameConfig()
        self.reward_funcs = tuple(reward_funcs) if reward_funcs is not None else None
        self.state = self.reset()
        self.renderer = SkiingRenderer(self.config)

    def action_space(self) -> spaces.Discrete:
        # Aktionen sind bei dir: NOOP=0, LEFT=1, RIGHT=2
        return spaces.Discrete(4)

    def observation_space(self):
        c = self.config

        # --- CHANGED: make helper actually produce float64 tensors (not float64)
        def f64(x):
            return jnp.array(x, dtype=jnp.float64)

        skier_space = spaces.Dict(collections.OrderedDict({
            "x":      spaces.Box(low=f64(0.0),               high=f64(float(c.screen_width)),  shape=(), dtype=jnp.float64),
            "y":      spaces.Box(low=f64(0.0),               high=f64(float(c.screen_height)), shape=(), dtype=jnp.float64),
            "width":  spaces.Box(low=f64(float(c.skier_width)),  high=f64(float(c.skier_width)),  shape=(), dtype=jnp.float64),
            "height": spaces.Box(low=f64(float(c.skier_height)), high=f64(float(c.skier_height)), shape=(), dtype=jnp.float64),
        }))

        flags_space = spaces.Box(low=f64([0.0, 0.0]),
                                 high=f64([float(c.screen_width), float(c.screen_height)]),
                                 shape=(c.max_num_flags, 2), dtype=jnp.float64)
        trees_space = spaces.Box(low=f64([0.0, 0.0]),
                                 high=f64([float(c.screen_width), float(c.screen_height)]),
                                 shape=(c.max_num_trees, 2), dtype=jnp.float64)
        rocks_space = spaces.Box(low=f64([0.0, 0.0]),
                                 high=f64([float(c.screen_width), float(c.screen_height)]),
                                 shape=(c.max_num_rocks, 2), dtype=jnp.float64)

        score_space = spaces.Box(low=jnp.array(0, dtype=jnp.int32),
                                 high=jnp.array(1_000_000, dtype=jnp.int32),
                                 shape=(), dtype=jnp.int32)

        return spaces.Dict(collections.OrderedDict({
            "skier": skier_space, "flags": flags_space, "trees": trees_space, "rocks": rocks_space, "score": score_space,
        }))

    def image_space(self):
        c = self.config
        return spaces.Box(low=0, high=255, shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs: SkiingObservation) -> jnp.ndarray:
        # --- CHANGED: return flattened float64
        skier_vec  = jnp.array(
            [obs.skier.x, obs.skier.y, obs.skier.width, obs.skier.height],
            dtype=jnp.float64
        ).reshape(-1)
    
        flags_flat = jnp.asarray(obs.flags, dtype=jnp.float64).reshape(-1)
        trees_flat = jnp.asarray(obs.trees, dtype=jnp.float64).reshape(-1)
        rocks_flat = jnp.asarray(obs.rocks, dtype=jnp.float64).reshape(-1)
        # Score is int32; keep as float64 in flat vector for consistency
        score_flat = jnp.asarray(obs.score, dtype=jnp.float64).reshape(-1)
    
        return jnp.concatenate([skier_vec, flags_flat, trees_flat, rocks_flat, score_flat], axis=0)

    def reset(self, key: jax.random.PRNGKey = jax.random.key(1701)) -> Tuple[SkiingObservation, GameState]:
        """Initialize a new game state deterministically from `key`."""
        c = self.config
        k_flags, k_trees, k_rocks, new_key = jax.random.split(key, 4)

        # Flags: y gleichmäßig verteilt, x zufällig
        y_spacing = (c.screen_height - 4 * c.flag_height) / c.max_num_flags
        i = jnp.arange(c.max_num_flags, dtype=jnp.float64)
        flags_y = (i + 1.0) * y_spacing + float(c.flag_height)
        flags_x = jax.random.randint(
            k_flags, (c.max_num_flags,),
            minval=int(c.flag_width),
            maxval=int(c.screen_width - c.flag_width - c.flag_distance) + 1
        ).astype(jnp.float64)
        flags = jnp.stack([
            flags_x, flags_y,
            jnp.full((c.max_num_flags,), float(c.flag_width),  dtype=jnp.float64),
            jnp.full((c.max_num_flags,), float(c.flag_height), dtype=jnp.float64)
        ], axis=1)

        # Trees
        trees_x = jax.random.randint(
            k_trees, (c.max_num_trees,),
            minval=int(c.tree_width),
            maxval=int(c.screen_width - c.tree_width) + 1
        ).astype(jnp.float64)
        trees_y = jax.random.randint(
            k_trees, (c.max_num_trees,),
            minval=int(c.tree_height),
            maxval=int(c.screen_height - c.tree_height) + 1
        ).astype(jnp.float64)
        trees = jnp.stack([
            trees_x, trees_y,
            jnp.full((c.max_num_trees,), float(c.tree_width),  dtype=jnp.float64),
            jnp.full((c.max_num_trees,), float(c.tree_height), dtype=jnp.float64)
        ], axis=1)

        # Rocks
        rocks_x = jax.random.randint(
            k_rocks, (c.max_num_rocks,),
            minval=int(c.rock_width),
            maxval=int(c.screen_width - c.rock_width) + 1
        ).astype(jnp.float64)
        rocks_y = jax.random.randint(
            k_rocks, (c.max_num_rocks,),
            minval=int(c.rock_height),
            maxval=int(c.screen_height - c.rock_height) + 1
        ).astype(jnp.float64)
        rocks = jnp.stack([
            rocks_x, rocks_y,
            jnp.full((c.max_num_rocks,), float(c.rock_width),  dtype=jnp.float64),
            jnp.full((c.max_num_rocks,), float(c.rock_height), dtype=jnp.float64)
        ], axis=1)

        state = GameState(
            skier_x=jnp.array(76.0),
            skier_pos=jnp.array(4, dtype=jnp.int32),
            skier_fell=jnp.array(0, dtype=jnp.int32),
            skier_x_speed=jnp.array(0.0),
            skier_y_speed=jnp.array(1.0),
            flags=flags,
            trees=trees,
            rocks=rocks,
            score=jnp.array(20, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            direction_change_counter=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False),
            key=new_key,
            collision_type=jnp.array(0, dtype=jnp.int32),
            flags_passed=jnp.zeros(c.max_num_flags, dtype=bool),
            collision_cooldown=jnp.array(0, dtype=jnp.int32),
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
            # neue x/y
            x_flag = jax.random.randint(
                k1.at[i].get(), [], 
                self.config.flag_width,
                self.config.screen_width - self.config.flag_width - self.config.flag_distance
            ).astype(jnp.float64)
            y = (self.consts.BOTTOM_BORDER + 
                 jax.random.randint(k1.at[3 - i].get(), [], 0, 100)).astype(jnp.float64)

            row_old = flags.at[i].get()                      # Shape (2,) oder (4,)
            row_new = row_old.at[0].set(x_flag).at[1].set(y) # gleiche Shape wie row_old

            cond = jnp.less(flags.at[i, 1].get(), self.consts.TOP_BORDER)
            out_row = jax.lax.cond(cond, lambda _: row_new, lambda _: row_old, operand=None)
            return flags.at[i].set(out_row)

        flags = jax.lax.fori_loop(0, 2, check_flags, new_flags)

        # ---- Trees ----
        k, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(k, 9)
        k1 = jnp.array([k1, k2, k3, k4, k5, k6, k7, k8])

        def check_trees(i, trees):
            x_tree = jax.random.randint(
                k1.at[i].get(), [], 
                self.config.tree_width,
                self.config.screen_width - self.config.tree_width
            ).astype(jnp.float64)
            y = (self.consts.BOTTOM_BORDER + 
                 jax.random.randint(k1.at[7 - i].get(), [], 0, 100)).astype(jnp.float64)

            row_old = trees.at[i].get()
            row_new = row_old.at[0].set(x_tree).at[1].set(y)

            cond = jnp.less(trees.at[i, 1].get(), self.consts.TOP_BORDER)
            out_row = jax.lax.cond(cond, lambda _: row_new, lambda _: row_old, operand=None)
            return trees.at[i].set(out_row)

        trees = jax.lax.fori_loop(0, 4, check_trees, new_trees)

        # ---- Rocks ----
        k, k1, k2, k3, k4, k5, k6 = jax.random.split(k, 7)
        k1 = jnp.array([k1, k2, k3, k4, k5, k6])

        def check_rocks(i, rocks):
            x_rock = jax.random.randint(
                k1.at[i].get(), [], 
                self.config.rock_width,
                self.config.screen_width - self.config.rock_width
            ).astype(jnp.float64)
            y = (self.consts.BOTTOM_BORDER + 
                 jax.random.randint(k1.at[5 - i].get(), [], 0, 100)).astype(jnp.float64)

            row_old = rocks.at[i].get()
            row_new = row_old.at[0].set(x_rock).at[1].set(y)

            cond = jnp.less(rocks.at[i, 1].get(), self.consts.TOP_BORDER)
            out_row = jax.lax.cond(cond, lambda _: row_new, lambda _: row_old, operand=None)
            return rocks.at[i].set(out_row)

        rocks = jax.lax.fori_loop(0, 3, check_rocks, new_rocks)

        return flags, trees, rocks, k

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: GameState, action: int
    ) -> tuple[SkiingObservation, GameState, float, bool, SkiingInfo]:
        #                              -->  --_      \     |     |    /    _-- <--
        side_speed = jnp.array([-1.0, -0.5, -0.333, 0.0, 0.0, 0.333, 0.5, 1.0], jnp.float64)
        #                              -->  --_   \     |    |     /    _--  <--
        down_speed = jnp.array([0.0, 0.5, 0.875, 1.0, 1.0, 0.875, 0.5, 0.0], jnp.float64)

        RECOVERY_FRAMES = jnp.int32(60)
        TREE_X_DIST = jnp.float64(3.0)
        ROCK_X_DIST = jnp.float64(1.0)
        Y_HIT_DIST  = jnp.float64(1.0)

        # 1) Eingabe -> Zielpose
        new_skier_pos = jax.lax.cond(jnp.equal(action, self.consts.LEFT),
                                     lambda _: state.skier_pos - 1,
                                     lambda _: state.skier_pos,
                                     operand=None)
        new_skier_pos = jax.lax.cond(jnp.equal(action, self.consts.RIGHT),
                                     lambda _: state.skier_pos + 1,
                                     lambda _: new_skier_pos,
                                     operand=None)
        skier_pos = jnp.clip(new_skier_pos, 0, 7)

        # Entprellung beibehalten
        skier_pos, direction_change_counter = jax.lax.cond(
            jnp.greater(state.direction_change_counter, 0),
            lambda _: (state.skier_pos, state.direction_change_counter - 1),
            lambda _: (skier_pos, jnp.array(0)),
            operand=None,
        )
        direction_change_counter = jax.lax.cond(
            jnp.logical_and(jnp.not_equal(skier_pos, state.skier_pos),
                            jnp.equal(direction_change_counter, 0)),
            lambda _: jnp.array(16),
            lambda _: direction_change_counter,
            operand=None,
        )

        # 2) Basisgeschwindigkeiten
        dx_target = side_speed.at[skier_pos].get()
        dy_target = down_speed.at[skier_pos].get()

        in_recovery = jnp.greater(state.skier_fell, 0)

        # Recovery: Front, x=0, y wie front
        skier_pos = jax.lax.select(in_recovery, jnp.array(3), skier_pos)
        dx_target = jax.lax.select(in_recovery, jnp.array(0.0, dtype=jnp.float64), dx_target)
        dy_target = jax.lax.select(in_recovery, down_speed.at[3].get(), dy_target)

        new_skier_x_speed_nom = jax.lax.select(
            in_recovery,
            jnp.array(0.0, dtype=jnp.float64),
            state.skier_x_speed + ((dx_target - state.skier_x_speed) * jnp.array(0.1, jnp.float64)),
        )
        new_skier_y_speed_nom = state.skier_y_speed + ((dy_target - state.skier_y_speed) * jnp.array(0.05, jnp.float64))

        min_x = self.config.skier_width / 2
        max_x = self.config.screen_width - self.config.skier_width / 2
        new_x_nom = jnp.clip(state.skier_x + new_skier_x_speed_nom, min_x, max_x)

        # 3) Welt – zunächst "nominal" bewegen (für Kollisionsprüfung),
        #    Freeze wird nach Kollisionsentscheidung angewandt.
        new_trees_nom = state.trees.at[:, 1].add(-new_skier_y_speed_nom)
        new_rocks_nom = state.rocks.at[:, 1].add(-new_skier_y_speed_nom)
        new_flags_nom = state.flags.at[:, 1].add(-new_skier_y_speed_nom)

        # 5) Kollisionen (seitliche Annäherung abfangen)
        skier_y_px = jnp.round(self.config.skier_y)

        def coll_tree(tree_pos, x_d=TREE_X_DIST, y_d=Y_HIT_DIST):
            x = tree_pos[..., 0]
            y = tree_pos[..., 1]
            dx = jnp.abs(new_x_nom - x)
            dy = jnp.abs(jnp.round(skier_y_px) - jnp.round(y))
            return jnp.logical_and(dx <= x_d, dy < y_d)

        def coll_rock(rock_pos, x_d=ROCK_X_DIST, y_d=Y_HIT_DIST):
            x = rock_pos[..., 0]
            y = rock_pos[..., 1]
            dx = jnp.abs(new_x_nom - x)
            dy = jnp.abs(jnp.round(skier_y_px) - jnp.round(y))
            return jnp.logical_and(dx < x_d, dy < y_d)

        def coll_flag(flag_pos, x_d=jnp.float64(1.0), y_d=Y_HIT_DIST):
            x = flag_pos[..., 0]
            y = flag_pos[..., 1]
            dx1 = jnp.abs(new_x_nom - x)
            dx2 = jnp.abs(new_x_nom - (x + self.config.flag_distance))
            dy  = jnp.abs(jnp.round(skier_y_px) - jnp.round(y))
            return jnp.logical_or(jnp.logical_and(dx1 <= x_d, dy < y_d),
                                  jnp.logical_and(dx2 <= x_d, dy < y_d))

        collisions_tree = jax.vmap(coll_tree)(jnp.array(new_trees_nom))
        collisions_rock = jax.vmap(coll_rock)(jnp.array(new_rocks_nom))
        collisions_flag = jax.vmap(coll_flag)(jnp.array(new_flags_nom))
        
        # Während Recovery ODER Cooldown keine neuen Kollisionen auslösen
        ignore_collisions = jnp.logical_or(in_recovery, jnp.greater(state.collision_cooldown, 0))
        collisions_tree = jnp.where(ignore_collisions, jnp.zeros_like(collisions_tree), collisions_tree)
        collisions_rock = jnp.where(ignore_collisions, jnp.zeros_like(collisions_rock), collisions_rock)
        collisions_flag = jnp.where(ignore_collisions, jnp.zeros_like(collisions_flag), collisions_flag)

        collided_tree = jnp.sum(collisions_tree) > 0
        collided_rock = jnp.sum(collisions_rock) > 0
        collided_flag = jnp.sum(collisions_flag) > 0

        # Recovery bei *jeder* Hinderniskollision (Baum/Stein/Flagge)
        start_recovery = jnp.logical_and(
            jnp.logical_not(in_recovery),
            jnp.logical_or(jnp.logical_or(collided_tree, collided_rock), collided_flag),
        )
        freeze = jnp.logical_or(in_recovery, start_recovery)
        # (removed) 6) Minimum-Separation block disabled to avoid pushback.


        # 7) skier_fell & collision_type aktualisieren
        new_skier_fell = jax.lax.cond(
            start_recovery,
            lambda _: RECOVERY_FRAMES,
            lambda _: jax.lax.cond(in_recovery,
                                   lambda __: jnp.maximum(state.skier_fell - 1, 0),
                                   lambda __: jnp.array(0, dtype=jnp.int32),
                                   operand=None),
            operand=None,
        )
        # Kollisions-Entprellung: Nach Recovery-Ende noch kurz Kollisionen ignorieren
        COOLDOWN_FRAMES = jnp.int32(6)
        new_collision_cooldown = jax.lax.cond(
            # Wenn gerade Recovery endet (vorher >0, jetzt ==0) → Cooldown setzen
            jnp.logical_and(in_recovery, jnp.equal(new_skier_fell, 0)),
            lambda _: COOLDOWN_FRAMES,
            # sonst Count-down laufen lassen (nicht negativ)
            lambda _: jnp.maximum(state.collision_cooldown - 1, 0),
            operand=None
        )
        new_collision_type = jax.lax.cond(
            start_recovery,
            lambda _: jnp.where(
                collided_tree, jnp.array(1, dtype=jnp.int32),
                jnp.where(
                    collided_rock, jnp.array(2, dtype=jnp.int32),
                    jnp.where(collided_flag, jnp.array(3, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
                )
            ),
            lambda _: state.collision_type,
            operand=None,
        )
        # Recompute freeze based on updated recovery counter
        freeze = jnp.greater(new_skier_fell, 0)

        # Apply freeze to speeds and world positions
        new_skier_x_speed = jax.lax.select(freeze, jnp.array(0.0, jnp.float64), new_skier_x_speed_nom)
        new_skier_y_speed = jax.lax.select(freeze, jnp.array(0.0, jnp.float64), new_skier_y_speed_nom)
        new_flags = jax.lax.select(freeze, state.flags, new_flags_nom)
        new_trees = jax.lax.select(freeze, state.trees, new_trees_nom)
        new_rocks = jax.lax.select(freeze, state.rocks, new_rocks_nom)
        # Freeze-aware skier X position (no pushback or lateral offset during recovery)
        new_x = jax.lax.select(freeze, state.skier_x, new_x_nom)


        
        # 8) Gate-Scoring & Missed-Penalty (erst JETZT, nach finalen Flag-Positionen)
        left_x  = state.flags[:, 0]
        right_x = left_x + self.config.flag_distance

        eligible = jnp.logical_and(new_x > left_x, new_x < right_x)
        crossed  = jnp.logical_and(state.flags[:, 1] > self.config.skier_y,
                                   new_flags[:, 1] <= self.config.skier_y)
        gate_pass = jnp.logical_and(eligible, jnp.logical_and(crossed, jnp.logical_not(state.flags_passed)))
        flags_passed = jnp.logical_or(state.flags_passed, gate_pass)

        # Despawn/Strafe nur anhand der "gefreezten" (finalen) Flags berechnen
        despawn_mask = new_flags[:, 1] < self.consts.TOP_BORDER
        missed_penalty_mask = jnp.logical_and(despawn_mask, jnp.logical_not(flags_passed))
        missed_penalty_count = jnp.sum(missed_penalty_mask)
        missed_penalty = missed_penalty_count * 300
        flags_passed = jnp.where(despawn_mask, False, flags_passed)

        # Respawns/Despawns nur, wenn NICHT gefreezt
        new_flags, new_trees, new_rocks, new_key = jax.lax.cond(
            freeze,
            lambda _: (new_flags, new_trees, new_rocks, state.key),
            lambda _: self._create_new_objs(state, new_flags, new_trees, new_rocks),
            operand=None
        )

        # Score/Time aktualisieren (nur Gates zählen)
        gates_scored = jnp.sum(gate_pass)
        new_score = state.score - gates_scored
        game_over = jax.lax.cond(jnp.equal(new_score, 0),
                                 lambda _: jnp.array(True),
                                 lambda _: jnp.array(False),
                                 operand=None)
        new_time = jax.lax.cond(
            jnp.greater(state.time, 9223372036854775807 / 2),
            lambda _: jnp.array(0, dtype=jnp.int32),
            lambda _: state.time + 1 + missed_penalty,
            operand=None,
        )

        new_state = GameState(
            skier_x=new_x,
            skier_pos=jnp.array(skier_pos),
            skier_fell=new_skier_fell,
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
            collision_type=new_collision_type,
            flags_passed=flags_passed,
            collision_cooldown=new_collision_cooldown,
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        reward = jnp.asarray(reward, dtype=jnp.float64)
        obs = self._get_observation(new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        # --- CHANGED: cast observation leaves to float64 (score stays int32)

        # Skier (float64 now)
        skier = EntityPosition(
            x=jnp.asarray(state.skier_x, dtype=jnp.float64),           # CHANGED
            y=jnp.asarray(self.config.skier_y, dtype=jnp.float64),     # CHANGED
            width=jnp.asarray(self.config.skier_width, dtype=jnp.float64),   # CHANGED
            height=jnp.asarray(self.config.skier_height, dtype=jnp.float64), # CHANGED
        )

        # Positionsspalten aus dem State holen
        flags_xy_f32 = jnp.asarray(state.flags, dtype=jnp.float64)[..., :2]
        trees_xy_f32 = jnp.asarray(state.trees, dtype=jnp.float64)[..., :2]
        rocks_xy_f32 = jnp.asarray(state.rocks, dtype=jnp.float64)[..., :2]

        # In-Space clippen (gegen Ausreißer wie y=240)
        W = jnp.float64(self.config.screen_width  - 1)
        H = jnp.float64(self.config.screen_height - 1)

        flags_xy_f32 = flags_xy_f32.at[:, 0].set(jnp.clip(flags_xy_f32[:, 0], 0.0, W))
        flags_xy_f32 = flags_xy_f32.at[:, 1].set(jnp.clip(flags_xy_f32[:, 1], 0.0, H))

        trees_xy_f32 = jnp.stack(
            [jnp.clip(trees_xy_f32[:, 0], 0.0, W),
             jnp.clip(trees_xy_f32[:, 1], 0.0, H)],
            axis=1
        )
        rocks_xy_f32 = jnp.stack(
            [jnp.clip(rocks_xy_f32[:, 0], 0.0, W),
             jnp.clip(rocks_xy_f32[:, 1], 0.0, H)],
            axis=1
        )

        # --- CHANGED: upcast clipped positions to float64 for the observation
        flags_xy = jnp.asarray(flags_xy_f32, dtype=jnp.float64)  # CHANGED
        trees_xy = jnp.asarray(trees_xy_f32, dtype=jnp.float64)  # CHANGED
        rocks_xy = jnp.asarray(rocks_xy_f32, dtype=jnp.float64)  # CHANGED

        return SkiingObservation(
            skier=skier,
            flags=flags_xy,
            trees=trees_xy,
            rocks=rocks_xy,
            score=jnp.asarray(state.score, dtype=jnp.int32),
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState, all_rewards: jnp.ndarray) -> SkiingInfo:
        return SkiingInfo(
            time=state.time,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState):
        return previous_state.score - state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        return jnp.equal(state.score, 0)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: GameState, state: GameState) -> jnp.ndarray:
        # Falls keine Liste übergeben wurde → 1-dimensionaler Nullvektor
        if self.reward_funcs is None or len(self.reward_funcs) == 0:
            return jnp.zeros((1,), dtype=jnp.float64)
        # Liste statisch → comprehension ist JIT-ok
        rewards = jnp.array([rf(previous_state, state) for rf in self.reward_funcs], dtype=jnp.float64)
        return rewards


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
    # False = JAX-Bitmap-Font
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
    dst_sub = jax.lax.dynamic_slice(dst_pad, (start_y, start_x, 0), (h, w, 4)).astype(jnp.float64)
    src_sub = src.astype(jnp.float64)

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
    
    is_fallen = jnp.logical_and(
        state.skier_fell > 0,
        jnp.logical_or(state.collision_type == 1,
                       jnp.logical_or(state.collision_type == 2, state.collision_type == 3))
    )
    skier_sprite = jax.lax.cond(is_fallen, lambda _: assets.skier_fallen, lambda _: skier_base, operand=None)

    skier_cx = to_px_x(state.skier_x)
    skier_cy = to_px_y(jnp.array(skier_y))

    # 4) Flags (links & rechts), jede 20. Gate rot, sonst blau
    #    centers sind Pixelcenter; Reihenfolge: Skier -> Flags -> Trees -> Rocks (wie zuvor)
    flags = state.flags  # (N,2) in Game-Koordinaten
    # Nur (x,y) verwenden – robust, egal ob flags (N,2) oder (N,4) ist
    flags_xy = flags[..., :2]  # -> (N,2)
    
    left_px  = jnp.round(flags_xy * scale_factor).astype(jnp.int32)
    right_px = jnp.round((flags_xy + jnp.array([float(flag_distance), 0.0], dtype=jnp.float64))
                         * scale_factor).astype(jnp.int32)
    
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

    # 8) UI/Text direkt in JAX – hier deaktiviert um identische Optik
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
        # Warmup
        # _ = self._render_jit(self._fake_state(), self.assets)

        # Font nur noch für UI-Fallback:
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
        # 2) UI-Text via Pygame oben drauf (identische Optik)
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
    consts = SkiingConstants()

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
            action = consts.NOOP
            VALID_ACTIONS = {consts.NOOP, consts.LEFT, consts.RIGHT}
            if keys[pygame.K_a]:
                action = consts.LEFT
            elif keys[pygame.K_d]:
                action = consts.RIGHT
            if action not in VALID_ACTIONS:
                action = consts.NOOP

            obs, state, reward, done, info = game.step(state, action)
            renderer.render(state)

            clock.tick(60)

        renderer.close()


if __name__ == "__main__":
    main()
