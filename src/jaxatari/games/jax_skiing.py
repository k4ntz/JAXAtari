from functools import partial
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
    flag_width: int = 10
    flag_height: int = 28
    flag_distance: int = 20
    gate_vertical_spacing: int = 90  # fixed vertical spacing between gates (in pixels)
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
    gates_seen: chex.Array  # Anzahl der bereits verarbeiteten Gates (despawned)


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
        return spaces.Discrete(8)

    def observation_space(self):
        c = self.config

        skier_space = spaces.Dict(collections.OrderedDict({
            "x":      spaces.Box(low=0.0,               high=float(c.screen_width),  shape=(), dtype=jnp.float32),
            "y":      spaces.Box(low=0.0,               high=float(c.screen_height), shape=(), dtype=jnp.float32),
            "width":  spaces.Box(low=float(c.skier_width),  high=float(c.skier_width),  shape=(), dtype=jnp.float32),
            "height": spaces.Box(low=float(c.skier_height), high=float(c.skier_height), shape=(), dtype=jnp.float32),
        }))

        flags_space = spaces.Box(low=[0.0, 0.0],
                                 high=[float(c.screen_width), float(c.screen_height)],
                                 shape=(c.max_num_flags, 2), dtype=jnp.float32)
        trees_space = spaces.Box(low=[0.0, 0.0],
                                 high=[float(c.screen_width), float(c.screen_height)],
                                 shape=(c.max_num_trees, 2), dtype=jnp.float32)
        rocks_space = spaces.Box(low=[0.0, 0.0],
                                 high=[float(c.screen_width), float(c.screen_height)],
                                 shape=(c.max_num_rocks, 2), dtype=jnp.float32)

        # nachher (alles float32):
        score_space = spaces.Box(low=jnp.array(0.0, dtype=jnp.float32),
                                 high=jnp.array(1_000_000.0, dtype=jnp.float32),
                                 shape=(), dtype=jnp.float32)

        return spaces.Dict(collections.OrderedDict({
            "skier": skier_space, "flags": flags_space, "trees": trees_space, "rocks": rocks_space, "score": score_space,
        }))

    def image_space(self):
        c = self.config
        return spaces.Box(low=0, high=255, shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def obs_to_flat_array(self, obs: SkiingObservation) -> jnp.ndarray:
        skier_vec  = jnp.array([obs.skier.x, obs.skier.y, obs.skier.width, obs.skier.height],
                               dtype=jnp.float32).reshape(-1)
        flags_flat = jnp.array(obs.flags, dtype=jnp.float32).reshape(-1)
        trees_flat = jnp.array(obs.trees, dtype=jnp.float32).reshape(-1)
        rocks_flat = jnp.array(obs.rocks, dtype=jnp.float32).reshape(-1)
        score_flat = jnp.array(obs.score, dtype=jnp.float32).reshape(-1)
        return jnp.concatenate([skier_vec, flags_flat, trees_flat, rocks_flat, score_flat], axis=0)

    def reset(self, key: jax.random.PRNGKey = jax.random.key(1701)) -> Tuple[SkiingObservation, GameState]:
        """Initialize a new game state deterministically from `key`."""
        c = self.config
        k_flags, k_trees, k_rocks, new_key = jax.random.split(key, 4)

        # Flags: y gleichmäßig verteilt, x zufällig
        y_spacing = float(c.gate_vertical_spacing)
        i = jnp.arange(c.max_num_flags, dtype=jnp.float32)
        flags_y = (i + 1.0) * y_spacing + float(c.flag_height)
        flags_x = jax.random.randint(
            k_flags, (c.max_num_flags,),
            minval=int(c.flag_width),
            maxval=int(c.screen_width - c.flag_width - c.flag_distance) + 1
        ).astype(jnp.float32)
        flags = jnp.stack([
            flags_x, flags_y,
            jnp.full((c.max_num_flags,), float(c.flag_width),  dtype=jnp.float32),
            jnp.full((c.max_num_flags,), float(c.flag_height), dtype=jnp.float32)
        ], axis=1)

        
        # Trees
        # Enforce min horizontal separation and no overlap among trees on spawn
        trees_x = jax.random.randint(
            k_trees, (c.max_num_trees,),
            minval=int(c.tree_width),
            maxval=int(c.screen_width - c.tree_width) + 1
        ).astype(jnp.float32)
        trees_y = jax.random.randint(
            k_trees, (c.max_num_trees,),
            minval=int(c.tree_height),
            maxval=int(c.screen_height - c.tree_height) + 1
        ).astype(jnp.float32)

        min_sep_tree = (jnp.float32(c.tree_width) + jnp.float32(c.tree_width)) * 0.5 + jnp.float32(8.0)
        xmin = jnp.float32(c.tree_width)
        xmax = jnp.float32(c.screen_width - c.tree_width)

        def adj_tree_i(i, tx):
            x0 = tx[i]
            x_adj = _enforce_min_sep_x(x0, tx, min_sep_tree, xmin, xmax, n_valid=jnp.array(i, dtype=jnp.int32))
            return tx.at[i].set(x_adj)

        trees_x = jax.lax.fori_loop(0, c.max_num_trees, adj_tree_i, trees_x)

        trees = jnp.stack([
            trees_x, trees_y,
            jnp.full((c.max_num_trees,), float(c.tree_width),  dtype=jnp.float32),
            jnp.full((c.max_num_trees,), float(c.tree_height), dtype=jnp.float32)
        ], axis=1)



        # Rocks
        rocks_x = jax.random.randint(
            k_rocks, (c.max_num_rocks,),
            minval=int(c.rock_width),
            maxval=int(c.screen_width - c.rock_width) + 1
        ).astype(jnp.float32)
        rocks_y = jax.random.randint(
            k_rocks, (c.max_num_rocks,),
            minval=int(c.rock_height),
            maxval=int(c.screen_height - c.rock_height) + 1
        ).astype(jnp.float32)

        # Enforce separation from trees and already placed rocks
        min_sep_rock_tree = (jnp.float32(c.rock_width) + jnp.float32(c.tree_width)) * 0.5 + jnp.float32(8.0)
        min_sep_rock_rock = (jnp.float32(c.rock_width) + jnp.float32(c.rock_width)) * 0.5 + jnp.float32(8.0)
        xmin_r = jnp.float32(c.rock_width)
        xmax_r = jnp.float32(c.screen_width - c.rock_width)

        tree_xs_fixed = trees[:, 0]

        def adj_rock_i(i, rx):
            x0 = rx[i]
            # push from all trees (all are valid)
            x1 = _enforce_min_sep_x(x0, tree_xs_fixed, min_sep_rock_tree, xmin_r, xmax_r, n_valid=jnp.array(tree_xs_fixed.shape[0], dtype=jnp.int32))
            # then from previously placed rocks (indices < i)
            x2 = _enforce_min_sep_x(x1, rx, min_sep_rock_rock, xmin_r, xmax_r, n_valid=jnp.array(i, dtype=jnp.int32))
            return rx.at[i].set(x2)

        rocks_x = jax.lax.fori_loop(0, c.max_num_rocks, adj_rock_i, rocks_x)

        rocks = jnp.stack([
            rocks_x, rocks_y,
            jnp.full((c.max_num_rocks,), float(c.rock_width),  dtype=jnp.float32),
            jnp.full((c.max_num_rocks,), float(c.rock_height), dtype=jnp.float32)
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
        
            gates_seen=jnp.array(0, dtype=jnp.int32),
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
            # neue x-Position innerhalb des gültigen Bereichs
            x_flag = jax.random.randint(
                k1.at[i].get(), [],
                self.config.flag_width,
                self.config.screen_width - self.config.flag_width - self.config.flag_distance
            ).astype(jnp.float32)

            # Konstanter Vertikalabstand: immer hinter die aktuell tiefste Flagge spawnen
            # Berücksichtigt sowohl bereits neu gesetzte Flags (new_flags) als auch bestehende (flags)
            base_existing = jnp.maximum(jnp.max(new_flags[:, 1]), jnp.max(flags[:, 1]))
            y = base_existing + jnp.float32(self.config.gate_vertical_spacing)

            row_old = flags.at[i].get()  # Shape (2,) oder (4,)
            row_new = row_old.at[0].set(x_flag).at[1].set(y)

            # Nur respawnen, wenn Flagge oberhalb TOP_BORDER despawned ist
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
            ).astype(jnp.float32)
            y = (jnp.max(new_flags[:, 1]) + jnp.float32(self.config.gate_vertical_spacing) / 2.0)

            # Enforce min separation from existing trees and rocks on respawn
            min_sep_tree_tree = (jnp.float32(self.config.tree_width) + jnp.float32(self.config.tree_width)) * 0.5 + jnp.float32(8.0)
            min_sep_tree_rock = (jnp.float32(self.config.tree_width) + jnp.float32(self.config.rock_width)) * 0.5 + jnp.float32(8.0)
            xmin_t = jnp.float32(self.config.tree_width)
            xmax_t = jnp.float32(self.config.screen_width - self.config.tree_width)
            taken_from_trees = trees[:, 0]
            taken_from_rocks = new_rocks[:, 0]
            x_tree = _enforce_min_sep_x(x_tree, taken_from_trees, min_sep_tree_tree, xmin_t, xmax_t, n_valid=jnp.array(i, dtype=jnp.int32))
            x_tree = _enforce_min_sep_x(x_tree, taken_from_rocks, min_sep_tree_rock, xmin_t, xmax_t, n_valid=jnp.array(taken_from_rocks.shape[0], dtype=jnp.int32))

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
            ).astype(jnp.float32)
            y = (jnp.max(new_flags[:, 1]) + jnp.float32(self.config.gate_vertical_spacing) / 2.0)

            # Enforce min separation from existing rocks and trees on respawn
            min_sep_rock_rock = (jnp.float32(self.config.rock_width) + jnp.float32(self.config.rock_width)) * 0.5 + jnp.float32(8.0)
            min_sep_rock_tree = (jnp.float32(self.config.rock_width) + jnp.float32(self.config.tree_width)) * 0.5 + jnp.float32(8.0)
            xmin_r = jnp.float32(self.config.rock_width)
            xmax_r = jnp.float32(self.config.screen_width - self.config.rock_width)
            taken_from_rocks = rocks[:, 0]
            taken_from_trees = new_trees[:, 0]
            x_rock = _enforce_min_sep_x(x_rock, taken_from_rocks, min_sep_rock_rock, xmin_r, xmax_r, n_valid=jnp.array(taken_from_rocks.shape[0], dtype=jnp.int32))
            x_rock = _enforce_min_sep_x(x_rock, taken_from_trees, min_sep_rock_tree, xmin_r, xmax_r, n_valid=jnp.array(taken_from_trees.shape[0], dtype=jnp.int32))

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
        side_speed = jnp.array([-1.0, -0.5, -0.333, 0.0, 0.0, 0.333, 0.5, 1.0], jnp.float32)
        #                              -->  --_   \     |    |     /    _--  <--
        down_speed = jnp.array([0.0, 0.5, 0.875, 1.0, 1.0, 0.875, 0.5, 0.0], jnp.float32)

        RECOVERY_FRAMES = jnp.int32(60)
        TREE_X_DIST = jnp.float32(3.0)
        ROCK_X_DIST = jnp.float32(1.0)
        Y_HIT_DIST  = jnp.float32(1.0)

        # 1) Eingabe -> Zielpose

        # Normalize action from get_human_action(): accept only A/D (JAXAtariAction LEFT=4, RIGHT=3).
        # Any other input (including SPACE/FIRE) becomes NOOP.
        is_left  = jnp.equal(action, jnp.int32(4))  # external LEFT (A)
        is_right = jnp.equal(action, jnp.int32(3)) # external RIGHT (D)
        norm_action = jax.lax.select(is_left,  self.consts.LEFT,
                    jax.lax.select(is_right, self.consts.RIGHT, self.consts.NOOP))
        new_skier_pos = jax.lax.cond(jnp.equal(norm_action, self.consts.LEFT),
                                     lambda _: state.skier_pos - 1,
                                     lambda _: state.skier_pos,
                                     operand=None)
        new_skier_pos = jax.lax.cond(jnp.equal(norm_action, self.consts.RIGHT),
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
        dx_target = jax.lax.select(in_recovery, jnp.array(0.0, dtype=jnp.float32), dx_target)
        dy_target = jax.lax.select(in_recovery, down_speed.at[3].get(), dy_target)

        new_skier_x_speed_nom = jax.lax.select(
            in_recovery,
            jnp.array(0.0, dtype=jnp.float32),
            dx_target * jnp.array(0.3, jnp.float32),  # no acceleration; slightly slower lateral speed
        )
        new_skier_y_speed_nom = state.skier_y_speed + ((dy_target - state.skier_y_speed) * jnp.array(0.05, jnp.float32))

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

        def coll_flag(flag_pos, x_d=jnp.float32(1.0), y_d=Y_HIT_DIST):
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
        # --- PATCH: make rocks non-collidable (requested)
        # We keep spawning/rendering rocks, but they never register a hit.
        collisions_rock = jnp.zeros_like(collisions_rock, dtype=collisions_rock.dtype)
        
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

        # Zusätzlich: Im Startframe der Recovery Kollisionen ignorieren,
        # um Doppel-Treffer ohne visuelle Separation zu vermeiden.
        mask_now = jnp.logical_or(ignore_collisions, start_recovery)
        collisions_tree = jnp.where(mask_now, jnp.zeros_like(collisions_tree), collisions_tree)
        collisions_rock = jnp.where(mask_now, jnp.zeros_like(collisions_rock), collisions_rock)
        collisions_flag = jnp.where(mask_now, jnp.zeros_like(collisions_flag), collisions_flag)
        # Freeze ohne Repositionierung: Hindernisse bleiben an Ort und Stelle
        freeze_flags = state.flags
        freeze_trees = state.trees
        freeze_rocks = state.rocks

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
        COOLDOWN_FRAMES = jnp.int32(10)
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
        new_skier_x_speed = jax.lax.select(freeze, jnp.array(0.0, jnp.float32), new_skier_x_speed_nom)
        new_skier_y_speed = jax.lax.select(freeze, jnp.array(0.0, jnp.float32), new_skier_y_speed_nom)
        new_flags = jax.lax.select(freeze, freeze_flags, new_flags_nom)
        new_trees = jax.lax.select(freeze, freeze_trees, new_trees_nom)
        new_rocks = jax.lax.select(freeze, freeze_rocks, new_rocks_nom)
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
        # missed_penalty_mask = jnp.logical_and(despawn_mask, jnp.logical_not(flags_passed))
        # missed_penalty_count = jnp.sum(missed_penalty_mask)
        # missed_penalty = missed_penalty_count * 300
        flags_passed = jnp.where(despawn_mask, False, flags_passed)


        # Gates-Zähler inkrementieren: jedes despawnte Gate zählt als gesehen
        gates_increment = jnp.sum(despawn_mask).astype(jnp.int32)
        new_gates_seen = state.gates_seen + gates_increment
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
        game_over = jnp.greater_equal(new_gates_seen, 20)
        new_time = jax.lax.cond(
            jnp.greater(state.time, 9223372036854775807 / 2),
            lambda _: jnp.array(0, dtype=jnp.int32),
            # lambda _: state.time + 1 + missed_penalty,
            lambda _: state.time + 1,
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
            gates_seen=new_gates_seen,
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        reward = jnp.array(reward, dtype=jnp.float32)
        obs = self._get_observation(new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        # --- CHANGED: cast observation leaves to float64 (score stays int32)

        # Skier (float64 now)
        skier = EntityPosition(
            x=jnp.array(state.skier_x, dtype=jnp.float32),           # CHANGED
            y=jnp.array(self.config.skier_y, dtype=jnp.float32),     # CHANGED
            width=jnp.array(self.config.skier_width, dtype=jnp.float32),   # CHANGED
            height=jnp.array(self.config.skier_height, dtype=jnp.float32), # CHANGED
        )

        # Positionsspalten aus dem State holen
        flags_xy_f32 = jnp.array(state.flags, dtype=jnp.float32)[..., :2]
        trees_xy_f32 = jnp.array(state.trees, dtype=jnp.float32)[..., :2]
        rocks_xy_f32 = jnp.array(state.rocks, dtype=jnp.float32)[..., :2]

        # In-Space clippen (gegen Ausreißer wie y=240)
        W = jnp.float32(self.config.screen_width  - 1)
        H = jnp.float32(self.config.screen_height - 1)

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

        flags_xy = jnp.array(flags_xy_f32, dtype=jnp.float32)
        trees_xy = jnp.array(trees_xy_f32, dtype=jnp.float32)
        rocks_xy = jnp.array(rocks_xy_f32, dtype=jnp.float32) 

        return SkiingObservation(
            skier=skier,
            flags=flags_xy,
            trees=trees_xy,
            rocks=rocks_xy,
            score=jnp.array(state.score, dtype=jnp.float32),
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState, all_rewards: jnp.ndarray | None = None) -> SkiingInfo:
        # Accept optional all_rewards so wrappers that call _get_info(state) still work.
        if all_rewards is None:
            all_rewards = jnp.zeros((1,), dtype=jnp.float32)
        return SkiingInfo(
            time=state.time,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState):
        return previous_state.score - state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        return jnp.greater_equal(state.gates_seen, 20)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: GameState, state: GameState) -> jnp.ndarray:
        # Falls keine Liste übergeben wurde → 1-dimensionaler Nullvektor
        if self.reward_funcs is None or len(self.reward_funcs) == 0:
            return jnp.zeros((1,), dtype=jnp.float32)
        # Liste statisch → comprehension ist JIT-ok
        rewards = jnp.array([rf(previous_state, state) for rf in self.reward_funcs], dtype=jnp.float32)
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
    use_pygame_text: bool = False

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
    return jax.device_put(jnp.array(arr, dtype=jnp.uint8))

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
    rgb_arr = jnp.array(jnp.array(rgb, dtype=jnp.uint8))[None, None, :].astype(jnp.uint8)
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

    top  = jnp.array(top,  jnp.int32)
    left = jnp.array(left, jnp.int32)

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


def _enforce_min_sep_x(x_init: jnp.ndarray, taken_xs: jnp.ndarray, min_sep: jnp.ndarray, xmin: jnp.ndarray, xmax: jnp.ndarray, n_valid: jnp.ndarray) -> jnp.ndarray:
    """Shift x_init away from up to the first `n_valid` entries in taken_xs so that |x - taken_x| >= min_sep.
    Uses fixed-size fori_loop (JAX-friendly)."""
    def body(j, x_curr):
        tx = taken_xs[j]
        dx = x_curr - tx
        too_close = jnp.abs(dx) < min_sep
        apply = jnp.less(j, n_valid)
        direction = jnp.where(dx >= 0.0, 1.0, -1.0)
        candidate = jnp.clip(tx + direction * min_sep, xmin, xmax)
        x_next = jnp.where(jnp.logical_and(apply, too_close), candidate, x_curr)
        return x_next
    x = jax.lax.fori_loop(0, taken_xs.shape[0], body, x_init)
    return jnp.clip(x, xmin, xmax)


    def body(i, x_curr):
        tx = taken_xs[i]
        dx = x_curr - tx
        too_close = jnp.abs(dx) < min_sep
        # Push to the side where we already are farther, default to right if exactly equal
        direction = jnp.where(dx >= 0.0, 1.0, -1.0)
        candidate = tx + direction * min_sep
        # Clamp to bounds
        candidate = jnp.clip(candidate, xmin, xmax)
        return jnp.where(too_close, candidate, x_curr)

    x = jax.lax.fori_loop(0, n, body, x)
    # final clamp
    x = jnp.clip(x, xmin, xmax)
    return x

# ---- Pure JAX Renderer -----------------------------------------------------------

# ---- Minimal JAX bitmap font (3x5) for digits/time UI -----------------------
# Glyph order: '0'-'9' -> 0..9, ':' -> 10, '.' -> 11, ' ' (blank) -> 12
_GLYPHS_BITS = jnp.array([
    # 0
    [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],
    # 1
    [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],
    # 2
    [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],
    # 3
    [[1,1,1],[0,0,1],[0,1,1],[0,0,1],[1,1,1]],
    # 4
    [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],
    # 5
    [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],
    # 6
    [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],
    # 7
    [[1,1,1],[0,0,1],[0,1,0],[1,0,0],[1,0,0]],
    # 8
    [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],
    # 9
    [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],
    # :
    [[0,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,0]],
    # .
    [[0,0,0],[0,0,0],[0,0,0],[0,1,0],[0,1,0]],
], dtype=jnp.uint8)

def _glyph_rgba(scale: int = 2) -> jnp.ndarray:
    """Return RGBA sprites for 12 glyphs, scaled with nearest-neighbor."""
    H, W = _GLYPHS_BITS.shape[1], _GLYPHS_BITS.shape[2]
    def upsample(bits):
        b = bits.astype(jnp.uint8)
        one = jnp.ones((scale, scale), dtype=jnp.uint8)
        up = jnp.kron(b, one)  # (H*scale, W*scale)
        rgb = jnp.zeros((up.shape[0], up.shape[1], 3), dtype=jnp.uint8)
        a = (up * 255).astype(jnp.uint8)
        return jnp.concatenate([rgb, a[..., None]], axis=-1)  # RGBA
    sprites = jax.vmap(upsample)(_GLYPHS_BITS)  # (12, hs, ws, 4)
    return sprites

def _center_positions(num_glyphs: int, glyph_w: int, y_top: int, screen_w: int, spacing: int=1):
    total_w = num_glyphs * glyph_w + (num_glyphs - 1) * spacing
    left = (screen_w - total_w) // 2
    xs = jnp.arange(num_glyphs) * (glyph_w + spacing) + left + glyph_w // 2
    ys = jnp.full((num_glyphs,), y_top + (glyph_w // 2), dtype=jnp.int32)
    centers = jnp.stack([xs.astype(jnp.int32), ys], axis=1)
    return centers

def _draw_digits_line(frame: jnp.ndarray, digit_codes: jnp.ndarray, y_top: int, scale: int = 2) -> jnp.ndarray:
    sprites = _glyph_rgba(scale)
    gh = sprites.shape[1]
    gw = sprites.shape[2]
    centers = _center_positions(digit_codes.shape[0], gw, y_top, frame.shape[1], spacing=1)
    chosen = sprites[digit_codes]
    return _scan_blit(frame, chosen, centers)

def _format_score_digits(score: jnp.ndarray) -> jnp.ndarray:
    # Two digits (00..99), clipped
    s_val = jnp.clip(score.astype(jnp.int32), 0, 99)
    tens = (s_val // 10) % 10
    ones = s_val % 10
    return jnp.stack([tens, ones], axis=0)
    val_out, arr_out = jax.lax.fori_loop(0, MAXD, body, (v, out))
    return arr_out

def _format_time_digits(t: jnp.ndarray) -> jnp.ndarray:
    # Zeit aus Frames berechnen: 60 FPS -> Sekunden (bei anderem Takt FPS anpassen)
    t = jnp.maximum(t.astype(jnp.float32), 0.0)
    FPS = jnp.float32(60.0)
    seconds_total = t / FPS

    # M:SS:MS  (M = Minuten einstellig, SS = Sekunden zweistellig, MS = Millisekunden zweistellig)
    minutes_digit = (jnp.floor(seconds_total / 60.0).astype(jnp.int32)) % 10
    seconds_int   = jnp.floor(jnp.mod(seconds_total, 60.0)).astype(jnp.int32)
    ms_int        = jnp.floor((seconds_total - jnp.floor(seconds_total)) * 1000.0).astype(jnp.int32)  # 0..999

    s_t  = (seconds_int // 10) % 10
    s_o  = seconds_int % 10
    ms_t = (ms_int // 10) % 10      # Zehner der Millisekunden (letzte zwei Stellen)
    ms_o = ms_int % 10

    colon = jnp.int32(10)  # ':'-Glyph
    return jnp.stack([minutes_digit, colon, s_t, s_o, colon, ms_t, ms_o], axis=0)
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
    right_px = jnp.round((flags_xy + jnp.array([float(flag_distance), 0.0], dtype=jnp.float32))
                         * scale_factor).astype(jnp.int32)
    
    # The 20th gate should be red. Identify the visible gate closest to the skier
    # and color it red only when 19 gates have already been seen.
    n_flags = flags.shape[0]
    dy_to_skier = jnp.abs(flags_xy[:, 1] - jnp.float32(skier_y))
    closest_idx = jnp.argmin(dy_to_skier)
    is_twentieth = jnp.greater_equal(state.gates_seen, jnp.int32(19))
    is_red = jnp.zeros((n_flags,), dtype=bool).at[closest_idx].set(is_twentieth)

    # For each gate we draw two flags (left & right) with the same color.
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

    # 9) Optional UI: score (top line) and time (second line), centered
    if draw_ui_jax:
        score_digits = _format_score_digits(state.score)
        time_digits  = _format_time_digits(state.time)
        frame = _draw_digits_line(frame, score_digits, y_top=2, scale=2)
        frame = _draw_digits_line(frame, time_digits,  y_top=2 + 5*2 + 2, scale=2)  # place below first line

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
            draw_ui_jax    = True,
        )

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state) -> jnp.ndarray:
        rgba = self._render_fn(state, self.assets)   # (210,160,4) uint8
        return rgba[..., :3]                         # -> (210,160,3) RGB