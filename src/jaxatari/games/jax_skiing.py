from functools import partial
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Tuple, NamedTuple, Callable, Sequence, Optional
import os
import numpy as np
import collections
import jaxatari.rendering.jax_rendering_utils as render_utils 
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces

def _create_static_procedural_sprites() -> dict:
    """Creates procedural sprites that don't depend on dynamic values."""
    # Procedural white background
    white_bg = jnp.full((210, 160, 4), 255, dtype=jnp.uint8)
    
    # Procedurally create text colors
    text_colors = jnp.array([
        [0, 0, 0, 255], # Black text
        [255, 0, 0, 255] # Red text
    ], dtype=jnp.uint8).reshape(-1, 1, 1, 4)
    
    # Create procedural digits from _GLYPHS_BITS
    def upsample(bits):
        b = bits.astype(jnp.uint8)
        scale = 2
        one = jnp.ones((scale, scale), dtype=jnp.uint8)
        up = jnp.kron(b, one)  # (H*scale, W*scale)
        
        # Create an RGBA sprite: Black (0,0,0) where bits=1, Transparent (0,0,0,0) where bits=0
        color = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        transparent = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)
        
        # Broadcast up to (H*scale, W*scale, 1) for comparison
        up_mask = up[..., None] > 0  # (H*scale, W*scale, 1) boolean
        return jnp.where(up_mask, color, transparent)
    
    # _GLYPHS_BITS is defined later in the file, but we'll reference it
    # For now, we'll create the digits in the function that uses it
    procedural_digits = None  # Will be set in _get_default_asset_config
    
    return {
        'background': white_bg,
        'ui_colors': text_colors,
    }

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Skiing.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    Note: Flag sprites need to be loaded from files, so they're added in renderer.
    """
    static_procedural = _create_static_procedural_sprites()
    
    # Create procedural digits from _GLYPHS_BITS
    def upsample(bits):
        b = bits.astype(jnp.uint8)
        scale = 2
        one = jnp.ones((scale, scale), dtype=jnp.uint8)
        up = jnp.kron(b, one)  # (H*scale, W*scale)
        
        color = jnp.array([0, 0, 0, 255], dtype=jnp.uint8)
        transparent = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)
        up_mask = up[..., None] > 0
        return jnp.where(up_mask, color, transparent)
    
    # Import _GLYPHS_BITS - it's defined later in the file
    # We need to reference it, but it's defined after this function
    # So we'll create it inline here
    _GLYPHS_BITS = jnp.array([
        [[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]],  # 0
        [[0,1,0],[1,1,0],[0,1,0],[0,1,0],[1,1,1]],  # 1
        [[1,1,1],[0,0,1],[1,1,1],[1,0,0],[1,1,1]],  # 2
        [[1,1,1],[0,0,1],[0,1,1],[0,0,1],[1,1,1]],  # 3
        [[1,0,1],[1,0,1],[1,1,1],[0,0,1],[0,0,1]],  # 4
        [[1,1,1],[1,0,0],[1,1,1],[0,0,1],[1,1,1]],  # 5
        [[1,1,1],[1,0,0],[1,1,1],[1,0,1],[1,1,1]],  # 6
        [[1,1,1],[0,0,1],[0,1,0],[1,0,0],[1,0,0]],  # 7
        [[1,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1]],  # 8
        [[1,1,1],[1,0,1],[1,1,1],[0,0,1],[1,1,1]],  # 9
        [[0,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,0]],  # : (10)
        [[0,0,0],[0,0,0],[0,0,0],[0,1,0],[0,1,0]],  # . (11)
    ], dtype=jnp.uint8)
    
    procedural_digits = jax.vmap(upsample)(_GLYPHS_BITS)  # (12, hs, ws, 4)
    
    return (
        # Background
        {'name': 'background', 'type': 'background', 'data': static_procedural['background']},

        # Skier Sprites (as a group for padding)
        {'name': 'skier_group', 'type': 'group', 'files': [
            "skiier_right.npy", # 0: skier_left
            "skiier_front.npy", # 1: skier_front
            "skiier_left.npy",  # 2: skier_right
            "skier_fallen.npy"  # 3: skier_fallen
        ]},

        # Obstacles
        {'name': 'tree', 'type': 'single', 'file': 'tree.npy'},
        {'name': 'rock', 'type': 'single', 'file': 'stone.npy'},
        
        # UI
        {'name': 'digits', 'type': 'procedural', 'data': procedural_digits},
        {'name': 'ui_colors', 'type': 'procedural', 'data': static_procedural['ui_colors']}
    )

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
    gate_vertical_spacing: int = 90
    tree_width: int = 16
    tree_height: int = 30
    rock_width: int = 16
    rock_height: int = 7
    # Separation margins (in pixels) used in X-separation checks
    sep_margin_tree_tree: float = 14.0
    sep_margin_rock_rock: float = 12.0
    sep_margin_tree_rock: float = 14.0
    # Small Y offset between rocks and trees to avoid identical rows
    min_y_offset_tree_vs_rock: float = 8.0
    max_num_flags: int = 2
    max_num_trees: int = 4
    max_num_rocks: int = 3
    speed: float = 1.0

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()


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


class JaxSkiing(JaxEnvironment[GameState, SkiingObservation, SkiingInfo, SkiingConstants]):
    def __init__(self, consts: SkiingConstants | None = None):
        consts = consts or SkiingConstants()
        super().__init__(consts)
        self.config = GameConfig()
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
        # [deterministic]         flags_x = jax.random.randint(
        # [deterministic]             k_flags, (c.max_num_flags,),
        # [deterministic]             minval=int(c.flag_width),
        # [deterministic]             maxval=int(c.screen_width - c.flag_width - c.flag_distance) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic left-flag x-position: scan across lane with fixed step
        min_fx = jnp.int32(self.config.flag_width)
        max_fx = jnp.int32(self.config.screen_width - self.config.flag_width - self.config.flag_distance)
        span_fx = max_fx - min_fx + 1
        # simple sawtooth pattern using step 13
        flags_x = (min_fx + ((jnp.arange(c.max_num_flags, dtype=jnp.int32) * 13) % span_fx)).astype(jnp.float32)
        flags = jnp.stack([
            flags_x, flags_y,
            jnp.full((c.max_num_flags,), float(c.flag_width),  dtype=jnp.float32),
            jnp.full((c.max_num_flags,), float(c.flag_height), dtype=jnp.float32)
        ], axis=1)

        
        # Trees
        # Enforce min horizontal separation and no overlap among trees on spawn
        # [deterministic]         trees_x = jax.random.randint(
        # [deterministic]             k_trees, (c.max_num_trees,),
        # [deterministic]             minval=int(c.tree_width),
        # [deterministic]             maxval=int(c.screen_width - c.tree_width) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic tree x-position: stride 17 across lane
        min_tx = jnp.int32(self.config.tree_width)
        max_tx = jnp.int32(self.config.screen_width - self.config.tree_width)
        span_tx = max_tx - min_tx + 1
        trees_x = (min_tx + ((jnp.arange(c.max_num_trees, dtype=jnp.int32) * 17) % span_tx)).astype(jnp.float32)
        # [deterministic]         trees_y = jax.random.randint(
        # [deterministic]             k_trees, (c.max_num_trees,),
        # [deterministic]             minval=int(c.tree_height),
        # [deterministic]             maxval=int(c.screen_height - c.tree_height) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic tree y-position: evenly spaced between gates (already unique)
        base_ty = flags_y[0] - jnp.float32(self.config.gate_vertical_spacing) * 0.5
        step_ty = jnp.float32(self.config.gate_vertical_spacing) / jnp.float32(max(1, c.max_num_trees))
        trees_y = (base_ty + jnp.arange(c.max_num_trees, dtype=jnp.float32) * step_ty).astype(jnp.float32)
        trees_y = jnp.round(trees_y).astype(jnp.float32)

        min_sep_tree = 0.5*(jnp.float32(c.tree_width)+jnp.float32(c.tree_width)) + jnp.float32(c.sep_margin_tree_tree)
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
        # [deterministic]         rocks_x = jax.random.randint(
        # [deterministic]             k_rocks, (c.max_num_rocks,),
        # [deterministic]             minval=int(c.rock_width),
        # [deterministic]             maxval=int(c.screen_width - c.rock_width) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic rock x-position: stride 19 across lane
        min_rx = jnp.int32(self.config.rock_width)
        max_rx = jnp.int32(self.config.screen_width - self.config.rock_width)
        span_rx = max_rx - min_rx + 1
        rocks_x = (min_rx + ((jnp.arange(c.max_num_rocks, dtype=jnp.int32) * 19) % span_rx)).astype(jnp.float32)
        # [deterministic]         rocks_y = jax.random.randint(
        # [deterministic]             k_rocks, (c.max_num_rocks,),
        # [deterministic]             minval=int(c.rock_height),
        # [deterministic]             maxval=int(c.screen_height - c.rock_height) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic rock y-position: offset from trees_y for alternation
        base_ry = flags_y[0] - jnp.float32(self.config.gate_vertical_spacing) * 0.25
        step_ry = jnp.float32(self.config.gate_vertical_spacing) / jnp.float32(max(1, c.max_num_rocks))
        rocks_y = (base_ry + jnp.arange(c.max_num_rocks, dtype=jnp.float32) * step_ry).astype(jnp.float32)
        rocks_y = jnp.round(rocks_y).astype(jnp.float32)
        # Enforce separation from trees and already placed rocks
        min_sep_rock_tree = 0.5*(jnp.float32(self.config.rock_width)+jnp.float32(self.config.tree_width)) + jnp.float32(self.config.sep_margin_tree_rock)
        min_sep_rock_rock = 0.5*(jnp.float32(self.config.rock_width)+jnp.float32(self.config.rock_width)) + jnp.float32(self.config.sep_margin_rock_rock)
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
            skier_y_speed=jnp.array(0.0),
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
        # [deterministic]         k, k1, k2, k3, k4 = jax.random.split(state.key, num=5)  # not used (deterministic respawn)
        # [deterministic]         k1 = jnp.array([k1, k2, k3, k4])
        k = state.key

        def check_flags(i, flags):
            # neue x-Position innerhalb des gültigen Bereichs
            # [deterministic]             x_flag = jax.random.randint(
            # [deterministic]                 k1.at[i].get(), [],
            # [deterministic]                 self.config.flag_width,
            # [deterministic]                 self.config.screen_width - self.config.flag_width - self.config.flag_distance
            # [deterministic]             ).astype(jnp.float32)
            # Deterministic left-flag x based on gates_seen and loop index
            min_fx = jnp.int32(self.config.flag_width)
            max_fx = jnp.int32(self.config.screen_width - self.config.flag_width - self.config.flag_distance)
            span_fx = max_fx - min_fx + 1
            step_fx = 13
            x_flag = (min_fx + (((state.gates_seen + i) * step_fx) % span_fx)).astype(jnp.float32)

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
        # [deterministic]         k, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(k, 9)
        # [deterministic]         k1 = jnp.array([k1, k2, k3, k4, k5, k6, k7, k8])

        def check_trees(i, trees):
            # [deterministic]             x_tree = jax.random.randint(
            # [deterministic]                 k1.at[i].get(), [], 
            # [deterministic]                 self.config.tree_width,
            # [deterministic]                 self.config.screen_width - self.config.tree_width
            # [deterministic]             ).astype(jnp.float32)
            # Deterministic tree x based on gates_seen and index i, different step to avoid overlap
            min_tx = jnp.int32(self.config.tree_width)
            max_tx = jnp.int32(self.config.screen_width - self.config.tree_width)
            span_tx = max_tx - min_tx + 1
            step_tx = 17
            x_tree = (min_tx + (((state.gates_seen + i) * step_tx) % span_tx)).astype(jnp.float32)

            base_y = (jnp.max(new_flags[:, 1]) 
                      + jnp.float32(self.config.gate_vertical_spacing) / 2.0 
                      + jnp.float32(self.config.min_y_offset_tree_vs_rock))
            delta_y = jnp.float32(15.0)  # small vertical stagger (pixels), deterministic
            y = base_y + (jnp.array(i, dtype=jnp.float32) * delta_y)

            # Enforce min separation from existing trees and rocks on respawn (X only)
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
        # [deterministic]         k, k1, k2, k3, k4, k5, k6 = jax.random.split(k, 7)
        # [deterministic]         k1 = jnp.array([k1, k2, k3, k4, k5, k6])

        def check_rocks(i, rocks):
            # [deterministic]             x_rock = jax.random.randint(
            # [deterministic]                 k1.at[i].get(), [], 
            # [deterministic]                 self.config.rock_width,
            # [deterministic]                 self.config.screen_width - self.config.rock_width
            # [deterministic]             ).astype(jnp.float32)
            # Deterministic rock x based on gates_seen and index i, different step
            min_rx = jnp.int32(self.config.rock_width)
            max_rx = jnp.int32(self.config.screen_width - self.config.rock_width)
            span_rx = max_rx - min_rx + 1
            step_rx = 19
            x_rock = (min_rx + (((state.gates_seen + i) * step_rx) % span_rx)).astype(jnp.float32)
            y = (jnp.max(new_flags[:, 1]) + jnp.float32(self.config.gate_vertical_spacing) / 2.0 + jnp.float32(self.config.min_y_offset_tree_vs_rock))

            # Enforce separation from existing rocks and trees on respawn
            min_sep_rock_rock = 0.5*(jnp.float32(self.config.rock_width)+jnp.float32(self.config.rock_width)) + jnp.float32(self.config.sep_margin_rock_rock)
            min_sep_rock_tree = 0.5*(jnp.float32(self.config.rock_width)+jnp.float32(self.config.tree_width)) + jnp.float32(self.config.sep_margin_tree_rock)
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

        # 1) Eingabe -> Zielpose (Atari-like: 8 discrete headings + tap/auto-repeat)
        # Normalize action from get_human_action(): accept only A/D (JAXAtariAction LEFT=4, RIGHT=3).
        # Any other input (including SPACE/FIRE) becomes NOOP.
        is_left  = jnp.equal(action, jnp.int32(4))  # external LEFT (A)
        is_right = jnp.equal(action, jnp.int32(3))  # external RIGHT (D)
        norm_action = jax.lax.select(is_left,  self.consts.LEFT,
                        jax.lax.select(is_right, self.consts.RIGHT, self.consts.NOOP))

        # Atari feel: stepwise heading with auto-repeat while holding the key.
        # We do NOT add new state fields; we re-use 'direction_change_counter' as a repeat timer.
        # Logic:
        # - When a LEFT/RIGHT key is held and the counter hits 0, advance one discrete step and reset the counter.
        # - While the counter > 0, it counts down each frame and no additional step happens.
        # - On NOOP, reset the counter to 0 (so the next tap is immediate).
        REPEAT_FRAMES = jnp.int32(4)  # small cadence to feel snappy (tap-friendly)

        # Count down when a direction key is pressed; else zero.
        # If NOOP: counter -> 0
        # If LEFT/RIGHT:
        #   if counter==0 -> allow step now and set counter=REPEAT_FRAMES
        #   else          -> just decrement by 1
        counter_now = state.direction_change_counter
        want_left   = jnp.equal(norm_action, self.consts.LEFT)
        want_right  = jnp.equal(norm_action, self.consts.RIGHT)
        want_turn   = jnp.logical_or(want_left, want_right)

        can_step_now = jnp.logical_and(want_turn, jnp.equal(counter_now, 0))

        delta = jnp.where(want_left, -1, jnp.where(want_right, +1, 0)).astype(jnp.int32)
        new_skier_pos = jnp.where(can_step_now, state.skier_pos + delta, state.skier_pos)
        new_skier_pos = jnp.clip(new_skier_pos, 0, 7)

        direction_change_counter = jnp.where(
            jnp.equal(norm_action, self.consts.NOOP),
            jnp.int32(0),
            jnp.where(
                jnp.equal(counter_now, 0),
                jnp.where(want_turn, REPEAT_FRAMES, jnp.int32(0)),
                jnp.maximum(counter_now - 1, 0),
            ),
        )
        skier_pos = new_skier_pos

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
        # --- First-frame jitter guard: freeze world & lateral motion on time==0
        first_frame = jnp.equal(state.time, jnp.int32(0))
        eff_x_speed_nom = jax.lax.select(first_frame, jnp.array(0.0, jnp.float32), new_skier_x_speed_nom)
        eff_y_speed_nom = jax.lax.select(first_frame, jnp.array(0.0, jnp.float32), new_skier_y_speed_nom)

        min_x = self.config.skier_width / 2
        max_x = self.config.screen_width - self.config.skier_width / 2
        new_x_nom = jnp.clip(state.skier_x + eff_x_speed_nom, min_x, max_x)

        # 3) Welt – zunächst "nominal" bewegen (für Kollisionsprüfung),
        #    Freeze wird nach Kollisionsentscheidung angewandt.
        # Tree-specific first-frames guard to avoid residual jitter

        first_two_frames = jnp.less_equal(state.time, jnp.int32(1))

        eff_y_speed_trees = jax.lax.select(first_two_frames, jnp.array(0.0, jnp.float32), eff_y_speed_nom)

        new_trees_nom = state.trees.at[:, 1].add(-eff_y_speed_trees)
        new_rocks_nom = state.rocks.at[:, 1].add(-eff_y_speed_nom)
        new_flags_nom = state.flags.at[:, 1].add(-eff_y_speed_nom)

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
        new_skier_x_speed = jax.lax.select(freeze, jnp.array(0.0, jnp.float32), eff_x_speed_nom)
        new_skier_y_speed = jax.lax.select(freeze, jnp.array(0.0, jnp.float32), eff_y_speed_nom)
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
        info = self._get_info(new_state)
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
    def _get_info(self, state: GameState) -> SkiingInfo:
        return SkiingInfo(
            time=state.time,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState):
        return previous_state.score - state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        return jnp.greater_equal(state.gates_seen, 20)


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

# Hard-coded bitmap font for procedural digit generation
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
    # : (10)
    [[0,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,0]],
    # . (11) - Not used but included for completeness
    [[0,0,0],[0,0,0],[0,0,0],[0,1,0],[0,1,0]],
], dtype=jnp.uint8)

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


class SkiingRenderer(JAXGameRenderer):
    def __init__(self, consts: GameConfig = None):
        super().__init__()
        self.config = consts or GameConfig()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/skiing"
        
        # 1. Configure the rendering utility
        self.render_config = render_utils.RendererConfig(
            game_dimensions=(self.config.screen_height, self.config.screen_width),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.render_config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.config.ASSET_CONFIG)
        
        # 3. Load and recolor flags (needs sprite path, so done here)
        flag_red_rgba = self._load_rgba_sprite("checkered_flag.npy")
        flag_blue_rgba = self._recolor_rgba(flag_red_rgba, (0, 96, 255))
        final_asset_config.append({'name': 'flag_red', 'type': 'procedural', 'data': flag_red_rgba})
        final_asset_config.append({'name': 'flag_blue', 'type': 'procedural', 'data': flag_blue_rgba})

        # 4. Make one call to load and process all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, self.sprite_path)

        # 5. Store key color/shape IDs
        self.RED_FLAG_MASK = self.SHAPE_MASKS['flag_red']
        self.BLUE_FLAG_MASK = self.SHAPE_MASKS['flag_blue']
        self.RED_FLAG_OFFSET = self.FLIP_OFFSETS['flag_red']
        self.BLUE_FLAG_OFFSET = self.FLIP_OFFSETS['flag_blue']
        
        # 6. Store glyph dimensions for UI
        # The digits mask is (N, H, W)
        self.glyph_height = self.SHAPE_MASKS['digits'].shape[1]
        self.glyph_width = self.SHAPE_MASKS['digits'].shape[2]
        self.glyph_spacing = 1  # From old _center_positions logic

    def _load_rgba_sprite(self, file_name: str) -> np.ndarray:
        """Helper to load and standardize sprites to RGBA."""
        path = os.path.join(self.sprite_path, file_name)
        rgba = np.load(path).astype(np.uint8)
        if rgba.shape[-1] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, np.uint8)
            rgba = np.concatenate([rgba, a], axis=-1)
        return rgba

    def _recolor_rgba(self, sprite_rgba: np.ndarray, rgb: Tuple[int,int,int]) -> np.ndarray:
        """Manually recolors an RGBA sprite. For setup only."""
        mask = (sprite_rgba[..., 3:4] > 0)
        rgb_arr = np.array(rgb, dtype=np.uint8)[None, None, :]
        new_rgb = np.where(mask, rgb_arr, sprite_rgba[..., :3])
        return np.concatenate([new_rgb, sprite_rgba[..., 3:4]], axis=-1)
        
    @partial(jax.jit, static_argnums=(0,))
    def _format_score_digits(self, score: jnp.ndarray) -> jnp.ndarray:
        s_val = jnp.clip(score.astype(jnp.int32), 0, 99)
        tens = (s_val // 10) % 10
        ones = s_val % 10
        return jnp.stack([tens, ones], axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _format_time_digits(self, t: jnp.ndarray) -> jnp.ndarray:
        t = jnp.maximum(t.astype(jnp.float32), 0.0)
        FPS = jnp.float32(60.0)
        seconds_total = t / FPS

        minutes_digit = (jnp.floor(seconds_total / 60.0).astype(jnp.int32)) % 10
        seconds_int   = jnp.floor(jnp.mod(seconds_total, 60.0)).astype(jnp.int32)
        ms_int        = jnp.floor((seconds_total - jnp.floor(seconds_total)) * 1000.0).astype(jnp.int32)

        s_t  = (seconds_int // 10) % 10
        s_o  = seconds_int % 10
        ms_t = (ms_int // 10) % 10
        ms_o = ms_int % 10

        # Index 10 is ':' in _GLYPHS_BITS
        colon = jnp.int32(10)
        return jnp.stack([minutes_digit, colon, s_t, s_o, colon, ms_t, ms_o], axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState) -> jnp.ndarray:
        # 1. Start with the white background
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # 2. Get skier sprite
        skier_masks = self.SHAPE_MASKS['skier_group']
        skier_offset = self.FLIP_OFFSETS['skier_group']
        
        pos = jnp.clip(state.skier_pos, 0, 7)
        # 0..2 = left (idx 0), 3..4 = front (idx 1), 5..7 = right (idx 2)
        skier_base_idx = jax.lax.select(
            pos <= 2, 0,
            jax.lax.select(pos >= 5, 2, 1)
        )
        skier_base = skier_masks[skier_base_idx]

        is_fallen = (state.skier_fell > 0) & \
                    ((state.collision_type == 1) | (state.collision_type == 2) | (state.collision_type == 3))
        
        skier_sprite = jax.lax.select(is_fallen, skier_masks[3], skier_base)  # idx 3 is 'skier_fallen'
        
        # Center coordinates
        skier_cx = state.skier_x
        skier_cy = jnp.array(self.config.skier_y)
        
        # Top-left coordinates
        skier_top = (skier_cy - (skier_sprite.shape[0] // 2)).astype(jnp.int32)
        skier_left = (skier_cx - (skier_sprite.shape[1] // 2)).astype(jnp.int32)

        # 3. Draw Skier
        raster = self.jr.render_at(
            raster, skier_left, skier_top, 
            skier_sprite, flip_offset=skier_offset
        )

        # 4. Draw Flags
        flags_xy = state.flags[..., :2]
        left_pos = flags_xy.astype(jnp.int32)
        right_pos = (flags_xy + jnp.array([self.config.flag_distance, 0.0])).astype(jnp.int32)
        
        n_flags = state.flags.shape[0]
        dy_to_skier = jnp.abs(flags_xy[:, 1] - jnp.float32(self.config.skier_y))
        closest_idx = jnp.argmin(dy_to_skier)
        is_twentieth = jnp.greater_equal(state.gates_seen, jnp.int32(19))
        is_red_mask = jnp.zeros((n_flags,), dtype=bool).at[closest_idx].set(is_twentieth)
        
        # Render flags one by one
        def draw_flag(i, r):
            is_red = is_red_mask[i]
            mask = jax.lax.select(is_red, self.RED_FLAG_MASK, self.BLUE_FLAG_MASK)
            offset = jax.lax.select(is_red, self.RED_FLAG_OFFSET, self.BLUE_FLAG_OFFSET)
            
            # Center coords
            cx_left, cy = left_pos[i]
            cx_right, _ = right_pos[i]
            
            # Top-left coords
            top = (cy - (mask.shape[0] // 2)).astype(jnp.int32)
            left_l = (cx_left - (mask.shape[1] // 2)).astype(jnp.int32)
            left_r = (cx_right - (mask.shape[1] // 2)).astype(jnp.int32)
            
            r = self.jr.render_at_clipped(r, left_l, top, mask, flip_offset=offset)
            r = self.jr.render_at_clipped(r, left_r, top, mask, flip_offset=offset)
            return r

        raster = jax.lax.fori_loop(0, self.config.max_num_flags, draw_flag, raster)

        # 5. Draw Trees
        tree_mask = self.SHAPE_MASKS['tree']
        tree_offset = self.FLIP_OFFSETS['tree']
        tree_h, tree_w = tree_mask.shape[0], tree_mask.shape[1]
        
        def draw_tree(i, r):
            cx, cy = state.trees[i, :2]
            top = (cy - (tree_h // 2)).astype(jnp.int32)
            left = (cx - (tree_w // 2)).astype(jnp.int32)
            return self.jr.render_at_clipped(r, left, top, tree_mask, flip_offset=tree_offset)
            
        raster = jax.lax.fori_loop(0, self.config.max_num_trees, draw_tree, raster)

        # 6. Draw Rocks
        rock_mask = self.SHAPE_MASKS['rock']
        rock_offset = self.FLIP_OFFSETS['rock']
        rock_h, rock_w = rock_mask.shape[0], rock_mask.shape[1]

        def draw_rock(i, r):
            cx, cy = state.rocks[i, :2]
            top = (cy - (rock_h // 2)).astype(jnp.int32)
            left = (cx - (rock_w // 2)).astype(jnp.int32)
            return self.jr.render_at_clipped(r, left, top, rock_mask, flip_offset=rock_offset)
            
        raster = jax.lax.fori_loop(0, self.config.max_num_rocks, draw_rock, raster)

        # 7. Draw UI
        score_digits = self._format_score_digits(state.score)
        time_digits  = self._format_time_digits(state.time)
        
        # Center score
        num_glyphs_score = score_digits.shape[0]
        total_w_score = num_glyphs_score * self.glyph_width + (num_glyphs_score - 1) * self.glyph_spacing
        left_score = (self.config.screen_width - total_w_score) // 2
        
        raster = self.jr.render_label(
            raster, left_score, 2, score_digits,
            self.SHAPE_MASKS['digits'], 
            spacing=self.glyph_width + self.glyph_spacing,
            max_digits=num_glyphs_score
        )
        
        # Center time
        num_glyphs_time = time_digits.shape[0]
        total_w_time = num_glyphs_time * self.glyph_width + (num_glyphs_time - 1) * self.glyph_spacing
        left_time = (self.config.screen_width - total_w_time) // 2
        
        raster = self.jr.render_label(
            raster, left_time, 2 + self.glyph_height + 2, time_digits,
            self.SHAPE_MASKS['digits'], 
            spacing=self.glyph_width + self.glyph_spacing,
            max_digits=num_glyphs_time
        )
        
        # 8. Final conversion
        return self.jr.render_from_palette(raster, self.PALETTE)