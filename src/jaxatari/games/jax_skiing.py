from functools import partial
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Tuple, NamedTuple, Callable, Sequence, Optional
import os
import numpy as np
import collections
from flax import struct
import jaxatari.rendering.jax_rendering_utils as render_utils 
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
import jaxatari.spaces as spaces
from jaxatari.modification import AutoDerivedConstants

ORIGINAL_SCORES = jnp.array([-1, -2, -2])
USE_ORIGINAL_ALE_REWARD = True

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
            "skiier_0.npy",
            "skiier_1.npy",
            "skiier_2.npy",
            "skiier_3.npy",
            "skiier_4.npy",
            "skiier_5.npy",
            "skiier_6.npy",
            "skiier_7.npy",
            "skier_fallen.npy"
        ]},

        # Obstacles
        {'name': 'tree_group', 'type': 'group', 'files': [
            'tree_0.npy',
            'tree_1.npy',
            'tree_2.npy',
            'tree_3.npy'
        ]},
        {'name': 'mogul', 'type': 'single', 'file': 'stone.npy'},
        
        # UI
        {'name': 'digits', 'type': 'procedural', 'data': procedural_digits},
        {'name': 'ui_colors', 'type': 'procedural', 'data': static_procedural['ui_colors']}
    )

class SkiingConstants(AutoDerivedConstants):
    NOOP: int = struct.field(pytree_node=False, default=0)
    LEFT: int = struct.field(pytree_node=False, default=1)
    RIGHT: int = struct.field(pytree_node=False, default=2)
    FIRE: int = struct.field(pytree_node=False, default=3)
    BOTTOM_BORDER: int = struct.field(pytree_node=False, default=176)
    TOP_BORDER: int = struct.field(pytree_node=False, default=-15)
    """Game configuration parameters"""
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    skier_width: int = struct.field(pytree_node=False, default=10)
    skier_height: int = struct.field(pytree_node=False, default=18)
    skier_y: int = struct.field(pytree_node=False, default=46)
    flag_width: int = struct.field(pytree_node=False, default=10)
    flag_height: int = struct.field(pytree_node=False, default=28)
    flag_distance: int = struct.field(pytree_node=False, default=32)
    gate_vertical_spacing: int = struct.field(pytree_node=False, default=90)
    tree_width: int = struct.field(pytree_node=False, default=16)
    tree_height: int = struct.field(pytree_node=False, default=30)
    mogul_width: int = struct.field(pytree_node=False, default=16)
    mogul_height: int = struct.field(pytree_node=False, default=7)
    # Separation margins (in pixels) used in X-separation checks
    sep_margin_tree_tree: float = struct.field(pytree_node=False, default=14.0)
    sep_margin_mogul_mogul: float = struct.field(pytree_node=False, default=12.0)
    sep_margin_tree_mogul: float = struct.field(pytree_node=False, default=14.0)
    # Small Y offset between moguls and trees to avoid identical rows
    min_y_offset_tree_vs_mogul: float = struct.field(pytree_node=False, default=8.0)
    max_num_flags: int = struct.field(pytree_node=False, default=2)
    max_num_trees: int = struct.field(pytree_node=False, default=4)
    max_num_moguls: int = struct.field(pytree_node=False, default=2)
    moguls_collidable: bool = struct.field(pytree_node=False, default=False)
    allow_jump: bool = struct.field(pytree_node=False, default=False)
    speed: float = struct.field(pytree_node=False, default=1.0)

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)


@struct.dataclass
class SkiingState:
    """Represents the current state of the game"""

    skier_x: chex.Array
    skier_pos: chex.Array  # --> --_  \   |   |   /  _-- <-- States are doubles in ALE
    skier_fell: chex.Array
    skier_x_speed: chex.Array
    skier_y_speed: chex.Array
    flags: chex.Array
    trees: chex.Array
    moguls: chex.Array
    successful_gates: chex.Array
    step_count: chex.Array
    direction_change_counter: chex.Array
    game_over: chex.Array
    key: chex.Array
    collision_type: chex.Array  # 0 = none, 1 = tree, 2 = mogul, 3 = flag
    flags_passed: chex.Array
    collision_cooldown: chex.Array  # Frames where collisions are ignored (Debounce after recovery)
    skier_just_respawned: chex.Array # Boolean indicating if skier is in post-recovery immunity
    jump_timer: chex.Array # Timer for jump duration and cooldown
    is_jumping: chex.Array # Boolean indicating if the skier is currently jumping
    gates_seen: chex.Array  # Number of already processed gates (despawned)

@struct.dataclass
class SkiingObservation:
    skier: ObjectObservation
    flags: ObjectObservation # n=2
    trees: ObjectObservation # n=4
    moguls: ObjectObservation # n=3
    successful_gates: jnp.ndarray


@struct.dataclass
class SkiingInfo:
    step_count: jnp.ndarray


class JaxSkiing(JaxEnvironment[SkiingState, SkiingObservation, SkiingInfo, SkiingConstants]):
    # ALE minimal action set: [NOOP, RIGHT, LEFT, FIRE]
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP,
        Action.RIGHT,
        Action.LEFT,
        Action.FIRE
    ], dtype=jnp.int32)

    def __init__(self, consts: SkiingConstants | None = None):
        self.consts = consts or SkiingConstants()
        super().__init__(self.consts)
        self.state = self.reset()
        self.renderer = SkiingRenderer(self.consts)

    def action_space(self) -> spaces.Discrete:
        # ALE actions: NOOP=0, RIGHT=1, LEFT=2
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        # Use self.consts directly, casting to int for concrete values
        h = int(self.consts.screen_height)
        w = int(self.consts.screen_width)
        screen_size = (h, w)
        
        single_obj = spaces.get_object_space(n=None, screen_size=screen_size)
        
        return spaces.Dict({
            "skier": single_obj,
            "flags": spaces.get_object_space(n=self.consts.max_num_flags, screen_size=screen_size),
            "trees": spaces.get_object_space(n=self.consts.max_num_trees, screen_size=screen_size),
            "moguls": spaces.get_object_space(n=self.consts.max_num_moguls, screen_size=screen_size),
            "successful_gates": spaces.Box(low=0.0, high=1_000_000.0, shape=(), dtype=jnp.float32),
        })

    def image_space(self):
        c = self.consts
        return spaces.Box(low=0, high=255, shape=(c.screen_height, c.screen_width, 3), dtype=jnp.uint8)

    def reset(self, key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(1701)) -> Tuple[SkiingObservation, SkiingState]:
        """Initialize a new game state deterministically from `key`."""
        c = self.consts
        k_flags, k_trees, k_moguls, new_key = jax.random.split(key, 4)

        row_spacing = jnp.float32(31.0)
        base_y = jnp.float32(60.0)

        # Flags: r = 3, 7 in the repeating sequence
        r_flags = jnp.array([3, 7], dtype=jnp.float32)
        flags_y = base_y + r_flags * row_spacing
        # [deterministic]         flags_x = jax.random.randint(
        # [deterministic]             k_flags, (c.max_num_flags,),
        # [deterministic]             minval=int(c.flag_width),
        # [deterministic]             maxval=int(c.screen_width - c.flag_width - c.flag_distance) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic left-flag x-position: between 50 and 110
        min_fx = jnp.int32(50)
        max_fx = jnp.int32(110)
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
        # Deterministic tree x-position: either [-6, 60] or [100, 170]
        # Use a large prime step (101) so they scatter across both sides
        tree_val = (jnp.arange(c.max_num_trees, dtype=jnp.int32) * 101) % 138
        trees_x = jnp.where(tree_val <= 66, -6.0 + tree_val, 100.0 + (tree_val - 67)).astype(jnp.float32)
        # [deterministic]         trees_y = jax.random.randint(
        # [deterministic]             k_trees, (c.max_num_trees,),
        # [deterministic]             minval=int(c.tree_height),
        # [deterministic]             maxval=int(c.screen_height - c.tree_height) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic tree y-position: pattern [0, 1, 4, 5] repeating every 8 rows
        trees_per_row = jnp.maximum(1, c.max_num_trees // 4)
        i_t = jnp.arange(c.max_num_trees, dtype=jnp.int32)
        row_idx_t = i_t // trees_per_row
        base_offsets_t = jnp.array([0, 1, 4, 5], dtype=jnp.float32)
        r_trees = (row_idx_t // 4) * 8.0 + jnp.take(base_offsets_t, row_idx_t % 4)
        trees_y = base_y + r_trees * row_spacing
        
        # Add a deterministic stagger (-7 to +7 pixels) so they are not perfectly aligned
        stagger_t = ((i_t * 7) % 15).astype(jnp.float32) - 7.0
        trees_y = trees_y + stagger_t

        min_sep_tree = 0.5*(jnp.float32(c.tree_width)+jnp.float32(c.tree_width)) + jnp.float32(c.sep_margin_tree_tree)
        xmin = jnp.float32(-6.0)
        xmax = jnp.float32(170.0)

        def adj_tree_i(i, tx):
            x0 = tx[i]
            x_adj = _enforce_min_sep_x(x0, tx, min_sep_tree, xmin, xmax, n_valid=jnp.array(i, dtype=jnp.int32))
            # Strictly enforce the forbidden gap (60, 100)
            x_adj = jnp.where((x_adj > 60.0) & (x_adj < 100.0), jnp.where(x_adj < 80.0, 60.0, 100.0), x_adj)
            return tx.at[i].set(x_adj)

        trees_x = jax.lax.fori_loop(0, c.max_num_trees, adj_tree_i, trees_x)

        trees_type = jnp.arange(c.max_num_trees, dtype=jnp.float32) % 4.0
        trees = jnp.stack([
            trees_x, trees_y,
            jnp.full((c.max_num_trees,), float(c.tree_width),  dtype=jnp.float32),
            jnp.full((c.max_num_trees,), float(c.tree_height), dtype=jnp.float32),
            trees_type
        ], axis=1)



        # moguls
        # [deterministic]         moguls_x = jax.random.randint(
        # [deterministic]             k_moguls, (c.max_num_moguls,),
        # [deterministic]             minval=int(c.mogul_width),
        # [deterministic]             maxval=int(c.screen_width - c.mogul_width) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic mogul x-position: between 50 and 110
        min_rx = jnp.int32(50)
        max_rx = jnp.int32(110)
        span_rx = max_rx - min_rx + 1
        moguls_x = (min_rx + ((jnp.arange(c.max_num_moguls, dtype=jnp.int32) * 19) % span_rx)).astype(jnp.float32)
        # [deterministic]         moguls_y = jax.random.randint(
        # [deterministic]             k_moguls, (c.max_num_moguls,),
        # [deterministic]             minval=int(c.mogul_height),
        # [deterministic]             maxval=int(c.screen_height - c.mogul_height) + 1
        # [deterministic]         ).astype(jnp.float32)
        # Deterministic mogul y-position: pattern [2, 6] repeating every 8 rows
        moguls_per_row = jnp.maximum(1, c.max_num_moguls // 2)
        i_r = jnp.arange(c.max_num_moguls, dtype=jnp.int32)
        row_idx_r = i_r // moguls_per_row
        base_offsets_r = jnp.array([2, 6], dtype=jnp.float32)
        r_moguls = (row_idx_r // 2) * 8.0 + jnp.take(base_offsets_r, row_idx_r % 2)
        moguls_y = base_y + r_moguls * row_spacing
        
        # Add a deterministic stagger (-7 to +7 pixels) so they are not perfectly aligned
        stagger_r = ((i_r * 11) % 15).astype(jnp.float32) - 7.0
        moguls_y = moguls_y + stagger_r
        # Enforce separation from trees and already placed moguls
        min_sep_mogul_tree = 0.5*(jnp.float32(self.consts.mogul_width)+jnp.float32(self.consts.tree_width)) + jnp.float32(self.consts.sep_margin_tree_mogul)
        min_sep_mogul_mogul = 0.5*(jnp.float32(self.consts.mogul_width)+jnp.float32(self.consts.mogul_width)) + jnp.float32(self.consts.sep_margin_mogul_mogul)
        xmin_r = jnp.float32(50.0)
        xmax_r = jnp.float32(110.0)

        tree_xs_fixed = trees[:, 0]

        def adj_mogul_i(i, rx):
            x0 = rx[i]
            # push from all trees (all are valid)
            x1 = _enforce_min_sep_x(x0, tree_xs_fixed, min_sep_mogul_tree, xmin_r, xmax_r, n_valid=jnp.array(tree_xs_fixed.shape[0], dtype=jnp.int32))
            # then from previously placed moguls (indices < i)
            x2 = _enforce_min_sep_x(x1, rx, min_sep_mogul_mogul, xmin_r, xmax_r, n_valid=jnp.array(i, dtype=jnp.int32))
            return rx.at[i].set(x2)

        moguls_x = jax.lax.fori_loop(0, c.max_num_moguls, adj_mogul_i, moguls_x)

        moguls = jnp.stack([
            moguls_x, moguls_y,
            jnp.full((c.max_num_moguls,), float(c.mogul_width),  dtype=jnp.float32),
            jnp.full((c.max_num_moguls,), float(c.mogul_height), dtype=jnp.float32)
        ], axis=1)


        state = SkiingState(
            skier_x=jnp.array(76.0),
            skier_pos=jnp.array(4, dtype=jnp.int32),
            skier_fell=jnp.array(0, dtype=jnp.int32),
            skier_x_speed=jnp.array(0.0),
            skier_y_speed=jnp.array(0.0),
            flags=flags,
            trees=trees,
            moguls=moguls,
            successful_gates=jnp.array(20, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
            direction_change_counter=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False),
            key=new_key,
            collision_type=jnp.array(0, dtype=jnp.int32),
            flags_passed=jnp.zeros(c.max_num_flags, dtype=bool),
            collision_cooldown=jnp.array(0, dtype=jnp.int32),
            skier_just_respawned=jnp.array(False, dtype=jnp.bool_),
            jump_timer=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(False, dtype=jnp.bool_),
            gates_seen=jnp.array(0, dtype=jnp.int32),
        )
        obs = self._get_observation(state)
        return obs, state
    
    def render(self, state: SkiingState) -> jnp.ndarray:
        """Delegates to SkiingRenderer so play.py receives an RGB frame."""
        return self.renderer.render(state)

    def _create_new_objs(self, state, new_flags, new_trees, new_moguls):
        # [deterministic]         k, k1, k2, k3, k4 = jax.random.split(state.key, num=5)  # not used (deterministic respawn)
        # [deterministic]         k1 = jnp.array([k1, k2, k3, k4])
        k = state.key

        def check_flags(i, flags):
            # new x-position within the valid range
            # [deterministic]             x_flag = jax.random.randint(
            # [deterministic]                 k1.at[i].get(), [],
            # [deterministic]                 self.config.flag_width,
            # [deterministic]                 self.config.screen_width - self.config.flag_width - self.config.flag_distance
            # [deterministic]             ).astype(jnp.float32)
            # Deterministic left-flag x based on gates_seen and loop index
            min_fx = jnp.int32(50)
            max_fx = jnp.int32(110)
            span_fx = max_fx - min_fx + 1
            step_fx = 13
            x_flag = (min_fx + (((state.gates_seen + i) * step_fx) % span_fx)).astype(jnp.float32)

            row_old = flags.at[i].get()  # Shape (2,) or (4,)

            # Constant vertical distance: always spawn exactly 8 rows (248.0 pixels) behind its current position
            # This perfectly preserves the sequence density even with multiple objects per row
            y = row_old.at[1].get() + jnp.float32(248.0)

            # Prevent spawning after the 20th gate (0 to 19).
            # When state.gates_seen >= 18, we have already spawned up to gate 19.
            y = jnp.where(state.gates_seen >= 18, jnp.float32(10000.0), y)

            row_new = row_old.at[0].set(x_flag).at[1].set(y)

            # Only respawn if the flag despawned above TOP_BORDER
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
            # Deterministic tree x based on gates_seen and index i, either [-6, 60] or [100, 170]
            step_tx = 101
            tree_val = ((state.gates_seen * 13 + i * 23) * step_tx) % 138
            x_tree = jnp.where(tree_val <= 66, -6.0 + tree_val, 100.0 + (tree_val - 67)).astype(jnp.float32)

            # Enforce min separation from existing trees and moguls on respawn (X only)
            min_sep_tree_tree = (jnp.float32(self.consts.tree_width) + jnp.float32(self.consts.tree_width)) * 0.5 + jnp.float32(8.0)
            min_sep_tree_mogul = (jnp.float32(self.consts.tree_width) + jnp.float32(self.consts.mogul_width)) * 0.5 + jnp.float32(8.0)
            xmin_t = jnp.float32(-6.0)
            xmax_t = jnp.float32(170.0)
            taken_from_trees = trees[:, 0]
            taken_from_moguls = new_moguls[:, 0]
            x_tree = _enforce_min_sep_x(x_tree, taken_from_trees, min_sep_tree_tree, xmin_t, xmax_t, n_valid=jnp.array(i, dtype=jnp.int32))
            x_tree = _enforce_min_sep_x(x_tree, taken_from_moguls, min_sep_tree_mogul, xmin_t, xmax_t, n_valid=jnp.array(taken_from_moguls.shape[0], dtype=jnp.int32))
            # Strictly enforce the forbidden gap (60, 100)
            x_tree = jnp.where((x_tree > 60.0) & (x_tree < 100.0), jnp.where(x_tree < 80.0, 60.0, 100.0), x_tree)

            row_old = trees.at[i].get()

            # Spawn exactly 8 rows behind current position to maintain sequence
            y = row_old.at[1].get() + jnp.float32(248.0)
            y = jnp.where(state.gates_seen >= 18, jnp.float32(10000.0), y)
            
            # Generate a pseudo-random tree type based on state.gates_seen and i
            new_type = ((state.gates_seen * 3 + i * 5) % 4).astype(jnp.float32)
            row_new = row_old.at[0].set(x_tree).at[1].set(y).at[4].set(new_type)

            cond = jnp.less(trees.at[i, 1].get(), self.consts.TOP_BORDER)
            out_row = jax.lax.cond(cond, lambda _: row_new, lambda _: row_old, operand=None)
            return trees.at[i].set(out_row)

        trees = jax.lax.fori_loop(0, 4, check_trees, new_trees)

        # ---- moguls ----
        # [deterministic]         k, k1, k2, k3, k4, k5, k6 = jax.random.split(k, 7)
        # [deterministic]         k1 = jnp.array([k1, k2, k3, k4, k5, k6])

        def check_moguls(i, moguls):
            # [deterministic]             x_mogul = jax.random.randint(
            # [deterministic]                 k1.at[i].get(), [], 
            # [deterministic]                 self.config.mogul_width,
            # [deterministic]                 self.config.screen_width - self.config.mogul_width
            # [deterministic]             ).astype(jnp.float32)
            # Deterministic mogul x based on gates_seen and index i, between 50 and 110
            min_rx = jnp.int32(50)
            max_rx = jnp.int32(110)
            span_rx = max_rx - min_rx + 1
            step_rx = 19
            x_mogul = (min_rx + (((state.gates_seen + i) * step_rx) % span_rx)).astype(jnp.float32)
            # Enforce separation from existing moguls and trees on respawn
            min_sep_mogul_mogul = 0.5*(jnp.float32(self.consts.mogul_width)+jnp.float32(self.consts.mogul_width)) + jnp.float32(self.consts.sep_margin_mogul_mogul)
            min_sep_mogul_tree = 0.5*(jnp.float32(self.consts.mogul_width)+jnp.float32(self.consts.tree_width)) + jnp.float32(self.consts.sep_margin_tree_mogul)
            xmin_r = jnp.float32(50.0)
            xmax_r = jnp.float32(110.0)
            taken_from_moguls = moguls[:, 0]
            taken_from_trees = new_trees[:, 0]
            x_mogul = _enforce_min_sep_x(x_mogul, taken_from_moguls, min_sep_mogul_mogul, xmin_r, xmax_r, n_valid=jnp.array(taken_from_moguls.shape[0], dtype=jnp.int32))
            x_mogul = _enforce_min_sep_x(x_mogul, taken_from_trees, min_sep_mogul_tree, xmin_r, xmax_r, n_valid=jnp.array(taken_from_trees.shape[0], dtype=jnp.int32))

            row_old = moguls.at[i].get()

            # Spawn exactly 8 rows behind current position to maintain sequence
            y = row_old.at[1].get() + jnp.float32(248.0)
            y = jnp.where(state.gates_seen >= 18, jnp.float32(10000.0), y)
            row_new = row_old.at[0].set(x_mogul).at[1].set(y)

            cond = jnp.less(moguls.at[i, 1].get(), self.consts.TOP_BORDER)
            out_row = jax.lax.cond(cond, lambda _: row_new, lambda _: row_old, operand=None)
            return moguls.at[i].set(out_row)

        moguls = jax.lax.fori_loop(0, 3, check_moguls, new_moguls)

        return flags, trees, moguls, k

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SkiingState, action: int
    ) -> tuple[SkiingObservation, SkiingState, float, bool, SkiingInfo]:
        #                              -->  --_      \     |     |    /    _-- <--
        side_speed = jnp.array([-1.0, -0.5, -0.333, 0.0, 0.0, 0.333, 0.5, 1.0], jnp.float32)
        #                              -->  --_   \     |    |     /    _--  <--
        down_speed = jnp.array([0.0, 0.5, 0.875, 1.0, 1.0, 0.875, 0.5, 0.0], jnp.float32)

        RECOVERY_FRAMES = jnp.int32(60)
        TREE_X_DIST = jnp.float32(8.0)
        mogul_X_DIST = jnp.float32(8.0)
        Y_HIT_DIST  = jnp.float32(4.0)

        # Translate agent action (0,1,2) to ALE action
        atari_action = jnp.take(self.ACTION_SET, action)
        
        # Map ALE action to internal constants
        norm_action = jnp.where(
            atari_action == Action.NOOP, self.consts.NOOP,
            jnp.where(
                atari_action == Action.RIGHT, self.consts.RIGHT,
                jnp.where(
                    atari_action == Action.LEFT, self.consts.LEFT,
                    jnp.where(
                        atari_action == Action.FIRE, self.consts.FIRE,
                        self.consts.NOOP  # fallback to NOOP
                    )
                )
            )
        )

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

        # Jumping logic: A jump lasts for 35 frames, followed by a 36-frame cooldown. Total = 71.
        want_jump = jnp.logical_and(jnp.equal(norm_action, self.consts.FIRE), self.consts.allow_jump)
        can_jump = jnp.equal(state.jump_timer, 0)
        
        # Start a new jump if requested and cooldown is over. Otherwise just decrement the timer.
        new_jump_timer = jax.lax.select(
            jnp.logical_and(want_jump, can_jump),
            jnp.int32(71),
            jnp.maximum(state.jump_timer - 1, 0)
        )
        
        # The skier is "jumping" (immune to moguls) for the first 35 frames of the 71-frame timer.
        new_is_jumping = jnp.greater(new_jump_timer, 36)

        # 2) Base speeds
        dx_target = side_speed.at[skier_pos].get()
        dy_target = down_speed.at[skier_pos].get()

        in_recovery = jnp.greater(state.skier_fell, 0)

        # Recovery: Face front, 0 horizontal speed, y speed same as front
        skier_pos = jax.lax.select(in_recovery, jnp.array(3), skier_pos)
        dx_target = jax.lax.select(in_recovery, jnp.array(0.0, dtype=jnp.float32), dx_target)
        dy_target = jax.lax.select(in_recovery, down_speed.at[3].get(), dy_target)

        friction_x = jnp.float32(0.04)
        friction_y = jnp.float32(0.01)
        max_speed = jnp.float32(1.2)

        # Calculate target orientation vector
        dir_norm = jnp.sqrt(dx_target**2 + dy_target**2) + 1e-6
        dir_x = dx_target / dir_norm
        dir_y = dy_target / dir_norm

        # Acceleration is proportional to how much the skier is facing down (dy_target)
        # Maximal when facing down (dy_target == 1.0), zero when parallel (dy_target == 0.0)
        accel_mag = dy_target * jnp.float32(0.05)

        # Distribute acceleration along the x and y axes
        # We boost the horizontal acceleration component by 2x so the skier accelerates more laterally when turning
        accel_x = accel_mag * dir_x * jnp.float32(2.0)
        accel_y = accel_mag * dir_y
        
        # Apply independent friction and acceleration to preserve momentum
        x_speed_next = state.skier_x_speed * (1.0 - friction_x) + accel_x
        y_speed_next = state.skier_y_speed * (1.0 - friction_y) + accel_y

        # Clip total speed vector to max_speed
        current_speed_next = jnp.sqrt(x_speed_next**2 + y_speed_next**2) + 1e-6
        scale = jnp.minimum(1.0, max_speed / current_speed_next)
        x_speed_next = x_speed_next * scale
        y_speed_next = y_speed_next * scale

        new_skier_x_speed_nom = jax.lax.select(
            in_recovery,
            jnp.array(0.0, dtype=jnp.float32),
            x_speed_next
        )
        new_skier_y_speed_nom = y_speed_next
        # --- First-frame jitter guard: freeze world & lateral motion on step_count==0
        first_frame = jnp.equal(state.step_count, jnp.int32(0))
        eff_x_speed_nom = jax.lax.select(first_frame, jnp.array(0.0, jnp.float32), new_skier_x_speed_nom)
        eff_y_speed_nom = jax.lax.select(first_frame, jnp.array(0.0, jnp.float32), new_skier_y_speed_nom)

        min_x = self.consts.skier_width / 2
        max_x = self.consts.screen_width - self.consts.skier_width / 2
        new_x_nom = jnp.clip(state.skier_x + eff_x_speed_nom, min_x, max_x)

        # 3) World - move "nominally" first (for collision detection),
        #    Freeze is applied after collision decision.
        # Tree-specific first-frames guard to avoid residual jitter

        first_two_frames = jnp.less_equal(state.step_count, jnp.int32(1))

        eff_y_speed_trees = jax.lax.select(first_two_frames, jnp.array(0.0, jnp.float32), eff_y_speed_nom)

        new_trees_nom = state.trees.at[:, 1].add(-eff_y_speed_trees)
        new_moguls_nom = state.moguls.at[:, 1].add(-eff_y_speed_nom)
        new_flags_nom = state.flags.at[:, 1].add(-eff_y_speed_nom)

        # 5) Collisions (intercept lateral approach)
        skier_y_px = jnp.round(self.consts.skier_y)

        def coll_tree(tree_pos, x_d=TREE_X_DIST, y_d=Y_HIT_DIST):
            x = tree_pos[..., 0]
            y = tree_pos[..., 1]
            dx = jnp.abs(new_x_nom - x)
            # Tree height is 30, center is y. Bottom is y+15.
            # Offset by +10 to target the trunk area.
            dy = jnp.abs(jnp.round(skier_y_px) - jnp.round(y + 10.0))
            return jnp.logical_and(dx <= x_d, dy < y_d)

        def coll_mogul(mogul_pos, x_d=mogul_X_DIST, y_d=Y_HIT_DIST):
            x = mogul_pos[..., 0]
            y = mogul_pos[..., 1]
            dx = jnp.abs(new_x_nom - x)
            dy = jnp.abs(jnp.round(skier_y_px) - jnp.round(y))
            return jnp.logical_and(dx < x_d, dy < y_d)

        def coll_flag(flag_pos, x_d=jnp.float32(2.0)):
            x = flag_pos[..., 0]
            y = flag_pos[..., 1]
            dx1 = jnp.abs(new_x_nom - 4 - x)
            dx2 = jnp.abs(new_x_nom - 4 - (x + self.consts.flag_distance))
            # The flag center is at y. Bottom of pole is y+14.
            # We move the hit zone higher up the pole to y+6.0 (8 pixels above the bottom).
            dy_off = skier_y_px - (y + 6.0)
            # Widen the vertical hit window to 4 pixels to prevent skipping at high speeds.
            dy_hit = (dy_off >= -2.0) & (dy_off <= 2.0)
            return jnp.logical_or(jnp.logical_and(dx1 <= x_d, dy_hit),
                                  jnp.logical_and(dx2 <= x_d, dy_hit))

        collisions_tree = jax.vmap(coll_tree)(jnp.array(new_trees_nom))
        collisions_mogul = jax.vmap(coll_mogul)(jnp.array(new_moguls_nom))
        collisions_flag = jax.vmap(coll_flag)(jnp.array(new_flags_nom))
        
        # Make moguls conditionally collidable (and ignore if jumping)
        collisions_mogul = jnp.where(
            jnp.logical_and(self.consts.moguls_collidable, jnp.logical_not(new_is_jumping)),
            collisions_mogul,
            jnp.zeros_like(collisions_mogul, dtype=collisions_mogul.dtype)
        )        
        # Do not trigger new collisions during recovery OR cooldown
        ignore_collisions = jnp.logical_or(in_recovery, jnp.greater(state.collision_cooldown, 0))
        collisions_tree = jnp.where(ignore_collisions, jnp.zeros_like(collisions_tree), collisions_tree)
        collisions_mogul = jnp.where(ignore_collisions, jnp.zeros_like(collisions_mogul), collisions_mogul)
        collisions_flag = jnp.where(ignore_collisions, jnp.zeros_like(collisions_flag), collisions_flag)

        collided_tree = jnp.sum(collisions_tree) > 0
        collided_mogul = jnp.sum(collisions_mogul) > 0
        collided_flag = jnp.sum(collisions_flag) > 0

        # Recovery on *any* obstacle collision (tree/mogul/flag)
        start_recovery = jnp.logical_and(
            jnp.logical_not(in_recovery),
            jnp.logical_or(jnp.logical_or(collided_tree, collided_mogul), collided_flag),
        )
        freeze = jnp.logical_or(in_recovery, start_recovery)

        # Additionally: Ignore collisions in the recovery start frame,
        # to avoid double hits without visual separation.
        mask_now = jnp.logical_or(ignore_collisions, start_recovery)
        collisions_tree = jnp.where(mask_now, jnp.zeros_like(collisions_tree), collisions_tree)
        collisions_mogul = jnp.where(mask_now, jnp.zeros_like(collisions_mogul), collisions_mogul)
        collisions_flag = jnp.where(mask_now, jnp.zeros_like(collisions_flag), collisions_flag)
        # Freeze without repositioning: Obstacles stay in place
        freeze_flags = state.flags
        freeze_trees = state.trees
        freeze_moguls = state.moguls

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
        # Collision debounce: Briefly ignore collisions after recovery ends
        COOLDOWN_FRAMES = jnp.int32(30)
        new_collision_cooldown = jax.lax.cond(
            # If recovery just ended (was >0, now ==0) -> set cooldown
            jnp.logical_and(in_recovery, jnp.equal(new_skier_fell, 0)),
            lambda _: COOLDOWN_FRAMES,
            # else countdown (not negative)
            lambda _: jnp.maximum(state.collision_cooldown - 1, 0),
            operand=None
        )
        new_skier_just_respawned = jnp.greater(new_collision_cooldown, 0)
        new_collision_type = jax.lax.cond(
            start_recovery,
            lambda _: jnp.where(
                collided_tree, jnp.array(1, dtype=jnp.int32),
                jnp.where(
                    collided_mogul, jnp.array(2, dtype=jnp.int32),
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
        new_moguls = jax.lax.select(freeze, freeze_moguls, new_moguls_nom)
        # Freeze-aware skier X position (no pushback or lateral offset during recovery)
        new_x = jax.lax.select(freeze, state.skier_x, new_x_nom)


        
        # 8) Gate scoring (happens NOW, after final flag positions)
        left_x  = state.flags[:, 0]
        right_x = left_x + self.consts.flag_distance

        eligible = jnp.logical_and(new_x > left_x, new_x < right_x)
        crossed  = jnp.logical_and(state.flags[:, 1] > self.consts.skier_y,
                                   new_flags[:, 1] <= self.consts.skier_y)
        gate_pass = jnp.logical_and(eligible, jnp.logical_and(crossed, jnp.logical_not(state.flags_passed)))
        flags_passed = jnp.logical_or(state.flags_passed, gate_pass)

        # Calculate despawn based only on the "frozen" (final) flags
        despawn_mask = new_flags[:, 1] < self.consts.TOP_BORDER
        # missed_penalty_mask = jnp.logical_and(despawn_mask, jnp.logical_not(flags_passed))
        # missed_penalty_count = jnp.sum(missed_penalty_mask)
        # missed_penalty = missed_penalty_count * 300
        flags_passed = jnp.where(despawn_mask, False, flags_passed)


        # Increment gate counter: every despawned gate counts as seen
        gates_increment = jnp.sum(despawn_mask).astype(jnp.int32)
        new_gates_seen = state.gates_seen + gates_increment
        # Respawns/Despawns only when NOT frozen
        new_flags, new_trees, new_moguls, new_key = jax.lax.cond(
            freeze,
            lambda _: (new_flags, new_trees, new_moguls, state.key),
            lambda _: self._create_new_objs(state, new_flags, new_trees, new_moguls),
            operand=None
        )

        # Update score/step_count (only gates count)
        gates_scored = jnp.sum(gate_pass)
        new_score = state.successful_gates - gates_scored
        game_over = jnp.greater_equal(new_gates_seen, 20)
        new_step_count = jax.lax.cond(
            jnp.greater(state.step_count, 9223372036854775807 / 2),
            lambda _: jnp.array(0, dtype=jnp.int32),
            # lambda _: state.step_count + 1 + missed_penalty,
            lambda _: state.step_count + 1,
            operand=None,
        )

        new_state = SkiingState(
            skier_x=new_x,
            skier_pos=jnp.array(skier_pos),
            skier_fell=new_skier_fell,
            skier_x_speed=new_skier_x_speed,
            skier_y_speed=new_skier_y_speed,
            flags=jnp.array(new_flags),
            trees=jnp.array(new_trees),
            moguls=jnp.array(new_moguls),
            successful_gates=new_score,
            step_count=new_step_count,
            direction_change_counter=direction_change_counter,
            game_over=game_over,
            key=new_key,
            collision_type=new_collision_type,
            flags_passed=flags_passed,
            collision_cooldown=new_collision_cooldown,
            skier_just_respawned=new_skier_just_respawned,
            jump_timer=new_jump_timer,
            is_jumping=new_is_jumping,
            gates_seen=new_gates_seen,
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        reward = jnp.array(reward, dtype=jnp.float32)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SkiingState) -> SkiingObservation:
        c = self.consts
        w, h = int(c.screen_width), int(c.screen_height)

        # --- Skier ---
        # Map skier_pos (0..7) to angles
        # 0->270, 1->292.5, 2->315, 3->337.5, 4->22.5, 5->45, 6->67.5, 7->90
        angles = jnp.array([270.0, 292.5, 315.0, 337.5, 22.5, 45.0, 67.5, 90.0], dtype=jnp.float32)
        skier_ori = angles[state.skier_pos]

        skier = ObjectObservation.create(
            x=jnp.clip(jnp.array(state.skier_x, dtype=jnp.int32), 0, w),
            y=jnp.clip(jnp.array(c.skier_y, dtype=jnp.int32), 0, h),
            width=jnp.array(c.skier_width, dtype=jnp.int32),
            height=jnp.array(c.skier_height, dtype=jnp.int32),
            active=jnp.array(1, dtype=jnp.int32),
            orientation=jnp.array(skier_ori, dtype=jnp.float32)
        )

        # --- Flags ---
        # Flags in state are [x, y, w, h]
        flags_xy = state.flags[..., :2].astype(jnp.int32)
        flags_active = (flags_xy[:, 1] < h).astype(jnp.int32) # Simple visibility check
        
        flags = ObjectObservation.create(
            x=jnp.clip(flags_xy[:, 0], 0, w),
            y=jnp.clip(flags_xy[:, 1], 0, h),
            width=jnp.full((c.max_num_flags,), c.flag_width, dtype=jnp.int32),
            height=jnp.full((c.max_num_flags,), c.flag_height, dtype=jnp.int32),
            active=flags_active,
            visual_id=jnp.zeros((c.max_num_flags,), dtype=jnp.int32) # Could encode Red/Blue if needed
        )

        # --- Trees ---
        trees_xy = state.trees[..., :2].astype(jnp.int32)
        trees_active = (trees_xy[:, 1] < h).astype(jnp.int32)
        
        trees = ObjectObservation.create(
            x=jnp.clip(trees_xy[:, 0], 0, w),
            y=jnp.clip(trees_xy[:, 1], 0, h),
            width=jnp.full((c.max_num_trees,), c.tree_width, dtype=jnp.int32),
            height=jnp.full((c.max_num_trees,), c.tree_height, dtype=jnp.int32),
            active=trees_active
        )

        # --- moguls ---
        moguls_xy = state.moguls[..., :2].astype(jnp.int32)
        moguls_active = (moguls_xy[:, 1] < h).astype(jnp.int32)
        
        moguls = ObjectObservation.create(
            x=jnp.clip(moguls_xy[:, 0], 0, w),
            y=jnp.clip(moguls_xy[:, 1], 0, h),
            width=jnp.full((c.max_num_moguls,), c.mogul_width, dtype=jnp.int32),
            height=jnp.full((c.max_num_moguls,), c.mogul_height, dtype=jnp.int32),
            active=moguls_active
        )

        return SkiingObservation(
            skier=skier,
            flags=flags,
            trees=trees,
            moguls=moguls,
            successful_gates=jnp.array(state.successful_gates, dtype=jnp.float32)
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SkiingState) -> SkiingInfo:
        return SkiingInfo(
            step_count=state.step_count,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: SkiingState, state: SkiingState):
        if USE_ORIGINAL_ALE_REWARD:
            done = self._get_done(state)
            # In ALE, the final reward incorporates a massive penalty for missed gates.
            # state.successful_gates tracks (20 - successfully_passed_gates), which represents the missed gates.
            missed_gates = 20 - state.successful_gates
            end_penalty = - missed_gates * 500
            
            step_reward = ORIGINAL_SCORES[state.step_count % 3]
            
            return jnp.where(done, end_penalty, step_reward).astype(jnp.float32)
            
        return (previous_state.successful_gates - state.successful_gates).astype(jnp.float32)


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SkiingState) -> bool:
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
    mogul_color: Tuple[int, int, int] = (128, 128, 128)
    game_over_color: Tuple[int, int, int] = (255, 0, 0)
    
    # Text overlay option: True = UI text via Pygame on finished JAX frame
    # False = JAX bitmap font
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
    def __init__(self, consts: SkiingConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or SkiingConstants()
        super().__init__(self.consts)
        self.sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "skiing")
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.screen_height, self.consts.screen_width),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 2. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)
        
        # 3. Load flags (needs sprite path, so done here)
        flag_red_rgba = self._load_rgba_sprite("checkered_flag_red.npy")
        flag_blue_rgba = self._load_rgba_sprite("checkered_flag_blue.npy")
        
        # Pad them so they have the same shape for jax.lax.select
        max_h = max(flag_red_rgba.shape[0], flag_blue_rgba.shape[0])
        max_w = max(flag_red_rgba.shape[1], flag_blue_rgba.shape[1])
        if flag_red_rgba.shape[:2] != (max_h, max_w):
            pad_h = max_h - flag_red_rgba.shape[0]
            pad_w = max_w - flag_red_rgba.shape[1]
            flag_red_rgba = np.pad(flag_red_rgba, ((0, pad_h), (0, pad_w), (0, 0)))
        if flag_blue_rgba.shape[:2] != (max_h, max_w):
            pad_h = max_h - flag_blue_rgba.shape[0]
            pad_w = max_w - flag_blue_rgba.shape[1]
            flag_blue_rgba = np.pad(flag_blue_rgba, ((0, pad_h), (0, pad_w), (0, 0)))

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
    def _format_step_count_digits(self, t: jnp.ndarray) -> jnp.ndarray:
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
    def render(self, state: SkiingState) -> jnp.ndarray:
        # 1. Start with the white background
        bg_raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = bg_raster

        # 2. Draw moguls (Moguls) first so they are behind the skier
        mogul_mask = self.SHAPE_MASKS['mogul']
        mogul_offset = self.FLIP_OFFSETS['mogul']
        mogul_h, mogul_w = mogul_mask.shape[0], mogul_mask.shape[1]

        def draw_mogul(i, r):
            cx, cy = state.moguls[i, :2]
            top = (cy - (mogul_h // 2)).astype(jnp.int32)
            left = (cx - (mogul_w // 2)).astype(jnp.int32)
            return self.jr.render_at_clipped(r, left, top, mogul_mask, flip_offset=mogul_offset)
            
        raster = jax.lax.fori_loop(0, self.consts.max_num_moguls, draw_mogul, raster)

        # 3. Get and Draw Skier
        skier_masks = self.SHAPE_MASKS['skier_group']
        skier_offset = self.FLIP_OFFSETS['skier_group']
        
        pos = jnp.clip(state.skier_pos, 0, 7)
        skier_base = skier_masks[pos]

        is_fallen = (state.skier_fell > 0) & \
                    ((state.collision_type == 1) | (state.collision_type == 2) | (state.collision_type == 3))
        
        skier_sprite = jax.lax.select(is_fallen, skier_masks[8], skier_base)  # idx 8 is 'skier_fallen'
        
        # Center coordinates
        skier_cx = state.skier_x
        skier_cy = jnp.array(self.consts.skier_y)
        
        # Top-left coordinates
        skier_top = (skier_cy - (skier_sprite.shape[0] // 2)).astype(jnp.int32)
        skier_left = (skier_cx - (skier_sprite.shape[1] // 2)).astype(jnp.int32)

        raster = self.jr.render_at(
            raster, skier_left, skier_top, 
            skier_sprite, flip_offset=skier_offset
        )

        # 4. Draw Flags (in front of skier)
        flags_xy = state.flags[..., :2]
        left_pos = flags_xy.astype(jnp.int32)
        right_pos = (flags_xy + jnp.array([self.consts.flag_distance, 0.0])).astype(jnp.int32)
        
        n_flags = state.flags.shape[0]
        # The 20th gate is always in slot 1, and is the last one spawned.
        # It should be red when it is spawned (gates_seen >= 18).
        is_twentieth_visible = jnp.greater_equal(state.gates_seen, jnp.int32(18))
        is_red_mask = jnp.zeros((n_flags,), dtype=bool).at[1].set(is_twentieth_visible)
        
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

        raster = jax.lax.fori_loop(0, self.consts.max_num_flags, draw_flag, raster)

        # 5. Draw Trees (in front of skier)
        tree_masks = self.SHAPE_MASKS['tree_group']
        tree_offset = self.FLIP_OFFSETS['tree_group']

        def draw_tree(i, r):
            cx, cy = state.trees[i, :2]
            tree_type = state.trees[i, 4].astype(jnp.int32)
            mask = tree_masks[tree_type]
            tree_h, tree_w = mask.shape[0], mask.shape[1]

            top = (cy - (tree_h // 2)).astype(jnp.int32)
            left = (cx - (tree_w // 2)).astype(jnp.int32)
            return self.jr.render_at_clipped(r, left, top, mask, flip_offset=tree_offset)            
        raster = jax.lax.fori_loop(0, self.consts.max_num_trees, draw_tree, raster)

        # 6.5 Apply 32px top and bottom white bands to hide scrolled objects
        raster = raster.at[:32, :].set(bg_raster[:32, :])
        raster = raster.at[-32:, :].set(bg_raster[-32:, :])

        # 7. Draw UI
        score_digits = self._format_score_digits(state.successful_gates)
        step_count_digits  = self._format_step_count_digits(state.step_count)
        
        # Center score
        num_glyphs_score = score_digits.shape[0]
        total_w_score = num_glyphs_score * self.glyph_width + (num_glyphs_score - 1) * self.glyph_spacing
        left_score = (self.consts.screen_width - total_w_score) // 2
        
        raster = self.jr.render_label(
            raster, left_score, 2, score_digits,
            self.SHAPE_MASKS['digits'], 
            spacing=self.glyph_width + self.glyph_spacing,
            max_digits=num_glyphs_score
        )
        
        # Center step_count
        num_glyphs_step_count = step_count_digits.shape[0]
        total_w_step_count = num_glyphs_step_count * self.glyph_width + (num_glyphs_step_count - 1) * self.glyph_spacing
        left_step_count = (self.consts.screen_width - total_w_step_count) // 2
        
        raster = self.jr.render_label(
            raster, left_step_count, 2 + self.glyph_height + 2, step_count_digits,
            self.SHAPE_MASKS['digits'], 
            spacing=self.glyph_width + self.glyph_spacing,
            max_digits=num_glyphs_step_count
        )
        
        # 8. Final conversion
        return self.jr.render_from_palette(raster, self.PALETTE)