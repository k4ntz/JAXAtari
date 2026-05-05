import jax
import jax.numpy as jnp
import chex
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin

class MoreTreesMod(JaxAtariInternalModPlugin):
    """
    Spawns more trees during the race.
    """
    constants_overrides = {
        "max_num_trees": 12,
    }

class TreesEverywhereMod(JaxAtariInternalModPlugin):
    """
    Allows trees to spawn anywhere across the entire horizontal axis,
    instead of forcing a central gap.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_initial_trees_x(self) -> chex.Array:
        c = self._env.consts
        tree_val_everywhere = (jnp.arange(c.max_num_trees, dtype=jnp.int32) * 101) % 176
        return -6.0 + tree_val_everywhere.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_new_tree_x(self, state, i: chex.Array) -> chex.Array:
        step_tx = 101
        tree_val_everywhere = ((state.gates_seen * 13 + i * 23) * step_tx) % 176
        return -6.0 + tree_val_everywhere.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _enforce_tree_gap(self, x_tree: chex.Array) -> chex.Array:
        return x_tree


class MoreMogulsMod(JaxAtariInternalModPlugin):
    """
    Spawns more moguls (rocks) during the race.
    """
    constants_overrides = {
        "max_num_moguls": 6,
    }

class DangerousMogulsMod(JaxAtariInternalModPlugin):
    """
    Makes colliding with moguls cause the skier to fall.
    """
    constants_overrides = {
        "moguls_collidable": True,
    }

class JumpToBreakMod(JaxAtariInternalModPlugin):
    """
    Allows the skier to jump over moguls using the FIRE action.
    This mod specifically causes the skier to stop moving while jumping.
    """
    constants_overrides = {
        "jump_speed_multiplier": 0.0,
    }

class SpeedBurstMod(JaxAtariInternalModPlugin):
    """
    Allows the skier to accelerate beyond the default maximum speed using the DOWN action.
    """
    constants_overrides = {
        "down_max_speed": 1.8,
        "down_accel": 0.15,
    }

class HallOfFameMod(JaxAtariInternalModPlugin):
    """
    Places the gates dead center and creates a corridor of trees.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _get_initial_flags_x(self) -> chex.Array:
        c = self._env.consts
        return jnp.full((c.max_num_flags,), 60.0, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_new_flag_x(self, state, i: chex.Array) -> chex.Array:
        return jnp.float32(60.0)

    @partial(jax.jit, static_argnums=(0,))
    def _get_initial_trees_x(self) -> chex.Array:
        c = self._env.consts
        return jnp.where(jnp.arange(c.max_num_trees, dtype=jnp.int32) % 2 == 0, 63, 90)

    @partial(jax.jit, static_argnums=(0,))
    def _get_new_tree_x(self, state, i: chex.Array) -> chex.Array:
        return jnp.where(i % 2 == 0, 63, 90)

    @partial(jax.jit, static_argnums=(0,))
    def _apply_tree_separation_initial(self, i: chex.Array, x0: chex.Array, tx: chex.Array, min_sep_tree: chex.Array, xmin: chex.Array, xmax: chex.Array) -> chex.Array:
        return x0

    @partial(jax.jit, static_argnums=(0,))
    def _apply_tree_separation_respawn(self, i: chex.Array, x_tree: chex.Array, taken_from_trees: chex.Array, taken_from_moguls: chex.Array, min_sep_tree_tree: chex.Array, min_sep_tree_mogul: chex.Array, xmin_t: chex.Array, xmax_t: chex.Array) -> chex.Array:
        return x_tree


class InvertFlagsMod(JaxAtariInternalModPlugin):
    """Swaps flag colors: blue gates become red and the special 20th gate becomes blue."""

    @partial(jax.jit, static_argnums=(0,))
    def _draw_flags(self, raster: jnp.ndarray, state) -> jnp.ndarray:
        renderer = self._env.renderer
        flags_xy = state.flags[..., :2]
        left_pos = flags_xy.astype(jnp.int32)
        right_pos = (flags_xy + jnp.array([self._env.consts.flag_distance, 0.0])).astype(jnp.int32)

        n_flags = state.flags.shape[0]
        is_twentieth_visible = jnp.greater_equal(state.gates_seen, jnp.int32(18))
        # Inverted: all red by default; the 20th gate slot switches to blue when visible
        is_red_mask = jnp.ones((n_flags,), dtype=bool).at[1].set(jnp.logical_not(is_twentieth_visible))

        def draw_flag(i, r):
            is_red = is_red_mask[i]
            mask = jax.lax.select(is_red, renderer.RED_FLAG_MASK, renderer.BLUE_FLAG_MASK)
            offset = jax.lax.select(is_red, renderer.RED_FLAG_OFFSET, renderer.BLUE_FLAG_OFFSET)
            cx_left, cy = left_pos[i]
            cx_right, _ = right_pos[i]
            top = (cy - (mask.shape[0] // 2)).astype(jnp.int32)
            left_l = (cx_left - (mask.shape[1] // 2)).astype(jnp.int32)
            left_r = (cx_right - (mask.shape[1] // 2)).astype(jnp.int32)
            r = renderer.jr.render_at_clipped(r, left_l, top, mask, flip_offset=offset)
            r = renderer.jr.render_at_clipped(r, left_r, top, mask, flip_offset=offset)
            return r

        return jax.lax.fori_loop(0, self._env.consts.max_num_flags, draw_flag, raster)
