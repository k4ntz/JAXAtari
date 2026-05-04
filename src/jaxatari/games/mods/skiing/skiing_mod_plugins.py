import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Optional, Tuple
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_skiing import _enforce_min_sep_x, SkiingState, SkiingObservation

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


class InvertFlagsMod(JaxAtariPostStepModPlugin):
    """
    Flips the orientation or state of the slalom flags on the course.
    """
    def run(self, prev_state, new_state):
        c = self._env.consts
        # Recalculate gate crossing with inverted logic
        left_x  = prev_state.flags[:, 0]
        right_x = left_x + c.flag_distance
        new_x = new_state.skier_x

        # Inverted: you MUST go OUTSIDE the flags
        eligible = jnp.logical_or(new_x <= left_x, new_x >= right_x)
        
        crossed = jnp.logical_and(prev_state.flags[:, 1] > c.skier_y,
                                   new_state.flags[:, 1] <= c.skier_y)
        
        gate_pass = jnp.logical_and(eligible, jnp.logical_and(crossed, jnp.logical_not(prev_state.flags_passed)))
        
        # Undo original scoring from step
        orig_eligible = jnp.logical_and(new_x > left_x, new_x < right_x)
        orig_gate_pass = jnp.logical_and(orig_eligible, jnp.logical_and(crossed, jnp.logical_not(prev_state.flags_passed)))
        
        corrected_score = new_state.successful_gates + jnp.sum(orig_gate_pass) - jnp.sum(gate_pass)
        new_flags_passed = jnp.logical_or(prev_state.flags_passed, gate_pass)
        
        return new_state.replace(successful_gates=corrected_score, flags_passed=new_flags_passed)


class InvertFlagColorsMod(JaxAtariInternalModPlugin):
    """
    Inverts the colors of the slalom flags (all flags become red, the last one becomes blue).
    """
    constants_overrides = {
        "invert_flag_colors": True,
    }


class MovingFlagsMod(JaxAtariPostStepModPlugin):
    """
    Causes the flags to slide horizontally across the screen while you ski.
    """
    def run(self, prev_state, new_state):
        c = self._env.consts
        # Move flags sinusoidally
        shift = jnp.sin(new_state.step_count.astype(jnp.float32) * 0.05) * 1.5
        new_flags_x = new_state.flags[:, 0] + shift
        
        # Keep them within bounds
        min_fx = c.border_left
        max_fx = c.screen_width - c.border_right - c.flag_distance
        new_flags_x = jnp.clip(new_flags_x, min_fx, max_fx)
        
        new_flags = new_state.flags.at[:, 0].set(new_flags_x)
        return new_state.replace(flags=new_flags)


class RandomFlagsMod(JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin):
    """
    Randomizes the horizontal placement of every flag.
    """
    def after_reset(self, obs, state):
        c = self._env.consts
        key, subkey = jax.random.split(state.key)
        min_fx = c.border_left + 20
        max_fx = c.screen_width - c.border_right - c.flag_distance - 20
        new_x = jax.random.uniform(subkey, (c.max_num_flags,), minval=min_fx, maxval=max_fx)
        new_flags = state.flags.at[:, 0].set(new_x)
        return obs, state.replace(flags=new_flags, key=key)

    def run(self, prev_state, new_state):
        c = self._env.consts
        # Detect if flags respawned
        respawned = new_state.flags[:, 1] > prev_state.flags[:, 1] + 100
        key, subkey = jax.random.split(new_state.key)
        min_fx = c.border_left + 20
        max_fx = c.screen_width - c.border_right - c.flag_distance - 20
        rand_x = jax.random.uniform(subkey, (c.max_num_flags,), minval=min_fx, maxval=max_fx)
        
        new_x = jnp.where(respawned, rand_x, new_state.flags[:, 0])
        new_flags = new_state.flags.at[:, 0].set(new_x)
        return new_state.replace(flags=new_flags, key=key)


class FlagFlurryMod(JaxAtariInternalModPlugin):
    """
    Dramatically increases the number of flags on the mountain.
    """
    constants_overrides = {
        "max_num_flags": 8,
    }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(1701)) -> Tuple[SkiingObservation, SkiingState]:
        c = self._env.consts
        _, new_key = jax.random.split(key, 2)

        row_spacing = jnp.float32(20.0) # More frequent
        base_y = jnp.float32(60.0)

        # Flags: more rows
        r_flags = jnp.arange(c.max_num_flags, dtype=jnp.float32) * 2.0 + 3.0
        flags_y = base_y + r_flags * row_spacing
        
        flags_x = self._env._get_initial_flags_x()
        flags = jnp.stack([flags_x, flags_y], axis=1)

        # Trees
        trees_x = self._env._get_initial_trees_x()
        trees_per_row = jnp.maximum(1, c.max_num_trees // 4)
        i_t = jnp.arange(c.max_num_trees, dtype=jnp.int32)
        row_idx_t = i_t // trees_per_row
        base_offsets_t = jnp.array([0, 1, 4, 5], dtype=jnp.float32)
        r_trees = (row_idx_t // 4) * 8.0 + jnp.take(base_offsets_t, row_idx_t % 4)
        trees_y = base_y + r_trees * jnp.float32(31.0) # Original row spacing for trees
        stagger_t = ((i_t * 7) % 15).astype(jnp.float32) - 7.0
        trees_y = trees_y + stagger_t

        min_sep_tree = 0.5*(jnp.float32(c.tree_width)+jnp.float32(c.tree_width)) + jnp.float32(c.sep_margin_tree_tree)
        xmin = jnp.float32(c.border_left)
        xmax = jnp.float32(c.screen_width - c.border_right)

        def adj_tree_i(i, tx):
            x0 = tx[i]
            x_adj = _enforce_min_sep_x(x0, tx, min_sep_tree, xmin, xmax, n_valid=jnp.array(i, dtype=jnp.int32))
            return tx.at[i].set(self._env._enforce_tree_gap(x_adj))

        trees_x = jax.lax.fori_loop(0, c.max_num_trees, adj_tree_i, trees_x)
        trees_type = jnp.arange(c.max_num_trees, dtype=jnp.float32) % 4.0
        trees = jnp.stack([trees_x, trees_y, trees_type], axis=1)

        # Moguls
        min_rx = jnp.int32(c.border_left + 50)
        max_rx = jnp.int32(c.screen_width - c.border_right - 50)
        span_rx = max_rx - min_rx + 1
        moguls_x = (min_rx + ((jnp.arange(c.max_num_moguls, dtype=jnp.int32) * 19) % span_rx)).astype(jnp.float32)
        moguls_per_row = jnp.maximum(1, c.max_num_moguls // 2)
        i_r = jnp.arange(c.max_num_moguls, dtype=jnp.int32)
        row_idx_r = i_r // moguls_per_row
        base_offsets_r = jnp.array([2, 6], dtype=jnp.float32)
        r_moguls = (row_idx_r // 2) * 8.0 + jnp.take(base_offsets_r, row_idx_r % 2)
        moguls_y = base_y + r_moguls * jnp.float32(31.0)
        stagger_r = ((i_r * 11) % 15).astype(jnp.float32) - 7.0
        moguls_y = moguls_y + stagger_r
        
        min_sep_mogul_tree = 0.5*(jnp.float32(c.mogul_width)+jnp.float32(c.tree_width)) + jnp.float32(c.sep_margin_tree_mogul)
        min_sep_mogul_mogul = 0.5*(jnp.float32(c.mogul_width)+jnp.float32(c.mogul_width)) + jnp.float32(c.sep_margin_mogul_mogul)
        xmin_r = jnp.float32(c.border_left + 50)
        xmax_r = jnp.float32(c.screen_width - c.border_right - 50)
        tree_xs_fixed = trees[:, 0]

        def adj_mogul_i(i, rx):
            x0 = rx[i]
            x1 = _enforce_min_sep_x(x0, tree_xs_fixed, min_sep_mogul_tree, xmin_r, xmax_r, n_valid=jnp.array(tree_xs_fixed.shape[0], dtype=jnp.int32))
            x2 = _enforce_min_sep_x(x1, rx, min_sep_mogul_mogul, xmin_r, xmax_r, n_valid=jnp.array(i, dtype=jnp.int32))
            return rx.at[i].set(x2)

        moguls_x = jax.lax.fori_loop(0, c.max_num_moguls, adj_mogul_i, moguls_x)
        moguls = jnp.stack([moguls_x, moguls_y], axis=1)

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
        obs = self._env._get_observation(state)
        return obs, state


class MogulsToTreesMod(JaxAtariInternalModPlugin):
    """
    Transforms all the small snow bumps (moguls) into solid trees.
    """
    constants_overrides = {
        "moguls_collidable": True,
        "mogul_height": 30,
    }
    asset_overrides = {
        "mogul": {'name': 'mogul', 'type': 'single', 'file': 'tree_0.npy'}
    }


class ClassicTreesMod(JaxAtariInternalModPlugin):
    """
    Replaces the default tree sprites with classic versions.
    """
    asset_overrides = {
        "tree_group": {
            'name': 'tree_group', 
            'type': 'group', 
            'files': [
                'classic_tree_0.npy',
                'classic_tree_1.npy',
                'classic_tree_2.npy',
                'classic_tree_3.npy'
            ]
        }
    }


class ThinMogulsMod(JaxAtariInternalModPlugin):
    """
    Replaces the standard moguls with thinner ones.
    """
    asset_overrides = {
        "mogul": {'name': 'mogul', 'type': 'single', 'file': 'thin_mogul.npy'}
    }


class BlueSkiierMod(JaxAtariInternalModPlugin):
    """
    Replaces the skier sprites with blue versions.
    """
    constants_overrides = {
        "blue_skier": True
    }


class GreenFlagsMod(JaxAtariInternalModPlugin):
    """
    Recolors all flags to green.
    """
    constants_overrides = {
        "green_flags": True
    }


class RewardAtGateMod(JaxAtariInternalModPlugin):
    """
    Changes the reward function to give +1 reward for passing a gate,
    instead of the original ALE reward (time penalty + massive end penalty).
    """
    constants_overrides = {
        "USE_ORIGINAL_ALE_REWARD": False,
    }

class InvertFlagsModOld(JaxAtariInternalModPlugin):
    """
    Swaps flag colors: blue gates become red and the special 20th gate becomes blue.
    NOTE: This mod had a name conflict with a new mod from a recent merge. To make sure we dont lose anything I will keep it in but unregistered
    """

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
