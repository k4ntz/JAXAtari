import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_pacman import PacmanState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin

# -------------------------------------------------------------
# Part 1: Simple Modifications
# -------------------------------------------------------------

class FasterPacmanMod(JaxAtariInternalModPlugin):
    """
    Faster Pac-Man (+20% Score):
    Change: Set PLAYER_SPEED=2 and add a 20% score multiplier (12 for dot, 60 for power).
    Effect: Makes Pac-Man highly agile, rewarding riskier, high-speed maneuvers.
    """
    # Simply override the constants at initialization
    constants_overrides = {
        "PLAYER_SPEED": 2,
        "PELLET_DOT_SCORE": 12,
        "PELLET_POWER_SCORE": 60
    }

class SlowerGhostsMod(JaxAtariInternalModPlugin):
    """
    Slower Ghosts (Easy Mode):
    Change: Set ghost speed to 0.5 (moves every other frame).
    Effect: Drastically reduces the threat level.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _ghost_step(self, state: PacmanState, keys: jax.Array) -> PacmanState:
        """
        Intercepts ghost updates. We only call the base logic if
        step_counter % 2 == 0; otherwise ghosts keep their previous state.
        """
        should_move = state.step_counter % 2 == 0
        return jax.lax.cond(
            should_move,
            lambda s: type(self._env)._ghost_step(self._env, s, keys),
            lambda s: s,
            state,
        )

class NoFrightMod(JaxAtariInternalModPlugin):
    """
    No Frightened State (Defensive Play):
    Change: Remove the power pellet effect completely.
    """
    # Overriding the duration to 0 means eating a power pellet grants score but instantly expires.
    constants_overrides = {
        "FRIGHTENED_DURATION": 0
    }

class HalfDotsMod(JaxAtariPostStepModPlugin):
    """
    Half Dots (Shorter Episodes):
    Change: Reduce total dots on map to ~120 by pre-collecting half of them.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        """
        Instantly collects half of the dots identically upon reset.
        """
        rng_key, pickup_key = jax.random.split(state.key)
        
        layout = self._env.consts.MAZE_LAYOUT
        valid_dots = (layout == 2)
        
        # Randomly select ~50% of the dots to be removed
        coin_flips = jax.random.bernoulli(pickup_key, p=0.5, shape=valid_dots.shape)
        dots_to_remove = jnp.logical_and(valid_dots, coin_flips)
        
        newly_removed = jnp.logical_and(state.pellets_collected == 0, dots_to_remove)
        new_pellets_collected = jnp.where(
            newly_removed,
            jnp.array(1, dtype=jnp.int32),
            state.pellets_collected,
        ).astype(jnp.int32)

        removed_count = jnp.sum(newly_removed)
        new_dots_remaining = state.dots_remaining - removed_count.astype(jnp.int32)
        
        new_state = state._replace(
            key=rng_key,
            pellets_collected=new_pellets_collected,
            dots_remaining=new_dots_remaining
        )
        
        return obs, new_state


class RandomStartMod(JaxAtariPostStepModPlugin):
    """
    Random Start Position (Robustness):
    Change: Pac-Man spawns at a random valid tile instead of the center.
    """
    def _find_nearest_node_idx_jax(self, x: jax.Array, y: jax.Array) -> jax.Array:
        node_x = jnp.asarray(self._env.node_positions_x, dtype=jnp.int32)
        node_y = jnp.asarray(self._env.node_positions_y, dtype=jnp.int32)
        dx = node_x - x.astype(jnp.int32)
        dy = node_y - y.astype(jnp.int32)
        dist_sq = dx * dx + dy * dy
        return jnp.argmin(dist_sq).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        rng_key, spawn_key = jax.random.split(state.key)
        
        layout = self._env.consts.MAZE_LAYOUT
        _, w = layout.shape
        valid_mask = (layout.flatten() == 0)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        flat_idx = jax.random.categorical(spawn_key, logits)
        
        spawn_y_tile = flat_idx // w
        spawn_x_tile = flat_idx % w
        spawn_px = (spawn_x_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        spawn_py = (spawn_y_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)

        spawn_node_idx = self._find_nearest_node_idx_jax(spawn_px, spawn_py)
        
        new_state = state._replace(
            key=rng_key,
            player_x=spawn_px.astype(jnp.int32),
            player_y=spawn_py.astype(jnp.int32),
            player_target_node_index=spawn_node_idx,
            player_current_node_index=spawn_node_idx
        )
        
        return obs, new_state

# -------------------------------------------------------------
# Part 2: Difficult Modifications (Requires Base Upgrades)
# -------------------------------------------------------------
class CoopMultiplayerMod(JaxAtariPostStepModPlugin):
    """
    Cooperative-style mode.
    Full multi-agent Pacman support requires broader base-env changes; this mod keeps
    gameplay JAX-safe by adding a support bonus (extra life) and randomizing spawn.
    """
    def _find_nearest_node_idx_jax(self, x: jax.Array, y: jax.Array) -> jax.Array:
        node_x = jnp.asarray(self._env.node_positions_x, dtype=jnp.int32)
        node_y = jnp.asarray(self._env.node_positions_y, dtype=jnp.int32)
        dx = node_x - x.astype(jnp.int32)
        dy = node_y - y.astype(jnp.int32)
        dist_sq = dx * dx + dy * dy
        return jnp.argmin(dist_sq).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        rng_key, spawn_key = jax.random.split(state.key)
        
        layout = self._env.consts.MAZE_LAYOUT
        _, w = layout.shape
        valid_mask = (layout.flatten() == 0)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        flat_idx = jax.random.categorical(spawn_key, logits)
        
        spawn_y_tile = flat_idx // w
        spawn_x_tile = flat_idx % w
        
        spawn_px = (spawn_x_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        spawn_py = (spawn_y_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        
        spawn_node_idx = self._find_nearest_node_idx_jax(spawn_px, spawn_py)
        
        new_state = state._replace(
            key=rng_key,
            lives=jnp.maximum(state.lives, jnp.array(2, dtype=jnp.int32)),
            player_x=spawn_px.astype(jnp.int32),
            player_y=spawn_py.astype(jnp.int32),
            player_current_node_index=spawn_node_idx.astype(jnp.int32),
            player_target_node_index=spawn_node_idx.astype(jnp.int32),
        )
        
        return self._env._get_observation(new_state), new_state
