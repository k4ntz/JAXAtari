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
    def _ghost_step(self,
                    px: jnp.int32, py: jnp.int32, pdir: jnp.int32,
                    g: jnp.ndarray,
                    state: PacmanState) -> jnp.ndarray:
        """
        Intercepts ghost step. We only call the base logic if step_counter % 2 == 0.
        Otherwise, we return the ghost state completely unmodified.
        """
        # Call the original method from the unwrapped environment instance
        # Python's `super` doesn't work easily here, so we get the _env ref
        base_step_result = type(self._env)._ghost_step(self._env, px, py, pdir, g, state)
        
        # Only move the ghosts every other frame to simulate 0.5 speed
        should_move = state.step_counter % 2 == 0
        
        return jax.lax.cond(
            should_move,
            lambda _: base_step_result,
            lambda _: g, # Return unmodified original ghost
            operand=None
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
        
        # Current valid dots (mask where layout == 2)
        layout = self._env.consts.MAZE_LAYOUT
        valid_dots = (layout == 2)
        
        # Randomly select ~50% of the dots to be removed
        coin_flips = jax.random.bernoulli(pickup_key, p=0.5, shape=valid_dots.shape)
        dots_to_remove = jnp.logical_and(valid_dots, coin_flips)
        
        # Update the pellets_collected mask
        new_pellets_collected = jnp.logical_or(state.pellets_collected, dots_to_remove)
        
        # Update dots_remaining
        removed_count = jnp.sum(dots_to_remove)
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
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        rng_key, spawn_key = jax.random.split(state.key)
        
        # Find all empty pathways (where layout == 0)
        layout = self._env.consts.MAZE_LAYOUT
        h, w = layout.shape
        
        # Create coordinates arrays
        y_coords, x_coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        
        # Flatten and filter for valid spawn points
        valid_mask = (layout.flatten() == 0)
        
        # Assign high weight to valid spots, 0 to invalid spots for categorical sampling
        logits = jnp.where(valid_mask, 0.0, -1e9)
        
        # Sample a random position
        flat_idx = jax.random.categorical(spawn_key, logits)
        
        # Convert back to 2D
        spawn_y_tile = flat_idx // w
        spawn_x_tile = flat_idx % w
        
        # Convert to pixels (center of tile)
        spawn_px = (spawn_x_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        spawn_py = (spawn_y_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        
        # Find nearest node for the queueing logic
        spawn_node_idx = self._env._find_nearest_node_idx(spawn_px, spawn_py)
        
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
    Cooperative Multiplayer Mode
    Overrides the reset to initialize 2 players instead of 1.
    Since `JaxPacman` uses `vmap` over all player indices internally for movement and collisions,
    all we need to do is spawn 2 initial player variables (x, y, directions, etc) upon reset.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: PacmanState):
        rng_key, spawn_key = jax.random.split(state.key)
        
        # Player 1 spawns at standard default node.
        # Let's spawn Player 2 randomly just like RandomStartMod.
        layout = self._env.consts.MAZE_LAYOUT
        h, w = layout.shape
        y_coords, x_coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        
        valid_mask = (layout.flatten() == 0)
        logits = jnp.where(valid_mask, 0.0, -1e9)
        flat_idx = jax.random.categorical(spawn_key, logits)
        
        spawn_y_tile = flat_idx // w
        spawn_x_tile = flat_idx % w
        
        spawn_px = (spawn_x_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        spawn_py = (spawn_y_tile * self._env.consts.TILE_SIZE) + (self._env.consts.TILE_SIZE // 2)
        
        spawn_node_idx = self._env._find_nearest_node_idx(spawn_px, spawn_py)
        
        # Append Player 2 state next to Player 1 state
        p2_x = jnp.array([spawn_px], dtype=jnp.int32)
        p2_y = jnp.array([spawn_py], dtype=jnp.int32)
        p2_dir = jnp.array([0], dtype=jnp.int32)
        p2_node = jnp.array([spawn_node_idx], dtype=jnp.int32)
        p2_next_dir = jnp.array([-1], dtype=jnp.int32)
        
        new_state = state._replace(
            key=rng_key,
            player_x=jnp.concatenate([state.player_x, p2_x]),
            player_y=jnp.concatenate([state.player_y, p2_y]),
            player_direction=jnp.concatenate([state.player_direction, p2_dir]),
            player_next_direction=jnp.concatenate([state.player_next_direction, p2_next_dir]),
            player_current_node_index=jnp.concatenate([state.player_current_node_index, p2_node]),
            player_target_node_index=jnp.concatenate([state.player_target_node_index, p2_node]),
            player_animation_frame=jnp.concatenate([state.player_animation_frame, jnp.array([0], dtype=jnp.int32)]),
        )
        
        # Obs will magically parse this correctly since our `_get_observation` maps natively to state shapes!
        return self._env._get_observation(new_state), new_state
