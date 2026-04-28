import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariPostStepModPlugin
from jaxatari.games.jax_mspacman import JaxPacman, GhostMode, reset_game

class CagedGhostsMod(JaxAtariPostStepModPlugin):
    def _jail_position(self, dtype):
        return jnp.array(self._env.consts.JAIL_POSITION, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self._cage_ghosts(state)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Always enforce caged ghosts
        return self._cage_ghosts(new_state)

    def _cage_ghosts(self, state):
        ghosts = state.ghosts
        new_modes = jnp.full_like(ghosts.modes, GhostMode.ENJAILED.value)
        new_positions = jnp.full_like(ghosts.positions, self._jail_position(ghosts.positions.dtype))
        new_timers = jnp.full_like(ghosts.timers, 9999)
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts)


class ConstantFruitsMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        new_fruit = new_state.fruit._replace(
            spawn=jnp.array(True, dtype=jnp.bool_),
            timer=jnp.array(9999, dtype=jnp.uint16)
        )
        return new_state.replace(fruit=new_fruit)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_fruit = state.fruit._replace(
            spawn=jnp.array(True, dtype=jnp.bool_),
            timer=jnp.array(9999, dtype=jnp.uint16)
        )
        new_state = state.replace(fruit=new_fruit)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state


class SetMaze1Mod(JaxAtariPostStepModPlugin):
    maze_level = 1
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), state.lives, state.score, state.key)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        transitioned = new_state.level.id != prev_state.level.id
        return jax.lax.cond(
            transitioned,
            lambda: reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), new_state.lives, new_state.score, new_state.key),
            lambda: new_state
        )

class SetMaze2Mod(JaxAtariPostStepModPlugin):
    maze_level = 3
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), state.lives, state.score, state.key)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        transitioned = new_state.level.id != prev_state.level.id
        return jax.lax.cond(
            transitioned,
            lambda: reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), new_state.lives, new_state.score, new_state.key),
            lambda: new_state
        )

class SetMaze3Mod(JaxAtariPostStepModPlugin):
    maze_level = 5
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), state.lives, state.score, state.key)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        transitioned = new_state.level.id != prev_state.level.id
        return jax.lax.cond(
            transitioned,
            lambda: reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), new_state.lives, new_state.score, new_state.key),
            lambda: new_state
        )

class SetMaze4Mod(JaxAtariPostStepModPlugin):
    maze_level = 7
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), state.lives, state.score, state.key)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        transitioned = new_state.level.id != prev_state.level.id
        return jax.lax.cond(
            transitioned,
            lambda: reset_game(self._env.consts, jnp.array(self.maze_level, dtype=jnp.uint8), new_state.lives, new_state.score, new_state.key),
            lambda: new_state
        )


class Only1GhostMod(JaxAtariPostStepModPlugin):
    num_ghosts = 1
    def _jail_position(self, dtype):
        return jnp.array(self._env.consts.JAIL_POSITION, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self._initialize_ghosts(state)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # We detect if the game has naturally reset the ghost timers
        # If new_state.step_count == 1, it means we just reset or died
        return jax.lax.cond(
            new_state.step_count == 1,
            lambda: self._initialize_ghosts(new_state),
            lambda: self._maintain_ghosts(new_state)
        )

    def _initialize_ghosts(self, state):
        key, subkey = jax.random.split(state.key)
        rand_vals = jax.random.uniform(subkey, shape=(4,))
        ranks = jnp.argsort(jnp.argsort(rand_vals))
        is_active = ranks < self.num_ghosts
        
        ghosts = state.ghosts
        new_modes = jnp.where(
            is_active,
            ghosts.modes,
            GhostMode.ENJAILED.value
        )
        new_positions = jnp.where(
            is_active[:, None],
            ghosts.positions,
            self._jail_position(ghosts.positions.dtype)
        )
        
        # Give active enjailed ghosts a very short timer so they leave immediately
        active_timers = jnp.where(
            ghosts.modes == GhostMode.ENJAILED.value,
            jnp.array(1, dtype=jnp.float16),
            ghosts.timers
        )
        
        new_timers = jnp.where(
            is_active,
            active_timers,
            jnp.array(9999, dtype=jnp.float16)
        )
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts, key=key)

    def _maintain_ghosts(self, state):
        ghosts = state.ghosts
        # Any ghost with a huge timer was marked as inactive
        is_inactive = ghosts.timers > 9000.0
        new_modes = jnp.where(is_inactive, GhostMode.ENJAILED.value, ghosts.modes)
        new_positions = jnp.where(is_inactive[:, None], self._jail_position(ghosts.positions.dtype), ghosts.positions)
        new_timers = jnp.where(is_inactive, jnp.array(9999, dtype=jnp.float16), ghosts.timers)
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts)

class Only2GhostMod(JaxAtariPostStepModPlugin):
    num_ghosts = 2
    def _jail_position(self, dtype):
        return jnp.array(self._env.consts.JAIL_POSITION, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self._initialize_ghosts(state)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return jax.lax.cond(
            new_state.step_count == 1,
            lambda: self._initialize_ghosts(new_state),
            lambda: self._maintain_ghosts(new_state)
        )

    def _initialize_ghosts(self, state):
        key, subkey = jax.random.split(state.key)
        rand_vals = jax.random.uniform(subkey, shape=(4,))
        ranks = jnp.argsort(jnp.argsort(rand_vals))
        is_active = ranks < self.num_ghosts
        
        ghosts = state.ghosts
        new_modes = jnp.where(
            is_active,
            ghosts.modes,
            GhostMode.ENJAILED.value
        )
        new_positions = jnp.where(
            is_active[:, None],
            ghosts.positions,
            self._jail_position(ghosts.positions.dtype)
        )
        
        # Give active enjailed ghosts a very short timer so they leave immediately
        active_timers = jnp.where(
            ghosts.modes == GhostMode.ENJAILED.value,
            jnp.array(1, dtype=jnp.float16),
            ghosts.timers
        )
        
        new_timers = jnp.where(
            is_active,
            active_timers,
            jnp.array(9999, dtype=jnp.float16)
        )
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts, key=key)

    def _maintain_ghosts(self, state):
        ghosts = state.ghosts
        is_inactive = ghosts.timers > 9000.0
        new_modes = jnp.where(is_inactive, GhostMode.ENJAILED.value, ghosts.modes)
        new_positions = jnp.where(is_inactive[:, None], self._jail_position(ghosts.positions.dtype), ghosts.positions)
        new_timers = jnp.where(is_inactive, jnp.array(9999, dtype=jnp.float16), ghosts.timers)
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts)

class Only3GhostMod(JaxAtariPostStepModPlugin):
    num_ghosts = 3
    def _jail_position(self, dtype):
        return jnp.array(self._env.consts.JAIL_POSITION, dtype=dtype)

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self._initialize_ghosts(state)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return jax.lax.cond(
            new_state.step_count == 1,
            lambda: self._initialize_ghosts(new_state),
            lambda: self._maintain_ghosts(new_state)
        )

    def _initialize_ghosts(self, state):
        key, subkey = jax.random.split(state.key)
        rand_vals = jax.random.uniform(subkey, shape=(4,))
        ranks = jnp.argsort(jnp.argsort(rand_vals))
        is_active = ranks < self.num_ghosts
        
        ghosts = state.ghosts
        new_modes = jnp.where(
            is_active,
            ghosts.modes,
            GhostMode.ENJAILED.value
        )
        new_positions = jnp.where(
            is_active[:, None],
            ghosts.positions,
            self._jail_position(ghosts.positions.dtype)
        )
        
        # Give active enjailed ghosts a very short timer so they leave immediately
        active_timers = jnp.where(
            ghosts.modes == GhostMode.ENJAILED.value,
            jnp.array(1, dtype=jnp.float16),
            ghosts.timers
        )
        
        new_timers = jnp.where(
            is_active,
            active_timers,
            jnp.array(9999, dtype=jnp.float16)
        )
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts, key=key)

    def _maintain_ghosts(self, state):
        ghosts = state.ghosts
        is_inactive = ghosts.timers > 9000.0
        new_modes = jnp.where(is_inactive, GhostMode.ENJAILED.value, ghosts.modes)
        new_positions = jnp.where(is_inactive[:, None], self._jail_position(ghosts.positions.dtype), ghosts.positions)
        new_timers = jnp.where(is_inactive, jnp.array(9999, dtype=jnp.float16), ghosts.timers)
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts)


class RandomGhostNavigationMod(JaxAtariPostStepModPlugin):
    """
    Mod that randomizes the navigation of all ghosts by forcing them into RANDOM mode
    whenever they would normally be in CHASE or SCATTER mode.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_state = self._randomize_ghosts(state)
        new_obs = JaxPacman.get_observation(new_state)
        return new_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return self._randomize_ghosts(new_state)

    def _randomize_ghosts(self, state):
        ghosts = state.ghosts
        # Force CHASE (1) and SCATTER (2) to RANDOM (0)
        new_modes = jnp.where(
            (ghosts.modes == GhostMode.CHASE.value) | (ghosts.modes == GhostMode.SCATTER.value),
            GhostMode.RANDOM.value,
            ghosts.modes
        )
        new_ghosts = ghosts._replace(modes=new_modes)
        return state.replace(ghosts=new_ghosts)
