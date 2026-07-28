import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_mspacman import JaxPacman, GhostMode, reset_game
from jaxatari.games.mspacman_mazes import MsPacmanMaze

class FruitGhostBonusMod(JaxAtariInternalModPlugin):
    """
    Mod that deactivates points for pellets and power pellets,
    but multiplies rewards for eating ghosts and fruits by 4.
    """
    constants_overrides = {
        "PELLET_POINTS": 0,
        "POWER_PELLET_POINTS": 0,
        "FRUIT_REWARDS": jnp.array([400, 800, 2000, 2800, 4000, 8000, 20000]),
        "EAT_GHOSTS_BASE_POINTS": 800,
    }

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


class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod: black background, green walls, green ghosts, white pacman."""
    name = "matrix_theme"
    
    constants_overrides = {
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_PACMAN': (255, 255, 255),
        'RGB_WALLS': (0, 200, 0),
        'RGB_PATH': (0, 0, 0),
        'RGB_PELLETS': (0, 255, 0),
        'RGB_GHOST_BLINKY': (50, 255, 50),
        'RGB_GHOST_PINKY': (0, 255, 100),
        'RGB_GHOST_INKY': (0, 180, 0),
        'RGB_GHOST_SUE': (100, 255, 100),
        'RGB_GHOST_FRIGHTENED': (0, 100, 0),
        'RGB_GHOST_BLINKING': (150, 255, 150),
        'RGB_FRUIT': (0, 255, 0),
        'RGB_SCORE': (0, 255, 0),
    }

# Simple Modifications
class RandomPowerPelletsMod(JaxAtariPostStepModPlugin):
    """New power pellets spawn randomly on the playing field at regular intervals."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Trigger every 500 steps
        spawn_step = (new_state.step_count % 500 == 0) & (new_state.step_count > 0)

        def spawn_pellet(state):
            key, subkey = jax.random.split(state.key)
            # Pick a random power pellet index (0 to 3)
            idx = jax.random.randint(subkey, (), 0, 4)

            # Reactivate it
            new_power_pellets = state.level.power_pellets.at[idx].set(True)
            new_level = state.level._replace(power_pellets=new_power_pellets)

            return state.replace(level=new_level, key=key)

        return jax.lax.cond(
            spawn_step,
            lambda: spawn_pellet(new_state),
            lambda: new_state
        )


class SpeedFrenzyMod(JaxAtariInternalModPlugin):
    """A chaotic survival challenge: ghosts frantically change directions, and you earn half the points per pellet."""
    constants_overrides = {
        "PELLET_POINTS": 5,
        "CHASE_DURATION": 10 * 20,    # Halved from 20 seconds
        "SCATTER_DURATION": 3 * 20,   # Halved from 7 seconds
    }

class FragileGhostsMod(JaxAtariInternalModPlugin):
    """After eating a power pellet, the ghosts remain in their edible state for twice as long."""
    constants_overrides = {
        "FRIGHTENED_DURATION": 520,
    }


class GreedyPacManMod(JaxAtariPostStepModPlugin):
    """Higher rewards for eaten dots, but a continuous penalty for every second that passes."""

    # Increase base reward
    constants_overrides = {
        "PELLET_POINTS": 20,
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Apply a 1 point penalty every 10 steps (approx. twice a second)
        penalty_step = (new_state.step_count % 10 == 0) & (new_state.step_count > 0)

        new_score = jax.lax.cond(
            penalty_step & (new_state.score > 0),  # Prevent score from going below 0 (uint32)
            lambda: new_state.score - 1,
            lambda: new_state.score
        )

        return new_state.replace(score=new_score)


class CozyStartMod(JaxAtariPostStepModPlugin):
    """The ghosts remain locked in their house for 500 steps at the start of the game."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        is_cozy = new_state.step_count < 500
        wake_up_step = new_state.step_count == 500

        return jax.lax.cond(
            is_cozy,
            lambda: self._cage_ghosts(new_state),
            lambda: jax.lax.cond(
                wake_up_step,
                lambda: self._release_ghosts(new_state),  # Reset them on exactly step 500
                lambda: new_state  # Let the engine run normally after 500
            )
        )

    def _cage_ghosts(self, state):
        ghosts = state.ghosts
        new_modes = jnp.full_like(ghosts.modes, GhostMode.ENJAILED.value)
        new_positions = jnp.full_like(ghosts.positions, self._jail_position(ghosts.positions.dtype))
        new_timers = jnp.full_like(ghosts.timers, 9999)
        new_ghosts = ghosts._replace(modes=new_modes, positions=new_positions, timers=new_timers)
        return state.replace(ghosts=new_ghosts)

    def _release_ghosts(self, state):
        """Restores the ghosts to their exact initial starting state so they activate properly."""
        ghosts = state.ghosts
        consts = self._env.consts

        # Array order: [Blinky, Pinky, Inky, Sue]
        new_modes = jnp.array([GhostMode.RANDOM, GhostMode.ENJAILED, GhostMode.ENJAILED, GhostMode.ENJAILED],
                              dtype=jnp.uint8)
        new_positions = consts.INITIAL_GHOSTS_POSITIONS
        new_timers = jnp.array([
            consts.SCATTER_DURATION,
            consts.PINKY_RELEASE_TIME,
            consts.INKY_RELEASE_TIME,
            consts.SUE_RELEASE_TIME
        ], dtype=jnp.float16)

        new_ghosts = ghosts._replace(
            modes=new_modes,
            positions=new_positions,
            timers=new_timers
        )
        return state.replace(ghosts=new_ghosts)

    def _jail_position(self, dtype):
        return jnp.array(self._env.consts.JAIL_POSITION, dtype=dtype)

# More Difficult Modifications
class InvisibleDotsMod(JaxAtariInternalModPlugin):
    """The dots are visually hidden, but can still be collected (the AI has to remember the layout)."""

    constants_overrides = {
        # Extract the array from MsPacmanMaze, convert it to a standard Python list, then to a tuple
        "RGB_PELLETS": tuple(MsPacmanMaze.PATH_COLOR.tolist()),
    }


class GhostMagnetismMod(JaxAtariPostStepModPlugin):
    """Once 70% of the dots have been collected, the ghosts pursue Ms. Pac-Man directly."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Determine which maze we are in to get the total required pellets
        maze_idx = jax.lax.switch(
            jnp.digitize(new_state.level.id, jnp.array([2, 4, 6, 8]), right=True).astype(jnp.int32),
            (lambda: 0, lambda: 1, lambda: 2, lambda: 3, lambda: 3)
        )
        total_pellets = self._env.consts.PELLETS_TO_COLLECT[maze_idx]

        # Check if collected pellets > 70%
        magnetism_active = new_state.level.collected_pellets >= (total_pellets * 0.7).astype(jnp.uint8)

        return jax.lax.cond(
            magnetism_active,
            lambda: self._magnetize_ghosts(new_state),
            lambda: new_state
        )

    def _magnetize_ghosts(self, state):
        ghosts = state.ghosts
        # Force all non-enjailed ghosts into CHASE mode
        new_modes = jnp.where(
            ghosts.modes != GhostMode.ENJAILED.value,
            GhostMode.CHASE.value,
            ghosts.modes
        )
        new_ghosts = ghosts._replace(modes=new_modes)
        return state.replace(ghosts=new_ghosts)


class TeleportationErrorMod(JaxAtariPostStepModPlugin):
    """Every 1,000 steps, Ms. Pac-Man and a random ghost suddenly swap positions."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        key, subkey = jax.random.split(new_state.key)

        teleport = (new_state.step_count % 1000 == 0) & (new_state.step_count > 0)

        def swap_positions(state):
            random_ghost_idx = jax.random.randint(subkey, (), 0, 4)

            # Use player, not pacman
            player_pos = state.player.position
            ghost_pos = state.ghosts.positions[random_ghost_idx]

            new_player_pos = ghost_pos
            new_ghost_positions = state.ghosts.positions.at[random_ghost_idx].set(player_pos)

            player = state.player._replace(position=new_player_pos)
            ghosts = state.ghosts._replace(positions=new_ghost_positions)

            return state.replace(player=player, ghosts=ghosts, key=key)

        return jax.lax.cond(
            teleport,
            lambda: swap_positions(new_state),
            lambda: new_state
        )
