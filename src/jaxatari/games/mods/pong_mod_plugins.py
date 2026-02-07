import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_pong import PongState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
from jaxatari.environment import JAXAtariAction as Action

# --- 1. Individual Mod Plugins ---
class LazyEnemyMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["random_enemy"]

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """
        Replaces the base _enemy_step logic.
        Access the environment via self._env (set by JaxAtariModController).
        """
        should_move = (state.step_counter % 8 != 0) & (state.ball_vel_x < 0)
        direction = jnp.sign(state.ball_y - state.enemy_y)
        new_y = state.enemy_y + (direction * self._env.consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        final_y = jax.lax.cond(should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None)
        return state.replace(enemy_y=final_y.astype(jnp.int32))

class RandomEnemyMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["lazy_enemy"]

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """
        Replaces the base _enemy_step logic.
        'self_env' is the bound JaxPong instance.
        'key' is now used for randomness.
        """
        # Split key: use one part for randomness, keep remainder for state
        rng_key, unused_key = jax.random.split(state.key)
        random_dir = jax.random.choice(rng_key, jnp.array([-1, 1]))
        random_cond = state.step_counter % 3 == 0
        new_y = state.enemy_y + (random_dir * self._env.consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        # Clamp to screen bounds
        new_y = jnp.clip(
            new_y,
            self._env.consts.WALL_TOP_Y + self._env.consts.WALL_TOP_HEIGHT - 10,
            self._env.consts.WALL_BOTTOM_Y - 4,
        )

        final_y = jax.lax.cond(random_cond, lambda _: new_y, lambda _: state.enemy_y, operand=None)
        # Return unused_key; step() will replace with new_state_key at the end
        return state.replace(enemy_y=final_y.astype(jnp.int32), key=unused_key)



class AlwaysZeroScoreMod(JaxAtariPostStepModPlugin):    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        return new_state.replace(
            player_score=jnp.array(0, dtype=jnp.int32),
            enemy_score=jnp.array(0, dtype=jnp.int32)
        )
    

class LinearMovementMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "PLAYER_ACCELERATION": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    }

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: PongState, action: chex.Array) -> PongState:
        up = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        down = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        # Direct movement: move 2 pixels per frame when input pressed
        move_amount = jnp.array(2, dtype=jnp.int32)
        
        new_player_y = state.player_y
        new_player_y = jax.lax.cond(
            up,
            lambda y: y - move_amount,
            lambda y: y,
            operand=new_player_y,
        )
        
        new_player_y = jax.lax.cond(
            down,
            lambda y: y + move_amount,
            lambda y: y,
            operand=new_player_y,
        )

        # Hard boundaries
        new_player_y = jnp.clip(
            new_player_y,
            self._env.consts.WALL_TOP_Y + self._env.consts.WALL_TOP_HEIGHT - 10,
            self._env.consts.WALL_BOTTOM_Y - 4,
        )

        return state.replace(
            player_y=new_player_y,
            player_speed=jnp.array(0, dtype=jnp.int32),
            acceleration_counter=jnp.array(0, dtype=jnp.int32),
            buffer=new_player_y
        )