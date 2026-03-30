import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin


class RandomStaticBlockersMod(JaxAtariPostStepModPlugin):
    """
    Places 1 or 2 neutral blocker voxels on random board cells after reset.
    """

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        # Split RNG from state
        key, count_key, pos_key, next_key = jax.random.split(state.key, 4)

        # choose 1 or 2 blockers
        num_blockers = jax.random.randint(
            count_key,
            shape=(),
            minval=1,
            maxval=3
        ).astype(jnp.int32)

        # choose 2 unique positions from 64 cells
        flat_positions = jax.random.choice(
            pos_key,
            a=64,
            shape=(2,),
            replace=False
        )

        flat_board = state.board.reshape(-1)

        # place first blocker
        flat_board = flat_board.at[flat_positions[0]].set(
            self._env.consts.BLOCKER
        )

        # optionally place second blocker
        flat_board = jax.lax.cond(
            num_blockers == 2,
            lambda b: b.at[flat_positions[1]].set(self._env.consts.BLOCKER),
            lambda b: b,
            flat_board
        )

        new_board = flat_board.reshape((4, 4, 4))

        # update state
        modified_state = state.replace(
            board=new_board,
            key=next_key
        )

        # IMPORTANT: regenerate observation properly
        modified_obs = self._env._get_observation(modified_state)

        return modified_obs, modified_state


class RandomTurnOrderMod(JaxAtariPostStepModPlugin):
    """
    Randomly assigns whether the agent starts first or the CPU starts first.
    """

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        key, coin_key, next_key = jax.random.split(state.key, 3)

        # True -> agent starts first
        agent_starts = jax.random.bernoulli(coin_key, 0.5)

        modified_state = state.replace(
            cpu_opening_pending=jnp.logical_not(agent_starts),
            key=next_key
        )

        modified_obs = self._env._get_observation(modified_state)
        return modified_obs, modified_state
    
    
class StrictIllegalMoveMod(JaxAtariInternalModPlugin):
    """Ends episode immediately with -1 reward on illegal FIRE (occupied cell)."""
    constants_overrides = {
        "STRICT_ILLEGAL_MOVES": True
    }

class SuddenDeathMod(JaxAtariPostStepModPlugin):
    """
    Hard cap on the episode length (e.g., 16 moves). 
    If no 4-in-a-row is achieved by the limit, a custom tie-breaker logic triggers, 
    awarding the win to the player with the most '3-in-a-row' potential lines.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def after_step(self, obs, state, reward, done, info):
        
        # --- 1. Define the Tie-Breaker Logic ---
        def tie_breaker(board):
            is_x = (board == self._env.consts.PLAYER_X).astype(jnp.int32)
            is_o = (board == self._env.consts.PLAYER_O).astype(jnp.int32)

            # Use the environment's pre-calculated masks to count pieces in winning lines
            x_counts = jnp.tensordot(self._env.consts.WIN_MASKS, is_x, axes=((1, 2, 3), (0, 1, 2)))
            o_counts = jnp.tensordot(self._env.consts.WIN_MASKS, is_o, axes=((1, 2, 3), (0, 1, 2)))

            # A "potential line" has 3 pieces of one player and 0 of the other
            x_potentials = jnp.sum(jnp.logical_and(x_counts == 3, o_counts == 0))
            o_potentials = jnp.sum(jnp.logical_and(o_counts == 3, x_counts == 0))

            # Return the ID of whoever has more potentials, or EMPTY for a tie
            return jnp.where(
                x_potentials > o_potentials, self._env.consts.PLAYER_X,
                jnp.where(o_potentials > x_potentials, self._env.consts.PLAYER_O, self._env.consts.EMPTY)
            )

        # --- 2. Check the Sudden Death Condition ---
        # Trigger ONLY if we hit 16 moves AND no one has won naturally yet
        trigger_sd = jnp.logical_and(
            state.move_count >= 16, 
            state.winner == self._env.consts.EMPTY
        )

        # Calculate the new winner (fallback to the current winner if SD isn't triggered)
        new_winner = jax.lax.cond(
            trigger_sd,
            lambda _: tie_breaker(state.board),
            lambda _: state.winner,
            operand=None
        )

        # --- 3. Hijack the Win Animation & Reward ---
        # If sudden death triggers, we manually force the game into Phase 1 of the win sequence
        new_win_phase = jnp.where(trigger_sd, jnp.int32(1), state.win_phase)
        new_win_timer = jnp.where(trigger_sd, jnp.int32(self._env.consts.BLACKOUT_FRAMES), state.win_timer)

        # Assign reward immediately so the RL agent learns from the tie-breaker
        new_reward = jnp.where(
            trigger_sd,
            jnp.where(new_winner == self._env.consts.PLAYER_X, 1.0,
            jnp.where(new_winner == self._env.consts.PLAYER_O, -1.0, 0.0)),
            reward
        )

        # Update the state
        modified_state = state.replace(
            winner=new_winner,
            win_phase=new_win_phase,
            win_timer=new_win_timer
        )

        # Regenerate observation to match the updated state
        modified_obs = self._env._get_observation(modified_state)
        
        return modified_obs, modified_state, new_reward, done, info