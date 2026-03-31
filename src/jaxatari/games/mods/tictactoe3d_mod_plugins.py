import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin




#########################################################################################################################################################################################
################################################################# SIMPLE MODIFICATION #################################################################################################
#########################################################################################################################################################################################




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
    awarding the win to the player with the most unblocked '3-in-a-row' potential lines.
    """
    
    # JIT compile the hook to run efficiently on the GPU/TPU.
    # static_argnums=(0,) tells JAX to treat the 'self' instance as static,
    # which is required when JIT-compiling class methods.
    @partial(jax.jit, static_argnums=(0,))
    def after_step(self, obs, state, reward, done, info):
        
        # --- 1. Define the Tie-Breaker Logic ---
        # We define this as an inner pure function to use within jax.lax.cond
        def tie_breaker(board):
            # Create binary masks for each player's pieces
            is_x = (board == self._env.consts.PLAYER_X).astype(jnp.int32)
            is_o = (board == self._env.consts.PLAYER_O).astype(jnp.int32)

            # Use JAX's highly optimized tensordot to compute how many pieces each player
            # has in every possible winning line simultaneously. 
            # WIN_MASKS contains the 76 intersecting lines of the 3D grid.
            x_counts = jnp.tensordot(self._env.consts.WIN_MASKS, is_x, axes=((1, 2, 3), (0, 1, 2)))
            o_counts = jnp.tensordot(self._env.consts.WIN_MASKS, is_o, axes=((1, 2, 3), (0, 1, 2)))

            # Define a "potential line": 3 pieces of one player and exactly 0 of the opponent.
            # If the opponent has a piece in that line (blocked), it does not count.
            x_potentials = jnp.sum(jnp.logical_and(x_counts == 3, o_counts == 0))
            o_potentials = jnp.sum(jnp.logical_and(o_counts == 3, x_counts == 0))

            # Return the ID of whoever has more potentials. 
            # If it's a tie, return EMPTY (0).
            return jnp.where(
                x_potentials > o_potentials, self._env.consts.PLAYER_X,
                jnp.where(o_potentials > x_potentials, self._env.consts.PLAYER_O, self._env.consts.EMPTY)
            )

        # --- 2. Check the Sudden Death Condition ---
        # We must use jnp.logical_and instead of Python's 'and' for JIT compatibility.
        # Trigger ONLY if the game has reached 16 moves AND no one has won naturally yet.
        trigger_sd = jnp.logical_and(
            state.move_count >= 16, 
            state.winner == self._env.consts.EMPTY
        )

        # We cannot use standard 'if/else' statements inside JIT.
        # jax.lax.cond ensures only the correct branch is executed based on trigger_sd.
        new_winner = jax.lax.cond(
            trigger_sd,
            lambda _: tie_breaker(state.board),
            lambda _: state.winner,
            operand=None
        )

        # --- 3. Hijack the Win Animation & Reward ---
        # If sudden death triggers, we manually force the game into Phase 1 of the win sequence
        # (the blackout phase) to give visual feedback that the game ended.
        new_win_phase = jnp.where(trigger_sd, jnp.int32(1), state.win_phase)
        new_win_timer = jnp.where(trigger_sd, jnp.int32(self._env.consts.BLACKOUT_FRAMES), state.win_timer)

        # Assign the reward immediately. This is crucial for RL agents (PPO/PQN).
        # By providing the +/- 1.0 reward based on the tie-breaker logic, the agent
        # learns to prioritize building 3-in-a-rows even if it can't finish the 4th.
        new_reward = jnp.where(
            trigger_sd,
            jnp.where(new_winner == self._env.consts.PLAYER_X, 1.0,
            jnp.where(new_winner == self._env.consts.PLAYER_O, -1.0, 0.0)),
            reward
        )

        # --- 4. Update the State and Observation ---
        # In functional programming, we don't mutate variables in place.
        # We create a new state object replacing only the modified fields.
        modified_state = state.replace(
            winner=new_winner,
            win_phase=new_win_phase,
            win_timer=new_win_timer
        )

        # Regenerate the observation dictionary so the CNN receives the updated
        # game_over and winner status correctly.
        modified_obs = self._env._get_observation(modified_state)
        
        return modified_obs, modified_state, new_reward, done, info
    


class TemporalPenaltyMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "STEP_PENALTY": -0.0003
    }

class MisereMod(JaxAtariPostStepModPlugin):
    """
    Misère (anti-win) mode:
    - If X completes 4-in-a-row, that becomes a LOSS for the agent.
    - If O completes 4-in-a-row, that becomes a WIN for the agent.
    - Non-terminal rewards remain unchanged.

    This version does not modify the base environment.
    It only rewrites reward / winner interpretation after the step.
    """

    @partial(jax.jit, static_argnums=(0,))
    def after_step(self, obs, state, reward, done, info):
        
        jax.debug.print("base_reward={} base_winner={} -> new_reward={} new_winner={}",reward, state.winner, new_reward, new_winner
)
        x_won = state.winner == self._env.consts.PLAYER_X
        o_won = state.winner == self._env.consts.PLAYER_O

        # Flip terminal reward semantics
        new_reward = jnp.where(
            x_won,
            jnp.float32(-1.0),
            jnp.where(
                o_won,
                jnp.float32(+1.0),
                reward
            )
        )

        # Swap displayed winner so logs / terminal observation match misère semantics
        new_winner = jnp.where(
            x_won,
            jnp.int32(self._env.consts.PLAYER_O),
            jnp.where(
                o_won,
                jnp.int32(self._env.consts.PLAYER_X),
                state.winner
            )
        )

        modified_state = state.replace(winner=new_winner)
        modified_obs = self._env._get_observation(modified_state)

        # Recompute info from the modified public state so winner/game phase stay consistent
        modified_info = self._env._get_info(modified_state, info["last_move_action"])

        return modified_obs, modified_state, new_reward, done, modified_info

    
#########################################################################################################################################################################################
################################################################# COMPLEX MODIFICATION #################################################################################################
#########################################################################################################################################################################################


class VariableIntelligenceMod(JaxAtariInternalModPlugin):
    """
    Overrides the CPU AI to use a vectorized Minimax-lite (Depth-2 Lookahead).
    Evaluates all 64 possible CPU moves, and simulates all 64 possible 
    Agent responses using jax.vmap, completely bypassing JAX's recursion limits.
    """

    # Updated signature to match the base environment: (self, board, key)
    @partial(jax.jit, static_argnums=(0,))
    def _compute_cpu_move(self, board, key):
        # 1. Create an array of all 64 possible board indices
        all_moves = jnp.arange(64)
        
        # --- DEPTH 1: Evaluate CPU Moves ---
        def evaluate_cpu_move(move_idx):
            # Is this move even legal? Use the passed 'board' array directly.
            is_valid = (board.flatten()[move_idx] == self._env.consts.EMPTY)
            
            # Simulate the board state if the CPU played here
            sim_board = board.flatten().at[move_idx].set(self._env.consts.PLAYER_O).reshape((4, 4, 4))
            
            # Did this move win the game instantly?
            is_o = (sim_board == self._env.consts.PLAYER_O).astype(jnp.int32)
            o_counts = jnp.tensordot(self._env.consts.WIN_MASKS, is_o, axes=((1, 2, 3), (0, 1, 2)))
            cpu_won = jnp.any(o_counts == 4)
            
            # --- DEPTH 2: Evaluate Agent's Best Response ---
            def evaluate_agent_response(resp_idx):
                resp_valid = (sim_board.flatten()[resp_idx] == self._env.consts.EMPTY)
                resp_board = sim_board.flatten().at[resp_idx].set(self._env.consts.PLAYER_X).reshape((4, 4, 4))
                
                is_x = (resp_board == self._env.consts.PLAYER_X).astype(jnp.int32)
                x_counts = jnp.tensordot(self._env.consts.WIN_MASKS, is_x, axes=((1, 2, 3), (0, 1, 2)))
                agent_won = jnp.any(x_counts == 4)
                
                # Return 1 if the agent can legally win here, else 0
                return jnp.where(jnp.logical_and(resp_valid, agent_won), 1, 0)
            
            # Vectorize the Agent's response over all 64 possible squares on the simulated board
            agent_wins = jax.vmap(evaluate_agent_response)(all_moves)
            # If any of the 64 responses result in an Agent win, this CPU move is highly dangerous!
            gives_agent_win = jnp.any(agent_wins == 1)
            
            # Heuristic fallback: Count how many 3-in-a-rows the CPU builds with this move
            cpu_3_in_a_row = jnp.sum(o_counts == 3)
            
            # --- Scoring System ---
            score = jnp.where(
                jnp.logical_not(is_valid), -9999,
                jnp.where(
                    cpu_won, 1000,
                    jnp.where(
                        gives_agent_win, -500,
                        cpu_3_in_a_row
                    )
                )
            )
            return score
        
        # 2. Vectorize the CPU evaluation over all 64 possible moves!
        move_scores = jax.vmap(evaluate_cpu_move)(all_moves)
        
        # 3. Pick the move with the absolute highest score
        best_flat_idx = jnp.argmax(move_scores)
        
        # Return the flat index as expected by the base environment
        return best_flat_idx
    



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
    
class VanishingPiecesMod(JaxAtariInternalModPlugin):
    """
    Pieces disappear after N full rounds.
    Decay happens only when the CPU (O) places a piece, to avoid penalizing the agent's strategic planning on its turn.
    """

    PIECE_LIFETIME = 10

    @partial(jax.jit, static_argnums=(0,))
    def _update_piece_timers_after_placement(self, prev_board, prev_piece_timers, new_board):
        empty = jnp.int32(self._env.consts.EMPTY)
        player_x = jnp.int32(self._env.consts.PLAYER_X)
        player_o = jnp.int32(self._env.consts.PLAYER_O)

        # Detect a newly placed X or O piece
        new_piece_mask = jnp.logical_and(
            prev_board == empty,
            jnp.logical_or(
                new_board == player_x,
                new_board == player_o,
            )
        )

        did_place = jnp.any(new_piece_mask)

        # Which piece was newly placed?
        placed_x = jnp.any(jnp.logical_and(new_piece_mask, new_board == player_x))
        placed_o = jnp.any(jnp.logical_and(new_piece_mask, new_board == player_o))

        def handle_placement(_):
            def decay_and_refresh(_):
                # 1. Decay all existing timers once per full round (on O move only)
                decayed_timers = jnp.maximum(prev_piece_timers - 1, 0)

                # 2. Remove expired X/O pieces from previous board
                board_after_decay = jnp.where(
                    decayed_timers > 0,
                    prev_board,
                    jnp.where(
                        jnp.logical_or(
                            prev_board == player_x,
                            prev_board == player_o
                        ),
                        jnp.uint8(empty),
                        prev_board
                    )
                )

                # 3. Apply the newly placed piece onto the decayed board
                refreshed_board = jnp.where(new_piece_mask, new_board, board_after_decay)

                # 4. Reset timer for the newly placed piece
                refreshed_timers = jnp.where(
                    new_piece_mask,
                    jnp.int32(self.PIECE_LIFETIME),
                    decayed_timers
                )

                return refreshed_board, refreshed_timers

            def refresh_without_decay(_):
                # X move: just place the piece and give it fresh lifetime
                refreshed_timers = jnp.where(
                    new_piece_mask,
                    jnp.int32(self.PIECE_LIFETIME),
                    prev_piece_timers
                )
                return new_board, refreshed_timers

            return jax.lax.cond(
                placed_o,
                decay_and_refresh,
                refresh_without_decay,
                operand=None
            )

        def no_change(_):
            return new_board, prev_piece_timers

        return jax.lax.cond(did_place, handle_placement, no_change, operand=None)