import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin


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