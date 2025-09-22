import pytest
import jax

@pytest.mark.requires_snapshot
class TestRegression:
    """
    A test suite for regression testing using snapshots.
    This ensures that game mechanics do not change unexpectedly.
    Make sure that syrupy is installed (TODO: add it to requirements if everything works as intended)
    """

    def test_trajectory_snapshot(self, wrapped_env, snapshot, game_name):
        """
        Verifies the exact final game state after a fixed sequence of actions.
        The action sequence is deterministically generated based on the environment's
        specific action space, making this test generalizable to any game.
        """
        # 1. Use a single, fixed master key for the entire deterministic process to ensure reproducibility.
        master_key = jax.random.key(42)
        action_key, env_key = jax.random.split(master_key)

        # 2. Deterministically generate valid action sequence for the specific environment.
        num_actions = wrapped_env.action_space().n
        num_steps = 500
        action_subkeys = jax.random.split(action_key, num_steps)
        actions = jax.vmap(
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=num_actions)
        )(action_subkeys)

        # 3. Run simulation using the environment key and the generated actions.
        _obs, state = wrapped_env.reset(env_key)

        for action in actions:
            env_key, step_key = jax.random.split(env_key)
            _obs, state, _reward, done, _info = wrapped_env.step(state, action)

            if done:
                break

        # 4. Assert final state against the saved baseline snapshot.
        assert state == snapshot