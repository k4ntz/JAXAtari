import pytest
import gymnasium
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Dict, Tuple, Union, Optional
import functools
import numpy as np
import collections
from gymnasium.utils import env_checker

from jaxatari.gym_wrapper import GymnasiumJaxAtariWrapper, JaxAtariFuncEnv, to_gymnasium_space


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def gym_env(raw_env):
    """
    Provides an instance of the GymnasiumJaxAtariWrapper for a given raw_env.
    The `raw_env` is automatically provided and parameterized by conftest.py,
    ensuring these tests run for every specified game.
    """
    return GymnasiumJaxAtariWrapper(raw_env, render_mode="rgb_array")


@pytest.fixture
def func_env(raw_env):
    """Provides direct access to the functional adapter for testing JAX transforms."""
    return JaxAtariFuncEnv(raw_env)


# ==============================================================================
# TEST SUITE
# ==============================================================================

class TestGymnasiumApiCompliance:
    """Tests core compliance with the Gymnasium API."""

    def test_gymnasium_env_checker(self, gym_env):
        """Validates the wrapper against Gymnasium's official environment checker."""
        try:
            env_checker.check_env(gym_env, skip_render_check=False)
        except Exception as e:
            pytest.fail(f"Gymnasium's check_env failed: {e}")

    def test_reset_method(self, gym_env):
        """Tests the reset method for correct return types and space containment."""
        obs, info = gym_env.reset(seed=42)

        assert isinstance(obs, np.ndarray), "Observation must be a NumPy array."
        assert isinstance(info, dict), "Info from reset must be a dictionary."
        assert gym_env.observation_space.contains(obs), "Observation from reset is not in the observation space."

    def test_step_method(self, gym_env):
        """Tests the step method for correct return types and values."""
        gym_env.reset(seed=42)
        action = gym_env.action_space.sample()
        obs, reward, terminated, truncated, info = gym_env.step(action)

        assert isinstance(obs, np.ndarray), "Observation from step must be a NumPy array."
        assert isinstance(reward, (float, np.floating)), "Reward must be a float or numpy float."
        assert isinstance(terminated, (bool, np.bool_)), "Terminated flag must be a numpy bool."
        assert isinstance(truncated, (bool, np.bool_)), "Truncated flag must be a numpy bool."
        assert isinstance(info, dict), "Info from step must be a dictionary."
        assert gym_env.observation_space.contains(obs), "Observation from step is not in the observation space."

    def test_render_method(self, gym_env):
        """Tests that rendering in 'rgb_array' mode returns a valid image."""
        gym_env.reset(seed=42)
        frame = gym_env.render()

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[2] == 3, "Rendered frame must be a HxWx3 NumPy array."
        assert frame.dtype == gym_env.observation_space.dtype, "Rendered frame dtype should match observation space dtype."

    def test_seeding_and_determinism(self, raw_env):
        """Ensures that seeding the environment produces deterministic trajectories."""
        # Create two separate wrapper instances from the same raw env
        env1 = GymnasiumJaxAtariWrapper(raw_env)
        env2 = GymnasiumJaxAtariWrapper(raw_env)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2, "Observations after a seeded reset should be identical.")

        # Use a deterministic sequence of actions
        actions = [1, 2, 0, 3, 1]
        for action in actions:
            obs1, reward1, term1, trunc1, _ = env1.step(action)
            obs2, reward2, term2, trunc2, _ = env2.step(action)

            np.testing.assert_array_equal(obs1, obs2, "Observations after a step should be identical.")
            assert reward1 == reward2, "Rewards should be identical."
            assert term1 == term2, "Termination flags should be identical."

        env1.close()
        env2.close()


class TestGymWrapperIntegration:
    """Tests that the environment correctly composes with standard Gymnasium wrappers."""

    def test_time_limit_wrapper(self, gym_env):
        """Tests integration with the TimeLimit wrapper."""
        max_steps = 15
        env = gymnasium.wrappers.TimeLimit(gym_env, max_episode_steps=max_steps)
        env.reset(seed=123)
        for i in range(max_steps - 1):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert not terminated and not truncated, "Episode should not end before the time limit."

        # This step should trigger truncation
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert truncated, "TimeLimit wrapper should signal truncation at max_episode_steps."
        env.close()

    def test_preprocessing_wrappers(self, gym_env):
        """Tests integration with observation preprocessing wrappers like Resize and Grayscale."""
        env = gymnasium.wrappers.ResizeObservation(gym_env, shape=(84, 84))
        env = gymnasium.wrappers.GrayscaleObservation(env, keep_dim=True)

        assert env.observation_space.shape == (84, 84, 1)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (84, 84, 1), "Observation shape should match the wrapped space."
        env.close()

    def test_frame_stack_wrapper(self, gym_env):
        """Tests integration with the FrameStack wrapper."""
        num_stack = 4
        env = gymnasium.wrappers.FrameStackObservation(gym_env, num_stack)

        original_shape = gym_env.observation_space.shape
        assert env.observation_space.shape == (num_stack,) + original_shape

        obs, _ = env.reset(seed=42)
        assert np.array(obs).shape == (num_stack,) + original_shape
        env.close()


class TestJaxTransforms:
    """Tests if the underlying functional environment is compatible with JAX transforms."""

    def test_jittable_transition(self, func_env):
        """Tests that the core transition function can be JIT-compiled."""
        jit_transition = jax.jit(func_env.transition)
        try:
            key = jax.random.PRNGKey(0)
            initial_state = func_env.initial(key)
            action = func_env.action_space.sample(key) # Sample a valid action
            _ = jit_transition(initial_state, action, key)
        except Exception as e:
            pytest.fail(f"JIT compilation of the functional transition failed: {e}")

    @pytest.mark.skip(reason="Skipping to debug memory issues in CI")
    def test_vmappable_environment(self, func_env):
        """Tests that the functional API can be vectorized with jax.vmap."""
        num_envs = 2
        keys = jax.random.split(jax.random.PRNGKey(1), num_envs)

        try:
            # Vmap initial state creation
            batch_initial_states = jax.vmap(func_env.initial)(keys)
            assert jax.tree.leaves(batch_initial_states)[0].shape[0] == num_envs

            # Vmap transition
            actions = jnp.arange(num_envs) % func_env.action_space.n
            batch_next_states = jax.vmap(func_env.transition)(
                batch_initial_states, actions, keys
            )
            assert jax.tree.leaves(batch_next_states)[0].shape[0] == num_envs

        except Exception as e:
            pytest.fail(f"Vmapping the functional API failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
