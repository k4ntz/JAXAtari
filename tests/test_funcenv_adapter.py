import pytest
import gymnasium
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Dict, Tuple, Union, Optional
import functools
import numpy as np
import collections
import jaxatari
from jaxatari.wrappers import AtariWrapper
from gymnasium.utils import env_checker

# Assume the new wrapper implementation is in a file named `thin_jaxatari_bridge.py`
from jaxatari.gym_wrapper import GymnasiumJaxAtariWrapper, JaxAtariFuncEnv, to_gymnasium_space


# ==============================================================================
# 2. TEST FIXTURES
# ==============================================================================

@pytest.fixture
def raw_jaxatari_env(request):
    """
    Provides a raw, unwrapped jaxatari environment instance.
    Uses the game specified by the --game option, or defaults to 'pong'.
    """
    game_name = request.config.getoption("--game")
    if game_name is None:
        game_name = "pong"
    return jaxatari.make(game_name)

@pytest.fixture
def gym_env(raw_jaxatari_env):
    """Provides the thin wrapper around the raw environment."""
    return GymnasiumJaxAtariWrapper(raw_jaxatari_env, render_mode="rgb_array")

@pytest.fixture
def func_env(raw_jaxatari_env):
    """Provides direct access to the functional adapter for testing JAX transforms."""
    return JaxAtariFuncEnv(raw_jaxatari_env)

# ==============================================================================
# 3. TEST SUITE
# ==============================================================================

class TestThinWrapperCompliance:
    """Tests basic API compliance and initialization logic."""

    def test_initialization_with_raw_env(self, raw_jaxatari_env):
        """Tests that the wrapper initializes correctly with a raw environment."""
        try:
            _ = GymnasiumJaxAtariWrapper(raw_jaxatari_env)
        except Exception as e:
            pytest.fail(f"Initialization with a raw environment failed: {e}")

    def test_initialization_fails_with_wrapped_env(self, raw_jaxatari_env):
        """Tests that the wrapper correctly rejects pre-wrapped environments."""
        pre_wrapped_env = AtariWrapper(raw_jaxatari_env)
        with pytest.raises(ValueError, match="only accepts raw, unwrapped"):
            _ = GymnasiumJaxAtariWrapper(pre_wrapped_env)

    def test_gymnasium_check_env(self, gym_env):
        """Tests if the thin wrapper passes Gymnasium's default environment checker."""
        try:
            # Our wrapper is now fully compliant and doesn't need unwrapping hacks.
            env_checker.check_env(gym_env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium's check_env failed on the thin wrapper: {e}")

    def test_spaces_are_converted(self, gym_env):
        """Tests that the custom jaxatari spaces are converted to gymnasium spaces."""
        assert isinstance(gym_env.observation_space, gymnasium.spaces.Box)
        assert isinstance(gym_env.action_space, gymnasium.spaces.Discrete)

class TestGymnasiumWrapperIntegration:
    """
    Tests that the thin bridge correctly composes with standard Gymnasium wrappers.
    This is the core validation of the new design philosophy.
    """

    def test_framestack_observation_wrapper(self, gym_env):
        """Tests integration with gymnasium.wrappers.FrameStackObservation."""
        num_stack = 4
        # Apply the standard FrameStack wrapper
        stacked_env = gymnasium.wrappers.FrameStackObservation(gym_env, num_stack)

        # Check that the observation space is updated correctly
        original_shape = gym_env.observation_space.shape
        assert stacked_env.observation_space.shape == (num_stack,) + original_shape

        # Check that the observation itself is stacked
        obs, _ = stacked_env.reset(seed=42)
        assert obs.shape == (num_stack,) + original_shape
        
        # After one step, the observation should still be stacked
        obs, _, _, _, _ = stacked_env.step(stacked_env.action_space.sample())
        assert obs.shape == (num_stack,) + original_shape

    def test_generic_preprocessing_wrappers(self, gym_env):
        """Tests integration with generic, composable preprocessing wrappers."""
        # The AtariPreprocessing wrapper is hard-coded to the ALE interface.
        # The correct approach for a custom env is to compose generic wrappers.
        env = gymnasium.wrappers.ResizeObservation(gym_env, shape=(84, 84))
        env = gymnasium.wrappers.GrayscaleObservation(env, keep_dim=True)

        # Check that observation space is correctly modified
        assert env.observation_space.shape == (84, 84, 1)
        assert env.observation_space.dtype == np.uint8

        # Check that actual observations match the new space
        obs, _ = env.reset(seed=42)
        assert obs.shape == (84, 84, 1)
        assert obs.dtype == np.uint8

    def test_normalize_reward_wrapper(self, gym_env):
        """Tests integration with gymnasium.wrappers.NormalizeReward."""
        # This wrapper keeps a running average of returns to normalize rewards
        normalized_env = gymnasium.wrappers.NormalizeReward(gym_env)
        
        normalized_env.reset(seed=42)
        # Run a few steps to see some rewards
        for _ in range(5):
            _, reward, _, _, _ = normalized_env.step(normalized_env.action_space.sample())
            # The normalized reward should be a float
            assert isinstance(reward, float)

    def test_timelimit_wrapper(self, gym_env):
        """Tests integration with gymnasium.wrappers.TimeLimit."""
        max_steps = 10
        # Apply the standard TimeLimit wrapper
        limited_env = gymnasium.wrappers.TimeLimit(gym_env, max_episode_steps=max_steps)

        limited_env.reset(seed=123)
        for i in range(max_steps - 1):
            _, _, terminated, truncated, _ = limited_env.step(limited_env.action_space.sample())
            # The episode should not end before the limit
            assert not terminated
            assert not truncated

        # This step should hit the time limit
        _, _, terminated, truncated, info = limited_env.step(limited_env.action_space.sample())

        # The underlying env is not done, so termination should be False
        assert not terminated
        # The TimeLimit wrapper should signal truncation
        assert truncated

class TestJaxTransformations:
    """Tests if the simplified functional core is still compatible with JAX transforms."""

    def test_is_jittable(self, func_env):
        """Tests that a single transition can be JIT-compiled."""
        jit_transition = jax.jit(func_env.transition)
        try:
            key = jax.random.PRNGKey(0)
            initial_state = func_env.initial(key)
            # jaxatari spaces don't have a sample method, so we create a valid action
            action = func_env.action_space.sample(key)
            _ = jit_transition(initial_state, action, key)
        except Exception as e:
            pytest.fail(f"JIT compilation of the thin transition function failed: {e}")

    def test_is_vmappable(self, func_env):
        """Tests that the core functions can be vectorized with jax.vmap."""
        num_envs = 8
        keys = jax.random.split(jax.random.PRNGKey(1), num_envs)

        try:
            # Vmap initial state creation
            batch_initial_states = jax.vmap(func_env.initial)(keys)
            # Check one of the pytree leaves for the batch dimension
            assert jax.tree.leaves(batch_initial_states)[0].shape[0] == num_envs

            # Vmap transition
            actions = jnp.arange(num_envs) % func_env.action_space.n
            batch_next_states = jax.vmap(func_env.transition, in_axes=(0, 0, 0))(
                batch_initial_states, actions, keys
            )
            assert jax.tree.leaves(batch_next_states)[0].shape[0] == num_envs

        except Exception as e:
            pytest.fail(f"Vmapping the thin functional API failed: {e}")


# ------------------------------ Generalized Tests --------------------------------

MAX_EPISODE_STEPS = 5000

# ---- TEST CASES ----

def test_env_creation_and_spaces(gym_env, raw_jaxatari_env):
    """
    Tests if the environment is created successfully and has valid Gymnasium spaces.
    """
    assert hasattr(gym_env, 'observation_space'), "Environment must have 'observation_space'"
    assert hasattr(gym_env, 'action_space'), "Environment must have 'action_space'"
    assert isinstance(gym_env.observation_space, gymnasium.spaces.Space), "observation_space must be a gymnasium.spaces.Space"
    assert isinstance(gym_env.action_space, gymnasium.spaces.Space), "action_space must be a gymnasium.spaces.Space"
    # Check that the action space size is correct
    assert gym_env.action_space.n == raw_jaxatari_env.num_actions, "Action space size should match the raw environment"

def test_reset_method(gym_env):
    """
    Tests the reset method for correct return types and that the observation fits the space.
    """
    obs, info = gym_env.reset()

    assert gym_env.observation_space.contains(obs), "Initial observation is not in the observation space"
    assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
    assert isinstance(info, dict), "Info returned by reset should be a dictionary"

def test_step_method(gym_env):
    """
    Tests the step method for correct return types and values.
    """
    gym_env.reset()
    action = gym_env.action_space.sample()  # Use a random valid action
    step_return = gym_env.step(action)

    assert isinstance(step_return, tuple) and len(step_return) == 5, "Step must return a 5-element tuple"
    obs, reward, terminated, truncated, info = step_return

    assert gym_env.observation_space.contains(obs), "Observation from step is not in the observation space"
    assert isinstance(reward, float), "Reward must be a float"
    assert isinstance(terminated, bool), "Terminated flag must be a bool"
    assert isinstance(truncated, bool), "Truncated flag must be a bool"
    assert isinstance(info, dict), "Info from step must be a dict"

def test_seeding_and_determinism(raw_jaxatari_env):
    """
    Tests that seeding the environment produces deterministic trajectories.
    We create two separate wrapper instances from the same raw env to test this.
    """
    # Create two separate wrapper instances to ensure no state crossover
    env1 = GymnasiumJaxAtariWrapper(raw_jaxatari_env)
    env2 = GymnasiumJaxAtariWrapper(raw_jaxatari_env)
    
    # Reset both with the same seed
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2, "Observations after a seeded reset should be identical")

    # Take a few identical actions and check for identical results
    for _ in range(10):
        # Sample an action using the first env's action space (they are identical)
        action = env1.action_space.sample()
        obs1, reward1, term1, trunc1, _ = env1.step(action)
        obs2, reward2, term2, trunc2, _ = env2.step(action)
        
        np.testing.assert_array_equal(obs1, obs2, "Observations after a step should be identical for seeded envs")
        assert reward1 == reward2, "Rewards after a step should be identical for seeded envs"
        assert term1 == term2, "Termination flags should be identical for seeded envs"
        assert trunc1 == trunc2, "Truncation flags should be identical for seeded envs"
    
    env1.close()
    env2.close()

def test_render_method(gym_env):
    """
    Tests the render method. The fixture already sets render_mode='rgb_array'.
    """
    gym_env.reset()
    frame = gym_env.render()

    assert isinstance(frame, np.ndarray), "render(mode='rgb_array') must return a numpy array"
    assert len(frame.shape) == 3, "RGB array must have 3 dimensions (H, W, C)"
    assert frame.shape[2] == 3, "RGB array must have 3 color channels"
    assert frame.dtype == np.uint8, "RGB array should have dtype uint8"


if __name__ == "__main__":
    pytest.main([__file__])