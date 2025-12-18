import pytest
import jax
import jax.numpy as jnp
import numpy as np
import importlib.util
import inspect
import os
import sys
from pathlib import Path
import collections

# --- Import Core Components ---
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.wrappers import (
    AtariWrapper,
    MultiRewardWrapper,
    PixelObsWrapper,
    ObjectCentricWrapper,
    PixelAndObjectCentricWrapper,
    FlattenObservationWrapper,
    NormalizeObservationWrapper,
    LogWrapper,
    MultiRewardLogWrapper,
)
import jaxatari.spaces as spaces
from conftest import load_game_environment, WRAPPER_RECIPES

# Add flax import for serialization tests
try:
    from flax.training.checkpoints import serialization
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_object_centric_obs_size(space: spaces.Dict) -> int:
    """Helper to correctly calculate the total flattened size of an object-centric space."""
    size = 0
    for leaf in jax.tree.leaves(space):
        size += np.prod(leaf.shape)
    return size

def deep_asdict(obj: any) -> any:
    """
    Recursively converts a Pytree of namedtuples into a Pytree of standard dicts.
    This is needed because obs is a namedtuple but the space is a Dict.
    """
    if hasattr(obj, '_asdict'): # It's a namedtuple
        return collections.OrderedDict(
            (key, deep_asdict(value)) for key, value in obj._asdict().items()
        )
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_asdict(item) for item in obj)
    else:
        return obj


# ==============================================================================
# INTEGRATION TEST SUITE
# ==============================================================================

class TestBasicAPI:
    """Tests the fundamental API of the raw, unwrapped environment."""

    def test_spaces(self, raw_env):
        """Should validate that observation and action spaces are correctly defined."""
        # Test observation space
        obs_space = raw_env.observation_space()
        assert obs_space is not None, "Observation space should not be None"
        assert isinstance(obs_space, spaces.Space), "Observation space should be a Space instance"
        
        # Test action space
        action_space = raw_env.action_space()
        assert action_space is not None, "Action space should not be None"
        assert isinstance(action_space, spaces.Space), "Action space should be a Space instance"
        
        # Test image space
        image_space = raw_env.image_space()
        assert image_space is not None, "Image space should not be None"
        assert isinstance(image_space, spaces.Space), "Image space should be a Space instance"
        
        # Test that spaces can sample valid observations
        key = jax.random.PRNGKey(0)
        sample_obs = obs_space.sample(key)
        assert sample_obs is not None, "Observation space sample should not be None"
        assert obs_space.contains(sample_obs), "Sampled observation should be contained in space"
        
        # Test action space sampling
        sample_action = action_space.sample(key)
        assert sample_action is not None, "Action space sample should not be None"
        assert action_space.contains(sample_action), "Sampled action should be contained in space"

    def test_reset(self, raw_env):
        """Should test the reset method for correct return types and observation validity."""
        key = jax.random.PRNGKey(0)
        
        # Test reset
        obs, state = raw_env.reset(key)
        assert obs is not None, "Observation should not be None"
        assert state is not None, "State should not be None"
        
        # Test that observation is valid
        obs_space = raw_env.observation_space()
        assert obs_space.contains(obs), "Reset observation should be contained in observation space"
        
        def is_pytree(obj):
            try:
                # jax.tree_util.tree_leaves raises a TypeError for non-pytree objects.
                jax.tree_util.tree_leaves(obj)
                return True
            except TypeError:
                return False

        # Test that state is a valid JAX array or structure
        assert is_pytree(state), "State should be a valid JAX structure (pytree)"
        
        # Test multiple resets with different keys
        key2 = jax.random.PRNGKey(1)
        obs2, state2 = raw_env.reset(key2)
        assert obs2 is not None, "Second reset observation should not be None"
        assert state2 is not None, "Second reset state should not be None"

    def test_step(self, raw_env):
        """Should test the step method for correct return types and value ranges."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        
        # Test step with valid action
        action_space = raw_env.action_space()
        action = action_space.sample(key)
        
        obs_step, state_step, reward, done, info = raw_env.step(state, action)
        
        # Test return types
        assert obs_step is not None, "Step observation should not be None"
        assert state_step is not None, "Step state should not be None"
        assert isinstance(reward, (float, jnp.ndarray)), "Reward should be float or jnp.ndarray"
        assert isinstance(done, (bool, jnp.ndarray)), "Done should be bool or jnp.ndarray"
        assert info is not None, "Info should not be None"
        
        # Test observation validity
        obs_space = raw_env.observation_space()
        assert obs_space.contains(obs_step), "Step observation should be contained in observation space"
        
        # Test reward range (should be finite)
        reward_float = float(reward)
        assert jnp.isfinite(reward_float), "Reward should be finite"
        
        # Test multiple steps
        for _ in range(10):
            action = action_space.sample(key)
            obs_step, state_step, reward, done, info = raw_env.step(state_step, action)
            assert obs_step is not None
            assert state_step is not None
            assert jnp.isfinite(float(reward))
            if done:
                break
        
    def test_determinism(self, raw_env):
        """Should ensure the environment is deterministic given the same key and actions."""
        key = jax.random.PRNGKey(42)
        
        # Run two identical episodes
        obs1, state1 = raw_env.reset(key)
        obs2, state2 = raw_env.reset(key)
        
        # States should be identical
        assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, state1, state2)), "States should be identical with same key"
        
        # Run same sequence of actions
        actions = [0, 1, 2, 0, 1]  # Fixed action sequence
        for action in actions:
            obs1, state1, reward1, done1, info1 = raw_env.step(state1, action)
            obs2, state2, reward2, done2, info2 = raw_env.step(state2, action)
            
            # Results should be identical
            assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, state1, state2)), "States should be identical after same action"
            assert jnp.array_equal(reward1, reward2), "Rewards should be identical after same action"
            assert jnp.array_equal(done1, done2), "Done flags should be identical after same action"

    def test_render(self, raw_env):
        """Should validate that the render() method returns a valid image array."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        
        # Test render function
        rendered_image = raw_env.render(state)
        assert rendered_image is not None, "Rendered image should not be None"
        assert isinstance(rendered_image, (tuple, jnp.ndarray)), "Rendered image should be tuple or jnp.ndarray"
        
        # If it's a tuple, check the first element
        if isinstance(rendered_image, tuple):
            assert len(rendered_image) > 0, "Rendered image tuple should not be empty"
            rendered_image = rendered_image[0]
        
        # Check that the rendered image is a valid array
        assert isinstance(rendered_image, jnp.ndarray), "Rendered image should be jnp.ndarray"
        assert rendered_image.ndim >= 2, "Rendered image should have at least 2 dimensions"
        
        # Test image space compatibility
        image_space = raw_env.image_space()
        assert image_space.contains(rendered_image), "Rendered image should be contained in image space"

    def test_obs_to_flat_array(self, raw_env):
        """Test that the obs_to_flat_array function works correctly."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        flat_obs = raw_env.obs_to_flat_array(obs)
        assert flat_obs is not None, "Flat observation should not be None"
        assert isinstance(flat_obs, jnp.ndarray), "Flat observation should be jnp.ndarray"
        assert flat_obs.ndim == 1, "Flat observation should be 1-dimensional"

    def test_episode_completion(self, raw_env):
        """Test that episodes can run to completion."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        action_space = raw_env.action_space()
        
        total_reward = 0.0
        step_count = 0
        done = False
        max_steps = 50  # Prevent infinite loops
        
        while not done and step_count < max_steps:
            action = action_space.sample(key)
            obs, state, reward, done, info = raw_env.step(state, action)
            
            total_reward += float(reward)
            step_count += 1
            
            # Update key for next random action
            key, _ = jax.random.split(key)
        
        assert step_count > 0, "Should have taken at least one step"
        assert step_count <= max_steps, "Should not exceed max steps"
        assert isinstance(total_reward, float), "Total reward should be float"


    ''' TODO: rewrite this and then reintroduce the test
    def test_jaxatari_vs_ale_image_and_action_space():
        for jax_name, ale_name in ALE_GAME_MAP.items():
            # JAXAtari env
            jax_env = jaxatari.make(jax_name)
            import jax
            key = jax.random.PRNGKey(0)
            jax_obs, jax_state = jax_env.reset(key)
            jax_frame = jax_env.render(jax_state)
            assert isinstance(jax_frame, jnp.ndarray)

            # ALE env
            ale_env = gym.make(f"ALE/{ale_name}", render_mode="rgb_array")
            ale_obs, _ = ale_env.reset(seed=0)
            ale_frame = ale_env.render()
            assert isinstance(ale_frame, np.ndarray)

            # Compare image shapes
            assert jax_frame.shape == ale_frame.shape, (
                f"Image shape mismatch for {jax_name}: "
                f"JAXAtari {jax_frame.shape} vs ALE {ale_frame.shape}"
            )
            assert jax_frame.dtype == ale_frame.dtype, (
                f"Image dtype mismatch for {jax_name}: "
                f"JAXAtari {jax_frame.dtype} vs ALE {ale_frame.dtype}"
            )

            # Compare action spaces
            jax_action_space = jax_env.action_space()
            ale_action_space = ale_env.action_space

            assert hasattr(jax_action_space, "n") and hasattr(ale_action_space, "n")
            assert jax_action_space.n == ale_action_space.n, (
                f"Action space size mismatch for {jax_name}: "
                f"JAXAtari {jax_action_space.n} vs ALE {ale_action_space.n}"
            )

            # Optionally, compare action meanings if available
            if hasattr(ale_env.unwrapped, "get_action_meanings"):
                ale_meanings = ale_env.unwrapped.get_action_meanings()
                if hasattr(jax_env, "get_action_meanings"):
                    jax_meanings = jax_env.get_action_meanings()
                    assert jax_meanings == ale_meanings, (
                        f"Action meanings mismatch for {jax_name}: "
                        f"JAXAtari {jax_meanings} vs ALE {ale_meanings}"
                    )

            ale_env.close()
    '''



class TestWrapperCompatibility:
    """Tests that the environment works correctly with all standard wrappers."""

    def test_wrapped_reset_and_step(self, wrapped_env):
        """
        Should test that reset() and step() work for the wrapped environment and
        that the returned observations are contained within the wrapper's space.
        """
        key = jax.random.PRNGKey(0)
        
        # Test reset
        obs, state = wrapped_env.reset(key)
        assert obs is not None, "Wrapped environment reset observation should not be None"
        assert state is not None, "Wrapped environment reset state should not be None"
        
        # Test observation space containment
        obs_space = wrapped_env.observation_space()
        assert obs_space.contains(obs), "Wrapped observation should be contained in wrapper's space"
        
        # Test step
        action_space = wrapped_env.action_space()
        action = action_space.sample(key)
        obs_step, state_step, reward, done, info = wrapped_env.step(state, action)
        
        assert obs_step is not None, "Wrapped environment step observation should not be None"
        assert state_step is not None, "Wrapped environment step state should not be None"
        assert isinstance(reward, (float, jnp.ndarray)), "Wrapped reward should be float or jnp.ndarray"
        assert isinstance(done, (bool, jnp.ndarray)), "Wrapped done should be bool or jnp.ndarray"
        assert info is not None, "Wrapped info should not be None"
        
        # Test observation space containment after step
        assert obs_space.contains(obs_step), "Wrapped step observation should be contained in wrapper's space"

    def test_observation_shape_and_type(self, wrapped_env):
        """
        Should verify that the observation's shape and dtype match what is
        expected from the specific wrapper being tested.
        """
        key = jax.random.PRNGKey(0)
        obs, state = wrapped_env.reset(key)
        obs_space = wrapped_env.observation_space()
        
        # Test shape consistency
        if isinstance(obs, tuple) and hasattr(obs, '_asdict') == False:
            # For tuple observations (like PixelAndObjectCentricWrapper) - but NOT named tuples
            assert len(obs) == len(obs_space.spaces), "Tuple observation length should match space length"
            for obs_part, space_part in zip(obs, obs_space.spaces):
                # Handle named tuples and other complex structures
                if hasattr(obs_part, 'shape'):
                    assert obs_part.shape == space_part.shape, f"Observation part shape {obs_part.shape} should match space shape {space_part.shape}"
                elif hasattr(obs_part, '_asdict'):
                    # For named tuples, check that the structure matches the space
                    obs_dict = obs_part._asdict()
                    # The space_part might be a string key or a space object
                    if isinstance(space_part, str):
                        # This is a key in a Dict space, skip shape checking
                        pass
                    elif isinstance(space_part, spaces.Dict):
                        assert set(obs_dict.keys()) == set(space_part.spaces.keys()), f"Named tuple keys should match space keys"
                    else:
                        # For other space types, just verify the observation is valid
                        assert obs_part is not None, f"Observation part should not be None"
                else:
                    # For other structures, just verify they're not None
                    assert obs_part is not None, f"Observation part should not be None"
        else:
            # For single observations (including named tuples)
            if hasattr(obs, 'shape'):
                assert obs.shape == obs_space.shape, f"Observation shape {obs.shape} should match space shape {obs_space.shape}"
                assert obs.dtype == obs_space.dtype, f"Observation dtype {obs.dtype} should match space dtype {obs_space.dtype}"
            elif hasattr(obs, '_asdict'):
                # For named tuples, check that the structure matches the space
                obs_dict = obs._asdict()
                assert isinstance(obs_space, spaces.Dict), f"Space should be Dict for named tuple observation"
                assert set(obs_dict.keys()) == set(obs_space.spaces.keys()), f"Named tuple keys should match space keys"
            else:
                # For other structures, just verify they're not None
                assert obs is not None, f"Observation should not be None"
        
        # Test step consistency
        action_space = wrapped_env.action_space()
        action = action_space.sample(key)
        obs_step, state_step, reward, done, info = wrapped_env.step(state, action)
        
        if isinstance(obs_step, tuple) and hasattr(obs_step, '_asdict') == False:
            assert len(obs_step) == len(obs_space.spaces), "Step tuple observation length should match space length"
            for obs_part, space_part in zip(obs_step, obs_space.spaces):
                # Handle named tuples and other complex structures
                if hasattr(obs_part, 'shape'):
                    assert obs_part.shape == space_part.shape, f"Step observation part shape {obs_part.shape} should match space shape {space_part.shape}"
                elif hasattr(obs_part, '_asdict'):
                    # For named tuples, check that the structure matches the space
                    obs_dict = obs_part._asdict()
                    # The space_part might be a string key or a space object
                    if isinstance(space_part, str):
                        # This is a key in a Dict space, skip shape checking
                        pass
                    elif isinstance(space_part, spaces.Dict):
                        assert set(obs_dict.keys()) == set(space_part.spaces.keys()), f"Named tuple step keys should match space keys"
                    else:
                        # For other space types, just verify the observation is valid
                        assert obs_part is not None, f"Step observation part should not be None"
                else:
                    # For other structures, just verify they're not None
                    assert obs_part is not None, f"Step observation part should not be None"
        else:
            if hasattr(obs_step, 'shape'):
                assert obs_step.shape == obs_space.shape, f"Step observation shape {obs_step.shape} should match space shape {obs_space.shape}"
            elif hasattr(obs_step, '_asdict'):
                # For named tuples, check that the structure matches the space
                obs_dict = obs_step._asdict()
                assert isinstance(obs_space, spaces.Dict), f"Space should be Dict for named tuple step observation"
                assert set(obs_dict.keys()) == set(obs_space.spaces.keys()), f"Named tuple step keys should match space keys"
            else:
                # For other structures, just verify they're not None
                assert obs_step is not None, f"Step observation should not be None"

    def test_wrapper_observation_spaces(self, wrapped_env):
        """Test that wrapper observation spaces are correctly defined."""
        obs_space = wrapped_env.observation_space()
        assert obs_space is not None, "Wrapper observation space should not be None"
        assert isinstance(obs_space, spaces.Space), "Wrapper observation space should be a Space instance"
        
        # Test space sampling
        key = jax.random.PRNGKey(0)
        sample_obs = obs_space.sample(key)
        assert sample_obs is not None, "Wrapper space sample should not be None"
        assert obs_space.contains(sample_obs), "Wrapper space sample should be contained in space"

    def test_wrapper_action_spaces(self, wrapped_env):
        """Test that wrapper action spaces are correctly defined."""
        action_space = wrapped_env.action_space()
        assert action_space is not None, "Wrapper action space should not be None"
        assert isinstance(action_space, spaces.Space), "Wrapper action space should be a Space instance"
        
        # Test space sampling
        key = jax.random.PRNGKey(0)
        sample_action = action_space.sample(key)
        assert sample_action is not None, "Wrapper action space sample should not be None"
        assert action_space.contains(sample_action), "Wrapper action space sample should be contained in space"

    def test_wrapper_determinism(self, wrapped_env):
        """Test that wrapped environments are deterministic."""
        key = jax.random.PRNGKey(42)
        
        # Run two identical episodes
        obs1, state1 = wrapped_env.reset(key)
        obs2, state2 = wrapped_env.reset(key)
        
        # States should be identical
        assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, state1, state2)), "Wrapped states should be identical with same key"
        
        # Run same sequence of actions
        actions = [0, 1, 2, 0, 1]  # Fixed action sequence
        for action in actions:
            obs1, state1, reward1, done1, info1 = wrapped_env.step(state1, action)
            obs2, state2, reward2, done2, info2 = wrapped_env.step(state2, action)
            
            # Results should be identical
            assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, state1, state2)), "Wrapped states should be identical after same action"
            assert jnp.array_equal(reward1, reward2), "Wrapped rewards should be identical after same action"
            assert jnp.array_equal(done1, done2), "Wrapped done flags should be identical after same action"


class TestJaxTransforms:
    """A stress test for JIT and VMAP compatibility on wrapped environments."""

    def _check_batch_dimension_recursive(self, obj, expected_batch_size, context_name):
        """
        Recursively check that all jnp.ndarray objects in a nested structure have the correct batch dimension.
        
        Args:
            obj: The object to check (can be jnp.ndarray, NamedTuple, tuple, list, dict, etc.)
            expected_batch_size: The expected batch size (first dimension)
            context_name: Context for error messages
        """
        if hasattr(obj, 'shape'):
            # Direct jnp.ndarray
            assert obj.shape[0] == expected_batch_size, f"{context_name} should have batch dimension {expected_batch_size}, got {obj.shape[0]}"
        elif hasattr(obj, '_asdict'):
            # NamedTuple - recursively check all fields
            for field_name, field_value in obj._asdict().items():
                self._check_batch_dimension_recursive(field_value, expected_batch_size, f"{context_name}.{field_name}")
        elif isinstance(obj, (tuple, list)):
            # Tuple or list - recursively check all elements
            for i, item in enumerate(obj):
                self._check_batch_dimension_recursive(item, expected_batch_size, f"{context_name}[{i}]")
        elif isinstance(obj, dict):
            # Dict - recursively check all values
            for key, value in obj.items():
                self._check_batch_dimension_recursive(value, expected_batch_size, f"{context_name}.{key}")
        # For other types (primitives, etc.), we don't need to check anything

    def test_jit_compilation(self, wrapped_env):
        """Should test that a full step can be JIT-compiled without error."""
        key = jax.random.PRNGKey(0)
        
        # JIT the reset and step functions
        jitted_reset = jax.jit(wrapped_env.reset)
        jitted_step = jax.jit(wrapped_env.step)
        
        # Test JIT reset
        obs, state = jitted_reset(key)
        assert obs is not None, "JIT reset observation should not be None"
        assert state is not None, "JIT reset state should not be None"
        
        # Test JIT step
        action_space = wrapped_env.action_space()
        action = action_space.sample(key)
        obs, state, reward, done, info = jitted_step(state, action)
        assert obs is not None, "JIT step observation should not be None"
        assert state is not None, "JIT step state should not be None"
        assert isinstance(reward, (float, jnp.ndarray)), "JIT step reward should be float or jnp.ndarray"
        assert isinstance(done, (bool, jnp.ndarray)), "JIT step done should be bool or jnp.ndarray"
        assert info is not None, "JIT step info should not be None"

    @pytest.mark.skip(reason="Skipping to debug memory issues in CI")
    def test_vmap_parallelization(self, wrapped_env):
        """Should test that the environment can be vectorized across a batch using vmap."""
        num_envs = 2
        key = jax.random.PRNGKey(42)
        
        # Vmap reset and step functions
        vmapped_reset = jax.vmap(wrapped_env.reset)
        vmapped_step = jax.vmap(wrapped_env.step)
        
        # Test vmap reset
        keys = jax.random.split(key, num_envs)
        obs, states = vmapped_reset(keys)
        
        # Check batch dimensions
        self._check_batch_dimension_recursive(obs, num_envs, "Observation")
        
        # Check state batch dimension
        state_leaves = jax.tree.leaves(states)
        for leaf in state_leaves:
            assert leaf.shape[0] == num_envs, f"State leaf should have batch dimension {num_envs}"
        
        # Test vmap step
        action_keys = jax.random.split(keys[0], num_envs)
        actions = jax.vmap(wrapped_env.action_space().sample)(action_keys)
        assert actions.shape == (num_envs,), f"Actions should have shape ({num_envs},)"
        
        new_obs, new_states, rewards, dones, infos = vmapped_step(states, actions)
        
        # Check batch dimensions for step results
        self._check_batch_dimension_recursive(new_obs, num_envs, "Step observation")
        
        new_state_leaves = jax.tree.leaves(new_states)
        for leaf in new_state_leaves:
            assert leaf.shape[0] == num_envs, f"Step state leaf should have batch dimension {num_envs}"
        
        assert rewards.shape == (num_envs,), f"Rewards should have shape ({num_envs},)"
        assert dones.shape == (num_envs,), f"Dones should have shape ({num_envs},)"

    @pytest.mark.skip(reason="Skipping to debug memory issues in CI")
    def test_jit_vmap_combination(self, wrapped_env):
        """Test that JIT and vmap can be combined."""
        num_envs = 2
        key = jax.random.PRNGKey(42)
        
        # Combine JIT and vmap
        jit_vmapped_reset = jax.jit(jax.vmap(wrapped_env.reset))
        jit_vmapped_step = jax.jit(jax.vmap(wrapped_env.step))
        
        # Test combined reset
        keys = jax.random.split(key, num_envs)
        obs, states = jit_vmapped_reset(keys)
        
        # Check batch dimensions
        self._check_batch_dimension_recursive(obs, num_envs, "JIT+vmap observation")
        
        # Test combined step
        action_keys = jax.random.split(keys[0], num_envs)
        actions = jax.vmap(wrapped_env.action_space().sample)(action_keys)
        
        new_obs, new_states, rewards, dones, infos = jit_vmapped_step(states, actions)
        
        # Check batch dimensions
        self._check_batch_dimension_recursive(new_obs, num_envs, "JIT+vmap step observation")
        
        assert rewards.shape == (num_envs,), f"JIT+vmap rewards should have shape ({num_envs},)"
        assert dones.shape == (num_envs,), f"JIT+vmap dones should have shape ({num_envs},)"


class TestAdvancedWrapperFeatures:
    """Tests advanced wrapper features and edge cases."""

    def test_flatten_observation_wrapper_space_structure(self, raw_env):
        """Test that FlattenObservationWrapper correctly flattens observation spaces."""
        key = jax.random.PRNGKey(0)
        
        # Create AtariWrapper first
        atari_env = AtariWrapper(raw_env, frame_stack_size=4)
        flatten_env = FlattenObservationWrapper(atari_env)
        
        # Test space transformation
        original_space = atari_env.observation_space()
        flattened_space = flatten_env.observation_space()
        
        # Get leaves from both spaces
        original_leaves = jax.tree.leaves(original_space)
        flattened_leaves = jax.tree.leaves(flattened_space)
        
        # Check each leaf was flattened correctly
        for original_leaf, flattened_leaf in zip(original_leaves, flattened_leaves):
            assert len(flattened_leaf.shape) == 1, "Flattened leaf should be 1D"
            expected_size = np.prod(original_leaf.shape)
            assert flattened_leaf.shape[0] == expected_size, f"Flattened size should be {expected_size}"
        
        # Test runtime observation transformation
        obs_flat, state = flatten_env.reset(key)
        assert obs_flat is not None, "Flattened observation should not be None"
        
        # Convert observation to dict for comparison
        obs_flat_dict = deep_asdict(obs_flat)
        obs_leaves = jax.tree.leaves(obs_flat_dict)
        space_leaves = jax.tree.leaves(flattened_space)
        
        assert len(obs_leaves) == len(space_leaves), "Number of observation and space leaves should match"
        
        for obs_leaf, space_leaf in zip(obs_leaves, space_leaves):
            assert isinstance(space_leaf, spaces.Box), "Space leaf should be Box"
            assert space_leaf.contains(obs_leaf), "Observation leaf should be contained in space"

    def test_normalize_observation_wrapper(self, raw_env):
        """Test that NormalizeObservationWrapper correctly normalizes observations."""
        key = jax.random.PRNGKey(0)
        
        # Create base environment stack
        base_env = AtariWrapper(raw_env, frame_stack_size=4)
        
        # Test different configurations
        configs = [
            (PixelObsWrapper, "PixelObs"),
            (ObjectCentricWrapper, "ObjectCentric"),
            (PixelAndObjectCentricWrapper, "PixelAndObjectCentric"),
        ]
        
        for wrapper_fn, desc in configs:
            for to_neg_one in [False, True]:
                # Setup environment
                unwrapped_env = wrapper_fn(base_env)
                env = NormalizeObservationWrapper(unwrapped_env, to_neg_one=to_neg_one)
                
                # Test space
                space = env.observation_space()
                expected_low = -1.0 if to_neg_one else 0.0
                
                def check_space_leaf(s):
                    assert isinstance(s, spaces.Box), f"[{desc}] Leaf space should be Box"
                    assert s.dtype == jnp.float16, f"[{desc}] Dtype should be float16"
                    assert jnp.all(s.low == expected_low), f"[{desc}] Space low bound is incorrect"
                    assert jnp.all(s.high == 1.0), f"[{desc}] Space high bound is incorrect"
                
                jax.tree.map(check_space_leaf, space, is_leaf=lambda x: isinstance(x, spaces.Box))
                
                # Test observation
                obs, state = env.reset(key)
                obs_step, _, _, _, _ = env.step(state, 2)  # Use non-NOOP action
                
                def check_obs_leaf(o):
                    assert jnp.all(o >= expected_low - 1e-6), f"[{desc}] Obs values are below lower bound"
                    assert jnp.all(o <= 1.0 + 1e-6), f"[{desc}] Obs values are above upper bound"
                    assert jnp.std(o) > 1e-6, f"[{desc}] Std of observation is zero"
                
                jax.tree.map(check_obs_leaf, obs)
                jax.tree.map(check_obs_leaf, obs_step)
                
                # Test containment
                assert space.contains(obs), f"[{desc}] Reset observation is not contained in space"
                assert space.contains(obs_step), f"[{desc}] Step observation is not contained in space"

    def test_log_wrapper_tracking(self, raw_env):
        """Test that LogWrapper correctly tracks episode statistics."""
        key = jax.random.PRNGKey(0)
        
        # Create environment with LogWrapper
        base_env = AtariWrapper(raw_env)
        env = LogWrapper(PixelObsWrapper(base_env))
        
        # Reset and check initial state
        obs, state = env.reset(key)
        assert state.episode_returns == 0.0, "Initial episode returns should be 0"
        assert state.episode_lengths == 0, "Initial episode lengths should be 0"
        
        # Take steps and accumulate rewards
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 100:
            obs, state, reward, done, info = env.step(state, 0)
            total_reward += reward
            steps += 1
            
            # Check running totals
            assert state.episode_returns == total_reward * (1 - done), "Episode returns should match accumulated reward"
            assert state.episode_lengths == steps * (1 - done), "Episode lengths should match step count"
            
            if done:
                # Check final episode statistics
                assert state.returned_episode_returns == total_reward, "Returned episode returns should match total reward"
                assert state.returned_episode_lengths == steps, "Returned episode lengths should match step count"
                assert info["returned_episode_returns"] == total_reward, "Info should contain correct episode returns"
                assert info["returned_episode_lengths"] == steps, "Info should contain correct episode lengths"
                assert info["returned_episode"] == True, "Info should indicate episode returned"

    def test_multi_reward_log_wrapper(self, raw_env):
        """Test that MultiRewardLogWrapper correctly tracks multiple reward types."""
        key = jax.random.PRNGKey(0)
        
        # Create environment with MultiRewardLogWrapper
        reward_funcs = [lambda state, prev_state: jnp.ones(1)]
        base_env = AtariWrapper(MultiRewardWrapper(raw_env, reward_funcs))
        env = MultiRewardLogWrapper(PixelAndObjectCentricWrapper(base_env))
        # Reset and check initial state
        obs, state = env.reset(key)
        assert state.episode_returns_env == 0.0, "Initial episode returns env should be 0"
        assert jnp.all(state.episode_returns == 0.0), "Initial episode returns should be all 0"
        assert state.episode_lengths == 0, "Initial episode lengths should be 0"
        
        # Take steps and accumulate rewards
        total_reward_env = 0.0
        total_rewards = jnp.zeros_like(state.episode_returns)
        steps = 0
        done = False
        
        while not done and steps < 100:
            obs, state, reward, done, info = env.step(state, 0)
            total_reward_env += reward
            total_rewards += info["all_rewards"]
            steps += 1
            
            # Check running totals
            assert state.episode_returns_env == total_reward_env * (1 - done), "Episode returns env should match accumulated reward"
            assert jnp.all(state.episode_returns == total_rewards * (1 - done)), "Episode returns should match accumulated rewards"
            assert state.episode_lengths == steps * (1 - done), "Episode lengths should match step count"
            
            if done:
                # Check final episode statistics
                assert state.returned_episode_returns_env == total_reward_env, "Returned episode returns env should match total reward"
                assert jnp.all(state.returned_episode_returns == total_rewards), "Returned episode returns should match total rewards"
                assert state.returned_episode_lengths == steps, "Returned episode lengths should match step count"
                assert info["returned_episode_env_returns"] == total_reward_env, "Info should contain correct episode returns env"
                for i, r in enumerate(total_rewards):
                    assert info[f"returned_episode_returns_{i}"] == r, f"Info should contain correct episode returns {i}"
                assert info["returned_episode_lengths"] == steps, "Info should contain correct episode lengths"
                assert info["returned_episode"] == True, "Info should indicate episode returned"

    def test_atari_wrapper_features(self, raw_env):
        """Tests specific features of the AtariWrapper like noop_reset."""
        key = jax.random.PRNGKey(0)
        
        # Test noop_reset feature
        env = AtariWrapper(raw_env, noop_reset=30)
        _, state = env.reset(key)

        # If noop_reset is active, the initial step count should be > 0
        assert state.step > 0, "noop_reset should result in a non-zero initial step count."
        
        # Test first_fire feature
        env_no_fire = AtariWrapper(raw_env, first_fire=False)
        env_with_fire = AtariWrapper(raw_env, first_fire=True)
        
        _, state_no_fire = env_no_fire.reset(key)
        _, state_with_fire = env_with_fire.reset(key)
        
        # The state with first_fire should have a different prev_action
        assert state_with_fire.prev_action == 1, "first_fire should set prev_action to FIRE (1)"
        assert state_no_fire.prev_action == 0, "first_fire=False should set prev_action to NOOP (0)"
        
        # Test sticky_actions feature
        env_sticky = AtariWrapper(raw_env, sticky_actions=True)
        _, state_sticky = env_sticky.reset(key)
        
        # Take a step and check that sticky actions can repeat the previous action
        obs, state_sticky, reward, done, info = env_sticky.step(state_sticky, 2)  # UP action
        assert state_sticky.prev_action == 2, "Sticky actions should update prev_action"
        
        # Test episodic_life feature
        # We'll just verify the wrapper accepts the parameter
        env_episodic = AtariWrapper(raw_env, episodic_life=True)
        _, state_episodic = env_episodic.reset(key)
        assert state_episodic is not None, "episodic_life=True should work"

    def test_log_wrapper_edge_cases(self, raw_env):
        """Test LogWrapper with edge cases like very short episodes."""
        key = jax.random.PRNGKey(0)
        
        # Create a mock environment that terminates immediately
        class MockImmediateDoneEnv:
            def __init__(self, base_env):
                self.base_env = base_env
                self._observation_space = base_env.observation_space()
                self._action_space = base_env.action_space()
            
            def observation_space(self):
                return self._observation_space
            
            def action_space(self):
                return self._action_space
            
            def reset(self, key):
                obs, state = self.base_env.reset(key)
                # Create a mock state that will cause immediate termination
                return obs, state
            
            def step(self, state, action):
                # Return done=True immediately
                obs, state, reward, done, info = self.base_env.step(state, action)
                return obs, state, reward, True, info  # Force done=True
        
        # Test with immediate termination
        mock_env = MockImmediateDoneEnv(raw_env)
        env = LogWrapper(AtariWrapper(mock_env))
        
        obs, state = env.reset(key)
        obs, state, reward, done, info = env.step(state, 0)
        
        # Check that logging still works correctly even with immediate termination
        assert done == True, "Mock environment should terminate immediately"
        assert state.returned_episode_returns == reward, "Episode returns should match the single step reward"
        assert state.returned_episode_lengths == 1, "Episode length should be 1 for immediate termination"
        assert info["returned_episode_returns"] == reward, "Info should contain correct episode returns"
        assert info["returned_episode_lengths"] == 1, "Info should contain correct episode lengths"
        assert info["returned_episode"] == True, "Info should indicate episode returned"


class TestEdgeCasesAndErrorHandling:
    """Tests edge cases and error handling scenarios."""

    def test_invalid_action_handling(self, raw_env):
        """Test that invalid actions are handled gracefully."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        action_space = raw_env.action_space()
        
        # Test with valid actions
        valid_action = action_space.sample(key)
        obs, state, reward, done, info = raw_env.step(state, valid_action)
        assert obs is not None, "Valid action should produce valid observation"
        
        # Test with out-of-bounds action (should be handled by action space)
        if hasattr(action_space, 'n'):
            invalid_action = action_space.n + 1
            # This should either be handled gracefully or raise an appropriate error
            try:
                obs, state, reward, done, info = raw_env.step(state, invalid_action)
                # If no error, check that the result is still valid
                assert obs is not None, "Invalid action should still produce valid observation"
            except (ValueError, IndexError, AssertionError):
                # Expected behavior for some environments
                pass

    def test_extreme_reward_values(self, raw_env):
        """Test that extreme reward values are handled correctly."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        action_space = raw_env.action_space()
        
        # Run for many steps to potentially encounter extreme rewards
        for _ in range(100):
            action = action_space.sample(key)
            obs, state, reward, done, info = raw_env.step(state, action)
            
            # Check that reward is finite
            reward_float = float(reward)
            assert jnp.isfinite(reward_float), "Reward should be finite"
            
            if done:
                break
            
            key, _ = jax.random.split(key)

    def test_state_consistency(self, raw_env):
        """Test that state remains consistent across operations."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        action_space = raw_env.action_space()
        
        # Test that state is a valid JAX structure
        assert jax.tree_util.tree_structure(state) is not None, "State should be a valid JAX tree structure"
        
        # Test state consistency across steps
        for _ in range(10):
            action = action_space.sample(key)
            obs, state, reward, done, info = raw_env.step(state, action)
            
            # State should remain a valid JAX structure
            assert jax.tree_util.tree_structure(state) is not None, "State should remain valid JAX tree structure"
            
            if done:
                break
            
            key, _ = jax.random.split(key)

    def test_observation_consistency(self, raw_env):
        """Test that observations remain consistent in shape and type."""
        key = jax.random.PRNGKey(0)
        obs, state = raw_env.reset(key)
        action_space = raw_env.action_space()
        
        initial_obs_shape = obs.shape if hasattr(obs, 'shape') else None
        initial_obs_type = type(obs)
        
        # Test observation consistency across steps
        for step in range(100):
            action = action_space.sample(key)
            obs, state, reward, done, info = raw_env.step(state, action)
            
            # Check observation type consistency
            assert isinstance(obs, initial_obs_type), f"Observation type should remain consistent at step {step}"
            
            # Check observation shape consistency (if applicable)
            if initial_obs_shape is not None:
                assert obs.shape == initial_obs_shape, f"Observation shape should remain consistent at step {step}"
            
            if done:
                break
            
            key, _ = jax.random.split(key)

    @pytest.mark.skipif(not FLAX_AVAILABLE, reason="flax.training.checkpoints not available")
    def test_state_serialization(self, wrapped_env):
        """Tests if the environment state can be serialized and restored."""
        key = jax.random.PRNGKey(0)
        _, state1 = wrapped_env.reset(key)

        # Test that we can extract and serialize the basic components
        # This is a simplified test that focuses on the core arrays rather than complex state objects
        
        # Extract arrays from the state
        def extract_arrays(state):
            """Extract all JAX arrays from a state object."""
            arrays = {}
            if hasattr(state, '_asdict'):
                # For named tuples
                for k, v in state._asdict().items():
                    if isinstance(v, jnp.ndarray):
                        arrays[k] = v
                    elif hasattr(v, '_asdict'):
                        # Recursively extract from nested objects
                        nested_arrays = extract_arrays(v)
                        for nk, nv in nested_arrays.items():
                            arrays[f"{k}_{nk}"] = nv
            elif hasattr(state, '__dict__'):
                # For dataclasses
                for k, v in state.__dict__.items():
                    if isinstance(v, jnp.ndarray):
                        arrays[k] = v
                    elif hasattr(v, '_asdict'):
                        # Recursively extract from nested objects
                        nested_arrays = extract_arrays(v)
                        for nk, nv in nested_arrays.items():
                            arrays[f"{k}_{nk}"] = nv
            return arrays

        # Extract arrays from both states
        arrays1 = extract_arrays(state1)
        _, state2 = wrapped_env.reset(key)
        arrays2 = extract_arrays(state2)

        # Test that we can serialize the arrays
        serialized_arrays = serialization.msgpack_serialize(arrays1)
        restored_arrays = serialization.msgpack_restore(serialized_arrays)

        # Verify that the arrays are preserved correctly
        assert len(arrays1) == len(restored_arrays), "Number of arrays should match"
        for key in arrays1:
            assert key in restored_arrays, f"Array key {key} should be in restored state"
            assert jnp.array_equal(arrays1[key], restored_arrays[key]), f"Array {key} should match"

if __name__ == "__main__":
    pytest.main([__file__])