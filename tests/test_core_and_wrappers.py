import collections
import jax
import jax.numpy as jnp
import pytest
import jaxatari
from jaxatari.environment import EnvInfo, EnvObs, EnvState
from jaxatari.wrappers import (
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    PixelAndObjectCentricWrapper,
    LogWrapper,
    MultiRewardLogWrapper, 
    FlattenObservationWrapper
)
import jaxatari.spaces as spaces
import numpy as np


def get_object_centric_obs_size(space: spaces.Dict) -> int:
    """Helper to correctly calculate the total flattened size of an object-centric space."""
    size = 0
    for leaf in jax.tree.leaves(space):
        size += np.prod(leaf.shape)
    return size

def test_base_jaxatari():
    """Test the base JAXAtari class with a simple game."""
    env = jaxatari.make("pong")
    key = jax.random.PRNGKey(0)
    
    # Test reset
    obs, state = env.reset(key)
    assert obs is not None
    assert state is not None
    
    # Test step
    action = 0  # NOOP
    obs, state, reward, done, info = env.step(state, action)
    assert obs is not None
    assert state is not None
    assert isinstance(reward, (float, jnp.ndarray))
    assert isinstance(done, (bool, jnp.ndarray))
    assert info is not None


def test_atari_wrapper():
    """Test the AtariWrapper."""
    base_env = jaxatari.make("pong")
    env = AtariWrapper(base_env)
    key = jax.random.PRNGKey(0)
    
    # Test reset
    obs, state = env.reset(key)
    assert obs is not None
    assert state.env_state is not None
    assert state.step == 0
    assert state.prev_action == 0
    assert state.obs_stack is not None
    assert state.key is not None  # Check that key is stored in state
    
    # Test step
    action = 0  # NOOP
    obs, state, reward, done, info = env.step(state, action)
    assert obs is not None
    assert state.env_state is not None
    assert state.step > 0
    assert state.prev_action is not None
    assert state.obs_stack is not None
    assert state.key is not None  # Check that key is still in state
    assert isinstance(reward, (float, jnp.ndarray))
    assert isinstance(done, (bool, jnp.ndarray))
    assert info is not None

def test_obs_to_flat_array_with_stacked_observations():
    """Test that obs_to_flat_array works correctly with stacked observations."""
    key = jax.random.PRNGKey(0)
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=4)
    env = ObjectCentricWrapper(atari_env)

    obs, state = env.reset(key)

    # The observation should maintain the stacked structure
    single_frame_size = get_object_centric_obs_size(atari_env._env.observation_space())
    assert obs.shape == (4, single_frame_size), f"Expected shape (4, {single_frame_size}), got {obs.shape}"
    
    # Run for a few steps to ensure frames are different
    # Use UP action (2) instead of NOOP (0) to ensure game state changes
    for _ in range(100):
        obs, state, _, _, _ = env.step(state, 2)  # Use UP action
    
    # Check that the start of the obs stack is different from the end
    first_frame = obs[0]
    last_frame = obs[-1]
    assert not jnp.array_equal(first_frame, last_frame), "Frames should be different after active gameplay"

def test_pixel_obs_wrapper_with_stacked_frames():
    """Test that PixelObsWrapper correctly handles stacked frames."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = jaxatari.make("pong")
    env = PixelObsWrapper(AtariWrapper(base_env))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify shape is (frame_stack_size, height, width, channels)
    # Pong dimensions are 160x210 with 3 color channels
    expected_shape = (env.frame_stack_size, 160, 210, 3)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    # Take a step and verify shape remains consistent
    obs, state, reward, done, info = env.step(state, 0)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"

    # perform 100 steps to get to a state in which the frames should be different
    for _ in range(100):
        obs, state, reward, done, info = env.step(state, 0)
    
    # Verify that frames are different (not just copies)
    # The first and last frame in the stack should be different
    # This is a basic check - in practice, they might be similar if the game state hasn't changed much
    assert not jnp.array_equal(obs[0], obs[-1]), "First and last frames should be different"
    
    # Verify that frames are in the correct range (0-255 for uint8)
    assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"

def test_pixel_and_object_centric_wrapper():
    """Test that PixelAndObjectCentricWrapper returns both pixel and flattened object-centric observations."""
    key = jax.random.PRNGKey(0)
    stack_size = 4
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=stack_size)
    env = PixelAndObjectCentricWrapper(atari_env)

    # 1. Test the space definition (This part is correct and passes)
    space = env.observation_space()
    assert isinstance(space, spaces.Tuple)
    assert len(space.spaces) == 2

    pixel_space, object_space = space.spaces
    assert isinstance(pixel_space, spaces.Box)
    assert pixel_space.shape == (stack_size, 160, 210, 3)

    assert isinstance(object_space, spaces.Box)
    single_frame_size = get_object_centric_obs_size(base_env.observation_space())
    assert object_space.shape == (stack_size, single_frame_size)

    # 2. Test the runtime output from reset() (This part is also correct)
    obs, state = env.reset(key)
    assert isinstance(obs, tuple)
    assert len(obs) == 2

    pixel_obs, object_obs = obs
    assert pixel_obs.shape == pixel_space.shape
    assert object_obs.shape == object_space.shape

    # 3. Test containment
    assert pixel_space.contains(pixel_obs), "Pixel observation is not contained in the pixel space"
    assert object_space.contains(object_obs), "Object observation is not contained in the object space"

    # 4. Test combined containment
    assert space.contains(obs), "Runtime observation is not contained in the defined space"


def test_object_centric_wrapper():
    """Test ObjectCentricWrapper returns a 2D stacked observation and its space is correct."""
    key = jax.random.PRNGKey(0)
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=4)
    env = ObjectCentricWrapper(atari_env)

    # 1. Test the space definition
    space = env.observation_space()
    assert isinstance(space, spaces.Box)
    single_frame_size = get_object_centric_obs_size(base_env.observation_space())
    # The space shape should be 2D: (stack_size, features)
    assert space.shape == (env.frame_stack_size, single_frame_size)

    # 2. Test the runtime output from reset()
    obs, state = env.reset(key)
    # The observation shape should match the space shape
    assert obs.shape == space.shape

    # 3. Test the runtime output from step()
    obs, state, _, _, _ = env.step(state, 2) # Use an action that causes change
    assert obs.shape == space.shape

    # 4. Verify that frames are different after several steps
    for _ in range(100):
        obs, state, _, _, _ = env.step(state, 2)
    
    first_frame = obs[0]
    last_frame = obs[-1]
    assert not jnp.array_equal(first_frame, last_frame), "Frames should be different after active gameplay"


def test_log_wrapper():
    """Test that LogWrapper correctly tracks episode returns and lengths."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = jaxatari.make("pong")
    env = LogWrapper(PixelObsWrapper(AtariWrapper(base_env)))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify initial state
    assert state.episode_returns == 0.0
    assert state.episode_lengths == 0
    assert state.returned_episode_returns == 0.0
    assert state.returned_episode_lengths == 0
    
    # Verify observation format (should match PixelObsWrapper)
    expected_shape = (env.frame_stack_size, 160, 210, 3)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"
    
    # Take some steps and accumulate rewards
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < 100:  # Limit steps to avoid infinite loops
        obs, state, reward, done, info = env.step(state, 0)  # Use NOOP action
        total_reward += reward
        steps += 1
        
        # Verify observation format remains consistent
        assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
        assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"
        
        # Verify running totals
        assert state.episode_returns == total_reward * (1 - done)
        assert state.episode_lengths == steps * (1 - done)
        
        if done:
            # Verify final episode statistics
            assert state.returned_episode_returns == total_reward
            assert state.returned_episode_lengths == steps
            assert info["returned_episode_returns"] == total_reward
            assert info["returned_episode_lengths"] == steps
            assert info["returned_episode"] == True

def test_multi_reward_log_wrapper():
    """Test that MultiRewardLogWrapper correctly tracks multiple reward types."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = jaxatari.make("pong")
    env = MultiRewardLogWrapper(PixelAndObjectCentricWrapper(AtariWrapper(base_env)))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify initial state
    assert state.episode_returns_env == 0.0
    assert jnp.all(state.episode_returns == 0.0)
    assert state.episode_lengths == 0
    assert state.returned_episode_returns_env == 0.0
    assert jnp.all(state.returned_episode_returns == 0.0)
    assert state.returned_episode_lengths == 0
    
    # Verify observation format (should match PixelAndObjectCentricWrapper)
    image_obs, object_obs = obs
    expected_image_shape = (env.frame_stack_size, 160, 210, 3)
    
    # The expected object shape is now a 2D tensor, get its size from the base env
    single_frame_object_size = get_object_centric_obs_size(base_env.observation_space())
    expected_object_shape = (env.frame_stack_size, single_frame_object_size)
    
    assert image_obs.shape == expected_image_shape, f"Expected image shape {expected_image_shape}, got {image_obs.shape}"
    assert object_obs.shape == expected_object_shape, f"Expected object shape {expected_object_shape}, got {object_obs.shape}"
    assert jnp.all(image_obs >= 0) and jnp.all(image_obs <= 255), "Pixel values should be in range [0, 255]"
    
    # Take some steps and accumulate rewards
    total_reward_env = 0.0
    total_rewards = jnp.zeros_like(state.episode_returns)
    steps = 0
    done = False
    
    while not done and steps < 100:  # Limit steps to avoid infinite loops
        obs, state, reward, done, info = env.step(state, 0)  # Use NOOP action
        total_reward_env += reward
        total_rewards += info["all_rewards"]
        steps += 1
        
        # Verify observation format remains consistent
        image_obs, object_obs = obs
        assert image_obs.shape == expected_image_shape, f"Expected image shape {expected_image_shape}, got {image_obs.shape}"
        assert object_obs.shape == expected_object_shape, f"Expected object shape {expected_object_shape}, got {object_obs.shape}"
        assert jnp.all(image_obs >= 0) and jnp.all(image_obs <= 255), "Pixel values should be in range [0, 255]"
        
        # Verify running totals
        assert state.episode_returns_env == total_reward_env * (1 - done)
        assert jnp.all(state.episode_returns == total_rewards * (1 - done))
        assert state.episode_lengths == steps * (1 - done)
        
        if done:
            # Verify final episode statistics
            assert state.returned_episode_returns_env == total_reward_env
            assert jnp.all(state.returned_episode_returns == total_rewards)
            assert state.returned_episode_lengths == steps
            assert info["returned_episode_env_returns"] == total_reward_env
            for i, r in enumerate(total_rewards):
                assert info[f"returned_episode_returns_{i}"] == r
            assert info["returned_episode_lengths"] == steps
            assert info["returned_episode"] == True


def test_flatten_observation_wrapper():
    """Test that FlattenObservationWrapper correctly flattens each observation type."""
    key = jax.random.PRNGKey(0)
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=4)

    # --- Test 1: Wrapping ObjectCentricWrapper ---
    unwrapped_oc = ObjectCentricWrapper(atari_env)
    env_oc = FlattenObservationWrapper(unwrapped_oc)
    
    unwrapped_obs_oc, _ = unwrapped_oc.reset(key)
    obs_oc, _ = env_oc.reset(key)
    
    assert obs_oc.ndim == 1, "OC obs should be a 1D array"
    assert obs_oc.shape[0] == 4 * get_object_centric_obs_size(base_env.observation_space())
    # Check order: first part of flattened obs should match first frame of unwrapped obs
    assert jnp.array_equal(obs_oc[:int(get_object_centric_obs_size(base_env.observation_space()))], unwrapped_obs_oc[0])

    # --- Test 2: Wrapping PixelObsWrapper ---
    unwrapped_pix = PixelObsWrapper(atari_env)
    env_pix = FlattenObservationWrapper(unwrapped_pix)

    unwrapped_obs_pix, _ = unwrapped_pix.reset(key)
    obs_pix, _ = env_pix.reset(key)

    assert obs_pix.ndim == 1, "Pixel obs should be a 1D array"
    assert obs_pix.shape[0] == 4 * 160 * 210 * 3
    # Check order: first part of flattened obs should match flattened first frame
    assert jnp.array_equal(obs_pix[:160*210*3], unwrapped_obs_pix[0].flatten())

    # --- Test 3: Wrapping PixelAndObjectCentricWrapper ---
    unwrapped_both = PixelAndObjectCentricWrapper(atari_env)
    env_both = FlattenObservationWrapper(unwrapped_both)

    unwrapped_obs_both, _ = unwrapped_both.reset(key)
    obs_both, _ = env_both.reset(key)

    assert isinstance(obs_both, tuple), "Combined obs should remain a tuple"
    
    # Check pixel part
    pix_part = obs_both[0]
    assert pix_part.ndim == 1, "Pixel part of combined obs should be 1D"
    assert pix_part.shape[0] == 4 * 160 * 210 * 3
    assert jnp.array_equal(pix_part[:160*210*3], unwrapped_obs_both[0][0].flatten())

    # Check OC part
    oc_part = obs_both[1]
    assert oc_part.ndim == 1, "OC part of combined obs should be 1D"
    assert oc_part.shape[0] == 4 * get_object_centric_obs_size(base_env.observation_space())
    assert jnp.array_equal(oc_part[:int(get_object_centric_obs_size(base_env.observation_space()))], unwrapped_obs_both[1][0])


def test_log_wrapper_with_flatten_observation():
    """Test that LogWrapper works correctly with FlattenObservationWrapper."""
    key = jax.random.PRNGKey(0)
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=4)
    
    # Test with a complex observation stack
    # Order: Observation generation (pixel and object-centric) -> Flattening -> Logging
    core_env = PixelAndObjectCentricWrapper(atari_env)
    flattened_env = FlattenObservationWrapper(core_env)
    env = LogWrapper(flattened_env)

    # Reset and check initial state
    obs, state = env.reset(key)
    assert state.episode_returns == 0.0
    assert state.episode_lengths == 0

    # Check that the observation from the logger is correctly flattened
    assert isinstance(obs, tuple)
    assert obs[0].ndim == 1 # Pixel part is 1D
    assert obs[1].ndim == 1 # OC part is 1D

    # Take one step
    obs, state, reward, done, info = env.step(state, 0)

    # Check that logging info is present and obs is still flattened
    assert "returned_episode" in info
    assert isinstance(obs, tuple)
    assert obs[0].ndim == 1
    assert obs[1].ndim == 1


def test_wrapper_observation_spaces():
    """
    Tests that all wrappers correctly modify and present their observation_space.
    """
    key = jax.random.PRNGKey(0)
    stack_size = 4
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=stack_size)

    # --- Test AtariWrapper ---
    assert isinstance(atari_env.observation_space(), spaces.Dict)
    original_leaves = jax.tree.leaves(base_env.observation_space())
    stacked_leaves = jax.tree.leaves(atari_env.observation_space())
    for original_leaf, stacked_leaf in zip(original_leaves, stacked_leaves):
        # Shape should have the stack_size as the new first dimension.
        assert stacked_leaf.shape == (stack_size,) + original_leaf.shape
        # Bounds and dtype should be preserved.
        assert jnp.all(stacked_leaf.low == original_leaf.low)
        assert jnp.all(stacked_leaf.high == original_leaf.high)
        assert stacked_leaf.dtype == original_leaf.dtype

    # --- Test PixelObsWrapper ---
    pixel_env = PixelObsWrapper(atari_env)
    space = pixel_env.observation_space()
    assert isinstance(space, spaces.Box)
    # Shape: (stack_size, H, W, C)
    assert space.shape == (stack_size, 160, 210, 3)
    # Bounds and dtype for pixel data.
    assert space.low == 0
    assert space.high == 255
    assert space.dtype == jnp.uint8

    # --- Test ObjectCentricWrapper ---
    oc_env = ObjectCentricWrapper(atari_env)
    space = oc_env.observation_space()
    assert isinstance(space, spaces.Box)
    
    # Check for the correct 2D shape
    single_frame_size = get_object_centric_obs_size(base_env.observation_space())
    assert space.shape == (stack_size, single_frame_size)

    # Check that the bounds are 1D and match a single frame
    assert space.low.shape == (single_frame_size,)
    assert space.high.shape == (single_frame_size,)

    # --- Test PixelAndObjectCentricWrapper ---
    both_env = PixelAndObjectCentricWrapper(atari_env)
    space = both_env.observation_space()
    assert isinstance(space, spaces.Tuple)
    assert len(space.spaces) == 2

    # The first element should be the stacked pixel space.
    pixel_part = space.spaces[0]
    assert pixel_part.shape == (stack_size, 160, 210, 3)
    assert pixel_part.low == 0
    assert pixel_part.high == 255

    # The second element should be the original stacked object-centric dict space.
    object_part = space.spaces[1]
    assert isinstance(object_part, spaces.Box)
    assert object_part.shape == (stack_size, get_object_centric_obs_size(base_env.observation_space()))


def test_flatten_observation_wrapper_space_structure():
    """
    Tests that FlattenObservationWrapper correctly flattens the leaf spaces
    of a given environment's observation space Pytree.
    """
    key = jax.random.PRNGKey(0)
    stack_size = 4
    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=stack_size)

    # Wrap the Atari environment with the FlattenObservationWrapper
    flatten_env = FlattenObservationWrapper(atari_env)

    # --- 1. Verify the Space Transformation ---

    # Access the original and flattened spaces as attributes
    original_space = atari_env.observation_space()
    flattened_space = flatten_env.observation_space()

    # The overall Pytree structure (the Dict keys) should be identical
    assert isinstance(flattened_space, spaces.Dict)
    assert original_space.spaces.keys() == flattened_space.spaces.keys()

    # Get the individual leaves from both space Pytrees
    original_leaves = jax.tree.leaves(original_space)
    flattened_leaves = jax.tree.leaves(flattened_space)

    # Check each leaf to ensure it was flattened correctly
    for original_leaf, flattened_leaf in zip(original_leaves, flattened_leaves):
        # a) The new leaf's shape should be 1D.
        assert len(flattened_leaf.shape) == 1, f"Expected 1D shape, got {len(flattened_leaf.shape)} dimensions"

        # b) The size of the new 1D leaf should equal the total number of elements in the original
        expected_size = np.prod(original_leaf.shape)
        assert flattened_leaf.shape[0] == expected_size, f"Expected size {expected_size}, got {flattened_leaf.shape[0]}"

        # c) The bounds should have been correctly broadcast and flattened
        expected_low = np.broadcast_to(original_leaf.low, original_leaf.shape).flatten()
        assert jnp.array_equal(flattened_leaf.low, expected_low)

    # --- 2. Verify the Runtime Observation Transformation ---

    # Get a sample observation from the flattened environment
    obs_flat, state = flatten_env.reset(key)

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

    # 1. Convert the observation from a (potentially nested) namedtuple to a
    #    standard dict Pytree. This is our data Pytree.
    obs_flat_dict = deep_asdict(obs_flat)

    # 2. Get a flat list of the leaves from both the data and the space.
    #    `jax.tree.leaves` reliably extracts the contents in the same order.
    obs_leaves = jax.tree.leaves(obs_flat_dict)
    space_leaves = jax.tree.leaves(flattened_space)

    # 3. First, verify the structures have the same number of leaves.
    assert len(obs_leaves) == len(
        space_leaves
    ), "Mismatch between number of observation leaves and space leaves."

    # 4. Now, iterate through the parallel lists and check each data leaf
    #    against its corresponding space leaf.
    for obs_leaf, space_leaf in zip(obs_leaves, space_leaves):
        # The space leaf should be a Box, and we check if the obs_leaf is contained in it.
        assert isinstance(space_leaf, spaces.Box)
        assert space_leaf.contains(
            obs_leaf
        ), f"Observation leaf with shape {obs_leaf.shape} not contained in space with shape {space_leaf.shape}"


if __name__ == "__main__":
    pytest.main([__file__]) 