import collections
import jax
import jax.numpy as jnp
import pytest
import jaxatari
from dataclasses import is_dataclass
from jaxatari.environment import EnvInfo, EnvObs, EnvState
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    MultiRewardWrapper,
    AtariWrapper,
    PixelAndObjectCentricWrapper,
    LogWrapper,
    MultiRewardLogWrapper, 
    FlattenObservationWrapper
)
import jaxatari.spaces as spaces
import numpy as np
import warnings

def get_object_centric_obs_size(space: spaces.Dict) -> int:
    """Helper to correctly calculate the total flattened size of an object-centric space."""
    size = 0
    for leaf in jax.tree.leaves(space):
        size += np.prod(leaf.shape)
    return size

@pytest.mark.integration
def test_pixel_obs_wrapper_with_stacked_frames(raw_env):
    """Test that PixelObsWrapper correctly handles stacked frames."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = raw_env
    env = PixelObsWrapper(AtariWrapper(base_env))
    
    # Get initial observation
    obs, state = env.reset(key)

    # Check for non-standard base shapes and issue a warning (not an error!)
    base_img_shape = base_env.image_space().shape
    if base_img_shape not in [(210, 160, 3), (250, 160, 3)]:
        warnings.warn(f"Running test with a non-standard Atari shape: {base_img_shape}. Be sure this is intended!", UserWarning)
    
    # Verify shape against the environment's observation space
    expected_shape = env.observation_space().shape
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    # Take a step and verify shape remains consistent
    obs, state, reward, done, _, info = env.step(state, 0)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"

    # perform 100 steps to get to a state in which the frames should be different
    for _ in range(100):
        obs, state, reward, done, _, info = env.step(state, 0)
    
    # Verify that frames are in the correct range (0-255 for uint8)
    assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"

def test_pixel_and_object_centric_wrapper(raw_env):
    """Test that PixelAndObjectCentricWrapper returns both pixel and flattened object-centric observations."""
    key = jax.random.PRNGKey(0)
    stack_size = 4
    base_env = raw_env
    atari_env = AtariWrapper(base_env)
    env = PixelAndObjectCentricWrapper(atari_env, frame_stack_size=stack_size)

    # 1. Test the space definition
    space = env.observation_space()
    assert isinstance(space, spaces.Tuple)
    assert len(space.spaces) == 2

    pixel_space, object_space = space.spaces
    assert isinstance(pixel_space, spaces.Box)

    # Check for non-standard base shapes and issue a warning
    base_img_shape = base_env.image_space().shape
    if base_img_shape not in [(210, 160, 3), (250, 160, 3)]:
        warnings.warn(f"Running test with a non-standard Atari shape: {base_img_shape}. Be sure this is intended!", UserWarning)

    # Check pixel space against the base environment's image space
    expected_pixel_shape = (stack_size,) + base_img_shape
    assert pixel_space.shape == expected_pixel_shape, f"Expected shape {expected_pixel_shape}, got {pixel_space.shape}"

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


def test_object_centric_wrapper(raw_env):
    """Test ObjectCentricWrapper returns a 2D stacked observation and its space is correct."""
    key = jax.random.PRNGKey(0)
    base_env = raw_env
    atari_env = AtariWrapper(base_env)
    env = ObjectCentricWrapper(atari_env, frame_stack_size=4)

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
    obs, state, _, _, _, _ = env.step(state, 2) # Use an action that causes change
    assert obs.shape == space.shape


def test_flatten_observation_wrapper(raw_env):
    """Test that FlattenObservationWrapper correctly flattens each observation type."""
    key = jax.random.PRNGKey(0)
    base_env = raw_env
    atari_env = AtariWrapper(base_env)

    # --- Test 1: Wrapping ObjectCentricWrapper ---
    unwrapped_oc = ObjectCentricWrapper(atari_env, frame_stack_size=4)
    env_oc = FlattenObservationWrapper(unwrapped_oc)
    
    unwrapped_obs_oc, _ = unwrapped_oc.reset(key)
    obs_oc, _ = env_oc.reset(key)
    
    assert obs_oc.ndim == 1, "OC obs should be a 1D array"
    assert obs_oc.shape[0] == 4 * get_object_centric_obs_size(base_env.observation_space())
    # Check order: first part of flattened obs should match first frame of unwrapped obs
    assert jnp.array_equal(obs_oc[:int(get_object_centric_obs_size(base_env.observation_space()))], unwrapped_obs_oc[0])

    # --- Test 2: Wrapping PixelObsWrapper ---
    unwrapped_pix = PixelObsWrapper(atari_env, frame_stack_size=4)
    env_pix = FlattenObservationWrapper(unwrapped_pix)

    unwrapped_obs_pix, _ = unwrapped_pix.reset(key)
    obs_pix, _ = env_pix.reset(key)

    assert obs_pix.ndim == 1, "Pixel obs should be a 1D array"

    # Check for non-standard base shapes and issue a warning
    base_img_shape = base_env.image_space().shape
    if base_img_shape not in [(210, 160, 3), (250, 160, 3)]:
        warnings.warn(f"Running test with a non-standard Atari shape: {base_img_shape}. Be sure this is intended!", UserWarning)

    # Generalize the shape check
    expected_pixel_size = 4 * np.prod(base_img_shape)
    assert obs_pix.shape[0] == expected_pixel_size
    # Check order: first part of flattened obs should match flattened first frame
    assert jnp.array_equal(obs_pix[:np.prod(base_img_shape)], unwrapped_obs_pix[0].flatten())

    # --- Test 3: Wrapping PixelAndObjectCentricWrapper ---
    unwrapped_both = PixelAndObjectCentricWrapper(atari_env, frame_stack_size=4)
    env_both = FlattenObservationWrapper(unwrapped_both)

    unwrapped_obs_both, _ = unwrapped_both.reset(key)
    obs_both, _ = env_both.reset(key)

    assert isinstance(obs_both, tuple), "Combined obs should remain a tuple"
    
    # Check pixel part
    pix_part = obs_both[0]
    assert pix_part.ndim == 1, "Pixel part of combined obs should be 1D"
    assert pix_part.shape[0] == expected_pixel_size # Use generalized size from above
    assert jnp.array_equal(pix_part[:np.prod(base_img_shape)], unwrapped_obs_both[0][0].flatten())

    # Check OC part
    oc_part = obs_both[1]
    assert oc_part.ndim == 1, "OC part of combined obs should be 1D"
    assert oc_part.shape[0] == 4 * get_object_centric_obs_size(base_env.observation_space())
    assert jnp.array_equal(oc_part[:int(get_object_centric_obs_size(base_env.observation_space()))], unwrapped_obs_both[1][0])


def test_log_wrapper_with_flatten_observation(raw_env):
    """Test that LogWrapper works correctly with FlattenObservationWrapper."""
    key = jax.random.PRNGKey(0)
    base_env = raw_env
    atari_env = AtariWrapper(base_env)
    
    # Test with a complex observation stack
    # Order: Observation generation (pixel and object-centric) -> Flattening -> Logging
    core_env = PixelAndObjectCentricWrapper(atari_env, frame_stack_size=4)
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
    obs, state, reward, done, _, info = env.step(state, 0)

    # Check that logging info is present and obs is still flattened
    assert "returned_episode" in info
    assert isinstance(obs, tuple)
    assert obs[0].ndim == 1
    assert obs[1].ndim == 1


@pytest.mark.integration
def test_native_downscaling_hot_swap(raw_env):
    """
    Verifies that enabling use_native_downscaling correctly hot-swaps the 
    renderer configuration and produces downscaled observations natively.
    """
    # 1. Setup: Load raw env and basic wrappers
    env = AtariWrapper(raw_env)
    
    # Define target shape
    TARGET_H, TARGET_W = 84, 84
    
    # 2. Apply Wrapper with Native Downscaling enabled
    # This triggers the "Hot Swap" logic in PixelObsWrapper.__init__
    try:
        env = PixelObsWrapper(
            env, 
            do_pixel_resize=True, 
            pixel_resize_shape=(TARGET_H, TARGET_W), 
            use_native_downscaling=True
        )
    except TypeError as e:
        pytest.fail(f"Game renderer likely hasn't updated its __init__ to accept 'config'. Error: {e}")

    # 3. Check Observation Space (Public Interface)
    # The wrapper should have patched image_space() or observation_space()
    obs_space = env.observation_space()
    
    # Expected: (Stack, H, W, 3) or (Stack, H, W, 1)
    # We assume default RGB (channels=3) for this test unless env forces grayscale
    expected_shape = (env.frame_stack_size, TARGET_H, TARGET_W, 3)
    
    assert obs_space.shape == expected_shape, \
        f"Observation space mismatch. Expected {expected_shape}, got {obs_space.shape}"

    # 4. Check Internals (The Hot Swap Verification)
    # We dig into the wrapped instance to verify the Renderer config was actually updated.
    # Stack: PixelObsWrapper -> AtariWrapper -> GameEnv
    base_env = env._env._env 
    
    # Ensure the renderer has the downscale config set
    assert base_env.renderer.config.downscale == (TARGET_H, TARGET_W), \
        f"Renderer config was not updated! It is {base_env.renderer.config.downscale}. Wrapper might be falling back to slow resizing."

    # 5. Check Runtime Output (JIT Compilation & Execution)
    # This ensures the JAX graph compiles with the new shapes
    key = jax.random.PRNGKey(0)
    
    # Test Reset
    obs, state = env.reset(key)
    assert obs.shape == expected_shape, f"Reset observation shape mismatch. Got {obs.shape}"
    
    # Test Step
    obs, state, reward, done, _, info = env.step(state, 0)
    assert obs.shape == expected_shape, f"Step observation shape mismatch. Got {obs.shape}"

    # 6. Verify Values (Sanity Check)
    # Ensure we aren't getting empty arrays or garbage
    assert obs.dtype == jnp.uint8
    assert jnp.max(obs) <= 255
    assert jnp.min(obs) >= 0


@pytest.mark.integration
def test_native_downscaling_grayscale(raw_env):
    """
    Specific test to ensure Native Downscaling handles Grayscale channel reduction correctly.
    """
    env = AtariWrapper(raw_env)
    TARGET_H, TARGET_W = 84, 84

    # Enable Grayscale + Native
    env = PixelObsWrapper(
        env, 
        do_pixel_resize=True, 
        pixel_resize_shape=(TARGET_H, TARGET_W), 
        grayscale=True,
        use_native_downscaling=True
    )

    # Expecting 1 channel now
    expected_shape = (env.frame_stack_size, TARGET_H, TARGET_W, 1)
    
    obs_space = env.observation_space()
    assert obs_space.shape == expected_shape
    
    # Verify Internal Config channels
    base_env = env._env._env
    assert base_env.renderer.config.channels == 1, "Renderer config channels should be 1 for grayscale"

    # Runtime check
    key = jax.random.PRNGKey(0)
    obs, _ = env.reset(key)
    assert obs.shape == expected_shape

if __name__ == "__main__":
    pytest.main([__file__])