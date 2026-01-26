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

def test_obs_to_flat_array_with_stacked_observations(raw_env):
    """Test that obs_to_flat_array works correctly with stacked observations."""
    key = jax.random.PRNGKey(0)
    base_env = raw_env
    atari_env = AtariWrapper(base_env, frame_stack_size=4)
    env = ObjectCentricWrapper(atari_env)

    obs, state = env.reset(key)

    # The observation should maintain the stacked structure
    single_frame_size = get_object_centric_obs_size(atari_env._env.observation_space())
    assert obs.shape == (4, single_frame_size), f"Expected shape (4, {single_frame_size}), got {obs.shape}"

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
    obs, state, reward, done, info = env.step(state, 0)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"

    # perform 100 steps to get to a state in which the frames should be different
    for _ in range(100):
        obs, state, reward, done, info = env.step(state, 0)
    
    # Verify that frames are in the correct range (0-255 for uint8)
    assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"

def test_pixel_and_object_centric_wrapper(raw_env):
    """Test that PixelAndObjectCentricWrapper returns both pixel and flattened object-centric observations."""
    key = jax.random.PRNGKey(0)
    stack_size = 4
    base_env = raw_env
    atari_env = AtariWrapper(base_env, frame_stack_size=stack_size)
    env = PixelAndObjectCentricWrapper(atari_env)

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

def test_log_wrapper(raw_env):
    """Test that LogWrapper correctly tracks episode returns and lengths."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = raw_env
    env = LogWrapper(PixelObsWrapper(AtariWrapper(base_env)))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify initial state
    assert state.episode_returns == 0.0
    assert state.episode_lengths == 0
    assert state.returned_episode_returns == 0.0
    assert state.returned_episode_lengths == 0

    # Check for non-standard base shapes and issue a warning
    base_img_shape = base_env.image_space().shape
    if base_img_shape not in [(210, 160, 3), (250, 160, 3)]:
        warnings.warn(f"Running test with a non-standard Atari shape: {base_img_shape}. Be sure this is intended!", UserWarning)

    # Verify observation format against the environment's observation space
    expected_shape = env.observation_space().shape
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

def test_multi_reward_log_wrapper(raw_env):
    """Test that MultiRewardLogWrapper correctly tracks multiple reward types."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = raw_env
    reward_funcs = [lambda state, prev_state: jnp.ones(1)]
    env = MultiRewardLogWrapper(PixelAndObjectCentricWrapper(AtariWrapper(MultiRewardWrapper(base_env, reward_funcs))))
    
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

    # Check for non-standard base shapes and issue a warning
    base_img_shape = base_env.image_space().shape
    if base_img_shape not in [(210, 160, 3), (250, 160, 3)]:
        warnings.warn(f"Running test with a non-standard Atari shape: {base_img_shape}. Be sure this is intended!", UserWarning)

    # Verify shapes against the environment's observation space
    expected_image_shape = env.observation_space().spaces[0].shape
    expected_object_shape = env.observation_space().spaces[1].shape
    
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


def test_flatten_observation_wrapper(raw_env):
    """Test that FlattenObservationWrapper correctly flattens each observation type."""
    key = jax.random.PRNGKey(0)
    base_env = raw_env
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
    unwrapped_both = PixelAndObjectCentricWrapper(atari_env)
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

def test_flatten_observation_wrapper_space_structure(raw_env):
    """
    Tests that FlattenObservationWrapper correctly flattens the leaf spaces
    of a given environment's observation space Pytree.
    """
    key = jax.random.PRNGKey(0)
    stack_size = 4
    base_env = raw_env
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
        Recursively converts a Pytree of namedtuples or dataclasses into a Pytree of standard dicts.
        This is needed because obs might be a namedtuple or dataclass but the space is a Dict.
        """
        if hasattr(obj, '_asdict'): # It's a namedtuple
            return collections.OrderedDict(
                (key, deep_asdict(value)) for key, value in obj._asdict().items()
            )
        elif is_dataclass(obj): # It's a dataclass
            from dataclasses import asdict
            return collections.OrderedDict(
                (key, deep_asdict(value)) for key, value in asdict(obj).items()
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

def test_atari_wrapper_features_and_pixel_preprocessing(raw_env):
    """Tests max-pooling, resizing, and grayscaling features."""
    key = jax.random.PRNGKey(0)
    
    # --- Test 1: AtariWrapper Max-Pooling and Sticky Actions ---
    class FakeEnv:
        def __init__(self):
            self.state = 0
            self._observation_space = spaces.Dict({"features": spaces.Box(low=0, high=255, shape=(2,2), dtype=jnp.uint8)})
        
        def observation_space(self) -> spaces.Space:
            return self._observation_space

        def reset(self, key):
            self.state = 0
            obs = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), self._observation_space)
            return obs, self.state
        
        def step(self, state, action):
            state += 1
            obs = jax.tree.map(lambda s: (jnp.ones(s.shape, s.dtype) * state), self._observation_space)
            return obs, state, 1.0, False, {"all_rewards": jnp.array([1.0])}
        
        def render(self, state):
             return jnp.zeros((210, 160, 3), dtype=jnp.uint8)

    base_env = AtariWrapper(FakeEnv(), frame_skip=4, max_pooling=True, first_fire=False, episodic_life=False, sticky_actions=True)
    _, state = base_env.reset(key)
    obs, state, _, _, _ = base_env.step(state, 0)
    
    expected_max_pooled_frame = jnp.ones((2,2), dtype=jnp.uint8) * 4
    
    assert jnp.array_equal(obs.spaces['features'][-1], expected_max_pooled_frame)

    # --- Test 2: PixelObsWrapper Preprocessing ---
    RESIZE_SHAPE = (84, 84)
    STACK_SIZE = 4
    atari_env = AtariWrapper(raw_env, frame_stack_size=STACK_SIZE)
    pixel_env = PixelObsWrapper(atari_env, do_pixel_resize=True, pixel_resize_shape=RESIZE_SHAPE, grayscale=True)
    
    space = pixel_env.observation_space()
    expected_shape = (STACK_SIZE, RESIZE_SHAPE[0], RESIZE_SHAPE[1], 1)
    assert space.shape == expected_shape
    assert space.dtype == jnp.uint8
    
    obs, state = pixel_env.reset(key)
    assert obs.shape == expected_shape

    # --- Test 3: PixelAndObjectCentricWrapper Preprocessing ---
    atari_env_2 = AtariWrapper(raw_env, frame_stack_size=STACK_SIZE)
    mixed_env = PixelAndObjectCentricWrapper(atari_env_2, do_pixel_resize=True, pixel_resize_shape=RESIZE_SHAPE, grayscale=True)

    pix_space, obj_space = mixed_env.observation_space().spaces
    assert pix_space.shape == expected_shape
    
    (pix_obs, obj_obs), state = mixed_env.reset(key)
    assert pix_obs.shape == expected_shape
    assert obj_obs.ndim == 2


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
    obs, state, reward, done, info = env.step(state, 0)
    assert obs.shape == expected_shape, f"Step observation shape mismatch. Got {obs.shape}"

    # 6. Verify Values (Sanity Check)
    # Ensure we aren't getting empty arrays or garbage
    assert obs.dtype == jnp.uint8
    assert jnp.max(obs) <= 255
    assert jnp.min(obs) >= 0


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