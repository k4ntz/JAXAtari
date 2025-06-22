import jax
import jax.numpy as jnp
import pytest
import jaxatari
from jaxatari.core import list_available_games
from jaxatari.environment import JAXAtariAction
from jaxatari.spaces import Space, Discrete, Box, Dict, Tuple, stack_space
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, PixelAndObjectCentricWrapper, FlattenObservationWrapper, LogWrapper, MultiRewardLogWrapper


def test_game_basic_functionality(game_name: str):
    """Test basic functionality of a game with 100 steps."""
    # Create environment
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    
    # Test reset
    obs, state = env.reset(key)
    assert obs is not None, f"Observation should not be None for {game_name}"
    assert state is not None, f"State should not be None for {game_name}"
    
    # Test action space
    action_space = env.action_space()
    assert action_space is not None, f"Action space should not be None for {game_name}"
    
    # Test 100 steps
    total_reward = 0.0
    step_count = 0
    done = False
    
    for step in range(100):
        # Choose a random action
        action = action_space.sample(key)
        
        # Take step
        obs, state, reward, done, info = env.step(state, action)
        
        # Basic assertions
        assert obs is not None, f"Observation should not be None at step {step} for {game_name}"
        assert state is not None, f"State should not be None at step {step} for {game_name}"
        assert isinstance(reward, (float, jnp.ndarray)), f"Reward should be float or jnp.ndarray at step {step} for {game_name}"
        assert isinstance(done, (bool, jnp.ndarray)), f"Done should be bool or jnp.ndarray at step {step} for {game_name}"
        assert info is not None, f"Info should not be None at step {step} for {game_name}"
        
        # Accumulate reward and count steps
        total_reward += float(reward)
        step_count += 1
        
        # Update key for next random action
        key, _ = jax.random.split(key)
        
        # If done, break early
        if done:
            break
    
    # Final assertions
    assert step_count > 0, f"Should have taken at least one step for {game_name}"
    assert step_count <= 100, f"Should not exceed 100 steps for {game_name}"
    assert isinstance(total_reward, float), f"Total reward should be float for {game_name}"


def test_game_observation_consistency(game_name: str):
    """Test that observations are consistent across steps for a game."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset(key)
    initial_obs_shape = obs.shape if hasattr(obs, 'shape') else None
    initial_obs_type = type(obs)
    
    # Take 100 steps and verify observation consistency
    action_space = env.action_space()
    
    for step in range(100):
        action = action_space.sample(key)
        obs, state, reward, done, info = env.step(state, action)
        
        # Check observation type consistency
        assert isinstance(obs, initial_obs_type), f"Observation type should remain consistent at step {step} for {game_name}"
        
        # Check observation shape consistency (if applicable)
        if initial_obs_shape is not None:
            assert obs.shape == initial_obs_shape, f"Observation shape should remain consistent at step {step} for {game_name}"
        
        # Update key
        key, _ = jax.random.split(key)
        
        if done:
            break


def test_game_state_transitions(game_name: str):
    """Test that state transitions are working correctly for a game."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset(key)
    
    # Take 100 steps and verify state changes
    action_space = env.action_space()
    
    for step in range(100):
        action = action_space.sample(key)
        obs, state, reward, done, info = env.step(state, action)
        
        # Check that state is not None
        assert state is not None, f"State should not be None at step {step} for {game_name}"
        
        # Update key
        key, _ = jax.random.split(key)
        
        if done:
            break


def test_game_reward_consistency(game_name: str):
    """Test that rewards are consistent and reasonable for a game."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset(key)
    
    # Take 100 steps and verify reward consistency
    action_space = env.action_space()
    rewards = []
    
    for step in range(100):
        action = action_space.sample(key)
        obs, state, reward, done, info = env.step(state, action)
        
        # Check reward type and value
        assert isinstance(reward, (float, jnp.ndarray)), f"Reward should be float or jnp.ndarray at step {step} for {game_name}"
        
        # Convert to float and store
        reward_float = float(reward)
        rewards.append(reward_float)
        
        # Check that reward is finite
        assert jnp.isfinite(reward_float), f"Reward should be finite at step {step} for {game_name}"
        
        # Update key
        key, _ = jax.random.split(key)
        
        if done:
            break
    
    # Verify we collected some rewards
    assert len(rewards) > 0, f"Should have collected rewards for {game_name}"


def test_game_render_functionality(game_name: str):
    """Test that the render function works correctly for a game."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset(key)
    
    # Test render function
    rendered_image = env.render(state)
    assert rendered_image is not None, f"Rendered image should not be None for {game_name}"
    assert isinstance(rendered_image, (tuple, jnp.ndarray)), f"Rendered image should be tuple or jnp.ndarray for {game_name}"
    
    # If it's a tuple, check the first element
    if isinstance(rendered_image, tuple):
        assert len(rendered_image) > 0, f"Rendered image tuple should not be empty for {game_name}"
        rendered_image = rendered_image[0]
    
    # Check that the rendered image is a valid array
    assert isinstance(rendered_image, jnp.ndarray), f"Rendered image should be jnp.ndarray for {game_name}"
    assert rendered_image.ndim >= 2, f"Rendered image should have at least 2 dimensions for {game_name}"


def test_game_observation_space(game_name: str):
    """Test that the observation space is correctly defined for a game."""
    env = jaxatari.make(game_name)
    obs_space = env.observation_space()
    assert obs_space is not None, f"Observation space should not be None for {game_name}"
    assert isinstance(obs_space, Space), f"Observation space should be a Space instance for {game_name}"
    key = jax.random.PRNGKey(0)
    sample_obs = obs_space.sample(key)
    assert sample_obs is not None, f"Observation space sample should not be None for {game_name}"


def test_game_image_space(game_name: str):
    """Test that the image space is correctly defined for a game."""
    env = jaxatari.make(game_name)
    image_space = env.image_space()
    assert image_space is not None, f"Image space should not be None for {game_name}"
    assert isinstance(image_space, Space), f"Image space should be a Space instance for {game_name}"
    key = jax.random.PRNGKey(0)
    sample_image = image_space.sample(key)
    assert sample_image is not None, f"Image space sample should not be None for {game_name}"


def test_game_obs_to_flat_array(game_name: str):
    """Test that the obs_to_flat_array function works correctly for a game."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    flat_obs = env.obs_to_flat_array(obs)
    assert flat_obs is not None, f"Flat observation should not be None for {game_name}"
    assert isinstance(flat_obs, jnp.ndarray), f"Flat observation should be jnp.ndarray for {game_name}"
    assert flat_obs.ndim == 1, f"Flat observation should be 1-dimensional for {game_name}"


def test_game_basic_wrapper_compatibility(game_name: str):
    """Test basic wrapper compatibility for a game."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    atari_wrapper = AtariWrapper(env)
    obs, state = atari_wrapper.reset(key)
    assert obs is not None, f"AtariWrapper reset should return valid observation for {game_name}"
    assert state is not None, f"AtariWrapper reset should return valid state for {game_name}"
    action_space = atari_wrapper.action_space()
    action = action_space.sample(key)
    obs, state, reward, done, info = atari_wrapper.step(state, action)
    assert obs is not None, f"AtariWrapper step should return valid observation for {game_name}"
    assert state is not None, f"AtariWrapper step should return valid state for {game_name}"


def test_jaxatari_action_constants():
    """Test that JAXAtariAction constants are correctly defined."""
    # Test all action constants
    assert JAXAtariAction.NOOP == 0
    assert JAXAtariAction.FIRE == 1
    assert JAXAtariAction.UP == 2
    assert JAXAtariAction.RIGHT == 3
    assert JAXAtariAction.LEFT == 4
    assert JAXAtariAction.DOWN == 5
    assert JAXAtariAction.UPRIGHT == 6
    assert JAXAtariAction.UPLEFT == 7
    assert JAXAtariAction.DOWNRIGHT == 8
    assert JAXAtariAction.DOWNLEFT == 9
    assert JAXAtariAction.UPFIRE == 10
    assert JAXAtariAction.RIGHTFIRE == 11
    assert JAXAtariAction.LEFTFIRE == 12
    assert JAXAtariAction.DOWNFIRE == 13
    assert JAXAtariAction.UPRIGHTFIRE == 14
    assert JAXAtariAction.UPLEFTFIRE == 15
    assert JAXAtariAction.DOWNRIGHTFIRE == 16
    assert JAXAtariAction.DOWNLEFTFIRE == 17
    
    # Test get_all_values method
    all_values = JAXAtariAction.get_all_values()
    assert isinstance(all_values, jnp.ndarray), "get_all_values should return jnp.ndarray"
    assert all_values.shape == (18,), "get_all_values should have shape (18,)"
    assert all_values.dtype == jnp.int32, "get_all_values should have dtype int32"


def test_spaces_functionality():
    """Test that the spaces module functions work correctly."""
    key = jax.random.PRNGKey(0)
    
    # Test Discrete space
    discrete_space = Discrete(6)
    assert discrete_space.n == 6
    assert discrete_space.shape == ()
    assert discrete_space.dtype == jnp.int32
    
    sample = discrete_space.sample(key)
    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == ()
    assert sample.dtype == jnp.int32
    assert discrete_space.contains(sample)
    
    low, high = discrete_space.range()
    assert low == 0
    assert high == 5
    
    # Test Box space
    box_space = Box(low=0, high=1, shape=(3, 4))
    assert box_space.shape == (3, 4)
    assert box_space.dtype == jnp.float32
    
    sample = box_space.sample(key)
    assert isinstance(sample, jnp.ndarray)
    assert sample.shape == (3, 4)
    assert box_space.contains(sample)
    
    low, high = box_space.range()
    assert jnp.array_equal(low, jnp.array(0))
    assert jnp.array_equal(high, jnp.array(1))
    
    # Test Dict space
    dict_space = Dict({
        "x": Box(low=0, high=10, shape=()),
        "y": Box(low=0, high=10, shape=())
    })
    
    sample = dict_space.sample(key)
    assert isinstance(sample, dict)
    assert "x" in sample
    assert "y" in sample
    assert dict_space.contains(sample)
    
    # Test Tuple space
    tuple_space = Tuple([
        Box(low=0, high=1, shape=(2,)),
        Discrete(5)
    ])
    
    sample = tuple_space.sample(key)
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert tuple_space.contains(sample)
    
    # Test stack_space function
    stacked_space = stack_space(box_space, 4)
    assert isinstance(stacked_space, Box)
    assert stacked_space.shape == (4, 3, 4)


def test_all_games_available():
    """Test that all games can be listed and created."""
    games = list_available_games()
    assert len(games) > 0, "Should have at least one game available"
    
    # Test that each game can be created
    for game_name in games:
        env = jaxatari.make(game_name)
        assert env is not None, f"Environment for {game_name} should not be None"


def test_game_names_are_valid():
    """Test that all game names are valid Python identifiers."""
    games = list_available_games()
    
    for game_name in games:
        # Check that game name is a valid Python identifier
        assert game_name.isidentifier(), f"Game name '{game_name}' should be a valid Python identifier"
        
        # Check that game name is not empty
        assert len(game_name) > 0, f"Game name should not be empty"
        
        # Check that game name contains only alphanumeric characters and underscores
        assert game_name.replace('_', '').isalnum(), f"Game name '{game_name}' should contain only alphanumeric characters and underscores"


def test_game_jit_compatibility(game_name: str):
    """Test that games work correctly with JAX JIT compilation."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(0)
    
    # JIT the reset and step functions
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    
    # Test JIT reset
    obs, state = jitted_reset(key)
    assert obs is not None
    assert state is not None
    
    # Test JIT step
    action = env.action_space().sample(key)
    obs, state, reward, done, info = jitted_step(state, action)
    assert obs is not None
    assert state is not None


def test_game_deterministic_behavior(game_name: str):
    """Test that games produce deterministic results with same key and actions."""
    env = jaxatari.make(game_name)
    key = jax.random.PRNGKey(42)
    
    # Run two identical episodes
    obs1, state1 = env.reset(key)
    obs2, state2 = env.reset(key)
    
    # States should be identical
    assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, state1, state2))
    
    # Run same sequence of actions
    actions = [0, 1, 2, 0, 1]  # Fixed action sequence
    for action in actions:
        obs1, state1, reward1, done1, info1 = env.step(state1, action)
        obs2, state2, reward2, done2, info2 = env.step(state2, action)
        
        # Results should be identical
        assert jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, state1, state2))
        assert jnp.array_equal(reward1, reward2)
        assert jnp.array_equal(done1, done2)


def run_all_game_tests():
    """Run tests for all available games."""
    print("Running tests for all available games...")
    available_games = list_available_games()
    print(f"Found {len(available_games)} games: {available_games}")
    passed_tests = 0
    failed_tests = 0
    total_tests = 0
    for game_name in available_games:
        print(f"\n--- Testing {game_name} ---")
        total_tests += 1
        try:
            test_game_basic_functionality(game_name)
            print(f"✓ {game_name} basic functionality: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} basic functionality: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_observation_consistency(game_name)
            print(f"✓ {game_name} observation consistency: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} observation consistency: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_state_transitions(game_name)
            print(f"✓ {game_name} state transitions: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} state transitions: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_reward_consistency(game_name)
            print(f"✓ {game_name} reward consistency: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} reward consistency: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_render_functionality(game_name)
            print(f"✓ {game_name} render functionality: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} render functionality: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_observation_space(game_name)
            print(f"✓ {game_name} observation space: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} observation space: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_image_space(game_name)
            print(f"✓ {game_name} image space: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} image space: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_obs_to_flat_array(game_name)
            print(f"✓ {game_name} obs_to_flat_array: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} obs_to_flat_array: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_basic_wrapper_compatibility(game_name)
            print(f"✓ {game_name} basic wrapper compatibility: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} basic wrapper compatibility: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_jit_compatibility(game_name)
            print(f"✓ {game_name} JIT compatibility: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} JIT compatibility: FAILED - {e}")
            failed_tests += 1
        total_tests += 1
        try:
            test_game_deterministic_behavior(game_name)
            print(f"✓ {game_name} deterministic behavior: PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"✗ {game_name} deterministic behavior: FAILED - {e}")
            failed_tests += 1
    # Test general functionality
    print(f"\n--- Testing general functionality ---")
    
    total_tests += 1
    try:
        test_all_games_available()
        print("✓ All games available: PASSED")
        passed_tests += 1
    except Exception as e:
        print(f"✗ All games available: FAILED - {e}")
        failed_tests += 1
    
    total_tests += 1
    try:
        test_game_names_are_valid()
        print("✓ Game names are valid: PASSED")
        passed_tests += 1
    except Exception as e:
        print(f"✗ Game names are valid: FAILED - {e}")
        failed_tests += 1
    
    # Test JAXAtariAction constants
    total_tests += 1
    try:
        test_jaxatari_action_constants()
        print("✓ JAXAtariAction constants: PASSED")
        passed_tests += 1
    except Exception as e:
        print(f"✗ JAXAtariAction constants: FAILED - {e}")
        failed_tests += 1
    
    # Test spaces functionality
    total_tests += 1
    try:
        test_spaces_functionality()
        print("✓ Spaces functionality: PASSED")
        passed_tests += 1
    except Exception as e:
        print(f"✗ Spaces functionality: FAILED - {e}")
        failed_tests += 1
    
    # Print summary
    print(f"\n--- Test Summary ---")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests > 0:
        return False
    else:
        print(f"\nAll tests passed!")
        return True


if __name__ == "__main__":
    # Run all tests when executed directly
    success = run_all_game_tests()
    exit(0 if success else 1)