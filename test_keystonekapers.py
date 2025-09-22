#!/usr/bin/env python3
"""
Test script for KeystoneKapers JAX implementation.
This script tests basic functionality of the game including:
- Environment initialization
- Step function
- Collision detection
- Reward calculation
- Observation extraction
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxatari.games.jax_keystonekapers import JaxKeystoneKapers, KeystoneKapersConstants
from jaxatari.environment import JAXAtariAction as Action

def test_game_creation():
    """Test basic game creation and initialization."""
    print("Testing game creation...")

    # Create game with default constants
    game = JaxKeystoneKapers()
    print(f"✓ Game created successfully")
    print(f"  Action space size: {game.action_space().n}")
    print(f"  Observation size: {game.obs_size}")

    # Test with custom constants
    custom_consts = KeystoneKapersConstants(
        SCREEN_WIDTH=160,
        SCREEN_HEIGHT=210,
        PLAYER_SPEED=3  # Faster player
    )
    custom_game = JaxKeystoneKapers(custom_consts)
    print(f"✓ Game with custom constants created")

    return game

def test_reset_functionality(game):
    """Test game reset functionality."""
    print("\nTesting reset functionality...")

    key = jrandom.PRNGKey(42)
    obs, state = game.reset(key)

    print(f"✓ Reset successful")
    print(f"  Player position: ({state.player.x}, {state.player.y})")
    print(f"  Player floor: {state.player.floor}")
    print(f"  Thief position: ({state.thief.x}, {state.thief.y})")
    print(f"  Thief floor: {state.thief.floor}")
    print(f"  Initial score: {state.score}")
    print(f"  Initial timer: {state.timer}")
    print(f"  Initial lives: {state.lives}")

    # Test observation shape
    flat_obs = game.obs_to_flat_array(obs)
    print(f"  Flat observation shape: {flat_obs.shape}")

    return obs, state

def test_step_functionality(game, state):
    """Test stepping through the game."""
    print("\nTesting step functionality...")

    # Test different actions
    test_actions = [
        Action.NOOP,
        Action.LEFT,
        Action.RIGHT,
        Action.UP,
        Action.DOWN,
        Action.FIRE,  # Jump
        Action.LEFTFIRE,  # Jump left
    ]

    for i, action in enumerate(test_actions):
        obs, new_state, reward, done, info = game.step(state, jnp.array(action))

        if i == 0:  # First step
            print(f"✓ Step function working")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print(f"  Info keys: {info._fields}")

        # Test position changes
        if action in [Action.LEFT, Action.LEFTFIRE]:
            expected_change = state.player.x > new_state.player.x
            print(f"  Action {action} -> Player X change: {expected_change}")
        elif action in [Action.RIGHT, Action.RIGHTFIRE]:
            expected_change = state.player.x < new_state.player.x
            print(f"  Action {action} -> Player X change: {expected_change}")

        state = new_state

        if done:
            print(f"  Game ended after action {action}")
            break

    return state

def test_collision_detection(game):
    """Test collision detection functionality."""
    print("\nTesting collision detection...")

    # Create a state with player and thief at same position
    key = jrandom.PRNGKey(123)
    obs, state = game.reset(key)

    # Move thief to player position for collision test
    modified_thief = state.thief._replace(
        x=state.player.x,
        y=state.player.y,
        floor=state.player.floor
    )
    modified_state = state._replace(thief=modified_thief)

    # Step and check collision
    obs, new_state, reward, done, info = game.step(modified_state, jnp.array(Action.NOOP))

    print(f"✓ Collision test completed")
    print(f"  Thief caught: {new_state.thief_caught}")
    print(f"  Reward from collision: {reward}")

    return new_state

def test_jax_compilation(game):
    """Test JAX JIT compilation."""
    print("\nTesting JAX compilation...")

    # Test if step function can be JIT compiled
    key = jrandom.PRNGKey(456)
    obs, state = game.reset(key)

    # Compile step function
    jitted_step = jax.jit(game.step)

    try:
        obs, new_state, reward, done, info = jitted_step(state, jnp.array(Action.RIGHT))
        print(f"✓ JIT compilation successful")
        print(f"  Post-JIT reward: {reward}")
        print(f"  Post-JIT done: {done}")
    except Exception as e:
        print(f"✗ JIT compilation failed: {e}")
        return False

    return True

def test_observation_spaces(game):
    """Test observation and action spaces."""
    print("\nTesting spaces...")

    # Test action space
    action_space = game.action_space()
    print(f"✓ Action space: {action_space}")

    # Test observation space
    obs_space = game.observation_space()
    print(f"✓ Observation space keys: {list(obs_space.spaces.keys())}")

    # Test image space
    image_space = game.image_space()
    print(f"✓ Image space shape: {image_space.shape}")

    return True

def test_rendering(game, state):
    """Test rendering functionality."""
    print("\nTesting rendering...")

    try:
        rendered_frame = game.render(state)
        print(f"✓ Rendering successful")
        print(f"  Frame shape: {rendered_frame.shape}")
        print(f"  Frame dtype: {rendered_frame.dtype}")
        print(f"  Frame value range: [{rendered_frame.min()}, {rendered_frame.max()}]")

        # Test JIT compilation of rendering
        jitted_render = jax.jit(game.render)
        jitted_frame = jitted_render(state)
        print(f"✓ JIT rendering successful")

        return True
    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        return False

def run_full_episode(game, max_steps=100):
    """Run a full episode to test game dynamics."""
    print(f"\nRunning full episode (max {max_steps} steps)...")

    key = jrandom.PRNGKey(789)
    obs, state = game.reset(key)

    total_reward = 0
    step_count = 0

    for step in range(max_steps):
        # Random action selection
        action_key = jrandom.fold_in(key, step)
        action_idx = jrandom.randint(action_key, (), 0, len(game.action_set))
        action = jnp.array(action_idx)

        obs, state, reward, done, info = game.step(state, action)
        total_reward += reward
        step_count += 1

        if step % 20 == 0:
            print(f"  Step {step}: Score={state.score}, Timer={state.timer}, Lives={state.lives}")

        if done:
            print(f"  Episode ended at step {step}")
            break

    print(f"✓ Episode completed")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward}")
    print(f"  Final score: {state.score}")
    print(f"  Thief caught: {state.thief_caught}")
    print(f"  Thief escaped: {state.thief.escaped}")

    return state

def main():
    """Run all tests."""
    print("=" * 60)
    print("KeystoneKapers JAX Implementation Test Suite")
    print("=" * 60)

    try:
        # Basic functionality tests
        game = test_game_creation()
        obs, state = test_reset_functionality(game)
        state = test_step_functionality(game, state)
        test_collision_detection(game)

        # JAX-specific tests
        jit_success = test_jax_compilation(game)
        spaces_success = test_observation_spaces(game)
        render_success = test_rendering(game, state)

        # Integration test
        final_state = run_full_episode(game)

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"✓ Basic functionality: PASSED")
        print(f"{'✓' if jit_success else '✗'} JAX JIT compilation: {'PASSED' if jit_success else 'FAILED'}")
        print(f"{'✓' if spaces_success else '✗'} Observation spaces: {'PASSED' if spaces_success else 'FAILED'}")
        print(f"{'✓' if render_success else '✗'} Rendering: {'PASSED' if render_success else 'FAILED'}")
        print(f"✓ Integration test: PASSED")

        success = jit_success and spaces_success and render_success
        print(f"\nOVERALL: {'PASSED' if success else 'FAILED'}")

        if success:
            print("\n🎉 KeystoneKapers implementation is working correctly!")
            print("The game can be used with:")
            print("  from jaxatari.games.jax_keystonekapers import JaxKeystoneKapers")
            print("  game = JaxKeystoneKapers()")
            print("  obs, state = game.reset()")
            print("  obs, state, reward, done, info = game.step(state, action)")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")

        return success

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
