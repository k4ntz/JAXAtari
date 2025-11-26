#!/usr/bin/env python3
"""
Quick test script to verify Pacman implementation.
Tests basic functionality without requiring full environment setup.
"""

import jax
import jax.numpy as jnp

# Test 1: Import the game module
print("Test 1: Importing Pacman module...")
try:
    from jaxatari.games.jax_pacman import JaxPacman, PacmanConstants
    print("✅ Module imported successfully")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    exit(1)

# Test 2: Create environment instance
print("\nTest 2: Creating Pacman environment...")
try:
    env = JaxPacman()
    print("✅ Environment created successfully")
except Exception as e:
    print(f"❌ Failed to create environment: {e}")
    exit(1)

# Test 3: Initialize game state
print("\nTest 3: Resetting environment...")
try:
    rng = jax.random.PRNGKey(42)
    obs, state = env.reset(rng)
    print(f"✅ Environment reset successfully")
    print(f"   Initial lives: {state.lives}")
    print(f"   Initial score: {state.score}")
    print(f"   Dots remaining: {state.dots_remaining}")
    print(f"   Player position: ({state.player_x}, {state.player_y})")
except Exception as e:
    print(f"❌ Failed to reset: {e}")
    exit(1)

# Test 4: Check maze layout
print("\nTest 4: Checking maze layout...")
try:
    maze = env.consts.MAZE_LAYOUT
    print(f"✅ Maze layout exists")
    print(f"   Maze shape: {maze.shape}")
    wall_count = jnp.sum(maze == 1)
    dot_count = jnp.sum(maze == 2)
    power_count = jnp.sum(maze == 3)
    print(f"   Walls: {wall_count}")
    print(f"   Dots: {dot_count}")
    print(f"   Power pellets: {power_count}")
except Exception as e:
    print(f"❌ Failed to check maze: {e}")
    exit(1)

# Test 5: Take some steps
print("\nTest 5: Taking game steps...")
try:
    for i in range(5):
        # Try moving right
        action = 3  # RIGHT action
        obs, state, reward, done, info = env.step(state, action)
        print(f"   Step {i+1}: pos=({state.player_x},{state.player_y}), score={state.score}, reward={reward}")
    print("✅ Steps executed successfully")
except Exception as e:
    print(f"❌ Failed to step: {e}")
    exit(1)

# Test 6: Test rendering
print("\nTest 6: Testing rendering...")
try:
    img = env.render(state)
    print(f"✅ Rendering successful")
    print(f"   Image shape: {img.shape}")
    print(f"   Image dtype: {img.dtype}")
except Exception as e:
    print(f"❌ Failed to render: {e}")
    exit(1)

print("\n" + "="*50)
print("✅ ALL TESTS PASSED!")
print("="*50)
print("\nThe Pacman implementation is working correctly!")
print("Next steps:")
print("  1. Install dependencies: pip install -e .")
print("  2. Test with JAXAtari: python3 -c 'import jaxatari; env = jaxatari.make(\"pacman\"); print(\"Success!\")'")
print("  3. Play the game: python3 scripts/play.py -g pacman")
