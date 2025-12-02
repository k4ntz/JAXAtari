"""
Pac-Man Test Suite

Runs verification tests for Pac-Man implementation.
Checks map layout, game logic, and physics.
"""

import jax
import jax.numpy as jnp
import pytest
import numpy as np
from jaxatari.games.jax_pacman import JaxPacman, PacmanConstants

@pytest.fixture
def env():
    return JaxPacman()

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)

def test_initialization(env, key):
    """
    Verifies initial game state.
    Checks lives, score, and starting position.
    """
    obs, state = env.reset(key)
    assert state.lives == 3
    assert state.score == 0
    # Check initial position (should be at start position)
    assert state.player_x > 0
    assert state.player_y > 0

def test_maze_layout(env):
    """
    Verifies maze layout properties.
    Checks for presence of walls, dots, power pellets, and ghost house.
    """
    maze = env.consts.MAZE_LAYOUT
    assert maze.shape == (25, 20)
    
    wall_count = jnp.sum(maze == 1)
    dot_count = jnp.sum(maze == 2)
    power_count = jnp.sum(maze == 3)
    
    assert wall_count > 0, "Maze should have walls"
    assert dot_count > 0, "Maze should have dots"
    assert power_count > 0, "Maze should have power pellets"
    
    # Check for ghost house tiles (4)
    house_count = jnp.sum(maze == 4)
    assert house_count > 0, "Maze should have ghost house tiles"
    
    # Check ghost house node index
    gh_idx = env.ghost_house_node_idx
    gh_x = env.node_positions_x[gh_idx]
    gh_y = env.node_positions_y[gh_idx]
    gh_tile_x = int(gh_x) // env.consts.TILE_SIZE
    gh_tile_y = int(gh_y) // env.consts.TILE_SIZE
    assert maze[gh_tile_y, gh_tile_x] == 4, f"Ghost house node should be on tile 4, got {maze[gh_tile_y, gh_tile_x]}"

def test_step_execution(env, key):
    """
    Verifies basic gameplay mechanics.
    Simulates steps to ensure state updates correctly.
    """
    obs, state = env.reset(key)
    initial_score = state.score
    
    # Take 5 steps moving right
    for _ in range(5):
        action = 3  # RIGHT
        obs, state, reward, done, info = env.step(state, action)
        
    # Score might change if we eat a dot
    assert state.score >= initial_score

def test_rendering(env, key):
    """
    Verifies renderer output.
    Ensures render function returns valid RGB array.
    """
    obs, state = env.reset(key)
    img = env.render(state)
    assert img.shape == (210, 160, 3)
    assert img.dtype == jnp.uint8

def test_ghost_wall_collision_strict(env, key):
    """
    Strict collision check.
    Verifies ghosts never overlap with wall tiles at any frame.
    """
    obs, state = env.reset(key)
    maze = env.consts.MAZE_LAYOUT
    
    # Run for a longer duration to ensure ghosts move around
    for step in range(200):
        # Use NOOP action
        obs, state, reward, done, info = env.step(state, 0)
        ghosts = state.ghosts
        
        for i in range(4):
            gx, gy = ghosts[i, 0], ghosts[i, 1]
            
            # Check bounding box (assuming 8x8 ghost size)
            # We check the 4 corners of the ghost
            corners = [
                (gx, gy),           # Top-left
                (gx + 7, gy),       # Top-right
                (gx, gy + 7),       # Bottom-left
                (gx + 7, gy + 7)    # Bottom-right
            ]
            
            for cx, cy in corners:
                tx, ty = int(cx // 8), int(cy // 8)
                
                # Boundary check
                if tx < 0 or tx >= 20 or ty < 0 or ty >= 25:
                    continue # Out of bounds is handled separately or allowed in tunnel
                    
                # Check if inside wall (1)
                is_wall = maze[ty, tx] == 1
                if is_wall:
                    print(f"FAILURE at step {step}: Ghost {i} at ({gx}, {gy}) overlaps wall at tile ({tx}, {ty})")
                    print(f"Ghost State: {ghosts[i]}")
                
                assert not is_wall, f"Ghost {i} at ({gx}, {gy}) overlaps wall at tile ({tx}, {ty}) during step {step}"

def test_ghost_revival(env, key):
    """
    Verifies ghost revival logic.
    Ensures eaten ghosts return to ghost house and reset state.
    """
    obs, state = env.reset(key)
    
    # Manually set a ghost to eaten state (2)
    eaten_ghost_idx = 0
    ghosts = state.ghosts
    ghosts = ghosts.at[eaten_ghost_idx, 3].set(2)
    state = state._replace(ghosts=ghosts)
    
    # Verify ghost is eaten
    assert state.ghosts[eaten_ghost_idx, 3] == 2
    
    # Step until revived (or timeout)
    revived = False
    for _ in range(500): # Allow enough steps to return to house
        # Use NOOP action
        obs, state, reward, done, info = env.step(state, 0)
        
        # Check state
        current_state = state.ghosts[eaten_ghost_idx, 3]
        if current_state == 0:
            revived = True
            break
    
    assert revived, f"Ghost {eaten_ghost_idx} failed to revive after 500 steps"

if __name__ == "__main__":
    pytest.main([__file__])
