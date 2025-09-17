#!/usr/bin/env python3
"""Debug script to test the world_to_screen_3d transformation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax.numpy as jnp
import math
import numpy as np

def debug_world_to_screen_3d(world_x, world_y, player_x, player_y, player_angle):
    """Debug version of world_to_screen_3d with detailed output."""
    print(f"\n=== Debug world_to_screen_3d ===")
    print(f"World position: ({world_x}, {world_y})")
    print(f"Player position: ({player_x}, {player_y})")
    print(f"Player angle: {player_angle} ({player_angle * 180 / math.pi:.1f}°)")
    
    # Translate to player-relative coordinates
    rel_x = world_x - player_x
    rel_y = world_y - player_y
    print(f"Relative position: ({rel_x}, {rel_y})")
    
    # Rotate by player angle to get view-relative coordinates
    cos_a = np.cos(player_angle)
    sin_a = np.sin(player_angle)
    print(f"cos(angle): {cos_a:.3f}, sin(angle): {sin_a:.3f}")
    
    # Current (incorrect) transformation
    view_x_current = rel_x * cos_a + rel_y * sin_a
    view_y_current = -rel_x * sin_a + rel_y * cos_a
    print(f"Current view space: ({view_x_current:.2f}, {view_y_current:.2f})")
    
    # Expected behavior analysis
    print(f"Expected behavior:")
    if player_angle == 0:  # Facing right
        print(f"  - Objects to the right should have positive view_y (forward)")
        print(f"  - Objects above should have negative view_x (left)")
        print(f"  - Objects below should have positive view_x (right)")
    elif player_angle == math.pi/2:  # Facing down
        print(f"  - Objects below should have positive view_y (forward)")
        print(f"  - Objects to the right should have negative view_x (left)")
        print(f"  - Objects to the left should have positive view_x (right)")
    
    return view_x_current, view_y_current

def test_coordinate_transformations():
    """Test various scenarios to understand the coordinate issue."""
    
    print("=== Testing Coordinate Transformations ===\n")
    
    # Test scenario 1: Player at origin, facing right, object to the right
    print("Scenario 1: Player at origin facing right, object to the right")
    debug_world_to_screen_3d(100, 0, 0, 0, 0)  # Object at (100,0), player at (0,0) facing right
    
    # Test scenario 2: Player at origin, facing right, object above
    print("\nScenario 2: Player at origin facing right, object above")
    debug_world_to_screen_3d(0, -100, 0, 0, 0)  # Object at (0,-100), player at (0,0) facing right
    
    # Test scenario 3: Player at origin, facing down, object below
    print("\nScenario 3: Player at origin facing down, object below")
    debug_world_to_screen_3d(0, 100, 0, 0, math.pi/2)  # Object at (0,100), player at (0,0) facing down
    
    # Test scenario 4: Player at origin, facing down, object to the right
    print("\nScenario 4: Player at origin facing down, object to the right")
    debug_world_to_screen_3d(100, 0, 0, 0, math.pi/2)  # Object at (100,0), player at (0,0) facing down

if __name__ == "__main__":
    test_coordinate_transformations()
