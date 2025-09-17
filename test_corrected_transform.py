#!/usr/bin/env python3
"""Test the corrected world_to_screen_3d transformation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax.numpy as jnp
import math
import numpy as np

def corrected_world_to_screen_3d(world_x, world_y, player_x, player_y, player_angle):
    """Corrected version of world_to_screen_3d with detailed output."""
    print(f"\n=== Corrected world_to_screen_3d ===")
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
    
    # CORRECTED transformation
    view_x = -rel_x * sin_a + rel_y * cos_a   # Right/left relative to player
    view_y = rel_x * cos_a + rel_y * sin_a    # Forward/back relative to player
    print(f"Corrected view space: ({view_x:.2f}, {view_y:.2f})")
    
    # Verify correctness
    print(f"Verification:")
    if view_y > 0:
        print(f"  ✓ Object is in front of player (view_y > 0)")
    else:
        print(f"  ✗ Object is behind player (view_y <= 0)")
    
    if view_x > 0:
        print(f"  → Object appears to the RIGHT of player's view")
    elif view_x < 0:
        print(f"  → Object appears to the LEFT of player's view")
    else:
        print(f"  → Object is directly ahead")
    
    return view_x, view_y

def test_movement_scenarios():
    """Test how objects move when player moves in different directions."""
    
    print("=== Testing Movement Scenarios ===\n")
    
    # Fixed object position
    obj_x, obj_y = 100, 100
    
    print("Testing with fixed obstacle at (100, 100):")
    
    # Test 1: Player moves right
    print("\n1. Player moves RIGHT (0° → 0°, position 0→10)")
    corrected_world_to_screen_3d(obj_x, obj_y, 0, 100, 0)  # Player at (0,100) facing right
    corrected_world_to_screen_3d(obj_x, obj_y, 10, 100, 0)  # Player moved to (10,100) facing right
    print("Expected: Object should move LEFT in view (negative view_x change)")
    
    # Test 2: Player moves left  
    print("\n2. Player moves LEFT (0° → 0°, position 10→0)")
    corrected_world_to_screen_3d(obj_x, obj_y, 10, 100, 0)  # Player at (10,100) facing right
    corrected_world_to_screen_3d(obj_x, obj_y, 0, 100, 0)   # Player moved to (0,100) facing right
    print("Expected: Object should move RIGHT in view (positive view_x change)")
    
    # Test 3: Player moves forward (right direction)
    print("\n3. Player moves FORWARD (facing right: 0°, position x-axis)")
    corrected_world_to_screen_3d(obj_x, obj_y, 90, 100, 0)  # Player at (90,100) facing right  
    corrected_world_to_screen_3d(obj_x, obj_y, 100, 100, 0) # Player moved to (100,100) facing right
    print("Expected: Object should move AWAY (smaller view_y)")
    
    # Test 4: Player moves backward (right direction)
    print("\n4. Player moves BACKWARD (facing right: 0°, position x-axis)")
    corrected_world_to_screen_3d(obj_x, obj_y, 100, 100, 0) # Player at (100,100) facing right
    corrected_world_to_screen_3d(obj_x, obj_y, 90, 100, 0)  # Player moved to (90,100) facing right  
    print("Expected: Object should move CLOSER (larger view_y)")

def test_direction_changes():
    """Test what happens when player changes direction."""
    
    print("\n\n=== Testing Direction Changes ===\n")
    
    # Fixed object and player position
    obj_x, obj_y = 100, 50
    player_x, player_y = 100, 100
    
    print(f"Fixed setup: Object at ({obj_x}, {obj_y}), Player at ({player_x}, {player_y})")
    
    # Test different facing directions
    directions = [
        (0, "RIGHT"),
        (math.pi/2, "DOWN"), 
        (math.pi, "LEFT"),
        (-math.pi/2, "UP")
    ]
    
    for angle, direction in directions:
        print(f"\nPlayer facing {direction} ({angle:.2f} rad):")
        view_x, view_y = corrected_world_to_screen_3d(obj_x, obj_y, player_x, player_y, angle)
        
        if view_y > 0:
            print(f"  Object is visible in front")
        else:
            print(f"  Object is behind player (not visible)")

if __name__ == "__main__":
    test_movement_scenarios()
    test_direction_changes()
