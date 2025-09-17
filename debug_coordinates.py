#!/usr/bin/env python3
"""Debug script to test coordinate transformations in BattleZone."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax.numpy as jnp
import math
from jaxatari.games.jax_battlezone import update_tank_position, Tank, create_bullet

def test_movement_coordinates():
    """Test that tank movement coordinates work as expected."""
    print("=== Testing Tank Movement Coordinates ===")
    
    # Create initial tank at origin
    tank = Tank(
        x=jnp.array(0.0),
        y=jnp.array(0.0), 
        angle=jnp.array(0.0),
        alive=jnp.array(1)
    )
    
    # Test each movement direction
    movements = [
        (3, "RIGHT"),  # RIGHT action
        (4, "LEFT"),   # LEFT action  
        (2, "UP"),     # UP action
        (5, "DOWN"),   # DOWN action
        (6, "UPRIGHT"), # UPRIGHT action
        (7, "UPLEFT"),  # UPLEFT action
        (8, "DOWNRIGHT"), # DOWNRIGHT action
        (9, "DOWNLEFT")  # DOWNLEFT action
    ]
    
    for action_id, action_name in movements:
        new_tank = update_tank_position(tank, jnp.array(action_id))
        print(f"{action_name:>10}: x={float(new_tank.x):6.2f}, y={float(new_tank.y):6.2f}, angle={float(new_tank.angle):6.2f} ({float(new_tank.angle) * 180 / math.pi:6.1f}°)")

def test_bullet_creation():
    """Test bullet creation and direction."""
    print("\n=== Testing Bullet Creation ===")
    
    # Test bullets for each facing direction
    angles = [
        (0.0, "RIGHT"),
        (math.pi/2, "DOWN"),
        (math.pi, "LEFT"),
        (-math.pi/2, "UP"),
        (math.pi/4, "DOWN-RIGHT"),
        (3*math.pi/4, "DOWN-LEFT"),
        (-math.pi/4, "UP-RIGHT"),
        (-3*math.pi/4, "UP-LEFT")
    ]
    
    for angle, direction in angles:
        tank = Tank(
            x=jnp.array(100.0),
            y=jnp.array(100.0),
            angle=jnp.array(angle),
            alive=jnp.array(1)
        )
        
        bullet = create_bullet(tank, jnp.array(0))
        print(f"{direction:>10}: spawn=({float(bullet.x):6.1f},{float(bullet.y):6.1f}), vel=({float(bullet.vel_x):5.1f},{float(bullet.vel_y):5.1f})")

def test_coordinate_system():
    """Test the overall coordinate system consistency."""
    print("\n=== Testing Coordinate System Consistency ===")
    
    # Create tank facing right
    tank_right = Tank(x=jnp.array(0.0), y=jnp.array(0.0), angle=jnp.array(0.0), alive=jnp.array(1))
    
    # Move right and create bullet
    moved_right = update_tank_position(tank_right, jnp.array(3))  # RIGHT
    bullet_right = create_bullet(moved_right, jnp.array(0))
    
    print(f"Tank facing RIGHT:")
    print(f"  Position after RIGHT move: ({float(moved_right.x)}, {float(moved_right.y)})")
    print(f"  Bullet velocity: ({float(bullet_right.vel_x)}, {float(bullet_right.vel_y)})")
    print(f"  Expected: positive X movement for both tank and bullet")
    
    # Create tank facing up  
    tank_up = Tank(x=jnp.array(0.0), y=jnp.array(0.0), angle=jnp.array(-math.pi/2), alive=jnp.array(1))
    
    # Move up and create bullet
    moved_up = update_tank_position(tank_up, jnp.array(2))  # UP
    bullet_up = create_bullet(moved_up, jnp.array(0))
    
    print(f"\nTank facing UP:")
    print(f"  Position after UP move: ({float(moved_up.x)}, {float(moved_up.y)})")
    print(f"  Bullet velocity: ({float(bullet_up.vel_x)}, {float(bullet_up.vel_y)})")
    print(f"  Expected: negative Y movement for both tank and bullet")

if __name__ == "__main__":
    test_movement_coordinates()
    test_bullet_creation()
    test_coordinate_system()
