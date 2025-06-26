
# import os
# from functools import partial
# from typing import NamedTuple, Tuple
# import jax.lax
# import jax.numpy as jnp
# import chex
# import pygame
# from gymnax.environments import spaces

# from jaxatari.rendering import atraJaxis as aj
# from jaxatari.environment import JaxEnvironment

# """
# Things that have been implemented:
# - Player movement
# - Player shooting
# - Enemy movement
# - Enemy shooting
# - Enemy splitting and collision detection
# - Mothership movement
# - Player lives and score tracking
# - Stage progression
# - Enemy spawning

# Things that need to be implemented:
# Emanuele
# - Enemy random direction changes -> done
# - Enemy invisibility -> done
# - Stop spawning enemies when 10 have been spawned -> done
# - Delays for enemy projectile firing -> done

# Milan
# - Heat tracking [x]
# - Setting correct constants for the game
# - Game over conditions -> done
# - Game accurate player projectile movement

# Things that "should" be implemented:
# - Use arrays for enemy positions and states instead of individual variables

# Things that could be implemented:
# - loading more sprites to be more similar to the original game
# - Transition to linking enemy y positions to enemy_index
# """

# WIDTH = 160
# HEIGHT = 210

# NOOP = 0
# FIRE = 1
# RIGHT = 2
# LEFT = 3
# RIGHTFIRE = 4
# LEFTFIRE = 5

# SPEED = 1
# MOTHERSHIP_Y = 32
# PLAYER_Y = 175
# MAX_HEAT = 15
# COOLDOWN_STEPS = 30
# MAX_LIVES = 3
# LIVES_Y = 200
# LIFE_ONE_X = 25
# LIFE_OFFSET = 20

# ENEMY_Y_POSITIONS = (64, 96, 128)

# PLAYER_SIZE = (8, 8)
# ENEMY_SIZE = (16, 8)
# Y_STEP_DELAY = 70
# MOTHERSHIP_SIZE = (32, 16)

# WINDOW_WIDTH = 160 * 3
# WINDOW_HEIGHT = 210 * 3

# STATE_TRANSLATOR: dict = {
#     0: "player_x",
#     1: "player_speed",
#     2: "enemy_projectile_x",
#     3: "enemy_projectile_y",
#     4: "enemy_projectile_dir",
#     5: "mothership_x",
#     6: "enemy_1_x",
#     7: "enemy_1_y",
#     8: "enemy_1_speed",
#     9: "enemy_1_split",
#     10: "enemy_2_x",
#     11: "enemy_2_y",
#     12: "enemy_2_speed",
#     13: "enemy_2_split",
#     14: "enemy_3_x",
#     15: "enemy_3_y",
#     16: "enemy_3_speed",
#     17: "enemy_3_split",
#     18: "enemy_4_x",
#     19: "enemy_4_y",
#     20: "enemy_4_speed",
#     21: "enemy_5_x",
#     22: "enemy_5_y",
#     23: "enemy_5_speed",
#     24: "enemy_6_x",
#     25: "enemy_6_y",
#     26: "enemy_6_speed",
#     27: "player_projectile_x",
#     28: "player_projectile_y",
#     29: "player_projectile_dir",
#     30: "score",
#     31: "player_lives",
#     32: "heat",
#     33: "stage",
#     34: "buffer",
# }

# def get_human_action() -> chex.Array:
#     """
#     Records if UP or DOWN is being pressed and returns the corresponding action.

#     Returns:
#         action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
#     """
#     keys = pygame.key.get_pressed()
#     if keys[pygame.K_a] and keys[pygame.K_SPACE]:
#         return jnp.array(LEFTFIRE)
#     elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
#         return jnp.array(RIGHTFIRE)
#     elif keys[pygame.K_a]:
#         return jnp.array(LEFT)
#     elif keys[pygame.K_d]:
#         return jnp.array(RIGHT)
#     elif keys[pygame.K_SPACE]:
#         return jnp.array(FIRE)
#     else:
#         return jnp.array(NOOP)

# class EntityPosition(NamedTuple):
#     x: jnp.ndarray
#     y: jnp.ndarray
#     width: jnp.ndarray
#     height: jnp.ndarray
#     invisible: jnp.ndarray

# # immutable state container
# class AssaultState(NamedTuple):
#     player_x: chex.Array
#     player_speed: chex.Array
#     enemy_projectile_x: chex.Array
#     enemy_projectile_y: chex.Array
#     enemy_projectile_dir: chex.Array
#     mothership_x: chex.Array
#     mothership_dir: chex.Array
#     enemy_1_x: chex.Array
#     enemy_1_y: chex.Array
#     enemy_1_dir: chex.Array
#     enemy_1_split: chex.Array
#     enemy_2_x: chex.Array
#     enemy_2_y: chex.Array
#     enemy_2_dir: chex.Array
#     enemy_2_split: chex.Array
#     enemy_3_x: chex.Array
#     enemy_3_y: chex.Array
#     enemy_3_dir: chex.Array
#     enemy_3_split: chex.Array
#     enemy_4_x: chex.Array
#     enemy_4_y: chex.Array
#     enemy_4_dir: chex.Array
#     enemy_5_x: chex.Array
#     enemy_5_y: chex.Array
#     enemy_5_dir: chex.Array
#     enemy_6_x: chex.Array
#     enemy_6_y: chex.Array
#     enemy_6_dir: chex.Array
#     player_projectile_x: chex.Array
#     player_projectile_y: chex.Array
#     player_projectile_dir: chex.Array
#     score: chex.Array
#     player_lives: chex.Array
#     heat: chex.Array
#     stage: chex.Array
#     buffer: chex.Array
#     obs_stack: chex.ArrayTree
#     occupied_y: chex.Array
#     step_counter: chex.Array
#     enemies_killed: chex.Array
#     current_stage:  chex.Array
#     enemies_spawned_this_stage: chex.Array
#     enemies_invisible: chex.Array
#     cooldown: chex.Array
#     fired: chex.Array


# class AssaultObservation(NamedTuple):
#     player: EntityPosition
#     mothership: EntityPosition
#     enemy_1: EntityPosition
#     enemy_2: EntityPosition
#     enemy_3: EntityPosition
#     enemy_4: EntityPosition
#     enemy_5: EntityPosition
#     enemy_6: EntityPosition
#     enemy_projectile: EntityPosition
#     lives: jnp.ndarray
#     score: jnp.ndarray
    


# class AssaultInfo(NamedTuple):
#     time: jnp.ndarray
#     all_rewards: chex.Array

# @jax.jit
# def player_step(
#     state, action: chex.Array
# ):
#     # Minimal: move left/right, clamp to screen
#     move_left = jnp.logical_or(action == LEFT, action == LEFTFIRE)
#     move_right = jnp.logical_or(action == RIGHT, action == RIGHTFIRE)
#     speed = jnp.where(move_left, -SPEED, jnp.where(move_right, SPEED, 0))
#     new_x = jnp.clip(state.player_x + speed, 0, 160 - int(PLAYER_SIZE[0]/2))
#     return state._replace(
#         player_x=new_x,
#         player_speed=speed
#     )

# @jax.jit
# def player_projectile_step(
#     state, action: chex.Array
# ):
#     # If projectile is inactive, check for fire action to spawn it
#     fire_up_action = action == FIRE
#     fire_left_action = action == LEFTFIRE
#     fire_right_action = action == RIGHTFIRE
#     fire_action = fire_up_action + fire_left_action * 2 + fire_right_action * 3
#     can_fire = state.player_projectile_y < 0
#     fire_action = fire_action * can_fire
#     spawn_proj = jnp.logical_and(fire_action>0, can_fire)
#     # Spawn at player's current x, just above the player
#     spawn_x = jnp.array([state.player_projectile_x, state.player_x, state.player_x + 12, state.player_x - 4])
#     spawn_y = jnp.array([state.player_projectile_y, PLAYER_Y - 4, PLAYER_Y, PLAYER_Y])
#     new_fired = jnp.where(spawn_proj, jnp.array(1), state.fired)
#     new_proj_x = spawn_x[fire_action]
#     new_proj_y = spawn_y[fire_action]
#     new_proj_dir = jnp.where(spawn_proj, fire_action, state.player_projectile_dir)
#     # Move projectile if active
#     moving_y = jnp.logical_and(new_proj_y >= 0, new_proj_dir == 1)
#     moving_x = jnp.logical_and.reduce(jnp.array([new_proj_x >= 0, new_proj_x <= WIDTH, new_proj_dir > 1]))
#     moved_proj_y = jnp.where(moving_y, new_proj_y - 6, new_proj_y)
#     x_dir = jnp.where(new_proj_dir == 2, -1, 1)
#     moved_proj_x = jnp.where(moving_x, new_proj_x + x_dir * 6, new_proj_x)
#     # Deactivate if off screen
#     out_of_bounds = jnp.logical_or.reduce(jnp.array([moved_proj_y < 0,moved_proj_x < 0, moved_proj_x > WIDTH]))
#     final_proj_y = jnp.where(out_of_bounds, jnp.array(-1), moved_proj_y)
#     final_proj_x = jnp.where(out_of_bounds, jnp.array(-1), moved_proj_x)
#     final_proj_dir = jnp.where(out_of_bounds, jnp.array(0), new_proj_dir)
#     return state._replace(
#         player_projectile_x=final_proj_x,
#         player_projectile_y=final_proj_y,
#         player_projectile_dir=final_proj_dir,
#         fired=new_fired
#     )
    
# @jax.jit
# def cooldown_step(
#     state
# ):
#     # Cooldown logic: if heat is above 0, decrease cooldown
#     new_cooldown = jnp.where(state.cooldown > 0, state.cooldown - 1, state.cooldown)
#     # If cooldown is 0 and heat is above 0, reset heat
#     new_heat = jnp.where(jnp.logical_and(state.fired,new_cooldown == 0), state.heat +1, state.heat)
#     new_heat = jnp.where(jnp.logical_and(jnp.logical_not(state.fired),new_cooldown == 0), jnp.maximum(state.heat-1,0), new_heat)
#     new_fired = jnp.where(new_cooldown == 0, jnp.array(0), state.fired)
#     overheat = new_heat > MAX_HEAT
#     # If overheat, reset heat and set cooldown to COOLDOWN_STEPS
#     new_heat = jnp.where(overheat, jnp.array(0), new_heat)
#     new_lives = jnp.where(overheat, jnp.maximum(state.player_lives - 1, 0), state.player_lives)
#     new_cooldown = jnp.where(new_cooldown==0, COOLDOWN_STEPS, new_cooldown)
#     return state._replace(
#         heat=new_heat,
#         cooldown=new_cooldown,
#         player_lives=new_lives,
#         fired=new_fired
#     )    

# @jax.jit
# def enemy_projectile_step(
#     state
# ):
#     # If projectile is inactive, check for random fire opportunity
#     can_fire = state.enemy_projectile_y < 0
    
#     # Random chance of firing (1% probability)
#     # Note: In a complete implementation, you would use a proper PRNG key
#     fire_action = jnp.equal(jnp.mod(state.step_counter, 180), 0)
#     #jax.debug.print(f"Can Fire : {can_fire}, Random fire chance: {fire_random}, Fire action: {fire_action}")
#     spawn_proj = jnp.logical_and(fire_action, can_fire)
    
#     # Initialize with default values
#     chosen_enemy_x = jnp.array(0)
#     chosen_enemy_y = jnp.array(-1)  # Default value if no enemies are active
    
#     # Check each enemy individually and keep track of the one with largest y
#     # Enemy 1
#     e1_active = state.enemy_1_y < HEIGHT
#     is_better = jnp.logical_and(e1_active, state.enemy_1_y > chosen_enemy_y)
#     chosen_enemy_x = jnp.where(is_better, state.enemy_1_x, chosen_enemy_x)
#     chosen_enemy_y = jnp.where(is_better, state.enemy_1_y, chosen_enemy_y)
    
#     # Enemy 2
#     e2_active = state.enemy_2_y < HEIGHT
#     is_better = jnp.logical_and(e2_active, state.enemy_2_y > chosen_enemy_y)
#     chosen_enemy_x = jnp.where(is_better, state.enemy_2_x, chosen_enemy_x)
#     chosen_enemy_y = jnp.where(is_better, state.enemy_2_y, chosen_enemy_y)
    
#     # Enemy 3
#     e3_active = state.enemy_3_y < HEIGHT
#     is_better = jnp.logical_and(e3_active, state.enemy_3_y > chosen_enemy_y)
#     chosen_enemy_x = jnp.where(is_better, state.enemy_3_x, chosen_enemy_x)
#     chosen_enemy_y = jnp.where(is_better, state.enemy_3_y, chosen_enemy_y)
    
#     # Enemy 4
#     e4_active = state.enemy_4_y < HEIGHT
#     is_better = jnp.logical_and(e4_active, state.enemy_4_y > chosen_enemy_y)
#     chosen_enemy_x = jnp.where(is_better, state.enemy_4_x, chosen_enemy_x)
#     chosen_enemy_y = jnp.where(is_better, state.enemy_4_y, chosen_enemy_y)
    
#     # Enemy 5
#     e5_active = state.enemy_5_y < HEIGHT
#     is_better = jnp.logical_and(e5_active, state.enemy_5_y > chosen_enemy_y)
#     chosen_enemy_x = jnp.where(is_better, state.enemy_5_x, chosen_enemy_x)
#     chosen_enemy_y = jnp.where(is_better, state.enemy_5_y, chosen_enemy_y)
    
#     # Enemy 6
#     e6_active = state.enemy_6_y < HEIGHT
#     is_better = jnp.logical_and(e6_active, state.enemy_6_y > chosen_enemy_y)
#     chosen_enemy_x = jnp.where(is_better, state.enemy_6_x, chosen_enemy_x)
#     chosen_enemy_y = jnp.where(is_better, state.enemy_6_y, chosen_enemy_y)
    
#     # Only spawn projectile if we have an active enemy
#     has_active_enemy = chosen_enemy_y >= 0
#     effective_spawn = jnp.logical_and(spawn_proj, has_active_enemy)
    
#     # Spawn projectile at the chosen enemy's position
#     new_proj_x = jnp.where(effective_spawn, chosen_enemy_x + ENEMY_SIZE[0]//2, state.enemy_projectile_x)
#     new_proj_y = jnp.where(effective_spawn, chosen_enemy_y + ENEMY_SIZE[1], state.enemy_projectile_y)
#     new_proj_dir = jnp.where(effective_spawn, 1, state.enemy_projectile_dir)  # Moving down
    
#     # Move projectile if active
#     moving = new_proj_y >= 0
#      # Check if this is a special stage (every third stage: 2, 5, 8, etc.)
#     # Since current_stage is 0-indexed (0=stage 1, 1=stage 2), we need stages where (current_stage + 1) % 3 == 2
#     is_special_stage = jnp.equal(jnp.mod(state.current_stage + 1, 4), 0)
    
#     # Check if projectile is near player's y-level
#     near_player_level = jnp.logical_and(
#         new_proj_y >= PLAYER_Y - 5,  # Within 20 pixels above player
#         new_proj_y <= PLAYER_Y + 5       # Not past player
#     )
    
#     # Should we apply horizontal tracking?
#     should_track = jnp.logical_and(is_special_stage, near_player_level)
    
#     # Calculate direction toward player (left=-1, right=1)
#     player_direction = jnp.sign(state.player_x - new_proj_x)
    
#     # Apply vertical movement
#     moved_proj_y = jnp.where(jnp.logical_and(moving,should_track), PLAYER_Y, new_proj_y + new_proj_dir * 4)
    
#     # Apply horizontal movement only if tracking is enabled
#     moved_proj_x = jnp.where(
#         jnp.logical_and(should_track, moving),
#         new_proj_x + player_direction * 2,  # Move 2 pixels toward player
#         new_proj_x
#     )
    
#     # Deactivate if off screen (below screen height)
#     final_proj_y = jnp.where(moved_proj_y > HEIGHT, -1, moved_proj_y)
#     final_proj_x = jnp.where(moved_proj_y > HEIGHT, -1, moved_proj_x)
#     final_proj_dir = jnp.where(moved_proj_y > HEIGHT, 0, new_proj_dir)
    
#     # Check for collision with player 
#     player_hit = check_collision(
#         final_proj_x, final_proj_y, 
#         state.player_x, PLAYER_Y, 
#         PLAYER_SIZE[0], PLAYER_SIZE[1]
#     )
    
#     # If hit player, deactivate projectile and reduce player lives
#     final_proj_y = jnp.where(player_hit, -1, final_proj_y)
#     final_proj_x = jnp.where(player_hit, -1, final_proj_x)
#     final_proj_dir = jnp.where(player_hit, 0, final_proj_dir)
#     new_lives = jnp.where(player_hit, state.player_lives - 1, state.player_lives)
    
#     return state._replace(
#         enemy_projectile_x=final_proj_x,
#         enemy_projectile_y=final_proj_y,
#         enemy_projectile_dir=final_proj_dir,
#         player_lives=new_lives
#     )
    
    


# @jax.jit
# def enemy_step(state):     
#     occupied_y = state.occupied_y
#     # Track if any enemy has moved down this frame
#     has_moved_down = jnp.array(0)

#     allow_y_movement = jnp.equal(jnp.mod(state.step_counter, 70), 0)
    
#     def move_enemy_x(x, dir, linked_enemy_x=WIDTH+1):
#         # If at left border, go right; if at right border, go left
#         at_left = jnp.greater_equal(0, x)
#         at_right = jnp.greater_equal(x, 160 - int(ENEMY_SIZE[0]/2))
#         new_dir = jnp.where(at_left, 1, jnp.where(at_right, -1, dir))

#         # check for linked enemy collision
#         collision = jnp.logical_not(jnp.logical_or(x > linked_enemy_x + int(ENEMY_SIZE[0]/2), x < linked_enemy_x - int(ENEMY_SIZE[0]/2)))
#         # If collision, reverse direction
#         new_dir = jnp.where(collision, -new_dir, new_dir)

#         # Randomly reverse direction
#         check_interval = 50 + jnp.mod(x * 7, 50)  # Enemy 1: every 32 frames, Enemy 2: every 39 frames, etc.
#         should_check_random = jnp.equal(jnp.mod(state.step_counter, check_interval), 0)
#         random_seed = jnp.mod(state.step_counter + x, 10000)
#         random_chance = jax.random.uniform(jax.random.PRNGKey(random_seed), shape=())
#         should_reverse = jnp.logical_and(should_check_random, random_chance < 0.5)  # Higher chance but less frequent


#         can_reverse = jnp.logical_not(jnp.logical_or.reduce(jnp.array([at_left, at_right, collision])))
#         new_dir = jnp.where(jnp.logical_and(should_reverse, can_reverse), -new_dir, new_dir)
#         new_x = jnp.clip(x + new_dir * SPEED, 0, 160 - int(ENEMY_SIZE[0]/2))
#         return new_x, new_dir

#     def move_enemy_y(y, occupied_y, has_moved, linked_enemy_lives=False):
#         # Check if enemy is inactive (outside screen)
#         is_inactive = jnp.greater_equal(y, HEIGHT)
        
#         # Check if it's at one of the defined row positions
#         matches = jnp.array(ENEMY_Y_POSITIONS) == y
#         has_match = jnp.any(matches)
#         idx = jnp.argmax(matches)  # Returns 0 if no match
        
#         can_spawn_more = jnp.less(state.enemies_spawned_this_stage, 10)


#         # Determine which action to take - only move down if no other enemy has moved
#         should_spawn = jnp.logical_and.reduce(jnp.array([
#             jnp.logical_and(is_inactive, occupied_y[0] == 0),
#             jnp.logical_and(has_moved == 0, allow_y_movement),
#             jnp.logical_not(linked_enemy_lives),
#             can_spawn_more
#         ]))
        
#         should_move_down = jnp.logical_and(
#             jnp.logical_and(
#                 has_match, 
#                 jnp.logical_and(idx < 2, occupied_y[idx + 1] == 0)
#             ),
#             jnp.logical_and(has_moved == 0, allow_y_movement)
#         )
        
#         # Define actions as separate functions
#         def spawn():
#             new_occupied = occupied_y.at[0].set(1)
#             # No downward movement occurred
#             # Return the random x as well
#             return ENEMY_Y_POSITIONS[0], new_occupied, jnp.array(1), jnp.array(1)
        
#         def move_down():
#             # Clear current row, mark next row
#             new_occupied = occupied_y.at[idx].set(0).at[idx + 1].set(1)
#             # Use jnp.take instead of direct indexing for JAX compatibility
#             next_position = jnp.take(jnp.array(ENEMY_Y_POSITIONS), idx + 1)
#             # Mark that a movement has occurred
#             # Return -1 as x to indicate no x change needed
#             return next_position, new_occupied, jnp.array(1), jnp.array(-1)
        
#         def no_change():
#             # Return -1 as x to indicate no x change needed
#             return y, occupied_y, has_moved, jnp.array(-1)
        
#         # Apply the appropriate action
#         result = jax.lax.cond(
#             should_spawn,
#             lambda _: spawn(),
#             lambda _: jax.lax.cond(
#                 should_move_down,
#                 lambda _: move_down(),
#                 lambda _: no_change(),
#                 operand=None
#             ),
#             operand=None
#         )
        
#         return result
    
#     e1_x, e1_dir = move_enemy_x(state.enemy_1_x, state.enemy_1_dir, jnp.where(state.enemy_4_y <= HEIGHT, state.enemy_4_x, WIDTH+1))
#     e2_x, e2_dir = move_enemy_x(state.enemy_2_x, state.enemy_2_dir, jnp.where(state.enemy_5_y <= HEIGHT, state.enemy_5_x, WIDTH+1))
#     e3_x, e3_dir = move_enemy_x(state.enemy_3_x, state.enemy_3_dir, jnp.where(state.enemy_6_y <= HEIGHT, state.enemy_6_x, WIDTH+1))
#     e4_x, e4_dir = move_enemy_x(state.enemy_4_x, state.enemy_4_dir, jnp.where(state.enemy_1_y <= HEIGHT, state.enemy_1_x, WIDTH+1))
#     e5_x, e5_dir = move_enemy_x(state.enemy_5_x, state.enemy_5_dir, jnp.where(state.enemy_2_y <= HEIGHT, state.enemy_2_x, WIDTH+1))
#     e6_x, e6_dir = move_enemy_x(state.enemy_6_x, state.enemy_6_dir, jnp.where(state.enemy_3_y <= HEIGHT, state.enemy_3_x, WIDTH+1))

#     # Pass and update the has_moved_down flag for each enemy
#     e1_y, occupied_y, has_moved_down, has_spawned = move_enemy_y(state.enemy_1_y, occupied_y, has_moved_down, jnp.less_equal(state.enemy_4_y, HEIGHT))
#     # Update e1_x with random_x if needed
#     e1_x = jnp.where(has_spawned >= 0, state.mothership_x, e1_x)
#     e1_split = jnp.where(has_spawned == 1, 0, state.enemy_1_split)
#     enemy_1_spawned = jnp.where(has_spawned == 1, 1, 0) # Count spawns for enemy 1

#     e2_y, occupied_y, has_moved_down, has_spawned = move_enemy_y(state.enemy_2_y, occupied_y, has_moved_down, jnp.less_equal(state.enemy_5_y, HEIGHT))
#     e2_x = jnp.where(has_spawned >= 0, state.mothership_x, e2_x)
#     e2_split = jnp.where(has_spawned == 1, 0, state.enemy_2_split)
#     enemy_2_spawned = jnp.where(has_spawned == 1, 1, 0) # Count spawns for enemy 2


#     e3_y, occupied_y, has_moved_down, has_spawned = move_enemy_y(state.enemy_3_y, occupied_y, has_moved_down, jnp.less_equal(state.enemy_6_y, HEIGHT))
#     e3_x = jnp.where(has_spawned >= 0, state.mothership_x, e3_x)
#     e3_split = jnp.where(has_spawned == 1, 0, state.enemy_3_split)
#     enemy_3_spawned = jnp.where(has_spawned == 1, 1, 0) # Count spawns for enemy 3

#     total_spawned_this_frame = enemy_1_spawned + enemy_2_spawned + enemy_3_spawned
#     new_enemies_spawned_this_stage = state.enemies_spawned_this_stage + total_spawned_this_frame

    
#     e4_y, occupied_y, has_moved_down, has_spawned = move_enemy_y(state.enemy_4_y, occupied_y, has_moved_down, True)
#     #e4_x = jnp.where(has_spawned >= 0, state.mothership_x, e4_x)
#     e4_y = jnp.where(jnp.logical_and(state.enemy_4_y < HEIGHT+1, e1_y < HEIGHT+1), e1_y, e4_y)

#     e5_y, occupied_y, has_moved_down, has_spawned = move_enemy_y(state.enemy_5_y, occupied_y, has_moved_down, True)
#     #e5_x = jnp.where(has_spawned >= 0, state.mothership_x, e5_x)
#     e5_y = jnp.where(jnp.logical_and(state.enemy_5_y < HEIGHT+1, e2_y < HEIGHT+1), e2_y, e5_y)

#     e6_y, occupied_y, has_moved_down, has_spawned = move_enemy_y(state.enemy_6_y, occupied_y, has_moved_down, True)
#     #e6_x = jnp.where(has_spawned >= 0, state.mothership_x, e6_x)
#     e6_y = jnp.where(jnp.logical_and(state.enemy_6_y < HEIGHT+1, e3_y < HEIGHT+1), e3_y, e6_y)

#     return state._replace(
#         enemy_1_x=e1_x, enemy_1_y=e1_y, enemy_1_dir=e1_dir,
#         enemy_2_x=e2_x, enemy_2_y=e2_y, enemy_2_dir=e2_dir,
#         enemy_3_x=e3_x, enemy_3_y=e3_y, enemy_3_dir=e3_dir,
#         enemy_4_x=e4_x, enemy_4_y=e4_y, enemy_4_dir=e4_dir,
#         enemy_5_x=e5_x, enemy_5_y=e5_y, enemy_5_dir=e5_dir,
#         enemy_6_x=e6_x, enemy_6_y=e6_y, enemy_6_dir=e6_dir,
#         enemy_1_split=e1_split, enemy_2_split=e2_split, enemy_3_split=e3_split,
#         occupied_y=occupied_y,
#         enemies_spawned_this_stage=new_enemies_spawned_this_stage  # Update spawn count here
#     )

# @jax.jit
# def mothership_step(state):
#     def move_mothership(x, dir):
#         # If at left border, go right; if at right border, go left
#         at_left = jnp.greater_equal(0, x)
#         at_right = jnp.greater_equal(x, 160 - MOTHERSHIP_SIZE[0])
#         new_dir = jnp.where(at_left, 1, jnp.where(at_right, -1, dir))
#         new_x = jnp.clip(x + new_dir * SPEED, 0, 160 - MOTHERSHIP_SIZE[0])
#         return new_x, new_dir

#     # Move mothership left/right, clamp to screen
#     mothership_x, mothership_dir = move_mothership(state.mothership_x, state.mothership_dir)
#     return state._replace(mothership_x=mothership_x, mothership_dir=mothership_dir)


# @jax.jit
# def check_collision(px, py, ex, ey, ew, eh):
#     # Returns True if (px, py) is inside the enemy box
#     return jnp.logical_and(
#         jnp.logical_and(px >= ex, px < ex + ew),
#         jnp.logical_and(py >= ey, py < ey + eh)
#     )

# class JaxAssault(JaxEnvironment[AssaultState, AssaultObservation, AssaultInfo]):

#     def __init__(self):
#         super().__init__()
#         self.frameskip = 1
#         self.frame_stack_size = 4
#         self.action_set = {NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE}
#         self.reward_funcs = None
#         self.occupied_y = jnp.array([0, 0, 0])

#     def reset(self) -> AssaultState:
#         # Minimal state initialization
#         state = AssaultState(
#             player_x=jnp.array(80).astype(jnp.int32),
#             player_speed=jnp.array(0).astype(jnp.int32),
#             enemy_projectile_x=jnp.array(0).astype(jnp.int32),
#             enemy_projectile_y=jnp.array(-1).astype(jnp.int32),
#             enemy_projectile_dir=jnp.array(0).astype(jnp.int32),
#             mothership_x=jnp.array(64).astype(jnp.int32),
#             mothership_dir=jnp.array(1).astype(jnp.int32),
#             enemy_1_x=jnp.array(-1).astype(jnp.int32),
#             enemy_1_y=jnp.array(HEIGHT+1).astype(jnp.int32),
#             enemy_1_dir=jnp.array(1).astype(jnp.int32),
#             enemy_1_split=jnp.array(0).astype(jnp.int32),
#             enemy_2_x=jnp.array(-1).astype(jnp.int32),
#             enemy_2_y=jnp.array(HEIGHT+1).astype(jnp.int32),
#             enemy_2_dir=jnp.array(1).astype(jnp.int32),
#             enemy_2_split=jnp.array(0).astype(jnp.int32),
#             enemy_3_x=jnp.array(-1).astype(jnp.int32),
#             enemy_3_y=jnp.array(HEIGHT+1).astype(jnp.int32),
#             enemy_3_dir=jnp.array(1).astype(jnp.int32),
#             enemy_3_split=jnp.array(0).astype(jnp.int32),
#             enemy_4_x=jnp.array(-1).astype(jnp.int32),
#             enemy_4_y=jnp.array(HEIGHT+1).astype(jnp.int32),
#             enemy_4_dir=jnp.array(1).astype(jnp.int32),
#             enemy_5_x=jnp.array(-1).astype(jnp.int32),
#             enemy_5_y=jnp.array(HEIGHT+1).astype(jnp.int32),
#             enemy_5_dir=jnp.array(1).astype(jnp.int32),
#             enemy_6_x=jnp.array(-1).astype(jnp.int32),
#             enemy_6_y=jnp.array(HEIGHT+1).astype(jnp.int32),
#             enemy_6_dir=jnp.array(1).astype(jnp.int32),
#             player_projectile_x=jnp.array(-1).astype(jnp.int32),
#             player_projectile_y=jnp.array(-1).astype(jnp.int32),
#             player_projectile_dir=jnp.array(0).astype(jnp.int32),
#             score=jnp.array(0).astype(jnp.int32),
#             player_lives=jnp.array(MAX_LIVES).astype(jnp.int32),
#             heat=jnp.array(0).astype(jnp.int32),
#             stage=jnp.array(1).astype(jnp.int32),
#             buffer=jnp.array(0).astype(jnp.int32),
#             obs_stack=None,
#             occupied_y=jnp.array([0, 0, 0]),
#             step_counter=jnp.array(0).astype(jnp.int32),
#             enemies_killed=jnp.array(0).astype(jnp.int32),
#             current_stage=jnp.array(0).astype(jnp.int32),
#             enemies_spawned_this_stage=jnp.array(0).astype(jnp.int32),
#             enemies_invisible=jnp.array(0).astype(jnp.int32),
#             cooldown=jnp.array(0).astype(jnp.int32),
#             fired=jnp.array(0).astype(jnp.int32)
#         )
#         obs = self._get_observation(state)
#         def expand_and_copy(x):
#             x_expanded = jnp.expand_dims(x, axis=0)
#             return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
#         obs_stack = jax.tree.map(expand_and_copy, obs)
#         state = state._replace(obs_stack=obs_stack)
#         return  obs_stack, state

#     @partial(jax.jit, static_argnums=(0,))
#     def step(self, state: AssaultState, action: chex.Array) -> Tuple[AssaultState, AssaultObservation, float, bool, AssaultInfo]:
#         # Player step
#         new_state = player_step(state, action)

#         new_state = player_projectile_step(new_state,action)
#         new_state = enemy_projectile_step(new_state)
#         new_state = enemy_step(new_state)
#         new_state = mothership_step(new_state)
#         new_state = cooldown_step(new_state)
#         occupied_y = new_state.occupied_y

#         player_proj_active = jnp.greater_equal(new_state.player_projectile_y, 0)
#         enemy_proj_active = jnp.greater_equal(new_state.enemy_projectile_y, 0)
#         enemy_proj_lateral = jnp.equal(new_state.enemy_projectile_y, PLAYER_Y)  # Only lateral projectiles

#         current_x_distance = new_state.player_projectile_x - new_state.enemy_projectile_x
#         enemy_horizontal_dir = jnp.sign(state.player_x - new_state.enemy_projectile_x)
#         enemy_prev_x = new_state.enemy_projectile_x - enemy_horizontal_dir * 2  # Enemy was 2 pixels away in opposite direction
        
#         # Calculate previous distances
#         prev_x_distance = new_state.player_projectile_x - enemy_prev_x
#         x_sign_changed = jnp.not_equal(jnp.sign(current_x_distance), jnp.sign(prev_x_distance))

#         projectiles_intersecting = jnp.logical_and.reduce(jnp.array([
#             player_proj_active,
#             enemy_proj_active,
#             enemy_proj_lateral,
#             x_sign_changed,      # X distance changed sign (crossed horizontally)
#         ]))
        
#         projectile_collision = jnp.logical_and.reduce(jnp.array([
#             player_proj_active,
#             enemy_proj_active,
#             enemy_proj_lateral,  # Only allow collision when enemy projectile is lateral
#             check_collision(
#                 new_state.player_projectile_x, new_state.player_projectile_y,
#                 new_state.enemy_projectile_x, new_state.enemy_projectile_y,
#                 4, 4  # Collision box size
#             )
#         ]))

#         new_player_proj_x = jnp.where(projectile_collision, -1, new_state.player_projectile_x)
#         new_player_proj_y = jnp.where(projectile_collision, -1, new_state.player_projectile_y)
#         new_player_proj_dir = jnp.where(projectile_collision, 0, new_state.player_projectile_dir)
        
#         new_enemy_proj_x = jnp.where(projectile_collision, -1, new_state.enemy_projectile_x)
#         new_enemy_proj_y = jnp.where(projectile_collision, -1, new_state.enemy_projectile_y)
#         new_enemy_proj_dir = jnp.where(projectile_collision, 0, new_state.enemy_projectile_dir)

#         def split_condition(stage):
#             return stage > -1

#         def kill_enemy(arr):
#             ex, ey, ew, eh, proj_x, proj_y, occupied_y, linked_y = arr
#             hit = check_collision(proj_x, proj_y, ex, ey, ew, eh)
#             matches = jnp.array(ENEMY_Y_POSITIONS) == ey
#             has_match = jnp.any(matches)
#             idx = jnp.argmax(matches)
            
#             new_occupied_y = jax.lax.cond(
#                 jnp.logical_and.reduce(jnp.array([hit, has_match, linked_y > HEIGHT])), 
#                 lambda _: occupied_y.at[idx].set(0), 
#                 lambda _: occupied_y, 
#                 operand=None
#             )

#             new_ex = jnp.where(hit, -1, ex)
#             new_ey = jnp.where(hit, HEIGHT+1, ey)
#             return new_ex, new_ey, hit, new_occupied_y
        
#         was_split = False
#         # Function to split enemy into two
#         def split_enemy(arr):
#             ex, ey, ew, eh, proj_x, proj_y, occupied_y, _ = arr
#             hit = check_collision(proj_x, proj_y, ex, ey, ew, eh)
#             new_ex = jnp.where(hit, ex-ENEMY_SIZE[0], ex)
            
#             return new_ex, ey, hit, occupied_y
#         def spawn_enemy(arr):
#             ex,ey = arr[:2]
#             new_ex = jnp.where(ex+3 >= WIDTH, WIDTH, ex+ENEMY_SIZE[0])
#             new_ey = ey
#             matches = jnp.array(ENEMY_Y_POSITIONS) == ey
#             idx = jnp.argmax(matches)
#             new_occupied_y = occupied_y.at[idx].set(occupied_y[idx]+1)

#             return new_ex, new_ey, False, new_occupied_y

#         splitting_enemies = split_condition(state.current_stage)

#         # Enemy 1
#         arg_1 = [new_state.enemy_1_x, new_state.enemy_1_y, ENEMY_SIZE[0], ENEMY_SIZE[1], new_state.player_projectile_x, new_state.player_projectile_y, occupied_y, new_state.enemy_4_y]
#         e1_x, e1_y, hit1, occupied_y = jax.lax.cond(jnp.logical_and(splitting_enemies, jnp.logical_not(state.enemy_1_split)),
#                                                     split_enemy,
#                                                     kill_enemy, 
#                                                     operand = arg_1)
#         e1_split = jnp.where(jnp.logical_and(hit1, e1_y < HEIGHT+1), 1, 0)
#         # Enemy 2
#         arg2 = [new_state.enemy_2_x, new_state.enemy_2_y, ENEMY_SIZE[0], ENEMY_SIZE[1], new_state.player_projectile_x, new_state.player_projectile_y, occupied_y, new_state.enemy_5_y]
#         e2_x, e2_y, hit2, occupied_y = jax.lax.cond(jnp.logical_and(splitting_enemies, jnp.logical_not(state.enemy_2_split)),
#                                                     split_enemy,
#                                                     kill_enemy,
#                                                     operand=arg2)
#         e2_split = jnp.where(jnp.logical_and(hit2, e2_y < HEIGHT+1), 1, 0)
#         # Enemy 3
#         arg3 = [new_state.enemy_3_x, new_state.enemy_3_y, ENEMY_SIZE[0], ENEMY_SIZE[1], new_state.player_projectile_x, new_state.player_projectile_y, occupied_y, new_state.enemy_6_y]
#         e3_x, e3_y, hit3, occupied_y = jax.lax.cond(jnp.logical_and(splitting_enemies, jnp.logical_not(state.enemy_3_split)),
#                                                     split_enemy,
#                                                     kill_enemy,
#                                                     operand=arg3)
#         e3_split = jnp.where(jnp.logical_and(hit3,e3_y < HEIGHT+1), 1, 0)
#         was_split = jnp.logical_or.reduce(jnp.array([e1_split, e2_split, e3_split]))
#         # Enemy 4
#         xy4 = jnp.array([new_state.enemy_4_x, new_state.enemy_4_y])
#         spawn4 = jnp.array([e1_x, e1_y])
#         arr4 = jnp.where(jnp.logical_and(splitting_enemies, jnp.logical_and(hit1, was_split)), spawn4, xy4)
#         arg4 = [arr4[0], arr4[1], ENEMY_SIZE[0], ENEMY_SIZE[1], new_state.player_projectile_x, new_state.player_projectile_y, occupied_y, e1_y]
        
#         e4_x, e4_y, hit4, occupied_y = jax.lax.cond(jnp.logical_and(hit1, was_split),
#                                                     spawn_enemy,
#                                                     kill_enemy,
#                                                     operand=arg4)
#         # Enemy 5

#         xy5 = jnp.array([new_state.enemy_5_x, new_state.enemy_5_y])
#         spawn5 = jnp.array([e2_x, e2_y])
#         arr5 = jnp.where(jnp.logical_and(splitting_enemies, jnp.logical_and(hit2, was_split)), spawn5, xy5)
#         arg5 = [arr5[0], arr5[1], ENEMY_SIZE[0], ENEMY_SIZE[1], new_state.player_projectile_x, new_state.player_projectile_y, occupied_y, e2_y]
#         e5_x, e5_y, hit5, occupied_y = jax.lax.cond(jnp.logical_and(hit2, was_split),
#                                                     spawn_enemy,
#                                                     kill_enemy,
#                                                     operand=arg5)
#         # Enemy 6
#         xy6 = jnp.array([new_state.enemy_6_x, new_state.enemy_6_y])
#         spawn6 = jnp.array([e3_x, e3_y])
#         arr6 = jnp.where(jnp.logical_and(splitting_enemies, jnp.logical_and(hit3, was_split)), spawn6, xy6)
#         arg6 = [arr6[0], arr6[1], ENEMY_SIZE[0], ENEMY_SIZE[1], new_state.player_projectile_x, new_state.player_projectile_y, occupied_y, e3_y]
#         e6_x, e6_y, hit6, occupied_y = jax.lax.cond(jnp.logical_and(hit3, was_split),
#                                                     spawn_enemy,
#                                                     kill_enemy,
#                                                     operand=arg6)

        
#         # If any enemy was hit, remove projectile
#         any_hit = hit1 | hit2 | hit3 | hit4 | hit5 | hit6
        
#         new_player_proj_x = jnp.where(jnp.logical_and(any_hit, jnp.logical_not(projectile_collision)), -1, new_state.player_projectile_x)
#         new_player_proj_y = jnp.where(jnp.logical_and(any_hit, jnp.logical_not(projectile_collision)), -1, new_state.player_projectile_y)
#         new_player_proj_dir = jnp.where(jnp.logical_and(any_hit, jnp.logical_not(projectile_collision)), 0, new_state.player_projectile_dir)


#         # Increase score for each enemy hit (e.g., +1 per enemy)
#         score_incr = hit1.astype(jnp.int32) + hit2.astype(jnp.int32) + hit3.astype(jnp.int32) + \
#                     hit4.astype(jnp.int32) + hit5.astype(jnp.int32) + hit6.astype(jnp.int32)
        
#         enemy_1_killed = jnp.logical_and(hit1, jnp.logical_not(state.enemy_1_split))
#         enemy_2_killed = jnp.logical_and(hit2, jnp.logical_not(state.enemy_2_split))
#         enemy_3_killed = jnp.logical_and(hit3, jnp.logical_not(state.enemy_3_split))

#         kills_incr = enemy_1_killed.astype(jnp.int32) + enemy_2_killed.astype(jnp.int32) + enemy_3_killed.astype(jnp.int32)

        
#         new_score = state.score + score_incr
#         new_enemies_killed = state.enemies_killed + kills_incr
#         all_rows_empty = jnp.array_equal(new_state.occupied_y, jnp.array([0, 0, 0]))

        

#         #stage_complete = jnp.logical_and(jnp.equal(jnp.mod(new_enemies_killed, 10), 0), (new_enemies_killed > 0))
#         stage_complete = jnp.logical_and.reduce(jnp.array([
#             jnp.greater(new_enemies_killed, 0),  # Enemies were killed this step
#             jnp.equal(jnp.mod(new_enemies_killed, 10), 0),  # Total kills is multiple of 10
#             all_rows_empty  # No enemies occupying any row
#         ]))
#         new_enemies_killed = jnp.where(stage_complete, 0, new_enemies_killed)
#         new_enemies_spawned_this_stage = jnp.where(stage_complete, 0, new_state.enemies_spawned_this_stage)
#         new_current_stage = jnp.where(stage_complete, state.current_stage + 1, state.current_stage)
        
#         invis_action = jax.lax.cond(
#             jnp.equal(state.current_stage, 1),
#             lambda _: jax.random.uniform(jax.random.PRNGKey(state.step_counter), shape=()) < 0.01,
#             lambda _: jnp.array(False),
#             operand=None
#         )
#         enemies_invisible = jnp.where(invis_action, jnp.logical_not(state.enemies_invisible), state.enemies_invisible)
#         enemies_invisible = jnp.where(stage_complete, jnp.array(0), enemies_invisible)

#         new_state = new_state._replace(
#             player_projectile_x=new_player_proj_x,
#             player_projectile_y=new_player_proj_y,
#             player_projectile_dir=new_player_proj_dir,
#             enemy_projectile_x=new_enemy_proj_x,
#             enemy_projectile_y=new_enemy_proj_y,
#             enemy_projectile_dir=new_enemy_proj_dir,
#             enemy_1_x=e1_x, enemy_1_y=e1_y,
#             enemy_1_split=jnp.logical_or(new_state.enemy_1_split, e1_split),
#             enemy_1_dir=jnp.where(e1_split,-1, new_state.enemy_1_dir),
#             enemy_2_x=e2_x, enemy_2_y=e2_y,
#             enemy_2_split=jnp.logical_or(new_state.enemy_2_split,e2_split),
#             enemy_2_dir=jnp.where(e2_split,-1, new_state.enemy_2_dir),
#             enemy_3_x=e3_x, enemy_3_y=e3_y,
#             enemy_3_split=jnp.logical_or(new_state.enemy_3_split,e3_split),
#             enemy_3_dir=jnp.where(e3_split,-1, new_state.enemy_3_dir),
#             enemy_4_x=e4_x, enemy_4_y=e4_y,
#             enemy_4_dir=jnp.where(e1_split,1, new_state.enemy_4_dir),
#             enemy_5_x=e5_x, enemy_5_y=e5_y,
#             enemy_5_dir=jnp.where(e2_split,1, new_state.enemy_5_dir),
#             enemy_6_x=e6_x, enemy_6_y=e6_y,
#             enemy_6_dir=jnp.where(e3_split,1, new_state.enemy_6_dir),
#             score=new_score,
#             enemies_killed=new_enemies_killed,
#             current_stage=new_current_stage,
#             occupied_y=occupied_y,
#             enemies_spawned_this_stage=new_enemies_spawned_this_stage,
#             enemies_invisible=enemies_invisible,
#             # TODO: update other fields as needed
#         )
        

#         # Reward: +1 if score increased, -1 if lost life
#         reward = jnp.where(new_state.score > state.score, 1.0, 0.0)
#         reward = jnp.where(new_state.player_lives < state.player_lives, -1.0, reward)
#         done = jnp.greater_equal(0,new_state.player_lives)
#         obs = self._get_observation(new_state)
#         obs_stack = jax.tree.map(lambda stack, o: jnp.concatenate([stack[1:], jnp.expand_dims(o, axis=0)], axis=0), state.obs_stack, obs)
#         new_state = new_state._replace(obs_stack=obs_stack)
#         info = AssaultInfo(time=jnp.array(0), all_rewards=jnp.zeros(1))

#         # Use jax.debug.print instead of Python's print for JIT compatibility
#         jax.debug.print("Enemy positions:")
#         jax.debug.print("Enemy 1: ({}, {})", state.enemy_1_x, state.enemy_1_y)
#         jax.debug.print("Enemy 2: ({}, {})", state.enemy_2_x, state.enemy_2_y)
#         jax.debug.print("Enemy 3: ({}, {})", state.enemy_3_x, state.enemy_3_y)
#         jax.debug.print("Enemy 4: ({}, {})", state.enemy_4_x, state.enemy_4_y)
#         jax.debug.print("Enemy 5: ({}, {})", state.enemy_5_x, state.enemy_5_y)
#         jax.debug.print("Enemy 6: ({}, {})", state.enemy_6_x, state.enemy_6_y)
        
#         # Print occupied_y using jax.debug.print
#         jax.debug.print("Occupied rows: {}", state.occupied_y)
#         jax.debug.print("current_stage: {}", state.current_stage)
#         jax.debug.print("Enemies killed: {}", state.enemies_killed)
#         jax.debug.print("----------------------------------------")

#         new_step_counter = jnp.mod(state.step_counter + 1, Y_STEP_DELAY * 100000)
#         new_state = new_state._replace(step_counter=new_step_counter)

#         return obs_stack, new_state, reward, done, info

#     @partial(jax.jit, static_argnums=(0,))
#     def _get_observation(self, state: AssaultState):
#         # Build observation from state
#         player = EntityPosition(
#             x=state.player_x,
#             y=jnp.array(PLAYER_Y),
#             width=jnp.array(PLAYER_SIZE[0]),
#             height=jnp.array(PLAYER_SIZE[1]),
#             invisible=jnp.array(0),
#         )
#         mothership = EntityPosition(
#             x=state.mothership_x,
#             y=jnp.array(MOTHERSHIP_Y),
#             width=jnp.array(MOTHERSHIP_SIZE[0]),
#             height=jnp.array(MOTHERSHIP_SIZE[1]),
#             invisible=jnp.array(0),
#         )
#         def enemy_entity(x, y):
#             return EntityPosition(
#                 x=x, y=y,
#                 width=jnp.array(ENEMY_SIZE[0]),
#                 height=jnp.array(ENEMY_SIZE[1]),
#                 invisible=jnp.array(0),
#             )
#         enemy_projectile=EntityPosition(
#                 x=state.enemy_projectile_x,
#                 y=state.enemy_projectile_y,
#                 width=jnp.array(2),
#                 height=jnp.array(4),
#                 invisible=jnp.array(0),
#             )
#         return AssaultObservation(
#             player=player,
#             mothership=mothership,
#             enemy_1=enemy_entity(state.enemy_1_x, state.enemy_1_y),
#             enemy_2=enemy_entity(state.enemy_2_x, state.enemy_2_y),
#             enemy_3=enemy_entity(state.enemy_3_x, state.enemy_3_y),
#             enemy_4=enemy_entity(state.enemy_4_x, state.enemy_4_y),
#             enemy_5=enemy_entity(state.enemy_5_x, state.enemy_5_y),
#             enemy_6=enemy_entity(state.enemy_6_x, state.enemy_6_y),
#             enemy_projectile=enemy_projectile,
#             lives=state.player_lives,
#             score=state.score,
#         )

#     @partial(jax.jit, static_argnums=(0,))
#     def obs_to_flat_array(self, obs: AssaultObservation) -> jnp.ndarray:
#         # Flatten all positions and stats into a 1D array
#         return jnp.concatenate([
#             obs.player.x.flatten(), obs.player.y.flatten(),
#             obs.player.width.flatten(), obs.player.height.flatten(),
#             obs.mothership.x.flatten(), obs.mothership.y.flatten(),
#             obs.enemy_1.x.flatten(), obs.enemy_1.y.flatten(),
#             obs.enemy_2.x.flatten(), obs.enemy_2.y.flatten(),
#             obs.enemy_3.x.flatten(), obs.enemy_3.y.flatten(),
#             obs.enemy_4.x.flatten(), obs.enemy_4.y.flatten(),
#             obs.enemy_5.x.flatten(), obs.enemy_5.y.flatten(),
#             obs.enemy_6.x.flatten(), obs.enemy_6.y.flatten(),
#             obs.enemy_projectile.x.flatten(), obs.enemy_projectile.y.flatten(),
#             obs.lives.flatten(), obs.score.flatten()
#         ])

#     def action_space(self) -> spaces.Discrete:
#         return spaces.Discrete(len(self.action_set))
    
#     def observation_space(self) -> spaces.Box:
#         return spaces.Box(
#             low=0,
#             high=255,
#             shape=None,
#             dtype=jnp.uint8,
#         )
    
#     @partial(jax.jit, static_argnums=(0,))
#     def _get_info(self, state: AssaultState, all_rewards: chex.Array) -> AssaultInfo:
#         return AssaultInfo(time=state.step_counter, all_rewards=all_rewards)

#     @partial(jax.jit, static_argnums=(0,))
#     def _get_env_reward(self, previous_state: AssaultState, state: AssaultState):
#         return (state.player_score - state.enemy_score) - (
#             previous_state.player_score - previous_state.enemy_score
#         )

#     @partial(jax.jit, static_argnums=(0,))
#     def _get_all_reward(self, previous_state: AssaultState, state: AssaultState):
#         if self.reward_funcs is None:
#             return jnp.zeros(1)
#         rewards = jnp.array(
#             [reward_func(previous_state, state) for reward_func in self.reward_funcs]
#         )
#         return rewards 

#     @partial(jax.jit, static_argnums=(0,))
#     def _get_done(self, state: AssaultState) -> bool:
#         return jnp.logical_or(
#             jnp.greater_equal(state.player_score, 20),
#             jnp.greater_equal(state.enemy_score, 20),
#         )
    


# def load_assault_sprites():
#     """
#     Load all sprites required for Assault rendering.
#     Assumes files are named enemy.npy, life.npy, mothership.npy, player.npy
#     and are located in sprites/assault relative to this file.
#     """
#     MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
#     SPRITES_DIR = os.path.join(MODULE_DIR, "sprites", "assault")

#     background = aj.loadFrame(os.path.join(SPRITES_DIR, "background.npy"), transpose=True)
#     enemy = aj.loadFrame(os.path.join(SPRITES_DIR, "enemy_0.npy"), transpose=True)
#     #life = aj.loadFrame(os.path.join(SPRITES_DIR, "life.npy"), transpose=True)
#     mothership = aj.loadFrame(os.path.join(SPRITES_DIR, "mothership_0.npy"), transpose=True)
#     player = aj.loadFrame(os.path.join(SPRITES_DIR, "player.npy"), transpose=True)
#     player_projectile = aj.loadFrame(os.path.join(SPRITES_DIR, "player_projectile.npy"), transpose=True)
#     enemy_projectile = aj.loadFrame(os.path.join(SPRITES_DIR, "enemy_projectile.npy"), transpose=True)
#     enemy_projectile = aj.loadFrame(os.path.join(SPRITES_DIR, "enemy_projectile.npy"), transpose=True)
#     enemy_rain = aj.loadFrame(os.path.join(SPRITES_DIR, "proj_wide.npy"), transpose=True)
#     enemy_sphere = aj.loadFrame(os.path.join(SPRITES_DIR, "proj_sphere.npy"), transpose=True)
#     enemy_projectile_lateral = aj.loadFrame(os.path.join(SPRITES_DIR, "proj_lateral.npy"), transpose=True)
#     life = aj.loadFrame(os.path.join(SPRITES_DIR, "life.npy"), transpose=True)
#     enemy_tiny = aj.loadFrame(os.path.join(SPRITES_DIR, "enemy_tiny.npy"), transpose=True)

#     # Optionally expand dims if you want a batch/frame dimension
#     BACKGROUND_SPRITE = jnp.expand_dims(background, axis=0)
#     ENEMY_SPRITE = jnp.expand_dims(enemy, axis=0)
#     #LIFE_SPRITE = jnp.expand_dims(life, axis=0)
#     MOTHERSHIP_SPRITE = jnp.expand_dims(mothership, axis=0)
#     PLAYER_SPRITE = jnp.expand_dims(player, axis=0)
#     PLAYER_PROJECTILE= jnp.expand_dims(player_projectile, axis=0)
#     ENEMY_PROJECTILE = jnp.expand_dims(enemy_projectile, axis=0)
#     ENEMY_RAIN = jnp.expand_dims(enemy_rain, axis=0)
#     ENEMY_SPHERE = jnp.expand_dims(enemy_sphere, axis=0)
#     ENEMY_PROJECTILE_LATERAL = jnp.expand_dims(enemy_projectile_lateral, axis=0)
#     LIFE_SPRITE = jnp.squeeze(life)
#     ENEMY_TINY = jnp.expand_dims(enemy_tiny, axis=0)

#     DIGIT_SPRITES = aj.load_and_pad_digits(
#         os.path.join(MODULE_DIR, os.path.join(SPRITES_DIR, "number_{}.npy")),
#         num_chars=10,
#     )

#     return BACKGROUND_SPRITE,ENEMY_SPRITE, MOTHERSHIP_SPRITE, PLAYER_SPRITE, DIGIT_SPRITES, PLAYER_PROJECTILE,ENEMY_PROJECTILE, ENEMY_RAIN, ENEMY_SPHERE, ENEMY_PROJECTILE_LATERAL, LIFE_SPRITE, ENEMY_TINY

# class Renderer_AtraJaxisAssault:
#     """JAX-based Assault game renderer, optimized with JIT compilation."""

#     def __init__(self):
#         (
#             self.SPRITE_BG,
#             self.SPRITE_ENEMY,
#             self.SPRITE_MOTHERSHIP,
#             self.SPRITE_PLAYER,
#             self.DIGIT_SPRITES,
#             self.PLAYER_PROJECTILE,
#             self.ENEMY_PROJECTILE,
#             self.ENEMY_RAIN,
#             self.ENEMY_SPHERE,
#             self.ENEMY_PROJECTILE_LATERAL,
#             self.LIFE_SPRITE,
#             self.ENEMY_TINY
#         ) = load_assault_sprites()  # You need to implement this in atraJaxis

#     @partial(jax.jit, static_argnums=(0,))
#     def apply_color_transform(self, sprite, stage):
#         """Apply a color transformation by permuting RGB channels based on stage."""
#         # Get the number of channels
#         num_channels = sprite.shape[-1]
        
#         # Extract channels
#         r = sprite[..., 0]
#         g = sprite[..., 1]
#         b = sprite[..., 2]
        
#         # Handle alpha channel if present
#         has_alpha = num_channels >= 4
#         alpha = jnp.ones_like(r) if not has_alpha else sprite[..., 3]
        
#         # Use modulo to cycle through 4 permutation patterns (0-3)
#         stage_mod = jnp.mod(stage, 4)
        
#         # Use conditional logic with 4 color patterns
#         final_r = jnp.where(stage_mod == 0, r, 
#                     jnp.where(stage_mod == 1, g, 
#                     jnp.where(stage_mod == 2, b, g)))
        
#         final_g = jnp.where(stage_mod == 0, g, 
#                     jnp.where(stage_mod == 1, r, 
#                     jnp.where(stage_mod == 2, g, b)))
        
#         final_b = jnp.where(stage_mod == 0, b, 
#                     jnp.where(stage_mod == 1, b, 
#                     jnp.where(stage_mod == 2, r, r)))
        
#         # Stack channels back together with alpha if it exists
#         if has_alpha:
#             return jnp.stack([final_r, final_g, final_b, alpha], axis=-1).astype(jnp.uint8)
#         else:
#             return jnp.stack([final_r, final_g, final_b], axis=-1).astype(jnp.uint8)
        
#     @partial(jax.jit, static_argnums=(0,))
#     def render(self, state):
#         """
#         Renders the current Assault game state using JAX operations.

#         Args:
#             state: An AssaultState object containing the current game state.

#         Returns:
#             A JAX array representing the rendered frame.
#         """
#         raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)

#         # last minute change to make render work with newest upstream version without changing all 12 calls
#         def render_at(raster, y, x, frame):
#             return aj.render_at(raster, x, y, frame)
        
#         # Render background
#         frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
#         raster = render_at(raster, 0, 0, frame_bg)

#         # Render mothership
#         frame_mothership = aj.get_sprite_frame(self.SPRITE_MOTHERSHIP, 0)
#         raster = render_at(raster, MOTHERSHIP_Y, state.mothership_x, frame_mothership)

#         # Render player
#         frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
#         raster = render_at(raster, PLAYER_Y, state.player_x, frame_player)

#         # Render enemies (unrolled manually for JIT compatibility)
#         frame_enemy_original = aj.get_sprite_frame(self.SPRITE_ENEMY, 0)
#         frame_enemy = self.apply_color_transform(frame_enemy_original, state.current_stage)
#         frame_enemy_tiny_original = aj.get_sprite_frame(self.ENEMY_TINY, 0)
#         frame_enemy_tiny = self.apply_color_transform(frame_enemy_tiny_original, state.current_stage)

#         def render_split_enemy(xy):
#             x,y, raster = xy
#             should_render = jnp.logical_and(y < HEIGHT+1, jnp.logical_not(state.enemies_invisible))
#             return jax.lax.cond(should_render, lambda _: render_at(raster, y, x, frame_enemy_tiny), lambda _: raster, operand=None)
        
#         def render_enemy(xy):
#             x,y, raster = xy
#             should_render = jnp.logical_and(y < HEIGHT+1, jnp.logical_not(state.enemies_invisible))
#             return jax.lax.cond(should_render, lambda _: render_at(raster, y, x, frame_enemy), lambda _: raster, operand=None)
        
#         def render_tiny_enemy(xy):
#             x, y, raster = xy
#             should_render = jnp.logical_and(y < HEIGHT+1, jnp.logical_not(state.enemies_invisible))
#             return jax.lax.cond(should_render, lambda _: render_at(raster, y, x, frame_enemy_tiny), lambda _: raster, operand=None)

#         raster = jax.lax.cond( state.enemy_1_split == 1, render_split_enemy,render_enemy, [state.enemy_1_x,state.enemy_1_y, raster])
#         raster = jax.lax.cond( state.enemy_2_split == 1, render_split_enemy,render_enemy, [state.enemy_2_x, state.enemy_2_y, raster])
#         raster = jax.lax.cond( state.enemy_3_split == 1, render_split_enemy,render_enemy, [state.enemy_3_x, state.enemy_3_y, raster])

#         raster = render_tiny_enemy([state.enemy_4_x, state.enemy_4_y, raster])
#         raster = render_tiny_enemy([state.enemy_5_x, state.enemy_5_y, raster])
#         raster = render_tiny_enemy([state.enemy_6_x, state.enemy_6_y, raster])        
#         self.PLAYER_PROJECTILE_SIDEWAYS = jnp.array([[[[236, 236, 236, 255]]]*7])
        
#         # Render player projectile using lax.cond
#         def render_player_proj(_):
            
#             frame_proj = aj.get_sprite_frame(self.PLAYER_PROJECTILE, 0)
#             return render_at(raster, state.player_projectile_y, state.player_projectile_x, frame_proj)
        
#         def render_player_proj_sideways(_):
#             frame_proj = aj.get_sprite_frame(self.PLAYER_PROJECTILE_SIDEWAYS, 0)
#             return render_at(raster, PLAYER_Y+2, state.player_projectile_x, frame_proj)

#         def skip_player_proj(_):
#             return raster

#         raster = jax.lax.cond(
#             jnp.logical_and(jnp.greater_equal(state.player_projectile_y, 0),jnp.not_equal(state.player_projectile_y, PLAYER_Y)),
#             render_player_proj,
#             skip_player_proj,
#             operand=None
#         )
#         raster = jax.lax.cond(
#             jnp.equal(state.player_projectile_y, PLAYER_Y),
#             render_player_proj_sideways,
#             lambda _: raster,
#             operand=None
#         )
#         # Render enemy projectile using lax.cond
#         def render_enemy_proj(_):
#             is_stage_4 = jnp.equal(jnp.mod(state.current_stage + 1, 4), 0)
#             is_stage_3 = jnp.equal(jnp.mod(state.current_stage + 2, 4), 0)
#             is_lateral = jnp.equal(state.enemy_projectile_y, PLAYER_Y)
            
#             # First, check if it's stage 3, since that takes priority
#             def stage3_proj(_):
#                 return render_at(raster, state.enemy_projectile_y, state.enemy_projectile_x, 
#                                 aj.get_sprite_frame(self.ENEMY_RAIN, 0))
            
#             def other_stages(_):
#                 # If not stage 3, then check if it's stage 4
#                 def stage4_proj(_):
#                     # For stage 4, check if it's lateral
#                     def lateral_proj(_):
#                         return render_at(raster, state.enemy_projectile_y, state.enemy_projectile_x,
#                                         aj.get_sprite_frame(self.ENEMY_PROJECTILE_LATERAL, 0))
                    
#                     def sphere_proj(_):
#                         return render_at(raster, state.enemy_projectile_y, state.enemy_projectile_x,
#                                         aj.get_sprite_frame(self.ENEMY_SPHERE, 0))
                    
#                     return jax.lax.cond(
#                         is_lateral,
#                         lateral_proj,
#                         sphere_proj,
#                         operand=None
#                     )
                
#                 def standard_proj(_):
#                     return render_at(raster, state.enemy_projectile_y, state.enemy_projectile_x,
#                                     aj.get_sprite_frame(self.ENEMY_PROJECTILE, 0))
                
#                 # Choose between stage 4 logic and standard
#                 return jax.lax.cond(
#                     is_stage_4,
#                     stage4_proj,
#                     standard_proj,
#                     operand=None
#                 )
            
#             # Final choice between stage 3 and other stages
#             return jax.lax.cond(
#                 is_stage_3,
#                 stage3_proj,
#                 other_stages,
#                 operand=None
#             )

#         def skip_enemy_proj(_):
#             return raster
        
#         raster = jax.lax.cond(
#             jnp.greater_equal(state.enemy_projectile_y, 0),
#             render_enemy_proj,
#             skip_enemy_proj,
#             operand=None
#         )

#         # Render score (top left)
#         score_digits = aj.int_to_digits(state.score, max_digits=4)
#         raster = aj.render_label_selective(
#             raster, 5, 5, score_digits, self.DIGIT_SPRITES, 0, len(score_digits), spacing=12
#         )

#         # Render lives (top right)
#         lives_digits = aj.int_to_digits(state.player_lives, max_digits=1)
#         raster = aj.render_label_selective(
#             raster, 5, WIDTH - 20, lives_digits, self.DIGIT_SPRITES, 0, len(lives_digits), spacing=12
#         )

#         # Render lives (bottom left)
#         def lives_fn(i, raster):
#             return render_at(raster, LIVES_Y, LIFE_ONE_X + i * LIFE_OFFSET, self.LIFE_SPRITE)
#         raster = jax.lax.fori_loop(0, state.player_lives, lives_fn, raster)

#         # Render heat bar(bottom right)
#         def heat_bar_fn(heat, raster):
#             color = (0, 255, 0, 255)
#             background_color = (0, 0, 0, 255)
#             return aj.render_bar(
#                 raster, WIDTH-60, LIVES_Y,heat+1,MAX_HEAT+1,48,5,color,background_color
#             )
#         raster = heat_bar_fn(state.heat, raster)
#         print(self.PLAYER_PROJECTILE)
#         return raster
    
# if __name__ == "__main__":
#     # Initialize Pygame
#     pygame.init()
#     screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
#     pygame.display.set_caption("Assault Game")
#     clock = pygame.time.Clock()

#     game = JaxAssault()

#     # Create the JAX renderer
#     renderer = Renderer_AtraJaxisAssault()

#     # Get jitted functions
#     jitted_step = jax.jit(game.step)
#     jitted_reset = jax.jit(game.reset)

#     obs, curr_state = jitted_reset()

#     # Game loop
#     running = True
#     frame_by_frame = False
#     frameskip = game.frameskip
#     counter = 1

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_f:
#                     frame_by_frame = not frame_by_frame
#             elif event.type == pygame.KEYDOWN or (
#                     event.type == pygame.KEYUP and event.key == pygame.K_n
#             ):
#                 if event.key == pygame.K_n and frame_by_frame:
#                     if counter % frameskip == 0:
#                         action = get_human_action()
#                         obs, curr_state, reward, done, info = jitted_step(
#                             curr_state, action
#                         )
#         game_over = jnp.logical_or(
#             jnp.less_equal(curr_state.player_lives, 0),  # Player has 0 or fewer lives
#             jnp.greater_equal(curr_state.score, 999999)  # Score reached 999999
#         )
#         if game_over:
#             print(f"Game Over! Final Score: {curr_state.score}, Lives: {curr_state.player_lives}")
#             running = True

#         if not frame_by_frame:
#             if counter % frameskip == 0:
#                 action = get_human_action()
#                 obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        

#         # Render and display
#         raster = renderer.render(curr_state)

#         aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

#         counter += 1
#         clock.tick(60)

#     pygame.quit()
