# JAX Enduro

## Gameplay Observations from playing the original ROM
### Driving
- Car speed is maintained, the gas does not need to be pressed like a pedal
    - once accelerated the car can never stop again, it always maintains a minimum speed (6)
    - slowest is 0,1 km per second
    - fastest is 0,5 km per second
    - It takes 5 second to get to full speed
    - The acceleration speed changes. It is calculated by 2 linear functions. See
    - The tire animation speed increases wit increasing speed
- With increasing speed the car moves forward, 1 pixel at a time
    - 5 presses of the trigger equals one pixel
    - Moving forwards lets the car collide "earlier"
- Breaking reduces your speed faster than accelerating (fire)
  - Config: `breaking_per_second`
- Drift is about 2-3 pixels per second
  - Config: `drift_per_second_pixels`
- Hitting the side of the road gives a small kickback and reduces you speed by 15
  - Kickback: `track_collision_kickback_pixels`
  - Speed reduction: `track_collision_speed_reduction`
- It takes ~3 seconds to steer from one edge of the track to the one on a straight section
  - Config: `steering_range_in_pixels`
  - Config: `steering_sensitivity`

### Opponent Cars
- Enemies do not change lane position
- They do have the same speed (24) that does not change
- Overtaking
  - They spawn in a way that is non-blocking which always let's the player overtake
  - If the player slows down they overtake the player again
  - They do not crash into the player when overtaking, they spawn at the other side to avoid collision
  - Cars overtaking you makes the counter go backwards
- Opponent collision
  - Hitting an opponent reduces speed to 6 (not zero) and creates a cooldown (~ 3 seconds)
  - during cooldown there is no steering and accelerating (encoded in `cooldown_handling()`)
- No red cars (encoded in `_generate_opponent_spawns(...)`)

### Environment
- Weather cycles are time based
  - They are not all the same length
  - The horizon colors only change during sunset
- Track
  - Curves are 1-15 km long and randomly distributed (`_build_whole_track()`)
  - Steering causes the track to move in the opposite direction, the car moves in the steering direction. This creates a more vivid steering experience.
- Every 400 m the Track has a "bumper"
- Level
  - 200 cars to overtake. Number increases by 100 per level
  - Max level = 5
- Clouds move in the opponent curve direction
  - Cloud movement depends on player speed
  - Clouds stop once the track is straight again
  - Clouds appear at the opposite side once they leave the screen on one side

## Debug renderer
The file contains a [debug-renderer](./jax_enduro_debug_renderer.py) with an overlay to display state variables for 
debugging and analysis.


## How to Modify Different Game Aspects

### Gameplay Mechanics

**Changing Car Physics/Movement:**
- Modify `_step_single()` method in `JaxEnduro`
- Adjust speed, acceleration, steering sensitivity in `GameConfig`
- Key functions: `regular_handling()`, `cooldown_handling()`

**Modifying Collision Behavior:**
- Track collisions: `_check_car_track_collision()`
- Car-to-car collisions: `_check_car_opponent_collision()`  
- Collision responses: Look for kickback and speed reduction logic in `_step_single()`

**Adjusting Difficulty/Progression:**
- Level progression: Modify `cars_to_pass_per_level` and related constants in `GameConfig`
- Speed limits: Change `min_speed`, `max_speed` in `GameConfig`
- Day/night cycle: Adjust `weather_starts_s` array in `GameConfig`

### Opponent System

**Changing Opponent Spawning:**
- Core logic: `generate_opponent_spawns()` method
- Spawn density: Modify `opponent_density` in `GameConfig`  
- Spawn patterns: Adjust the constraint logic that prevents lane-blocking triplets
- Colors: Modify `generate_non_red_color()` function within spawning logic

**Opponent Behavior:**
- Movement speed: Change `opponent_speed` in `GameConfig`
- Overtaking mechanics: Modify `_adjust_opponent_positions_when_overtaking()`
- Visibility calculation: Adjust `get_visible_opponent_positions()`

### Track System

**Track Generation:**
- Track layout: Modify `build_whole_track()` for different curve patterns
- Track width/curvature: Adjust `track_width`, `track_max_curvature_width` in `GameConfig`
- Curve dynamics: Change `generate_viewable_track()` for different perspective effects

**Track Rendering:**
- Visual appearance: Modify `_render_track_from_state()` in renderer
- Track boundaries: Adjust `_generate_track_spaces()` for width progression

### Visual System

**Sprite Management:**
- Loading: Modify `_load_sprites()` in `EnduroRenderer`
- Format changes: Adjust `load_frame_with_animation()` in utilities
- Collision masks: Modify `VehicleSpec` class initialization

**Weather/Time Effects:**
- Weather progression: Adjust `weather_starts_s` array in `GameConfig`
- Visual colors: Modify `weather_color_codes` array for different weather appearances
- Fog effects: Change `_render_fog()` in renderer

**UI Elements:**
- Odometer: Modify `_render_distance_odometer()` for different display formats
- Score display: Adjust `_render_cars_to_overtake_score()` and `_render_level_score()`
- Debug overlay: Modify `render_debug_overlay()` in the demo script

