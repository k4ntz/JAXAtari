# JAX Enduro

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

