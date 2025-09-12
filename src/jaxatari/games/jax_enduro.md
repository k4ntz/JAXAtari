# JAX Enduro

## Gameplay Observations from playing the original ROM
### Driving
- Speed
  - Car speed is maintained, the gas does not need to be pressed like a pedal
  - once accelerated the car can never stop again, it always maintains a minimum speed of 6 (from RAM state 22)
    - Config: ``min_speed``
  - The maximum speed is 120 (from RAM state 22)
    - Config: ``max_speed``
  - Minimum km progression is measured by starting the original game and letting the car progress with min speed for 5 km (2:23 min)
    - Config: ``km_per_second_per_speed_unit``
  - The tire animation speed increases wit increasing speed
  - With increasing speed the car moves forward, 1 pixel at a time
      - Moving forwards lets the car collide "earlier"
- Acceleration
  - It takes 5 second to get to full speed
  - The car accelerates in 2 linear ways. From measuring the RAM states the acceleration can be approximated with this function, where t = number of seconds:
  
    *f(t) = 10.5t where f <= 46*
  
    *f(t) = 3.75t where f > 46*
    - Config: ``acceleration_per_frame``, `slower_acceleration_per_frame`, `acceleration_slow_down_threshold`
- Breaking 
  - Reduces your speed faster than accelerating (fire)
    - Config: `breaking_per_second`
- Drift
  - In curves the car drifts "out of the curve"
  - Drift is about 2-3 pixels per second
    - Config: `drift_per_second_pixels`
- Track collision
  - Hitting the side of the road gives a small kickback and reduces you speed by 15 (RAM state measured)
    - Kickback: `track_collision_kickback_pixels`
    - Speed reduction: `track_collision_speed_reduction`
- Steering
  - It takes ~3 seconds to steer from one edge of the track to the one on a straight section
    - Config: `steering_range_in_pixels`
    - Config: `steering_sensitivity`

### Opponent Cars
- Behavior
  - Enemies do not change lane position
  - They do have the same speed (24) that does not change
    - Config: ``opponent_speed``
- Opponent Generation
  - All opponents are generated upfront with a random seed
    - Stored in: ``state.opponent_pos_and_color``
    - Implemented in: ``_generate_opponent_spawns(...)``
  - They are generated in a way that is non-blocking which always let's the player overtake
- In-Game Spawning
  - The original game has 7 opponent slots for the visible opponents (from near to far)
    - Stored in: ``state.visible_opponent_positions``
    - Implemented in: ``_get_visible_opponent_positions()``
  - Each slot contains the color and lane position of the opponent
  - The slots do not have the same size. Slots closer to the player (smaller index) have more pixels. This creates the effect of the perspective.
  - The absolute x-Positions for each of the 7 slots are stored in the config.
    - Config: ``opponent_slot_ys``
  - This implementation pre-calculates an opponent array and moves over it depending on player speed. The player can 
  "see" a 7 slot window of that array.
  
- Overtaking
  - If the player slows down they overtake the player again 
  - Cars overtaking you makes the counter go backwards
  - They do not crash into the player when overtaking, they spawn at the other side to avoid collision. This is handled by changing the ``opponent_pos_and_color`` array on demand.
    - Implemented in: ``_adjust_opponent_positions_when_overtaking``
- Opponent collision
  - Hitting an opponent reduces speed to 6 (not zero) and creates a cooldown (~ 3 seconds)
    - Config: ``car_crash_cooldown_seconds``
  - During cooldown there is no steering and accelerating
    - Implemented in: `cooldown_handling()`
- Colors
  - Cars have random colors
  - No red cars (encoded in `_generate_opponent_spawns(...)`)

### Environment
- Weather
  - Weather cycles are time based
  - They are not all the same length (see `weather_starts_s`)
  - The horizon colors only change during sunset
  - Weather colors are encoded in ``weather_color_codes``
- Track
  - Curves are 1-15 km long and randomly distributed (`_build_whole_track()`)
  - Steering causes the track to move in the opposite direction, the car moves in the steering direction. This creates a more vivid steering experience.
  - Every 400 m the Track has a "bumper"
- Level
  - 200 cars to overtake during a whole day/night cycle in order to pass a level.
    - Config: ``cars_to_pass_per_level``
  - Number increases by 100 per level
    - Config: ``cars_increase_per_level``
  - Max level = 5
    - Config: ``max_increase_level``
- Clouds 
  - Clouds move in the opponent curve direction
  - Cloud movement depends on player speed
    - Config: ``mountain_pixel_movement_per_frame_per_speed_unit``
  - Clouds stop once the track is straight again
  - Clouds appear at the opposite side once they leave the screen on one side

## Debug renderer
The file contains a [debug-renderer](./jax_enduro_debug_renderer.py) with an overlay to display state variables for 
debugging and analysis.
