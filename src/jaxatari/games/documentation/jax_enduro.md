# JAX Enduro

## Fine-tuning:

### Opponent Spawning
- ``opponent_spawn_seed`` The seed for the opponent spawning
- ``length_of_opponent_array`` The total length of the opponent array. If the end of the array is reached the player progresses through the same pattern.
- ``opponent_density`` How many oppoents should be in the array. Must be >= 0.0 and < 1.0
- ``opponent_delay_slots`` How many slots should be left empty at the beginning of the array for the start of the game.
- ``lane_ratios`` Determines the x position of a lane that an opponent is on. First position is the left lane, 2nd the middle lane and 3rd position the right lane.

### Track generation
- ``track_seed`` The sees for the track
- ``min_track_section_length`` The minimum length of a curve or straight section
- ``max_track_section_length`` The maximum length of a curve or straight section

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
  - Hitting the track-bumpers is also a collision (so they narrow the track a bit)
  - Speed reduction depends on speed (25%)
    - Kickback: `track_collision_kickback_pixels`
    - Speed reduction: `track_collision_speed_reduction_per_speed_unit`
- Steering
  - Steering sensitivity depends on player speed.
  - By measuring the time it takes from one end to the other of a track the following function emerges:
    
    *time(speed) = 8 - 0.15s where s <= 32*
  
    *time(speed) = 4.86 - 0.567s where s > 32*
      - Config: `steering_range_in_pixels`
      - Config: `steering_sensitivity`
  - Steering sensitivity is halved during snowy weather (all white)
    - Config: ``steering_snow_factor``

### Opponent Cars
- Behavior
  - Enemies do not change lane position
  - They do have the same speed (24) that does not change
    - Config: ``opponent_speed``
  - Their relative speed to the player can be configured with:
    - ``opponent_relative_speed_factor``
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
- Collision
  - The original implementation has a pixel perfect collision
  - The cars at night are smaller than during day, only the car lights collide
  
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

## Limitations
### Integer vs. floats
The original game uses integer for speed, but since we need a frame based acceleration logic for the JAX implementation, this is not feasible. This has thw downside, that the player cannot really match the opponent speed (the float is always slightly above or below).
If this really represents an issue the speed could be encoded as an integer and the sub-speed could be implemented as a cooldown (wait n-steps before the next speed increase).

## Debug renderer
The file contains a [debug-renderer](../jax_enduro_debug_renderer.py) with an overlay to display state variables for 
debugging and analysis.

## RAM States
The following RAM states could be identified

| RAM field | Function                                | Values                                                            |
| --------: | --------------------------------------- | ----------------------------------------------------------------- |
|         0 |                                         |                                                                   |
|         1 |                                         |                                                                   |
|         2 |                                         |                                                                   |
|         3 | Car position 0 ?                        | default 4                                                         |
|         4 | Car position 1 ?                        | default 5                                                         |
|         5 | Car position 2 ?                        | default 6                                                         |
|         6 | Car position 3 ?                        | default 7                                                         |
|         7 | Car position 4 ?                        | default 8                                                         |
|         8 | Car position 5 ?                        | default 9                                                         |
|         9 | Car position 6 ?                        | default 10                                                        |
|        10 |                                         |                                                                   |
|        11 |                                         |                                                                   |
|        12 |                                         |                                                                   |
|        13 |                                         |                                                                   |
|        14 |                                         |                                                                   |
|        15 |                                         |                                                                   |
|        16 |                                         |                                                                   |
|        17 |                                         |                                                                   |
|        18 |                                         |                                                                   |
|        19 |                                         |                                                                   |
|        20 |                                         |                                                                   |
|        21 |                                         |                                                                   |
|        22 | Speed                                   | 6–120                                                             |
|        23 |                                         |                                                                   |
|        24 | ?                                       | −6 per frame                                                      |
|        25 |                                         |                                                                   |
|        26 |                                         |                                                                   |
|        27 | Car position 0 color                    |                                                                   |
|        28 | Car position 1 color                    |                                                                   |
|        29 | Car position 2 color                    |                                                                   |
|        30 | Car position 4 color                    |                                                                   |
|        31 | Car position 5 color                    |                                                                   |
|        32 | Car position 6 color                    |                                                                   |
|        33 | Car position 7 color                    |                                                                   |
|        34 | Track direction                         | 0 = straight, 50 = right, 205 = left                              |
|        35 | Kickback                                | Car = 160; Track boundary = 65–80; Speed reduction from wall = 15 |
|        36 | km of current track direction remaining | 0.05 km per number value                                          |
|        37 |                                         |                                                                   |
|        38 |                                         |                                                                   |
|        39 | Counts up 3 per frame                   | +3 per frame                                                      |
|        40 |                                         |                                                                   |
|        41 |                                         |                                                                   |
|        42 |                                         |                                                                   |
|        43 | Cars overtaken                          |                                                                   |
|        44 |                                         |                                                                   |
|        45 |                                         |                                                                   |
|        46 | Player car x position                   |                                                                   |
|        47 |                                         |                                                                   |
|        48 | Counts down 1 per frame                 |                                                                   |
|        49 | Counts down 1 per cycle of field 48     |                                                                   |
|        50 |                                         |                                                                   |
|        51 |                                         |                                                                   |
|        52 |                                         |                                                                   |
|        53 |                                         |                                                                   |
|        54 |                                         |                                                                   |
|        55 |                                         |                                                                   |
|        56 |                                         |                                                                   |
|        57 |                                         |                                                                   |
|        58 |                                         |                                                                   |
|        59 |                                         |                                                                   |
|        60 |                                         |                                                                   |
|        61 |                                         |                                                                   |
|        62 |                                         |                                                                   |
|        63 |                                         |                                                                   |
|        64 |                                         |                                                                   |
|        65 |                                         |                                                                   |
|        66 |                                         |                                                                   |
|        67 |                                         |                                                                   |
|        68 |                                         |                                                                   |
|        69 |                                         |                                                                   |
|        70 |                                         |                                                                   |
|        71 | Something with overtaking cars          |                                                                   |
|        72 | track                                   |                                                                   |
|        73 | track                                   |                                                                   |
|        74 | track                                   |                                                                   |
|        75 | track                                   |                                                                   |
|        76 | track                                   |                                                                   |
|        77 | track                                   |                                                                   |
|        78 | track                                   |                                                                   |
|        79 | track                                   |                                                                   |
|        80 | track                                   |                                                                   |
|        81 | track                                   |                                                                   |
|        82 | track                                   |                                                                   |
|        83 | track                                   |                                                                   |
|        84 | track                                   |                                                                   |
|        85 | track                                   |                                                                   |
|        86 | track                                   |                                                                   |
|        87 | track                                   |                                                                   |
|        88 | track                                   |                                                                   |
|        89 | track                                   |                                                                   |
|        90 | track                                   |                                                                   |
|        91 | track                                   |                                                                   |
|        92 | track                                   |                                                                   |
|        93 | track                                   |                                                                   |
|        94 | track                                   |                                                                   |
|        95 | track                                   |                                                                   |
|        96 | track                                   |                                                                   |
|        97 | track                                   |                                                                   |
|        98 | track                                   |                                                                   |
|        99 | track                                   |                                                                   |
|       100 | track                                   |                                                                   |
|       101 | track                                   |                                                                   |
|       102 | track                                   |                                                                   |
|       103 |                                         |                                                                   |
|       104 |                                         |                                                                   |
|       105 |                                         |                                                                   |
|       106 |                                         |                                                                   |
|       107 |                                         |                                                                   |
|       108 |                                         |                                                                   |
|       109 |                                         |                                                                   |
|       110 |                                         |                                                                   |
|       111 |                                         |                                                                   |
|       112 |                                         |                                                                   |
|       113 |                                         |                                                                   |
|       114 |                                         |                                                                   |
|       115 |                                         |                                                                   |
|       116 |                                         |                                                                   |
|       117 |                                         |                                                                   |
|       118 |                                         |                                                                   |
|       119 |                                         |                                                                   |
|       120 |                                         |                                                                   |
|       121 |                                         |                                                                   |
|       122 |                                         |                                                                   |
|       123 |                                         |                                                                   |
|       124 |                                         |                                                                   |
|       125 |                                         |                                                                   |
|       126 |                                         |                                                                   |
|       127 |                                         |                                                                   |

