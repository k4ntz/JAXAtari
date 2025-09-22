# KeystoneKapers JAX Implementation

## Overview

This is a complete JAX-based implementation of the classic Atari 2600 game **KeystoneKapers**. The implementation follows functional programming principles with JAX optimizations for GPU acceleration.

## Game Description

KeystoneKapers is a department store chase game where Officer Kelly (the player) pursues a thief through a multi-floor building. The player must navigate through 3 floors plus a roof using elevators and escalators while avoiding obstacles and collecting items, all within a time limit.

## Key Features

### Architecture
- **Functional Programming**: Pure functions with immutable state using JAX arrays
- **GPU Acceleration**: JIT compilation with `@partial(jax.jit)` for performance
- **Vectorized Operations**: Efficient collision detection and obstacle management
- **Modular Design**: Separate concerns for physics, AI, rendering, and game logic

### Game Mechanics
- **Multi-floor Navigation**: 3 playable floors + roof with escalators and elevators
- **Dynamic Obstacles**: Shopping carts, bouncing balls, and toy planes with level-based scaling
- **Thief AI**: Intelligent thief movement with escalator usage and escape behavior
- **Timer System**: Countdown timer with collision penalties and time bonuses
- **Scoring**: Points for catching thief, collecting items, and time bonuses
- **Difficulty Progression**: Level-based speed and spawn rate scaling

### Technical Implementation

#### State Management
```python
GameState = NamedTuple containing:
├── PlayerState (position, floor, jumping, elevator/escalator usage)
├── ThiefState (position, AI behavior, escape status)
├── ObstacleState (vectorized obstacles: carts, balls, planes)
├── ElevatorState (position, timing, open/closed state)
└── Game Variables (score, lives, level, timer, items)
```

#### JAX Optimizations
- `@partial(jax.jit)` compilation for all performance-critical functions
- Vectorized collision detection using `jax.vmap`
- Functional state updates without side effects
- Efficient array operations for obstacle management

#### Game Constants
All game parameters are configurable via `KeystoneKapersConstants`:
- Floor positions and dimensions
- Movement speeds and scaling factors
- Obstacle spawn rates and behavior
- Timing and scoring parameters
- Visual colors and entity sizes

## Usage

### Basic Usage
```python
from jaxatari.games.jax_keystonekapers import JaxKeystoneKapers

# Create game instance
game = JaxKeystoneKapers()

# Initialize game
key = jax.random.PRNGKey(42)
observation, state = game.reset(key)

# Game loop
while not done:
    observation, state, reward, done, info = game.step(state, action)
    rendered_frame = game.render(state)
```

### With JAXAtari Play Script
```bash
python scripts/play.py -g keystonekapers
```

### Custom Configuration
```python
from jaxatari.games.jax_keystonekapers import JaxKeystoneKapers, KeystoneKapersConstants

# Custom game parameters
custom_consts = KeystoneKapersConstants(
    PLAYER_SPEED=3,           # Faster player
    THIEF_BASE_SPEED=1.5,     # Faster thief
    BASE_TIMER=4800,          # Longer time limit (80 seconds)
    MAX_OBSTACLES=12,         # More obstacles
    COLLISION_PENALTY=180     # Smaller time penalty (3 seconds)
)

game = JaxKeystoneKapers(consts=custom_consts)
```

## Game Rules Implementation

### Player Controls
- **Movement**: LEFT/RIGHT for horizontal movement
- **Floors**: UP/DOWN on escalators and elevators to change floors
- **Jumping**: FIRE button to jump over obstacles (provides temporary invulnerability)
- **Combined**: LEFTFIRE/RIGHTFIRE for diagonal movement with jumping

### Thief Behavior
- Starts on top floor, moves horizontally toward escape
- Uses escalators when moving right and available
- Speed increases with each level: `base_speed * (1 + 0.10 * level)`
- Escapes when reaching the roof area

### Obstacles
1. **Shopping Carts** (Floors 1-3)
   - Horizontal movement across floors
   - Spawn interval: 1.5-3.0 seconds (decreases with level)
   - Speed scales with level

2. **Bouncing Balls** (Floors 2-3)
   - Horizontal movement with vertical bouncing
   - Bounce physics with gravity simulation
   - Spawn interval: 2.0-4.0 seconds

3. **Toy Planes** (Floor 3 primarily)
   - Fast horizontal movement
   - Spawn interval: 3.0-6.0 seconds
   - Highest speed increase per level

### Scoring System
- **Catch Thief**: 3000 points + time bonus (remaining_time * 50)
- **Collect Items**: 100 points each
- **Jump Over Obstacles**: 50 points
- **Extra Life**: Every 10,000 points

### Level Progression
- **Thief Speed**: Increases by 10% per level (capped)
- **Obstacle Spawn Rate**: Increases by 12% per level
- **Obstacle Speed**: Varies by type (8-12% increase per level)
- **Timer**: Optional reduction of 5 seconds per level

## Collision Detection

The implementation uses efficient vectorized collision detection:

```python
# Player-obstacle collisions (vectorized)
cart_collisions = jax.vmap(collision_function)(obstacle_positions)
obstacle_hit = jnp.any(cart_collisions)

# Player has invulnerability while jumping
collision_applies = jnp.logical_not(player.is_jumping)
```

## Rendering

The renderer creates a simple but functional visual representation:
- **Background**: Solid color with floor platforms
- **Escalators/Elevators**: Fixed position structural elements
- **Entities**: Colored rectangles for player, thief, obstacles, items
- **HUD**: Score, timer, lives display (simplified)

For production use, sprite-based rendering can be implemented by:
1. Creating `.npy` sprite files in `sprites/keystonekapers/`
2. Loading sprites in `KeystoneKapersRenderer.__init__`
3. Using `jr.render_at()` for sprite positioning

## Testing and Validation

The implementation includes comprehensive testing:
- **Structure Validation**: Ensures proper JAXAtari patterns
- **Functional Testing**: Verifies game mechanics work correctly
- **Performance Testing**: Confirms JAX JIT compilation
- **Integration Testing**: Full episode simulation

Run tests with:
```bash
python validate_keystonekapers.py  # Structure validation
python test_keystonekapers.py      # Full functional testing (requires JAX)
```

## Compatibility

This implementation is fully compatible with:
- **JAXAtari Framework**: Inherits from `JaxEnvironment` with proper interfaces
- **Gymnasium/ALE API**: Standard observation/action spaces
- **JAX Ecosystem**: JIT compilation, vectorization, GPU acceleration
- **Reinforcement Learning**: Standard reward/observation patterns

## Performance Considerations

- **JIT Compilation**: All critical functions are JIT-compiled for speed
- **Memory Efficiency**: Uses fixed-size arrays to avoid dynamic allocation
- **Vectorization**: Obstacle updates and collisions are vectorized
- **GPU Ready**: All operations use JAX arrays compatible with GPU execution

## Extension Points

The modular design allows easy extension:
- **Custom Reward Functions**: Pass `reward_funcs` to constructor
- **Modified Game Rules**: Subclass and override specific methods
- **Enhanced AI**: Improve thief behavior in `_update_thief`
- **Visual Improvements**: Enhance renderer with proper sprites
- **Additional Obstacles**: Extend `ObstacleState` and spawning logic

## Implementation Compliance

This implementation strictly follows the requirements:
- ✅ **JAX Functional Programming**: Pure functions, immutable state
- ✅ **GPU Acceleration**: JIT compilation and JAX arrays
- ✅ **Game Mechanics**: All specified mechanics implemented
- ✅ **Difficulty Scaling**: Level-based progression with specified formulas
- ✅ **JAXAtari Compatibility**: Proper inheritance and interface implementation
- ✅ **Modular Design**: Clean separation of concerns
- ✅ **Performance Optimization**: Vectorization and JIT where appropriate
- ✅ **Configurable Parameters**: All constants easily adjustable

The implementation provides a solid foundation for both research and gaming applications while maintaining high performance through JAX optimizations.
