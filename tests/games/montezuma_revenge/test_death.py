import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge

def test_death_by_falling():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Use Room 12 (index 5)
    # Middle platform is at y=26 (feet at 45). Next floor is at 88 (feet at 87).
    # Distance = 87 - 20 + 1 - 26 = 68 - 26 = 42. 42 > 33.
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    
    initial_lives = state.lives
    
    # Place player in the air off the middle platform in room 12
    # Middle platform is roughly 60 to 100 in X.
    # Place at x=50.
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(26, dtype=jnp.int32),
        fall_distance=jnp.array(0, dtype=jnp.int32),
        is_falling=jnp.array(1, dtype=jnp.int32)
    )
    
    # Step until hitting the floor (or dying)
    for _ in range(100):
        obs, state, reward, done, info = env.step(state, 0) # NOOP
        if state.death_timer > 0:
            break
            
    assert state.death_timer == env.consts.DEATH_TIMER_FRAMES
    assert state.death_type == 1 # died_from_fall
    assert state.lives == initial_lives - 1

def test_death_by_enemy():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Place an enemy and player at the same position
    # Use Room 12, but ensure enemy is active
    state = state.replace(
        player_x=jnp.array(50, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32),
        enemies_x=state.enemies_x.at[0].set(50),
        enemies_y=state.enemies_y.at[0].set(100),
        enemies_active=state.enemies_active.at[0].set(1),
        enemies_type=state.enemies_type.at[0].set(1) # Skull
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    assert state.death_timer == env.consts.DEATH_TIMER_FRAMES
    assert state.death_type == 2 # died_from_enemy
    assert state.lives == initial_lives - 1

def test_death_by_laser():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    initial_lives = state.lives
    
    # Laser is active when laser_cycle is [0, 92)
    # Laser room is room 14
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(14, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(40, dtype=jnp.int32),
        player_y=jnp.array(20, dtype=jnp.int32),
        lasers_x=state.lasers_x.at[0].set(40),
        lasers_active=state.lasers_active.at[0].set(1),
        laser_cycle=jnp.array(0, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    assert state.death_timer == env.consts.DEATH_TIMER_FRAMES
    assert state.death_type == 3 # died_from_laser
    assert state.lives == initial_lives - 1

def test_respawn_after_death():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Trigger death with custom entry coordinates
    state = state.replace(
        entry_x=jnp.array(35, dtype=jnp.int32),
        entry_y=jnp.array(60, dtype=jnp.int32),
        entry_is_climbing=jnp.array(1, dtype=jnp.int32),
        entry_last_ladder=jnp.array(0, dtype=jnp.int32),
        death_timer=jnp.array(1, dtype=jnp.int32),
        death_type=jnp.array(1, dtype=jnp.int32)
    )
    
    # One more step to respawn
    obs, state, reward, done, info = env.step(state, 0)
    
    assert state.death_timer == 0
    assert state.player_x == 35
    assert state.player_y == 60
    assert state.is_climbing == 1
    assert state.last_ladder == 0
