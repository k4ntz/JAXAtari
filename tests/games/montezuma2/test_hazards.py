import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2

def test_laser_inactive_cycle():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Laser is inactive when laser_cycle >= 92
    # Laser room is room 14
    from jaxatari.games.montezuma2.rooms import load_room
    state = load_room(jnp.array(14, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(40, dtype=jnp.int32),
        player_y=jnp.array(20, dtype=jnp.int32),
        lasers_x=state.lasers_x.at[0].set(40),
        lasers_active=state.lasers_active.at[0].set(1),
        laser_cycle=jnp.array(95, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Laser is inactive, player should not die
    assert state.death_timer == 0

def test_ladder_delay():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4, ladder at x=72.
    from jaxatari.games.montezuma2.rooms import load_room
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    # Place player on ladder, previously climbing
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32),
        player_y=jnp.array(55, dtype=jnp.int32),
        is_climbing=jnp.array(1, dtype=jnp.int32)
    )
    
    # Jump off to trigger delay
    RIGHTFIRE = 11
    obs, state, reward, done, info = env.step(state, RIGHTFIRE)
    
    assert state.out_of_ladder_delay == env.consts.OUT_OF_LADDER_DELAY
    
    # Try to catch ladder immediately with UP
    obs, state, reward, done, info = env.step(state, 2)
    
    # Delay should decrease, but climbing is blocked
    assert state.out_of_ladder_delay == env.consts.OUT_OF_LADDER_DELAY - 1
    assert state.is_climbing == 0
