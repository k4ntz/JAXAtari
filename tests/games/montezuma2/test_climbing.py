import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2

def test_climb_ladder():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a ladder at x=72, top=49, bottom=88
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxatari.games.montezuma2.rooms import load_room
    state = load_room(state.room_id, state, env.consts)
    
    # Place player at the bottom of the ladder
    # Player height is 20. Feet at 88 means y = 88 - 20 + 1 = 69
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32), # mid is 80, ladder x=72, ladder width is 16. Mid is 80.
        player_y=jnp.array(69, dtype=jnp.int32)
    )
    
    # UP action is 2
    UP_ACTION = 2
    
    # Step UP to catch the ladder
    obs, state, reward, done, info = env.step(state, UP_ACTION)
    assert state.is_climbing == 1
    
    # Climb up
    initial_y = state.player_y
    obs, state, reward, done, info = env.step(state, UP_ACTION)
    assert state.player_y < initial_y
    
    # DOWN action is 5
    DOWN_ACTION = 5
    obs, state, reward, done, info = env.step(state, DOWN_ACTION)
    assert state.player_y == initial_y
    
def test_climb_rope():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a rope at x=111, top=49, bottom=88
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxatari.games.montezuma2.rooms import load_room
    state = load_room(state.room_id, state, env.consts)
    
    # Place player at the rope
    # Player width is 7. Player mid should be around rope_x=111.
    # 111 - 3 = 108
    state = state.replace(
        player_x=jnp.array(108, dtype=jnp.int32),
        player_y=jnp.array(60, dtype=jnp.int32)
    )
    
    # Catch the rope
    obs, state, reward, done, info = env.step(state, 2) # UP
        
    assert state.is_climbing == 1
    
    # Climb up
    initial_y = state.player_y
    obs, state, reward, done, info = env.step(state, 2) # UP
    assert state.player_y < initial_y

def test_exit_ladder_onto_platform():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4, ladder at x=72, top=49.
    # Platform is at Y=46..48.
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxatari.games.montezuma2.rooms import load_room
    state = load_room(state.room_id, state, env.consts)
    
    # Place player on ladder near top
    # Ground at 46. Feet should be at 45.
    # y = 45 - 20 + 1 = 26
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32),
        player_y=jnp.array(26, dtype=jnp.int32),
        is_climbing=jnp.array(1, dtype=jnp.int32)
    )
    
    # Move RIGHT (3) to exit ladder
    obs, state, reward, done, info = env.step(state, 3)
    
    # Should no longer be climbing and should be on the platform
    assert state.is_climbing == 0
    assert state.is_falling == 0
    assert state.player_y == 26
