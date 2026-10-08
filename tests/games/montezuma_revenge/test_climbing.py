import jax
import jax.numpy as jnp

def test_climb_ladder(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a ladder at x=72, top=49, bottom=88
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxtari.games.montezuma_revenge.rooms import load_room
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
    
def test_climb_rope(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a rope at x=112, top=49, bottom=88
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = load_room(state.room_id, state, env.consts)
    
    # Place player at the rope
    # Player width is 7. Player mid should be around rope_x=112.
    # 112 - 3 = 109
    state = state.replace(
        player_x=jnp.array(109, dtype=jnp.int32),
        player_y=jnp.array(60, dtype=jnp.int32)
    )
    
    # Catch the rope
    obs, state, reward, done, info = env.step(state, 2) # UP
        
    assert state.is_climbing == 1
    
    # Climb up
    initial_y = state.player_y
    obs, state, reward, done, info = env.step(state, 2) # UP
    assert state.player_y < initial_y

def test_no_drop_ladder_onto_platform(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Room 4 (load_room_0_4): ladder[0] at x=72, top=50, bottom=88.
    # Horizontal input must not disengage climbing while still in the ladder zone.
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = load_room(state.room_id, state, env.consts)
    
    # Enter the ladder from the platform. This is an actual attached pose;
    # an injected y=26 pose lies above the ladder zone and has already exited.
    obs, state, reward, done, info = env.step(state, 5)  # DOWN
    assert state.is_climbing == 1
    initial_y = state.player_y
    
    # Move RIGHT (3) to do nothing 
    obs, state, reward, done, info = env.step(state, 3)

    assert state.is_climbing == 1
    assert state.is_falling == 0
    assert state.player_y == initial_y
