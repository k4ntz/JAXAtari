import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2
from jaxatari.games.montezuma2.rooms import load_room

def test_room_transitions_horizontal():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # 1. Room 4 -> 3 (Left)
    state = load_room(jnp.array(4, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(0, dtype=jnp.int32), player_y=jnp.array(100, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 4) # LEFT
    assert state.room_id == 3
    assert state.player_x >= 140 # Should be teleported to the right side of new room
    
    # 2. Room 3 -> 4 (Right)
    state = load_room(jnp.array(3, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(155, dtype=jnp.int32), player_y=jnp.array(100, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 3) # RIGHT
    assert state.room_id == 4
    assert state.player_x <= 10 # Should be teleported to the left side of new room

    # 3. Room 4 -> 5 (Right)
    state = load_room(jnp.array(4, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(155, dtype=jnp.int32), player_y=jnp.array(100, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 3) # RIGHT
    assert state.room_id == 5
    assert state.player_x <= 10
    
    # 4. Room 12 -> 11 (Left)
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(0, dtype=jnp.int32), player_y=jnp.array(100, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 4) # LEFT
    assert state.room_id == 11
    assert state.player_x >= 140

def test_room_transitions_vertical():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # 1. Room 4 -> 12 (Down)
    # Room 4 has a hole for ladder down at x=72:88, y=148
    state = load_room(jnp.array(4, dtype=jnp.int32), state, env.consts)
    # transition_down triggers when new_y >= 148
    state = state.replace(player_x=jnp.array(77, dtype=jnp.int32), player_y=jnp.array(147, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 5) # DOWN
    assert state.room_id == 12
    assert state.player_y <= 20 # Should be at the top of the new room
    
    # 2. Room 12 -> 4 (Up)
    state = load_room(jnp.array(12, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(77, dtype=jnp.int32), player_y=jnp.array(0, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 0) # NOOP (will fall to y=2)
    assert state.room_id == 4
    assert state.player_y >= 130 # Should be at the bottom of the new room

    # 3. Room 10 -> 18 (Down)
    state = load_room(jnp.array(10, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(77, dtype=jnp.int32), player_y=jnp.array(147, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 5) # DOWN
    assert state.room_id == 18
    
    # 4. Room 18 -> 10 (Up)
    state = load_room(jnp.array(18, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(77, dtype=jnp.int32), player_y=jnp.array(0, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 0) # NOOP (will fall to y=2)
    assert state.room_id == 10

    # 5. Room 11 (ROOM_1_3) -> 19 (ROOM_2_3) (Down)
    state = load_room(jnp.array(11, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(77, dtype=jnp.int32), player_y=jnp.array(147, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 5) # DOWN
    assert state.room_id == 19
    assert state.player_y <= 20
    
    # 6. Room 19 (ROOM_2_3) -> 11 (ROOM_1_3) (Up)
    state = load_room(jnp.array(19, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(77, dtype=jnp.int32), player_y=jnp.array(3, dtype=jnp.int32),
                          is_climbing=jnp.array(1, dtype=jnp.int32), last_ladder=jnp.array(0, dtype=jnp.int32))
    obs, state, reward, done, info = env.step(state, 2) # UP
    assert state.room_id == 11
    assert state.player_y >= 130

def test_all_implemented_rooms_loadable():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    
    implemented_room_ids = [3, 4, 5, 10, 11, 12, 13, 14, 18, 19]
    
    for rid in implemented_room_ids:
        new_state = load_room(jnp.array(rid, dtype=jnp.int32), state, env.consts)
        assert new_state.room_id == rid
        # Render to ensure no crashes
        frame = env.render(new_state)
        assert frame.shape == (210, 160, 3)
