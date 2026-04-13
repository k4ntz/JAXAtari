import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge

def test_conveyor_movement():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a conveyor at y=88 (surface), x=60, direction 1.
    # Feet at 87 -> player_y = 87 - 20 + 1 = 68.
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(65, dtype=jnp.int32),
        player_y=jnp.array(68, dtype=jnp.int32)
    )
    
    initial_x = state.player_x
    
    # Step NOOP several times to see conveyor movement
    # Conveyor moves 1 pixel every 2 frames
    for _ in range(4):
        obs, state, reward, done, info = env.step(state, 0)
        
    assert state.player_x < initial_x
    assert state.player_y == 68

def test_wall_collision():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 5 has a right wall at x=156
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(5, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    # Place player near the wall
    # Player width is 7. 156 - 7 = 149.
    state = state.replace(
        player_x=jnp.array(148, dtype=jnp.int32),
        player_y=jnp.array(100, dtype=jnp.int32)
    )
    
    # Move RIGHT
    obs, state, reward, done, info = env.step(state, 3)
    
    # Player_x shouldn't exceed 149
    assert state.player_x <= 149

def test_jump_off_ladder():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4, ladder at x=72.
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32),
        player_y=jnp.array(55, dtype=jnp.int32),
        is_climbing=jnp.array(1, dtype=jnp.int32)
    )
    
    # RightFire action is 11
    RIGHTFIRE = 11
    
    obs, state, reward, done, info = env.step(state, RIGHTFIRE)
    
    # Should abort the ladder, but NOT jump
    assert state.is_climbing == 0
    assert state.is_jumping == 0
    # On the next step, player should be falling
    obs, state, reward, done, info = env.step(state, 0)
    assert state.is_falling == 1

def test_transition_landing_overlap():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Start in Room 18 (ROOM_2_2)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(18, jnp.int32))
    state = load_room(jnp.array(18, jnp.int32), state, env.consts)
    
    # Position player at the left edge, about to transition to Room 17 (ROOM_2_1)
    # Floor in Room 17 at px=144 is py=56.
    # Let's put the player at py=56 - 19 = 37. Feet at 56 (inside the platform).
    # Transition should push them up to py=36 (feet at 55).
    state = state.replace(player_x=jnp.array(0, jnp.int32), player_y=jnp.array(37, jnp.int32))
    
    # Action LEFT (4) to trigger transition
    # JAXAtariAction.LEFT is 4
    obs, state, r, d, i = env.step(state, 4)
    
    assert state.room_id == 17
    assert state.player_x == 148
    # If the fix works, it should have pushed the player up to 36.
    # If it didn't, they'd stay at 37.
    assert state.player_y == 36
    assert state.player_y + 19 == 55 # Feet at 55, platform at 56.

def test_jump_descent_overlap():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 19 (ROOM_2_3)
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = state.replace(room_id=jnp.array(19, jnp.int32))
    state = load_room(jnp.array(19, jnp.int32), state, env.consts)
    
    # Fixed platform on the right side. 
    # Let's find a solid pixel in the room_col_map.
    # In room_col_2_3, there should be a platform around y=48.
    
    room_idx = jnp.where(state.room_id == 19, 10, 0)
    room_col_map = env.ROOM_COLLISION_MAPS[room_idx]
    
    # Let's find the floor at x=140.
    floor_y = -1
    for y in range(40, 140):
        if room_col_map[y, 140] == 1 and room_col_map[y-1, 140] == 0:
            floor_y = y
            break
    
    assert floor_y != -1, "Could not find floor in ROOM_2_3"
    
    # Position player above the floor, in a jump descent with dy=3.
    # JUMP_Y_OFFSETS: [3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3]
    # Index 17 is -3. So we need jump_counter = 17.
    state = state.replace(
        player_x=jnp.array(140 - 3, jnp.int32), 
        player_y=jnp.array(floor_y - 19 - 3, jnp.int32), # 3 pixels above where they'd be on the floor
        is_jumping=jnp.array(1, jnp.int32),
        jump_counter=jnp.array(17, jnp.int32)
    )
    
    # One step should land them 1 pixel above floor_y.
    # Before the fix (with loop limit 3), they would land AT floor_y (overlap).
    # With the fix (limit 5), they should land at floor_y - 1.
    obs, state, r, d, i = env.step(state, 0)
    
    assert state.player_y + 19 == floor_y - 1
    assert state.is_jumping == 0
