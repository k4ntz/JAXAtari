import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2

def test_conveyor_movement():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a conveyor at y=88 (surface), x=60, direction 1.
    # Feet at 87 -> player_y = 87 - 20 + 1 = 68.
    from jaxatari.games.montezuma2.rooms import load_room
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
        
    assert state.player_x > initial_x
    assert state.player_y == 68

def test_jump_hit_ceiling():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4 has a platform at y=48.
    # Player starts jumping from below it, say feet at 65 -> y=46
    from jaxatari.games.montezuma2.rooms import load_room
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    state = load_room(state.room_id, state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32),
        player_y=jnp.array(50, dtype=jnp.int32)
    )
    
    # Jump (UPFIRE = 10 or just UP = 2 if in air? No, FIRE = 1 if on ground.
    # To start a jump, is_jumping can be forced to 1.
    state = state.replace(
        is_jumping=jnp.array(1, dtype=jnp.int32),
        jump_counter=jnp.array(0, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0)
    
    # Should have hit the ceiling at y=48 (player_y=48 or 49)
    # The jump dy is usually negative, but hit_ceiling should cancel it.
    assert state.is_jumping == 0
    assert state.player_y >= 48

def test_wall_collision():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 5 has a right wall at x=156
    from jaxatari.games.montezuma2.rooms import load_room
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
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Room 4, ladder at x=72.
    from jaxatari.games.montezuma2.rooms import load_room
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
