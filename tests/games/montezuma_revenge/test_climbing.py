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
    
    # Room 4 has a rope at x=111, top=49, bottom=88
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxtari.games.montezuma_revenge.rooms import load_room
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

def test_no_drop_ladder_onto_platform(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Room 4 (load_room_0_4): ladder[0] at x=72, top=50, bottom=88.
    # Horizontal input must not disengage climbing while still in the ladder zone.
    state = state.replace(room_id=jnp.array(4, dtype=jnp.int32))
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = load_room(state.room_id, state, env.consts)

    # Place player on that ladder mid-climb (feet ≈ 60 → y = 60 - 20 + 1 = 41).
    # Mid-x of ladder is 72+8=80; player_x=77 centers width-7 on the ladder.
    player_y = jnp.array(41, dtype=jnp.int32)
    state = state.replace(
        player_x=jnp.array(77, dtype=jnp.int32),
        player_y=player_y,
        is_climbing=jnp.array(1, dtype=jnp.int32),
        last_ladder=jnp.array(0, dtype=jnp.int32),
    )

    # Move RIGHT (3): must stay climbing (ladders are vertical-only).
    obs, state, reward, done, info = env.step(state, 3)

    assert state.is_climbing == 1
    assert state.is_falling == 0
    assert state.player_y == player_y
