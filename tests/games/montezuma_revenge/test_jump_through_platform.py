import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezumarevenge import JaxMontezumaRevenge

def test_jump_through_dynamic_platform():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Place player on platform at y=76 (Platform index 2 in ROOM_2_1)
    # player_y = 76 - 1 (to be on top) - 19 (height offset) = 56
    state = state.replace(
        room_id=jnp.array(17, dtype=jnp.int32),
        player_x=jnp.array(4, dtype=jnp.int32),
        player_y=jnp.array(56, dtype=jnp.int32), # Feet at 75
        fall_distance=jnp.array(0, dtype=jnp.int32),
        is_falling=jnp.array(0, dtype=jnp.int32),
        is_jumping=jnp.array(0, dtype=jnp.int32)
    )
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(17, dtype=jnp.int32), state, env.consts)

    # Ensure we are on platform 2 (y=76)
    assert state.platforms_y[2] == 76

    # Jump UP (Action 1 is FIRE, which triggers jump if on ground)
    obs, state, reward, done, info = env.step(state, 1)
    assert state.is_jumping == 1

    # We want to jump through platform 1 (y=66).
    # We start at feet=75.
    # Frame 0: dy=-3 -> feet=72
    # Frame 1: dy=-3 -> feet=69
    # Frame 2: dy=-3 -> feet=66
    # Frame 3: dy=-2 -> feet=64
    # At Frame 3, feet cross y=66. If not semi-permeable, hit_ceiling would stop the jump.

    for _ in range(4):
        obs, state, reward, done, info = env.step(state, 0)

    # Should be at player_y = 56 - 13 = 43 (feet at 62)
    assert state.player_y == 43
    assert state.is_jumping == 1

    # Now wait until landing. Should land on platform 1 (y=66), so player_y = 66 - 1 - 19 = 46.
    landed = False
    for _ in range(50):
        obs, state, reward, done, info = env.step(state, 0)
        if state.is_jumping == 0 and state.is_falling == 0:
            landed = True
            break

    assert landed
    assert state.player_y == 46
    assert state.player_y + 19 == 65 # Feet just above platform 1

def test_jump_through_static_platform():
    env = JaxMontezumaRevenge()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Room 17 has static platforms at y=46, 47, 48.
    # Place player on dynamic platform at y=56 (Platform 0 in ROOM_2_1)
    # player_y = 56 - 1 - 19 = 36 (Feet at 55)
    state = state.replace(
        room_id=jnp.array(17, dtype=jnp.int32),
        player_x=jnp.array(10, dtype=jnp.int32),
        player_y=jnp.array(36, dtype=jnp.int32),
        fall_distance=jnp.array(0, dtype=jnp.int32),
        is_falling=jnp.array(0, dtype=jnp.int32),
        is_jumping=jnp.array(0, dtype=jnp.int32)
    )
    from jaxatari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(17, dtype=jnp.int32), state, env.consts)

    # Jump UP
    obs, state, reward, done, info = env.step(state, 1)
    assert state.is_jumping == 1

    # Wait until apex or until we land
    landed = False
    for _ in range(50):
        obs, state, reward, done, info = env.step(state, 0)
        if state.is_jumping == 0 and state.is_falling == 0:
            landed = True
            break

    assert landed
    # We should land on the static platform top surface, which is y=46.
    # Snapped y: 46 - 20 = 26.
    assert state.player_y == 26
    assert state.player_y + 19 == 45 # Feet just above y=46
