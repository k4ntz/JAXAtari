import jax
import jax.numpy as jnp
from jaxatari.games.jax_montezuma2 import JaxMontezuma2

def test_climb_down_ladder_from_platform():
    env = JaxMontezuma2()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Room 17 has a ladder at x=20 (wait, let's find a ladder)
    # Let's check rooms.py for Room 2_1 (Room 17) ladders.
    # Ah, load_room_2_1 doesn't have ladders!
    # Let's use Room 4 (ROOM_0_4).
    # lx = 72, lt = 49, lb = 88.
    
    state = state.replace(
        room_id=jnp.array(4, dtype=jnp.int32),
        player_x=jnp.array(72 - 4, dtype=jnp.int32), # Aligned with ladder (l_x + 8 - PLAYER_WIDTH//2 = 72 + 8 - 3 = 77? Wait. Ladder mid x is l_x + 8. player mid x is player_x + 3. 72+8-3=77)
        player_y=jnp.array(29, dtype=jnp.int32), # Feet at 48. Ladder top is 49.
        is_falling=jnp.array(0, dtype=jnp.int32),
        is_jumping=jnp.array(0, dtype=jnp.int32)
    )
    from jaxatari.games.montezuma2.rooms import load_room
    state = load_room(jnp.array(4, dtype=jnp.int32), state, env.consts)

    # Check that ladder 0 is at x=72, top=49
    assert state.ladders_x[0] == 72
    assert state.ladders_top[0] == 49
    
    # Move player to exact alignment
    ladder_mid_x = 72 + 8
    player_x = ladder_mid_x - 3
    state = state.replace(player_x=jnp.array(player_x, dtype=jnp.int32))

    # Action 5 is DOWN
    obs, state, reward, done, info = env.step(state, 5)
    
    # Should start climbing
    assert state.is_climbing == 1
    assert state.player_y == 31 # Moved down 2 pixels (PLAYER_SPEED=2)

    # Move down again
    obs, state, reward, done, info = env.step(state, 5)
    assert state.is_climbing == 1
    assert state.player_y == 33 # Moved down 2 more pixels

if __name__ == "__main__":
    test_climb_down_ladder_from_platform()
