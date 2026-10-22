import jax
import jax.numpy as jnp
import pytest
from jaxtari.games.montezuma_revenge.rooms import load_room


def _assert_room_change(env, state, action, target, entry_x, entry_y):
    old_position = (int(state.player_x), int(state.player_y))
    _, state, reward, done, _ = env.step(state, action)

    # ALE changes the room in mode 2 while retaining the old player anchor.
    assert (int(state.room_id), int(state.native_mode)) == (target, 2)
    assert (int(state.player_x), int(state.player_y)) == old_position
    assert not done

    # The next native frame initializes the entry position and resumes mode 4.
    _, state, reward, done, _ = env.step(state, 0)
    assert (int(state.room_id), int(state.native_mode)) == (target, 4)
    assert (int(state.player_x), int(state.player_y)) == (entry_x, entry_y)
    assert not done


@pytest.mark.parametrize("source,target,action,start_x,entry_x", [
    (4, 3, 4, 4, 151),
    (3, 4, 3, 148, 0),
    (4, 5, 3, 148, 0),
    (12, 11, 4, 4, 151),
])
def test_room_transitions_horizontal(montezuma_env, source, target, action, start_x, entry_x):
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.array(source, dtype=jnp.int32), state, env.consts)
    # Native side exits are in the upper corridor, at internal y=26.
    state = state.replace(player_x=jnp.array(start_x, dtype=jnp.int32),
                          player_y=jnp.array(26, dtype=jnp.int32))
    _assert_room_change(env, state, action, target, entry_x, 26)


@pytest.mark.parametrize("source,target,action,start_y,entry_y", [
    # ALE's ladder connects rooms 5 and 13, rather than rooms 4 and 12.
    (5, 13, 5, 127, 7),
    (13, 5, 2, 7, 127),
    (10, 18, 5, 127, 7),
    (18, 10, 2, 7, 127),
    (11, 19, 5, 127, 7),
    (19, 11, 2, 7, 127),
])
def test_room_transitions_vertical(montezuma_env, source, target, action, start_y, entry_y):
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.array(source, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(76, dtype=jnp.int32),
                          player_y=jnp.array(start_y, dtype=jnp.int32),
                          is_climbing=jnp.array(1, dtype=jnp.int32),
                          last_ladder=jnp.array(0, dtype=jnp.int32))
    _assert_room_change(env, state, action, target, 76, entry_y)


@pytest.mark.parametrize("source,target,action,start_x,entry_x", [
    (31, 32, 3, 148, 0),
    (32, 31, 4, 4, 151),
    (31, 30, 4, 4, 151),
    (30, 31, 3, 148, 0),
])
def test_room_3_8_transition(montezuma_env, source, target, action, start_x, entry_x):
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.array(source, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(start_x, dtype=jnp.int32),
                          player_y=jnp.array(26, dtype=jnp.int32))
    _assert_room_change(env, state, action, target, entry_x, 26)


@pytest.mark.parametrize("source,target,action,start_y,entry_y", [
    (22, 30, 5, 127, 7),
    (30, 22, 2, 7, 127),
])
def test_room_2_6_to_3_6_transition(montezuma_env, source, target, action, start_y, entry_y):
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.array(source, dtype=jnp.int32), state, env.consts)
    state = state.replace(player_x=jnp.array(76, dtype=jnp.int32),
                          player_y=jnp.array(start_y, dtype=jnp.int32),
                          is_climbing=jnp.array(1, dtype=jnp.int32),
                          last_ladder=jnp.array(0, dtype=jnp.int32))
    _assert_room_change(env, state, action, target, 76, entry_y)


def test_all_implemented_rooms_loadable(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    
    implemented_room_ids = [3, 4, 5, 10, 11, 12, 13, 14, 18, 17, 19, 20, 21, 22, 23, 30, 31, 32]
    
    for rid in implemented_room_ids:
        new_state = load_room(jnp.array(rid, dtype=jnp.int32), state, env.consts)
        assert new_state.room_id == rid
        # Render to ensure no crashes
        frame = env.render(new_state)
        assert frame.shape == (210, 160, 3)
