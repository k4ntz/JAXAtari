import jax
import jax.numpy as jnp
import pytest

def test_laser_inactive_cycle(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    
    # Laser is inactive when laser_cycle >= 92
    # Laser room is room 14
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(14, dtype=jnp.int32), state, env.consts)
    
    state = state.replace(
        player_x=jnp.array(40, dtype=jnp.int32),
        player_y=jnp.array(20, dtype=jnp.int32),
        lasers_x=state.lasers_x.at[0].set(40),
        lasers_active=state.lasers_active.at[0].set(1),
        # Phase 95 belongs to native tick 64. The free-running native clock
        # determines laser phase, including when a test initializes a room.
        native_frame_count=jnp.array(64, dtype=jnp.int32),
        laser_cycle=jnp.array(95, dtype=jnp.int32)
    )
    
    obs, state, reward, done, info = env.step(state, 0) # NOOP
    
    # Laser is inactive, player should not die
    assert state.laser_cycle == 96
    assert state.death_timer == 0


@pytest.mark.parametrize("native_frame,visible", [(59, True), (60, False), (95, False), (96, True)])
def test_laser_observation_matches_final_clock(montezuma_env, native_frame, visible):
    from jaxtari.games.montezuma_revenge.rooms import load_room
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.int32(14), state, env.consts).replace(
        native_frame_count=jnp.int32(native_frame), player_x=jnp.int32(76), player_y=jnp.int32(26))
    obs, state, _, _, _ = env.step(state, 0)
    assert bool(obs.lasers.active[0]) == visible


def test_platform_observation_on_room_entry(montezuma_env):
    from jaxtari.games.montezuma_revenge.rooms import load_room
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.int32(18), state, env.consts).replace(
        player_x=jnp.int32(4), player_y=jnp.int32(26),
        native_frame_count=jnp.int32(70), platform_cycle=jnp.int32(92))
    obs, state, _, _, _ = env.step(state, 4)
    assert state.room_id == 17
    assert (obs.platforms.active == 1).all()
