import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.environment import JAXAtariAction as Action


def test_surround_start_positions():
    env = jaxatari.make("surround")
    _obs, state = env.reset()
    assert tuple(state.p1_pos.tolist()) == env.consts.P1_START_POS
    assert tuple(state.p2_pos.tolist()) == env.consts.P2_START_POS


def test_surround_crossing_no_crash():
    env = jaxatari.make("surround")
    _obs, state = env.reset()
    actions = jnp.array([Action.RIGHT, Action.LEFT], dtype=jnp.int32)
    # Move players towards each other until they swap positions
    for _ in range(15):
        _obs, state, reward, done, _info = env.step(state, actions)
        assert not bool(done)


def test_surround_wall_crash():
    env = jaxatari.make("surround")
    _obs, state = env.reset()
    actions = jnp.array([Action.LEFT, Action.UP], dtype=jnp.int32)
    done = False
    for _ in range(6):
        _obs, state, reward, done, _info = env.step(state, actions)
        if done:
            break
    assert bool(done)
    assert state.p2_score == 1
    assert reward == -1.0
