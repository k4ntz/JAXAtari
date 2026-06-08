import jax
import jax.numpy as jnp

from jaxatari.core import GAME_MODULES
from jaxatari.games.jax_jamesbond import JaxJamesBond


def test_jamesbond_registered():
    assert GAME_MODULES["jamesbond"] == "jaxatari.games.jax_jamesbond"


def test_jamesbond_skeleton_reset_step_and_render():
    env = JaxJamesBond()
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    assert bool(env.observation_space().contains(obs))
    assert int(state.lives) == env.consts.MAX_LIVES
    assert int(state.score) == 0
    assert int(state.step_count) == 0

    obs, state, reward, done, info = env.step(state, jnp.array(0, dtype=jnp.int32))
    assert bool(env.observation_space().contains(obs))
    assert float(reward) == 0.0
    assert not bool(done)
    assert not bool(info.collision_happened)
    assert int(state.step_count) == 1

    frame = env.render(state)
    assert frame.shape == env.image_space().shape
    assert bool(env.image_space().contains(frame))
