"""Independent ALE sequences for D6 completion before standing-anchor restore."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtari.environment import JaxtariAction as Action
from jaxtari.games.montezuma_revenge.core import NativeSupportPhase
from test_native_ale_regressions import _measured, _state

REFERENCE = json.loads(Path(__file__).with_name('dispatch_ale_oracles.json').read_text())
FIELDS = ['x', 'y', 'room', 'mode', 'level', 'lives', 'tick', 'rng_a', 'rng_b',
          'keys', 'swords', 'torch', 'amulet'] + [f'history_{i}' for i in range(24)] + [
          'score', 'done', 'fire_latch', 'momentum', 'rng_carry']


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: c['label'])
def test_native_support_dispatch(montezuma_env, case):
    _, state = montezuma_env.reset(jax.random.PRNGKey(REFERENCE['provenance']['seed']))
    state = _state(jax.tree.map(jnp.asarray, state), case['initial'])
    state = jax.tree.map(lambda v: jnp.asarray(v, dtype=np.asarray(v).dtype), state)
    assert len(case['actions']) == len(case['expected']) == len(case['rewards'])
    for tick, (action, expected, expected_reward) in enumerate(zip(
            case['actions'], case['expected'], case['rewards'])):
        _, state, reward, done, _ = montezuma_env.step(state, jnp.int32(action))
        measured = _measured(state, done)
        assert [int(measured[k]) for k in FIELDS] == expected, (case['label'], tick)
        assert float(reward) == expected_reward


def test_support_dispatch_preserves_input_and_handles_batched_phases(montezuma_env):
    """No horizontal/diagonal entry; an entered climb persists on release."""
    _, initial = montezuma_env.reset(jax.random.PRNGKey(0))
    phases = jnp.array([1, 1, 1, 2, 0], dtype=jnp.int32)
    states = jax.tree.map(lambda v: jnp.broadcast_to(v, (5,) + jnp.shape(v)), initial)
    states = states.replace(native_support_phase=phases, is_jumping=jnp.ones(5, jnp.int32),
                            last_rope=jnp.zeros(5, jnp.int32))
    actions = jnp.array([Action.UP, Action.UPRIGHT, Action.NOOP, Action.NOOP, Action.UP])
    result = jax.jit(jax.vmap(montezuma_env._prepare_native_dispatch))(states, actions)
    np.testing.assert_array_equal(result.native_support_phase, [2, 1, 1, 2, 0])
    np.testing.assert_array_equal(result.is_jumping, [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(result.last_rope, [-1, 0, 0, -1, 0])
    np.testing.assert_array_equal(states.is_jumping, [1, 1, 1, 1, 1])
    assert int(initial.native_support_phase) == NativeSupportPhase.NORMAL
