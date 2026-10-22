"""Key/momentum regressions from independent single-frame ALE measurements.

Frozen fixtures are mechanism controls, not a claim of regular reachability.
The accompanying local report separately records unsynchronized reset replay.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_native_ale_regressions import _measured, _state

ORACLES = json.loads(Path(__file__).with_name("key_momentum_ale_oracles.json").read_text())


@pytest.mark.parametrize("case", ORACLES["cases"], ids=lambda c: c["name"])
def test_key_momentum_native_frame(montezuma_env, case):
    assert len(case["actions"]) == len(case["expected"])
    _, state = montezuma_env.reset(jax.random.PRNGKey(100001790))
    state = _state(jax.tree.map(jnp.asarray, state), case["state"])
    for tick, (action, expected) in enumerate(zip(case["actions"], case["expected"])):
        _, state, _, done, _ = montezuma_env.step(state, jnp.int32(action))
        measured = _measured(state, done)
        assert {k: measured[k] for k in expected} == expected, (case["name"], tick)
        if "expected_key_active" in case:
            assert int(state.global_items_active[4, 0]) == case["expected_key_active"][tick]
