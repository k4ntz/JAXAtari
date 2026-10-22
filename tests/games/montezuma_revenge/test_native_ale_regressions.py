"""Measured native-frame regressions, with frozen mechanism fixtures.

These tests need installed sprite assets but do not load a ROM or ALE. Numeric
oracles and player masks were recorded from actual one-frame ALE execution.
They verify the fields named per case; they do not assert hidden-state parity
or regular reachability of a frozen state. Full reset/prefix replay is separate.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ORACLES = json.loads(Path(__file__).with_name("native_ale_oracles.json").read_text())


@pytest.fixture(scope="session")
def native_step(montezuma_env):
    return montezuma_env.step


@pytest.fixture(scope="session")
def native_render(montezuma_env):
    return jax.jit(montezuma_env.render)


@pytest.fixture(scope="session")
def native_reset_state(montezuma_env):
    _, state = montezuma_env.reset(jax.random.PRNGKey(100001790))
    return jax.tree.map(jnp.asarray, state)


def _state(state, values):
    if values is not None:
        state = state.replace(**{
            k: jnp.asarray(v, dtype=np.asarray(getattr(state, k)).dtype)
            for k, v in values.items()
        })
    return state


def _measured(state, done):
    result = dict(
        x=int(state.player_x), y=int(state.player_y), room=int(state.room_id),
        mode=int(state.native_mode), level=int(state.level), lives=int(state.lives),
        rng_a=int(state.native_rng_a), rng_b=int(state.native_rng_b),
        score=int(np.asarray(state.score).sum()) % 1000000,
        fire_latch=int(state.prev_is_fire),
        rng_carry=int(state.native_rng_carry),
        tick=int((state.native_frame_count + 67) % 256),
        momentum=int(state.native_jump_momentum),
        done=int(done),
    )
    result.update(zip(["keys", "swords", "torch", "amulet"], np.asarray(state.inventory).astype(int)))
    result.update({f"history_{i}": int(v) for i, v in enumerate(state.native_room_history)})
    return result


@pytest.mark.parametrize("case", ORACLES["cases"], ids=lambda c: c["name"])
def test_native_frame_mechanism(native_reset_state, native_step, case):
    assert len(case["actions"]) == len(case["expected"])
    state = _state(native_reset_state, case["state"])
    for tick, (action, expected) in enumerate(zip(case["actions"], case["expected"])):
        _, state, _, done, _ = native_step(state, jnp.int32(action))
        measured = _measured(state, done)
        assert {k: measured[k] for k in expected} == expected, (case["name"], tick)


@pytest.mark.parametrize("case", ORACLES["render"], ids=lambda c: c["name"])
def test_native_player_silhouette(native_reset_state, native_step, native_render, case):
    assert len(case["actions"]) == len(case["expected_masks"]) == len(case["anchors"])
    state = _state(native_reset_state, case["state"])
    colors = np.asarray([[200, 72, 72], [210, 182, 86], [228, 111, 111]])
    frames = zip(case["actions"], case["expected_masks"], case["anchors"])
    for tick, (action, expected, (x, y)) in enumerate(frames):
        _, state, _, _, _ = native_step(state, jnp.int32(action))
        frame = np.asarray(native_render(state))
        crop = frame[max(0, y + 43):min(frame.shape[0], y + 73),
                     max(0, x - 4):min(frame.shape[1], x + 12)]
        mask = np.any(np.all(crop[:, :, None, :] == colors[None, None, :, :], axis=3), axis=2)
        target = np.asarray([[c == "#" for c in row] for row in expected])
        np.testing.assert_array_equal(mask, target, err_msg=f"{case['name']}, native frame {tick}")
