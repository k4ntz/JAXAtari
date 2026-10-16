"""Tests for jaxtari.detection (YOLO detector + YOLOObjectCentricWrapper)."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import flatten_util

import jaxtari
from jaxtari.detection import MISSING, YOLOObjectCentricWrapper, load_detector
from jaxtari.detection import labels
from jaxtari.wrappers import AtariWrapper, ObjectCentricWrapper

from jaxtari.paths import get_base_detector_dir

WEIGHTS = Path(get_base_detector_dir())  # installed detector checkpoints (<storage>/detectors)
PONG = WEIGHTS / "gray96" / "pong.npz"
needs_weights = pytest.mark.skipif(not PONG.exists(), reason="detector checkpoints not installed")


def test_label_padding_and_yolo_line():
    assert labels.pad_label({}, 10, 20, 2, 4) == (7, 18, 8, 8)              # floor of 8 px, centred
    assert labels.pad_label({"cell_w": 0, "cell_h": 4}, 10, 20, 2, 4) == (10, 20, 2, 4)  # opted out
    c, x, y, w, h = labels.to_yolo_line(3, 0, 0, 160, 210).split()
    assert (c, float(x), float(y), float(w), float(h)) == ("3", 0.5, 0.5, 1.0, 1.0)


def test_registry_covers_checkpoints():
    registry = labels.load_registry()
    for ck in WEIGHTS.glob("*/*.npz"):
        assert ck.stem in registry, ck


@pytest.mark.smoke
@needs_weights
def test_detector_finds_pong_objects():
    det = load_detector(PONG)
    assert det.input_mode == "native" and det.frame_size == (96, 96)
    env = jaxtari.make("pong")
    from jaxtari.modification import apply_native_downscaling
    apply_native_downscaling(env, (96, 96), True)
    _, state = env.reset(jax.random.PRNGKey(0))
    for _ in range(40):
        _, state, *_ = env.step(state, 0)
    out = jax.jit(det.detect)(det.params, env.render(state))
    classes = set(np.asarray(out.cls)[np.asarray(out.valid)].tolist())
    assert {0, 1}.issubset(classes)  # both paddles


@pytest.fixture(scope="module")
def pong_env():
    return YOLOObjectCentricWrapper(AtariWrapper(jaxtari.make("pong")), PONG, calibration_steps=60)


@pytest.mark.smoke
@needs_weights
def test_wrapper_matches_object_centric_layout(pong_env):
    oc = ObjectCentricWrapper(AtariWrapper(jaxtari.make("pong")))
    obs, state = pong_env.reset(jax.random.PRNGKey(1))
    oc_obs, _ = oc.reset(jax.random.PRNGKey(1))
    assert obs.shape == oc_obs.shape == pong_env.observation_space().shape
    for _ in range(30):
        obs, state, *_ = pong_env.step(state, jnp.int32(0))
    # the scores are not detected -> MISSING; the player paddle is detected within a few pixels
    template = jax.tree.map(lambda a: np.asarray(a, np.float32), pong_env._template)
    pred = flatten_util.ravel_pytree(template)[1](np.asarray(obs[-1]))
    gt = pong_env._env._env._get_observation(state.atari_state.env_state)
    assert float(pred.score_player) == MISSING and float(pred.score_enemy) == MISSING
    assert float(pred.player.active) == 1
    assert abs(float(pred.player.y) - float(gt.player.y)) <= 3
    assert float(pred.player.visual_id) == MISSING


@pytest.mark.smoke
@needs_weights
def test_wrapper_jit_and_vmap(pong_env):
    reset, step = jax.jit(jax.vmap(pong_env.reset)), jax.jit(jax.vmap(pong_env.step))
    obs, state = reset(jax.random.split(jax.random.PRNGKey(0), 3))
    obs, state, reward, term, trunc, info = step(state, jnp.zeros(3, jnp.int32))
    assert obs.shape == (3, 4, pong_env.num_features)
    assert reward.shape == (3,) and state.detections.boxes.shape[0] == 3

