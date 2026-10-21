"""GPU parity checks: wrappers must not inflate Pong step time vs a fair baseline.

Excluded from the normal GitHub pytest pipeline (CPU-forced, no CUDA).
Run locally with::

    JAXTARI_GPU_TESTS=1 pytest -m gpu tests/test_wrapper_performance.py -s

Non-pixel wrappers are compared to the raw env.
Pixel wrappers are compared to raw env + ``render`` (same frame count).
"""
from __future__ import annotations

import os
import time

import pytest

# Opt-in only — default CI / conftest stays on CPU and never sets this.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.serial,
    pytest.mark.skipif(
        os.environ.get("JAXTARI_GPU_TESTS") != "1",
        reason="GPU wrapper parity test; run with JAXTARI_GPU_TESTS=1 on a CUDA machine",
    ),
]

import jax
import jax.numpy as jnp

import jaxtari
from jaxtari.wrappers import (
    AtariWrapper,
    ContinuousActionWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    MultiRewardLogWrapper,
    MultiRewardWrapper,
    NoiseWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelAndObjectCentricWrapper,
    PixelAndObjectObsWrapper,
    PixelObsWrapper,
)


NUM_ENVS = 128
NUM_STEPS = 2000
TRIALS = 3
# GPU: OC speculative autoreset / obs packing leave a small residual; catch the old nested-scan / reset bugs.
MAX_NON_PIXEL_OVERHEAD = 2.0
# Pixel path also resizes/grayscales/stacks beyond a bare render.
MAX_PIXEL_OVERHEAD = 4.0




def _median_ms(compiled, init) -> float:
    out = compiled(init)
    jax.tree_util.tree_leaves(out)[0].block_until_ready()
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        out = compiled(init)
        jax.tree_util.tree_leaves(out)[0].block_until_ready()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e3


def _compile_discrete_rollout(env, length: int):
    keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    _, states = jax.vmap(env.reset)(keys)
    actions = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)
    vm = jax.vmap(env.step)

    def body(s, _):
        return vm(s, actions)[1], None

    @jax.jit
    def rollout(s):
        s2, _ = jax.lax.scan(body, s, xs=None, length=length)
        return s2

    return rollout.lower(states).compile(), states


def _compile_continuous_rollout(env, length: int):
    keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    _, states = jax.vmap(env.reset)(keys)
    actions = jnp.zeros((NUM_ENVS, 3), dtype=jnp.float32)
    vm = jax.vmap(env.step)

    def body(s, _):
        return vm(s, actions)[1], None

    @jax.jit
    def rollout(s):
        s2, _ = jax.lax.scan(body, s, xs=None, length=length)
        return s2

    return rollout.lower(states).compile(), states


def _compile_base_plus_render(env, length: int):
    """Raw step + render; image checksum kept in carry so XLA cannot DCE render."""
    keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    _, states = jax.vmap(env.reset)(keys)
    actions = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)
    vm = jax.vmap(env.step)
    render = jax.vmap(env.render)

    def body(carry, _):
        s, acc = carry
        _, ns, _, _, _ = vm(s, actions)
        img = render(ns)
        return (ns, acc + img.astype(jnp.float32).sum()), None

    @jax.jit
    def rollout(carry):
        (s2, acc), _ = jax.lax.scan(body, carry, xs=None, length=length)
        return s2, acc

    init = (states, jnp.float32(0))
    return rollout.lower(init).compile(), init


def _atari():
    return AtariWrapper(jaxtari.make("pong"), noop_max=0, first_fire=False)


def _time_discrete(env, length: int) -> float:
    compiled, states = _compile_discrete_rollout(env, length)
    return _median_ms(compiled, states)


@pytest.fixture(scope="module")
def gpu_ready():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("No GPU device visible to JAX")
    return True


def test_pong_wrapper_gpu_step_parity(gpu_ready):
    """Wrappers stay within a small factor of a fair GPU baseline on Pong."""
    base = jaxtari.make("pong")
    base_ms = _time_discrete(base, NUM_STEPS)

    compiled_rr, init_rr = _compile_base_plus_render(base, NUM_STEPS)
    render_ms = _median_ms(compiled_rr, init_rr)

    results = {"base": base_ms, "base+render": render_ms}

    non_pixel = {
        "atari": _atari(),
        "oc_fs1": ObjectCentricWrapper(
            _atari(), frame_skip=1, frame_stack_size=1, clip_reward=False
        ),
        "flatten_oc": FlattenObservationWrapper(
            ObjectCentricWrapper(_atari(), frame_skip=1, frame_stack_size=1, clip_reward=False)
        ),
        "normalize_oc": NormalizeObservationWrapper(
            FlattenObservationWrapper(
                ObjectCentricWrapper(_atari(), frame_skip=1, frame_stack_size=1, clip_reward=False)
            )
        ),
        "log_oc": LogWrapper(
            NormalizeObservationWrapper(
                FlattenObservationWrapper(
                    ObjectCentricWrapper(
                        _atari(), frame_skip=1, frame_stack_size=1, clip_reward=False
                    )
                )
            )
        ),
        "noise": NoiseWrapper(_atari(), detect_prob=1.0, std_dev=0.0),
        "multi_reward": MultiRewardWrapper(
            jaxtari.make("pong"), reward_funcs=[lambda _prev, _cur: jnp.float32(0.0)]
        ),
        # MultiRewardLog expects Atari-style (terminated, truncated) underneath.
        "multi_reward_log": MultiRewardLogWrapper(
            ObjectCentricWrapper(
                AtariWrapper(
                    MultiRewardWrapper(
                        jaxtari.make("pong"),
                        reward_funcs=[lambda _prev, _cur: jnp.float32(0.0)],
                    ),
                    noop_max=0,
                    first_fire=False,
                ),
                frame_skip=1,
                frame_stack_size=1,
                clip_reward=False,
            )
        ),

        # Equal base-env work: frame_skip=4 => NUM_STEPS // 4 outer steps (Python-unrolled skip).
        "oc_fs4": (
            ObjectCentricWrapper(_atari(), frame_skip=4, frame_stack_size=4, clip_reward=True),
            NUM_STEPS // 4,
        ),
    }




    for name, env in non_pixel.items():
        length = NUM_STEPS
        if isinstance(env, tuple):
            env, length = env
        results[name] = _time_discrete(env, length)

    cont = ContinuousActionWrapper(
        AtariWrapper(
            jaxtari.make("pong"), full_action_space=True, noop_max=0, first_fire=False
        )
    )
    compiled_c, states_c = _compile_continuous_rollout(cont, NUM_STEPS)
    results["continuous"] = _median_ms(compiled_c, states_c)

    pixel = {
        "pixel_fs1": PixelObsWrapper(
            _atari(),
            frame_skip=1,
            frame_stack_size=1,
            do_pixel_resize=True,
            grayscale=True,
            max_pooling=False,
            clip_reward=False,
        ),
        "pixel_and_oc_fs1": PixelAndObjectCentricWrapper(
            _atari(),
            frame_skip=1,
            frame_stack_size=1,
            do_pixel_resize=True,
            grayscale=True,
            max_pooling=False,
            clip_reward=False,
        ),
        "pixel_and_oc_obs_fs1": PixelAndObjectObsWrapper(
            _atari(),
            frame_skip=1,
            frame_stack_size=1,
            do_pixel_resize=True,
            grayscale=True,
            max_pooling=False,
            clip_reward=False,
        ),
    }

    for name, env in pixel.items():
        results[name] = _time_discrete(env, NUM_STEPS)

    report = ", ".join(f"{k}={v:.2f}ms" for k, v in results.items())
    print(f"\nGPU Pong wrapper medians ({NUM_ENVS} envs × {NUM_STEPS} steps): {report}")

    for name in (
        "atari",
        "oc_fs1",
        "flatten_oc",
        "normalize_oc",
        "log_oc",
        "noise",
        "multi_reward",
        "multi_reward_log",
        "oc_fs4",
        "continuous",
    ):
        ratio = results[name] / base_ms
        assert ratio <= MAX_NON_PIXEL_OVERHEAD, (
            f"{name} is {ratio:.2f}× base on GPU ({results[name]:.2f}ms vs {base_ms:.2f}ms). [{report}]"
        )



    for name in ("pixel_fs1", "pixel_and_oc_fs1", "pixel_and_oc_obs_fs1"):
        ratio = results[name] / render_ms
        assert ratio <= MAX_PIXEL_OVERHEAD, (
            f"{name} is {ratio:.2f}× base+render on GPU "
            f"({results[name]:.2f}ms vs {render_ms:.2f}ms). [{report}]"
        )

