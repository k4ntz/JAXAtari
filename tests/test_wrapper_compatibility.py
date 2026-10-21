"""Behavioral API checks independent of GPU timing thresholds."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxtari
from jaxtari import wrappers as w
import wrapper_reference as reference

OBS_WRAPPERS = (
    "ObjectCentricWrapper", "PixelObsWrapper",
    "PixelAndObjectCentricWrapper", "PixelAndObjectObsWrapper",
)


def assert_tree_equal(a, b):
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    assert ta == tb
    for (pa, x), (pb, y) in zip(la, lb):
        assert pa == pb
        assert x.shape == y.shape and x.dtype == y.dtype
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", OBS_WRAPPERS)
@pytest.mark.parametrize("noop_max,frame_skip", [(0, 1), (3, 4)])
def test_rollout_matches_scan_cond_reference(name, noop_max, frame_skip):
    def build(module):
        atari = module.AtariWrapper(
            jaxtari.make("pong"), noop_max=noop_max, first_fire=True,
            max_frames_per_episode=12,
        )
        return getattr(module, name)(atari, frame_skip=frame_skip, frame_stack_size=2)

    def rollout(env):
        keys = jax.random.split(jax.random.key(42), (2, 3))
        obs, state = jax.vmap(jax.vmap(env.reset))(keys)
        # Force a mixed truncation batch, followed by further autoresets.
        counters = jnp.array([[0, 11, 12], [12, 0, 11]], jnp.int32)
        state = state.replace(atari_state=state.atari_state.replace(step=counters))
        def body(state, action):
            out = jax.vmap(jax.vmap(env.step))(state, jnp.full((2, 3), action))
            return out[1], out
        return obs, jax.lax.scan(body, state, jnp.arange(5, dtype=jnp.int32) % 3)

    build_ref, build_new = build(reference), build(w)
    expected = jax.jit(lambda: rollout(build_ref))()
    actual = jax.jit(lambda: rollout(build_new))()
    assert_tree_equal(expected, actual)


@pytest.mark.parametrize("name", ("AtariWrapper", "NoiseWrapper") + OBS_WRAPPERS)
def test_public_reset_signature_and_legacy_override(name):
    cls = getattr(w, name)
    assert list(inspect.signature(cls.reset).parameters) == ["self", "key"]
    class Legacy(cls):
        def reset(self, key):
            obs, state = super().reset(key)
            # Deliberately change the observation independently of state.
            return jax.tree.map(lambda x: jnp.zeros_like(x), obs), state

    base = jaxtari.make("pong")
    if name == "AtariWrapper":
        env = w.ObjectCentricWrapper(Legacy(base, noop_max=0, first_fire=False,
                                            max_frames_per_episode=1), frame_skip=1)
    elif name == "NoiseWrapper":
        env = w.ObjectCentricWrapper(Legacy(w.AtariWrapper(
            base, noop_max=0, first_fire=False, max_frames_per_episode=1)), frame_skip=1)
    else:
        env = Legacy(w.AtariWrapper(base, noop_max=0, first_fire=False,
                                    max_frames_per_episode=1), frame_skip=1)
    _, state = env.reset(jax.random.PRNGKey(7))
    out = env.step(state, 0)
    assert bool(out[4])
    # Every reset override returns zeros; autoreset must retain its returned obs.
    for leaf in jax.tree.leaves(out[0]):
        np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))


def test_speculative_cond_autodiff_matches_cond():
    pred = lambda x: x > 0
    rare, common = jnp.sqrt, lambda x: x * x
    f = w.speculative_cond(pred, rare, common)
    ref = lambda x: jax.lax.cond(pred(x), rare, common, x)
    for x in (-1., 2.):
        assert_tree_equal(jax.jvp(f, (jnp.float32(x),), (jnp.float32(1),)),
                          jax.jvp(ref, (jnp.float32(x),), (jnp.float32(1),)))
        np.testing.assert_array_equal(jax.grad(f)(x), jax.grad(ref)(x))
        np.testing.assert_array_equal(jax.grad(jax.grad(f))(x), jax.grad(jax.grad(ref))(x))

    f = w.speculative_cond(pred, lambda x: x * x, lambda x: 3 * x)
    ref = lambda x: jax.lax.cond(pred(x), lambda x: x*x, lambda x: 3*x, x)
    x = jnp.array([[-2., 1.], [3., -4.]])
    for transform in (lambda g: g, jax.grad, jax.jacfwd):
        a, b = jax.vmap(jax.vmap(transform(f))), jax.vmap(jax.vmap(transform(ref)))
        assert_tree_equal(jax.jit(a)(x), jax.jit(b)(x))
    a, b = jax.vmap(jax.vmap(f)), jax.vmap(jax.vmap(ref))
    assert_tree_equal(jax.jvp(a, (x,), (jnp.ones_like(x),)),
                      jax.jvp(b, (x,), (jnp.ones_like(x),)))
    assert_tree_equal(jax.grad(lambda x: a(x).sum())(x),
                      jax.grad(lambda x: b(x).sum())(x))


def test_speculative_guard_survives_nested_vmap():
    f = w.speculative_cond(lambda x: x > 0, lambda x: jnp.sin(x), lambda x: x)
    lowered = jax.jit(jax.vmap(jax.vmap(f))).lower(jnp.ones((2, 3)))
    assert "stablehlo.case" in lowered.as_text()


def test_wrapper_jvp_matches_reference():
    def function(module):
        env = module.ObjectCentricWrapper(module.AtariWrapper(
            jaxtari.make("pong"), noop_max=3, first_fire=False), frame_skip=1)
        _, state = env.reset(jax.random.PRNGKey(0))
        return lambda x: env.step(state.replace(obs_stack=state.obs_stack * x), 0)[0].sum()
    args = (jnp.float32(1),), (jnp.float32(1),)
    assert_tree_equal(jax.jvp(function(w), *args), jax.jvp(function(reference), *args))


def test_terminal_noop_restores_seeded_reset():
    # A pure terminal-every-step environment forces the reset-inside-noop path.
    from flax import struct
    @struct.dataclass
    class State:
        value: jax.Array
    class TerminalEnv:
        ACTION_SET = jnp.array([0])
        def observation_space(self): return w.spaces.Box(0, 1000, (1,))
        def reset(self, key):
            value = jax.random.randint(key, (), 0, 1000)
            return value[None], State(value)
        def step(self, state, action):
            return (state.value + 1)[None], State(state.value + 1), jnp.float32(0), jnp.bool_(True), {}
    for module in (reference, w):
        env = module.AtariWrapper(TerminalEnv(), noop_max=5, first_fire=False)
        result = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(9), 8))
        if module is reference: expected = result
        else: assert_tree_equal(expected, result)


def test_mixed_axes_and_grad_through_jit():
    def pred(x, scale): return x["value"] > 0
    def rare(x, scale): return {"result": x["value"] ** 2 * scale}
    def common(x, scale): return {"result": 3 * x["value"] * scale}
    optimized = w.speculative_cond(pred, rare, common)
    reference = lambda x, scale: jax.lax.cond(pred(x, scale), rare, common, x, scale)
    # Nonleading mapped axes, a pytree argument, and an unmapped operand.
    def batched(fn):
        return jax.jit(jax.vmap(jax.vmap(fn, in_axes=(0, None)), in_axes=(1, None)))
    a, b = batched(optimized), batched(reference)
    for values in (jnp.ones((2, 3)), -jnp.ones((2, 3)),
                   jnp.array([[-1., 2., -3.], [4., -5., 6.]])):
        x, scale = {"value": values}, jnp.float32(2)
        assert_tree_equal(a(x, scale), b(x, scale))
        assert_tree_equal(jax.grad(lambda x, s: a(x, s)["result"].sum(), (0, 1))(x, scale),
                          jax.grad(lambda x, s: b(x, s)["result"].sum(), (0, 1))(x, scale))
