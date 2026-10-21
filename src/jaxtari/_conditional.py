"""Transformation boundary for speculative conditionals.

Only this operation has custom rules; no existing JAX rules are modified.
Execution uses the optimized function. AD uses the equivalent ordinary cond,
with any batching already applied, preserving JAX's own derivative semantics.
"""
import jax
from jax.extend import core
from jax.interpreters import ad, batching, mlir


_call = core.Primitive("jaxtari_conditional")
_call.multiple_results = True


def _impl(*args, forward, reference, out_avals):
    del reference, out_avals
    return forward(*args)


def _abstract(*args, forward, reference, out_avals):
    del args, forward, reference
    return out_avals


def _lower(ctx, *args, forward, reference, out_avals):
    del reference, out_avals
    return mlir.lower_fun(forward, multiple_results=True)(ctx, *args)


def _batch(args, axes, *, forward, reference, out_avals):
    size = next(x.shape[axis] for x, axis in zip(args, axes) if axis is not None)
    batched_avals = tuple(aval.update(shape=(size,) + aval.shape) for aval in out_avals)
    result = _call.bind(
        *args, forward=jax.vmap(forward, in_axes=axes),
        reference=jax.vmap(reference, in_axes=axes), out_avals=batched_avals,
    )
    return result, (0,) * len(result)


def _jvp(primals, tangents, *, forward, reference, out_avals):
    del forward, out_avals
    return jax.jvp(reference, tuple(primals), tuple(map(ad.instantiate_zeros, tangents)))


_call.def_impl(_impl)
_call.def_abstract_eval(_abstract)
mlir.register_lowering(_call, _lower)
batching.primitive_batchers[_call] = _batch
ad.primitive_jvps[_call] = _jvp


def conditional_call(forward, reference, *args):
    flat_args, in_tree = jax.tree.flatten(args)
    shape = jax.eval_shape(reference, *args)
    shape_leaves, out_tree = jax.tree.flatten(shape)
    out_avals = tuple(jax.core.ShapedArray(x.shape, x.dtype, weak_type=x.weak_type)
                      for x in shape_leaves)

    def flatten(fn):
        def call(*leaves):
            return jax.tree.leaves(fn(*jax.tree.unflatten(in_tree, leaves)))
        return call

    result = _call.bind(*flat_args, forward=flatten(forward),
                        reference=flatten(reference), out_avals=out_avals)
    return jax.tree.unflatten(out_tree, result)
