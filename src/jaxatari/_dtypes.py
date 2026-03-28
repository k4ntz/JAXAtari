"""Shared dtype helpers for counters that may exceed 32-bit ranges."""

import jax
import jax.numpy as jnp


COUNTER_DTYPE = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


def counter_array(value):
    return jnp.asarray(value, dtype=COUNTER_DTYPE)
