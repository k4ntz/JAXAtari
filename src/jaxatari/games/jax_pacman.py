"""
Pacman implementation for JAXAtari.
Based on the original Atari 2600 version.
"""

from enum import IntEnum
import os
from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


def get_level_maze(level: chex.Array):
    return jnp.mod(level - 1, 4).astype(jnp.int32)


MAZE = jnp.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=bool)

MAZE2 = jnp.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=bool)

class PacmanMaze:
    MAZE0 = MAZE
    MAZES = jnp.array([MAZE0, MAZE2], dtype=jnp.bool_)
    TILE_SCALE = 4
    WIDTH = 160
    HEIGHT = 160
    
    WALL_COLOR = jnp.array([223, 192, 111], dtype=jnp.uint8)
    PATH_COLOR = jnp.array([50, 50, 176], dtype=jnp.uint8)

    # We reuse MsPacman maze 0 pellets for compatibility
    BASE_PELLETS = jnp.array([ 
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        ], dtype=bool).T

    @staticmethod
    def precompute_dof(maze_id: int):
        maze = PacmanMaze.MAZES[maze_id]
        sum_horizontal_strip = (
            jnp.roll(maze, -1, axis=1) +
            maze
        )
        sum_vertical_strip = (
            jnp.roll(maze, 1, axis=0) +
            maze +
            jnp.roll(maze, -1, axis=0) +
            jnp.roll(maze, -2, axis=0)
        )
        no_wall_above = jnp.roll(sum_horizontal_strip, 2, axis=0) == 0
        no_wall_below = jnp.roll(sum_horizontal_strip, -3, axis=0) == 0
        no_wall_left = jnp.roll(sum_vertical_strip, 1, axis=1) == 0
        no_wall_right = jnp.roll(sum_vertical_strip, -2, axis=1) == 0
        dof_grid = jnp.stack([no_wall_above, no_wall_right, no_wall_left, no_wall_below], axis=-1)
        dof_grid = jnp.transpose(dof_grid, (1, 0, 2))
        return dof_grid
    
    @staticmethod
    def load_background(maze_id: int):
        maze = PacmanMaze.MAZES[maze_id]
        maze_expanded = jnp.repeat(jnp.repeat(maze, PacmanMaze.TILE_SCALE, axis=0), PacmanMaze.TILE_SCALE, axis=1)
        background = jnp.where(
            maze_expanded[..., None],
            PacmanMaze.WALL_COLOR,
            PacmanMaze.PATH_COLOR
        )
        pad_height = 210 - background.shape[0]
        background = jnp.pad(background, ((0, pad_height), (0, 0), (0, 0)))
        return jnp.swapaxes(background, 0, 1)



# -------- Constants --------
class PacmanConstants(struct.PyTreeNode):
    # GENERAL
    RESET_LEVEL: int = struct.field(pytree_node=False, default=1)
    TIME_SCALE: int = struct.field(pytree_node=False, default=20)
    INITIAL_LIVES: int = struct.field(pytree_node=False, default=4) # Pacman starts with 4 lives
    MAX_LIVE_COUNT: int = struct.field(pytree_node=False, default=8)
    MAX_SCORE_DIGITS: int = struct.field(pytree_node=False, default=6)
    BONUS_LIFE_SCORE: int = struct.field(pytree_node=False, default=1000000) # Effectively disabled by score, logic uses maze clear
    COLLISION_THRESHOLD: int = struct.field(pytree_node=False, default=6)
    PELLETS_TO_COLLECT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([130, 130, 130, 130]))

    # GHOST TIMINGS
    SUE_RELEASE_TIME: int = struct.field(pytree_node=False, default=1*20)
    INKY_RELEASE_TIME: int = struct.field(pytree_node=False, default=5*20)
    PINKY_RELEASE_TIME: int = struct.field(pytree_node=False, default=7*20)
    RESET_TIMER: int = struct.field(pytree_node=False, default=4*20)
    CHASE_DURATION: int = struct.field(pytree_node=False, default=20*20)
    SCATTER_DURATION: int = struct.field(pytree_node=False, default=7*20)
    FRIGHTENED_DURATION: int = struct.field(pytree_node=False, default=13*20)
    BLINKING_DURATION: int = struct.field(pytree_node=False, default=4*20)
    ENJAILED_DURATION: int = struct.field(pytree_node=False, default=10*20)
    FRIGHTENED_REDUCTION: float = struct.field(pytree_node=False, default=0.85)
    RETURN_DURATION: int = struct.field(pytree_node=False, default=int(20/2))
    MAX_CHASE_OFFSET: float = struct.field(pytree_node=False, default=20*20/10)
    MAX_SCATTER_OFFSET: float = struct.field(pytree_node=False, default=7*20/10)
    TUNNEL_DELAY: int = struct.field(pytree_node=False, default=76)

    # VITAMINS (Fruit)
    VITAMINS_SPAWN_THRESHOLDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([50, 100]))
    VITAMINS_DURATION: int = struct.field(pytree_node=False, default=10*20) # Stationary for a few moments
    VITAMINS_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([76, 103])) # Center of playfield

    # WEAPONS (Updated for Pacman)
    POWER_PELLET_TILES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[1, 3], [36, 3], [1, 39], [36, 39]]))
    POWER_PELLET_HITBOXES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[1, 5], [36, 5], [1, 40], [36, 40], [0, 5], [35, 5], [0, 40], [35, 40]]))

    JAIL_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([73, 78]))
    INITIAL_GHOST_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([73, 78]))
    INITIAL_PACMAN_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([75, 150]))
    SCATTER_TARGETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[PacmanMaze.WIDTH - 1, 0], [0, 0], [PacmanMaze.WIDTH - 1, PacmanMaze.HEIGHT - 1], [0, PacmanMaze.HEIGHT - 1]]))

    # ACTIONS
    DIRECTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN]))
    ACTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([(0, 0), (0, 0), (0, -1), (1, 0), (-1, 0), (0, 1)]))
    INITIAL_ACTION: int = struct.field(pytree_node=False, default=Action.LEFT)
    HORIZONTAL_SPEED: int = struct.field(pytree_node=False, default=1)
    VERTICAL_SPEED: int = struct.field(pytree_node=False, default=2)
    SLOW_VERTICAL_SPEED: int = struct.field(pytree_node=False, default=1)

    def slow_vertical_speed(self):
        return self.replace(VERTICAL_SPEED=self.SLOW_VERTICAL_SPEED)

    # POINTS (Updated for Pacman)
    PELLET_POINTS: int = struct.field(pytree_node=False, default=1)
    POWER_PELLET_POINTS: int = struct.field(pytree_node=False, default=5)
    VITAMINS_REWARD: int = struct.field(pytree_node=False, default=100)
    EAT_GHOSTS_POINTS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([20, 40, 80, 160], dtype=jnp.uint32))
    LEVEL_COMPLETED_POINTS: int = struct.field(pytree_node=False, default=0)

    # COLORS
    PATH_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([50, 50, 176], dtype=jnp.uint8))
    WALL_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([223, 192, 111], dtype=jnp.uint8))
    PELLET_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([223, 192, 111], dtype=jnp.uint8))
    POWER_PELLET_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([252, 144, 200], dtype=jnp.uint8))
    PACMAN_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([210, 164, 74, 255], dtype=jnp.uint8))
    PALE_BLUE_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([144, 144, 252], dtype=jnp.uint8))
    GREEN_BAND_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([72, 176, 110], dtype=jnp.uint8))
    GREEN_BAND_HEIGHT: int = struct.field(pytree_node=False, default=9)

CONSTS = PacmanConstants()

# We must define the module-level variables used by pathfind and others, mapping to CONSTS.
ACTIONS = CONSTS.ACTIONS
DIRECTIONS = CONSTS.DIRECTIONS
COLLISION_THRESHOLD = CONSTS.COLLISION_THRESHOLD

class GhostMode(IntEnum):
    RANDOM = 0
    CHASE = 1
    SCATTER = 2
    FRIGHTENED = 3
    BLINKING = 4
    RETURNING = 5
    ENJAILED = 6

class LevelState(NamedTuple):
    id: chex.Array                  # Int - Number of the current level, starts at 1
    eaten_pellets: chex.Array       # Int - Number of collected pellets
    dofmaze: chex.Array             # Bool[x][y][4] - Precomputed degree of freedom maze layout
    pellets: chex.Array             # Bool[x][y] - 2D grid of 0 (empty) or 1 (pellet)
    power_pellets: chex.Array       # Bool[4] - Indicates wheter the power pellet is available
    loaded: chex.Array

class GhostsState(NamedTuple):
    positions: chex.Array           # Tuple - (x, y)
    actions: chex.Array             # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    modes: chex.Array               # Enum - 0: RANDOM, 1: CHASE, 2: SCTATTER, 3: FRIGHTENED, 4: BLINKING, 5: RETURNING, 6: ENJAILED
    timers: chex.Array

class PlayerState(NamedTuple):
    position: chex.Array            # Tuple - (x, y)
    action: chex.Array              # Enum - 0: NOOP, 1: FURE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    has_pellet: chex.Array          # Bool - Indicates if pacman just collected a pellet
    eaten_ghosts: chex.Array        # Int - Indicates the number of ghosts eaten since the last power pellet
    tunnel_timer: chex.Array = jnp.array(0, dtype=jnp.int32)
    last_horiz_dir: chex.Array = jnp.array(2, dtype=jnp.int32)

class FruitState(NamedTuple):
    position: chex.Array            # Tuple - (x, y)
    exit: chex.Array                # Tuple - (x, y) Position of the tunnel through which it will exit
    type: chex.Array                # Enum - 0: CHERRY, 1: STRAWBERRY, 2: ORANGE, 3: PRETZEL, 4: APPLE, 5: PEAR, 6: BANANA, 7: NONE
    action: chex.Array              # Enum - 0: NOOP, 1: FIRE, 2: UP, 3: RIGHT, 4: LEFT, 5: DOWN
    spawn: chex.Array               # Bool - Indicates wether a fruit should spawn into the maze as soon as possible
    spawned: chex.Array             # Bool - Indicates wether a fruit is currently present within the maze
    timer: chex.Array

class PacmanState(NamedTuple):
    level: LevelState               # LevelState
    player: PlayerState             # PlayerState
    ghosts: GhostsState             # GhostStates
    fruit: FruitState               # FruitState
    lives: chex.Array               # Int - Number of lives left
    score: chex.Array               # Int - Total score reached
    score_changed: chex.Array       # Bool[] - Indicates which score digit changed since the last step
    freeze_timer: chex.Array        # Int - Time until game is unfrozen, decrements every step
    step_count: chex.Array          # Int - Number of steps made in the current level
    key: chex.PRNGKey

class PacmanObservation(NamedTuple):
    player_position: chex.Array
    player_action: chex.Array
    ghost_positions: chex.Array
    ghost_actions: chex.Array
    fruit_position: chex.Array
    fruit_action: chex.Array
    fruit_type: chex.Array
    pellets: chex.Array
    power_pellets: chex.Array

class PacmanInfo(NamedTuple):
    level: chex.Array
    score: chex.Array
    lives: chex.Array


def get_digit_count(number: chex.Array):
    """Returns the number of digits in a given decimal number."""
    number = jnp.abs(number)
    return jax.lax.cond(
        number == 0,
        lambda: jnp.array(1, dtype=jnp.uint8),
        lambda: jnp.floor(jnp.log10(number) + 1).astype(jnp.uint8)
    )

def act_to_dir(action: chex.Array):
    """Converts a JAXAtari action into the corresponding DIRECTION index.
    If conversion is not possible -1 is returned.

    action:     2 (UP)  3 (RIGHT)   4 (LEFT)    5 (DOWN)    ELSE
    direction:  0       1           2           3           -1
    """
    return jax.lax.cond(
        (action >= 2) & (action < 6),
        lambda: jnp.array(action - 2, dtype=jnp.int8),
        lambda: jnp.array(-1, dtype=jnp.int8)
    )

def dir_to_act(direction: chex.Array):
    """Converts a DIRECTION index into the corresponding JAXAtari action.
    If conversion is not possible -1 is returned.

    direction:  0       1           2           3           ELSE
    action:     2 (UP)  3 (RIGHT)   4 (LEFT)    5 (DOWN)    -1
    """
    return jax.lax.cond(
        (direction >= 0) & (direction < 4),
        lambda: jnp.array(direction + 2, dtype=jnp.int8),
        lambda: jnp.array(-1, dtype=jnp.int8)
    )

def last_pressed_action(action, prev_action):
    """Returns the last pressed action in cases where both actions are pressed"""
    return jax.lax.cond(
        action == Action.UPRIGHT,
        lambda: jax.lax.cond(
            prev_action == Action.UP,
            lambda: Action.RIGHT,
            lambda: Action.UP
        ),
        lambda: jax.lax.cond(
            action == Action.UPLEFT,
            lambda: jax.lax.cond(
                prev_action == Action.UP,
                lambda: Action.LEFT,
                lambda: Action.UP
            ),
            lambda: jax.lax.cond(
                action == Action.DOWNRIGHT,
                lambda: jax.lax.cond(
                    prev_action == Action.DOWN,
                    lambda: Action.RIGHT,
                    lambda: Action.DOWN
                ),
                lambda: jax.lax.cond(
                    action == Action.DOWNLEFT,
                    lambda: jax.lax.cond(
                        prev_action == Action.DOWN,
                        lambda: Action.LEFT,
                        lambda: Action.DOWN
                    ),
                    lambda: action
                )
            )
        )
    )

def pathfind(position: chex.Array, direction: chex.Array, target: chex.Array, allowed: chex.Array, key: chex.Array):
    """
    Returns the direction which should be taken to approach the target.
    If multiple options exist the direction is chosen that minimizes the distance on the longer axis - horizontal or vertical.
    If both distances are equal or multiple options exist on the same axis, the direction is chosen randomly.
    """
    valid_mask = allowed != 0
    n_allowed = jnp.sum(valid_mask)

    # If no direction allowed - Continue forward
    def no_allowed():
        return direction.astype(allowed.dtype)

    # If one direction allowed - Take it
    def one_allowed():
        return allowed[0].astype(allowed.dtype)

    # If multiple directions allowed - Get cost of all possible steps and determine advantageous directions
    def multi_allowed():
        new_positions = position + ACTIONS[allowed]
        costs = jnp.abs(new_positions - target).sum(axis=1)  # Manhattan distances
        costs = jnp.where(valid_mask, costs, jnp.iinfo(jnp.int32).max)
        min_cost = jnp.min(costs)
        min_mask = costs == min_cost
        min_dirs = jnp.compress(min_mask, allowed, size=DIRECTIONS.shape[0])
        n_min = jnp.sum(min_dirs != 0)

        # If one direction advantageous - Take it
        def one_min():
            return min_dirs[0].astype(allowed.dtype)

        # If multiple directions advantageous - Prioritize the longer axis
        def multi_min():
            h_dist = jnp.abs(position[0] - target[0])
            v_dist = jnp.abs(position[1] - target[1])
            h_dirs = jnp.array([int(Action.LEFT), int(Action.RIGHT)], dtype=jnp.int32)
            v_dirs = jnp.array([int(Action.DOWN), int(Action.UP)], dtype=jnp.int32)
            h_mask = jnp.isin(min_dirs, h_dirs)
            v_mask = jnp.isin(min_dirs, v_dirs)
            prefer_h = h_dist >= v_dist
            prefer_v = v_dist >= h_dist
            prefered = (h_mask & prefer_h) | (v_mask & prefer_v)
            n_prefered = jnp.sum(prefered)

            # If no direction advantageous on longer axis - Choose randomly
            def no_long_axis():
                return min_dirs[jax.random.randint(key, (), 0, n_min)].astype(allowed.dtype)

            # If one direction advantageous on longer axis - Take it
            def one_long_axis():
                return min_dirs[jnp.argmax(prefered)].astype(allowed.dtype)
            
            # If multiple directions advantageous on longer or equal axis - Choose randomly with mask
            def multi_long_axis():
                prefered_dirs = jnp.compress(prefered, min_dirs, size=DIRECTIONS.shape[0])
                return prefered_dirs[jax.random.randint(key, (), 0, n_prefered)].astype(allowed.dtype)

            # Check for advantageous directions on longer axis
            return jax.lax.cond(
                n_prefered == 0,
                no_long_axis,
                lambda: jax.lax.cond(
                    n_prefered == 1,
                    one_long_axis,
                    multi_long_axis
                )
            )

        # Check for advantageous directions
        return jax.lax.cond(
            n_min == 1,
            one_min,
            multi_min
        )

    # Check for allowed directions
    return jax.lax.cond(
        n_allowed == 0,
        no_allowed,
        lambda: jax.lax.cond(
            n_allowed == 1,
            one_allowed,
            multi_allowed
        )
    )

def reverse_action(dir_idx: chex.Array):
    """Inverts the direction if possible."""
    # Mapping for actions: 0->0, 1->1, 2->5, 3->4, 4->3, 5->2
    inv_map = jnp.array([0, 1, 5, 4, 3, 2], dtype=jnp.uint8)
    idx = jnp.array(dir_idx, dtype=jnp.uint8)
    in_range = (idx >= 0) & (idx < inv_map.shape[0])
    return jnp.where(in_range, inv_map[idx], idx).astype(idx.dtype)

def detect_collision(position_1: chex.Array, position_2: chex.Array):
    """Checks if the two positions are closer than the collision threshold."""
    return jnp.all(abs(position_1.astype(jnp.int32) - position_2.astype(jnp.int32)) < COLLISION_THRESHOLD)


class PacmanRenderer(JAXGameRenderer):
    """JAX-based Pacman game renderer, optimized with JIT compilation."""

    def __init__(self, consts: PacmanConstants = None, config: render_utils.RendererConfig = None, sprite_dir_name: str = "pacman"):
        super().__init__(consts)
        self.consts = consts or PacmanConstants()
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)

        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), sprite_dir_name)

        # Define asset config
        asset_config = [
            {'name': 'dummy_bg', 'type': 'background', 'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)},
            {'name': 'pacman', 'type': 'group', 'files': ['pacman_0.npy', 'pacman_1.npy', 'pacman_2.npy']},
            {'name': 'ghosts', 'type': 'group', 'files': [
                'ghost_0.npy', 'ghost_1.npy', 'ghost_2.npy', 'ghost_3.npy'
            ], 'recolorings': {
                'frightened': tuple(map(int, self.consts.PALE_BLUE_COLOR.tolist())),
                'white': (255, 255, 255)
            }},
            {'name': 'life', 'type': 'single', 'file': 'life.npy'},
            {'name': 'fruit', 'type': 'group', 'files': ['vitamins.npy']},
            {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        ]

        # Include background colors in the palette (Path, Wall, Black, Pink Power Pellet, Pale Blue)
        bg_colors = jnp.stack([
            self.consts.PATH_COLOR, 
            self.consts.WALL_COLOR, 
            jnp.array([0, 0, 0], dtype=jnp.uint8),
            self.consts.POWER_PELLET_COLOR,
            self.consts.PALE_BLUE_COLOR,
            self.consts.GREEN_BAND_COLOR
        ])
        bg_colors = jnp.concatenate([bg_colors, jnp.full((6, 1), 255, dtype=jnp.uint8)], axis=1)
        asset_config.append({'name': 'bg_colors', 'type': 'procedural', 'data': bg_colors[:, None, :]})

        (self.PALETTE, self.SHAPE_MASKS, _, self.COLOR_TO_ID, self.FLIP_OFFSETS) = \
            self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Concatenate recolored ghosts to the main ghosts group
        # Index 4: Frightened (Pale Blue), Index 5: White (Blinking)
        self.SHAPE_MASKS['ghosts'] = jnp.concatenate([
            self.SHAPE_MASKS['ghosts'],
            self.SHAPE_MASKS['ghosts_frightened'][0:1],
            self.SHAPE_MASKS['ghosts_white'][0:1]
        ], axis=0)

        # Make digits black
        black_id = self.COLOR_TO_ID[(0, 0, 0)]
        transparent_id = self.jr.TRANSPARENT_ID
        self.SHAPE_MASKS['digits'] = jnp.where(
            self.SHAPE_MASKS['digits'] != transparent_id,
            black_id,
            self.SHAPE_MASKS['digits']
        )

        # Pre-rotate Pacman masks: 0: UP, 1: RIGHT, 2: LEFT, 3: DOWN
        # Original masks are RIGHT looking
        pacman_raw = self.SHAPE_MASKS['pacman']
        h, w = pacman_raw.shape[1], pacman_raw.shape[2]
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        right = jnp.pad(pacman_raw, ((0, 0), (pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), constant_values=self.jr.TRANSPARENT_ID)
        
        left = jnp.flip(right, axis=2)
        up = jnp.rot90(right, k=1, axes=(1, 2))
        down = jnp.rot90(right, k=3, axes=(1, 2))
        self.PACMAN_MASKS = jnp.stack([up, right, left, down])
        self.LIFE_MASK = self.SHAPE_MASKS['life']
        self.pacman_pad = jnp.array([pad_w, pad_h], dtype=jnp.int32)

        # Pre-calculate backgrounds
        self.MAZE_BACKGROUNDS = self._create_all_backgrounds()

    def _create_all_backgrounds(self):
        bgs = []
        for i in range(4): # 4 mazes for Pacman mods
            bg = PacmanMaze.load_background(i) # Returns (W, H, 3)
            bg = jnp.transpose(bg, (1, 0, 2)) # Convert to (H, W, 3)
            if bg.shape[2] == 3:
                bg = jnp.concatenate([bg, jnp.full((*bg.shape[:2], 1), 255, dtype=jnp.uint8)], axis=2)

            bg_id = self.jr._create_background_raster(bg, self.COLOR_TO_ID)
            bgs.append(bg_id)
        return jnp.stack(bgs)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: PacmanState):
        maze_idx = get_level_maze(state.level.id)
        background = self.MAZE_BACKGROUNDS[maze_idx]
        raster = self.jr.create_object_raster(background)
        
        # 1. Render Pellets
        wall_id = self.COLOR_TO_ID[tuple(map(int, self.consts.WALL_COLOR.tolist()))]
        raster = self.render_pellets(raster, state.level.pellets, wall_id)

        # 2. Power Pellets
        pink_id = self.COLOR_TO_ID[tuple(map(int, self.consts.POWER_PELLET_COLOR.tolist()))]
        pale_blue_id = self.COLOR_TO_ID[tuple(map(int, self.consts.PALE_BLUE_COLOR.tolist()))]
        
        # Check if power pellet effect is active (any ghost frightened or blinking)
        power_pellet_active = jnp.any((state.ghosts.modes == GhostMode.FRIGHTENED) | (state.ghosts.modes == GhostMode.BLINKING))
        power_pellet_color_id = jax.lax.select(power_pellet_active, pale_blue_id, pink_id)
        
        raster = self.render_power_pellets(raster, state, power_pellet_color_id)
        
        # 3. Pacman
        # Pacman should only face LEFT or RIGHT, even when moving UP or DOWN
        orientation = state.player.last_horiz_dir
        
        cycle = (state.step_count // 4) % 4
        frame = jnp.array([0, 1, 2, 1])[cycle]
        pacman_mask = self.PACMAN_MASKS[orientation.astype(jnp.int32), frame.astype(jnp.int32)]
        
        raster = jax.lax.cond(
            state.player.tunnel_timer == 0,
            lambda r: self.jr.render_at(r, state.player.position[0].astype(jnp.int32), state.player.position[1].astype(jnp.int32) - 1, pacman_mask),
            lambda r: r,
            raster
        )
        
        # 4. Ghosts
        raster = self.render_ghosts(raster, state)
        
        # 5. Fruit (Vitamin)
        raster = jax.lax.cond(
            state.fruit.spawned,
            lambda r: self.jr.render_at(r, state.fruit.position[0].astype(jnp.int32), state.fruit.position[1].astype(jnp.int32) - 1, self.SHAPE_MASKS['fruit'][state.fruit.type.astype(jnp.int32)]),
            lambda r: r,
            raster
        )
        
        # 6. UI
        raster = self.render_ui(raster, state)
        
        return self.jr.render_from_palette(raster, self.PALETTE)

    @partial(jax.jit, static_argnums=(0,))
    def render_ghosts(self, raster, state):
        anim_frame = (state.step_count & 0b10000) >> 4
        
        def render_one(i, r):
            mode = state.ghosts.modes[i]
            is_frightened = (mode == GhostMode.FRIGHTENED) | (mode == GhostMode.BLINKING)
            
            ghost_idx = jax.lax.cond(
                is_frightened,
                lambda: jnp.array(4, dtype=jnp.int32), 
                lambda: i
            ).astype(jnp.int32)
            
            mask = self.SHAPE_MASKS['ghosts'][ghost_idx]
            flip = anim_frame == 1
            return self.jr.render_at(r, state.ghosts.positions[i][0].astype(jnp.int32) + 3, state.ghosts.positions[i][1].astype(jnp.int32) - 2, mask, flip_horizontal=flip)

        return jax.lax.fori_loop(0, 4, render_one, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_pellets(self, raster, pellets, color_id):
        x_range, y_range = jnp.nonzero(pellets, size=pellets.size)
        n_pellets = jnp.sum(pellets)
        mask = jnp.arange(pellets.size) < n_pellets

        x_positions = x_range * 8 + 8
        x_positions = jnp.where(x_positions > 74, x_positions + 4, x_positions)
        x_positions = jnp.where(y_range % 2, x_positions, jnp.where(60 < x_positions, jnp.where(x_positions < 76,  x_positions + 4, x_positions), x_positions))
        x_positions = jnp.where(y_range % 2, x_positions,  jnp.where(76 < x_positions, jnp.where(x_positions < 100,  x_positions - 4, x_positions), x_positions))
        y_positions = y_range * 24 + 11

        positions = jnp.stack([x_positions, y_positions], axis=1).astype(jnp.int32)
        positions = jnp.where(mask[:, None], positions, -1)
        sizes = jnp.tile(jnp.array([4, 2], dtype=jnp.int32), (pellets.size, 1))

        return self.jr.draw_rects(raster, positions, sizes, color_id)

    @partial(jax.jit, static_argnums=(0,))
    def render_power_pellets(self, raster, state, color_id):
        # 10x4 sprite (10 height, 4 width)
        sprite = jnp.full((10, 4), color_id, dtype=raster.dtype)
        
        def render_one(i, r):
            # Reactivated: removed blinking to make them more visible
            should_draw = state.level.power_pellets[i]
            
            # Left pellets (i=0, 2) shift left by 2px, Right pellets (i=1, 3) shift right by 2px
            x_offset = jnp.where(i % 2 == 0, -2, 2)
            
            x = (self.consts.POWER_PELLET_TILES[i][0] * 4 + 4 + x_offset).astype(jnp.int32)
            y = (self.consts.POWER_PELLET_TILES[i][1] * 4 + 7).astype(jnp.int32)
            return jax.lax.cond(should_draw, 
                                lambda r_in: self.jr.render_at(r_in, x, y, sprite),
                                lambda r_in: r_in,
                                r)
        
        return jax.lax.fori_loop(0, 4, render_one, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_ui(self, raster, state):
        # Green Band
        green_id = self.COLOR_TO_ID[tuple(map(int, self.consts.GREEN_BAND_COLOR.tolist()))]
        raster = self.jr.draw_rects(raster, jnp.array([[0, 193]]), jnp.array([[160, self.consts.GREEN_BAND_HEIGHT]]), green_id)

        # Score
        digits = self.jr.int_to_digits(state.score, max_digits=self.consts.MAX_SCORE_DIGITS)
        digit_count = get_digit_count(state.score).astype(jnp.int32)
        start_index = self.consts.MAX_SCORE_DIGITS - digit_count
        render_x = 60 + start_index * 8
        raster = self.jr.render_label_selective(raster, render_x, 194, digits, self.SHAPE_MASKS['digits'], start_index, digit_count, spacing=8, max_digits_to_render=self.consts.MAX_SCORE_DIGITS)
        
        # Lives
        raster = self.jr.render_indicator(raster, 12, 203, (state.lives - 1).astype(jnp.int32), self.LIFE_MASK, spacing=14, max_value=self.consts.MAX_LIVE_COUNT)
        
        # Fruit indicator
        # fruit_mask = self.SHAPE_MASKS['fruit'][state.fruit.type.astype(jnp.int32)]
        # raster = self.jr.render_at(raster, 128, 182, fruit_mask)
        
        return raster

def dof(pos: chex.Array, dofmaze: chex.Array, is_ghost: bool = False):
    """Degree of freedom of the object, can it move up, right, left, down"""
    x, y = pos
    grid_x = jnp.clip((x + 5) // 4, 0, 39)
    grid_y = jnp.clip((y + 3) // 4, 0, 47)
    up, right, left, down = dofmaze[grid_x, grid_y]

    # Restrict vertical wrap (Atari 2600 Pacman has vertical tunnels only at the center)
    # The tunnel is roughly at X = 71 to 74
    is_in_tunnel = (x >= 71) & (x <= 74)
    is_at_top = (y <= 7)
    is_at_bottom = (y >= 184)

    # Disable movement that would wrap if not in the tunnel
    up = jnp.where(is_at_top & ~is_in_tunnel, False, up)
    down = jnp.where(is_at_bottom & ~is_in_tunnel, False, down)

    # Fix precomputed dofmaze bug: wrapper maps Y=0 UP to row 46 which has walls!
    # Force it to True for Pacman if in tunnel.
    up = jnp.where(~jnp.array(is_ghost, dtype=jnp.bool_) & is_at_top & is_in_tunnel, True, up)
    down = jnp.where(~jnp.array(is_ghost, dtype=jnp.bool_) & is_at_bottom & is_in_tunnel, True, down)

    # For ghosts, if they are not allowed in the tunnel, block them at the boundary
    up = jnp.where(jnp.array(is_ghost, dtype=jnp.bool_) & is_at_top & is_in_tunnel, False, up)
    down = jnp.where(jnp.array(is_ghost, dtype=jnp.bool_) & is_at_bottom & is_in_tunnel, False, down)

    # Ghosts can exit the cage through the center door
    is_cage_door = (x >= 71) & (x <= 75) & (y >= 73) & (y <= 78)
    up = jnp.where(jnp.array(is_ghost, dtype=jnp.bool_) & is_cage_door, True, up)

    return up, right, left, down

def available_directions(pos: chex.Array, dofmaze: chex.Array, is_ghost: bool = False):
    """
    What direction Pacman or the ghosts can take when at an intersection.
    """
    x, y = pos
    on_vertical_grid = x % 4 == 1
    on_horizontal_grid = y % 12 == 6
    
    up, right, left, down = dof(pos, dofmaze, is_ghost)
    
    return jnp.array([
        up & on_vertical_grid,
        right & on_horizontal_grid,
        left & on_horizontal_grid,
        down & on_vertical_grid
    ], dtype=jnp.bool_)

def get_allowed_directions(position: chex.Array, action: chex.Array, dofmaze: chex.Array, is_ghost: bool = False):
    """
    Returns an array of all directions (JAXAtari actions) in which movement is possible.
    """
    direction_count = 4 # UP, RIGHT, LEFT, DOWN
    
    def at_center(_):
        avail = available_directions(position, dofmaze, is_ghost)
        
        # Directions that are not the reverse of current action
        not_reverse_mask = jnp.arange(direction_count) != act_to_dir(reverse_action(action))
        
        allowed_mask = avail & not_reverse_mask
        allowed_actions = jnp.where(allowed_mask, CONSTS.DIRECTIONS, 0)
        return jnp.compress(allowed_actions != 0, allowed_actions, size=direction_count).astype(jnp.uint8)

    def not_at_center(_):
        return jnp.zeros(direction_count, dtype=jnp.uint8).at[0].set(jnp.array(action, dtype=jnp.uint8))

    # Check if the position is at the center of a tile
    at_tile_center = (position[0] % 4 == 1) | (position[1] % 12 == 6)
    return jax.lax.cond(
        at_tile_center,
        at_center,
        not_at_center,
        None
    )

def stop_wall(pos: chex.Array, dofmaze: chex.Array):
    """
    What directions are blocked for Pacman or the ghosts when at an intersection.
    """
    x, y = pos
    on_vertical_grid = x % 4 == 1
    on_horizontal_grid = y % 12 == 6
    up, right, left, down = dof(pos, dofmaze, is_ghost=False)
    return jnp.array([
        ~up & on_horizontal_grid,
        ~right & on_vertical_grid,
        ~left & on_vertical_grid,
        ~down & on_horizontal_grid
    ], dtype=jnp.bool_)

class JaxPacman(JaxEnvironment[PacmanState, PacmanObservation, PacmanInfo, PacmanConstants]):
    def __init__(self, consts: PacmanConstants = None):
        consts = consts or PacmanConstants()
        super().__init__(consts)
        self.frame_stack_size = 1
        self.action_set = [
            Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
            Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        ]
        self.renderer = PacmanRenderer(self.consts, sprite_dir_name="pacman")

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(10)

    def reset(self, key=None) -> Tuple[PacmanObservation, PacmanState]:
        if key is None:
            key = jax.random.PRNGKey(0)
        state = reset_game(self.consts.RESET_LEVEL, self.consts.INITIAL_LIVES, 0, key)
        return self.get_observation(state), state

    def render(self, state: PacmanState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: PacmanState, action: chex.Array) -> tuple[
        PacmanObservation, PacmanState, jax.Array, jax.Array, PacmanInfo]:
        key, step_key = jax.random.split(state.key)
        
        (new_state, frozen, done) = self.death_step(state, step_key)
        
        (
            player_position, player_action, pellets, has_pellet,
            eaten_pellets, power_pellets, ate_power_pellet,
            pellet_reward, level_id, new_tunnel_timer
        ) = self.player_step(state, action)

        (fruit_state, fruit_reward) = self.fruit_step(state, player_position, eaten_pellets, step_key)

        (
            ghost_positions, ghost_actions, ghost_modes, ghost_timers,
            eaten_ghosts, new_lives, new_death_timer, ghosts_reward
        ) = self.ghosts_step(state, ate_power_pellet, step_key)

        reward = pellet_reward + fruit_reward + ghosts_reward
        new_score = state.score + reward
        score_changed = self.flag_score_change(state.score, new_score)
        
        new_state = jax.lax.cond(
            frozen,
            lambda: new_state._replace(key=key),
            lambda: jax.lax.cond(
                level_id != state.level.id,
                lambda: reset_maze(level_id, state.lives, new_score, key),
                lambda: PacmanState(
                    level = LevelState(
                        id=level_id,
                        eaten_pellets=eaten_pellets,
                        dofmaze=state.level.dofmaze,
                        pellets=pellets,
                        power_pellets=power_pellets,
                        loaded=jax.lax.cond(state.level.loaded < 2, lambda: state.level.loaded + 1, lambda: state.level.loaded)
                    ),
                    player = PlayerState(
                        position=player_position, 
                        action=player_action, 
                        has_pellet=has_pellet, 
                        eaten_ghosts=eaten_ghosts,
                        last_horiz_dir=jax.lax.cond(
                            (player_action == Action.LEFT) | (player_action == Action.RIGHT),
                            lambda: act_to_dir(player_action).astype(jnp.int32),
                            lambda: state.player.last_horiz_dir
                        ),
                        tunnel_timer=new_tunnel_timer
                    ),
                    ghosts = GhostsState(positions=ghost_positions, actions=ghost_actions, modes=ghost_modes, timers=ghost_timers),
                    fruit=fruit_state,
                    lives=new_lives,
                    score=new_score,
                    score_changed=score_changed,
                    freeze_timer=new_death_timer,
                    step_count=state.step_count + 1,
                    key=key
                )
            )
        )

        observation = self.get_observation(new_state)
        info = self.get_info(new_state)
        reward_val = jax.lax.cond(frozen, lambda: jnp.array(0, dtype=jnp.uint32), lambda: jnp.array(reward, dtype=jnp.uint32))
        return observation, new_state, reward_val, done, info
    
    @staticmethod
    @jax.jit
    def get_observation(state: PacmanState):
        return PacmanObservation(
            player_position=state.player.position,
            player_action=state.player.action,
            ghost_positions=state.ghosts.positions,
            ghost_actions=state.ghosts.actions,
            fruit_position=state.fruit.position,
            fruit_action=state.fruit.position,
            fruit_type=state.fruit.type,
            pellets=state.level.pellets,
            power_pellets=state.level.power_pellets
        )

    @staticmethod
    @jax.jit
    def get_info(state: PacmanState):
        return PacmanInfo(level=state.level.id, score=state.score, lives=state.lives)
    
    @staticmethod
    def death_step(state: PacmanState, key: chex.PRNGKey):
        def decrement_timer(state: PacmanState):
            return state._replace(freeze_timer=state.freeze_timer - 1)

        return jax.lax.cond(
            state.freeze_timer == 0,
            lambda: (state, False, False),
            lambda: jax.lax.cond(
                state.freeze_timer > 1,
                lambda: (decrement_timer(state), True, False),
                lambda: jax.lax.cond(
                    state.lives == 0,
                    lambda: (decrement_timer(state), True, True),
                    lambda: (reset_entities(decrement_timer(state), key), True, False)
                )
            )
        )

    @staticmethod
    def player_step(state: PacmanState, action: chex.Array):
        action = jnp.array(action, dtype=jnp.int32)
        action = last_pressed_action(action, state.player.action)
        action = jax.lax.cond((action < 0) | (action > len(CONSTS.ACTIONS) - 1), lambda: jnp.array(Action.NOOP, dtype=jnp.int32), lambda: action)

        def normal_step(_):
            available = available_directions(state.player.position, state.level.dofmaze)
            new_action = jax.lax.cond((action != Action.NOOP) & (action != Action.FIRE) & available[act_to_dir(action)], lambda: action, lambda: state.player.action)
            
            # Check for tunnel entry
            is_in_tunnel_x = (state.player.position[0] >= 71) & (state.player.position[0] <= 74)
            # Wrap points: moving UP from top or moving DOWN from bottom
            next_y = state.player.position[1] + CONSTS.ACTIONS[new_action][1] * CONSTS.VERTICAL_SPEED
            will_wrap_up = (next_y < 0) & (new_action == Action.UP)
            will_wrap_down = (next_y >= 192) & (new_action == Action.DOWN)
            is_tunnel_entry = is_in_tunnel_x & (will_wrap_up | will_wrap_down)
            
            def tunnel_entry_step(_):
                return state.player.position, new_action, jnp.array(CONSTS.TUNNEL_DELAY, dtype=jnp.int32)
            
            def standard_step(_):
                is_blocked = stop_wall(state.player.position, state.level.dofmaze)[act_to_dir(new_action)]
                res_pos = jax.lax.cond(is_blocked, lambda: state.player.position, lambda: get_new_position(state.player.position, new_action, speed=jnp.array([CONSTS.HORIZONTAL_SPEED, CONSTS.VERTICAL_SPEED])))
                return res_pos, new_action, jnp.array(0, dtype=jnp.int32)
                
            return jax.lax.cond(is_tunnel_entry, tunnel_entry_step, standard_step, None)

        def tunnel_timer_step(_):
            new_timer = state.player.tunnel_timer - 1
            
            def teleport(_):
                # Wrapped position
                wrapped_pos = get_new_position(state.player.position, state.player.action, speed=jnp.array([CONSTS.HORIZONTAL_SPEED, CONSTS.VERTICAL_SPEED]))
                return wrapped_pos, state.player.action, jnp.array(0, dtype=jnp.int32)
                
            def wait(_):
                return state.player.position, state.player.action, new_timer
                
            return jax.lax.cond(new_timer == 0, teleport, wait, None)

        player_pos, player_action, tunnel_timer = jax.lax.cond(
            state.player.tunnel_timer > 0,
            tunnel_timer_step,
            normal_step,
            None
        )

        (pellets, has_pellet, eaten_pellets, power_pellets, ate_power_pellet, reward, level_id) = JaxPacman.pellet_step(state, player_pos)
        
        # Disable interactions if in tunnel
        reward = jax.lax.select(state.player.tunnel_timer > 0, 0, reward)
        has_pellet = jax.lax.select(state.player.tunnel_timer > 0, False, has_pellet)
        
        return (player_pos, player_action, pellets, has_pellet, eaten_pellets, power_pellets, ate_power_pellet, reward, level_id, tunnel_timer)

    @staticmethod
    def pellet_step(state: PacmanState, new_pacman_pos: chex.Array):
        def check_power_pellet(idx: chex.Array, power_pellets: chex.Array):
            return jax.lax.cond(idx < 0, lambda: False, lambda: power_pellets[idx % 4])
        
        def eat_power_pellet(idx: chex.Array, power_pellets: chex.Array):
            return power_pellets.at[idx % 4].set(False)
        
        def check_pellet(pos: chex.Array):
            return pos[1] % 24 == 6

        def eat_pellet(pos: chex.Array, pellets: chex.Array):
            adjusted_x = jax.lax.select(pos[0] > 85, pos[0] - 4, pos[0])
            tile_y = (pos[1] + 4) // 24
            
            tile_x1 = (adjusted_x - 2) // 8
            tile_x2 = (adjusted_x + 1) // 8
            
            in_bounds1 = (tile_x1 >= 0) & (tile_x1 < pellets.shape[0]) & (tile_y >= 0) & (tile_y < pellets.shape[1])
            in_bounds2 = (tile_x2 >= 0) & (tile_x2 < pellets.shape[0]) & (tile_y >= 0) & (tile_y < pellets.shape[1])
            
            has_1 = pellets[tile_x1, tile_y] & in_bounds1
            has_2 = pellets[tile_x2, tile_y] & in_bounds2
            
            return jax.lax.cond(
                has_1,
                lambda: (pellets.at[tile_x1, tile_y].set(False), True),
                lambda: jax.lax.cond(
                    has_2,
                    lambda: (pellets.at[tile_x2, tile_y].set(False), True),
                    lambda: (pellets, False)
                )
            )
        pellets, ate_pellet = jax.lax.cond(check_pellet(new_pacman_pos), lambda: eat_pellet(new_pacman_pos, state.level.pellets), lambda: (state.level.pellets, False))
        power_pellet_hit = jnp.where(jnp.all(jnp.round(new_pacman_pos / PacmanMaze.TILE_SCALE) == CONSTS.POWER_PELLET_HITBOXES, axis=1), size=1, fill_value=-1)[0][0]
        power_pellets, ate_power_pellet = jax.lax.cond(check_power_pellet(power_pellet_hit, state.level.power_pellets), lambda: (eat_power_pellet(power_pellet_hit, state.level.power_pellets), True), lambda: (state.level.power_pellets, False))
        
        reward = jax.lax.cond(ate_power_pellet, lambda: CONSTS.POWER_PELLET_POINTS, lambda: jax.lax.cond(ate_pellet, lambda: CONSTS.PELLET_POINTS, lambda: 0))
        has_pellet = ate_power_pellet | ate_pellet
        eaten_pellets = jax.lax.cond(has_pellet, lambda: state.level.eaten_pellets + 1, lambda: state.level.eaten_pellets)
        
        level_id, reward = jax.lax.cond(eaten_pellets >= CONSTS.PELLETS_TO_COLLECT[get_level_maze(state.level.id)], lambda: (state.level.id + 1, reward + CONSTS.LEVEL_COMPLETED_POINTS), lambda: (state.level.id, reward))
        return (pellets, has_pellet, eaten_pellets, power_pellets, ate_power_pellet, reward, level_id)

    @staticmethod
    def ghosts_step(state: PacmanState, ate_power_pellet: chex.Array, common_key: chex.Array):
        def update_ghost_mode(mode, action, timer, step_count, ate_power_pellet):
            new_timer = jax.lax.cond(timer > 0, lambda: jnp.array(timer - 1.0, dtype=jnp.float16), lambda: jnp.array(timer, dtype=jnp.float16))
            timing_factor = jax.lax.cond(state.level.id == 1, lambda: 1.0, lambda: CONSTS.FRIGHTENED_REDUCTION ** (state.level.id - 1))
            
            def start_scatter(action, step_count):
                return (jnp.array(GhostMode.SCATTER, dtype=jnp.uint8), jnp.array(action, dtype=jnp.uint8), jnp.array(CONSTS.SCATTER_DURATION, dtype=jnp.float16), False)
            def start_chase(action, step_count):
                return (jnp.array(GhostMode.CHASE, dtype=jnp.uint8), jnp.array(action, dtype=jnp.uint8), jnp.array(CONSTS.CHASE_DURATION, dtype=jnp.float16), False)
            def start_blinking(action, step_count):
                # Transition directly to chase/returning, effectively removing the blinking state
                return start_returned(action, step_count)
            def start_returning(action, step_count):
                return (jnp.array(GhostMode.RETURNING, dtype=jnp.uint8), jnp.array(Action.UP, dtype=jnp.uint8), jnp.array(CONSTS.RETURN_DURATION, dtype=jnp.float16), True)
            def start_returned(action, step_count):
                return (jnp.array(GhostMode.CHASE, dtype=jnp.uint8), jnp.array(action, dtype=jnp.uint8), jnp.array(CONSTS.CHASE_DURATION, dtype=jnp.float16), False)

            return jax.lax.cond(
                ate_power_pellet & (mode != GhostMode.ENJAILED) & (mode != GhostMode.RETURNING),
                lambda: (jnp.array(GhostMode.FRIGHTENED, dtype=jnp.uint8), jnp.array(reverse_action(action), dtype=jnp.uint8), jnp.array((CONSTS.FRIGHTENED_DURATION + CONSTS.BLINKING_DURATION) * timing_factor, dtype=jnp.float16), True),
                lambda: jax.lax.cond(
                    (timer > 0) & (new_timer <= 0),
                    lambda: jax.lax.switch(mode, (start_returned, start_scatter, start_chase, start_blinking, start_returned, start_returned, start_returning), action, step_count),
                    lambda: (mode, action, new_timer, False)
                )
            )

        def ghost_step(ghost_index: int, new_ghost_states: Tuple):
            new_mode, new_action, new_timer, skip = update_ghost_mode(state.ghosts.modes[ghost_index], state.ghosts.actions[ghost_index], state.ghosts.timers[ghost_index], state.step_count, ate_power_pellet)
            allowed = get_allowed_directions(state.ghosts.positions[ghost_index], new_action, state.level.dofmaze, is_ghost=True)
            n_allowed = jnp.sum(allowed != 0)
            
            chase_target = jax.lax.cond(new_mode == GhostMode.CHASE, lambda: get_chase_target(state.player.position), lambda: CONSTS.SCATTER_TARGETS[ghost_index])
            new_action = jax.lax.cond(skip | (new_mode == GhostMode.ENJAILED) | (new_mode == GhostMode.RETURNING), lambda: new_action, lambda: jax.lax.cond(n_allowed == 0, lambda: jax.lax.cond((state.ghosts.positions[ghost_index][0] >= 71) & (state.ghosts.positions[ghost_index][0] <= 74) & ((state.ghosts.positions[ghost_index][1] <= 7) | (state.ghosts.positions[ghost_index][1] >= 184)), lambda: reverse_action(new_action), lambda: new_action), lambda: jax.lax.cond(n_allowed == 1, lambda: allowed[0], lambda: jax.lax.cond((new_mode == GhostMode.FRIGHTENED) | (new_mode == GhostMode.BLINKING), lambda: allowed[jax.random.randint(ghost_keys[ghost_index], (), 0, n_allowed)], lambda: pathfind(state.ghosts.positions[ghost_index], new_action, chase_target, allowed, ghost_keys[ghost_index])))))
            
            slow_down = ((new_mode == GhostMode.FRIGHTENED) | (new_mode == GhostMode.BLINKING) | (new_mode == GhostMode.RETURNING)) & (state.step_count % 2 == 0)
            new_pos = jax.lax.cond(slow_down, lambda: state.ghosts.positions[ghost_index], lambda: get_new_position(state.ghosts.positions[ghost_index], new_action))
            
            return (new_ghost_states[0].at[ghost_index].set(new_mode), new_ghost_states[1].at[ghost_index].set(new_action), new_ghost_states[2].at[ghost_index].set(new_pos), new_ghost_states[3].at[ghost_index].set(new_timer))

        ghost_keys = jax.random.split(common_key, 4)
        (new_modes, new_actions, new_positions, new_timers) = jax.lax.fori_loop(0, 4, ghost_step, (jnp.zeros(4, dtype=jnp.uint8), jnp.zeros(4, dtype=jnp.uint8), jnp.zeros((4, 2), dtype=jnp.int32), jnp.zeros(4, dtype=jnp.float16)))
        
        (new_positions, new_actions, new_modes, new_timers, eaten_ghosts, new_lives, new_death_timer, reward) = JaxPacman.ghosts_collision(new_positions, new_actions, new_modes, new_timers, state.player.position, state.player.eaten_ghosts, ate_power_pellet, state.lives, state.player.tunnel_timer)
        return new_positions, new_actions, new_modes, new_timers, eaten_ghosts, new_lives, new_death_timer, reward

    @staticmethod
    def ghosts_collision(ghost_positions, ghost_actions, ghost_modes, ghost_timers, new_pacman_pos, eaten_ghosts, ate_power_pellet, lives, tunnel_timer):
        class GhostStates(NamedTuple):
            pacman_position: chex.Array
            reward: chex.Array
            ghost_positions: chex.Array
            ghost_actions: chex.Array
            ghost_modes: chex.Array
            ghost_timers: chex.Array
            eaten_ghosts: chex.Array
            deadly_collision: chex.Array

        def handle_ghost_collision(ghost_index: int, ghost_states: GhostStates):
            def handle_eaten():
                reward_inc = CONSTS.EAT_GHOSTS_POINTS[jnp.minimum(ghost_states.eaten_ghosts, 3)]
                return GhostStates(ghost_states.pacman_position, ghost_states.reward + reward_inc, ghost_states.ghost_positions.at[ghost_index].set(CONSTS.JAIL_POSITION), ghost_states.ghost_actions.at[ghost_index].set(Action.NOOP), ghost_states.ghost_modes.at[ghost_index].set(GhostMode.ENJAILED.value), ghost_states.ghost_timers.at[ghost_index].set(CONSTS.ENJAILED_DURATION), ghost_states.eaten_ghosts + 1, False)
            def handle_death():
                return ghost_states._replace(deadly_collision=True)
            
            is_active = (ghost_states.ghost_modes[ghost_index] != GhostMode.ENJAILED.value) & \
                        (ghost_states.ghost_modes[ghost_index] != GhostMode.RETURNING.value)
            is_collision = (tunnel_timer == 0) & is_active & detect_collision(ghost_states.pacman_position, ghost_states.ghost_positions[ghost_index])
            return jax.lax.cond(is_collision, lambda: jax.lax.cond((ghost_states.ghost_modes[ghost_index] == GhostMode.FRIGHTENED) | (ghost_states.ghost_modes[ghost_index] == GhostMode.BLINKING), handle_eaten, handle_death), lambda: ghost_states)

        new_eaten = jax.lax.cond(ate_power_pellet, lambda: jnp.array(0, dtype=jnp.uint8), lambda: jnp.array(eaten_ghosts, dtype=jnp.uint8))
        res = jax.lax.fori_loop(0, 4, handle_ghost_collision, GhostStates(new_pacman_pos, jnp.array(0, dtype=jnp.uint32), ghost_positions, ghost_actions, ghost_modes, ghost_timers, new_eaten, False))
        
        new_lives = (lives - jnp.where(res.deadly_collision, 1, 0)).astype(jnp.int8)
        new_death_timer = jnp.where(res.deadly_collision, CONSTS.RESET_TIMER, 0).astype(jnp.uint32)
        return res.ghost_positions, res.ghost_actions, res.ghost_modes, res.ghost_timers, res.eaten_ghosts, new_lives, new_death_timer, res.reward

    @staticmethod
    def fruit_step(state: PacmanState, new_pacman_pos: chex.Array, eaten_pellets: chex.Array, key: chex.Array):
        def spawn_vitamin():
            return FruitState(position=CONSTS.VITAMINS_POSITION.astype(jnp.uint8), exit=CONSTS.VITAMINS_POSITION.astype(jnp.uint8), type=jnp.array(0, dtype=jnp.uint8), action=jnp.array(Action.NOOP, dtype=jnp.uint8), spawn=jnp.array(False, dtype=jnp.bool), spawned=jnp.array(True, dtype=jnp.bool), timer=jnp.array(CONSTS.VITAMINS_DURATION, dtype=jnp.uint16)), 0
        def consume_vitamin():
            return FruitState(jnp.zeros(2, dtype=jnp.uint8), jnp.zeros(2, dtype=jnp.uint8), jnp.array(0, dtype=jnp.uint8), jnp.array(Action.NOOP, dtype=jnp.uint8), state.fruit.spawn, jnp.array(False, dtype=jnp.bool), jnp.array(CONSTS.VITAMINS_DURATION, dtype=jnp.uint16)), CONSTS.VITAMINS_REWARD
        def step_vitamin():
            new_timer = jax.lax.cond(state.fruit.timer > 0, lambda: state.fruit.timer - 1, lambda: state.fruit.timer)
            return state.fruit._replace(timer=new_timer), 0
        def clear_vitamin():
            return state.fruit._replace(spawned=False), 0

        fruit_spawn = jnp.any(CONSTS.VITAMINS_SPAWN_THRESHOLDS == eaten_pellets) & state.player.has_pellet
        
        return jax.lax.cond(
            state.fruit.spawned,
            lambda: jax.lax.cond(detect_collision(new_pacman_pos, state.fruit.position), consume_vitamin, lambda: jax.lax.cond(state.fruit.timer == 0, clear_vitamin, step_vitamin)),
            lambda: jax.lax.cond(fruit_spawn | state.fruit.spawn, spawn_vitamin, lambda: (state.fruit, 0))
        )

    @staticmethod
    def flag_score_change(current_score: chex.Array, new_score: chex.Array):
        def int_to_digits(n, max_digits):
            n = jnp.maximum(n, 0)
            def scan_body(carry, _):
                return carry // 10, carry % 10
            _, digits = jax.lax.scan(scan_body, n, None, length=max_digits)
            return jnp.flip(digits, axis=0)
        return int_to_digits(new_score, CONSTS.MAX_SCORE_DIGITS) != int_to_digits(current_score, CONSTS.MAX_SCORE_DIGITS)

def get_chase_target(player_pos: chex.Array) -> chex.Array:
    """
    Compute the chase-mode target for each ghost.
    In Pacman, all ghosts just target the player directly.
    """
    return player_pos.astype(jnp.int32)

def get_new_position(position: chex.Array, action: chex.Array, speed: chex.Array = jnp.array([1, 1])):
    new_position = position + CONSTS.ACTIONS[action] * speed
    # Atari 2600 Pacman has vertical tunnels (wrapping Y), not horizontal (wrapping X)
    # The maze height is 48 rows * 4 pixels = 192 pixels.
    return new_position.at[1].set(new_position[1] % 192).astype(position.dtype)

# -------- Reset functions --------
def reset_game(level: chex.Array, lives: chex.Array, score: chex.Array, key: chex.PRNGKey):
    return PacmanState(level=reset_level(level), player=reset_player(), ghosts=reset_ghosts(), fruit=reset_fruit(level, key), lives=jnp.array(lives, dtype=jnp.int8), score=jnp.array(score, dtype=jnp.uint32), score_changed=jnp.zeros(6, dtype=jnp.bool_), freeze_timer=jnp.array(0, dtype=jnp.uint32), step_count=jnp.array(0, dtype=jnp.uint32), key=key)

def reset_level(level: chex.Array):
    return LevelState(id=jnp.array(level, dtype=jnp.uint8), eaten_pellets=jnp.array(0, dtype=jnp.uint8), dofmaze=PacmanMaze.precompute_dof(get_level_maze(level)), pellets=jnp.copy(PacmanMaze.BASE_PELLETS), power_pellets=jnp.ones(4, dtype=jnp.bool_), loaded=jnp.array(0, dtype=jnp.uint8))

def reset_maze(level: chex.Array, lives: chex.Array, score: chex.Array, key: chex.PRNGKey):
    """Resets the maze after clearing it: pellets reset, extra life, positions reset."""
    new_lives = jnp.minimum(lives + 1, CONSTS.MAX_LIVE_COUNT).astype(jnp.int8)
    return reset_game(level, new_lives, score, key)

def reset_player():
    return PlayerState(
        position=CONSTS.INITIAL_PACMAN_POSITION.astype(jnp.uint8), 
        action=jnp.array(Action.LEFT, dtype=jnp.int32), 
        has_pellet=jnp.array(False, dtype=jnp.bool_), 
        eaten_ghosts=jnp.array(0, dtype=jnp.uint8),
        last_horiz_dir=jnp.array(2, dtype=jnp.int32),
        tunnel_timer=jnp.array(0, dtype=jnp.int32)
    )

def reset_ghosts():
    return GhostsState(positions=jnp.tile(CONSTS.INITIAL_GHOST_POSITION, (4, 1)).astype(jnp.int32), actions=jnp.array([Action.LEFT, Action.NOOP, Action.NOOP, Action.NOOP], dtype=jnp.uint8), modes=jnp.array([GhostMode.RANDOM, GhostMode.ENJAILED, GhostMode.ENJAILED, GhostMode.ENJAILED], dtype=jnp.uint8), timers=jnp.array([CONSTS.SCATTER_DURATION, CONSTS.PINKY_RELEASE_TIME, CONSTS.INKY_RELEASE_TIME, CONSTS.SUE_RELEASE_TIME], dtype=jnp.float16))

def reset_fruit(level: chex.Array, key: chex.PRNGKey):
    return FruitState(position=jnp.zeros(2, dtype=jnp.uint8), exit=jnp.zeros(2, dtype=jnp.uint8), type=jnp.array(0, dtype=jnp.uint8), action=jnp.array(Action.NOOP, dtype=jnp.uint8), spawn=jnp.array(False, dtype=jnp.bool_), spawned=jnp.array(False, dtype=jnp.bool_), timer=jnp.array(CONSTS.VITAMINS_DURATION, dtype=jnp.uint16))

def reset_entities(state: PacmanState, key: chex.PRNGKey):
    return state._replace(player=reset_player(), ghosts=reset_ghosts(), fruit=reset_fruit(state.level.id, key), step_count=jnp.array(0, dtype=jnp.uint32))
