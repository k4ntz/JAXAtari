"""
Pacman implementation for JAXAtari.
Based on the original Atari 2600 version.
"""

from enum import IntEnum
import os
from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.games.jax_mspacman import (
    GhostMode, GhostType, FruitType, 
    LevelState, GhostsState, PlayerState, FruitState, PacmanState,
    PacmanObservation, PacmanInfo, MsPacmanRenderer,
    available_directions, stop_wall, get_allowed_directions,
    pathfind,
    reverse_action, detect_collision, act_to_dir, dir_to_act,
    last_pressed_action, get_digit_count
)

def get_level_maze(level: chex.Array):
    return jnp.array(0, dtype=jnp.int32)


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
# MAZE = jnp.array([
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
#         [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ], dtype=bool)

class PacmanMaze:
    MAZE0 = MAZE
    MAZES = jnp.array([MAZE0], dtype=jnp.bool_)
    TILE_SCALE = 4
    WIDTH = 160
    HEIGHT = 160
    
    WALL_COLOR = jnp.array([228, 111, 111], dtype=jnp.uint8)
    PATH_COLOR = jnp.array([0, 28, 136], dtype=jnp.uint8)

    # We reuse MsPacman maze 0 pellets for compatibility
    BASE_PELLETS = jnp.array([ 
       	[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
       	], dtype=bool)

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
        no_wall_below = jnp.roll(sum_horizontal_strip, -2, axis=0) == 0
        no_wall_left = jnp.roll(sum_vertical_strip, 2, axis=1) == 0
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
    PELLETS_TO_COLLECT: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([154, 150, 158, 154]))

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

    # VITAMINS (Fruit)
    VITAMIN_SPAWN_THRESHOLDS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([50, 100]))
    VITAMIN_DURATION: int = struct.field(pytree_node=False, default=10*20) # Stationary for a few moments
    VITAMIN_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([75, 75])) # Center of playfield

    # POSITIONS
    POWER_PELLET_TILES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[1, 3], [36, 3], [1, 36], [36, 36]]))
    POWER_PELLET_HITBOXES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[1, 3], [36, 3], [1, 36], [36, 36], [1, 4], [36, 4], [1, 37], [36, 37]]))
    JAIL_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([75, 75]))
    INITIAL_GHOSTS_POSITIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[75, 54], [75, 75], [75, 75], [75, 75]]))
    INITIAL_PACMAN_POSITION: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([75, 102]))
    SCATTER_TARGETS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([[PacmanMaze.WIDTH - 1, 0], [0, 0], [PacmanMaze.WIDTH - 1, PacmanMaze.HEIGHT - 1], [0, PacmanMaze.HEIGHT - 1]]))

    # ACTIONS
    DIRECTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN]))
    ACTIONS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([(0, 0), (0, 0), (0, -1), (1, 0), (-1, 0), (0, 1)]))
    INITIAL_ACTION: int = struct.field(pytree_node=False, default=Action.LEFT)

    # POINTS (Updated for Pacman)
    PELLET_POINTS: int = struct.field(pytree_node=False, default=1)
    POWER_PELLET_POINTS: int = struct.field(pytree_node=False, default=5)
    VITAMIN_REWARD: int = struct.field(pytree_node=False, default=100)
    EAT_GHOSTS_POINTS: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([20, 40, 80, 160], dtype=jnp.uint32))
    LEVEL_COMPLETED_POINTS: int = struct.field(pytree_node=False, default=0)

    # COLORS
    PATH_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([0, 28, 136], dtype=jnp.uint8))
    WALL_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([228, 111, 111], dtype=jnp.uint8))
    PELLET_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([228, 111, 111], dtype=jnp.uint8))
    POWER_PELLET_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([252, 144, 200], dtype=jnp.uint8))
    PACMAN_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([210, 164, 74, 255], dtype=jnp.uint8))
    PALE_BLUE_COLOR: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([144, 144, 252], dtype=jnp.uint8))

CONSTS = PacmanConstants()

class PacmanRenderer(MsPacmanRenderer):
    """JAX-based Pacman game renderer, optimized with JIT compilation."""

    def __init__(self, consts: PacmanConstants = None, config: render_utils.RendererConfig = None, sprite_dir_name: str = "pacman"):
        super(MsPacmanRenderer, self).__init__(consts)
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
        vitamin_color = jnp.array([252, 144, 200, 255], dtype=jnp.uint8)
        vitamin_data = jnp.zeros((10, 10, 4), dtype=jnp.uint8)
        vitamin_data = vitamin_data.at[3:7, 1:9].set(vitamin_color)
        vitamin_data = vitamin_data.at[1:9, 3:7].set(vitamin_color)

        asset_config = [
            {'name': 'dummy_bg', 'type': 'background', 'data': jnp.zeros((210, 160, 4), dtype=jnp.uint8)},
            {'name': 'pacman', 'type': 'group', 'files': ['pacman_0.npy', 'pacman_1.npy', 'pacman_2.npy']},
            {'name': 'ghosts', 'type': 'group', 'files': [
                'ghost_0.npy', 'ghost_1.npy', 'ghost_2.npy', 'ghost_3.npy'
            ], 'recolorings': {
                'frightened': tuple(map(int, self.consts.PALE_BLUE_COLOR.tolist()))
            }},
            {'name': 'life', 'type': 'single', 'file': 'life.npy'},
            {'name': 'fruit', 'type': 'group', 'data': [vitamin_data]},
            {'name': 'digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        ]

        # Include background colors in the palette (Path, Wall, Black, Pink Power Pellet, Pale Blue)
        bg_colors = jnp.stack([
            self.consts.PATH_COLOR, 
            self.consts.WALL_COLOR, 
            jnp.array([0, 0, 0], dtype=jnp.uint8),
            self.consts.POWER_PELLET_COLOR,
            self.consts.PALE_BLUE_COLOR
        ])
        bg_colors = jnp.concatenate([bg_colors, jnp.full((5, 1), 255, dtype=jnp.uint8)], axis=1)
        asset_config.append({'name': 'bg_colors', 'type': 'procedural', 'data': bg_colors[:, None, :]})

        (self.PALETTE, self.SHAPE_MASKS, _, self.COLOR_TO_ID, self.FLIP_OFFSETS) = \
            self.jr.load_and_setup_assets(asset_config, sprite_path)

        # Pacman masks are just right looking: 0, 1, 2
        self.PACMAN_MASKS = self.SHAPE_MASKS['pacman']
        self.LIFE_MASK = self.SHAPE_MASKS['life']

        # Pre-calculate backgrounds
        self.MAZE_BACKGROUNDS = self._create_all_backgrounds()

    def _create_all_backgrounds(self):
        bgs = []
        for i in range(1): # Only one maze for Pacman
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
        # raster = self.render_pellets(raster, state.level.pellets, wall_id)

        # 2. Power Pellets
        pink_id = self.COLOR_TO_ID[tuple(map(int, self.consts.POWER_PELLET_COLOR.tolist()))]
        pale_blue_id = self.COLOR_TO_ID[tuple(map(int, self.consts.PALE_BLUE_COLOR.tolist()))]
        
        # Check if power pellet effect is active (any ghost frightened or blinking)
        power_pellet_active = jnp.any((state.ghosts.modes == GhostMode.FRIGHTENED) | (state.ghosts.modes == GhostMode.BLINKING))
        power_pellet_color_id = jax.lax.select(power_pellet_active, pale_blue_id, pink_id)
        
        raster = self.render_power_pellets(raster, state, power_pellet_color_id)
        
        # 3. Pacman
        is_left = state.player.last_horiz_dir == 2 # 2 is LEFT, 1 is RIGHT
        cycle = (state.step_count // 4) % 4
        frame = jnp.array([0, 1, 2, 1])[cycle]
        pacman_mask = self.PACMAN_MASKS[frame.astype(jnp.int32)]
        
        raster = self.jr.render_at(raster, state.player.position[0].astype(jnp.int32), state.player.position[1].astype(jnp.int32) - 1, pacman_mask, flip_horizontal=is_left)
        
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
        # Animation frame changes every 8 frames
        sprite_idx = (state.step_count // 8) % 4
        normal_mask = self.SHAPE_MASKS['ghosts'][sprite_idx]
        frightened_mask = self.SHAPE_MASKS['ghosts_frightened'][sprite_idx]

        def render_one(i, r):
            pos = state.ghosts.positions[i]
            mode = state.ghosts.modes[i]
            is_frightened = (mode == GhostMode.FRIGHTENED) | (mode == GhostMode.BLINKING)
            
            # Blinking logic (every 8 frames toggle color)
            # Actually just using frightened mask if they are frightened/blinking.
            # If we want blinking, we can toggle between normal and frightened mask based on step_count
            is_blinking_frame = is_frightened & (mode == GhostMode.BLINKING) & ((state.step_count // 8) % 2 == 0)
            use_normal = (~is_frightened) | is_blinking_frame
            mask = jax.lax.select(use_normal, normal_mask, frightened_mask)
            
            # Only draw if not enjailed or returning? No, usually enjailed just means eyes, but we only have 1 sprite.
            # Keep as original for now, just apply mask
            return self.jr.render_at(r, pos[0].astype(jnp.int32), pos[1].astype(jnp.int32) - 1, mask)

        return jax.lax.fori_loop(0, 4, render_one, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_power_pellets(self, raster, state, color_id):
        # 10x4 sprite (10 height, 4 width)
        sprite = jnp.full((10, 4), color_id, dtype=raster.dtype)
        
        def render_one(i, r):
            should_draw = state.level.power_pellets[i] & (((state.step_count & 0b1000) >> 3) == 1)
            x = (self.consts.POWER_PELLET_TILES[i][0] * 4 + 4).astype(jnp.int32)
            y = (self.consts.POWER_PELLET_TILES[i][1] * 4 + 4).astype(jnp.int32)
            return jax.lax.cond(should_draw, 
                                lambda r_in: self.jr.render_at(r_in, x, y, sprite),
                                lambda r_in: r_in,
                                r)
        
        return jax.lax.fori_loop(0, 4, render_one, raster)

    @partial(jax.jit, static_argnums=(0,))
    def render_ui(self, raster, state):
        # Score
        digits = self.jr.int_to_digits(state.score, max_digits=self.consts.MAX_SCORE_DIGITS)
        digit_count = get_digit_count(state.score).astype(jnp.int32)
        start_index = self.consts.MAX_SCORE_DIGITS - digit_count
        render_x = 60 + start_index * 8
        raster = self.jr.render_label_selective(raster, render_x, 190, digits, self.SHAPE_MASKS['digits'], start_index, digit_count, spacing=8, max_digits_to_render=self.consts.MAX_SCORE_DIGITS)
        
        # Lives
        raster = self.jr.render_indicator(raster, 12, 182, (state.lives - 1).astype(jnp.int32), self.LIFE_MASK, spacing=14, max_value=self.consts.MAX_LIVE_COUNT)
        
        # Fruit indicator
        fruit_mask = self.SHAPE_MASKS['fruit'][state.fruit.type.astype(jnp.int32)]
        raster = self.jr.render_at(raster, 128, 182, fruit_mask)
        
        return raster

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
            collected_pellets, power_pellets, ate_power_pellet,
            pellet_reward, level_id
        ) = self.player_step(state, action)

        (fruit_state, fruit_reward) = self.fruit_step(state, player_position, collected_pellets, step_key)

        (
            ghost_positions, ghost_actions, ghost_modes, ghost_timers,
            eaten_ghosts, new_lives, new_death_timer, ghosts_reward
        ) = self.ghosts_step(state, ate_power_pellet, step_key)

        reward = pellet_reward + fruit_reward + ghosts_reward
        new_score = state.score + reward
        score_changed = self.flag_score_change(state.score, new_score)
        
        # Pacman logic: extra life when clearing a maze
        new_lives = jax.lax.cond(
            level_id > state.level.id,
            lambda: jnp.minimum(new_lives + 1, self.consts.MAX_LIVE_COUNT).astype(jnp.int8),
            lambda: new_lives
        )
        
        new_state = jax.lax.cond(
            frozen,
            lambda: new_state._replace(key=key),
            lambda: jax.lax.cond(
                level_id != state.level.id,
                lambda: reset_game(level_id, new_lives, new_score, key),
                lambda: PacmanState(
                    level = LevelState(
                        id=level_id,
                        collected_pellets=collected_pellets,
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
                        )
                    ),
                    ghosts = GhostsState(positions=ghost_positions, types=state.ghosts.types, actions=ghost_actions, modes=ghost_modes, timers=ghost_timers),
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
        available = available_directions(state.player.position, state.level.dofmaze)
        new_action = jax.lax.cond((action != Action.NOOP) & (action != Action.FIRE) & available[act_to_dir(action)], lambda: action, lambda: state.player.action)
        new_pos = jax.lax.cond(stop_wall(state.player.position, state.level.dofmaze)[act_to_dir(state.player.action)], lambda: state.player.position, lambda: get_new_position(state.player.position, new_action)) 
        
        (pellets, has_pellet, collected_pellets, power_pellets, ate_power_pellet, reward, level_id) = JaxPacman.pellet_step(state, new_pos)
        return (new_pos, new_action, pellets, has_pellet, collected_pellets, power_pellets, ate_power_pellet, reward, level_id)

    @staticmethod
    def pellet_step(state: PacmanState, new_pacman_pos: chex.Array):
        def check_power_pellet(idx: chex.Array, power_pellets: chex.Array):
            return jax.lax.cond(idx < 0, lambda: False, lambda: power_pellets[idx % 4])
        
        def eat_power_pellet(idx: chex.Array, power_pellets: chex.Array):
            return power_pellets.at[idx % 4].set(False)
        
        def check_pellet(pos: chex.Array):
            x_offset = jax.lax.cond(pos[0] < 75, lambda: 5, lambda: 1)
            return (pos[0] % 8 == x_offset) & (pos[1] % 12 == 6)

        def eat_pellet(pos: chex.Array, pellets: chex.Array):
            tile_x, tile_y = (pos[0] - 2) // 8, (pos[1] + 4) // 12
            in_bounds = (tile_x >= 0) & (tile_x < pellets.shape[0]) & (tile_y >= 0) & (tile_y < pellets.shape[1])
            return jax.lax.cond(pellets[tile_x, tile_y] & in_bounds, lambda: (pellets.at[tile_x, tile_y].set(False), True), lambda: (pellets, False))
        pellets, ate_pellet = jax.lax.cond(check_pellet(new_pacman_pos), lambda: eat_pellet(new_pacman_pos, state.level.pellets), lambda: (state.level.pellets, False))
        power_pellet_hit = jnp.where(jnp.all(jnp.round(new_pacman_pos / PacmanMaze.TILE_SCALE) == CONSTS.POWER_PELLET_HITBOXES, axis=1), size=1, fill_value=-1)[0][0]
        power_pellets, ate_power_pellet = jax.lax.cond(check_power_pellet(power_pellet_hit, state.level.power_pellets), lambda: (eat_power_pellet(power_pellet_hit, state.level.power_pellets), True), lambda: (state.level.power_pellets, False))
        
        reward = jax.lax.cond(ate_power_pellet, lambda: CONSTS.POWER_PELLET_POINTS, lambda: jax.lax.cond(ate_pellet, lambda: CONSTS.PELLET_POINTS, lambda: 0))
        has_pellet = ate_power_pellet | ate_pellet
        collected_pellets = jax.lax.cond(has_pellet, lambda: state.level.collected_pellets + 1, lambda: state.level.collected_pellets)
        
        level_id, reward = jax.lax.cond(collected_pellets >= CONSTS.PELLETS_TO_COLLECT[get_level_maze(state.level.id)], lambda: (state.level.id + 1, reward + CONSTS.LEVEL_COMPLETED_POINTS), lambda: (state.level.id, reward))
        return (pellets, has_pellet, collected_pellets, power_pellets, ate_power_pellet, reward, level_id)

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
                return (jnp.array(GhostMode.BLINKING, dtype=jnp.uint8), jnp.array(action, dtype=jnp.uint8), jnp.array(jnp.round(CONSTS.BLINKING_DURATION * timing_factor), dtype=jnp.float16), False)
            def start_returning(action, step_count):
                return (jnp.array(GhostMode.RETURNING, dtype=jnp.uint8), jnp.array(Action.UP, dtype=jnp.uint8), jnp.array(CONSTS.RETURN_DURATION, dtype=jnp.float16), True)
            def start_returned(action, step_count):
                return (jnp.array(GhostMode.CHASE, dtype=jnp.uint8), jnp.array(action, dtype=jnp.uint8), jnp.array(CONSTS.CHASE_DURATION, dtype=jnp.float16), False)

            return jax.lax.cond(
                ate_power_pellet & (mode != GhostMode.ENJAILED) & (mode != GhostMode.RETURNING),
                lambda: (jnp.array(GhostMode.FRIGHTENED, dtype=jnp.uint8), jnp.array(reverse_action(action), dtype=jnp.uint8), jnp.array(CONSTS.FRIGHTENED_DURATION * timing_factor, dtype=jnp.float16), True),
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
            
            chase_target = jax.lax.cond(new_mode == GhostMode.CHASE, lambda: get_chase_target(state.ghosts.types[ghost_index], state.ghosts.positions[ghost_index], state.ghosts.positions[0], state.player.position, state.player.action), lambda: CONSTS.SCATTER_TARGETS[ghost_index])
            new_action = jax.lax.cond(skip | (new_mode == GhostMode.ENJAILED) | (new_mode == GhostMode.RETURNING), lambda: new_action, lambda: jax.lax.cond(n_allowed == 0, lambda: new_action, lambda: jax.lax.cond(n_allowed == 1, lambda: allowed[0], lambda: jax.lax.cond((new_mode == GhostMode.FRIGHTENED) | (new_mode == GhostMode.BLINKING), lambda: allowed[jax.random.randint(ghost_keys[ghost_index], (), 0, n_allowed)], lambda: pathfind(state.ghosts.positions[ghost_index], new_action, chase_target, allowed, ghost_keys[ghost_index])))))
            
            slow_down = ((new_mode == GhostMode.FRIGHTENED) | (new_mode == GhostMode.BLINKING) | (new_mode == GhostMode.RETURNING)) & (state.step_count % 2 == 0)
            new_pos = jax.lax.cond(slow_down, lambda: state.ghosts.positions[ghost_index], lambda: get_new_position(state.ghosts.positions[ghost_index], new_action))
            
            return (new_ghost_states[0].at[ghost_index].set(new_mode), new_ghost_states[1].at[ghost_index].set(new_action), new_ghost_states[2].at[ghost_index].set(new_pos), new_ghost_states[3].at[ghost_index].set(new_timer))

        ghost_keys = jax.random.split(common_key, 4)
        (new_modes, new_actions, new_positions, new_timers) = jax.lax.fori_loop(0, 4, ghost_step, (jnp.zeros(4, dtype=jnp.uint8), jnp.zeros(4, dtype=jnp.uint8), jnp.zeros((4, 2), dtype=jnp.int32), jnp.zeros(4, dtype=jnp.float16)))
        
        (new_positions, new_actions, new_modes, new_timers, eaten_ghosts, new_lives, new_death_timer, reward) = JaxPacman.ghosts_collision(new_positions, new_actions, new_modes, new_timers, state.player.position, state.player.eaten_ghosts, ate_power_pellet, state.lives)
        return new_positions, new_actions, new_modes, new_timers, eaten_ghosts, new_lives, new_death_timer, reward

    @staticmethod
    def ghosts_collision(ghost_positions, ghost_actions, ghost_modes, ghost_timers, new_pacman_pos, eaten_ghosts, ate_power_pellet, lives):
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
            
            is_collision = detect_collision(ghost_states.pacman_position, ghost_states.ghost_positions[ghost_index])
            return jax.lax.cond(is_collision, lambda: jax.lax.cond((ghost_states.ghost_modes[ghost_index] == GhostMode.FRIGHTENED) | (ghost_states.ghost_modes[ghost_index] == GhostMode.BLINKING), handle_eaten, handle_death), lambda: ghost_states)

        new_eaten = jax.lax.cond(ate_power_pellet, lambda: jnp.array(0, dtype=jnp.uint8), lambda: jnp.array(eaten_ghosts, dtype=jnp.uint8))
        res = jax.lax.fori_loop(0, 4, handle_ghost_collision, GhostStates(new_pacman_pos, jnp.array(0, dtype=jnp.uint32), ghost_positions, ghost_actions, ghost_modes, ghost_timers, new_eaten, False))
        
        new_lives = (lives - jnp.where(res.deadly_collision, 1, 0)).astype(jnp.int8)
        new_death_timer = jnp.where(res.deadly_collision, CONSTS.RESET_TIMER, 0).astype(jnp.uint32)
        return res.ghost_positions, res.ghost_actions, res.ghost_modes, res.ghost_timers, res.eaten_ghosts, new_lives, new_death_timer, res.reward

    @staticmethod
    def fruit_step(state: PacmanState, new_pacman_pos: chex.Array, collected_pellets: chex.Array, key: chex.Array):
        def spawn_vitamin():
            return FruitState(position=CONSTS.VITAMIN_POSITION.astype(jnp.uint8), exit=CONSTS.VITAMIN_POSITION.astype(jnp.uint8), type=jnp.array(0, dtype=jnp.uint8), action=jnp.array(Action.NOOP, dtype=jnp.uint8), spawn=jnp.array(False, dtype=jnp.bool), spawned=jnp.array(True, dtype=jnp.bool), timer=jnp.array(CONSTS.VITAMIN_DURATION, dtype=jnp.uint16)), 0
        def consume_vitamin():
            return FruitState(jnp.zeros(2, dtype=jnp.uint8), jnp.zeros(2, dtype=jnp.uint8), jnp.array(0, dtype=jnp.uint8), jnp.array(Action.NOOP, dtype=jnp.uint8), state.fruit.spawn, jnp.array(False, dtype=jnp.bool), jnp.array(CONSTS.VITAMIN_DURATION, dtype=jnp.uint16)), CONSTS.VITAMIN_REWARD
        def step_vitamin():
            new_timer = jax.lax.cond(state.fruit.timer > 0, lambda: state.fruit.timer - 1, lambda: state.fruit.timer)
            return state.fruit._replace(timer=new_timer), 0
        def clear_vitamin():
            return state.fruit._replace(spawned=False), 0

        fruit_spawn = jnp.any(CONSTS.VITAMIN_SPAWN_THRESHOLDS == collected_pellets) & state.player.has_pellet
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

def get_chase_target(ghost: GhostType,
                     ghost_position: chex.Array, blinky_pos: chex.Array,
                     player_pos: chex.Array, player_dir: chex.Array) -> chex.Array:
    """
    Compute the chase-mode target for each ghost:
    0=Red (Blinky), 1=Pink (Pinky), 2=Blue (Inky), 3=Orange (Sue)
    """
    def get_blinky_target(_):
        return player_pos.astype(jnp.int32)
    
    def get_pinky_target(_):
        return (player_pos.astype(jnp.int32) + 4*PacmanMaze.TILE_SCALE * CONSTS.ACTIONS[player_dir]).astype(jnp.int32)
    
    def get_inky_target(_):
        two_ahead = player_pos.astype(jnp.int32) + 2*PacmanMaze.TILE_SCALE * CONSTS.ACTIONS[player_dir]
        vect = two_ahead - blinky_pos.astype(jnp.int32)
        return (blinky_pos.astype(jnp.int32) + 2 * vect).astype(jnp.int32)
    
    def get_sue_target(_):
        dist = jnp.linalg.norm(ghost_position.astype(jnp.float32) - player_pos.astype(jnp.float32))
        return jax.lax.cond(dist > 8*PacmanMaze.TILE_SCALE, lambda: player_pos.astype(jnp.int32), lambda: CONSTS.SCATTER_TARGETS[GhostType.SUE].astype(jnp.int32))
    
    return jax.lax.switch(
        ghost,
        (
            get_blinky_target,  # GhostType.BLINKY
            get_pinky_target,   # GhostType.PINKY
            get_inky_target,    # GhostType.INKY
            get_sue_target      # GhostType.SUE
        ),
        None
    )

def get_new_position(position: chex.Array, action: chex.Array):
    new_position = position + CONSTS.ACTIONS[action]
    return new_position.at[0].set(new_position[0] % 160).astype(position.dtype)

# -------- Reset functions --------
def reset_game(level: chex.Array, lives: chex.Array, score: chex.Array, key: chex.PRNGKey):
    return PacmanState(level=reset_level(level), player=reset_player(), ghosts=reset_ghosts(), fruit=reset_fruit(level, key), lives=jnp.array(lives, dtype=jnp.int8), score=jnp.array(score, dtype=jnp.uint32), score_changed=jnp.zeros(6, dtype=jnp.bool_), freeze_timer=jnp.array(0, dtype=jnp.uint32), step_count=jnp.array(0, dtype=jnp.uint32), key=key)

def reset_level(level: chex.Array):
    return LevelState(id=jnp.array(level, dtype=jnp.uint8), collected_pellets=jnp.array(0, dtype=jnp.uint8), dofmaze=PacmanMaze.precompute_dof(get_level_maze(level)), pellets=jnp.copy(PacmanMaze.BASE_PELLETS), power_pellets=jnp.ones(4, dtype=jnp.bool_), loaded=jnp.array(0, dtype=jnp.uint8))

def reset_player():
    return PlayerState(
        position=CONSTS.INITIAL_PACMAN_POSITION.astype(jnp.uint8), 
        action=jnp.array(Action.LEFT, dtype=jnp.int32), 
        has_pellet=jnp.array(False, dtype=jnp.bool_), 
        eaten_ghosts=jnp.array(0, dtype=jnp.uint8),
        last_horiz_dir=jnp.array(2, dtype=jnp.int32)
    )

def reset_ghosts():
    return GhostsState(positions=CONSTS.INITIAL_GHOSTS_POSITIONS.astype(jnp.int32), types=jnp.array([GhostType.BLINKY, GhostType.PINKY, GhostType.INKY, GhostType.SUE], dtype=jnp.uint8), actions=jnp.array([Action.LEFT, Action.NOOP, Action.NOOP, Action.NOOP], dtype=jnp.uint8), modes=jnp.array([GhostMode.RANDOM, GhostMode.ENJAILED, GhostMode.ENJAILED, GhostMode.ENJAILED], dtype=jnp.uint8), timers=jnp.array([CONSTS.SCATTER_DURATION, CONSTS.PINKY_RELEASE_TIME, CONSTS.INKY_RELEASE_TIME, CONSTS.SUE_RELEASE_TIME], dtype=jnp.float16))

def reset_fruit(level: chex.Array, key: chex.PRNGKey):
    return FruitState(position=jnp.zeros(2, dtype=jnp.uint8), exit=jnp.zeros(2, dtype=jnp.uint8), type=jnp.array(0, dtype=jnp.uint8), action=jnp.array(Action.NOOP, dtype=jnp.uint8), spawn=jnp.array(False, dtype=jnp.bool_), spawned=jnp.array(False, dtype=jnp.bool_), timer=jnp.array(CONSTS.VITAMIN_DURATION, dtype=jnp.uint16))

def reset_entities(state: PacmanState, key: chex.PRNGKey):
    return state._replace(player=reset_player(), ghosts=reset_ghosts(), fruit=reset_fruit(state.level.id, key), step_count=jnp.array(0, dtype=jnp.uint32))
