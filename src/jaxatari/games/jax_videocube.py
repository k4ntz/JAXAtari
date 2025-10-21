#! /usr/bin/python3
# -*- coding: utf-8 -*-
#
# JAX VideoCube
#
# Simulates the Atari VideoCube game
#
# Authors:
# - Xarion99
# - Keksmo
# - Embuer
# - Snocember
import os
from typing import NamedTuple, Tuple
from functools import partial

import chex
import jax
import jax.numpy as jnp

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils_legacy as aj
from jaxatari.spaces import Space, Discrete, Box, Dict


class VideoCubeConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    SELECTED_CUBE = 1
    """ The selected cube. 1 - 50 are the standard cubes and 51 a random generated cube """
    GAME_VARIATION = 0
    """ The selected game variation: 0 = normal game, 1 = all tiles are blacked out, 2 = only up and right are allowed """
    CUBES = [
        # Cube 1
        [2, 4, 1, 3, 5, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 3, 0, 5, 4, 2, 1, 3, 1, 2, 3, 0, 1, 4, 2, 2, 5, 4, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 3, 5, 4, 1, 2],
        # Cube 2
        [2, 4, 1, 3, 1, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 3, 0, 5, 4, 2, 1, 3, 2, 2, 3, 3, 1, 4, 4, 2, 5, 0, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 5, 5, 4, 1, 2],
        # Cube 3
        [2, 4, 1, 3, 2, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 3, 0, 5, 4, 2, 1, 3, 4, 2, 3, 5, 1, 4, 0, 2, 5, 3, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 1, 5, 4, 1, 2],
        # Cube 4
        [4, 4, 1, 3, 3, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 5, 0, 5, 4, 2, 2, 3, 5, 2, 3, 4, 1, 4, 1, 2, 5, 2, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 0, 5, 0, 1, 2],
        # Cube 5
        [4, 4, 1, 3, 5, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 5, 0, 5, 4, 2, 2, 3, 1, 2, 3, 0, 1, 4, 2, 2, 5, 4, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 3, 5, 0, 1, 2],
        # Cube 6
        [4, 4, 1, 3, 1, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 5, 0, 5, 4, 2, 2, 3, 2, 2, 3, 3, 1, 4, 4, 2, 5, 0, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 5, 5, 0, 1, 2],
        # Cube 7
        [4, 4, 1, 3, 2, 1, 0, 5, 4, 1, 4, 3, 0, 0, 1, 5, 0, 5, 4, 2, 2, 3, 4, 2, 3, 5, 1, 4, 0, 2, 5, 3, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 3, 5, 5, 0, 3, 2, 1, 5, 0, 1, 2],
        # Cube 8
        [4, 4, 1, 3, 3, 1, 3, 5, 4, 2, 4, 3, 0, 0, 1, 5, 0, 1, 0, 2, 2, 3, 5, 2, 3, 4, 1, 4, 1, 2, 5, 2, 0, 3, 5, 4, 1,
         4, 4, 2, 1, 0, 3, 3, 5, 5, 0, 5, 2, 0, 5, 0, 1, 2],
        # Cube 9
        [4, 4, 1, 3, 5, 1, 3, 5, 4, 2, 4, 3, 0, 0, 1, 5, 0, 1, 0, 2, 2, 3, 1, 2, 3, 0, 1, 4, 2, 2, 5, 4, 0, 3, 5, 4, 1,
         4, 4, 2, 1, 0, 3, 3, 5, 5, 0, 5, 2, 3, 5, 0, 1, 2],
        # Cube 10
        [2, 4, 1, 3, 3, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 0, 5, 4, 2, 1, 3, 5, 2, 3, 4, 1, 4, 1, 4, 5, 2, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 2, 0, 1, 4, 1, 2],
        # Cube 11
        [2, 4, 1, 3, 5, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 0, 5, 4, 2, 1, 3, 1, 2, 3, 0, 1, 4, 2, 4, 5, 4, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 2, 3, 1, 4, 1, 2],
        # Cube 12
        [2, 4, 1, 3, 1, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 0, 5, 5, 4, 0, 3, 2, 2, 3, 3, 1, 4, 4, 4, 0, 4, 1, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 4, 2, 5, 0, 3, 2, 5, 1, 4, 1, 2],
        # Cube 13
        [2, 4, 1, 3, 2, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 0, 5, 4, 2, 1, 3, 4, 2, 3, 5, 1, 4, 0, 4, 5, 3, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 2, 1, 1, 4, 1, 2],
        # Cube 14
        [4, 4, 1, 3, 3, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 0, 5, 4, 2, 2, 3, 5, 2, 3, 4, 1, 4, 1, 4, 5, 2, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 2, 0, 1, 0, 1, 2],
        # Cube 15
        [4, 4, 1, 3, 5, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 0, 5, 4, 2, 2, 3, 1, 2, 3, 0, 1, 4, 2, 4, 5, 4, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 2, 3, 1, 0, 1, 2],
        # Cube 16
        [4, 4, 1, 3, 1, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 0, 5, 5, 4, 0, 3, 2, 2, 3, 3, 1, 4, 4, 4, 0, 4, 1, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 5, 4, 2, 5, 0, 3, 2, 5, 1, 0, 1, 2],
        # Cube 17
        [4, 4, 1, 3, 2, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 0, 5, 2, 4, 5, 3, 4, 2, 3, 5, 1, 4, 0, 4, 1, 0, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 0, 4, 5, 5, 0, 3, 2, 1, 1, 0, 1, 2],
        # Cube 18
        [4, 4, 1, 3, 3, 2, 3, 5, 4, 2, 0, 3, 0, 3, 1, 5, 0, 1, 1, 4, 0, 3, 5, 2, 3, 4, 1, 4, 1, 4, 0, 1, 1, 3, 5, 4, 1,
         4, 4, 2, 1, 0, 5, 4, 2, 5, 0, 5, 2, 0, 1, 0, 1, 2],
        # Cube 19
        [4, 4, 1, 3, 5, 2, 3, 5, 4, 2, 0, 3, 0, 3, 1, 5, 0, 1, 0, 2, 2, 3, 1, 2, 3, 0, 1, 4, 2, 4, 5, 4, 0, 3, 5, 4, 1,
         4, 4, 2, 1, 0, 3, 5, 5, 5, 0, 5, 2, 3, 1, 0, 1, 2],
        # Cube 20
        [2, 0, 1, 3, 3, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 2, 1, 5, 5, 2, 3, 4, 2, 4, 1, 4, 1, 2, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 0, 1, 4, 1, 2],
        # Cube 21
        [2, 0, 1, 3, 5, 2, 0, 5, 4, 1, 0, 3, 3, 1, 1, 3, 3, 5, 4, 2, 1, 5, 1, 2, 1, 3, 2, 4, 2, 4, 1, 4, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 3, 1, 4, 1, 2],
        # Cube 22
        [2, 0, 1, 3, 1, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 5, 4, 0, 5, 2, 2, 3, 3, 2, 4, 4, 4, 3, 4, 1, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 4, 2, 5, 0, 3, 4, 5, 1, 4, 1, 2],
        # Cube 23
        [2, 0, 1, 3, 2, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 2, 1, 5, 4, 2, 3, 5, 2, 4, 0, 4, 1, 3, 0, 0, 5, 4, 5,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 1, 1, 4, 1, 2],
        # Cube 24
        [4, 0, 1, 3, 3, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 2, 2, 5, 5, 2, 3, 4, 2, 4, 1, 4, 1, 2, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 0, 1, 0, 1, 2],
        # Cube 25
        [4, 0, 1, 3, 5, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 2, 2, 5, 1, 2, 3, 0, 2, 4, 2, 4, 1, 4, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 3, 1, 0, 1, 2],
        # Cube 26
        [4, 0, 1, 3, 1, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 2, 2, 5, 2, 2, 3, 3, 2, 4, 4, 4, 1, 0, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 5, 1, 0, 1, 2],
        # Cube 27
        [4, 0, 1, 5, 2, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 2, 2, 5, 4, 2, 3, 5, 2, 4, 0, 4, 1, 3, 0, 3, 5, 4, 1,
         4, 2, 2, 1, 0, 3, 5, 5, 5, 0, 3, 4, 1, 1, 0, 1, 2],
        # Cube 28
        [4, 0, 1, 3, 3, 2, 3, 5, 4, 2, 0, 3, 0, 3, 1, 5, 3, 1, 0, 2, 2, 5, 5, 2, 3, 4, 2, 4, 1, 4, 1, 2, 0, 3, 5, 4, 1,
         4, 4, 2, 1, 0, 3, 5, 5, 5, 0, 5, 4, 0, 1, 0, 1, 2],
        # Cube 29
        [4, 0, 1, 3, 5, 2, 3, 5, 4, 2, 0, 3, 0, 3, 1, 5, 3, 1, 0, 2, 2, 5, 1, 2, 3, 0, 2, 4, 2, 4, 1, 4, 0, 3, 5, 4, 1,
         4, 4, 2, 1, 0, 3, 5, 5, 5, 0, 5, 4, 3, 1, 0, 1, 2],
        # Cube 30
        [2, 0, 1, 5, 3, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 5, 2, 3, 4, 2, 4, 1, 4, 1, 2, 0, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 0, 1, 4, 1, 2],
        # Cube 31
        [2, 0, 1, 5, 5, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 1, 2, 3, 0, 2, 4, 2, 4, 1, 4, 0, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 3, 1, 4, 1, 2],
        # Cube 32
        [2, 0, 1, 5, 1, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 2, 2, 3, 3, 2, 4, 4, 4, 1, 0, 0, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 5, 1, 4, 1, 2],
        # Cube 33
        [2, 0, 1, 5, 2, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 4, 2, 3, 5, 2, 4, 0, 4, 1, 3, 0, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 1, 1, 4, 1, 2],
        # Cube 34
        [4, 0, 1, 5, 3, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 5, 2, 3, 4, 2, 4, 1, 4, 1, 2, 0, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 0, 1, 0, 1, 2],
        # Cube 35
        [4, 0, 1, 5, 5, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 1, 2, 3, 0, 2, 4, 2, 4, 1, 4, 0, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 3, 1, 0, 1, 2],
        # Cube 36
        [4, 0, 1, 5, 1, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 2, 2, 3, 3, 2, 4, 4, 4, 1, 0, 0, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 5, 1, 0, 1, 2],
        # Cube 37
        [5, 0, 1, 5, 2, 2, 0, 5, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 4, 2, 3, 5, 2, 4, 0, 4, 1, 3, 0, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 1, 1, 0, 1, 2],
        # Cube 38
        [4, 0, 1, 5, 3, 2, 3, 5, 4, 2, 0, 3, 0, 3, 1, 5, 3, 1, 0, 4, 2, 5, 5, 2, 3, 4, 2, 4, 1, 4, 1, 2, 0, 3, 1, 4, 1,
         0, 4, 2, 2, 0, 3, 5, 5, 5, 3, 5, 4, 0, 1, 0, 1, 2],
        # Cube 39
        [4, 0, 1, 5, 5, 2, 3, 5, 4, 2, 0, 3, 0, 3, 1, 5, 3, 1, 0, 4, 2, 5, 1, 2, 3, 0, 2, 4, 2, 4, 1, 4, 0, 3, 1, 4, 1,
         0, 4, 2, 2, 0, 3, 5, 5, 5, 3, 5, 4, 3, 1, 0, 1, 2],
        # Cube 40
        [2, 0, 1, 5, 3, 2, 0, 1, 4, 1, 0, 3, 5, 5, 0, 4, 1, 0, 1, 3, 5, 5, 5, 4, 0, 4, 3, 2, 3, 5, 4, 2, 5, 0, 1, 4, 2,
         2, 1, 1, 0, 2, 4, 1, 3, 5, 3, 3, 4, 0, 1, 4, 2, 2],
        # Cube 41
        [2, 0, 1, 5, 5, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 1, 4, 5, 0, 2, 0, 2, 4, 1, 4, 3, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 3, 1, 4, 2, 2],
        # Cube 42
        [2, 0, 1, 5, 1, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 2, 4, 5, 3, 2, 0, 4, 4, 1, 0, 3, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 5, 1, 4, 2, 2],
        # Cube 43
        [2, 0, 1, 5, 2, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 3, 3, 5, 4, 4, 1, 5, 4, 4, 5, 5, 2, 0, 0, 4, 1, 3, 3, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 1, 1, 4, 2, 2],
        # Cube 44
        [4, 0, 1, 5, 3, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 5, 4, 5, 4, 2, 0, 1, 4, 1, 2, 3, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 0, 1, 0, 2, 2],
        # Cube 45
        [4, 0, 1, 5, 5, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 1, 4, 5, 0, 2, 0, 2, 4, 1, 4, 3, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 3, 1, 0, 2, 2],
        # Cube 46
        [4, 0, 1, 5, 1, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 2, 4, 5, 3, 2, 0, 4, 4, 1, 0, 3, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 5, 1, 0, 2, 2],
        # Cube 47
        [4, 0, 1, 5, 2, 2, 0, 1, 4, 1, 0, 3, 0, 3, 1, 5, 3, 5, 4, 4, 2, 5, 4, 4, 5, 5, 2, 0, 0, 4, 1, 3, 3, 3, 1, 4, 1,
         0, 2, 2, 2, 0, 3, 5, 5, 5, 3, 3, 4, 1, 1, 0, 2, 2],
        # Cube 48
        [4, 0, 1, 5, 3, 2, 3, 1, 4, 2, 0, 3, 0, 3, 1, 5, 3, 1, 3, 4, 2, 5, 5, 4, 5, 4, 2, 0, 1, 4, 1, 2, 3, 3, 1, 4, 1,
         0, 4, 2, 2, 0, 3, 5, 5, 5, 3, 5, 4, 0, 1, 0, 2, 2],
        # Cube 49
        [4, 0, 1, 5, 5, 2, 3, 1, 4, 2, 0, 3, 0, 3, 1, 5, 3, 1, 0, 4, 2, 5, 1, 4, 5, 0, 2, 0, 2, 4, 1, 4, 3, 3, 1, 4, 1,
         0, 4, 2, 2, 0, 3, 5, 5, 5, 3, 5, 4, 3, 1, 0, 2, 2],
        # Cube 50
        [2, 0, 1, 5, 3, 4, 0, 1, 4, 1, 3, 3, 0, 5, 1, 3, 3, 5, 4, 4, 1, 5, 5, 4, 5, 4, 2, 0, 1, 0, 1, 2, 3, 0, 1, 4, 5,
         0, 2, 2, 2, 0, 3, 1, 5, 5, 3, 3, 4, 0, 2, 4, 2, 2],
        # Cube 51 (randomly generated) in reset!
    ]
    """ The color values of the 50 standard cubes """
    PLAYER_COLORS = [
        # Cube 1
        2,
        # Cube 2
        4,
        # Cube 3
        0,
        # Cube 4
        1,
        # Cube 5
        2,
        # Cube 6
        4,
        # Cube 7
        0,
        # Cube 8
        1,
        # Cube 9
        2,
        # Cube 10
        1,
        # Cube 11
        2,
        # Cube 12
        4,
        # Cube 13
        0,
        # Cube 14
        1,
        # Cube 15
        2,
        # Cube 16
        4,
        # Cube 17
        0,
        # Cube 18
        1,
        # Cube 19
        2,
        # Cube 20
        1,
        # Cube 21
        2,
        # Cube 22
        4,
        # Cube 23
        0,
        # Cube 24
        1,
        # Cube 25
        2,
        # Cube 26
        4,
        # Cube 27
        0,
        # Cube 28
        1,
        # Cube 29
        2,
        # Cube 30
        1,
        # Cube 31
        2,
        # Cube 32
        4,
        # Cube 33
        0,
        # Cube 34
        1,
        # Cube 35
        2,
        # Cube 36
        4,
        # Cube 37
        0,
        # Cube 38
        1,
        # Cube 39
        2,
        # Cube 40
        1,
        # Cube 41
        2,
        # Cube 42
        4,
        # Cube 43
        0,
        # Cube 44
        1,
        # Cube 45
        2,
        # Cube 46
        4,
        # Cube 47
        0,
        # Cube 48
        1,
        # Cube 49
        2,
        # Cube 50
        1,
        # Cube 51 (randomly generated) in reset!
    ]
    """ The color of the player, depending on the selected cube """
    INITIAL_PLAYER_POS = 49
    """ The initial player position
    Important: the initial player position and the initial current side must fit together
    """
    INITIAL_CURRENT_SIDE = 1
    """ The initial side the player is starting on 
    Important: the initial player position and the initial current side must fit together
    """
    INITIAL_ORIENTATION = 0
    """ The initial orientation of the player """
    INITIAL_PLAYER_SCORE = 0
    """ The initial score of the player """
    INITIAL_PLAYER_LOOKING_DIRECTION = jnp.array(Action.RIGHT)
    """ The initial looking direction of the player 
    Only up, down, right and left are possible
    """
    CUBE_SIDES = jnp.array([
        [24, 25, 26, 14, 2, 1, 0, 12, 13],
        [60, 61, 62, 50, 38, 37, 36, 48, 49],
        [63, 64, 65, 53, 41, 40, 39, 51, 52],
        [66, 67, 68, 56, 44, 43, 42, 54, 55],
        [69, 70, 71, 59, 47, 46, 45, 57, 58],
        [96, 97, 98, 86, 74, 73, 72, 84, 85]
    ])

class MovementState(NamedTuple):
    is_moving_on_one_side: chex.Numeric
    """ Tells if the player is moving on one side of the cube """
    is_moving_between_two_sides: chex.Numeric
    """ Tells if the player is moving between two sides of the cube """
    moving_counter: chex.Numeric
    """ The counter for the animation of the player if he is moving """


class VideoCubeState(NamedTuple):
    player_pos: chex.Numeric
    """ The global position of the player """
    player_color: chex.Numeric
    """ The color of the player: 0 = red, 1 = green, 2 = blue, 3 = orange, 4 = purple, 5 = white, 6 = undefined (or black when game_variation is 1) """
    cube_orientation: chex.Numeric
    """ The orientation of the cube: 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270° (clockwise) """
    cube_current_side: chex.Numeric
    """ The view side from 0 to 5
    5 u u u
    1 2 3 4
    0 u u u
    (u = undefined) 
    """
    cube: chex.Array
    """ The representation of the cube: an array not matrix
    96, 97, 98 | 99,100,101 | 102,103,104| 105,106,107
    84, 85, 86 | 87, 88, 89 | 90, 91, 92 | 93, 94, 95
    72, 73, 74 | 75, 76. 77 | 78, 79, 80 | 81, 82, 83
    -----------+------------+------------+-----------
    60, 61, 62 | 63, 64, 65 | 66, 67, 68 | 69, 70, 71
    48, 49, 50 | 51, 52, 53 | 54, 55, 56 | 57, 58, 59
    36, 37, 38 | 39, 40, 41 | 42, 43, 44 | 45, 46, 47
    -----------+------------+------------+-----------
    24, 25, 26 | 27, 28, 29 | 30, 31, 32 | 33, 34, 35
    12, 13, 14 | 15, 16, 17 | 18, 19, 20 | 21, 22, 23
    00, 01, 02 | 03, 04, 05 | 06, 07, 08 | 09, 10, 11 
    """
    step_counter: chex.Numeric
    """ The counter for the current step """
    player_score: chex.Numeric
    """ The score of the player """
    last_action: chex.Array
    """ The last selected action """
    can_move: chex.Numeric
    """ Tells if the selected action can be executed """
    movement_state: MovementState
    """ Manages the values for the movement animations of the player """
    last_player_pos: chex.Numeric
    """ The last position of the player """
    last_cube_orientation: chex.Numeric
    """ The last cube orientation """
    last_cube_current_side: chex.Numeric
    """ The last current side of the cube """
    skip_step: chex.Numeric
    """ Contains a target tick count. The game will not continue until the tick count is reached """


class VideoCubeObservation(NamedTuple):
    cube_current_view: jnp.ndarray
    """ The current view on the cube """
    player_score: jnp.ndarray
    """ The score of the player """
    player_color: jnp.ndarray
    """ The color of the player """
    player_x: jnp.ndarray
    """ The x coordinate of the player """
    player_y: jnp.ndarray
    """ The y coordinate of the player """


class VideoCubeInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: jnp.ndarray


@jax.jit
def get_player_position(cube_current_side, cube_orientation, player_pos, consts):
    """ Computes from the global player position the position of the player on one side of the cube

    :param cube_current_side: the number of the side the player is looking at
    :param cube_orientation: the rotation of the current side
    :param player_pos: the global player position
    """

    # Get the current side of the cube as an array
    current_side = consts.CUBE_SIDES[cube_current_side]
    # Shifts the array by cube_orientation
    rotated_array = jnp.roll(current_side[0:8], -cube_orientation * 2)
    # Organizes the output matrix correctly
    result = jnp.array([
        [rotated_array[0], rotated_array[1], rotated_array[2]],
        [rotated_array[7], current_side[8], rotated_array[3]],
        [rotated_array[6], rotated_array[5], rotated_array[4]],
    ])
    # Compute the position localy as x,y coordinate
    flat_index = jnp.argmax(result == player_pos)
    index = jnp.unravel_index(flat_index, result.shape)

    x_coordinate = index[0].astype(jnp.int32)
    y_coordinate = index[1].astype(jnp.int32)
    return y_coordinate, x_coordinate


@jax.jit
def get_view(cube, cube_current_side, cube_orientation, game_variation, is_moving_between_two_sides):
    """ Returns a 3x3 matrix which contains the color of the tiles the player is looking at

    :param cube: the representation of the cube
    :param cube_current_side: the number of the side the player is looking at
    :param cube_orientation: the rotation of the current side
    :param game_variation: the selected game variation
    :param is_moving_between_two_sides: Whether the player is moving between the two sides (needed for game_variation 1)
    """

    @jax.jit
    def return_true_val(params):
        return jnp.array([[6, 6, 6], [6, 6, 6], [6, 6, 6]]).astype(jnp.int32)

    @jax.jit
    def return_false_val(params):
        # Get the current side of the cube as an array
        cube = params[0]
        cube_current_side = params[1]
        cube_orientation = params[2]

        current_side = jnp.array([
            [cube[24], cube[25], cube[26], cube[14], cube[2], cube[1], cube[0], cube[12], cube[13]],
            [cube[60], cube[61], cube[62], cube[50], cube[38], cube[37], cube[36], cube[48], cube[49]],
            [cube[63], cube[64], cube[65], cube[53], cube[41], cube[40], cube[39], cube[51], cube[52]],
            [cube[66], cube[67], cube[68], cube[56], cube[44], cube[43], cube[42], cube[54], cube[55]],
            [cube[69], cube[70], cube[71], cube[59], cube[47], cube[46], cube[45], cube[57], cube[58]],
            [cube[96], cube[97], cube[98], cube[86], cube[74], cube[73], cube[72], cube[84], cube[85]]
        ])
        # Shifts the array by cube_orientation
        rotated_array = jnp.roll(current_side[cube_current_side][0:8], -cube_orientation * 2)
        # Organizes the output matrix correctly
        result = jnp.array([
            [rotated_array[0], rotated_array[1], rotated_array[2]],
            [rotated_array[7], current_side[cube_current_side][8], rotated_array[3]],
            [rotated_array[6], rotated_array[5], rotated_array[4]],
        ]).astype(jnp.int32)
        return result


    # When game variation 1 is selected, the current side of the cube is only shown when switching between sides
    return jax.lax.cond(
        pred=jnp.logical_and(game_variation == 1, is_moving_between_two_sides == 0),
        true_fun=return_true_val,
        false_fun=return_false_val,
        operand=(cube, cube_current_side, cube_orientation)
    )


@jax.jit
def movement_controller(state: VideoCubeState):
    """ Sets all needed values for the movement animation of the player

    :param state: the current state of the game
    """
    movement_state = state.movement_state
    # Calculates if only rendering should be executed
    render_only = jnp.where(jnp.logical_or(movement_state.is_moving_on_one_side, movement_state.is_moving_between_two_sides), 1, 0)

    # Calculates the new moving_counter
    moving_counter = jnp.where(jnp.logical_and(jnp.logical_or(movement_state.is_moving_on_one_side, movement_state.is_moving_between_two_sides), state.step_counter % 4 == 0),
                               (movement_state.moving_counter + 1) % 6,
                               movement_state.moving_counter)

    # The movement has finished if moving_counter equal 0
    condition = moving_counter == 0
    is_moving_on_one_side = jnp.where(condition, 0, movement_state.is_moving_on_one_side)
    is_moving_between_two_sides = jnp.where(condition, 0, movement_state.is_moving_between_two_sides)


    return MovementState(is_moving_on_one_side, is_moving_between_two_sides, moving_counter), render_only


class JaxVideoCube(JaxEnvironment[VideoCubeState, VideoCubeObservation, VideoCubeInfo, VideoCubeConstants]):
    def __init__(self, consts: VideoCubeConstants = None, reward_funcs: list[callable] = None):
        """ Initialisation of VideoCube Game

        :param consts: all constants needed for the game
        :param reward_funcs: list of functions used to compute rewards
        """

        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.renderer = VideoCubeRenderer()
        self.consts = consts or VideoCubeConstants()
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        self.obs_size = 5

    def reset(self, key=jax.random.PRNGKey(int.from_bytes(os.urandom(3), byteorder='big'))) -> Tuple[VideoCubeObservation, VideoCubeState]:
        if self.consts.SELECTED_CUBE == 51:
            cube = jax.random.permutation(key, jnp.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5]))
            player_color = jax.random.permutation(key, jnp.array([0, 1, 2, 3, 4, 5]))[0],
        else:
            cube = self.consts.CUBES[self.consts.SELECTED_CUBE - 1]
            player_color = self.consts.PLAYER_COLORS[self.consts.SELECTED_CUBE - 1]

        new_state = VideoCubeState(
            player_pos=self.consts.INITIAL_PLAYER_POS,
            player_color=player_color,
            cube_orientation=self.consts.INITIAL_ORIENTATION,
            cube_current_side=self.consts.INITIAL_CURRENT_SIDE,
            cube= jnp.array([
                    cube[0],  cube[1],  cube[2], 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    cube[3],  cube[4],  cube[5], 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    cube[6],  cube[7],  cube[8], 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    cube[9],  cube[10], cube[11], cube[12], cube[13], cube[14], cube[15], cube[16], cube[17], cube[18], cube[19], cube[20],
                    cube[21], cube[22], cube[23], cube[24], cube[25], cube[26], cube[27], cube[28], cube[29], cube[30], cube[31], cube[32],
                    cube[33], cube[34], cube[35], cube[36], cube[37], cube[38], cube[39], cube[40], cube[41], cube[42], cube[43], cube[44],
                    cube[45], cube[46], cube[47], 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    cube[48], cube[49], cube[50], 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    cube[51], cube[52], cube[53], 6, 6, 6, 6, 6, 6, 6, 6, 6
            ]),
            step_counter=0,
            player_score=self.consts.INITIAL_PLAYER_SCORE,
            last_action=self.consts.INITIAL_PLAYER_LOOKING_DIRECTION,
            can_move=1,
            movement_state=MovementState(
                is_moving_on_one_side=0,
                is_moving_between_two_sides=0,
                moving_counter=0
            ),
            last_player_pos=self.consts.INITIAL_PLAYER_POS,
            last_cube_orientation=self.consts.INITIAL_ORIENTATION,
            last_cube_current_side=self.consts.INITIAL_CURRENT_SIDE,
            skip_step=0
        )

        initial_obs = self._get_observation(new_state)

        return initial_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: VideoCubeState, action: chex.Array) -> Tuple[VideoCubeObservation, VideoCubeState, float, bool, VideoCubeInfo]:
        action = jnp.array([Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.RIGHT, Action.LEFT, Action.RIGHT,
                           Action.LEFT, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.RIGHT, Action.LEFT, Action.RIGHT, Action.LEFT])[action]

        # Set action to Noop if game_variation 2 (only up and right) is selected and something else was pressed
        condition = jnp.logical_or(self.consts.GAME_VARIATION != 2, jnp.logical_not(jnp.logical_or(jnp.logical_or(action == Action.NOOP, action == Action.FIRE), jnp.logical_or(action == Action.UP, action == Action.RIGHT))))
        action = jnp.where(condition, action, Action.NOOP)

        skip_step = jnp.where(jnp.logical_and(action != Action.NOOP, state.skip_step == state.step_counter), state.skip_step + 21, state.skip_step)

        movement_state, render_only = movement_controller(state)

        render_only = jnp.where(skip_step - state.step_counter <= 20, 1, render_only)

        skip_step = jnp.where(skip_step != state.step_counter, skip_step, state.skip_step + 1)

        # Updates the last selected action and ignores fire and noop
        last_action = jnp.where(jnp.logical_or(jnp.logical_or(action == Action.NOOP, action == Action.FIRE), render_only == 1), state.last_action, action)

        # check if color should be changed when Action.FIRE is pressed else do nothing if fire is pressed change player_color and cube but do not create a new state
        color_change_condition = jnp.logical_and(jnp.equal(action, Action.FIRE), render_only == 0)
        cube = jnp.where(color_change_condition, state.cube.at[state.player_pos].set(state.player_color), state.cube)
        player_color = jnp.where(color_change_condition, state.cube[state.player_pos], state.player_color)

        @jax.jit
        def move_player_false_val(params):
            return (state.player_pos, state.cube_current_side, state.cube_orientation, state.can_move, movement_state.is_moving_on_one_side, movement_state.is_moving_between_two_sides, movement_state.moving_counter)

        # Move player
        player_position, cube_current_side, cube_orientation, can_move, is_moving_on_one_side, is_moving_between_two_sides, movement_counter = jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(action != Action.NOOP, action != Action.FIRE), render_only == 0),
            true_fun= self.move,
            false_fun= move_player_false_val,
            operand=(state, action)
        )

        # Calculate player score
        player_score = jnp.where(jnp.logical_and(action != Action.NOOP, render_only == 0), state.player_score + 1, state.player_score)

        # Update last_player_position, last_cube_orientation, last_cube_current_side
        condition = movement_counter != 0
        last_player_position = jnp.where(condition, state.last_player_pos, player_position)
        last_cube_orientation = jnp.where(condition, state.last_cube_orientation, cube_orientation)
        last_cube_current_side = jnp.where(condition, state.last_cube_current_side, cube_current_side)

        new_state = VideoCubeState(
            player_pos=player_position,
            player_color=player_color,
            cube_orientation=cube_orientation,
            cube_current_side=cube_current_side,
            cube=cube,
            step_counter=state.step_counter + 1,
            player_score=player_score,
            last_action=last_action,
            can_move=can_move,
            movement_state=MovementState(
                moving_counter=movement_counter,
                is_moving_on_one_side=is_moving_on_one_side,
                is_moving_between_two_sides=is_moving_between_two_sides
            ),
            last_player_pos=last_player_position,
            last_cube_orientation=last_cube_orientation,
            last_cube_current_side=last_cube_current_side,
            skip_step=skip_step
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def move(self, params):
        """ Moves the player

        :params params: Is an array containing the current state and the current action
        """

        state = params[0]
        action = params[1]

        def get_step_value(rotation, relative_direction):
            absolute = (rotation + relative_direction) % 4
            dirs = jnp.array([(0, 1), (1, 0), (0, -1), (-1, 0)])
            dx, dy = dirs[absolute]
            return dy * 12 + dx * 1

        # move_direction: up = 0, right = 1, down = 2, left = 3
        move_direction = jnp.array([0, 1, 3, 2])[action - 2]
        new_step_value = get_step_value(state.cube_orientation, move_direction)
        new_pos = state.player_pos + new_step_value

        # noinspection PyTypeChecker
        player_position, cube_current_side, cube_orientation, can_move, is_moving_on_one_side, is_moving_between_two_sides, movement_counter = jnp.select(
            condlist=[
                jnp.logical_and(jnp.logical_and(self.get_cube_side_index(state.player_pos) == self.get_cube_side_index(new_pos), state.cube[new_pos] != 6), state.player_color == state.cube[state.player_pos + get_step_value(state.cube_orientation, move_direction)]),
                jnp.logical_and(self.get_cube_side_index(state.player_pos) == self.get_cube_side_index(new_pos), state.cube[new_pos] != 6),
                state.cube[self.move_side(state, new_step_value)[0]] == state.player_color,
                state.cube[self.move_side(state, new_step_value)[0]] != state.player_color
            ],
            choicelist=jnp.array([
                (state.player_pos, state.cube_current_side, state.cube_orientation, 0, 0, 0, 0),
                (new_pos, state.cube_current_side, state.cube_orientation, 1, 1, 0, 1),
                (state.player_pos, state.cube_current_side, state.cube_orientation, 0, 0, 0, 0),
                (*self.move_side(state, new_step_value), 1, 0, 1, 1)
            ]).astype(jnp.int32),
        )

        return player_position, cube_current_side, cube_orientation, can_move, is_moving_on_one_side, is_moving_between_two_sides, movement_counter

    @partial(jax.jit, static_argnums=(0,))
    def get_cube_side_index(self, n: int, width: int = 12, block_size: int = 3):
        """ Returns the side of the cube on the given field. Values that are outside the cube return unspecified values due to efficiency reasons! """
        row = n // width          # row index (counting from the bottom)
        col = n % width           # column index (counting from left)
        block_x = col // block_size
        block_y = row // block_size

        return jnp.where(block_y == 2, 5, block_x + block_y)

    @partial(jax.jit, static_argnums=(0,))
    def move_side(self, state: VideoCubeState, move_Value: chex.Numeric):
        """ Moves the player between two sides and returns the new player position and current side of the cube and the rotation of the cube

        :param state: the current game state
        :param move_Value: the movement of the index (or the player position) in the cube representation
        """
        cube_side_index_old = self.get_cube_side_index(state.player_pos)
        new_pos = state.player_pos + move_Value

        # noinspection PyTypeChecker
        return jnp.select(
            condlist=[
                # Check if movement happens without rotation change
                jnp.logical_and(jnp.logical_and(jnp.logical_and(new_pos > -1, new_pos < 108), state.cube[new_pos] != 6), jnp.logical_not(jnp.logical_or(jnp.logical_and(cube_side_index_old % 4 == 1, move_Value == -1), jnp.logical_and(cube_side_index_old == 4, move_Value == 1)))),
                jnp.logical_and(jnp.logical_and(state.cube_current_side > 0, state.cube_current_side < 5), jnp.logical_and(state.cube_current_side % 3 == 1, abs(move_Value) == 1)),
                jnp.logical_and(jnp.logical_and(state.cube_current_side > 0, state.cube_current_side < 5), new_pos >= 72),
                jnp.logical_and(jnp.logical_and(state.cube_current_side > 0, state.cube_current_side < 5), new_pos <  72),
                state.cube_current_side == 0,
                state.cube_current_side != 0,
            ],
            choicelist=jnp.array([
                # No rotation
                (new_pos, self.get_cube_side_index(new_pos), state.cube_orientation),
                (state.player_pos - move_Value * 11, state.cube_current_side - move_Value * 3, state.cube_orientation),
                jax.lax.switch(
                    index=state.cube_current_side - 2,
                    branches=[
                        lambda op: (74 + (12 * (op[0] - 63)), 5, (op[1] + 3) % 4),
                        lambda op: (98 - (op[0] - 66), 5, (op[1] + 2) % 4),
                        lambda op: (96 - (12 * (op[0] - 69)), 5, (op[1] + 1) % 4)
                    ],
                    operand=(state.player_pos, state.cube_orientation)
                ),
                jax.lax.switch(
                    index=state.cube_current_side - 2,
                    branches=[
                        lambda op: (26 - (12 * (op[0] - 39)), 0, (op[1] + 1) % 4),
                        lambda op: (2 - (op[0] - 42), 0, (op[1] + 2) % 4),
                        lambda op: (0 + (12 * (op[0] - 45)), 0, (op[1] + 3) % 4)
                    ],
                    operand=(state.player_pos, state.cube_orientation)
                ),
                jax.lax.switch(
                    index=jnp.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3])[move_Value + 12],
                    branches=[
                        lambda op: (44 - op[0], 3, (op[1] + 2) % 4),
                        lambda op: (45 + (op[0] / 12).astype("int32"), 4, (op[1] + 1) % 4),
                        lambda op: (41 - ((op[0] - 2) / 12).astype("int32"), 2, (op[1] + 3) % 4),
                        lambda _: (-20, 10, 9)  # unreachable case
                    ],
                    operand=(state.player_pos, state.cube_orientation)
                ),
                jax.lax.switch(
                    index=jnp.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3])[move_Value + 12],
                    branches=[
                        lambda _: (-20, 10, 9),  # unreachable case
                        lambda op: (71 - ((op[0] - 72) / 12).astype("int32"), 4, (op[1] + 3) % 4),
                        lambda op: (63 + ((op[0] - 74) / 12).astype("int32"), 2, (op[1] + 1) % 4),
                        lambda op: (66 + (98 - op[0]), 3, (op[1] + 2) % 4)
                    ],
                    operand=(state.player_pos, state.cube_orientation)
                ),
            ]).astype(jnp.int32)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: VideoCubeState, state: VideoCubeState):
        return previous_state.player_score - state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: VideoCubeState, state: VideoCubeState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state)
             for reward_func in self.reward_funcs]
        )
        return rewards

    def render(self, state: VideoCubeState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> Discrete:
        return Discrete(len(self.action_set))

    def image_space(self) -> Box:
        return Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> Dict:
        return Dict({
            "cube_current_view": Box(0, 6, (3, 3), jnp.int32),
            "player_score": Box(0, 100000, (), jnp.int32),
            "player_color": Box(0, 6, (), jnp.int32),
            "player_x": Box(0, 2, (), jnp.int32),
            "player_y": Box(0, 2, (), jnp.int32)
        })

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: VideoCubeObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.cube_current_view.flatten(),
            obs.player_score.flatten(),
            obs.player_color.flatten(),
            obs.player_x.flatten(),
            obs.player_y.flatten()
        ])

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoCubeState):
        return VideoCubeObservation(
            cube_current_view=get_view(state.cube, state.cube_current_side, state.cube_orientation, self.consts.GAME_VARIATION, state.movement_state.is_moving_between_two_sides),
            player_score=state.player_score.astype(jnp.int32),
            player_color=state.player_color.astype(jnp.int32),
            player_x=get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos, self.consts)[0],
            player_y=get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos, self.consts)[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoCubeState, all_rewards: jnp.ndarray = None) -> VideoCubeInfo:
        return VideoCubeInfo(state.step_counter, all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def is_side_solved(self, state, side_idx: chex.Numeric) -> bool:
        """Checks if a single side of the cube is solved."""
        view = get_view(state.cube, side_idx, 0, 0, state.movement_state.is_moving_between_two_sides)
        # A side is solved if all its tiles have the same color as the top-left tile.
        return jnp.all(view == view[0, 0])

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoCubeState) -> bool:
        """
        Determines if the game is done.
        The game is done if the player has solved all sides of the cube or has a player score of 100000.
        """
        # Vectorize the check for a single side being solved.
        # We check for each of the 6 sides.
        all_sides_solved = jnp.all(jax.vmap(self.is_side_solved, in_axes=(None, 0))(state, jnp.arange(6)))
        return jnp.logical_or(all_sides_solved, state.player_score >= 100000)


def load_sprites():
    """ Loads all sprites required for Blackjack rendering """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/background.npy"), transpose=False)
    background_switch_sides_vertically = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/background_switch_sides_vertically.npy"), transpose=False)
    tile_blue = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_blue.npy"), transpose=False)
    tile_green = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_green.npy"), transpose=False)
    tile_orange = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_orange.npy"), transpose=False)
    tile_purple = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_purple.npy"), transpose=False)
    tile_red = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_red.npy"), transpose=False)
    tile_white = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_white.npy"), transpose=False)
    tile_black = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/videocube/tile/tile_black.npy"), transpose=False)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITE_BACKGROUND = aj.get_sprite_frame(jnp.expand_dims(background, axis=0), 0)
    SPRITE_BACKGROUND_SWITCH_SIDES_VERTICALLY = aj.get_sprite_frame(jnp.expand_dims(background_switch_sides_vertically, axis=0), 0)
    SPRITE_TILE_BLUE = aj.get_sprite_frame(jnp.expand_dims(tile_blue, axis=0), 0)
    SPRITE_TILE_GREEN = aj.get_sprite_frame(jnp.expand_dims(tile_green, axis=0), 0)
    SPRITE_TILE_ORANGE = aj.get_sprite_frame(jnp.expand_dims(tile_orange, axis=0), 0)
    SPRITE_TILE_PURPLE = aj.get_sprite_frame(jnp.expand_dims(tile_purple, axis=0), 0)
    SPRITE_TILE_RED = aj.get_sprite_frame(jnp.expand_dims(tile_red, axis=0), 0)
    SPRITE_TILE_WHITE = aj.get_sprite_frame(jnp.expand_dims(tile_white, axis=0), 0)
    SPRITE_TILE_BLACK = aj.get_sprite_frame(jnp.expand_dims(tile_black, axis=0), 0)

    # Load digits for cube selection
    CUBE_DIGIT_SPRITES = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/videocube/cube_digit/cube_digit_{}.npy"), num_chars=10)

    # Load all Atari labels
    atari_labels = []
    for i in range(1, 65):
        path = "".join(["sprites/videocube/label/label_", str(i), ".npy"])
        frame = aj.loadFrame(os.path.join(MODULE_DIR, path), transpose=False)
        sprite = jnp.expand_dims(frame, axis=0)
        atari_labels.append(aj.get_sprite_frame(sprite, 0))
    LABEL_SPRITES = jnp.array(atari_labels)

    @partial(jax.jit, static_argnames=["color_index"])
    def load_score_digits(color_index):
        """ Modified version of atraJaxis load_and_pad_digits
        :param color_index: Color of digit (1 - 64)
        """

        digits = []
        max_width, max_height = 0, 0

        # Load digits assuming loadFrame returns (H, W, C)
        for k in range(0, 10):
            # Load with transpose=True (default) assuming source is H, W, C
            path_from_digits = "".join(["sprites/videocube/score/score_", str(k), "/score_", str(k), "_", str(color_index), ".npy"])
            digit = aj.loadFrame(os.path.join(MODULE_DIR, path_from_digits), transpose=False)
            max_width = max(max_width, digit.shape[1])  # Axis 1 is Width
            max_height = max(max_height, digit.shape[0])  # Axis 0 is Height
            digits.append(digit)

        # Pad digits to max dimensions (H, W)
        padded_digits = []
        for digit in digits:
            pad_w = max_width - digit.shape[1]  # Pad width (axis 1)
            pad_h = max_height - digit.shape[0]  # Pad height (axis 0)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            # Padding order for HWC: ((pad_H_before, after), (pad_W_before, after), ...)
            padded_digit = jnp.pad(
                digit,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_digits.append(padded_digit)

        return jnp.array(padded_digits)

    # Load all score digits
    score_digits = []
    for i in range(1, 65):
        score_digits.append(load_score_digits(i))
    SCORE_DIGIT_SPRITES = jnp.array(score_digits)

    # Load all player animations
    orientations = ["down_up", "right", "left"]
    colors = ["red", "green", "blue", "orange", "purple", "white"]
    player_animations = []
    for orientation in orientations:
        animations_orientation = []
        for color in colors:
            animations_color = []
            for i in range(1, 7):
                path = "".join(["sprites/videocube/player_animation/", orientation, "/", color, "/", orientation, "_", color, "_", str(i), ".npy"])
                frame = aj.loadFrame(os.path.join(MODULE_DIR, path), transpose=False)
                sprite = jnp.expand_dims(frame, axis=0)
                animations_color.append(aj.get_sprite_frame(sprite, 0))
            animations_orientation.append(jnp.array(animations_color))
        player_animations.append(jnp.array(animations_orientation))
    PLAYER_ANIMATIONS_SPRITES = jnp.array(player_animations)

    # Load tiles with different width
    widths = [7, 8, 9, 15, 17, 18, 20, 21, 22, 23, 24]
    horizontal_animation = []
    for color in colors:
        tmp_array = []
        for width in widths:
            path = "".join(["sprites/videocube/switch_sides_animation/left_right/tile_", color, "_width_", str(width), ".npy"])
            frame = aj.loadFrame(os.path.join(MODULE_DIR, path), transpose=False)
            sprite = jnp.expand_dims(frame, axis=0)
            tmp_array.append(aj.get_sprite_frame(sprite, 0))
        horizontal_animation.append(jnp.array(tmp_array))
    HORIZONTAL_ANIMATIONS_SPRITES = jnp.array(horizontal_animation)

    # Load tiles with different heights
    heights = [15, 24, 30, 36, 41]
    vertical_animation = []
    for color in colors:
        tmp_array = []
        for height in heights:
            path = "".join(["sprites/videocube/switch_sides_animation/up_down/tile_", color, "_hight_", str(height), ".npy"])
            frame = aj.loadFrame(os.path.join(MODULE_DIR, path), transpose=False)
            sprite = jnp.expand_dims(frame, axis=0)
            tmp_array.append(aj.get_sprite_frame(sprite, 0))
        vertical_animation.append(jnp.array(tmp_array))
    VERTICAL_ANIMATIONS_SPRITES = jnp.array(vertical_animation)

    return (
        SPRITE_BACKGROUND,
        SPRITE_BACKGROUND_SWITCH_SIDES_VERTICALLY,
        SPRITE_TILE_BLUE,
        SPRITE_TILE_GREEN,
        SPRITE_TILE_ORANGE,
        SPRITE_TILE_PURPLE,
        SPRITE_TILE_RED,
        SPRITE_TILE_WHITE,
        SPRITE_TILE_BLACK,
        HORIZONTAL_ANIMATIONS_SPRITES,
        VERTICAL_ANIMATIONS_SPRITES,
        CUBE_DIGIT_SPRITES,
        LABEL_SPRITES,
        SCORE_DIGIT_SPRITES,
        PLAYER_ANIMATIONS_SPRITES
    )


class VideoCubeRenderer(JAXGameRenderer):
    def __init__(self, consts: VideoCubeConstants = None):
        super().__init__()
        self.consts = consts or VideoCubeConstants()
        (
            self.SPRITE_BACKGROUND,
            self.SPRITE_BACKGROUND_SWITCH_SIDES_VERTICALLY,
            self.SPRITE_TILE_BLUE,
            self.SPRITE_TILE_GREEN,
            self.SPRITE_TILE_ORANGE,
            self.SPRITE_TILE_PURPLE,
            self.SPRITE_TILE_RED,
            self.SPRITE_TILE_WHITE,
            self.SPRITE_TILE_BLACK,
            self.HORIZONTAL_ANIMATIONS_SPRITES,
            self.VERTICAL_ANIMATIONS_SPRITES,
            self.CUBE_DIGIT_SPRITES,
            self.LABEL_SPRITES,
            self.SCORE_DIGIT_SPRITES,
            self.PLAYER_ANIMATIONS_SPRITES
        ) = load_sprites()

        # Position of every tile when rotating horizontally
        self.TILE_POSITIONS_HORIZONTAL_ROTATION = jnp.array([
            # Step 1
            [
                [[32, 28], [55, 28], [79, 28], [103, 28], [112, 28], [121, 28]],
                [[32, 78], [55, 78], [79, 78], [103, 78], [112, 78], [121, 78]],
                [[32, 128], [55, 128], [79, 128], [103, 128], [112, 128], [121, 128]]
            ],
            # Step 2
            [
                [[28, 28], [49, 28], [70, 28], [91, 28], [106, 28], [121, 28]],
                [[28, 78], [49, 78], [70, 78], [91, 78], [106, 78], [121, 78]],
                [[28, 128], [49, 128], [70, 128], [91, 128], [106, 128], [121, 128]]
            ],
            # Step 3
            [
                [[28, 28], [46, 28], [64, 28], [82, 28], [100, 28], [118, 28]],
                [[28, 78], [46, 78], [64, 78], [82, 78], [100, 78], [118, 78]],
                [[28, 128], [46, 128], [64, 128], [82, 128], [100, 128], [118, 128]]
            ],
            # Step 4
            [
                [[28, 28], [43, 28], [58, 28], [74, 28], [94, 28], [115, 28]],
                [[28, 78], [43, 78], [58, 78], [74, 78], [94, 78], [115, 78]],
                [[28, 128], [43, 128], [58, 128], [74, 128], [94, 128], [115, 128]]
            ],
            # Step 5
            [
                [[32, 28], [40, 28], [49, 28], [59, 28], [82, 28], [106, 28]],
                [[32, 78], [40, 78], [49, 78], [59, 78], [82, 78], [106, 78]],
                [[32, 128], [40, 128], [49, 128], [59, 128], [82, 128], [106, 128]],
            ]
        ])

        # Position of every tile when rotating vertically
        self.TILE_POSITIONS_VERTICAL_ROTATION = jnp.array([
            # Step 1
            [
                [[40, 14], [67, 14], [94, 12]],
                [[40, 31], [67, 31], [94, 32]],
                [[40, 48], [67, 48], [94, 48]],
                [[40, 66], [67, 66], [94, 66]],
                [[40, 109], [67, 109], [94, 109]],
                [[40, 152], [67, 152], [94, 152]]
            ],
            # Step 2
            [
                [[40, 8], [67, 8], [94, 8]],
                [[40, 34], [67, 34], [94, 34]],
                [[40, 60], [67, 60], [94, 60]],
                [[40, 87], [67, 87], [94, 87]],
                [[40, 125], [67, 125], [94, 125]],
                [[40, 163], [67, 163], [94, 163]]
            ],
            # Step 3
            [
                [[40, 8], [67, 8], [94, 8]],
                [[40, 40], [67, 40], [94, 40]],
                [[40, 72], [67, 72], [94, 72]],
                [[40, 105], [67, 105], [94, 105]],
                [[40, 137], [67, 137], [94, 137]],
                [[40, 169], [67, 169], [94, 169]]
            ],
            # Step 4
            [
                [[40, 8], [67, 8], [94, 8]],
                [[40, 46], [67, 46], [94, 46]],
                [[40, 84], [67, 84], [94, 84]],
                [[40, 123], [67, 123], [94, 123]],
                [[40, 149], [67, 149], [94, 149]],
                [[40, 175], [67, 175], [94, 175]]
            ],
            # Step 5
            [
                [[40, 14], [67, 14], [94, 14]],
                [[40, 57], [67, 57], [94, 57]],
                [[40, 100], [67, 100], [94, 100]],
                [[40, 144], [67, 144], [94, 144]],
                [[40, 161], [67, 161], [94, 161]],
                [[40, 178], [67, 178], [94, 178]]
            ]
        ])

        # Position of every tile when not rotating
        self.TILE_POSITIONS = jnp.array([
            [[40, 28], [67, 28], [94, 28]],
            [[40, 78], [67, 78], [94, 78]],
            [[40, 128], [67, 128], [94, 128]],
        ])

        # Widths of tiles when rotating horizontally
        self.WIDTHS = jnp.array([
            # Step 1
            [23, 24, 23, 9, 9, 7],
            # Step 2
            [21, 21, 20, 15, 15, 15],
            # Step 3
            [18, 18, 17, 18, 18, 18],
            # Step 4
            [15, 15, 15, 20, 21, 21],
            # Step 5
            [8, 9, 9, 23, 24, 22]
        ])

        # Heights of tiles when rotating vertically
        self.HEIGHTS = jnp.array([
            # Step 1
            [15, 15, 15, 41, 41, 41],
            # Step 2
            [24, 24, 24, 36, 36, 36],
            # Step 3
            [30, 30, 30, 30, 30, 30],
            # Step 4
            [36, 36, 36, 24, 24, 24],
            # Step 5
            [41, 41, 41, 15, 15, 15]
        ])

        # All possible positions of the player on one side of the cube
        self.PLAYER_POSITIONS = jnp.array([
            # Positions when player is looking up or down
            [
                [[46, 48], [73, 48], [100, 48]],
                [[46, 98], [73, 98], [100, 98]],
                [[46, 148], [73, 148], [100, 148]]
            ],
            # Positions when player is looking left or right
            [
                [[46, 44], [73, 44], [100, 44]],
                [[46, 94], [73, 94], [100, 94]],
                [[46, 144], [73, 144], [100, 144]]
            ]
        ])

        # The movement of the player on one side relative to the start position
        self.PLAYER_MOVEMENT_ON_ONE_SIDE = jnp.array([
            # Moving up
            [[0, -8], [0, -16], [0, -26], [0, -34], [0, -42], [0, -50]],
            # Moving right
            [[7, 1], [10, 0], [16, 1], [18, 0], [25, 1], [27, 0]],
            # Moving left
            [[-7, 1], [-10, 0], [-16, 1], [-18, 0], [-25, 1], [-27, 0]],
            # Moving down
            [[0, 8], [0, 16], [0, 26], [0, 34], [0, 42], [0, 50]]
        ])

        # The movement of the player between two sides relative to the start position
        self.PLAYER_MOVEMENT_BETWEEN_TWO_SIDES = jnp.array([
            # Moving up
            [[0, 6], [0, 16], [0, 32], [0, 48], [0, 70], [0, 100]],
            # Moving right
            [[-7, 0], [-17, 0], [-25, 0], [-35, 0], [-42, 0], [-54, 0]],
            # Moving left
            [[12, 0], [19, 0], [29, 0], [37, 0], [47, 0], [54, 0]],
            # Moving down
            [[0, -30], [0, -52], [0, -68], [0, -84], [0, -94], [0, -100]]
        ])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: VideoCubeState):
        """ Responsible for the graphical representation of the game

        :param state: the current game state
        """
        # Create empty raster with CORRECT orientation for atraJaxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        # Render background - (0, 0) is top-left corner
        raster = aj.render_at(raster, 0, 0, jnp.where(jnp.logical_and(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN),
                              state.movement_state.is_moving_between_two_sides == 1), self.SPRITE_BACKGROUND_SWITCH_SIDES_VERTICALLY, self.SPRITE_BACKGROUND))

        # Render Atari label
        # 1. Calculate the index for the label according to step_counter (the label color changes every 8 ticks)
        label_index = jnp.floor(state.step_counter / 8).astype("int32") % 64
        raster = jnp.where(jnp.logical_and(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN),
                           state.movement_state.is_moving_between_two_sides == 1), raster, aj.render_at(raster, 55, 5, self.LABEL_SPRITES[label_index]))

        # Render number of selected cube
        # 1. Get digit array (always 2 digits)
        selected_cube_digits = aj.int_to_digits(self.consts.SELECTED_CUBE, max_digits=2)

        # 2. Determine parameters for selected cube rendering
        is_selected_cube_single_digit = self.consts.SELECTED_CUBE < 10
        selected_cube_start_index = jax.lax.select(is_selected_cube_single_digit, 1, 0)  # Start at index 1 if single, 0 if double
        selected_cube_num_to_render = jax.lax.select(is_selected_cube_single_digit, 1, 2)  # Render 1 digit if single, 2 if double

        # 3. Render selected cube number using the selective renderer
        raster = aj.render_label_selective(raster, 96, 191, selected_cube_digits, self.CUBE_DIGIT_SPRITES,
                                           selected_cube_start_index, selected_cube_num_to_render, spacing=5)

        # Render player_score
        # 1. Get digit array (always 3 digits)
        player_score_digits = aj.int_to_digits(state.player_score, max_digits=6)
        # 2. Determine parameters for player score rendering
        player_score_conditions = jnp.array([
            state.player_score < 10,
            jnp.logical_and(state.player_score >= 10, state.player_score < 100),
            jnp.logical_and(state.player_score >= 100, state.player_score < 1000),
            jnp.logical_and(state.player_score >= 1000, state.player_score < 10000),
            jnp.logical_and(state.player_score >= 10000, state.player_score < 100000),
            state.player_score >= 100000
        ], dtype=bool)
        # Start at index 3 if single, 2 if double, 1 if triple, 0 if quadrupel
        player_score_start_index = jnp.select(player_score_conditions, jnp.array([5, 4, 3, 2, 1, 0]))
        # Render 1 digit if single, 2 if double, 3 if triple, 4 if quadrupel
        player_score_num_to_render = jnp.select(player_score_conditions, jnp.array([1, 2, 3, 4, 5, 6]))

        # 3. Render player score using the selective renderer
        raster = aj.render_label_selective(raster, 95, 180, player_score_digits,
                                           self.SCORE_DIGIT_SPRITES[label_index], player_score_start_index, player_score_num_to_render, spacing=10)

        @jax.jit
        def get_index(value, array):
            """ Calculate the index of the value in the given array

            :param value: the given value of the index to find
            :param array: the given array
            """

            index = jax.lax.fori_loop(
                lower=0,
                upper=array.size,
                body_fun=lambda i, val: jnp.where(array[i] == value, i, val),
                init_val=0
            )
            return index

        last_view = get_view(state.cube, state.last_cube_current_side, state.last_cube_orientation, 0, state.movement_state.is_moving_between_two_sides)

        @jax.jit
        def tiles_move_on_one_side():
            """ Returns the raster containing the tiles in the correct colors """
            tiles = jnp.array([
                self.SPRITE_TILE_RED,
                self.SPRITE_TILE_GREEN,
                self.SPRITE_TILE_BLUE,
                self.SPRITE_TILE_ORANGE,
                self.SPRITE_TILE_PURPLE,
                self.SPRITE_TILE_WHITE
            ])

            result_raster = raster
            result_raster = jax.lax.fori_loop(
                lower=0,
                upper=3,
                body_fun=lambda i, val1: jax.lax.fori_loop(
                    lower=0,
                    upper=3,
                    body_fun=lambda j, val2: aj.render_at(val2, self.TILE_POSITIONS[i][j][0], self.TILE_POSITIONS[i][j][1], jnp.where(self.consts.GAME_VARIATION == 1, self.SPRITE_TILE_BLACK, tiles[view[i][j]])),
                    init_val=val1
                ),
                init_val=result_raster
            )
            return result_raster

        @jax.jit
        def tiles_move_between_two_sides_vertically():
            """ Returns the raster containing the vertical rotation of the cube """
            result_raster = raster
            heights = jnp.array([15, 24, 30, 36, 41])
            # Array containing last_view and the current view
            combined_view = jax.lax.cond(
                pred=state.last_action == Action.UP,
                true_fun=lambda: jnp.concat((view, last_view), axis=0),
                false_fun=lambda: jnp.concat((last_view, view), axis=0),
            )
            # Tells which step of the animation is needed
            counter = jax.lax.cond(
                pred=state.last_action == Action.UP,
                true_fun=lambda: state.movement_state.moving_counter - 1,
                false_fun=lambda: 6 - state.movement_state.moving_counter - 1
            )
            # Render the tiles
            result_raster = jax.lax.fori_loop(
                lower=0,
                upper=6,
                body_fun=lambda i, val1: jax.lax.fori_loop(
                    lower=0,
                    upper=3,
                    body_fun=lambda j, val2: aj.render_at(
                        val2,
                        self.TILE_POSITIONS_VERTICAL_ROTATION[counter][i][j][0],
                        self.TILE_POSITIONS_VERTICAL_ROTATION[counter][i][j][1],
                        self.VERTICAL_ANIMATIONS_SPRITES[combined_view[i][j]][get_index(self.HEIGHTS[counter][i], heights)]),
                    init_val=val1
                ),
                init_val=result_raster
            )
            return result_raster

        @jax.jit
        def tiles_move_between_two_sides_horizontally():
            """ Returns the raster containing the horizontal rotation of the cube """
            result_raster = raster
            widths = jnp.array([7, 8, 9, 15, 17, 18, 20, 21, 22, 23, 24])
            # Array containing last_view and the current view
            combined_view = jax.lax.cond(
                pred=state.last_action == Action.RIGHT,
                true_fun=lambda: jnp.hstack((last_view, view)),
                false_fun=lambda: jnp.hstack((view, last_view)),
            )
            # Tells which step of the animation is needed
            counter = jax.lax.cond(
                pred=state.last_action == Action.RIGHT,
                true_fun=lambda: state.movement_state.moving_counter - 1,
                false_fun=lambda: 6 - state.movement_state.moving_counter - 1
            )
            # Render the tiles
            result_raster = jax.lax.fori_loop(
                lower=0,
                upper=3,
                body_fun=lambda i, val1: jax.lax.fori_loop(
                    lower=0,
                    upper=6,
                    body_fun=lambda j, val2: aj.render_at(
                        val2,
                        self.TILE_POSITIONS_HORIZONTAL_ROTATION[counter][i][j][0],
                        self.TILE_POSITIONS_HORIZONTAL_ROTATION[counter][i][j][1],
                        self.HORIZONTAL_ANIMATIONS_SPRITES[combined_view[i][j]][get_index(self.WIDTHS[counter][j], widths)]
                    ),
                    init_val=val1
                ),
                init_val=result_raster
            )
            return result_raster

        # Render the tiles of the cube
        # 1. Get the current cube side in consideration of the rotation
        view = get_view(state.cube, state.cube_current_side, state.cube_orientation, 0, state.movement_state.is_moving_between_two_sides)
        # 2. Differentiate between moving on one side and between two sides
        raster = jax.lax.cond(
            pred=jnp.logical_or(state.movement_state.is_moving_on_one_side, state.movement_state.moving_counter == 0),
            # Move on one side
            true_fun=lambda: tiles_move_on_one_side(),
            # Move between two sides
            false_fun=lambda: jax.lax.cond(
                pred=jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN),
                # Move vertically
                true_fun=lambda: tiles_move_between_two_sides_vertically(),
                # Move horizontally
                false_fun=lambda: tiles_move_between_two_sides_horizontally()
            ),
        )
        # Render player
        player_position = get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos, self.consts)
        last_player_position = get_player_position(state.last_cube_current_side, state.last_cube_orientation, state.last_player_pos, self.consts)
        sprite_on_one_side_indices = jnp.array([0, 1, 2, 0])
        # 1. Check if player can move
        raster = jax.lax.cond(
            pred=jnp.logical_and(state.can_move == 1, state.movement_state.moving_counter != 0),
            true_fun=lambda: jax.lax.cond(
                # Check if player is moving
                pred=jnp.logical_or(state.movement_state.is_moving_on_one_side, state.movement_state.is_moving_between_two_sides),
                true_fun=lambda: jax.lax.cond(
                    # Check if player is moving on one side ore between two sides
                    pred=state.movement_state.is_moving_on_one_side,
                    true_fun=lambda: aj.render_at(raster,
                                          self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1)][last_player_position[1]][last_player_position[0]][0] + self.PLAYER_MOVEMENT_ON_ONE_SIDE[state.last_action - 2][state.movement_state.moving_counter][0],
                                          self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1)][last_player_position[1]][last_player_position[0]][1] + self.PLAYER_MOVEMENT_ON_ONE_SIDE[state.last_action - 2][state.movement_state.moving_counter][1],
                                          self.PLAYER_ANIMATIONS_SPRITES[sprite_on_one_side_indices[state.last_action - 2]][state.player_color][state.movement_state.moving_counter]
                                          ),
                    false_fun=lambda: aj.render_at(raster,
                                            self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1)][last_player_position[1]][last_player_position[0]][0] + self.PLAYER_MOVEMENT_BETWEEN_TWO_SIDES[state.last_action - 2][state.movement_state.moving_counter][0],
                                            self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1)][last_player_position[1]][last_player_position[0]][1] + self.PLAYER_MOVEMENT_BETWEEN_TWO_SIDES[state.last_action - 2][state.movement_state.moving_counter][1],
                                            self.PLAYER_ANIMATIONS_SPRITES[sprite_on_one_side_indices[state.last_action - 2]][state.player_color][state.movement_state.moving_counter]
                                           )
                ),
                false_fun=lambda: raster
            ),
            false_fun=lambda: aj.render_at(raster,
                                           self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1)][player_position[1]][player_position[0]][0],
                                           self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1)][player_position[1]][player_position[0]][1],
                                           self.PLAYER_ANIMATIONS_SPRITES[sprite_on_one_side_indices[state.last_action - 2]][state.player_color][5]),
        )

        return raster