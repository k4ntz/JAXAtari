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
from jaxatari.rendering import jax_rendering_utils as aj
from jaxatari.spaces import Space, Discrete, Box, Dict


class VideoCubeConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
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
    INITIAL_PLAYER_POS = 49 # Important: the initial player position and the initial current side must fit together (look at comment of cube in VideoCubeState)
    INITIAL_CURRENT_SIDE = 1 # Important: the initial player position and the initial current side must fit together (look at comment of cube in VideoCubeState)
    INITIAL_ORIENTATION = 0
    INITIAL_PLAYER_SCORE = 0
    INITIAL_PLAYER_LOOKING_DIRECTION = jnp.array(Action.RIGHT) # Only up, down, right and left possible
    CUBE_SIDES = jnp.array([
        [24, 25, 26, 14, 2, 1, 0, 12, 13],
        [60, 61, 62, 50, 38, 37, 36, 48, 49],
        [63, 64, 65, 53, 41, 40, 39, 51, 52],
        [66, 67, 68, 56, 44, 43, 42, 54, 55],
        [69, 70, 71, 59, 47, 46, 45, 57, 58],
        [96, 97, 98, 86, 74, 73, 72, 84, 85]
    ])

class MovementState(NamedTuple):
    # Tells if the player is moving on one side of the cube
    is_moving_on_one_side: chex.Numeric
    # Tells if the player is moving between two sides of the cube
    is_moving_between_two_sides: chex.Numeric
    # The counter for the animation of the player if he is moving
    moving_counter: chex.Numeric


class VideoCubeState(NamedTuple):
    # The global position of the player
    player_pos: chex.Numeric
    # The color of the player: 0 = red, 1 = green, 2 = blue, 3 = orange, 4 = purple, 5 = white, 6 = undefined (or black when game_variation is 1)
    player_color: chex.Numeric
    # The orientation of the cube: 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270° (clockwise)
    cube_orientation: chex.Numeric
    # The view side from 0 to 5
    # 5 u u u
    # 1 2 3 4
    # 0 u u u
    # (u = undefined)
    cube_current_side: chex.Numeric
    # The representation of the cube: an array not matrix
    # 96, 97, 98 | 99,100,101 | 102,103,104| 105,106,107
    # 84, 85, 86 | 87, 88, 89 | 90, 91, 92 | 93, 94, 95
    # 72, 73, 74 | 75, 76. 77 | 78, 79, 80 | 81, 82, 83
    # -----------+------------+------------+-----------
    # 60, 61, 62 | 63, 64, 65 | 66, 67, 68 | 69, 70, 71
    # 48, 49, 50 | 51, 52, 53 |	54, 55, 56 | 57, 58, 59
    # 36, 37, 38 | 39, 40, 41 |	42, 43, 44 | 45, 46, 47
    # -----------+------------+------------+-----------
    # 24, 25, 26 | 27, 28, 29 | 30, 31, 32 | 33, 34, 35
    # 12, 13, 14 | 15, 16, 17 | 18, 19, 20 | 21, 22, 23
    # 00, 01, 02 | 03, 04, 05 | 06, 07, 08 | 09, 10, 11
    cube: chex.Array
    # The counter for the current step
    step_counter: chex.Numeric
    # The score of the player
    player_score: chex.Numeric
    # The last selected action
    last_action: chex.Array
    # The selected cube: 1 - 50 are the cubes from the game, 51 is randomly generated
    selected_cube: chex.Numeric
    # Tells if the selected action can be executed
    can_move: chex.Numeric
    # Manages the values for the movement animations of the player
    movement_state: MovementState
    # The last position of the player
    last_player_pos: chex.Numeric
    # The last cube orientation
    last_cube_orientation: chex.Numeric
    # The last current side of the cube
    last_cube_current_side: chex.Numeric
    # The selected game variation (0: normal game, 1: all tiles are blacked out, 2: only up and right are allowed)
    game_variation: chex.Numeric


class VideoCubeObservation(NamedTuple):
    # The current view on the cube
    cube_current_view: chex.Array
    # The score of the player
    player_score: chex.Numeric
    # The color of the player
    player_color: chex.Numeric
    # The x coordinate of the player
    player_x: chex.Numeric
    # The y coordinate of the player
    player_y: chex.Numeric


class VideoCubeInfo(NamedTuple):
    # The score of the player
    player_score: chex.Numeric
    # The counter for the current step
    step_counter: chex.Numeric


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

    x_coordinate = index[0]
    y_coordinate = index[1]
    return y_coordinate, x_coordinate


@jax.jit
def change_color(player_pos, player_color, cube):
    """ Changes the color of the tile the player is on and the color of the player to the color of the tile
    :param player_pos: the global player position
    :param player_color: the color of the player
    :param cube: the representation of the cube
    """
    new_player_color = cube[player_pos]
    # Change the color of the tile on the cube
    new_cube = cube.at[player_pos].set(player_color)

    return new_cube, new_player_color


@jax.jit
def get_view(cube, cube_current_side, cube_orientation, game_variation, is_moving_between_two_sides):
    """ Returns a 3x3 matrix which contains the color of the tiles the player is looking at

    :param cube: the representation of the cube
    :param cube_current_side: the number of the side the player is looking at
    :param cube_orientation: the rotation of the current side
    :param game_variation: the selected game variation
    :param is_moving_between_two_sides: Wether the player is moving between the two sides (needed for game_variation 1)
    """
    # Get the current side of the cube as an array
    current_side = jnp.array(jax.lax.switch(
        index=cube_current_side,
        branches=[
            lambda: [cube[24], cube[25], cube[26], cube[14], cube[2], cube[1], cube[0], cube[12], cube[13]],
            lambda: [cube[60], cube[61], cube[62], cube[50], cube[38], cube[37], cube[36], cube[48], cube[49]],
            lambda: [cube[63], cube[64], cube[65], cube[53], cube[41], cube[40], cube[39], cube[51], cube[52]],
            lambda: [cube[66], cube[67], cube[68], cube[56], cube[44], cube[43], cube[42], cube[54], cube[55]],
            lambda: [cube[69], cube[70], cube[71], cube[59], cube[47], cube[46], cube[45], cube[57], cube[58]],
            lambda: [cube[96], cube[97], cube[98], cube[86], cube[74], cube[73], cube[72], cube[84], cube[85]]
        ],
    ))
    # Shifts the array by cube_orientation
    rotated_array = jnp.roll(current_side[0:8], -cube_orientation * 2)
    # Organizes the output matrix correctly
    result = jnp.array([
        [rotated_array[0], rotated_array[1], rotated_array[2]],
        [rotated_array[7], current_side[8], rotated_array[3]],
        [rotated_array[6], rotated_array[5], rotated_array[4]],
    ])

    # When game variation 1 is selected, the current side of the cube is only shown when switching between sides
    return jax.lax.cond(
        pred=jnp.logical_and(game_variation == 1, is_moving_between_two_sides == 0),
        true_fun=lambda: jnp.array([[6, 6, 6], [6, 6, 6], [6, 6, 6]]),
        false_fun=lambda: result
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
    moving_counter = jax.lax.cond(
        pred=jnp.logical_and(jnp.logical_or(movement_state.is_moving_on_one_side, movement_state.is_moving_between_two_sides), state.step_counter % 4 == 0),
        true_fun=lambda: (movement_state.moving_counter + 1) % 6,
        false_fun=lambda: movement_state.moving_counter,
    )
    # The movement has finished if moving_counter equal 0
    is_moving_on_one_side, is_moving_between_two_sides = jax.lax.cond(
        pred=moving_counter == 0,
        true_fun=lambda: (0, 0),
        false_fun=lambda: (movement_state.is_moving_on_one_side, movement_state.is_moving_between_two_sides)
    )
    return MovementState(is_moving_on_one_side, is_moving_between_two_sides, moving_counter), render_only


class JaxVideoCube(JaxEnvironment[VideoCubeState, VideoCubeObservation, VideoCubeInfo, VideoCubeConstants]):
    def __init__(self, consts: VideoCubeConstants = None, reward_funcs: list[callable] = None, key: jax.random.PRNGKey = jax.random.key(int.from_bytes(os.urandom(3), byteorder='big')), selected_cube: int = 1, play_manually: bool = True, game_variation: int = 0):
        """ Initialisation of VideoCube Game

        :param consts: all constants needed for the game
        :param reward_funcs: list of functions used to compute rewards
        :param key: a key for generating random numbers
        :param selected_cube: index of the cube to play: 1 to 50 are the cubes from the game and 51 is randomly generated
        :param play_manually: whether to play the game manually and render the game
        :param game_variation: the number of game variation (0: normal game, 1: all tiles are blacked out, 2: only up and right are allowed)
        """

        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.renderer = VideoCubeRenderer
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
        self.key = key
        self.selected_cube = selected_cube
        self.play_manually = play_manually
        self.game_variation = game_variation

    def reset(self, key=None) -> Tuple[VideoCubeObservation, VideoCubeState]:
        if self.selected_cube == 51:
            cube = jax.random.permutation(self.key, jnp.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5]))
            player_color = jax.random.permutation(self.key, jnp.array([0, 1, 2, 3, 4, 5]))[0],
        else:
            cube = self.consts.CUBES[self.selected_cube - 1]
            player_color = self.consts.PLAYER_COLORS[self.selected_cube - 1]

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
            selected_cube=self.selected_cube,
            can_move=1,
            movement_state=MovementState(
                is_moving_on_one_side=0,
                is_moving_between_two_sides=0,
                moving_counter=0
            ),
            last_player_pos=self.consts.INITIAL_PLAYER_POS,
            last_cube_orientation=self.consts.INITIAL_ORIENTATION,
            last_cube_current_side=self.consts.INITIAL_CURRENT_SIDE,
            game_variation=self.game_variation
        )

        initial_obs = self._get_observation(new_state)

        return initial_obs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: VideoCubeState, action: chex.Array) -> Tuple[VideoCubeObservation, VideoCubeState, float, bool, VideoCubeInfo]:
        action = jnp.array([Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.RIGHT, Action.LEFT, Action.RIGHT,
                           Action.LEFT, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN, Action.RIGHT, Action.LEFT, Action.RIGHT, Action.LEFT])[action]

        # Set action to Noop if game_variation 2 (only up and right) is selected and something else was pressed
        action = jax.lax.cond(
            pred=state.game_variation == 2,
            true_fun=lambda: jax.lax.cond(
                pred=jnp.logical_or(jnp.logical_or(action == Action.NOOP, action == Action.FIRE), jnp.logical_or(action == Action.UP, action == Action.RIGHT)),
                true_fun=lambda: action,
                false_fun=lambda: Action.NOOP
            ),
            false_fun=lambda: action
        )

        # Determines if some parts of the code which are only relevant for rendering should be executed
        movement_state, render_only = jax.lax.cond(
            pred=self.play_manually,
            true_fun=lambda: movement_controller(state),
            false_fun=lambda: (state.movement_state, 0)
        )

        # Updates the last selected action and ignores fire and noop
        last_action = jax.lax.cond(
            pred=jnp.logical_or(jnp.logical_or(action == Action.NOOP, action == Action.FIRE), render_only == 1),
            true_fun=lambda: state.last_action,
            false_fun=lambda: action
        )

        # check if color should be changed when Action.FIRE is pressed else do nothing if fire is pressed change player_color and cube but do not create a new state
        cube, player_color = jax.lax.cond(
            pred=jnp.logical_and(jnp.equal(action, Action.FIRE), render_only == 0),
            true_fun=lambda: change_color(state.player_pos, state.player_color, state.cube),
            false_fun=lambda: (state.cube, state.player_color)
        )

        # Move player
        player_position, cube_current_side, cube_orientation, can_move, is_moving_on_one_side, is_moving_between_two_sides, movement_counter = jax.lax.cond(
            pred=jnp.logical_and(jnp.logical_and(action != Action.NOOP, action != Action.FIRE), render_only == 0),
            true_fun=lambda: (self.move(state, action)),
            false_fun=lambda: (state.player_pos, state.cube_current_side, state.cube_orientation, state.can_move,
                               movement_state.is_moving_on_one_side, movement_state.is_moving_between_two_sides, movement_state.moving_counter),
        )

        # Calculate player score
        player_score = jax.lax.cond(
            pred=jnp.logical_and(action != Action.NOOP, render_only == 0),
            true_fun=lambda: state.player_score + 1,
            false_fun=lambda: state.player_score
        )

        # Update last_player_position, last_cube_orientation, last_cube_current_side
        last_player_position, last_cube_orientation, last_cube_current_side = jax.lax.cond(
            pred=movement_counter != 0,
            true_fun=lambda: (state.last_player_pos, state.last_cube_orientation, state.last_cube_current_side),
            false_fun=lambda: (player_position, cube_orientation, cube_current_side)
        )

        new_state = VideoCubeState(
            player_pos=player_position,
            player_color=player_color,
            cube_orientation=cube_orientation,
            cube_current_side=cube_current_side,
            cube=cube,
            step_counter=state.step_counter + 1,
            player_score=player_score,
            last_action=last_action,
            selected_cube=state.selected_cube,
            can_move=can_move,
            movement_state=MovementState(
                moving_counter=movement_counter,
                is_moving_on_one_side=is_moving_on_one_side,
                is_moving_between_two_sides=is_moving_between_two_sides
            ),
            last_player_pos=last_player_position,
            last_cube_orientation=last_cube_orientation,
            last_cube_current_side=last_cube_current_side,
            game_variation=state.game_variation
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def move(self, state: VideoCubeState, action: chex.Array):
        """ Moves the player

        :param state: the current game state
        :param action: the selected action
        """

        def get_step_value(rotation, relative_direction):
            absolute = (rotation + relative_direction) % 4
            dirs = jnp.array([(0, 1), (1, 0), (0, -1), (-1, 0)])
            dx, dy = dirs[absolute]
            return dy * 12 + dx * 1

        # move_direction: up = 0, right = 1, down = 2, left = 3
        move_direction = jnp.array([0, 1, 3, 2])[action - 2]
        new_step_value = get_step_value(state.cube_orientation, move_direction)
        new_pos = state.player_pos + new_step_value

        return_values = jax.lax.cond(
            pred=jnp.logical_and(self.get_cube_side_index(state.player_pos) == self.get_cube_side_index(new_pos), state.cube[new_pos] != 6),
            true_fun=lambda: jax.lax.cond(
                pred=state.player_color == state.cube[state.player_pos + get_step_value(state.cube_orientation, move_direction)],
                true_fun=lambda: (state.player_pos, state.cube_current_side, state.cube_orientation, 0, 0, 0, 0),
                false_fun=lambda: (new_pos, state.cube_current_side, state.cube_orientation, 1, 1, 0, 1)
            ),
            false_fun=lambda: jax.lax.cond(
                pred=state.cube[self.move_side(state, new_step_value)[0]] == state.player_color,
                true_fun=lambda: (state.player_pos, state.cube_current_side, state.cube_orientation, 0, 0, 0, 0),
                false_fun=lambda: (*self.move_side(state, new_step_value), 1, 0, 1, 1)
            )
        )

        return return_values

    @partial(jax.jit, static_argnums=(0,))
    def get_cube_side_index(self, n: int, width: int = 12, block_size: int = 3) -> int:
        """ Returns the side of the cube on the given field. Values that are outside the cube return unspecified values due to efficiency reasons! """
        row = n // width          # row index (counting from the bottom)
        col = n % width           # column index (counting from left)
        block_x = col // block_size
        block_y = row // block_size

        return jax.lax.cond(
            pred=block_y == 2,
            true_fun=lambda: 5,
            false_fun=lambda: block_x + block_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def move_side(self, state: VideoCubeState, move_Value: chex.Numeric):
        """ Moves the player between two sides and returns the new player position and current side of the cube and the rotation of the cube

        :param state: the current game state
        :param move_Value: the movement of the index (or the player position) in the cube representation
        """
        cube_side_index_old = self.get_cube_side_index(state.player_pos)
        new_pos = state.player_pos + move_Value

        return jax.lax.cond(
            # Check if movement happens without rotation change
            pred = jnp.logical_and(jnp.logical_and(jnp.logical_and(new_pos > -1, new_pos < 108), state.cube[new_pos] != 6), jnp.logical_not(jnp.logical_or(jnp.logical_and(cube_side_index_old % 4 == 1, move_Value == -1), jnp.logical_and(cube_side_index_old == 4, move_Value == 1)))),
            # No rotation
            true_fun = lambda: (new_pos, self.get_cube_side_index(new_pos), state.cube_orientation),
            # Movement results in rotation
            false_fun = lambda: jax.lax.cond(
                # Check if movement stays in the second row of the cube representation or not
                pred = jnp.logical_and(state.cube_current_side > 0, state.cube_current_side < 5),
                # Moving away from a side in the second row
                true_fun = lambda: jax.lax.cond(
                    # Check if movement happens on side 1, 2, 3, 4 or 0, 5
                    pred = jnp.logical_and(state.cube_current_side % 3 == 1, abs(move_Value) == 1),
                    # Moving to field 1, 2, 3, 4
                    true_fun = lambda: (state.player_pos - move_Value * 11, state.cube_current_side - move_Value * 3, state.cube_orientation),  # moving between field 1 and 4 efficiently
                    # Moving to field 0 or 5
                    false_fun = lambda: jax.lax.cond(
                        # Checks if moving to field 5 or 0
                        pred = new_pos >= 72,
                        # Moving to field 5
                        true_fun = lambda: jax.lax.switch(
                            index = state.cube_current_side - 2,
                            branches = [
                                lambda: (74 + (12 * (state.player_pos - 63)), 5, (state.cube_orientation + 3) % 4),
                                lambda: (98 - (state.player_pos - 66), 5, (state.cube_orientation + 2) % 4),
                                lambda: (96 - (12 * (state.player_pos - 69)), 5, (state.cube_orientation + 1) % 4)
                            ]
                        ),
                        # Moving to field 0
                        false_fun=lambda: jax.lax.switch(
                            index = state.cube_current_side - 2,
                            branches = [
                                lambda: (26 - (12 * (state.player_pos - 39)), 0, (state.cube_orientation + 1) % 4),
                                lambda: (2 - (state.player_pos - 42), 0, (state.cube_orientation + 2) % 4),
                                lambda: (0 + (12 * (state.player_pos - 45)), 0, (state.cube_orientation + 3) % 4)
                            ]
                        ),
                    )
                ),
                # Moving away from a side in the bottom or top row
                false_fun = lambda: jax.lax.cond(
                    # Determines if movement is to the bottom side or top side
                    pred=state.cube_current_side == 0,
                    # Bottom side
                    true_fun=lambda: jax.lax.switch(
                        index = jnp.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3])[move_Value + 12],
                        branches = [
                            lambda: (44 - state.player_pos, 3, (state.cube_orientation + 2) % 4),
                            lambda: (45 + (state.player_pos / 12).astype("int32"), 4, (state.cube_orientation + 1) % 4),
                            lambda: (41 - ((state.player_pos - 2) / 12).astype("int32"), 2, (state.cube_orientation + 3) % 4),
                            lambda: (-20, 10, 9)  # unreachable case
                        ]
                    ),
                    # Top side
                    false_fun=lambda: jax.lax.switch(
                        index = jnp.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3])[move_Value + 12],
                        branches = [
                            lambda: (-20, 10, 9),  # unreachable case
                            lambda: (71 - ((state.player_pos - 72) /12).astype("int32"),4, (state.cube_orientation + 3) % 4),
                            lambda: (63 + ((state.player_pos - 74)/12).astype("int32"), 2, (state.cube_orientation + 1) % 4),
                            lambda: (66 + (98 - state.player_pos), 3, (state.cube_orientation + 2) % 4)
                        ]
                    ),
                ),
            ),
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

    def action_space(self) -> Space:
        return Discrete(18)

    def image_space(self) -> Space:
        return Box(0, 255, shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

    def observation_space(self) -> Space:
        return Dict({
            "cube_current_view": Box(0, 6, (3, 3), jnp.int32),
            "player_score": Box(0, 1000, (1,), jnp.int32),
            "player_color": Box(0, 6, (1,), jnp.int32),
            "player_x": Box(0, 2, (1,), jnp.int32),
            "player_y": Box(0, 2, (1,), jnp.int32)
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
            cube_current_view=get_view(state.cube, state.cube_current_side, state.cube_orientation, self.game_variation, state.movement_state.is_moving_between_two_sides),
            player_score=state.player_score,
            player_color=state.player_color,
            player_x=get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos, self.consts)[0],
            player_y=get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos, self.consts)[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoCubeState, all_rewards: jnp.ndarray) -> VideoCubeInfo:
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
        The game is done if the player has solved all sides of the cube or has a player score of 1000.
        """
        # Vectorize the check for a single side being solved.
        # We check for each of the 6 sides.
        all_sides_solved = jnp.all(jax.vmap(self.is_side_solved, in_axes=(None, 0))(state, jnp.arange(6)))
        return jnp.logical_or(all_sides_solved, state.player_score >= 1000)


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
        selected_cube_digits = aj.int_to_digits(state.selected_cube, max_digits=2)

        # 2. Determine parameters for selected cube rendering
        is_selected_cube_single_digit = state.selected_cube < 10
        selected_cube_start_index = jax.lax.select(is_selected_cube_single_digit, 1, 0)  # Start at index 1 if single, 0 if double
        selected_cube_num_to_render = jax.lax.select(is_selected_cube_single_digit, 1, 2)  # Render 1 digit if single, 2 if double

        # 3. Render selected cube number using the selective renderer
        raster = aj.render_label_selective(raster, 96, 191, selected_cube_digits, self.CUBE_DIGIT_SPRITES,
                                           selected_cube_start_index, selected_cube_num_to_render, spacing=5)

        # Render player_score
        # 1. Get digit array (always 3 digits)
        player_score_digits = aj.int_to_digits(state.player_score, max_digits=4)
        # 2. Determine parameters for player score rendering
        player_score_conditions = jnp.array([
            state.player_score < 10,
            jnp.logical_and(state.player_score >= 10, state.player_score < 100),
            jnp.logical_and(state.player_score >= 100, state.player_score < 1000),
            state.player_score >= 1000
        ], dtype=bool)
        # Start at index 3 if single, 2 if double, 1 if triple, 0 if quadrupel
        player_score_start_index = jnp.select(player_score_conditions, jnp.array([3, 2, 1, 0]))
        # Render 1 digit if single, 2 if double, 3 if triple, 4 if quadrupel
        player_score_num_to_render = jnp.select(player_score_conditions, jnp.array([1, 2, 3, 4]))

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
                body_fun=lambda i, val: jax.lax.cond(
                    pred=array[i] == value,
                    true_fun=lambda: i,
                    false_fun=lambda: val
                ),
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
                    body_fun=lambda j, val2: aj.render_at(val2, self.TILE_POSITIONS[i][j][0], self.TILE_POSITIONS[i][j][1],
                                                          jnp.where(state.game_variation == 1, self.SPRITE_TILE_BLACK, tiles[view[i][j]])),
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