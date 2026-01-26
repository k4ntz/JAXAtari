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
import numpy as np
from flax import struct

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.spaces import Space, Discrete, Box, Dict
from jaxatari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for VideoCube.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    # Helper lists for generating file paths
    label_files = [f'label/label_{i}.npy' for i in range(1, 65)]
    
    score_digit_files = []
    for i in range(1, 65):
        for k in range(10):
            score_digit_files.append(f'score/score_{k}/score_{k}_{i}.npy')
    
    player_anim_files = []
    orientations = ["down_up", "right", "left"]
    colors = ["red", "green", "blue", "orange", "purple", "white"]
    for orientation in orientations:
        for color in colors:
            for i in range(1, 7):
                player_anim_files.append(f'player_animation/{orientation}/{color}/{orientation}_{color}_{i}.npy')
    horiz_anim_files = []
    widths = [7, 8, 9, 15, 17, 18, 20, 21, 22, 23, 24]
    for color in colors:
        for width in widths:
            horiz_anim_files.append(f'switch_sides_animation/left_right/tile_{color}_width_{width}.npy')
    
    vert_anim_files = []
    heights = [15, 24, 30, 36, 41]
    for color in colors:
        for height in heights:
            vert_anim_files.append(f'switch_sides_animation/up_down/tile_{color}_hight_{height}.npy')

    return (
        # Backgrounds
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'background_switch', 'type': 'single', 'file': 'background_switch_sides_vertically.npy'},
        
        # Tile colors (Indices 0-6 match game logic: R,G,B,O,P,W,Black)
        {'name': 'tiles', 'type': 'group', 'files': [
            'tile/tile_red.npy',
            'tile/tile_green.npy',
            'tile/tile_blue.npy',
            'tile/tile_orange.npy',
            'tile/tile_purple.npy',
            'tile/tile_white.npy',
            'tile/tile_black.npy'
        ]},
        
        # Digits
        {'name': 'cube_digits', 'type': 'digits', 'pattern': 'cube_digit/cube_digit_{}.npy'},
        
        # Groups of sprites
        {'name': 'labels', 'type': 'group', 'files': label_files},
        {'name': 'score_digits_all', 'type': 'group', 'files': score_digit_files},
        {'name': 'player_anims', 'type': 'group', 'files': player_anim_files},
        {'name': 'horiz_anims', 'type': 'group', 'files': horiz_anim_files},
        {'name': 'vert_anims', 'type': 'group', 'files': vert_anim_files},
    )


class VideoCubeConstants(AutoDerivedConstants):
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    SELECTED_CUBE: int = struct.field(pytree_node=False, default=1)
    """ The selected cube. 1 - 50 are the standard cubes and 51 a random generated cube """
    GAME_VARIATION: int = struct.field(pytree_node=False, default=0)
    """ The selected game variation: 0 = normal game, 1 = all tiles are blacked out, 2 = only up and right are allowed """
    CUBES: list[list[int]] = struct.field(pytree_node=False, default_factory=lambda: [
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
    ])
    """ The color values of the 50 standard cubes """
    PLAYER_COLORS: list[int] = struct.field(pytree_node=False, default_factory=lambda: [
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
    ])
    """ The color of the player, depending on the selected cube """
    INITIAL_PLAYER_POS: int = struct.field(pytree_node=False, default=49)
    """ The initial player position
    Important: the initial player position and the initial current side must fit together
    """
    INITIAL_CURRENT_SIDE: int = struct.field(pytree_node=False, default=1)
    """ The initial side the player is starting on 
    Important: the initial player position and the initial current side must fit together
    """
    INITIAL_ORIENTATION: int = struct.field(pytree_node=False, default=0)
    """ The initial orientation of the player """
    INITIAL_PLAYER_SCORE: int = struct.field(pytree_node=False, default=0)
    """ The initial score of the player """
    INITIAL_PLAYER_LOOKING_DIRECTION: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array(Action.RIGHT))
    """ The initial looking direction of the player 
    Only up, down, right and left are possible
    """
    CUBE_SIDES: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        [24, 25, 26, 14, 2, 1, 0, 12, 13],
        [60, 61, 62, 50, 38, 37, 36, 48, 49],
        [63, 64, 65, 53, 41, 40, 39, 51, 52],
        [66, 67, 68, 56, 44, 43, 42, 54, 55],
        [69, 70, 71, 59, 47, 46, 45, 57, 58],
        [96, 97, 98, 86, 74, 73, 72, 84, 85]
    ]))
    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple[dict, ...] = struct.field(pytree_node=False, default=_get_default_asset_config())



@struct.dataclass
class MovementState:
    is_moving_on_one_side: chex.Numeric
    """ Tells if the player is moving on one side of the cube """
    is_moving_between_two_sides: chex.Numeric
    """ Tells if the player is moving between two sides of the cube """
    moving_counter: chex.Numeric
    """ The counter for the animation of the player if he is moving """


@struct.dataclass
class VideoCubeState:
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


@struct.dataclass
class VideoCubeObservation:
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


@struct.dataclass
class VideoCubeInfo:
    time: jnp.ndarray


@jax.jit
def get_player_position_helper(cube_current_side, cube_orientation, player_pos, cube_sides):
    """ Helper function for renderer to compute player position.
    The main implementation is in JaxVideoCube.get_player_position.
    """
    # Get the current side of the cube as an array
    current_side = cube_sides[cube_current_side]
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
    # Minimal ALE action set for Video Cube (from scripts/action_space_helper.py)
    ACTION_SET: jnp.ndarray = jnp.array(
        [
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
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )

    def __init__(self, consts: VideoCubeConstants = None):
        """ Initialisation of VideoCube Game

        :param consts: all constants needed for the game
        """
        consts = consts or VideoCubeConstants()
        super().__init__(consts)
        self.consts = consts
        self.renderer = VideoCubeRenderer(self.consts)
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
        # Translate compact agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, action.astype(jnp.int32))

        # Treat FIRE-combo actions as their base directions for movement/aiming.
        # (Only plain FIRE triggers the color-change mechanic below.)
        action = jnp.select(
            condlist=[
                atari_action == Action.UPFIRE,
                atari_action == Action.RIGHTFIRE,
                atari_action == Action.LEFTFIRE,
                atari_action == Action.DOWNFIRE,
                atari_action == Action.UPRIGHTFIRE,
                atari_action == Action.UPLEFTFIRE,
                atari_action == Action.DOWNRIGHTFIRE,
                atari_action == Action.DOWNLEFTFIRE,
            ],
            choicelist=[
                Action.UP,
                Action.RIGHT,
                Action.LEFT,
                Action.DOWN,
                Action.UPRIGHT,
                Action.UPLEFT,
                Action.DOWNRIGHT,
                Action.DOWNLEFT,
            ],
            default=atari_action,
        )

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
        info = self._get_info(new_state)
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
        # Map diagonal actions to their primary axis direction.
        move_direction = jnp.select(
            condlist=[
                jnp.logical_or(action == Action.UP, jnp.logical_or(action == Action.UPRIGHT, action == Action.UPLEFT)),
                jnp.logical_or(action == Action.RIGHT, jnp.logical_or(action == Action.UPRIGHT, action == Action.DOWNRIGHT)),
                jnp.logical_or(action == Action.DOWN, jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNLEFT)),
                jnp.logical_or(action == Action.LEFT, jnp.logical_or(action == Action.UPLEFT, action == Action.DOWNLEFT)),
            ],
            choicelist=[0, 1, 2, 3],
            default=0,
        ).astype(jnp.int32)
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
    def get_player_position(self, cube_current_side, cube_orientation, player_pos):
        """ Computes from the global player position the position of the player on one side of the cube

        :param cube_current_side: the number of the side the player is looking at
        :param cube_orientation: the rotation of the current side
        :param player_pos: the global player position
        """
        # Get the current side of the cube as an array
        current_side = self.consts.CUBE_SIDES[cube_current_side]
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

    def render(self, state: VideoCubeState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> Discrete:
        return Discrete(len(self.ACTION_SET))

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
            player_x=self.get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos)[0],
            player_y=self.get_player_position(state.cube_current_side, state.cube_orientation, state.player_pos)[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: VideoCubeState) -> VideoCubeInfo:
        return VideoCubeInfo(state.step_counter)

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




class VideoCubeRenderer(JAXGameRenderer):
    def __init__(self, consts: VideoCubeConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or VideoCubeConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/videocube"
        
        # 2. Asset manifest is now loaded from constants
        # The large local 'asset_config' list has been removed.
        # 3. Load all assets
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(list(self.consts.ASSET_CONFIG), self.sprite_path)

        # 4. Store animation helper arrays (unchanged game logic)
        self.TILE_POSITIONS_HORIZONTAL_ROTATION = jnp.array([
            [[[32, 28], [55, 28], [79, 28], [103, 28], [112, 28], [121, 28]],
             [[32, 78], [55, 78], [79, 78], [103, 78], [112, 78], [121, 78]],
             [[32, 128], [55, 128], [79, 128], [103, 128], [112, 128], [121, 128]]],
            [[[28, 28], [49, 28], [70, 28], [91, 28], [106, 28], [121, 28]],
             [[28, 78], [49, 78], [70, 78], [91, 78], [106, 78], [121, 78]],
             [[28, 128], [49, 128], [70, 128], [91, 128], [106, 128], [121, 128]]],
            [[[28, 28], [46, 28], [64, 28], [82, 28], [100, 28], [118, 28]],
             [[28, 78], [46, 78], [64, 78], [82, 78], [100, 78], [118, 78]],
             [[28, 128], [46, 128], [64, 128], [82, 128], [100, 128], [118, 128]]],
            [[[28, 28], [43, 28], [58, 28], [74, 28], [94, 28], [115, 28]],
             [[28, 78], [43, 78], [58, 78], [74, 78], [94, 78], [115, 78]],
             [[28, 128], [43, 128], [58, 128], [74, 128], [94, 128], [115, 128]]],
            [[[32, 28], [40, 28], [49, 28], [59, 28], [82, 28], [106, 28]],
             [[32, 78], [40, 78], [49, 78], [59, 78], [82, 78], [106, 78]],
             [[32, 128], [40, 128], [49, 128], [59, 128], [82, 128], [106, 128]]]
        ])

        self.TILE_POSITIONS_VERTICAL_ROTATION = jnp.array([
            [[[40, 14], [67, 14], [94, 12]],
             [[40, 31], [67, 31], [94, 32]],
             [[40, 48], [67, 48], [94, 48]],
             [[40, 66], [67, 66], [94, 66]],
             [[40, 109], [67, 109], [94, 109]],
             [[40, 152], [67, 152], [94, 152]]],
            [[[40, 8], [67, 8], [94, 8]],
             [[40, 34], [67, 34], [94, 34]],
             [[40, 60], [67, 60], [94, 60]],
             [[40, 87], [67, 87], [94, 87]],
             [[40, 125], [67, 125], [94, 125]],
             [[40, 163], [67, 163], [94, 163]]],
            [[[40, 8], [67, 8], [94, 8]],
             [[40, 40], [67, 40], [94, 40]],
             [[40, 72], [67, 72], [94, 72]],
             [[40, 105], [67, 105], [94, 105]],
             [[40, 137], [67, 137], [94, 137]],
             [[40, 169], [67, 169], [94, 169]]],
            [[[40, 8], [67, 8], [94, 8]],
             [[40, 46], [67, 46], [94, 46]],
             [[40, 84], [67, 84], [94, 84]],
             [[40, 123], [67, 123], [94, 123]],
             [[40, 149], [67, 149], [94, 149]],
             [[40, 175], [67, 175], [94, 175]]],
            [[[40, 14], [67, 14], [94, 14]],
             [[40, 57], [67, 57], [94, 57]],
             [[40, 100], [67, 100], [94, 100]],
             [[40, 144], [67, 144], [94, 144]],
             [[40, 161], [67, 161], [94, 161]],
             [[40, 178], [67, 178], [94, 178]]]
        ])

        self.TILE_POSITIONS = jnp.array([
            [[40, 28], [67, 28], [94, 28]],
            [[40, 78], [67, 78], [94, 78]],
            [[40, 128], [67, 128], [94, 128]]
        ])

        self.WIDTHS = jnp.array([
            [23, 24, 23, 9, 9, 7],
            [21, 21, 20, 15, 15, 15],
            [18, 18, 17, 18, 18, 18],
            [15, 15, 15, 20, 21, 21],
            [8, 9, 9, 23, 24, 22]
        ])

        self.HEIGHTS = jnp.array([
            [15, 15, 15, 41, 41, 41],
            [24, 24, 24, 36, 36, 36],
            [30, 30, 30, 30, 30, 30],
            [36, 36, 36, 24, 24, 24],
            [41, 41, 41, 15, 15, 15]
        ])

        self.PLAYER_POSITIONS = jnp.array([
            [
                [[46, 48], [73, 48], [100, 48]],
                [[46, 98], [73, 98], [100, 98]],
                [[46, 148], [73, 148], [100, 148]]
            ],
            [
                [[46, 44], [73, 44], [100, 44]],
                [[46, 94], [73, 94], [100, 94]],
                [[46, 144], [73, 144], [100, 144]]
            ]
        ])

        self.PLAYER_MOVEMENT_ON_ONE_SIDE = jnp.array([
            [[0, -8], [0, -16], [0, -26], [0, -34], [0, -42], [0, -50]],
            [[7, 1], [10, 0], [16, 1], [18, 0], [25, 1], [27, 0]],
            [[-7, 1], [-10, 0], [-16, 1], [-18, 0], [-25, 1], [-27, 0]],
            [[0, 8], [0, 16], [0, 26], [0, 34], [0, 42], [0, 50]]
        ])

        self.PLAYER_MOVEMENT_BETWEEN_TWO_SIDES = jnp.array([
            [[0, 6], [0, 16], [0, 32], [0, 48], [0, 70], [0, 100]],
            [[-7, 0], [-17, 0], [-25, 0], [-35, 0], [-42, 0], [-54, 0]],
            [[12, 0], [19, 0], [29, 0], [37, 0], [47, 0], [54, 0]],
            [[0, -30], [0, -52], [0, -68], [0, -84], [0, -94], [0, -100]]
        ])

        # 5. *** NEW: PRECOMPUTATION ***
        # Create static numpy arrays for lookup
        widths_np = np.array([7, 8, 9, 15, 17, 18, 20, 21, 22, 23, 24])
        heights_np = np.array([15, 24, 30, 36, 41])
        
        # Create python-side hash maps for fast lookups
        width_map = {val: idx for idx, val in enumerate(widths_np)}
        height_map = {val: idx for idx, val in enumerate(heights_np)}
        
        # Use np.vectorize to apply the map to the static JAX arrays
        static_widths_table = np.array(self.WIDTHS)
        static_heights_table = np.array(self.HEIGHTS)
        
        lookup_w_np = np.vectorize(width_map.get)(static_widths_table)
        lookup_h_np = np.vectorize(height_map.get)(static_heights_table)
        
        # Store the final precomputed lookup tables as JAX arrays
        self.WIDTH_INDEX_LOOKUP = jnp.array(lookup_w_np, dtype=jnp.int32)
        self.HEIGHT_INDEX_LOOKUP = jnp.array(lookup_h_np, dtype=jnp.int32)
        
        # 6. Store digit mask dimensions
        self.SCORE_DIGIT_H = self.SHAPE_MASKS['score_digits_all'].shape[1]
        self.SCORE_DIGIT_W = self.SHAPE_MASKS['score_digits_all'].shape[2]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: VideoCubeState):
        """ Responsible for the graphical representation of the game

        :param state: the current game state
        """
        # Create empty raster
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render background
        bg_mask = jnp.where(
            jnp.logical_and(
                jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN),
                state.movement_state.is_moving_between_two_sides == 1
            ),
            self.SHAPE_MASKS['background_switch'],
            self.BACKGROUND  # Use the ID mask of the background, not the sprite
        )

        # We must stamp the background *over* the default black
        raster = self.jr.render_at(raster, 0, 0, bg_mask)

        # Render Atari label
        # 1. Calculate the index for the label
        label_index = jnp.floor(state.step_counter / 8).astype(jnp.int32) % 64
        label_mask = self.SHAPE_MASKS['labels'][label_index]
        
        raster = jnp.where(
            jnp.logical_and(
                jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN),
                state.movement_state.is_moving_between_two_sides == 1
            ), 
            raster, 
            self.jr.render_at(raster, 55, 5, label_mask)
        )

        # Render number of selected cube
        selected_cube_digits = self.jr.int_to_digits(self.consts.SELECTED_CUBE, max_digits=2)
        is_selected_cube_single_digit = self.consts.SELECTED_CUBE < 10
        selected_cube_start_index = jax.lax.select(is_selected_cube_single_digit, 1, 0)
        selected_cube_num_to_render = jax.lax.select(is_selected_cube_single_digit, 1, 2)

        raster = self.jr.render_label_selective(
            raster, 96, 191, selected_cube_digits, self.SHAPE_MASKS['cube_digits'],
            selected_cube_start_index, selected_cube_num_to_render, 
            spacing=5, # static value
            max_digits_to_render=2 # static value
        )

        # Render player_score
        player_score_digits = self.jr.int_to_digits(state.player_score, max_digits=6)
        player_score_conditions = jnp.array([
            state.player_score < 10,
            jnp.logical_and(state.player_score >= 10, state.player_score < 100),
            jnp.logical_and(state.player_score >= 100, state.player_score < 1000),
            jnp.logical_and(state.player_score >= 1000, state.player_score < 10000),
            jnp.logical_and(state.player_score >= 10000, state.player_score < 100000),
            state.player_score >= 100000
        ], dtype=bool)
        player_score_start_index = jnp.select(player_score_conditions, jnp.array([5, 4, 3, 2, 1, 0]))
        player_score_num_to_render = jnp.select(player_score_conditions, jnp.array([1, 2, 3, 4, 5, 6]))

        # Dynamically slice the correct set of 10 digit masks for the current label color
        all_score_masks = self.SHAPE_MASKS['score_digits_all']
        current_score_masks = jax.lax.dynamic_slice(
            all_score_masks, 
            (label_index * 10, 0, 0), 
            (10, self.SCORE_DIGIT_H, self.SCORE_DIGIT_W)
        )
        
        raster = self.jr.render_label_selective(
            raster, 95, 180, player_score_digits,
            current_score_masks, player_score_start_index, player_score_num_to_render, 
            spacing=10, # static value
            max_digits_to_render=6 # static value
        )

        # *** REMOVED get_index *** # It is no longer needed in the render function.
        last_view = get_view(state.cube, state.last_cube_current_side, state.last_cube_orientation, 0, state.movement_state.is_moving_between_two_sides)

        @jax.jit
        def tiles_move_on_one_side():
            """ Returns the raster containing the tiles in the correct colors """
            tile_masks = self.SHAPE_MASKS['tiles'] # Shape (7, H, W)
            result_raster = raster
            def body_i(i, val1):
                def body_j(j, val2):
                    # Select the correct tile mask based on view
                    color_index = view[i, j]
                    mask = jax.lax.select(
                        self.consts.GAME_VARIATION == 1,
                        tile_masks[6], # Black tile
                        tile_masks[color_index]
                    )
                    return self.jr.render_at(
                        val2, 
                        self.TILE_POSITIONS[i, j, 0], 
                        self.TILE_POSITIONS[i, j, 1], 
                        mask
                    )
                return jax.lax.fori_loop(0, 3, body_j, val1)
            
            result_raster = jax.lax.fori_loop(0, 3, body_i, result_raster)
            return result_raster

        @jax.jit
        def tiles_move_between_two_sides_vertically():
            """ Returns the raster containing the vertical rotation of the cube """
            result_raster = raster
            
            combined_view = jax.lax.cond(
                pred=state.last_action == Action.UP,
                true_fun=lambda: jnp.concat((view, last_view), axis=0),
                false_fun=lambda: jnp.concat((last_view, view), axis=0),
            )
            counter = jax.lax.cond(
                pred=state.last_action == Action.UP,
                true_fun=lambda: state.movement_state.moving_counter - 1,
                false_fun=lambda: 6 - state.movement_state.moving_counter - 1
            )
            
            def body_i(i, val1):
                def body_j(j, val2):
                    color_index = combined_view[i, j]
                    
                    # *** OPTIMIZATION ***
                    # Directly look up the precomputed index instead of searching
                    height_index = self.HEIGHT_INDEX_LOOKUP[counter, i]
                    
                    # flat_index = color_index * num_heights + height_index
                    flat_index = color_index * 5 + height_index
                    mask = self.SHAPE_MASKS['vert_anims'][flat_index]
                    
                    return self.jr.render_at(
                        val2,
                        self.TILE_POSITIONS_VERTICAL_ROTATION[counter, i, j, 0],
                        self.TILE_POSITIONS_VERTICAL_ROTATION[counter, i, j, 1],
                        mask
                    )
                return jax.lax.fori_loop(0, 3, body_j, val1)
            
            result_raster = jax.lax.fori_loop(0, 6, body_i, result_raster)
            return result_raster

        @jax.jit
        def tiles_move_between_two_sides_horizontally():
            """ Returns the raster containing the horizontal rotation of the cube """
            result_raster = raster
            
            combined_view = jax.lax.cond(
                pred=state.last_action == Action.RIGHT,
                true_fun=lambda: jnp.hstack((last_view, view)),
                false_fun=lambda: jnp.hstack((view, last_view)),
            )
            counter = jax.lax.cond(
                pred=state.last_action == Action.RIGHT,
                true_fun=lambda: state.movement_state.moving_counter - 1,
                false_fun=lambda: 6 - state.movement_state.moving_counter - 1
            )
            
            def body_i(i, val1):
                def body_j(j, val2):
                    color_index = combined_view[i, j]
                    # *** OPTIMIZATION ***
                    # Directly look up the precomputed index instead of searching
                    width_index = self.WIDTH_INDEX_LOOKUP[counter, j]
                    
                    # flat_index = color_index * num_widths + width_index
                    flat_index = color_index * 11 + width_index
                    mask = self.SHAPE_MASKS['horiz_anims'][flat_index]
                    
                    return self.jr.render_at(
                        val2,
                        self.TILE_POSITIONS_HORIZONTAL_ROTATION[counter, i, j, 0],
                        self.TILE_POSITIONS_HORIZONTAL_ROTATION[counter, i, j, 1],
                        mask
                    )
                return jax.lax.fori_loop(0, 6, body_j, val1)
            result_raster = jax.lax.fori_loop(0, 3, body_i, result_raster)
            return result_raster

        # Render the tiles of the cube
        view = get_view(state.cube, state.cube_current_side, state.cube_orientation, 0, state.movement_state.is_moving_between_two_sides)
        raster = jax.lax.cond(
            pred=jnp.logical_or(state.movement_state.is_moving_on_one_side, state.movement_state.moving_counter == 0),
            true_fun=lambda: tiles_move_on_one_side(),
            false_fun=lambda: jax.lax.cond(
                pred=jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN),
                true_fun=lambda: tiles_move_between_two_sides_vertically(),
                false_fun=lambda: tiles_move_between_two_sides_horizontally()
            ),
        )
        
        # Render player
        player_position = get_player_position_helper(state.cube_current_side, state.cube_orientation, state.player_pos, self.consts.CUBE_SIDES)
        last_player_position = get_player_position_helper(state.last_cube_current_side, state.last_cube_orientation, state.last_player_pos, self.consts.CUBE_SIDES)
        sprite_on_one_side_indices = jnp.array([0, 1, 2, 0])
        
        @jax.jit
        def get_player_mask_and_pos():
            # Check if player is moving
            def moving_true():
                # Check if player is moving on one side or between two sides
                def move_on_one_side():
                    orientation_index = sprite_on_one_side_indices[state.last_action - 2]
                    color_index = state.player_color
                    frame_index = state.movement_state.moving_counter
                    # flat_index = orientation_index * (6*6) + color_index * 6 + frame_index
                    flat_index = orientation_index * 36 + color_index * 6 + frame_index
                    
                    mask = self.SHAPE_MASKS['player_anims'][flat_index]
                    pos_x = self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1), last_player_position[1], last_player_position[0], 0] + self.PLAYER_MOVEMENT_ON_ONE_SIDE[state.last_action - 2, frame_index, 0]
                    pos_y = self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1), last_player_position[1], last_player_position[0], 1] + self.PLAYER_MOVEMENT_ON_ONE_SIDE[state.last_action - 2, frame_index, 1]
                    return mask, pos_x, pos_y
                def move_between_sides():
                    orientation_index = sprite_on_one_side_indices[state.last_action - 2]
                    color_index = state.player_color
                    frame_index = state.movement_state.moving_counter
                    # flat_index = orientation_index * (6*6) + color_index * 6 + frame_index
                    flat_index = orientation_index * 36 + color_index * 6 + frame_index
                    
                    mask = self.SHAPE_MASKS['player_anims'][flat_index]
                    pos_x = self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1), last_player_position[1], last_player_position[0], 0] + self.PLAYER_MOVEMENT_BETWEEN_TWO_SIDES[state.last_action - 2, frame_index, 0]
                    pos_y = self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1), last_player_position[1], last_player_position[0], 1] + self.PLAYER_MOVEMENT_BETWEEN_TWO_SIDES[state.last_action - 2, frame_index, 1]
                    return mask, pos_x, pos_y
                return jax.lax.cond(
                    state.movement_state.is_moving_on_one_side,
                    move_on_one_side,
                    move_between_sides
                )
            # Player is not moving (or anim is over), draw idle frame
            def moving_false():
                orientation_index = sprite_on_one_side_indices[state.last_action - 2]
                color_index = state.player_color
                frame_index = 5 # Idle frame
                # flat_index = orientation_index * (6*6) + color_index * 6 + frame_index
                flat_index = orientation_index * 36 + color_index * 6 + frame_index
                
                mask = self.SHAPE_MASKS['player_anims'][flat_index]
                pos_x = self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1), player_position[1], player_position[0], 0]
                pos_y = self.PLAYER_POSITIONS[jnp.where(jnp.logical_or(state.last_action == Action.UP, state.last_action == Action.DOWN), 0, 1), player_position[1], player_position[0], 1]
                return mask, pos_x, pos_y

            return jax.lax.cond(
                pred=jnp.logical_and(state.can_move == 1, state.movement_state.moving_counter != 0),
                true_fun=moving_true,
                false_fun=moving_false
            )

        player_mask, player_x, player_y = get_player_mask_and_pos()
        raster = self.jr.render_at(
            raster, 
            player_x, 
            player_y, 
            player_mask, 
            flip_offset=self.FLIP_OFFSETS['player_anims']
        )

        return self.jr.render_from_palette(raster, self.PALETTE)