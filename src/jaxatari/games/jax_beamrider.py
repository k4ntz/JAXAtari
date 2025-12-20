"Authors: Lasse Reith, Benedikt Schwarz, Shir Nussbaum"
import os
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Optional, Tuple
import chex
import jax
import jax.lax
import jax.numpy as jnp

import jaxatari.spaces as spaces

from jaxatari.environment import JAXAtariAction as Action, JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

BLUE_LINE_INIT_TABLE = jnp.array([[45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 124, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 136, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 60, 74, 92, 124, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 150], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 124, -1]])
BLUE_LINE_LOOP_TABLE = jnp.array([[44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 110, 154], [44, 48, 56, 68, 84, 110, 154], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 110, 154], [44, 48, 56, 68, 84, 110, 154], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 58, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 60, 74, 92, 124, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 96, 130, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 102, 140, -1], [45, 56, 66, 82, 104, 144, -1], [45, 56, 66, 82, 104, 144, -1], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 108, 152], [44, 48, 56, 68, 84, 110, 154], [44, 48, 56, 68, 84, 110, 154], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 114, 162], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 118, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 90, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 72, 92, 120, -1], [44, 50, 60, 74, 92, 122, -1], [44, 50, 60, 74, 92, 122, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 124, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 74, 94, 126, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 94, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 96, 128, -1], [44, 52, 62, 76, 98, 132, -1], [44, 52, 62, 76, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 132, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 98, 134, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 136, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 78, 100, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 64, 80, 102, 138, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 102, 140, -1], [45, 54, 66, 80, 104, 142, -1], [45, 54, 66, 80, 104, 142, -1], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 146], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 66, 82, 106, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 82, 108, 148], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 108, 150], [44, 48, 56, 68, 84, 110, 154], [44, 48, 56, 68, 84, 110, 154], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 68, 86, 110, 156], [44, 48, 58, 70, 86, 112, 158], [44, 48, 58, 70, 86, 112, 158], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 86, 114, 160], [44, 50, 58, 70, 88, 116, -1], [44, 50, 58, 70, 88, 116, -1]])


PROJECTILE_STEP_FRAMES = 2
PROJECTILE_Y_TABLE_LASER = jnp.array(
    [155, 141, 129, 118, 109, 102, 95, 89, 84, 79, 75, 71, 67, 64, 800],
    dtype=jnp.float32,
)
PROJECTILE_Y_TABLE_TORPEDO = jnp.array(
    [155, 141, 129, 118, 109, 102, 95, 89, 84, 79, 75, 71, 67, 64, 61, 58, 56, 53, 800, 49, 800],
    dtype=jnp.float32,
)
LASER_PROJECTILE_FRAMES = PROJECTILE_Y_TABLE_LASER.shape[0] * PROJECTILE_STEP_FRAMES
TORPEDO_PROJECTILE_FRAMES = PROJECTILE_Y_TABLE_TORPEDO.shape[0] * PROJECTILE_STEP_FRAMES


class WhiteUFOPattern(IntEnum):
    """Behavioral patterns used by the white UFO enemies."""

    IDLE = 0
    DROP_STRAIGHT = 1
    DROP_RIGHT = 2
    DROP_LEFT = 3
    RETREAT = 4
    SHOOT = 5
    MOVE_BACK = 6
    KAMIKAZE = 7
    TRIPLE_SHOT_RIGHT = 8
    TRIPLE_SHOT_LEFT = 9


class BouncerState(IntEnum):
    SWITCHING = 0  # Also used for entry
    DOWN = 1
    UP = 2


class LaneBlockerState(IntEnum):
    DESCEND = 0
    HOLD = 1
    SINK = 2
    RETREAT = 3


class BeamriderConstants(NamedTuple):

    WHITE_UFOS_PER_SECTOR: int = 3

    RENDER_SCALE_FACTOR: int = 4
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    PLAYER_COLOR: Tuple[int, int, int] = (223, 183, 85)
    LEFT_CLIP_PLAYER: int = 27
    RIGHT_CLIP_PLAYER: int = 142
    BOTTOM_OF_LANES: Tuple[int, int, int, int, int] = (27,52,77,102,127)
    TOP_OF_LANES: Tuple[int, int, int, int, int] = (38,61,71,81,91,102,123)  #lane 0,6 are connected to points in middle of the map, not to bottom lane points
    
    TOP_TO_BOTTOM_LANE_VECTORS: Tuple[Tuple[float, float],Tuple[float, float],Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2,4),(-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4),(2,4))


    MAX_LASER_Y: int = 64
    MIN_BULLET_Y:int =155
    MAX_TORPEDO_Y: int = 49
    MAX_TORPEDO_Y_MOTHERSHIP_SCENE: int = 45
    BOTTOM_TO_TOP_LANE_VECTORS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4))

    PLAYER_POS_Y: int = 164
    PLAYER_SPEED: float = 2.5
    # ALE uses projectile existence as the main limiter.
    # We use a shared recovery window after hit/destruction (measured ~8 frames in ALE).
    PLAYER_SHOT_RECOVERY: int = 6
    # Delay from button press to shot appearance (measured ~3 frames in ALE).
    PLAYER_SHOT_LAUNCH_DELAY: int = 2

    BOTTOM_CLIP:int = 164
    TOP_CLIP:int=43
    LASER_ID: int = 1
    TORPEDO_ID: int = 2
    BULLET_OFFSCREEN_POS: Tuple[int, int] = (800.0, 800.0)
    ENEMY_OFFSCREEN_POS: Tuple[int,int] = (-100, -100)
    MIN_BLUE_LINE_POS: int = 46
    MAX_BLUE_LINE_POS: int = 160
    WHITE_UFO_RETREAT_DURATION: int = 28
    ####PATTERNS:                                                           IDLE | DROP_STRAIGHT | DROP_RIGHT | DROP_LEFT | RETREAT | SHOOT | MOVE_BACK | KAMIKAZE | TRIPLE_SHOT_RIGHT | TRIPLE_SHOT_LEFT
    WHITE_UFO_PATTERN_DURATIONS: Tuple[int, ...] =                          (0,          42,            42,         42,         28,     0,      42,         100,            123,                123)
    WHITE_UFO_PATTERN_PROBS: Tuple[float, ...] =                            (            0.3,           0.2,        0.2,                0.2,    0.1,        0.3,            0.2,                0.2) #these probas are not 1:1, as some patterns have activation conditions
    WHITE_UFO_SPEED_FACTOR: float = 0.1
    WHITE_UFO_SHOT_SPEED_FACTOR: float = 0.8
    WHITE_UFO_RETREAT_P_MIN: float = 0.005
    WHITE_UFO_RETREAT_P_MAX: float = 0.1
    WHITE_UFO_RETREAT_ALPHA: float = 0.01
    WHITE_UFO_RETREAT_SPEED_MULT: float = 2.5
    WHITE_UFO_TOP_LANE_MIN_SPEED: float = 0.3
    WHITE_UFO_TOP_LANE_TURN_SPEED: float = 0.5
    WHITE_UFO_ATTACK_P_MIN: float = 0.0004
    WHITE_UFO_ATTACK_P_MAX: float = 0.8
    WHITE_UFO_ATTACK_ALPHA: float = 0.0002
    ENEMY_EXPLOSION_FRAMES: int = 21
    KAMIKAZE_Y_THRESHOLD: float = 86.0

    # Bouncer constants
    BOUNCER_SPAWN_HEIGHT: float = 75.0
    BOUNCER_START_LEVEL: int = 8
    BOUNCER_SPEED_PATTERN: Tuple[int, int, int, int] = (0, 3, 0, 3)
    BOUNCER_SWITCH_SPEED_X: float = 1.2
    BOUNCER_SWITCH_SPEED_Y: float = 1.0

    # Mothership Explosion
    # Sequence: 1, 2, 1, 2, 3, 2, 3, 2, 3 (indices 0, 1, 0, 1, 2, 1, 2, 1, 2)
    # 8 frames per sprite step -> total 9 steps * 8 frames = 72 frames
    MOTHERSHIP_EXPLOSION_SEQUENCE: Tuple[int, ...] = (0, 1, 0, 1, 2, 1, 2, 1, 2)
    MOTHERSHIP_EXPLOSION_STEP_DURATION: int = 8
    MOTHERSHIP_HITBOX_SIZE: int = 16

    # Sprite sizes (H, W)
    PLAYER_SPRITE_SIZE: Tuple[int, int] = (16, 15)
    UFO_SPRITE_SIZES: Tuple[Tuple[int, int], ...] = (
        (1, 1), (2, 3), (2, 4), (2, 4), (4, 6), (4, 7), (6, 8)
    )
    BOUNCER_SPRITE_SIZE: Tuple[int, int] = (7, 8)
    METEOROID_SPRITE_SIZE: Tuple[int, int] = (7, 7)
    BULLET_SPRITE_SIZES: Tuple[Tuple[int, int], ...] = (
        (5, 8), (1, 1), (3, 3), (5, 5) # Laser, Torpedo 3, 2, 1
    )
    ENEMY_SHOT_SPRITE_SIZES: Tuple[Tuple[int, int], ...] = (
        (6, 2), (2, 4) # Vertical, Horizontal
    )
    REJUVENATOR_SPRITE_SIZES: Tuple[Tuple[int, int], ...] = (
        (2, 3), (4, 3), (5, 4), (7, 5), (7, 7) # Stage 1-4, Dead
    )
    FALLING_ROCK_SPRITE_SIZES: Tuple[Tuple[int, int], ...] = (
        (3, 4), (4, 5), (5, 6), (6, 8)
    )
    LANE_BLOCKER_SPRITE_SIZES: Tuple[Tuple[int, int], ...] = (
        (3, 3), (4, 5), (7, 8), (7, 8)
    )
    MOTHERSHIP_SPRITE_SIZE: Tuple[int, int] = (7, 16)

    # Blue line constants
    BLUE_LINE_OFFSCREEN_Y = 500

    CHASING_METEOROID_MAX: int = 8
    CHASING_METEOROID_WAVE_MIN: int = 2
    CHASING_METEOROID_WAVE_MAX: int = 8
    CHASING_METEOROID_SPAWN_INTERVAL_MIN: int = 2
    CHASING_METEOROID_SPAWN_INTERVAL_MAX: int = 40
    CHASING_METEOROID_SPAWN_Y: float = 54.0
    CHASING_METEOROID_LANE_SPEED: float = 0.9
    CHASING_METEOROID_ACCEL: float = 0.045
    CHASING_METEOROID_LANE_ALIGN_THRESHOLD: float = 1.5
    CHASING_METEOROID_CYCLE_DX: Tuple[int, ...] = (2, 0, 1, 0, 2, 0, 2, 0)
    CHASING_METEOROID_CYCLE_DY: Tuple[int, ...] = (1, 0, 0, 0, 1, 0, 0, 0)
    MOTHERSHIP_OFFSCREEN_POS: int = 500
    MOTHERSHIP_ANIM_X: Tuple[int, int, int, int, int, int, int] = (9, 9, 10, 10, 11, 12, 12)
    MOTHERSHIP_HEIGHT: int = 7
    MOTHERSHIP_EMERGE_Y: int = 44
    REJUVENATOR_SPAWN_PROB: float = 1/4500
    REJUVENATOR_STAGE_2_Y: float = 62.0
    REJUVENATOR_STAGE_3_Y: float = 93.0
    REJUVENATOR_STAGE_4_Y: float = 112.0
    DEATH_DURATION: int = 120

    # Falling Rock constants
    FALLING_ROCK_MAX: int = 3
    FALLING_ROCK_SPAWN_PROB: float = 0.0065
    FALLING_ROCK_SPAWN_Y: float = 43.0
    FALLING_ROCK_BOTTOM_CLIP: float = 164.0
    FALLING_ROCK_INIT_VEL: float = 0.07
    FALLING_ROCK_ACCEL: float = 0.02

    # Lane blocker constants
    LANE_BLOCKER_MAX: int = 3
    LANE_BLOCKER_START_LEVEL: int = 10
    LANE_BLOCKER_SPAWN_PROB: float = 1 / 1800
    LANE_BLOCKER_SPAWN_Y: float = 43.0
    LANE_BLOCKER_BOTTOM_Y: float = 155.0
    LANE_BLOCKER_HOLD_FRAMES: int = 122
    LANE_BLOCKER_INIT_VEL: float = 0.02
    LANE_BLOCKER_SINK_INTERVAL: int = 2
    LANE_BLOCKER_WIDTH: int = 8
    LANE_BLOCKER_HEIGHT: int = 7
    LANE_BLOCKER_RETREAT_SPEED_MULT: float = 2.5

    # Kamikaze constants
    KAMIKAZE_MAX: int = 1
    KAMIKAZE_SPAWN_INTERVAL: int = 250
    KAMIKAZE_START_SECTOR: int = 12
    KAMIKAZE_START_Y: float = 43.0
    KAMIKAZE_TRACK_Y: float = 60.0
    KAMIKAZE_SPRITE_Y_THRESHOLDS: Tuple[float, float, float, float] = (43.0, 65.0, 72.0, 95.0)

    # Coin constants
    COIN_MAX: int = 3
    COIN_SPAWN_PROB: float = 1/300
    COIN_SPAWN_Y: float = 55.0
    COIN_EXIT_Y: float = 95.0
    COIN_SPAWN_X_LEFT: float = 5.0
    COIN_SPAWN_X_RIGHT: float = 155.0
    COIN_SPEED_Y: float = 0.5
    COIN_SPEED_X: float = 1.9375
    COIN_ANIM_SEQ: Tuple[int, ...] = (3, 2, 1, 0, 1, 2)
    COIN_SPRITE_SIZE: Tuple[int, int] = (7, 8)


def _get_index_ufo(pos: chex.Array) -> chex.Array:
    stage_1 = (pos >= 0).astype(jnp.int32)
    stage_2 = (pos >= 48).astype(jnp.int32)
    stage_3 = (pos >= 57).astype(jnp.int32)
    stage_4 = (pos >= 62).astype(jnp.int32)  # in reference game he chills there for a frame, only then switches
    stage_5 = (pos >= 69).astype(jnp.int32)  # in reference game he chills there for a frame, only then switches
    stage_6 = (pos >= 86).astype(jnp.int32)  # ab hier werden die schneller
    stage_7 = (pos >= 121).astype(jnp.int32)
    return stage_1 + stage_2 + stage_3 + stage_4 + stage_5 + stage_6 + stage_7


def _get_index_rejuvenator(pos: chex.Array) -> chex.Array:
    stage_1 = (pos >= 0).astype(jnp.int32)
    stage_2 = (pos >= 62).astype(jnp.int32)
    stage_3 = (pos >= 93).astype(jnp.int32)
    stage_4 = (pos >= 112).astype(jnp.int32)
    return stage_1 + stage_2 + stage_3 + stage_4


def _get_index_falling_rock(pos: chex.Array) -> chex.Array:
    stage_1 = (pos >= 43).astype(jnp.int32)
    stage_2 = (pos >= 64).astype(jnp.int32)
    stage_3 = (pos >= 85).astype(jnp.int32)
    stage_4 = (pos >= 111).astype(jnp.int32)
    return stage_1 + stage_2 + stage_3 + stage_4


def _get_index_lane_blocker(pos: chex.Array) -> chex.Array:
    stage_1 = (pos >= 43).astype(jnp.int32)
    stage_2 = (pos >= 64).astype(jnp.int32)
    stage_3 = (pos >= 84).astype(jnp.int32)
    stage_4 = (pos >= 111).astype(jnp.int32)
    return stage_1 + stage_2 + stage_3 + stage_4


def _get_index_kamikaze(pos: chex.Array) -> chex.Array:
    stage_1 = (pos >= 43).astype(jnp.int32)
    stage_2 = (pos >= 65).astype(jnp.int32)
    stage_3 = (pos >= 72).astype(jnp.int32)
    stage_4 = (pos >= 95).astype(jnp.int32)
    return stage_1 + stage_2 + stage_3 + stage_4


def _get_ufo_alignment(pos: chex.Array) -> chex.Array:
    stage_1 = (pos >= 0).astype(jnp.int32)
    stage_2 = (pos >= 48).astype(jnp.int32)
    stage_3 = (pos >= 57).astype(jnp.int32)
    stage_4 = (pos >= 62).astype(jnp.int32)
    stage_5 = (pos >= 69).astype(jnp.int32)
    stage_6 = (pos >= 86).astype(jnp.int32)
    stage_7 = (pos >= 121).astype(jnp.int32)
    return 4 - (stage_1 + stage_2 + stage_3 + stage_5 + stage_7)


def _get_index_bullet(pos: chex.Array, bullet_type: chex.Array, laser_id: int) -> chex.Array:
    is_laser = bullet_type == laser_id
    large = (pos >= 79).astype(jnp.int32)
    medium = (pos >= 56).astype(jnp.int32)
    torpedo_idx = jnp.where(large, 3, jnp.where(medium, 2, 1))
    return jnp.where(is_laser, 0, torpedo_idx)


def _get_bullet_alignment(pos: chex.Array, bullet_type: chex.Array, laser_id: int) -> chex.Array:
    is_laser = bullet_type == laser_id
    large = (pos >= 79).astype(jnp.int32)
    medium = (pos >= 56).astype(jnp.int32)
    torpedo_offset = jnp.where(large, 2, jnp.where(medium, 3, 4))
    return jnp.where(is_laser, 0, torpedo_offset)


def _get_player_shot_screen_x(
    player_shot_pos: chex.Array,
    player_shot_vel: chex.Array,
    bullet_type: chex.Array,
    laser_id: int,
) -> chex.Array:
    # Match ALE lane rounding: left lanes (vx < 0) ceil, right lanes (vx > 0) floor.
    shot_dir = jnp.sign(player_shot_vel[0])
    shot_x_base = player_shot_pos[0]
    shot_x_rounded = jnp.where(
        shot_dir < 0,
        jnp.ceil(shot_x_base),
        jnp.where(shot_dir > 0, jnp.floor(shot_x_base), jnp.round(shot_x_base)),
    )
    return shot_x_rounded + _get_bullet_alignment(player_shot_pos[1], bullet_type, laser_id)


class LevelState(NamedTuple):
    player_pos: chex.Array
    player_vel: chex.Array
    white_ufo_left: chex.Array
    mothership_position: chex.Array
    mothership_timer: chex.Array
    mothership_stage: chex.Array
    player_shot_pos: chex.Array
    player_shot_vel: chex.Array
    player_shot_frame: chex.Array
    torpedoes_left: chex.Array
    shooting_cooldown: chex.Array
    shooting_delay: chex.Array
    bullet_type: chex.Array
    shot_type_pending: chex.Array

    # enemies
    enemy_type: chex.Array
    white_ufo_pos: chex.Array
    white_ufo_vel: chex.Array
    enemy_shot_pos: chex.Array  # (2, 9) - 3 shots per ufo
    enemy_shot_vel: chex.Array  # (9,) - lane index
    enemy_shot_timer: chex.Array # (9,)
    enemy_shot_explosion_frame: chex.Array # (9,)
    enemy_shot_explosion_pos: chex.Array   # (2, 9)
    white_ufo_time_on_lane: chex.Array
    white_ufo_attack_time: chex.Array
    white_ufo_pattern_id: chex.Array
    white_ufo_pattern_timer: chex.Array
    ufo_explosion_frame: chex.Array
    ufo_explosion_pos: chex.Array
    chasing_meteoroid_explosion_frame: chex.Array
    chasing_meteoroid_explosion_pos: chex.Array
    chasing_meteoroid_pos: chex.Array
    chasing_meteoroid_active: chex.Array
    chasing_meteoroid_vel_y: chex.Array
    chasing_meteoroid_phase: chex.Array
    chasing_meteoroid_frame: chex.Array
    chasing_meteoroid_lane: chex.Array
    chasing_meteoroid_side: chex.Array
    chasing_meteoroid_spawn_timer: chex.Array
    chasing_meteoroid_remaining: chex.Array
    chasing_meteoroid_wave_active: chex.Array

    rejuvenator_pos: chex.Array
    rejuvenator_active: chex.Array
    rejuvenator_dead: chex.Array
    rejuvenator_frame: chex.Array
    rejuvenator_lane: chex.Array

    falling_rock_pos: chex.Array
    falling_rock_active: chex.Array
    falling_rock_vel_y: chex.Array
    falling_rock_lane: chex.Array
    falling_rock_explosion_frame: chex.Array
    falling_rock_explosion_pos: chex.Array

    lane_blocker_pos: chex.Array
    lane_blocker_active: chex.Array
    lane_blocker_vel_y: chex.Array
    lane_blocker_lane: chex.Array
    lane_blocker_phase: chex.Array
    lane_blocker_timer: chex.Array
    lane_blocker_explosion_frame: chex.Array
    lane_blocker_explosion_pos: chex.Array

    line_positions: chex.Array
    blue_line_counter: chex.Array
    
    death_timer: chex.Array

    # Bouncer dedicated fields
    bouncer_pos: chex.Array
    bouncer_vel: chex.Array
    bouncer_state: chex.Array
    bouncer_timer: chex.Array
    bouncer_active: chex.Array
    bouncer_lane: chex.Array
    bouncer_step_index: chex.Array
    bouncer_explosion_frame: chex.Array
    bouncer_explosion_pos: chex.Array

    coin_pos: chex.Array
    coin_active: chex.Array
    coin_timer: chex.Array
    coin_side: chex.Array
    coin_explosion_frame: chex.Array
    coin_explosion_pos: chex.Array
    coin_spawn_count: chex.Array

    kamikaze_pos: chex.Array
    kamikaze_active: chex.Array
    kamikaze_lane: chex.Array
    kamikaze_vel_y: chex.Array
    kamikaze_tracking: chex.Array
    kamikaze_spawn_timer: chex.Array
    kamikaze_explosion_frame: chex.Array
    kamikaze_explosion_pos: chex.Array


class BeamriderState(NamedTuple):
    level: LevelState
    score: chex.Array
    sector: chex.Array
    level_finished: chex.Array
    reset_coords: chex.Array
    lives: chex.Array
    steps: chex.Array
    rng: chex.Array


class BeamriderInfo(NamedTuple):
    score: chex.Array
    sector: chex.Array


class BeamriderObservation(NamedTuple):
    pos: chex.Array
    shooting_cd: chex.Array
    torpedoes_left: chex.Array
    player_shots_pos: chex.Array
    player_shots_vel: chex.Array
    white_ufo_left: chex.Array

    # enemies
    enemy_type: chex.Array
    white_ufo_pos: chex.Array
    white_ufo_vel: chex.Array
    enemy_shot_pos: chex.Array
    enemy_shot_vel: chex.Array

class WhiteUFOUpdate(NamedTuple):
    """Aggregated quantities needed after updating all white UFOs."""

    pos: chex.Array
    vel: chex.Array
    time_on_lane: chex.Array
    attack_time: chex.Array
    pattern_id: chex.Array
    pattern_timer: chex.Array


class JaxBeamrider(JaxEnvironment[BeamriderState, BeamriderObservation, BeamriderInfo, BeamriderConstants]):
    def __init__(self, consts: Optional[BeamriderConstants] = None):
        super().__init__(consts)
        self.consts = consts or BeamriderConstants()
        self.renderer = BeamriderRenderer(self.consts)
        self.obs_size = 111
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
            Action.DOWNLEFTFIRE,
        ]
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[BeamriderObservation, BeamriderState]:
        state = self.reset_level(12)
        observation = self._get_observation(state)
        return observation, state

    def _create_level_state(self, white_ufo_left=None, torpedoes_left=None, shooting_cooldown=None, shooting_delay=None, shot_type_pending=None, coin_spawn_count=None) -> LevelState:
        white_ufo_left = white_ufo_left if white_ufo_left is not None else jnp.array(self.consts.WHITE_UFOS_PER_SECTOR)
        torpedoes_left = torpedoes_left if torpedoes_left is not None else jnp.array(3)
        shooting_cooldown = shooting_cooldown if shooting_cooldown is not None else jnp.array(0)
        shooting_delay = shooting_delay if shooting_delay is not None else jnp.array(0)
        shot_type_pending = shot_type_pending if shot_type_pending is not None else jnp.array(self.consts.LASER_ID)
        coin_spawn_count = coin_spawn_count if coin_spawn_count is not None else jnp.array(0, dtype=jnp.int32)
        
        active_count = jnp.minimum(white_ufo_left.astype(jnp.int32), 3)
        active_mask = jnp.arange(3, dtype=jnp.int32) < active_count
        ufo_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 3),
        )
        initial_ufo_pos = jnp.array([[77.0, 77.0, 77.0], [43.0, 43.0, 43.0]])

        enemy_shot_offscreen = jnp.tile(
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 9),
        )

        return LevelState(
            player_pos=jnp.array(77.0),
            player_vel=jnp.array(0.0),
            white_ufo_left=white_ufo_left,
            mothership_position=jnp.array(self.consts.MOTHERSHIP_OFFSCREEN_POS, dtype=jnp.float32),
            mothership_timer=jnp.array(0, dtype=jnp.int32),
            mothership_stage=jnp.array(0, dtype=jnp.int32),
            player_shot_pos=jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32),
            player_shot_vel=jnp.zeros((2,), dtype=jnp.float32),
            player_shot_frame=jnp.array(-1, dtype=jnp.int32),
            torpedoes_left=torpedoes_left,
            shooting_cooldown=shooting_cooldown,
            shooting_delay=shooting_delay,
            bullet_type=jnp.array(self.consts.LASER_ID),
            shot_type_pending=shot_type_pending,
            enemy_type=jnp.array([0, 0, 0]),
            white_ufo_pos=jnp.where(active_mask[None, :], initial_ufo_pos, ufo_offscreen),
            white_ufo_vel=jnp.zeros((2, 3), dtype=jnp.float32),
            enemy_shot_pos=enemy_shot_offscreen,
            enemy_shot_vel=jnp.zeros((9,), dtype=jnp.int32),
            enemy_shot_timer=jnp.zeros((9,), dtype=jnp.int32),
            enemy_shot_explosion_frame=jnp.zeros((9,), dtype=jnp.int32),
            enemy_shot_explosion_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, 9),
            ),
            white_ufo_time_on_lane=jnp.array([0, 0, 0]),
            white_ufo_attack_time=jnp.zeros((3,), dtype=jnp.int32),
            white_ufo_pattern_id=jnp.zeros(3, dtype=jnp.int32),
            white_ufo_pattern_timer=jnp.zeros(3, dtype=jnp.int32),
            ufo_explosion_frame=jnp.zeros((3,), dtype=jnp.int32),
            ufo_explosion_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, 3),
            ),
            chasing_meteoroid_explosion_frame=jnp.zeros((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_explosion_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.CHASING_METEOROID_MAX),
            ),
            chasing_meteoroid_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.CHASING_METEOROID_MAX),
            ),
            chasing_meteoroid_active=jnp.zeros((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.bool_),
            chasing_meteoroid_vel_y=jnp.zeros((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.float32),
            chasing_meteoroid_phase=jnp.zeros((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_frame=jnp.zeros((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_lane=jnp.zeros((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_side=jnp.ones((self.consts.CHASING_METEOROID_MAX,), dtype=jnp.int32),
            chasing_meteoroid_spawn_timer=jnp.array(0, dtype=jnp.int32),
            chasing_meteoroid_remaining=jnp.array(0, dtype=jnp.int32),
            chasing_meteoroid_wave_active=jnp.array(False),
            rejuvenator_pos=jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32),
            rejuvenator_active=jnp.array(False),
            rejuvenator_dead=jnp.array(False),
            rejuvenator_frame=jnp.array(0, dtype=jnp.int32),
            rejuvenator_lane=jnp.array(0, dtype=jnp.int32),
            falling_rock_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.FALLING_ROCK_MAX),
            ),
            falling_rock_active=jnp.zeros((self.consts.FALLING_ROCK_MAX,), dtype=jnp.bool_),
            falling_rock_vel_y=jnp.zeros((self.consts.FALLING_ROCK_MAX,), dtype=jnp.float32),
            falling_rock_lane=jnp.zeros((self.consts.FALLING_ROCK_MAX,), dtype=jnp.int32),
            falling_rock_explosion_frame=jnp.zeros((self.consts.FALLING_ROCK_MAX,), dtype=jnp.int32),
            falling_rock_explosion_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.FALLING_ROCK_MAX),
            ),
            lane_blocker_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.LANE_BLOCKER_MAX),
            ),
            lane_blocker_active=jnp.zeros((self.consts.LANE_BLOCKER_MAX,), dtype=jnp.bool_),
            lane_blocker_vel_y=jnp.zeros((self.consts.LANE_BLOCKER_MAX,), dtype=jnp.float32),
            lane_blocker_lane=jnp.zeros((self.consts.LANE_BLOCKER_MAX,), dtype=jnp.int32),
            lane_blocker_phase=jnp.zeros((self.consts.LANE_BLOCKER_MAX,), dtype=jnp.int32),
            lane_blocker_timer=jnp.zeros((self.consts.LANE_BLOCKER_MAX,), dtype=jnp.int32),
            lane_blocker_explosion_frame=jnp.zeros((self.consts.LANE_BLOCKER_MAX,), dtype=jnp.int32),
            lane_blocker_explosion_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.LANE_BLOCKER_MAX),
            ),
            line_positions=BLUE_LINE_INIT_TABLE[0],
            blue_line_counter=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            bouncer_pos=jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32),
            bouncer_vel=jnp.zeros((2,), dtype=jnp.float32),
            bouncer_state=jnp.array(int(BouncerState.SWITCHING), dtype=jnp.int32),
            bouncer_timer=jnp.array(0, dtype=jnp.int32),
            bouncer_active=jnp.array(False),
            bouncer_lane=jnp.array(0, dtype=jnp.int32),
            bouncer_step_index=jnp.array(0, dtype=jnp.int32),
            bouncer_explosion_frame=jnp.array(0, dtype=jnp.int32),
            bouncer_explosion_pos=jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32),
            coin_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.COIN_MAX),
            ),
            coin_active=jnp.zeros((self.consts.COIN_MAX,), dtype=jnp.bool_),
            coin_timer=jnp.zeros((self.consts.COIN_MAX,), dtype=jnp.int32),
            coin_side=jnp.zeros((self.consts.COIN_MAX,), dtype=jnp.int32),
            coin_explosion_frame=jnp.zeros((self.consts.COIN_MAX,), dtype=jnp.int32),
            coin_explosion_pos=jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
                (1, self.consts.COIN_MAX),
            ),
            coin_spawn_count=coin_spawn_count,
            kamikaze_pos=jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            kamikaze_active=jnp.array([False]),
            kamikaze_lane=jnp.array([0], dtype=jnp.int32),
            kamikaze_vel_y=jnp.array([0.0], dtype=jnp.float32),
            kamikaze_tracking=jnp.array([False]),
            kamikaze_spawn_timer=jnp.array([0], dtype=jnp.int32),
            kamikaze_explosion_frame=jnp.array([0], dtype=jnp.int32),
            kamikaze_explosion_pos=jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
        )

    def reset_level(self, next_level=1) -> BeamriderState:
        level_state = self._create_level_state()
        
        return BeamriderState(
            level=level_state,
            score=jnp.array(0),
            sector=jnp.array(next_level),
            level_finished=jnp.array(0),
            reset_coords=jnp.array(False),
            lives=jnp.array(3),
            steps=jnp.array(0),
            rng=jnp.array(jax.random.key(42)),
        )
    

    def _get_observation(self, state: BeamriderState) -> BeamriderObservation:
        level = state.level
        is_init = level.blue_line_counter < len(BLUE_LINE_INIT_TABLE)
        
        ufo_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=level.white_ufo_pos.dtype).reshape(2, 1),
            (1, 3),
        )
        enemy_shot_offscreen = jnp.tile(
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=level.enemy_shot_pos.dtype).reshape(2, 1),
            (1, 9),
        )
        
        return BeamriderObservation(
            pos=level.player_pos,
            shooting_cd=level.shooting_cooldown,
            torpedoes_left=level.torpedoes_left,
            player_shots_pos=level.player_shot_pos,
            player_shots_vel=level.player_shot_vel,
            white_ufo_left=level.white_ufo_left,
            enemy_type=level.enemy_type,
            white_ufo_pos=jnp.where(is_init, ufo_offscreen, level.white_ufo_pos),
            white_ufo_vel=jnp.where(is_init, 0.0, level.white_ufo_vel),
            enemy_shot_pos=jnp.where(is_init, enemy_shot_offscreen, level.enemy_shot_pos),
            enemy_shot_vel=jnp.where(is_init, 0, level.enemy_shot_vel),
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(
        self,
        state: BeamriderState,
        action: chex.Array,
    ) -> Tuple[BeamriderObservation, BeamriderState, float, bool, BeamriderInfo]:
        
        # Map action index to semantic value
        action = jnp.take(jnp.array(self.action_set), action)

        # --- 1. Advance Blue Lines (Always happens) ---
        line_positions, blue_line_counter = self._line_step(state)
        
        # --- 2. Check for Initialization Phase ---
        # The game logic only starts after the BLUE_LINE_INIT_TABLE is exhausted.
        is_init = blue_line_counter < len(BLUE_LINE_INIT_TABLE)
        
        def handle_normal(_):
            (
                player_x,
                vel_x,
                player_shot_position,
                player_shot_velocity,
                player_shot_frame,
                torpedos_left,
                bullet_type,
                shooting_cooldown,
                shooting_delay,
                shot_type_pending,
            ) = self._player_step(state, action)

            rngs = jax.random.split(state.rng, 10)
            next_rng = rngs[0]
            ufo_keys = rngs[1:4]
            meteoroid_key = rngs[4]
            rejuvenator_key = rngs[5]
            falling_rock_key = rngs[6]
            lane_blocker_key = rngs[7]
            bouncer_key = rngs[8]
            coin_key = rngs[8]
            kamikaze_key = rngs[9]

            ufo_update = self._advance_white_ufos(state, ufo_keys)
            (
                white_ufo_pos,
                player_shot_position,
                white_ufo_pattern_id,
                white_ufo_pattern_timer,
                white_ufo_left,
                score,
                hit_mask_ufo,
                hit_exists_ufo,
            ) = self._collision_handler(
                state,
                ufo_update.pos,
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                ufo_update.pattern_id,
                ufo_update.pattern_timer,
            )
            (
                bouncer_pos,
                bouncer_vel,
                bouncer_state,
                bouncer_timer,
                bouncer_active,
                bouncer_lane,
                bouncer_step_index,
            ) = self._bouncer_dedicated_step(state, bouncer_key)

            # Bouncer collision check (Shot)
            # Torpedoes only
            bouncer_pos_screen = bouncer_pos[0] + _get_ufo_alignment(bouncer_pos[1])
            shot_x = _get_player_shot_screen_x(
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                self.consts.LASER_ID,
            )
            shot_y = player_shot_position[1]

            bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
            bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
            bouncer_size = jnp.array(self.consts.BOUNCER_SPRITE_SIZE)

            bouncer_hit = bouncer_active & \
                          (bouncer_pos_screen < shot_x + bullet_size[1]) & (shot_x < bouncer_pos_screen + bouncer_size[1]) & \
                          (bouncer_pos[1] < shot_y + bullet_size[0]) & (shot_y < bouncer_pos[1] + bouncer_size[0])

            # Destroy bouncer only if torpedo
            bullet_type_is_laser = bullet_type == self.consts.LASER_ID
            bouncer_destroyed = bouncer_hit & jnp.logical_not(bullet_type_is_laser)

            pre_collision_bouncer_pos = bouncer_pos
            bouncer_pos = jnp.where(bouncer_destroyed, jnp.array(self.consts.ENEMY_OFFSCREEN_POS), bouncer_pos)
            bouncer_active = jnp.where(bouncer_destroyed, False, bouncer_active)
            player_shot_position = jnp.where(
                bouncer_hit,
                jnp.array(self.consts.BULLET_OFFSCREEN_POS),
                player_shot_position,
            )

            score = jnp.where(bouncer_destroyed, score + 80, score)

            bouncer_explosion_frame, bouncer_explosion_pos = self._update_enemy_explosions(
                state.level.bouncer_explosion_frame[None],
                state.level.bouncer_explosion_pos[:, None],
                bouncer_destroyed[None],
                pre_collision_bouncer_pos[:, None],
            )
            bouncer_explosion_frame = bouncer_explosion_frame[0]
            bouncer_explosion_pos = bouncer_explosion_pos[:, 0]

            (
                chasing_meteoroid_pos,
                chasing_meteoroid_active,
                chasing_meteoroid_vel_y,
                chasing_meteoroid_phase,
                chasing_meteoroid_frame,
                chasing_meteoroid_lane,
                chasing_meteoroid_side,
                chasing_meteoroid_spawn_timer,
                chasing_meteoroid_remaining,
                chasing_meteoroid_wave_active,
            ) = self._chasing_meteoroid_step(state, player_x, vel_x, white_ufo_left, meteoroid_key)
            pre_collision_meteoroid_pos = chasing_meteoroid_pos
            (
                chasing_meteoroid_pos,
                chasing_meteoroid_active,
                chasing_meteoroid_vel_y,
                chasing_meteoroid_phase,
                chasing_meteoroid_frame,
                chasing_meteoroid_lane,
                chasing_meteoroid_side,
                player_shot_position,
                chasing_meteoroid_hit_mask,
                hit_exists_meteoroid,
                collision_exists,
            ) = self._chasing_meteoroid_bullet_collision(
                chasing_meteoroid_pos,
                chasing_meteoroid_active,
                chasing_meteoroid_vel_y,
                chasing_meteoroid_phase,
                chasing_meteoroid_frame,
                chasing_meteoroid_lane,
                chasing_meteoroid_side,
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                white_ufo_left,
            )
            
            (
                falling_rock_pos,
                falling_rock_active,
                falling_rock_lane,
                falling_rock_vel_y,
            ) = self._falling_rock_step(state, falling_rock_key)
            pre_collision_rock_pos = falling_rock_pos
            (
                falling_rock_pos,
                falling_rock_active,
                player_shot_position,
                falling_rock_hit_mask,
                hit_exists_rock,
            ) = self._falling_rock_bullet_collision(
                falling_rock_pos,
                falling_rock_active,
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                white_ufo_left,
            )

            (
                lane_blocker_pos,
                lane_blocker_active,
                lane_blocker_lane,
                lane_blocker_vel_y,
                lane_blocker_phase,
                lane_blocker_timer,
            ) = self._lane_blocker_step(state, lane_blocker_key)
            pre_collision_lane_blocker_pos = lane_blocker_pos
            (
                lane_blocker_pos,
                lane_blocker_active,
                lane_blocker_phase,
                lane_blocker_timer,
                lane_blocker_vel_y,
                player_shot_position,
                lane_blocker_hit_mask,
                hit_exists_lane_blocker,
            ) = self._lane_blocker_bullet_collision(
                lane_blocker_pos,
                lane_blocker_active,
                lane_blocker_phase,
                lane_blocker_timer,
                lane_blocker_vel_y,
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                white_ufo_left,
            )

            (
                kamikaze_pos,
                kamikaze_active,
                kamikaze_lane,
                kamikaze_vel_y,
                kamikaze_tracking,
                kamikaze_spawn_timer,
            ) = self._kamikaze_step(state, kamikaze_key)
            pre_collision_kamikaze_pos = kamikaze_pos
            (
                kamikaze_pos,
                kamikaze_active,
                player_shot_position,
                kamikaze_destroyed,
                hit_exists_kamikaze,
            ) = self._kamikaze_bullet_collision(
                kamikaze_pos,
                kamikaze_active,
                player_shot_position,
                player_shot_velocity,
                bullet_type,
            )
            
            (
                coin_pos,
                coin_active,
                coin_timer,
                coin_side,
                coin_spawn_count,
            ) = self._coin_step(state, coin_key)
            pre_collision_coin_pos = coin_pos
            
            (
                hit_mask_coin,
                hit_exists_coin,
            ) = self._coin_bullet_collision(
                coin_pos,
                coin_active,
                player_shot_position,
                bullet_type,
            )
            
            coin_pos = jnp.where(hit_mask_coin[None, :], jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=coin_pos.dtype)[:, None], coin_pos)
            coin_active = jnp.where(hit_mask_coin, False, coin_active)
            player_shot_position = jnp.where(hit_exists_coin, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)

            clamped_sector = jnp.minimum(state.sector, 89)
            ms_score_val = 300 + 30 * clamped_sector
            hp_bonus_per_life_val = 100 + 10 * clamped_sector
            hp_bonus = jnp.maximum(state.lives - 1, 0) * hp_bonus_per_life_val
            
            score = jnp.where(hit_exists_coin, score + ms_score_val + hp_bonus, score)

            (
                rejuv_pos,
                rejuv_active,
                rejuv_dead,
                rejuv_frame,
                rejuv_lane,
            ) = self._rejuvenator_step(state, rejuvenator_key)
            
            # Rejuvenator-Shot collision
            rejuv_x_screen = rejuv_pos[0] + _get_ufo_alignment(rejuv_pos[1])
            rejuv_y = rejuv_pos[1]
            shot_x_screen = _get_player_shot_screen_x(
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                self.consts.LASER_ID,
            )
            shot_y = player_shot_position[1]
            
            bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
            bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)

            rejuv_indices = jnp.where(rejuv_dead, 4, jnp.clip(_get_index_rejuvenator(rejuv_y) - 1, 0, 3))
            rejuv_sizes = jnp.take(jnp.array(self.consts.REJUVENATOR_SPRITE_SIZES), rejuv_indices, axis=0)

            rejuv_hit_by_shot = jnp.logical_and.reduce(jnp.array([
                rejuv_active,
                jnp.logical_not(rejuv_dead),
                (rejuv_x_screen < shot_x_screen + bullet_size[1]) & (shot_x_screen < rejuv_x_screen + rejuv_sizes[1]),
                (rejuv_y < shot_y + bullet_size[0]) & (shot_y < rejuv_y + rejuv_sizes[0]),
                shot_y < self.consts.BOTTOM_CLIP
            ]))
            
            rejuv_dead = jnp.logical_or(rejuv_dead, rejuv_hit_by_shot)
            # If the rejuvenator was just killed by a shot, we don't reset its position yet,
            # because it should drop down as a "Dead" sprite.
            # But if it hit the player, it's already reset in the later collision logic.
            
            player_shot_position = jnp.where(rejuv_hit_by_shot, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)

            # --- Mothership Collision Check ---
            ms_stage = state.level.mothership_stage
            ms_pos = state.level.mothership_position
            ms_y = self.consts.MOTHERSHIP_EMERGE_Y - self.consts.MOTHERSHIP_HEIGHT
            
            ms_size = jnp.array(self.consts.MOTHERSHIP_SPRITE_SIZE)
            
            shot_x = _get_player_shot_screen_x(
                player_shot_position,
                player_shot_velocity,
                bullet_type,
                self.consts.LASER_ID,
            )
            shot_y = player_shot_position[1]
            
            bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
            bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)

            shot_active = shot_y < self.consts.BOTTOM_CLIP 
            is_torpedo = bullet_type == self.consts.TORPEDO_ID
            ms_vulnerable = ms_stage == 2
            
            ms_square = jnp.max(ms_size)
            hit_mothership = (ms_pos < shot_x + bullet_size[1]) & (shot_x < ms_pos + ms_square) & \
                             (ms_y < shot_y + bullet_size[0]) & (shot_y < ms_y + ms_square + 3) & \
                             shot_active & is_torpedo & ms_vulnerable
            player_shot_position = jnp.where(hit_mothership, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)
            
            enemy_shot_pos, enemy_shot_lane, enemy_shot_timer, shot_hit_count = self._enemy_shot_step(
                state,
                white_ufo_pos,
                white_ufo_pattern_id,
                white_ufo_pattern_timer,
            )

            # --- Enemy Shot Collision Check (Torpedoes only) ---
            hit_mask_shot, hit_exists_shot = self._enemy_shot_bullet_collision(
                enemy_shot_pos, enemy_shot_timer, player_shot_position, player_shot_velocity, bullet_type
            )
            enemy_shot_pos_pre_collision = enemy_shot_pos
            enemy_shot_offscreen = jnp.tile(jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=enemy_shot_pos.dtype).reshape(2, 1), (1, 9))
            enemy_shot_pos = jnp.where(hit_mask_shot[None, :], enemy_shot_offscreen, enemy_shot_pos)
            enemy_shot_timer = jnp.where(hit_mask_shot, 0, enemy_shot_timer)
            player_shot_position = jnp.where(hit_exists_shot, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)

            # Check if projectile was resolved this frame
            projectile_at_horizon = self._projectile_resolved(state)
            
            # Any collision that destroys the bullet
            bullet_hit_any = hit_exists_ufo | bouncer_hit | hit_exists_meteoroid | hit_exists_rock | hit_exists_lane_blocker | rejuv_hit_by_shot | hit_mothership | hit_exists_shot | hit_exists_coin
            
            projectile_resolved_now = projectile_at_horizon | bullet_hit_any

            player_shot_frame = jnp.where(
                projectile_resolved_now,
                jnp.array(-1, dtype=player_shot_frame.dtype),
                player_shot_frame,
            )
            
            # If resolved, trigger recovery window
            shooting_cooldown = jnp.where(
                projectile_resolved_now,
                self.consts.PLAYER_SHOT_RECOVERY,
                shooting_cooldown
            )

            ufo_explosion_frame, ufo_explosion_pos = self._update_enemy_explosions(
                state.level.ufo_explosion_frame,
                state.level.ufo_explosion_pos,
                hit_mask_ufo,
                ufo_update.pos,
            )
            chasing_meteoroid_explosion_frame, chasing_meteoroid_explosion_pos = self._update_enemy_explosions(
                state.level.chasing_meteoroid_explosion_frame,
                state.level.chasing_meteoroid_explosion_pos,
                chasing_meteoroid_hit_mask,
                pre_collision_meteoroid_pos,
            )
            falling_rock_explosion_frame, falling_rock_explosion_pos = self._update_enemy_explosions(
                state.level.falling_rock_explosion_frame,
                state.level.falling_rock_explosion_pos,
                falling_rock_hit_mask,
                pre_collision_rock_pos,
            )
            lane_blocker_explosion_frame, lane_blocker_explosion_pos = self._update_enemy_explosions(
                state.level.lane_blocker_explosion_frame,
                state.level.lane_blocker_explosion_pos,
                lane_blocker_hit_mask,
                pre_collision_lane_blocker_pos,
            )
            enemy_shot_explosion_frame, enemy_shot_explosion_pos = self._update_enemy_explosions(
                state.level.enemy_shot_explosion_frame,
                state.level.enemy_shot_explosion_pos,
                hit_mask_shot,
                enemy_shot_pos_pre_collision,
            )
            coin_explosion_frame, coin_explosion_pos = self._update_enemy_explosions(
                state.level.coin_explosion_frame,
                state.level.coin_explosion_pos,
                hit_mask_coin,
                pre_collision_coin_pos,
            )
            kamikaze_explosion_frame, kamikaze_explosion_pos = self._update_enemy_explosions(
                state.level.kamikaze_explosion_frame,
                state.level.kamikaze_explosion_pos,
                kamikaze_destroyed,
                pre_collision_kamikaze_pos,
            )

            # Player-UFO collision check
            ufo_x = white_ufo_pos[0] + _get_ufo_alignment(white_ufo_pos[1])
            ufo_y = white_ufo_pos[1]
            player_x_topleft = player_x
            player_y_topleft = float(self.consts.PLAYER_POS_Y)
            player_size = jnp.array(self.consts.PLAYER_SPRITE_SIZE)

            ufo_indices = jnp.clip(_get_index_ufo(ufo_y) - 1, 0, len(self.consts.UFO_SPRITE_SIZES) - 1)
            ufo_sizes = jnp.take(jnp.array(self.consts.UFO_SPRITE_SIZES), ufo_indices, axis=0)

            ufo_hits = (ufo_x < player_x_topleft + player_size[1]) & (player_x_topleft < ufo_x + ufo_sizes[:, 1]) & \
                       (ufo_y < player_y_topleft + player_size[0]) & (player_y_topleft < ufo_y + ufo_sizes[:, 0])

            ufo_hit_count = jnp.sum(ufo_hits.astype(jnp.int32))
            
            # Player-Bouncer collision check
            bouncer_size = jnp.array(self.consts.BOUNCER_SPRITE_SIZE)
            bouncer_hits = bouncer_active & \
                           (bouncer_pos_screen < player_x_topleft + player_size[1]) & (player_x_topleft < bouncer_pos_screen + bouncer_size[1]) & \
                           (bouncer_pos[1] < player_y_topleft + player_size[0]) & (player_y_topleft < bouncer_pos[1] + bouncer_size[0])

            bouncer_hit_count = jnp.sum(bouncer_hits.astype(jnp.int32))
            bouncer_active = jnp.where(bouncer_hits, False, bouncer_active)
            bouncer_pos = jnp.where(
                bouncer_hits,
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=bouncer_pos.dtype),
                bouncer_pos,
            )

            chasing_meteoroid_x = chasing_meteoroid_pos[0] + _get_ufo_alignment(chasing_meteoroid_pos[1]).astype(chasing_meteoroid_pos.dtype)
            chasing_meteoroid_y = chasing_meteoroid_pos[1]
            meteoroid_size = jnp.array(self.consts.METEOROID_SPRITE_SIZE)

            chasing_meteoroid_hits = chasing_meteoroid_active & \
                                     (chasing_meteoroid_x < player_x_topleft + player_size[1]) & (player_x_topleft < chasing_meteoroid_x + meteoroid_size[1]) & \
                                     (chasing_meteoroid_y < player_y_topleft + player_size[0]) & (player_y_topleft < chasing_meteoroid_y + meteoroid_size[0])

            chasing_meteoroid_hit_count = jnp.sum(chasing_meteoroid_hits.astype(jnp.int32))
            
            rejuv_indices = jnp.where(rejuv_dead, 4, jnp.clip(_get_index_rejuvenator(rejuv_y) - 1, 0, 3))
            rejuv_sizes = jnp.take(jnp.array(self.consts.REJUVENATOR_SPRITE_SIZES), rejuv_indices, axis=0)

            rejuv_hit_player = rejuv_active & \
                               (rejuv_x_screen < player_x_topleft + player_size[1]) & (player_x_topleft < rejuv_x_screen + rejuv_sizes[1]) & \
                               (rejuv_y < player_y_topleft + player_size[0]) & (player_y_topleft < rejuv_y + rejuv_sizes[0])
            
            gain_life = jnp.logical_and(rejuv_hit_player, jnp.logical_not(rejuv_dead))
            lose_life_rejuv = jnp.logical_and(rejuv_hit_player, rejuv_dead)
            
            rejuv_active = jnp.where(rejuv_hit_player, False, rejuv_active)
            rejuv_pos = jnp.where(rejuv_hit_player, jnp.array(self.consts.ENEMY_OFFSCREEN_POS), rejuv_pos)

            rock_x = falling_rock_pos[0] + _get_ufo_alignment(falling_rock_pos[1]).astype(falling_rock_pos.dtype)
            rock_y = falling_rock_pos[1]
            rock_indices = jnp.clip(_get_index_falling_rock(rock_y) - 1, 0, len(self.consts.FALLING_ROCK_SPRITE_SIZES) - 1)
            rock_sizes = jnp.take(jnp.array(self.consts.FALLING_ROCK_SPRITE_SIZES), rock_indices, axis=0)
            
            rock_hits = falling_rock_active & \
                        (rock_x < player_x_topleft + player_size[1]) & (player_x_topleft < rock_x + rock_sizes[:, 1]) & \
                        (rock_y < player_y_topleft + player_size[0]) & (player_y_topleft < rock_y + rock_sizes[:, 0])

            rock_hit_count = jnp.sum(rock_hits.astype(jnp.int32))

            bottom_lanes = jnp.array(self.consts.BOTTOM_OF_LANES, dtype=player_x_topleft.dtype)
            safe_lane_idx = jnp.clip(lane_blocker_lane, 1, 5) - 1
            lane_blocker_lane_x = bottom_lanes[safe_lane_idx]
            lane_blocker_on_bottom = lane_blocker_active & (lane_blocker_pos[1] >= self.consts.LANE_BLOCKER_BOTTOM_Y)
            lane_blocker_on_bottom = lane_blocker_on_bottom & (lane_blocker_phase != int(LaneBlockerState.RETREAT))
            lane_blocker_hits = lane_blocker_on_bottom & (player_x_topleft == lane_blocker_lane_x)
            lane_blocker_hit_count = jnp.sum(lane_blocker_hits.astype(jnp.int32))

            # Player-Kamikaze collision check
            kamikaze_x_col = kamikaze_pos[0, 0] + _get_ufo_alignment(kamikaze_pos[1, 0])
            kamikaze_y_col = kamikaze_pos[1, 0]
            
            kamikaze_indices_col = jnp.clip(_get_index_kamikaze(kamikaze_y_col) - 1, 0, 3)
            kamikaze_sizes_col = jnp.take(jnp.array(self.consts.LANE_BLOCKER_SPRITE_SIZES), kamikaze_indices_col, axis=0)
            
            kamikaze_hits_player = kamikaze_active[0] & \
                            (kamikaze_x_col < player_x_topleft + player_size[1]) & (player_x_topleft < kamikaze_x_col + kamikaze_sizes_col[1]) & \
                            (kamikaze_y_col < player_y_topleft + player_size[0]) & (player_y_topleft < kamikaze_y_col + kamikaze_sizes_col[0])
            
            kamikaze_hit_count = kamikaze_hits_player.astype(jnp.int32)
            kamikaze_active = jnp.where(kamikaze_hits_player, jnp.array([False]), kamikaze_active)
            kamikaze_pos = jnp.where(
                kamikaze_hits_player,
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=kamikaze_pos.dtype).reshape(2, 1),
                kamikaze_pos,
            )

            chasing_meteoroid_offscreen = jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=chasing_meteoroid_pos.dtype).reshape(2, 1),
                (1, self.consts.CHASING_METEOROID_MAX),
            )
            chasing_meteoroid_active = jnp.where(chasing_meteoroid_hits, False, chasing_meteoroid_active)
            chasing_meteoroid_pos = jnp.where(chasing_meteoroid_hits[None, :], chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_vel_y = jnp.where(chasing_meteoroid_hits, 0.0, chasing_meteoroid_vel_y)
            chasing_meteoroid_phase = jnp.where(chasing_meteoroid_hits, 0, chasing_meteoroid_phase)
            chasing_meteoroid_frame = jnp.where(chasing_meteoroid_hits, 0, chasing_meteoroid_frame)
            chasing_meteoroid_lane = jnp.where(chasing_meteoroid_hits, 0, chasing_meteoroid_lane)
            chasing_meteoroid_side = jnp.where(chasing_meteoroid_hits, 1, chasing_meteoroid_side)
            reached_player = jnp.logical_and(
                chasing_meteoroid_active,
                chasing_meteoroid_pos[1] >= player_y_topleft,
            )
            chasing_meteoroid_active = jnp.where(reached_player, False, chasing_meteoroid_active)
            chasing_meteoroid_pos = jnp.where(reached_player[None, :], chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_vel_y = jnp.where(reached_player, 0.0, chasing_meteoroid_vel_y)
            chasing_meteoroid_phase = jnp.where(reached_player, 0, chasing_meteoroid_phase)
            chasing_meteoroid_frame = jnp.where(reached_player, 0, chasing_meteoroid_frame)
            chasing_meteoroid_lane = jnp.where(reached_player, 0, chasing_meteoroid_lane)
            chasing_meteoroid_side = jnp.where(reached_player, 1, chasing_meteoroid_side)

            hit_count = (
                shot_hit_count
                + ufo_hit_count
                + chasing_meteoroid_hit_count
                + bouncer_hit_count
                + lose_life_rejuv.astype(jnp.int32)
                + rock_hit_count
                + lane_blocker_hit_count
                + kamikaze_hit_count
            )

            current_death_timer = state.level.death_timer
            is_hit = hit_count > 0
            start_dying = jnp.logical_and(is_hit, current_death_timer == 0)
            next_death_timer = jnp.where(start_dying, self.consts.DEATH_DURATION, jnp.maximum(current_death_timer - 1, 0))
            is_dying_sequence = next_death_timer > 0
            just_died = jnp.logical_and(current_death_timer > 0, next_death_timer == 0)

            mothership_position, mothership_timer, mothership_stage, sector_advanced_m = self._mothership_step(
                state, white_ufo_left, ufo_explosion_frame, hit_mothership
            )

            clamped_sector = jnp.minimum(state.sector, 89)
            ms_score_val = 300 + 30 * clamped_sector
            hp_bonus_per_life_val = 100 + 10 * clamped_sector
            hp_bonus = jnp.maximum(state.lives - 1, 0) * hp_bonus_per_life_val
            score = jnp.where(hit_mothership, score + ms_score_val + hp_bonus, score)

            died_after_clearing_ufos = jnp.logical_and(just_died, white_ufo_left == 0)
            sector_advanced = jnp.logical_or(died_after_clearing_ufos, sector_advanced_m)
            sector = state.sector + sector_advanced.astype(jnp.int32)

            white_ufo_left = jnp.where(sector_advanced, self.consts.WHITE_UFOS_PER_SECTOR, white_ufo_left)
            torpedos_left = jnp.where(sector_advanced, 3, torpedos_left)

            white_ufo_vel = jnp.where(hit_mask_ufo[None, :], 0.0, ufo_update.vel)
            white_ufo_time_on_lane = jnp.where(hit_mask_ufo, 0, ufo_update.time_on_lane)
            white_ufo_attack_time = jnp.where(hit_mask_ufo, 0, ufo_update.attack_time)

            enemy_shot_offscreen = jnp.tile(jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=enemy_shot_pos.dtype).reshape(2, 1), (1, 9))
            # Removing masking of enemy shots by active_mask to allow shots to persist after UFO death
            # enemy_shot_pos = jnp.where(active_mask[None, :], enemy_shot_pos, enemy_shot_offscreen)
            # enemy_shot_timer = jnp.where(active_mask, enemy_shot_timer, 0)
            # enemy_shot_lane = jnp.where(active_mask, enemy_shot_lane, 0)

            enemy_shot_pos = jnp.where(sector_advanced, enemy_shot_offscreen, enemy_shot_pos)
            enemy_shot_timer = jnp.where(sector_advanced, 0, enemy_shot_timer)
            enemy_shot_lane = jnp.where(sector_advanced, 0, enemy_shot_lane)
            chasing_meteoroid_pos = jnp.where(sector_advanced, chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_active = jnp.where(sector_advanced, False, chasing_meteoroid_active)
            chasing_meteoroid_vel_y = jnp.where(sector_advanced, 0.0, chasing_meteoroid_vel_y)
            chasing_meteoroid_phase = jnp.where(sector_advanced, 0, chasing_meteoroid_phase)
            chasing_meteoroid_frame = jnp.where(sector_advanced, 0, chasing_meteoroid_frame)
            chasing_meteoroid_lane = jnp.where(sector_advanced, 0, chasing_meteoroid_lane)
            chasing_meteoroid_side = jnp.where(sector_advanced, 1, chasing_meteoroid_side)
            chasing_meteoroid_spawn_timer = jnp.where(sector_advanced, 0, chasing_meteoroid_spawn_timer)
            chasing_meteoroid_remaining = jnp.where(sector_advanced, 0, chasing_meteoroid_remaining)
            chasing_meteoroid_wave_active = jnp.where(sector_advanced, False, chasing_meteoroid_wave_active)
            
            rejuv_pos = jnp.where(sector_advanced, jnp.array(self.consts.ENEMY_OFFSCREEN_POS), rejuv_pos)
            rejuv_active = jnp.where(sector_advanced, False, rejuv_active)
            rejuv_dead = jnp.where(sector_advanced, False, rejuv_dead)
            rejuv_frame = jnp.where(sector_advanced, 0, rejuv_frame)
            rejuv_lane = jnp.where(sector_advanced, 0, rejuv_lane)

            falling_rock_offscreen = jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=falling_rock_pos.dtype).reshape(2, 1),
                (1, self.consts.FALLING_ROCK_MAX),
            )
            falling_rock_pos = jnp.where(sector_advanced, falling_rock_offscreen, falling_rock_pos)
            falling_rock_active = jnp.where(sector_advanced, False, falling_rock_active)
            falling_rock_lane = jnp.where(sector_advanced, 0, falling_rock_lane)
            falling_rock_vel_y = jnp.where(sector_advanced, 0.0, falling_rock_vel_y)

            lane_blocker_offscreen = jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=lane_blocker_pos.dtype).reshape(2, 1),
                (1, self.consts.LANE_BLOCKER_MAX),
            )
            lane_blocker_pos = jnp.where(sector_advanced, lane_blocker_offscreen, lane_blocker_pos)
            lane_blocker_active = jnp.where(sector_advanced, False, lane_blocker_active)
            lane_blocker_lane = jnp.where(sector_advanced, 0, lane_blocker_lane)
            lane_blocker_vel_y = jnp.where(sector_advanced, 0.0, lane_blocker_vel_y)
            lane_blocker_phase = jnp.where(sector_advanced, 0, lane_blocker_phase)
            lane_blocker_timer = jnp.where(sector_advanced, 0, lane_blocker_timer)

            kamikaze_pos = jnp.where(sector_advanced, jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=kamikaze_pos.dtype).reshape(2, 1), kamikaze_pos)
            kamikaze_active = jnp.where(sector_advanced, jnp.array([False]), kamikaze_active)
            kamikaze_lane = jnp.where(sector_advanced, 0, kamikaze_lane)
            kamikaze_vel_y = jnp.where(sector_advanced, 0.0, kamikaze_vel_y)
            kamikaze_tracking = jnp.where(sector_advanced, jnp.array([False]), kamikaze_tracking)
            kamikaze_spawn_timer = jnp.where(sector_advanced, 0, kamikaze_spawn_timer)

            coin_offscreen = jnp.tile(jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=coin_pos.dtype).reshape(2, 1), (1, self.consts.COIN_MAX))
            coin_pos = jnp.where(sector_advanced, coin_offscreen, coin_pos)
            coin_active = jnp.where(sector_advanced, False, coin_active)
            coin_timer = jnp.where(sector_advanced, 0, coin_timer)
            coin_side = jnp.where(sector_advanced, 0, coin_side)
            coin_explosion_frame = jnp.where(sector_advanced, jnp.zeros_like(coin_explosion_frame), coin_explosion_frame)
            coin_explosion_pos = jnp.where(sector_advanced, coin_offscreen, coin_explosion_pos)
            coin_spawn_count = jnp.where(sector_advanced, 0, coin_spawn_count)

            ufo_offscreen = jnp.tile(jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=white_ufo_pos.dtype).reshape(2, 1), (1, 3))
            white_ufo_pos = jnp.where(is_dying_sequence, ufo_offscreen, white_ufo_pos)
            enemy_shot_pos = jnp.where(is_dying_sequence, enemy_shot_offscreen, enemy_shot_pos)
            chasing_meteoroid_pos = jnp.where(is_dying_sequence, chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_active = jnp.where(is_dying_sequence, False, chasing_meteoroid_active)
            chasing_meteoroid_vel_y = jnp.where(is_dying_sequence, 0.0, chasing_meteoroid_vel_y)
            chasing_meteoroid_phase = jnp.where(is_dying_sequence, 0, chasing_meteoroid_phase)
            chasing_meteoroid_frame = jnp.where(is_dying_sequence, 0, chasing_meteoroid_frame)
            chasing_meteoroid_lane = jnp.where(is_dying_sequence, 0, chasing_meteoroid_lane)
            chasing_meteoroid_side = jnp.where(is_dying_sequence, 1, chasing_meteoroid_side)
            chasing_meteoroid_spawn_timer = jnp.where(is_dying_sequence, 0, chasing_meteoroid_spawn_timer)
            chasing_meteoroid_remaining = jnp.where(is_dying_sequence, 0, chasing_meteoroid_remaining)
            chasing_meteoroid_wave_active = jnp.where(is_dying_sequence, False, chasing_meteoroid_wave_active)
            
            rejuv_pos = jnp.where(is_dying_sequence, jnp.array(self.consts.ENEMY_OFFSCREEN_POS), rejuv_pos)
            rejuv_active = jnp.where(is_dying_sequence, False, rejuv_active)
            rejuv_dead = jnp.where(is_dying_sequence, False, rejuv_dead)
            rejuv_frame = jnp.where(is_dying_sequence, 0, rejuv_frame)
            rejuv_lane = jnp.where(is_dying_sequence, 0, rejuv_lane)
            
            falling_rock_pos = jnp.where(is_dying_sequence, falling_rock_offscreen, falling_rock_pos)
            falling_rock_active = jnp.where(is_dying_sequence, False, falling_rock_active)
            falling_rock_lane = jnp.where(is_dying_sequence, 0, falling_rock_lane)
            falling_rock_vel_y = jnp.where(is_dying_sequence, 0.0, falling_rock_vel_y)

            lane_blocker_pos = jnp.where(is_dying_sequence, lane_blocker_offscreen, lane_blocker_pos)
            lane_blocker_active = jnp.where(is_dying_sequence, False, lane_blocker_active)
            lane_blocker_lane = jnp.where(is_dying_sequence, 0, lane_blocker_lane)
            lane_blocker_vel_y = jnp.where(is_dying_sequence, 0.0, lane_blocker_vel_y)
            lane_blocker_phase = jnp.where(is_dying_sequence, 0, lane_blocker_phase)
            lane_blocker_timer = jnp.where(is_dying_sequence, 0, lane_blocker_timer)

            kamikaze_pos = jnp.where(is_dying_sequence, jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=kamikaze_pos.dtype).reshape(2, 1), kamikaze_pos)
            kamikaze_active = jnp.where(is_dying_sequence, jnp.array([False]), kamikaze_active)
            kamikaze_lane = jnp.where(is_dying_sequence, 0, kamikaze_lane)
            kamikaze_vel_y = jnp.where(is_dying_sequence, 0.0, kamikaze_vel_y)
            kamikaze_tracking = jnp.where(is_dying_sequence, jnp.array([False]), kamikaze_tracking)
            kamikaze_spawn_timer = jnp.where(is_dying_sequence, 0, kamikaze_spawn_timer)

            coin_pos = jnp.where(is_dying_sequence, coin_offscreen, coin_pos)
            coin_active = jnp.where(is_dying_sequence, False, coin_active)
            coin_timer = jnp.where(is_dying_sequence, 0, coin_timer)
            coin_side = jnp.where(is_dying_sequence, 0, coin_side)
            coin_explosion_frame = jnp.where(is_dying_sequence, jnp.zeros_like(coin_explosion_frame), coin_explosion_frame)
            coin_explosion_pos = jnp.where(is_dying_sequence, coin_offscreen, coin_explosion_pos)
            coin_spawn_count = jnp.where(is_dying_sequence, 0, coin_spawn_count)

            mothership_position = jnp.where(is_dying_sequence, self.consts.MOTHERSHIP_OFFSCREEN_POS, mothership_position)
            mothership_timer = jnp.where(is_dying_sequence, 0, mothership_timer)
            mothership_stage = jnp.where(is_dying_sequence, 0, mothership_stage)
            vel_x = jnp.where(is_dying_sequence, 0.0, vel_x)
            player_shot_position = jnp.where(is_dying_sequence, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)
            player_shot_frame = jnp.where(
                is_dying_sequence,
                jnp.array(-1, dtype=player_shot_frame.dtype),
                player_shot_frame,
            )
            bouncer_pos = jnp.where(
                is_dying_sequence,
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=bouncer_pos.dtype),
                bouncer_pos,
            )
            bouncer_active = jnp.where(is_dying_sequence, False, bouncer_active)

            next_step = state.steps + 1
            new_level_state = LevelState(
                player_pos=player_x,
                player_vel=vel_x,
                white_ufo_left=white_ufo_left,
                mothership_position=mothership_position,
                mothership_timer=mothership_timer,
                mothership_stage=mothership_stage,
                player_shot_pos=player_shot_position,
                player_shot_vel=player_shot_velocity,
                player_shot_frame=player_shot_frame,
                torpedoes_left=torpedos_left,
                shooting_cooldown=shooting_cooldown,
                shooting_delay=shooting_delay,
                bullet_type=bullet_type,
                shot_type_pending=shot_type_pending,
                enemy_type=jnp.array([0, 0, 0]),
                white_ufo_pos=white_ufo_pos,
                white_ufo_vel=white_ufo_vel,
                enemy_shot_pos=enemy_shot_pos,
                enemy_shot_vel=enemy_shot_lane,
                enemy_shot_timer=enemy_shot_timer,
                enemy_shot_explosion_frame=enemy_shot_explosion_frame,
                enemy_shot_explosion_pos=enemy_shot_explosion_pos,
                white_ufo_time_on_lane=white_ufo_time_on_lane,
                white_ufo_attack_time=white_ufo_attack_time,
                white_ufo_pattern_id=white_ufo_pattern_id,
                white_ufo_pattern_timer=white_ufo_pattern_timer,
                ufo_explosion_frame=ufo_explosion_frame,
                ufo_explosion_pos=ufo_explosion_pos,
                chasing_meteoroid_explosion_frame=chasing_meteoroid_explosion_frame,
                chasing_meteoroid_explosion_pos=chasing_meteoroid_explosion_pos,
                chasing_meteoroid_pos=chasing_meteoroid_pos,
                chasing_meteoroid_active=chasing_meteoroid_active,
                chasing_meteoroid_vel_y=chasing_meteoroid_vel_y,
                chasing_meteoroid_phase=chasing_meteoroid_phase,
                chasing_meteoroid_frame=chasing_meteoroid_frame,
                chasing_meteoroid_lane=chasing_meteoroid_lane,
                chasing_meteoroid_side=chasing_meteoroid_side,
                chasing_meteoroid_spawn_timer=chasing_meteoroid_spawn_timer,
                chasing_meteoroid_remaining=chasing_meteoroid_remaining,
                chasing_meteoroid_wave_active=chasing_meteoroid_wave_active,
                rejuvenator_pos=rejuv_pos,
                rejuvenator_active=rejuv_active,
                rejuvenator_dead=rejuv_dead,
                rejuvenator_frame=rejuv_frame,
                rejuvenator_lane=rejuv_lane,
                falling_rock_pos=falling_rock_pos,
                falling_rock_active=falling_rock_active,
                falling_rock_vel_y=falling_rock_vel_y,
                falling_rock_lane=falling_rock_lane,
                falling_rock_explosion_frame=falling_rock_explosion_frame,
                falling_rock_explosion_pos=falling_rock_explosion_pos,
                lane_blocker_pos=lane_blocker_pos,
                lane_blocker_active=lane_blocker_active,
                lane_blocker_vel_y=lane_blocker_vel_y,
                lane_blocker_lane=lane_blocker_lane,
                lane_blocker_phase=lane_blocker_phase,
                lane_blocker_timer=lane_blocker_timer,
                lane_blocker_explosion_frame=lane_blocker_explosion_frame,
                lane_blocker_explosion_pos=lane_blocker_explosion_pos,
                line_positions=line_positions,
                blue_line_counter=blue_line_counter,
                death_timer=next_death_timer,
                bouncer_pos=bouncer_pos,
                bouncer_vel=bouncer_vel,
                bouncer_state=bouncer_state,
                bouncer_timer=bouncer_timer,
                bouncer_active=bouncer_active,
                bouncer_lane=bouncer_lane,
                bouncer_step_index=bouncer_step_index,
                bouncer_explosion_frame=bouncer_explosion_frame,
                bouncer_explosion_pos=bouncer_explosion_pos,
                coin_pos=coin_pos,
                coin_active=coin_active,
                coin_timer=coin_timer,
                coin_side=coin_side,
                coin_explosion_frame=coin_explosion_frame,
                coin_explosion_pos=coin_explosion_pos,
                coin_spawn_count=coin_spawn_count,
                kamikaze_pos=kamikaze_pos,
                kamikaze_active=kamikaze_active,
                kamikaze_lane=kamikaze_lane,
                kamikaze_vel_y=kamikaze_vel_y,
                kamikaze_tracking=kamikaze_tracking,
                kamikaze_spawn_timer=kamikaze_spawn_timer,
                kamikaze_explosion_frame=kamikaze_explosion_frame,
                kamikaze_explosion_pos=kamikaze_explosion_pos,
            )

            reset_level_state = self._create_level_state(
                white_ufo_left=white_ufo_left, 
                torpedoes_left=torpedos_left,
                shooting_cooldown=shooting_cooldown,
                shooting_delay=shooting_delay,
                shot_type_pending=shot_type_pending,
                coin_spawn_count=coin_spawn_count
            )
            final_level_state = jax.tree_util.tree_map(
                lambda normal, reset: jnp.where(jnp.logical_or(just_died, sector_advanced), reset, normal),
                new_level_state, reset_level_state
            )
            
            lives_after_gain = jnp.minimum(state.lives + gain_life.astype(jnp.int32), 14)
            new_lives = jnp.where(just_died, jnp.maximum(lives_after_gain - 1, 0), lives_after_gain)

            new_state = BeamriderState(
                level=final_level_state, score=score, sector=sector,
                level_finished=jnp.array(0), reset_coords=jnp.array(False),
                lives=new_lives, steps=next_step, rng=next_rng,
            )
            
            done = jnp.array(self._get_done(new_state), dtype=jnp.bool_)
            env_reward = jnp.array(self._get_reward(state, new_state), dtype=jnp.float32)
            info = self._get_info(new_state)
            observation = self._get_observation(new_state)
            return observation, new_state, env_reward, done, info

        def handle_init(_):
            # During init, we ONLY update the blue line positions and counter.
            # We ALSO advance the RNG so the game doesn't start identically every time.
            rngs = jax.random.split(state.rng, 2)
            new_level = state.level._replace(
                line_positions=line_positions,
                blue_line_counter=blue_line_counter,
                shooting_cooldown=jnp.maximum(state.level.shooting_cooldown - 1, 0),
                shooting_delay=jnp.maximum(state.level.shooting_delay - 1, 0)
            )
            new_state = state._replace(level=new_level, steps=state.steps + 1, rng=rngs[0])
            
            done = jnp.array(False, dtype=jnp.bool_)
            env_reward = jnp.array(0.0, dtype=jnp.float32)
            info = self._get_info(new_state)
            observation = self._get_observation(new_state)
            return observation, new_state, env_reward, done, info

        return jax.lax.cond(is_init, handle_init, handle_normal, operand=None)


    def _player_step(self, state: BeamriderState, action: chex.Array):
        #level_constants = self._get_level_constants(state.sector)
        x = state.level.player_pos
        v = state.level.player_vel

        is_dead = state.level.death_timer > 0
        action = jnp.where(is_dead, Action.NOOP, action)
        v = jnp.where(is_dead, 0.0, v)

        # Actions with UP component trigger torpedo
        press_up = jnp.isin(action, jnp.array([
            Action.UP, Action.UPRIGHT, Action.UPLEFT, 
            Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE
        ]))

        # Actions with FIRE component trigger laser
        press_fire = jnp.isin(action, jnp.array([
            Action.FIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
            Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ]))

        is_in_lane = jnp.isin(x, jnp.array(self.consts.BOTTOM_OF_LANES)) # predicate: x is one of LANES

        v = jax.lax.cond(
            is_in_lane,          
            lambda v_: jnp.zeros_like(v_),          # then -> 0
            lambda v_: v_,                          # else -> keep v
            v,                                      # operand
        )
        
        # Check for Left/Right movement components
        press_left = jnp.isin(action, jnp.array([
            Action.LEFT, Action.UPLEFT, Action.DOWNLEFT,
            Action.LEFTFIRE, Action.UPLEFTFIRE, Action.DOWNLEFTFIRE
        ]))
        press_right = jnp.isin(action, jnp.array([
            Action.RIGHT, Action.UPRIGHT, Action.DOWNRIGHT,
            Action.RIGHTFIRE, Action.UPRIGHTFIRE, Action.DOWNRIGHTFIRE
        ]))

        v = jax.lax.cond(
            jnp.logical_or(press_left, press_right),
            lambda v_: (press_right.astype(v.dtype) - press_left.astype(v.dtype)) * self.consts.PLAYER_SPEED,    
            lambda v_: v_,
            v,
        )
        x_before_change = x
        x = jnp.clip(x + v, self.consts.LEFT_CLIP_PLAYER, self.consts.RIGHT_CLIP_PLAYER - self.consts.PLAYER_SPRITE_SIZE[1])

        ####### Corrected Shot Logic
        bullet_exists = self._bullet_infos(state)
        shooting_cooldown = state.level.shooting_cooldown
        shooting_delay = state.level.shooting_delay
        shot_type_pending = state.level.shot_type_pending

        # 1. Update recovery cooldown
        shooting_cooldown = jnp.maximum(shooting_cooldown - 1, 0)

        # 2. Handle firing input (only if no bullet exists and not in recovery/delay)
        can_initiate_launch = jnp.logical_and.reduce(jnp.array([
            jnp.logical_not(bullet_exists),
            is_in_lane,
            shooting_cooldown == 0,
            shooting_delay == 0
        ]))

        torpedo_actions = jnp.array([
            Action.UP, Action.UPRIGHT, Action.UPLEFT, 
            Action.UPFIRE, Action.UPRIGHTFIRE, Action.UPLEFTFIRE
        ])
        laser_actions = jnp.array([
            Action.FIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
            Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
        ])
        
        want_torpedo = jnp.any(jnp.isin(action, torpedo_actions)) & (state.level.torpedoes_left >= 1)
        want_laser = jnp.any(jnp.isin(action, laser_actions)) & jnp.logical_not(want_torpedo)
        
        initiate_launch = can_initiate_launch & (want_torpedo | want_laser)
        
        # 3. Advance launch delay and spawn bullet on transition 1 -> 0
        new_bullet = (shooting_delay == 1)
        
        # Update timers
        delay_hold = new_bullet & jnp.logical_not(is_in_lane)
        shooting_delay = jnp.where(
            initiate_launch,
            self.consts.PLAYER_SHOT_LAUNCH_DELAY,
            jnp.where(delay_hold, shooting_delay, jnp.maximum(shooting_delay - 1, 0)),
        )
        shot_type_pending = jnp.where(initiate_launch, 
                                      jnp.where(want_torpedo, self.consts.TORPEDO_ID, self.consts.LASER_ID),
                                      shot_type_pending)

        lanes = jnp.array(self.consts.BOTTOM_OF_LANES)
        lane_velocities = jnp.array(self.consts.BOTTOM_TO_TOP_LANE_VECTORS, dtype=jnp.float32)
        lane_index = jnp.argmax(x_before_change == lanes) 
        lane_velocity = lane_velocities[lane_index]

        spawn_bullet = new_bullet & is_in_lane
        shot_velocity = jnp.where(
            spawn_bullet,
            lane_velocity,
            state.level.player_shot_vel,
        )

        bullet_type = jnp.where(spawn_bullet, shot_type_pending, state.level.bullet_type)
        shot_frame = state.level.player_shot_frame
        reset_frame = jnp.full_like(shot_frame, -1)
        shot_frame = jnp.where(
            spawn_bullet,
            jnp.zeros_like(shot_frame),
            jnp.where(bullet_exists, shot_frame + 1, reset_frame),
        )

        is_laser = bullet_type == self.consts.LASER_ID
        max_frames = jnp.where(is_laser, LASER_PROJECTILE_FRAMES, TORPEDO_PROJECTILE_FRAMES)
        active = (shot_frame >= 0) & (shot_frame < max_frames)
        step_idx = jnp.floor_divide(jnp.maximum(shot_frame, 0), PROJECTILE_STEP_FRAMES).astype(jnp.int32)
        laser_y = jnp.take(PROJECTILE_Y_TABLE_LASER, step_idx, mode="clip")
        torpedo_y = jnp.take(PROJECTILE_Y_TABLE_TORPEDO, step_idx, mode="clip")
        shot_y = jnp.where(is_laser, laser_y, torpedo_y)
        shot_y = jnp.where(active, shot_y, self.consts.BULLET_OFFSCREEN_POS[1])

        lane_dy = jnp.where(shot_velocity[1] == 0, 1.0, shot_velocity[1])
        lane_slope = shot_velocity[0] / lane_dy
        delta_y = shot_y - state.level.player_shot_pos[1]
        shot_x = state.level.player_shot_pos[0] + lane_slope * delta_y
        shot_x = jnp.where(spawn_bullet, state.level.player_pos + 3, shot_x)
        shot_x = jnp.where(active, shot_x, self.consts.BULLET_OFFSCREEN_POS[0])
        shot_position = jnp.array([shot_x, shot_y])

        shot_frame = jnp.where(active, shot_frame, reset_frame)

        # Torpedo consumed ONLY when actually spawned
        new_torpedo_spawned = spawn_bullet & (shot_type_pending == self.consts.TORPEDO_ID)
        torpedos_left = state.level.torpedoes_left - new_torpedo_spawned.astype(jnp.int32)

        #####
        return(
            x, v, shot_position, shot_velocity, shot_frame, torpedos_left, bullet_type,
            shooting_cooldown, shooting_delay, shot_type_pending
        )

    def _collision_handler(
        self,
        state: BeamriderState,
        new_white_ufo_pos: chex.Array,
        new_shot_pos: chex.Array,
        new_shot_vel: chex.Array,
        new_bullet_type: chex.Array,
        current_patterns: chex.Array,
        current_timers: chex.Array,
    ):
        enemies_raw = new_white_ufo_pos.T

        # Collision should match what the player sees on screen:
        # - white UFO x-pos is shifted based on its y-pos (_get_ufo_alignment)
        # - player bullet x-pos is shifted based on its y-pos, type, and lane rounding (_get_player_shot_screen_x)
        ufo_x = new_white_ufo_pos[0, :] + _get_ufo_alignment(new_white_ufo_pos[1, :])
        ufo_y = new_white_ufo_pos[1, :]
        
        ufo_indices = jnp.clip(_get_index_ufo(ufo_y) - 1, 0, len(self.consts.UFO_SPRITE_SIZES) - 1)
        ufo_sizes = jnp.take(jnp.array(self.consts.UFO_SPRITE_SIZES), ufo_indices, axis=0)
        # ufo_sizes is (3, 2) -> [H, W]
        
        shot_x = _get_player_shot_screen_x(
            new_shot_pos,
            new_shot_vel,
            new_bullet_type,
            self.consts.LASER_ID,
        )
        shot_y = new_shot_pos[1]
        
        bullet_idx = _get_index_bullet(shot_y, new_bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
        # bullet_size is (2,) -> [H, W]

        # AABB collision check
        # x-overlap: (ufo_x < shot_x + bullet_w) & (shot_x < ufo_x + ufo_w)
        # y-overlap: (ufo_y < shot_y + bullet_h) & (shot_y < ufo_y + ufo_h)
        hit_mask_ufo = (ufo_x < shot_x + bullet_size[1]) & (shot_x < ufo_x + ufo_sizes[:, 1]) & \
                       (ufo_y < shot_y + bullet_size[0]) & (shot_y < ufo_y + ufo_sizes[:, 0])
        
        hit_index = jnp.argmax(hit_mask_ufo)
        hit_exists_ufo = jnp.any(hit_mask_ufo)
        hit_index = jnp.argmax(hit_mask_ufo)
        hit_exists_ufo = jnp.any(hit_mask_ufo)

        # If there are more than 3 UFOs, the hit one respawns.
        # Otherwise, it stays offscreen.
        should_respawn = state.level.white_ufo_left > 3
        respawn_pos = jnp.array([77.0, 43.0], dtype=enemies_raw.dtype)
        offscreen_pos = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=enemies_raw.dtype)
        target_pos = jnp.where(should_respawn, respawn_pos, offscreen_pos)

        enemy_pos_after_hit = enemies_raw.at[hit_index].set(target_pos).T

        # Reset pattern and timer for hit UFO
        new_patterns = jnp.where(hit_mask_ufo, int(WhiteUFOPattern.IDLE), current_patterns)
        new_timers = jnp.where(hit_mask_ufo, 0, current_timers)

        player_shot_pos = jnp.where(hit_exists_ufo, jnp.array(self.consts.BULLET_OFFSCREEN_POS), new_shot_pos)
        enemy_pos = jnp.where(hit_exists_ufo, enemy_pos_after_hit, new_white_ufo_pos)
        white_ufo_left = jnp.where(
            hit_exists_ufo,
            jnp.maximum(state.level.white_ufo_left - 1, 0),
            state.level.white_ufo_left,
        )
        clamped_sector = jnp.minimum(state.sector, 89)
        ufo_score = 40 + 4 * clamped_sector
        score = jnp.where(hit_exists_ufo, state.score + ufo_score, state.score)
        return (enemy_pos, player_shot_pos, new_patterns, new_timers, white_ufo_left, score, hit_mask_ufo, hit_exists_ufo)
    
    def _update_enemy_explosions(
        self,
        current_frames: chex.Array,
        current_positions: chex.Array,
        hit_mask: chex.Array,
        enemy_positions: chex.Array,
    ):
        """Advance current explosion animations and start new ones when enemies get hit."""
        offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=enemy_positions.dtype).reshape(2, 1),
            (1, current_frames.shape[0]),
        )

        active_frames = current_frames > 0
        advanced_frames = jnp.where(active_frames, current_frames + 1, current_frames)
        finished = advanced_frames > self.consts.ENEMY_EXPLOSION_FRAMES
        cleared_frames = jnp.where(finished, 0, advanced_frames)
        cleared_positions = jnp.where(finished[None, :], offscreen, current_positions)

        next_frames = jnp.where(hit_mask, jnp.ones_like(cleared_frames), cleared_frames)
        next_positions = jnp.where(hit_mask[None, :], enemy_positions, cleared_positions)
        return next_frames, next_positions

    def entropy_heat_prob(self, steps_static, alpha=0.0005, p_min=0.0002, p_max=0.8):
        steps =steps_static/10
        # steps_static: scalar integer or array
        heat = 1.0 - jnp.exp(-alpha * steps)
        p_swap = p_min + (p_max - p_min) * heat
        return p_swap
    
    def _advance_white_ufos(self, state: BeamriderState, keys: chex.Array) -> WhiteUFOUpdate:
        """Advance all white UFOs in lockstep for clearer logic inside step()."""

        updates = [self._white_ufo_step(state, idx, keys[idx]) for idx in range(3)]
        positions = jnp.stack([update[0] for update in updates], axis=1)
        vel_x = jnp.array([update[1] for update in updates])
        vel_y = jnp.array([update[2] for update in updates])
        velocities = jnp.stack([vel_x, vel_y])
        time_on_lane = jnp.array([update[3] for update in updates])
        attack_time = jnp.array([update[4] for update in updates])
        pattern_id = jnp.array([update[5] for update in updates], dtype=jnp.int32)
        pattern_timer = jnp.array([update[6] for update in updates], dtype=jnp.int32)
        return WhiteUFOUpdate(
            pos=positions,
            vel=velocities,
            time_on_lane=time_on_lane,
            attack_time=attack_time,
            pattern_id=pattern_id,
            pattern_timer=pattern_timer,
        )
    def _white_ufo_step(self, state: BeamriderState, index: int, key: chex.Array):
        white_ufo_position = jnp.array([state.level.white_ufo_pos[0][index], state.level.white_ufo_pos[1][index]])
        white_ufo_vel_x = state.level.white_ufo_vel[0][index]
        white_ufo_vel_y = state.level.white_ufo_vel[1][index]
        time_on_lane = state.level.white_ufo_time_on_lane[index]
        attack_time = state.level.white_ufo_attack_time[index]
        pattern_id = state.level.white_ufo_pattern_id[index]
        pattern_timer = state.level.white_ufo_pattern_timer[index]

        offscreen_pos = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=white_ufo_position.dtype)
        is_offscreen = jnp.all(white_ufo_position == offscreen_pos)

        key_pattern, key_motion = jax.random.split(key)
        pattern_id, pattern_timer, time_on_lane, attack_time = self._white_ufo_update_pattern_state(
            state.sector, white_ufo_position, time_on_lane, attack_time, pattern_id, pattern_timer, key_pattern
        )

        requires_lane_motion = self._white_ufo_pattern_requires_lane_motion(pattern_id)

        def follow_lane(_):
            return self._white_ufo_normal(white_ufo_position, white_ufo_vel_x, white_ufo_vel_y, pattern_id)

        def stay_on_top(_):
            return self._white_ufo_top_lane(white_ufo_position, white_ufo_vel_x, pattern_id, key_motion)

        white_ufo_vel_x, white_ufo_vel_y = jax.lax.cond(
            requires_lane_motion,
            follow_lane,
            stay_on_top,
            operand=None
        )

        new_x = white_ufo_position[0] + white_ufo_vel_x
        new_y = white_ufo_position[1] + white_ufo_vel_y
        
        # Only clip horizontally if on top lane
        on_top_lane = new_y <= self.consts.TOP_CLIP
        clipped_x = jnp.clip(new_x, self.consts.LEFT_CLIP_PLAYER, self.consts.RIGHT_CLIP_PLAYER)
        new_x = jnp.where(on_top_lane, clipped_x, new_x)

        new_y = jnp.clip(new_y, self.consts.TOP_CLIP, self.consts.PLAYER_POS_Y + 1.0)
        
        # Only respawn if it was not already offscreen (i.e. it was active)
        should_respawn = jnp.logical_and(
            jnp.logical_not(is_offscreen),
            jnp.logical_or(
                new_x < 0,
                jnp.logical_or(
                    new_x > self.consts.SCREEN_WIDTH,
                    new_y > self.consts.PLAYER_POS_Y
                )
            )
        )

        white_ufo_position = jnp.where(should_respawn, jnp.array([77.0, 43.0]), jnp.array([new_x, new_y]))
        white_ufo_vel_x = jnp.where(should_respawn, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(should_respawn, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(should_respawn, 0, time_on_lane)
        attack_time = jnp.where(should_respawn, 0, attack_time)
        pattern_id = jnp.where(should_respawn, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(should_respawn, 0, pattern_timer)

        # Ensure that if it was offscreen, it stays offscreen and its state is reset
        white_ufo_position = jnp.where(is_offscreen, offscreen_pos, white_ufo_position)
        white_ufo_vel_x = jnp.where(is_offscreen, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(is_offscreen, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(is_offscreen, 0, time_on_lane)
        attack_time = jnp.where(is_offscreen, 0, attack_time)
        pattern_id = jnp.where(is_offscreen, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(is_offscreen, 0, pattern_timer)

        return (
            white_ufo_position,
            white_ufo_vel_x,
            white_ufo_vel_y,
            time_on_lane,
            attack_time,
            pattern_id,
            pattern_timer,
        )

    def _white_ufo_pattern_requires_lane_motion(self, pattern_id: chex.Array) -> chex.Array:
        drop_straight = pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT)
        drop_left = pattern_id == int(WhiteUFOPattern.DROP_LEFT)
        drop_right = pattern_id == int(WhiteUFOPattern.DROP_RIGHT)
        retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        kamikaze = pattern_id == int(WhiteUFOPattern.KAMIKAZE)
        triple_right = pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)
        triple_left = pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT)
        return jnp.logical_or(
            drop_straight,
            jnp.logical_or(
                drop_left,
                jnp.logical_or(
                    drop_right,
                    jnp.logical_or(retreat, jnp.logical_or(move_back, jnp.logical_or(kamikaze, jnp.logical_or(triple_right, triple_left))))
                )
            ),
        )

    def _white_ufo_update_pattern_state(
        self,
        sector: chex.Array,
        position: chex.Array,
        time_on_lane: chex.Array,
        attack_time: chex.Array,
        pattern_id: chex.Array,
        pattern_timer: chex.Array,
        key: chex.Array,
    ):
        on_top_lane = position[1] <= self.consts.TOP_CLIP
        time_on_lane = jnp.where(on_top_lane, time_on_lane + 1, 0)
        attack_time = jnp.where(on_top_lane, 0, attack_time)

        # Calculate closest lane and distance to it to determine if we are "on a lane"
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

        ufo_x = position[0].astype(jnp.float32)
        ufo_y = position[1].astype(jnp.float32)
        lane_x_at_ufo_y = lanes_top_x + lane_dx_over_dy * (ufo_y - float(self.consts.TOP_CLIP))
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x)).astype(jnp.int32)

        closest_lane_x = lane_x_at_ufo_y[closest_lane_id]
        dist_to_lane = jnp.abs(closest_lane_x - ufo_x)
        is_on_lane = dist_to_lane <= 0.25

        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        
        # Triple Shot State Machine:
        # pattern_timer bits: [shoot_now (1 bit) | last_lane (4 bits) | shots_left (3 bits)]
        shots_left = pattern_timer & 7
        last_lane = (pattern_timer >> 3) & 15
        shoot_now = (pattern_timer >> 7) & 1

        def update_triple(_):
            new_shoot_now = jnp.array(0, dtype=jnp.int32)
            new_shots_left = shots_left
            new_last_lane = last_lane
            
            # If we just signaled a shot, clear it
            # If we have shots left and just reached a new lane, signal a shot
            can_shoot = (shots_left > 0) & is_on_lane & (closest_lane_id != last_lane)
            
            new_shoot_now = jnp.where(shoot_now == 1, 0, jnp.where(can_shoot, 1, 0))
            new_shots_left = jnp.where(can_shoot, shots_left - 1, shots_left)
            new_last_lane = jnp.where(can_shoot, closest_lane_id, last_lane)
            
            return (new_shoot_now << 7) | (new_last_lane << 3) | new_shots_left

        pattern_timer = jnp.where(
            is_triple,
            update_triple(None),
            jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer))
        )

        shootable_lane = jnp.logical_and(closest_lane_id > 0, closest_lane_id < 6)
        allow_shoot = jnp.logical_and(jnp.logical_not(on_top_lane), shootable_lane)

        is_drop_pattern = jnp.logical_or(
            pattern_id == int(WhiteUFOPattern.DROP_STRAIGHT),
            jnp.logical_or(
                pattern_id == int(WhiteUFOPattern.DROP_LEFT),
                jnp.logical_or(
                    pattern_id == int(WhiteUFOPattern.DROP_RIGHT),
                    pattern_id == int(WhiteUFOPattern.MOVE_BACK)
                ),
            ),
        )
        is_triple_shot_pattern = is_triple
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = is_drop_pattern | is_shoot_pattern | is_triple_shot_pattern
        attack_time = jnp.where(
            jnp.logical_and(jnp.logical_not(on_top_lane), is_engagement_pattern),
            attack_time + 1,
            attack_time,
        )

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        movement_finished = jnp.logical_and(jnp.logical_or(is_retreat, is_move_back), on_top_lane)
        pattern_id = jnp.where(movement_finished, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(movement_finished, 0, pattern_timer)
        attack_time = jnp.where(movement_finished, 0, attack_time)

        triple_finished = is_triple & ((pattern_timer & 7) == 0) & jnp.logical_not((pattern_timer >> 7) & 1) & is_on_lane

        # Stuck check: if target lane is same as current lane due to clipping in stage 6/7, finish triple shot.
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)
        in_restricted_stage = position[1] >= 86.0
        min_lane = jnp.where(in_restricted_stage, 1, 0)
        max_lane = jnp.where(in_restricted_stage, 5, 6)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, min_lane, max_lane)
        
        triple_stuck = is_triple & is_on_lane & (shots_left > 0) & (target_lane_id == closest_lane_id) & (closest_lane_id == last_lane)
        triple_finished = triple_finished | triple_stuck

        pattern_finished_off_top = jnp.logical_and.reduce(jnp.array([
            jnp.logical_not(on_top_lane),
            is_engagement_pattern,
            jnp.where(is_triple, triple_finished, pattern_timer == 0),
            is_on_lane,
        ]))

        key_start_roll, key_start_choice, key_retreat_roll, key_chain_choice, key_kamikaze_roll = jax.random.split(key, 5)
        retreat_roll = jax.random.uniform(key_retreat_roll)
        retreat_prob = self._white_ufo_retreat_prob(attack_time)
        retreat_now = jnp.logical_and(pattern_finished_off_top, retreat_roll < retreat_prob)
        pattern_id = jnp.where(retreat_now, int(WhiteUFOPattern.RETREAT), pattern_id)
        pattern_timer = jnp.where(retreat_now, self.consts.WHITE_UFO_RETREAT_DURATION, pattern_timer)
        attack_time = jnp.where(retreat_now, 0, attack_time)

        chain_next = jnp.logical_and(pattern_finished_off_top, jnp.logical_not(retreat_now))

        ufo_stage = _get_index_ufo(position[1])

        def choose_chain_pattern(_):
            is_kamikaze_zone = position[1] >= self.consts.KAMIKAZE_Y_THRESHOLD
            pattern, duration = self._white_ufo_choose_pattern(
                key_chain_choice, 
                allow_shoot=allow_shoot, 
                prev_pattern=pattern_id,
                is_kamikaze_zone=is_kamikaze_zone,
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane
            )
            # If pattern is triple shot, initialize bits 3-6 (last_lane) to 15 (all ones)
            # so that it fires on the first lane it is on.
            is_new_triple = (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
            init_timer = jnp.where(is_new_triple, duration | (15 << 3), duration)
            return pattern, init_timer

        def keep_after_chain(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(
            chain_next,
            choose_chain_pattern,
            keep_after_chain,
            operand=None,
        )

        should_choose_new = jnp.logical_and.reduce(jnp.array([
            on_top_lane,
            pattern_id == int(WhiteUFOPattern.IDLE),
            pattern_timer == 0,
        ]))
        p_start = self.entropy_heat_prob(
            time_on_lane,
            alpha=self.consts.WHITE_UFO_ATTACK_ALPHA,
            p_min=self.consts.WHITE_UFO_ATTACK_P_MIN,
            p_max=self.consts.WHITE_UFO_ATTACK_P_MAX,
        )
        start_roll = jax.random.uniform(key_start_roll)
        start_attack = jnp.logical_and(should_choose_new, start_roll < p_start)

        def choose_new_pattern(_):
            pattern, duration = self._white_ufo_choose_pattern(
                key_start_choice, 
                allow_shoot=jnp.array(False), 
                prev_pattern=pattern_id,
                is_kamikaze_zone=jnp.array(False),
                sector=sector,
                stage=ufo_stage,
                lane=closest_lane_id,
                is_on_lane=is_on_lane
            )
            is_new_triple = (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
            init_timer = jnp.where(is_new_triple, duration | (15 << 3), duration)
            return pattern, init_timer

        def keep_pattern(_):
            return pattern_id, pattern_timer

        pattern_id, pattern_timer = jax.lax.cond(
            start_attack,
            choose_new_pattern,
            keep_pattern,
            operand=None,
        )

        return pattern_id, pattern_timer, time_on_lane, attack_time

    def _white_ufo_retreat_prob(self, attack_time: chex.Array) -> chex.Array:
        t = attack_time.astype(jnp.float32)
        alpha = jnp.array(self.consts.WHITE_UFO_RETREAT_ALPHA, dtype=jnp.float32)
        p_min = jnp.array(self.consts.WHITE_UFO_RETREAT_P_MIN, dtype=jnp.float32)
        p_max = jnp.array(self.consts.WHITE_UFO_RETREAT_P_MAX, dtype=jnp.float32)
        heat = 1.0 - jnp.exp(-alpha * t)
        return p_min + (p_max - p_min) * heat

    def _white_ufo_choose_pattern(
        self, 
        key: chex.Array, 
        *, 
        allow_shoot: chex.Array, 
        prev_pattern: chex.Array,
        is_kamikaze_zone: chex.Array,
        sector: chex.Array,
        stage: chex.Array,
        lane: chex.Array,
        is_on_lane: chex.Array
    ):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.MOVE_BACK),
                int(WhiteUFOPattern.KAMIKAZE),
                int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT),
                int(WhiteUFOPattern.TRIPLE_SHOT_LEFT),
            ],
            dtype=jnp.int32,
        )
        # Probabilities for IDLE, DROP_STRAIGHT, DROP_RIGHT, DROP_LEFT, RETREAT, SHOOT, MOVE_BACK, KAMIKAZE, TRIPLE_RIGHT, TRIPLE_LEFT
        pattern_probs = jnp.array(self.consts.WHITE_UFO_PATTERN_PROBS, dtype=jnp.float32)

        # Restriction: Cannot follow MOVE_BACK with DROP_STRAIGHT (idx 0)
        is_move_back = (prev_pattern == int(WhiteUFOPattern.MOVE_BACK))
        chain_mask = jnp.ones_like(pattern_probs).at[0].set(jnp.where(is_move_back, 0.0, 1.0))
        pattern_probs = pattern_probs * chain_mask

        # Mask out SHOOT (index 3) if not allowed.
        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)
        
        # Mask out KAMIKAZE (index 5) if not in zone
        kamikaze_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(is_kamikaze_zone, pattern_probs, pattern_probs * kamikaze_mask)

        # Triple shot conditions
        # sector >= 7
        # stage in [4, 5, 6]
        # triple right: lane in [1, 2, 3]
        # triple left: lane in [5, 4, 3]
        can_triple = sector >= 7
        can_triple = can_triple & (stage >= 4) & (stage <= 6) & is_on_lane

        # Further restriction for Stage 6: restricted to lanes 1-5.
        # If in Stage 6, TRIPLE_SHOT_RIGHT (ends at lane+2) must have lane+2 <= 5 => lane <= 3.
        # TRIPLE_SHOT_LEFT (ends at lane-2) must have lane-2 >= 1 => lane >= 3.
        # This matches the requested lanes (1,2,3 for right; 5,4,3 for left).
        
        can_triple_right = can_triple & (lane >= 1) & (lane <= 3)
        can_triple_left = can_triple & (lane >= 3) & (lane <= 5)
        
        triple_right_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
        triple_left_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        
        pattern_probs = jnp.where(can_triple_right, pattern_probs, pattern_probs * triple_right_mask)
        pattern_probs = jnp.where(can_triple_left, pattern_probs, pattern_probs * triple_left_mask)
        
        # Avoid division by zero if all probs masked
        prob_sum = jnp.sum(pattern_probs)
        pattern_probs = jnp.where(prob_sum > 0, pattern_probs / prob_sum, pattern_probs)
        
        pattern = jax.random.choice(key, pattern_choices, shape=(), p=pattern_probs)
        pattern_durations = jnp.array(self.consts.WHITE_UFO_PATTERN_DURATIONS)
        duration = pattern_durations[pattern]
        return pattern, duration

    def _white_ufo_top_lane(self, white_ufo_pos, white_ufo_vel_x, pattern_id, key: chex.Array):
        hold_position = jnp.logical_or(
            pattern_id == int(WhiteUFOPattern.SHOOT),
            white_ufo_pos[1] > float(self.consts.TOP_CLIP),
        )
        min_speed = float(self.consts.WHITE_UFO_TOP_LANE_MIN_SPEED)
        turn_speed = float(self.consts.WHITE_UFO_TOP_LANE_TURN_SPEED)

        vx = jnp.where(hold_position, 0.0, white_ufo_vel_x)
        need_boost = jnp.logical_and(jnp.logical_not(hold_position), jnp.abs(vx) < min_speed)
        random_sign = jnp.where(jax.random.uniform(key) < 0.5, -1.0, 1.0)
        direction = jnp.where(vx == 0.0, random_sign, jnp.sign(vx))
        vx = jnp.where(need_boost, direction * min_speed, vx)

        do_bounce = jnp.logical_not(hold_position)
        vx = jnp.where(
            jnp.logical_and(do_bounce, white_ufo_pos[0] >= self.consts.RIGHT_CLIP_PLAYER),
            -turn_speed,
            vx,
        )
        vx = jnp.where(
            jnp.logical_and(do_bounce, white_ufo_pos[0] <= self.consts.LEFT_CLIP_PLAYER),
            turn_speed,
            vx,
        )
        return vx, 0.0
    
    def _white_ufo_normal(self, white_ufo_pos, white_ufo_vel_x, white_ufo_vel_y, pattern_id): #velocities not used anymore
        speed_factor = self.consts.WHITE_UFO_SPEED_FACTOR
        retreat_mult = self.consts.WHITE_UFO_RETREAT_SPEED_MULT
        x, y = white_ufo_pos[0], white_ufo_pos[1]
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)

        lane_dx = lane_vectors[:, 0]
        lane_dy = lane_vectors[:, 1]
        lane_x_at_y = lanes_top_x + (lane_dx / lane_dy) * (y - float(self.consts.TOP_CLIP))

        # 1. Identify the current lane
        closest_lane_id = jnp.argmin(jnp.abs(lane_x_at_y - x))

        # 2. Determine index offset based on pattern
        # DROP_RIGHT (+1), DROP_LEFT (-1), others (0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_RIGHT), 1, 0)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.DROP_LEFT), -1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT), 1, lane_offset)
        lane_offset = jnp.where(pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT), -1, lane_offset)

        # 3. Apply offset and clip to valid lane indices (0 to 6)
        # Stage 6 starts at y=86. When in Stage 6 or 7, restrict to lanes 1-5.
        in_restricted_stage = y >= 86.0
        min_lane = jnp.where(in_restricted_stage, 1, 0)
        max_lane = jnp.where(in_restricted_stage, 5, 6)
        target_lane_id = jnp.clip(closest_lane_id + lane_offset, min_lane, max_lane)

        lane_vector = lane_vectors[target_lane_id]
        target_lane_x = lane_x_at_y[target_lane_id]

        is_retreat = pattern_id == int(WhiteUFOPattern.RETREAT)
        is_move_back = pattern_id == int(WhiteUFOPattern.MOVE_BACK)
        is_kamikaze = pattern_id == int(WhiteUFOPattern.KAMIKAZE)
        is_triple = (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        
        cross_track = target_lane_x - x
        distance_to_lane = jnp.abs(cross_track)
        direction = jnp.sign(cross_track)

        def seek_lane(_):
            attack_vx = jnp.where(direction == 0, 0.0, direction * 0.5)
            retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)
            
            # For Kamikaze and Triple Shot, use retreat lateral speed (fast seek)
            new_vx = jnp.where(is_retreat | is_kamikaze | is_triple, retreat_vx, attack_vx)
            
            normal_vy = 0.25
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
            move_back_vy = -lane_vector[1] * speed_factor
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult
            triple_vy = 0.25 # Stick to normal vertical speed for triple shot

            new_vy = jnp.where(is_retreat, retreat_vy, normal_vy)
            new_vy = jnp.where(is_move_back, move_back_vy, new_vy)
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)
            
            return new_vx, new_vy

        def follow_lane(_):
            normal_vx = lane_vector[0] * speed_factor
            normal_vy = lane_vector[1] * speed_factor
            
            retreat_vx = -lane_vector[0] * speed_factor * retreat_mult
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
            
            move_back_vx = -lane_vector[0] * speed_factor
            move_back_vy = -lane_vector[1] * speed_factor
            
            kamikaze_vx = lane_vector[0] * speed_factor * retreat_mult
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult
            
            triple_vy = 0.25

            new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_move_back, move_back_vx, normal_vx))
            new_vx = jnp.where(is_kamikaze, kamikaze_vx, new_vx)
            
            new_vy = jnp.where(is_retreat, retreat_vy, jnp.where(is_move_back, move_back_vy, normal_vy))
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            new_vy = jnp.where(is_triple, triple_vy, new_vy)
            
            return new_vx, new_vy

        return jax.lax.cond(
            distance_to_lane <= 0.25,
            follow_lane,
            seek_lane,
            operand=None,
        )

    def _enemy_shot_step(
        self,
        state: BeamriderState,
        white_ufo_pos: chex.Array,
        white_ufo_pattern_id: chex.Array,
        white_ufo_pattern_timer: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

        offscreen_xy = jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32)
        offscreen = jnp.tile(offscreen_xy.reshape(2, 1), (1, 9))

        shot_pos = state.level.enemy_shot_pos.astype(jnp.float32)
        shot_lane = state.level.enemy_shot_vel.astype(jnp.int32)
        shot_timer = state.level.enemy_shot_timer.astype(jnp.int32)

        shot_active = shot_pos[1] <= float(self.consts.BOTTOM_CLIP)
        shot_timer = jnp.where(shot_active, shot_timer + 1, 0)

        shoot_duration = jnp.array(self.consts.WHITE_UFO_PATTERN_DURATIONS, dtype=jnp.int32)[
            int(WhiteUFOPattern.SHOOT)
        ]
        
        is_triple = (white_ufo_pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_RIGHT)) | (white_ufo_pattern_id == int(WhiteUFOPattern.TRIPLE_SHOT_LEFT))
        shoot_now_triple = (white_ufo_pattern_timer >> 7) & 1

        wants_spawn = jnp.logical_and(
            white_ufo_pattern_id == int(WhiteUFOPattern.SHOOT),
            white_ufo_pattern_timer == shoot_duration,
        )
        wants_spawn = wants_spawn | (is_triple & (shoot_now_triple == 1))

        ufo_on_screen = white_ufo_pos[1] <= float(self.consts.BOTTOM_CLIP)
        ufo_not_on_top_lane = white_ufo_pos[1] > float(self.consts.TOP_CLIP)
        ufo_x = white_ufo_pos[0].astype(jnp.float32)
        ufo_y = white_ufo_pos[1].astype(jnp.float32)
        
        lane_x_at_ufo_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            ufo_y[None, :] - float(self.consts.TOP_CLIP)
        )
        closest_lane = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x[None, :]), axis=0).astype(jnp.int32)
        allowed_shot_lane = jnp.logical_and(closest_lane > 0, closest_lane < 6)
        
        # Each UFO has 3 slots: [3*i, 3*i+1, 3*i+2]
        # We find the first inactive slot for each UFO.
        ufo_shot_active = jnp.reshape(shot_active, (3, 3)) # UFO, Slot
        first_inactive_slot = jnp.argmax(jnp.logical_not(ufo_shot_active), axis=1) # UFO
        has_inactive_slot = jnp.any(jnp.logical_not(ufo_shot_active), axis=1)
        
        spawn = jnp.logical_and.reduce(
            jnp.array([
                wants_spawn,
                ufo_on_screen,
                ufo_not_on_top_lane,
                allowed_shot_lane,
                has_inactive_slot,
            ])
        )

        spawn_y = jnp.clip(ufo_y + 4.0, float(self.consts.TOP_CLIP), float(self.consts.BOTTOM_CLIP))
        spawn_x = jnp.take(lanes_top_x, closest_lane) + jnp.take(lane_dx_over_dy, closest_lane) * (
            spawn_y - float(self.consts.TOP_CLIP)
        )
        
        # Global slot indices for each UFO's best spawn candidate
        spawn_slots = jnp.arange(3) * 3 + first_inactive_slot
        spawn_mask = (jax.nn.one_hot(spawn_slots, 9) * spawn[:, None]).sum(axis=0).astype(jnp.bool_) # (9,)
        
        # Update shot state for spawned shots
        spawn_x_expanded = jnp.repeat(spawn_x, 3) # (9,)
        spawn_y_expanded = jnp.repeat(spawn_y, 3) # (9,)
        spawn_pos_expanded = jnp.stack([spawn_x_expanded, spawn_y_expanded]) # (2, 9)

        shot_pos = jnp.where(spawn_mask[None, :], spawn_pos_expanded, shot_pos)
        
        closest_lane_expanded = jnp.repeat(closest_lane, 3) # (9,)
        shot_lane = jnp.where(spawn_mask, closest_lane_expanded, shot_lane)
        shot_timer = jnp.where(spawn_mask, 0, shot_timer)
        shot_active = jnp.logical_or(shot_active, spawn_mask)

        # Per-shot cadence (frame-by-frame):
        should_move = jnp.logical_and(shot_active, (shot_timer % 4) == 2)
        speed = float(self.consts.WHITE_UFO_SHOT_SPEED_FACTOR)
        lane_dy = jnp.take(lane_vectors[:, 1], shot_lane)
        y_after = shot_pos[1] + jnp.where(should_move, lane_dy * speed, 0.0)
        x_after = jnp.take(lanes_top_x, shot_lane) + jnp.take(lane_dx_over_dy, shot_lane) * (
            y_after - float(self.consts.TOP_CLIP)
        )
        shot_pos = jnp.where(shot_active, jnp.stack([x_after, y_after]), shot_pos)

        moved_offscreen = shot_pos[1] > float(self.consts.BOTTOM_CLIP)
        shot_pos = jnp.where(moved_offscreen, offscreen, shot_pos)
        shot_timer = jnp.where(moved_offscreen, 0, shot_timer)
        shot_active = jnp.logical_and(shot_active, jnp.logical_not(moved_offscreen))

        player_left = state.level.player_pos.astype(jnp.float32)
        player_y = float(self.consts.PLAYER_POS_Y)
        player_size = jnp.array(self.consts.PLAYER_SPRITE_SIZE)

        shot_x = shot_pos[0] + _get_ufo_alignment(shot_pos[1])
        shot_y = shot_pos[1]
        
        timer = shot_timer
        sprite_idx = (jnp.floor_divide(timer, 4) % 2).astype(jnp.int32)
        shot_sizes = jnp.take(jnp.array(self.consts.ENEMY_SHOT_SPRITE_SIZES), sprite_idx, axis=0)

        hits = (shot_active) & \
               (shot_x < player_left + player_size[1]) & (player_left < shot_x + shot_sizes[:, 1]) & \
               (shot_y < player_y + player_size[0]) & (player_y < shot_y + shot_sizes[:, 0])

        hit_count = jnp.sum(hits.astype(jnp.int32))
        
        shot_pos = jnp.where(hits[None, :], offscreen, shot_pos)
        shot_timer = jnp.where(hits, 0, shot_timer)
        return shot_pos, shot_lane, shot_timer, hit_count

    def _chasing_meteoroid_step(
        self,
        state: BeamriderState,
        player_x: chex.Array,
        player_vel: chex.Array,
        white_ufo_left: chex.Array,
        key: chex.Array,
    ):
        pos = state.level.chasing_meteoroid_pos
        active = state.level.chasing_meteoroid_active
        vel_y = state.level.chasing_meteoroid_vel_y
        phase = state.level.chasing_meteoroid_phase
        frame = state.level.chasing_meteoroid_frame
        lane = state.level.chasing_meteoroid_lane
        side = state.level.chasing_meteoroid_side
        spawn_timer = state.level.chasing_meteoroid_spawn_timer
        remaining = state.level.chasing_meteoroid_remaining
        wave_active = state.level.chasing_meteoroid_wave_active

        is_sector_6_plus = state.sector >= 6
        key_start, key_wave, key_interval, key_side = jax.random.split(key, 4)

        ms_stage = state.level.mothership_stage
        # Stop spawning if mothership is descending (3) or exploding (5)
        # This ensures they finish their tracing before the mothership disappears.
        can_spawn_in_ms = (white_ufo_left == 0) & (ms_stage < 3)

        # Trigger waves of chasing meteoroids:
        # 1. Periodically when all UFOs are cleared (all sectors)
        # 2. Periodically during normal play (Sector 6+)
        # Frequency for normal gameplay: 0.0021 probability per frame (~once per 8 seconds).
        # Clear-UFO waves are more frequent: 0.05 probability per frame.
        start_chance = jnp.where(can_spawn_in_ms, 0.05, jnp.where(is_sector_6_plus & (white_ufo_left > 0), 0.0021, 0.0))
        
        start_wave = jnp.logical_and(
            jax.random.uniform(key_start) < start_chance,
            jnp.logical_not(wave_active)
        )
        
        wave_active = jnp.where(start_wave, True, wave_active)
        
        # Chunk size: max 3 during normal play, full range during Mothership phase
        min_w = jnp.where(white_ufo_left == 0, self.consts.CHASING_METEOROID_WAVE_MIN, 1)
        max_w = jnp.where(white_ufo_left == 0, self.consts.CHASING_METEOROID_WAVE_MAX, 3)
        
        wave_count = jax.random.randint(
            key_wave,
            (),
            min_w,
            max_w + 1,
        )
        remaining = jnp.where(start_wave, wave_count, remaining)
        spawn_timer = jnp.where(start_wave, 0, spawn_timer)

        # Reset wave state when moving to a sector where they shouldn't appear normally,
        # and UFOs are present.
        should_cancel = jnp.logical_and(jnp.logical_not(is_sector_6_plus), white_ufo_left > 0)
        # Also stop spawning waves if it's too late in the Mothership phase
        should_cancel = jnp.logical_or(should_cancel, (white_ufo_left == 0) & (ms_stage >= 3))
        
        # Also reset wave_active when finished so it can repeat.
        wave_finished = wave_active & (remaining == 0) & jnp.all(jnp.logical_not(active))
        
        wave_active = jnp.where(should_cancel | wave_finished, False, wave_active)
        remaining = jnp.where(should_cancel, 0, remaining)
        spawn_timer = jnp.where(should_cancel, 0, spawn_timer)

        spawn_timer = jnp.where(
            jnp.logical_and(wave_active, remaining > 0),
            jnp.maximum(spawn_timer - 1, 0),
            spawn_timer,
        )

        should_spawn = jnp.logical_and.reduce(jnp.array([
            wave_active,
            remaining > 0,
            spawn_timer == 0,
        ]))
        has_slot = jnp.any(jnp.logical_not(active))
        should_spawn = jnp.logical_and(should_spawn, has_slot)

        spawn_interval = jax.random.randint(
            key_interval,
            (),
            self.consts.CHASING_METEOROID_SPAWN_INTERVAL_MIN,
            self.consts.CHASING_METEOROID_SPAWN_INTERVAL_MAX + 1,
        )
        spawn_side = jnp.where(jax.random.uniform(key_side) < 0.5, 1, -1)
        spawn_x = jnp.where(
            spawn_side == 1,
            float(self.consts.LEFT_CLIP_PLAYER),
            float(self.consts.RIGHT_CLIP_PLAYER),
        )
        spawn_y = float(self.consts.CHASING_METEOROID_SPAWN_Y)

        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask.astype(jnp.int32))
        one_hot = jax.nn.one_hot(slot, self.consts.CHASING_METEOROID_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)
        pos_spawned = pos + (spawn_pos[:, None] - pos) * one_hot[None, :]
        active_spawned = jnp.where(one_hot_bool, True, active)
        vel_y_spawned = jnp.where(one_hot_bool, 0.0, vel_y)
        phase_spawned = jnp.where(one_hot_bool, 0, phase)
        frame_spawned = jnp.where(one_hot_bool, 0, frame)
        lane_spawned = jnp.where(one_hot_bool, 0, lane)
        side_spawned = jnp.where(one_hot_bool, spawn_side, side)

        pos = jnp.where(should_spawn, pos_spawned, pos)
        active = jnp.where(should_spawn, active_spawned, active)
        vel_y = jnp.where(should_spawn, vel_y_spawned, vel_y)
        phase = jnp.where(should_spawn, phase_spawned, phase)
        frame = jnp.where(should_spawn, frame_spawned, frame)
        lane = jnp.where(should_spawn, lane_spawned, lane)
        side = jnp.where(should_spawn, side_spawned, side)
        remaining = jnp.where(should_spawn, remaining - 1, remaining)
        spawn_timer = jnp.where(should_spawn, spawn_interval, spawn_timer)

        cycle_dx = jnp.array(self.consts.CHASING_METEOROID_CYCLE_DX, dtype=pos.dtype)
        cycle_dy = jnp.array(self.consts.CHASING_METEOROID_CYCLE_DY, dtype=pos.dtype)
        dy_a = jnp.take(cycle_dy, frame)
        new_y_a = pos[1] + dy_a

        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=pos.dtype)
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=pos.dtype)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

        lane_x_at_current_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            new_y_a[None, :] - float(self.consts.TOP_CLIP)
        )
        dx_dir = side.astype(pos.dtype)
        dx = jnp.take(cycle_dx, frame) * dx_dir
        new_x_a = pos[0] + dx

        new_vel_y = vel_y + self.consts.CHASING_METEOROID_ACCEL
        new_y_b = pos[1] + new_vel_y
        lane_x_at_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            new_y_b[None, :] - float(self.consts.TOP_CLIP)
        )
        lane_x_b = jnp.take_along_axis(lane_x_at_y, lane[None, :], axis=0).squeeze(0)
        new_x_b = lane_x_b

        phase_descend = phase == 2
        phase_horizontal = jnp.logical_not(phase_descend)
        new_x = jnp.where(phase_descend, new_x_b, new_x_a)
        new_y = jnp.where(phase_descend, new_y_b, new_y_a)

        player_left = player_x
        player_right = player_x + float(self.consts.PLAYER_SPRITE_SIZE[1])
        align_x = _get_ufo_alignment(new_y_a).astype(new_x_a.dtype)
        aligned_x_a = new_x_a + align_x
        chasing_meteoroid_left = aligned_x_a
        chasing_meteoroid_right = aligned_x_a + float(self.consts.METEOROID_SPRITE_SIZE[1])
        hits_x = jnp.logical_and(chasing_meteoroid_right >= player_left, chasing_meteoroid_left <= player_right)
        player_center = player_x + float(self.consts.PLAYER_SPRITE_SIZE[1]) / 2.0
        bottom_lanes = jnp.array(self.consts.BOTTOM_OF_LANES, dtype=jnp.float32)
        nearest_lane_idx = jnp.argmin(jnp.abs(bottom_lanes - player_center)).astype(jnp.int32)
        
        # Consider only inner 5 lanes (indices 1-5 in TOP_OF_LANES)
        # playable_lanes_x shape: (5, N)
        playable_lanes_x = lane_x_at_current_y[1:6]
        
        # Alignment check for all 5 inner lanes
        # dist_to_lanes shape: (5, N)
        dist_to_lanes = jnp.abs(playable_lanes_x - new_x_a[None, :])
        is_aligned = dist_to_lanes <= float(self.consts.CHASING_METEOROID_LANE_ALIGN_THRESHOLD)
        
        # Valid drop logic:
        # Lane index i (0..4) corresponds to Lanes 1..5
        # Drop if:
        #   Moving Right (side > 0) AND (Lane >= PlayerLane OR Lane == 4 (Last))
        #   Moving Left  (side < 0) AND (Lane <= PlayerLane OR Lane == 0 (Last))
        # Note: PlayerLane is 0..4 relative to inner lanes.
        # "Lane >= PlayerLane" covers "Lane == 4" if PlayerLane <= 4 (always true).
        
        lane_indices = jnp.arange(5)[:, None] # (5, 1)
        
        # Broadcase side and player_lane
        # side: (N,), nearest_lane_idx: scalar
        side_broad = side[None, :]
        player_idx_broad = nearest_lane_idx
        
        should_drop_right = (side_broad > 0) & (lane_indices >= player_idx_broad)
        should_drop_left = (side_broad < 0) & (lane_indices <= player_idx_broad)
        
        # Force drop at boundaries if somehow passed? 
        # Actually, "latest at lane 5" means if we are AT lane 5 (index 4) and moving right, we MUST drop.
        # indices >= player_idx includes 4 (since player_idx <= 4).
        # So checking >= player is sufficient to cover "next lane" and "last lane".
        
        is_valid_drop_lane = should_drop_right | should_drop_left
        
        # Valid trigger mask: (5, N)
        trigger_mask = is_aligned & is_valid_drop_lane
        
        # Check if ANY lane triggers for each meteoroid
        should_descend = jnp.any(trigger_mask, axis=0)
        
        # Identify WHICH lane to drop on.
        # If multiple align (unlikely), pick the one that triggered.
        # We can use argmax to get the index. 
        # If none, argmax returns 0, but should_descend will be False so it doesn't matter.
        target_lane_idx_0_4 = jnp.argmax(trigger_mask, axis=0).astype(jnp.int32)
        
        # Map back to 1-5
        chosen_lane = target_lane_idx_0_4 + 1
        
        # State transition
        # If active & phase==0 & should_descend -> Start Phase 2 directly?
        # Or Phase 1 then 2? 
        # Original code: Phase 0 -> Phase 1 (Arm) -> Phase 2 (Descend if on lane).
        # Since we confirmed "is_aligned" above, we are "on lane". We can descend immediately.
        
        start_descend_now = active & (phase == 0) & should_descend
        
        new_phase = jnp.where(start_descend_now, 2, phase)
        new_lane = jnp.where(start_descend_now, chosen_lane, lane)
        
        # Update target X to lock onto the chosen lane center perfectly (optional but cleaner)
        # target_lane_x_final = jnp.take_along_axis(lane_x_at_current_y, new_lane[None, :], axis=0).squeeze(0)
        # new_x = jnp.where(start_descend_now, target_lane_x_final, new_x) 
        # (For now keeping original movement logic to preserve momentum/physics unless requested otherwise)

        # Legacy compatibility: if phase was 1, check if aligned with *current* lane
        # lane is 1-5. Index is lane-1.
        current_lane_idx = jnp.clip(lane - 1, 0, 4)
        current_lane_x = playable_lanes_x[current_lane_idx, jnp.arange(self.consts.CHASING_METEOROID_MAX)]
        is_on_current_lane = jnp.abs(current_lane_x - new_x_a) <= float(self.consts.CHASING_METEOROID_LANE_ALIGN_THRESHOLD)

        start_descend = start_descend_now | ((phase == 1) & is_on_current_lane) 
        new_phase = jnp.where(start_descend, 2, new_phase)
        
        # When starting descend, initialize velocity
        new_vel_y = jnp.where(start_descend, float(self.consts.CHASING_METEOROID_LANE_SPEED), vel_y)
        # Apply acceleration if already descending
        new_vel_y = jnp.where(phase_descend, vel_y + self.consts.CHASING_METEOROID_ACCEL, new_vel_y)

        new_frame = jnp.where(phase_horizontal, (frame + 1) % 8, frame)
        new_frame = jnp.where(start_descend, 0, new_frame)

        new_pos = jnp.stack([new_x, new_y])
        offscreen = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=pos.dtype)
        new_pos = jnp.where(active[None, :], new_pos, offscreen[:, None])

        out_of_bounds = jnp.logical_and(
            active,
            jnp.logical_or(new_x < 0.0, new_x > float(self.consts.SCREEN_WIDTH)),
        )
        new_active = jnp.where(out_of_bounds, False, active)
        new_pos = jnp.where(new_active[None, :], new_pos, offscreen[:, None])
        new_vel_y = jnp.where(new_active, new_vel_y, 0.0)
        new_phase = jnp.where(new_active, new_phase, 0)
        new_frame = jnp.where(new_active, new_frame, 0)
        new_lane = jnp.where(new_active, new_lane, 0)
        new_side = jnp.where(new_active, side, 1)

        return (
            new_pos,
            new_active,
            new_vel_y,
            new_phase,
            new_frame,
            new_lane,
            new_side,
            spawn_timer,
            remaining,
            wave_active,
        )

    def _chasing_meteoroid_bullet_collision(
        self,
        chasing_meteoroid_pos: chex.Array,
        chasing_meteoroid_active: chex.Array,
        chasing_meteoroid_vel_y: chex.Array,
        chasing_meteoroid_phase: chex.Array,
        chasing_meteoroid_frame: chex.Array,
        chasing_meteoroid_lane: chex.Array,
        chasing_meteoroid_side: chex.Array,
        player_shot_pos: chex.Array,
        player_shot_vel: chex.Array,
        bullet_type: chex.Array,
        white_ufo_left: chex.Array,
    ):
        is_torpedo = bullet_type == self.consts.TORPEDO_ID
        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        chasing_meteoroid_x = chasing_meteoroid_pos[0] + _get_ufo_alignment(chasing_meteoroid_pos[1]).astype(chasing_meteoroid_pos.dtype)
        chasing_meteoroid_y = chasing_meteoroid_pos[1]

        bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
        
        meteoroid_size = jnp.array(self.consts.METEOROID_SPRITE_SIZE)
        
        # AABB collision check
        collision_mask = (
            chasing_meteoroid_active
            & shot_active
            & (chasing_meteoroid_x < shot_x + bullet_size[1]) & (shot_x < chasing_meteoroid_x + meteoroid_size[1])
            & (chasing_meteoroid_y < shot_y + bullet_size[0]) & (shot_y < chasing_meteoroid_y + meteoroid_size[0])
        )
        
        # Meteoroid is only destroyed if it's hit by a torpedo
        hit_mask = collision_mask & is_torpedo
        
        hit_exists_meteoroid = jnp.any(hit_mask)
        collision_exists = jnp.any(collision_mask)
        
        hit_index = jnp.argmax(hit_mask)
        hit_one_hot = jax.nn.one_hot(hit_index, self.consts.CHASING_METEOROID_MAX, dtype=chasing_meteoroid_pos.dtype)
        hit_one_hot_bool = hit_one_hot.astype(jnp.bool_)

        offscreen = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=chasing_meteoroid_pos.dtype)
        pos_after_hit = chasing_meteoroid_pos + (offscreen[:, None] - chasing_meteoroid_pos) * hit_one_hot[None, :]
        active_after_hit = jnp.where(hit_one_hot_bool, False, chasing_meteoroid_active)
        vel_y_after_hit = jnp.where(hit_one_hot_bool, 0.0, chasing_meteoroid_vel_y)
        phase_after_hit = jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_phase)
        frame_after_hit = jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_frame)
        lane_after_hit = jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_lane)
        side_after_hit = jnp.where(hit_one_hot_bool, 1, chasing_meteoroid_side)

        chasing_meteoroid_pos = jnp.where(hit_exists_meteoroid, pos_after_hit, chasing_meteoroid_pos)
        chasing_meteoroid_active = jnp.where(hit_exists_meteoroid, active_after_hit, chasing_meteoroid_active)
        chasing_meteoroid_vel_y = jnp.where(hit_exists_meteoroid, vel_y_after_hit, chasing_meteoroid_vel_y)
        chasing_meteoroid_phase = jnp.where(hit_exists_meteoroid, phase_after_hit, chasing_meteoroid_phase)
        chasing_meteoroid_frame = jnp.where(hit_exists_meteoroid, frame_after_hit, chasing_meteoroid_frame)
        chasing_meteoroid_lane = jnp.where(hit_exists_meteoroid, lane_after_hit, chasing_meteoroid_lane)
        chasing_meteoroid_side = jnp.where(hit_exists_meteoroid, side_after_hit, chasing_meteoroid_side)
        
        # Shot is removed if it hits a meteoroid (blocking behavior)
        player_shot_pos = jnp.where(
            collision_exists,
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=player_shot_pos.dtype),
            player_shot_pos,
        )

        return (
            chasing_meteoroid_pos,
            chasing_meteoroid_active,
            chasing_meteoroid_vel_y,
            chasing_meteoroid_phase,
            chasing_meteoroid_frame,
            chasing_meteoroid_lane,
            chasing_meteoroid_side,
            player_shot_pos,
            hit_mask,
            hit_exists_meteoroid,
            collision_exists
        )

    def _rejuvenator_step(self, state: BeamriderState, key: chex.Array):
        pos = state.level.rejuvenator_pos
        active = state.level.rejuvenator_active
        dead = state.level.rejuvenator_dead
        frame = state.level.rejuvenator_frame
        lane = state.level.rejuvenator_lane

        key_spawn, key_lane = jax.random.split(key)
        
        # Spawning logic: only spawn if not already active
        spawn_roll = jax.random.uniform(key_spawn)
        should_spawn = jnp.logical_and.reduce(jnp.array([
            jnp.logical_not(active),
            spawn_roll < self.consts.REJUVENATOR_SPAWN_PROB,
            state.level.white_ufo_left > 0
        ]))
        
        # Lanes 1 to 5
        spawn_lane = jax.random.randint(key_lane, (), 1, 6)
        
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        spawn_x = lanes_top_x[spawn_lane]
        spawn_y = float(self.consts.TOP_CLIP)
        
        pos = jnp.where(should_spawn, jnp.array([spawn_x, spawn_y]), pos)
        lane = jnp.where(should_spawn, spawn_lane, lane)
        active = jnp.where(should_spawn, True, active)
        dead = jnp.where(should_spawn, False, dead)
        frame = jnp.where(should_spawn, 0, frame)

        # Movement logic: 
        # Phase 1: pixel moves alternate 2 and 4 frames (dy=1)
        # Phase 2: moves every 2 frames (dy=1)
        # Phase 3: moves every 2 frames, cyclic dy [1, 1, 2]
        # Phase 4: moves every 2 frames, cyclic dy [2, 3]
        # Dead: always every frame (dy based on current stage or 1.0)
        y = pos[1]
        stage = _get_index_rejuvenator(y)
        
        should_move_normal = jax.lax.switch(
            jnp.clip(stage - 1, 0, 3),
            [
                lambda: (state.steps % 6 == 0) | (state.steps % 6 == 2), # Phase 1
                lambda: (state.steps % 2) == 0,                         # Phase 2
                lambda: (state.steps % 2) == 0,                         # Phase 3
                lambda: (state.steps % 2) == 0,                         # Phase 4
            ]
        )
        
        should_move = jnp.logical_and(active, jnp.where(dead, True, should_move_normal))
        
        # dy logic:
        dy = jax.lax.switch(
            jnp.clip(stage - 1, 0, 3),
            [
                lambda: 1.0,                                       # Phase 1
                lambda: 1.0,                                       # Phase 2
                lambda: jnp.take(jnp.array([1.0, 1.0, 2.0]), frame % 3), # Phase 3: 1, 1, 2
                lambda: jnp.take(jnp.array([2.0, 3.0]), frame % 2),     # Phase 4: 2, 3
            ]
        )
        
        new_y = y + jnp.where(should_move, dy, 0.0)
        
        # Update X based on lane
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]
        
        new_x = jnp.take(lanes_top_x, lane) + jnp.take(lane_dx_over_dy, lane) * (new_y - float(self.consts.TOP_CLIP))
        
        pos = jnp.where(active, jnp.array([new_x, new_y]), pos)
        frame = jnp.where(should_move, frame + 1, frame)
        
        # Deactivate if off-screen
        off_screen = new_y > self.consts.PLAYER_POS_Y + 1.0
        active = jnp.where(off_screen, False, active)
        offscreen_pos = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32)
        pos = jnp.where(active, jnp.array([new_x, new_y]), offscreen_pos)
        
        return pos, active, dead, frame, lane

    def _falling_rock_step(self, state: BeamriderState, key: chex.Array):
        pos = state.level.falling_rock_pos
        active = state.level.falling_rock_active
        lane = state.level.falling_rock_lane
        vel_y = state.level.falling_rock_vel_y

        key_spawn, key_lane = jax.random.split(key)

        # Spawning logic: Level 2 onwards
        is_level_2_plus = state.sector >= 2
        spawn_roll = jax.random.uniform(key_spawn)
        can_spawn = jnp.logical_and.reduce(jnp.array([
            is_level_2_plus,
            jnp.sum(active.astype(jnp.int32)) < self.consts.FALLING_ROCK_MAX,
            state.level.white_ufo_left > 0
        ]))
        should_spawn = jnp.logical_and(can_spawn, spawn_roll < self.consts.FALLING_ROCK_SPAWN_PROB)

        # Random lane from 1 to 5 (inner lanes)
        spawn_lane = jax.random.randint(key_lane, (), 1, 6)

        # Find first inactive slot
        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask.astype(jnp.int32))
        one_hot = jax.nn.one_hot(slot, self.consts.FALLING_ROCK_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        spawn_x = lanes_top_x[spawn_lane]
        spawn_y = float(self.consts.FALLING_ROCK_SPAWN_Y)
        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)

        pos = jnp.where(should_spawn, pos + (spawn_pos[:, None] - pos) * one_hot[None, :], pos)
        active = jnp.where(should_spawn, jnp.where(one_hot_bool, True, active), active)
        lane = jnp.where(should_spawn, jnp.where(one_hot_bool, spawn_lane, lane), lane)
        vel_y = jnp.where(should_spawn, jnp.where(one_hot_bool, self.consts.FALLING_ROCK_INIT_VEL, vel_y), vel_y)

        # Movement logic: Stage-based acceleration
        # Slower during sprite 1 (y < 64) and 2 (y < 85)
        y = pos[1]
        accel = jnp.where(y < 64, 0.004, 
                jnp.where(y < 85, 0.008, 
                self.consts.FALLING_ROCK_ACCEL))
        
        new_vel_y = vel_y + accel
        new_y = y + new_vel_y
        
        # Update X to stay centered on lane
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]
        
        target_lane_dx_over_dy = jnp.take(lane_dx_over_dy, lane)
        target_lanes_top_x = jnp.take(lanes_top_x, lane)
        new_x = target_lanes_top_x + target_lane_dx_over_dy * (new_y - float(self.consts.TOP_CLIP))
        
        pos = jnp.where(active[None, :], jnp.stack([new_x, new_y]), pos)
        vel_y = jnp.where(active, new_vel_y, vel_y)
        
        # Deactivate if off-screen
        off_screen = new_y > self.consts.FALLING_ROCK_BOTTOM_CLIP
        active = jnp.where(off_screen, False, active)
        offscreen_pos = jnp.tile(jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1), (1, self.consts.FALLING_ROCK_MAX))
        pos = jnp.where(active[None, :], pos, offscreen_pos)
        vel_y = jnp.where(active, vel_y, 0.0)
        
        return pos, active, lane, vel_y

    def _coin_step(self, state: BeamriderState, key: chex.Array):
        pos = state.level.coin_pos
        active = state.level.coin_active
        timer = state.level.coin_timer
        side = state.level.coin_side
        spawn_count = state.level.coin_spawn_count

        key_spawn, key_side = jax.random.split(key)

        # Spawning logic: Sector 4 onwards
        is_sector_4_plus = state.sector >= 4
        can_spawn = jnp.logical_and.reduce(jnp.array([
            is_sector_4_plus,
            jnp.sum(active.astype(jnp.int32)) < self.consts.COIN_MAX,
            state.level.white_ufo_left > 0,
        ]))
        should_spawn = jnp.logical_and(can_spawn, jax.random.uniform(key_spawn) < self.consts.COIN_SPAWN_PROB)

        # 50/50 chance for side choice: 0 for Left, 1 for Right
        spawn_side_idx = (jax.random.uniform(key_side) < 0.5).astype(jnp.int32)
        spawn_x = jnp.where(spawn_side_idx == 0, self.consts.COIN_SPAWN_X_LEFT, self.consts.COIN_SPAWN_X_RIGHT)
        spawn_y = self.consts.COIN_SPAWN_Y
        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)

        # Find first inactive slot
        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask.astype(jnp.int32))
        one_hot = jax.nn.one_hot(slot, self.consts.COIN_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        pos = jnp.where(should_spawn, pos + (spawn_pos[:, None] - pos) * one_hot[None, :], pos)
        active = jnp.where(should_spawn, jnp.where(one_hot_bool, True, active), active)
        side = jnp.where(should_spawn, jnp.where(one_hot_bool, spawn_side_idx, side), side)
        timer = jnp.where(should_spawn, jnp.where(one_hot_bool, 0, timer), timer)
        spawn_count = jnp.where(should_spawn, spawn_count + 1, spawn_count)

        # Movement logic: every second frame
        should_move = (state.steps % 2 == 0)
        
        # side == 0 (Left start) -> moves Right (DX > 0)
        # side == 1 (Right start) -> moves Left (DX < 0)
        dx = jnp.where(side == 0, self.consts.COIN_SPEED_X, -self.consts.COIN_SPEED_X)
        dy = self.consts.COIN_SPEED_Y
        
        new_x = pos[0] + jnp.where(should_move, dx, 0.0)
        new_y = pos[1] + jnp.where(should_move, dy, 0.0)
        
        pos = jnp.where(active[None, :], jnp.stack([new_x, new_y]), pos)
        timer = jnp.where(active, timer + 1, timer) # timer increments every frame for animation

        # Deactivate if off-screen (Y >= 95)
        off_screen = pos[1] >= self.consts.COIN_EXIT_Y
        active = jnp.where(off_screen, False, active)
        offscreen_pos = jnp.tile(jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1), (1, self.consts.COIN_MAX))
        pos = jnp.where(active[None, :], pos, offscreen_pos)
        
        return pos, active, timer, side, spawn_count

    def _coin_bullet_collision(
        self,
        coin_pos: chex.Array,
        coin_active: chex.Array,
        player_shot_pos: chex.Array,
        bullet_type: chex.Array,
    ):
        shot_x = player_shot_pos[0] + _get_bullet_alignment(
            player_shot_pos[1], bullet_type, self.consts.LASER_ID
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        coin_x_screen = coin_pos[0] + _get_ufo_alignment(coin_pos[1]).astype(coin_pos.dtype)
        coin_y = coin_pos[1]

        bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
        
        coin_size = jnp.array(self.consts.COIN_SPRITE_SIZE)
        
        # AABB collision check
        hit_mask = (
            coin_active
            & shot_active
            & (coin_x_screen < shot_x + bullet_size[1]) & (shot_x < coin_x_screen + coin_size[1])
            & (coin_y < shot_y + bullet_size[0]) & (shot_y < coin_y + coin_size[0])
        )
        hit_exists = jnp.any(hit_mask)
        
        return hit_mask, hit_exists

    def _falling_rock_bullet_collision(
        self,
        falling_rock_pos: chex.Array,
        falling_rock_active: chex.Array,
        player_shot_pos: chex.Array,
        player_shot_vel: chex.Array,
        bullet_type: chex.Array,
        white_ufo_left: chex.Array,
    ):
        is_torpedo = bullet_type == self.consts.TORPEDO_ID
        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        rock_x = falling_rock_pos[0] + _get_ufo_alignment(falling_rock_pos[1]).astype(falling_rock_pos.dtype)
        rock_y = falling_rock_pos[1]

        bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
        
        rock_indices = jnp.clip(_get_index_falling_rock(rock_y) - 1, 0, len(self.consts.FALLING_ROCK_SPRITE_SIZES) - 1)
        rock_sizes = jnp.take(jnp.array(self.consts.FALLING_ROCK_SPRITE_SIZES), rock_indices, axis=0)
        
        # AABB collision check
        hit_mask = (
            falling_rock_active
            & shot_active
            & (rock_x < shot_x + bullet_size[1]) & (shot_x < rock_x + rock_sizes[:, 1])
            & (rock_y < shot_y + bullet_size[0]) & (shot_y < rock_y + rock_sizes[:, 0])
        )
        hit_exists_rock = jnp.any(hit_mask)
        
        # Rock is only destroyed by torpedo
        rock_should_die = hit_mask & is_torpedo
        
        offscreen = jnp.tile(jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=falling_rock_pos.dtype).reshape(2, 1), (1, self.consts.FALLING_ROCK_MAX))
        pos = jnp.where(rock_should_die[None, :], offscreen, falling_rock_pos)
        active = jnp.where(rock_should_die, False, falling_rock_active)
        
        player_shot_pos = jnp.where(
            hit_exists_rock,
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=player_shot_pos.dtype),
            player_shot_pos,
        )

        return pos, active, player_shot_pos, rock_should_die, hit_exists_rock

    def _lane_blocker_step(self, state: BeamriderState, key: chex.Array):
        pos = state.level.lane_blocker_pos
        active = state.level.lane_blocker_active
        lane = state.level.lane_blocker_lane
        vel_y = state.level.lane_blocker_vel_y
        phase = state.level.lane_blocker_phase
        hold_timer = state.level.lane_blocker_timer

        key_spawn, key_lane = jax.random.split(key)

        # Spawning logic: Level 10 onwards
        is_level_10_plus = state.sector >= self.consts.LANE_BLOCKER_START_LEVEL
        spawn_roll = jax.random.uniform(key_spawn)
        can_spawn = jnp.logical_and.reduce(jnp.array([
            is_level_10_plus,
            jnp.sum(active.astype(jnp.int32)) < self.consts.LANE_BLOCKER_MAX,
            state.level.white_ufo_left > 0
        ]))
        should_spawn = jnp.logical_and(can_spawn, spawn_roll < self.consts.LANE_BLOCKER_SPAWN_PROB)

        # Random lane from 1 to 5 (inner lanes)
        spawn_lane = jax.random.randint(key_lane, (), 1, 6)

        # Find first inactive slot
        inactive_mask = jnp.logical_not(active)
        slot = jnp.argmax(inactive_mask.astype(jnp.int32))
        one_hot = jax.nn.one_hot(slot, self.consts.LANE_BLOCKER_MAX, dtype=pos.dtype)
        one_hot_bool = one_hot.astype(jnp.bool_)

        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        spawn_x = lanes_top_x[spawn_lane]
        spawn_y = float(self.consts.LANE_BLOCKER_SPAWN_Y)
        spawn_pos = jnp.array([spawn_x, spawn_y], dtype=pos.dtype)

        pos = jnp.where(should_spawn, pos + (spawn_pos[:, None] - pos) * one_hot[None, :], pos)
        active = jnp.where(should_spawn, jnp.where(one_hot_bool, True, active), active)
        lane = jnp.where(should_spawn, jnp.where(one_hot_bool, spawn_lane, lane), lane)
        vel_y = jnp.where(should_spawn, jnp.where(one_hot_bool, self.consts.LANE_BLOCKER_INIT_VEL, vel_y), vel_y)
        phase = jnp.where(should_spawn, jnp.where(one_hot_bool, int(LaneBlockerState.DESCEND), phase), phase)
        hold_timer = jnp.where(should_spawn, jnp.where(one_hot_bool, 0, hold_timer), hold_timer)

        y = pos[1]
        bottom_y = float(self.consts.LANE_BLOCKER_BOTTOM_Y)

        descend = active & (phase == int(LaneBlockerState.DESCEND))
        hold = active & (phase == int(LaneBlockerState.HOLD))
        sink = active & (phase == int(LaneBlockerState.SINK))
        retreat = active & (phase == int(LaneBlockerState.RETREAT))

        accel = jnp.where(y < 64, 0.004,
                jnp.where(y < 85, 0.008,
                self.consts.FALLING_ROCK_ACCEL))
        new_vel_y = vel_y + accel
        new_y_descend = y + new_vel_y
        reached_bottom = new_y_descend >= bottom_y
        y_descend = jnp.where(reached_bottom, bottom_y, new_y_descend)
        vel_descend = jnp.where(reached_bottom, 0.0, new_vel_y)
        phase_descend = jnp.where(reached_bottom, int(LaneBlockerState.HOLD), int(LaneBlockerState.DESCEND))
        hold_timer_descend = jnp.where(reached_bottom, self.consts.LANE_BLOCKER_HOLD_FRAMES, hold_timer)

        hold_timer_next = jnp.maximum(hold_timer - 1, 0)
        start_sink = hold_timer_next == 0
        phase_hold = jnp.where(start_sink, int(LaneBlockerState.SINK), int(LaneBlockerState.HOLD))

        sink_step = (state.steps % self.consts.LANE_BLOCKER_SINK_INTERVAL) == 0
        sink_dy = jnp.where(sink_step, 1.0, 0.0)
        y_sink = y + sink_dy

        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        retreat_speed = lane_vectors[:, 1] * self.consts.WHITE_UFO_SPEED_FACTOR * self.consts.WHITE_UFO_RETREAT_SPEED_MULT
        retreat_speed = retreat_speed * self.consts.LANE_BLOCKER_RETREAT_SPEED_MULT
        lane_retreat_speed = jnp.take(retreat_speed, lane)
        y_retreat = y - lane_retreat_speed

        new_y = jnp.where(descend, y_descend, y)
        new_y = jnp.where(hold, bottom_y, new_y)
        new_y = jnp.where(sink, y_sink, new_y)
        new_y = jnp.where(retreat, y_retreat, new_y)

        new_vel_y = vel_y
        new_vel_y = jnp.where(descend, vel_descend, new_vel_y)
        new_vel_y = jnp.where(hold | sink | retreat, 0.0, new_vel_y)

        new_phase = phase
        new_phase = jnp.where(descend, phase_descend, new_phase)
        new_phase = jnp.where(hold, phase_hold, new_phase)
        new_phase = jnp.where(sink, int(LaneBlockerState.SINK), new_phase)
        new_phase = jnp.where(retreat, int(LaneBlockerState.RETREAT), new_phase)

        new_timer = hold_timer
        new_timer = jnp.where(descend, hold_timer_descend, new_timer)
        new_timer = jnp.where(hold, hold_timer_next, new_timer)
        new_timer = jnp.where(sink | retreat, 0, new_timer)

        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]
        target_lane_dx_over_dy = jnp.take(lane_dx_over_dy, lane)
        target_lanes_top_x = jnp.take(lanes_top_x, lane)
        new_x = target_lanes_top_x + target_lane_dx_over_dy * (new_y - float(self.consts.TOP_CLIP))

        offscreen_pos = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, self.consts.LANE_BLOCKER_MAX),
        )
        retreat_done = retreat & (new_y <= float(self.consts.LANE_BLOCKER_SPAWN_Y))
        sink_done = sink & ((new_y - bottom_y) >= float(self.consts.LANE_BLOCKER_HEIGHT))
        done = retreat_done | sink_done

        active = jnp.where(done, False, active)
        new_phase = jnp.where(done, int(LaneBlockerState.DESCEND), new_phase)
        new_timer = jnp.where(done, 0, new_timer)
        new_vel_y = jnp.where(done, 0.0, new_vel_y)

        pos = jnp.where(active[None, :], jnp.stack([new_x, new_y]), offscreen_pos)
        new_vel_y = jnp.where(active, new_vel_y, 0.0)

        return pos, active, lane, new_vel_y, new_phase, new_timer

    def _lane_blocker_bullet_collision(
        self,
        lane_blocker_pos: chex.Array,
        lane_blocker_active: chex.Array,
        lane_blocker_phase: chex.Array,
        lane_blocker_timer: chex.Array,
        lane_blocker_vel_y: chex.Array,
        player_shot_pos: chex.Array,
        player_shot_vel: chex.Array,
        bullet_type: chex.Array,
        white_ufo_left: chex.Array,
    ):
        is_torpedo = bullet_type == self.consts.TORPEDO_ID
        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        blocker_x = lane_blocker_pos[0] + _get_ufo_alignment(lane_blocker_pos[1]).astype(lane_blocker_pos.dtype)
        blocker_y = lane_blocker_pos[1]

        bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)

        blocker_indices = jnp.clip(_get_index_lane_blocker(blocker_y) - 1, 0, len(self.consts.LANE_BLOCKER_SPRITE_SIZES) - 1)
        blocker_indices = jnp.where(blocker_indices == 2, 3, blocker_indices)
        blocker_sizes = jnp.take(jnp.array(self.consts.LANE_BLOCKER_SPRITE_SIZES), blocker_indices, axis=0)

        hit_mask = (
            lane_blocker_active
            & shot_active
            & (blocker_x < shot_x + bullet_size[1]) & (shot_x < blocker_x + blocker_sizes[:, 1])
            & (blocker_y < shot_y + bullet_size[0]) & (shot_y < blocker_y + blocker_sizes[:, 0])
        )
        hit_exists = jnp.any(hit_mask)

        blocker_destroyed = hit_mask & is_torpedo
        blocker_retreat = hit_mask & jnp.logical_not(is_torpedo)

        offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=lane_blocker_pos.dtype).reshape(2, 1),
            (1, self.consts.LANE_BLOCKER_MAX),
        )
        pos = jnp.where(blocker_destroyed[None, :], offscreen, lane_blocker_pos)
        active = jnp.where(blocker_destroyed, False, lane_blocker_active)

        phase = jnp.where(blocker_retreat, int(LaneBlockerState.RETREAT), lane_blocker_phase)
        phase = jnp.where(blocker_destroyed, int(LaneBlockerState.DESCEND), phase)
        timer = jnp.where(blocker_retreat, 0, lane_blocker_timer)
        timer = jnp.where(blocker_destroyed, 0, timer)
        vel_y = jnp.where(blocker_retreat, 0.0, lane_blocker_vel_y)
        vel_y = jnp.where(blocker_destroyed, 0.0, vel_y)

        player_shot_pos = jnp.where(
            hit_exists,
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=player_shot_pos.dtype),
            player_shot_pos,
        )

        return pos, active, phase, timer, vel_y, player_shot_pos, blocker_destroyed, hit_exists

    def _enemy_shot_bullet_collision(
        self,
        enemy_shot_pos: chex.Array,
        enemy_shot_timer: chex.Array,
        player_shot_pos: chex.Array,
        player_shot_vel: chex.Array,
        bullet_type: chex.Array,
    ):
        is_torpedo = bullet_type == self.consts.TORPEDO_ID
        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        enemy_shot_x = enemy_shot_pos[0, :] + _get_ufo_alignment(enemy_shot_pos[1, :])
        enemy_shot_y = enemy_shot_pos[1, :]
        
        sprite_idx = (jnp.floor_divide(enemy_shot_timer, 4) % 2).astype(jnp.int32)
        enemy_shot_sizes = jnp.take(jnp.array(self.consts.ENEMY_SHOT_SPRITE_SIZES), sprite_idx, axis=0)

        bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
        
        # AABB collision check
        shot_on_screen = enemy_shot_y <= float(self.consts.BOTTOM_CLIP)
        
        hit_mask = (
            shot_on_screen
            & shot_active
            & (enemy_shot_x < shot_x + bullet_size[1]) & (shot_x < enemy_shot_x + enemy_shot_sizes[:, 1])
            & (enemy_shot_y < shot_y + bullet_size[0]) & (shot_y < enemy_shot_y + enemy_shot_sizes[:, 0])
        )
        
        # Only destroyed by torpedo
        hit_mask_torpedo = hit_mask & is_torpedo
        hit_exists = jnp.any(hit_mask_torpedo)
        
        return hit_mask_torpedo, hit_exists

    def _mothership_step(self, state: BeamriderState, white_ufo_left: chex.Array, enemy_explosion_frame: chex.Array, is_hit: chex.Array):
        """Spawn and move the mothership once all white UFOs are cleared."""
        stage = state.level.mothership_stage
        timer = state.level.mothership_timer
        pos_x = state.level.mothership_position
        sector = state.sector
        
        is_ltr = (sector % 2) != 0

        def idle_logic():
            explosions_finished = jnp.all(enemy_explosion_frame == 0)
            start = (white_ufo_left == 0) & explosions_finished
            # Start at stage 1, timer 1
            return jnp.where(start, 1, 0), jnp.where(start, 1, 0), pos_x.astype(jnp.float32)

        def emergence_logic():
            next_timer = timer + 1
            finished = next_timer > 15
            
            # Calculate pos_x based on timer (2..15)
            s = jnp.clip((timer - 1) // 2, 0, 6)
            rel_x = jnp.take(jnp.array(self.consts.MOTHERSHIP_ANIM_X), s)
            calculated_pos = jnp.where(is_ltr, rel_x.astype(jnp.float32), (160 - 16 - rel_x + 8).astype(jnp.float32))
            
            return jnp.where(finished, 2, 1), jnp.where(finished, 1, next_timer), calculated_pos

        def moving_logic():
            # Interval 2, 4, 2, 4...
            # timer=2: move (after 2)
            # timer=6: move (after 4)
            # timer=8: move (after 2)
            # timer=12: move (after 4)
            # should_move if timer % 6 in {2, 0}
            should_move = (timer % 6 == 2) | (timer % 6 == 0)
            dx = jnp.where(is_ltr, 1.0, -1.0)
            next_pos_x = pos_x + jnp.where(should_move, dx, 0.0)
            
            stable_rel_x = jnp.array(self.consts.MOTHERSHIP_ANIM_X)[6].astype(jnp.float32)
            target_x = jnp.where(is_ltr, 160.0 - 16.0 - stable_rel_x + 8.0, stable_rel_x)
            reached = jnp.where(is_ltr, next_pos_x >= target_x, next_pos_x <= target_x)
            
            next_stage = jnp.where(reached, 3, 2)
            next_timer = jnp.where(reached, 1, timer + 1)
            
            # Transition to explosion if hit
            final_stage = jnp.where(is_hit, 5, next_stage)
            final_timer = jnp.where(is_hit, 0, next_timer)
            
            return final_stage, final_timer, next_pos_x

        def descending_logic():
            next_timer = timer + 1
            finished = next_timer > 15
            
            # Calculate pos_x based on timer (1..14)
            s = jnp.clip(6 - (timer - 1) // 2, 0, 6)
            rel_x = jnp.take(jnp.array(self.consts.MOTHERSHIP_ANIM_X), s)
            # During descending, we are at the opposite side from where we started
            # If we started LTR (Left), we are now at Right.
            calculated_pos = jnp.where(is_ltr, (160 - 16 - rel_x + 8).astype(jnp.float32), rel_x.astype(jnp.float32))
            
            return jnp.where(finished, 4, 3), jnp.where(finished, 0, next_timer), calculated_pos

        def done_logic():
            return 0, 0, jnp.array(self.consts.MOTHERSHIP_OFFSCREEN_POS, dtype=jnp.float32)
            
        def exploding_logic():
            # 9 steps * 8 frames = 72 frames total
            duration = 9 * self.consts.MOTHERSHIP_EXPLOSION_STEP_DURATION
            finished = timer >= duration
            return jnp.where(finished, 4, 5), timer + 1, pos_x

        new_stage, new_timer, new_pos = jax.lax.switch(
            stage.astype(jnp.int32),
            [idle_logic, emergence_logic, moving_logic, descending_logic, done_logic, exploding_logic]
        )
        
        # Advance sector when mothership finishes descending (3->4) OR finishes exploding (5->4)
        sector_advance = (new_stage == 4)
        
        return new_pos, new_timer, new_stage, sector_advance



    def _line_step(self, state: BeamriderState):
        counter = state.level.blue_line_counter + 1
        
        # Determine current table and index
        # Transition from INIT to LOOP
        # Use length-1 to avoid out of bounds during the very last frame of init
        is_init = counter < len(BLUE_LINE_INIT_TABLE)
        
        def get_init_pos(c):
            return BLUE_LINE_INIT_TABLE[jnp.minimum(c, len(BLUE_LINE_INIT_TABLE) - 1)]
            
        def get_loop_pos(c):
            loop_idx = (c - len(BLUE_LINE_INIT_TABLE)) % len(BLUE_LINE_LOOP_TABLE)
            return BLUE_LINE_LOOP_TABLE[loop_idx]
            
        positions = jax.lax.cond(is_init, get_init_pos, get_loop_pos, counter)
        
        return positions, counter

    def _bullet_infos(self, state: BeamriderState):
        shot_frame = state.level.player_shot_frame
        bullet_type = state.level.bullet_type

        is_laser = bullet_type == self.consts.LASER_ID
        max_frames = jnp.where(is_laser, LASER_PROJECTILE_FRAMES, TORPEDO_PROJECTILE_FRAMES)
        return (shot_frame >= 0) & (shot_frame < max_frames)

    def _projectile_resolved(self, state: BeamriderState):
        """Check if the active projectile just finished its lifecycle (horizon)."""
        shot_frame = state.level.player_shot_frame
        bullet_type = state.level.bullet_type
        is_laser = bullet_type == self.consts.LASER_ID
        max_frames = jnp.where(is_laser, LASER_PROJECTILE_FRAMES, TORPEDO_PROJECTILE_FRAMES)
        return (shot_frame >= 0) & (shot_frame >= (max_frames - 1))

    def _bouncer_dedicated_step(self, state: BeamriderState, key: chex.Array):
        level = state.level
        pos = level.bouncer_pos
        vel = level.bouncer_vel
        lane_idx = level.bouncer_lane
        step_index = level.bouncer_step_index
        state_id = level.bouncer_state
        timer = level.bouncer_timer
        active = level.bouncer_active

        def switching_logic(p, l_idx, s_idx, direction):
            is_odd = (s_idx % 2) != 0
            dy = jnp.where(is_odd, 1.0, 0.0)

            target_lane = jnp.where(l_idx == -1, 1,
                                    jnp.where(l_idx == 7, 5,
                                              jnp.clip(l_idx + direction.astype(jnp.int32), 0, 6)))

            lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
            lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
            lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

            target_x = lanes_top_x[target_lane] + lane_dx_over_dy[target_lane] * (p[1] - float(self.consts.TOP_CLIP))

            dist_x = target_x - p[0]
            dx = jnp.clip(dist_x, -self.consts.BOUNCER_SWITCH_SPEED_X, self.consts.BOUNCER_SWITCH_SPEED_X)

            finished_switching = jnp.abs(dist_x) < 1.0

            next_state = jnp.where(finished_switching, int(BouncerState.DOWN), int(BouncerState.SWITCHING))
            next_s_idx = jnp.where(finished_switching, 0, s_idx + 1)
            next_lane = jnp.where(finished_switching, target_lane, l_idx)

            return dx, dy, next_state, next_s_idx, next_lane

        def lane_logic(p, l_idx, s_idx, is_down):
            speed_pattern = jnp.array(self.consts.BOUNCER_SPEED_PATTERN)
            speed = speed_pattern[s_idx % 4]

            lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
            safe_lane = jnp.clip(l_idx, 0, 6)
            vec = lane_vectors[safe_lane]

            vy = speed.astype(jnp.float32)
            vx = vy * (vec[0] / vec[1])

            vy = jnp.where(is_down, vy, -vy)
            vx = jnp.where(is_down, vx, -vx)

            can_bounce = (safe_lane > 0) & (safe_lane < 6)
            limit_y = jnp.where(can_bounce, float(self.consts.PLAYER_POS_Y), 220.0)

            hit_bottom = (p[1] >= limit_y) & is_down
            hit_top = (p[1] <= self.consts.BOUNCER_SPAWN_HEIGHT) & (jnp.logical_not(is_down))

            next_state = jnp.where(is_down,
                                   jnp.where(hit_bottom, int(BouncerState.UP), int(BouncerState.DOWN)),
                                   jnp.where(hit_top, int(BouncerState.SWITCHING), int(BouncerState.UP)))

            next_s_idx = jnp.where(hit_top & (jnp.logical_not(is_down)), 0, s_idx + 1)
            next_state = jnp.where(jnp.logical_not(can_bounce) & is_down, int(BouncerState.DOWN), next_state)

            return vx, vy, next_state, next_s_idx, l_idx

        def move_active(_):
            direction = jnp.where(timer == 0, jnp.sign(vel[0]), timer.astype(jnp.float32))
            direction = jnp.where(direction == 0, 1.0, direction)
            direction = jnp.sign(direction)

            is_switching = state_id == int(BouncerState.SWITCHING)
            is_down = state_id == int(BouncerState.DOWN)

            dx_s, dy_s, ns_s, ni_s, nl_s = switching_logic(pos, lane_idx, step_index, direction)
            dx_l, dy_l, ns_l, ni_l, nl_l = lane_logic(pos, lane_idx, step_index, is_down)

            final_dx = jnp.where(is_switching, dx_s, dx_l)
            final_dy = jnp.where(is_switching, dy_s, dy_l)
            final_state = jnp.where(is_switching, ns_s, ns_l)
            final_s_idx = jnp.where(is_switching, ni_s, ni_l)
            final_lane = jnp.where(is_switching, nl_s, nl_l)

            new_pos = pos + jnp.array([final_dx, final_dy])

            should_deactivate = jnp.logical_or(
                new_pos[1] > float(self.consts.BOTTOM_CLIP) + 5.0,
                jnp.logical_or(new_pos[0] < 0, new_pos[0] > float(self.consts.SCREEN_WIDTH))
            )

            return (
                new_pos,
                jnp.array([final_dx, final_dy]),
                final_state,
                timer,
                jnp.logical_not(should_deactivate),
                final_lane,
                final_s_idx
            )

        def spawn_inactive(_):
            key_spawn, key_side = jax.random.split(key)
            # Average every 5 seconds = 1/150 chance per step
            spawn_prob = 1.0 / 150.0
            should_spawn = (jax.random.uniform(key_spawn) < spawn_prob) & (state.sector >= self.consts.BOUNCER_START_LEVEL)

            side_left = jax.random.uniform(key_side) < 0.5
            spawn_x = jnp.where(side_left, 5.0, 155.0)
            spawn_y = self.consts.BOUNCER_SPAWN_HEIGHT
            spawn_dir = jnp.where(side_left, 1.0, -1.0)
            spawn_lane = jnp.where(side_left, -1, 7)

            return (
                jnp.where(should_spawn, jnp.array([spawn_x, spawn_y]), jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32)),
                jnp.where(should_spawn, jnp.array([spawn_dir, 0.0]), jnp.zeros(2, dtype=jnp.float32)),
                jnp.where(should_spawn, int(BouncerState.SWITCHING), int(BouncerState.SWITCHING)),
                jnp.where(should_spawn, spawn_dir.astype(jnp.int32), 0),
                should_spawn,
                jnp.where(should_spawn, spawn_lane, 0),
                jnp.where(should_spawn, 0, 0)
            )

        return jax.lax.cond(active, move_active, spawn_inactive, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BeamriderState):
        is_init = state.level.blue_line_counter < len(BLUE_LINE_INIT_TABLE)
        
        ufo_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 3),
        )
        enemy_shot_offscreen = jnp.tile(
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 9),
        )
        chasing_meteoroid_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, self.consts.CHASING_METEOROID_MAX),
        )
        falling_rock_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, self.consts.FALLING_ROCK_MAX),
        )
        lane_blocker_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, self.consts.LANE_BLOCKER_MAX),
        )
        coin_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, self.consts.COIN_MAX),
        )
        rejuv_offscreen = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32)
        
        # Create a state for rendering where enemies are offscreen if initializing
        render_level = state.level._replace(
            white_ufo_pos=jnp.where(is_init, ufo_offscreen, state.level.white_ufo_pos),
            enemy_shot_pos=jnp.where(is_init, enemy_shot_offscreen, state.level.enemy_shot_pos),
            enemy_shot_explosion_pos=jnp.where(is_init, enemy_shot_offscreen, state.level.enemy_shot_explosion_pos),
            chasing_meteoroid_pos=jnp.where(is_init, chasing_meteoroid_offscreen, state.level.chasing_meteoroid_pos),
            rejuvenator_pos=jnp.where(is_init, rejuv_offscreen, state.level.rejuvenator_pos),
            falling_rock_pos=jnp.where(is_init, falling_rock_offscreen, state.level.falling_rock_pos),
            falling_rock_explosion_pos=jnp.where(is_init, falling_rock_offscreen, state.level.falling_rock_explosion_pos),
            lane_blocker_pos=jnp.where(is_init, lane_blocker_offscreen, state.level.lane_blocker_pos),
            lane_blocker_explosion_pos=jnp.where(is_init, lane_blocker_offscreen, state.level.lane_blocker_explosion_pos),
            lane_blocker_active=jnp.where(is_init, False, state.level.lane_blocker_active),
            coin_pos=jnp.where(is_init, coin_offscreen, state.level.coin_pos),
            coin_explosion_pos=jnp.where(is_init, coin_offscreen, state.level.coin_explosion_pos),
            bouncer_pos=jnp.where(is_init, jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32), state.level.bouncer_pos),
            bouncer_active=jnp.where(is_init, False, state.level.bouncer_active),
        )
        render_state = state._replace(level=render_level)
        
        return self.renderer.render(render_state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    
    def _get_info(self, state: BeamriderState) -> BeamriderInfo:
        return BeamriderInfo(
            score=state.score,
            sector=state.sector,
        )

    def _get_reward(
        self, previous_state: BeamriderState, state: BeamriderState
    ) -> float:
        return state.score - previous_state.score

    def _get_done(self, state: BeamriderState) -> bool:
        return jnp.logical_or(state.lives <= 0, state.sector > 14)

    def _kamikaze_step(self, state: BeamriderState, key: chex.Array):
        pos = state.level.kamikaze_pos
        active = state.level.kamikaze_active[0]
        lane = state.level.kamikaze_lane[0]
        vel_y = state.level.kamikaze_vel_y[0]
        tracking = state.level.kamikaze_tracking[0]
        spawn_timer = state.level.kamikaze_spawn_timer[0]

        # Spawning logic: Sector 12 onwards, every 250 frames
        is_level_12_plus = state.sector >= self.consts.KAMIKAZE_START_SECTOR
        
        # Advance timer if in sector 12+ and not currently active
        spawn_timer = jnp.where(is_level_12_plus & jnp.logical_not(active), 
                                spawn_timer + 1, 
                                0)
        
        should_spawn = is_level_12_plus & jnp.logical_not(active) & (spawn_timer >= self.consts.KAMIKAZE_SPAWN_INTERVAL)
        
        spawn_lane = jax.random.randint(key, (), 1, 6) # Inner lanes 1-5
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=jnp.float32)
        spawn_x = lanes_top_x[spawn_lane]
        spawn_y = float(self.consts.KAMIKAZE_START_Y)
        
        pos = jnp.where(should_spawn, jnp.array([[spawn_x], [spawn_y]]), pos)
        active = jnp.where(should_spawn, True, active)
        lane = jnp.where(should_spawn, spawn_lane, lane)
        vel_y = jnp.where(should_spawn, self.consts.LANE_BLOCKER_INIT_VEL, vel_y)
        tracking = jnp.where(should_spawn, True, tracking)
        spawn_timer = jnp.where(should_spawn, 0, spawn_timer)

        y = pos[1, 0]
        
        # Acceleration like Alien Blocker
        accel = jnp.where(y < 64, 0.004,
                jnp.where(y < 85, 0.008,
                self.consts.FALLING_ROCK_ACCEL))
        
        new_vel_y = vel_y + accel
        dy = new_vel_y
        new_y = y + dy

        # Tracking logic: starts at 60y
        player_x = state.level.player_pos
        bottom_lanes = jnp.array(self.consts.BOTTOM_OF_LANES, dtype=jnp.float32)
        player_lane = jnp.argmin(jnp.abs(bottom_lanes - player_x)) + 1 # 1-5
        
        should_track = active & (new_y >= self.consts.KAMIKAZE_TRACK_Y) & tracking
        
        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=jnp.float32)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]
        
        # Current lane position
        current_lane_x = lanes_top_x[lane] + lane_dx_over_dy[lane] * (new_y - float(self.consts.TOP_CLIP))
        
        # Target lane position
        target_lane_x = lanes_top_x[player_lane] + lane_dx_over_dy[player_lane] * (new_y - float(self.consts.TOP_CLIP))
        
        lateral_dist = target_lane_x - pos[0, 0]
        dx_dir = jnp.sign(lateral_dist)
        dx_lateral = 2.0 * dy * dx_dir
        
        # If we are close enough to the target lane, snap to it and stop tracking
        reached_lane = jnp.abs(lateral_dist) <= jnp.abs(dx_lateral)
        
        new_x = jnp.where(should_track, 
                          jnp.where(reached_lane, target_lane_x, pos[0, 0] + dx_lateral),
                          current_lane_x)
        
        # Update lane index if reached
        lane = jnp.where(should_track & reached_lane, player_lane, lane)
        tracking = jnp.where(should_track & reached_lane, False, tracking)
        
        # Deactivate if off-screen
        off_screen = new_y > self.consts.PLAYER_POS_Y + 5.0
        active = jnp.where(off_screen, False, active)
        
        offscreen_pos = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1)
        final_pos = jnp.where(active, jnp.array([[new_x], [new_y]]), offscreen_pos)
        
        return final_pos, jnp.array([active]), jnp.array([lane]), jnp.array([new_vel_y]), jnp.array([tracking]), jnp.array([spawn_timer])

    def _kamikaze_bullet_collision(
        self,
        kamikaze_pos: chex.Array,
        kamikaze_active: chex.Array,
        player_shot_pos: chex.Array,
        player_shot_vel: chex.Array,
        bullet_type: chex.Array,
    ):
        is_torpedo = bullet_type == self.consts.TORPEDO_ID
        shot_x = _get_player_shot_screen_x(
            player_shot_pos,
            player_shot_vel,
            bullet_type,
            self.consts.LASER_ID,
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        kamikaze_x = kamikaze_pos[0, 0] + _get_ufo_alignment(kamikaze_pos[1, 0]).astype(kamikaze_pos.dtype)
        kamikaze_y = kamikaze_pos[1, 0]

        bullet_idx = _get_index_bullet(shot_y, bullet_type, self.consts.LASER_ID)
        bullet_size = jnp.take(jnp.array(self.consts.BULLET_SPRITE_SIZES), bullet_idx, axis=0)
        
        kamikaze_indices = jnp.clip(_get_index_kamikaze(kamikaze_y) - 1, 0, 3)
        kamikaze_sizes = jnp.take(jnp.array(self.consts.LANE_BLOCKER_SPRITE_SIZES), kamikaze_indices, axis=0)
        
        # AABB collision check
        hit_mask = (
            kamikaze_active[0]
            & shot_active
            & (kamikaze_x < shot_x + bullet_size[1]) & (shot_x < kamikaze_x + kamikaze_sizes[1])
            & (kamikaze_y < shot_y + bullet_size[0]) & (shot_y < kamikaze_y + kamikaze_sizes[0])
        )
        
        # Kamikaze is destroyed only by torpedo
        destroyed = hit_mask & is_torpedo
        
        offscreen = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=kamikaze_pos.dtype).reshape(2, 1)
        new_pos = jnp.where(destroyed, offscreen, kamikaze_pos)
        new_active = jnp.where(destroyed, False, kamikaze_active[0])
        
        # Shot is always removed if it hits (blocking behavior)
        new_player_shot_pos = jnp.where(
            hit_mask,
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=player_shot_pos.dtype),
            player_shot_pos,
        )

        return new_pos, jnp.array([new_active]), new_player_shot_pos, jnp.array([destroyed]), jnp.array([hit_mask])

class BeamriderRenderer(JAXGameRenderer):
    def __init__(self, consts=None):
        super().__init__()
        self.consts = consts or BeamriderConstants()
        self.rendering_config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )

        self.jr = render_utils.JaxRenderingUtils(self.rendering_config)

        # 1. Create procedural assets:
        # background_sprite = self._create_background_sprite()
        # player_sprite = self._create_player_sprite()

        #2 Update asset config to include sprites 
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/beamrider"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        self._enemy_explosion_sprite_seq = jnp.array(
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5],
            dtype=jnp.int32,
        )
        self._enemy_explosion_y_offsets = jnp.array(
            [0, 0, 0, 0, -1, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3, -4, -4, -4, -4, -5],
            dtype=jnp.int32,
        )

    # def _create_player_sprite(self) -> jnp.ndarray:
    #     """Procedurally creates an RGBA sprite for the background"""
    #     player_color_rgba = (214, 239, 30, 0.7) # e.g., (236, 236, 236, 255)
    #     player_dimensions = (self.consts.PLAYER_SPRITE_SIZE[0], self.consts.PLAYER_SPRITE_SIZE[1], 4)
    #     player_sprite = jnp.tile(jnp.array(player_color_rgba, dtype=jnp.uint8), (*player_dimensions[:2], 1))
    #     return player_sprite
    
    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [
            {'name': 'background_sprite', 'type': 'background', 'file': 'new_background.npy'},
            {'name': 'player_sprite', 'type': 'group', 'files': [f'Player/Player_{i}.npy' for i in range(1, 17)]},
            {'name': 'dead_player', 'type': 'single', 'file': 'Dead_Player.npy'},
            {'name': 'white_ufo', 'type': 'group', 'files': ['White_Ufo_Stage_1.npy', 'White_Ufo_Stage_2.npy', 'White_Ufo_Stage_3.npy', 'White_Ufo_Stage_4.npy', 'White_Ufo_Stage_5.npy', 'White_Ufo_Stage_6.npy', 'White_Ufo_Stage_7.npy']},
            {'name': 'coin', 'type': 'group', 'files': ['Coin/Coin1.npy', 'Coin/Coin2.npy', 'Coin/Coin3.npy', 'Coin/Coin4.npy']},
            {'name': 'bouncer', 'type': 'single', 'file': 'Bouncer.npy'},
            {'name': 'enemy_explosion', 'type': 'group', 'files': [
                'White_Ufo_Explosion/White_Ufo_Explosion_1.npy',
                'White_Ufo_Explosion/White_Ufo_Explosion_2.npy',
                'White_Ufo_Explosion/White_Ufo_Explosion_3.npy',
                'White_Ufo_Explosion/White_Ufo_Explosion_4.npy',
                'White_Ufo_Explosion/White_Ufo_Explosion_5.npy',
                'White_Ufo_Explosion/White_Ufo_Explosion_6.npy',
            ]},
            {'name': 'chasing_meteoroid', 'type': 'single', 'file': 'Chasing_Meteoroid.npy'},
            {'name': 'laser_sprite', 'type': 'single', 'file': 'Laser.npy'},
            {'name': 'bullet_sprite', 'type': 'group', 'files': ['Laser.npy', 'Torpedo/Torpedo_3.npy', 'Torpedo/Torpedo_2.npy', 'Torpedo/Torpedo_1.npy']},
            {'name': 'enemy_shot', 'type': 'group', 'files': ['Enemy_Shot/Enemy_Shot_Vertical.npy', 'Enemy_Shot/Enemy_Shot_Horizontal.npy']},
            {'name': 'rejuvenator', 'type': 'group', 'files': [f'Rejuvenator/Rejuvenator_{i}.npy' for i in range(1, 5)] + ['Rejuvenator/Rejuvenator_Dead.npy']},
            {'name': 'falling_rocks', 'type': 'group', 'files': ['Falling Rocks/Rock_1.npy', 'Falling Rocks/Rock_2.npy', 'Falling Rocks/Rock_3.npy', 'Falling Rocks/Rock_4.npy']},
            {'name': 'lane_blocker', 'type': 'group', 'files': [
                'AlienBlocker/AlienBlocker_1.npy',
                'AlienBlocker/AlienBlocker_2.npy',
                'AlienBlocker/AlienBlocker_3.npy',
                'AlienBlocker/AlienBlocker_4.npy',
            ]},
            {'name': 'kamikaze', 'type': 'group', 'files': [
                'Kamikaze/Kamikaze_1.npy',
                'Kamikaze/Kamikaze_2.npy',
                'Kamikaze/Kamikaze_3.npy',
                'Kamikaze/Kamikaze_4.npy',
            ]},
            {'name': 'blue_line', 'type': 'single', 'file': 'blue_line.npy'},
            {'name': 'torpedos_left', 'type': 'single', 'file': 'torpedos_left.npy'},
            {'name': 'green_numbers', 'type': 'digits', 'pattern': 'green_nums/green_{}.npy'},
            {'name': 'yellow_numbers', 'type': 'digits', 'pattern': 'yellow_nums/yellow_{}.npy'},
            {'name': 'live', 'type': 'single', 'file': 'lives.npy'},
            {'name': 'mothership', 'type': 'single', 'file': 'Mothership.npy'},
            {'name': 'mothership_explosion', 'type': 'group', 'files': [
                'Mothership/Mothership_explosion1.npy',
                'Mothership/Mothership_explosion2.npy',
                'Mothership/Mothership_explosion3.npy',
            ]},
        ]
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state:BeamriderState) -> chex.Array:
        raster = self.jr.create_object_raster(self.BACKGROUND)
        raster = self._render_blue_lines(raster, state)
        raster = self._render_player_and_bullet(raster, state)
        raster = self._render_enemy_shots(raster, state)
        raster = self._render_white_ufos(raster, state)
        raster = self._render_bouncer(raster, state)
        raster = self._render_chasing_meteoroids(raster, state)
        raster = self._render_falling_rocks(raster, state)
        raster = self._render_lane_blockers(raster, state)
        raster = self._render_kamikaze(raster, state)
        raster = self._render_coins(raster, state)
        raster = self._render_rejuvenator(raster, state)
        raster = self._render_hud(raster, state)
        raster = self._render_mothership(raster, state)
        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_kamikaze(self, raster, state):
        kamikaze_masks = self.SHAPE_MASKS["kamikaze"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        
        explosion_frame = state.level.kamikaze_explosion_frame[0]
        active = state.level.kamikaze_active[0]

        def render_explosion(r_in):
            sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
            sprite = explosion_masks[sprite_idx]
            x_pos = state.level.kamikaze_explosion_pos[0][0] + _get_ufo_alignment(
                state.level.kamikaze_explosion_pos[1][0]
            )
            y_pos = state.level.kamikaze_explosion_pos[1][0] + y_offset
            return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

        def render_kamikaze(r_in):
            def render_active(r_inner):
                sprite_idx = _get_index_kamikaze(state.level.kamikaze_pos[1][0]) - 1
                sprite_idx = jnp.clip(sprite_idx, 0, 3)
                sprite = kamikaze_masks[sprite_idx]
                x_pos = state.level.kamikaze_pos[0][0] + _get_ufo_alignment(
                    state.level.kamikaze_pos[1][0]
                )
                y_pos = state.level.kamikaze_pos[1][0]
                return self.jr.render_at_clipped(r_inner, x_pos, y_pos, sprite)
            
            return jax.lax.cond(active, render_active, lambda r: r, r_in)

        raster = jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_kamikaze,
            raster,
        )
        return raster

    def _render_coins(self, raster, state):
        coin_masks = self.SHAPE_MASKS["coin"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        for idx in range(self.consts.COIN_MAX):
            active = state.level.coin_active[idx]
            timer = state.level.coin_timer[idx]
            pos = state.level.coin_pos[:, idx]
            explosion_frame = state.level.coin_explosion_frame[idx]

            def render_explosion(r_in):
                sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
                sprite = explosion_masks[sprite_idx]
                x_pos = state.level.coin_explosion_pos[0][idx] + _get_ufo_alignment(
                    state.level.coin_explosion_pos[1][idx]
                )
                y_pos = state.level.coin_explosion_pos[1][idx] + y_offset
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            def render_coin(r_in):
                # Animation sequence: (3, 2, 1, 0, 1, 2)
                anim_idx = (timer // 4) % len(self.consts.COIN_ANIM_SEQ)
                sprite_idx = jnp.array(self.consts.COIN_ANIM_SEQ)[anim_idx]
                mask = coin_masks[sprite_idx]
                
                # Adjust X for screen position
                x_pos = pos[0] + _get_ufo_alignment(pos[1])
                y_pos = jnp.where(active, pos[1], 500)
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, mask)

            raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_coin,
                raster,
            )
        return raster

    def _render_falling_rocks(self, raster, state):
        falling_rock_masks = self.SHAPE_MASKS["falling_rocks"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        for idx in range(self.consts.FALLING_ROCK_MAX):
            explosion_frame = state.level.falling_rock_explosion_frame[idx]

            def render_explosion(r_in):
                sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
                sprite = explosion_masks[sprite_idx]
                x_pos = state.level.falling_rock_explosion_pos[0][idx] + _get_ufo_alignment(
                    state.level.falling_rock_explosion_pos[1][idx]
                )
                y_pos = state.level.falling_rock_explosion_pos[1][idx] + y_offset
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            def render_rock(r_in):
                sprite_idx = _get_index_falling_rock(state.level.falling_rock_pos[1][idx]) - 1
                sprite = falling_rock_masks[sprite_idx]
                x_pos = state.level.falling_rock_pos[0][idx] + _get_ufo_alignment(
                    state.level.falling_rock_pos[1][idx]
                )
                y_pos = state.level.falling_rock_pos[1][idx]
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_rock,
                raster,
            )
        return raster

    def _render_lane_blockers(self, raster, state):
        lane_blocker_masks = self.SHAPE_MASKS["lane_blocker"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        for idx in range(self.consts.LANE_BLOCKER_MAX):
            explosion_frame = state.level.lane_blocker_explosion_frame[idx]

            def render_explosion(r_in):
                sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
                sprite = explosion_masks[sprite_idx]
                x_pos = state.level.lane_blocker_explosion_pos[0][idx] + _get_ufo_alignment(
                    state.level.lane_blocker_explosion_pos[1][idx]
                )
                y_pos = state.level.lane_blocker_explosion_pos[1][idx] + y_offset
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            def render_blocker(r_in):
                sprite_idx = _get_index_lane_blocker(state.level.lane_blocker_pos[1][idx]) - 1
                sprite_idx = jnp.clip(sprite_idx, 0, lane_blocker_masks.shape[0] - 1)
                # Asset 3 is oversized; reuse stage 4 visuals for mid-height range.
                sprite_idx = jnp.where(sprite_idx == 2, 3, sprite_idx)
                sprite = lane_blocker_masks[sprite_idx]
                x_pos = state.level.lane_blocker_pos[0][idx] + _get_ufo_alignment(
                    state.level.lane_blocker_pos[1][idx]
                )
                y_pos = state.level.lane_blocker_pos[1][idx]

                clip_rows = jnp.maximum(
                    0, jnp.round(y_pos - self.consts.LANE_BLOCKER_BOTTOM_Y).astype(jnp.int32)
                )
                is_sinking = state.level.lane_blocker_phase[idx] == int(LaneBlockerState.SINK)

                def clip_mask(mask):
                    height = mask.shape[0]
                    visible_rows = jnp.maximum(0, height - clip_rows)
                    row_idx = jnp.arange(height)
                    row_mask = row_idx < visible_rows
                    transparent = jnp.array(self.jr.TRANSPARENT_ID, dtype=mask.dtype)
                    return jnp.where(row_mask[:, None], mask, transparent)

                sprite = jax.lax.cond(is_sinking, clip_mask, lambda m: m, sprite)
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_blocker,
                raster,
            )
        return raster

    def _render_rejuvenator(self, raster, state):
        rejuv_pos = state.level.rejuvenator_pos
        active = state.level.rejuvenator_active
        dead = state.level.rejuvenator_dead
        
        # Determine mask
        rejuv_masks = self.SHAPE_MASKS["rejuvenator"]
        
        stage = _get_index_rejuvenator(rejuv_pos[1])
        # stage 1..4 maps to masks 0..3
        # dead state is mask index 4
        sprite_idx = jnp.where(dead, 4, jnp.clip(stage - 1, 0, 3))
        
        mask = rejuv_masks[sprite_idx]
        
        # Handle offscreen for rendering: ensure they are definitely outside [0, 160/210]
        final_y = jnp.where(active, rejuv_pos[1], -50.0)
        final_x = jnp.where(active, rejuv_pos[0], -50.0)
        
        return self.jr.render_at_clipped(raster, final_x, final_y, mask)

    def _render_blue_lines(self, raster, state):
        """Draw the scrolling foreground lines."""

        blue_line_mask = self.SHAPE_MASKS["blue_line"]
        for idx in range(7):
            y_pos = state.level.line_positions[idx]
            # y_pos is -1 if line is inactive
            final_y = jnp.where(y_pos >= 0, y_pos, self.consts.BLUE_LINE_OFFSCREEN_Y)
            raster = self.jr.render_at_clipped(
                raster, 8, final_y, blue_line_mask
            )
        return raster

    def _render_hud(self, raster, state):
        raster = self._render_torpedo_icons(raster, state)
        raster = self._render_white_ufo_counter(raster, state)
        raster = self._render_score(raster, state)
        raster = self._render_sector(raster, state)
        raster = self._render_lives(raster, state)
        return raster

    def _render_torpedo_icons(self, raster, state):
        """Render the torpedo inventory indicator on the HUD."""

        torpedo_mask = self.SHAPE_MASKS["torpedos_left"]
        icon_config = [(3, 128), (2, 136), (1, 144)]
        for threshold, y in icon_config:
            y_pos = jnp.where(state.level.torpedoes_left >= threshold, y, 500)
            raster = self.jr.render_at_clipped(raster, y_pos, 32, torpedo_mask)
        return raster

    def _render_white_ufo_counter(self, raster, state):
        # Hide counter if mothership animation is active
        should_hide = state.level.mothership_stage > 0
        
        ufos_left_digits = self.jr.int_to_digits(state.level.white_ufo_left, max_digits=2)
        ufos_left_masks = self.SHAPE_MASKS["green_numbers"]
        is_double_digit = state.level.white_ufo_left >= 10
        start_index = jnp.where(is_double_digit, 0, 1)
        numbers_to_render = jnp.where(is_double_digit, 2, 1)
        
        final_numbers_to_render = jnp.where(should_hide, 0, numbers_to_render)
        
        return self.jr.render_label_selective(
            raster, 19, 32, ufos_left_digits, ufos_left_masks, start_index, final_numbers_to_render, spacing=6
        )

    def _render_score(self, raster, state):
        yellow_digits = self.jr.int_to_digits(state.score, max_digits=6)
        score_masks = self.SHAPE_MASKS["yellow_numbers"]
        return self.jr.render_label_selective(
            raster, 61, 10, yellow_digits, score_masks, 0, 6, spacing=8, max_digits_to_render=6
        )

    def _render_sector(self, raster, state):
        sector_digits = self.jr.int_to_digits(state.sector, max_digits=2)
        score_masks = self.SHAPE_MASKS["yellow_numbers"]
        return self.jr.render_label_selective(
            raster, 93, 21, sector_digits, score_masks, 0, 2, spacing=8, max_digits_to_render=2
        )

    def _render_lives(self, raster, state):
        hp_mask = self.SHAPE_MASKS["live"]
        death_timer = state.level.death_timer
        is_dead = death_timer > 0
        
        # Flashing logic: 8 frames on, 8 frames off.
        flash_visible = (death_timer // 8) % 2 == 0
        
        # Supporting up to 14 lives means up to 13 icons (lives-1)
        for idx in range(13):
            # Normal logic: render if idx < state.lives - 1 (Bonus HP display)
            
            # Flashing logic: if is_dead, we are about to lose a life.
            # We want to flash the icon that represents the life we are losing.
            # The icon disappearing is at index state.lives - 2.
            
            is_last_life = (idx == state.lives - 2)
            should_flash = jnp.logical_and(is_dead, is_last_life)
            
            visible_normally = (state.lives - 1) > idx
            
            # If flashing, visibility depends on flash_visible.
            # If not flashing, depends on visible_normally.
            
            is_visible = jnp.where(
                should_flash,
                jnp.logical_and(visible_normally, flash_visible),
                visible_normally
            )
            
            # Icons start at x=32, spaced by 9 pixels
            pos_x = jnp.where(is_visible, 32 + (idx * 9), -100)
            raster = self.jr.render_at_clipped(raster, pos_x, 183, hp_mask)
        return raster

    def _render_player_and_bullet(self, raster, state):
        player_masks = self.SHAPE_MASKS["player_sprite"]
        dead_player_mask = self.SHAPE_MASKS["dead_player"]
        
        # Determine which mask to use
        is_dead = state.level.death_timer > 0
        is_init = state.level.blue_line_counter < len(BLUE_LINE_INIT_TABLE)
        
        def render_alive(r):
            sprite_idx = jnp.where(is_init, 9, ((state.steps // 2) + 1) % 16)
            mask = player_masks[sprite_idx]
            return self.jr.render_at(r, state.level.player_pos, self.consts.PLAYER_POS_Y, mask)

        def render_dead(r):
            return self.jr.render_at(r, state.level.player_pos, self.consts.PLAYER_POS_Y, dead_player_mask)

        raster = jax.lax.cond(is_dead, render_dead, render_alive, raster)

        bullet_mask = self.SHAPE_MASKS["bullet_sprite"][
            _get_index_bullet(state.level.player_shot_pos[1], state.level.bullet_type, self.consts.LASER_ID)
        ]
        shot_x_screen = _get_player_shot_screen_x(
            state.level.player_shot_pos,
            state.level.player_shot_vel,
            state.level.bullet_type,
            self.consts.LASER_ID,
        )
        raster = self.jr.render_at_clipped(
            raster,
            shot_x_screen,
            state.level.player_shot_pos[1],
            bullet_mask,
        )
        return raster

    def _render_enemy_shots(self, raster, state):
        enemy_shot_masks = self.SHAPE_MASKS["enemy_shot"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        for idx in range(9):
            explosion_frame = state.level.enemy_shot_explosion_frame[idx]

            def render_explosion(r_in):
                sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
                sprite = explosion_masks[sprite_idx]
                x_pos = state.level.enemy_shot_explosion_pos[0][idx] + _get_ufo_alignment(
                    state.level.enemy_shot_explosion_pos[1][idx]
                )
                y_pos = state.level.enemy_shot_explosion_pos[1][idx] + y_offset
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            def render_shot(r_in):
                timer = state.level.enemy_shot_timer[idx]
                sprite_idx = (jnp.floor_divide(timer, 4) % 2).astype(jnp.int32)
                y_pos = jnp.where(
                    state.level.enemy_shot_pos[1][idx] <= self.consts.BOTTOM_CLIP,
                    state.level.enemy_shot_pos[1][idx],
                    500,
                )
                return self.jr.render_at_clipped(
                    r_in, state.level.enemy_shot_pos[0][idx] + _get_ufo_alignment(y_pos), y_pos, enemy_shot_masks[sprite_idx]
                )

            raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_shot,
                raster,
            )
        return raster

    def _render_white_ufos(self, raster, state):
        white_ufo_masks = self.SHAPE_MASKS["white_ufo"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        for idx in range(3):
            explosion_frame = state.level.ufo_explosion_frame[idx]

            def render_explosion(r_in):
                sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
                sprite = explosion_masks[sprite_idx]
                x_pos = state.level.ufo_explosion_pos[0][idx] + _get_ufo_alignment(
                    state.level.ufo_explosion_pos[1][idx]
                )
                y_pos = state.level.ufo_explosion_pos[1][idx] + y_offset
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            def render_ufo(r_in):
                sprite_idx = _get_index_ufo(state.level.white_ufo_pos[1][idx]) - 1
                sprite = white_ufo_masks[sprite_idx]
                x_pos = state.level.white_ufo_pos[0][idx] + _get_ufo_alignment(
                    state.level.white_ufo_pos[1][idx]
                )
                y_pos = state.level.white_ufo_pos[1][idx]
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_ufo,
                raster,
            )
        return raster

    def _render_bouncer(self, raster, state):
        bouncer_mask = self.SHAPE_MASKS["bouncer"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        explosion_frame = state.level.bouncer_explosion_frame

        def render_explosion(r_in):
            sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
            sprite = explosion_masks[sprite_idx]
            x_pos = state.level.bouncer_explosion_pos[0] + _get_ufo_alignment(
                state.level.bouncer_explosion_pos[1]
            )
            y_pos = state.level.bouncer_explosion_pos[1] + y_offset
            return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

        def render_active_bouncer(r_in):
            is_active = state.level.bouncer_active
            y_pos = jnp.where(is_active, state.level.bouncer_pos[1], 500)
            x_pos = jnp.where(
                is_active,
                state.level.bouncer_pos[0] + _get_ufo_alignment(y_pos),
                500,
            )
            return self.jr.render_at_clipped(r_in, x_pos, y_pos, bouncer_mask)

        return jax.lax.cond(
            explosion_frame > 0,
            render_explosion,
            render_active_bouncer,
            raster,
        )

    def _render_chasing_meteoroids(self, raster, state):
        meteoroid_mask = self.SHAPE_MASKS["chasing_meteoroid"]
        explosion_masks = self.SHAPE_MASKS["enemy_explosion"]
        for idx in range(self.consts.CHASING_METEOROID_MAX):
            explosion_frame = state.level.chasing_meteoroid_explosion_frame[idx]

            def render_explosion(r_in):
                sprite_idx, y_offset = self._get_enemy_explosion_visuals(explosion_frame)
                sprite = explosion_masks[sprite_idx]
                x_pos = state.level.chasing_meteoroid_explosion_pos[0][idx] + _get_ufo_alignment(
                    state.level.chasing_meteoroid_explosion_pos[1][idx]
                )
                y_pos = state.level.chasing_meteoroid_explosion_pos[1][idx] + y_offset
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, sprite)

            def render_meteoroid(r_in):
                is_active = state.level.chasing_meteoroid_active[idx]
                y_pos = jnp.where(is_active, state.level.chasing_meteoroid_pos[1][idx], 500)
                x_pos = jnp.where(
                    is_active,
                    state.level.chasing_meteoroid_pos[0][idx] + _get_ufo_alignment(y_pos),
                    500,
                )
                return self.jr.render_at_clipped(r_in, x_pos, y_pos, meteoroid_mask)

            raster = jax.lax.cond(
                explosion_frame > 0,
                render_explosion,
                render_meteoroid,
                raster,
            )
        return raster

    def _render_mothership(self, raster, state):
        stage = state.level.mothership_stage
        timer = state.level.mothership_timer
        pos_x = state.level.mothership_position
        
        mask = self.SHAPE_MASKS["mothership"]

        def render_none(r):
            return r

        def render_clipping(r, num_lines):
            # Clip the mask to only show top num_lines
            y_indices = jnp.arange(self.consts.MOTHERSHIP_HEIGHT)[:, None]
            active_rows = y_indices < num_lines
            effective_mask = jnp.where(active_rows, mask, self.jr.TRANSPARENT_ID)
            y = self.consts.MOTHERSHIP_EMERGE_Y - num_lines
            return self.jr.render_at_clipped(r, pos_x, y, effective_mask)

        def render_emergence(r):
            num_lines = jnp.clip(timer // 2, 0, self.consts.MOTHERSHIP_HEIGHT)
            return jax.lax.cond(num_lines > 0, 
                               lambda r_: render_clipping(r_, num_lines),
                               lambda r_: r_, r)

        def render_moving(r):
            return self.jr.render_at_clipped(r, pos_x, self.consts.MOTHERSHIP_EMERGE_Y - self.consts.MOTHERSHIP_HEIGHT, mask)

        def render_descending(r):
            s = jnp.clip(6 - (timer - 1) // 2, 0, 6)
            num_lines = s + 1
            return render_clipping(r, num_lines)

        def render_exploding(r):
            explosion_masks = self.SHAPE_MASKS["mothership_explosion"]
            step_duration = self.consts.MOTHERSHIP_EXPLOSION_STEP_DURATION
            step_idx = timer // step_duration
            step_idx = jnp.clip(step_idx, 0, 8)
            seq = jnp.array(self.consts.MOTHERSHIP_EXPLOSION_SEQUENCE)
            sprite_idx = seq[step_idx]
            exp_mask = explosion_masks[sprite_idx]
            y = self.consts.MOTHERSHIP_EMERGE_Y - self.consts.MOTHERSHIP_HEIGHT
            return self.jr.render_at_clipped(r, pos_x, y, exp_mask)

        return jax.lax.switch(
            stage.astype(jnp.int32),
            [render_none, render_emergence, render_moving, render_descending, render_none, render_exploding],
            raster
        )

    def _get_enemy_explosion_visuals(self, frame: chex.Array) -> Tuple[chex.Array, chex.Array]:
        clamped = jnp.clip(frame - 1, 0, self._enemy_explosion_sprite_seq.shape[0] - 1)
        sprite_idx = self._enemy_explosion_sprite_seq[clamped]
        y_offset = self._enemy_explosion_y_offsets[clamped]
        return sprite_idx, y_offset
