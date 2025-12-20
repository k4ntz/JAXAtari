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


class BeamriderConstants(NamedTuple):

    WHITE_UFOS_PER_SECTOR: int = 1

    RENDER_SCALE_FACTOR: int = 4
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    PLAYER_WIDTH: int = 10
    PLAYER_HEIGHT: int = 10
    ENEMY_WIDTH: int = 4
    ENEMY_HEIGHT: int = 4
    PLAYER_COLOR: Tuple[int, int, int] = (223, 183, 85)
    LEFT_CLIP_PLAYER: int = 27
    RIGHT_CLIP_PLAYER: int = 137
    BOTTOM_OF_LANES: Tuple[int, int, int, int, int] = (27,52,77,102,127)
    TOP_OF_LANES: Tuple[int, int, int, int, int] = (38,61,71,81,91,102,123)  #lane 0,6 are connected to points in middle of the map, not to bottom lane points
    
    TOP_TO_BOTTOM_LANE_VECTORS: Tuple[Tuple[float, float],Tuple[float, float],Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2,4),(-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4),(2,4))


    MAX_LASER_Y: int = 67
    MIN_BULLET_Y:int =156
    MAX_TORPEDO_Y: int = 35
    BOTTOM_TO_TOP_LANE_VECTORS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-1, 4), (-0.52, 4), (0,4), (0.52, 4), (1, 4))

    PLAYER_POS_Y: int = 164
    PLAYER_SPEED: float = 2.5
    PLAYER_LASER_COOLDOWN: int = 23
    PLAYER_TORPEDO_COOLDOWN: int = 31

    BOTTOM_CLIP:int = 175
    TOP_CLIP:int=43
    LASER_HIT_RADIUS: Tuple[int, int] = (7, 2)
    TORPEDO_HIT_RADIUS: Tuple[int, int] = (4, 2)
    LASER_ID: int = 1
    TORPEDO_ID: int = 2
    BULLET_OFFSCREEN_POS: Tuple[int, int] = (800.0, 800.0)
    ENEMY_OFFSCREEN_POS: Tuple[int,int] = (-100, -100)
    MIN_BLUE_LINE_POS: int = 46
    MAX_BLUE_LINE_POS: int = 160
    WHITE_UFO_RETREAT_DURATION: int = 28
    ####PATTERNS:                                                           IDLE | DROP_STRAIGHT | DROP_RIGHT | DROP_LEFT | RETREAT | SHOOT | MOVE_BACK | KAMIKAZE
    WHITE_UFO_PATTERN_DURATIONS: Tuple[int, ...] =                          (0,          42,            42,         42,         28,     0,      42,         100)
    WHITE_UFO_PATTERN_PROBS: Tuple[float, ...] =                            (            0.3,           0.2,        0.2,                0.2,    0.1,        0.3) #these probas are not 1:1, as some patterns have activation conditions
    WHITE_UFO_SPEED_FACTOR: float = 0.1
    WHITE_UFO_SHOT_SPEED_FACTOR: float = 0.8
    WHITE_UFO_RETREAT_P_MIN: float = 0.005
    WHITE_UFO_RETREAT_P_MAX: float = 0.1
    WHITE_UFO_RETREAT_ALPHA: float = 0.01
    WHITE_UFO_RETREAT_SPEED_MULT: float = 2.5
    WHITE_UFO_TOP_LANE_MIN_SPEED: float = 0.3
    WHITE_UFO_TOP_LANE_TURN_SPEED: float = 0.5
    WHITE_UFO_ATTACK_P_MIN: float = 0.0002
    WHITE_UFO_ATTACK_P_MAX: float = 0.8
    WHITE_UFO_ATTACK_ALPHA: float = 0.0001
    ENEMY_EXPLOSION_FRAMES: int = 21
    KAMIKAZE_Y_THRESHOLD: float = 86.0

    # Mothership Explosion
    # Sequence: 1, 2, 1, 2, 3, 2, 3, 2, 3 (indices 0, 1, 0, 1, 2, 1, 2, 1, 2)
    # 8 frames per sprite step -> total 9 steps * 8 frames = 72 frames
    MOTHERSHIP_EXPLOSION_SEQUENCE: Tuple[int, ...] = (0, 1, 0, 1, 2, 1, 2, 1, 2)
    MOTHERSHIP_EXPLOSION_STEP_DURATION: int = 8
    MOTHERSHIP_HITBOX_SIZE: int = 16

    # Blue line constants
    BLUE_LINE_OFFSCREEN_Y = 500

    CHASING_METEOROID_MAX: int = 8
    CHASING_METEOROID_WAVE_MIN: int = 2
    CHASING_METEOROID_WAVE_MAX: int = 8
    CHASING_METEOROID_SPAWN_INTERVAL_MIN: int = 2
    CHASING_METEOROID_SPAWN_INTERVAL_MAX: int = 40
    CHASING_METEOROID_SPAWN_Y: float = 54.0
    CHASING_METEOROID_LANE_SPEED: float = 2.0
    CHASING_METEOROID_LANE_ALIGN_THRESHOLD: float = 1.5
    CHASING_METEOROID_CYCLE_DX: Tuple[int, ...] = (2, 0, 1, 0, 2, 0, 2, 0)
    CHASING_METEOROID_CYCLE_DY: Tuple[int, ...] = (1, 0, 0, 0, 1, 0, 0, 0)
    MOTHERSHIP_OFFSCREEN_POS: int = 500
    MOTHERSHIP_ANIM_X: Tuple[int, int, int, int, int, int, int] = (9, 9, 10, 10, 11, 12, 12)
    MOTHERSHIP_HEIGHT: int = 7
    MOTHERSHIP_EMERGE_Y: int = 44
    REJUVENATOR_SPAWN_PROB: float = 1/1500
    REJUVENATOR_STAGE_2_Y: float = 62.0
    REJUVENATOR_STAGE_3_Y: float = 93.0
    REJUVENATOR_STAGE_4_Y: float = 112.0
    DEATH_DURATION: int = 120


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
    stage_1 = (pos >= 100).astype(jnp.int32)
    stage_2 = (pos >= 80).astype(jnp.int32)
    stage_3 = (pos >= 0).astype(jnp.int32)
    result = jnp.where(bullet_type == laser_id, 0, stage_1 + stage_2 + stage_3)
    return result


def _get_bullet_alignment(pos: chex.Array, bullet_type: chex.Array, laser_id: int) -> chex.Array:
    stage_1 = (pos >= 100).astype(jnp.int32)
    stage_2 = (pos >= 80).astype(jnp.int32)
    stage_3 = (pos >= 0).astype(jnp.int32)
    # default alignment if smallest torpedo is +3
    # if bullet is laser, no offset
    result = jnp.where(bullet_type == laser_id, 0, 4 - (stage_1 + stage_2 + stage_3))
    return result


class LevelState(NamedTuple):
    player_pos: chex.Array
    player_vel: chex.Array
    white_ufo_left: chex.Array
    mothership_position: chex.Array
    mothership_timer: chex.Array
    mothership_stage: chex.Array
    player_shot_pos: chex.Array
    player_shot_vel: chex.Array
    torpedoes_left: chex.Array
    shooting_cooldown: chex.Array
    bullet_type: chex.Array

    # enemies
    enemy_type: chex.Array
    white_ufo_pos: chex.Array
    white_ufo_vel: chex.Array
    enemy_shot_pos: chex.Array
    enemy_shot_vel: chex.Array
    enemy_shot_timer: chex.Array
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

    line_positions: chex.Array
    blue_line_counter: chex.Array
    
    death_timer: chex.Array


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
        state = self.reset_level(1)
        observation = self._get_observation(state)
        return observation, state

    def _create_level_state(self, white_ufo_left=None) -> LevelState:
        white_ufo_left = white_ufo_left if white_ufo_left is not None else jnp.array(self.consts.WHITE_UFOS_PER_SECTOR)
        
        enemy_shot_offscreen = jnp.tile(
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 3),
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
            torpedoes_left=jnp.array(3),
            shooting_cooldown=jnp.array(0),
            bullet_type=jnp.array(self.consts.LASER_ID),
            enemy_type=jnp.array([0, 0, 0]),
            white_ufo_pos=jnp.array([[77.0, 77.0, 77.0], [43.0, 43.0, 43.0]]),
            white_ufo_vel=jnp.array([[-0.5, 0.5, 0.3], [0.0, 0.0, 0.0]]),
            enemy_shot_pos=enemy_shot_offscreen,
            enemy_shot_vel=jnp.zeros((3,), dtype=jnp.int32),
            enemy_shot_timer=jnp.zeros((3,), dtype=jnp.int32),
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
            line_positions=BLUE_LINE_INIT_TABLE[0],
            blue_line_counter=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
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
            (1, 3),
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
                torpedos_left,
                bullet_type,
                shooting_cooldown,
            ) = self._player_step(state, action)

            rngs = jax.random.split(state.rng, 6)
            next_rng = rngs[0]
            ufo_keys = rngs[1:4]
            meteoroid_key = rngs[4]
            rejuvenator_key = rngs[5]

            ufo_update = self._advance_white_ufos(state, ufo_keys)
            (
                white_ufo_pos,
                player_shot_position,
                white_ufo_pattern_id,
                white_ufo_pattern_timer,
                white_ufo_left,
                score,
                hit_mask,
            ) = self._collision_handler(
                state,
                ufo_update.pos,
                player_shot_position,
                bullet_type,
                ufo_update.pattern_id,
                ufo_update.pattern_timer,
            )

            (
                chasing_meteoroid_pos,
                chasing_meteoroid_active,
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
                chasing_meteoroid_phase,
                chasing_meteoroid_frame,
                chasing_meteoroid_lane,
                chasing_meteoroid_side,
                player_shot_position,
                chasing_meteoroid_hit_mask,
            ) = self._chasing_meteoroid_bullet_collision(
                chasing_meteoroid_pos,
                chasing_meteoroid_active,
                chasing_meteoroid_phase,
                chasing_meteoroid_frame,
                chasing_meteoroid_lane,
                chasing_meteoroid_side,
                player_shot_position,
                bullet_type,
            )
            
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
            shot_x_screen = player_shot_position[0] + _get_bullet_alignment(player_shot_position[1], bullet_type, self.consts.LASER_ID)
            shot_y = player_shot_position[1]
            
            rejuv_shot_dist_x = jnp.abs(rejuv_x_screen - shot_x_screen)
            rejuv_shot_dist_y = jnp.abs(rejuv_y - shot_y)
            
            is_laser = bullet_type == self.consts.LASER_ID
            bullet_radius = jnp.where(is_laser, jnp.array(self.consts.LASER_HIT_RADIUS), jnp.array(self.consts.TORPEDO_HIT_RADIUS))
            
            rejuv_hit_by_shot = jnp.logical_and.reduce(jnp.array([
                rejuv_active,
                jnp.logical_not(rejuv_dead),
                rejuv_shot_dist_x <= bullet_radius[0] + 2.0,
                rejuv_shot_dist_y <= bullet_radius[1] + 2.0,
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
            
            box_size = self.consts.MOTHERSHIP_HITBOX_SIZE
            ms_center_x = ms_pos + 8.0 
            ms_center_y = ms_y + 3.5 
            
            shot_x = player_shot_position[0] + _get_bullet_alignment(player_shot_position[1], bullet_type, self.consts.LASER_ID)
            shot_y = player_shot_position[1]
            
            dx = jnp.abs(shot_x - ms_center_x)
            dy = jnp.abs(shot_y - ms_center_y)
            
            shot_active = shot_y < self.consts.BOTTOM_CLIP 
            is_torpedo = bullet_type == self.consts.TORPEDO_ID
            ms_vulnerable = ms_stage == 2
            
            half_size = box_size / 2.0
            hit_mothership = (dx < half_size) & (dy < half_size) & shot_active & is_torpedo & ms_vulnerable
            player_shot_position = jnp.where(hit_mothership, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)
            
            ufo_explosion_frame, ufo_explosion_pos = self._update_enemy_explosions(
                state.level.ufo_explosion_frame,
                state.level.ufo_explosion_pos,
                hit_mask,
                ufo_update.pos,
            )
            chasing_meteoroid_explosion_frame, chasing_meteoroid_explosion_pos = self._update_enemy_explosions(
                state.level.chasing_meteoroid_explosion_frame,
                state.level.chasing_meteoroid_explosion_pos,
                chasing_meteoroid_hit_mask,
                pre_collision_meteoroid_pos,
            )
            enemy_shot_pos, enemy_shot_lane, enemy_shot_timer, shot_hit_count = self._enemy_shot_step(
                state,
                white_ufo_pos,
                white_ufo_pattern_id,
                white_ufo_pattern_timer,
            )

            # Player-UFO collision check
            ufo_x = white_ufo_pos[0] + _get_ufo_alignment(white_ufo_pos[1])
            ufo_y = white_ufo_pos[1]
            player_left = player_x
            player_right = player_x + self.consts.PLAYER_WIDTH
            player_y = float(self.consts.PLAYER_POS_Y)

            ufo_hits = jnp.logical_and.reduce(jnp.array([
                ufo_y >= player_y - 4.0,
                ufo_y <= player_y + 10.0,
                ufo_x >= player_left - 2.0,
                ufo_x <= player_right + 2.0,
            ]))
            ufo_hit_count = jnp.sum(ufo_hits.astype(jnp.int32))
            
            chasing_meteoroid_x = chasing_meteoroid_pos[0] + _get_ufo_alignment(chasing_meteoroid_pos[1]).astype(chasing_meteoroid_pos.dtype)
            chasing_meteoroid_left = chasing_meteoroid_x
            chasing_meteoroid_right = chasing_meteoroid_x + float(self.consts.ENEMY_WIDTH)
            chasing_meteoroid_top = chasing_meteoroid_pos[1]
            chasing_meteoroid_bottom = chasing_meteoroid_pos[1] + float(self.consts.ENEMY_HEIGHT)
            player_bottom = player_y + float(self.consts.PLAYER_HEIGHT)
            chasing_meteoroid_hits = jnp.logical_and.reduce(jnp.array([
                chasing_meteoroid_active,
                chasing_meteoroid_right >= player_left,
                chasing_meteoroid_left <= player_right,
                chasing_meteoroid_bottom >= player_y,
                chasing_meteoroid_top <= player_bottom,
            ]))
            chasing_meteoroid_hit_count = jnp.sum(chasing_meteoroid_hits.astype(jnp.int32))
            
            rejuv_hit_player = jnp.logical_and.reduce(jnp.array([
                rejuv_active,
                rejuv_y >= player_y - 4.0,
                rejuv_y <= player_y + 10.0,
                rejuv_x_screen >= player_left - 2.0,
                rejuv_x_screen <= player_right + 2.0,
            ]))
            
            gain_life = jnp.logical_and(rejuv_hit_player, jnp.logical_not(rejuv_dead))
            lose_life_rejuv = jnp.logical_and(rejuv_hit_player, rejuv_dead)
            
            rejuv_active = jnp.where(rejuv_hit_player, False, rejuv_active)
            rejuv_pos = jnp.where(rejuv_hit_player, jnp.array(self.consts.ENEMY_OFFSCREEN_POS), rejuv_pos)

            chasing_meteoroid_offscreen = jnp.tile(
                jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=chasing_meteoroid_pos.dtype).reshape(2, 1),
                (1, self.consts.CHASING_METEOROID_MAX),
            )
            chasing_meteoroid_active = jnp.where(chasing_meteoroid_hits, False, chasing_meteoroid_active)
            chasing_meteoroid_pos = jnp.where(chasing_meteoroid_hits[None, :], chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_phase = jnp.where(chasing_meteoroid_hits, 0, chasing_meteoroid_phase)
            chasing_meteoroid_frame = jnp.where(chasing_meteoroid_hits, 0, chasing_meteoroid_frame)
            chasing_meteoroid_lane = jnp.where(chasing_meteoroid_hits, 0, chasing_meteoroid_lane)
            chasing_meteoroid_side = jnp.where(chasing_meteoroid_hits, 1, chasing_meteoroid_side)
            reached_player = jnp.logical_and(
                chasing_meteoroid_active,
                chasing_meteoroid_pos[1] >= player_y,
            )
            chasing_meteoroid_active = jnp.where(reached_player, False, chasing_meteoroid_active)
            chasing_meteoroid_pos = jnp.where(reached_player[None, :], chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_phase = jnp.where(reached_player, 0, chasing_meteoroid_phase)
            chasing_meteoroid_frame = jnp.where(reached_player, 0, chasing_meteoroid_frame)
            chasing_meteoroid_lane = jnp.where(reached_player, 0, chasing_meteoroid_lane)
            chasing_meteoroid_side = jnp.where(reached_player, 1, chasing_meteoroid_side)

            hit_count = shot_hit_count + ufo_hit_count + chasing_meteoroid_hit_count + lose_life_rejuv.astype(jnp.int32)

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

            white_ufo_vel = jnp.where(hit_mask[None, :], 0.0, ufo_update.vel)
            white_ufo_time_on_lane = jnp.where(hit_mask, 0, ufo_update.time_on_lane)
            white_ufo_attack_time = jnp.where(hit_mask, 0, ufo_update.attack_time)

            active_count = jnp.minimum(white_ufo_left.astype(jnp.int32), 3)
            active_mask = jnp.arange(3, dtype=jnp.int32) < active_count
            ufo_offscreen = jnp.tile(jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=white_ufo_pos.dtype).reshape(2, 1), (1, 3))
            
            white_ufo_pos = jnp.where(active_mask[None, :], white_ufo_pos, ufo_offscreen)
            white_ufo_vel = jnp.where(active_mask[None, :], white_ufo_vel, 0.0)
            white_ufo_time_on_lane = jnp.where(active_mask, white_ufo_time_on_lane, 0)
            white_ufo_attack_time = jnp.where(active_mask, white_ufo_attack_time, 0)
            white_ufo_pattern_id = jnp.where(active_mask, white_ufo_pattern_id, int(WhiteUFOPattern.IDLE))
            white_ufo_pattern_timer = jnp.where(active_mask, white_ufo_pattern_timer, 0)

            enemy_shot_offscreen = jnp.tile(jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=enemy_shot_pos.dtype).reshape(2, 1), (1, 3))
            enemy_shot_pos = jnp.where(active_mask[None, :], enemy_shot_pos, enemy_shot_offscreen)
            enemy_shot_timer = jnp.where(active_mask, enemy_shot_timer, 0)
            enemy_shot_lane = jnp.where(active_mask, enemy_shot_lane, 0)

            enemy_shot_pos = jnp.where(sector_advanced, enemy_shot_offscreen, enemy_shot_pos)
            enemy_shot_timer = jnp.where(sector_advanced, 0, enemy_shot_timer)
            enemy_shot_lane = jnp.where(sector_advanced, 0, enemy_shot_lane)
            chasing_meteoroid_pos = jnp.where(sector_advanced, chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_active = jnp.where(sector_advanced, False, chasing_meteoroid_active)
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

            white_ufo_pos = jnp.where(is_dying_sequence, ufo_offscreen, white_ufo_pos)
            enemy_shot_pos = jnp.where(is_dying_sequence, enemy_shot_offscreen, enemy_shot_pos)
            chasing_meteoroid_pos = jnp.where(is_dying_sequence, chasing_meteoroid_offscreen, chasing_meteoroid_pos)
            chasing_meteoroid_active = jnp.where(is_dying_sequence, False, chasing_meteoroid_active)
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

            mothership_position = jnp.where(is_dying_sequence, self.consts.MOTHERSHIP_OFFSCREEN_POS, mothership_position)
            mothership_timer = jnp.where(is_dying_sequence, 0, mothership_timer)
            mothership_stage = jnp.where(is_dying_sequence, 0, mothership_stage)
            vel_x = jnp.where(is_dying_sequence, 0.0, vel_x)
            player_shot_position = jnp.where(is_dying_sequence, jnp.array(self.consts.BULLET_OFFSCREEN_POS), player_shot_position)

            next_step = state.steps + 1
            new_level_state = LevelState(
                player_pos=player_x, player_vel=vel_x, white_ufo_left=white_ufo_left,
                mothership_position=mothership_position, mothership_timer=mothership_timer,
                mothership_stage=mothership_stage, player_shot_pos=player_shot_position,
                player_shot_vel=player_shot_velocity, torpedoes_left=torpedos_left,
                shooting_cooldown=shooting_cooldown, bullet_type=bullet_type,
                enemy_type=jnp.array([0, 0, 0]), white_ufo_pos=white_ufo_pos,
                white_ufo_vel=white_ufo_vel, enemy_shot_pos=enemy_shot_pos,
                enemy_shot_vel=enemy_shot_lane, enemy_shot_timer=enemy_shot_timer,
                white_ufo_time_on_lane=white_ufo_time_on_lane, white_ufo_attack_time=white_ufo_attack_time,
                white_ufo_pattern_id=white_ufo_pattern_id, white_ufo_pattern_timer=white_ufo_pattern_timer,
                ufo_explosion_frame=ufo_explosion_frame, ufo_explosion_pos=ufo_explosion_pos,
                chasing_meteoroid_explosion_frame=chasing_meteoroid_explosion_frame,
                chasing_meteoroid_explosion_pos=chasing_meteoroid_explosion_pos,
                chasing_meteoroid_pos=chasing_meteoroid_pos, chasing_meteoroid_active=chasing_meteoroid_active,
                chasing_meteoroid_phase=chasing_meteoroid_phase, chasing_meteoroid_frame=chasing_meteoroid_frame,
                chasing_meteoroid_lane=chasing_meteoroid_lane, chasing_meteoroid_side=chasing_meteoroid_side,
                chasing_meteoroid_spawn_timer=chasing_meteoroid_spawn_timer,
                chasing_meteoroid_remaining=chasing_meteoroid_remaining,
                chasing_meteoroid_wave_active=chasing_meteoroid_wave_active,
                rejuvenator_pos=rejuv_pos, rejuvenator_active=rejuv_active,
                rejuvenator_dead=rejuv_dead, rejuvenator_frame=rejuv_frame,
                rejuvenator_lane=rejuv_lane,
                line_positions=line_positions, blue_line_counter=blue_line_counter,
                death_timer=next_death_timer,
            )

            reset_level_state = self._create_level_state(white_ufo_left=white_ufo_left)
            final_level_state = jax.tree_util.tree_map(
                lambda normal, reset: jnp.where(jnp.logical_or(just_died, sector_advanced), reset, normal),
                new_level_state, reset_level_state
            )
            
            lives_after_gain = state.lives + gain_life.astype(jnp.int32)
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
                blue_line_counter=blue_line_counter
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

        # Get inputs
        press_right = jnp.any(
            jnp.array(
                [action == Action.RIGHT,action == Action.UPRIGHT,action == Action.DOWNRIGHT, action == Action.RIGHTFIRE]
            )
        )

        press_left = jnp.any(
            jnp.array(
                [action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT, action == Action.LEFTFIRE]
            )
        )

        press_up = jnp.any(
            jnp.array(
                [action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT]
            )
        )
        fire_types = jnp.array([Action.FIRE, Action.DOWNFIRE, Action.LEFTFIRE, Action.RIGHTFIRE, Action.UPLEFTFIRE, Action.UPRIGHTFIRE, Action.DOWNLEFTFIRE, Action.DOWNRIGHTFIRE])
        press_fire = jnp.any(
            jnp.array(
                jnp.isin(action, fire_types)
            )
        )

        is_in_lane = jnp.isin(x, jnp.array(self.consts.BOTTOM_OF_LANES)) # predicate: x is one of LANES

        v = jax.lax.cond(
            is_in_lane,          
            lambda v_: jnp.zeros_like(v_),          # then -> 0
            lambda v_: v_,                          # else -> keep v
            v,                                      # operand
        )
        
        v = jax.lax.cond(
            jnp.logical_or(press_left,press_right),
            lambda v_: (press_right.astype(v.dtype) - press_left.astype(v.dtype)) * self.consts.PLAYER_SPEED,    
            lambda v_: v_,
            v,
        )
        x_before_change = x
        x = jnp.clip(x + v, self.consts.LEFT_CLIP_PLAYER, self.consts.RIGHT_CLIP_PLAYER - self.consts.PLAYER_WIDTH)

        ####### Ab hier von Lasse shot gedöns
        bullet_exists= self._bullet_infos(state)
        shooting_cooldown = state.level.shooting_cooldown

        can_spawn_bullet = jnp.logical_and(jnp.logical_not(bullet_exists), is_in_lane)
        can_spawn_bullet = jnp.logical_and(can_spawn_bullet, shooting_cooldown == 0)
        
        can_spawn_torpedo = jnp.logical_and(can_spawn_bullet, state.level.torpedoes_left >= 1)

        new_laser = jnp.logical_and(press_fire, can_spawn_bullet)
        new_torpedo = jnp.logical_and(press_up, can_spawn_torpedo)
        new_bullet = jnp.logical_or(new_torpedo, new_laser)

        lanes = jnp.array(self.consts.BOTTOM_OF_LANES)
        lane_velocities = jnp.array(self.consts.BOTTOM_TO_TOP_LANE_VECTORS, dtype=jnp.float32)
        lane_index = jnp.argmax(x_before_change == lanes) 
        lane_velocity = lane_velocities[lane_index]

        shot_velocity = jnp.where(
            new_bullet,
            lane_velocity,
            state.level.player_shot_vel,
        )

 
        pos_if_no_new = jnp.where(bullet_exists, (state.level.player_shot_pos - shot_velocity), jnp.array(self.consts.BULLET_OFFSCREEN_POS))
        shot_position = jnp.where(new_bullet, jnp.array([state.level.player_pos +3 , self.consts.MIN_BULLET_Y]), pos_if_no_new)

        bullet_exists = jnp.any(jnp.array([new_laser, bullet_exists, new_torpedo]))
        torpedos_left = state.level.torpedoes_left - new_torpedo
        bullet_type_if_new = jnp.where(new_laser, self.consts.LASER_ID, self.consts.TORPEDO_ID)
        bullet_type = jnp.where(new_bullet, bullet_type_if_new, state.level.bullet_type)

        current_cooldown = jnp.where(
            bullet_type_if_new == self.consts.LASER_ID,
            self.consts.PLAYER_LASER_COOLDOWN,
            self.consts.PLAYER_TORPEDO_COOLDOWN
        )

        shooting_cooldown = jnp.where(
            new_bullet,
            current_cooldown,
            jnp.maximum(shooting_cooldown - 1, 0)
        )

        #####
        return(x,v,shot_position, shot_velocity, torpedos_left, bullet_type, shooting_cooldown)

    def _collision_handler(
        self,
        state: BeamriderState,
        new_white_ufo_pos: chex.Array,
        new_shot_pos: chex.Array,
        new_bullet_type: chex.Array,
        current_patterns: chex.Array,
        current_timers: chex.Array,
    ):
        enemies_raw = new_white_ufo_pos.T

        # Collision should match what the player sees on screen:
        # - white UFO x-pos is shifted based on its y-pos (_get_ufo_alignment)
        # - player bullet x-pos is shifted based on its y-pos and type (_get_bullet_alignment)
        ufo_pos_screen = new_white_ufo_pos.at[0, :].add(
            _get_ufo_alignment(new_white_ufo_pos[1, :])
        )
        shot_pos_screen = new_shot_pos.at[0].add(
            _get_bullet_alignment(new_shot_pos[1], new_bullet_type, self.consts.LASER_ID)
        )

        distance_to_bullet = jnp.abs(ufo_pos_screen.T - shot_pos_screen)
        bullet_type_is_laser = new_bullet_type == self.consts.LASER_ID
        bullet_radius = jnp.where(
            bullet_type_is_laser,
            jnp.array(self.consts.LASER_HIT_RADIUS),
            jnp.array(self.consts.TORPEDO_HIT_RADIUS),
        )
        distance_bullet_radius = distance_to_bullet - bullet_radius
        hit_mask = jnp.array((distance_bullet_radius[:, 0] <= 0) & (distance_bullet_radius[:, 1] <= 0))
        hit_index = jnp.argmax(hit_mask)
        hit_exists = jnp.any(hit_mask)
        enemy_pos_after_hit = enemies_raw.at[hit_index].set(
            jnp.array([77.0, 43.0], dtype=enemies_raw.dtype)
        ).T

        # Reset pattern and timer for hit UFO
        new_patterns = jnp.where(hit_mask, int(WhiteUFOPattern.IDLE), current_patterns)
        new_timers = jnp.where(hit_mask, 0, current_timers)

        player_shot_pos = jnp.where(hit_exists, jnp.array(self.consts.BULLET_OFFSCREEN_POS), new_shot_pos)
        enemy_pos = jnp.where(hit_exists, enemy_pos_after_hit, new_white_ufo_pos)
        white_ufo_left = jnp.where(
            hit_exists,
            jnp.maximum(state.level.white_ufo_left - 1, 0),
            state.level.white_ufo_left,
        )
        clamped_sector = jnp.minimum(state.sector, 89)
        ufo_score = 40 + 4 * clamped_sector
        score = jnp.where(hit_exists, state.score + ufo_score, state.score)
        return (enemy_pos, player_shot_pos, new_patterns, new_timers, white_ufo_left, score, hit_mask)
    
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

        key_pattern, key_motion = jax.random.split(key)
        pattern_id, pattern_timer, time_on_lane, attack_time = self._white_ufo_update_pattern_state(
            white_ufo_position, time_on_lane, attack_time, pattern_id, pattern_timer, key_pattern
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
        
        should_respawn = jnp.logical_or(
            new_x < 0,
            jnp.logical_or(
                new_x > self.consts.SCREEN_WIDTH,
                new_y > self.consts.PLAYER_POS_Y
            )
        )

        white_ufo_position = jnp.where(should_respawn, jnp.array([77.0, 43.0]), jnp.array([new_x, new_y]))
        white_ufo_vel_x = jnp.where(should_respawn, 0.0, white_ufo_vel_x)
        white_ufo_vel_y = jnp.where(should_respawn, 0.0, white_ufo_vel_y)
        time_on_lane = jnp.where(should_respawn, 0, time_on_lane)
        attack_time = jnp.where(should_respawn, 0, attack_time)
        pattern_id = jnp.where(should_respawn, int(WhiteUFOPattern.IDLE), pattern_id)
        pattern_timer = jnp.where(should_respawn, 0, pattern_timer)

        active_count = jnp.minimum(state.level.white_ufo_left.astype(jnp.int32), 3)
        is_active = jnp.array(index, dtype=jnp.int32) < active_count
        offscreen_pos = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=white_ufo_position.dtype)
        white_ufo_position = jnp.where(is_active, white_ufo_position, offscreen_pos)
        white_ufo_vel_x = jnp.where(is_active, white_ufo_vel_x, 0.0)
        white_ufo_vel_y = jnp.where(is_active, white_ufo_vel_y, 0.0)
        time_on_lane = jnp.where(is_active, time_on_lane, 0)
        attack_time = jnp.where(is_active, attack_time, 0)
        pattern_id = jnp.where(is_active, pattern_id, int(WhiteUFOPattern.IDLE))
        pattern_timer = jnp.where(is_active, pattern_timer, 0)

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
        return jnp.logical_or(
            drop_straight,
            jnp.logical_or(
                drop_left,
                jnp.logical_or(
                    drop_right,
                    jnp.logical_or(retreat, jnp.logical_or(move_back, kamikaze))
                )
            ),
        )

    def _white_ufo_update_pattern_state(
        self,
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
        pattern_timer = jnp.maximum(pattern_timer - 1, jnp.zeros_like(pattern_timer))

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
        is_shoot_pattern = pattern_id == int(WhiteUFOPattern.SHOOT)
        is_engagement_pattern = jnp.logical_or(is_drop_pattern, is_shoot_pattern)
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

        pattern_finished_off_top = jnp.logical_and.reduce(jnp.array([
            jnp.logical_not(on_top_lane),
            is_engagement_pattern,
            pattern_timer == 0,
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

        def choose_chain_pattern(_):
            is_kamikaze_zone = position[1] >= self.consts.KAMIKAZE_Y_THRESHOLD
            pattern, duration = self._white_ufo_choose_pattern(
                key_chain_choice, 
                allow_shoot=allow_shoot, 
                prev_pattern=pattern_id,
                is_kamikaze_zone=is_kamikaze_zone
            )
            return pattern, duration

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
                is_kamikaze_zone=jnp.array(False)
            )
            return pattern, duration

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
        is_kamikaze_zone: chex.Array
    ):
        pattern_choices = jnp.array(
            [
                int(WhiteUFOPattern.DROP_STRAIGHT),
                int(WhiteUFOPattern.DROP_LEFT),
                int(WhiteUFOPattern.DROP_RIGHT),
                int(WhiteUFOPattern.SHOOT),
                int(WhiteUFOPattern.MOVE_BACK),
                int(WhiteUFOPattern.KAMIKAZE),
            ],
            dtype=jnp.int32,
        )
        pattern_probs = jnp.array(self.consts.WHITE_UFO_PATTERN_PROBS, dtype=jnp.float32)

        # Restriction: Cannot follow MOVE_BACK with DROP_STRAIGHT (idx 0)
        is_move_back = (prev_pattern == int(WhiteUFOPattern.MOVE_BACK))
        # Mask DROP_STRAIGHT (index 0) if prev was MOVE_BACK
        # Since we added Kamikaze at index 5, mask is length 6
        chain_mask = jnp.ones_like(pattern_probs).at[0].set(jnp.where(is_move_back, 0.0, 1.0))
        pattern_probs = pattern_probs * chain_mask

        # Mask out SHOOT (index 3) if not allowed. Allow others including Kamikaze (index 5)
        shoot_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
        pattern_probs = jnp.where(allow_shoot, pattern_probs, pattern_probs * shoot_mask)
        
        # Mask out KAMIKAZE (index 5) if not in zone
        kamikaze_mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
        pattern_probs = jnp.where(is_kamikaze_zone, pattern_probs, pattern_probs * kamikaze_mask)
        
        # Avoid division by zero if all probs masked (shouldn't happen with standard probs, but for safety/testing)
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
        
        cross_track = target_lane_x - x
        distance_to_lane = jnp.abs(cross_track)
        direction = jnp.sign(cross_track)

        def seek_lane(_):
            attack_vx = jnp.where(direction == 0, 0.0, direction * 0.5)
            retreat_vx = jnp.where(direction == 0, 0.0, direction * speed_factor * retreat_mult * 2.0)
            
            # For Kamikaze, use retreat lateral speed (fast seek)
            new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_kamikaze, retreat_vx, attack_vx))
            
            normal_vy = 0.25
            retreat_vy = -lane_vector[1] * speed_factor * retreat_mult
            move_back_vy = -lane_vector[1] * speed_factor
            kamikaze_vy = lane_vector[1] * speed_factor * retreat_mult

            new_vy = jnp.where(is_retreat, retreat_vy, normal_vy)
            new_vy = jnp.where(is_move_back, move_back_vy, new_vy)
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            
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
            
            new_vx = jnp.where(is_retreat, retreat_vx, jnp.where(is_move_back, move_back_vx, normal_vx))
            new_vx = jnp.where(is_kamikaze, kamikaze_vx, new_vx)
            
            new_vy = jnp.where(is_retreat, retreat_vy, jnp.where(is_move_back, move_back_vy, normal_vy))
            new_vy = jnp.where(is_kamikaze, kamikaze_vy, new_vy)
            
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
        offscreen = jnp.tile(offscreen_xy.reshape(2, 1), (1, 3))

        shot_pos = state.level.enemy_shot_pos.astype(jnp.float32)
        shot_lane = state.level.enemy_shot_vel.astype(jnp.int32)
        shot_timer = state.level.enemy_shot_timer.astype(jnp.int32)

        shot_active = shot_pos[1] <= float(self.consts.BOTTOM_CLIP)
        shot_timer = jnp.where(shot_active, shot_timer + 1, 0)

        shoot_duration = jnp.array(self.consts.WHITE_UFO_PATTERN_DURATIONS, dtype=jnp.int32)[
            int(WhiteUFOPattern.SHOOT)
        ]
        wants_spawn = jnp.logical_and(
            white_ufo_pattern_id == int(WhiteUFOPattern.SHOOT),
            white_ufo_pattern_timer == shoot_duration,
        )
        ufo_on_screen = white_ufo_pos[1] <= float(self.consts.BOTTOM_CLIP)
        ufo_not_on_top_lane = white_ufo_pos[1] > float(self.consts.TOP_CLIP)
        ufo_x = white_ufo_pos[0].astype(jnp.float32)
        ufo_y = white_ufo_pos[1].astype(jnp.float32)
        lane_x_at_ufo_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            ufo_y[None, :] - float(self.consts.TOP_CLIP)
        )
        closest_lane = jnp.argmin(jnp.abs(lane_x_at_ufo_y - ufo_x[None, :]), axis=0).astype(jnp.int32)
        allowed_shot_lane = jnp.logical_and(closest_lane > 0, closest_lane < 6)
        spawn = jnp.logical_and.reduce(
            jnp.array([
                wants_spawn,
                ufo_on_screen,
                ufo_not_on_top_lane,
                allowed_shot_lane,
                jnp.logical_not(shot_active),
            ])
        )

        spawn_y = jnp.clip(ufo_y + 4.0, float(self.consts.TOP_CLIP), float(self.consts.BOTTOM_CLIP))
        spawn_x = jnp.take(lanes_top_x, closest_lane) + jnp.take(lane_dx_over_dy, closest_lane) * (
            spawn_y - float(self.consts.TOP_CLIP)
        )
        spawn_pos = jnp.stack([spawn_x, spawn_y])
        shot_pos = jnp.where(spawn[None, :], spawn_pos, shot_pos)
        shot_lane = jnp.where(spawn, closest_lane, shot_lane)
        shot_timer = jnp.where(spawn, 0, shot_timer)
        shot_active = jnp.logical_or(shot_active, spawn)

        # Per-shot cadence (frame-by-frame):
        # sprite1 -> stand still -> move -> stand still -> sprite2 -> stand still -> move -> stand still -> ...
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
        player_right = player_left + float(self.consts.PLAYER_WIDTH)
        player_y = float(self.consts.PLAYER_POS_Y)

        shot_x = shot_pos[0] + _get_ufo_alignment(shot_pos[1])
        shot_y = shot_pos[1]

        hits = jnp.logical_and.reduce(jnp.array([
            shot_active,
            shot_y >= player_y - 7.0,
            shot_x >= player_left,
            shot_x <= player_right,
        ]))


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
        phase = state.level.chasing_meteoroid_phase
        frame = state.level.chasing_meteoroid_frame
        lane = state.level.chasing_meteoroid_lane
        side = state.level.chasing_meteoroid_side
        spawn_timer = state.level.chasing_meteoroid_spawn_timer
        remaining = state.level.chasing_meteoroid_remaining
        wave_active = state.level.chasing_meteoroid_wave_active

        spawn_window = white_ufo_left == 0

        key_wave, key_interval, key_side = jax.random.split(key, 3)
        wave_count = jax.random.randint(
            key_wave,
            (),
            self.consts.CHASING_METEOROID_WAVE_MIN,
            self.consts.CHASING_METEOROID_WAVE_MAX + 1,
        )
        start_wave = jnp.logical_and.reduce(jnp.array([
            spawn_window,
            jnp.logical_not(wave_active),
            state.level.mothership_stage == 0,
        ]))
        wave_active = jnp.where(start_wave, True, wave_active)
        remaining = jnp.where(start_wave, wave_count, remaining)
        spawn_timer = jnp.where(start_wave, 0, spawn_timer)

        wave_active = jnp.where(spawn_window, wave_active, False)
        remaining = jnp.where(spawn_window, remaining, 0)
        spawn_timer = jnp.where(spawn_window, spawn_timer, 0)

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
        phase_spawned = jnp.where(one_hot_bool, 0, phase)
        frame_spawned = jnp.where(one_hot_bool, 0, frame)
        lane_spawned = jnp.where(one_hot_bool, 0, lane)
        side_spawned = jnp.where(one_hot_bool, spawn_side, side)

        pos = jnp.where(should_spawn, pos_spawned, pos)
        active = jnp.where(should_spawn, active_spawned, active)
        phase = jnp.where(should_spawn, phase_spawned, phase)
        frame = jnp.where(should_spawn, frame_spawned, frame)
        lane = jnp.where(should_spawn, lane_spawned, lane)
        side = jnp.where(should_spawn, side_spawned, side)
        remaining = jnp.where(should_spawn, remaining - 1, remaining)
        spawn_timer = jnp.where(should_spawn, spawn_interval, spawn_timer)

        cycle_dx = jnp.array(self.consts.CHASING_METEOROID_CYCLE_DX, dtype=pos.dtype)
        cycle_dy = jnp.array(self.consts.CHASING_METEOROID_CYCLE_DY, dtype=pos.dtype)
        dy = jnp.take(cycle_dy, frame)
        new_y_a = pos[1] + dy

        lane_vectors = jnp.array(self.consts.TOP_TO_BOTTOM_LANE_VECTORS, dtype=pos.dtype)
        lanes_top_x = jnp.array(self.consts.TOP_OF_LANES, dtype=pos.dtype)
        lane_dx_over_dy = lane_vectors[:, 0] / lane_vectors[:, 1]

        lane_x_at_current_y = lanes_top_x[:, None] + lane_dx_over_dy[:, None] * (
            new_y_a[None, :] - float(self.consts.TOP_CLIP)
        )
        dx_dir = side.astype(pos.dtype)
        dx = jnp.take(cycle_dx, frame) * dx_dir
        new_x_a = pos[0] + dx

        new_y_b = pos[1] + float(self.consts.CHASING_METEOROID_LANE_SPEED)
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
        player_right = player_x + float(self.consts.PLAYER_WIDTH)
        align_x = _get_ufo_alignment(new_y_a).astype(new_x_a.dtype)
        aligned_x_a = new_x_a + align_x
        chasing_meteoroid_left = aligned_x_a
        chasing_meteoroid_right = aligned_x_a + float(self.consts.ENEMY_WIDTH)
        hits_x = jnp.logical_and(chasing_meteoroid_right >= player_left, chasing_meteoroid_left <= player_right)
        player_center = player_x + float(self.consts.PLAYER_WIDTH) / 2.0
        bottom_lanes = jnp.array(self.consts.BOTTOM_OF_LANES, dtype=jnp.float32)
        nearest_lane_idx = jnp.argmin(jnp.abs(bottom_lanes - player_center)).astype(jnp.int32)
        move_dir = jnp.sign(player_vel).astype(jnp.int32)
        target_lane_idx = jnp.clip(
            nearest_lane_idx + move_dir,
            0,
            bottom_lanes.shape[0] - 1,
        )
        preferred_lane_idx = target_lane_idx + 1
        preferred_lane_x = lane_x_at_current_y[preferred_lane_idx]
        preferred_ahead = jnp.where(
            side > 0,
            preferred_lane_x >= new_x_a,
            preferred_lane_x <= new_x_a,
        )
        ahead_mask = jnp.where(
            side[None, :] > 0,
            lane_x_at_current_y >= new_x_a[None, :],
            lane_x_at_current_y <= new_x_a[None, :],
        )
        lane_dist = jnp.abs(lane_x_at_current_y - player_center)
        masked_dist = jnp.where(ahead_mask, lane_dist, jnp.inf)
        ahead_lane_idx = jnp.argmin(masked_dist, axis=0).astype(jnp.int32)
        any_ahead = jnp.any(ahead_mask, axis=0)
        target_lane = jnp.where(
            preferred_ahead,
            preferred_lane_idx,
            jnp.where(any_ahead, ahead_lane_idx, preferred_lane_idx),
        )
        arm_now = active & (phase == 0) & hits_x
        new_phase = jnp.where(arm_now, 1, phase)
        new_lane = jnp.where(arm_now, target_lane, lane)

        target_lane_x = jnp.take_along_axis(lane_x_at_current_y, new_lane[None, :], axis=0).squeeze(0)
        dist_to_lane = jnp.abs(target_lane_x - new_x_a)
        is_on_lane = dist_to_lane <= float(self.consts.CHASING_METEOROID_LANE_ALIGN_THRESHOLD)

        start_descend = active & (new_phase >= 1) & is_on_lane
        new_phase = jnp.where(start_descend, 2, new_phase)

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
        new_phase = jnp.where(new_active, new_phase, 0)
        new_frame = jnp.where(new_active, new_frame, 0)
        new_lane = jnp.where(new_active, new_lane, 0)
        new_side = jnp.where(new_active, side, 1)

        return (
            new_pos,
            new_active,
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
        chasing_meteoroid_phase: chex.Array,
        chasing_meteoroid_frame: chex.Array,
        chasing_meteoroid_lane: chex.Array,
        chasing_meteoroid_side: chex.Array,
        player_shot_pos: chex.Array,
        bullet_type: chex.Array,
    ):
        is_torpedo = bullet_type == self.consts.TORPEDO_ID
        shot_x = player_shot_pos[0] + _get_bullet_alignment(
            player_shot_pos[1], bullet_type, self.consts.LASER_ID
        )
        shot_y = player_shot_pos[1]
        shot_active = shot_y < float(self.consts.BOTTOM_CLIP)

        chasing_meteoroid_x = chasing_meteoroid_pos[0] + _get_ufo_alignment(chasing_meteoroid_pos[1]).astype(chasing_meteoroid_pos.dtype)
        chasing_meteoroid_screen_pos = jnp.stack([chasing_meteoroid_x, chasing_meteoroid_pos[1]]).T
        distance_to_bullet = jnp.abs(chasing_meteoroid_screen_pos - jnp.array([shot_x, shot_y], dtype=chasing_meteoroid_pos.dtype))
        bullet_radius = jnp.array(self.consts.TORPEDO_HIT_RADIUS, dtype=chasing_meteoroid_pos.dtype)
        chasing_meteoroid_radius = jnp.array(
            [self.consts.ENEMY_WIDTH / 2.0, self.consts.ENEMY_HEIGHT / 2.0],
            dtype=chasing_meteoroid_pos.dtype,
        )
        hit_radius = bullet_radius + chasing_meteoroid_radius
        hit_mask = (
            chasing_meteoroid_active
            & is_torpedo
            & shot_active
            & (distance_to_bullet[:, 0] <= hit_radius[0])
            & (distance_to_bullet[:, 1] <= hit_radius[1])
        )
        hit_exists = jnp.any(hit_mask)
        hit_index = jnp.argmax(hit_mask)
        hit_one_hot = jax.nn.one_hot(hit_index, self.consts.CHASING_METEOROID_MAX, dtype=chasing_meteoroid_pos.dtype)
        hit_one_hot_bool = hit_one_hot.astype(jnp.bool_)

        offscreen = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=chasing_meteoroid_pos.dtype)
        pos_after_hit = chasing_meteoroid_pos + (offscreen[:, None] - chasing_meteoroid_pos) * hit_one_hot[None, :]
        active_after_hit = jnp.where(hit_one_hot_bool, False, chasing_meteoroid_active)
        phase_after_hit = jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_phase)
        frame_after_hit = jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_frame)
        lane_after_hit = jnp.where(hit_one_hot_bool, 0, chasing_meteoroid_lane)
        side_after_hit = jnp.where(hit_one_hot_bool, 1, chasing_meteoroid_side)

        chasing_meteoroid_pos = jnp.where(hit_exists, pos_after_hit, chasing_meteoroid_pos)
        chasing_meteoroid_active = jnp.where(hit_exists, active_after_hit, chasing_meteoroid_active)
        chasing_meteoroid_phase = jnp.where(hit_exists, phase_after_hit, chasing_meteoroid_phase)
        chasing_meteoroid_frame = jnp.where(hit_exists, frame_after_hit, chasing_meteoroid_frame)
        chasing_meteoroid_lane = jnp.where(hit_exists, lane_after_hit, chasing_meteoroid_lane)
        chasing_meteoroid_side = jnp.where(hit_exists, side_after_hit, chasing_meteoroid_side)
        player_shot_pos = jnp.where(
            hit_exists,
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=player_shot_pos.dtype),
            player_shot_pos,
        )

        return (
            chasing_meteoroid_pos,
            chasing_meteoroid_active,
            chasing_meteoroid_phase,
            chasing_meteoroid_frame,
            chasing_meteoroid_lane,
            chasing_meteoroid_side,
            player_shot_pos,
            hit_mask,
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
        
        # FOR TESTING ONLY: spawn immediately after init
        TEST_SPAWN = True
        if TEST_SPAWN:
            is_first_frame_after_init = state.level.blue_line_counter == len(BLUE_LINE_INIT_TABLE)
            should_spawn = jnp.logical_and(jnp.logical_not(active), jnp.logical_or(spawn_roll < self.consts.REJUVENATOR_SPAWN_PROB, is_first_frame_after_init))
        else:
            should_spawn = jnp.logical_and(jnp.logical_not(active), spawn_roll < self.consts.REJUVENATOR_SPAWN_PROB)
        
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
        shot_y = state.level.player_shot_pos[1]
        bullet_type = state.level.bullet_type

        laser_exists = jnp.all(jnp.array([shot_y >= self.consts.MAX_LASER_Y, 
                                          shot_y <= self.consts.MIN_BULLET_Y,
                                          bullet_type == self.consts.LASER_ID]))
        torpedo_exists = jnp.all(jnp.array([shot_y >= self.consts.MAX_TORPEDO_Y,
                                           shot_y <= self.consts.MIN_BULLET_Y,
                                           bullet_type == self.consts.TORPEDO_ID]))
        bullet_exists = jnp.logical_or(torpedo_exists, laser_exists)
        
        return(bullet_exists)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BeamriderState):
        is_init = state.level.blue_line_counter < len(BLUE_LINE_INIT_TABLE)
        
        ufo_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 3),
        )
        enemy_shot_offscreen = jnp.tile(
            jnp.array(self.consts.BULLET_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, 3),
        )
        chasing_meteoroid_offscreen = jnp.tile(
            jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32).reshape(2, 1),
            (1, self.consts.CHASING_METEOROID_MAX),
        )
        rejuv_offscreen = jnp.array(self.consts.ENEMY_OFFSCREEN_POS, dtype=jnp.float32)
        
        # Create a state for rendering where enemies are offscreen if initializing
        render_level = state.level._replace(
            white_ufo_pos=jnp.where(is_init, ufo_offscreen, state.level.white_ufo_pos),
            enemy_shot_pos=jnp.where(is_init, enemy_shot_offscreen, state.level.enemy_shot_pos),
            chasing_meteoroid_pos=jnp.where(is_init, chasing_meteoroid_offscreen, state.level.chasing_meteoroid_pos),
            rejuvenator_pos=jnp.where(is_init, rejuv_offscreen, state.level.rejuvenator_pos)
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
    #     player_dimensions = (self.consts.PLAYER_HEIGHT, self.consts.PLAYER_WIDTH, 4)
    #     player_sprite = jnp.tile(jnp.array(player_color_rgba, dtype=jnp.uint8), (*player_dimensions[:2], 1))
    #     return player_sprite
    
    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [
            {'name': 'background_sprite', 'type': 'background', 'file': 'new_background.npy'},
            {'name': 'player_sprite', 'type': 'group', 'files': [f'Player/Player_{i}.npy' for i in range(1, 17)]},
            {'name': 'dead_player', 'type': 'single', 'file': 'Dead_Player.npy'},
            {'name': 'white_ufo', 'type': 'group', 'files': ['White_Ufo_Stage_1.npy', 'White_Ufo_Stage_2.npy', 'White_Ufo_Stage_3.npy', 'White_Ufo_Stage_4.npy', 'White_Ufo_Stage_5.npy', 'White_Ufo_Stage_6.npy', 'White_Ufo_Stage_7.npy']},
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
        raster = self._render_chasing_meteoroids(raster, state)
        raster = self._render_rejuvenator(raster, state)
        raster = self._render_hud(raster, state)
        raster = self._render_mothership(raster, state)
        return self.jr.render_from_palette(raster, self.PALETTE)

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
        
        for idx in range(3):
            # Normal logic: render if idx < state.lives - 1 (Bonus HP display)
            # If lives=3, we render indices 0, 1 (2 icons).
            # If lives=1, we render nothing.
            
            # Flashing logic: if is_dead, we are about to lose a life.
            # We want to flash the icon that represents the life we are losing.
            # If we have 3 lives (2 icons), we lose one -> 2 lives (1 icon).
            # The icon disappearing is at index 1.
            # So we flash if idx == state.lives - 2.
            
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
            
            pos_x = jnp.where(is_visible, 32 + (idx * 9), 500)
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
        raster = self.jr.render_at_clipped(
            raster,
            state.level.player_shot_pos[0] + _get_bullet_alignment(state.level.player_shot_pos[1], state.level.bullet_type, self.consts.LASER_ID),
            state.level.player_shot_pos[1],
            bullet_mask,
        )
        return raster

    def _render_enemy_shots(self, raster, state):
        enemy_shot_masks = self.SHAPE_MASKS["enemy_shot"]
        for idx in range(3):
            timer = state.level.enemy_shot_timer[idx]
            sprite_idx = (jnp.floor_divide(timer, 4) % 2).astype(jnp.int32)
            y_pos = jnp.where(
                state.level.enemy_shot_pos[1][idx] <= self.consts.BOTTOM_CLIP,
                state.level.enemy_shot_pos[1][idx],
                500,
            )
            raster = self.jr.render_at_clipped(
                raster, state.level.enemy_shot_pos[0][idx] + _get_ufo_alignment(y_pos), y_pos, enemy_shot_masks[sprite_idx]
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
