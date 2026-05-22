import os
from functools import partial
from typing import List, NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
from jax import Array
from flax import struct

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

