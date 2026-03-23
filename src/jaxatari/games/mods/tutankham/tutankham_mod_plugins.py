import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_tutankham import TutankhamState


class NightModeMod(JaxAtariPostStepModPlugin):
    pass


class MimicModeMod(JaxAtariPostStepModPlugin):
    pass


class UpsideDownMod(JaxAtariInternalModPlugin):
    pass


class WhipMod(JaxAtariInternalModPlugin):
    pass
    