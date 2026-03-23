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
    constants_overrides = {
        "MAP_CHECKPOINTS": jnp.array([
            # MAP 1
            [
                [588, 800, 18, 684],
                [405, 587, 80, 586],
                [201, 404, 12, 403],
                [0,   200, 78, 199],
            ],
            # MAP 2
            [
                [573, 800, 19, 634],
                [425, 572, 24, 572],
                [261, 426, 78, 426],
                [0,   260, 78, 259],
            ],
            # MAP 3
            [
                [553, 800,  107, 715],
                [401, 552,  98,  550],
                [269, 400,  78,  396],
                [0,   268,  39,  248],
            ],
            # MAP 4
            [
                [531, 800, 77,  719],
                [391, 532, 119, 531],
                [204, 392, 18,  391],
                [0 ,  203, 30,  203],
            ],
        ], dtype=jnp.int32),

        "MAP_GOAL_POSITIONS": jnp.array([
            [[134, 61]],  # MAP 1
            [[136, 60]],  # MAP 2
            [[16,  93]],  # MAP 3
            [[82,  95]]   # MAP 4
        ], dtype=jnp.int32)
    }


class WhipMod(JaxAtariInternalModPlugin):
    pass
    