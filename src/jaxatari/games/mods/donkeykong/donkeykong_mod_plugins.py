import os
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_donkeykong import Ladder
from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir

class SpeedrunnerMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "MARIO_MOVING_SPEED": jnp.float32(0.67),
        "MARIO_CLIMBING_SPEED": jnp.float32(0.666),
    }

class AggressiveBarrelsMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "BARREL_MOVING_SPEED": 2,
        "BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_1": jnp.float32(0.35),
        "BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN_ROUND_2": jnp.float32(0.60),
    }

class PacifistMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "LEVEL_1_HAMMER_Y": -100,
        "LEVEL_1_HAMMER_X": -100,
        "LEVEL_2_HAMMER_Y": -100,
        "LEVEL_2_HAMMER_X": -100,
    }

class NoBarrelsMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "SPAWN_STEP_COUNTER_BARREL": 9999999,
        # Set start Y (horizontal coordinate) to far left so it never enters screen
        "BARREL_START_Y": -9999,
        "BARREL_START_X": -9999,
    }


class ShiftedLaddersMod(JaxAtariInternalModPlugin):
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "file": "donkeyKong_background_level_1_shifted.npy",
        },
        "background_level_2": {
            "name": "background_level_2",
            "type": "single",
            "file": "donkeyKong_background_level_2_shifted.npy",
        }
    }

    @partial(jax.jit, static_argnums=(0,))
    def init_ladders_for_level(self, level: int) -> Ladder:
        Ladder_level_1 = Ladder(
            stage=jnp.array([6, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1, -1, -1, -1], dtype=jnp.int32),
            climbable=jnp.array([True, False, True, True, True, False, False, True, True, True, True, False, True, False, False, False]),
            start_y=jnp.array([60, 84, 83, 111, 112, 115, 143, 139, 137, 167, 169, 193, 193, -1, -1, -1], dtype=jnp.int32),
            start_x=jnp.array([60, 84, 96, 56, 76, 108, 52, 96, 116, 56, 88, 60, 116, -1, -1, -1], dtype=jnp.int32),
            end_y=jnp.array([35, 60, 60, 86, 80, 76, 110, 114, 116, 142, 132, 167, 172, -1, -1, -1], dtype=jnp.int32),
            end_x=jnp.array([60, 84, 96, 56, 76, 108, 52, 96, 116, 56, 88, 60, 116, -1, -1, -1], dtype=jnp.int32),
        )

        Ladder_level_2 = Ladder(
            stage=jnp.array([4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1], dtype=jnp.int32),
            climbable=jnp.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]),
            start_y=jnp.array([172, 172, 172, 172, 144, 144, 143, 144, 116, 116, 115, 116, 88, 88, 87, 88], dtype=jnp.int32),
            start_x=jnp.array([50, 70, 106, 126, 50, 70, 106, 126, 50, 70, 106, 126, 50, 70, 106, 126], dtype=jnp.int32),
            end_y=jnp.array([144, 144, 143, 144, 116, 116, 115, 148, 88, 88, 87, 88, 60, 60, 59, 60], dtype=jnp.int32),
            end_x=jnp.array([50, 70, 106, 126, 50, 70, 106, 126, 50, 70, 106, 126, 50, 70, 106, 126], dtype=jnp.int32),
        )

        return jax.lax.cond(
            level == 1,
            lambda _: Ladder_level_1,
            lambda _: Ladder_level_2,
            operand=None
        )
