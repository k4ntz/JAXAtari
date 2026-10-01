import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.games.jax_donkeykong import Ladder

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
        "ENABLE_BARRELS": False,
        "SPAWN_STEP_COUNTER_BARREL": 9999999,
    }


class ShiftedLaddersMod(JaxAtariInternalModPlugin):
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


class CenterLaddersMod(JaxAtariInternalModPlugin):
    """
    Shifts ladders inward toward the screen center in both Level 1 and Level 2.
    - Level 1: Outer climbable ascent ladders shift inward toward the center,
      snapping vertical endpoints to the sloped girders. Inner ladders are adjusted
      for spacing.
    - Level 2: All 4 ladder columns shift inward symmetrically: [40, 60, 96, 116] -> [52, 68, 88, 104].
    """
    @partial(jax.jit, static_argnums=(0,))
    def init_ladders_for_level(self, level: int) -> Ladder:
        from jaxatari.games.jax_donkeykong import JaxDonkeyKong
        base_ladders = JaxDonkeyKong.init_ladders_for_level(self._env, level)

        # Level 1 adjustments:
        # Outer climbable ladders shifted inward:
        #   Stage 1 outer (Ladder 12): x=106 -> 94 (render segment 17: [96, 172], size [4, 21])
        #   Stage 2 outer (Ladder 9):  x=46 -> 58  (render segment 13: [60, 144], size [4, 21])
        #   Stage 3 outer (Ladder 8):  x=106 -> 94 (render segment 12: [96, 116], size [4, 21])
        #   Stage 3 inner (Ladder 7):  x=86 -> 82  (render segment 11: [84, 116], size [4, 21])
        #   Stage 4 outer (Ladder 3):  x=46 -> 58  (render segment 5:  [60, 88],  size [4, 21])
        #   Stage 4 inner (Ladder 4):  x=66 -> 70  (render segment 6:  [72, 88],  size [4, 21])
        #   Stage 5 outer (Ladder 2):  x=106 -> 94 (render segment 4:  [96, 64],  size [4, 17])
        l1_start_x = base_ladders.start_x.at[12].set(94).at[9].set(58).at[8].set(94).at[7].set(82).at[3].set(58).at[4].set(70).at[2].set(94)
        l1_end_x   = base_ladders.end_x.at[12].set(94).at[9].set(58).at[8].set(94).at[7].set(82).at[3].set(58).at[4].set(70).at[2].set(94)
        l1_start_y = base_ladders.start_y.at[12].set(185).at[9].set(159).at[8].set(131).at[7].set(132).at[3].set(103).at[4].set(104).at[2].set(75)
        l1_end_y   = base_ladders.end_y.at[12].set(162).at[9].set(134).at[8].set(106).at[7].set(106).at[3].set(78).at[4].set(78).at[2].set(52)

        l1_render_pos = base_ladders.render_positions.at[17].set(jnp.array([96, 172])).at[13].set(jnp.array([60, 144])).at[12].set(jnp.array([96, 116])).at[11].set(jnp.array([84, 116])).at[5].set(jnp.array([60, 88])).at[6].set(jnp.array([72, 88])).at[4].set(jnp.array([96, 64]))
        l1_render_sizes = base_ladders.render_sizes.at[17].set(jnp.array([4, 21])).at[13].set(jnp.array([4, 21])).at[12].set(jnp.array([4, 21])).at[5].set(jnp.array([4, 21])).at[4].set(jnp.array([4, 17]))

        ladders_level_1 = base_ladders.replace(
            start_x=l1_start_x,
            end_x=l1_end_x,
            start_y=l1_start_y,
            end_y=l1_end_y,
            render_positions=l1_render_pos,
            render_sizes=l1_render_sizes,
        )

        # Level 2 adjustments:
        # Center even further to clear the trap holes at x=52..56 and x=104..108:
        # Columns centered symmetrically between traps: [62, 74, 86, 98]
        l2_cols = jnp.array([62, 74, 86, 98], dtype=jnp.int32)
        l2_all_cols = jnp.tile(l2_cols, 5)
        l2_start_x = l2_all_cols[:16]
        l2_end_x = l2_all_cols[:16]
        l2_render_pos = base_ladders.render_positions.at[:20, 0].set(l2_all_cols)

        ladders_level_2 = base_ladders.replace(
            start_x=l2_start_x,
            end_x=l2_end_x,
            render_positions=l2_render_pos,
        )

        return jax.lax.cond(
            level == 1,
            lambda _: ladders_level_1,
            lambda _: ladders_level_2,
            operand=None
        )


class StartTopPlatformMod(JaxAtariInternalModPlugin):
    """
    Mod where Mario starts from the platform just below the top one (Stage 6, on the right side).
    """
    constants_overrides = {
        "LEVEL_1_MARIO_START_X": jnp.float32(43.0),
        "LEVEL_1_MARIO_START_Y": jnp.float32(106.0),
        "LEVEL_1_MARIO_START_STAGE": 6,
        "LEVEL_1_MARIO_START_DIRECTION": 3,  # MOVING_LEFT
    }


