import jax
import jax.numpy as jnp
from functools import partial

from jaxtari.modification import JaxtariPostStepModPlugin, JaxtariInternalModPlugin
from jaxtari.games.jax_journeyescape import JourneyEscapeState


class BackgroundStaticMod(JaxtariPostStepModPlugin):
    """Makes the background static (disables background animation)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: JourneyEscapeState, new_state: JourneyEscapeState) -> JourneyEscapeState:
        return new_state.replace(bg_frames=prev_state.bg_frames)


class SpeedUpPlayerMod(JaxtariInternalModPlugin):
    """Increases the player's movement speed."""

    constants_overrides = {
        "player_speed": 2,
    }


class SpeedUpObstaclesMod(JaxtariInternalModPlugin):
    """Increase the obstacles' movement speed."""

    constants_overrides = {
        "obstacle_speed_px_per_frame": 2,
    }


class ReducePlayerSizeMod(JaxtariInternalModPlugin):
    """Reduced the size of the player sprite."""

    asset_overrides = {
        "player": {
            "name": "player",
            "type": "group",
            "files": [
                "smaller_player_walk_front_1.npy",
                "smaller_player_walk_front_0.npy",
                "smaller_player_run_right_0.npy",
                "smaller_player_run_right_1.npy",
                "smaller_player_run_left_0.npy",
                "smaller_player_run_left_1.npy",
            ],
        }
    }

    constants_overrides = {
        "player_width": 5,
        "player_height": 15,
    }


class RestrictPlayerMovementMod(JaxtariPostStepModPlugin):
    """Restrict movement to four directions (no diagonals)."""

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: JourneyEscapeState, new_state: JourneyEscapeState) -> JourneyEscapeState:
        diagonal_movement = (prev_state.player_y != new_state.player_y) & (
            prev_state.player_x != new_state.player_x
        )
        new_player_y = jnp.where(
            diagonal_movement,
            prev_state.player_y,
            new_state.player_y,
        ).astype(jnp.int32)
        return new_state.replace(player_y=new_player_y)


# --- Level shortcuts ---------------------------------------------------------
# These skip the concurrent env to a given difficulty_level. Always-diagonal
# (prob 1.0) for non-barrier enemies — matching ALE later stages.


class ObstacleDiagonalMovementMod(JaxtariInternalModPlugin):
    """Skip to Level 2: enemies always move diagonally and bounce."""

    constants_overrides = {
        "start_difficulty_level": 1,
        "lock_difficulty_level": True,
        "use_level_profiles": True,
    }


class ObstacleSteepDiagonalMovementMod(JaxtariInternalModPlugin):
    """Skip to Level 3: steep diagonal (1px/frame, every 16th frame vertical-only)."""

    constants_overrides = {
        "start_difficulty_level": 2,
        "lock_difficulty_level": True,
        "use_level_profiles": True,
    }


class ObstacleAcceleratingBounceMod(JaxtariInternalModPlugin):
    """Skip to Level 4: diagonal with speed alternating on wall bounces."""

    constants_overrides = {
        "start_difficulty_level": 3,
        "lock_difficulty_level": True,
        "use_level_profiles": True,
    }


class ObstacleRandomDirectionSwitchMod(JaxtariInternalModPlugin):
    """Skip to Level 5: diagonal with random direction flips."""

    constants_overrides = {
        "start_difficulty_level": 4,
        "lock_difficulty_level": True,
        "use_level_profiles": True,
    }


class ObstacleChaoticMovementMod(JaxtariInternalModPlugin):
    """Chaos Mode: locked at max difficulty + faster spawns + per-obstacle flips."""

    constants_overrides = {
        "start_difficulty_level": 4,
        "lock_difficulty_level": True,
        "use_level_profiles": True,
        "diagonal_switch_per_obstacle": True,
        "diagonal_random_switch_prob": 0.8,  # used when profiles off; profiles still drive base
        "diagonal_random_switch_cooldown": 10,
        "diagonal_random_switch_cooldown_range": 40,
        "row_spawn_period_frames": 19,
        # Override level-4 switch rate upward for chaos
        "level_random_switch_prob": (0.0, 0.0, 0.0, 0.0, 0.8),
        "level_enemy_diagonal_prob": (0.0, 1.0, 1.0, 1.0, 1.0),
    }
