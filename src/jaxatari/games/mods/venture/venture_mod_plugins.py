from jaxatari.modification import JaxAtariInternalModPlugin
import jax.numpy as jnp


class FastWinkyMod(JaxAtariInternalModPlugin):
    """Increase Winky's movement speed."""

    constants_overrides = {
        "PLAYER_SPEED": 2.0,
    }


class SlowMonstersMod(JaxAtariInternalModPlugin):
    """Decrease the movement speed of all monsters, including the hallway chaser."""

    constants_overrides = {
        "MONSTER_SPEEDS": jnp.array([0.5, 0.75, 1.0, 1.25], dtype=jnp.float32),
        "CHASER_SPEED": 0.2,
    }


class WealthyVentureMod(JaxAtariInternalModPlugin):
    """Significantly increase the points awarded for collecting treasures."""

    constants_overrides = {
        "CHEST_SCORE": 1000,
    }


class PatientChaserMod(JaxAtariInternalModPlugin):
    """Increase the time before the hallway chaser appears in a room."""

    constants_overrides = {
        "CHASER_SPAWN_FRAMES": 10000,
    }


class FastArrowsMod(JaxAtariInternalModPlugin):
    """Increase the speed of Winky's arrows."""

    constants_overrides = {
        "PROJECTILE_SPEED": 4.0,
    }


class LongRangeArrowsMod(JaxAtariInternalModPlugin):
    """Increase the distance arrows travel before disappearing."""

    constants_overrides = {
        "PROJECTILE_LIFETIME_FRAMES": 60,
    }


class GodModeMod(JaxAtariInternalModPlugin):
    """Winky is immune to collisions with hazards."""

    def _check_player_hazard_collision(self, player_state, monster_state, chaser_state, laser_state, current_level, world_level):
        return jnp.array(False)
