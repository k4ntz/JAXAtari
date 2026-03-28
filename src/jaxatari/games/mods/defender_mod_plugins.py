import jax
import jax.numpy as jnp
from functools import partial
import chex
from jaxatari.games.jax_defender import DefenderState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.environment import JAXAtariAction as Action
from flax import struct

# --- Simple Mods ---


class SmartBombsUnlimitedMod(JaxAtariPostStepModPlugin):
    """
    Prevents Smart Bomb count from decreasing, effectively giving infinite smart bombs.
    """

    constants_overrides = {
        "SPACE_SHIP_INIT_BOMBS": 99,  # Set initial bombs to
    }


class BulletTimewarpMod(JaxAtariInternalModPlugin):
    """
    Reduces the speed of enemy bullets.
    """

    constants_overrides = {
        "BULLET_SPEED": 1.5,  # Original speed is 3.0, so this is half speed
    }


class ParachutesEquippedMod(JaxAtariInternalModPlugin):
    """
    Humans can survive falling even from big heights by parachuting down instead of dying immediately.
    """

    constants_overrides = {
        "HUMAN_DEADLY_FALL_HEIGHT": 300.0,  # Original is 80.0, so this allows safe falling from any height
        "HUMAN_FALLING_SPEED": 0.25,  # Original is 5.0, so this makes falling much slower and safer
    }


class EnemyEmpMod(JaxAtariInternalModPlugin):
    """
    Enemies do not move.
    """

    constants_overrides = {"SHIP_SPEED_INFLUENCE_ON_SPEED": 0, "ENEMY_SPEED": 0}


class NoBackupMod(JaxAtariInternalModPlugin):
    """
    Reduces the number of enemies required to clear a level.
    """

    constants_overrides = {
        "LANDER_LEVEL_AMOUNT": struct.field(
            pytree_node=True, default_factory=lambda: jnp.array([8, 9, 10, 10, 10])
        )
    }


# --- Difficult Mods ---


class MissingFundingMod(JaxAtariInternalModPlugin):
    """
    Start with only 1 life and 1 bomb.
    """

    constants_overrides = {
        "SPACE_SHIP_INIT_LIVES": 1,
        "SPACE_SHIP_INIT_BOMBS": 1,
    }


class EnemiesOnSpeedMod(JaxAtariInternalModPlugin):
    """
    Increases enemy and bullet speeds.
    """

    constants_overrides = {"ENEMY_SPEED": 0.24, "BULLET_SPEED": 5.0}


class NoBreaksInSpaceMod(JaxAtariInternalModPlugin):
    """
    Ship cannot brake / stop.
    """

    constants_overrides = {"SPACE_SHIP_BREAK": 0.0}

