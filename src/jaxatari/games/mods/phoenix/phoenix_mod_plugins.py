from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import jax.numpy as jnp
from jaxatari.games.jax_phoenix import PhoenixState


class BossLateMissilesMod(JaxAtariInternalModPlugin):
    """
    Make boss missiles appear a few pixels after spawn so they are visible
    later (e.g., after leaving dense boss-block area).
    """

    constants_overrides = {
        "BOSS_PROJECTILE_RENDER_DELAY_PX": 8,
    }


class InfiniteLivesMod(JaxAtariInternalModPlugin):
    """
    Set player lives to 99.
    """
    constants_overrides = {
        "PLAYER_LIVES": 99,
    }


class FastPlayerMod(JaxAtariInternalModPlugin):
    """
    Increases player movement speed.
    """
    constants_overrides = {
        "PLAYER_STEP_SIZE": 2,
    }


class InvinciblePlayerMod(JaxAtariPostStepModPlugin):
    """
    Player is always invincible.
    """
    def run(self, prev_state: PhoenixState, new_state: PhoenixState) -> PhoenixState:
        return new_state.replace(invincibility=jnp.array(True))


class FastEnemyBulletsMod(JaxAtariInternalModPlugin):
    """
    Increases speed of enemy projectiles.
    """
    constants_overrides = {
        "ENEMY_PROJECTILE_SPEED": 4,
    }


class NoAbilityCooldownMod(JaxAtariInternalModPlugin):
    """
    Removes cooldown for the special ability (shield).
    """
    constants_overrides = {
        "ABILITY_COOLDOWN": 0,
    }

