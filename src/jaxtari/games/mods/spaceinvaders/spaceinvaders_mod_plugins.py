import jax
import jax.numpy as jnp
from functools import partial
from jaxtari.games.jax_spaceinvaders import SpaceInvadersState
from jaxtari.modification import JaxtariInternalModPlugin, JaxtariPostStepModPlugin

# --- Shield Modifications ---

class DisableShieldLeftMod(JaxtariPostStepModPlugin):
    """
    Erases the left bunker from the screen by clearing its specific memory data.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            barricade_health=new_state.barricade_health.at[0].set(0)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[0].set(0)
        )
        return self._env._get_observation(state), state

class DisableShieldMiddleMod(JaxtariPostStepModPlugin):
    """
    Erases the middle bunker from the screen by clearing its specific memory data.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            barricade_health=new_state.barricade_health.at[1].set(0)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[1].set(0)
        )
        return self._env._get_observation(state), state

class DisableShieldRightMod(JaxtariPostStepModPlugin):
    """
    Erases the right bunker from the screen by clearing its specific memory data.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            barricade_health=new_state.barricade_health.at[2].set(0)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[2].set(0)
        )
        return self._env._get_observation(state), state

class ShiftShieldsMod(JaxtariInternalModPlugin):
    """
    Teleports all bunkers to new horizontal positions.
    """
    constants_overrides = {
        # Shifting all bunkers 10 pixels to the right from [41, 73, 105] to [51, 83, 115]
        "BARRICADE_POS": (jnp.array([51, 83, 115], dtype=jnp.int32), 157) # 210 - 53 = 157
    }

# --- Weapon & Gameplay Modifications ---

class ControllableMissileMod(JaxtariPostStepModPlugin):
    """
    Forces the fired missile's horizontal position to match the player's tank,
    allowing you to "steer" shots after firing.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        return new_state.replace(
            bullet_x=jnp.where(
                new_state.bullet_active,
                new_state.player_x - (self._env.consts.PLAYER_SIZE[0] // 2),
                new_state.bullet_x
            )
        )

class NoDangerMod(JaxtariPostStepModPlugin):
    """
    Removes all player shields and neutralizes incoming enemy projectiles.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Remove all shields
        new_barricade_health = new_state.barricade_health.at[:].set(0)
        # Neutralize enemy projectiles (deactivate them)
        new_enemy_bullets_active = jnp.zeros_like(new_state.enemy_bullets_active, dtype=jnp.bool_)
        
        return new_state.replace(
            barricade_health=new_barricade_health,
            enemy_bullets_active=new_enemy_bullets_active
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        state = state.replace(
            barricade_health=state.barricade_health.at[:].set(0),
            enemy_bullets_active=jnp.zeros_like(state.enemy_bullets_active, dtype=jnp.bool_)
        )
        return self._env._get_observation(state), state
