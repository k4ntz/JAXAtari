import jax
import jax.numpy as jnp
from functools import partial
import chex
from jaxatari.games.jax_defender import DefenderState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.environment import JAXAtariAction as Action

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

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: DefenderState) -> DefenderState:
        def _enemy_move_switch(
            enemy: chex.Array, state: DefenderState
        ) -> DefenderState:
            enemy_type = enemy[2]

            # Use original delete logic
            delete_state, score = self._env._delete_enemy(enemy)

            return jax.lax.switch(
                jnp.array(enemy_type, int),
                [
                    lambda: enemy,
                    lambda: enemy, 
                    lambda: enemy,
                    lambda: enemy,
                    lambda: enemy,
                    lambda: enemy,
                    lambda: enemy, 
                    lambda: delete_state, 
                ],
            )

        def _enemy_move_switch_wrapped(enemy: chex.Array) -> chex.Array:
            return _enemy_move_switch(enemy, state)

        enemy_states_updated = jax.vmap(_enemy_move_switch_wrapped, in_axes=(0,))(
            state.enemy_states
        )
        state = state._replace(enemy_states=enemy_states_updated)

        return state


class NoBackupMod(JaxAtariInternalModPlugin):
    """
    Reduces the number of enemies required to clear a level.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _check_level_done(self, state: DefenderState) -> DefenderState:
        enemy_killed = state.enemy_killed
        needed_kills = jnp.asarray(
            [
                self._env.consts.LANDER_LEVEL_AMOUNT[state.level],
                self._env.consts.POD_LEVEL_AMOUNT[state.level],
                self._env.consts.BOMBER_LEVEL_AMOUNT[state.level],
            ]
        )

        # Reduced requirement: Max 1 kill needed per type (if amount > 0)
        reduced_goals = jnp.minimum(needed_kills, 5)

        is_done = jnp.all(enemy_killed >= reduced_goals)

        state = jax.lax.cond(
            is_done, lambda: self._env._end_level(state), lambda: state
        )
        return state


# --- Difficult Mods ---

class MissingFundingMod(JaxAtariInternalModPlugin):
    """
    Start with only 1 life and 1 bomb.
    """

    constants_overrides = {
        "SPACE_SHIP_INIT_LIVES": 1,
        "SPACE_SHIP_INIT_BOMBS": 0,
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