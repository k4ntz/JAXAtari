import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_boxing2 import BoxingState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin

class CenterEnemyMod(JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin):
    """
    Forces the enemy (black boxer) to stay at the center of the ring.
    The enemy cannot move, but can only punch.
    """

    @partial(jax.jit, static_argnums=(0,))
    def _cpu_logic(self, state: BoxingState):
        """
        Replace the CPU logic to ONLY punch, without moving.
        """
        p1_pos = state.pos[0]
        p2_pos = state.pos[1]
        
        # Punch if close
        dist = jnp.linalg.norm(p1_pos - p2_pos)
        should_punch = jnp.logical_and(dist < 30.0, jax.random.uniform(state.key, ()) < 0.1)
        act = jnp.where(should_punch, Action.FIRE, Action.NOOP)
        
        return act

    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: BoxingState):
        """
        Snap the enemy to the center immediately after reset.
        """
        center_x = (self._env.consts.XMIN + self._env.consts.XMAX) / 2.0
        center_y = (self._env.consts.YMIN + self._env.consts.YMAX) / 2.0
        
        new_pos = state.pos.at[1].set(jnp.array([center_x, center_y]))
        state = replace(state, pos=new_pos)
        
        # The wrapper doesn't auto-recompute obs after after_reset, so we must do it
        # Actually JaxAtariModWrapper's after_reset might just return what we return
        # so let's update obs.
        new_obs = self._env._get_observation(state)
        
        return new_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BoxingState, new_state: BoxingState):
        """
        Snap the enemy back to the center after every step, neutralizing knockback.
        """
        center_x = (self._env.consts.XMIN + self._env.consts.XMAX) / 2.0
        center_y = (self._env.consts.YMIN + self._env.consts.YMAX) / 2.0
        
        new_pos = new_state.pos.at[1].set(jnp.array([center_x, center_y]))
        modified_state = replace(new_state, pos=new_pos)
        
        return modified_state


class AlwaysPunchEnemyMod(JaxAtariInternalModPlugin):
    """
    Forces the enemy (black boxer) to constantly punch while maintaining its regular movement logic.
    """
    
    @partial(jax.jit, static_argnums=(0,))
    def _cpu_logic(self, state: BoxingState):
        """
        Replace the CPU logic to perform regular movement but ALWAYS include FIRE.
        """
        p1_pos = state.pos[0]
        p2_pos = state.pos[1]
        
        # Simple AI: Track P1's Y, stay at distance on X
        target_x = p1_pos[0] + jnp.where(p2_pos[0] > p1_pos[0], 20.0, -20.0)
        target_y = p1_pos[1]
        
        dx = jnp.where(p2_pos[0] < target_x - 2, Action.RIGHT, jnp.where(p2_pos[0] > target_x + 2, Action.LEFT, Action.NOOP))
        dy = jnp.where(p2_pos[1] < target_y - 2, Action.DOWN, jnp.where(p2_pos[1] > target_y + 2, Action.UP, Action.NOOP))
        
        # Combine into action and append FIRE
        act = Action.FIRE
        act = jnp.where(jnp.logical_and(dx == Action.RIGHT, dy == Action.UP), Action.UPRIGHTFIRE, act)
        act = jnp.where(jnp.logical_and(dx == Action.LEFT, dy == Action.UP), Action.UPLEFTFIRE, act)
        act = jnp.where(jnp.logical_and(dx == Action.RIGHT, dy == Action.DOWN), Action.DOWNRIGHTFIRE, act)
        act = jnp.where(jnp.logical_and(dx == Action.LEFT, dy == Action.DOWN), Action.DOWNLEFTFIRE, act)
        act = jnp.where(jnp.logical_and(act == Action.FIRE, dx == Action.RIGHT), Action.RIGHTFIRE, act)
        act = jnp.where(jnp.logical_and(act == Action.FIRE, dx == Action.LEFT), Action.LEFTFIRE, act)
        act = jnp.where(jnp.logical_and(act == Action.FIRE, dy == Action.UP), Action.UPFIRE, act)
        act = jnp.where(jnp.logical_and(act == Action.FIRE, dy == Action.DOWN), Action.DOWNFIRE, act)
        
        return act


class DifficultyEasyMod(JaxAtariInternalModPlugin):
    """
    Easy Difficulty Mod:
    - Target updates slowly (every ~8 frames on average).
    - Low punching probability.
    - Long retreat/dancing phase.
    """
    constants_overrides = {
        "DIFFICULTY_PRESET": "easy",
        "CPU_UPDATE_MASK": 7,
        "CPU_AGGR_WINNING": 20,
        "CPU_AGGR_LOSING": 10,
        "CPU_DANCING_DURATION": 60,
    }


class DifficultyMediumMod(JaxAtariInternalModPlugin):
    """
    Medium / Normal Difficulty Mod:
    - Moderate reaction speed and punching probability.
    - Standard defensive/dancing behavior.
    """
    constants_overrides = {
        "DIFFICULTY_PRESET": "normal",
        "CPU_UPDATE_MASK": 3,
        "CPU_AGGR_WINNING": 55,
        "CPU_AGGR_LOSING": 35,
        "CPU_DANCING_DURATION": 30,
    }


class DifficultyHardMod(JaxAtariInternalModPlugin):
    """
    Hard Difficulty Mod:
    - Fast target updates (~2 frames).
    - High punching probability.
    - Brief retreat/dancing phase.
    """
    constants_overrides = {
        "DIFFICULTY_PRESET": "hard",
        "CPU_UPDATE_MASK": 1,
        "CPU_AGGR_WINNING": 90,
        "CPU_AGGR_LOSING": 70,
        "CPU_DANCING_DURATION": 10,
    }


class DifficultyImpossibleMod(JaxAtariInternalModPlugin):
    """
    Impossible Difficulty Mod:
    - Instant target updates (every frame).
    - Relentless pressure and punching.
    - No retreat/dancing phase at all.
    """
    constants_overrides = {
        "DIFFICULTY_PRESET": "impossible",
        "CPU_UPDATE_MASK": 0,
        "CPU_AGGR_WINNING": 120,
        "CPU_AGGR_LOSING": 100,
        "CPU_DANCING_DURATION": 0,
    }


class PeacefulEnemyMod(JaxAtariInternalModPlugin):
    """
    Peaceful Enemy Mod:
    - The enemy (black boxer) is strictly not allowed to punch.
    - Compatible with all other movement or logic mods.
    """
    constants_overrides = {
        "ENEMY_PEACEFUL": True,
    }
