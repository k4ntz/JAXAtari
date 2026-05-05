import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
from jaxatari.environment import JAXAtariAction as Action
from flax import struct

# --- 1. Individual Mod Plugins ---
class LastEggMod(JaxAtariInternalModPlugin):
    """Only one egg is remaining."""
    key = jax.random.key(-1)
    egg_selected = jax.random.choice(key, 106)
    egg_positions = [
        [
            [18,14,0,4],   [26,16,0,4],   [34,18,0,4],   [48,14,0,1],   [56,16,0,1],   [64,18,0,5],
            [18,26,0,4],                                 [48,26,0,2],
            [18,38,0,4],   [26,40,0,4],   [34,42,0,4],   [48,38,0,2],   [56,40,0,2],   [64,42,0,2],
            [18,50,0,4],   [26,52,0,4],                                                [64,54,0,2],
            [18,62,0,4],   [26,64,0,4],                  [48,62,0,2],   [56,64,0,2],   [64,66,0,2],
                           [26,76,0,4],                  [48,74,0,2],
            [18,86,0,4],   [26,88,0,4],   [34,90,0,4],   [48,86,0,0],   [56,88,0,0],   [64,90,0,0],
                           [26,100,0,4],                                               [64,102,0,0],
            [18,110,0,4],  [26,112,0,4],  [34,114,0,4],  [48,110,0,0],  [56,112,0,0],  [64,114,0,0],
            [18,122,0,4],                                [48,122,0,0],
            [18,134,0,4],  [26,136,0,4],  [34,138,0,4],  [48,134,0,0],  [56,136,0,0],  [64,138,0,0],
            [18,146,0,4],                 [34,150,0,4],
            [18,158,0,4],  [26,160,0,4],  [34,162,0,4],  [48,158,0,0],  [56,160,0,0]
        ],
        [
            [81,14,0,4],   [89,16,0,4],   [97,18,0,4],   [110,14,0,1],  [118,16,0,1],  [126,18,0,5],
                                          [97,30,0,4],                                 [126,30,0,2],
            [81,38,0,4],   [89,40,0,4],   [97,42,0,4],   [110,38,0,2],  [118,40,0,2],  [126,42,0,2],
            [81,50,0,4],                                                [118,52,0,2],  [126,54,0,2],
            [81,62,0,4],   [89,64,0,4],   [97,66,0,4],                  [118,64,0,2],  [126,66,0,2],
                                          [97,78,0,4],                  [118,76,0,2],
            [81,86,0,4],   [89,88,0,4],   [97,90,0,4],   [110,86,0,0],  [118,88,0,0],  [126,90,0,0],
            [81,98,0,4],                                                [118,100,0,0],
            [81,110,0,4],  [89,112,0,4],  [97,114,0,4],  [110,110,0,0], [118,112,0,0], [126,114,0,0],
                                          [97,126,0,4],                                [126,126,0,0],
            [81,134,0,4],  [89,136,0,4],  [97,138,0,4],  [110,134,0,0], [118,136,0,0], [126,138,0,0],
                                                         [110,146,0,0],                [126,150,0,0],
                           [89,160,0,4],  [97,162,0,4],  [110,158,0,0], [118,160,0,0], [126,162,0,0]
        ]]
    egg_positions[egg_selected//53][egg_selected%53][2] = 1
    constants_overrides = {
        "EGG_ARRAY": jnp.array(egg_positions, dtype=jnp.int32)
    }

class EndGameMod(JaxAtariInternalModPlugin):
    """Only one area has remaining eggs."""
    key = jax.random.key(-1)
    cluster_selected = jax.random.choice(key, 2, (2, ))
    cluster = list(range(38, 53)) if cluster_selected[1] else list(range(16))
    egg_positions = [
        [
            [18,14,0,4],   [26,16,0,4],   [34,18,0,4],   [48,14,0,1],   [56,16,0,1],   [64,18,0,5],
            [18,26,0,4],                                 [48,26,0,2],
            [18,38,0,4],   [26,40,0,4],   [34,42,0,4],   [48,38,0,2],   [56,40,0,2],   [64,42,0,2],
            [18,50,0,4],   [26,52,0,4],                                                [64,54,0,2],
            [18,62,0,4],   [26,64,0,4],                  [48,62,0,2],   [56,64,0,2],   [64,66,0,2],
                           [26,76,0,4],                  [48,74,0,2],
            [18,86,0,4],   [26,88,0,4],   [34,90,0,4],   [48,86,0,0],   [56,88,0,0],   [64,90,0,0],
                           [26,100,0,4],                                               [64,102,0,0],
            [18,110,0,4],  [26,112,0,4],  [34,114,0,4],  [48,110,0,0],  [56,112,0,0],  [64,114,0,0],
            [18,122,0,4],                                [48,122,0,0],
            [18,134,0,4],  [26,136,0,4],  [34,138,0,4],  [48,134,0,0],  [56,136,0,0],  [64,138,0,0],
            [18,146,0,4],                 [34,150,0,4],
            [18,158,0,4],  [26,160,0,4],  [34,162,0,4],  [48,158,0,0],  [56,160,0,0]
        ],
        [
            [81,14,0,4],   [89,16,0,4],   [97,18,0,4],   [110,14,0,1],  [118,16,0,1],  [126,18,0,5],
                                          [97,30,0,4],                                 [126,30,0,2],
            [81,38,0,4],   [89,40,0,4],   [97,42,0,4],   [110,38,0,2],  [118,40,0,2],  [126,42,0,2],
            [81,50,0,4],                                                [118,52,0,2],  [126,54,0,2],
            [81,62,0,4],   [89,64,0,4],   [97,66,0,4],                  [118,64,0,2],  [126,66,0,2],
                                          [97,78,0,4],                  [118,76,0,2],
            [81,86,0,4],   [89,88,0,4],   [97,90,0,4],   [110,86,0,0],  [118,88,0,0],  [126,90,0,0],
            [81,98,0,4],                                                [118,100,0,0],
            [81,110,0,4],  [89,112,0,4],  [97,114,0,4],  [110,110,0,0], [118,112,0,0], [126,114,0,0],
                                          [97,126,0,4],                                [126,126,0,0],
            [81,134,0,4],  [89,136,0,4],  [97,138,0,4],  [110,134,0,0], [118,136,0,0], [126,138,0,0],
                                                         [110,146,0,0],                [126,150,0,0],
                           [89,160,0,4],  [97,162,0,4],  [110,158,0,0], [118,160,0,0], [126,162,0,0]
        ]]
    for egg in cluster:
        egg_positions[cluster_selected[0]][egg][2] = 1
    constants_overrides = {
        "EGG_ARRAY": jnp.array(egg_positions, dtype=jnp.int32)
    }

class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod: black background, green walls, green enemies, green eggs."""
    name = "matrix_theme"
    
    constants_overrides = {
        'RGB_BACKGROUND': (0, 0, 0),
        'RGB_BASIC_BLUE': (0, 200, 0),   # Walls, Player, UI
        'RGB_ORANGE': (0, 255, 0),       # Flame, etc
        'RGB_PINK': (50, 255, 50),
        'RGB_GREEN': (0, 255, 100),
        'RGB_YELLOW': (100, 255, 100),
        'RGB_FRIGHTENED': (0, 100, 0),
    }

class PacifistMod(JaxAtariInternalModPlugin):
    """Aliens are never frightened, forcing a pure evasion playstyle."""
    name = "pacifist_mode"
    constants_overrides = {
        "FRIGHTENED_DURATION": 0,
        "FLAME_FRIGHTENED_DURATION": 0
    }

class AggressiveSwarmMod(JaxAtariInternalModPlugin):
    """Aliens spend almost all their time actively chasing the player."""
    name = "aggressive_swarm"
    constants_overrides = {
        "SCATTER_DURATION_1": 10,
        "SCATTER_DURATION_2": 10,
        "SCATTER_DURATION_3": 10,
        "CHASE_DURATION_1": 1000,
        "CHASE_DURATION_2": 1000,
        "CHASE_DURATION_3": 1000,
    }

class DontKillMod(JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin):
    """Punishes for killing and shooting enemies."""
    name = "dont_kill"
    constants_overrides = {
        "EGG_SCORE_MULTIPLYER": 50,
        "ENEMY_KILL_SCORE": jnp.array([-1000, -2000, -5000], dtype=jnp.int32)
    }

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Punish for shooting (pulsar usage)
        # Every frame the flame is active, we subtract points.
        is_shooting = new_state.player.flame.flame_flag > 0
        shooting_punishment = jnp.where(is_shooting, 2, 0).astype(jnp.uint16)
        
        # Reward 10 points every 100 frames
        survival_reward = jnp.where(
            jnp.logical_and(new_state.step_counter > 0, new_state.step_counter % 100 == 0), 
            10, 
            0
        ).astype(jnp.uint16)
        
        new_score = new_state.level.score - shooting_punishment + survival_reward
        
        return new_state.replace(level=new_state.level.replace(score=new_score))

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
        # Calculate signed difference with wrap-around handling for uint16 score
        curr_score = state.level.score[0].astype(jnp.int32)
        prev_score = previous_state.level.score[0].astype(jnp.int32)
        
        diff = curr_score - prev_score
        # Handle uint16 wrap: if diff is very large positive, it's likely a negative change
        # (e.g., 500 - 1000 = -500 -> 65036. 65036 - 500 = 64536 > 32768)
        diff = jnp.where(diff > 32768, diff - 65536, diff)
        diff = jnp.where(diff < -32768, diff + 65536, diff)
        
        return diff.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state, state):
        return self._get_reward(previous_state, state)

class ShortCircuitMod(JaxAtariInternalModPlugin):
    """Vulnerability window is drastically reduced."""
    name = "short_circuit"
    constants_overrides = {
        "FRIGHTENED_DURATION": 30,
        "FLAME_FRIGHTENED_DURATION": 2
    }

class ExtraLivesMod(JaxAtariPostStepModPlugin):
    """Start with many extra lives."""
    name = "extra_lives"
    
    constants_overrides = {
        "MAX_LIVES_RENDERED": 9
    }
    
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state):
        new_level = state.level.replace(lifes=jnp.array(9, dtype=jnp.int32))
        return obs, state.replace(level=new_level)
