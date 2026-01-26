import jax
import jax.numpy as jnp
import chex
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_seaquest import SeaquestState, SpawnState


class DisableEnemiesMod(JaxAtariPostStepModPlugin):
    """Disable enemies in the environment."""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Zero out all enemy positions
        return new_state.replace(
            shark_positions=jnp.zeros_like(new_state.shark_positions),
            sub_positions=jnp.zeros_like(new_state.sub_positions),
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions),
            surface_sub_position=jnp.zeros_like(new_state.surface_sub_position)
        )


class NoDiversMod(JaxAtariInternalModPlugin):
    """
    Internal mod to remove Divers from the game.
    It suppresses the logic that updates/spawns divers and disables their rendering.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step_diver_movement(self,
            diver_positions: chex.Array,
            shark_positions: chex.Array,
            state_player_x: chex.Array,
            state_player_y: chex.Array,
            state_divers_collected: chex.Array,
            spawn_state: SpawnState,
            step_counter: chex.Array,
            rng: chex.PRNGKey
        ):
        """
        Override for _diver_step (or equivalent logic function).
        We return off-screen positions and inactive flags.
        """
        
        # We assume the diver step returns: 
        # (new_positions, new_actives, new_timers, score_addition)
        
        return (
            jnp.full_like(diver_positions, -1), 
            state_divers_collected,  
            spawn_state,
            rng
        )

    @partial(jax.jit, static_argnums=(0,))
    def _draw_divers(self, raster: jnp.ndarray, state: SeaquestState):
        """
        Override for the renderer to skip drawing divers.
        """
        # Simply return the raster without drawing the sprite
        return raster


class EnemyMinesMod(JaxAtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

    asset_overrides = {
        "shark_base": {
            'name': 'shark_base',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        },
        "enemy_sub": {
            'name': 'enemy_sub',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        }
    }

    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([[128, 128, 128]] * 5),
    }


class FireBallsMod(JaxAtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

    asset_overrides = {
        "enemy_torp": {
            'name': 'enemy_torp',
            'type': 'single',
            'file': 'mods/fireball.npy'
        }
    }

