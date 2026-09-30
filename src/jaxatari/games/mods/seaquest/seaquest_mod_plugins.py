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

class UnlimitedOxygenMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        return new_state.replace(oxygen=jnp.array(64, dtype=jnp.int32))

class GravityMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        new_player_y = jnp.where(
            new_state.step_counter % 4 == 0,
            jnp.minimum(new_state.player_y + 1, self._env.consts.PLAYER_BOUNDS[1, 1]),
            new_state.player_y
        )
        return new_state.replace(player_y=new_player_y)

class RandomColorEnemiesMod(JaxAtariInternalModPlugin):
    pass


class PenalizeDiverShootingMod(JaxAtariPostStepModPlugin):
    """
    Penalizes the player for shooting divers with the torpedo.

    When the player's missile hits a diver:
    - The diver is killed (removed from play)
    - The missile is consumed (like hitting any other entity)
    - Score is penalized proportionally to difficulty:
        penalty = min(PENALTY_BASE + PENALTY_STEP * successful_rescues, PENALTY_MAX)
    """

    PENALTY_BASE = 50
    PENALTY_STEP = 25  # per successful rescue
    PENALTY_MAX = 500

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        missile_pos = new_state.player_missile_position
        missile_active = missile_pos[2] != 0

        # Calculate penalty based on successful rescues (difficulty proxy)
        penalty = jnp.minimum(
            self.PENALTY_BASE + self.PENALTY_STEP * new_state.successful_rescues,
            self.PENALTY_MAX
        )

        missile_xy = missile_pos[:2]
        missile_size = self._env.consts.MISSILE_SIZE
        diver_size = self._env.consts.DIVER_SIZE

        def check_diver(i, carry):
            state, missile_gone = carry
            diver_pos = state.diver_positions[i]

            should_check = jnp.logical_and(
                diver_pos[2] != 0,
                jnp.logical_not(missile_gone)
            )

            collision = self._env.check_collision_single(
                missile_xy, missile_size,
                jnp.array([diver_pos[0], diver_pos[1]]), diver_size
            )
            hit = jnp.logical_and(should_check, collision)

            state = state.replace(
                diver_positions=state.diver_positions.at[i].set(
                    jnp.where(hit, jnp.zeros(3, dtype=jnp.int32), diver_pos)
                ),
                score=jnp.where(hit, state.score - penalty, state.score),
                player_missile_position=jnp.where(
                    hit, jnp.zeros(3, dtype=jnp.int32), state.player_missile_position
                ),
            )

            return state, jnp.logical_or(missile_gone, hit)

        final_state = jax.lax.cond(
            missile_active,
            lambda s: jax.lax.fori_loop(0, 4, check_diver, (s, jnp.array(False)))[0],
            lambda s: s,
            new_state,
        )

        return final_state
