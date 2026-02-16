import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_doubledunk import DunkGameState

class TimerMod(JaxAtariInternalModPlugin):
    """
    Ends the game after 1 minute (3600 frames at 60fps)
    instead of the default score limit (24).
    """
    def _get_done(self, state: DunkGameState) -> bool:
        # 1 minute = 60 seconds * 60 frames = 3600 frames
        return state.step_counter >= 3600

class SuperDunkMod(JaxAtariPostStepModPlugin):
    """
    Makes dunks (close range shots) worth 5 points.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # Detect score change
        p_score_diff = new_state.scores.player - prev_state.scores.player
        e_score_diff = new_state.scores.enemy - prev_state.scores.enemy
        
        # Check distance
        # We need to access basket pos from consts
        basket_x, basket_y = self._env.consts.BASKET_POSITION
        
        sx = prev_state.ball.shooter_pos_x
        sy = prev_state.ball.shooter_pos_y
        
        dist = jnp.sqrt((sx - basket_x)**2 + (sy - basket_y)**2)
        
        # Check if it was a close shot (Dunk range)
        is_close = dist < self._env.consts.DUNK_RADIUS
        
        # Player
        new_p_score = jax.lax.select(
            jnp.logical_and(p_score_diff > 0, is_close),
            prev_state.scores.player + 5,
            new_state.scores.player
        )
        
        # Enemy
        new_e_score = jax.lax.select(
            jnp.logical_and(e_score_diff > 0, is_close),
            prev_state.scores.enemy + 5,
            new_state.scores.enemy
        )
        
        new_scores = new_state.scores.replace(player=new_p_score, enemy=new_e_score)
        return new_state.replace(scores=new_scores)