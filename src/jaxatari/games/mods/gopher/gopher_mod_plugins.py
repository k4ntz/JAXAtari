import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_gopher import GopherState
from jaxatari.games.jax_gopher import GopherAction
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
from jaxatari.environment import JAXAtariAction as Action


class GreedyGopherMod(JaxAtariInternalModPlugin):
    """
    Mod that makes the gopher almost always try to steal 
    once it reaches the surface.
    Tests extreme enemy aggression.
    """
    constants_overrides = {
        "PROB_STEAL_NORMAL": 0.99,  # Original: 0.7
        "PROB_STEAL_SMART": 1.0,    # Original: 0.9
        "TIME_TO_PEEK": 5,          
    }
    
class MirrorButtonMod(JaxAtariInternalModPlugin):
    """
    Mod switch the move left and right button
    Tests spatial understanding or action memorization
    """
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state, action):
        """
        Handles player horizontal movement and bonking input.
        """

        real_action = jnp.array(self._env.action_set)[action]

        # --- Player walking ---
        left = jnp.logical_or(real_action == Action.RIGHT, real_action == Action.RIGHTFIRE) # switched left to right
        right = jnp.logical_or(real_action == Action.LEFT, real_action == Action.LEFTFIRE) # switched right to left
        new_speed = jax.lax.select(left, -self._env.consts.PLAYER_SPEED, jax.lax.select(right, self._env.consts.PLAYER_SPEED, 0.0))

        # Wall Collision
        touch_left_wall = state.player_x <= self._env.consts.LEFT_WALL
        touch_right_wall = state.player_x + self._env.consts.PLAYER_SIZE[0] >= self._env.consts.RIGHT_WALL
        final_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(left, touch_left_wall), jnp.logical_and(right, touch_right_wall)),
            lambda _: 0.0, lambda _: new_speed, operand = None
        )

        # Freeze player if Game is in Start Delay
        is_frozen = state.gopher_action == GopherAction.START_DELAY
        final_speed = jax.lax.select(is_frozen, 0.0, final_speed)
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            self._env.consts.LEFT_WALL,
            self._env.consts.RIGHT_WALL - self._env.consts.PLAYER_SIZE[0],
        )

        # --- Fire bonk logic ---
        fire_pressed = (real_action == Action.FIRE) | (real_action == Action.DOWNFIRE) | \
                      (real_action == Action.LEFTFIRE) | (real_action == Action.RIGHTFIRE)
        valid_fire = fire_pressed & jnp.logical_not(is_frozen)
        is_bonking = state.bonk_timer > 0
        
        
        current_fire_down = fire_pressed
       
        is_fresh_press = current_fire_down & jnp.logical_not(state.prev_fire_pressed)
        
        
        valid_start = is_fresh_press & jnp.logical_not(is_frozen) & (state.bonk_timer == 0)

        
        def handle_timer(t):
            return jax.lax.cond(
                t == 0,
                lambda _: jax.lax.select(valid_start, 1, 0), 
                lambda _: jax.lax.select(t >= 5, 0, t + 1),  
                operand=None
            )

        new_bonk_timer = handle_timer(state.bonk_timer)
        new_state = state.replace(
            player_x = proposed_player_x, 
            player_speed = final_speed,
            bonk_timer = new_bonk_timer,
            prev_fire_pressed = current_fire_down
        )
        return new_state, fire_pressed


class PinkGopherMod(JaxAtariInternalModPlugin):
    """
    Mod change gopher color to pink, check if agent heavily relies on color features than movement
    Tests visual/feature reliance
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        pink_color = jnp.array([255, 105, 180], dtype=jnp.uint8)
        new_palette = env.renderer.PALETTE.at[11].set(pink_color) # Gopher color ID: 11
        env.renderer.PALETTE = new_palette
        if hasattr(self._env, "_patched_renderer_methods"):
            self._env._patched_renderer_methods.append("PALETTE")

class HeavyShovelMod(JaxAtariInternalModPlugin):
    """
    Increase the timer for bonk, player get locked if he holds bonk button all the time
    """
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state, action):
        """
        Handles player horizontal movement and bonking input.
        """

        real_action = jnp.array(self._env.action_set)[action]

        # --- Player walking ---
        left =  jnp.logical_or(real_action == Action.LEFT, real_action == Action.LEFTFIRE)
        right = jnp.logical_or(real_action == Action.RIGHT, real_action == Action.RIGHTFIRE)
        new_speed = jax.lax.select(left, -self._env.consts.PLAYER_SPEED, jax.lax.select(right, self._env.consts.PLAYER_SPEED, 0.0))

        # Wall Collision
        touch_left_wall = state.player_x <= self._env.consts.LEFT_WALL
        touch_right_wall = state.player_x + self._env.consts.PLAYER_SIZE[0] >= self._env.consts.RIGHT_WALL
        final_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(left, touch_left_wall), jnp.logical_and(right, touch_right_wall)),
            lambda _: 0.0, lambda _: new_speed, operand = None
        )

        # Freeze player if Game is in Start Delay
        is_frozen = state.gopher_action == GopherAction.START_DELAY
        final_speed = jax.lax.select(is_frozen, 0.0, final_speed)
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            self._env.consts.LEFT_WALL,
            self._env.consts.RIGHT_WALL - self._env.consts.PLAYER_SIZE[0],
        )

        # --- Fire bonk logic ---
        fire_pressed = (real_action == Action.FIRE) | (real_action == Action.DOWNFIRE) | \
                      (real_action == Action.LEFTFIRE) | (real_action == Action.RIGHTFIRE)
        valid_fire = fire_pressed & jnp.logical_not(is_frozen)
        is_bonking = state.bonk_timer > 0
        
        
        current_fire_down = fire_pressed
       
        is_fresh_press = current_fire_down & jnp.logical_not(state.prev_fire_pressed)
        
        
        valid_start = is_fresh_press & jnp.logical_not(is_frozen) & (state.bonk_timer == 0)

        
        def handle_timer(t):
            return jax.lax.cond(
                t == 0,
                lambda _: jax.lax.select(valid_start, 1, 0), 
                lambda _: jax.lax.select(t >= 25, 0, t + 1),  # increase the bonk timer
                operand=None
            )

        new_bonk_timer = handle_timer(state.bonk_timer)
        new_state = state.replace(
            player_x = proposed_player_x, 
            player_speed = final_speed,
            bonk_timer = new_bonk_timer,
            prev_fire_pressed = current_fire_down
        )
        return new_state, fire_pressed


class FastSeedMod(JaxAtariInternalModPlugin):
    """
    Mod double the seed falling speed, check if agent remebers the original falling speed or 
    it adapts to the current situation. Tests robustness to altered physics/speed.
    """
    constants_overrides = {     
        "SEED_DROP_SPEED": 2.0           # Original:1.0
    }
    
