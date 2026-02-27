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
    """
    constants_overrides = {
        "PROB_STEAL_NORMAL": 0.99,  # Original: 0.7
        "PROB_STEAL_SMART": 1.0,    # Original: 0.9
        "TIME_TO_PEEK": 5,          # Make it peek faster
    }
    
class MirrorButtonMod(JaxAtariInternalModPlugin):
    """
    Switch the move left and right button
    """
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state, action):
        """
        Handles player horizontal movement and bonking input.
        """

        real_action = jnp.array(self._env.action_set)[action]

        # --- Player walking ---
        left = jnp.logical_or(real_action == Action.RIGHT, real_action == Action.RIGHTFIRE)
        right = jnp.logical_or(real_action == Action.LEFT, real_action == Action.LEFTFIRE)
        new_speed = jax.lax.select(left, -self._env.consts.PLAYER_SPEED, jax.lax.select(right, self._env.consts.PLAYER_SPEED, 0.0))

        # Wall Collision
        touch_left_wall = state.player_x <= self.consts.LEFT_WALL
        touch_right_wall = state.player_x + self.consts.PLAYER_SIZE[0] >= self.consts.RIGHT_WALL
        final_speed = jax.lax.cond(
            jnp.logical_or(jnp.logical_and(left, touch_left_wall), jnp.logical_and(right, touch_right_wall)),
            lambda _: 0.0, lambda _: new_speed, operand = None
        )

        # Freeze player if Game is in Start Delay
        is_frozen = state.gopher_action == GopherAction.START_DELAY
        final_speed = jax.lax.select(is_frozen, 0.0, final_speed)
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            self.consts.LEFT_WALL,
            self.consts.RIGHT_WALL - self.consts.PLAYER_SIZE[0],
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


class RewardInversionMod(JaxAtariPostStepModPlugin):
    """
    Rewards the agent for every frame it remains still AND does not swing the shovel.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: GopherState, new_state: GopherState):
        is_still = new_state.player_speed == 0.0
            
        # 2. Is the player NOT swinging the shovel?
        # (bonk_timer is 0 when the shovel is not in motion)
        is_not_firing = new_state.bonk_timer == 0
            
        # 3. Calculate reward
        # We give a small 'patience' reward for each condition met
        patience_reward = jnp.where(is_still, 0.1, 0.0)
        patience_reward += jnp.where(is_not_firing, 0.1, 0.0)
            
        return new_state, patience_reward


class RewardShapingMod(JaxAtariPostStepModPlugin):
    """
    Adds custom rewards for catching seeds and planting carrots
    without changing the visual score.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        # 1. Detect if a seed was caught (went from 0 to 1 in inventory)
        caught_seed = (new_state.player_has_seed == 1) & (prev_state.player_has_seed == 0)

        # 2. Detect if a carrot was planted (carrot count increased)
        planted_carrot = jnp.sum(new_state.carrots_present) > jnp.sum(prev_state.carrots_present)

        # 3. Calculate extra reward
        extra = jnp.where(caught_seed, 10.0, 0.0)
        extra += jnp.where(planted_carrot, 20.0, 0.0)

        # In this framework, the 'run' method in a PostStep plugin
        # returns the modified state. Since we want to affect the 'reward'
        # specifically, we ensure the agent 'sees' this in the training loop.
        return new_state, extra
