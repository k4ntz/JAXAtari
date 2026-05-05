import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_gopher import GopherState
from jaxatari.games.jax_gopher import GopherAction
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_gopher import JaxGopher
import chex
from jaxatari.environment import JAXAtariAction as Action

# ----- Simple Mods -----
class GreedyGopherMod(JaxAtariInternalModPlugin):
    """
    Mod that makes the gopher almost always try to steal 
    once it reaches the surface.
    Tests extreme enemy aggression.
    """
    constants_overrides = {
        "PROB_STEAL_NORMAL": 0.99,  # Original: 0.7
        "PROB_STEAL_SMART": 1.0,    # Original: 0.9
        "TIME_TO_PEEK": 5,          # Original: 24
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


class EnergyDrainMod(JaxAtariInternalModPlugin):
    """
    Subtracts a tiny amount of reward every frame.
    Tests if the agent can optimize for speed, rather than just camping
    """
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, next_state, reward, done, info = JaxGopher.step(self._env, state, action)
        
        tired_reward = reward - 0.05
        
        return obs, next_state, tired_reward, done, info


class HeavyShovelMod(JaxAtariInternalModPlugin):
    """
    Increase the timer for bonk, player bonk slower, and get locked if he holds bonk button all the time
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
    

# ----- Hard Mods -----
class InvisibleGopherMod(JaxAtariInternalModPlugin):
    """
    Dynamically hides the Gopher when it is underground.
    When the Gopher reaches the surface, it becomes visible again.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state):
        # Check if gopher is underground
        is_underground = state.gopher_position[1] > 150.0

        # Make gopher out of screen(invisible)
        hidden_position = jnp.array([-100.0, -100.0], dtype = jnp.float32)

        # Choose invisible gopher when it's underground
        fake_position = jax.lax.select(
            is_underground,
            hidden_position,
            state.gopher_position
        )

        fake_state = state.replace(gopher_position=fake_position)
        return JaxGopher._get_observation(self._env, fake_state)

    
class WindGopherMod(JaxAtariInternalModPlugin):
    """
    Applies a constant wind force pushing the farmer to the left.
    The agent must constantly pressing right button to stay centered
    """
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # Run the normal game
        obs, next_state, reward, done, info = JaxGopher.step(self._env, state, action)

        # Apply the wind force on the result
        wind_force = 0.8
        after_pushed_x = next_state.player_x - wind_force

        # Wall collision check
        clamped_x = jnp.clip(after_pushed_x, 0.0, 147.0)

        # Update the state
        windy_state = next_state.replace(player_x = clamped_x)
        windy_obs = self._env._get_observation(windy_state)

        return windy_obs, windy_state, reward, done, info
    


class DizzyFarmerMod(JaxAtariInternalModPlugin):
    """
    The Gopher's perceived X coordinate wobbles back and forth
    Tests the agent's robustness to noisy observations and adversarial perturbations
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state):

        wobble = jnp.sin(state.player_x * 0.2) * 12.0
        
        # Add this hallucination to the Gopher's true X coordinate
        fake_gopher_x = state.gopher_position[0] + wobble
        
        # Create the fake coordinate array 
        fake_position = jnp.array([fake_gopher_x, state.gopher_position[1]], dtype=jnp.float32)
        
        # Update fake position
        fake_state = state.replace(gopher_position=fake_position)
        return JaxGopher._get_observation(self._env, fake_state)