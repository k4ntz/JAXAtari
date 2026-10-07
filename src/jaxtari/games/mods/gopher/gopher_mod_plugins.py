import jax
import jax.numpy as jnp
from functools import partial
from jaxtari.games.jax_gopher import GopherAction
from jaxtari.modification import JaxtariInternalModPlugin
from jaxtari.games.jax_gopher import JaxGopher
from jaxtari.environment import JaxtariAction as Action


# ----- Simple Mods -----
class GreedyGopherMod(JaxtariInternalModPlugin):
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


class MirrorButtonMod(JaxtariInternalModPlugin):
    """
    Swap left/right. Tests spatial understanding / action memorization.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state, action):
        real_action = jnp.array(self._env.action_set)[action]

        left = jnp.logical_or(real_action == Action.RIGHT, real_action == Action.RIGHTFIRE)
        right = jnp.logical_or(real_action == Action.LEFT, real_action == Action.LEFTFIRE)
        new_speed = jax.lax.select(
            left, -self._env.consts.PLAYER_SPEED,
            jax.lax.select(right, self._env.consts.PLAYER_SPEED, 0.0),
        )

        touch_left_wall = state.player_x <= self._env.consts.LEFT_WALL
        touch_right_wall = state.player_x >= self._env.consts.PLAYER_MAX_X
        final_speed = jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(left, touch_left_wall),
                jnp.logical_and(right, touch_right_wall),
            ),
            lambda _: 0.0, lambda _: new_speed, operand=None,
        )

        is_frozen = state.gopher_action == GopherAction.START_DELAY
        final_speed = jax.lax.select(is_frozen, 0.0, final_speed)
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            float(self._env.consts.LEFT_WALL),
            float(self._env.consts.PLAYER_MAX_X),
        )

        fire_pressed = (
            (real_action == Action.FIRE) | (real_action == Action.DOWNFIRE)
            | (real_action == Action.LEFTFIRE) | (real_action == Action.RIGHTFIRE)
        )
        current_fire_down = fire_pressed
        is_fresh_press = current_fire_down & jnp.logical_not(state.prev_fire_pressed)
        valid_start = is_fresh_press & jnp.logical_not(is_frozen) & (state.bonk_timer == 0)

        def handle_timer(t):
            return jax.lax.cond(
                t == 0,
                lambda _: jax.lax.select(valid_start, 1, 0),
                lambda _: jax.lax.select(t >= 5, 0, t + 1),
                operand=None,
            )

        new_bonk_timer = handle_timer(state.bonk_timer)
        return state.replace(
            player_x=proposed_player_x,
            player_speed=final_speed,
            bonk_timer=new_bonk_timer,
            prev_fire_pressed=current_fire_down,
        ), fire_pressed


class EnergyDrainMod(JaxtariInternalModPlugin):
    """
    Continuous energy tax: -0.05 reward every frame, with the on-screen score
    falling by 1 every 20 frames (same average rate).

    Note: JaxtariModWrapper recomputes reward via ``_get_reward(prev, next)``,
    so the dense -0.05 signal must live there — not only in a patched ``step``.
    """
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, next_state, reward, done, info = JaxGopher.step(self._env, state, action)
        drain_points = jnp.where(
            (next_state.frame_count % 20) == 0, 1, 0
        ).astype(next_state.score.dtype)
        next_state = next_state.replace(
            score=jnp.maximum(0, next_state.score - drain_points)
        )
        obs = JaxGopher._get_observation(self._env, next_state)
        return obs, next_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state, state):
        base = jnp.float32(state.score - previous_state.score)
        # Undo the discrete score tick so reward stays a smooth -0.05 / frame.
        drained = jnp.where((state.frame_count % 20) == 0, 1.0, 0.0)
        return base + drained - jnp.float32(0.05)


class HeavyShovelMod(JaxtariInternalModPlugin):
    """Slower bonk + lock if fire is held."""
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state, action):
        real_action = jnp.array(self._env.action_set)[action]

        left = jnp.logical_or(real_action == Action.LEFT, real_action == Action.LEFTFIRE)
        right = jnp.logical_or(real_action == Action.RIGHT, real_action == Action.RIGHTFIRE)
        new_speed = jax.lax.select(
            left, -self._env.consts.PLAYER_SPEED,
            jax.lax.select(right, self._env.consts.PLAYER_SPEED, 0.0),
        )

        touch_left_wall = state.player_x <= self._env.consts.LEFT_WALL
        touch_right_wall = state.player_x >= self._env.consts.PLAYER_MAX_X
        final_speed = jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(left, touch_left_wall),
                jnp.logical_and(right, touch_right_wall),
            ),
            lambda _: 0.0, lambda _: new_speed, operand=None,
        )

        is_frozen = state.gopher_action == GopherAction.START_DELAY
        final_speed = jax.lax.select(is_frozen, 0.0, final_speed)
        proposed_player_x = jnp.clip(
            state.player_x + final_speed,
            float(self._env.consts.LEFT_WALL),
            float(self._env.consts.PLAYER_MAX_X),
        )

        fire_pressed = (
            (real_action == Action.FIRE) | (real_action == Action.DOWNFIRE)
            | (real_action == Action.LEFTFIRE) | (real_action == Action.RIGHTFIRE)
        )
        current_fire_down = fire_pressed
        is_fresh_press = current_fire_down & jnp.logical_not(state.prev_fire_pressed)
        valid_start = is_fresh_press & jnp.logical_not(is_frozen) & (state.bonk_timer == 0)

        def handle_timer(t):
            return jax.lax.cond(
                t == 0,
                lambda _: jax.lax.select(valid_start, 1, 0),
                lambda _: jax.lax.select(t >= 25, 0, t + 1),
                operand=None,
            )

        new_bonk_timer = handle_timer(state.bonk_timer)
        return state.replace(
            player_x=proposed_player_x,
            player_speed=final_speed,
            bonk_timer=new_bonk_timer,
            prev_fire_pressed=current_fire_down,
        ), fire_pressed


class FastSeedMod(JaxtariInternalModPlugin):
    """Double seed falling speed."""
    constants_overrides = {
        "SEED_DROP_SPEED": 2.0,  # Original: 1.0
    }


# ----- Hard Mods -----
class InvisibleGopherMod(JaxtariInternalModPlugin):
    """
    Hide the gopher while underground (pixel render + object obs).
    Visible again once it reaches the surface.
    """
    constants_overrides = {
        "SHOW_GOPHER_UNDERGROUND": False,
    }

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state):
        is_underground = state.gopher_position[1] > 150.0
        hidden_position = jnp.array([-100.0, -100.0], dtype=jnp.float32)
        fake_position = jax.lax.select(
            is_underground,
            hidden_position,
            state.gopher_position,
        )
        fake_state = state.replace(gopher_position=fake_position)
        return JaxGopher._get_observation(self._env, fake_state)


class WindGopherMod(JaxtariInternalModPlugin):
    """Constant leftward wind on the farmer."""
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        obs, next_state, reward, done, info = JaxGopher.step(self._env, state, action)

        wind_force = 0.8
        after_pushed_x = next_state.player_x - wind_force
        clamped_x = jnp.clip(
            after_pushed_x,
            float(self._env.consts.LEFT_WALL),
            float(self._env.consts.PLAYER_MAX_X),
        )
        windy_state = next_state.replace(player_x=clamped_x)
        windy_obs = self._env._get_observation(windy_state)
        return windy_obs, windy_state, reward, done, info


class DizzyFarmerMod(JaxtariInternalModPlugin):
    """Wobble the gopher's perceived X in object observations."""
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state):
        wobble = jnp.sin(state.player_x * 0.2) * 12.0
        fake_gopher_x = state.gopher_position[0] + wobble
        fake_position = jnp.array([fake_gopher_x, state.gopher_position[1]], dtype=jnp.float32)
        fake_state = state.replace(gopher_position=fake_position)
        return JaxGopher._get_observation(self._env, fake_state)
