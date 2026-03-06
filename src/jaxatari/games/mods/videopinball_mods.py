import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jaxatari.games.jax_videopinball import VideoPinballState

from jaxatari.wrappers import JaxatariWrapper


class NeverActivateTiltMode(JaxatariWrapper):
    """Prevent entering tilt mode by resetting the tilt counter."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def prevent_tilt(self, state: VideoPinballState) -> VideoPinballState:
        new_state = state._replace(
            tilt_mode_active=jnp.array(False, dtype=jnp.bool_),
        )
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: Union[int, float]
    ) -> Tuple[chex.Array, VideoPinballState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.prevent_tilt(new_state)

        return new_obs, new_state, reward, done, info


class NoScoringCooldown(JaxatariWrapper):
    """Remove the cooldown period after hitting a target."""

    @functools.partial(jax.jit, static_argnums=(0,))
    def remove_cooldown(self, state: VideoPinballState) -> VideoPinballState:
        new_state = state._replace(
            target_cooldown=jnp.array(-1, dtype=jnp.int32),
            special_target_cooldown=jnp.array(-1, dtype=jnp.int32),
            active_targets=jnp.array([True, True, True, True], dtype=jnp.bool_),
            rollover_enabled=jnp.array(True, dtype=jnp.bool_),
        )

        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: Union[int, float]
    ) -> Tuple[chex.Array, VideoPinballState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.remove_cooldown(new_state)

        return new_obs, new_state, reward, done, info


class LowTiltEffect(JaxatariWrapper):
    """Decreases the tilt effect when nudging."""

    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self._env.consts._replace(NUDGE_EFFECT_AMOUNT=jnp.array(0.15))


class ConstantBallDynamics(JaxatariWrapper):
    """Removes all/most of the player's ability to change the ball trajectory."""
    
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        # Overrides the _decide_on_ball_vel from env
        self._env._decide_on_ball_vel = self._decide_on_ball_vel.__get__(self._env, self._env.__class__) 

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decide_on_ball_vel(
        self,
        ball_in_play: chex.Array,
        any_collision: chex.Array,
        ball_vel_x: chex.Array,
        ball_vel_y: chex.Array,
        original_ball_speed: chex.Array,
        collision_vel_x: chex.Array,
        collision_vel_y: chex.Array,
    ):
        """
        In the original game, the angle of the ball trajectory after a collision is determined by its
        reflection angle (as one would expect). However, to avoid numerical instabilities, the
        game will keep the angle of the pre-collision trajectory if the reflected velocity is too small
        to avoid numerical instabilities. This enables us to remove the player's ability to influence
        the reflection angle (e.g. by tilting), by simply always setting the reflected angle to the
        pre-collision angle.
        """
        return jnp.abs(ball_vel_x), jnp.abs(ball_vel_y), original_ball_speed