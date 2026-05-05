import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_pong import PongState
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir


def _recolor_sprite(filename: str, original_rgb: tuple, new_rgb: tuple) -> np.ndarray:
    """Load a pong sprite .npy and replace original_rgb with new_rgb (alpha preserved)."""
    sprite_path = os.path.join(get_base_sprite_dir(), "pong", filename)
    sprite = np.load(sprite_path).copy()
    original = np.array([*original_rgb, 255], dtype=np.uint8)
    replacement = np.array([*new_rgb, 255], dtype=np.uint8)
    mask = np.all(sprite == original, axis=-1)
    sprite[mask] = replacement
    return sprite


def _make_recolored_background(new_color: tuple) -> np.ndarray:
    return _recolor_sprite("background.npy", (144, 72, 17), new_color)

# --- 1. Individual Mod Plugins ---
class LazyEnemyMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["random_enemy"]

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """
        Replaces the base _enemy_step logic.
        Access the environment via self._env (set by JaxAtariModController).
        """
        should_move = (state.step_counter % 8 != 0) & (state.ball_vel_x < 0)
        direction = jnp.sign(state.ball_y - state.enemy_y)
        new_y = state.enemy_y + (direction * self._env.consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        final_y = jax.lax.cond(should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None)
        return state.replace(enemy_y=final_y.astype(jnp.int32))

class RandomEnemyMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["lazy_enemy"]

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """
        Replaces the base _enemy_step logic.
        'self_env' is the bound JaxPong instance.
        'key' is now used for randomness.
        """
        # Split key: use one part for randomness, keep remainder for state
        rng_key, unused_key = jax.random.split(state.key)
        random_dir = jax.random.choice(rng_key, jnp.array([-1, 1]))
        random_cond = state.step_counter % 3 == 0
        new_y = state.enemy_y + (random_dir * self._env.consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        # Clamp to screen bounds
        new_y = jnp.clip(
            new_y,
            self._env.consts.WALL_TOP_Y + self._env.consts.WALL_TOP_HEIGHT - 10,
            self._env.consts.WALL_BOTTOM_Y - 4,
        )

        final_y = jax.lax.cond(random_cond, lambda _: new_y, lambda _: state.enemy_y, operand=None)
        # Return unused_key; step() will replace with new_state_key at the end
        return state.replace(enemy_y=final_y.astype(jnp.int32), key=unused_key)



class AlwaysZeroScoreMod(JaxAtariPostStepModPlugin):    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        return new_state.replace(
            player_score=jnp.array(0, dtype=jnp.int32),
            enemy_score=jnp.array(0, dtype=jnp.int32)
        )
    

class LinearMovementMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: PongState, action: chex.Array) -> PongState:
        up = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        down = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        # Direct movement: move 2 pixels per frame when input pressed
        move_amount = jnp.array(2.0, dtype=jnp.float32)

        new_player_y = state.player_y
        new_player_y = jax.lax.cond(
            up,
            lambda y: y - move_amount,
            lambda y: y,
            operand=new_player_y,
        )

        new_player_y = jax.lax.cond(
            down,
            lambda y: y + move_amount,
            lambda y: y,
            operand=new_player_y,
        )

        # Hard boundaries using the analog paddle limits
        new_player_y = jnp.clip(
            new_player_y,
            self._env.consts.PADDLE_MIN_Y,
            self._env.consts.PADDLE_MAX_Y,
        )

        return state.replace(
            player_y=new_player_y,
            player_speed=jnp.array(0.0, dtype=jnp.float32),
        )

class ShiftPlayerMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "PLAYER_X": 136,
    }

class ShiftEnemyMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "ENEMY_X": 20,
    }


class NoFireMod(JaxAtariInternalModPlugin):
    attribute_overrides = {
        "ACTION_SET": jnp.array([Action.NOOP, Action.RIGHT, Action.LEFT], dtype=jnp.int32),
    }


class ChangeBackgroundColorMod(JaxAtariInternalModPlugin):
    """Changes the playfield background color. Default: navy blue (0, 0, 128)."""
    _NEW_BG_COLOR = (0, 0, 128)

    constants_overrides = {"BACKGROUND_COLOR": _NEW_BG_COLOR}
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_recolored_background(_NEW_BG_COLOR),
        }
    }


class ChangePlayerColorMod(JaxAtariInternalModPlugin):
    """Changes the player paddle color. Default: red (255, 0, 0)."""
    _NEW_PLAYER_COLOR = (255, 0, 0)

    constants_overrides = {"PLAYER_COLOR": _NEW_PLAYER_COLOR}
    asset_overrides = {
        "player": {
            "name": "player",
            "type": "single",
            "data": _recolor_sprite("player.npy", (92, 186, 92), _NEW_PLAYER_COLOR),
        }
    }


class RewardAccelerationMod(JaxAtariInternalModPlugin):
    """Rewards (+2) the player each time the ball is accelerated by the player."""
    constants_overrides = {"ACCELERATION_REWARD": 2.0}