import os
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_pong import PongState, PongObservation
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
from jaxatari import spaces
from jaxatari.environment import JAXAtariAction as Action, ObjectObservation
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


def _recolor_sprite_keep_alpha(filename: str, new_rgb: tuple) -> np.ndarray:
    """Load a pong sprite and recolor every opaque pixel to ``new_rgb``.

    Unlike :func:`_recolor_sprite` this does not depend on the exact original
    color, which makes it robust to sprites that use slightly different shades
    (e.g. the ball is recolored to give each of the 3 balls a distinct color).
    """
    sprite_path = os.path.join(get_base_sprite_dir(), "pong", filename)
    sprite = np.load(sprite_path).copy()
    opaque = sprite[..., 3] > 128
    sprite[opaque, 0] = new_rgb[0]
    sprite[opaque, 1] = new_rgb[1]
    sprite[opaque, 2] = new_rgb[2]
    return sprite

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


# --- 3-Ball / 3-Paddle Mod ---

def _enemy_zones(consts) -> Tuple[chex.Array, chex.Array]:
    """Split the vertical play area into 3 equal zones for the enemy paddles.

    Paddle ``i`` may only move inside ``[zone_mins[i], zone_maxs[i]]``. The
    zones are adjacent, so each paddle can move exactly until the spot where
    the neighbouring paddle takes over.
    """
    n = 3
    min_y = float(consts.PADDLE_MIN_Y)
    max_y = float(consts.PADDLE_MAX_Y)
    edges = min_y + (max_y - min_y) * jnp.arange(n + 1) / n
    return edges[:-1].astype(jnp.int32), edges[1:].astype(jnp.int32)


class TriplePongMod(JaxAtariInternalModPlugin):
    """Pong with 3 enemy paddles and 3 balls instead of 1 each.

    - The 3 enemy paddles split the vertical play area into 3 equally sized
      zones; each paddle only moves within its own zone (up to the spot where
      the neighbouring paddle takes over) and tracks the closest incoming ball
      to intercept it.
    - The 3 balls spawn staggered in time (``SPAWN_DELAY`` steps apart) and
      are drawn in distinct colors so they are easy to tell apart. Balls that
      have not spawned yet are held (and hidden) at the center.
    """

    NUM_BALLS = 3
    NUM_ENEMIES = 3
    SPAWN_DELAY = 90

    # Colors of the three balls (index 0 keeps the original sprite).
    BALL_COLORS = [
        (236, 236, 236),  # white (original)
        (255, 0, 0),      # red
        (0, 128, 255),    # blue
    ]

    asset_overrides = {
        "ball_red": {
            "name": "ball_red",
            "type": "single",
            "data": _recolor_sprite_keep_alpha("ball.npy", (255, 0, 0)),
        },
        "ball_blue": {
            "name": "ball_blue",
            "type": "single",
            "data": _recolor_sprite_keep_alpha("ball.npy", (0, 128, 255)),
        },
    }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[PongObservation, PongState]:
        """Reset with 3 balls / 3 enemy paddles.

        Ball 0 is live immediately (matching the base game). Balls 1 and 2 are
        held dormant at the center and spawn after ``SPAWN_DELAY`` /
        ``2 * SPAWN_DELAY`` steps. The enemy paddles start in the middle of
        their own zone.
        """
        consts = self._env.consts
        n = self.NUM_BALLS

        state_key, _ = jax.random.split(key)

        ball_x = jnp.full((n,), consts.BALL_START_X, dtype=jnp.int32)
        ball_y = jnp.full((n,), consts.BALL_START_Y, dtype=jnp.int32)
        ball_vel_x = jnp.zeros((n,), dtype=jnp.int32)
        ball_vel_y = jnp.zeros((n,), dtype=jnp.int32)
        ball_vel_x = ball_vel_x.at[0].set(consts.BALL_SPEED[0])
        ball_vel_y = ball_vel_y.at[0].set(consts.BALL_SPEED[1])

        zone_mins, zone_maxs = _enemy_zones(consts)
        enemy_y = ((zone_mins + zone_maxs) // 2).astype(jnp.int32)

        state = PongState(
            player_y=jnp.array(96.0, dtype=jnp.float32),
            player_speed=jnp.array(0.0, dtype=jnp.float32),
            ball_x=ball_x,
            ball_y=ball_y,
            enemy_y=enemy_y,
            enemy_speed=jnp.zeros((n,), dtype=jnp.int32),
            ball_vel_x=ball_vel_x,
            ball_vel_y=ball_vel_y,
            player_score=jnp.array(0, dtype=jnp.int32),
            enemy_score=jnp.array(0, dtype=jnp.int32),
            step_counter=jnp.array(0, dtype=jnp.int32),
            key=state_key,
        )
        obs = self._env._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """Move the 3 enemy paddles to intercept incoming balls.

        Each paddle tracks the ball that is currently approaching the enemy
        side (``ball_vel_x < 0``) and is closest to its own position, then
        moves toward it. The zone clamp keeps every paddle in its own lane, so
        together the three paddles cover the whole play area (a ball crossing
        near zone ``i`` is always closest to paddle ``i``). Balls that are not
        approaching (or have not spawned yet) are ignored, so the paddles hold
        position while the ball is on the player's side.
        """
        consts = self._env.consts
        should_move = state.step_counter % 8 != 0

        incoming = state.ball_vel_x < 0                                     # (n,)
        # Distance from every paddle to every ball (n x n), ignoring balls
        # that are not approaching the enemy side.
        dist = jnp.abs(state.ball_y[None, :] - state.enemy_y[:, None])       # (n, n)
        dist = jnp.where(incoming[None, :], dist, jnp.iinfo(jnp.int32).max)
        target_y = state.ball_y[jnp.argmin(dist, axis=1)]                    # (n,)

        any_incoming = jnp.any(incoming)
        direction = jnp.where(any_incoming, jnp.sign(target_y - state.enemy_y), 0)
        new_y = state.enemy_y + (direction * consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        enemy_y = jax.lax.cond(
            should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None
        )

        # Clamp each paddle to its own zone (it can only move until the spot
        # where the neighbouring paddle takes over).
        zone_mins, zone_maxs = _enemy_zones(consts)
        enemy_y = jnp.clip(enemy_y, zone_mins, zone_maxs)

        return state.replace(enemy_y=enemy_y.astype(jnp.int32))

    @partial(jax.jit, static_argnums=(0,))
    def _ball_step(self, state: PongState, action: chex.Array) -> PongState:
        """Vectorized ball physics for 3 balls, with staggered spawning."""
        consts = self._env.consts
        n = self.NUM_BALLS

        # --- Spawn schedule: balls activate SPAWN_DELAY steps apart ---
        spawn_times = self.SPAWN_DELAY * jnp.arange(n, dtype=jnp.int32)
        dormant = state.ball_vel_x == 0
        should_activate = dormant & (state.step_counter >= spawn_times)
        init_vx = jnp.array([-1, 1, -1], dtype=jnp.int32)
        init_vy = jnp.array([-1, 1, 1], dtype=jnp.int32)
        ball_vel_x = jnp.where(should_activate, init_vx, state.ball_vel_x)
        ball_vel_y = jnp.where(should_activate, init_vy, state.ball_vel_y)

        # --- Position integration ---
        ball_x = state.ball_x + ball_vel_x
        ball_y = state.ball_y + ball_vel_y

        # --- Wall bounces ---
        wall_bounce = jnp.logical_or(
            ball_y <= consts.WALL_TOP_Y + consts.WALL_TOP_HEIGHT - consts.BALL_SIZE[1],
            ball_y >= consts.WALL_BOTTOM_Y,
        )
        ball_vel_y = jnp.where(wall_bounce, -ball_vel_y, ball_vel_y)

        # --- Paddle collisions (player + enemy) ---
        player_paddle_hit = jnp.logical_and(
            jnp.logical_and(
                consts.PLAYER_X <= ball_x,
                ball_x <= consts.PLAYER_X + consts.PLAYER_SIZE[0],
            ),
            ball_vel_x > 0,
        )
        player_paddle_hit = jnp.logical_and(
            player_paddle_hit,
            jnp.logical_and(
                state.player_y - consts.BALL_SIZE[1] <= ball_y,
                ball_y <= state.player_y + consts.PLAYER_SIZE[1] + consts.BALL_SIZE[1],
            ),
        )

        # A ball hits the enemy side if ANY of the 3 enemy paddles covers its y
        # (all paddles sit at the same x = ENEMY_X, so the ball can reach any of
        # them). The covering paddle closest to the ball's y handles the bounce.
        enemy_cover = jnp.logical_and(
            state.enemy_y[None, :] - consts.BALL_SIZE[1] <= ball_y[:, None],
            ball_y[:, None] <= state.enemy_y[None, :] + consts.ENEMY_SIZE[1] + consts.BALL_SIZE[1],
        )  # (n_balls, n_enemies)
        enemy_x_check = jnp.logical_and(
            jnp.logical_and(
                consts.ENEMY_X <= ball_x,
                ball_x <= consts.ENEMY_X + consts.ENEMY_SIZE[0] - 1,
            ),
            ball_vel_x < 0,
        )
        enemy_paddle_hit = jnp.logical_and(enemy_x_check, jnp.any(enemy_cover, axis=1))

        paddle_hit = jnp.logical_or(player_paddle_hit, enemy_paddle_hit)

        # --- Hit position -> outgoing angle (mirrors the base paddle logic) ---
        section_height = consts.PLAYER_SIZE[1] / 5
        enemy_dist = jnp.abs(state.enemy_y[None, :] - ball_y[:, None])  # (n_balls, n_enemies)
        enemy_dist = jnp.where(enemy_cover, enemy_dist, jnp.iinfo(jnp.int32).max)
        nearest_enemy_y = state.enemy_y[jnp.argmin(enemy_dist, axis=1)]  # (n_balls,)
        paddle_y_ref = jnp.where(player_paddle_hit, state.player_y, nearest_enemy_y)
        rel = (ball_y - paddle_y_ref) / section_height
        hit_position = jnp.clip(jnp.floor(rel).astype(jnp.int32), 0, 4) - 2
        ball_vel_y = jnp.where(paddle_hit, hit_position, ball_vel_y)

        # --- Boost on FIRE / max-speed hits (player paddle only) ---
        boost_triggered = jnp.logical_and(
            player_paddle_hit,
            jnp.logical_or(
                jnp.logical_or(action == Action.LEFTFIRE, action == Action.RIGHTFIRE),
                action == Action.FIRE,
            ),
        )
        player_max_hit = jnp.logical_and(
            player_paddle_hit,
            jnp.abs(state.player_speed) >= consts.PADDLE_MAX_SPEED,
        )
        ball_vel_x = jnp.where(
            jnp.logical_or(boost_triggered, player_max_hit),
            ball_vel_x + jnp.sign(ball_vel_x),
            ball_vel_x,
        )
        ball_vel_x = jnp.where(paddle_hit, -ball_vel_x, ball_vel_x)

        return state.replace(
            ball_x=ball_x.astype(jnp.int32),
            ball_y=ball_y.astype(jnp.int32),
            ball_vel_x=ball_vel_x.astype(jnp.int32),
            ball_vel_y=ball_vel_y.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _score_and_reset(self, state: PongState) -> PongState:
        """Per-ball scoring: only the ball that left the field is re-served.

        The step counter keeps increasing so the staggered spawn schedule stays
        stable across goals, and the enemy paddles keep their zone positions.
        """
        consts = self._env.consts

        player_goal = state.ball_x < 4
        enemy_goal = state.ball_x > 156
        ball_reset = jnp.logical_or(player_goal, enemy_goal)

        player_score = state.player_score + jnp.sum(player_goal.astype(jnp.int32))
        enemy_score = state.enemy_score + jnp.sum(enemy_goal.astype(jnp.int32))

        # Re-serve each ball that scored, matching the base serve rules.
        serve_vx = jnp.where(enemy_goal, 1, -1)
        serve_vy = jnp.where(state.ball_y > consts.BALL_START_Y, 1, -1)
        ball_x = jnp.where(ball_reset, consts.BALL_START_X, state.ball_x)
        ball_y = jnp.where(ball_reset, consts.BALL_START_Y, state.ball_y)
        ball_vel_x = jnp.where(ball_reset, serve_vx, state.ball_vel_x)
        ball_vel_y = jnp.where(ball_reset, serve_vy, state.ball_vel_y)

        return state.replace(
            ball_x=ball_x.astype(jnp.int32),
            ball_y=ball_y.astype(jnp.int32),
            ball_vel_x=ball_vel_x.astype(jnp.int32),
            ball_vel_y=ball_vel_y.astype(jnp.int32),
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=state.step_counter + 1,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PongState) -> PongObservation:
        """Vectorized observation: 3 balls and 3 enemy paddles.

        The ``active`` flag of a ball is 1 once it has spawned; ``visual_id``
        is the ball/paddle index (ball colors match the index).
        """
        consts = self._env.consts
        n = self.NUM_BALLS

        player = ObjectObservation.create(
            x=jnp.array(consts.PLAYER_X, dtype=jnp.int32),
            y=state.player_y,
            width=jnp.array(consts.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(consts.PLAYER_SIZE[1], dtype=jnp.int32),
        )

        enemy = ObjectObservation.create(
            x=jnp.full((n,), consts.ENEMY_X, dtype=jnp.int32),
            y=state.enemy_y,
            width=jnp.full((n,), consts.ENEMY_SIZE[0], dtype=jnp.int32),
            height=jnp.full((n,), consts.ENEMY_SIZE[1], dtype=jnp.int32),
            active=jnp.ones((n,), dtype=jnp.int32),
            visual_id=jnp.arange(n, dtype=jnp.int32),
        )

        ball = ObjectObservation.create(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.full((n,), consts.BALL_SIZE[0], dtype=jnp.int32),
            height=jnp.full((n,), consts.BALL_SIZE[1], dtype=jnp.int32),
            active=(state.ball_vel_x != 0).astype(jnp.int32),
            visual_id=jnp.arange(n, dtype=jnp.int32),
        )

        return PongObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            score_player=state.player_score,
            score_enemy=state.enemy_score,
        )

    def observation_space(self) -> spaces.Dict:
        """Vectorized observation space for 3 balls and 3 enemy paddles."""
        consts = self._env.consts
        screen_size = (consts.HEIGHT, consts.WIDTH)
        return spaces.Dict({
            "player": spaces.get_object_space(n=None, screen_size=screen_size),
            "enemy": spaces.get_object_space(n=self.NUM_ENEMIES, screen_size=screen_size),
            "ball": spaces.get_object_space(n=self.NUM_BALLS, screen_size=screen_size),
            "score_player": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
            "score_enemy": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
        })