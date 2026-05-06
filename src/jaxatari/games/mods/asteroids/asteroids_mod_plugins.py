import jax.numpy as jnp
import numpy as np
import os
from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_asteroids import AsteroidsState
from jaxatari.environment import JAXAtariAction as Action
import jax.lax
from functools import partial
import jax
from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir

class DontShootMod(JaxAtariPostStepModPlugin):
    """
    Mod that provides a reward of 20 every 300 frames but penalizes shooting by -5.
    This mod updates the state's score to reflect these changes.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: AsteroidsState, new_state: AsteroidsState):
        periodic_reward = jnp.where(
            (new_state.step_counter % 300 == 0) & (new_state.step_counter > 0),
            20,
            0
        )
        
        missile_lifespan = self._env.consts.MISSILE_LIFESPAN
        shot_fired = jnp.any(
            (new_state.missile_states[:, 5] == missile_lifespan) & 
            (prev_state.missile_states[:, 5] != missile_lifespan)
        )
        penalty = jnp.where(shot_fired, -5, 0)
        
        return new_state.replace(
            score=new_state.score + periodic_reward + penalty
        )

def _recolor_all(sprite: np.ndarray, new_rgb: tuple) -> np.ndarray:
    """Replaces all non-transparent pixels with new_rgb."""
    sprite = sprite.copy()
    if sprite.shape[-1] == 3:
        is_transparent = (sprite[:, :, 0] == 0) & (sprite[:, :, 1] == 0) & (sprite[:, :, 2] == 0)
        alpha = np.where(is_transparent, 0, 255).astype(np.uint8)
        sprite = np.concatenate([sprite, alpha[..., None]], axis=-1)
    mask = sprite[..., 3] > 128
    sprite[mask, :3] = new_rgb
    return sprite

def _load_and_recolor_group(filenames, new_rgb, transpose=False) -> list:
    base_dir = os.path.join(get_base_sprite_dir(), "asteroids")
    sprites = []
    for f in filenames:
        sprite = np.load(os.path.join(base_dir, f))
        if transpose:
            sprite = np.transpose(sprite, (1, 0, 2))
        sprites.append(jnp.array(_recolor_all(sprite, new_rgb)))
    return sprites

def _load_and_recolor_single(filename, new_rgb, transpose=False) -> jnp.ndarray:
    base_dir = os.path.join(get_base_sprite_dir(), "asteroids")
    sprite = np.load(os.path.join(base_dir, filename))
    if transpose:
        sprite = np.transpose(sprite, (1, 0, 2))
    return jnp.array(_recolor_all(sprite, new_rgb))

def _get_player_group_recolored():
    player_files = [f'player_pos{i}.npy' for i in range(16)] + [f'death_player{i}.npy' for i in range(3)]
    return _load_and_recolor_group(player_files, (150, 255, 150))

def _get_asteroid_group_recolored():
    asteroid_files = []
    for size in ['big1', 'big2', 'medium', 'small']:
        for color in ['brown', 'grey', 'lightblue', 'lightyellow', 'pink', 'purple', 'red', 'yellow']:
            asteroid_files.append(f'asteroid_{size}_{color}.npy')
    for size in ['big', 'medium', 'small']:
        for color in ['pink', 'yellow']:
            asteroid_files.append(f'death_{size}_{color}.npy')
    return _load_and_recolor_group(asteroid_files, (0, 200, 0))

def _get_digits_recolored() -> jnp.ndarray:
    sprites = _load_and_recolor_group([f'{i}.npy' for i in range(10)], (0, 255, 0))
    max_height = max(s.shape[0] for s in sprites)
    max_width = max(s.shape[1] for s in sprites)
    padded_digits = []
    for digit in sprites:
        digit = np.array(digit)
        pad_h = max_height - digit.shape[0]
        pad_w = max_width - digit.shape[1]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_digit = np.pad(
            digit,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        padded_digits.append(padded_digit)
    return jnp.stack([jnp.array(p) for p in padded_digits])

class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod for Asteroids: black background, green elements."""
    name = "matrix_theme"

    constants_overrides = {
        'WALL_COLOR': (0, 100, 0),
    }
    
    asset_overrides = {
        'player_group': {
            'name': 'player_group',
            'type': 'group',
            'data': _get_player_group_recolored()
        },
        'asteroid_group': {
            'name': 'asteroid_group',
            'type': 'group',
            'data': _get_asteroid_group_recolored()
        },
        'missile1': {
            'name': 'missile1',
            'type': 'single',
            'data': _load_and_recolor_single('missile1.npy', (0, 255, 0))
        },
        'missile2': {
            'name': 'missile2',
            'type': 'single',
            'data': _load_and_recolor_single('missile2.npy', (0, 255, 0))
        },
        'digits': {
            'name': 'digits',
            'type': 'digits',
            'data': _get_digits_recolored()
        },
        'minus_sign': {
            'name': 'minus_sign',
            'type': 'procedural',
            'data': jnp.zeros((10, 12, 4), dtype=jnp.uint8).at[4:6, 2:10, :].set(jnp.array([0, 255, 0, 255], dtype=jnp.uint8))
        },
        'wall_color': {
            'name': 'wall_color',
            'type': 'procedural',
            'data': jnp.array([0, 100, 0, 255], dtype=jnp.uint8).reshape(1, 1, 4)
        }
    }

class SlowAsteroidsMod(JaxAtariInternalModPlugin):
    """
    Mod that slows down asteroids by only updating their positions every 2nd frame.
    """
    @partial(jax.jit, static_argnums=(0,))
    def asteroids_step(self, asteroids_state: AsteroidsState):
        should_move = asteroids_state.step_counter % 2 == 0
        def _move_logic(_):
            asteroid_states = asteroids_state.asteroid_states
            side_step_counter = asteroids_state.side_step_counter
            rng_key, subkey = jax.random.split(asteroids_state.rng_key)
            counter_step = jax.random.randint(subkey, [], 7, 10)
            side_step = jnp.logical_and(side_step_counter <= counter_step, side_step_counter != 0)
            new_side_step_counter = jax.lax.cond(
                side_step_counter < counter_step,
                lambda: 115 + side_step_counter - counter_step,
                lambda: side_step_counter - counter_step
            )
            @jax.jit
            def update_asteroid(i, asteroid_states):
                ret = jnp.copy(asteroid_states)
                axis_directions = jax.lax.switch(
                    ret[i][2],
                    [
                        lambda: (self._env.consts.ASTEROID_SPEED[0], self._env.consts.ASTEROID_SPEED[1]),
                        lambda: (-self._env.consts.ASTEROID_SPEED[0], self._env.consts.ASTEROID_SPEED[1]),
                        lambda: (-self._env.consts.ASTEROID_SPEED[0], -self._env.consts.ASTEROID_SPEED[1]),
                        lambda: (self._env.consts.ASTEROID_SPEED[0], -self._env.consts.ASTEROID_SPEED[1])
                    ]
                )
                return ret.at[i].set(jax.lax.cond(
                    ret[i][3] != self._env.consts.INACTIVE,
                    lambda: jnp.array([self._env.final_pos(self._env.consts.MIN_ENTITY_X,
                                                      self._env.consts.MAX_ENTITY_X,
                                                      jax.lax.cond(
                                                          side_step,
                                                          lambda: ret[i][0] + axis_directions[0],
                                                          lambda: ret[i][0])),
                                       self._env.final_pos(self._env.consts.MIN_ENTITY_Y,
                                                      self._env.consts.MAX_ENTITY_Y,
                                                      ret[i][1] + axis_directions[1]),
                                       ret[i][2], ret[i][3], ret[i][4]]),
                    lambda: ret[i]
                ))
            new_asteroid_states = jax.lax.fori_loop(0, self._env.consts.MAX_NUMBER_OF_ASTEROIDS, update_asteroid, asteroid_states)
            return new_asteroid_states, new_side_step_counter, rng_key

        def _no_move_logic(_):
            return asteroids_state.asteroid_states, asteroids_state.side_step_counter, asteroids_state.rng_key

        return jax.lax.cond(should_move, _move_logic, _no_move_logic, operand=None)

class InstantTurnMod(JaxAtariInternalModPlugin):
    """Directly places the ship in the direction given by the action and applies thrust."""
    name = "instant_turn"

    attribute_overrides = {
        "ACTION_SET": jnp.array(
            [
                Action.NOOP,
                Action.FIRE,
                Action.UP,
                Action.RIGHT,
                Action.LEFT,
                Action.DOWN,
                Action.UPRIGHT,
                Action.UPLEFT,
                Action.DOWNRIGHT,
                Action.DOWNLEFT,
                Action.UPFIRE,
                Action.RIGHTFIRE,
                Action.LEFTFIRE,
                Action.DOWNFIRE,
                Action.UPRIGHTFIRE,
                Action.UPLEFTFIRE,
                Action.DOWNRIGHTFIRE,
                Action.DOWNLEFTFIRE,
            ],
            dtype=jnp.int32,
        )
    }

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state_player_x,
        state_player_y,
        state_player_speed_x,
        state_player_speed_y,
        state_player_rotation,
        action,
        state_respawn_timer,
        rng_key
    ):
        # 1. Parse actions into logical directions
        left = jnp.logical_or(jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE),
                              jnp.logical_or(jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE),
                                             jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE)))
        right = jnp.logical_or(jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE),
                               jnp.logical_or(jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE),
                                              jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)))
        up = jnp.logical_or(jnp.logical_or(action == Action.UP, action == Action.UPFIRE),
                            jnp.logical_or(jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE),
                                           jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE)))
        down = jnp.logical_or(jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE),
                              jnp.logical_or(jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE),
                                             jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)))

        any_direction = jnp.logical_or(jnp.logical_or(up, down), jnp.logical_or(right, left))

        # 2. Determine new rotation (instant)
        # UP=0, UPLEFT=2, LEFT=4, DOWNLEFT=6, DOWN=8, DOWNRIGHT=10, RIGHT=12, UPRIGHT=14
        new_rotation = jax.lax.cond(
            up,
            lambda: jax.lax.cond(left, lambda: 2, lambda: jax.lax.cond(right, lambda: 14, lambda: 0)),
            lambda: jax.lax.cond(
                down,
                lambda: jax.lax.cond(left, lambda: 6, lambda: jax.lax.cond(right, lambda: 10, lambda: 8)),
                lambda: jax.lax.cond(
                    left, lambda: 4, lambda: jax.lax.cond(right, lambda: 12, lambda: state_player_rotation)
                )
            )
        )

        player_rotation = jax.lax.cond(
            any_direction,
            lambda: new_rotation,
            lambda: state_player_rotation
        )

        # 3. Apply physics based on the new rotation
        player_x = state_player_x
        player_y = state_player_y
        player_speed_x = state_player_speed_x
        player_speed_y = state_player_speed_y

        decel_x = self._env.decel_func(player_speed_x)
        decel_y = self._env.decel_func(player_speed_y)

        accel_x = self._env.consts.ACCEL_PER_ROTATION[player_rotation][0]
        accel_y = self._env.consts.ACCEL_PER_ROTATION[player_rotation][1]

        # In instant turn mod, pressing any direction triggers thrust
        is_thrusting = any_direction

        adj_speed_x = jnp.logical_and(
            jnp.logical_and(is_thrusting, jnp.abs(player_speed_x + accel_x) < self._env.consts.MAX_PLAYER_SPEED),
            jnp.logical_not(player_rotation%8 == 0))
        adj_speed_y = jnp.logical_and(
            jnp.logical_and(is_thrusting, jnp.abs(player_speed_y + accel_y) < self._env.consts.MAX_PLAYER_SPEED),
            jnp.logical_not((player_rotation-4)%8 == 0))

        # calculate new player speed
        player_speed_x = jax.lax.cond(
            adj_speed_x,
            lambda: player_speed_x + accel_x,
            lambda: player_speed_x
        )
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_x), jnp.abs(player_speed_x) > jnp.abs(decel_x)),
            lambda: player_speed_x + decel_x,
            lambda: player_speed_x
        )
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_x), jnp.abs(player_speed_x) <= jnp.abs(decel_x)),
            lambda: 0,
            lambda: player_speed_x
        )

        player_speed_y = jax.lax.cond(
            adj_speed_y,
            lambda: player_speed_y + accel_y,
            lambda: player_speed_y
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_y), jnp.abs(player_speed_y) > jnp.abs(decel_y)),
            lambda: player_speed_y + decel_y,
            lambda: player_speed_y
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_y), jnp.abs(player_speed_y) <= jnp.abs(decel_y)),
            lambda: 0,
            lambda: player_speed_y
        )

        displace_x = self._env.speed_func(player_speed_x)
        displace_y = self._env.speed_func(player_speed_y)

        player_x = jnp.int32(self._env.final_pos(self._env.consts.MIN_PLAYER_X, self._env.consts.MAX_PLAYER_X, player_x + displace_x))
        player_y = jnp.int32(self._env.final_pos(self._env.consts.MIN_PLAYER_Y, self._env.consts.MAX_PLAYER_Y, player_y + displace_y))

        # We remove hyperspace (down) entirely so you can fly down without teleporting
        
        return jax.lax.cond(
            state_respawn_timer <= 0,
            lambda: (player_x, player_y, player_speed_x, player_speed_y,
                     player_rotation, state_respawn_timer, rng_key),
            lambda: (state_player_x, state_player_y, state_player_speed_x, state_player_speed_y,
                     state_player_rotation, state_respawn_timer, rng_key)
        )
