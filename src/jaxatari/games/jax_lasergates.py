"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""
import os
from functools import partial
from typing import Tuple, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxatari.rendering.atraJaxis as aj
import pygame
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import AtraJaxisRenderer

# -------- Game constants --------
WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 4

SCROLL_SPEED = 1

# -------- Mountains constants --------
MOUNTAIN_SIZE = (60, 12) # Width, Height

LOWER_MOUNTAINS_Y = 82 # Y Spawn position of lower mountains. This does not change
UPPER_MOUNTAINS_Y = 21 # Y Spawn position of upper mountains. This does not change

LOWER_MOUNTAINS_START_X = -44 # X Spawn position of lower mountains.
UPPER_MOUNTAINS_START_X = -4 # X Spawn position of upper mountains.

MOUNTAINS_DISTANCE = 20 # Distance between two given mountains

UPDATE_EVERY = 4 # The mountain position is updated every UPDATE_EVERY-th frame.

# -------- Player constants --------
PLAYER_SIZE = (8, 6) # Width, Height
PLAYER_COLOR = (85, 92, 197, 255)
PLAYER_BOUNDS = (20, WIDTH - 20 - PLAYER_SIZE[0]), (21, 88)

PLAYER_START_X = 20 # X Spawn position of player
PLAYER_START_Y = 52 # Y Spawn position of player

PLAYER_VELOCITY_Y = 1.5 # Y Velocity of player
PLAYER_VELOCITY_X = 1.5 # X Velocity of player

# -------- Player missile constants --------
PLAYER_MISSILE_SIZE = (16, 1)
PLAYER_MISSILE_COLOR = (54, 46, 200, 255)

PLAYER_MISSILE_INITIAL_VELOCITY = 2.5
PLAYER_MISSILE_VELOCITY_MULTIPLIER = 1.1

# -------- Enemy Missile constants --------
ENEMY_MISSILE_COLOR = (85, 92, 197, 255)

# -------- GUI constants --------


# -------- States --------
class MountainState(NamedTuple):
    x1: chex.Array
    x2: chex.Array
    x3: chex.Array
    y: chex.Array

class PlayerMissileState(NamedTuple):
    x: chex.Array
    y: chex.Array
    direction: chex.Array
    velocity: chex.Array

class LaserGatesState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_facing_direction: chex.Array
    player_missile: PlayerMissileState
    lower_mountains: MountainState
    upper_mountains: MountainState
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    # TODO: fill

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class LaserGatesObservation(NamedTuple):
    player: EntityPosition
    # enemy1
    # enemy2: EntityPosition
    # ...: EntityPosition
    # ...: EntityPosition
    # TODO: fill

class LaserGatesInfo(NamedTuple):
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

# -------- Render Constants --------
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    background = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/background.npy"))
    lower_mountain = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/lower_mountain.npy"))
    upper_mountain = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/background/mountains/upper_mountain.npy"))
    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/player/player.npy"))
    player_missile = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/lasergates/missiles/player_missile.npy"))


    SPRITE_BACKGROUND = background
    SPRITE_PLAYER = player
    SPRITE_PLAYER_MISSILE = player_missile
    SPRITE_LOWER_MOUNTAIN = lower_mountain
    SPRITE_UPPER_MOUNTAIN = upper_mountain

    return (
        SPRITE_BACKGROUND,
        SPRITE_PLAYER,
        SPRITE_PLAYER_MISSILE,
        SPRITE_LOWER_MOUNTAIN,
        SPRITE_UPPER_MOUNTAIN,
    )

(
    SPRITE_BACKGROUND,
    SPRITE_PLAYER,
    SPRITE_PLAYER_MISSILE,
    SPRITE_LOWER_MOUNTAIN,
    SPRITE_UPPER_MOUNTAIN,
) = load_sprites()

# -------- Game Logic --------

@jax.jit
def mountains_step(
        mountain_state: MountainState, step_counter: jnp.ndarray
) -> MountainState:

    # If this is true, update the position
    update_tick = step_counter % UPDATE_EVERY == 0

    # Update x positions
    new_x1 = jnp.where(update_tick, mountain_state.x1 - UPDATE_EVERY * SCROLL_SPEED, mountain_state.x1)
    new_x2 = jnp.where(update_tick, mountain_state.x2 - UPDATE_EVERY * SCROLL_SPEED, mountain_state.x2)
    new_x3 = jnp.where(update_tick, mountain_state.x3 - UPDATE_EVERY * SCROLL_SPEED, mountain_state.x3)

    # If completely behind left border, set x position to the right again
    new_x1 = jnp.where(new_x1 < 0 - MOUNTAIN_SIZE[0], new_x3 + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE, new_x1)
    new_x2 = jnp.where(new_x2 < 0 - MOUNTAIN_SIZE[0], new_x1 + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE, new_x2)
    new_x3 = jnp.where(new_x3 < 0 - MOUNTAIN_SIZE[0], new_x2 + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE, new_x3)

    return MountainState(x1=new_x1, x2=new_x2, x3=new_x3, y=mountain_state.y)


@jax.jit
def player_step(
        state: LaserGatesState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    up = jnp.isin(action, jnp.array([
        Action.UP,
        Action.UPRIGHT,
        Action.UPLEFT,
        Action.UPFIRE,
        Action.UPRIGHTFIRE,
        Action.UPLEFTFIRE
    ]))
    down = jnp.isin(action, jnp.array([
        Action.DOWN,
        Action.DOWNRIGHT,
        Action.DOWNLEFT,
        Action.DOWNFIRE,
        Action.DOWNRIGHTFIRE,
        Action.DOWNLEFTFIRE
    ]))
    left = jnp.isin(action, jnp.array([
        Action.LEFT,
        Action.UPLEFT,
        Action.DOWNLEFT,
        Action.LEFTFIRE,
        Action.UPLEFTFIRE,
        Action.DOWNLEFTFIRE
    ]))
    right = jnp.isin(action, jnp.array([
        Action.RIGHT,
        Action.UPRIGHT,
        Action.DOWNRIGHT,
        Action.RIGHTFIRE,
        Action.UPRIGHTFIRE,
        Action.DOWNRIGHTFIRE
    ]))

    # Move x
    delta_x = jnp.where(left, -PLAYER_VELOCITY_X, jnp.where(right, PLAYER_VELOCITY_X, 0))
    player_x = jnp.clip(state.player_x + delta_x, PLAYER_BOUNDS[0][0], PLAYER_BOUNDS[0][1])

    # Move y
    delta_y = jnp.where(up, -PLAYER_VELOCITY_Y, jnp.where(down, PLAYER_VELOCITY_Y, 0))
    player_y = jnp.clip(state.player_y + delta_y, PLAYER_BOUNDS[1][0], PLAYER_BOUNDS[1][1])

    # Player facing direction
    new_player_facing_direction = jnp.where(right, 1, jnp.where(left, -1, state.player_facing_direction))

    no_x_input = jnp.logical_and(
        jnp.logical_not(left), jnp.logical_not(right)
        )

    # SCROLL LEFT
    player_x = jnp.where(no_x_input, player_x - SCROLL_SPEED, player_x)

    return player_x, player_y, new_player_facing_direction

@jax.jit
def player_missile_step(
        state: LaserGatesState, action: chex.Array
) -> PlayerMissileState:

    fire = jnp.isin(action, jnp.array([
        Action.FIRE,
        Action.UPFIRE,
        Action.RIGHTFIRE,
        Action.LEFTFIRE,
        Action.DOWNFIRE,
        Action.UPRIGHTFIRE,
        Action.UPLEFTFIRE,
        Action.DOWNRIGHTFIRE,
        Action.DOWNLEFTFIRE
    ]))


    is_alive = state.player_missile.direction != 0
    out_of_bounds = jnp.logical_or(
        state.player_missile.x < 0 - PLAYER_MISSILE_SIZE[0],
        state.player_missile.x > WIDTH
    )
    kill = jnp.logical_and(is_alive, out_of_bounds)

    # Kill missile
    new_x = jnp.where(kill, 0, state.player_missile.x)
    new_y = jnp.where(kill, 0, state.player_missile.y)
    new_direction = jnp.where(kill, 0, state.player_missile.direction)
    new_velocity = jnp.where(kill, 0, state.player_missile.velocity)

    # Move missile
    new_x = jnp.where(
        is_alive,
        new_x + jnp.where(new_direction > 0, state.player_missile.velocity, -state.player_missile.velocity),
        new_x
    ) # Move by the velocity in state
    new_velocity = jnp.where(
        is_alive,
        new_velocity * PLAYER_MISSILE_VELOCITY_MULTIPLIER,
        new_velocity
    ) # Multiply velocity by given constant

    # Spawn missile
    spawn = jnp.logical_and(jnp.logical_not(is_alive), fire)
    new_x = jnp.where(spawn, jnp.where(
        state.player_facing_direction > 0,
        state.player_x + PLAYER_SIZE[0],
        state.player_x - 2 * PLAYER_SIZE[0] - 1
    ), new_x)
    new_y = jnp.where(spawn, state.player_y + 4, new_y)
    new_direction = jnp.where(spawn, state.player_facing_direction, new_direction)
    new_velocity = jnp.where(spawn, PLAYER_MISSILE_INITIAL_VELOCITY, new_velocity)

    return PlayerMissileState(x=new_x, y=new_y, direction=new_direction, velocity=new_velocity)


class JaxLaserGates(JaxEnvironment[LaserGatesState, LaserGatesObservation, LaserGatesInfo]):
    def __init__(self, reward_funcs: list[callable] =None):
        super().__init__()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
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
            Action.DOWNLEFTFIRE
        ]
        # self.frame_stack_size = 4 # ???
        # self.obs_size = 1024 # ???

    # TODO: add other funtions if needed

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: LaserGatesState) -> LaserGatesObservation:
        # TODO: fill
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(0),
            height=jnp.array(0), # TODO: Import sizes
            active=jnp.array(1),
        )

        return LaserGatesObservation(
            player=player,
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_info(self, state: LaserGatesState, all_rewards: jnp.ndarray) -> LaserGatesInfo:
        # TODO: fill
        return LaserGatesInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @jax.jit
    def _get_env_reward(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: LaserGatesState, state: LaserGatesState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @jax.jit
    def _get_done(self, state: LaserGatesState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self) -> Tuple[LaserGatesObservation, LaserGatesState]:
        """Initialize game state"""

        initial_lower_mountains = MountainState(
            x1=jnp.array(LOWER_MOUNTAINS_START_X),
            x2=jnp.array(LOWER_MOUNTAINS_START_X + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE),
            x3=jnp.array(LOWER_MOUNTAINS_START_X + 2 * MOUNTAIN_SIZE[0] + 2 * MOUNTAINS_DISTANCE),
            y=jnp.array(LOWER_MOUNTAINS_Y)
        )

        initial_upper_mountains = MountainState(
            x1=jnp.array(UPPER_MOUNTAINS_START_X),
            x2=jnp.array(UPPER_MOUNTAINS_START_X + MOUNTAIN_SIZE[0] + MOUNTAINS_DISTANCE),
            x3=jnp.array(UPPER_MOUNTAINS_START_X + 2 * MOUNTAIN_SIZE[0] + 2 * MOUNTAINS_DISTANCE),
            y=jnp.array(UPPER_MOUNTAINS_Y)
        )

        initial_player_missile = PlayerMissileState(
            x=jnp.array(0),
            y=jnp.array(0),
            direction=jnp.array(0),
            velocity=jnp.array(0),
        )

        reset_state = LaserGatesState( # TODO: fill
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_facing_direction=jnp.array(1),
            player_missile=initial_player_missile,
            lower_mountains=initial_lower_mountains,
            upper_mountains=initial_upper_mountains,
            score=jnp.array(0),
            lives=jnp.array(3),
            step_counter=jnp.array(0),
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
            self, state: LaserGatesState, action: Action
    ) -> Tuple[LaserGatesObservation, LaserGatesState, float, bool, LaserGatesInfo]:
        # TODO: fill

        # -------- Move player --------
        new_player_x, new_player_y, new_player_facing_direction = player_step(state, action)

        new_lower_mountains_state = mountains_step(state.lower_mountains, state.step_counter)
        new_upper_mountains_state = mountains_step(state.upper_mountains, state.step_counter)

        new_player_missile_state = player_missile_step(state, action)

        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_facing_direction=new_player_facing_direction,
            player_missile=new_player_missile_state,
            lower_mountains=new_lower_mountains_state,
            upper_mountains=new_upper_mountains_state,
            step_counter=state.step_counter + 1
        )

        obs = self._get_observation(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, 0.0, False, info

class LaserGatesRenderer(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        def recolor_sprite(sprite: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
            assert sprite.ndim == 3 and sprite.shape[2] in (3, 4), "Sprite must be HxWx3 or HxWx4"
            assert color.shape[0] == sprite.shape[2], "Color channels must match sprite channels"

            # Define a visibility mask: pixel is visible if any of its channels > 0
            visible_mask = jnp.any(sprite != 0, axis=-1)  # (H, W)
            visible_mask = visible_mask[:, :, None]  # (H, W, 1) for broadcasting

            # Broadcast color to the same shape as sprite
            color_broadcasted = jnp.broadcast_to(color, sprite.shape)

            # Where visible, use the new color; otherwise keep black (zeros)
            return jnp.where(visible_mask, color_broadcasted, 0)

        # -------- Render background --------
        raster = aj.render_at(
            raster,
            0,
            0,
            SPRITE_BACKGROUND,
        )

        # -------- Render mountains --------

        # Lower mountains
        raster = aj.render_at(
            raster,
            state.lower_mountains.x1,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.lower_mountains.x2,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.lower_mountains.x3,
            state.lower_mountains.y,
            SPRITE_LOWER_MOUNTAIN,
        )

        # Upper mountains
        raster = aj.render_at(
            raster,
            state.upper_mountains.x1,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.upper_mountains.x2,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        raster = aj.render_at(
            raster,
            state.upper_mountains.x3,
            state.upper_mountains.y,
            SPRITE_UPPER_MOUNTAIN,
        )

        # -------- Render player --------
        frame_player = recolor_sprite(SPRITE_PLAYER, jnp.array(PLAYER_COLOR))
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_player,
            flip_horizontal=state.player_facing_direction < 0,
        )

        # -------- Render player missile --------
        frame_player = recolor_sprite(SPRITE_PLAYER_MISSILE, jnp.array(PLAYER_MISSILE_COLOR))

        raster = jnp.where(state.player_missile.direction != 0,
                      aj.render_at(
                      raster,
                      state.player_missile.x,
                      state.player_missile.y,
                      frame_player,
                      flip_horizontal=state.player_missile.direction < 0,
                      ),
                  raster
                  )

        return raster

def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)

if __name__ == "__main__":
    # Initialize game and renderer
    game = JaxLaserGates()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()

    renderer_AtraJaxis = LaserGatesRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )
                        print(f"Observations: {curr_obs}")
                        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_obs, curr_state, reward, done, info = jitted_step(
                    curr_state, action
                )

        # render and update pygame
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1
        clock.tick(60)

    pygame.quit()