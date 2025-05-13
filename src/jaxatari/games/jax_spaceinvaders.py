import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

WIDTH = 160
HEIGHT = 210

FPS = 30 
SCALING_FACTOR = 4
WINDOW_WIDTH = WIDTH * SCALING_FACTOR
WINDOW_HEIGHT = HEIGHT * SCALING_FACTOR

BULLET_SPEED = 1

NUMBER_YELLOW_OFFSET = 83

WALL_LEFT_X = 34
WALL_RIGHT_X = 123

MAX_SPEED = 1
ACCELERATION = 1
    
PATH_SPRITES = "sprites/spaceinvaders"

# Sizes
PLAYER_SIZE = (7, 10)
WALL_SIZE = (2, 4)
BACKGROUND_SIZE = (WIDTH, 15)
NUMBER_SIZE = (12, 9)
OPPONENT_SIZE = (8, 10)
START_OPPONENT = (23, 31)
OFFSET_OPPONENT = (8, 8)

PLAYER_Y = HEIGHT - PLAYER_SIZE[1] - BACKGROUND_SIZE[1]

def get_human_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_LEFT]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    return jnp.array(Action.NOOP)

class SpaceInvadersState(NamedTuple):
    player_x: chex.Array
    player_speed: chex.Array
    step_counter: chex.Array 
    player_score: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class SpaceInvadersObservation(NamedTuple):
    player: EntityPosition
    score_player: jnp.ndarray

class SpaceInvadersInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


@jax.jit
def player_step(state_player_x, state_player_speed, action: chex.Array):
    left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
    right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
    
    bounds_left = WALL_LEFT_X + PLAYER_SIZE[0]
    bounds_right = WALL_RIGHT_X 

    touches_wall_left = state_player_x <= bounds_left
    touches_wall_right = state_player_x >= bounds_right  
    touches_wall = jnp.logical_or(touches_wall_left, touches_wall_right)

    player_speed = jax.lax.cond(
        jnp.logical_or(jnp.logical_not(jnp.logical_or(left, right)), touches_wall),
        lambda s: 0,
        lambda s: s,
        operand=state_player_speed,
    )

    player_speed = jax.lax.cond(
        right,
        lambda s: jnp.minimum(s + ACCELERATION, MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    player_speed = jax.lax.cond(
        left,
        lambda s: jnp.maximum(s - ACCELERATION, -MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    player_x = jnp.clip(
        state_player_x + player_speed,
        bounds_left,
        bounds_right 
    )
    
    return player_x, player_speed


class JaxSpaceInvaders(JaxEnvironment[SpaceInvadersState, SpaceInvadersObservation, SpaceInvadersInfo]):
    def __init__(self, reward_funcs: list[callable]=None):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ]
        self.obs_size = 3*4+1+1 

    def reset(self, key=None) -> Tuple[SpaceInvadersObservation, SpaceInvadersState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = SpaceInvadersState(
            player_x=jnp.array(96).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SpaceInvadersState, action: chex.Array) -> Tuple[SpaceInvadersObservation, SpaceInvadersState, float, bool, SpaceInvadersInfo]:
        new_player_x, new_player_speed = player_step(
            state.player_x, state.player_speed, action
        )

        new_player_x, new_player_speed = jax.lax.cond(
            state.step_counter % 2 == 0,
            lambda _: (new_player_x, new_player_speed),
            lambda _: (state.player_x, state.player_speed),
            operand=None,
        )

        step_counter = jax.lax.cond(
            state.step_counter > 255,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        new_state = SpaceInvadersState(
            player_x=new_player_x,
            player_speed=new_player_speed,
            step_counter=step_counter,
            player_score=state.player_score,
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SpaceInvadersState):
        player = EntityPosition(
            x=state.player_x,
            y=PLAYER_Y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
        )
        
        return SpaceInvadersObservation(
            player=player,
            score_player=state.player_score,
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: SpaceInvadersState, all_rewards: chex.Array) -> SpaceInvadersInfo:
        return SpaceInvadersInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: SpaceInvadersState, state: SpaceInvadersState):
        return state.player_score - previous_state.player_score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: SpaceInvadersState, state: SpaceInvadersState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: SpaceInvadersState) -> bool:
        return jnp.greater_equal(state.player_score, 20)


def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Player
    player = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "player.npy"), transpose=True)
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)

    # Background
    background = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "background.npy"), transpose=True)
    SPRITE_BACKGROUND = jnp.expand_dims(background, axis=0)

    # Score
    zero_green = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "numbers/zero_green.npy"), transpose=True)
    SPRITE_ZERO_GREEN = jnp.expand_dims(zero_green, axis=0)
    zero_yellow = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "numbers/zero_yellow.npy"), transpose=True)
    SPRITE_ZERO_YELLOW = jnp.expand_dims(zero_yellow, axis=0)

    # Defense
    defense = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "defense.npy"), transpose=True)
    SPRITE_DEFENSE = jnp.expand_dims(defense, axis=0)
    
    # Enemies
    opponent_1 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_1.npy"), transpose=True)
    SPRITE_OPPONENT_1 = jnp.expand_dims(opponent_1, axis=0)
    opponent_2 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_2_a.npy"), transpose=True)
    SPRITE_OPPONENT_2 = jnp.expand_dims(opponent_2, axis=0)
    opponent_3 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_3_a.npy"), transpose=True)
    SPRITE_OPPONENT_3 = jnp.expand_dims(opponent_3, axis=0)
    opponent_4 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_4_a.npy"), transpose=True)
    SPRITE_OPPONENT_4 = jnp.expand_dims(opponent_4, axis=0)
    opponent_5 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_5.npy"), transpose=True)
    SPRITE_OPPONENT_5 = jnp.expand_dims(opponent_5, axis=0)
    opponent_6 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_6.npy"), transpose=True)
    SPRITE_OPPONENT_6 = jnp.expand_dims(opponent_6, axis=0)

    return (SPRITE_BACKGROUND, SPRITE_PLAYER, SPRITE_ZERO_GREEN, SPRITE_ZERO_YELLOW, SPRITE_DEFENSE, SPRITE_OPPONENT_1, SPRITE_OPPONENT_2, SPRITE_OPPONENT_3, SPRITE_OPPONENT_4, SPRITE_OPPONENT_5, SPRITE_OPPONENT_6)

(
    SPRITE_BACKGROUND, 
    SPRITE_PLAYER,
    SPRITE_ZERO_GREEN,
    SPRITE_ZERO_YELLOW,
    SPRITE_DEFENSE,
    SPRITE_OPPONENT_1,
    SPRITE_OPPONENT_2,
    SPRITE_OPPONENT_3,
    SPRITE_OPPONENT_4,
    SPRITE_OPPONENT_5,
    SPRITE_OPPONENT_6
) = load_sprites()

class SpaceInvadersRenderer(AtraJaxisRenderer):
    """
    Renderer for the Space Invaders environment.
    """

    def __init__(self):
        self.SPRITE_PLAYER = SPRITE_PLAYER

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SpaceInvadersState) -> chex.Array:
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Load Background
        background = aj.get_sprite_frame(SPRITE_BACKGROUND, 0)
        raster = aj.render_at(raster, 0, HEIGHT - BACKGROUND_SIZE[1], background)

        # Load Player
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = aj.render_at(raster, state.player_x - PLAYER_SIZE[0], PLAYER_Y, frame_player)

        # Load Initial Score
        # Green Numbers
        frame_zero_green = aj.get_sprite_frame(SPRITE_ZERO_GREEN, 0)
        raster = aj.render_at(raster, 3, 9, frame_zero_green)
        raster = aj.render_at(raster, 6 + NUMBER_SIZE[0], 10, frame_zero_green)
        raster = aj.render_at(raster, 9 + 2 * NUMBER_SIZE[0], 10, frame_zero_green)
        raster = aj.render_at(raster, 12 + 3 * NUMBER_SIZE[0], 9, frame_zero_green)

        # Yellow Numbers
        frame_zero_yellow = aj.get_sprite_frame(SPRITE_ZERO_YELLOW, 0)
        raster = aj.render_at(raster, NUMBER_YELLOW_OFFSET, 9, frame_zero_yellow)
        raster = aj.render_at(raster, NUMBER_YELLOW_OFFSET + 3 + NUMBER_SIZE[0], 10, frame_zero_yellow)
        raster = aj.render_at(raster, NUMBER_YELLOW_OFFSET + 6 + 2 * NUMBER_SIZE[0], 10, frame_zero_yellow)
        raster = aj.render_at(raster, NUMBER_YELLOW_OFFSET + 9 + 3 * NUMBER_SIZE[0], 9, frame_zero_yellow)

        # Load Defense Object
        frame_defense = aj.get_sprite_frame(SPRITE_DEFENSE, 0)
        raster = aj.render_at(raster, 41, HEIGHT - 53, frame_defense)
        raster = aj.render_at(raster, 73, HEIGHT - 53, frame_defense)
        raster = aj.render_at(raster, 105, HEIGHT - 53, frame_defense)

        # Load Opponent Sprites
        frame_opponent_1 = aj.get_sprite_frame(SPRITE_OPPONENT_1, 0)
        frame_opponent_2 = aj.get_sprite_frame(SPRITE_OPPONENT_2, 0)
        frame_opponent_3 = aj.get_sprite_frame(SPRITE_OPPONENT_3, 0)
        frame_opponent_4 = aj.get_sprite_frame(SPRITE_OPPONENT_4, 0)
        frame_opponent_5 = aj.get_sprite_frame(SPRITE_OPPONENT_5, 0)
        frame_opponent_6 = aj.get_sprite_frame(SPRITE_OPPONENT_6, 0)

        def body(i, raster):
            raster = aj.render_at(raster, START_OPPONENT[0] + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0]), START_OPPONENT[1], frame_opponent_1)
            raster = aj.render_at(raster, START_OPPONENT[0] + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0]), START_OPPONENT[1] + (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), frame_opponent_2)
            raster = aj.render_at(raster, START_OPPONENT[0] + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0]), START_OPPONENT[1] + 2 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), frame_opponent_3)
            raster = aj.render_at(raster, START_OPPONENT[0] + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0]), START_OPPONENT[1] + 3 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), frame_opponent_4)
            raster = aj.render_at(raster, START_OPPONENT[0] + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0]), START_OPPONENT[1] + 4 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), frame_opponent_5)
            raster = aj.render_at(raster, START_OPPONENT[0] + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0]), START_OPPONENT[1] + 5 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), frame_opponent_6)

            return raster
        
        raster = jax.lax.fori_loop(0, 6, body, raster)

        return raster 



if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SpaceInvaders Game")
    clock = pygame.time.Clock()

    game = JaxSpaceInvaders()

    # Create the JAX renderer
    renderer = SpaceInvadersRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    obs, curr_state = jitted_reset()

    # Game loop
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
                        obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)

        counter += 1
        clock.tick(FPS)

    pygame.quit()