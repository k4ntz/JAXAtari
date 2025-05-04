
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

WIDTH = 160
HEIGHT = 210

NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5

SPEED = 2
MOTHERSHIP_Y = 32
PLAYER_Y = 192
MAX_HEAT = 100
MAX_LIVES = 3

ENEMY_Y_POSITIONS = (64, 96, 128)

PLAYER_SIZE = (4, 16)
ENEMY_SIZE = (4, 16)
MOTHERSHIP_SIZE = (32, 16)

WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

STATE_TRANSLATOR: dict = {
    0: "player_x",
    1: "player_speed",
    2: "enemy_projectile_x",
    3: "enemy_projectile_y",
    4: "enemy_projectile_dir",
    5: "mothership_x",
    6: "enemy_1_x",
    7: "enemy_1_y",
    8: "enemy_1_speed",
    9: "enemy_2_x",
    10: "enemy_2_y",
    11: "enemy_2_speed",
    12: "enemy_3_x",
    13: "enemy_3_y",
    14: "enemy_3_speed",
    15: "enemy_4_x",
    16: "enemy_4_y",
    17: "enemy_4_speed",
    18: "enemy_5_x",
    19: "enemy_5_y",
    20: "enemy_5_speed",
    21: "enemy_6_x",
    22: "enemy_6_y",
    23: "enemy_6_speed",
    24: "player_projectile_x",
    25: "player_projectile_y",
    26: "player_projectile_dir",
    27: "score",
    28: "player_lives",
    29: "bottom_health",
    30: "stage",
    31: "buffer",
}

def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    elif keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    invisible: jnp.ndarray

# immutable state container
class AssaultState(NamedTuple):
    player_x: chex.Array
    player_speed: chex.Array
    enemy_projectile_x: chex.Array
    enemy_projectile_y: chex.Array
    enemy_projectile_dir: chex.Array
    mothership_x: chex.Array
    mothership_dir: chex.Array
    enemy_1_x: chex.Array
    enemy_1_y: chex.Array
    enemy_1_dir: chex.Array
    enemy_2_x: chex.Array
    enemy_2_y: chex.Array
    enemy_2_dir: chex.Array
    enemy_3_x: chex.Array
    enemy_3_y: chex.Array
    enemy_3_dir: chex.Array
    enemy_4_x: chex.Array
    enemy_4_y: chex.Array
    enemy_4_dir: chex.Array
    enemy_5_x: chex.Array
    enemy_5_y: chex.Array
    enemy_5_dir: chex.Array
    enemy_6_x: chex.Array
    enemy_6_y: chex.Array
    enemy_6_dir: chex.Array
    player_projectile_x: chex.Array
    player_projectile_y: chex.Array
    player_projectile_dir: chex.Array
    score: chex.Array
    player_lives: chex.Array
    heat: chex.Array
    stage: chex.Array
    buffer: chex.Array
    obs_stack: chex.ArrayTree


class AssaultObservation(NamedTuple):
    player: EntityPosition
    mothership: EntityPosition
    enemy_1: EntityPosition
    enemy_2: EntityPosition
    enemy_3: EntityPosition
    enemy_4: EntityPosition
    enemy_5: EntityPosition
    enemy_6: EntityPosition
    enemy_projectile: EntityPosition
    lives: jnp.ndarray
    score: jnp.ndarray
    


class AssaultInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

@jax.jit
def player_step(
    state_player_x, action: chex.Array
):
    # Minimal: move left/right, clamp to screen
    move_left = jnp.logical_or(action == LEFT, action == LEFTFIRE)
    move_right = jnp.logical_or(action == RIGHT, action == RIGHTFIRE)
    speed = jnp.where(move_left, -SPEED, jnp.where(move_right, SPEED, 0))
    new_x = jnp.clip(state_player_x + speed, 0, 160 - PLAYER_SIZE[0])
    return new_x, speed

@jax.jit
def enemy_step(state):
    def move_enemy(x, dir):
        # If at left border, go right; if at right border, go left
        at_left = jnp.greater_equal(0, x)
        at_right = jnp.greater_equal(x, 160 - ENEMY_SIZE[0])
        new_dir = jnp.where(at_left, 1, jnp.where(at_right, -1, dir))
        new_x = jnp.clip(x + new_dir * SPEED, 0, 160 - ENEMY_SIZE[0])
        return new_x, new_dir

    e1_x, e1_dir = move_enemy(state.enemy_1_x, state.enemy_1_dir)
    e2_x, e2_dir = move_enemy(state.enemy_2_x, state.enemy_2_dir)
    e3_x, e3_dir = move_enemy(state.enemy_3_x, state.enemy_3_dir)
    e4_x, e4_dir = move_enemy(state.enemy_4_x, state.enemy_4_dir)
    e5_x, e5_dir = move_enemy(state.enemy_5_x, state.enemy_5_dir)
    e6_x, e6_dir = move_enemy(state.enemy_6_x, state.enemy_6_dir)

    return state._replace(
        enemy_1_x=e1_x, enemy_1_dir=e1_dir,
        enemy_2_x=e2_x, enemy_2_dir=e2_dir,
        enemy_3_x=e3_x, enemy_3_dir=e3_dir,
        enemy_4_x=e4_x, enemy_4_dir=e4_dir,
        enemy_5_x=e5_x, enemy_5_dir=e5_dir,
        enemy_6_x=e6_x, enemy_6_dir=e6_dir,
    )

@jax.jit
def mothership_step(state):
    def move_mothership(x, dir):
        # If at left border, go right; if at right border, go left
        at_left = jnp.greater_equal(0, x)
        at_right = jnp.greater_equal(x, 160 - MOTHERSHIP_SIZE[0])
        new_dir = jnp.where(at_left, 1, jnp.where(at_right, -1, dir))
        new_x = jnp.clip(x + new_dir * SPEED, 0, 160 - MOTHERSHIP_SIZE[0])
        return new_x, new_dir
    # Move mothership left/right, clamp to screen
    mothership_x, mothership_dir = move_mothership(state.mothership_x, state.mothership_dir)
    return state._replace(mothership_x=mothership_x, mothership_dir=mothership_dir)

class JaxAssault(JaxEnvironment[AssaultState, AssaultObservation, AssaultInfo]):

    def __init__(self):
        super().__init__()
        self.frameskip = 1
        self.frame_stack_size = 4
        self.action_set = {NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE}
        self.reward_funcs = None

    def reset(self) -> AssaultState:
        # Minimal state initialization
        state = AssaultState(
            player_x=jnp.array(80).astype(jnp.int32),
            player_speed=jnp.array(0).astype(jnp.int32),
            enemy_projectile_x=jnp.array(0).astype(jnp.int32),
            enemy_projectile_y=jnp.array(0).astype(jnp.int32),
            enemy_projectile_dir=jnp.array(0).astype(jnp.int32),
            mothership_x=jnp.array(64).astype(jnp.int32),
            mothership_dir=jnp.array(1).astype(jnp.int32),
            enemy_1_x=jnp.array(20).astype(jnp.int32),
            enemy_1_y=jnp.array(ENEMY_Y_POSITIONS[0]).astype(jnp.int32),
            enemy_1_dir=jnp.array(1).astype(jnp.int32),
            enemy_2_x=jnp.array(40).astype(jnp.int32),
            enemy_2_y=jnp.array(ENEMY_Y_POSITIONS[0]).astype(jnp.int32),
            enemy_2_dir=jnp.array(1).astype(jnp.int32),
            enemy_3_x=jnp.array(60).astype(jnp.int32),
            enemy_3_y=jnp.array(ENEMY_Y_POSITIONS[1]).astype(jnp.int32),
            enemy_3_dir=jnp.array(1).astype(jnp.int32),
            enemy_4_x=jnp.array(80).astype(jnp.int32),
            enemy_4_y=jnp.array(ENEMY_Y_POSITIONS[1]).astype(jnp.int32),
            enemy_4_dir=jnp.array(1).astype(jnp.int32),
            enemy_5_x=jnp.array(100).astype(jnp.int32),
            enemy_5_y=jnp.array(ENEMY_Y_POSITIONS[2]).astype(jnp.int32),
            enemy_5_dir=jnp.array(1).astype(jnp.int32),
            enemy_6_x=jnp.array(120).astype(jnp.int32),
            enemy_6_y=jnp.array(ENEMY_Y_POSITIONS[2]).astype(jnp.int32),
            enemy_6_dir=jnp.array(1).astype(jnp.int32),
            player_projectile_x=jnp.array(-1).astype(jnp.int32),
            player_projectile_y=jnp.array(-1).astype(jnp.int32),
            player_projectile_dir=jnp.array(0).astype(jnp.int32),
            score=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(MAX_LIVES).astype(jnp.int32),
            heat=jnp.array(0).astype(jnp.int32),
            stage=jnp.array(1).astype(jnp.int32),
            buffer=jnp.array(0).astype(jnp.int32),
            obs_stack=None,
        )
        obs = self._get_observation(state)
        def expand_and_copy(x):
            x_expanded = jnp.expand_dims(x, axis=0)
            return jnp.concatenate([x_expanded] * self.frame_stack_size, axis=0)
        obs_stack = jax.tree.map(expand_and_copy, obs)
        state = state._replace(obs_stack=obs_stack)
        return state, obs_stack

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AssaultState, action: chex.Array) -> Tuple[AssaultState, AssaultObservation, float, bool, AssaultInfo]:
        # Player step
        new_player_x, new_player_speed = player_step(state.player_x, action)
        # Enemy step (stub)

        new_state = state._replace(
            player_x=new_player_x,
            player_speed=new_player_speed,
            # TODO: update enemies, projectiles, collisions, score, etc.
        )
        new_state = enemy_step(new_state)
        new_state = mothership_step(new_state)

        # Reward: +1 if score increased, -1 if lost life
        reward = jnp.where(new_state.score > state.score, 1.0, 0.0)
        reward = jnp.where(new_state.player_lives < state.player_lives, -1.0, reward)
        done = jnp.greater_equal(0,new_state.player_lives)
        obs = self._get_observation(new_state)
        obs_stack = jax.tree.map(lambda stack, o: jnp.concatenate([stack[1:], jnp.expand_dims(o, axis=0)], axis=0), state.obs_stack, obs)
        new_state = new_state._replace(obs_stack=obs_stack)
        info = AssaultInfo(time=jnp.array(0), all_rewards=jnp.zeros(1))
        return new_state, obs_stack, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: AssaultState):
        # Build observation from state
        player = EntityPosition(
            x=state.player_x,
            y=jnp.array(PLAYER_Y),
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
            invisible=jnp.array(0),
        )
        mothership = EntityPosition(
            x=state.mothership_x,
            y=jnp.array(MOTHERSHIP_Y),
            width=jnp.array(MOTHERSHIP_SIZE[0]),
            height=jnp.array(MOTHERSHIP_SIZE[1]),
            invisible=jnp.array(0),
        )
        def enemy_entity(x, y):
            return EntityPosition(
                x=x, y=y,
                width=jnp.array(ENEMY_SIZE[0]),
                height=jnp.array(ENEMY_SIZE[1]),
                invisible=jnp.array(0),
            )
        enemy_projectile=EntityPosition(
                x=state.enemy_projectile_x,
                y=state.enemy_projectile_y,
                width=jnp.array(2),
                height=jnp.array(4),
                invisible=jnp.array(0),
            )
        return AssaultObservation(
            player=player,
            mothership=mothership,
            enemy_1=enemy_entity(state.enemy_1_x, state.enemy_1_y),
            enemy_2=enemy_entity(state.enemy_2_x, state.enemy_2_y),
            enemy_3=enemy_entity(state.enemy_3_x, state.enemy_3_y),
            enemy_4=enemy_entity(state.enemy_4_x, state.enemy_4_y),
            enemy_5=enemy_entity(state.enemy_5_x, state.enemy_5_y),
            enemy_6=enemy_entity(state.enemy_6_x, state.enemy_6_y),
            enemy_projectile=enemy_projectile,
            lives=state.player_lives,
            score=state.score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AssaultObservation) -> jnp.ndarray:
        # Flatten all positions and stats into a 1D array
        return jnp.concatenate([
            obs.player.x.flatten(), obs.player.y.flatten(),
            obs.player.width.flatten(), obs.player.height.flatten(),
            obs.mothership.x.flatten(), obs.mothership.y.flatten(),
            obs.enemy_1.x.flatten(), obs.enemy_1.y.flatten(),
            obs.enemy_2.x.flatten(), obs.enemy_2.y.flatten(),
            obs.enemy_3.x.flatten(), obs.enemy_3.y.flatten(),
            obs.enemy_4.x.flatten(), obs.enemy_4.y.flatten(),
            obs.enemy_5.x.flatten(), obs.enemy_5.y.flatten(),
            obs.enemy_6.x.flatten(), obs.enemy_6.y.flatten(),
            obs.enemy_projectile.x.flatten(), obs.enemy_projectile.y.flatten(),
            obs.lives.flatten(), obs.score.flatten()
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AssaultState, all_rewards: chex.Array) -> AssaultInfo:
        return AssaultInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: AssaultState, state: AssaultState):
        return (state.player_score - state.enemy_score) - (
            previous_state.player_score - previous_state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: AssaultState, state: AssaultState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards 

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AssaultState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.player_score, 20),
            jnp.greater_equal(state.enemy_score, 20),
        )
    


def load_assault_sprites():
    """
    Load all sprites required for Assault rendering.
    Assumes files are named enemy.npy, life.npy, mothership.npy, player.npy
    and are located in sprites/assault relative to this file.
    """
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    SPRITES_DIR = os.path.join(MODULE_DIR, "sprites", "assault")

    background = aj.loadFrame(os.path.join(SPRITES_DIR, "background.npy"), transpose=True)
    enemy = aj.loadFrame(os.path.join(SPRITES_DIR, "enemy_0.npy"), transpose=True)
    #life = aj.loadFrame(os.path.join(SPRITES_DIR, "life.npy"), transpose=True)
    mothership = aj.loadFrame(os.path.join(SPRITES_DIR, "mothership_0.npy"), transpose=True)
    player = aj.loadFrame(os.path.join(SPRITES_DIR, "player.npy"), transpose=True)
    player_projectile = aj.loadFrame(os.path.join(SPRITES_DIR, "player_projectile.npy"), transpose=True)
    enemy_projectile = aj.loadFrame(os.path.join(SPRITES_DIR, "enemy_projectile.npy"), transpose=True)

    # Optionally expand dims if you want a batch/frame dimension
    BACKGROUND_SPRITE = jnp.expand_dims(background, axis=0)
    ENEMY_SPRITE = jnp.expand_dims(enemy, axis=0)
    #LIFE_SPRITE = jnp.expand_dims(life, axis=0)
    MOTHERSHIP_SPRITE = jnp.expand_dims(mothership, axis=0)
    PLAYER_SPRITE = jnp.expand_dims(player, axis=0)
    PLAYER_PROJECTILE= jnp.expand_dims(player_projectile, axis=0)
    ENEMY_PROJECTILE = jnp.expand_dims(enemy_projectile, axis=0)

    DIGIT_SPRITES = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, os.path.join(SPRITES_DIR, "number_{}.npy")),
        num_chars=10,
    )

    return BACKGROUND_SPRITE,ENEMY_SPRITE, MOTHERSHIP_SPRITE, PLAYER_SPRITE, DIGIT_SPRITES, PLAYER_PROJECTILE,ENEMY_PROJECTILE

class Renderer_AtraJaxisAssault:
    """JAX-based Assault game renderer, optimized with JIT compilation."""

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_ENEMY,
            self.SPRITE_MOTHERSHIP,
            self.SPRITE_PLAYER,
            self.DIGIT_SPRITES,
            self.PLAYER_PROJECTILE,
            self.ENEMY_PROJECTILE
        ) = load_assault_sprites()  # You need to implement this in atraJaxis

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current Assault game state using JAX operations.

        Args:
            state: An AssaultState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        raster = jnp.zeros((WIDTH, HEIGHT, 3), dtype=jnp.uint8)

        # Render background
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render mothership
        frame_mothership = aj.get_sprite_frame(self.SPRITE_MOTHERSHIP, 0)
        raster = aj.render_at(raster, MOTHERSHIP_Y, state.mothership_x, frame_mothership)

        # Render player
        frame_player = aj.get_sprite_frame(self.SPRITE_PLAYER, 0)
        raster = aj.render_at(raster, PLAYER_Y, state.player_x, frame_player)

        # Render enemies (unrolled manually for JIT compatibility)
        frame_enemy = aj.get_sprite_frame(self.SPRITE_ENEMY, 0)
        raster = aj.render_at(raster, state.enemy_1_y, state.enemy_1_x, frame_enemy)
        raster = aj.render_at(raster, state.enemy_2_y, state.enemy_2_x, frame_enemy)
        raster = aj.render_at(raster, state.enemy_3_y, state.enemy_3_x, frame_enemy)
        raster = aj.render_at(raster, state.enemy_4_y, state.enemy_4_x, frame_enemy)
        raster = aj.render_at(raster, state.enemy_5_y, state.enemy_5_x, frame_enemy)
        raster = aj.render_at(raster, state.enemy_6_y, state.enemy_6_x, frame_enemy)
         # Render player projectile using lax.cond
        def render_player_proj(_):
            frame_proj = aj.get_sprite_frame(self.PLAYER_PROJECTILE, 0)
            return aj.render_at(raster, state.player_projectile_y, state.player_projectile_x, frame_proj)

        def skip_player_proj(_):
            return raster

        raster = jax.lax.cond(
            jnp.greater_equal(state.player_projectile_y, 0),
            render_player_proj,
            skip_player_proj,
            operand=None
        )

        # Render enemy projectile using lax.cond
        def render_enemy_proj(_):
            frame_proj = aj.get_sprite_frame(self.ENEMY_PROJECTILE, 0)
            return aj.render_at(raster, state.enemy_projectile_y, state.enemy_projectile_x, frame_proj)

        def skip_enemy_proj(_):
            return raster

        raster = jax.lax.cond(
            jnp.greater_equal(state.enemy_projectile_y, 0),
            render_enemy_proj,
            skip_enemy_proj,
            operand=None
        )

        # Render score (top left)
        score_digits = aj.int_to_digits(state.score, max_digits=4)
        raster = aj.render_label_selective(
            raster, 5, 5, score_digits, self.DIGIT_SPRITES, 0, len(score_digits), spacing=12
        )

        # Render lives (top right)
        lives_digits = aj.int_to_digits(state.player_lives, max_digits=1)
        raster = aj.render_label_selective(
            raster, 5, WIDTH - 20, lives_digits, self.DIGIT_SPRITES, 0, len(lives_digits), spacing=12
        )

        return raster
    
if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Assault Game")
    clock = pygame.time.Clock()

    game = JaxAssault()

    # Create the JAX renderer
    renderer = Renderer_AtraJaxisAssault()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state, obs = jitted_reset()

    # Game loop
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
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
                        curr_state, obs, reward, done, info = jitted_step(
                            curr_state, action
                        )

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_state, obs, reward, done, info = jitted_step(curr_state, action)

        # Render and display
        raster = renderer.render(curr_state)

        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)

        counter += 1
        clock.tick(60)

    pygame.quit()