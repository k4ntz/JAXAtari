import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


WIDTH = 160
HEIGHT = 210

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Donkey Kong position
DONKEYKONG_X = 33
DONKEYKONG_Y = 14

# Girlfriend position
GIRLFRIEND_X = 62
GIRLFRIEND_Y = 17

# Life Bar positions
LEVEL_1_LIFE_BAR_1_X = 116
LEVEL_1_LIFE_BAR_2_X = 124
LEVEL_2_LIFE_BAR_1_X = 112
LEVEL_2_LIFE_BAR_2_X = 120
LIFE_BAR_Y = 23

# Hammer default position
LEVEL_1_HAMMER_X = 39
LEVEL_1_HAMMER_Y = 68
LEVEL_2_HAMMER_X = 78
LEVEL_2_HAMMER_Y = 68

# Drop Pit position
DP_LEFT_X = 52
DP_RIGHT_X = 104
DP_FLOOR_2_Y = 144
DP_FLOOR_3_Y = 116
DP_FLOOR_4_Y = 88
DP_FLOOR_5_Y = 60

# Digits position
DIGIT_Y = 7
FIRST_DIGIT_X = 96
DISTANCE_DIGIT_X = 8
NUMBER_OF_DIGITS_FOR_GAME_SCORE = 6
NUMBER_OF_DIGITS_FOR_TIMER_SCORE = 4

# Mario start position
MARIO_START_X = 48
MARIO_START_Y = 162

# Barrel Start Position
BARREL_START_X = 34
BARREL_START_Y = 53

# Barrel Sprite Index
BARREL_SPRITE_FALL = 0
BARREL_SPRITE_RIGHT = 1
BARREL_SPRITE_LEFT = 2

# Prob Barrel rolls down a ladder
BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN = 0.6

# Hit Boxes RANGES
MARIO_HIT_BOX_X = 7
MARIO_HIT_BOX_Y = 17
BARREL_HIT_BOX_X = 8
BARREL_HIT_BOX_Y = 8

# Moving Directions
MOVING_UP = 0
MOVING_RIGHT = 1
MOVING_DOWN = 2  # Falling Barrel
MOVING_LEFT = 3

# Bar Start and End Positions
BAR_LEFT_X = 32
BAR_RIGHT_X = 120
BAR_1_LEFT_Y = 185
BAR_1_RIGHT_Y = 185
BAR_2_LEFT_Y = 157
BAR_2_RIGHT_Y = 164
BAR_3_LEFT_Y = 136
BAR_3_RIGHT_Y = 129
BAR_4_LEFT_Y = 101
BAR_4_RIGHT_Y = 108
BAR_5_LEFT_Y = 80
BAR_5_RIGHT_Y = 73
BAR_6_LEFT_Y = 52
BAR_6_RIGHT_Y = 52

# Steps counter until next Barrel will spawn
SPAWN_STEP_COUNTER_BARREL = 236




# Bars as lienar functions
def bar_linear_equation(stage, x):
    x_1 = BAR_LEFT_X
    x_2 = BAR_RIGHT_X

    y_1_values = [BAR_1_LEFT_Y, BAR_2_LEFT_Y, BAR_3_LEFT_Y, BAR_4_LEFT_Y, BAR_5_LEFT_Y, BAR_6_LEFT_Y]
    y_2_values = [BAR_1_RIGHT_Y, BAR_2_RIGHT_Y, BAR_3_RIGHT_Y, BAR_4_RIGHT_Y, BAR_5_RIGHT_Y, BAR_6_RIGHT_Y]

    index = stage - 1
    branches = [lambda _, v=val: jnp.array(v) for val in y_1_values]
    y_1 = jax.lax.switch(index, branches, operand=None)
    branches = [lambda _, v=val: jnp.array(v) for val in y_2_values]
    y_2 = jax.lax.switch(index, branches, operand=None)

    m = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - m * x_1

    y = m * x + b
    return y



def get_human_action() -> chex.Array:
    """
    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP, UP, DOWN).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(Action.LEFTFIRE)
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(Action.RIGHTFIRE)
    elif keys[pygame.K_a]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_s]:
        return jnp.array(Action.DOWN)
    elif keys[pygame.K_w]:
        return jnp.array(Action.UP)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    else:
        return jnp.array(Action.NOOP)

class Ladder(NamedTuple):
    stage: chex.Array
    climbable: chex.Array
    start_x: chex.Array
    start_y: chex.Array
    end_x: chex.Array
    end_y: chex.Array

@jax.jit
def init_ladders_for_level(level: int) -> Ladder:
    # Ladder positions for level 1
    Ladder_level_1 = Ladder(
        stage=jnp.array([5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1], dtype=jnp.int32),
        climbable=jnp.array([False, True, True, True, False, False, True, True, True, True, False, True]),
        start_x=jnp.array([74, 106, 46, 66, 98, 62, 86, 106, 46, 78, 70, 106], dtype=jnp.int32),
        start_y=jnp.array([77, 74, 102, 104, 106, 134, 132, 130, 158, 161, 185, 185], dtype=jnp.int32),
        end_x=jnp.array([74, 106, 46, 66, 98, 62, 86, 106, 46, 78, 70, 106], dtype=jnp.int32),
        end_y=jnp.array([53, 53, 79, 78, 76, 104, 106, 108, 135, 133, 161, 164], dtype=jnp.int32),
    )

    # Ladder positions for level 2
    Ladder_level_2 = Ladder_level_1

    return jax.lax.cond(
        level == 1,
        lambda _: Ladder_level_1,
        lambda _: Ladder_level_2,
        operand=None
    )


class BarrelPosition(NamedTuple):
    barrel_x: chex.Array
    barrel_y: chex.Array
    sprite: chex.Array
    moving_direction: chex.Array
    stage: chex.Array
    reached_the_end: chex.Array


class DonkeyKongState(NamedTuple):
    level: chex.Array
    step_counter: chex.Array
    mario_x: chex.Array
    mario_y: chex.Array
    barrels: BarrelPosition
    ladders: Ladder
    random_key: int
    frames_since_last_barrel_spawn: int


class DonkeyKongObservation(NamedTuple):
    total_score: jnp.ndarray


class DonkeyKongInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


@jax.jit
def barrel_step(state):
    step_counter = state.step_counter
    
    # pick other sprite for animation after 8 frames
    should_pick_next_sprite = step_counter % 8 == 0
    
    new_state = state
    # calculate new position
    def update_single_barrel(x, y, direction, sprite, stage, reached_the_end):
        ladders = state.ladders

        # change sprite animation
        def flip_sprite(sprite):
            return jax.lax.cond(
                sprite == BARREL_SPRITE_RIGHT,
                lambda _: BARREL_SPRITE_LEFT,
                lambda _: BARREL_SPRITE_RIGHT,
                operand=None
            )
        
        sprite = jax.lax.cond(
            jnp.logical_and(should_pick_next_sprite, direction != MOVING_DOWN), 
            lambda _: flip_sprite(sprite),
            lambda _: sprite,
            operand=None
        )

        # change y position if the barrel is still falling
        # if barrel is landed on the down stage, change the moving direction
        def change_y_if_barrel_is_falling(x, y, direction, sprite, stage):
            new_y = y + 2

            bar_y = jnp.round(bar_linear_equation(stage, x)).astype(int)
            new_direction = jax.lax.cond(
                new_y >= bar_y,
                lambda _: jax.lax.cond(
                    stage % 2 == 0,
                    lambda _: MOVING_RIGHT,
                    lambda _: MOVING_LEFT,
                    operand=None
                ),
                lambda _: direction,
                operand=None
            )
            new_sprite = jax.lax.cond(
                new_y >= bar_y,
                lambda _: BARREL_SPRITE_RIGHT,
                lambda _: sprite,
                operand=None
            )

            return jax.lax.cond(
                direction == MOVING_DOWN,
                lambda _: (x, new_y, new_direction, new_sprite, stage),
                lambda _: (x, y, direction, sprite, stage),
                operand=None
            )
        x, y, direction, sprite, stage = change_y_if_barrel_is_falling(x, y, direction, sprite, stage)

        # change position
        # check if barrel can fall (ladder or end of bar)
        def check_if_barrel_will_fall(x, y, direction, sprite, stage):        
            prob_barrel_rolls_down_a_ladder = BASE_PROBABILITY_BARREL_ROLLING_A_LADDER_DOWN
            
            # check first if barrel is positioned on top of a ladder
            curr_stage = stage - 1
            mask = jnp.logical_and(ladders.stage == curr_stage, ladders.end_x == x)
            barrel_is_on_ladder = jnp.any(mask)
            roll_down_prob = jax.random.bernoulli(state.random_key, prob_barrel_rolls_down_a_ladder)

            new_direction = MOVING_DOWN
            new_sprite = BARREL_SPRITE_FALL
            new_y = y + 1
            new_stage = stage - 1

            # check secondly if barrel is positioned at the end of a bar
            bar_x = jax.lax.cond(
                stage % 2 == 0,
                lambda _: BAR_RIGHT_X,
                lambda _: BAR_LEFT_X,
                operand=None
            )
            new_direction_2 = MOVING_DOWN
            new_stage_2 = stage - 1
            barrel_is_over_the_bar = jax.lax.cond(
                stage % 2 == 0,
                lambda _: jax.lax.cond(
                    x >= BAR_RIGHT_X,
                    lambda _: True,
                    lambda _: False,
                    operand=None
                ),
                lambda _: jax.lax.cond(
                    x <= BAR_LEFT_X,
                    lambda _: True,
                    lambda _: False,
                    operand=None
                ),
                operand=None
            )

            return jax.lax.cond(
                jnp.logical_and(barrel_is_on_ladder, jnp.logical_and(direction != MOVING_DOWN, roll_down_prob)),
                lambda _: (x, new_y, new_direction, new_sprite, new_stage),
                lambda _: jax.lax.cond(
                    barrel_is_over_the_bar,
                    lambda _: (x, y, new_direction_2, sprite, new_stage_2),
                    lambda _: (x, y, direction, sprite, stage),
                    operand=None
                ),
                operand=None
            )
        x, y, direction, sprite, stage = check_if_barrel_will_fall(x, y, direction, sprite, stage)

        # change x (y) positions when barrel is rolling on bar
        def barrel_rolling_on_a_bar(x, y, direction, sprite, stage):
            new_x = jax.lax.cond(
                direction == MOVING_RIGHT,
                lambda _: x + 1,
                lambda _: x - 1,
                operand=None
            )
            new_y = jnp.round(bar_linear_equation(stage, new_x)).astype(int)
            return jax.lax.cond(
                direction != MOVING_DOWN,
                lambda _: (new_x, new_y, direction, sprite, stage),
                lambda _: (x, y, direction, sprite, stage),
                operand=None
            )
        x, y, direction, sprite, stage = barrel_rolling_on_a_bar(x, y, direction, sprite, stage)

        # mark x = y = -1 as a barrel reaches the end
        def mark_barrel_if_cheached_end(x, y, direction, sprite, stage, reached_the_end):
            return jax.lax.cond(
                jnp.logical_and(stage == 1, x <= BAR_LEFT_X),
                lambda _: (-1, -1, direction, sprite, stage, True),
                lambda _: (x, y, direction, sprite, stage, reached_the_end),
                operand=None
            )
        x, y, direction, sprite, stage, reached_the_end = mark_barrel_if_cheached_end(x, y, direction, sprite, stage, reached_the_end)

        return jax.lax.cond(
            reached_the_end == False,
            lambda _: (x, y, direction, sprite, stage, reached_the_end),
            lambda _: (-1, -1, direction, sprite, stage, reached_the_end),
            operand=None
        )
    update_all_barrels = jax.vmap(update_single_barrel)

    barrels = new_state.barrels
    new_barrel_x, new_barrel_y, new_barrel_moving_direction, new_sprite, new_stage, new_reached_the_end = update_all_barrels(
        barrels.barrel_x, barrels.barrel_y, barrels.moving_direction, barrels.sprite, barrels.stage, barrels.reached_the_end
    )
    barrels = barrels._replace(
        barrel_x = new_barrel_x,
        barrel_y = new_barrel_y,
        moving_direction = new_barrel_moving_direction,
        sprite = new_sprite,
        stage=new_stage,
        reached_the_end=new_reached_the_end
    )
    new_state = new_state._replace(
        barrels=barrels
    )

    # new random key
    key, subkey = jax.random.split(state.random_key)
    new_state = new_state._replace(random_key=key)


    # Skip every second frame
    should_move = step_counter % 2 == 0

    # spawn a new barrel if posible
    def spawn_new_barrel(state):
        barrels = state.barrels

        new_barrel_x = BARREL_START_X
        new_barrel_y = BARREL_START_Y
        new_moving_direction = MOVING_RIGHT
        new_stage = 6
        new_sprite = BARREL_SPRITE_RIGHT
        new_reached_the_end = False

        barrel_x = jnp.append(barrels.barrel_x, new_barrel_x)
        barrel_y = jnp.append(barrels.barrel_y, new_barrel_y)
        moving_direction = jnp.append(barrels.moving_direction, new_moving_direction)
        sprite = jnp.append(barrels.sprite, new_sprite)
        stage = jnp.append(barrels.stage, new_stage)
        reached_the_end = jnp.append(barrels.reached_the_end, new_reached_the_end)

        barrels = barrels._replace(
            barrel_x=barrel_x,
            barrel_y=barrel_y,
            moving_direction=moving_direction,
            sprite=sprite,
            stage=stage,
        )

        new_state = state._replace(
            barrels=barrels,
            frames_since_last_barrel_spawn=0,
        )
        

        return state
    new_state = spawn_new_barrel(new_state)

    # return either new position or old position because of frame skip/ step counter
    return jax.lax.cond(
        should_move, lambda _: new_state, lambda _: state, operand=None
    )


class JaxDonkeyKong(JaxEnvironment[DonkeyKongState, DonkeyKongObservation, DonkeyKongInfo]):
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
            Action.UP,
            Action.DOWN,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
        ]
        self.obs_size = 0

    def reset(self, key = [0,0]) -> Tuple[DonkeyKongObservation, DonkeyKongState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        ladders = init_ladders_for_level(level=1)
        state = DonkeyKongState(
            level = 1,
            step_counter=jnp.array(1).astype(jnp.int32),
            frames_since_last_barrel_spawn=jnp.array(0).astype(jnp.int32),
            mario_x=jnp.array([MARIO_START_X]).astype(jnp.int32),
            mario_y=jnp.array([MARIO_START_Y]).astype(jnp.int32),

            barrels = BarrelPosition(
                barrel_x = jnp.array([BARREL_START_X]).astype(jnp.int32),
                barrel_y = jnp.array([BARREL_START_Y]).astype(jnp.int32), 
                sprite = jnp.array([BARREL_SPRITE_RIGHT]).astype(jnp.int32),
                moving_direction = jnp.array([MOVING_RIGHT]).astype(jnp.int32),
                stage = jnp.array([6]).astype(jnp.int32),
                reached_the_end=jnp.array([False]).astype(bool)
            ),

            ladders=ladders,
            random_key = jax.random.PRNGKey(key[0]),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: DonkeyKongState, action: chex.Array) -> Tuple[DonkeyKongObservation, DonkeyKongState, float, bool, DonkeyKongInfo]:
        # First search for colision

        new_state = state

        # If there is no colision: game will continue
        new_state = barrel_step(state)
  

        new_state = new_state._replace(step_counter=new_state.step_counter+1)
        

        observation = self._get_observation(new_state)
        return observation, new_state, None, False, None

    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: DonkeyKongState):
        
        return DonkeyKongObservation(
            total_score = 0,
        )



def load_sprites():
    """Load all sprites required for Pong rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load sprites
    bg_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background_level_1.npy"), transpose=True)
    bg_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong_background_level_2.npy"), transpose=True)

    donkeyKong_pose_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong1.npy"), transpose=True)
    donkeyKong_pose_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/donkeyKong2.npy"), transpose=True)

    girlfriend = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/girlfriend.npy"), transpose=True)

    level_1_life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/level_1_life_bar.npy"), transpose=True)
    level_2_life_bar = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/level_2_life_bar.npy"), transpose=True)

    mario_standing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_right.npy"), transpose=True)
    mario_standing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_standing_left.npy"), transpose=True)
    mario_jumping_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_right.npy"), transpose=True)
    mario_jumping_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_jumping_left.npy"), transpose=True)
    mario_walking_1_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_1_right.npy"), transpose=True)
    mario_walking_1_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_1_left.npy"), transpose=True)
    mario_walking_2_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_2_right.npy"), transpose=True)
    mario_walking_2_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_walking_2_left.npy"), transpose=True)
    mario_climbing_right = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_climbing_right.npy"), transpose=True)
    mario_climbing_left = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/mario_climbing_left.npy"), transpose=True)

    hammer_up_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_up_level_1.npy"), transpose=True)
    hammer_up_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_up_level_2.npy"), transpose=True)
    hammer_down_right_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_right_level_1.npy"), transpose=True)
    hammer_down_left_level_1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_left_level_1.npy"), transpose=True)
    hammer_down_right_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_right_level_2.npy"), transpose=True)
    hammer_down_left_level_2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/hammer_down_left_level_2.npy"), transpose=True)

    ghost = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/ghost.npy"), transpose=True)

    drop_pit = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/donkeyKong/drop_pit.npy"), transpose=True)

    # Convert all sprites to the expected format (add frame dimension)
    SPRITES_BG = jnp.stack([bg_level_1, bg_level_2], axis=0)
    SPRITES_DONKEYKONG = jnp.stack([donkeyKong_pose_1, donkeyKong_pose_2], axis=0)
    SPRITE_GIRLFRIEND = jnp.expand_dims(girlfriend, axis=0)
    SPRITES_LIFEBAR = jnp.stack([level_1_life_bar, level_2_life_bar], axis=0)
    SPRITES_MARIO_STANDING = jnp.stack([mario_standing_right, mario_standing_left], axis=0)
    SPRITES_MARIO_JUMPING = jnp.stack([mario_jumping_right, mario_jumping_left], axis=0)
    SPRITES_MARIO_WALKING_1 = jnp.stack([mario_walking_1_right, mario_walking_1_left], axis=0)
    SPRITES_MARIO_WALKING_2 = jnp.stack([mario_walking_2_right, mario_walking_2_left], axis=0)
    SPRITES_MARIO_CLIMBING = jnp.stack([mario_climbing_right, mario_climbing_left], axis=0)
    SPRITES_HAMMER_UP = jnp.stack([hammer_up_level_1, hammer_up_level_2], axis=0)
    SPRITES_HAMMER_DOWN = jnp.stack([hammer_down_right_level_1, hammer_down_left_level_1, hammer_down_right_level_2, hammer_down_left_level_2], axis=0)
    SPRITE_GHOST = jnp.expand_dims(ghost, axis=0)
    SPRITE_DROP_PIT = jnp.expand_dims(drop_pit, axis=0)

    SPRITES_BARREL = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/donkeyKong/barrel{}.npy"),
        num_chars=3,
    )

    SPRITES_BLUE_DIGITS = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/donkeyKong/digits/blue_score_{}.npy"),
        num_chars=10,
    )
    SPRITES_YELLOW_DIGITS = aj.load_and_pad_digits(
        os.path.join(MODULE_DIR, "sprites/donkeyKong/digits/yellow_score_{}.npy"),
        num_chars=10,
    )

    return (
        SPRITES_BG,
        SPRITES_DONKEYKONG,
        SPRITE_GIRLFRIEND,
        SPRITES_BARREL,
        SPRITES_LIFEBAR,
        SPRITES_MARIO_STANDING,
        SPRITES_MARIO_JUMPING,
        SPRITES_MARIO_WALKING_1,
        SPRITES_MARIO_WALKING_2,
        SPRITES_MARIO_CLIMBING,
        SPRITES_HAMMER_UP,
        SPRITES_HAMMER_DOWN,
        SPRITE_GHOST,
        SPRITE_DROP_PIT,
        SPRITES_BLUE_DIGITS,
        SPRITES_YELLOW_DIGITS,
    )


class DonkeyKongRenderer(AtraJaxisRenderer):
    """JAX-based Pong game renderer, optimized with JIT compilation."""

    def __init__(self):
        (
            self.SPRITES_BG,
            self.SPRITES_DONKEYKONG,
            self.SPRITE_GIRLFRIEND,
            self.SPRITES_BARREL,
            self.SPRITES_LIFEBAR,
            self.SPRITES_MARIO_STANDING,
            self.SPRITES_MARIO_JUMPING,
            self.SPRITES_MARIO_WALKING_1,
            self.SPRITES_MARIO_WALKING_2,
            self.SPRITES_MARIO_CLIMBING,
            self.SPRITES_HAMMER_UP,
            self.SPRITES_HAMMER_DOWN,
            self.SPRITE_GHOST,
            self.SPRITE_DROP_PIT,
            self.SPRITES_BLUE_DIGITS,
            self.SPRITES_YELLOW_DIGITS,
        ) = load_sprites()

    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A DonkeyKongState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """

        def render_at_transparent(raster, x, y, sprite):
            if sprite.shape[-1] > 3:
                sprite = sprite[:, :, :3]
            
            h, w, _ = sprite.shape
            sub_raster = jax.lax.dynamic_slice(raster, (x, y, 0), (h, w, 3))

            # Transparent Pixel = every channel with value 0
            mask = jnp.any(sprite != 0, axis=-1, keepdims=True)
            mask = jnp.broadcast_to(mask, sprite.shape)

            blended = jnp.where(mask, sprite, sub_raster)

            return jax.lax.dynamic_update_slice(raster, blended, (x, y, 0))

        def create_bg_raster_for_level_2_regarding_drop_pits(raster):
            frame_bg = aj.get_sprite_frame(self.SPRITES_BG, 1)
            raster = aj.render_at(raster, 0, 0, frame_bg)
            frame_drop_pit = aj.get_sprite_frame(self.SPRITE_DROP_PIT, 0)

            # some drop pits might be already triggered - in that case, drop pits at those position will not be rendered
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_2_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_3_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_4_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_LEFT_X, DP_FLOOR_5_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_2_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_3_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_4_Y, frame_drop_pit)
            raster = aj.render_at(raster, DP_RIGHT_X, DP_FLOOR_5_Y, frame_drop_pit)
            return raster            

        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Background raster
        level = state.level
        raster = jax.lax.cond(
            level == 1,
            lambda _: aj.render_at(raster, 0, 0, aj.get_sprite_frame(self.SPRITES_BG, 0)),
            lambda x: create_bg_raster_for_level_2_regarding_drop_pits(x),
            raster 
        )

        # DonkeyKong
        frame_donkeyKong = aj.get_sprite_frame(self.SPRITES_DONKEYKONG, 0)
        raster = aj.render_at(raster, DONKEYKONG_X, DONKEYKONG_Y, frame_donkeyKong)

        # Girlfriend
        frame_girlfriend = aj.get_sprite_frame(self.SPRITE_GIRLFRIEND, 0)
        raster = aj.render_at(raster, GIRLFRIEND_X, GIRLFRIEND_Y, frame_girlfriend)

        # Life Bars - depending if lifes are still given 
        frame_life_bar = aj.get_sprite_frame(self.SPRITES_LIFEBAR, 0)
        raster = aj.render_at(raster, LEVEL_1_LIFE_BAR_1_X, LIFE_BAR_Y, frame_life_bar)
        raster = aj.render_at(raster, LEVEL_1_LIFE_BAR_2_X, LIFE_BAR_Y, frame_life_bar)

        # Mario
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 0)
        raster = aj.render_at(raster, state.mario_x, state.mario_y, frame_mario)

        # Barrels if there are some on the field
        barrels = state.barrels
        for barrel_x, barrel_y, sprite_id, reached_the_end in zip(barrels.barrel_x, barrels.barrel_y, barrels.sprite, barrels.reached_the_end):
            frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, sprite_id)
            raster = jax.lax.cond(
                reached_the_end,
                lambda _: raster,
                lambda _: render_at_transparent(raster, barrel_x, barrel_y, frame_barrel),
                operand=None
            )

        # Hammer
        frame_hammer = aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 0)
        raster = aj.render_at(raster, LEVEL_1_HAMMER_X, LEVEL_1_HAMMER_Y, frame_hammer)


        # Scores
        score = 5000
        show_game_score = False
        def create_score_in_raster(i, raster):
            digit = score // (10 ** i)
            pos_x = FIRST_DIGIT_X - DISTANCE_DIGIT_X * i
            pos_y = DIGIT_Y
            return jax.lax.cond(
                show_game_score == True,
                lambda _: aj.render_at(raster, pos_x, pos_y, aj.get_sprite_frame(self.SPRITES_BLUE_DIGITS, digit)),
                lambda _: aj.render_at(raster, pos_x, pos_y, aj.get_sprite_frame(self.SPRITES_YELLOW_DIGITS, digit)),
                operand=None
            )
        raster = jax.lax.cond(
            show_game_score == True,
            lambda x: jax.lax.fori_loop(0, NUMBER_OF_DIGITS_FOR_GAME_SCORE, create_score_in_raster, x),
            lambda x: jax.lax.fori_loop(0, NUMBER_OF_DIGITS_FOR_TIMER_SCORE, create_score_in_raster, x),
            raster
        )

        # Barrels - example for now
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 0)
        raster = aj.render_at(raster, 5, 5, frame_barrel)
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 1)
        raster = aj.render_at(raster, 5, 15, frame_barrel)
        frame_barrel = aj.get_sprite_frame(self.SPRITES_BARREL, 2)
        raster = aj.render_at(raster, 5, 25, frame_barrel)

        # Mario - example raster on the left of the screen
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 0)
        raster = aj.render_at(raster, 5, 35, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_STANDING, 1)
        raster = aj.render_at(raster, 15, 35, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_JUMPING, 0)
        raster = aj.render_at(raster, 5, 55, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_JUMPING, 1)
        raster = aj.render_at(raster, 15, 55, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_1, 0)
        raster = aj.render_at(raster, 5, 75, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_1, 1)
        raster = aj.render_at(raster, 15, 75, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_2, 0)
        raster = aj.render_at(raster, 5, 95, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_WALKING_2, 1)
        raster = aj.render_at(raster, 15, 95, frame_mario)

        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_CLIMBING, 0)
        raster = aj.render_at(raster, 5, 115, frame_mario)
        frame_mario = aj.get_sprite_frame(self.SPRITES_MARIO_CLIMBING, 1)
        raster = aj.render_at(raster, 15, 115, frame_mario)

        # Hammer down examples
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 0)
        raster = aj.render_at(raster, 5, 135, frame_hammer_down)
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 1)
        raster = aj.render_at(raster, 15, 135, frame_hammer_down)
        frame_hammer_up_level_2 = aj.get_sprite_frame(self.SPRITES_HAMMER_UP, 1)
        raster = aj.render_at(raster, 5, 145, frame_hammer_up_level_2)
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 2)
        raster = aj.render_at(raster, 5, 155, frame_hammer_down)
        frame_hammer_down = aj.get_sprite_frame(self.SPRITES_HAMMER_DOWN, 3)
        raster = aj.render_at(raster, 15, 155, frame_hammer_down)

        # Ghost
        frame_ghost = aj.get_sprite_frame(self.SPRITE_GHOST, 0)
        raster = aj.render_at(raster, 5, 165, frame_ghost)

        return raster


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Donkey Kong Game")
    clock = pygame.time.Clock()

    game = JaxDonkeyKong()

    # Create the JAX renderer
    renderer = DonkeyKongRenderer()

    # Get jitted functions
    jitted_reset = jax.jit(game.reset)
    jitted_step = jax.jit(game.step)

    obs, curr_state = jitted_reset()

    # Game Loop
    running = True
    frame_by_frame = False
    while running:

        if not frame_by_frame:
            action = None
            obs, curr_state, reward, done, info = jitted_step(curr_state, action)

        # Render and Display
        raster = renderer.render(state=curr_state)
        aj.update_pygame(screen, raster, 3, WIDTH, HEIGHT)
        clock.tick(30)

    pygame.quit()