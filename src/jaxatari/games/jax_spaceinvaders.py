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

NUMBER_YELLOW_OFFSET = 83

WALL_LEFT_X = 34
WALL_RIGHT_X = 123

# Borders for opponent movement
OPPONENT_LIMIT_X = (22, 136)
OPPONENT_LIMIT_Y = (31, None)

MAX_SPEED = 1
ACCELERATION = 1
BULLET_SPEED = 1     
ENEMY_BULLET_SPEED = 1

PATH_SPRITES = "sprites/spaceinvaders"
ENEMY_ROWS = 6
ENEMY_COLS = 6
MAX_ENEMY_BULLETS = 3

# Rate of Opponent Movement and Animation
MOVEMENT_RATE = 32
ENEMY_FIRE_RATE = 60

# Sizes
PLAYER_SIZE = (7, 10)
BULLET_SIZE = (1, 10)
WALL_SIZE = (2, 4)
BACKGROUND_SIZE = (WIDTH, 15)
NUMBER_SIZE = (12, 9)
OPPONENT_SIZE = (8, 10)
OFFSET_OPPONENT = (8, 8)

PLAYER_Y = HEIGHT - PLAYER_SIZE[1] - BACKGROUND_SIZE[1]

def get_human_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(Action.RIGHTFIRE)
    elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(Action.LEFTFIRE)
    elif keys[pygame.K_RIGHT]:
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
    player_lives: chex.Array
    # Holds a list of destroyed opponents by their index (row, column)
    destroyed: chex.Array
    opponent_current_x: int
    opponent_current_y: int
    # Defines a bounding rect around the visible opponents
    opponent_bounding_rect: NamedTuple
    opponent_direction: int
    # Player bullet (there can only be one at a time)
    bullet_active: chex.Array   
    bullet_x: chex.Array
    bullet_y: chex.Array
    # Enemy bullets
    enemy_bullets_active: chex.Array
    enemy_bullets_x: chex.Array
    enemy_bullets_y: chex.Array
    enemy_fire_cooldown: chex.Array

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
def get_enemy_position(opponent_current_x, opponent_current_y, row, col):
    x = opponent_current_x + col * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0])
    y = opponent_current_y + row * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1])
    return x, y

@jax.jit
def check_collision(bullet_x, bullet_y, target_x, target_y, target_width, target_height):
    bullet_right = bullet_x + BULLET_SIZE[0]
    bullet_bottom = bullet_y + BULLET_SIZE[1]
    target_right = target_x + target_width
    target_bottom = target_y + target_height
    
    collision = (bullet_x < target_right) & (bullet_right > target_x) & (bullet_y < target_bottom) & (bullet_bottom > target_y)
    return collision

@jax.jit
def check_bullet_enemy_collisions(state: SpaceInvadersState):
    def check_loop(carry):
        def body(i, carry):
            destroyed, score, bullet_active = carry

            row = i // ENEMY_COLS
            col = i % ENEMY_COLS
            idx = row * ENEMY_COLS + col

            enemy_x, enemy_y = get_enemy_position(
                state.opponent_current_x,
                state.opponent_current_y,
                row,
                col
            )

            enemy_alive = jnp.logical_not(destroyed[idx])
            collision = jnp.logical_and(
                jnp.logical_and(enemy_alive, bullet_active),
                check_collision(state.bullet_x, state.bullet_y, enemy_x, enemy_y, OPPONENT_SIZE[0], OPPONENT_SIZE[1])
            )

            destroyed = destroyed.at[idx].set(jnp.logical_or(destroyed[idx], collision))
            score += jnp.where(collision, 10, 0) # +10 score 
            bullet_active = jnp.where(collision, False, bullet_active)

            return destroyed, score, bullet_active

        return jax.lax.fori_loop(0, ENEMY_ROWS * ENEMY_COLS, body, carry)

    init = (state.destroyed, state.player_score, state.bullet_active)

    return jax.lax.cond(
        state.bullet_active,
        check_loop,
        lambda carry: carry,
        init
    )


@jax.jit
def check_enemy_bullet_player_collisions(state: SpaceInvadersState):
    def check_bullet(i, carry):
        lives, enemy_bullets_active = carry
        
        bullet_active = enemy_bullets_active[i]
        collision = jnp.logical_and(
            bullet_active,
            check_collision(
                state.enemy_bullets_x[i], 
                state.enemy_bullets_y[i], 
                state.player_x - PLAYER_SIZE[0], 
                PLAYER_Y, 
                PLAYER_SIZE[0], 
                PLAYER_SIZE[1]
            )
        )
        
        # If collision, deactivate bullet and reduce lives
        new_bullet_active = jnp.where(collision, False, bullet_active)
        enemy_bullets_active = enemy_bullets_active.at[i].set(new_bullet_active)
        lives = jnp.where(collision, lives - 1, lives)
        
        return lives, enemy_bullets_active
    
    init = (state.player_lives, state.enemy_bullets_active)
    return jax.lax.fori_loop(0, MAX_ENEMY_BULLETS, check_bullet, init)

@jax.jit
def get_bottom_enemies(state: SpaceInvadersState):
    def check_column(col):
        def check_row(row):
            idx = row * ENEMY_COLS + col
            return jnp.logical_not(state.destroyed[idx])
        
        # check from bottom to top
        has_enemy = jnp.array([check_row(row) for row in range(ENEMY_ROWS-1, -1, -1)])
        # find first alive enemy from bottom
        bottom_row = jnp.where(has_enemy, jnp.arange(ENEMY_ROWS-1, -1, -1), -1)
        # get the bottom row with an enemy
        actual_bottom = jnp.max(jnp.where(bottom_row >= 0, bottom_row, -1))
        
        return jnp.where(actual_bottom >= 0, actual_bottom, -1)
    
    return jnp.array([check_column(col) for col in range(ENEMY_COLS)])


@jax.jit
def update_enemy_bullets(state: SpaceInvadersState, key):    
    new_y = state.enemy_bullets_y + ENEMY_BULLET_SPEED
    new_active = jnp.where(new_y >= HEIGHT, False, state.enemy_bullets_active)
    
    new_cooldown = jnp.maximum(0, state.enemy_fire_cooldown - 1)
    should_fire = (new_cooldown == 0) & (jnp.sum(new_active) < MAX_ENEMY_BULLETS)
    
    def spawn_bullet():
        bottom_enemies = get_bottom_enemies(state)
        
        # valid column is with alive enemies
        valid_columns = jnp.where(bottom_enemies >= 0, jnp.arange(ENEMY_COLS), -1)
        mask = valid_columns >= 0
        indices = jnp.nonzero(mask, size=ENEMY_COLS)[0]
        valid_columns = valid_columns[indices]
        num_valid = valid_columns.shape[0]
        
        def fire_bullet():
            # choose random column
            col_idx = jax.random.randint(key, (), 0, num_valid)
            firing_col = valid_columns[col_idx]
            firing_row = bottom_enemies[firing_col]
            
            enemy_x, enemy_y = get_enemy_position(
                state.opponent_current_x,
                state.opponent_current_y,
                firing_row,
                firing_col
            )
            
            # find first inactive bullet slot
            def find_slot(i, slot):
                return jnp.where((slot == -1) & (new_active[i] == False), i, slot)
            
            bullet_slot = jax.lax.fori_loop(0, MAX_ENEMY_BULLETS, find_slot, -1)
            
            # spawn bullet if slot found
            spawn_active = jnp.where(
                bullet_slot >= 0,
                new_active.at[bullet_slot].set(True),
                new_active
            )
            spawn_x = jnp.where(
                bullet_slot >= 0,
                state.enemy_bullets_x.at[bullet_slot].set(enemy_x + OPPONENT_SIZE[0] // 2),
                state.enemy_bullets_x
            )
            spawn_y = jnp.where(
                bullet_slot >= 0,
                new_y.at[bullet_slot].set(enemy_y + OPPONENT_SIZE[1]),
                new_y
            )
            
            return spawn_active, spawn_x, spawn_y, ENEMY_FIRE_RATE
        
        def no_fire():
            return new_active, state.enemy_bullets_x, new_y, new_cooldown
        
        return jax.lax.cond(num_valid > 0, fire_bullet, no_fire)
    
    def no_spawn():
        return new_active, state.enemy_bullets_x, new_y, new_cooldown
    
    return jax.lax.cond(should_fire, spawn_bullet, no_spawn)

@jax.jit
def player_step(state_player_x, state_player_speed, action: chex.Array):
    left = (action == Action.LEFT) | (action == Action.LEFTFIRE)
    right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)
    
    bounds_left = WALL_LEFT_X + PLAYER_SIZE[0]
    bounds_right = WALL_RIGHT_X 

    touches_wall_left = state_player_x <= bounds_left
    touches_wall_right = state_player_x >= bounds_right  
    touches_wall = jnp.logical_or(touches_wall_left, touches_wall_right)

    player_speed = jax.lax.cond(
        jnp.logical_not(jnp.logical_or(left, right)) | touches_wall,
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

@jax.jit
def player_bullet_step(state: SpaceInvadersState, action: chex.Array):
    fired = (action == Action.FIRE) | (action == Action.RIGHTFIRE) | (action == Action.LEFTFIRE)
    
    new_bullet_active, new_bullet_x, new_bullet_y = jax.lax.cond(
        state.bullet_active,
        # if bullet is active: update position and check boundaries
        lambda: jax.lax.cond(
            (state.bullet_y - BULLET_SPEED) < 0,
            lambda: (False, state.bullet_x, state.bullet_y),
            lambda: (True, state.bullet_x, state.bullet_y - BULLET_SPEED)
        ),
        # if bullet is inactive: check fire action
        lambda: jax.lax.cond(
            fired,
            lambda: (
                True,
                state.player_x - (PLAYER_SIZE[0] // 2),
                PLAYER_Y - PLAYER_SIZE[1]
            ),
            lambda: (False, state.bullet_x, state.bullet_y)
        )
    )
    
    return new_bullet_active, new_bullet_x, new_bullet_y


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
            player_x = jnp.array(96).astype(jnp.int32),
            player_speed = jnp.array(0.0).astype(jnp.int32),
            step_counter = jnp.array(0).astype(jnp.int32),
            player_score = jnp.array(0).astype(jnp.int32),
            player_lives = jnp.array(3).astype(jnp.int32),
            destroyed = jnp.zeros((ENEMY_ROWS * ENEMY_COLS,), dtype=jnp.bool),
            opponent_current_x = OPPONENT_LIMIT_X[0],
            opponent_current_y = OPPONENT_LIMIT_Y[0],
            opponent_bounding_rect = (OPPONENT_SIZE[0] * 6 + OFFSET_OPPONENT[0] * 5, OPPONENT_SIZE[1] * 6 + OFFSET_OPPONENT[1] * 5),
            opponent_direction = 1,
            bullet_active=jnp.array(0).astype(jnp.int32),
            bullet_x=jnp.array(78).astype(jnp.int32),
            bullet_y=jnp.array(78).astype(jnp.int32),
            enemy_bullets_active=jnp.zeros(MAX_ENEMY_BULLETS, dtype=jnp.bool),
            enemy_bullets_x=jnp.zeros(MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_bullets_y=jnp.zeros(MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_fire_cooldown=jnp.array(ENEMY_FIRE_RATE).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SpaceInvadersState, action: chex.Array, key = None) -> Tuple[SpaceInvadersObservation, SpaceInvadersState, float, bool, SpaceInvadersInfo]:
        if key is None:
            key = jax.random.PRNGKey(state.step_counter)

        new_player_x, new_player_speed = player_step(
            state.player_x, state.player_speed, action
        )

        new_player_x, new_player_speed = jax.lax.cond(
            state.step_counter % 2 == 0,
            lambda _: (new_player_x, new_player_speed),
            lambda _: (state.player_x, state.player_speed),
            operand=None,
        )

        new_bullet_active, new_bullet_x, new_bullet_y = player_bullet_step(state, action)

        new_bullet_state = state._replace(
            bullet_active=new_bullet_active,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y
        )
        new_destroyed, new_score, final_bullet_active = check_bullet_enemy_collisions(new_bullet_state)

        enemy_bullets_active, enemy_bullets_x, enemy_bullets_y, enemy_fire_cooldown = update_enemy_bullets(
            state._replace(
                destroyed=new_destroyed,
                enemy_bullets_active=state.enemy_bullets_active,
                enemy_bullets_x=state.enemy_bullets_x,
                enemy_bullets_y=state.enemy_bullets_y,
                enemy_fire_cooldown=state.enemy_fire_cooldown
            ), 
            key
        )

        new_lives, final_enemy_bullets_active = check_enemy_bullet_player_collisions(
            state._replace(
                player_x=new_player_x,
                enemy_bullets_active=enemy_bullets_active,
                enemy_bullets_x=enemy_bullets_x,
                enemy_bullets_y=enemy_bullets_y
            )
        )

        step_counter = jax.lax.cond(
            state.step_counter > 255,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        def get_opponent_position(): 
            """
            Calculates the opponents movement depending on the current state, the limits and the current bounding rect.
            The Bounding Rect should be the minimum size of a box required to contain all the (not destroyed) opponent sprites.

            @return Returns the direction that the opponents should move now and the new position.
            """
            direction = jax.lax.cond(
                state.opponent_direction < 0,
                # Checking left side borders as opponents are moving to the left
                lambda: jax.lax.cond(
                    OPPONENT_LIMIT_X[0] < state.opponent_current_x,
                    lambda: -1,
                    lambda: 1
                ),
                # Checking right side borders as opponents are moving to the right
                lambda: jax.lax.cond(
                    OPPONENT_LIMIT_X[1] > state.opponent_current_x + state.opponent_bounding_rect[0],
                    lambda: 1,
                    lambda: -1
                )
            )

            new_position = state.opponent_current_x + direction

            return (direction, new_position)

        # Opponents should not move in every game step
        is_opponent_step = state.step_counter % MOVEMENT_RATE == 0
        (direction, position) = jax.lax.cond(is_opponent_step, lambda: get_opponent_position(), lambda: (state.opponent_direction, state.opponent_current_x))

        new_state = SpaceInvadersState(
            player_x = new_player_x,
            player_speed = new_player_speed,
            step_counter = step_counter,
            player_score = new_score,
            player_lives = new_lives,
            destroyed = new_destroyed,
            opponent_current_x = position,
            opponent_current_y = state.opponent_current_y,
            opponent_bounding_rect = state.opponent_bounding_rect,
            opponent_direction = direction,
            bullet_active=final_bullet_active,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y,
            enemy_bullets_active=final_enemy_bullets_active,
            enemy_bullets_x=enemy_bullets_x,
            enemy_bullets_y=enemy_bullets_y,
            enemy_fire_cooldown=enemy_fire_cooldown
        )

        # TODO: Adjust the opponenet_bounding_rect depending on the destroyed enemies. If all the enemies of a outer column are destroyed the rect should shrink accordingly.

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

    # Score - All Green Numbers (0-9)
    green_numbers = []
    number_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    for i, name in enumerate(number_names):
        number_sprite = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, f"numbers/{name}_green.npy"), transpose=True)
        green_numbers.append(jnp.expand_dims(number_sprite, axis=0))
    
    SPRITE_ZERO_GREEN = green_numbers[0]
    SPRITE_ONE_GREEN = green_numbers[1]
    SPRITE_TWO_GREEN = green_numbers[2]
    SPRITE_THREE_GREEN = green_numbers[3]
    SPRITE_FOUR_GREEN = green_numbers[4]
    SPRITE_FIVE_GREEN = green_numbers[5]
    SPRITE_SIX_GREEN = green_numbers[6]
    SPRITE_SEVEN_GREEN = green_numbers[7]
    SPRITE_EIGHT_GREEN = green_numbers[8]
    SPRITE_NINE_GREEN = green_numbers[9]
    
    # Yellow numbers
    zero_yellow = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "numbers/zero_yellow.npy"), transpose=True)
    SPRITE_ZERO_YELLOW = jnp.expand_dims(zero_yellow, axis=0)

    # Defense
    defense = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "defense.npy"), transpose=True)
    SPRITE_DEFENSE = jnp.expand_dims(defense, axis=0)
    
    # Enemies
    opponent_1_a = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_1_a.npy"), transpose=True)
    SPRITE_OPPONENT_1_A = jnp.expand_dims(opponent_1_a, axis=0)
    opponent_1_b = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_1_b.npy"), transpose=True)
    SPRITE_OPPONENT_1_B = jnp.expand_dims(opponent_1_b, axis=0)
    opponent_2_a = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_2_a.npy"), transpose=True)
    SPRITE_OPPONENT_2_A = jnp.expand_dims(opponent_2_a, axis=0)
    opponent_2_b = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_2_b.npy"), transpose=True)
    SPRITE_OPPONENT_2_B = jnp.expand_dims(opponent_2_b, axis=0)
    opponent_3_a = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_3_a.npy"), transpose=True)
    SPRITE_OPPONENT_3_A = jnp.expand_dims(opponent_3_a, axis=0)
    opponent_3_b = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_3_b.npy"), transpose=True)
    SPRITE_OPPONENT_3_B = jnp.expand_dims(opponent_3_b, axis=0)
    SPRITE_OPPONENT_3_B = jax.numpy.pad(SPRITE_OPPONENT_3_B, ((0,0), (0,0), (0,1), (0,0)))
    opponent_4_a = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_4_a.npy"), transpose=True)
    SPRITE_OPPONENT_4_A = jnp.expand_dims(opponent_4_a, axis=0)
    opponent_4_b = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_4_b.npy"), transpose=True)
    SPRITE_OPPONENT_4_B = jnp.expand_dims(opponent_4_b, axis=0)
    SPRITE_OPPONENT_4_B = jax.numpy.pad(SPRITE_OPPONENT_4_B, ((0,0), (0,0), (0,1), (0,0)))
    opponent_5 = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_5.npy"), transpose=True)
    SPRITE_OPPONENT_5 = jnp.expand_dims(opponent_5, axis=0)
    opponent_6_a = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_6_a.npy"), transpose=True)
    SPRITE_OPPONENT_6_A = jnp.expand_dims(opponent_6_a, axis=0)
    opponent_6_b = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "opponents/opponent_6_b.npy"), transpose=True)
    SPRITE_OPPONENT_6_B = jnp.expand_dims(opponent_6_b, axis=0)

    bullet = aj.loadFrame(os.path.join(MODULE_DIR, PATH_SPRITES, "bullet.npy"), transpose=True)
    SPRITE_BULLET = jnp.expand_dims(bullet, axis=0) 

    return (SPRITE_BACKGROUND, SPRITE_PLAYER, SPRITE_BULLET, 
            SPRITE_ZERO_GREEN, SPRITE_ONE_GREEN, SPRITE_TWO_GREEN, SPRITE_THREE_GREEN, 
            SPRITE_FOUR_GREEN, SPRITE_FIVE_GREEN, SPRITE_SIX_GREEN, SPRITE_SEVEN_GREEN, 
            SPRITE_EIGHT_GREEN, SPRITE_NINE_GREEN, SPRITE_ZERO_YELLOW, SPRITE_DEFENSE, 
            SPRITE_OPPONENT_1_A, SPRITE_OPPONENT_1_B, SPRITE_OPPONENT_2_A, SPRITE_OPPONENT_2_B, 
            SPRITE_OPPONENT_3_A, SPRITE_OPPONENT_3_B, SPRITE_OPPONENT_4_A, SPRITE_OPPONENT_4_B, 
            SPRITE_OPPONENT_5, SPRITE_OPPONENT_6_A, SPRITE_OPPONENT_6_B)

(
    SPRITE_BACKGROUND, 
    SPRITE_PLAYER,
    SPRITE_BULLET,
    SPRITE_ZERO_GREEN,
    SPRITE_ONE_GREEN,
    SPRITE_TWO_GREEN,
    SPRITE_THREE_GREEN,
    SPRITE_FOUR_GREEN,
    SPRITE_FIVE_GREEN,
    SPRITE_SIX_GREEN,
    SPRITE_SEVEN_GREEN,
    SPRITE_EIGHT_GREEN,
    SPRITE_NINE_GREEN,
    SPRITE_ZERO_YELLOW,
    SPRITE_DEFENSE,
    SPRITE_OPPONENT_1_A,
    SPRITE_OPPONENT_1_B,
    SPRITE_OPPONENT_2_A,
    SPRITE_OPPONENT_2_B,
    SPRITE_OPPONENT_3_A,
    SPRITE_OPPONENT_3_B,
    SPRITE_OPPONENT_4_A,
    SPRITE_OPPONENT_4_B,
    SPRITE_OPPONENT_5,
    SPRITE_OPPONENT_6_A,
    SPRITE_OPPONENT_6_B,
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

        # Render bullet every even frame (TODO not sure if the blinking is just a render or logic thing)
        frame_bullet = aj.get_sprite_frame(SPRITE_BULLET, 0)
        raster = jax.lax.cond(
            (state.step_counter % 2 == 0) & (state.bullet_active),
            lambda r: aj.render_at(r, state.bullet_x, state.bullet_y, frame_bullet),
            lambda r: r,
            raster
        )

        # Render Score - Convert score to 4 digits
        score = state.player_score
        digit_1 = (score // 1000) % 10
        digit_2 = (score // 100) % 10
        digit_3 = (score // 10) % 10
        digit_4 = score % 10

        green_sprites = jnp.array([
            aj.get_sprite_frame(SPRITE_ZERO_GREEN, 0),
            aj.get_sprite_frame(SPRITE_ONE_GREEN, 0),
            aj.get_sprite_frame(SPRITE_TWO_GREEN, 0),
            aj.get_sprite_frame(SPRITE_THREE_GREEN, 0),
            aj.get_sprite_frame(SPRITE_FOUR_GREEN, 0),
            aj.get_sprite_frame(SPRITE_FIVE_GREEN, 0),
            aj.get_sprite_frame(SPRITE_SIX_GREEN, 0),
            aj.get_sprite_frame(SPRITE_SEVEN_GREEN, 0),
            aj.get_sprite_frame(SPRITE_EIGHT_GREEN, 0),
            aj.get_sprite_frame(SPRITE_NINE_GREEN, 0)
        ])

        # Load Initial Score
        # Green Numbers
        raster = aj.render_at(raster, 3, 9, green_sprites[digit_1])
        raster = aj.render_at(raster, 6 + NUMBER_SIZE[0], 10, green_sprites[digit_2])
        raster = aj.render_at(raster, 9 + 2 * NUMBER_SIZE[0], 10, green_sprites[digit_3])
        raster = aj.render_at(raster, 12 + 3 * NUMBER_SIZE[0], 9, green_sprites[digit_4])

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
        fo_5 = aj.get_sprite_frame(SPRITE_OPPONENT_5, 0)
        
        # Defines wether or not the sprites is getting flipped
        flip = jax.numpy.floor(state.step_counter / MOVEMENT_RATE) % 2 == 1

        # Loading sprites depending on the current animation step
        (fo_1, fo_2, fo_3, fo_4, fo_6) = jax.lax.cond(
            flip, 
            lambda: (
                aj.get_sprite_frame(SPRITE_OPPONENT_1_B, 0), 
                aj.get_sprite_frame(SPRITE_OPPONENT_2_B, 0), 
                aj.get_sprite_frame(SPRITE_OPPONENT_3_B, 0),
                aj.get_sprite_frame(SPRITE_OPPONENT_4_B, 0),
                aj.get_sprite_frame(SPRITE_OPPONENT_6_B, 0)
            ),
            lambda: (
                aj.get_sprite_frame(SPRITE_OPPONENT_1_A, 0), 
                aj.get_sprite_frame(SPRITE_OPPONENT_2_A, 0), 
                aj.get_sprite_frame(SPRITE_OPPONENT_3_A, 0),
                aj.get_sprite_frame(SPRITE_OPPONENT_4_A, 0),
                aj.get_sprite_frame(SPRITE_OPPONENT_6_A, 0)
            )
        )

        def render_enemies(i, raster):
            base_x = state.opponent_current_x + i * (OFFSET_OPPONENT[0] + OPPONENT_SIZE[0])

            def render_if_alive(row, sprite, y, raster, do_flip=False):
                idx = row * ENEMY_COLS + i
                alive = jnp.logical_not(state.destroyed[idx])
                return jax.lax.cond(
                    alive,
                    lambda r: aj.render_at(r, base_x, y, sprite, do_flip),
                    lambda r: r,
                    raster
                )

            raster = render_if_alive(0, fo_1, state.opponent_current_y + 0 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), raster)
            raster = render_if_alive(1, fo_2, state.opponent_current_y + 1 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), raster)
            raster = render_if_alive(2, fo_3, state.opponent_current_y + 2 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), raster)
            raster = render_if_alive(3, fo_4, state.opponent_current_y + 3 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), raster)
            raster = render_if_alive(4, fo_5, state.opponent_current_y + 4 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), raster, do_flip=flip)
            raster = render_if_alive(5, fo_6, state.opponent_current_y + 5 * (OFFSET_OPPONENT[1] + OPPONENT_SIZE[1]), raster)

            return raster
        raster = jax.lax.fori_loop(0, 6, render_enemies, raster)
   
        frame_enemy_bullet = aj.get_sprite_frame(SPRITE_BULLET, 0)
        def render_enemy_bullet_body(i, current_raster):
            def draw_bullet(r):
                return aj.render_at(r, state.enemy_bullets_x[i], state.enemy_bullets_y[i], frame_enemy_bullet)

            should_render = jnp.logical_and(
                state.enemy_bullets_active[i],
                state.step_counter % 2 == 0
            )

            return jax.lax.cond(
                should_render,
                draw_bullet,
                lambda r: r,
                current_raster
            )
        raster = jax.lax.fori_loop(0, MAX_ENEMY_BULLETS, render_enemy_bullet_body, raster)

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