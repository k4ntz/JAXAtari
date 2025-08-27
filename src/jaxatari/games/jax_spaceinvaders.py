#
# JAX SpaceInvaders
#
# Simulates the Space Invaders game using JAX
#
# Authors:
# - Luca Philippi 
# - Julian Bayer 
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class SpaceInvadersConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210

    SCALING_FACTOR: int = 4
    WINDOW_WIDTH: int = WIDTH * SCALING_FACTOR
    WINDOW_HEIGHT: int = HEIGHT * SCALING_FACTOR

    NUMBER_YELLOW_OFFSET: int = 83

    WALL_LEFT_X: int = 34
    WALL_RIGHT_X: int = 123

    OPPONENT_LIMIT_X: Tuple[int, int] = (22, 136)
    OPPONENT_LIMIT_Y: Tuple[int, None] = (31, None)

    MAX_SPEED: int = 1
    ACCELERATION: int = 1
    BULLET_SPEED: int = 1
    ENEMY_BULLET_SPEED: int = 1

    PATH_SPRITES: str = "sprites/spaceinvaders"
    ENEMY_ROWS: int = 6
    ENEMY_COLS: int = 6
    MAX_ENEMY_BULLETS: int = 3

    EXPLOSION_FRAMES: jnp.array = jnp.array([3, 11, 19, 27], dtype=jnp.int32)
    PLAYER_EXPLOSION_FRAMES: int = 4
    PLAYER_EXPLOSION_DURATION: int = 126
    PLAYER_RESET_DURATION: int = 64
    PLAYER_RESET_FRAMES: int = 8
    FULL_PAUSE_DURATION: int = PLAYER_EXPLOSION_DURATION + 1 + PLAYER_RESET_DURATION # +1 for the frame between the end of the explosion and the start of the blinking animation

    POSITION_LIFE_X: int = 83

    MOVEMENT_RATE: int = 32
    ENEMY_FIRE_RATE: int = 60

    INITIAL_LIVES: int = 3
    INITIAL_PLAYER_X: int = 41
    INITIAL_BULLET_POS: int = 78
    INITIAL_OPPONENT_DIRECTION: int = 1
    WIN_SCORE: int = 1000 # actually infinite 

    PLAYER_SIZE: Tuple[int, int] = (7, 10)
    BULLET_SIZE: Tuple[int, int] = (1, 10)
    WALL_SIZE: Tuple[int, int] = (2, 4)
    BACKGROUND_SIZE: Tuple[int, int] = (WIDTH, 15)
    NUMBER_SIZE: Tuple[int, int] = (12, 9)
    OPPONENT_SIZE: Tuple[int, int] = (8, 10)
    OPPONENT_VERTICAL_STEP_SIZE : int = 10 # vertical moving distance of opponents
    OFFSET_OPPONENT: Tuple[int, int] = (8, 8)

    PLAYER_Y: int = HEIGHT - PLAYER_SIZE[1] - BACKGROUND_SIZE[1]

C = SpaceInvadersConstants()

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
    player_dead: int
    step_counter: chex.Array
    player_score: chex.Array
    player_lives: chex.Array
    destroyed: chex.Array
    opponent_current_x: int
    opponent_current_y: int
    opponent_bounding_rect: NamedTuple
    opponent_direction: int
    bullet_active: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
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
    player: EntityPosition  # Player position and size
    enemies: jnp.ndarray  # Array of enemy positions and states
    player_bullet: EntityPosition  # Player bullet position and size
    enemy_bullets: jnp.ndarray  # Array of enemy bullet positions and states
    score_player: jnp.ndarray  # Player score
    lives: jnp.ndarray  # Player lives

class SpaceInvadersInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array

class JaxSpaceInvaders(JaxEnvironment[SpaceInvadersState, SpaceInvadersObservation, SpaceInvadersInfo, SpaceInvadersConstants]):
    def __init__(self, consts: SpaceInvadersConstants = None, reward_funcs: list[callable] = None):
        consts = consts or SpaceInvadersConstants()
        super().__init__(consts)
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
        self.obs_size = 3 * 4 + 1 + 1

    @partial(jax.jit, static_argnums=(0,))
    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.array([entity.x]),
            jnp.array([entity.y]),
            jnp.array([entity.width]),
            jnp.array([entity.height])
        ])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: SpaceInvadersObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_entity_position(obs.player),          # 4 values
            obs.enemies.flatten(),                             # ENEMY_ROWS * ENEMY_COLS * 5 values
            self.flatten_entity_position(obs.player_bullet),   # 4 values
            obs.enemy_bullets.flatten(),                       # MAX_ENEMY_BULLETS * 5 values
            jnp.array([obs.score_player]),                     # 1 value
            jnp.array([obs.lives])                            # 1 value
        ])

    def get_flat_obs_size(self) -> int:
        """Get the size of the flattened observation array."""
        player_size = 4  # x, y, width, height
        enemies_size = self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS * 5  # x, y, width, height, alive for each enemy
        player_bullet_size = 4  # x, y, width, height (no active field needed since bullet existence is implicit)
        enemy_bullets_size = self.consts.MAX_ENEMY_BULLETS * 5  # x, y, width, height, active for each bullet
        score_size = 1
        lives_size = 1
        
        return player_size + enemies_size + player_bullet_size + enemy_bullets_size + score_size + lives_size

    @partial(jax.jit, static_argnums=(0,))
    def _get_enemy_position(self, opponent_current_x, opponent_current_y, row, col):
        x = opponent_current_x + col * (self.consts.OFFSET_OPPONENT[0] + self.consts.OPPONENT_SIZE[0])
        y = opponent_current_y + row * (self.consts.OFFSET_OPPONENT[1] + self.consts.OPPONENT_SIZE[1])
        return x, y

    @partial(jax.jit, static_argnums=(0,))
    def _check_collision(self, bullet_x, bullet_y, target_x, target_y, target_width, target_height):
        bullet_right = bullet_x + self.consts.BULLET_SIZE[0]
        bullet_bottom = bullet_y + self.consts.BULLET_SIZE[1]
        target_right = target_x + target_width
        target_bottom = target_y + target_height

        collision = (bullet_x < target_right) & (bullet_right > target_x) & (bullet_y < target_bottom) & (
                    bullet_bottom > target_y)
        return collision

    @partial(jax.jit, static_argnums=(0,))
    def _check_bullet_enemy_collisions(self, state: SpaceInvadersState):
        def check_single_enemy(i):
            row = i // self.consts.ENEMY_COLS
            col = i % self.consts.ENEMY_COLS
            
            enemy_x, enemy_y = self._get_enemy_position(
                state.opponent_current_x,
                state.opponent_current_y,
                row,
                col
            )
            
            enemy_alive = jnp.logical_not(state.destroyed[i])
            collision = jnp.logical_and(
                jnp.logical_and(enemy_alive, state.bullet_active),
                self._check_collision(state.bullet_x, state.bullet_y, enemy_x, enemy_y, 
                                self.consts.OPPONENT_SIZE[0], self.consts.OPPONENT_SIZE[1])
            )
            
            new_destroyed = jnp.where((state.destroyed[i] == 0) & collision, 1, state.destroyed[i])
            score_contrib = jnp.where(collision, 10, 0)
            bullet_hit = collision
            
            # Returns if destroyed, the contribution to the score and if the bullet hit 
            return new_destroyed, score_contrib, bullet_hit

        def check_all_enemies():
            # vmap over all enemy indices
            enemy_indices = jnp.arange(self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS)
            destroyed_vals, score_contribs, bullet_hits = jax.vmap(check_single_enemy)(enemy_indices)
            
            # Update destroyed states (handle explosion animation)
            destroyed_updated = jnp.where(state.destroyed != 0, jnp.minimum(state.destroyed + 1, 29), 0)
            final_destroyed = jnp.where(destroyed_vals > state.destroyed, destroyed_vals, destroyed_updated)
            
            total_score = state.player_score + jnp.sum(score_contribs)
            bullet_active = jnp.logical_not(jnp.any(bullet_hits))
            
            return final_destroyed, total_score, bullet_active

        def no_bullet():
            destroyed = jnp.where(state.destroyed != 0, jnp.minimum(state.destroyed + 1, 29), 0)
            return destroyed, state.player_score, state.bullet_active

        return jax.lax.cond(
            state.bullet_active,
            check_all_enemies,
            no_bullet
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_enemy_bullet_player_collisions(self, state: SpaceInvadersState):
        def check_bullet(i, carry):
            lives, enemy_bullets_active = carry

            bullet_active = enemy_bullets_active[i]
            collision = jnp.logical_and(
                bullet_active,
                self._check_collision(
                    state.enemy_bullets_x[i],
                    state.enemy_bullets_y[i],
                    state.player_x - self.consts.PLAYER_SIZE[0],
                    self.consts.PLAYER_Y,
                    self.consts.PLAYER_SIZE[0],
                    self.consts.PLAYER_SIZE[1]
                )
            )

            new_bullet_active = jnp.where(collision, False, bullet_active)
            enemy_bullets_active = enemy_bullets_active.at[i].set(new_bullet_active)
            lives = jnp.where(collision, lives - 1, lives)

            return lives, enemy_bullets_active

        init = (state.player_lives, state.enemy_bullets_active)
        return jax.lax.fori_loop(0, self.consts.MAX_ENEMY_BULLETS, check_bullet, init)

    @partial(jax.jit, static_argnums=(0,))
    def _get_bottom_enemies(self, state: SpaceInvadersState):
        def check_column(col):
            def check_row(row):
                idx = row * self.consts.ENEMY_COLS + col
                return jnp.logical_not(state.destroyed[idx])

            has_enemy = jnp.array([check_row(row) for row in range(self.consts.ENEMY_ROWS - 1, -1, -1)])
            bottom_row = jnp.where(has_enemy, jnp.arange(self.consts.ENEMY_ROWS - 1, -1, -1), -1)
            actual_bottom = jnp.max(jnp.where(bottom_row >= 0, bottom_row, -1))

            return jnp.where(actual_bottom >= 0, actual_bottom, -1)

        return jnp.array([check_column(col) for col in range(self.consts.ENEMY_COLS)])

    @partial(jax.jit, static_argnums=(0,))
    def _update_enemy_bullets(self, state: SpaceInvadersState, key):
        new_y = state.enemy_bullets_y + self.consts.ENEMY_BULLET_SPEED
        new_active = jnp.where(new_y >= self.consts.HEIGHT, False, state.enemy_bullets_active)

        new_cooldown = jnp.maximum(0, state.enemy_fire_cooldown - 1)
        should_fire = (new_cooldown == 0) & (jnp.sum(new_active) < self.consts.MAX_ENEMY_BULLETS)

        def spawn_bullet():
            bottom_enemies = self._get_bottom_enemies(state)

            valid_columns = jnp.where(bottom_enemies >= 0, jnp.arange(self.consts.ENEMY_COLS), -1)
            mask = valid_columns >= 0
            indices = jnp.nonzero(mask, size=self.consts.ENEMY_COLS)[0]
            valid_columns = valid_columns[indices]
            num_valid = valid_columns.shape[0]

            def fire_bullet():
                col_idx = jax.random.randint(key, (), 0, num_valid)
                firing_col = valid_columns[col_idx]
                firing_row = bottom_enemies[firing_col]

                enemy_x, enemy_y = self._get_enemy_position(
                    state.opponent_current_x,
                    state.opponent_current_y,
                    firing_row,
                    firing_col
                )

                def find_slot(i, slot):
                    return jnp.where((slot == -1) & (new_active[i] == False), i, slot)

                bullet_slot = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_BULLETS, find_slot, -1)

                spawn_active = jnp.where(
                    bullet_slot >= 0,
                    new_active.at[bullet_slot].set(True),
                    new_active
                )
                spawn_x = jnp.where(
                    bullet_slot >= 0,
                    state.enemy_bullets_x.at[bullet_slot].set(enemy_x + self.consts.OPPONENT_SIZE[0] // 2),
                    state.enemy_bullets_x
                )
                spawn_y = jnp.where(
                    bullet_slot >= 0,
                    new_y.at[bullet_slot].set(enemy_y + self.consts.OPPONENT_SIZE[1]),
                    new_y
                )

                return spawn_active, spawn_x, spawn_y, self.consts.ENEMY_FIRE_RATE

            def no_fire():
                return new_active, state.enemy_bullets_x, new_y, new_cooldown

            return jax.lax.cond(num_valid > 0, fire_bullet, no_fire)

        def no_spawn():
            return new_active, state.enemy_bullets_x, new_y, new_cooldown

        return jax.lax.cond(should_fire, spawn_bullet, no_spawn)

    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state_player_x, state_player_speed, action: chex.Array):
        left = (action == Action.LEFT) | (action == Action.LEFTFIRE)
        right = (action == Action.RIGHT) | (action == Action.RIGHTFIRE)

        bounds_left = self.consts.WALL_LEFT_X + self.consts.PLAYER_SIZE[0]
        bounds_right = self.consts.WALL_RIGHT_X

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
            lambda s: jnp.minimum(s + self.consts.ACCELERATION, self.consts.MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        player_speed = jax.lax.cond(
            left,
            lambda s: jnp.maximum(s - self.consts.ACCELERATION, -self.consts.MAX_SPEED),
            lambda s: s,
            operand=player_speed,
        )

        player_x = jnp.clip(
            state_player_x + player_speed,
            bounds_left,
            bounds_right
        )

        return player_x, player_speed

    @partial(jax.jit, static_argnums=(0,))
    def _player_bullet_step(self, state: SpaceInvadersState, action: chex.Array):
        fired = (action == Action.FIRE) | (action == Action.RIGHTFIRE) | (action == Action.LEFTFIRE)

        new_bullet_active, new_bullet_x, new_bullet_y = jax.lax.cond(
            state.bullet_active,
            lambda: jax.lax.cond(
                (state.bullet_y - self.consts.BULLET_SPEED) < 0,
                lambda: (False, state.bullet_x, state.bullet_y),
                lambda: (True, state.bullet_x, state.bullet_y - self.consts.BULLET_SPEED)
            ),
            lambda: jax.lax.cond(
                fired,
                lambda: (
                    True,
                    state.player_x - (self.consts.PLAYER_SIZE[0] // 2),
                    self.consts.PLAYER_Y - self.consts.PLAYER_SIZE[1]
                ),
                lambda: (False, state.bullet_x, state.bullet_y)
            )
        )

        return new_bullet_active, new_bullet_x, new_bullet_y

    def reset(self, key=None) -> Tuple[SpaceInvadersObservation, SpaceInvadersState]:
        opponent_rect_width = self.consts.OPPONENT_SIZE[0] * self.consts.ENEMY_COLS + self.consts.OFFSET_OPPONENT[0] * (self.consts.ENEMY_COLS - 1)
        opponent_rect_height = self.consts.OPPONENT_SIZE[1] * self.consts.ENEMY_ROWS + self.consts.OFFSET_OPPONENT[1] * (self.consts.ENEMY_ROWS - 1)

        state = SpaceInvadersState(
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            player_dead=self.consts.PLAYER_EXPLOSION_DURATION,
            step_counter=jnp.array(0).astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            player_lives=jnp.array(self.consts.INITIAL_LIVES).astype(jnp.int32),
            destroyed=jnp.zeros((self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS,), dtype=jnp.int32), # If 0 its alive, after it counts up to 28 each frame showing a different animation state depending on this value. Starting with 29 its gone 
            opponent_current_x=self.consts.OPPONENT_LIMIT_X[0],
            opponent_current_y=self.consts.OPPONENT_LIMIT_Y[0],
            opponent_bounding_rect=(opponent_rect_width, opponent_rect_height),
            opponent_direction=self.consts.INITIAL_OPPONENT_DIRECTION,
            bullet_active=jnp.array(0).astype(jnp.int32),
            bullet_x=jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            bullet_y=jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            enemy_bullets_active=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.bool),
            enemy_bullets_x=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_bullets_y=jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_fire_cooldown=jnp.array(self.consts.ENEMY_FIRE_RATE).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    def reset_level(self, state: SpaceInvadersState) -> SpaceInvadersState:
        # new_state = SpaceInvadersState(
        state = state._replace(
            player_x = jnp.array(self.consts.INITIAL_PLAYER_X).astype(jnp.int32),
            player_speed = jnp.array(0.0).astype(jnp.int32),
            player_dead = state.player_dead,
            step_counter = state.step_counter,
            player_score = state.player_score,
            player_lives = state.player_lives,
            destroyed = jnp.zeros((self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS,), dtype=jnp.int32),
            opponent_current_x = self.consts.OPPONENT_LIMIT_X[0],
            opponent_current_y = self.consts.OPPONENT_LIMIT_Y[0],
            opponent_bounding_rect = state.opponent_bounding_rect,
            opponent_direction = self.consts.INITIAL_OPPONENT_DIRECTION,
            bullet_active = jnp.array(0).astype(jnp.int32),
            bullet_x = jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            bullet_y = jnp.array(self.consts.INITIAL_BULLET_POS).astype(jnp.int32),
            enemy_bullets_active = jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.bool),
            enemy_bullets_x = jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_bullets_y = jnp.zeros(self.consts.MAX_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_fire_cooldown = jnp.array(self.consts.ENEMY_FIRE_RATE).astype(jnp.int32)
        )

        return state

    def step_paused(self, state: SpaceInvadersState) -> SpaceInvadersState:
        new_player_dead = jax.lax.cond(
            state.player_dead + 1 > self.consts.FULL_PAUSE_DURATION,
            lambda _: 0,
            lambda _: state.player_dead + 1,
            None
        )

        # Resets the level only once during the pause animation
        state = jax.lax.cond(
            state.player_dead == self.consts.PLAYER_EXPLOSION_DURATION + 1, 
            lambda: self.reset_level(state), 
            lambda: state
        )
        state = state._replace(player_dead = new_player_dead)

        return state
    
    def step_running(self, state: SpaceInvadersState, action: chex.Array, key=None) -> SpaceInvadersState:
        new_player_x, new_player_speed = self._player_step(state.player_x, state.player_speed, action)

        new_player_x, new_player_speed = jax.lax.cond(
            jnp.logical_and(state.step_counter % 2 == 0, state.player_dead == 0),
            lambda _: (new_player_x, new_player_speed),
            lambda _: (state.player_x, state.player_speed),
            operand=None,
        )

        new_bullet_active, new_bullet_x, new_bullet_y = self._player_bullet_step(state, action)

        new_bullet_state = state._replace(
            bullet_active=new_bullet_active,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y
        )
        new_destroyed, new_score, final_bullet_active = self._check_bullet_enemy_collisions(new_bullet_state)

        enemy_bullets_active, enemy_bullets_x, enemy_bullets_y, enemy_fire_cooldown = self._update_enemy_bullets(
            state._replace(
                destroyed=new_destroyed,
                enemy_bullets_active=state.enemy_bullets_active,
                enemy_bullets_x=state.enemy_bullets_x,
                enemy_bullets_y=state.enemy_bullets_y,
                enemy_fire_cooldown=state.enemy_fire_cooldown,
            ),
            key
        )

        new_lives, final_enemy_bullets_active = self._check_enemy_bullet_player_collisions(
            state._replace(
                player_x=new_player_x,
                enemy_bullets_active=enemy_bullets_active,
                enemy_bullets_x=enemy_bullets_x,
                enemy_bullets_y=enemy_bullets_y
            )
        )

        new_player_dead = jax.lax.cond(
            jnp.logical_and(new_lives != state.player_lives, state.player_dead == 0),
            lambda _: 1,
            lambda _: jnp.where(state.player_dead > 0, state.player_dead + 1, state.player_dead),
            None
        )

        def get_opponent_position():
            direction = jax.lax.cond(
                state.opponent_direction < 0,
                lambda: jax.lax.cond(
                    self.consts.OPPONENT_LIMIT_X[0] < state.opponent_current_x,
                    lambda: -1,
                    lambda: 1
                ),
                lambda: jax.lax.cond(
                    self.consts.OPPONENT_LIMIT_X[1] > state.opponent_current_x + state.opponent_bounding_rect[0],
                    lambda: 1,
                    lambda: -1
                )
            )

            new_position_y = jax.lax.cond(
                direction != state.opponent_direction,
                lambda y: y + self.consts.OPPONENT_VERTICAL_STEP_SIZE,
                lambda y: y,
                state.opponent_current_y
            )

            new_position_x = state.opponent_current_x + direction
            return (direction, new_position_x, new_position_y)

        is_opponent_step = state.step_counter % self.consts.MOVEMENT_RATE == 0
        (direction, new_position_x, new_position_y) = jax.lax.cond(
            is_opponent_step, 
            lambda: get_opponent_position(),
            lambda: (state.opponent_direction, state.opponent_current_x, state.opponent_current_y)
        )

        new_state = SpaceInvadersState(
            player_x=new_player_x,
            player_speed=new_player_speed,
            player_dead=new_player_dead,
            step_counter=state.step_counter,
            player_score=new_score,
            player_lives=new_lives,
            destroyed=new_destroyed,
            opponent_current_x=new_position_x,
            opponent_current_y=new_position_y,
            opponent_bounding_rect=state.opponent_bounding_rect,
            opponent_direction=direction,
            bullet_active=final_bullet_active.astype(jnp.int32),
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y,
            enemy_bullets_active=final_enemy_bullets_active,
            enemy_bullets_x=enemy_bullets_x,
            enemy_bullets_y=enemy_bullets_y,
            enemy_fire_cooldown=enemy_fire_cooldown
        )

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SpaceInvadersState, action: chex.Array, key=None) -> Tuple[SpaceInvadersObservation, SpaceInvadersState, float, bool, SpaceInvadersInfo]:
        if key is None:
            key = jax.random.PRNGKey(state.step_counter)

        new_step_counter = jax.lax.cond(
            state.step_counter > 255,
            lambda _: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        state = state._replace(step_counter = new_step_counter)

        new_state: SpaceInvadersState = jax.lax.cond(
            state.player_dead > 0,
            lambda: self.step_paused(state),
            lambda: self.step_running(state, action, key)
        )   

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info
    
    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for SpaceInvaders.
        The observation contains:
        - player: EntityPosition (x, y, width, height) - player position and size
        - enemies: array of shape (ENEMY_ROWS * ENEMY_COLS, 5) with x, y, width, height, alive for each enemy
        - player_bullet: EntityPosition (x, y, width, height) - player's bullet
        - enemy_bullets: array of shape (MAX_ENEMY_BULLETS, 5) with x, y, width, height, active for each enemy bullet
        - score_player: int - current player score
        - lives: int - remaining player lives
        """
        total_enemies = self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS
        
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "enemies": spaces.Box(
                low=0, 
                high=max(self.consts.WIDTH, self.consts.HEIGHT), 
                shape=(total_enemies, 5), 
                dtype=jnp.int32
            ),
            "player_bullet": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "enemy_bullets": spaces.Box(
                low=0, 
                high=max(self.consts.WIDTH, self.consts.HEIGHT), 
                shape=(self.consts.MAX_ENEMY_BULLETS, 5), 
                dtype=jnp.int32
            ),
            "score_player": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.INITIAL_LIVES, shape=(), dtype=jnp.int32),
        })

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SpaceInvadersState):
        player = EntityPosition(
            x=state.player_x,
            y=jnp.array(self.consts.PLAYER_Y),
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )
        
        # Create enemy observations
        def get_enemy_obs(i):
            row = i // self.consts.ENEMY_COLS
            col = i % self.consts.ENEMY_COLS
            
            enemy_x, enemy_y = self._get_enemy_position(
                state.opponent_current_x,
                state.opponent_current_y,
                row,
                col
            )
            
            alive = jnp.logical_not(state.destroyed[i]).astype(jnp.int32)
            return jnp.array([
                enemy_x,
                enemy_y,
                self.consts.OPPONENT_SIZE[0],
                self.consts.OPPONENT_SIZE[1],
                alive
            ])
        
        total_enemies = self.consts.ENEMY_ROWS * self.consts.ENEMY_COLS
        enemies = jnp.array([get_enemy_obs(i) for i in range(total_enemies)])
        
        # Player bullet observation
        player_bullet = EntityPosition(
            x=state.bullet_x,
            y=state.bullet_y,
            width=jnp.array(self.consts.BULLET_SIZE[0]),
            height=jnp.array(self.consts.BULLET_SIZE[1]),
        )
        
        # Enemy bullets observation
        def get_enemy_bullet_obs(i):
            active = state.enemy_bullets_active[i].astype(jnp.int32)
            return jnp.array([
                state.enemy_bullets_x[i],
                state.enemy_bullets_y[i],
                self.consts.BULLET_SIZE[0],
                self.consts.BULLET_SIZE[1],
                active
            ])
        
        enemy_bullets = jnp.array([get_enemy_bullet_obs(i) for i in range(self.consts.MAX_ENEMY_BULLETS)])
        
        return SpaceInvadersObservation(
            player=player,
            enemies=enemies,
            player_bullet=player_bullet,
            enemy_bullets=enemy_bullets,
            score_player=state.player_score,
            lives=state.player_lives,
        )
        
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

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
        return jnp.greater_equal(state.player_score, self.consts.WIN_SCORE)

def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    SPRITES_DIR = os.path.join(MODULE_DIR, C.PATH_SPRITES)
    
    # Player
    player = aj.loadFrame(os.path.join(SPRITES_DIR, "player.npy"))
    player_invisible = aj.loadFrame(os.path.join(SPRITES_DIR, "player_invisible.npy"))
    SPRITE_PLAYER = jnp.expand_dims(player, axis=0)
    SPRITE_PLAYER_INVISIBLE = jnp.expand_dims(player_invisible, axis=0)

    # Background
    background = aj.loadFrame(os.path.join(SPRITES_DIR, "background.npy"))
    SPRITE_BACKGROUND = jnp.expand_dims(background, axis=0)

    # Score
    green_numbers = []
    number_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for name in number_names:
        number_sprite = aj.loadFrame(os.path.join(SPRITES_DIR, f"numbers/{name}_green.npy"))
        green_numbers.append(jnp.expand_dims(number_sprite, axis=0))
    
    zero_yellow = aj.loadFrame(os.path.join(SPRITES_DIR, "numbers/zero_yellow.npy"))
    SPRITE_ZERO_YELLOW = jnp.expand_dims(zero_yellow, axis=0)
    
    # Defense
    defense = aj.loadFrame(os.path.join(SPRITES_DIR, "defense.npy"))
    SPRITE_DEFENSE = jnp.expand_dims(defense, axis=0)

    # Enemies
    opponents = {}
    opponent_files = [
        "opponent_1_a", "opponent_1_b", "opponent_2_a", "opponent_2_b",
        "opponent_3_a", "opponent_3_b", "opponent_4_a", "opponent_4_b",
        "opponent_5", "opponent_6_a", "opponent_6_b"
    ]
    for name in opponent_files:
        sprite = aj.loadFrame(os.path.join(SPRITES_DIR, f"opponents/{name}.npy"))
        opponents[name] = jnp.expand_dims(sprite, axis=0)

    opponents["opponent_3_b"] = jax.numpy.pad(opponents["opponent_3_b"], ((0,0), (0,1), (0,0), (0,0)))
    opponents["opponent_4_b"] = jax.numpy.pad(opponents["opponent_4_b"], ((0,0), (0,1), (0,0), (0,0)))

    # Bullet
    bullet = aj.loadFrame(os.path.join(SPRITES_DIR, "bullet.npy"))
    SPRITE_BULLET = jnp.expand_dims(bullet, axis=0)

    # Explosions
    explosions = {}
    explosion_files = [
        "explosion_1", "explosion_2", "explosion_3", "explosion_4", "explosion_purple_a", "explosion_purple_b", "exp_player_1", "exp_player_2"
    ]

    for name in explosion_files:
        sprite = aj.loadFrame(os.path.join(SPRITES_DIR, f"explosions/{name}.npy"))
        explosions[name] = jnp.expand_dims(sprite, axis=0)

    # Lives
    lifes = {}
    life_files = ["one", "two", "three"]

    for name in life_files:
        sprite = aj.loadFrame(os.path.join(SPRITES_DIR, f"lifes/{name}.npy"))
        lifes[name] = jnp.expand_dims(sprite, axis=0)

    return (SPRITE_BACKGROUND, SPRITE_PLAYER, SPRITE_PLAYER_INVISIBLE, SPRITE_BULLET, *green_numbers, SPRITE_ZERO_YELLOW, SPRITE_DEFENSE, *opponents.values(), *explosions.values(), *lifes.values())

(
    SPRITE_BACKGROUND, SPRITE_PLAYER, SPRITE_PLAYER_INVISIBLE, SPRITE_BULLET,
    SPRITE_ZERO_GREEN, SPRITE_ONE_GREEN, SPRITE_TWO_GREEN, SPRITE_THREE_GREEN,
    SPRITE_FOUR_GREEN, SPRITE_FIVE_GREEN, SPRITE_SIX_GREEN, SPRITE_SEVEN_GREEN,
    SPRITE_EIGHT_GREEN, SPRITE_NINE_GREEN, SPRITE_ZERO_YELLOW, SPRITE_DEFENSE,
    SPRITE_OPPONENT_1_A, SPRITE_OPPONENT_1_B, SPRITE_OPPONENT_2_A, SPRITE_OPPONENT_2_B,
    SPRITE_OPPONENT_3_A, SPRITE_OPPONENT_3_B, SPRITE_OPPONENT_4_A, SPRITE_OPPONENT_4_B,
    SPRITE_OPPONENT_5, SPRITE_OPPONENT_6_A, SPRITE_OPPONENT_6_B, SPRITE_EXPLOSION_1, 
    SPRITE_EXPLOSION_2, SPRITE_EXPLOSION_3, SPRITE_EXPLOSION_4, 
    SPRITE_EXPLOSION_PURPLE_A, SPRITE_EXPLOSION_PURPLE_B, PLAYER_EXPLOSION_A, PLAYER_EXPLOSION_B,
    LIFE_1, LIFE_2, LIFE_3
) = load_sprites()

class SpaceInvadersRenderer(JAXGameRenderer):
    life_sprites: chex.Array

    def __init__(self, consts: SpaceInvadersConstants = None):
        super().__init__()
        self.consts = consts or SpaceInvadersConstants()
        self.SPRITE_PLAYER = SPRITE_PLAYER

        self.init_sprites()

    def init_sprites(self):
        self.life_sprites = jnp.array([aj.get_sprite_frame(LIFE_1, 0), aj.get_sprite_frame(LIFE_2, 0), aj.get_sprite_frame(LIFE_3, 0)])

    def get_player_sprite(self, state: SpaceInvadersState):
        return jax.lax.cond(
            state.player_dead > self.consts.PLAYER_EXPLOSION_DURATION,
            lambda _: jnp.where(jnp.floor((state.player_dead - 1) / self.consts.PLAYER_RESET_FRAMES) % 2 == 0, SPRITE_PLAYER, SPRITE_PLAYER_INVISIBLE),
            lambda _: jnp.where(jnp.floor(state.player_dead / self.consts.PLAYER_EXPLOSION_FRAMES) % 2 == 0, PLAYER_EXPLOSION_A, PLAYER_EXPLOSION_B),
            None
        )
    
    def render_life(self, state: SpaceInvadersState, raster): 
        sprite = self.life_sprites[state.player_lives - 1]
        raster = aj.render_at(raster, self.consts.POSITION_LIFE_X, self.consts.PLAYER_Y, sprite)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: SpaceInvadersState) -> chex.Array:
        raster = aj.create_initial_frame(width=self.consts.WIDTH, height=self.consts.HEIGHT)

        # Load Background
        background = aj.get_sprite_frame(SPRITE_BACKGROUND, 0)
        raster = aj.render_at(raster, 0, self.consts.HEIGHT - self.consts.BACKGROUND_SIZE[1], background)

        # Load Player
        sprite_player = self.get_player_sprite(state)
        sprite_player = jax.lax.cond(
            state.player_dead == 0,
            lambda _: self.SPRITE_PLAYER,
            lambda _: self.get_player_sprite(state),
            None
        )

        frame_player = aj.get_sprite_frame(sprite_player, 0)
        raster = aj.render_at(raster, state.player_x - self.consts.PLAYER_SIZE[0], self.consts.PLAYER_Y, frame_player)

        raster = jax.lax.cond(
            state.player_dead > self.consts.PLAYER_EXPLOSION_DURATION + 1,
            lambda r: self.render_life(state, r),
            lambda r: r,
            raster
        )

        # Render bullet every even frame 
        frame_bullet = aj.get_sprite_frame(SPRITE_BULLET, 0)
        raster = jax.lax.cond(
            (state.step_counter % 2 == 0) & (state.bullet_active),
            lambda r: aj.render_at(r, state.bullet_x, state.bullet_y, frame_bullet),
            lambda r: r,
            raster
        )

        # Render Score - Convert score to 4 digits
        score = state.player_score
        digits = [
            (score // 1000) % 10,
            (score // 100) % 10,
            (score // 10) % 10,
            score % 10
        ]
        
        green_sprites = jnp.array([
            aj.get_sprite_frame(s, 0) for s in [
                SPRITE_ZERO_GREEN, SPRITE_ONE_GREEN, SPRITE_TWO_GREEN, SPRITE_THREE_GREEN, SPRITE_FOUR_GREEN,
                SPRITE_FIVE_GREEN, SPRITE_SIX_GREEN, SPRITE_SEVEN_GREEN, SPRITE_EIGHT_GREEN, SPRITE_NINE_GREEN
            ]
        ])

        # Load Initial Score
        score_pos_x = [3, 6, 9, 12]
        score_pos_y = [9, 10, 10, 9]
        for i in range(4):
            raster = aj.render_at(raster, score_pos_x[i] + i * self.consts.NUMBER_SIZE[0], score_pos_y[i], green_sprites[digits[i]])

        frame_zero_yellow = aj.get_sprite_frame(SPRITE_ZERO_YELLOW, 0)
        for i in range(4):
            raster = aj.render_at(raster, self.consts.NUMBER_YELLOW_OFFSET + score_pos_x[i] + i * self.consts.NUMBER_SIZE[0], score_pos_y[i], frame_zero_yellow)

        # Load Defense Object
        frame_defense = aj.get_sprite_frame(SPRITE_DEFENSE, 0)
        defense_pos_x = [41, 73, 105]
        for x_pos in defense_pos_x:
            raster = aj.render_at(raster, x_pos, self.consts.HEIGHT - 53, frame_defense)

        # Load Opponent Sprites
        flip = jax.lax.cond(
            state.player_dead != 0,
            lambda: False,
            lambda: jax.numpy.floor(state.step_counter / self.consts.MOVEMENT_RATE) % 2 == 1
        )

        opponent_sprites_a = [SPRITE_OPPONENT_1_A, SPRITE_OPPONENT_2_A, SPRITE_OPPONENT_3_A, SPRITE_OPPONENT_4_A, SPRITE_OPPONENT_5, SPRITE_OPPONENT_6_A]
        opponent_sprites_b = [SPRITE_OPPONENT_1_B, SPRITE_OPPONENT_2_B, SPRITE_OPPONENT_3_B, SPRITE_OPPONENT_4_B, SPRITE_OPPONENT_5, SPRITE_OPPONENT_6_B]
        
        sprites_to_render = jax.lax.cond(
            flip,
            lambda: jnp.stack([aj.get_sprite_frame(s, 0) for s in opponent_sprites_b]),
            lambda: jnp.stack([aj.get_sprite_frame(s, 0) for s in opponent_sprites_a])
        )

        explosion_sprites = [SPRITE_EXPLOSION_1, SPRITE_EXPLOSION_2, SPRITE_EXPLOSION_3, SPRITE_EXPLOSION_4]
        explosion_purple_sprites = [SPRITE_EXPLOSION_PURPLE_A, SPRITE_EXPLOSION_PURPLE_B]
        explosion_player_sprites = [PLAYER_EXPLOSION_A, PLAYER_EXPLOSION_B]

        explosion_sprites = jnp.array([aj.get_sprite_frame(s, 0) for s in explosion_sprites])

        def render_explosion(idx, raster, x, y):
            destroyed_frame = state.destroyed[idx] # value between [1;28]

            #TODO state._replace(exploded_frames=frames)

            sprite_id = jnp.argmax(destroyed_frame < self.consts.EXPLOSION_FRAMES)
            sprite = explosion_sprites[sprite_id]

            raster = aj.render_at(raster, x, y, sprite, False)
            #jax.debug.print("Drawing Explosion at x: {x}, y: {y} with Frame: {frame} and Sprite: {sprite} for IDX: {idx}", x=x, y=y, frame=frames[idx], sprite=sprite_id, idx=idx)

            return raster


        def render_enemies(i, raster):
            base_x = state.opponent_current_x + i * (self.consts.OFFSET_OPPONENT[0] + self.consts.OPPONENT_SIZE[0])
            
            def render_row(j, raster):
                idx = j * self.consts.ENEMY_COLS + i
                y_pos = state.opponent_current_y + j * (self.consts.OFFSET_OPPONENT[1] + self.consts.OPPONENT_SIZE[1])
                sprite = sprites_to_render[j]
                
                is_alive = jnp.logical_not(state.destroyed[idx])
                raster = jax.lax.cond(
                    jnp.logical_and(state.destroyed[idx] != 0, state.destroyed[idx] < 29), # Starting with the 29th frame the explosion animation is finished and doesnt need to be shown anymore 
                    lambda r: render_explosion(idx, r, base_x, y_pos),
                    lambda r: r,
                    raster
                )
                
                # Special flip logic for opponent 5
                do_flip = jnp.where(j == 4, flip, False)

                return jax.lax.cond(
                    is_alive,
                    lambda r: aj.render_at(r, base_x, y_pos, sprite, do_flip),
                    lambda r: r,
                    raster
                )
            
            raster = jax.lax.fori_loop(0, self.consts.ENEMY_ROWS, render_row, raster)
            return raster
        
        raster = jax.lax.fori_loop(0, self.consts.ENEMY_COLS, render_enemies, raster)

        frame_enemy_bullet = aj.get_sprite_frame(SPRITE_BULLET, 0)
        def render_enemy_bullet_body(i, current_raster):
            should_render = state.enemy_bullets_active[i] & (state.step_counter % 2 == 0)
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(r, state.enemy_bullets_x[i], state.enemy_bullets_y[i], frame_enemy_bullet),
                lambda r: r,
                current_raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMY_BULLETS, render_enemy_bullet_body, raster)
        return raster

