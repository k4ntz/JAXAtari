from turtle import width
from typing import NamedTuple, Tuple, Optional, Dict, Any
from functools import partial
import jax
import jax.numpy as jnp
import chex
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from gymnax.environments import spaces
import pygame
import os

from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr

# Group: Kaan Yilmaz, Jonathan Frey
# Game: Berzerk
# Tested on Ubuntu Virtual Machine

class BerzerkConstants(NamedTuple):
    WIDTH = 160
    HEIGHT = 210
    SCALING_FACTOR = 3

    PLAYER_SIZE = (6, 20)
    PLAYER_SPEED = 0.2

    ENEMY_SIZE = (8, 16)
    NUM_ENEMIES = 5
    MOVEMENT_PROB = 0.0025  # Value for testing, has to be adjusted
    ENEMY_SPEED = 0.05
    ENEMY_SHOOT_PROB = 0.001

    BULLET_SIZE_HORIZONTAL = (4, 2)
    BULLET_SIZE_VERTICAL = (1, 6)
    BULLET_SPEED = 1
    MAX_BULLETS = 1

    WALL_THICKNESS = 4
    WALL_OFFSET = (4, 4, 4, 30) # left, top, right, bottom
    EXIT_WIDTH = 40
    EXIT_HEIGHT = 64

    SCORE_OFFSET_X = WIDTH - 58 - 6  # window width - distance to the right - digit width 
    SCORE_OFFSET_Y = HEIGHT - 20 - 7  # window height - distance to the bottom - digit height 

    UI_OFFSET = 30  # pixels reserved for score at bottom
    PLAYER_BOUNDS = (
        (WALL_THICKNESS + WALL_OFFSET[0], WIDTH - WALL_THICKNESS - WALL_OFFSET[2]),
        (WALL_THICKNESS + WALL_OFFSET[1], HEIGHT - WALL_THICKNESS - WALL_OFFSET[3])
    )


class BerzerkState(NamedTuple):
    player_pos: chex.Array             # (2,)
    lives: chex.Array
    bullets: chex.Array                # (MAX_BULLETS, 2)
    bullet_dirs: chex.Array            # (MAX_BULLETS, 2)
    bullet_active: chex.Array          # (MAX_BULLETS,)
    enemy_pos: chex.Array              # (NUM_ENEMIES, 2)
    enemy_move_axis: chex.Array        # (NUM_ENEMIES,)
    enemy_move_dir: chex.Array         # (NUM_ENEMIES,)
    enemy_alive: chex.Array            # (NUM_ENEMIES,)
    enemy_bullets: chex.Array          # (NUM_ENEMIES, 2)
    enemy_bullet_dirs: chex.Array      # (NUM_ENEMIES, 2)
    enemy_bullet_active: chex.Array    # (NUM_ENEMIES,)
    enemy_move_prob: chex.Array        # (1,)
    last_dir: chex.Array               # (2,)
    rng: chex.PRNGKey
    score: chex.Array
    animation_counter: chex.Array
    enemy_animation_counter: chex.Array
    
class BerzerkObservation(NamedTuple):
    player: chex.Array
    bullets: chex.Array
    bullet_dirs: chex.Array
    bullet_active: chex.Array

class BerzerkInfo(NamedTuple):
    dummy: chex.Array  # placeholder (will be added later on)



class JaxBerzerk(JaxEnvironment[BerzerkState, BerzerkObservation, BerzerkInfo, BerzerkConstants]):
    def __init__(self, consts: BerzerkConstants = None, frameskip: int = 1, reward_funcs: list[callable]=None):
        super().__init__(consts)
        self.frameskip = frameskip
        self.frame_stack_size = 4
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
        self.consts = consts or BerzerkConstants()
        self.obs_size = 111
        self.renderer = BerzerkRenderer(self.consts)

    def is_moving_action(self, action):
        moving_actions = jnp.array([
        Action.UP, 
        Action.DOWN, 
        Action.LEFT, 
        Action.RIGHT,
        Action.UPLEFT, 
        Action.UPRIGHT, 
        Action.DOWNLEFT, 
        Action.DOWNRIGHT,
        Action.UPFIRE, 
        Action.DOWNFIRE, 
        Action.LEFTFIRE, 
        Action.RIGHTFIRE,
        Action.UPLEFTFIRE, 
        Action.UPRIGHTFIRE, 
        Action.DOWNLEFTFIRE, 
        Action.DOWNRIGHTFIRE
    ])
        return jnp.any(action == moving_actions)

    
    @partial(jax.jit, static_argnums=(0, ))
    def player_step(
        self, state: BerzerkState, action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        # implement all the possible movement directions for the player, the mapping is:
        # anything with left in it, add -1 to the x position
        # anything with right in it, add 1 to the x position
        # anything with up in it, add -1 to the y position
        # anything with down in it, add 1 to the y position
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        dx = jnp.where(right, 1, jnp.where(left, -1, 0))
        dy = jnp.where(down, 1, jnp.where(up, -1, 0))

        # movement scaled
        player_x = state.player_pos[0] + dx * self.consts.PLAYER_SPEED
        player_y = state.player_pos[1] + dy * self.consts.PLAYER_SPEED

        # set the direction according to the movement
        player_direction = jnp.where(
                (dx != 0) | (dy != 0),
                jnp.array([dx, dy]),
                state.last_dir
            )
        
        # perform out of bounds checks
        #player_x = jnp.where(
        #    player_x < PLAYER_BOUNDS[0][0],
        #    PLAYER_BOUNDS[0][0],  # Clamp to min player bound
        #    jnp.where(
        #        player_x > PLAYER_BOUNDS[0][1],
        #        PLAYER_BOUNDS[0][1],  # Clamp to max player bound
        #        player_x,
        #    ),
        #)

        #player_y = jnp.where(
        #    player_y < PLAYER_BOUNDS[1][0],
        #    PLAYER_BOUNDS[1][0],
        #    jnp.where(player_y > PLAYER_BOUNDS[1][1], PLAYER_BOUNDS[1][1], player_y),
        #)

        return player_x, player_y, player_direction


    @partial(jax.jit, static_argnums=0)
    def check_exit_crossing(self, player_pos: chex.Array) -> chex.Array:
        """Return True if player touches an exit region (centered on wall)."""
        x, y = player_pos[0], player_pos[1]

        # Top exit
        top = (self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 - self.consts.EXIT_WIDTH / 2,
            self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 + self.consts.EXIT_WIDTH / 2 - self.consts.PLAYER_SIZE[0])
        top_exit = (x > top[0]) & (x < top[1]) & (y < self.consts.PLAYER_BOUNDS[1][0])
        # Bottom exit
        bottom_exit = (x > top[0]) & (x < top[1]) & (y > self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_SIZE[1])

        # Left exit
        left = (self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 - self.consts.EXIT_HEIGHT / 2,
                self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 + self.consts.EXIT_HEIGHT / 2 - self.consts.PLAYER_SIZE[1])
        left_exit = (y > left[0]) & (y < left[1]) & (x < self.consts.PLAYER_BOUNDS[0][0])

        # Right exit
        right_exit = (y > left[0]) & (y < left[1]) & (x > self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_SIZE[0])

        return top_exit | bottom_exit | left_exit | right_exit


    @partial(jax.jit, static_argnums=(0, ))
    def update_enemies(self, player_pos, enemy_pos, enemy_axis, enemy_dir, rng, move_prob):
        enemy_rngs = jax.random.split(rng, self.consts.NUM_ENEMIES)

        def update_one_enemy(_, inputs):
            rng, pos, axis, dir_, prob = inputs
            new_pos, new_axis, new_dir, _ = self.update_enemy_position(
                player_pos, pos, axis, dir_, rng, prob
            )
            return None, (new_pos, new_axis, new_dir)

        _, (positions, axes, dirs) = jax.lax.scan(
            update_one_enemy,
            None,
            (enemy_rngs, enemy_pos, enemy_axis, enemy_dir, move_prob)
        )

        return positions, axes, dirs


    @partial(jax.jit, static_argnums=(0, ))
    def update_enemy_position(self, player_pos: chex.Array, enemy_pos: chex.Array, 
                            enemy_move_axis: chex.Array, enemy_move_dir: chex.Array,
                            rng: chex.PRNGKey, move_prob: float):
        """
        Update enemy position with movement probability.
        Once started moving, continues until aligned with player.
        
        Args:
            enemy_move_axis: 0 for x-axis, 1 for y-axis movement
            enemy_move_dir: 1 for positive, -1 for negative direction
        """
        rng, move_rng = jax.random.split(rng)
        
        # Check if already moving (axis != -1)
        is_moving = enemy_move_axis != -1
        
        # If not moving, decide whether to start moving
        start_moving = jax.random.bernoulli(move_rng, move_prob)
        should_move = is_moving | (~is_moving & start_moving)
        
        def start_new_movement(_):
            # Choose random axis (0=x, 1=y) and direction (1 or -1)
            rng1, rng2 = jax.random.split(move_rng)
            axis = jax.random.bernoulli(rng1, 0.5).astype(jnp.int32)  # 0 or 1
            dir_ = jnp.where(player_pos[axis] > enemy_pos[axis], 1, -1)
            return axis, dir_
        
        # If not moving but should start, initialize movement
        new_axis, new_dir = jax.lax.cond(
            ~is_moving & should_move,
            start_new_movement,
            lambda _: (enemy_move_axis, enemy_move_dir),
            operand=None
        )
        
        # Update position if moving
        new_pos = jnp.where(
            new_axis == 0,  # moving in x-axis
            enemy_pos.at[0].add(new_dir * self.consts.ENEMY_SPEED),
            enemy_pos
        )
        new_pos = jnp.where(
            new_axis == 1,  # moving in y-axis
            new_pos.at[1].add(new_dir * self.consts.ENEMY_SPEED),
            new_pos
        )
        
        # Check if aligned with player in movement axis
        aligned_x = jnp.abs(player_pos[0] - new_pos[0]) < 5
        aligned_y = jnp.abs(player_pos[1] - new_pos[1]) < 5
        aligned = jax.lax.select(new_axis == 0, aligned_x, aligned_y)

        # Stop moving if aligned
        final_axis = jnp.where(aligned, -1, new_axis)
        final_dir = jnp.where(aligned, 0, new_dir)
        
        return new_pos, final_axis, final_dir, rng
        
    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: BerzerkState) -> BerzerkObservation:
        return BerzerkObservation(
            player=state.player_pos,
            bullets=state.bullets,
            bullet_dirs=state.bullet_dirs, 
            bullet_active=state.bullet_active
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BerzerkState) -> bool:
        return state.lives < 0
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BerzerkState) -> BerzerkInfo: # later as params: self, state: BerzerkState, all_rewards: chex.Array
        return BerzerkInfo(
            jnp.array(0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BerzerkObservation, BerzerkState]:
        pos = jnp.array([self.consts.WIDTH // 2, self.consts.HEIGHT // 2], dtype=jnp.float32)
        lives = jnp.array(2)
        bullets = jnp.zeros((self.consts.MAX_BULLETS, 2), dtype=jnp.float32)
        bullet_dirs = jnp.zeros((self.consts.MAX_BULLETS, 2), dtype=jnp.float32)
        active = jnp.zeros((self.consts.MAX_BULLETS,), dtype=bool)
        enemy_pos = jax.random.uniform( # TODO: Don't let enemies spawn in player
            rng, shape=(self.consts.NUM_ENEMIES, 2),
            minval=jnp.array([self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[1][0]]),
            maxval=jnp.array([self.consts.PLAYER_BOUNDS[0][1] - self.consts.ENEMY_SIZE[0], self.consts.PLAYER_BOUNDS[1][1] - self.consts.ENEMY_SIZE[1]])
        )
        enemy_move_axis = -jnp.ones((self.consts.NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_dir = jnp.zeros((self.consts.NUM_ENEMIES,), dtype=jnp.int32)
        enemy_alive = jnp.ones((self.consts.NUM_ENEMIES,), dtype=bool)
        enemy_bullets = jnp.zeros((self.consts.NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_dirs = jnp.zeros((self.consts.NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_active = jnp.zeros((self.consts.NUM_ENEMIES,), dtype=bool)
        enemy_move_prob = jnp.full((self.consts.NUM_ENEMIES,), self.consts.MOVEMENT_PROB, dtype=jnp.float32)
        last_dir = jnp.array([0.0, -1.0])  # default = up
        score = jnp.array(0, dtype=jnp.int32)
        animation_counter = jnp.array(0, dtype=jnp.int32)
        enemy_animation_counter = jnp.zeros((self.consts.NUM_ENEMIES,), dtype=jnp.int32)
        state = BerzerkState(player_pos=pos, 
                             lives=lives, 
                             bullets=bullets, bullet_dirs=bullet_dirs, 
                             bullet_active=active, 
                             enemy_pos=enemy_pos, 
                             enemy_move_axis=enemy_move_axis, 
                             enemy_move_dir=enemy_move_dir, 
                             enemy_alive=enemy_alive,
                             enemy_bullets=enemy_bullets,
                             enemy_bullet_dirs=enemy_bullet_dirs,
                             enemy_bullet_active=enemy_bullet_active,
                             enemy_move_prob=enemy_move_prob, 
                             last_dir=last_dir, 
                             rng=rng, 
                             score=score, 
                             animation_counter=animation_counter,
                             enemy_animation_counter=enemy_animation_counter)
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=0)
    def step(self, state: BerzerkState, action: chex.Array) -> Tuple[BerzerkObservation, BerzerkState, float, bool, BerzerkInfo]:
        # 1. Player movement
        player_x, player_y, move_dir = self.player_step(state, action)
        new_pos = jnp.array([player_x, player_y])

        moving = self.is_moving_action(action)

        animation_counter = jnp.where(
            moving,
            state.animation_counter + 1,
            0
        )

        _, reset_state = self.reset()

        def rects_overlap(pos_a, size_a, pos_b, size_b):
            left_a, top_a = pos_a
            right_a = pos_a[0] + size_a[0]
            bottom_a = pos_a[1] + size_a[1]

            left_b, top_b = pos_b
            right_b = pos_b[0] + size_b[0]
            bottom_b = pos_b[1] + size_b[1]

            overlap_x = (left_a < right_b) & (right_a > left_b)
            overlap_y = (top_a < bottom_b) & (bottom_a > top_b)
            return overlap_x & overlap_y

        def object_hits_enemy(object_pos, object_size, enemy_pos):
            object_left   = object_pos[0]
            object_right  = object_pos[0] + object_size[0]
            object_top    = object_pos[1]
            object_bottom = object_pos[1] + object_size[1]

            enemy_left   = enemy_pos[0]
            enemy_right  = enemy_pos[0] + self.consts.ENEMY_SIZE[0]
            enemy_top    = enemy_pos[1]
            enemy_bottom = enemy_pos[1] + self.consts.ENEMY_SIZE[1]

            overlap_x = (object_left < enemy_right) & (object_right > enemy_left)
            overlap_y = (object_top < enemy_bottom) & (object_bottom > enemy_top)

            return overlap_x & overlap_y

        # 1a. Check wallcollision and exit collision
        def object_hits_wall(object_pos, object_size):
            object_hits_wall = (
                ((object_pos[0] < self.consts.PLAYER_BOUNDS[0][0]) & (object_pos[0] > 0)) |
                ((object_pos[0] + object_size[0] > self.consts.PLAYER_BOUNDS[0][1]) & (object_pos[0] < self.consts.WIDTH)) |
                ((object_pos[1] < self.consts.PLAYER_BOUNDS[1][0]) & (object_pos[0] > 0)) |
                ((object_pos[1] + object_size[1] > self.consts.PLAYER_BOUNDS[1][1]) & (object_pos[1] < self.consts.HEIGHT))
            )

            return object_hits_wall
        
        hit_exit = self.check_exit_crossing(new_pos)
        hit_wall = object_hits_wall((player_x, player_y), self.consts.PLAYER_SIZE) & ~hit_exit

        # 2. Update position and direction of enemies
        rng, enemy_rng = jax.random.split(state.rng)

        updated_enemy_pos, updated_enemy_axis, updated_enemy_dir = self.update_enemies(
            new_pos, state.enemy_pos, state.enemy_move_axis, state.enemy_move_dir,
            enemy_rng, state.enemy_move_prob
        )

        enemy_moving = updated_enemy_axis != -1
        enemy_animation_counter = jnp.where(
            enemy_moving,
            state.enemy_animation_counter + 1,
            jnp.zeros_like(state.enemy_animation_counter)
        )

        # Only move living enemies
        updated_enemy_pos = jnp.where(state.enemy_alive[:, None], updated_enemy_pos, state.enemy_pos)
        updated_enemy_axis = jnp.where(state.enemy_alive, updated_enemy_axis, state.enemy_move_axis)
        updated_enemy_dir = jnp.where(state.enemy_alive, updated_enemy_dir, state.enemy_move_dir)

        enemy_hits_wall = jax.vmap(
            lambda enemy_pos: object_hits_wall(enemy_pos, self.consts.ENEMY_SIZE)
        )(updated_enemy_pos)

        # Enemies shoot
        def enemy_fire_logic(pos, alive, axis, dir_, rng):
            should_fire = jax.random.uniform(rng) < self.consts.ENEMY_SHOOT_PROB

            def direction_when_moving():
                # Bewegung entlang x (0) oder y (1)
                dx = jnp.where(axis == 0, dir_, 0)
                dy = jnp.where(axis == 1, dir_, 0)
                return jnp.array([dx, dy], dtype=jnp.float32)

            def direction_when_idle():
                # Zielt grob in Richtung Spieler
                delta = new_pos - pos  # Spieler - Gegner
                abs_dx = jnp.abs(delta[0])
                abs_dy = jnp.abs(delta[1])
                axis = jnp.where(abs_dx > abs_dy, 0, 1)
                dir_ = jnp.where(delta[axis] > 0, 1.0, -1.0)
                dx = jnp.where(axis == 0, dir_, 0.0)
                dy = jnp.where(axis == 1, dir_, 0.0)
                return jnp.array([dx, dy], dtype=jnp.float32)

            is_moving = axis != -1
            direction = jax.lax.cond(
                is_moving,
                direction_when_moving,
                direction_when_idle
            )

            return (
                pos,
                direction,
                should_fire & alive
            )

        enemy_rngs = jax.random.split(rng, self.consts.NUM_ENEMIES)
        enemy_bullets_new, dirs_new, active_new = jax.vmap(enemy_fire_logic)(
            updated_enemy_pos,
            state.enemy_alive,
            updated_enemy_axis,
            updated_enemy_dir,
            enemy_rngs
        )

        # Nur feuern, wenn nicht bereits aktiv
        enemy_bullets = jnp.where(
            ~state.enemy_bullet_active[:, None] & active_new[:, None],
            enemy_bullets_new,
            state.enemy_bullets
        )
        enemy_bullet_dirs = jnp.where(
            ~state.enemy_bullet_active[:, None] & active_new[:, None],
            dirs_new,
            state.enemy_bullet_dirs
        )
        enemy_bullet_active = state.enemy_bullet_active | active_new

        enemy_bullet_sizes = jax.vmap(
            lambda d: jax.lax.select(
                d[0] == 0,
                jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
            )
        )(enemy_bullet_dirs)

        enemy_bullets = enemy_bullets + enemy_bullet_dirs * self.consts.BULLET_SPEED * enemy_bullet_active[:, None]

        # Deaktiviere wenn außerhalb
        enemy_bullet_active = enemy_bullet_active & (
            (enemy_bullets[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
            (enemy_bullets[:, 0] + enemy_bullet_sizes[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
            (enemy_bullets[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
            (enemy_bullets[:, 1] + enemy_bullet_sizes[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
        )

        enemy_bullet_hits_player = jax.vmap(
            lambda b_pos, b_size: rects_overlap(b_pos, b_size, new_pos, self.consts.PLAYER_SIZE)
        )(enemy_bullets, enemy_bullet_sizes)

        hit_by_enemy_bullet = jnp.any(enemy_bullet_hits_player)

        # 3. Check collision of player with enemy
        player_pos = jnp.array([player_x, player_y])
        player_hits = jax.vmap(
            lambda enemy_pos: object_hits_enemy(player_pos, self.consts.PLAYER_SIZE, enemy_pos)
        )(updated_enemy_pos)
        
        hit_by_enemy = jnp.any(player_hits)
        hit_something = hit_by_enemy | hit_wall | hit_by_enemy_bullet
        lives_after = jnp.where(hit_something, state.lives - 1, state.lives)

        # 4. Shoot bullets of player (enemies can't shoot yet)
        def shoot_bullet(state):
            def try_spawn(i, carry):
                bullets, directions, active = carry
                return jax.lax.cond(
                    ~active[i],
                    lambda _: (
                        bullets.at[i].set(new_pos),
                        directions.at[i].set(move_dir),
                        active.at[i].set(True),
                    ),
                    lambda _: (bullets, directions, active),
                    operand=None
                )
            return jax.lax.fori_loop(0, self.consts.MAX_BULLETS, try_spawn, (state.bullets, state.bullet_dirs, state.bullet_active))

        is_shooting = jnp.any(jnp.array([
            action == Action.FIRE,
            action == Action.UPRIGHTFIRE,
            action == Action.UPLEFTFIRE,
            action == Action.DOWNFIRE,
            action == Action.DOWNRIGHTFIRE,
            action == Action.DOWNLEFTFIRE,
            action == Action.RIGHTFIRE,
            action == Action.LEFTFIRE,
            action == Action.UPFIRE,
        ]))

        bullets, bullet_dirs, bullet_active = jax.lax.cond(
            is_shooting,
            lambda _: shoot_bullet(state),
            lambda _: (state.bullets, state.bullet_dirs, state.bullet_active),
            operand=None
        )

        # 5. Move bullets
        # Choose bullet size (depending on direction)
        bullet_sizes = jax.vmap(
            lambda d: jax.lax.select(
                d[0] == 0,
                jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
            )
        )(bullet_dirs)
        
        bullets += bullet_dirs * self.consts.BULLET_SPEED * bullet_active[:, None]
        bullet_active = bullet_active & (
            (bullets[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) & # left
            (bullets[:, 0] + bullet_sizes[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) & # right
            (bullets[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) & # top
            (bullets[:, 1] + bullet_sizes[:, 1] <= self.consts.PLAYER_BOUNDS[1][1]) # bottom
        )

        # 6. Check collision of bullet and enemy
        def bullet_hits_enemy(bullet_pos, bullet_size, enemy_pos):
            return object_hits_enemy(bullet_pos, bullet_size, enemy_pos)
        
        # 6b. Check collision of enemy bullets with other enemies (friendly fire)
        def enemy_bullet_hits_enemy(bullet_pos, bullet_size, target_pos, shooter_pos):
            # Treffer, wenn Rechtecke überlappen UND nicht auf sich selbst schießen
            return rects_overlap(bullet_pos, bullet_size, target_pos, jnp.array(self.consts.ENEMY_SIZE, dtype=jnp.float32)) & ~jnp.all(target_pos == shooter_pos)

        enemy_friendly_fire_hits = jax.vmap(
            lambda bullet_pos, bullet_size, shooter_pos: jax.vmap(
                lambda target_pos: enemy_bullet_hits_enemy(bullet_pos, bullet_size, target_pos, shooter_pos)
            )(updated_enemy_pos)
        )(enemy_bullets, enemy_bullet_sizes, updated_enemy_pos)

        enemy_hit_by_friendly_fire = jnp.any(enemy_friendly_fire_hits, axis=0)  # (NUM_ENEMIES,)

        all_hits = jax.vmap(
            lambda enemy_pos: jax.vmap(
                lambda bullet, size: bullet_hits_enemy(bullet, size, enemy_pos)
            )(bullets, bullet_sizes)
        )(updated_enemy_pos)

        enemy_hit = jnp.any(all_hits, axis=1)
        enemy_alive = state.enemy_alive & ~enemy_hit & ~enemy_hits_wall & ~enemy_hit_by_friendly_fire

        bullet_vs_bullet_hits = jax.vmap(
            lambda b_pos, b_size, b_active: jax.vmap(
                lambda e_pos, e_size, e_active: 
                    rects_overlap(b_pos, b_size, e_pos, e_size) & b_active & e_active
            )(enemy_bullets, enemy_bullet_sizes, enemy_bullet_active)
        )(bullets, bullet_sizes, bullet_active)

        # Spieler-Schüsse, die eine Gegner-Bullet treffen
        player_bullet_hit_enemy_bullet = jnp.any(bullet_vs_bullet_hits, axis=1)  # (player_bullets,)
        # Gegner-Schüsse, die von Spieler-Bullets getroffen werden
        enemy_bullet_hit_by_player = jnp.any(bullet_vs_bullet_hits, axis=0)      # (enemy_bullets,)

        enemy_bullet_hit_enemy = jnp.any(enemy_friendly_fire_hits, axis=1)  # Shape: (NUM_ENEMIES,)
        bullet_active = bullet_active & ~player_bullet_hit_enemy_bullet
        enemy_bullet_active = enemy_bullet_active & ~enemy_bullet_hit_enemy

        score_after = jnp.where(
            jnp.any(enemy_hit | enemy_hits_wall),
            state.score + 50,
            state.score
        )

        # 7. For now simply teleport enemies out of area
        invisible = jnp.array([-100.0, -100.0])
        updated_enemy_pos = jnp.where(enemy_alive[:, None], updated_enemy_pos, invisible)

        updated_enemy_axis = jnp.where(enemy_alive, updated_enemy_axis, 0)

        # 8. Deactivate bullets on hit
        bullet_hit = jnp.any(all_hits, axis=0)
        bullet_active = bullet_active & ~bullet_hit

        # 9. New state
        new_state = BerzerkState(
            player_pos=new_pos,
            lives=lives_after,
            bullets=bullets,
            bullet_dirs=bullet_dirs,
            bullet_active=bullet_active,
            enemy_pos=updated_enemy_pos,
            enemy_move_axis=updated_enemy_axis,
            enemy_move_dir=updated_enemy_dir,
            enemy_alive=enemy_alive,
            enemy_bullets=enemy_bullets,
            enemy_bullet_dirs=enemy_bullet_dirs,
            enemy_bullet_active=enemy_bullet_active,
            enemy_move_prob = state.enemy_move_prob,
            last_dir=move_dir,
            rng=rng,
            score=score_after,
            animation_counter=animation_counter,
            enemy_animation_counter=enemy_animation_counter
        )

        # New level if exit reached
        reset_for_new_level = BerzerkState(
            player_pos=jnp.array([self.consts.WIDTH // 2, self.consts.HEIGHT // 2], dtype=jnp.float32),
            lives=state.lives,
            bullets=jnp.zeros_like(state.bullets),
            bullet_dirs=jnp.zeros_like(state.bullet_dirs),
            bullet_active=jnp.zeros_like(state.bullet_active),
            enemy_pos=jax.random.uniform(
                state.rng, shape=(self.consts.NUM_ENEMIES, 2),
                minval=jnp.array([self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[1][0]]),
                maxval=jnp.array([
                    self.consts.PLAYER_BOUNDS[0][1] - self.consts.ENEMY_SIZE[0],
                    self.consts.PLAYER_BOUNDS[1][1] - self.consts.ENEMY_SIZE[1]
                ])
            ),
            enemy_move_axis=-jnp.ones_like(state.enemy_move_axis),
            enemy_move_dir=jnp.zeros_like(state.enemy_move_dir),
            enemy_alive=jnp.ones_like(state.enemy_alive),
            enemy_bullets=jnp.zeros((self.consts.NUM_ENEMIES, 2), dtype=jnp.float32),
            enemy_bullet_dirs=jnp.zeros((self.consts.NUM_ENEMIES, 2), dtype=jnp.float32),
            enemy_bullet_active=jnp.zeros((self.consts.NUM_ENEMIES,), dtype=bool),
            enemy_move_prob=state.enemy_move_prob,
            last_dir=jnp.array([0.0, -1.0], dtype=jnp.float32),
            rng=rng,
            score=score_after,
            animation_counter=jnp.zeros_like(state.animation_counter),
            enemy_animation_counter=jnp.zeros_like(state.enemy_animation_counter)
        )

        _, reset_state = self.reset(rng)

        jax.debug.print("EXIT: {}", hit_exit)
        new_state = jax.lax.cond(
            hit_exit,
            lambda: reset_for_new_level,
            lambda: jax.lax.cond(
                hit_something,
                lambda: reset_state._replace(lives=state.lives - 1, score=score_after),  # Tod
                lambda: new_state
            )
        )

        # 10. Observation + Info + Reward/Done
        observation = self._get_observation(new_state)
        info = self._get_info(new_state)
        reward = 0.0
        done = jnp.equal(lives_after, -1) 
        
        jax.debug.print("Leben: {}", lives_after)

        return observation, new_state, reward, done, info
    

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
         return spaces.Dict(
            {
            "player": spaces.Box(0, 255, (2,), jnp.float32),
            "bullets": spaces.Box(0, 255, (self.consts.MAX_BULLETS, 2), jnp.float32),
            "bullet_active": spaces.Box(0, 1, (self.consts.MAX_BULLETS,), jnp.bool_),
            }
        )


class BerzerkRenderer(JAXGameRenderer):
    # Type hint for sprites dictionary
    sprites: Dict[str, Any]
    pivots: Dict[str, Any]

    def __init__(self, consts=None):
        """
        Initializes the renderer by loading sprites, including level backgrounds.

        Args:
            sprite_path: Path to the directory containing sprite .npy files.
        """
        self.consts = consts or BerzerkConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/berzerk"
        self.sprites, self.pivots = self._load_sprites()

    def _load_sprites(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads all necessary sprites from .npy files and returns (padded sprites, render offsets)."""
        sprites: Dict[str, Any] = {}
        pad_offsets: Dict[str, Any] = {}

        def _load_sprite_frame(name: str) -> chex.Array:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim == 2:
                frame = jnp.stack([frame]*3, axis=-1)  # grayscale → RGB
            if frame.shape[-1] == 3:
                frame = jnp.pad(frame, ((0, 0), (0, 0), (0, 1)))  # RGB → RGBA
            return frame.astype(jnp.uint8)

        # Sprites to load
        sprite_names = [
            'player_idle', 'player_move_1', 'player_move_2',
            'enemy_idle_1', 'enemy_move_horizontal_1', 'enemy_move_horizontal_2','level_outer_walls',
            'bullet_horizontal', 'bullet_vertical'
        ]
        for name in sprite_names:
            sprites[name] = _load_sprite_frame(name)

        score_digit_path = os.path.join(self.sprite_path, 'score_{}.npy')
        digits = jr.load_and_pad_digits(score_digit_path, num_chars=10)
        sprites['digits'] = digits

        # Add padding to player sprites for same size
        player_keys = ['player_idle', 'player_move_1', 'player_move_2']
        player_frames = [sprites[k] for k in player_keys]

        player_sprites_padded, player_offsets = jr.pad_to_match(player_frames)
        for i, key in enumerate(player_keys):
            sprites[key] = jnp.expand_dims(player_sprites_padded[i], axis=0)
            pad_offsets[key] = player_offsets[i]

        # Expand other sprites
        for key in sprites.keys():
            if key not in player_keys:
                if isinstance(sprites[key], (list, tuple)):
                    sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
                else:
                    sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites, pad_offsets


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

        # Draw walls (assuming fixed positions based on bounds)
        wall_sprite = jr.get_sprite_frame(self.sprites['level_outer_walls'], 0)
        raster = jr.render_at(raster, self.consts.WALL_OFFSET[0], self.consts.WALL_OFFSET[1], wall_sprite)


        # Draw bullets
        for i in range(state.bullets.shape[0]):
            is_active = state.bullet_active[i]
            bullet_pos = state.bullets[i]
            bullet_dir = state.bullet_dirs[i]

            def draw_bullet(raster):
                dx = bullet_dir[0]

                type_idx = jax.lax.select(dx != 0, 0, 1)  # 0=horizontal, 1=vertical

                def render_horizontal(r):
                    sprite = jr.get_sprite_frame(self.sprites['bullet_horizontal'], 0)
                    return jr.render_at(r, bullet_pos[0], bullet_pos[1], sprite)

                def render_vertical(r):
                    sprite = jr.get_sprite_frame(self.sprites['bullet_vertical'], 0)
                    return jr.render_at(r, bullet_pos[0], bullet_pos[1], sprite)

                return jax.lax.switch(
                    type_idx,
                    [render_horizontal, render_vertical],
                    raster
                )

            raster = jax.lax.cond(is_active, draw_bullet, lambda r: r, raster)

                # Draw enemy bullets
        for i in range(state.enemy_bullets.shape[0]):
            is_active = state.enemy_bullet_active[i]
            bullet_pos = state.enemy_bullets[i]
            bullet_dir = state.enemy_bullet_dirs[i]

            def draw_enemy_bullet(raster):
                dx = bullet_dir[0]

                type_idx = jax.lax.select(dx != 0, 0, 1)  # 0=horizontal, 1=vertical

                def render_horizontal(r):
                    sprite = jr.get_sprite_frame(self.sprites['bullet_horizontal'], 0)
                    return jr.render_at(r, bullet_pos[0], bullet_pos[1], sprite)

                def render_vertical(r):
                    sprite = jr.get_sprite_frame(self.sprites['bullet_vertical'], 0)
                    return jr.render_at(r, bullet_pos[0], bullet_pos[1], sprite)

                return jax.lax.switch(
                    type_idx,
                    [render_horizontal, render_vertical],
                    raster
                )

            raster = jax.lax.cond(is_active, draw_enemy_bullet, lambda r: r, raster)


        # Draw player animation
        def get_player_sprite():
            return jax.lax.cond(
                state.animation_counter == 0,
                lambda: self.sprites['player_idle'],
                lambda: jax.lax.switch(
                    (state.animation_counter - 1) % 12,
                    [
                        lambda: self.sprites['player_move_1'],
                        lambda: self.sprites['player_move_1'],
                        lambda: self.sprites['player_move_1'],
                        lambda: self.sprites['player_move_1'],
                        lambda: self.sprites['player_move_2'],
                        lambda: self.sprites['player_move_2'],
                        lambda: self.sprites['player_move_2'],
                        lambda: self.sprites['player_move_2'],
                        lambda: self.sprites['player_idle'],
                        lambda: self.sprites['player_idle'],
                        lambda: self.sprites['player_idle'],
                        lambda: self.sprites['player_idle']
                    ]
                )
            )

        player_sprite = get_player_sprite()
        raster = jr.render_at(raster, state.player_pos[0], state.player_pos[1], jr.get_sprite_frame(player_sprite, 0))


        def get_enemy_sprite(i):
            counter = state.enemy_animation_counter[i]

            return jax.lax.cond(
                counter == 0,
                lambda: self.sprites["enemy_idle_1"],
                lambda: jax.lax.switch(
                    (counter - 1) % 28,
                    [
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_1"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        lambda: self.sprites["enemy_move_horizontal_2"],
                        
                        ]
                    
                )
            )

        for i in range(state.enemy_pos.shape[0]):
            sprite = get_enemy_sprite(i)
            raster = jr.render_at(raster, state.enemy_pos[i][0], state.enemy_pos[i][1], jr.get_sprite_frame(sprite, 0))



        # Draw score
        score_spacing = 8  # Spacing between digits 
        max_score_digits = 5  # Maximal displayed digits

        digit_sprites_raw = self.sprites.get('digits', None)
        digit_sprites = (
            jnp.squeeze(digit_sprites_raw, axis=0)  # from (1,10,H,W,C) → (10,H,W,C)
            if digit_sprites_raw is not None else None
        )

        def render_scores(raster_to_update):
            """
            Render the score on the screen for Berzerk.

            Args:
                raster_to_update: The current frame to update.
            """

            # Convert score to digits, zero-padded (e.g. 50 -> [0,0,5,0])
            score_digits = jr.int_to_digits(state.score, max_digits=max_score_digits)

            # Remove leading zeros dynamically
            def find_start_index(digits):
                # Return the first non-zero index (or max_digits-1 if score == 0)
                is_non_zero = digits != 0
                first_non_zero = jnp.argmax(is_non_zero)
                return jax.lax.select(jnp.any(is_non_zero), first_non_zero, max_score_digits - 1)

            start_idx = find_start_index(score_digits)

            # Number of digits to render
            num_to_render = max_score_digits - start_idx

            # Adjust x-position to align right
            render_start_x = self.consts.SCORE_OFFSET_X - score_spacing * (num_to_render - 1)

            # Render selective digits
            raster_updated = jr.render_label_selective(
                raster_to_update,
                render_start_x,
                self.consts.SCORE_OFFSET_Y,
                score_digits,
                digit_sprites,
                start_idx,
                num_to_render,
                spacing=score_spacing
            )

            return raster_updated

        raster = jax.lax.cond(
            digit_sprites is not None,
            render_scores,
            lambda r: r,
            raster
        )

        return raster

# TODO: Refactor input
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