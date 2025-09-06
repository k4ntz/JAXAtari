from turtle import width
from typing import NamedTuple, Tuple, Optional, Dict, Any
from functools import partial
import jax
import jax.numpy as jnp
import chex
from numpy import logical_or
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

    EXTRA_LIFE_AT = 1000

    ENEMY_SIZE = (8, 16)
    NUM_ENEMIES = 7
    MIN_NUM_ENEMIES = 5
    MOVEMENT_PROB = 0.0025  # Value for testing, has to be adjusted
    ENEMY_SPEED = 0.05
    ENEMY_SHOOT_PROB = 0.005
    ENEMY_BULLET_SPEED = 0.235

    BULLET_SIZE_HORIZONTAL = (4, 2)
    BULLET_SIZE_VERTICAL = (1, 6)
    BULLET_SPEED = 1
    MAX_BULLETS = 1

    WALL_THICKNESS = 4
    WALL_OFFSET = (4, 4, 4, 30) # left, top, right, bottom
    EXIT_WIDTH = 40
    EXIT_HEIGHT = 64

    DEATH_ANIMATION_FRAMES = 256
    ENEMY_DEATH_ANIMATION_FRAMES = 16
    
    TRANSITION_ANIMATION_FRAMES = 128

    GAME_OVER_FRAMES = 64

    SCORE_OFFSET_X = WIDTH - 58 - 6  # window width - distance to the right - digit width 
    SCORE_OFFSET_Y = HEIGHT - 20 - 7  # window height - distance to the bottom - digit height 

    UI_OFFSET = 30  # pixels reserved for score at bottom
    PLAYER_BOUNDS = (
        (WALL_THICKNESS + WALL_OFFSET[0], WIDTH - WALL_THICKNESS - WALL_OFFSET[2]),
        (WALL_THICKNESS + WALL_OFFSET[1], HEIGHT - WALL_THICKNESS - WALL_OFFSET[3])
    )

    EVIL_OTTO_SIZE = (8, 7)
    EVIL_OTTO_SPEED = 0.2
    EVIL_OTTO_DELAY = 900
    


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
    death_timer: chex.Array
    room_counter: chex.Array
    entry_direction: chex.Array
    player_is_firing: chex.Array
    room_transition_timer: chex.Array
    enemy_clear_bonus_given: chex.Array
    extra_life_counter: chex.Array
    enemy_death_timer: chex.Array
    enemy_death_pos: chex.Array
    game_over_timer: chex.Array
    num_enemies: chex.Array
    otto_pos: chex.Array        # (2,)
    otto_active: bool
    otto_timer: int
    otto_anim_counter: int


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
    ])
        return jnp.any(action == moving_actions)


    @staticmethod
    def get_room_index(room_num):
            def get_random_index(room_num):
                prev = (room_num - 1) % 3
                offset = room_num + 1
                next_idx = (prev + offset) % 3
                return next_idx + 1
            return jax.lax.cond(
                room_num == 0,
                lambda: jnp.array(0, dtype=jnp.int32),
                lambda: jnp.array(get_random_index(room_num), dtype=jnp.int32)
            )
    
    @staticmethod
    def get_enemy_bullet_speed(level: jnp.ndarray) -> jnp.ndarray:
        base_bullet_speed = BerzerkConstants.ENEMY_BULLET_SPEED
        bullet_speed_increment = 0.065  # anpassen nach Bedarf

        # capped_level = min(level, 13)
        capped_level = jnp.minimum(level + 1, 13)

        # Schrittzahl (2,3 = 0; 4,5 = 1; …; 12,13 = 5)
        step = (capped_level - 1) // 2
        step = jnp.maximum(step, 0)

        value = base_bullet_speed + step * bullet_speed_increment

        return value

    @staticmethod
    def get_enemy_speed(level: jnp.ndarray) -> jnp.ndarray:
        base_enemy_speed = BerzerkConstants.ENEMY_SPEED
        enemy_speed_increment = 0.007  # anpassen nach Bedarf

        # Schrittzahl berechnen, rotiert nach 8 Stufen
        step = ((level + 1) // 2) % 8

        return base_enemy_speed + step * enemy_speed_increment

    
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
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                ]
            )
        )
        

        dx = jnp.where(right, 1, jnp.where(left, -1, 0))
        dy = jnp.where(down, 1, jnp.where(up, -1, 0))


        # movement scaled
        player_x = state.player_pos[0] + dx * self.consts.PLAYER_SPEED
        player_y = state.player_pos[1] + dy * self.consts.PLAYER_SPEED



        player_direction = jnp.select(
            [
                action == Action.UPFIRE,
                action == Action.DOWNFIRE,
                action == Action.LEFTFIRE,
                action == Action.RIGHTFIRE,
                action == Action.UP,
                action == Action.DOWN,
                action == Action.LEFT,
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.UPLEFT,
                action == Action.DOWNRIGHT,
                action == Action.DOWNLEFT,
                action == Action.UPRIGHTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNRIGHTFIRE,
                action == Action.DOWNLEFTFIRE,
            ],
            [
                jnp.array([0, -1]),   # UPFIRE
                jnp.array([0, 1]),    # DOWNFIRE
                jnp.array([-1, 0]),   # LEFTFIRE
                jnp.array([1, 0]),    # RIGHTFIRE
                jnp.array([0, -1]),   # UP
                jnp.array([0, 1]),    # DOWN
                jnp.array([-1, 0]),   # LEFT
                jnp.array([1, 0]),    # RIGHT
                jnp.array([1, -1]),   # UPRIGHT
                jnp.array([-1, -1]),  # UPLEFT
                jnp.array([1, 1]),    # DOWNRIGHT
                jnp.array([-1, 1]),   # DOWNLEFT
                jnp.array([1, -1]),   # UPRIGHTFIRE
                jnp.array([-1, -1]),  # UPLEFTFIRE
                jnp.array([1, 1]),    # DOWNRIGHTFIRE
                jnp.array([-1, 1]),   # DOWNLEFTFIRE
            ],
        default=state.last_dir
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
    def object_hits_enemy(self, object_pos, object_size, enemy_pos):
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
    @partial(jax.jit, static_argnums=0)
    def object_hits_wall(self, object_pos, object_size, room_counter, entry_direction, num_points_per_side=10):
        # Aktuelle Raum-ID (0–3 → mid_walls_1 bis _4)
        room_idx = JaxBerzerk.get_room_index(room_counter)

        # Hole passende Maske (True = Kollision)
        def load_mask(idx):
            return jax.lax.switch(idx, [
                lambda: self.renderer.room_collision_masks['mid_walls_1'],
                lambda: self.renderer.room_collision_masks['mid_walls_2'],
                lambda: self.renderer.room_collision_masks['mid_walls_3'],
                lambda: self.renderer.room_collision_masks['mid_walls_4'],
            ])
        
        # Hole Basismasken
        mid_mask = load_mask(room_idx)
        outer_mask = self.renderer.room_collision_masks['level_outer_walls']

        # Hole Türmasken
        left_mask = self.renderer.room_collision_masks['door_vertical_left']
        right_mask = self.renderer.room_collision_masks['door_vertical_right']
        top_mask = self.renderer.room_collision_masks['door_horizontal_up']
        bottom_mask = self.renderer.room_collision_masks['door_horizontal_down']

        # entry_direction: 0=oben, 1=unten, 2=links, 3=rechts
        entry = entry_direction

        # Dynamisch festlegen, welche Türen "geschlossen" (= blockieren) sind
        block_left   = (entry == 2) | (entry == 3)
        block_right  = (entry == 2) | (entry == 3)
        block_top    = (entry == 1)
        block_bottom = (entry == 0)

        # Nur diese Türmasken zulassen
        collision_mask = mid_mask | outer_mask
        collision_mask = jax.lax.cond(block_left,   lambda: collision_mask | left_mask,   lambda: collision_mask)
        collision_mask = jax.lax.cond(block_right,  lambda: collision_mask | right_mask,  lambda: collision_mask)
        collision_mask = jax.lax.cond(block_top,    lambda: collision_mask | top_mask,    lambda: collision_mask)
        collision_mask = jax.lax.cond(block_bottom, lambda: collision_mask | bottom_mask, lambda: collision_mask)


        mask_height, mask_width = collision_mask.shape

        # Check Kollision: alle Eckpunkte
        def point_hits(px, py):
            i = jnp.floor(py).astype(jnp.int32)
            j = jnp.floor(px).astype(jnp.int32)
            in_bounds = (i >= 0) & (i < mask_height) & (j >= 0) & (j < mask_width)
            return jax.lax.select(in_bounds, collision_mask[i, j], False)

        x0, y0 = object_pos
        w, h = object_size
        top_edge = [(x0 + dx, y0) for dx in jnp.linspace(0, w, num_points_per_side)]
        right_edge = [(x0 + w, y0 + dy) for dy in jnp.linspace(0, h, num_points_per_side)]
        bottom_edge = [(x0 + dx, y0 + h) for dx in jnp.linspace(w, 0, num_points_per_side)]
        left_edge = [(x0, y0 + dy) for dy in jnp.linspace(h, 0, num_points_per_side)]

        all_edge_points = top_edge + right_edge + bottom_edge + left_edge
        return jnp.any(jnp.array([point_hits(x, y) for x, y in all_edge_points]))
    
    @partial(jax.jit, static_argnums=0)
    def check_exit_crossing(self, player_pos: chex.Array) -> chex.Array:
        """Return True if player touches an exit region (centered on wall)."""
        x, y = player_pos[0], player_pos[1]

        # Top exit
        top = (self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 - self.consts.EXIT_WIDTH / 2,
            self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 + self.consts.EXIT_WIDTH / 2 - self.consts.PLAYER_SIZE[0])
        top_exit = (x > top[0]) & (x < top[1]) & (y < self.consts.PLAYER_BOUNDS[1][0] - self.consts.WALL_THICKNESS)
        # Bottom exit
        bottom_exit = (x > top[0]) & (x < top[1]) & (y > self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_SIZE[1] + self.consts.WALL_THICKNESS)

        # Left exit
        left = (self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 - self.consts.EXIT_HEIGHT / 2,
                self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 + self.consts.EXIT_HEIGHT / 2 - self.consts.PLAYER_SIZE[1])
        left_exit = (y > left[0]) & (y < left[1]) & (x < self.consts.PLAYER_BOUNDS[0][0] - self.consts.WALL_THICKNESS)

        # Right exit
        right_exit = (y > left[0]) & (y < left[1]) & (x > self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_SIZE[0] + self.consts.WALL_THICKNESS)

        return top_exit | bottom_exit | left_exit | right_exit
    
    @partial(jax.jit, static_argnums=(0,))
    def get_exit_direction(self, player_pos: chex.Array) -> jnp.ndarray:
        """Returns direction index: 0=top, 1=bottom, 2=left, 3=right, -1=none"""
        x, y = player_pos[0], player_pos[1]
        top = (self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 - self.consts.EXIT_WIDTH / 2,
            self.consts.PLAYER_BOUNDS[0][0] + (self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_BOUNDS[0][0]) / 2 + self.consts.EXIT_WIDTH / 2 - self.consts.PLAYER_SIZE[0])
        left = (self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 - self.consts.EXIT_HEIGHT / 2,
                self.consts.PLAYER_BOUNDS[1][0] + (self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_BOUNDS[1][0]) / 2 + self.consts.EXIT_HEIGHT / 2 - self.consts.PLAYER_SIZE[1])
        
        return jax.lax.select(
            (x > top[0]) & (x < top[1]) & (y < self.consts.PLAYER_BOUNDS[1][0]), jnp.int32(0),
            jax.lax.select(
                (x > top[0]) & (x < top[1]) & (y > self.consts.PLAYER_BOUNDS[1][1] - self.consts.PLAYER_SIZE[1]), jnp.int32(1),
                jax.lax.select(
                    (y > left[0]) & (y < left[1]) & (x < self.consts.PLAYER_BOUNDS[0][0]), jnp.int32(2),
                    jax.lax.select(
                        (y > left[0]) & (y < left[1]) & (x > self.consts.PLAYER_BOUNDS[0][1] - self.consts.PLAYER_SIZE[0]), jnp.int32(3),
                        jnp.int32(-1)
                    )
                )
            )
        )



    @partial(jax.jit, static_argnums=(0, ))
    def update_enemies(self, player_pos, enemy_pos, enemy_axis, enemy_dir, rng, move_prob, room_counter):
        enemy_rngs = jax.random.split(rng, self.consts.NUM_ENEMIES)

        def update_one_enemy(_, inputs):
            rng, pos, axis, dir_, prob = inputs
            new_pos, new_axis, new_dir, _ = self.update_enemy_position(
                player_pos, pos, axis, dir_, rng, prob, enemy_pos, room_counter
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
                            rng: chex.PRNGKey, move_prob: float, all_enemy_pos: chex.Array, room_counter: chex.Array):
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
        
         # Bewegungsvektor berechnen
        move_vec = jnp.array([
            jnp.where(new_axis == 0, new_dir * self.get_enemy_speed(room_counter), 0),
            jnp.where(new_axis == 1, new_dir * self.get_enemy_speed(room_counter), 0),
        ])
        proposed_pos = enemy_pos + move_vec


        # Prüfen ob er in einen anderen Gegner laufen würde (mit 3px Sicherheitsabstand)
        def too_close(ep):
            # simulierter Schritt + 5px extra Puffer in Bewegungsrichtung
            offset = jnp.array([
                jnp.where(new_axis == 0, new_dir * 5, 0),
                jnp.where(new_axis == 1, new_dir * 5, 0),
            ])
            future_pos = enemy_pos + move_vec + offset
            return (
                self.object_hits_enemy(future_pos, self.consts.ENEMY_SIZE, ep)
                & ~jnp.all(ep == enemy_pos)  # sich selbst ausschließen
            )
        overlap = jnp.any(jax.vmap(too_close)(all_enemy_pos))

        # Wenn overlap → gar nicht bewegen
        final_pos = jax.lax.select(overlap, enemy_pos, proposed_pos)

        # Bewegung stoppen wie bei Alignment
        stop_due_to_block = overlap

        # Prüfen, ob mit Spieler ausgerichtet → Bewegung stoppen
        aligned_x = jnp.abs(player_pos[0] - final_pos[0]) < 5
        aligned_y = jnp.abs(player_pos[1] - final_pos[1]) < 5
        aligned = jax.lax.select(new_axis == 0, aligned_x, aligned_y)

        # Bewegung stoppen entweder wenn blockiert ODER aligned
        final_axis = jnp.where(stop_due_to_block | aligned, -1, new_axis)
        final_dir  = jnp.where(stop_due_to_block | aligned, 0, new_dir)
        
        return final_pos, final_axis, final_dir, rng
    
    @partial(jax.jit, static_argnums=(0, ))
    def spawn_enemies(self, state, rng):
        # Gegneranzahl: 5–7
        rng, sub_num, sub_spawn = jax.random.split(rng, 3)
        num_enemies = jax.random.randint(sub_num, (), self.consts.MIN_NUM_ENEMIES, self.consts.NUM_ENEMIES+1)  # 8 exklusiv → 5, 6, 7

        # Alle Plätze initial leer
        placed_init = jnp.full((self.consts.NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)

        def sample_pos(r):
            return jax.random.uniform(
                r, shape=(2,),
                minval=jnp.array([self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[1][0]]),
                maxval=jnp.array([self.consts.PLAYER_BOUNDS[0][1] - self.consts.ENEMY_SIZE[0],
                                self.consts.PLAYER_BOUNDS[1][1] - self.consts.ENEMY_SIZE[1]])
            )

        def cond_fn(carry2):
            pos, rng2, attempts, placed = carry2
            in_wall = self.object_hits_wall(pos, self.consts.ENEMY_SIZE,
                                            state.room_counter, state.entry_direction)
            on_player = self.object_hits_enemy(state.player_pos, self.consts.PLAYER_SIZE, pos)
            overlap_enemy = jnp.any(jax.vmap(lambda ep: self.object_hits_enemy(pos, self.consts.ENEMY_SIZE, ep))(placed))
            invalid = in_wall | on_player | overlap_enemy
            return jnp.logical_and(invalid, attempts < 200)

        def body2(carry2):
            _, rng2, attempts, placed = carry2
            rng2, sub2 = jax.random.split(rng2)
            return sample_pos(sub2), rng2, attempts + 1, placed

        def body_fun(i, carry):
            placed, rng_inner = carry
            rng_inner, sub = jax.random.split(rng_inner)
            pos0 = sample_pos(sub)
            pos, rng_after, _, _ = jax.lax.while_loop(cond_fn, body2, (pos0, sub, jnp.int32(0), placed))
            placed = placed.at[i].set(pos)
            return (placed, rng_after)

        final_carry = jax.lax.fori_loop(0, num_enemies, body_fun, (placed_init, sub_spawn))
        placed_final, _ = final_carry
        enemy_alive = jnp.arange(self.consts.NUM_ENEMIES) < num_enemies
        return state._replace(enemy_pos=placed_final, enemy_alive=enemy_alive, num_enemies=num_enemies)


        
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
        pos = jnp.array([
            self.consts.PLAYER_BOUNDS[0][0] + 2,
            self.consts.PLAYER_BOUNDS[1][1] // 2
        ], dtype=jnp.float32)
        lives = jnp.array(2, dtype=jnp.int32)
        bullets = jnp.zeros((self.consts.MAX_BULLETS, 2), dtype=jnp.float32)
        bullet_dirs = jnp.zeros((self.consts.MAX_BULLETS, 2), dtype=jnp.float32)
        active = jnp.zeros((self.consts.MAX_BULLETS,), dtype=bool)

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
        death_timer = jnp.array(0, dtype=jnp.int32)
        room_counter = jnp.array(0, dtype=jnp.int32)
        entry_direction= jnp.array(3, dtype=jnp.int32)
        player_is_firing= jnp.array(False)
        room_transition_timer= jnp.array(0, dtype=jnp.int32)
        enemy_clear_bonus_given=jnp.array(False)
        extra_life_counter = jnp.array(0, dtype=jnp.int32)
        enemy_death_timer = jnp.zeros((self.consts.NUM_ENEMIES,), dtype=jnp.int32)
        enemy_death_pos = jnp.full((self.consts.NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        num_enemies = jnp.array(self.consts.NUM_ENEMIES, dtype=jnp.int32)
        game_over_timer = jnp.array(0, dtype=jnp.int32)

        enemy_pos = jnp.full((self.consts.NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)

        otto_pos = jnp.array([-100.0, -100.0], dtype=jnp.float32)
        otto_active = jnp.array(False)
        otto_timer = self.consts.EVIL_OTTO_DELAY
        otto_anim_counter = jnp.array(0, dtype=jnp.int32)


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
                             enemy_animation_counter=enemy_animation_counter,
                             death_timer=death_timer,
                             room_counter=room_counter,
                             entry_direction=entry_direction,
                             player_is_firing=player_is_firing,
                             room_transition_timer=room_transition_timer,
                             enemy_clear_bonus_given=enemy_clear_bonus_given,
                             extra_life_counter=extra_life_counter,
                             enemy_death_timer=enemy_death_timer,
                             enemy_death_pos=enemy_death_pos,
                             game_over_timer=game_over_timer,
                             num_enemies=num_enemies,
                             otto_pos=otto_pos,
                             otto_active=otto_active,
                             otto_timer=otto_timer,
                             otto_anim_counter=otto_anim_counter,
                            )
        
        state = self.spawn_enemies(state, jax.random.split(rng)[0])
        return self._get_observation(state), state
    
    @partial(jax.jit, static_argnums=0)
    def step(self, state: BerzerkState, action: chex.Array) -> Tuple[BerzerkObservation, BerzerkState, float, bool, BerzerkInfo]:
        # 0. Handle death animation phase
        is_dead = state.death_timer > 0
        death_timer = jnp.maximum(state.death_timer - 1, 0)

        def handle_death(_):
            # Timer runterzählen
            new_state = state._replace(death_timer=death_timer)

            # Nur wenn Tod vorbei, Leben verringern
            lives_after = jnp.where(death_timer == 0, state.lives - 1, state.lives)
            score_after = jnp.where(lives_after == -1, 0, state.score)

            # Basis-Update nach Tod
            base_state = state._replace(
                death_timer=0,
                lives=lives_after,
                score=score_after,
                entry_direction=3
            )

            # Wenn Todesanimation noch läuft → einfach weiter
            still_dying = death_timer > 0
            def during_death():
                return (
                    self._get_observation(new_state),
                    new_state,
                    0.0,
                    False,
                    self._get_info(new_state)
                )

            # Wenn Tod vorbei → entscheiden: Game Over oder Transition
            def after_death():
                base_state_with_timer = jax.lax.cond(
                    lives_after == -1,
                    # Game Over
                    lambda: base_state._replace(game_over_timer=self.consts.GAME_OVER_FRAMES),
                    # Normale Raum-Transition
                    lambda: base_state._replace(room_transition_timer=self.consts.TRANSITION_ANIMATION_FRAMES)
                )
                return (
                    self._get_observation(base_state_with_timer),
                    base_state_with_timer,
                    0.0,
                    False,
                    self._get_info(base_state_with_timer)
                )

            return jax.lax.cond(still_dying, during_death, after_death)


        game_over_active = state.game_over_timer > 0
        game_over_timer = jnp.maximum(state.game_over_timer - 1, 0)

        def handle_game_over(_):
            new_state = state._replace(game_over_timer=game_over_timer)
            return (
                self._get_observation(new_state),
                new_state,
                0.0,
                game_over_timer == 0,
                self._get_info(new_state),
            )


        room_transition_active = state.room_transition_timer > 0
        transition_timer = jnp.maximum(state.room_transition_timer - 1, 0)

        def handle_room_transition(_):
            new_state = state._replace(room_transition_timer=transition_timer)
            #jax.debug.print("{gi}",gi=new_state.room_transition_timer)
            def finished_transition():
                #TODO: Positions need to be changed to match original.
                player_spawn_pos = jax.lax.switch(
                    new_state.entry_direction,
                    [
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2,
                            self.consts.PLAYER_BOUNDS[1][1] - 25
                            ], dtype=jnp.float32),  # oben → unten spawnen
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2, 
                            self.consts.PLAYER_BOUNDS[1][0] + 5], dtype=jnp.float32),  # unten → oben spawnen
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] - 12, 
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                            ], dtype=jnp.float32),  # links → rechts
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32),  # rechts → links
                    ],
                    jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32)  # fallback
                )
                # Neues Level laden (ähnlich wie beim Tod)
                new_rng = jax.random.split(state.rng)[1]
                obs, base_state = self.reset(new_rng)
                base_state = base_state._replace(
                    player_pos=player_spawn_pos,
                    room_counter=state.room_counter + 1,
                    lives=state.lives,
                    score=state.score,
                    entry_direction=state.entry_direction,
                    extra_life_counter=state.extra_life_counter
                )

                # Jetzt Gegner spawnen mit den neuen Werten
                next_state = self.spawn_enemies(base_state, jax.random.split(new_rng)[1])

                return (
                    self._get_observation(next_state),
                    next_state,
                    0.0,
                    False,
                    self._get_info(next_state),
                )

            def in_transition():
                return (
                    self._get_observation(new_state),
                    new_state,
                    0.0,
                    False,
                    self._get_info(new_state),
                )

            return jax.lax.cond(
                transition_timer == 0,
                finished_transition,
                in_transition
            )


        def handle_normal(_):
            # 1. Player movement
            player_x, player_y, move_dir = self.player_step(state, action)

            new_pos = jnp.array([player_x, player_y])

            moving = self.is_moving_action(action)

            animation_counter = jnp.where(
                moving,
                state.animation_counter + 1,
                0
            )

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

            

            
            hit_exit = self.check_exit_crossing(new_pos)

            otto_active = jnp.where(hit_exit, False, state.otto_active)
            new_otto_timer = jnp.where(hit_exit, self.consts.EVIL_OTTO_DELAY, state.otto_timer)
            otto_pos = jnp.where(hit_exit, jnp.array([-100.0, -100.0], dtype=jnp.float32), state.otto_pos)

            transition_timer = jnp.where(hit_exit, self.consts.TRANSITION_ANIMATION_FRAMES, state.room_transition_timer)
            entry_direction = jnp.where(hit_exit, self.get_exit_direction(new_pos), state.entry_direction)
            hit_wall = self.object_hits_wall((player_x, player_y), self.consts.PLAYER_SIZE, state.room_counter, state.entry_direction) & ~hit_exit
            #jax.debug.print("Exzr: {hit_exit}", hit_exit=hit_exit)
            #jax.debug.print("Room: {new_room_counter}", new_room_counter=state.room_counter)
            # 2. Update position and direction of enemies
            rng, enemy_rng = jax.random.split(state.rng)

            updated_enemy_pos, updated_enemy_axis, updated_enemy_dir = self.update_enemies(
                new_pos, state.enemy_pos, state.enemy_move_axis, state.enemy_move_dir,
                enemy_rng, state.enemy_move_prob, state.room_counter
            )

            #enemy_moving = updated_enemy_axis != -1
            #enemy_animation_counter = jnp.where(
            #    enemy_moving,
            #    state.enemy_animation_counter + 1,
            #    jnp.zeros_like(state.enemy_animation_counter)
            #)
            enemy_animation_counter = state.enemy_animation_counter + 1

            # Only move living enemies
            updated_enemy_pos = jnp.where(state.enemy_alive[:, None], updated_enemy_pos, state.enemy_pos)
            updated_enemy_axis = jnp.where(state.enemy_alive, updated_enemy_axis, state.enemy_move_axis)
            updated_enemy_dir = jnp.where(state.enemy_alive, updated_enemy_dir, state.enemy_move_dir)

            enemy_hits_wall = jax.vmap(
                lambda enemy_pos: self.object_hits_wall(enemy_pos, self.consts.ENEMY_SIZE, state.room_counter, state.entry_direction)
            )(updated_enemy_pos)

            # Enemies shoot
            def enemy_fire_logic(pos, alive, axis, dir_, rng):
                # Nur schießen, wenn nicht in Bewegung
                is_moving = axis != -1

                aligned_x = jnp.abs(pos[0] - new_pos[0]) < 5
                aligned_y = jnp.abs(pos[1] - new_pos[1]) < 5
                aligned = aligned_x | aligned_y

                can_shoot = (~is_moving) & aligned & alive
                should_fire = jax.random.uniform(rng) < self.consts.ENEMY_SHOOT_PROB

                # Richtung in die geschossen wird (zur Spielerposition entlang Achse)
                dx = jnp.where(aligned_x, 0.0, jnp.sign(new_pos[0] - pos[0]))
                dy = jnp.where(aligned_y, 0.0, jnp.sign(new_pos[1] - pos[1]))
                direction = jnp.array([dx, dy], dtype=jnp.float32)

                # mögliche Richtungen
                dirs = jnp.array([
                    [0, -1],   # up
                    [1, 0],    # right
                    [0, 1],    # down
                    [-1, 0],   # left
                ], dtype=jnp.float32)

                # passende Offsets relativ zur Gegnerposition
                offsets = jnp.array([
                    [self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1] // 2 - 7],# up (Mitte oben)
                    [self.consts.ENEMY_SIZE[0], self.consts.ENEMY_SIZE[1] // 2],# right (rechts Mitte)
                    [0.0, self.consts.ENEMY_SIZE[1] // 2 + 2],# down (Mitte unten)
                    [0.0, self.consts.ENEMY_SIZE[1] // 2],                      # left (links Mitte)
                ], dtype=jnp.float32)

                # checke welche Richtung matched
                conds = jnp.all(dirs == direction[None, :], axis=1)

                # fallback Mitte, falls keine Richtung passt
                default_offset = jnp.array([self.consts.ENEMY_SIZE[0] // 2,
                                            self.consts.ENEMY_SIZE[1] // 2], dtype=jnp.float32)

                # wähle Offset
                offset = jnp.select(conds, offsets, default_offset)

                # Startposition des Schusses = Gegnerposition + Offset
                spawn_pos = pos + offset

                return (
                    spawn_pos,
                    direction,
                    should_fire & can_shoot
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
            can_shoot = (state.room_counter > 0)
            can_shoot_mask = jnp.broadcast_to(can_shoot, active_new.shape)  # gleiche Länge wie NUM_ENEMIES

            enemy_bullets = jnp.where(
                ~state.enemy_bullet_active[:, None] & active_new[:, None] & can_shoot_mask[:, None],
                enemy_bullets_new,
                state.enemy_bullets
            )

            enemy_bullet_dirs = jnp.where(
                ~state.enemy_bullet_active[:, None] & active_new[:, None] & can_shoot_mask[:, None],
                dirs_new,
                state.enemy_bullet_dirs
            )

            enemy_bullet_active = state.enemy_bullet_active | (active_new & can_shoot_mask)


            enemy_bullet_sizes = jax.vmap(
                lambda d: jax.lax.select(
                    d[0] == 0,
                    jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                    jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
                )
            )(enemy_bullet_dirs)

            enemy_bullets = enemy_bullets + enemy_bullet_dirs * self.get_enemy_bullet_speed(state.room_counter) * enemy_bullet_active[:, None]

            # Deaktiviere wenn außerhalb
            enemy_bullet_active = enemy_bullet_active & (
                (enemy_bullets[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
                (enemy_bullets[:, 0] + enemy_bullet_sizes[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
                (enemy_bullets[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
                (enemy_bullets[:, 1] + enemy_bullet_sizes[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
            )

            enemy_bullet_hits_player = jax.vmap(
                lambda b_pos, b_size, b_active: rects_overlap(b_pos, b_size, new_pos, self.consts.PLAYER_SIZE) & b_active
            )(enemy_bullets, enemy_bullet_sizes, enemy_bullet_active)

            hit_by_enemy_bullet = jnp.any(enemy_bullet_hits_player)

            # 3. Check collision of player with enemy
            player_pos = jnp.array([player_x, player_y])
            player_hits = jax.vmap(
                lambda enemy_pos: self.object_hits_enemy(player_pos, self.consts.PLAYER_SIZE, enemy_pos)
            )(updated_enemy_pos)

            otto_hits_player = rects_overlap(
                otto_pos, jnp.array(self.consts.EVIL_OTTO_SIZE, dtype=jnp.float32),
                new_pos, self.consts.PLAYER_SIZE
            )
            
            hit_by_enemy = jnp.any(player_hits)
            hit_something = hit_by_enemy | hit_wall | hit_by_enemy_bullet | otto_hits_player
            death_timer = jnp.where(hit_something, self.consts.DEATH_ANIMATION_FRAMES, state.death_timer)


            # 4. Shoot bullets of player (enemies can't shoot yet)
            def shoot_bullet(state):

                # alle gültigen Richtungen
                dirs = jnp.array([
                    [0, -1],   # up
                    [1, -1],   # upright
                    [1, 0],    # right
                    [1, 1],    # downright
                    [0, 1],    # down
                    [-1, 1],   # downleft
                    [-1, 0],   # left
                    [-1, -1],  # upleft
                ], dtype=jnp.int32)

                # passende Handpositionen (relativ zur Spieler-Mitte)
                offsets = jnp.array([
                    [self.consts.PLAYER_SIZE[0] // 2, 0.0],    # up
                    [self.consts.PLAYER_SIZE[0] // 2, 4.0],    # upright
                    [3.0, self.consts.PLAYER_SIZE[1] // 2 - 4],    # right
                    [self.consts.PLAYER_SIZE[0] // 2 + 1.0, self.consts.PLAYER_SIZE[1] - 10.0],   # downright
                    [self.consts.PLAYER_SIZE[0] // 2 + 2.0, self.consts.PLAYER_SIZE[1] - 10.0],    # down
                    [self.consts.PLAYER_SIZE[0] // 2 - 6.0, self.consts.PLAYER_SIZE[1] - 10.0],    # downleft
                    [-3.0, self.consts.PLAYER_SIZE[1] // 2 - 4],    # left
                    [self.consts.PLAYER_SIZE[0] -4.0 // 2 - 6.0, 4.0],     # upleft
                ], dtype=jnp.float32)

                # prüfe, welche Richtung aktiv ist (8 Bedingungen → shape (8,))
                conds = jnp.all(dirs == move_dir[None, :], axis=1)

                # fallback: wenn keine passt (z. B. move_dir = [0,0])
                default_offset = jnp.array([8.0, 8.0], dtype=jnp.float32)

                # wähle das passende offset (shape (2,))
                offset = jnp.select(conds, offsets, default_offset)

                # Startposition ist Spielerposition + Offset (Handposition)
                spawn_pos = new_pos + offset
                def try_spawn(i, carry):
                    bullets, directions, active = carry
                    return jax.lax.cond(
                        ~active[i],
                        lambda _: (
                            bullets.at[i].set(spawn_pos),
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

            player_is_firing = is_shooting.astype(jnp.bool_)

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
            # only 1 player bullet
            bullet_active = bullet_active & (~self.object_hits_wall(bullets[0], bullet_sizes[0], state.room_counter, state.entry_direction)) & (
                (bullets[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
                (bullets[:, 0] + bullet_sizes[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
                (bullets[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
                (bullets[:, 1] + bullet_sizes[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
            )

            # 6. Check collision of bullet and enemy
            def bullet_hits_enemy(bullet_pos, bullet_size, enemy_pos):
                return self.object_hits_enemy(bullet_pos, bullet_size, enemy_pos)
            
            # 6b. Check collision of enemy bullets with other enemies (friendly fire)
            def enemy_bullet_hits_enemy(bullet_pos, bullet_size, target_pos, shooter_pos, active):
                # Treffer, wenn Rechtecke überlappen UND nicht auf sich selbst schießen
                return rects_overlap(bullet_pos, bullet_size, target_pos, jnp.array(self.consts.ENEMY_SIZE, dtype=jnp.float32)) & active & ~jnp.all(target_pos == shooter_pos)

            enemy_friendly_fire_hits = jax.vmap(
                lambda bullet_pos, bullet_size, shooter_pos, active: jax.vmap(
                    lambda target_pos: enemy_bullet_hits_enemy(bullet_pos, bullet_size, target_pos, shooter_pos, active)
                )(updated_enemy_pos)
            )(enemy_bullets, enemy_bullet_sizes, updated_enemy_pos, enemy_bullet_active)

            enemy_hit_by_friendly_fire = jnp.any(enemy_friendly_fire_hits, axis=0)  # (NUM_ENEMIES,)

            all_hits = jax.vmap(
                lambda enemy_pos: jax.vmap(
                    lambda bullet, size: bullet_hits_enemy(bullet, size, enemy_pos)
                )(bullets, bullet_sizes)
            )(updated_enemy_pos)

            # Gegner-Gegner-Kollision
            enemy_touch_hits = jax.vmap(
                lambda pos_a, alive_a: jax.vmap(
                    lambda pos_b, alive_b: (
                        self.object_hits_enemy(pos_a, self.consts.ENEMY_SIZE, pos_b) &  # Kollision
                        ~jnp.all(pos_a == pos_b) &                                 # nicht derselbe Gegner
                        alive_a & alive_b                                          # beide lebendig
                    )
                )(updated_enemy_pos, state.enemy_alive)
            )(updated_enemy_pos, state.enemy_alive)

            # Reduziere zu einer "wird berührt"-Maske pro Gegner
            enemy_hit_enemy = jnp.any(enemy_touch_hits, axis=1)

            enemy_hit = jnp.any(all_hits, axis=1)
            enemy_alive = (
                state.enemy_alive
                & ~enemy_hit
                & ~enemy_hits_wall
                & ~enemy_hit_by_friendly_fire
                & ~enemy_hit_enemy
            )


            # Neue Todes-Timer setzen, wenn Gegner gerade getroffen wurden
            enemy_dies = enemy_hit | enemy_hits_wall | enemy_hit_by_friendly_fire | enemy_hit_enemy
            new_enemy_death_timer = jnp.where(enemy_dies, self.consts.ENEMY_DEATH_ANIMATION_FRAMES, state.enemy_death_timer)

            new_enemy_death_pos = jnp.where(enemy_dies[:, None], updated_enemy_pos, state.enemy_death_pos)

            # Timer herunterzählen
            enemy_death_timer_next = jnp.maximum(new_enemy_death_timer - 1, 0)

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
            enemy_bullet_active = enemy_bullet_active & ~enemy_bullet_hit_enemy & ~enemy_bullet_hit_by_player
            enemy_bullet_hits_wall = jax.vmap(
                lambda pos, size: self.object_hits_wall(pos, size, state.room_counter, state.entry_direction)
            )(enemy_bullets, enemy_bullet_sizes)

            enemy_bullet_active = enemy_bullet_active & (~enemy_bullet_hits_wall)

            give_bonus = (~jnp.any(enemy_alive)) & (~state.enemy_clear_bonus_given)
            bonus_score = jnp.where(give_bonus, state.num_enemies * 10, 0)

            # Maske aller toten Gegner in diesem Frame
            enemy_dies_mask = (
                enemy_hit |
                enemy_hits_wall |
                enemy_bullet_hit_enemy |
                hit_by_enemy |
                enemy_hit_enemy
            )



            # Punkte berechnen: 50 pro gestorbenem Gegner
            score_after = state.score + jnp.sum(enemy_dies_mask) * 50

            score_after += bonus_score
            enemy_clear_bonus_given = state.enemy_clear_bonus_given | give_bonus

            lives_after = state.lives

            extra_lives_given_last_score = state.extra_life_counter * self.consts.EXTRA_LIFE_AT
            give_extra_life = score_after >= extra_lives_given_last_score + self.consts.EXTRA_LIFE_AT

            lives_after = jnp.where(give_extra_life, state.lives + 1, state.lives)
            extra_life_counter_after = jnp.where(give_extra_life, state.extra_life_counter + 1, state.extra_life_counter)

            # 7. For now simply teleport enemies out of area
            invisible = jnp.array([-100.0, -100.0])
            updated_enemy_pos = jnp.where(enemy_alive[:, None], updated_enemy_pos, invisible)

            updated_enemy_axis = jnp.where(enemy_alive, updated_enemy_axis, 0)

            # 8. Deactivate bullets on hit
            bullet_hit = jnp.any(all_hits, axis=0)
            bullet_active = bullet_active & ~bullet_hit

            # 1. Timer runterzählen
            new_otto_timer = jnp.maximum(state.otto_timer - 1, 0)

            # 2. Spawn wenn Timer abgelaufen und noch nicht aktiv
            spawn_otto = jnp.logical_not(new_otto_timer) & jnp.logical_not(state.otto_active)

            otto_spawn_pos = jax.lax.switch(
                    state.entry_direction,
                    [
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2,
                            self.consts.PLAYER_BOUNDS[1][1] - 25
                            ], dtype=jnp.float32),  # oben → unten spawnen
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2, 
                            self.consts.PLAYER_BOUNDS[1][0] + 5], dtype=jnp.float32),  # unten → oben spawnen
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] - 12, 
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                            ], dtype=jnp.float32),  # links → rechts
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32),  # rechts → links
                    ],
                    jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32)  # fallback
                )



            spawn_pos = jnp.where(spawn_otto, otto_spawn_pos, state.otto_pos)

            otto_active = state.otto_active | spawn_otto

            # 3. Bewegung Richtung Spieler, wenn aktiv
            def move_otto(otto_pos, player_pos):
                direction = player_pos - otto_pos
                norm = jnp.linalg.norm(direction) + 1e-6
                new_otto_pos = otto_pos + (direction / norm) * self.consts.EVIL_OTTO_SPEED

                otto_animation_counter = state.otto_anim_counter + 1

                # Sprungbewegung: Otto wippt auf/ab
                jump_phase = (otto_animation_counter // 18) % 6  # 0 oder 1
                jump_offset = jnp.where(jump_phase == 0, 0.5, 
                                        jnp.where(jump_phase == 5, 0.8, 
                                                  jnp.where(jump_phase == 1, -0.7, -0.2)))  # springt hoch/runter
                otto_pos_with_jump = new_otto_pos.at[1].add(jump_offset)

                return otto_pos_with_jump
            
            

            otto_pos = jnp.where(otto_active, move_otto(spawn_pos, new_pos), spawn_pos)

            # 4. Animation Counter
            otto_anim_counter = jnp.where(otto_active, state.otto_anim_counter + 1, 0)

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
                enemy_animation_counter=enemy_animation_counter,
                death_timer=death_timer,
                room_counter=state.room_counter,
                entry_direction=entry_direction,
                player_is_firing=player_is_firing,
                room_transition_timer=transition_timer,
                enemy_clear_bonus_given=enemy_clear_bonus_given,
                extra_life_counter=extra_life_counter_after,
                enemy_death_timer=enemy_death_timer_next,
                enemy_death_pos=new_enemy_death_pos,
                game_over_timer=game_over_timer,
                num_enemies=state.num_enemies,
                otto_pos=otto_pos,
                otto_active=otto_active,
                otto_timer=new_otto_timer,
                otto_anim_counter=otto_anim_counter,
            )

            # 10. Observation + Info + Reward/Done
            observation = self._get_observation(new_state)
            info = self._get_info(new_state)
            reward = 0.0
            done = jnp.equal(state.lives, -1) 

            return observation, new_state, reward, done, info
        
        return jax.lax.cond(
            game_over_active,
            handle_game_over,
            lambda _: jax.lax.cond(
                is_dead,
                handle_death,
                lambda _: jax.lax.cond(
                    room_transition_active,
                    handle_room_transition,
                    handle_normal,
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )



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
        self.room_collision_masks = self._generate_room_collision_masks()

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
            'player_idle', 'player_move_1', 'player_move_2', 'player_death',
            'player_shoot_up', 'player_shoot_right', 'player_shoot_down',
            'player_shoot_left', 'player_shoot_up_left', 'player_shoot_down_left',
            'enemy_idle_1', 'enemy_idle_2', 'enemy_idle_3', 'enemy_idle_4', 'enemy_idle_5', 'enemy_idle_6', 'enemy_idle_7', 'enemy_idle_8',
            'enemy_death_1', 'enemy_death_2', 'enemy_death_3',
            'enemy_move_horizontal_1', 'enemy_move_horizontal_2',
            'enemy_move_vertical_1', 'enemy_move_vertical_2', 'enemy_move_vertical_3',
            'bullet_horizontal', 'bullet_vertical',
            'door_vertical_left', 'door_vertical_right',
            'door_horizontal_up', 'door_horizontal_down',
            'level_outer_walls', 'mid_walls_1', 'mid_walls_2', 'mid_walls_3', 'mid_walls_4',
            'life', 'start_title',
            'evil_otto', 'evil_otto_2'
        ]
        for name in sprite_names:
            sprites[name] = _load_sprite_frame(name)

        score_digit_path = os.path.join(self.sprite_path, 'score_{}.npy')
        digits = jr.load_and_pad_digits(score_digit_path, num_chars=10)
        sprites['digits'] = digits

        # Add padding to player sprites for same size
        player_keys = ['player_idle', 'player_move_1', 'player_move_2', 'player_death', 
                       'player_shoot_up', 'player_shoot_right', 'player_shoot_down',
                       'player_shoot_left', 'player_shoot_up_left', 'player_shoot_down_left']
        player_frames = [sprites[k] for k in player_keys]

        player_sprites_padded, player_offsets = jr.pad_to_match(player_frames)
        for i, key in enumerate(player_keys):
            sprites[key] = jnp.expand_dims(player_sprites_padded[i], axis=0)
            pad_offsets[key] = player_offsets[i]

        # Add padding to enemy sprites for same size
        def pad_and_store(keys: list[str]):
            enemy_frames = [sprites[k] for k in keys]
            padded_frames, offsets = jr.pad_to_match(enemy_frames)
            for i, key in enumerate(keys):
                sprites[key] = jnp.expand_dims(padded_frames[i], axis=0)
                pad_offsets[key] = offsets[i]

        enemy_keys = [
            'enemy_idle_1', 'enemy_idle_2', 'enemy_idle_3', 'enemy_idle_4',
            'enemy_idle_5', 'enemy_idle_6', 'enemy_idle_7', 'enemy_idle_8',
            'enemy_move_horizontal_1', 'enemy_move_horizontal_2',
            'enemy_move_vertical_1', 'enemy_move_vertical_2', 'enemy_move_vertical_3',
            'enemy_death_1', 'enemy_death_2', 'enemy_death_3',
        ]
        pad_and_store(enemy_keys)

        # Add padding to otto sprites for same size
        otto_keys = ['evil_otto', 'evil_otto_2']
        otto_frames = [sprites[k] for k in otto_keys]

        otto_sprites_padded, otto_offsets = jr.pad_to_match(otto_frames)
        for i, key in enumerate(otto_keys):
            sprites[key] = jnp.expand_dims(otto_sprites_padded[i], axis=0)
            pad_offsets[key] = otto_offsets[i]

        # Pad mid_walls sprites to same shape
        mid_keys = ['mid_walls_1', 'mid_walls_2', 'mid_walls_3', 'mid_walls_4', 'level_outer_walls', 
                    'door_vertical_left', 'door_vertical_right', 'door_horizontal_up', 'door_horizontal_down',]
        mid_frames = [sprites[k] for k in mid_keys]
        mid_padded, mid_offsets = jr.pad_to_match(mid_frames)
        for i, key in enumerate(mid_keys):
            sprites[key] = jnp.expand_dims(mid_padded[i], axis=0)
            pad_offsets[key] = mid_offsets[i]

        # Expand other sprites
        for key in sprites.keys():
            if key not in player_keys and key not in enemy_keys and key not in mid_keys and key not in otto_keys:
                if isinstance(sprites[key], (list, tuple)):
                    sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
                else:
                    sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites, pad_offsets


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        death_anim = state.death_timer > 0
        room_transition_anim = state.room_transition_timer > 0
        game_over_anim = state.game_over_timer > 0
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

        # Draw walls (assuming fixed positions based on bounds)
        room_idx = JaxBerzerk.get_room_index(state.room_counter)

        # --- Lade passendes Sprite ---
        def load_room_sprite(idx):
            return jax.lax.switch(
                idx,
                [
                    lambda: jr.get_sprite_frame(self.sprites["mid_walls_1"], 0),
                    lambda: jr.get_sprite_frame(self.sprites["mid_walls_2"], 0),
                    lambda: jr.get_sprite_frame(self.sprites["mid_walls_3"], 0),
                    lambda: jr.get_sprite_frame(self.sprites["mid_walls_4"], 0),
                ]
            )

        mid_sprite = load_room_sprite(room_idx)
        raster = jr.render_at(raster, 0, 0, mid_sprite)


        # --- Außenwände immer oben drauf ---
        outer_walls = jr.get_sprite_frame(self.sprites['level_outer_walls'], 0)
        raster = jr.render_at(raster, 0, 0, outer_walls)


        def draw_entry_block(raster):
            wall_vl = jr.get_sprite_frame(self.sprites['door_vertical_left'], 0)
            wall_hu = jr.get_sprite_frame(self.sprites['door_horizontal_up'], 0)
            wall_vr = jr.get_sprite_frame(self.sprites['door_vertical_right'], 0)
            wall_hd = jr.get_sprite_frame(self.sprites['door_horizontal_down'], 0)

            def block_top(r):
                return jr.render_at(r, 0, 0, wall_hu)

            def block_bottom(r):
                return jr.render_at(r, 0, 0, wall_hd)

            def block_left(r):
                return jr.render_at(r, 0, 0, wall_vl)

            def block_right(r):
                return jr.render_at(r, 0, 0, wall_vr)

            # Falls von links oder rechts → blockiere beide
            cond_lr = jnp.logical_and(state.entry_direction == 2, room_transition_anim == 0)
            raster = jax.lax.cond(cond_lr, lambda r: block_left(block_right(r)), lambda r: r, raster)

            cond_ll = jnp.logical_and(state.entry_direction == 3, room_transition_anim == 0)
            raster = jax.lax.cond(cond_ll, lambda r: block_left(block_right(r)), lambda r: r, raster)

            # Falls von oben → blockiere unten
            cond_top = jnp.logical_and(state.entry_direction == 0, room_transition_anim == 0)
            raster = jax.lax.cond(cond_top, block_bottom, lambda r: r, raster)

            # Falls von unten → blockiere oben
            cond_bottom = jnp.logical_and(state.entry_direction == 1, room_transition_anim == 0)
            raster = jax.lax.cond(cond_bottom, block_top, lambda r: r, raster)


            return raster
        
        raster = draw_entry_block(raster)


        def draw_enemy_wall_lines(raster, state):
            wall_x = 2
            line_height = 1
            line_length = 6

            draw_lines = ~room_transition_anim  # Nur zeichnen, wenn keine Animation

            def draw_line(raster, enemy_y):
                y = jnp.clip(enemy_y.astype(jnp.int32) - 1, 0, raster.shape[0] - line_height)
                line = jnp.zeros((line_height, line_length, raster.shape[-1]), dtype=raster.dtype)
                return jax.lax.dynamic_update_slice(raster, line, (y, wall_x, 0))

            def maybe_draw(i, raster):
                is_alive = state.enemy_alive[i]
                enemy_y = state.enemy_pos[i][1]
                should_draw = is_alive & draw_lines
                return jax.lax.cond(
                    should_draw,
                    lambda _: draw_line(raster, enemy_y),
                    lambda _: raster,
                    operand=None
                )

            return jax.lax.fori_loop(0, state.enemy_pos.shape[0], maybe_draw, raster)

        
        raster = draw_enemy_wall_lines(raster, state)


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

            cond = jnp.logical_and(is_active, jnp.logical_not(room_transition_anim))
            raster = jax.lax.cond(cond, draw_bullet, lambda r: r, raster)


        # Draw player animation
        def get_player_sprite():
            def death_animation():
                return jax.lax.switch(
                    (state.death_timer - 1) % 8,
                        [lambda: self.sprites['player_idle']] * 4 +
                        [lambda: self.sprites['player_death']] * 4
                )
            dir = state.last_dir

            return jax.lax.cond(
                death_anim,
                death_animation,
                lambda: jax.lax.cond(
                    state.player_is_firing,
                    lambda: jax.lax.switch(
                        jnp.select(
                            [
                                (dir[0] == 0) & (dir[1] == -1),     # up
                                (dir[0] == 1) & (dir[1] == -1),     # upright
                                (dir[0] == 1) & (dir[1] == 0),      # right
                                (dir[0] == 1) & (dir[1] == 1),      # downright
                                (dir[0] == 0) & (dir[1] == 1),      # down
                                (dir[0] == -1) & (dir[1] == 1),     # downleft
                                (dir[0] == -1) & (dir[1] == 0),     # left
                                (dir[0] == -1) & (dir[1] == -1),    # upleft
                            ],
                            jnp.arange(8),
                            default=2  # fallback = right
                        ),
                        [
                            lambda: self.sprites['player_shoot_up'],
                            lambda: self.sprites['player_shoot_up'],
                            lambda: self.sprites['player_shoot_right'],
                            lambda: self.sprites['player_shoot_down'],
                            lambda: self.sprites['player_shoot_down'],
                            lambda: self.sprites["player_shoot_down_left"],
                            lambda: self.sprites["player_shoot_left"],
                            lambda: self.sprites["player_shoot_up_left"],
                        ]
                    ),
                    lambda: jax.lax.cond(
                        state.animation_counter > 0,
                        lambda: jax.lax.switch((state.animation_counter - 1) % 12,
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
                            lambda: self.sprites['player_idle'],
                        ]
                    ),
                    lambda: self.sprites['player_idle']
                )
            )
        )


        player_sprite = get_player_sprite()

        player_frame_right = jr.get_sprite_frame(player_sprite, 0)

        player_frame = jax.lax.cond(
            (state.last_dir[0] < 0) & (~state.player_is_firing),
            lambda: jnp.flip(player_frame_right, axis=1),  # Horizontal spiegeln
            lambda: player_frame_right
        )
        raster = jax.lax.cond(
            room_transition_anim,
            lambda r: r,
            lambda r: jr.render_at(r, state.player_pos[0], state.player_pos[1], player_frame),
            raster
        )

        # Draw enemies
        color_cycle = ["yellow", "orange", "white", "green", "red", "blue", "yellow2", "pink"]

        def get_enemy_color_index(room_counter: jnp.ndarray) -> jnp.ndarray:
            num_colors = len(color_cycle)
            return jax.lax.cond(
                room_counter == 0,
                lambda: jnp.array(0, dtype=jnp.int32),
                lambda: ((room_counter - 1) // 2 + 1) % num_colors
            )

        color_names = jnp.array([
            [210, 210, 91, 255],    # yellow
            [186, 112, 69, 255],    # orange
            [214, 214, 214, 255],  # white
            [109, 210, 111, 255],      # green
            [239, 127, 128, 255],      # red
            [102, 158, 193, 255],      # blue
            [227, 205, 115, 255],  # yellow2
            [185, 96, 175, 255],  # pink
        ], dtype=jnp.uint8)

        def get_new_color(color_idx: jnp.ndarray) -> jnp.ndarray:
            # color_idx ist ein JAX tracer int32
            return color_names[color_idx]


        def recolor_sprite(sprite, color_idx, original_color):
            # Beispiel: Gelb (RGBA)
            new_color = get_new_color(color_idx)  # z.B. grün, rot, etc.

            # Maske (Vergleich mit original_color)
            mask = jnp.all(sprite == original_color, axis=-1)  # shape (H, W)

            recolored = jnp.where(mask[..., None], new_color, sprite)

            return recolored.astype(jnp.uint8)
        

        def maybe_recolor(sprite: chex.Array, color_idx: int, original_color) -> chex.Array:
            return jax.lax.cond(
                color_idx == 0,
                lambda: sprite,
                lambda: recolor_sprite(sprite, color_idx, original_color)
            )

        def get_enemy_sprite(i):
            counter = state.enemy_animation_counter[i]
            axis = state.enemy_move_axis[i]
            death_timer = state.enemy_death_timer[i]
            
            color_idx = get_enemy_color_index(state.room_counter)

            def recolor(name: str):
                original_color = jnp.array([210, 210, 64, 255], dtype=jnp.uint8)
                return maybe_recolor(self.sprites[name], color_idx, original_color)

            def death_animation():
                return jax.lax.switch(
                    (death_timer - 1) % 16,
                    [lambda: recolor("enemy_death_3")] * 8 +
                    [lambda: recolor("enemy_death_2")] * 4 + 
                    [lambda: recolor("enemy_death_1")] * 4
                )

            def normal_animation():
                return jax.lax.switch(
                    jnp.clip(axis + 1, 0, 2),
                    [
                        lambda: jax.lax.switch(
                            (counter - 1) % 64,
                            [lambda: recolor("enemy_idle_1")] * 8 +
                            [lambda: recolor("enemy_idle_2")] * 8 +
                            [lambda: recolor("enemy_idle_3")] * 8 +
                            [lambda: recolor("enemy_idle_4")] * 8 +
                            [lambda: recolor("enemy_idle_5")] * 8 +
                            [lambda: recolor("enemy_idle_6")] * 8 +
                            [lambda: recolor("enemy_idle_7")] * 8 +
                            [lambda: recolor("enemy_idle_8")] * 8
                        ),
                        lambda: jax.lax.switch(
                            (counter - 1) % 28,
                            [lambda: recolor("enemy_move_horizontal_1")] * 14 +
                            [lambda: recolor("enemy_move_horizontal_2")] * 14
                        ),
                        lambda: jax.lax.switch(
                            (counter - 1) % 48,
                            [lambda: recolor("enemy_move_vertical_1")] * 12 +
                            [lambda: recolor("enemy_move_vertical_2")] * 12 +
                            [lambda: recolor("enemy_move_vertical_1")] * 12 +
                            [lambda: recolor("enemy_move_vertical_3")] * 12
                        ),
                    ]
                )

            return jax.lax.cond(death_timer > 0, death_animation, normal_animation)
        
        for i in range(state.enemy_pos.shape[0]):
            is_dying = state.enemy_death_timer[i] > 0
            pos = jax.lax.cond(is_dying, lambda: state.enemy_death_pos[i], lambda: state.enemy_pos[i])
            sprite = get_enemy_sprite(i)
            frame = jr.get_sprite_frame(sprite, 0)
            
            frame = jax.lax.cond(
                state.enemy_move_dir[i] < 0,
                lambda: jnp.flip(frame, axis=1),
                lambda: frame
            )

            #TODO: Death Sprites arent centered yet
            raster = jax.lax.cond(
                room_transition_anim,
                lambda r: r,
                lambda r: jr.render_at(r, pos[0], pos[1], frame),
                raster
            )

        # yellow (only 1x) -> orange -> white -> green -> red -> other yellow -> pink -> yellow -> ...

        # Draw enemy bullets
        color_idx = get_enemy_color_index(state.room_counter)

        for i in range(state.enemy_bullets.shape[0]):
            is_active = state.enemy_bullet_active[i]
            bullet_pos = state.enemy_bullets[i]
            bullet_dir = state.enemy_bullet_dirs[i]

            def draw_enemy_bullet(raster):
                dx = bullet_dir[0]
                original_color = jnp.array([240, 170, 103, 255], dtype=jnp.uint8)

                def draw_horizontal(r):
                    raw_sprite = jr.get_sprite_frame(self.sprites['bullet_horizontal'], 0)
                    recolored = recolor_sprite(raw_sprite, color_idx, original_color)
                    return jr.render_at(r, bullet_pos[0], bullet_pos[1], recolored)

                def draw_vertical(r):
                    raw_sprite = jr.get_sprite_frame(self.sprites['bullet_vertical'], 0)
                    recolored = recolor_sprite(raw_sprite, color_idx, original_color)
                    return jr.render_at(r, bullet_pos[0], bullet_pos[1], recolored)

                return jax.lax.cond(dx != 0, draw_horizontal, draw_vertical, raster)

            cond = jnp.logical_and(is_active, jnp.logical_not(room_transition_anim))
            raster = jax.lax.cond(cond, draw_enemy_bullet, lambda r: r, raster)


        otto_sprites = self.sprites.get('evil_otto')
        otto_sprites = jax.lax.cond(
            (state.otto_anim_counter // 18) % 6,
            lambda s: s.get('evil_otto'), 
            lambda s: s.get('evil_otto_2'),
            self.sprites)

        otto_frame = jr.get_sprite_frame(otto_sprites, 0)
        jr.render_at(raster, state.otto_pos[0], state.otto_pos[1], otto_frame)

        raster = jax.lax.cond(
                state.otto_active,
                lambda r: jr.render_at(r, state.otto_pos[0], state.otto_pos[1], otto_frame),
                lambda r: r,
                raster
            )


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

            def skip_render():
                return raster_to_update
            
            def draw_score(value, offset_x):
                # Convert score to digits, zero-padded (e.g. 50 -> [0,0,5,0])
                score_digits = jr.int_to_digits(value, max_digits=max_score_digits)

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
                render_start_x = offset_x - score_spacing * (num_to_render - 1)

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
            
            show_bonus = state.enemy_clear_bonus_given
            #jax.debug.print("Enemy alive: {test}", test= state.enemy_alive)
            #jax.debug.print("Enemy bonus: {test}", test= state.enemy_clear_bonus_given)

            return jax.lax.cond(
                (state.score == 0) & (~show_bonus),
                skip_render,
                lambda: jax.lax.cond(
                    show_bonus,
                    lambda: draw_score(state.num_enemies * 10, self.consts.SCORE_OFFSET_X - 31),  # Bonus weiter links
                    lambda: draw_score(state.score, self.consts.SCORE_OFFSET_X)               # Normal an Standardposition
                )
            )

        raster = jax.lax.cond(
            jnp.logical_not(room_transition_anim),
            render_scores,
            lambda r: r,
            raster
        )

        # ---- Titel ----
        title_sprite = self.sprites.get('start_title', None)
        title_sprite = jnp.squeeze(title_sprite, axis=0)

        x = (self.consts.WIDTH - title_sprite.shape[1]) // 2 + 2
        y = self.consts.SCORE_OFFSET_Y

        def render_title(r):
            return jr.render_at(r, x, y, title_sprite)

        raster = jax.lax.cond(state.score == 0, render_title, lambda r: r, raster)

        def apply_bar_overlay(raster, progress: jnp.ndarray, mode_idx: int):
            total_height, width = raster.shape[0], raster.shape[1]
            playfield_height = total_height - self.consts.WALL_OFFSET[3]  # ignoring margin at top
            covered_rows = jnp.floor(progress * playfield_height).astype(jnp.int32)
            rows = jnp.arange(total_height)

            def top_down_mask():
                return rows[:, None] < covered_rows

            def bottom_up_mask():
                return rows[:, None] >= (playfield_height - covered_rows)

            def center_inward_mask():
                top = rows[:, None] < (covered_rows // 2)
                bottom = rows[:, None] >= (playfield_height - covered_rows // 2)
                return top | bottom

            mask = jax.lax.switch(
                mode_idx,
                [top_down_mask, bottom_up_mask, center_inward_mask]
            )

            mask_3c = jnp.repeat(mask, width, axis=1)[..., None]
            return jnp.where(mask_3c, 0, raster)


        # Fortschritt berechnen
        progress_transition = 1.0 - (state.room_transition_timer.astype(jnp.float32) / self.consts.TRANSITION_ANIMATION_FRAMES)
        #jax.debug.print("hgi {test}", test=progress_transition)
        raster = jax.lax.cond(
            room_transition_anim,
            lambda r: jax.lax.switch(
            state.entry_direction,
            [
                lambda: apply_bar_overlay(raster, progress_transition, 0),  # oben
                lambda: apply_bar_overlay(raster, progress_transition, 1),  # unten
                lambda: apply_bar_overlay(raster, progress_transition, 2),  # rechts
                lambda: apply_bar_overlay(raster, progress_transition, 2),  # links
            ]
            ),
            lambda r: r,
            raster
        )

        # render lives when in transition animation
        life_sprite = self.sprites.get('life', None)
        life_sprite = jnp.squeeze(life_sprite, axis=0)
        def render_lives(raster_to_update):
            """
            Render player lives using life_sprite during room transition or death.
            """
            life_spacing = 8
            start_x = self.consts.SCORE_OFFSET_X
            start_y = self.consts.SCORE_OFFSET_Y

            # Entscheide, wie viele Leben angezeigt werden sollen
            num_lives_to_draw = jax.lax.cond(
                death_anim,
                lambda: jnp.maximum(state.lives - 1, 0),
                lambda: state.lives
            )

            def draw_life(i, r):
                x = start_x - i * life_spacing
                y = start_y
                return jr.render_at(r, x, y, life_sprite)

            return jax.lax.fori_loop(0, num_lives_to_draw, draw_life, raster_to_update)


        raster = jax.lax.cond(
            room_transition_anim,
            render_lives,
            lambda r: r,
            raster
        )

        raster = jax.lax.cond(
            game_over_anim,
            lambda _: jnp.zeros_like(raster),  # Schwarzer Bildschirm
            lambda _: raster,
            operand=None
        )

        return raster
    
    def _generate_room_collision_masks(self) -> Dict[str, chex.Array]:
        def extract_mask(sprite, wall_color=jnp.array([84, 92, 214, 255])):
            return jnp.all(sprite[0] == wall_color, axis=-1)  # shape (H, W)

        return {
            name: extract_mask(self.sprites[name])
            for name in ['mid_walls_1', 'mid_walls_2', 'mid_walls_3', 'mid_walls_4', 
                         'level_outer_walls', 
                         'door_vertical_left', 'door_horizontal_up', 'door_vertical_right', 'door_horizontal_down']
        }


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