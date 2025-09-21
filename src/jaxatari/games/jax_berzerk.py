from typing import NamedTuple, Tuple, Dict, Any
from functools import partial
import jax
import jax.numpy as jnp
import chex
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
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
    PLAYER_SPEED = 0.4

    EXTRA_LIFE_AT = 1000

    ENEMY_SIZE = (8, 16)
    MAX_NUM_ENEMIES = 7
    MIN_NUM_ENEMIES = 5
    MOVEMENT_PROB = 0.0025  # probability for enemy to move
    ENEMY_SPEED = 0.1
    ENEMY_SHOOT_PROB = 0.005
    ENEMY_BULLET_SPEED = 0.47

    BULLET_SIZE_HORIZONTAL = (4, 2)
    BULLET_SIZE_VERTICAL = (1, 6)
    BULLET_SPEED = 2
    MAX_BULLETS = 1

    WALL_THICKNESS = 4
    WALL_OFFSET = (4, 4, 4, 30) # left, top, right, bottom
    EXIT_WIDTH = 40
    EXIT_HEIGHT = 64

    DEATH_ANIMATION_FRAMES = 128
    ENEMY_DEATH_ANIMATION_FRAMES = 8
    
    TRANSITION_ANIMATION_FRAMES = 64

    GAME_OVER_FRAMES = 32

    SCORE_OFFSET_X = WIDTH - 58 - 6  # window width - distance to the right - digit width 
    SCORE_OFFSET_Y = HEIGHT - 20 - 7  # window height - distance to the bottom - digit height 

    UI_OFFSET = 30  # pixels reserved for score at bottom
    PLAYER_BOUNDS = (
        (WALL_THICKNESS + WALL_OFFSET[0], WIDTH - WALL_THICKNESS - WALL_OFFSET[2]),
        (WALL_THICKNESS + WALL_OFFSET[1], HEIGHT - WALL_THICKNESS - WALL_OFFSET[3])
    )

    # Variations Evil Otto 
    ENABLE_EVIL_OTTO = False    # Variation 1: enable immortal evil otto
    MORTAL_EVIL_OTTO = False    # Variation 2: enable mortal evil otto (ENABLE_EVIL_OTTO has to be True)
    EVIL_OTTO_SIZE = (8, 7)
    EVIL_OTTO_SPEED = 0.4
    EVIL_OTTO_DELAY = 450
    
class PlayerState(NamedTuple):
    pos: chex.Array                     # (2,)
    last_dir: chex.Array                # (2,)
    animation_counter: chex.Array       # (1,)
    is_firing: chex.Array               # (1,)
    bullet: chex.Array                 # (2,)
    bullet_dir: chex.Array             # (2,)
    bullet_active: chex.Array           # (1,)
    death_timer: chex.Array

class EnemyState(NamedTuple):
    pos: chex.Array                     # (NUM_ENEMIES, 2)
    move_axis: chex.Array               # (NUM_ENEMIES,)
    move_dir: chex.Array                # (NUM_ENEMIES,)
    alive: chex.Array                   # (NUM_ENEMIES,)
    bullets: chex.Array                 # (NUM_ENEMIES, 2)
    bullet_dirs: chex.Array             # (NUM_ENEMIES, 2)
    bullet_active: chex.Array           # (NUM_ENEMIES,)
    move_prob: chex.Array               # (1,)
    clear_bonus_given: chex.Array       # (1,)
    death_timer: chex.Array             # (NUM_ENEMIES,)
    death_pos: chex.Array               # (NUM_ENEMIES,)
    animation_counter: chex.Array       # (NUM_ENEMIES,)

class OttoState(NamedTuple):
    pos: chex.Array                     # (2,)
    active: chex.Array                  # (1,)
    timer: chex.Array                   # (1,)
    anim_counter: chex.Array            # (1,)
    alive: chex.Array                   # (1,)

class BerzerkState(NamedTuple):
    player: PlayerState             
    enemy: EnemyState
    otto: OttoState
    rng: chex.PRNGKey                   # (1,)
    score: chex.Array                   # (1,)
    lives: chex.Array                   # (1,)
    room_counter: chex.Array            # (1,)
    extra_life_counter: chex.Array      # (1,)
    game_over_timer: chex.Array         # (1,)
    num_enemies: chex.Array             # (1,)
    entry_direction: chex.Array         # (1,)
    room_transition_timer: chex.Array   # (1,)


class BerzerkObservation(NamedTuple):
    # Player
    player_pos: chex.Array          # (2,)
    player_dir: chex.Array          # (2,)
    player_is_firing: chex.Array    # (1,)
    player_bullet: chex.Array       # (2,)
    player_bullet_dir: chex.Array   # (2,)
    player_bullet_active: chex.Array# (1,)

    # Enemies
    enemy_pos: chex.Array           # (NUM_ENEMIES, 2)
    enemy_bullets: chex.Array       # (NUM_ENEMIES, 2)
    enemy_bullet_dirs: chex.Array   # (NUM_ENEMIES, 2)
    enemy_bullet_active: chex.Array # (NUM_ENEMIES,)

    # Otto
    otto_pos: chex.Array            # (2,)

    # Game-level
    score: chex.Array               # (1,)
    lives: chex.Array               # (1,)


class BerzerkInfo(NamedTuple):
    all_rewards: chex.Array       # (1,)
    enemies_killed: chex.Array      # (1,)
    level_cleared: chex.Array       # (1,)


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


    @staticmethod   # has to be static to work for renderer
    def get_room_index(room_num):
        def get_current_index(room_num):
            prev = (room_num - 1) % 3
            offset = room_num + 1
            next_idx = (prev + offset) % 3
            return next_idx + 1
        
        return jax.lax.cond(
            room_num == 0,
            lambda: jnp.array(0, dtype=jnp.int32),
            lambda: jnp.array(get_current_index(room_num), dtype=jnp.int32)
        )


    @partial(jax.jit, static_argnums=(0, ))
    def rects_overlap(self, pos_a, size_a, pos_b, size_b):
        left_a, top_a = pos_a
        right_a = pos_a[0] + size_a[0]
        bottom_a = pos_a[1] + size_a[1]

        left_b, top_b = pos_b
        right_b = pos_b[0] + size_b[0]
        bottom_b = pos_b[1] + size_b[1]

        overlap_x = (left_a < right_b) & (right_a > left_b)
        overlap_y = (top_a < bottom_b) & (bottom_a > top_b)
        return overlap_x & overlap_y


    @partial(jax.jit, static_argnums=(0, ))
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
        player_x = state.player.pos[0] + dx * self.consts.PLAYER_SPEED
        player_y = state.player.pos[1] + dy * self.consts.PLAYER_SPEED

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
        default=state.player.last_dir
        )

        return player_x, player_y, player_direction
    

    @partial(jax.jit, static_argnums=(0, ))
    def shoot_bullet(self, state, player_pos, player_move_dir):
        # all possible directions
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

        # corresponding spawn positions (relative to player middle coordinates)
        offsets = jnp.array([
            [self.consts.PLAYER_SIZE[0] // 2, 0.0],                                     # up
            [self.consts.PLAYER_SIZE[0] // 2, 4.0],                                     # upright
            [3.0, self.consts.PLAYER_SIZE[1] // 2 - 4],                                 # right
            [self.consts.PLAYER_SIZE[0] // 2 + 1.0, self.consts.PLAYER_SIZE[1] - 10.0], # downright
            [self.consts.PLAYER_SIZE[0] // 2 + 2.0, self.consts.PLAYER_SIZE[1] - 10.0], # down
            [self.consts.PLAYER_SIZE[0] // 2 - 6.0, self.consts.PLAYER_SIZE[1] - 10.0], # downleft
            [-3.0, self.consts.PLAYER_SIZE[1] // 2 - 4],                                # left
            [self.consts.PLAYER_SIZE[0] -4.0 // 2 - 6.0, 4.0],                          # upleft
        ], dtype=jnp.float32)

        conds = jnp.all(dirs == player_move_dir[None, :], axis=1)   # check for active player direction

        default_offset = jnp.array([8.0, 8.0], dtype=jnp.float32)   # fallback (this should not happen) 

        offset = jnp.select(conds, offsets, default_offset)

        spawn_pos = player_pos + offset
        
        return jax.lax.cond(
            ~state.player.bullet_active[0],
            lambda _: (
                state.player.bullet.at[0].set(spawn_pos),
                state.player.bullet_dir.at[0].set(player_move_dir),
                state.player.bullet_active.at[0].set(True),
            ),
            lambda _: (state.player.bullet, state.player.bullet_dir, state.player.bullet_active),
            operand=None
        )


    @partial(jax.jit, static_argnums=(0, ))
    def object_hits_wall(self, object_pos, object_size, room_counter, entry_direction, num_points_per_side=3):
        # get current room id (0–3 → mid_walls_1 to _4)
        room_idx = JaxBerzerk.get_room_index(room_counter)

        # get respective wall mask (True = collision)
        def load_mask(idx):
            masks = jnp.array([
                self.renderer.room_collision_masks['mid_walls_1'],
                self.renderer.room_collision_masks['mid_walls_2'],
                self.renderer.room_collision_masks['mid_walls_3'],
                self.renderer.room_collision_masks['mid_walls_4'],
            ])
            return masks[idx]

        mid_mask = load_mask(room_idx)
        outer_mask = self.renderer.room_collision_masks['level_outer_walls']

        # get respective wall mask
        left_mask = self.renderer.room_collision_masks['door_vertical_left']
        right_mask = self.renderer.room_collision_masks['door_vertical_right']
        top_mask = self.renderer.room_collision_masks['door_horizontal_up']
        bottom_mask = self.renderer.room_collision_masks['door_horizontal_down']

        # calculate which doors should be opened
        block_left   = (entry_direction == 2) | (entry_direction == 3)
        block_right  = (entry_direction == 2) | (entry_direction == 3)
        block_top    = (entry_direction == 1)
        block_bottom = (entry_direction == 0)

        # get closed door masks
        collision_mask = mid_mask | outer_mask
        collision_mask = jax.lax.cond(block_left,   lambda: collision_mask | left_mask,   lambda: collision_mask)
        collision_mask = jax.lax.cond(block_right,  lambda: collision_mask | right_mask,  lambda: collision_mask)
        collision_mask = jax.lax.cond(block_top,    lambda: collision_mask | top_mask,    lambda: collision_mask)
        collision_mask = jax.lax.cond(block_bottom, lambda: collision_mask | bottom_mask, lambda: collision_mask)

        mask_height, mask_width = collision_mask.shape

        # check collision at all hit detection points
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


    @partial(jax.jit, static_argnums=(0, ))
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


    @partial(jax.jit, static_argnums=(0, ))
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
    def update_enemy_positions(self, player_pos, enemy_pos, enemy_axis, enemy_dir, rng, move_prob, room_counter):
        """
        Update enemy positions with movement probability.
        Once started moving, continues until aligned with player.
        
        Args:
            enemy_move_axis: 0 for x-axis, 1 for y-axis movement
            enemy_move_dir: 1 for positive, -1 for negative direction
        """
        enemy_rngs = jax.random.split(rng, self.consts.MAX_NUM_ENEMIES)

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
        rng, move_rng = jax.random.split(rng)
        
        # check if already moving (axis != -1)
        is_moving = enemy_move_axis != -1
        
        # if not moving, decide whether to start moving
        start_moving = jax.random.bernoulli(move_rng, move_prob)
        should_move = is_moving | (~is_moving & start_moving)
        
        def start_new_movement(_):
            # choose random axis (0=x, 1=y) and direction (1 or -1)
            rng1, _ = jax.random.split(move_rng)
            axis = jax.random.bernoulli(rng1, 0.5).astype(jnp.int32)
            dir_ = jnp.where(player_pos[axis] > enemy_pos[axis], 1, -1)
            return axis, dir_
        
        # if not moving but should start, initialize movement
        new_axis, new_dir = jax.lax.cond(
            ~is_moving & should_move,
            start_new_movement,
            lambda _: (enemy_move_axis, enemy_move_dir),
            operand=None
        )
        
         # calculate movement vector
        move_vec = jnp.array([
            jnp.where(new_axis == 0, new_dir * self.get_enemy_speed(room_counter), 0),
            jnp.where(new_axis == 1, new_dir * self.get_enemy_speed(room_counter), 0),
        ])
        proposed_pos = enemy_pos + move_vec

        # check if enemy would walk in other enemy (5px puffer)
        def too_close(enemy_position):
            offset = jnp.array([
                jnp.where(new_axis == 0, new_dir * 5, 0),
                jnp.where(new_axis == 1, new_dir * 5, 0),
            ])
            future_pos = enemy_pos + move_vec + offset
            return (
                self.rects_overlap(future_pos, self.consts.ENEMY_SIZE, enemy_position, self.consts.ENEMY_SIZE)
                & ~jnp.all(enemy_position == enemy_pos)  # sich selbst ausschließen
            )
        overlap = jnp.any(jax.vmap(too_close)(all_enemy_pos))

        # if movement would lead to collision with other enemy stop
        final_pos = jax.lax.select(overlap, enemy_pos, proposed_pos)

        # stop if enemy axis aligned with player axis
        aligned_x = jnp.abs(player_pos[0] - final_pos[0]) < 5
        aligned_y = jnp.abs(player_pos[1] - final_pos[1]) < 5
        aligned = jax.lax.select(new_axis == 0, aligned_x, aligned_y)

        final_axis = jnp.where(overlap | aligned, -1, new_axis)
        final_dir  = jnp.where(overlap | aligned, 0, new_dir)
        
        return final_pos, final_axis, final_dir, rng
    

    @partial(jax.jit, static_argnums=(0, ))
    def get_enemy_bullet_speed(self, level: jnp.ndarray) -> jnp.ndarray:
        base_bullet_speed = BerzerkConstants.ENEMY_BULLET_SPEED
        bullet_speed_increment = 0.065      # best-guess-estimate of true value

        # cap bullet speed after 13 levels
        capped_level = jnp.minimum(level + 1, 13)

        # increment every 2 levels (2,3 = 0; 4,5 = 1; …; 12,13 = 5)
        step = (capped_level - 1) // 2
        step = jnp.maximum(step, 0)

        value = base_bullet_speed + step * bullet_speed_increment

        return value


    @partial(jax.jit, static_argnums=(0, ))
    def get_enemy_speed(self, level: jnp.ndarray) -> jnp.ndarray:
        base_enemy_speed = BerzerkConstants.ENEMY_SPEED
        enemy_speed_increment = 0.007       # best-guess-estimate of true value

        # increment speed every 2 levels and repeat after 16 levels
        step = ((level + 1) // 2) % 8

        return base_enemy_speed + step * enemy_speed_increment


    @partial(jax.jit, static_argnums=(0, ))
    def enemy_fire_logic(self, player_pos, enemy_pos, enemy_size, enemy_shoot_prob, alive, axis, rng):
        # only shoot if not moving
        is_moving = axis != -1

        aligned_x = jnp.abs(enemy_pos[0] - player_pos[0]) < 5
        aligned_y = jnp.abs(enemy_pos[1] - player_pos[1]) < 5
        aligned = aligned_x | aligned_y

        can_shoot = (~is_moving) & aligned & alive
        should_fire = jax.random.uniform(rng) < enemy_shoot_prob

        dx = jnp.where(aligned_x, 0.0, jnp.sign(player_pos[0] - enemy_pos[0]))
        dy = jnp.where(aligned_y, 0.0, jnp.sign(player_pos[1] - enemy_pos[1]))
        direction = jnp.array([dx, dy], dtype=jnp.float32)

        dirs = jnp.array([
            [0, -1],   # up
            [1, 0],    # right
            [0, 1],    # down
            [-1, 0],   # left
        ], dtype=jnp.float32)

        # bullet spawn offsets relative to enemy position
        offsets = jnp.array([
            [enemy_size[0], enemy_size[1] // 2 - 7],    # up
            [enemy_size[0], enemy_size[1] // 2],        # right
            [0.0, enemy_size[1] // 2 + 2],              # down
            [0.0, enemy_size[1] // 2],                  # left
        ], dtype=jnp.float32)

        conds = jnp.all(dirs == direction[None, :], axis=1)

        # fallback (should not be used)
        default_offset = jnp.array([enemy_size[0] // 2,
                                    enemy_size[1] // 2], dtype=jnp.float32)

        offset = jnp.select(conds, offsets, default_offset)

        spawn_pos = enemy_pos + offset

        return (
            spawn_pos,
            direction,
            should_fire & can_shoot
        )


    @partial(jax.jit, static_argnums=(0, ))
    def spawn_enemies(self, state, rng):
        # number of enemy: self.consts.MIN_NUM_ENEMIES - self.consts.MAX_NUM_ENEMIES
        rng, sub_num, sub_spawn = jax.random.split(rng, 3)
        num_enemies = jax.random.randint(sub_num, (), self.consts.MIN_NUM_ENEMIES, self.consts.MAX_NUM_ENEMIES+1)

        # initialize empty spawn vector
        placed_init = jnp.full((self.consts.MAX_NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)

        def sample_pos(r):
            return jax.random.uniform(
                r, shape=(2,),
                minval=jnp.array([self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[1][0]]),
                maxval=jnp.array([self.consts.PLAYER_BOUNDS[0][1] - self.consts.ENEMY_SIZE[0],
                                self.consts.PLAYER_BOUNDS[1][1] - self.consts.ENEMY_SIZE[1]])
            )

        def cond_fn(carry2):
            pos, _, attempts, placed = carry2
            in_wall = self.object_hits_wall(pos, self.consts.ENEMY_SIZE,
                                            state.room_counter, state.entry_direction)
            on_player = self.rects_overlap(state.player.pos, self.consts.PLAYER_SIZE, pos, self.consts.ENEMY_SIZE)
            overlap_enemy = jnp.any(
                jax.vmap(lambda enemy_position: self.rects_overlap(
                    pos, self.consts.ENEMY_SIZE, enemy_position, self.consts.ENEMY_SIZE))(placed))
            invalid = in_wall | on_player | overlap_enemy
            return jnp.logical_and(invalid, attempts < 2)

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
        enemy_alive = jnp.arange(self.consts.MAX_NUM_ENEMIES) < num_enemies
        return state._replace(
            enemy=state.enemy._replace(pos=placed_final, 
                                      alive=enemy_alive),
            num_enemies=num_enemies)


    @partial(jax.jit, static_argnums=(0, ))
    def move_otto(self, otto_pos, player_pos, otto_speed, otto_animation_counter):
                direction = player_pos - otto_pos
                norm = jnp.linalg.norm(direction) + 1e-6
                new_otto_pos = otto_pos + (direction / norm) * otto_speed

                otto_animation_counter += 1

                # jump animation for otto (best-guess-estimate of true values)
                jump_phase = (otto_animation_counter // 15) % 5
                jump_offset = jnp.where(jump_phase == 0, 0.5,
                                        jnp.where(jump_phase == 4, 0.8, 
                                                  jnp.where(jump_phase == 1, -0.7, -0.3)))
                otto_pos_with_jump = new_otto_pos.at[1].add(jump_offset)

                return otto_pos_with_jump

       
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state) -> BerzerkObservation:
        NUM_ENEMIES = int(self.consts.MAX_NUM_ENEMIES)

        # Player
        player_pos = state.player.pos.astype(jnp.float32)
        player_dir = state.player.last_dir.astype(jnp.float32)
        player_is_firing = jnp.array([state.player.is_firing], dtype=jnp.float32)
        player_bullet = state.player.bullet.astype(jnp.float32)
        player_bullet_dir = state.player.bullet_dir.astype(jnp.float32)
        player_bullet_active = jnp.array([state.player.bullet_active], dtype=jnp.float32)

        # Enemies (pad/truncate to NUM_ENEMIES)
        enemy_pos = jnp.full((NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        enemy_bullets = jnp.full((NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        enemy_bullet_dirs = jnp.zeros((NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_active = jnp.zeros((NUM_ENEMIES,), dtype=jnp.float32)

        # state.enemy.pos etc. have shape (num_enemies,2)
        n_state_enemies = min(NUM_ENEMIES, state.enemy.pos.shape[0])
        enemy_pos = enemy_pos.at[:n_state_enemies].set(state.enemy.pos[:n_state_enemies].astype(jnp.float32))
        enemy_bullets = enemy_bullets.at[:n_state_enemies].set(state.enemy.bullets[:n_state_enemies].astype(jnp.float32))
        enemy_bullet_dirs = enemy_bullet_dirs.at[:n_state_enemies].set(state.enemy.bullet_dirs[:n_state_enemies].astype(jnp.float32))
        enemy_bullet_active = enemy_bullet_active.at[:n_state_enemies].set(state.enemy.bullet_active[:n_state_enemies].astype(jnp.float32))

        # Otto
        otto_pos = state.otto.pos.astype(jnp.float32)

        # Game-level
        score = jnp.array([state.score], dtype=jnp.float32)
        lives = jnp.array([state.lives], dtype=jnp.float32)

        return BerzerkObservation(
            player_pos=player_pos,
            player_dir=player_dir,
            player_is_firing=player_is_firing,
            player_bullet=player_bullet,
            player_bullet_dir=player_bullet_dir,
            player_bullet_active=player_bullet_active,
            enemy_pos=enemy_pos,
            enemy_bullets=enemy_bullets,
            enemy_bullet_dirs=enemy_bullet_dirs,
            enemy_bullet_active=enemy_bullet_active,
            otto_pos=otto_pos,
            score=score,
            lives=lives
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BerzerkState, state: BerzerkState) -> jnp.ndarray:
        return state.score - previous_state.score


    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: BerzerkState, state: BerzerkState) -> jnp.ndarray:
        points_reward = state.score - previous_state.score
        enemies_killed_reward = jnp.sum(previous_state.enemy_alive - state.enemy_alive)
        
        return jnp.array([points_reward, enemies_killed_reward], dtype=jnp.float32)


    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: BerzerkState, previous_state: BerzerkState = None) -> BerzerkInfo:
        if previous_state is None:
            all_rewards = jnp.zeros(2, dtype=jnp.float32)
            enemies_killed = jnp.array(0, dtype=jnp.int32)
        else:
            # Rewards
            all_rewards = self._get_all_rewards(previous_state, state)

            prev_alive = jnp.array(previous_state.enemy_alive, dtype=jnp.int32)
            curr_alive = jnp.array(state.enemy_alive, dtype=jnp.int32)
            enemies_killed = jnp.sum(prev_alive - curr_alive)

        level_cleared = jnp.array([state.room_counter], dtype=jnp.int32)

        return BerzerkInfo(
            all_rewards=all_rewards,
            enemies_killed=enemies_killed,
            level_cleared=level_cleared
        )


    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: BerzerkState) -> bool:
        return state.lives < 0


    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: BerzerkObservation) -> chex.Array:
        return jnp.concatenate([
            obs.player_pos.flatten(),
            obs.player_dir.flatten(),
            obs.player_is_firing.flatten(),
            obs.player_bullet.flatten(),
            obs.player_bullet_dir.flatten(),
            obs.player_bullet_active.flatten(),
            obs.enemy_pos.flatten(),
            obs.enemy_bullets.flatten(),
            obs.enemy_bullet_dirs.flatten(),
            obs.enemy_bullet_active.flatten(),
            obs.otto_pos.flatten(),
            obs.score.flatten(),
            obs.lives.flatten()
        ])


    @partial(jax.jit, static_argnums=(0,))
    def info_to_flat_array(self, info: BerzerkInfo) -> chex.Array:
        return jnp.concatenate([
            info.all_rewards.flatten(),
            info.enemies_killed.flatten(),
            info.level_cleared.flatten()
        ])


    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: BerzerkState) -> jnp.ndarray:
        return self.renderer.render(state)


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[BerzerkObservation, BerzerkState]:
        # --- Player init ---
        pos = jnp.array([
            self.consts.PLAYER_BOUNDS[0][0] + 2,
            self.consts.PLAYER_BOUNDS[1][1] // 2
        ], dtype=jnp.float32)
        last_dir = jnp.array([0.0, -1.0])  # default = up
        bullets = jnp.zeros((1, 2), dtype=jnp.float32)
        bullet_dirs = jnp.zeros((1, 2), dtype=jnp.float32)
        bullet_active = jnp.zeros((1,), dtype=jnp.bool_)
        animation_counter = jnp.array(0, dtype=jnp.int32)
        death_timer = jnp.array(0, dtype=jnp.int32)
        player_is_firing = jnp.array(False)

        player_state = PlayerState(
            pos=pos,
            last_dir=last_dir,
            animation_counter=animation_counter,
            is_firing=player_is_firing,
            bullet=bullets,
            bullet_dir=bullet_dirs,
            bullet_active=bullet_active,
            death_timer=death_timer,
        )

        # --- Enemy init ---
        enemy_pos = jnp.full((self.consts.MAX_NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        enemy_move_axis = -jnp.ones((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_dir = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_alive = jnp.ones((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_bullets = jnp.zeros((self.consts.MAX_NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_dirs = jnp.zeros((self.consts.MAX_NUM_ENEMIES, 2), dtype=jnp.float32)
        enemy_bullet_active = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_move_prob = jnp.full((self.consts.MAX_NUM_ENEMIES,), self.consts.MOVEMENT_PROB, dtype=jnp.float32)
        enemy_clear_bonus_given = jnp.array(False)
        enemy_death_timer = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)
        enemy_death_pos = jnp.full((self.consts.MAX_NUM_ENEMIES, 2), -100.0, dtype=jnp.float32)
        enemy_animation_counter = jnp.zeros((self.consts.MAX_NUM_ENEMIES,), dtype=jnp.int32)

        enemy_state = EnemyState(
            pos=enemy_pos,
            move_axis=enemy_move_axis,
            move_dir=enemy_move_dir,
            alive=enemy_alive,
            bullets=enemy_bullets,
            bullet_dirs=enemy_bullet_dirs,
            bullet_active=enemy_bullet_active,
            move_prob=enemy_move_prob,
            clear_bonus_given=enemy_clear_bonus_given,
            death_timer=enemy_death_timer,
            death_pos=enemy_death_pos,
            animation_counter=enemy_animation_counter,
        )

        # --- Otto init ---
        otto_pos = jnp.array([-100.0, -100.0], dtype=jnp.float32)
        otto_active = jnp.array(False)
        otto_timer = self.consts.EVIL_OTTO_DELAY
        otto_anim_counter = jnp.array(0, dtype=jnp.int32)
        otto_alive = jnp.array(True)

        otto_state = OttoState(
            pos=otto_pos,
            active=otto_active,
            timer=otto_timer,
            anim_counter=otto_anim_counter,
            alive=otto_alive,
        )

        # --- Global game state ---
        lives = jnp.array(2, dtype=jnp.int32)
        score = jnp.array(0, dtype=jnp.int32)
        room_counter = jnp.array(0, dtype=jnp.int32)
        extra_life_counter = jnp.array(0, dtype=jnp.int32)
        game_over_timer = jnp.array(0, dtype=jnp.int32)
        num_enemies = jnp.array(self.consts.MAX_NUM_ENEMIES, dtype=jnp.int32)
        entry_direction = jnp.array(3, dtype=jnp.int32)
        room_transition_timer = jnp.array(0, dtype=jnp.int32)

        state = BerzerkState(
            player=player_state,
            enemy=enemy_state,
            otto=otto_state,
            rng=rng,
            score=score,
            lives=lives,
            room_counter=room_counter,
            extra_life_counter=extra_life_counter,
            game_over_timer=game_over_timer,
            num_enemies=num_enemies,
            entry_direction=entry_direction,
            room_transition_timer=room_transition_timer,
        )

        # Spawn enemies in valid positions
        state = self.spawn_enemies(state, jax.random.split(rng)[0])
        return self._get_observation(state), state

    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: BerzerkState, action: chex.Array) -> Tuple[BerzerkObservation, BerzerkState, float, bool, BerzerkInfo]:
        # Handle game over animation phase
        game_over_active = state.game_over_timer > 0
        game_over_timer = jnp.maximum(state.game_over_timer - 1, 0)

        def handle_game_over():
            new_state = state._replace(game_over_timer=game_over_timer)
            return (
                self._get_observation(new_state),
                new_state,
                0.0,
                game_over_timer == 0,
                self._get_info(new_state),
            )


        # Handle room transition animation phase
        room_transition_active = state.room_transition_timer > 0
        transition_timer = jnp.maximum(state.room_transition_timer - 1, 0)

        def handle_room_transition():
            new_state = state._replace(room_transition_timer=transition_timer)

            def finished_transition():
                player_spawn_pos = jax.lax.switch(
                    new_state.entry_direction,
                    [
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2,
                            self.consts.PLAYER_BOUNDS[1][1] - 25
                            ], dtype=jnp.float32),  # top -> spawn at bottom
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] // 2, 
                            self.consts.PLAYER_BOUNDS[1][0] + 5], dtype=jnp.float32),  # bottom -> spawn at top
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][1] - 12, 
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                            ], dtype=jnp.float32),  # left -> spawn at right
                        
                        lambda _: jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32),  # right -> spawn at left
                    ],
                    jnp.array([
                            self.consts.PLAYER_BOUNDS[0][0] + 2,
                            self.consts.PLAYER_BOUNDS[1][1] // 2
                        ], dtype=jnp.float32)  # fallback
                )

                # load new level
                new_rng = jax.random.split(state.rng)[1]
                obs, base_state = self.reset(new_rng)
                base_state = base_state._replace(
                    player=base_state.player._replace(pos=player_spawn_pos),
                    room_counter=state.room_counter + 1,
                    lives=state.lives,
                    score=state.score,
                    entry_direction=state.entry_direction,
                    extra_life_counter=state.extra_life_counter
                )

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


        # Handle normal gameplay
        def handle_normal():
            #######################################################
            # 1. Update Player
            #######################################################

            player_alive = state.player.death_timer == 0
            # forbid movement during death animation
            player_x, player_y, move_dir = jax.lax.cond(
                player_alive,
                lambda _: self.player_step(state, action),
                lambda _: (state.player.pos[0], state.player.pos[1], state.player.last_dir),
                operand=None
            )

            new_player_pos = jnp.array([player_x, player_y])

            moving = self.is_moving_action(action)

            animation_counter = jnp.where(
                player_alive & moving,
                state.player.animation_counter + 1,
                0
            )

            # handle player bullet
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
            is_shooting = player_alive & is_shooting

            player_is_firing = is_shooting.astype(jnp.bool_)

            player_bullet, player_bullet_dir, player_bullet_active = jax.lax.cond(
                is_shooting,
                lambda _: self.shoot_bullet(state, new_player_pos, move_dir),
                lambda _: (state.player.bullet, state.player.bullet_dir, state.player.bullet_active),
                operand=None
            )

            # choose bullet size (depending on direction)
            player_bullet_size = jax.vmap(
                lambda d: jax.lax.select(
                    d[0] == 0,
                    jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                    jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
                )
            )(player_bullet_dir)
            
            player_bullet += player_bullet_dir * self.consts.BULLET_SPEED * player_bullet_active[:, None]
            # only 1 player bullet
            player_bullet_active = player_bullet_active & (~self.object_hits_wall(player_bullet[0], player_bullet_size[0], state.room_counter, state.entry_direction)) & (
                (player_bullet[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
                (player_bullet[:, 0] + player_bullet_size[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
                (player_bullet[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
                (player_bullet[:, 1] + player_bullet_size[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
            )


            #######################################################
            # 2. Update enemies
            #######################################################

            rng, enemy_rng = jax.random.split(state.rng)

            updated_enemy_pos, updated_enemy_axis, updated_enemy_dir = self.update_enemy_positions(
                new_player_pos, state.enemy.pos, state.enemy.move_axis, state.enemy.move_dir,
                enemy_rng, state.enemy.move_prob, state.room_counter
            )

            enemy_animation_counter = state.enemy.animation_counter + 1

            # only move living enemies
            updated_enemy_pos = jnp.where(state.enemy.alive[:, None], updated_enemy_pos, state.enemy.pos)
            updated_enemy_axis = jnp.where(state.enemy.alive, updated_enemy_axis, state.enemy.move_axis)
            updated_enemy_dir = jnp.where(state.enemy.alive, updated_enemy_dir, state.enemy.move_dir)

            # handle enemy bullets
            enemy_rngs = jax.random.split(rng, self.consts.MAX_NUM_ENEMIES)
            enemy_bullets_new, dirs_new, active_new = jax.vmap(self.enemy_fire_logic,
                                                               in_axes=(None, 0, None, None, 0, 0, 0))(
                new_player_pos,
                updated_enemy_pos,
                self.consts.ENEMY_SIZE,
                self.consts.ENEMY_SHOOT_PROB,
                state.enemy.alive,
                updated_enemy_axis,
                enemy_rngs
            )

            # only fire if no active bullet of this enemy already
            can_shoot = (state.room_counter > 0)
            can_shoot_mask = jnp.broadcast_to(can_shoot, active_new.shape)  # cast to length of NUM_ENEMIES

            enemy_bullets = jnp.where(
                ~state.enemy.bullet_active[:, None] & active_new[:, None] & can_shoot_mask[:, None],
                enemy_bullets_new,
                state.enemy.bullets
            )

            enemy_bullet_dirs = jnp.where(
                ~state.enemy.bullet_active[:, None] & active_new[:, None] & can_shoot_mask[:, None],
                dirs_new,
                state.enemy.bullet_dirs
            )

            enemy_bullet_active = state.enemy.bullet_active | (active_new & can_shoot_mask)

            enemy_bullet_sizes = jax.vmap(
                lambda d: jax.lax.select(
                    d[0] == 0,
                    jnp.array(self.consts.BULLET_SIZE_VERTICAL, dtype=jnp.float32),
                    jnp.array(self.consts.BULLET_SIZE_HORIZONTAL, dtype=jnp.float32)
                )
            )(enemy_bullet_dirs)

            enemy_bullets = enemy_bullets + enemy_bullet_dirs * self.get_enemy_bullet_speed(state.room_counter) * enemy_bullet_active[:, None]

            # deactivate bullets that are out of bounds
            enemy_bullet_active = enemy_bullet_active & (
                (enemy_bullets[:, 0] >= self.consts.PLAYER_BOUNDS[0][0]) &
                (enemy_bullets[:, 0] + enemy_bullet_sizes[:, 0] <= self.consts.PLAYER_BOUNDS[0][1]) &
                (enemy_bullets[:, 1] >= self.consts.PLAYER_BOUNDS[1][0]) &
                (enemy_bullets[:, 1] + enemy_bullet_sizes[:, 1] <= self.consts.PLAYER_BOUNDS[1][1])
            )


            #######################################################
            # 3. Collision Checks
            #######################################################

            player_hit_exit = self.check_exit_crossing(new_player_pos)  # player reached exit

            player_hit_wall = self.object_hits_wall((player_x, player_y), self.consts.PLAYER_SIZE, state.room_counter, state.entry_direction) & ~player_hit_exit

            player_hits_enemy = jax.vmap(
                lambda enemy_pos: self.rects_overlap(new_player_pos, self.consts.PLAYER_SIZE, enemy_pos, self.consts.ENEMY_SIZE)
            )(updated_enemy_pos)
            player_hit_by_enemy = jnp.any(player_hits_enemy)

            enemy_bullet_hits_player = jax.vmap(
                lambda b_pos, b_size, b_active: self.rects_overlap(b_pos, b_size, new_player_pos, self.consts.PLAYER_SIZE) & b_active
            )(enemy_bullets, enemy_bullet_sizes, enemy_bullet_active)
            player_hit_by_enemy_bullet = jnp.any(enemy_bullet_hits_player)

            enemy_hits_wall = jax.vmap(
                lambda enemy_pos: self.object_hits_wall(enemy_pos, self.consts.ENEMY_SIZE, state.room_counter, state.entry_direction)
            )(updated_enemy_pos)

            all_enemy_bullet_hits = jax.vmap(
                lambda enemy_pos: jax.vmap(
                    lambda bullet, size: self.rects_overlap(bullet, size, enemy_pos, self.consts.ENEMY_SIZE)
                )(player_bullet, player_bullet_size)
            )(updated_enemy_pos)
            enemy_hit_by_player_bullet = jnp.any(all_enemy_bullet_hits, axis=1)

            enemy_friendly_fire_hits = jax.vmap(
                lambda bullet_pos, bullet_size, shooter_pos, active: jax.vmap(
                    lambda target_pos: self.rects_overlap(
                        bullet_pos, bullet_size, target_pos, jnp.array(self.consts.ENEMY_SIZE, dtype=jnp.float32))
                          & active 
                          & ~jnp.all(target_pos == shooter_pos)
                )(updated_enemy_pos)
            )(enemy_bullets, enemy_bullet_sizes, updated_enemy_pos, enemy_bullet_active)

            enemy_touch_hits = jax.vmap(
                lambda pos_a, alive_a: jax.vmap(
                    lambda pos_b, alive_b: (
                        self.rects_overlap(pos_a, self.consts.ENEMY_SIZE, pos_b, self.consts.ENEMY_SIZE) &
                        ~jnp.all(pos_a == pos_b) &   # not the same enemy
                        alive_a & alive_b            # both alive
                    )
                )(updated_enemy_pos, state.enemy.alive)
            )(updated_enemy_pos, state.enemy.alive)
            enemy_hit_enemy = jnp.any(enemy_touch_hits, axis=1)

            # remove bullets on hit
            bullet_vs_bullet_hits = jax.vmap(
                lambda b_pos, b_size, b_active: jax.vmap(
                    lambda e_pos, e_size, e_active: 
                        self.rects_overlap(b_pos, b_size, e_pos, e_size) & b_active & e_active
                )(enemy_bullets, enemy_bullet_sizes, enemy_bullet_active)
            )(player_bullet, player_bullet_size, player_bullet_active)
            player_bullet_hit_enemy_bullet = jnp.any(bullet_vs_bullet_hits, axis=1)     # (player_bullets,)
            enemy_bullet_hit_by_player_bullet = jnp.any(bullet_vs_bullet_hits, axis=0)  # (enemy_bullets,)

            enemy_bullet_hit_enemy = jnp.any(enemy_friendly_fire_hits, axis=1)
            player_bullet_active = player_bullet_active & ~player_bullet_hit_enemy_bullet
            enemy_bullet_active = enemy_bullet_active & ~enemy_bullet_hit_enemy & ~enemy_bullet_hit_by_player_bullet

            enemy_bullet_hits_wall = jax.vmap(
                lambda pos, size: self.object_hits_wall(pos, size, state.room_counter, state.entry_direction)
            )(enemy_bullets, enemy_bullet_sizes)

            enemy_bullet_active = enemy_bullet_active & (~enemy_bullet_hits_wall)

            bullet_hit = jnp.any(all_enemy_bullet_hits, axis=0)
            player_bullet_active = player_bullet_active & ~bullet_hit


            #######################################################
            # 4. Handle Evil Otto (if active)
            #######################################################

            otto_pos = jnp.where(player_hit_exit, jnp.array([-100.0, -100.0], dtype=jnp.float32), state.otto.pos)
            otto_hits_player = self.rects_overlap(
                otto_pos, jnp.array(self.consts.EVIL_OTTO_SIZE, dtype=jnp.float32),
                new_player_pos, self.consts.PLAYER_SIZE
            )

            # spawn otto after timer has reached 0
            new_otto_timer = jnp.maximum(state.otto.timer - 1, 0)
            otto_active = jnp.logical_and(jnp.logical_not(new_otto_timer), jnp.logical_not(state.otto.active))

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

            spawn_pos = jnp.where(otto_active, otto_spawn_pos, state.otto.pos)

            # keep otto active after spawn
            otto_active = jnp.logical_or(state.otto.active, otto_active) 

            # move to player once active
            otto_pos = jnp.where(
                otto_active, 
                self.move_otto(spawn_pos, new_player_pos, self.consts.EVIL_OTTO_SPEED, state.otto.anim_counter), 
                spawn_pos)
            otto_anim_counter = jnp.where(otto_active, state.otto.anim_counter + 1, 0)

            otto_hit_by_bullet = jnp.any(
                jax.vmap(lambda b_pos, b_size, b_active:
                    self.rects_overlap(b_pos, b_size, otto_pos, self.consts.EVIL_OTTO_SIZE) & b_active
                )(player_bullet, player_bullet_size, player_bullet_active)
            )

            otto_alive = jax.lax.cond(
                self.consts.MORTAL_EVIL_OTTO,
                lambda _: state.otto.alive & (~otto_hit_by_bullet),
                lambda _: True,
                operand=None
            )

            otto_removed = otto_active & (~otto_alive)

            otto_pos = jax.lax.cond(
                otto_removed,
                lambda _: jnp.array([-100.0, -100.0], dtype=jnp.float32),
                lambda _: otto_pos,
                operand=None
            )

            player_bullet_active = jnp.where(self.consts.ENABLE_EVIL_OTTO, ~otto_hit_by_bullet & player_bullet_active, player_bullet_active)


            #######################################################
            # 5. Handle Death, Score, Extra Lives
            #######################################################

            # player death
            hit_something = player_hit_by_enemy | player_hit_wall | player_hit_by_enemy_bullet | otto_hits_player
            death_timer = jnp.where(hit_something & (state.player.death_timer == 0), self.consts.DEATH_ANIMATION_FRAMES + 2, state.player.death_timer)
            death_timer = jnp.maximum(death_timer - 1, 0)

            # enemy death
            enemy_kill_events = (
                enemy_hit_by_player_bullet |
                enemy_hits_wall |
                enemy_hit_enemy |
                enemy_bullet_hit_enemy |
                player_hits_enemy
            )

            enemy_dies = state.enemy.alive & enemy_kill_events

            enemy_alive = state.enemy.alive & ~enemy_dies

            updated_enemy_axis = jnp.where(enemy_alive, updated_enemy_axis, 0)
            new_enemy_death_timer = jnp.where(enemy_dies, self.consts.ENEMY_DEATH_ANIMATION_FRAMES + 2, state.enemy.death_timer)
            new_enemy_death_pos = jnp.where(enemy_dies[:, None], updated_enemy_pos, state.enemy.death_pos)
            
            enemy_death_timer_next = jnp.maximum(new_enemy_death_timer - 1, 0)

            invisible = jnp.array([-100.0, -100.0])     # teleport dead enemies out of view
            updated_enemy_pos = jnp.where(enemy_alive[:, None], updated_enemy_pos, invisible)

            # calculate score: 50 points per dead enemy
            score_after = state.score + jnp.sum(enemy_dies) * 50

            # bonus score for killing all enemies in level
            give_bonus = (~jnp.any(enemy_alive)) & (~state.enemy.clear_bonus_given)
            bonus_score = jnp.where(give_bonus, state.num_enemies * 10, 0)

            score_after += bonus_score
            enemy_clear_bonus_given = state.enemy.clear_bonus_given | give_bonus

            # Handle live logic
            lives_after = jnp.where((death_timer == 0) & hit_something, state.lives - 1, state.lives)
            score_after = jnp.where(lives_after == -1, 0, score_after)

            # Trigger Room Transition oder Game Over automatisch
            transition_timer = jax.lax.cond(
                death_timer == 1,
                lambda: jax.lax.cond(
                    lives_after == -1,
                    lambda: jnp.array(self.consts.GAME_OVER_FRAMES, dtype=jnp.int32),
                    lambda: jnp.array(self.consts.TRANSITION_ANIMATION_FRAMES, dtype=jnp.int32)
                ),
                lambda: state.room_transition_timer
            )
            game_over_timer = jax.lax.cond(
                (death_timer == 1) & (lives_after == -1),
                lambda: self.consts.GAME_OVER_FRAMES,
                lambda: state.game_over_timer
            )

            extra_lives_given_last_score = state.extra_life_counter * self.consts.EXTRA_LIFE_AT
            give_extra_life = score_after >= extra_lives_given_last_score + self.consts.EXTRA_LIFE_AT

            lives_after = jnp.where(give_extra_life, state.lives + 1, state.lives)
            extra_life_counter_after = jnp.where(give_extra_life, state.extra_life_counter + 1, state.extra_life_counter)


            #######################################################
            # 5. Update State
            #######################################################

            transition_timer = jnp.where(player_hit_exit, self.consts.TRANSITION_ANIMATION_FRAMES, transition_timer)
            entry_direction = jnp.where(player_hit_exit, self.get_exit_direction(new_player_pos), state.entry_direction)

            new_state = BerzerkState(
                player=PlayerState(
                    pos=new_player_pos,
                    last_dir=move_dir,
                    animation_counter=animation_counter,
                    is_firing=player_is_firing,
                    bullet=player_bullet,
                    bullet_dir=player_bullet_dir,
                    bullet_active=player_bullet_active,
                    death_timer=death_timer,
                ),
                enemy=EnemyState(
                    pos=updated_enemy_pos,
                    move_axis=updated_enemy_axis,
                    move_dir=updated_enemy_dir,
                    alive=enemy_alive,
                    bullets=enemy_bullets,
                    bullet_dirs=enemy_bullet_dirs,
                    bullet_active=enemy_bullet_active,
                    move_prob=state.enemy.move_prob,
                    clear_bonus_given=enemy_clear_bonus_given,
                    death_timer=enemy_death_timer_next,
                    death_pos=new_enemy_death_pos,
                    animation_counter=enemy_animation_counter,
                ),
                otto=OttoState(
                    pos=otto_pos,
                    active=otto_active,
                    timer=new_otto_timer,
                    anim_counter=otto_anim_counter,
                    alive=otto_alive,
                ),
                rng=rng,
                score=score_after,
                lives=lives_after,
                room_counter=state.room_counter,
                extra_life_counter=extra_life_counter_after,
                game_over_timer=game_over_timer,
                num_enemies=state.num_enemies,
                entry_direction=entry_direction,
                room_transition_timer=transition_timer,
            )


            #######################################################
            # 5. Observation + Info + Reward/Done
            #######################################################

            observation = self._get_observation(new_state)
            info = self._get_info(new_state)
            reward = 0.0
            done = jnp.equal(state.lives, -1) 

            return observation, new_state, reward, done, info
        

        # call appropriate step handler for current frame
        return jax.lax.cond(
            game_over_active,
            lambda _: handle_game_over(),
            lambda _: jax.lax.cond(
                room_transition_active,
                lambda _: handle_room_transition(),
                lambda _: handle_normal(),
                operand=None
            ),
            operand=None
        )


    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))


    def observation_space(self) -> spaces.Dict:
        """Returns the simplified observation space for the agent."""
        NUM = int(self.consts.MAX_NUM_ENEMIES)
        # allow -100 sentinel for "invisible" / teleported entities
        POS_LOW = -100.0
        POS_HIGH = 255.0

        return spaces.Dict({
            # Player
            "player_pos": spaces.Box(POS_LOW, POS_HIGH, (2,), jnp.float32),
            "player_dir": spaces.Box(-1.0, 1.0, (2,), jnp.float32),
            "player_is_firing": spaces.Box(0.0, 1.0, (1,), jnp.float32),
            "player_bullet": spaces.Box(POS_LOW, POS_HIGH, (2,), jnp.float32),
            "player_bullet_dir": spaces.Box(-1.0, 1.0, (2,), jnp.float32),
            "player_bullet_active": spaces.Box(0.0, 1.0, (1,), jnp.float32),

            # Enemies
            "enemy_pos": spaces.Box(POS_LOW, POS_HIGH, (NUM, 2), jnp.float32),
            # if you don't track enemy_alive keep bullets entries (they can be -100 when unused)
            "enemy_bullets": spaces.Box(POS_LOW, POS_HIGH, (NUM, 2), jnp.float32),
            "enemy_bullet_dirs": spaces.Box(-1.0, 1.0, (NUM, 2), jnp.float32),
            "enemy_bullet_active": spaces.Box(0.0, 1.0, (NUM,), jnp.float32),

            # Otto (can be teleported off-screen -> -100)
            "otto_pos": spaces.Box(POS_LOW, POS_HIGH, (2,), jnp.float32),

            # Game-level
            "score": spaces.Box(0.0, 1e6, (1,), jnp.float32),
            "lives": spaces.Box(0.0, 10.0, (1,), jnp.float32),  # lives small; float for compat
        })

    

    def image_space(self) -> spaces.Box:
        """
        Returns the pixel observation space of the environment.
        For now, we assume the game screen is 160x210 RGB.
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
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
    def render(self, state: BerzerkState) -> chex.Array:
        death_anim = state.player.death_timer > 0
        room_transition_anim = state.room_transition_timer > 0
        game_over_anim = state.game_over_timer > 0
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3), dtype=jnp.uint8)

        # Draw walls (assuming fixed positions based on bounds)
        room_idx = JaxBerzerk.get_room_index(state.room_counter)

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

            # if player entered from left or right -> block both side entries
            cond_lr = jnp.logical_and(state.entry_direction == 2, room_transition_anim == 0)
            raster = jax.lax.cond(cond_lr, lambda r: block_left(block_right(r)), lambda r: r, raster)

            cond_ll = jnp.logical_and(state.entry_direction == 3, room_transition_anim == 0)
            raster = jax.lax.cond(cond_ll, lambda r: block_left(block_right(r)), lambda r: r, raster)

            # if player entered from top -> block bottom entry
            cond_top = jnp.logical_and(state.entry_direction == 0, room_transition_anim == 0)
            raster = jax.lax.cond(cond_top, block_bottom, lambda r: r, raster)

            # if player entered from bottom -> block top entry
            cond_bottom = jnp.logical_and(state.entry_direction == 1, room_transition_anim == 0)
            raster = jax.lax.cond(cond_bottom, block_top, lambda r: r, raster)

            return raster
        
        raster = draw_entry_block(raster)


        def draw_enemy_wall_lines(raster, state):
            wall_x = 2
            line_height = 1
            line_length = 6

            draw_lines = ~room_transition_anim

            def draw_line(raster, enemy_y):
                y = jnp.clip(enemy_y.astype(jnp.int32) - 1, 0, raster.shape[0] - line_height)
                line = jnp.zeros((line_height, line_length, raster.shape[-1]), dtype=raster.dtype)
                return jax.lax.dynamic_update_slice(raster, line, (y, wall_x, 0))

            def maybe_draw(i, raster):
                is_alive = state.enemy.alive[i]
                enemy_y = state.enemy.pos[i][1]
                should_draw = jnp.logical_and(is_alive, draw_lines)
                return jax.lax.cond(
                    should_draw,
                    lambda _: draw_line(raster, enemy_y),
                    lambda _: raster,
                    operand=None
                )

            return jax.lax.fori_loop(0, state.enemy.pos.shape[0], maybe_draw, raster)

        raster = draw_enemy_wall_lines(raster, state)


        # Draw player bullet
        is_active = state.player.bullet_active[0]
        bullet_pos = state.player.bullet[0]
        bullet_dir = state.player.bullet_dir[0]

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


        def get_player_sprite():
            def death_animation():
                idx = (state.player.death_timer - 1) % 4
                return jnp.where(idx < 2, self.sprites['player_idle'], self.sprites['player_death'])

            dir = state.player.last_dir

            shoot_idx = jnp.select(
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
                default=2
            )

            shoot_frames = [
                lambda: self.sprites['player_shoot_up'],
                lambda: self.sprites['player_shoot_up'],
                lambda: self.sprites['player_shoot_right'],
                lambda: self.sprites['player_shoot_down'],
                lambda: self.sprites['player_shoot_down'],
                lambda: self.sprites['player_shoot_down_left'],
                lambda: self.sprites['player_shoot_left'],
                lambda: self.sprites['player_shoot_up_left'],
            ]

            move_frames = [
                lambda: self.sprites['player_move_1'],
                lambda: self.sprites['player_move_1'],
                lambda: self.sprites['player_move_2'],
                lambda: self.sprites['player_move_2'],
                lambda: self.sprites['player_idle'],
                lambda: self.sprites['player_idle'],
            ]

            def normal_or_shoot():
                return jax.lax.cond(
                    state.player.is_firing,
                    lambda: jax.lax.switch(shoot_idx, shoot_frames),
                    lambda: jax.lax.cond(
                        state.player.animation_counter > 0,
                        lambda: jax.lax.switch((state.player.animation_counter - 1) % 6, move_frames),
                        lambda: self.sprites['player_idle']
                    )
                )

            return jax.lax.cond(death_anim, death_animation, normal_or_shoot)


        player_sprite = get_player_sprite()

        player_frame_right = jr.get_sprite_frame(player_sprite, 0)

        player_frame = jax.lax.cond(
            jnp.logical_and(state.player.last_dir[0] < 0, ~state.player.is_firing),
            lambda: jnp.flip(player_frame_right, axis=1),  # flip horizontally
            lambda: player_frame_right
        )
        raster = jax.lax.cond(
            room_transition_anim,
            lambda r: r,
            lambda r: jr.render_at(r, state.player.pos[0], state.player.pos[1], player_frame),
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
            [214, 214, 214, 255],   # white
            [109, 210, 111, 255],   # green
            [239, 127, 128, 255],   # red
            [102, 158, 193, 255],   # blue
            [227, 205, 115, 255],   # yellow2
            [185, 96, 175, 255],    # pink
        ], dtype=jnp.uint8)

        def get_new_color(color_idx: jnp.ndarray) -> jnp.ndarray:
            return color_names[color_idx]

        def recolor_sprite(sprite, color_idx, original_color):
            new_color = get_new_color(color_idx)  # e.g. green, red, etc.

            # Mask (compare with original_color to get enemy pixels)
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
            counter = state.enemy.animation_counter[i]
            axis = state.enemy.move_axis[i]
            death_timer = state.enemy.death_timer[i]

            color_idx = get_enemy_color_index(state.room_counter)

            def recolor(name: str):
                original_color = jnp.array([210, 210, 64, 255], dtype=jnp.uint8)
                return maybe_recolor(self.sprites[name], color_idx, original_color)

            # Death Animation Frames
            death_sprites = (
                [recolor("enemy_death_3")] * 4 +
                [recolor("enemy_death_2")] * 2 +
                [recolor("enemy_death_1")] * 2
            )
            death_frames = [lambda sprite=s: sprite for s in death_sprites]

            # Normal Animation Frames
            idle_sprites = (
                [recolor("enemy_idle_1")] * 4 +
                [recolor("enemy_idle_2")] * 4 +
                [recolor("enemy_idle_3")] * 4 +
                [recolor("enemy_idle_4")] * 4 +
                [recolor("enemy_idle_5")] * 4 +
                [recolor("enemy_idle_6")] * 4 +
                [recolor("enemy_idle_7")] * 4 +
                [recolor("enemy_idle_8")] * 4
            )
            idle_frames = [lambda sprite=s: sprite for s in idle_sprites]

            move_horizontal_sprites = (
                [recolor("enemy_move_horizontal_1")] * 7 +
                [recolor("enemy_move_horizontal_2")] * 7
            )
            move_horizontal_frames = [lambda sprite=s: sprite for s in move_horizontal_sprites]

            move_vertical_sprites = (
                [recolor("enemy_move_vertical_1")] * 6 +
                [recolor("enemy_move_vertical_2")] * 6 +
                [recolor("enemy_move_vertical_1")] * 6 +
                [recolor("enemy_move_vertical_3")] * 6
            )
            move_vertical_frames = [lambda sprite=s: sprite for s in move_vertical_sprites]

            # Animation functions
            def death_animation():
                idx = (death_timer - 1) % 8
                return jax.lax.switch(idx, death_frames)

            def normal_animation():
                axis_idx = jnp.clip(axis + 1, 0, 2)  # 0=idle, 1=horizontal, 2=vertical
                return jax.lax.switch(
                    axis_idx,
                    [
                        lambda: jax.lax.switch((counter - 1) % 32, idle_frames),
                        lambda: jax.lax.switch((counter - 1) % 14, move_horizontal_frames),
                        lambda: jax.lax.switch((counter - 1) % 24, move_vertical_frames),
                    ]
                )

            # decide if death or normal animation
            return jax.lax.cond(death_timer > 0, death_animation, normal_animation)

        for i in range(state.enemy.pos.shape[0]):
            is_dying = state.enemy.death_timer[i] > 0
            pos = jax.lax.cond(is_dying, lambda: state.enemy.death_pos[i], lambda: state.enemy.pos[i])
            sprite = get_enemy_sprite(i)
            frame = jr.get_sprite_frame(sprite, 0)
            
            frame = jax.lax.cond(
                state.enemy.move_dir[i] < 0,
                lambda: jnp.flip(frame, axis=1),
                lambda: frame
            )

            raster = jax.lax.cond(
                room_transition_anim,
                lambda r: r,
                lambda r: jr.render_at(r, pos[0], pos[1], frame),
                raster
            )


        # Draw enemy bullets
        color_idx = get_enemy_color_index(state.room_counter)

        for i in range(state.enemy.bullets.shape[0]):
            is_active = state.enemy.bullet_active[i]
            bullet_pos = state.enemy.bullets[i]
            bullet_dir = state.enemy.bullet_dirs[i]

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
            (state.otto.anim_counter // 15) % 5,
            lambda s: s.get('evil_otto'), 
            lambda s: s.get('evil_otto_2'),
            self.sprites)

        otto_frame = jr.get_sprite_frame(otto_sprites, 0)
        jr.render_at(raster, state.otto.pos[0], state.otto.pos[1], otto_frame)

        raster = jax.lax.cond(
                state.otto.active,
                lambda r: jr.render_at(r, state.otto.pos[0], state.otto.pos[1], otto_frame),
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

                # Remove leading zeros
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
            
            show_bonus = state.enemy.clear_bonus_given

            return jax.lax.cond(
                jnp.logical_and(state.score == 0, ~show_bonus),
                skip_render,
                lambda: jax.lax.cond(
                    show_bonus,
                    lambda: draw_score(state.num_enemies * 10, self.consts.SCORE_OFFSET_X - 31),  # draw bonus further to left
                    lambda: draw_score(state.score, self.consts.SCORE_OFFSET_X)
                )
            )

        raster = jax.lax.cond(
            jnp.logical_not(room_transition_anim),
            render_scores,
            lambda r: r,
            raster
        )


        # Draw title
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


        # draw transition animation
        progress_transition = 1.0 - (state.room_transition_timer.astype(jnp.float32) / self.consts.TRANSITION_ANIMATION_FRAMES)
        raster = jax.lax.cond(
            room_transition_anim,
            lambda _: jax.lax.switch(
            state.entry_direction,
            [
                lambda: apply_bar_overlay(raster, progress_transition, 0),  # top
                lambda: apply_bar_overlay(raster, progress_transition, 1),  # bottom
                lambda: apply_bar_overlay(raster, progress_transition, 2),  # right
                lambda: apply_bar_overlay(raster, progress_transition, 2),  # left
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

        # draw black screen upon game over
        raster = jax.lax.cond(
            game_over_anim,
            lambda _: jnp.zeros_like(raster),
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