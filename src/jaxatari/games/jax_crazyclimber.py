import os
import pygame
import sys
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex
import numpy as np

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


class CrazyClimberState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_lives: chex.Array
    player_score: chex.Array
    last_move_frame: chex.Array
    
    level: chex.Array
    level_banner_timer: chex.Array
    
    last_hand: chex.Array
    left_hand_x: chex.Array
    right_hand_x: chex.Array
    left_hand_y: chex.Array
    right_hand_y: chex.Array
    
    camera_y: chex.Array
    
    step_counter: chex.Array
    game_won: chex.Array
    respawn_timer: chex.Array
    death_timer: chex.Array
    # Falling objects (Level 2 hazard)
    obj_x: chex.Array  # (N,) int32 world x
    obj_y: chex.Array  # (N,) int32 world y
    obj_active: chex.Array  # (N,) bool


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class CrazyClimberObservation(NamedTuple):
    player: EntityPosition
    score: jnp.ndarray
    lives: jnp.ndarray


class CrazyClimberInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


class CrazyClimberConstants(NamedTuple):
    """Game-wide tunable constants (pixels, timing, layout, colors)."""
    WIDTH: int = 160
    HEIGHT: int = 210
    HUD_HEIGHT: int = 40
    
    BUILDING_HEIGHT: int = 800
    
    PLAYER_SIZE: Tuple[int, int] = (10, 14)
    PLAYER_START_X: int = 80
    PLAYER_START_Y: int = 790
    
    HAND_SPAN: int = 16
    
    NUM_BUILDINGS: int = 2
    BUILDING_WIDTH: int = 40
    BUILDING_COLS: int = 2
    BUILDING_ROWS: int = 40
    WINDOW_WIDTH: int = 14
    WINDOW_HEIGHT: int = 10
    LEDGE_HEIGHT: int = 3
    GAP_WIDTH: int = 40
    MARGINS: int = 20
    BOTTOM_MARGIN: int = 40
    CONNECTION_HEIGHT: int = 30
    
    ROW_HEIGHT: int = 11
    CLIMB_SPEED: int = 1
    MOVE_SPEED: int = 1
    MOVEMENT_THROTTLE: int = 2
    MAX_LIVES: int = 5
    WIN_HEIGHT: int = 15
    LEVEL1_TOP_Y: int = 350
    
    FALLING_OBJECT_SPEED: int = 2
    FALLING_OBJECT_SIZE: Tuple[int, int] = (4, 4)
    MAX_FALLING_OBJECTS: int = 16
    
    WINDOW_OPEN: int = 0
    WINDOW_CLOSING: int = 1
    WINDOW_CLOSED: int = 2
    
    WINDOW_CLOSE_TIMER: int = 240
    WINDOW_REOPEN_TIMER: int = 180
    FALLING_OBJECT_SPAWN: int = 180
    
    BACKGROUND_COLOR: Tuple[int, int, int] = (0, 0, 0)
    BUILDING_COLOR: Tuple[int, int, int] = (66, 72, 200)
    LEDGE_COLOR: Tuple[int, int, int] = (66, 72, 200)
    WINDOW_OPEN_COLOR: Tuple[int, int, int] = (0, 0, 0)
    WINDOW_CLOSING_COLOR: Tuple[int, int, int] = (30, 30, 30)
    WINDOW_CLOSED_COLOR: Tuple[int, int, int] = (66, 72, 200)
    PLAYER_COLOR: Tuple[int, int, int] = (210, 210, 64)
    FALLING_OBJECT_COLOR: Tuple[int, int, int] = (255, 255, 255)
    SINGLE_BUILDING_COLOR: Tuple[int, int, int] = (230, 210, 50)

    WINDOW_ANIM_INTERVAL: int = 60
    WINDOW_ANIM_DURATION: int = 150
    WINDOW_ANIM_MAX_ACTIVE: int = 12
    CLOSED_TOP_ROWS: int = 3


class JaxCrazyClimber(JaxEnvironment):
    """JAX-friendly Crazy Climber environment with pygame-based human loop."""
    def __init__(self, consts=None):
        consts = consts or CrazyClimberConstants()
        super().__init__(consts)
        self.renderer = CrazyClimberRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
        ]

    def reset(self, key=None):
        """Reset game to initial state."""
        # Always start horizontally centered on screen (works for both levels)
        start_x = self.consts.WIDTH // 2
        start_y = self.consts.PLAYER_START_Y
        # Start camera near the bottom so the spawn is visible immediately
        start_cam = int(np.clip(start_y - (self.consts.HEIGHT // 2), 0, self.consts.BUILDING_HEIGHT - self.consts.HEIGHT))
        
        # Falling objects init arrays
        nobj = self.consts.MAX_FALLING_OBJECTS
        zx = jnp.zeros((nobj,), dtype=jnp.int32)
        zy = jnp.zeros((nobj,), dtype=jnp.int32)
        za = jnp.zeros((nobj,), dtype=bool)

        state = CrazyClimberState(
            player_x=jnp.array(start_x, dtype=jnp.int32),
            player_y=jnp.array(start_y, dtype=jnp.int32),
            player_lives=jnp.array(self.consts.MAX_LIVES, dtype=jnp.int32),
            player_score=jnp.array(0, dtype=jnp.int32),
            last_move_frame=jnp.array(0, dtype=jnp.int32),
            
            level=jnp.array(1, dtype=jnp.int32),
            level_banner_timer=jnp.array(0, dtype=jnp.int32),
            
            last_hand=jnp.array(0, dtype=jnp.int32),
            left_hand_x=jnp.array(start_x - self.consts.HAND_SPAN // 2, dtype=jnp.int32),
            right_hand_x=jnp.array(start_x + self.consts.HAND_SPAN // 2, dtype=jnp.int32),
            left_hand_y=jnp.array(start_y, dtype=jnp.int32),
            right_hand_y=jnp.array(start_y, dtype=jnp.int32),
            
            camera_y=jnp.array(start_cam, dtype=jnp.int32),
            
            step_counter=jnp.array(0, dtype=jnp.int32),
            game_won=jnp.array(False, dtype=bool),
            respawn_timer=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            obj_x=zx,
            obj_y=zy,
            obj_active=za,
        )
        
        return self._get_observation(state), state
        
    def _get_observation(self, state):
        """Convert game state to observation for the agent."""
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0], dtype=jnp.int32),
            height=jnp.array(self.consts.PLAYER_SIZE[1], dtype=jnp.int32),
        )
        
        return CrazyClimberObservation(
            player=player,
            score=state.player_score,
            lives=state.player_lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """Execute one game step with authentic hand-over-hand climbing."""
        
        # Throttle input: allow moves only every N frames
        can_move_this_frame = (state.step_counter - state.last_move_frame) >= self.consts.MOVEMENT_THROTTLE
        
        # Ignore down actions (up-only game)
        action = jnp.where(
            jnp.isin(action, jnp.array([Action.DOWN, Action.DOWNLEFT, Action.DOWNRIGHT])),
            Action.NOOP,
            action
        )
        
        action = jnp.where(
            jnp.logical_and(jnp.logical_not(can_move_this_frame), action != Action.NOOP),
            Action.NOOP,
            action
        )
        
        
        new_left_x = state.left_hand_x
        new_left_y = state.left_hand_y
        new_right_x = state.right_hand_x
        new_right_y = state.right_hand_y
        new_last_hand = state.last_hand
        
        # One step climbs exactly one row height
        climb_distance = self.consts.CLIMB_SPEED * self.consts.ROW_HEIGHT
        
        
        def process_upleft():
            can_move = (state.last_hand == 1)
            target_y = state.left_hand_y - climb_distance
            
            can_move_up = jnp.logical_and(can_move, target_y > self.consts.WIN_HEIGHT)
            new_left_x = state.left_hand_x
            new_left_y = jnp.where(can_move_up, target_y, state.left_hand_y)
            new_last_hand = jnp.where(can_move_up, jnp.array(0, dtype=jnp.int32), state.last_hand)
            
            return new_left_x, new_left_y, new_last_hand
        
        def process_upright():
            can_move = (state.last_hand == 0)
            target_y = state.right_hand_y - climb_distance
            
            can_move_up = jnp.logical_and(can_move, target_y > self.consts.WIN_HEIGHT)
            new_right_x = state.right_hand_x
            new_right_y = jnp.where(can_move_up, target_y, state.right_hand_y)
            new_last_hand = jnp.where(can_move_up, jnp.array(1, dtype=jnp.int32), state.last_hand)
            
            return new_right_x, new_right_y, new_last_hand
        
        def process_left_level1():
            new_left_x_temp = state.left_hand_x - self.consts.MOVE_SPEED
            new_right_x_temp = state.right_hand_x - self.consts.MOVE_SPEED
            left_building_start = self.consts.MARGINS
            left_building_end = self.consts.MARGINS + self.consts.BUILDING_WIDTH
            right_building_start = self.consts.MARGINS + self.consts.BUILDING_WIDTH + self.consts.GAP_WIDTH
            current_y = state.player_y
            in_bottom_connection = current_y >= self.consts.BUILDING_HEIGHT - self.consts.BOTTOM_MARGIN
            if_in_gap = jnp.logical_and(new_left_x_temp > left_building_end, new_left_x_temp < right_building_start)
            can_enter_gap = jnp.logical_or(in_bottom_connection, jnp.logical_not(if_in_gap))
            valid_move = jnp.logical_and(new_left_x_temp >= left_building_start, can_enter_gap)
            new_left_x = jnp.where(valid_move, new_left_x_temp, state.left_hand_x)
            new_right_x = jnp.where(valid_move, new_right_x_temp, state.right_hand_x)
            return new_left_x, new_right_x

        def process_left_level2():
            new_left_x_temp = state.left_hand_x - self.consts.MOVE_SPEED
            new_right_x_temp = state.right_hand_x - self.consts.MOVE_SPEED
            building_start = (self.consts.WIDTH - self.consts.BUILDING_WIDTH) // 2
            building_end = building_start + self.consts.BUILDING_WIDTH
            valid_move = jnp.logical_and(new_left_x_temp >= building_start, new_right_x_temp < building_end)
            new_left_x = jnp.where(valid_move, new_left_x_temp, state.left_hand_x)
            new_right_x = jnp.where(valid_move, new_right_x_temp, state.right_hand_x)
            return new_left_x, new_right_x

        def process_left():
            return jax.lax.cond(
                state.level == 1,
                process_left_level1,
                process_left_level2
            )
        
        def process_right_level1():
            new_left_x_temp = state.left_hand_x + self.consts.MOVE_SPEED
            new_right_x_temp = state.right_hand_x + self.consts.MOVE_SPEED
            left_building_end = self.consts.MARGINS + self.consts.BUILDING_WIDTH
            right_building_start = self.consts.MARGINS + self.consts.BUILDING_WIDTH + self.consts.GAP_WIDTH
            right_building_end = self.consts.MARGINS + (2 * self.consts.BUILDING_WIDTH) + self.consts.GAP_WIDTH
            current_y = state.player_y
            in_bottom_connection = current_y >= self.consts.BUILDING_HEIGHT - self.consts.BOTTOM_MARGIN
            if_in_gap = jnp.logical_and(new_left_x_temp > left_building_end, new_left_x_temp < right_building_start)
            can_enter_gap = jnp.logical_or(in_bottom_connection, jnp.logical_not(if_in_gap))
            valid_move = jnp.logical_and(new_right_x_temp < right_building_end, can_enter_gap)
            new_left_x = jnp.where(valid_move, new_left_x_temp, state.left_hand_x)
            new_right_x = jnp.where(valid_move, new_right_x_temp, state.right_hand_x)
            return new_left_x, new_right_x

        def process_right_level2():
            new_left_x_temp = state.left_hand_x + self.consts.MOVE_SPEED
            new_right_x_temp = state.right_hand_x + self.consts.MOVE_SPEED
            building_start = (self.consts.WIDTH - self.consts.BUILDING_WIDTH) // 2
            building_end = building_start + self.consts.BUILDING_WIDTH
            valid_move = jnp.logical_and(new_left_x_temp >= building_start, new_right_x_temp < building_end)
            new_left_x = jnp.where(valid_move, new_left_x_temp, state.left_hand_x)
            new_right_x = jnp.where(valid_move, new_right_x_temp, state.right_hand_x)
            return new_left_x, new_right_x

        def process_right():
            return jax.lax.cond(
                state.level == 1,
                process_right_level1,
                process_right_level2
            )
        
        # Alternate hands: UP moves the other hand; diagonals force a side
        def should_move_left_hand():
            return jnp.logical_or(
                jnp.logical_and(action == Action.UP, state.last_hand == 1),
                action == Action.UPLEFT
            )
            
        def should_move_right_hand():
            return jnp.logical_or(
                jnp.logical_and(action == Action.UP, state.last_hand == 0),
                action == Action.UPRIGHT
            )
        
        left_x, left_y, last_hand_upleft = jax.lax.cond(
            should_move_left_hand(),
            process_upleft,
            lambda: (state.left_hand_x, state.left_hand_y, state.last_hand)
        )
        
        right_x, right_y, last_hand_upright = jax.lax.cond(
            should_move_right_hand(),
            process_upright,
            lambda: (state.right_hand_x, state.right_hand_y, state.last_hand)
        )
        
        is_left_action = jnp.logical_or(action == Action.LEFT, action == Action.DOWNLEFT)
        is_right_action = jnp.logical_or(action == Action.RIGHT, action == Action.DOWNRIGHT)
        
        left_x_h, right_x_h = jax.lax.cond(
            is_left_action,
            process_left,
            lambda: (left_x, right_x)
        )
        
        left_x_h, right_x_h = jax.lax.cond(
            is_right_action,
            process_right,
            lambda: (left_x_h, right_x_h)
        )
        
        new_left_x = left_x_h
        new_left_y = left_y
        new_right_x = right_x_h
        new_right_y = right_y
        new_last_hand = jnp.where(should_move_left_hand(), last_hand_upleft, 
                               jnp.where(should_move_right_hand(), last_hand_upright, state.last_hand))
        
        new_player_x = (new_left_x + new_right_x) // 2
        new_player_y = jnp.minimum(new_left_y, new_right_y)
        
        # Camera follows the higher hand; clamp to texture bounds
        target_camera_y = new_player_y - (self.consts.HEIGHT // 2)
        max_camera_y = self.consts.BUILDING_HEIGHT - self.consts.HEIGHT
        new_camera_y = jnp.clip(target_camera_y, 0, max_camera_y)
        
        
    # Detect overlap with any actively-closing window
        window_collision = jnp.array(False, dtype=bool)
        
        def check_collision_for_group(ag_offset, collision_state):
            ag = (state.step_counter // self.consts.WINDOW_ANIM_INTERVAL) - ag_offset
            phase = state.step_counter - (ag * self.consts.WINDOW_ANIM_INTERVAL)
            
            # Is this group within its 0..2*duration active window?
            group_active = jnp.logical_and(ag >= 0, 
                          jnp.logical_and(phase >= 0, phase < (2 * self.consts.WINDOW_ANIM_DURATION)))
            
            # Triangle-wave shutter progress f in [0,1] then [1,0]
            f_closing = phase / self.consts.WINDOW_ANIM_DURATION
            f_opening = 1.0 - (phase - self.consts.WINDOW_ANIM_DURATION) / self.consts.WINDOW_ANIM_DURATION
            f = jnp.where(phase < self.consts.WINDOW_ANIM_DURATION, f_closing, f_opening)
            
            # Only dangerous when shutters > 10% closed (more responsive)
            should_check = jnp.logical_and(group_active, f > 0.1)
            
            rows = self.consts.BUILDING_ROWS
            cols_per = self.consts.BUILDING_COLS
            row_idx = (ag * 37 + 11) % rows
            # level-aware building index and base position
            building_idx_level1 = (ag * 53 + 7) % 2
            building_idx = jnp.where(state.level == 1, building_idx_level1, jnp.array(0, dtype=jnp.int32))
            col_idx = (ag * 97 + 3) % cols_per
            
            # World-space window rectangle (anchored to building, not camera)
            margin = self.consts.MARGINS
            b_width = self.consts.BUILDING_WIDTH
            gap = self.consts.GAP_WIDTH
            bottom_margin = self.consts.BOTTOM_MARGIN
            row_h = self.consts.ROW_HEIGHT
            w_w = self.consts.WINDOW_WIDTH
            w_h = self.consts.WINDOW_HEIGHT
            
            building1_start = margin
            building2_start = building1_start + b_width + gap
            single_start = (self.consts.WIDTH - b_width) // 2
            base_x_level1 = jnp.where(building_idx == 0, building1_start, building2_start)
            base_x = jnp.where(state.level == 1, base_x_level1, single_start)
            
            spacing = 4
            total_w_width = cols_per * w_w + (cols_per - 1) * spacing
            left_pad = (b_width - total_w_width) // 2
            win_x = base_x + left_pad + col_idx * (w_w + spacing)
            win_y = self.consts.BUILDING_HEIGHT - bottom_margin - (row_idx * row_h) - w_h
            
            player_left = new_player_x - self.consts.PLAYER_SIZE[0] // 2
            player_right = player_left + self.consts.PLAYER_SIZE[0]
            player_top = new_player_y
            player_bottom = player_top + self.consts.PLAYER_SIZE[1]
            
            win_right = win_x + w_w
            win_bottom = win_y + w_h
            
            # AABB overlap vs player sprite
            x_overlap = jnp.logical_and(player_left < win_right, player_right > win_x)
            y_overlap = jnp.logical_and(player_top < win_bottom, player_bottom > win_y)
            
            group_collision = jnp.logical_and(should_check, jnp.logical_and(x_overlap, y_overlap))
            
            return jnp.logical_or(collision_state, group_collision)
        
        for offset in range(self.consts.WINDOW_ANIM_MAX_ACTIVE):
            window_collision = check_collision_for_group(offset, window_collision)

        # Also treat level-1 closed top rows as deadly, regardless of animation
        def check_static_top_collision():
            cols_per = self.consts.BUILDING_COLS
            margin = self.consts.MARGINS
            b_width = self.consts.BUILDING_WIDTH
            gap = self.consts.GAP_WIDTH
            bottom_margin = self.consts.BOTTOM_MARGIN
            row_h = self.consts.ROW_HEIGHT
            w_w = self.consts.WINDOW_WIDTH
            w_h = self.consts.WINDOW_HEIGHT
            rows = self.consts.BUILDING_ROWS
            top_rows = self.consts.CLOSED_TOP_ROWS

            building1_start = margin
            building2_start = building1_start + b_width + gap

            spacing = 4
            total_w_width = cols_per * w_w + (cols_per - 1) * spacing
            left_pad = (b_width - total_w_width) // 2

            coll = jnp.array(False, dtype=bool)
            for b_idx in [0, 1]:
                base_x = building1_start if b_idx == 0 else building2_start
                for row in range(rows - top_rows, rows):
                    win_y = self.consts.BUILDING_HEIGHT - bottom_margin - (row * row_h) - w_h
                    for col in range(cols_per):
                        win_x = base_x + left_pad + col * (w_w + spacing)
                        win_right = win_x + w_w
                        win_bottom = win_y + w_h
                        player_left = new_player_x - self.consts.PLAYER_SIZE[0] // 2
                        player_right = player_left + self.consts.PLAYER_SIZE[0]
                        player_top = new_player_y
                        player_bottom = player_top + self.consts.PLAYER_SIZE[1]
                        x_overlap = jnp.logical_and(player_left < win_right, player_right > win_x)
                        y_overlap = jnp.logical_and(player_top < win_bottom, player_bottom > win_y)
                        coll = jnp.logical_or(coll, jnp.logical_and(x_overlap, y_overlap))
            return coll

        static_top_collision = jnp.where(state.level == 1, check_static_top_collision(), jnp.array(False, dtype=bool))
        window_collision = jnp.logical_or(window_collision, static_top_collision)

    # Start death effect; block re-collision during respawn grace period
        life_lost = jnp.logical_and(window_collision, state.respawn_timer == 0)
        death_started = jnp.logical_and(life_lost, state.death_timer == 0)

        # old death timer/respawn values will be recomputed after merging emptiness collisions

        # Also die if in empty space (outside building footprints or in the gap above the bottom bridge)
        def empty_collision_level1(px, py):
            margin = self.consts.MARGINS
            bw = self.consts.BUILDING_WIDTH
            gap = self.consts.GAP_WIDTH
            bottom_margin = self.consts.BOTTOM_MARGIN
            b1_start = margin
            b1_end = b1_start + bw
            b2_start = b1_end + gap
            b2_end = b2_start + bw
            in_left = jnp.logical_and(px >= b1_start, px < b1_end)
            in_right = jnp.logical_and(px >= b2_start, px < b2_end)
            in_gap = jnp.logical_and(px >= b1_end, px < b2_start)
            in_bottom_bridge = (py >= (self.consts.BUILDING_HEIGHT - bottom_margin - 1))
            # empty when not in either tower, and not in the gap within the bottom bridge band
            return jnp.logical_and(jnp.logical_not(jnp.logical_or(in_left, in_right)),
                                   jnp.logical_not(jnp.logical_and(in_gap, in_bottom_bridge)))

        def empty_collision_level2(px, py):
            bw = self.consts.BUILDING_WIDTH
            single_start = (self.consts.WIDTH - bw) // 2
            single_end = single_start + bw
            return jnp.logical_or(px < single_start, px >= single_end)

        empty_space_collision = jax.lax.cond(
            state.level == 1,
            lambda: empty_collision_level1(new_player_x, new_player_y),
            lambda: empty_collision_level2(new_player_x, new_player_y)
        )

        # Falling objects update (active only on level 2)
        def update_falling_objects():
            nobj = self.consts.MAX_FALLING_OBJECTS
            speed = self.consts.FALLING_OBJECT_SPEED
            size_w, size_h = self.consts.FALLING_OBJECT_SIZE

            # Move existing active objects down
            new_y = state.obj_y + jnp.where(state.obj_active, jnp.array(speed, dtype=jnp.int32), 0)
            # Deactivate those that go past bottom
            still_on_screen = new_y < self.consts.BUILDING_HEIGHT
            new_active = jnp.logical_and(state.obj_active, still_on_screen)

            # Determine spawn opportunity
            spawn_interval = self.consts.FALLING_OBJECT_SPAWN
            group = state.step_counter // jnp.array(spawn_interval, dtype=jnp.int32)
            # spawn one object per interval when group changed
            should_spawn = (state.step_counter % jnp.array(spawn_interval, dtype=jnp.int32)) == 0

            # Choose window position deterministically similar to shutters
            rows = self.consts.BUILDING_ROWS
            cols_per = self.consts.BUILDING_COLS
            row_idx = (group * 29 + 5) % rows
            col_idx = (group * 41 + 3) % cols_per

            bw = self.consts.BUILDING_WIDTH
            w_w = self.consts.WINDOW_WIDTH
            w_h = self.consts.WINDOW_HEIGHT
            spacing = 4
            total_w_width = cols_per * w_w + (cols_per - 1) * spacing
            left_pad = (bw - total_w_width) // 2
            single_start = (self.consts.WIDTH - bw) // 2
            base_x = single_start
            bottom_margin = self.consts.BOTTOM_MARGIN
            row_h = self.consts.ROW_HEIGHT
            spawn_win_x = base_x + left_pad + col_idx * (w_w + spacing)
            spawn_win_y = self.consts.BUILDING_HEIGHT - bottom_margin - (row_idx * row_h) - w_h
            spawn_x = spawn_win_x + (w_w // 2) - (size_w // 2)
            spawn_y = spawn_win_y

            # Find first inactive slot
            inactive_mask = jnp.logical_not(new_active)
            any_slot = jnp.any(inactive_mask)
            first_idx = jnp.argmax(inactive_mask.astype(jnp.int32))

            def do_spawn(args):
                x_arr, y_arr, a_arr = args
                x_arr = x_arr.at[first_idx].set(jnp.array(spawn_x, dtype=jnp.int32))
                y_arr = y_arr.at[first_idx].set(jnp.array(spawn_y, dtype=jnp.int32))
                a_arr = a_arr.at[first_idx].set(jnp.array(True, dtype=bool))
                return x_arr, y_arr, a_arr

            new_x = state.obj_x
            new_x, new_y2, new_active2 = new_x, new_y, new_active
            # Only spawn on level 2 and if time and slot
            do_it = jnp.logical_and(should_spawn, any_slot)
            new_x, new_y2, new_active2 = jax.lax.cond(
                do_it,
                do_spawn,
                lambda args: args,
                operand=(new_x, new_y2, new_active2)
            )

            # Collision with player
            player_left = new_player_x - self.consts.PLAYER_SIZE[0] // 2
            player_right = player_left + self.consts.PLAYER_SIZE[0]
            player_top = new_player_y
            player_bottom = player_top + self.consts.PLAYER_SIZE[1]

            # Compute overlaps across objects
            obj_left = new_x
            obj_right = new_x + size_w
            obj_top = new_y2
            obj_bottom = new_y2 + size_h
            x_overlap = jnp.logical_and(obj_left < player_right, obj_right > player_left)
            y_overlap = jnp.logical_and(obj_top < player_bottom, obj_bottom > player_top)
            hit_any = jnp.any(jnp.logical_and(new_active2, jnp.logical_and(x_overlap, y_overlap)))

            return new_x, new_y2, new_active2, hit_any

        def passthrough_objects():
            return state.obj_x, state.obj_y, state.obj_active, jnp.array(False, dtype=bool)

        obj_x_new, obj_y_new, obj_active_new, object_hit = jax.lax.cond(
            state.level == 2,
            update_falling_objects,
            passthrough_objects
        )

        # Merge emptiness death with window/top and object collisions
        life_lost = jnp.logical_and(
            jnp.logical_or(jnp.logical_or(window_collision, empty_space_collision), object_hit),
            state.respawn_timer == 0
        )
        death_started = jnp.logical_and(life_lost, state.death_timer == 0)

        # Level transition and respawn handling
        level1_top = self.consts.LEVEL1_TOP_Y
        win_thresh = jnp.where(
            state.level == 1,
            jnp.array(level1_top, dtype=jnp.int32),
            jnp.array(self.consts.WIN_HEIGHT, dtype=jnp.int32)
        )
        reached_top = (new_player_y <= win_thresh)
        transition_to_level2 = jnp.logical_and(reached_top, state.level == 1)

        # Compute level-2 start positions (centered single building)
        lvl2_center_x = (self.consts.WIDTH // 2)
        lvl2_start_x = lvl2_center_x
        lvl2_start_y = self.consts.PLAYER_START_Y
        lvl2_left_x = lvl2_start_x - self.consts.HAND_SPAN // 2
        lvl2_right_x = lvl2_start_x + self.consts.HAND_SPAN // 2

        # Hard reset position/camera/hands either for death respawn or level transition
        base_x_after = jnp.where(transition_to_level2, jnp.array(lvl2_start_x, dtype=jnp.int32), new_player_x)
        base_y_after = jnp.where(transition_to_level2, jnp.array(lvl2_start_y, dtype=jnp.int32), new_player_y)
        # If transitioning to level 2, compute its starting camera at the bottom
        lvl2_cam_after = jnp.clip(lvl2_start_y - (self.consts.HEIGHT // 2), 0, self.consts.BUILDING_HEIGHT - self.consts.HEIGHT)
        base_cam_after = jnp.where(transition_to_level2, lvl2_cam_after, new_camera_y)

        # Recompute death timer and respawn based on merged collision flags
        new_death_timer = jnp.where(
            death_started,
            jnp.array(60, dtype=jnp.int32),
            jnp.maximum(state.death_timer - 1, 0)
        )

        # Respawn only when the death timer expires
        should_respawn = jnp.logical_and(state.death_timer == 1, life_lost)
        new_lives = jnp.where(should_respawn, state.player_lives - 1, state.player_lives)

        center_x = jnp.array(self.consts.WIDTH // 2, dtype=jnp.int32)
        respawn_x = jnp.where(should_respawn, center_x, base_x_after)
        respawn_y = jnp.where(should_respawn, self.consts.PLAYER_START_Y, base_y_after)
        # If we respawn, aim camera to bottom spawn as well (not the very top)
        respawn_cam_target = jnp.clip(self.consts.PLAYER_START_Y - (self.consts.HEIGHT // 2), 0, self.consts.BUILDING_HEIGHT - self.consts.HEIGHT)
        respawn_camera = jnp.where(should_respawn, respawn_cam_target, base_cam_after)

        left_x_after = jnp.where(transition_to_level2, jnp.array(lvl2_left_x, dtype=jnp.int32), new_left_x)
        right_x_after = jnp.where(transition_to_level2, jnp.array(lvl2_right_x, dtype=jnp.int32), new_right_x)
        left_y_after = jnp.where(transition_to_level2, jnp.array(lvl2_start_y, dtype=jnp.int32), new_left_y)
        right_y_after = jnp.where(transition_to_level2, jnp.array(lvl2_start_y, dtype=jnp.int32), new_right_y)
        last_hand_after = jnp.where(transition_to_level2, jnp.array(0, dtype=jnp.int32), new_last_hand)

        respawn_left_x = jnp.where(should_respawn, center_x - self.consts.HAND_SPAN // 2, left_x_after)
        respawn_right_x = jnp.where(should_respawn, center_x + self.consts.HAND_SPAN // 2, right_x_after)
        respawn_left_y = jnp.where(should_respawn, self.consts.PLAYER_START_Y, left_y_after)
        respawn_right_y = jnp.where(should_respawn, self.consts.PLAYER_START_Y, right_y_after)
        respawn_last_hand = jnp.where(should_respawn, jnp.array(0, dtype=jnp.int32), last_hand_after)

        # Clear all falling objects on respawn or level transition
        clear_objs = jnp.logical_or(should_respawn, transition_to_level2)
        obj_x_after = jnp.where(clear_objs, jnp.zeros_like(obj_x_new), obj_x_new)
        obj_y_after = jnp.where(clear_objs, jnp.zeros_like(obj_y_new), obj_y_new)
        obj_active_after = jnp.where(clear_objs, jnp.zeros_like(obj_active_new), obj_active_new)

        new_respawn_timer = jnp.where(
            should_respawn,
            jnp.array(60, dtype=jnp.int32),
            jnp.maximum(state.respawn_timer - 1, 0)
        )

        # Win only after level 2
        win_condition = jnp.logical_and(reached_top, state.level == 2)

        # Sparse reward: +1 per upward row climbed; +100 on win
        height_change = state.player_y - new_player_y
        height_reward = jnp.where(height_change > 0, 1.0, 0.0)
        win_reward = jnp.where(win_condition, 100.0, 0.0)

        reward = height_reward + win_reward

        new_score = state.player_score + jnp.where(height_change > 0, height_change, 0)

        # Track last frame with movement for throttling
        movement_occurred = jnp.logical_not(action == Action.NOOP)
        new_last_move_frame = jnp.where(
            movement_occurred,
            state.step_counter + 1,
            state.last_move_frame
        )

        new_level = jnp.where(transition_to_level2, jnp.array(2, dtype=jnp.int32), state.level)
        new_level_banner = jnp.where(
            transition_to_level2,
            jnp.array(90, dtype=jnp.int32),
            jnp.maximum(state.level_banner_timer - 1, 0)
        )

        new_state = CrazyClimberState(
            player_x=respawn_x,
            player_y=respawn_y,
            player_lives=new_lives,
            player_score=new_score,
            last_move_frame=new_last_move_frame,
            level=new_level,
            level_banner_timer=new_level_banner,

            last_hand=respawn_last_hand,
            left_hand_x=respawn_left_x,
            right_hand_x=respawn_right_x,
            left_hand_y=respawn_left_y,
            right_hand_y=respawn_right_y,

            camera_y=respawn_camera,

            step_counter=state.step_counter + 1,
            game_won=jnp.logical_or(state.game_won, win_condition),
            respawn_timer=new_respawn_timer,
            death_timer=new_death_timer,
            obj_x=obj_x_after,
            obj_y=obj_y_after,
            obj_active=obj_active_after,
        )

        done = self.is_terminal(new_state)

        info = CrazyClimberInfo(
            time=state.step_counter,
            all_rewards=jnp.array([reward]),
        )

        return self._get_observation(new_state), new_state, reward, done, info
    
    def is_terminal(self, state):
        """Check if game is over."""
        all_lives_lost = state.player_lives <= 0
        game_completed = state.game_won
        return jnp.logical_or(all_lives_lost, game_completed)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
    
    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.BUILDING_HEIGHT, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.WIDTH, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.BUILDING_HEIGHT, shape=(), dtype=jnp.int32),
            }),
            "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
            "lives": spaces.Box(low=0, high=self.consts.MAX_LIVES, shape=(), dtype=jnp.int32),
        })
    
    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CrazyClimberObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.player.x.flatten(),
            obs.player.y.flatten(),
            obs.player.width.flatten(),
            obs.player.height.flatten(),
            obs.score.flatten(),
            obs.lives.flatten(),
        ])
    
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """Render the game state using the renderer."""
        return self.renderer.render(state)
    
    def _get_obs(self, state):
        """Get observation from state."""
        return self._get_observation(state)
    
    def _get_reward(self, previous_state, state):
        """Calculate reward for this step."""
        height_change = previous_state.player_y - state.player_y
        height_reward = jnp.where(height_change > 0, 1.0, 0.0)
        win_reward = jnp.where(state.game_won, 100.0, 0.0)
        return height_reward + win_reward
    
    def _get_all_rewards(self, state, new_state, action):
        """Get all reward components for this step."""
        reward = self._get_reward(state, new_state)
        return jnp.array([reward])
    
    def image_space(self) -> spaces:
        """Returns the image space of the environment."""
        return spaces.Box(
            low=0, 
            high=255, 
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3), 
            dtype=jnp.uint8
        )
    
    def _get_info(self, state, all_rewards=None):
        """Extracts information from the environment state."""
        return CrazyClimberInfo(
            time=state.step_counter,
            all_rewards=all_rewards if all_rewards is not None else jnp.array([0.0])
        )
    
    def _get_done(self, state):
        """Determines if the environment state is terminal."""
        # Game is done if no lives left or game is won
        return jnp.logical_or(state.player_lives <= 0, state.game_won)
    
class CrazyClimberRenderer(JAXGameRenderer):
    def __init__(self, consts: CrazyClimberConstants = None):
        super().__init__()
        self.consts = consts or CrazyClimberConstants()
        # Pre-generate backgrounds for both levels
        self.background_level1 = self._generate_background(level=1)
        self.background_level2 = self._generate_background(level=2)
    
    def _generate_background(self, level=1):
        WIDTH = self.consts.WIDTH
        HEIGHT = self.consts.BUILDING_HEIGHT
        HUD = self.consts.HUD_HEIGHT
        
        BLACK = jnp.array([0, 0, 0], dtype=jnp.uint8)
        BLUE = jnp.array([66, 72, 200], dtype=jnp.uint8)
        YELLOW = jnp.array(list(self.consts.SINGLE_BUILDING_COLOR), dtype=jnp.uint8)
        LEDGE_BLUE = jnp.array([86, 92, 220], dtype=jnp.uint8)
        LEDGE_YELLOW = jnp.array([240, 230, 100], dtype=jnp.uint8)
        
        bg = jnp.full((HEIGHT, WIDTH, 3), BLACK, dtype=jnp.uint8)
        
        building_width = self.consts.BUILDING_WIDTH
        gap_width = self.consts.GAP_WIDTH
        margin = self.consts.MARGINS
        bottom_margin = self.consts.BOTTOM_MARGIN
        connection_height = self.consts.CONNECTION_HEIGHT
        
        building1_start = margin
        building1_end = building1_start + building_width
        building2_start = building1_end + gap_width
        building2_end = building2_start + building_width
        
        if level == 1:
            for start_x, end_x in [(building1_start, building1_end), (building2_start, building2_end)]:
                bg = bg.at[HUD+connection_height:HEIGHT-bottom_margin, start_x:end_x].set(BLUE)
            for start_x, end_x in [(building1_start, building1_end), (building2_start, building2_end)]:
                bg = bg.at[HUD:HUD+connection_height, start_x:end_x].set(BLUE)
            # Connect the two buildings at the very top across the gap
            bg = bg.at[HUD:HUD+connection_height, building1_start:building2_end].set(BLUE)
            bg = bg.at[HEIGHT-bottom_margin:, building1_start:building2_end].set(BLUE)
        else:
            single_start = (WIDTH - building_width) // 2
            single_end = single_start + building_width
            bg = bg.at[HUD:HEIGHT-bottom_margin, single_start:single_end].set(YELLOW)
            # Extend single building down to the very bottom like level 1's bridge
            bg = bg.at[HEIGHT-bottom_margin:, single_start:single_end].set(YELLOW)
        
        window_width = self.consts.WINDOW_WIDTH
        window_height = self.consts.WINDOW_HEIGHT
        ledge_height = self.consts.LEDGE_HEIGHT
        row_height = self.consts.ROW_HEIGHT
        rows = self.consts.BUILDING_ROWS
        cols_per_building = self.consts.BUILDING_COLS
        if level == 1:
            for start_x, end_x in [(building1_start, building1_end), (building2_start, building2_end)]:
                for row in range(rows + 1):
                    ledge_y = HEIGHT - bottom_margin - (row * row_height)
                    if ledge_y >= HUD+connection_height and ledge_y < HEIGHT - bottom_margin:
                        bg = bg.at[ledge_y:ledge_y + ledge_height, start_x:end_x].set(LEDGE_BLUE)
            for row in range(rows + 1):
                ledge_y = HEIGHT - bottom_margin - (row * row_height)
                if ledge_y >= HEIGHT - bottom_margin:
                    bg = bg.at[ledge_y:ledge_y + ledge_height, building1_start:building2_end].set(LEDGE_BLUE)
            closed_top_rows = 3
            for building_idx, (start_x, end_x) in enumerate([(building1_start, building1_end), (building2_start, building2_end)]):
                for row in range(rows):
                    for col in range(cols_per_building):
                        total_windows_width = cols_per_building * window_width + (cols_per_building - 1) * 4
                        left_padding = (self.consts.BUILDING_WIDTH - total_windows_width) // 2
                        window_x = start_x + left_padding + col * (window_width + 4)
                        window_y = HEIGHT - bottom_margin - (row * row_height) - window_height
                        if (window_x + window_width <= end_x and 
                            window_y >= HUD+connection_height and 
                            window_y + window_height <= HEIGHT - bottom_margin):
                            if row >= rows - closed_top_rows:
                                # top finish: blackspace instead of closed sash
                                bg = bg.at[window_y:window_y + window_height, window_x:window_x + window_width].set(BLACK)
                            else:
                                bg = bg.at[window_y:window_y + window_height, window_x:window_x + window_width].set(BLACK)
            connection_cols = 8
            connection_rows = 5
            for row in range(connection_rows):
                for col in range(connection_cols):
                    span = (building2_end - building1_start)
                    total_windows_width = connection_cols * window_width + (connection_cols - 1) * 4
                    left_padding = (span - total_windows_width) // 2
                    window_x = building1_start + left_padding + col * (window_width + 4)
                    window_y = HEIGHT - bottom_margin + 3 + row * (window_height + 4)
                    if (window_x + window_width <= building2_end - 3 and 
                        window_y + window_height <= HEIGHT - 3):
                        bg = bg.at[window_y:window_y + window_height, 
                                   window_x:window_x + window_width].set(BLACK)
        else:
            single_start = (WIDTH - building_width) // 2
            single_end = single_start + building_width
            for row in range(rows + 1):
                ledge_y = HEIGHT - bottom_margin - (row * row_height)
                if ledge_y >= HUD and ledge_y < HEIGHT - bottom_margin:
                    bg = bg.at[ledge_y:ledge_y + ledge_height, single_start:single_end].set(LEDGE_YELLOW)
            for row in range(rows):
                for col in range(cols_per_building):
                    total_windows_width = cols_per_building * window_width + (cols_per_building - 1) * 4
                    left_padding = (self.consts.BUILDING_WIDTH - total_windows_width) // 2
                    window_x = single_start + left_padding + col * (window_width + 4)
                    window_y = HEIGHT - bottom_margin - (row * row_height) - window_height
                    if (window_x + window_width <= single_end and 
                        window_y >= HUD and 
                        window_y + window_height <= HEIGHT - bottom_margin):
                        bg = bg.at[window_y:window_y + window_height, 
                                   window_x:window_x + window_width].set(BLACK)
            # Add bottom part windows under the single building (like level 1 connection area)
            conn_cols = 6
            conn_rows = 4
            for r in range(conn_rows):
                for c in range(conn_cols):
                    total_windows_width = conn_cols * window_width + (conn_cols - 1) * 4
                    left_padding = (building_width - total_windows_width) // 2
                    window_x = single_start + left_padding + c * (window_width + 4)
                    window_y = HEIGHT - bottom_margin + 3 + r * (window_height + 4)
                    if (window_x + window_width <= single_end - 3 and 
                        window_y + window_height <= HEIGHT - 3):
                        bg = bg.at[window_y:window_y + window_height, 
                                   window_x:window_x + window_width].set(BLACK)
        
        return bg

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        max_cam = self.consts.BUILDING_HEIGHT - (self.consts.HEIGHT)
        camera_y = jnp.clip(state.camera_y, 0, max_cam)
        bg = jnp.where(
            state.level == 1,
            self.background_level1,
            self.background_level2
        )
        raster = jax.lax.dynamic_slice(
            bg,
            (camera_y, 0, 0),
            (self.consts.HEIGHT, self.consts.WIDTH, 3)
        )
        
        player_viewport_y = state.player_y - camera_y
        safe_y = jnp.clip(player_viewport_y, 0, self.consts.HEIGHT - self.consts.PLAYER_SIZE[1])
        # Clamp player drawing to the screen bounds (not building bounds)
        safe_x = jnp.clip(
            state.player_x - self.consts.PLAYER_SIZE[0] // 2,
            0,
            self.consts.WIDTH - self.consts.PLAYER_SIZE[0]
        )
        
        skin_color = jnp.array([220, 180, 140], dtype=jnp.uint8)
        face_color = jnp.array([230, 190, 150], dtype=jnp.uint8)
        shirt_color = jnp.array([200, 50, 50], dtype=jnp.uint8)
        pants_color = jnp.array([80, 80, 160], dtype=jnp.uint8)
        hair_color = jnp.array([100, 70, 40], dtype=jnp.uint8)
        shoe_color = jnp.array([60, 40, 20], dtype=jnp.uint8)
        eye_color = jnp.array([50, 50, 50], dtype=jnp.uint8)
        
        player_patch = jnp.full((self.consts.PLAYER_SIZE[1], self.consts.PLAYER_SIZE[0], 3), 
                               skin_color, dtype=jnp.uint8)
        
        if self.consts.PLAYER_SIZE[1] >= 3 and self.consts.PLAYER_SIZE[0] >= 6:
            player_patch = player_patch.at[0:2, 1:-1].set(hair_color)
            player_patch = player_patch.at[1:3, 0].set(hair_color)
            player_patch = player_patch.at[1:3, -1].set(hair_color)
        
        if self.consts.PLAYER_SIZE[1] >= 4 and self.consts.PLAYER_SIZE[0] >= 6:
            player_patch = player_patch.at[1:4, 1:-1].set(face_color)
            
        if self.consts.PLAYER_SIZE[1] >= 3 and self.consts.PLAYER_SIZE[0] >= 8:
            player_patch = player_patch.at[2:3, 2:3].set(eye_color)
            player_patch = player_patch.at[2:3, -3:-2].set(eye_color)
        
        shirt_start_y = 4
        shirt_end_y = 9
        if shirt_end_y <= self.consts.PLAYER_SIZE[1]:
            player_patch = player_patch.at[shirt_start_y:shirt_end_y, 1:-1].set(shirt_color)
        
        if self.consts.PLAYER_SIZE[0] >= 6 and shirt_end_y <= self.consts.PLAYER_SIZE[1]:
            player_patch = player_patch.at[shirt_start_y:shirt_end_y, 0].set(skin_color)
            player_patch = player_patch.at[shirt_start_y:shirt_end_y, -1].set(skin_color)
        
        pants_start_y = 9
        pants_end_y = self.consts.PLAYER_SIZE[1] - 2
        if pants_start_y < pants_end_y:
            player_patch = player_patch.at[pants_start_y:pants_end_y, 1:-1].set(pants_color)
            player_patch = player_patch.at[pants_start_y:pants_end_y, 0].set(pants_color)
            player_patch = player_patch.at[pants_start_y:pants_end_y, -1].set(pants_color)
        
        if self.consts.PLAYER_SIZE[1] >= 2:
            shoe_start_y = self.consts.PLAYER_SIZE[1] - 2
            player_patch = player_patch.at[shoe_start_y:, :].set(shoe_color)
        
        raster = jax.lax.dynamic_update_slice(raster, player_patch, (safe_y, safe_x, 0))

        return raster


class CrazyClimberPygameRenderer:
    """Pygame-based human playable wrapper for CrazyClimber."""
    
    def __init__(self, game, scale=3):
        self.game = game
        self.scale = scale
        pygame.init()
        self.screen = pygame.display.set_mode((game.consts.WIDTH * scale, game.consts.HEIGHT * scale))
        pygame.display.set_caption("Crazy Climber")
        self.clock = pygame.time.Clock()
    
    def render_frame(self, state):
        """Render a single frame with overlays."""
        # Get base frame from environment renderer
        frame = self.game.renderer.render(state)
        
        frame_np = jnp.array(frame)
        scaled_frame = jnp.repeat(jnp.repeat(frame_np, self.scale, axis=0), self.scale, axis=1)
        surface = pygame.surfarray.make_surface(jnp.transpose(scaled_frame, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Add window shutter overlays
        self._render_window_overlays(state)
        
        # Add game overlays (death, level banner, HUD)
        self._render_game_overlays(state)
        
        pygame.display.flip()
    
    def _render_window_overlays(self, state):
        """Render animated window shutters."""
        consts = self.game.consts
        ticks = int(state.step_counter)
        interval = int(consts.WINDOW_ANIM_INTERVAL)
        duration = int(consts.WINDOW_ANIM_DURATION)
        max_active = int(consts.WINDOW_ANIM_MAX_ACTIVE)

        if interval > 0 and duration > 0:
            rows = int(consts.BUILDING_ROWS)
            cols_per = int(consts.BUILDING_COLS)
            total_windows = rows * 2 * cols_per

            margin = int(consts.MARGINS)
            b_width = int(consts.BUILDING_WIDTH)
            gap = int(consts.GAP_WIDTH)
            bottom_margin = int(consts.BOTTOM_MARGIN)
            row_h = int(consts.ROW_HEIGHT)
            w_w = int(consts.WINDOW_WIDTH)
            w_h = int(consts.WINDOW_HEIGHT)

            spacing = 4
            total_w_width = cols_per * w_w + (cols_per - 1) * spacing
            left_pad = (b_width - total_w_width) // 2

            building1_start = margin
            building1_end = building1_start + b_width
            building2_start = building1_end + gap
            building2_end = building2_start + b_width
            single_start = (consts.WIDTH - b_width) // 2
            single_end = single_start + b_width

            def window_x_for(building_idx: int, col: int) -> int:
                if state.level == 1:
                    base_x = building1_start if building_idx == 0 else building2_start
                else:
                    base_x = single_start
                return base_x + left_pad + col * (w_w + spacing)

            group = ticks // interval
            active_groups = [group - k for k in range(max_active)]

            cam_y = int(state.camera_y)

            bcolor = tuple(consts.BUILDING_COLOR) if state.level == 1 else tuple(consts.SINGLE_BUILDING_COLOR)

            for ag in active_groups:
                if ag < 0 or total_windows <= 0:
                    continue
                row_idx = (ag * 37 + 11) % rows
                building_idx = (ag * 53 + 7) % 2 if state.level == 1 else 0
                col_idx = (ag * 97 + 3) % cols_per
                phase = ticks - (ag * interval)
                if phase < 0 or phase >= (2 * duration):
                    continue

                win_x = window_x_for(building_idx, col_idx)
                win_y = consts.BUILDING_HEIGHT - bottom_margin - (row_idx * row_h) - w_h

                if win_y < 0 or win_x < 0:
                    continue

                view_y = win_y - cam_y
                if view_y >= consts.HEIGHT or (view_y + w_h) <= 0:
                    continue

                if phase < duration:
                    f = phase / duration
                else:
                    f = 1.0 - (phase - duration) / duration

                h = int(w_h * f)
                if h <= 0:
                    continue

                xs = win_x * self.scale
                ys = view_y * self.scale
                ww = w_w * self.scale
                wh = w_h * self.scale
                hs = h * self.scale

                pygame.draw.rect(self.screen, bcolor, pygame.Rect(xs, ys, ww, hs))
                pygame.draw.rect(self.screen, bcolor, pygame.Rect(xs, ys + wh - hs, ww, hs))
        
        # Render falling objects on level 2
        if state.level == 2:
            size_w, size_h = self.game.consts.FALLING_OBJECT_SIZE
            cam_y = int(state.camera_y)
            for i in range(int(state.obj_x.shape[0])):
                if not bool(state.obj_active[i]):
                    continue
                ox = int(state.obj_x[i])
                oy = int(state.obj_y[i]) - cam_y
                if oy + size_h < 0 or oy >= self.game.consts.HEIGHT:
                    continue
                pygame.draw.rect(
                    self.screen,
                    tuple(self.game.consts.FALLING_OBJECT_COLOR),
                    pygame.Rect(ox * self.scale, oy * self.scale, size_w * self.scale, size_h * self.scale)
                )
    
    def _render_game_overlays(self, state):
        """Render death effects, level banners, and HUD."""
        consts = self.game.consts
        
        # Death overlay (Level 1 only)
        if state.death_timer > 0 and state.level == 1:
            flash_intensity = int(state.death_timer * 2)
            flash_surface = pygame.Surface((consts.WIDTH * self.scale, consts.HEIGHT * self.scale))
            flash_surface.set_alpha(min(flash_intensity, 100))
            flash_surface.fill((255, 50, 50))
            self.screen.blit(flash_surface, (0, 0))
            
            death_font = pygame.font.SysFont("Courier", 32, bold=True)
            death_text = death_font.render("CRUSHED!", True, (255, 255, 255))
            text_x = consts.WIDTH * self.scale // 2 - death_text.get_width() // 2
            text_y = consts.HEIGHT * self.scale // 2 - death_text.get_height() // 2
            self.screen.blit(death_text, (text_x, text_y))
        
        # Level 2 banner
        elif state.level == 2 and state.level_banner_timer > 0:
            banner_font = pygame.font.SysFont("Courier", 32, bold=True)
            banner_text = banner_font.render("LEVEL 2", True, (255, 255, 0))
            bx = consts.WIDTH * self.scale // 2 - banner_text.get_width() // 2
            by = consts.HEIGHT * self.scale // 2 - banner_text.get_height() // 2
            self.screen.blit(banner_text, (bx, by))
        
        # HUD
        hud_h = consts.HUD_HEIGHT * self.scale
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(0, 0, consts.WIDTH * self.scale, hud_h))
        
        hud_color = (128, 255, 128)
        font = pygame.font.SysFont("Courier", 24, bold=True)
        top_score_text = font.render(f"{int(state.player_score):05d}", True, hud_color)
        cur_score_text = font.render(f"{int(state.player_score):06d}", True, hud_color)
        center_x = consts.WIDTH * self.scale // 2
        self.screen.blit(top_score_text, (center_x - top_score_text.get_width() // 2, 4))
        self.screen.blit(cur_score_text, (center_x - cur_score_text.get_width() // 2, 4 + top_score_text.get_height()))
        
        # Lives indicator
        life_w, life_h = 12, 10
        for i in range(int(state.player_lives)):
            x = 10 + i * (life_w + 8)
            y = 6
            pygame.draw.rect(self.screen, hud_color, pygame.Rect(x, y, life_w, life_h), 2)
        
        # Win/Game Over messages
        if state.game_won:
            win_text = font.render("YOU WIN!", True, (0, 255, 0))
            self.screen.blit(win_text, (consts.WIDTH * self.scale // 2 - 80, consts.HEIGHT * self.scale // 2))
        elif state.player_lives <= 0:
            over_text = font.render("GAME OVER", True, (255, 0, 0))
            self.screen.blit(over_text, (consts.WIDTH * self.scale // 2 - 100, consts.HEIGHT * self.scale // 2))


def main():
    """Human playable version."""
    game = JaxCrazyClimber()
    obs, state = game.reset()
    
    renderer = CrazyClimberPygameRenderer(game)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        action = Action.NOOP
        
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            action = Action.UPLEFT
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            action = Action.UPRIGHT
        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
            action = Action.DOWNLEFT
        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
            action = Action.DOWNRIGHT
        elif keys[pygame.K_UP]:
            action = Action.UP
        elif keys[pygame.K_LEFT]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Action.RIGHT
        elif keys[pygame.K_DOWN]:
            action = Action.DOWN
        
        obs, state, reward, done, info = game.step(state, action)
        
        # Render using the pygame renderer
        renderer.render_frame(state)
        
        clock.tick(45)
        
        if done:
            pygame.time.wait(3000)
            running = False
    
    pygame.quit()


if __name__ == "__main__":
    main()