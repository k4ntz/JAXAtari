import jax
import jax.numpy as jnp
import jax.random as jrandom
from typing import Tuple
from functools import partial
import os

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.spaces import Discrete
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

from .montezuma2.core import Montezuma2Constants, Montezuma2State, Montezuma2Observation, Montezuma2Info
from .montezuma2.renderer import Montezuma2Renderer
from .montezuma2.rooms import load_room

class JaxMontezuma2(JaxEnvironment[Montezuma2State, Montezuma2Observation, Montezuma2Info, Montezuma2Constants]):
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
    ], dtype=jnp.int32)

    def __init__(self, consts: Montezuma2Constants = None):
        consts = consts or Montezuma2Constants()
        super().__init__(consts)
        self.renderer = Montezuma2Renderer(self.consts)
        
        sprite_path_0 = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma", "backgrounds", "base_collision_map.npy")
        col_map_0 = jnp.load(sprite_path_0)[:149, :, 0]
        room_col_0 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_0 = room_col_0.at[6:, 0:4].set(1) # Left wall only for room_0_0
        room_col_0 = room_col_0.at[147:149, 72:88].set(0) # Hole for ladder down to ROOM_1_1

        sprite_path_1 = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma", "backgrounds", "mid_room_collision_level_0.npy")
        col_map_1 = jnp.load(sprite_path_1)[:149, :, 0] # (149, 160)
        room_col_1 = jnp.where(col_map_1 > 0, 1, 0).astype(jnp.int32)
        # No side walls for room_0_1 as it's connected on both sides

        room_col_2 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_2 = room_col_2.at[6:, 156:160].set(1) # Right wall only for room_0_2
        
        room_col_3 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        self.ROOM_COLLISION_MAPS = jnp.stack([room_col_0, room_col_1, room_col_2, room_col_3])

    def reset(self, key: jrandom.PRNGKey) -> Tuple[Montezuma2Observation, Montezuma2State]:
        state = Montezuma2State(
            room_id=jnp.array(1, dtype=jnp.int32),
            lives=jnp.array(5, dtype=jnp.int32),
            score=jnp.array([0], dtype=jnp.int32),
            frame_count=jnp.array(0, dtype=jnp.int32),
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.INITIAL_PLAYER_Y, dtype=jnp.int32),
            player_vx=jnp.array(0, dtype=jnp.int32),
            player_vy=jnp.array(0, dtype=jnp.int32),
            player_dir=jnp.array(1, dtype=jnp.int32),
            is_jumping=jnp.array(0, dtype=jnp.int32),
            is_falling=jnp.array(0, dtype=jnp.int32),
            fall_start_y=jnp.array(0, dtype=jnp.int32),
            jump_counter=jnp.array(0, dtype=jnp.int32),
            is_climbing=jnp.array(0, dtype=jnp.int32),
            out_of_ladder_delay=jnp.array(0, dtype=jnp.int32),
            last_rope=jnp.array(-1, dtype=jnp.int32),
            last_ladder=jnp.array(-1, dtype=jnp.int32),
            enemies_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_y=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_direction=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_type=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_min_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_max_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_bouncing=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            ladders_x=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_top=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_bottom=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_active=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ropes_x=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_top=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_bottom=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_active=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            items_x=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_y=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_active=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_type=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            doors_x=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            doors_y=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            doors_active=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            conveyors_x=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_y=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_active=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_direction=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            lasers_x=jnp.zeros(self.consts.MAX_LASERS_PER_ROOM, dtype=jnp.int32),
            lasers_active=jnp.zeros(self.consts.MAX_LASERS_PER_ROOM, dtype=jnp.int32),
            laser_cycle=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            death_type=jnp.array(0, dtype=jnp.int32),
            inventory=jnp.zeros(1, dtype=jnp.int32),
            global_enemies_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ENEMIES_PER_ROOM), dtype=jnp.int32),
            global_enemies_type=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ENEMIES_PER_ROOM), dtype=jnp.int32),
            global_items_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ITEMS_PER_ROOM), dtype=jnp.int32),
            global_items_type=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_ITEMS_PER_ROOM), dtype=jnp.int32),
            global_doors_active=jnp.zeros((self.consts.MAX_ROOMS, self.consts.MAX_DOORS_PER_ROOM), dtype=jnp.int32),
            key=key
        )
        
        gia = state.global_items_active
        gia = gia.at[0, 0].set(1)
        gia = gia.at[1, 0].set(1)
        
        gda = state.global_doors_active
        gda = gda.at[1, 0].set(1)
        gda = gda.at[1, 1].set(1)
        
        gea = state.global_enemies_active
        gea = gea.at[1, 0].set(1)
        gea = gea.at[2, 0].set(1)
        gea = gea.at[2, 1].set(1)
        gea = gea.at[3, 0].set(1)

        gety = state.global_enemies_type
        # ROLL_SKULL = 1, BOUNCE_SKULL = 2, SPIDER = 3
        gety = gety.at[1, 0].set(1)
        gety = gety.at[2, 0].set(1)
        gety = gety.at[2, 1].set(1)
        gety = gety.at[3, 0].set(3)

        giy = state.global_items_type
        giy = giy.at[0, 0].set(1) # Gem in room 0
        state = state.replace(global_items_active=gia, global_doors_active=gda, global_enemies_active=gea, global_enemies_type=gety, global_items_type=giy)
        
        state = load_room(jnp.array(1, dtype=jnp.int32), state, self.consts)
        obs = self._get_observation(state)
        return obs, state
    def step(self, state: Montezuma2State, action: int) -> Tuple[Montezuma2Observation, Montezuma2State, float, bool, Montezuma2Info]:
        room_col_map = self.ROOM_COLLISION_MAPS[state.room_id]
        previous_score = state.score
        is_up = jnp.logical_or(action == Action.UP, jnp.logical_or(action == Action.UPRIGHT, action == Action.UPLEFT))
        is_up = jnp.logical_or(is_up, jnp.logical_or(action == Action.UPFIRE, jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE)))
        is_down = jnp.logical_or(action == Action.DOWN, jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNLEFT))
        is_down = jnp.logical_or(is_down, jnp.logical_or(action == Action.DOWNFIRE, jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE)))
        is_right = jnp.logical_or(action == Action.RIGHT, jnp.logical_or(action == Action.UPRIGHT, action == Action.DOWNRIGHT))
        is_right = jnp.logical_or(is_right, jnp.logical_or(action == Action.RIGHTFIRE, jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE)))
        is_left = jnp.logical_or(action == Action.LEFT, jnp.logical_or(action == Action.UPLEFT, action == Action.DOWNLEFT))
        is_left = jnp.logical_or(is_left, jnp.logical_or(action == Action.LEFTFIRE, jnp.logical_or(action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE)))
        is_fire = jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.UPFIRE, jnp.logical_or(action == Action.DOWNFIRE, jnp.logical_or(action == Action.RIGHTFIRE, action == Action.LEFTFIRE))))
        is_fire = jnp.logical_or(is_fire, jnp.logical_or(action == Action.UPRIGHTFIRE, jnp.logical_or(action == Action.UPLEFTFIRE, jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE))))
        
        # Player Velocity is locked during jump and falls
        is_in_air = jnp.logical_or(state.is_jumping == 1, state.is_falling == 1)
        new_vx = jax.lax.select(is_right, self.consts.PLAYER_SPEED, 0)
        new_vx = jax.lax.select(is_left, -self.consts.PLAYER_SPEED, new_vx)
        dx = jnp.where(is_in_air, state.player_vx, new_vx)
        new_player_dir = jnp.where(is_right, 1, jnp.where(is_left, -1, state.player_dir))

        player_mid_x = state.player_x + self.consts.PLAYER_WIDTH // 2
        player_feet_y = state.player_y + self.consts.PLAYER_HEIGHT - 1
        
        # 0. Ladder and Rope Climbing Logic
        new_out_of_ladder_delay = jnp.where(state.out_of_ladder_delay > 0, state.out_of_ladder_delay - 1, 0)

        def check_ladder(i, carry):
            c_on_ladder, c_ladder_idx = carry
            l_x = state.ladders_x[i]
            ladder_mid_x = l_x + 8
            l_top = state.ladders_top[i]
            l_bottom = state.ladders_bottom[i]

            is_aligned = jnp.logical_and(state.ladders_active[i] == 1, jnp.abs(player_mid_x - ladder_mid_x) <= 4)
            is_aligned = jnp.logical_and(is_aligned, new_out_of_ladder_delay == 0)

            # To get ON from top: must press DOWN near top
            get_on_top = jnp.logical_and(is_aligned, jnp.logical_and(is_down, jnp.abs(player_feet_y - l_top) <= 5))
            # To get ON from bottom: must press UP near bottom
            get_on_bottom = jnp.logical_and(is_aligned, jnp.logical_and(is_up, jnp.abs(player_feet_y - l_bottom) <= 5))

            # To stay ON: must be within the vertical bounds
            ladder_bottom_bound = jnp.where(l_bottom >= 148, 170, l_bottom + 1)
            ladder_top_bound = jnp.where(l_top <= 6, 0, l_top - 4)
            in_ladder_zone = jnp.logical_and(is_aligned, jnp.logical_and(player_feet_y >= ladder_top_bound, player_feet_y <= ladder_bottom_bound))

            on_this_ladder = jnp.where(state.is_climbing == 1, jnp.logical_and(in_ladder_zone, jnp.logical_or(state.last_ladder == i, state.last_ladder == -1)), jnp.logical_or(get_on_top, get_on_bottom))

            new_on_ladder = jnp.logical_or(c_on_ladder, on_this_ladder)
            new_ladder_idx = jnp.where(on_this_ladder, i, c_ladder_idx)
            return new_on_ladder, new_ladder_idx
        def check_rope(i, carry):
            c_on_rope, c_rope_idx = carry
            r_x = state.ropes_x[i]
            r_top = state.ropes_top[i]
            r_bottom = state.ropes_bottom[i]
            
            is_aligned = jnp.logical_and(state.ropes_active[i] == 1, jnp.abs(player_mid_x - r_x) <= 4)
            
            player_top_y = state.player_y
            intersect_y = jnp.logical_and(player_feet_y >= r_top, player_top_y <= r_bottom)
            
            catch_rope = jnp.logical_and(
                state.is_climbing == 0,
                jnp.logical_and(is_aligned, jnp.logical_and(intersect_y, state.last_rope != i))
            )

            get_on_top = jnp.logical_and(is_aligned, jnp.logical_and(is_down, jnp.abs(player_feet_y - r_top) <= 5))
            get_on_bottom = jnp.logical_and(is_aligned, jnp.logical_and(is_up, jnp.abs(player_feet_y - r_bottom) <= 5))

            in_rope_zone = jnp.logical_and(is_aligned, jnp.logical_and(player_feet_y >= r_top, player_feet_y <= r_bottom + 10))

            on_this_rope = jnp.where(state.is_climbing == 1, jnp.logical_and(in_rope_zone, jnp.logical_or(state.last_rope == i, state.last_rope == -1)), jnp.logical_or(catch_rope, jnp.logical_or(get_on_top, get_on_bottom)))

            new_on_rope = jnp.logical_or(c_on_rope, on_this_rope)
            new_rope_idx = jnp.where(on_this_rope, i, c_rope_idx)
            return new_on_rope, new_rope_idx

        can_ladder, ladder_idx = jax.lax.fori_loop(0, self.consts.MAX_LADDERS_PER_ROOM, check_ladder, (False, -1))
        can_rope, rope_idx = jax.lax.fori_loop(0, self.consts.MAX_ROPES_PER_ROOM, check_rope, (False, -1))

        is_jumping_off_ladder = jnp.logical_and(can_ladder, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(is_fire, jnp.logical_or(is_left, is_right))))
        is_moving_off_ladder = jnp.logical_and(can_ladder, jnp.logical_and(state.is_climbing == 1, jnp.logical_or(is_left, is_right)))
        abort_ladder = jnp.logical_or(is_jumping_off_ladder, is_moving_off_ladder)

        is_jumping_off_rope = jnp.logical_and(can_rope, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(is_fire, jnp.logical_or(is_left, is_right))))
        abort_rope = is_jumping_off_rope

        is_climbing_ladder = jnp.logical_and(can_ladder, jnp.logical_not(abort_ladder))
        is_climbing_rope = jnp.logical_and(can_rope, jnp.logical_not(abort_rope))

        is_climbing = jnp.where(jnp.logical_or(is_climbing_ladder, is_climbing_rope), 1, 0)

        # If we are aborting the ladder, or simply falling off, start the delay
        # But only if we were previously climbing!
        started_delay = jnp.logical_and(state.is_climbing == 1, is_climbing == 0)
        new_out_of_ladder_delay = jnp.where(started_delay, self.consts.OUT_OF_LADDER_DELAY, new_out_of_ladder_delay)

        target_climb_x = state.player_x
        target_climb_x = jnp.where(ladder_idx != -1, state.ladders_x[ladder_idx] + 8 - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        target_climb_x = jnp.where(rope_idx != -1, state.ropes_x[rope_idx] - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        
        current_x = jnp.where(is_climbing == 1, target_climb_x, state.player_x)
        
        def check_platform(y, x):
            x_m3 = jnp.clip(x - 3, 0, self.consts.WIDTH - 1)
            x_m2 = jnp.clip(x - 2, 0, self.consts.WIDTH - 1)
            x_m1 = jnp.clip(x - 1, 0, self.consts.WIDTH - 1)
            x_p1 = jnp.clip(x + 1, 0, self.consts.WIDTH - 1)
            x_p2 = jnp.clip(x + 2, 0, self.consts.WIDTH - 1)
            x_p3 = jnp.clip(x + 3, 0, self.consts.WIDTH - 1)
            return jnp.logical_or(
                room_col_map[y, x_m3] == 1,
                jnp.logical_or(
                    room_col_map[y, x_m2] == 1,
                    jnp.logical_or(
                        room_col_map[y, x_m1] == 1,
                        jnp.logical_or(
                            room_col_map[y, x] == 1,
                            jnp.logical_or(
                                room_col_map[y, x_p1] == 1,
                                jnp.logical_or(
                                    room_col_map[y, x_p2] == 1,
                                    room_col_map[y, x_p3] == 1
                                )
                            )
                        )
                    )
                )
            )
        
        # 1. Check if strictly on ground
        safe_x = jnp.clip(current_x + self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)
        safe_y = jnp.clip(player_feet_y + 1, 0, 148)
        on_ground = check_platform(safe_y, safe_x)
        
        def check_conveyor(i, on_grnd):
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            is_on_conveyor = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(player_feet_y == c_y, jnp.logical_and(player_mid_x >= c_x - 3, player_mid_x < c_x + 43))
            )
            return jnp.logical_or(on_grnd, is_on_conveyor)
        
        on_ground = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, check_conveyor, on_ground)

        # Update last_rope and last_ladder
        new_last_rope = jnp.where(on_ground, -1, state.last_rope)
        new_last_rope = jnp.where(is_jumping_off_rope, rope_idx, new_last_rope)

        new_last_ladder = jnp.where(on_ground, -1, state.last_ladder)
        new_last_ladder = jnp.where(is_climbing == 1, ladder_idx, new_last_ladder)

        # 2. Process Jump Initiation
        start_jump_normal = jnp.logical_and(is_fire, jnp.logical_and(on_ground, jnp.logical_and(state.is_jumping == 0, is_climbing == 0)))
        start_jump = jnp.logical_or(start_jump_normal, is_jumping_off_rope)
        is_jumping = jnp.where(start_jump, 1, state.is_jumping)
        is_jumping = jnp.where(is_climbing == 1, 0, is_jumping) # cancel jump
        jump_counter = jnp.where(start_jump, 0, state.jump_counter)

        # Horizontal velocity to carry over (used for momentum and animation tracking)
        current_vx = dx

        # 3. Calculate DY
        def get_jump_dy():
            dy_jump = -self.consts.JUMP_Y_OFFSETS[jump_counter]
            return dy_jump, jump_counter + 1, 1
            
        def get_fall_dy():
            pixel_1_below = check_platform(safe_y, safe_x)
            pixel_2_below = check_platform(jnp.clip(player_feet_y + 2, 0, 148), safe_x)
            
            def check_c_pixel2(i, p2b):
                c_x = state.conveyors_x[i]
                c_y = state.conveyors_y[i] - 1
                is_on = jnp.logical_and(
                    state.conveyors_active[i] == 1,
                    jnp.logical_and(player_feet_y + 1 == c_y, jnp.logical_and(safe_x >= c_x - 3, safe_x < c_x + 43))
                )
                return jnp.logical_or(p2b, is_on)
            pixel_2_below = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, check_c_pixel2, pixel_2_below)

            fall_dist = jnp.where(on_ground, 0, jnp.where(pixel_2_below, 1, self.consts.GRAVITY))
            return fall_dist, 0, 0
            
        def get_climb_dy():
            climb_dist = jnp.where(is_down, self.consts.PLAYER_SPEED, jnp.where(is_up, -self.consts.PLAYER_SPEED, 0))
            # Zero out vertical speed on the frame we catch the rope
            just_caught_rope = jnp.logical_and(state.is_climbing == 0, rope_idx != -1)
            climb_dist = jnp.where(just_caught_rope, 0, climb_dist)
            return climb_dist, 0, 0

        dy, new_jump_counter, new_is_jumping = jax.lax.cond(
            is_climbing == 1,
            get_climb_dy,
            lambda: jax.lax.cond(
                is_jumping == 1,
                get_jump_dy,
                get_fall_dy
            )
        )
        
        new_is_jumping = jnp.where(new_jump_counter >= 20, 0, new_is_jumping)
        
        # 4. Resolve Vertical Collision
        new_y = state.player_y + dy
        new_y = jnp.where(jnp.logical_and(is_climbing == 1, rope_idx != -1), jnp.maximum(new_y, state.ropes_top[rope_idx]), new_y)
        new_feet_y = new_y + self.consts.PLAYER_HEIGHT - 1
        
        new_top_y = jnp.clip(new_y, 0, 148)
        hit_ceiling = jnp.logical_and(dy < 0, jnp.logical_and(check_platform(new_top_y, safe_x), is_climbing == 0))
        new_y = jnp.where(hit_ceiling, state.player_y, new_y)
        new_is_jumping = jnp.where(hit_ceiling, 0, new_is_jumping)
        
        hit_floor_rm = jnp.logical_and(dy > 0, jnp.logical_and(is_climbing == 0, check_platform(jnp.clip(new_feet_y, 0, 148), safe_x)))
        snapped_y_rm = jnp.clip(new_feet_y, 0, 148) - self.consts.PLAYER_HEIGHT
        
        def check_c_hit_floor(i, carry):
            h_f, s_y = carry
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            crossed = jnp.logical_and(player_feet_y < c_y, new_feet_y >= c_y)
            is_hit = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(dy > 0, jnp.logical_and(crossed, jnp.logical_and(safe_x >= c_x - 3, safe_x < c_x + 43)))
            )
            return jnp.logical_or(h_f, is_hit), jnp.where(is_hit, c_y - self.consts.PLAYER_HEIGHT + 1, s_y)

        hit_floor, snapped_y = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, check_c_hit_floor, (hit_floor_rm, snapped_y_rm))

        new_y = jnp.where(hit_floor, snapped_y, new_y)

        # Stop climbing if we hit a floor (e.g. landing on a conveyor belt)
        is_climbing = jnp.where(hit_floor, 0, is_climbing)

        # Set is_falling state
        new_is_falling = jnp.where(jnp.logical_and(new_is_jumping == 0, hit_floor == False), jnp.where(dy > 0, 1, 0), 0)
        new_is_falling = jnp.where(is_climbing == 1, 0, new_is_falling)
        
        # 5. Resolve Horizontal with Wall Collision
        raw_new_x = current_x + dx
        transition_left = jnp.logical_and(raw_new_x < 0, jnp.logical_or(state.room_id == 1, state.room_id == 2))
        transition_right = jnp.logical_and(raw_new_x + self.consts.PLAYER_WIDTH > self.consts.WIDTH, jnp.logical_or(state.room_id == 0, state.room_id == 1))
        transition_down = jnp.logical_and(new_y >= 148, state.room_id == 0)
        transition_up = jnp.logical_and(new_y <= 2, state.room_id == 3)

        new_x = jnp.clip(raw_new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        new_left_x = jnp.clip(new_x, 0, self.consts.WIDTH - 1)
        new_right_x = jnp.clip(new_x + self.consts.PLAYER_WIDTH - 1, 0, self.consts.WIDTH - 1)
        
        front_x = jnp.where(dx > 0, new_right_x, new_left_x)
        
        check_y_top = jnp.clip(new_y, 0, 148)
        check_y_mid = jnp.clip(new_y + self.consts.PLAYER_HEIGHT // 2, 0, 148)
        check_y_bot = jnp.clip(new_y + self.consts.PLAYER_HEIGHT - 1, 0, 148)
        
        hit_wall = jnp.logical_or(
            room_col_map[check_y_top, front_x] == 1,
            jnp.logical_or(
                room_col_map[check_y_mid, front_x] == 1,
                room_col_map[check_y_bot, front_x] == 1
            )
        )
        
        # 5.5 Item Collection
        def collect_item(i, carry):
            keys_collected, items_active, current_score = carry
            i_x = state.items_x[i]
            i_y = state.items_y[i]
            i_active = items_active[i] == 1
            
            overlap_x = jnp.logical_and(new_left_x < i_x + 6, new_right_x >= i_x)
            overlap_y = jnp.logical_and(check_y_top < i_y + 8, check_y_bot >= i_y)
            overlap = jnp.logical_and(overlap_x, overlap_y)
            
            collect = jnp.logical_and(i_active, overlap)
            
            is_key = state.items_type[i] == 0
            new_keys = jnp.where(jnp.logical_and(collect, is_key), keys_collected + 1, keys_collected)
            new_items_active = jnp.where(collect, items_active.at[i].set(0), items_active)
            item_score = jnp.where(is_key, 100, 1000)
            new_score = jnp.where(collect, current_score + item_score, current_score)
            
            return new_keys, new_items_active, new_score

        current_keys, new_items_active, new_score = jax.lax.fori_loop(
            0, self.consts.MAX_ITEMS_PER_ROOM, collect_item,
            (state.inventory[0], state.items_active, state.score)
        )

        def check_door(i, carry):
            hit, keys_left, doors_active, current_score = carry
            d_x = state.doors_x[i]
            d_y = state.doors_y[i]
            d_active = doors_active[i] == 1
            in_x = jnp.logical_and(front_x >= d_x, front_x < d_x + 4)
            in_y_top = jnp.logical_and(check_y_top >= d_y, check_y_top < d_y + 38)
            in_y_mid = jnp.logical_and(check_y_mid >= d_y, check_y_mid < d_y + 38)
            in_y_bot = jnp.logical_and(check_y_bot >= d_y, check_y_bot < d_y + 38)
            in_y = jnp.logical_or(in_y_top, jnp.logical_or(in_y_mid, in_y_bot))

            hit_this_door = jnp.logical_and(d_active, jnp.logical_and(in_x, in_y))
            open_it = jnp.logical_and(hit_this_door, keys_left > 0)
            hit_as_wall = jnp.logical_and(hit_this_door, jnp.logical_not(open_it))

            new_hit = jnp.logical_or(hit, hit_as_wall)
            new_keys = jnp.where(open_it, keys_left - 1, keys_left)
            new_doors_active = jnp.where(open_it, doors_active.at[i].set(0), doors_active)
            new_score = jnp.where(open_it, current_score + 300, current_score)

            return new_hit, new_keys, new_doors_active, new_score

        hit_wall, current_keys, new_doors_active, new_score = jax.lax.fori_loop(
            0, self.consts.MAX_DOORS_PER_ROOM, check_door,
            (hit_wall, current_keys, state.doors_active, new_score)
        )
        new_x = jnp.where(jnp.logical_or(hit_wall, is_climbing == 1), current_x, new_x)
        
        new_mid_x = new_x + self.consts.PLAYER_WIDTH // 2
        new_feet_y_after = new_y + self.consts.PLAYER_HEIGHT - 1
        
        def apply_conveyor_physics(i, p_x):
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            is_on_conveyor = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(new_feet_y_after == c_y, jnp.logical_and(new_mid_x >= c_x - 3, new_mid_x < c_x + 43))
            )
            conveyor_velocity = jnp.mod(state.frame_count, 2) * state.conveyors_direction[i]
            return jax.lax.select(jnp.logical_and(is_on_conveyor, is_climbing == 0), p_x + conveyor_velocity, p_x)

        new_x = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, apply_conveyor_physics, new_x)
        new_x = jnp.clip(new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        
        current_vx = jnp.where(jnp.logical_or(hit_wall, hit_floor), 0, current_vx)
        
        # 6. Enemy Movement
        def move_enemy(i, carry):
            e_x, e_dir = carry
            current_x = e_x[i]
            current_dir = e_dir[i]
            
            # Move 1 pixel every 2 frames
            speed = jnp.where(jnp.mod(state.frame_count, 2) == 0, 1, 0)
            new_x = current_x + current_dir * speed
            
            # Bounce off walls
            e_y = state.enemies_y[i]
            hit_wall_left = jnp.logical_and(current_dir < 0, room_col_map[e_y + 8, jnp.clip(new_x, 0, self.consts.WIDTH - 1)] == 1)
            hit_wall_right = jnp.logical_and(current_dir > 0, room_col_map[e_y + 8, jnp.clip(new_x + 8, 0, self.consts.WIDTH - 1)] == 1)
            
            # Or boundaries
            hit_left = new_x <= state.enemies_min_x[i]
            hit_right = new_x >= state.enemies_max_x[i]
            
            bounce = jnp.logical_or(hit_left, jnp.logical_or(hit_right, jnp.logical_or(hit_wall_left, hit_wall_right)))
            
            final_dir = jnp.where(bounce, -current_dir, current_dir)
            final_x = jnp.where(bounce, current_x, new_x)
            
            return e_x.at[i].set(final_x), e_dir.at[i].set(final_dir)
            
        new_enemies_x, new_enemies_dir = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES_PER_ROOM, move_enemy, (state.enemies_x, state.enemies_direction))
        
        # 7. Dying Mechanism (Fall Damage & Enemy Collision)
        new_fall_start_y = jnp.where(
            jnp.logical_and(state.is_falling == 0, new_is_falling == 1),
            state.player_y,
            state.fall_start_y
        )
        fall_stopped = jnp.logical_and(state.is_falling == 1, new_is_falling == 0)
        fall_distance = new_y - state.fall_start_y
        died_from_fall = jnp.logical_and(fall_stopped, fall_distance > self.consts.MAX_FALL_DISTANCE)
        
        # Reset fall_start_y when not falling and not just stopped falling
        new_fall_start_y = jnp.where(
            jnp.logical_and(new_is_falling == 0, fall_stopped == 0),
            new_y,
            new_fall_start_y
        )
        
        def check_enemy_collision(i, carry):
            hit, enemies_active = carry
            e_x = new_enemies_x[i]
            bounce_offset = jax.lax.select(state.enemies_bouncing[i] == 1, self.consts.BOUNCE_OFFSETS[jnp.mod(state.frame_count // 4, 22)], 0)
            e_y = state.enemies_y[i] - bounce_offset
            e_active = enemies_active[i] == 1
            
            overlap_x = jnp.logical_and(new_left_x < e_x + 7, new_right_x >= e_x + 1)
            overlap_y = jnp.logical_and(check_y_top < e_y + 15, check_y_bot >= e_y + 1)
            overlap = jnp.logical_and(overlap_x, overlap_y)
            
            this_hit = jnp.logical_and(e_active, overlap)
            new_hit = jnp.logical_or(hit, this_hit)
            new_enemies_active = jnp.where(this_hit, enemies_active.at[i].set(0), enemies_active)
            
            return new_hit, new_enemies_active
            
        died_from_enemy, new_enemies_active = jax.lax.fori_loop(
            0, self.consts.MAX_ENEMIES_PER_ROOM, check_enemy_collision, 
            (False, state.enemies_active)
        )
        
        # Laser Collision
        new_laser_cycle = jnp.mod(state.laser_cycle + 1, 128)
        laser_active_now = jnp.logical_and(jnp.greater_equal(state.laser_cycle, 0), jnp.less(state.laser_cycle, 92))
        
        def check_laser_collision(i, hit):
            l_x = state.lasers_x[i]
            l_active = jnp.logical_and(state.lasers_active[i] == 1, laser_active_now)
            
            overlap_x = jnp.logical_and(new_left_x < l_x + 4, new_right_x >= l_x)
            overlap_y = jnp.logical_and(check_y_top < 46, check_y_bot >= 7)
            overlap = jnp.logical_and(overlap_x, overlap_y)
            
            return jnp.logical_or(hit, jnp.logical_and(l_active, overlap))
            
        died_from_laser = jax.lax.fori_loop(0, self.consts.MAX_LASERS_PER_ROOM, check_laser_collision, False)
        
        player_died = jnp.logical_or(died_from_fall, jnp.logical_or(died_from_enemy, died_from_laser))
        
        start_death = jnp.logical_and(state.death_timer == 0, player_died)
        new_death_timer = jnp.where(start_death, self.consts.DEATH_TIMER_FRAMES, 
                                    jnp.where(state.death_timer > 0, state.death_timer - 1, 0))
        
        death_type = jnp.where(died_from_fall, 1, jnp.where(died_from_enemy, 2, jnp.where(died_from_laser, 3, 0)))
        new_death_type = jnp.where(start_death, death_type, jnp.where(new_death_timer == 0, 0, state.death_type))

        respawn_now = jnp.logical_and(state.death_timer == 1, new_death_timer == 0)
        
        spawn_x = jnp.where(state.room_id == 0, 146, self.consts.INITIAL_PLAYER_X)
        spawn_y = jnp.where(state.room_id == 0, 27, self.consts.INITIAL_PLAYER_Y)
        
        new_lives = jnp.where(start_death, state.lives - 1, state.lives)
        final_x = jnp.where(respawn_now, spawn_x, jnp.where(new_death_timer > 0, state.player_x, new_x))
        final_y = jnp.where(respawn_now, spawn_y, jnp.where(new_death_timer > 0, state.player_y, new_y))
        final_vx = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, current_vx)
        final_vy = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, dy)
        final_player_dir = jnp.where(respawn_now, 1, jnp.where(new_death_timer > 0, state.player_dir, new_player_dir))
        final_is_jumping = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_is_jumping)
        final_is_falling = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_is_falling)
        final_is_climbing = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, is_climbing)
        final_jump_counter = jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_jump_counter)
        final_fall_start_y = jnp.where(respawn_now, spawn_y, jnp.where(new_death_timer > 0, state.fall_start_y, new_fall_start_y))
        
        state = state.replace(
            lives=new_lives,
            score=new_score,
            player_x=final_x,
            player_y=final_y,
            player_vx=final_vx,
            player_vy=final_vy,
            player_dir=final_player_dir,
            is_jumping=final_is_jumping,
            jump_counter=final_jump_counter,
            is_climbing=final_is_climbing,
            out_of_ladder_delay=jnp.where(jnp.logical_or(respawn_now, new_death_timer > 0), 0, new_out_of_ladder_delay),
            last_rope=new_last_rope,
            last_ladder=new_last_ladder,
            is_falling=final_is_falling,
            fall_start_y=final_fall_start_y,
            frame_count=state.frame_count + 1,
            enemies_x=new_enemies_x,
            enemies_active=new_enemies_active,
            enemies_direction=new_enemies_dir,
            inventory=jnp.array([current_keys], dtype=jnp.int32),
            items_active=new_items_active,
            doors_active=new_doors_active,
            laser_cycle=new_laser_cycle,
            death_timer=new_death_timer,
            death_type=new_death_type
        )

        transition_any = jnp.logical_or(jnp.logical_or(transition_left, transition_right), jnp.logical_or(transition_down, transition_up))
        new_room_id = jnp.where(transition_left, state.room_id - 1, jnp.where(transition_right, state.room_id + 1, jnp.where(transition_down, 3, jnp.where(transition_up, 0, state.room_id))))

        def transition_fn(state_in):
            st = state_in.replace(
                global_doors_active=state_in.global_doors_active.at[state_in.room_id].set(state_in.doors_active),
                global_items_active=state_in.global_items_active.at[state_in.room_id].set(state_in.items_active),
                global_enemies_active=state_in.global_enemies_active.at[state_in.room_id].set(state_in.enemies_active)
            )
            st = load_room(new_room_id, st, self.consts)
            new_px = jnp.where(transition_left, 148, jnp.where(transition_right, 4, current_x))
            new_py = jnp.where(transition_down, 6, jnp.where(transition_up, 140, jnp.where(new_room_id == 0, 27, 26)))
            return st.replace(
                player_x=new_px,
                player_y=new_py,
                fall_start_y=new_py,
                last_ladder=jnp.array(-1, dtype=jnp.int32),
                last_rope=jnp.array(-1, dtype=jnp.int32)
            )

        state = jax.lax.cond(transition_any, transition_fn, lambda x: x, state)

        obs = self._get_observation(state)
        reward = self._get_reward(previous_score, state.score)
        done = self._get_done(state)
        info = self._get_info(state)

        return obs, state, reward, done, info
    def action_space(self) -> Discrete:
        return Discrete(len(self.ACTION_SET))

    def observation_space(self) -> Discrete:
        return Discrete(1)
        
    def image_space(self) -> Discrete:
        return Discrete(1)

    def render(self, state: Montezuma2State) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: Montezuma2State) -> Montezuma2Observation:
        player_obs = ObjectObservation.create(
            x=jnp.array([state.player_x]),
            y=jnp.array([state.player_y]),
            width=jnp.array([self.consts.PLAYER_WIDTH]),
            height=jnp.array([self.consts.PLAYER_HEIGHT]),
            active=jnp.array([1])
        )
        
        enemies_obs = ObjectObservation.create(
            x=state.enemies_x + 1,
            y=state.enemies_y + 1 - jnp.where(state.enemies_bouncing == 1, self.consts.BOUNCE_OFFSETS[jnp.mod(state.frame_count // 4, 22)], 0),
            width=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 6),
            height=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 14),
            active=state.enemies_active
        )
        
        items_obs = ObjectObservation.create(
            x=state.items_x,
            y=state.items_y,
            width=jnp.full(self.consts.MAX_ITEMS_PER_ROOM, 6),
            height=jnp.full(self.consts.MAX_ITEMS_PER_ROOM, 8),
            active=state.items_active
        )
        
        conveyors_obs = ObjectObservation.create(
            x=state.conveyors_x,
            y=state.conveyors_y,
            width=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 40),
            height=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 5),
            active=state.conveyors_active
        )

        doors_obs = ObjectObservation.create(
            x=state.doors_x,
            y=state.doors_y,
            width=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 4),
            height=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 38),
            active=state.doors_active
        )

        ropes_obs = ObjectObservation.create(
            x=state.ropes_x,
            y=state.ropes_top,
            width=jnp.full(self.consts.MAX_ROPES_PER_ROOM, 1),
            height=state.ropes_bottom - state.ropes_top,
            active=state.ropes_active
        )

        return Montezuma2Observation(player=player_obs, enemies=enemies_obs, items=items_obs, conveyors=conveyors_obs, doors=doors_obs, ropes=ropes_obs)
    def _get_info(self, state: Montezuma2State) -> Montezuma2Info:
        return Montezuma2Info(lives=state.lives, room_id=state.room_id)

    def _get_reward(self, previous_score: jnp.ndarray, score: jnp.ndarray) -> float:
        return jnp.sum(score - previous_score).astype(jnp.float32)

    def _get_done(self, state: Montezuma2State) -> bool:
        return state.lives < 0
