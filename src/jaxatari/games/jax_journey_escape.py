import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class JourneyEscapeConstants(NamedTuple):

    starting_score: int = 50000

    # frame countdown for timer
    countdown_frame: int = 50  # countdown decreases by one second every 50 frames
    start_countdown: int = 59  # for a 59-second countdown

    screen_width: int = 160
    screen_height: int = 210

    background_frame_switch: int = 4 # set to 0 for static background
    background_frames_amount: int = 8 # defined order of 2*4 sprites

    player_width: int = 8
    player_height: int = 28
    start_player_x: int = 44  # Fixed x position
    start_player_y: int = 160  # Fixed y position
    player_speed: int = 1
    player_frame_switch: int = 16 # should match ALE

    # border of the valid game space
    top_border: int = 33
    bottom_border: int = screen_height - player_height - 47
    left_border: int = 8
    right_border: int = screen_width - 8

    # Line where the obstacles disappear behind
    bottom_blue_area: int = screen_height - 24

    # player position rules
    min_player_position_y: int = top_border + (screen_height // 4)

    # Standard sizes
    obstacle_width: int = 8
    obstacle_height: int = 10

    # Big sizes (2x)
    big_obstacle_width: int = 16
    big_obstacle_height: int = 20

    obstacle_frame_switch: int = 17  # should match ALE
    obstacle_speed_px_per_frame: int = 1
    row_spawn_period_frames: int = 50  # spawn every N frames # ToDo: calibrate
    hit_cooldown_frames: int = 17

    # Define the Width and Height for every ID (0 to 9)
    # 0: Fence, 1: Robot, 2: Heart, 3: Manager, 4: Light, 5: BigRobot, 6: BigHeart, 7: BigManager, 8: BigLight, 9: BigFireFace
    TYPE_WIDTHS: Tuple[int, ...] = (32, 8, 8, 8, 8, 16, 16, 16, 16, 17)
    TYPE_HEIGHTS: Tuple[int, ...] = (15, 15, 15, 15, 15, 15, 15, 15, 15, 15)

    MAX_OBS = 64

    """
        0: Fence
        1: Blue Robot
        2: Heart
        3: Manager
        4: Lightbulb
        5: Big Blue Robot
        6: Big Heart
        7: Big Manager
        8: Big Lightbulb
        9: Big Fire Face
    """

    # Blinking Effect
    lightbulb_on_duration: int = 17
    lightbulb_off_duration: int = 49

    # Invincible Effect
    INV_DURATION_ROBOT: int = 6 * countdown_frame # 6 seconds @ 50fps
    INV_DURATION_FIREFACE: int = 100000 # (longer than the max possible game time of ~60s)

    # True if the object stops movement / drags player
    IS_SOLID: chex.Array = jnp.array([
        True,           # 0: Fence
        False,          # 1: Blue Robot
        True,           # 2: Heart
        True,           # 3: Manager
        True,           # 4: Lightbulb
        False,          # 5: Big Blue Robot
        True,           # 6: Big Heart
        True,           # 7: Big Manager
        True,           # 8: Big Lightbulb
        False,          # 9: Big Fire Face
    ])

    # Points deducted on contact
    SCORE_PENALTIES: chex.Array = jnp.array([
        0,              # 0: Fence
        0,              # 1: Blue Robot
        -300,           # 2: Heart
        -2000,          # 3: Manager
        -600,           # 4: Lightbulb
        0,              # 5: Big Blue Robot
        -300,           # 6: Big Heart
        -2000,          # 7: Big Manager
        -600,           # 8: Big Lightbulb
        9900,           # 9: Big Fire Face
    ])
    # ---------------------CHANGE END-----------------------

    # predefined groups: [type, amount, spacing in px]
    obstacle_groups: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 0),      # Fence
        (1, 2, 20),     # Blue Robots
        (5, 1, 0),      # Big Robot (1)

        (2, 1, 0),      # Heart (1)
        (2, 2, 55),     # Hearts (2, Wide spacing)
        (2, 3, 10),     # Hearts (3, Tight spacing)
        (2, 3, 45),     # Hearts (3, Wide spacing)
        (6, 1, 0),      # Big Heart (1)

        (3, 1, 0),      # Manager (1)
        (3, 3, 15),     # Manager (3, Tight spacing)
        (3, 2, 55),     # Manager (2, Wide spacing)
        (7, 1, 0),      # Big Manager (1)

        (4, 3, 20),     # Lightbulbs (3, Tight spacing)
        (4, 2, 70),     # Lightbulbs (2, Very wide spacing)
        (8, 1, 0),      # Big Lightbulb (1)

        (9, 1, 0),      # Big Fire Face (1)
    )
    # SPAWN PROBABILITIES
    spawn_weights: chex.Array = jnp.array([
        0.07017544,     # 0: (0, 1, 0)   Fence
        0.0175,         # 1: (1, 2, 20) Robot
        0.0175,         # 2: (5, 1, 0)  Big Robot

        0.08771930,     # 3: (2, 1, 0)   Heart (1)
        0.22807018,     # 4: (2, 2, 55)  Hearts (2)
        0.10526316,     # 5: (2, 3, 10)  Hearts (3,T)
        0.10526316,     # 6: (2, 3, 45)  Hearts (3,W)
        0.03508772,     # 7: (6, 1, 0)   Big Heart (1)

        0.05263158,     # 8: (3, 1, 0)   Manager (1)
        0.08771930,     # 9: (3, 3, 15)  Manager (3,T)
        0.08771930,     # 10: (3, 2, 55)  Manager (2,W)
        0.05263158,     # 11: (7, 1, 0)  Big Manager

        0.05263158,     # 12: (4, 3, 20)  Lightbulbs (3)
        0.05263158,     # 13: (4, 2, 70)  Lightbulbs (2)
        0.05263158,     # 14: (8, 1, 0)   Big Lightbulb

        0.0001,       # 15: (9, 1, 0) Big Fire Face
    ])


class JourneyEscapeState(NamedTuple):
    """Represents the current state of the game"""

    player_y: chex.Array
    player_x: chex.Array
    score: chex.Array
    time: chex.Array
    walking_frames: chex.Array
    walking_direction: chex.Array  # can be {0, 1, 2} for {up/down, right, left}
    game_over: chex.Array

    row_timer: chex.Array  # int32
    obstacles: chex.Array  # (MAX_OBS, 5) -> x, y, w, h, type_idx | [pool]
    obstacle_frames: chex.Array
    invincibility_timer: chex.Array
    spawn_count: chex.Array
    rng_key: chex.Array  # PRNGKey

    hit_cooldown: chex.Array  # int32

    countdown: chex.Array

    bg_frames: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class JourneyEscapeObservation(NamedTuple):
    player: EntityPosition
    obstacles: chex.Array


class JourneyEscapeInfo(NamedTuple):
    time: jnp.ndarray


class JaxJourneyEscape(
    JaxEnvironment[JourneyEscapeState, JourneyEscapeObservation, JourneyEscapeInfo, JourneyEscapeConstants]):
    def __init__(self, consts: JourneyEscapeConstants = None, reward_funcs: list[callable] = None):
        if consts is None:
            consts = JourneyEscapeConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.state = self.reset()
        self.renderer = JourneyEscapeRenderer()

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[JourneyEscapeObservation, JourneyEscapeState]:
        """Initialize a new game state"""
        # Start player at bottom
        player_y = self.consts.start_player_y
        player_x = self.consts.start_player_x

        empty_boxes = jnp.zeros((self.consts.MAX_OBS, 5), dtype=jnp.int32)
        rng_key = jax.random.PRNGKey(0)

        state = JourneyEscapeState(
            player_y=jnp.array(player_y, dtype=jnp.int32),
            player_x=jnp.array(player_x, dtype=jnp.int32),
            score=jnp.array(self.consts.starting_score, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            walking_frames=jnp.array(0, dtype=jnp.int32),
            walking_direction=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            row_timer=jnp.array(0, dtype=jnp.int32),
            obstacles=empty_boxes,
            obstacle_frames=jnp.array(0, dtype=jnp.int32),
            invincibility_timer=jnp.array(0, dtype=jnp.int32),
            spawn_count=jnp.array(0, dtype=jnp.int32),
            rng_key=rng_key,
            hit_cooldown=jnp.array(0, dtype=jnp.int32),
            countdown=jnp.array(self.consts.start_countdown, dtype=jnp.int32),
            bg_frames=jnp.array(0, dtype=jnp.int32),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: JourneyEscapeState, action: int) -> tuple[
        JourneyEscapeObservation, JourneyEscapeState, float, bool, JourneyEscapeInfo]:
        """Take a step in the game given an action"""

        # ---BACKGROUND ANIMATION---
        cycle_length = self.consts.background_frame_switch * self.consts.background_frames_amount
        new_bg_frames = (state.bg_frames + 1) % cycle_length

        # ---RAW PLAYER INPUT---
        # Compute vertical movement
        dy_int = jnp.where(
            (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT),
            -self.consts.player_speed,
            jnp.where(
                (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT),
                self.consts.player_speed,
                0
            ),
        )

        # Compute horizontal movement
        dx_int = jnp.where(
            (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT),
            -self.consts.player_speed,
            jnp.where(
                (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT),
                self.consts.player_speed,
                0
            ),
        )

        # determine walking direction of player
        player_move_right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)
        player_move_left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        vertical = jnp.logical_not(player_move_right) & jnp.logical_not(player_move_left)

        new_walking_direction = jnp.where(
            vertical,
            0,  # up/down
            jnp.where(player_move_right,
                      1,  # right
                      2)  # left
        )

        # advance walking animation every frame, independent of input
        new_walking_frames = (state.walking_frames + 1) % self.consts.player_frame_switch

        # Effective player movement boundaries
        player_min_y = self.consts.min_player_position_y
        player_max_y = self.consts.bottom_border + self.consts.player_height - 3
        player_min_x = self.consts.left_border
        player_max_x = self.consts.right_border - self.consts.player_width - 1

        # "Proposed" movement from input only (used for collision detection)
        pre_y = jnp.clip(state.player_y + dy_int, player_min_y, player_max_y).astype(jnp.int32)
        pre_x = jnp.clip(state.player_x + dx_int, player_min_x, player_max_x).astype(jnp.int32)

        #---OBSTACLES---

        # move & cull

        boxes = state.obstacles
        active = boxes[:, 3] > 0  # Active mask: entries with height > 0 are “alive”

        # obstacle speed +1 when player hits top border
        obstacles_dy_int = jnp.where(state.player_y == self.consts.min_player_position_y, 1, 0)

        # Move down by constant speed only for active entries
        dy_obs = jnp.where(active, self.consts.obstacle_speed_px_per_frame + obstacles_dy_int,
                           0)  # speed for active, 0 for inactive
        boxes = boxes.at[:, 1].set(boxes[:, 1] + dy_obs)

        # Cull: if baseline y >= screen_height, deactivate by zeroing height
        cull_y = self.consts.screen_height + 20
        offscreen = boxes[:, 1] >= cull_y
        new_heights = jnp.where(offscreen, 0, boxes[:, 3])  # int32[N]
        boxes = boxes.at[:, 3].set(new_heights)

        # carry-through for new fields (no behavior change yet)
        new_row_timer = (state.row_timer + 1) % self.consts.row_spawn_period_frames
        new_rng = state.rng_key  # unchanged key

        # Trigger: every row_spawn_period_frames frames
        spawn_now = (new_row_timer == 0)

        def spawn_if_cadence(carry):
            boxes_in, rng_in, sp_count = carry
            rng_in, r1, r2 = jax.random.split(rng_in, 3)

            # Random Selection based on weights.
            logits = jnp.log(self.consts.spawn_weights)
            random_idx = jax.random.categorical(r1, logits)

            # Get Data
            presets = jnp.array(self.consts.obstacle_groups, dtype=jnp.int32)
            group_data = presets[random_idx]

            type_idx = group_data[0]  # Sprite ID
            amount = group_data[1]
            spacing = group_data[2]

            # Lookup dimensions based on type_idx
            # We convert the tuples to JAX arrays for lookup
            width_table = jnp.array(self.consts.TYPE_WIDTHS, dtype=jnp.int32)
            height_table = jnp.array(self.consts.TYPE_HEIGHTS, dtype=jnp.int32)

            this_w = width_table[type_idx]
            this_h = height_table[type_idx]

            # Calculate total width for spacing logic
            total_w = (this_w + spacing) * (amount - 1) + this_w

            # Choose spawn_x so the whole row is inside the screen horizontally
            min_x = self.consts.left_border
            max_x = self.consts.right_border - total_w
            span = (max_x - min_x) + 1
            spawn_x = min_x + jax.random.randint(r2, (), 0, span)

            # Pool capacity check: if not enough free slots, skip this spawn entirely.
            inactive = boxes_in[:, 3] == 0
            available = jnp.sum(inactive)
            enough_space = available >= amount

            def do_spawn(_):
                MAX_GROUP = 3  # no more than 3 obstacles should be in one group

                # First MAX_GROUP free indices (static shape)
                free_idx = jnp.nonzero(inactive, size=MAX_GROUP, fill_value=0)[0]

                # Build row positions
                xs = spawn_x + jnp.arange(MAX_GROUP) * (this_w + spacing)

                ws = jnp.full((MAX_GROUP,), this_w, dtype=jnp.int32)
                hs = jnp.full((MAX_GROUP,), this_h, dtype=jnp.int32)
                ts = jnp.full((MAX_GROUP,), type_idx, dtype=jnp.int32)

                ys = jnp.full((MAX_GROUP,), self.consts.top_border, dtype=jnp.int32) - hs

                # Place exactly `amount` entries; for t >= amount, do nothing
                def body(t, b):
                    def place_one(bb):
                        i = free_idx[t]
                        return bb.at[i].set(jnp.array([xs[t], ys[t], ws[t], hs[t], ts[t]], dtype=jnp.int32))

                    return jax.lax.cond(t < amount, place_one, lambda bb: bb, b)

                boxes_out = jax.lax.fori_loop(0, MAX_GROUP, body, boxes_in)
                return (boxes_out, rng_in, sp_count + 1)

            def skip_spawn(_):
                return (boxes_in, rng_in, sp_count)

            return jax.lax.cond(enough_space, do_spawn, skip_spawn, operand=None)

        def no_spawn(carry):
            return carry

        boxes, new_rng, new_spawn_count = jax.lax.cond(
            spawn_now,
            spawn_if_cadence,
            no_spawn,
            operand=(boxes, state.rng_key, state.spawn_count)
        )

        new_obstacle_frames = (state.obstacle_frames + 1) % self.consts.obstacle_frame_switch

        # --- COLLISIONS---

        # We treat any obstacle with h > 0 as active.
        # boxes has shape (MAX_OBS, 5): [x, y, w, h, type_idx]

        def get_collision_data(box):
            b_x, b_y, b_w, b_h, b_type = box

            is_active = b_h > 0

            # Blink Logic (Ghost State for Lightbulbs)
            # IDs: 4 = Lightbulb, 8 = Big Lightbulb
            is_lightbulb = (b_type == 4) | (b_type == 8)
            # Calculate cycle position
            cycle_len = self.consts.lightbulb_on_duration + self.consts.lightbulb_off_duration
            cycle_pos = state.time % cycle_len

            # It is a ghost (invisible/pass-through) if we are past the ON duration
            is_ghost = is_lightbulb & (cycle_pos >= self.consts.lightbulb_on_duration)

            # AABB Collision
            p_x, p_y = pre_x, pre_y
            p_w, p_h = self.consts.player_width, self.consts.player_height

            overlap_x = (p_x < b_x + b_w) & (p_x + p_w > b_x)
            overlap_y = (p_y < b_y + b_h) & (p_y + p_h > b_y)

            hit = is_active & jnp.logical_not(is_ghost) & overlap_x & overlap_y

            # Relative X Position (Center to Center)
            # Positive = Player is to the Right of Obstacle
            # Negative = Player is to the Left of Obstacle
            p_center_x = p_x + (p_w // 2)
            b_center_x = b_x + (b_w // 2)
            rel_x = (p_center_x - b_center_x).astype(jnp.int32)

            return hit, b_type, rel_x

        # Vectorize over all obstacles
        # collision_mask: bool[MAX_OBS]
        # type_mask: int32[MAX_OBS]
        # relative_x: int32[MAX_OBS]
        collision_mask, type_mask, relative_x = jax.vmap(get_collision_data)(boxes)

        # --- Consumables, Physics & Scoring ---

        # Distinguish Collision Types
        is_solid_type = self.consts.IS_SOLID[type_mask]
        solid_collisions = collision_mask & is_solid_type
        consumable_collisions = collision_mask & jnp.logical_not(is_solid_type)

        def check_anchor(box):
            """
            Checks if the player is *currently* physically inside a solid obstacle.

            Standard `solid_collisions` checks the PROPOSED position (`pre_y`).
            If the player is at the bottom edge of an obstacle and presses DOWN,
            `pre_y` might project a position just *outside* the hitbox.
            Without this check, the game would think the path is clear, release the
            'sticky' drag, and allow the player to strafe away (not possible in ALE).

            This function ensures that if the player's CURRENT coordinates overlap,
            the drag physics remain active.
            """
            b_x, b_y, b_w, b_h, b_type = box
            is_active = b_h > 0
            is_lightbulb = (b_type == 4) | (b_type == 8)

            cycle_len = self.consts.lightbulb_on_duration + self.consts.lightbulb_off_duration
            cycle_pos = state.time % cycle_len
            is_ghost = is_lightbulb & (cycle_pos >= self.consts.lightbulb_on_duration)
            is_solid = self.consts.IS_SOLID[b_type]

            overlap_x = (state.player_x < b_x + b_w) & (state.player_x + self.consts.player_width > b_x)
            overlap_y = (state.player_y < b_y + b_h) & (state.player_y + self.consts.player_height > b_y)
            return is_active & is_solid & jnp.logical_not(is_ghost) & overlap_x & overlap_y

        anchor_collisions = jax.vmap(check_anchor)(boxes)
        is_stuck = jnp.any(solid_collisions | anchor_collisions)

        # Handle Consumables (whole row disapears)

        # Identify the Y-coordinates of consumed items
        hit_y_values = jnp.where(consumable_collisions, boxes[:, 1], -999)

        # Does box[i].y match ANY of the hit_y_values?
        # We compare every box Y against every Hit Y.
        # Matrix: (MAX_OBS, MAX_OBS) -> [i, j] is True if Box i has same Y as Hit Box j
        all_y = boxes[:, 1]
        match_matrix = (all_y[:, None] == hit_y_values[None, :])

        # If a box matches ANY hit Y, it is part of the group.
        # We also ensure we only cull active items that are essentially "linked".
        is_part_of_group = jnp.any(match_matrix, axis=1)

        # Set height to 0 if it is part of a consumed group.
        current_heights = boxes[:, 3]
        new_heights_after_eat = jnp.where(is_part_of_group, 0, current_heights)
        boxes = boxes.at[:, 3].set(new_heights_after_eat)

        # Invincibility Logic (Variable Duration)

        # Identify Specific Power-up Hits
        hit_fireface = jnp.any(consumable_collisions & (type_mask == 9))
        hit_robot = jnp.any(consumable_collisions & ((type_mask == 1) | (type_mask == 5)))

        # Determine Duration to Set
        added_duration = jnp.where(
            hit_fireface,
            self.consts.INV_DURATION_FIREFACE,
            jnp.where(hit_robot, self.consts.INV_DURATION_ROBOT, 0)
        )

        # Update Invincible Timer
        new_inv_timer = jnp.maximum(
            jnp.maximum(state.invincibility_timer - 1, 0),
            added_duration
        )

        is_invincible = new_inv_timer > 0

        # Override Physics (The "Ghost" Effect)
        # If invincible, we are effectively never stuck.
        is_stuck_final = is_stuck & jnp.logical_not(is_invincible)

        # Scoring Logic

        # - Solid/Damage
        cooling_down = state.hit_cooldown > 0
        solid_score_effect = jnp.min(
            jnp.where(solid_collisions | anchor_collisions, self.consts.SCORE_PENALTIES[type_mask], 0)
        ).astype(jnp.int32)

        apply_damage = (solid_score_effect < 0) & jnp.logical_not(cooling_down) & jnp.logical_not(is_invincible)

        # - Consumable/Reward
        consumable_score_effect = jnp.sum(
            jnp.where(consumable_collisions, self.consts.SCORE_PENALTIES[type_mask], 0)
        ).astype(jnp.int32)

        # Update Score
        damage_delta = jnp.where(apply_damage, solid_score_effect, 0)
        total_delta = damage_delta + consumable_score_effect
        new_score = jnp.maximum(state.score + total_delta, 0)

        # Update Cooldown
        new_hit_cooldown = jnp.where(
            apply_damage,
            self.consts.hit_cooldown_frames,
            jnp.maximum(state.hit_cooldown - 1, 0)
        )

        # [Debugging]

        jax.lax.cond(
            apply_damage,
            lambda _: jax.debug.print("Hit! Effect: {}, New Score: {}", total_delta, new_score),
            lambda _: None,
            operand=None
        )

        # --- Movement Physics ("Sticky" Logic) ---

        # Reduce left and right movement speed on collision
        move_tick = (state.time % 4 == 0).astype(jnp.int32)
        reduced_dx = dx_int * move_tick

        drag_speed = self.consts.obstacle_speed_px_per_frame

        # Y-Axis
        new_y_raw = jnp.where(
            is_stuck_final,
            state.player_y + drag_speed,  # Strict Drag
            pre_y  # Normal Movement
        ).astype(jnp.int32)

        # Clip to screen
        new_y = jnp.clip(new_y_raw, player_min_y, player_max_y)

        # X-Axis
        # Determine Blocking
        # We use the union of collision masks to ensure blocking works for both cases
        combined_collisions = solid_collisions | anchor_collisions

        block_right = jnp.any(combined_collisions & (relative_x < 0))
        block_left = jnp.any(combined_collisions & (relative_x > 0))

        # Determine Speed
        effective_dx = jnp.where(is_stuck_final, reduced_dx, dx_int)

        # Apply Blocking
        can_move_x_raw = jnp.logical_not(
            (block_right & (effective_dx > 0)) |
            (block_left & (effective_dx < 0))
        )

        can_move_x = can_move_x_raw | is_invincible

        final_dx = jnp.where(can_move_x, effective_dx, 0)

        # Bottom Push Out Edge Case
        at_bottom_edge = (state.player_y >= player_max_y - 1)

        def check_static_overlap(box):
            """
            Checks overlap using the player's CURRENT position.

            The main collision logic is predictive (uses the proposed position after input).
            This check catches cases where an obstacle has already moved into the player,
            so we don't incorrectly release drag or allow sideways escape.
            """
            b_x, b_y, b_w, b_h, b_type = box

            p_x, p_y = state.player_x, state.player_y
            p_w, p_h = self.consts.player_width, self.consts.player_height

            overlap_x = (p_x < b_x + b_w) & (p_x + p_w > b_x)
            overlap_y = (p_y < b_y + b_h) & (p_y + p_h > b_y)

            is_active = b_h > 0
            is_solid = self.consts.IS_SOLID[b_type]

            return is_active & is_solid & overlap_x & overlap_y

        # Compute static mask
        static_collisions = jax.vmap(check_static_overlap)(boxes)
        is_crushed = jnp.any(static_collisions)

        # Determine Push Direction with Bias (25% Left / 75% Right) (as it is in ALE)

        # Get the width of the obstacle we are currently crushed by
        # boxes[:, 2] is width. We sum the widths of colliding boxes (usually just 1).
        crushed_width = jnp.sum(jnp.where(static_collisions, boxes[:, 2], 0))

        # Calculate the split threshold
        # Normal center split is at 0.
        # Since 0 is 50%, 25% corresponds to -Width/4 relative to center.
        split_threshold = -(crushed_width // 4).astype(jnp.int32)

        # Get relative position
        crushed_rel_x = jnp.sum(jnp.where(static_collisions, relative_x, 0))

        # Determine Direction
        # If we are to the right of the 25% mark -> Push Right (+1)
        # Otherwise -> Push Left (-1)
        push_dir = jnp.where(crushed_rel_x >= split_threshold, 1, -1)

        # Trigger Force Push ONLY if we are physically inside AND at the bottom
        should_eject = is_crushed & at_bottom_edge

        force_push_val = jnp.where(should_eject, push_dir, 0)

        # Apply: If ejecting, override normal movement input
        dx_applied = jnp.where(should_eject, force_push_val, final_dx)

        new_x = jnp.clip(state.player_x + dx_applied, player_min_x, player_max_x).astype(jnp.int32)

        # Update time
        new_time = (state.time + 1).astype(jnp.int32)

        # [Debugging]
        # Print when we hit specific Powerups
        jax.lax.cond(
            hit_fireface,
            lambda _: jax.debug.print(">> HIT FIREFACE! (Infinite Invincibility) Score: {}", consumable_score_effect),
            lambda _: None,
            operand=None
        )

        jax.lax.cond(
            hit_robot,
            lambda _: jax.debug.print(">> HIT ROBOT! (6s Invincibility)"),
            lambda _: None,
            operand=None
        )

        # Print Timer status periodically (e.g., every 60 frames) if active
        jax.lax.cond(
            (new_inv_timer > 0) & (new_time % 60 == 0),
            lambda _: jax.debug.print("... Invincibility Active. Timer: {}", new_inv_timer),
            lambda _: None,
            operand=None
        )

        # Update countdown
        update_countdown = (new_time % self.consts.countdown_frame == 0)
        new_countdown = jnp.where(update_countdown, state.countdown - 1, state.countdown)

        # Check game over
        game_over = jnp.where(
            new_countdown <= 0,
            jnp.array(True),
            state.game_over,
        )

        new_state = JourneyEscapeState(
            player_y=new_y,
            player_x=new_x,
            score=new_score,
            time=new_time,
            walking_frames=new_walking_frames.astype(jnp.int32),
            walking_direction=new_walking_direction.astype(jnp.int32),
            game_over=game_over,
            row_timer=new_row_timer.astype(jnp.int32),
            obstacles=boxes.astype(jnp.int32),  # updated pool
            obstacle_frames=new_obstacle_frames.astype(jnp.int32),
            invincibility_timer=new_inv_timer,
            spawn_count=new_spawn_count,
            rng_key=new_rng,
            hit_cooldown=new_hit_cooldown.astype(jnp.int32),
            countdown=new_countdown.astype(jnp.int32),
            bg_frames=new_bg_frames.astype(jnp.int32)
        )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        def after_timer_end(_):
            obs2, st2 = self.reset()
            return obs2, st2, jnp.array(0, dtype=jnp.int32), jnp.array(False, dtype=jnp.bool_), JourneyEscapeInfo(time=jnp.array(0, dtype=jnp.int32))

        obs, _, env_reward, done, info = jax.lax.cond(state.game_over, after_timer_end, lambda _: (obs, state, env_reward, done, info), state)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: JourneyEscapeState):
        # create player
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(self.consts.player_width, dtype=jnp.int32),
            height=jnp.array(self.consts.player_height, dtype=jnp.int32),
        )

        # create obstacle
        obstacles = jnp.zeros((self.consts.MAX_OBS, 4), dtype=jnp.int32)
        for i in range(self.consts.MAX_OBS):
            ob = state.obstacles.at[i].get()
            obstacles = obstacles.at[i].set(
                jnp.array(
                    [
                        ob.at[0].get(),
                        ob.at[1].get(),
                        self.consts.obstacle_width,
                        self.consts.obstacle_height
                    ],
                    dtype=jnp.int32
                )
            )
        return JourneyEscapeObservation(player=player, obstacles=obstacles)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: JourneyEscapeState) -> JourneyEscapeInfo:
        return JourneyEscapeInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: JourneyEscapeState, state: JourneyEscapeState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: JourneyEscapeState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for JourneyEscape.
        Actions are:
        0: NOOP
        1: UP
        2: DOWN
        """
        return spaces.Discrete(3)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for JourneyEscape.
        The observation contains:
        - player: EntityPosition (x, y, width, height)
        - obstacles: array of shape (10, 4) with x,y,width,height for each obstacle
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            })
            , "obstacles": spaces.Box(low=0, high=210, shape=(self.consts.MAX_OBS, 4), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for JourneyEscape.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    def render(self, state: JourneyEscapeState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def obs_to_flat_array(self, obs: JourneyEscapeObservation) -> jnp.ndarray:
        """Convert observation to a flat array."""
        # Flatten player position and dimensions
        player_flat = jnp.concatenate([
            obs.player.x.reshape(-1),
            obs.player.y.reshape(-1),
            obs.player.width.reshape(-1),
            obs.player.height.reshape(-1)
        ])

        obstacles_flat = obs.obstacles.reshape(-1)

        # Concatenate all components
        return jnp.concatenate([player_flat, obstacles_flat]).astype(jnp.int32)


class JourneyEscapeRenderer(JAXGameRenderer):
    def __init__(self, consts: JourneyEscapeConstants = None):
        super().__init__()
        self.consts = consts or JourneyEscapeConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Load and setup assets
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/journey_escape"

        # --- ASSET GENERATION ---

        # Colors (R, G, B)
        COLOR_BLACK = (0, 0, 0)
        COLOR_BLUE = (24, 26, 167)
        #COLOR_WHITE = (255, 255, 255) # testing

        # Side Bars (8px wide, full height)
        side_bar_sprite = self._create_solid_block(
            width=8,
            height=self.consts.screen_height,
            color=COLOR_BLACK
        )

        # Header
        header_height = self.consts.top_border
        header_sprite = self._create_solid_block(
            width=self.consts.screen_width,
            height=header_height,
            color=COLOR_BLUE
        )

        # Footer
        # Starts at bottom_blue_area -> Ends at screen_height
        footer_height = self.consts.screen_height - self.consts.bottom_blue_area
        footer_sprite = self._create_solid_block(
            width=self.consts.screen_width,
            height=footer_height,
            color=COLOR_BLUE
        )

        # Background (full wide, full height)
        background_sprite = self._create_solid_block(
            width=self.consts.screen_width,
            height=self.consts.screen_height,
            color=COLOR_BLACK
        )

        # Add to manifest
        asset_config.append({'name': 'black_bar', 'type': 'procedural', 'data': side_bar_sprite})
        asset_config.append({'name': 'header', 'type': 'procedural', 'data': header_sprite})
        asset_config.append({'name': 'footer', 'type': 'procedural', 'data': footer_sprite})
        asset_config.append({'name': 'background', 'type': 'procedural', 'data': background_sprite})

        # --- LOAD ASSETS ---
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        self.BACKGROUND = self.SHAPE_MASKS['background']

    def _create_solid_block(self, width: int, height: int, color: Tuple[int, int, int]) -> jnp.ndarray:
        """Creates a solid color sprite with full alpha."""
        # Shape: (H, W, 4)
        block = jnp.zeros((height, width, 4), dtype=jnp.uint8)

        # Set RGB
        block = block.at[:, :, 0].set(color[0])
        block = block.at[:, :, 1].set(color[1])
        block = block.at[:, :, 2].set(color[2])

        # Set Alpha to 255 (Opaque)
        block = block.at[:, :, 3].set(255)
        return block

    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game."""

        return [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {
                'name': 'backgrounds', 'type': 'group',
                'files': [
                    'background_0.npy',
                    'background_1.npy',
                    'background_2.npy',
                    'background_1.npy',
                    'background_2.npy',
                    'background_3.npy',
                    'background_2.npy',
                    'background_1.npy',
                ]
            },
            {
                'name': 'player', 'type': 'group',
                'files': ['player_walk_front_0.npy', 'player_walk_front_1.npy',
                          'player_run_right_0.npy', 'player_run_right_1.npy',
                          'player_run_left_0.npy', 'player_run_left_1.npy']
            },
            {
                'name': 'obstacle_face', 'type': 'group',
                'files': ['3_Manager_0.npy', '3_Manager_1.npy']
            },
            {
                'name': 'obstacle_heart', 'type': 'group',
                'files': ['2_Heart_0.npy', '2_Heart_1.npy']
            },
            {
                'name': 'obstacle_robot', 'type': 'group',
                'files': ['1_Blue_Robot_0.npy', '1_Blue_Robot_1.npy']
            },
            {
                'name': 'obstacle_heart_big', 'type': 'group',
                'files': ['6_Big_Heart_0.npy', '6_Big_Heart_1.npy']
            },
            {
                'name': 'obstacle_face_big', 'type': 'group',
                'files': ['7_Big_Manager_0.npy', '7_Big_Manager_1.npy']
            },
            {
                'name': 'obstacle_robot_big', 'type': 'group',
                'files': ['5_Big_Blue_Robot_0.npy', '5_Big_Blue_Robot_1.npy']
            },
            {
                'name': 'obstacle_fire_face_big', 'type': 'group',
                'files': ['9_Big_Fire_Face_0.npy', '9_Big_Fire_Face_1.npy']
            },
            {'name': 'obstacle_fence', 'type': 'single', 'file': '0_Fence.npy'},
            {'name': 'obstacle_light', 'type': 'single', 'file': '4_Lightbulb.npy'},
            {'name': 'obstacle_light_big', 'type': 'single', 'file': '8_Big_Lightbulb.npy'},
            {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
            {'name': 'dollar', 'type': 'single', 'file': 'dollar.npy'},
            {'name': 'timer_digits', 'type': 'digits', 'pattern': 'timer_{}.npy'},
            {'name': 'timer_colon', 'type': 'single', 'file': 'timer_colon.npy'},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render Background
        frame_idx = state.bg_frames // self.consts.background_frame_switch
        bg_mask = self.SHAPE_MASKS["backgrounds"][frame_idx]
        raster = self.jr.render_at(raster, 0, 20, bg_mask)

        # Render obstacles
        # state.obstacles has shape (MAX_OBS, 5): [x, y, w, h, type_idx]
        # We consider entries with h > 0 as "active".
        obs_boxes = state.obstacles  # (MAX_OBS, 5)

        """
        0: Fence
        1: Blue Robot
        2: Heart
        3: Manager
        4: Lightbulb
        5: Big Blue Robot
        6: Big Heart
        7: Big Manager
        8: Big Lightbulb
        9: Big Fire Face
        """

        # Fence (ID 0) - Returns 32x15
        FENCE_MASK = self.SHAPE_MASKS["obstacle_fence"] # 0

        # Table for Small Items (IDs 1-4) - Returns 8x15
        # Note: These lambda functions expect indices 0, 1, 2, 3, so we will subtract 1 from the ID
        SMALL_TABLE = [
            lambda frame: self.SHAPE_MASKS["obstacle_robot"][frame],  # 1 -> 0
            lambda frame: self.SHAPE_MASKS["obstacle_heart"][frame],  # 2 -> 1
            lambda frame: self.SHAPE_MASKS["obstacle_face"][frame],  # 3 -> 2
            lambda frame: self.SHAPE_MASKS["obstacle_light"],  # 4 -> 3
        ]

        # Table for Big Items (IDs 5-8) - Returns 16x15
        # Note: These lambda functions expect indices 0, 1, 2, so we will subtract 5 from the ID
        BIG_TABLE = [
            lambda frame: self.SHAPE_MASKS["obstacle_robot_big"][frame],  # 5 -> 0
            lambda frame: self.SHAPE_MASKS["obstacle_heart_big"][frame],  # 6 -> 1
            lambda frame: self.SHAPE_MASKS["obstacle_face_big"][frame],  # 7 -> 2
            lambda frame: self.SHAPE_MASKS["obstacle_light_big"],  # 8 -> 3
        ]

        FIRE_FACE_MASK = lambda frame: self.SHAPE_MASKS["obstacle_fire_face_big"][frame]  # 9

        def draw_fence(r, x, y):
            mask = FENCE_MASK
            return self.jr.render_at_clipped(r, x, y, mask)

        def draw_small(r, x, y, type_idx, frame_idx):
            mask = jax.lax.switch(type_idx - 1, SMALL_TABLE, frame_idx)
            return self.jr.render_at_clipped(r, x, y, mask)

        def draw_big(r, x, y, type_idx, frame_idx):
            # Map global ID (5,6,7,8) to local table index (0,1,2,3)
            mask = jax.lax.switch(type_idx - 5, BIG_TABLE, frame_idx)
            return self.jr.render_at_clipped(r, x, y, mask)

        def draw_fire_face(r, x, y, frame_idx):
            mask = FIRE_FACE_MASK(frame_idx)
            return self.jr.render_at_clipped(r, x, y, mask)

        def body(i, r):
            box = obs_boxes[i]
            box_h = box[3]
            obs_type = box[4]

            does_exist = box_h > 0

            is_lightbulb = (obs_type == 4) | (obs_type == 8)

            cycle_len = self.consts.lightbulb_on_duration + self.consts.lightbulb_off_duration
            cycle_pos = state.time % cycle_len

            is_ghost = is_lightbulb & (cycle_pos >= self.consts.lightbulb_on_duration)

            # Final Decision: Draw if existing AND not a ghost
            should_draw = does_exist & jnp.logical_not(is_ghost)

            x, y = box[0], box[1]
            obs_frame_idx = jnp.where(state.obstacle_frames >= (self.consts.obstacle_frame_switch // 2), 0, 1)

            def render_op(curr_raster):
                return jax.lax.cond(
                    obs_type >= 5,
                    lambda _r: jax.lax.cond(
                                    obs_type == 9,
                                    lambda _r: draw_fire_face(_r, x, y, obs_frame_idx),
                                    lambda _r: draw_big(_r, x, y, obs_type, obs_frame_idx),
                                    _r
                                ),
                    lambda _r: jax.lax.cond(
                                    obs_type == 0,
                                    lambda _r: draw_fence(_r, x, y),
                                    lambda _r: draw_small(_r, x, y, obs_type, obs_frame_idx),
                                    _r
                                ),
                    curr_raster
                )

            return jax.lax.cond(should_draw, render_op, lambda _r: _r, r)

        # Iterate all possible slots
        raster = jax.lax.fori_loop(0, obs_boxes.shape[0], body, raster)

        # Select player sprite based on walking frames and direction
        use_idle = state.walking_frames < (self.consts.player_frame_switch // 2)
        sprite_index = state.walking_direction * 2
        player_frame_index = jax.lax.select(use_idle, sprite_index, sprite_index + 1)

        player_mask = self.SHAPE_MASKS["player"][player_frame_index]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        # Render Header (Top Blue)
        header_mask = self.SHAPE_MASKS["header"]
        raster = self.jr.render_at(raster, 0, 0, header_mask)

        # Render Footer (Bottom Blue)
        footer_y = self.consts.bottom_blue_area
        footer_mask = self.SHAPE_MASKS["footer"]
        raster = self.jr.render_at(raster, 0, footer_y, footer_mask)

        # Render Score (On top of Blue Header)
        score_digits = self.jr.int_to_digits(state.score, max_digits=5)
        score_digit_masks = self.SHAPE_MASKS["score_digits"]
        num_to_render = (
            1 
            + (state.score >= 10).astype(jnp.int32)
            + (state.score >= 100).astype(jnp.int32)
            + (state.score >= 1000).astype(jnp.int32)
            + (state.score >= 10000).astype(jnp.int32)
        )
        start_index = 5 - num_to_render
        render_x_pos = ((self.consts.screen_width // 2) + 20) - (num_to_render * 8)
        raster = self.jr.render_at(raster, render_x_pos - 9, 5, self.SHAPE_MASKS["dollar"])
        raster = self.jr.render_label_selective(raster, render_x_pos, 5,
                                                score_digits,
                                                score_digit_masks, start_index,num_to_render, spacing=8, 
                                                max_digits_to_render=5)

        # Render Countdown (On top of Blue Header)
        countdown_digits = self.jr.int_to_digits(state.countdown, max_digits=2)
        countdown_digit_masks = self.SHAPE_MASKS["timer_digits"]
        num_to_render = 2
        render_x_pos = (self.consts.screen_width // 2) - 2
        raster = self.jr.render_at(raster, render_x_pos - 9, 20, self.SHAPE_MASKS["timer_digits"][0]) # Renders the fixed leading 0
        raster = self.jr.render_at(raster, render_x_pos - 3, 20, self.SHAPE_MASKS["timer_colon"]) 
        raster = self.jr.render_label_selective(raster, render_x_pos, 20,
                                                countdown_digits, # the remaining seconds
                                                countdown_digit_masks, 0, num_to_render, spacing=7)

        # Render Side Bars (Black)
        black_bar_mask = self.SHAPE_MASKS["black_bar"]
        # left bar
        raster = self.jr.render_at(raster, 0, 0, black_bar_mask)
        # right bar
        raster = self.jr.render_at(raster, self.consts.screen_width - 8, 0, black_bar_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
