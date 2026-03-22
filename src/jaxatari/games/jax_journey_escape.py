import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, Dict, Optional, Any
from flax import struct

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

def get_default_asset_config() -> tuple:
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
            'files': ['player_walk_front_1.npy', 'player_walk_front_0.npy',
                        'player_run_right_0.npy', 'player_run_right_1.npy',
                        'player_run_left_0.npy', 'player_run_left_1.npy']
        },
        {
            'name': 'promoter', 'type': 'group',
            'files': ['3_Promoter_0.npy', '3_Promoter_1.npy']
        },
        {
            'name': 'groupies', 'type': 'group',
            'files': ['2_Groupies_0.npy', '2_Groupies_1.npy']
        },
        {
            'name': 'roadie', 'type': 'group',
            'files': ['1_Loyal_Roadie_0.npy', '1_Loyal_Roadie_1.npy']
        },
        {
            'name': 'big_groupies', 'type': 'group',
            'files': ['6_Big_Groupies_0.npy', '6_Big_Groupies_1.npy']
        },
        {
            'name': 'big_promoter', 'type': 'group',
            'files': ['7_Big_Promoter_0.npy', '7_Big_Promoter_1.npy']
        },
        {
            'name': 'big_roadie', 'type': 'group',
            'files': ['5_Big_Loyal_Roadie_0.npy', '5_Big_Loyal_Roadie_1.npy']
        },
        {
            'name': 'big_manager', 'type': 'group',
            'files': ['9_Big_Mighty_Manager_0.npy', '9_Big_Mighty_Manager_1.npy']
        },
        {'name': 'barrier', 'type': 'single', 'file': '0_Stage_Barrier.npy'},
        {'name': 'photographer', 'type': 'single', 'file': '4_Photographer.npy'},
        {'name': 'big_photographer', 'type': 'single', 'file': '8_Big_Photographer.npy'},
        {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        {'name': 'dollar', 'type': 'single', 'file': 'dollar.npy'},
        {'name': 'timer_digits', 'type': 'digits', 'pattern': 'timer_{}.npy'},
        {'name': 'timer_colon', 'type': 'single', 'file': 'timer_colon.npy'},
    ]

class JourneyEscapeConstants(AutoDerivedConstants):

    starting_score: int = struct.field(pytree_node=False, default=50000)

    # frame countdown for timer
    countdown_frame: int = struct.field(pytree_node=False, default=50)  # countdown decreases by one second every 50 frames
    start_countdown: int = struct.field(pytree_node=False, default=59)  # for a 59-second countdown

    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=230)

    background_frame_switch: int = struct.field(pytree_node=False, default=4) # set to 0 for static background
    background_frames_amount: int = struct.field(pytree_node=False, default=8) # defined order of 2*4 sprites

    player_width: int = struct.field(pytree_node=False, default=8)
    player_height: int = struct.field(pytree_node=False, default=25)
    start_player_x: int = struct.field(pytree_node=False, default=93)  # Fixed x position
    start_player_y: int = struct.field(pytree_node=False, default=172)  # Fixed y position
    player_speed: int = struct.field(pytree_node=False, default=1)
    player_frame_switch: int = struct.field(pytree_node=False, default=16) # should match ALE

    # top and bottom blue areas
    top_blue_area_height: int = struct.field(pytree_node=False, default=27)
    bottom_blue_area_height: int = struct.field(pytree_node=False, default=28)
    # paddings
    top_padding: int = struct.field(pytree_node=False, default=3)
    bottom_padding: int = struct.field(pytree_node=False, default=4)
    # border of the valid game space
    top_border: int = struct.field(pytree_node=False, default=3 + 27) # top_padding + top_blue_area_height
    bottom_border: int = struct.field(pytree_node=False, default=230 - 28 - 4) # screen_height - bottom_blue_area_height - bottom_padding
    left_border: int = struct.field(pytree_node=False, default=8)
    right_border: int = struct.field(pytree_node=False, default=160 - 8) # screen_width - left_border

    # player position rules
    min_player_position_y: int = struct.field(pytree_node=False, default=30 + (230 // 4)) # top_border + (screen_height // 4)

    # Standard sizes
    obstacle_width: int = struct.field(pytree_node=False, default=8)
    obstacle_height: int = struct.field(pytree_node=False, default=10)

    # Big sizes (2x)
    big_obstacle_width: int = struct.field(pytree_node=False, default=16)
    big_obstacle_height: int = struct.field(pytree_node=False, default=20)

    obstacle_frame_switch: int = struct.field(pytree_node=False, default=17)  # should match ALE
    obstacle_speed_px_per_frame: int = struct.field(pytree_node=False, default=1)
    row_spawn_period_frames: int = struct.field(pytree_node=False, default=50)  # spawn every N frames # ToDo: calibrate
    hit_cooldown_frames: int = struct.field(pytree_node=False, default=17)

    # Define the Width and Height for every ID (0 to 9)
        #   0: Stage Barriers
        #   1: Loyal Roadie
        #   2: Love-Crazed Groupies
        #   3: Shifty-Eyed Promoter
        #   4: Sneaky Photographer
        #   5: Big Loyal Roadie
        #   6: Big Love-Crazed Groupies
        #   7: Big Shifty-Eyed Promoter
        #   8: Big Sneaky Photographer
        #   9: Big Mighty Manager
    TYPE_WIDTHS: Tuple[int, ...] = struct.field(pytree_node=False, default_factory=lambda: (32, 8, 8, 8, 8, 16, 16, 16, 16, 17))
    TYPE_HEIGHTS: Tuple[int, ...] = struct.field(pytree_node=False, default_factory=lambda: (15, 15, 15, 15, 15, 15, 15, 15, 15, 15))

    MAX_OBS: int = struct.field(pytree_node=False, default=64)

    # Blinking Effect
    photographer_on_duration: int = struct.field(pytree_node=False, default=17)
    photographer_off_duration: int = struct.field(pytree_node=False, default=49)

    # Invincible Effect
    INV_DURATION_ROADIE: int = struct.field(pytree_node=False, default=6 * 50) # 6 seconds @ 50fps
    INV_DURATION_MANAGER: int = struct.field(pytree_node=False, default=100000) # (longer than the max possible game time of ~60s)

    # True if the object stops movement / drags player
    IS_SOLID: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        True,           # 0 barrier
        False,          # 1 roadie
        True,           # 2 groupies
        True,           # 3 promoter
        True,           # 4 photographer
        False,          # 5 big roadie
        True,           # 6 big groupies
        True,           # 7 big promoter
        True,           # 8 big photographer
        False,          # 9 big Manager
    ]))

    # Points deducted on contact
    SCORE_PENALTIES: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        0,              # 0 barriers
        0,              # 1 roadie
        -300,           # 2 groupies
        -2000,          # 3 promoter
        -600,           # 4 photographer
        0,              # 5 big roadie
        -300,           # 6 big groupies
        -2000,          # 7 big promoter
        -600,           # 8 big photographer
        9900,           # 9 big manager
    ]))

    # predefined groups: [type, amount, spacing in px]
    obstacle_groups: Tuple[Tuple[int, int, int], ...] = struct.field(pytree_node=False, default_factory=lambda: (
        (0, 1, 0),      # barriers
        (1, 2, 20),     # roadies
        (5, 1, 0),      # big roadie (1)

        (2, 1, 0),      # groupies (1)
        (2, 2, 55),     # groupies (2, Wide spacing)
        (2, 3, 10),     # groupies (3, Tight spacing)
        (2, 3, 45),     # groupies (3, Wide spacing)
        (6, 1, 0),      # big groupies (1)

        (3, 1, 0),      # promoter (1)
        (3, 3, 15),     # promoter (3, Tight spacing)
        (3, 2, 55),     # promoter (2, Wide spacing)
        (7, 1, 0),      # big promoter (1)

        (4, 3, 20),     # photographers (3, Tight spacing)
        (4, 2, 70),     # photographers (2, Very wide spacing)
        (8, 1, 0),      # big photographer (1)

        (9, 1, 0),      # big manager (1)
    ))
    # SPAWN PROBABILITIES
    spawn_weights: chex.Array = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        0.07017544,     # 0: (0, 1, 0)  barriers
        0.0175,         # 1: (1, 2, 20) roadie
        0.0175,         # 2: (5, 1, 0)  big roadie

        0.08771930,     # 3: (2, 1, 0)   groupies (1)
        0.22807018,     # 4: (2, 2, 55)  groupies (2)
        0.10526316,     # 5: (2, 3, 10)  groupies (3,T)
        0.10526316,     # 6: (2, 3, 45)  groupies (3,W)
        0.03508772,     # 7: (6, 1, 0)   big groupies (1)

        0.05263158,     # 8: (3, 1, 0)   promoter (1)
        0.08771930,     # 9: (3, 3, 15)  promoter (3,T)
        0.08771930,     # 10: (3, 2, 55) promoter (2,W)
        0.05263158,     # 11: (7, 1, 0)  big promoter

        0.05263158,     # 12: (4, 3, 20)  photographers (3)
        0.05263158,     # 13: (4, 2, 70)  photographers (2)
        0.05263158,     # 14: (8, 1, 0)   big photographer

        0.0085,         # 15: (9, 1, 0) big manager
    ]))
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=get_default_asset_config)

    # Diagonal movement: per-group probability of spawning with horizontal velocity.
    # One entry per group in obstacle_groups. 0.0 = always vertical, 1.0 = always diagonal.
    diagonal_probabilities: Tuple[float, ...] = struct.field(pytree_node=False, default_factory=lambda: (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ))
    # Horizontal move period for diagonal obstacles (move 1px every N frames)
    obstacle_horizontal_move_period: int = struct.field(pytree_node=False, default=2)
    # When True, horizontal move period alternates between 2 and 1 on each wall bounce
    diagonal_speed_alternates: bool = struct.field(pytree_node=False, default=False)
    # Per-frame probability of spontaneous direction flip (0.0 = never, checked every frame)
    diagonal_random_switch_prob: float = struct.field(pytree_node=False, default=0.0)
    # Cooldown frames after a random/wall-bounce switch before another can happen
    diagonal_random_switch_cooldown: int = struct.field(pytree_node=False, default=20)
    # Random range added to cooldown (actual cooldown = base + random(0..range))
    diagonal_random_switch_cooldown_range: int = struct.field(pytree_node=False, default=40)
    # When True, random direction switches are per-obstacle instead of per-group (obstacles can split)
    diagonal_switch_per_obstacle: bool = struct.field(pytree_node=False, default=False)

@struct.dataclass
class JourneyEscapeState:
    """Represents the current state of the game"""

    player_y: chex.Array
    player_x: chex.Array
    score: chex.Array
    time: chex.Array
    walking_frames: chex.Array
    walking_direction: chex.Array  # can be {0, 1, 2} for {up/down, right, left}
    game_over: chex.Array

    row_timer: chex.Array  # int32
    obstacles: chex.Array  # (MAX_OBS, 8) -> x, y, w, h, type_idx, dx, move_period, switch_cd | [pool]
    obstacle_frames: chex.Array
    invincibility_timer: chex.Array
    spawn_count: chex.Array
    rng_key: chex.Array  # PRNGKey

    hit_cooldown: chex.Array  # int32

    countdown: chex.Array

    bg_frames: chex.Array

@struct.dataclass
class EntityPosition:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

@struct.dataclass
class JourneyEscapeObservation:
    player: EntityPosition
    obstacles: chex.Array

@struct.dataclass
class JourneyEscapeInfo:
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
        self.renderer = JourneyEscapeRenderer(self.consts)

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[JourneyEscapeObservation, JourneyEscapeState]:
        """Initialize a new game state"""
        # Start player at bottom
        player_y = self.consts.start_player_y
        player_x = self.consts.start_player_x

        empty_boxes = jnp.zeros((self.consts.MAX_OBS, 8), dtype=jnp.int32)
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
        up_normal = (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT)
        up_fire_moves = (action == Action.UPLEFTFIRE) | (action == Action.UPRIGHTFIRE)
        going_up = up_normal | up_fire_moves

        down_normal = (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT)
        down_fire = (action == Action.DOWNFIRE) | (action == Action.DOWNLEFTFIRE) | (action == Action.DOWNRIGHTFIRE)

        left_normal = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        left_fire = (action == Action.LEFTFIRE) | (action == Action.UPLEFTFIRE) | (action == Action.DOWNLEFTFIRE)

        right_normal = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)
        right_fire = (action == Action.RIGHTFIRE) | (action == Action.UPRIGHTFIRE) | (action == Action.DOWNRIGHTFIRE)

        # Compute vertical movement
        dy_int = jnp.where(
            going_up,
            -self.consts.player_speed,
            jnp.where(
                down_normal,
                self.consts.player_speed,
                jnp.where(
                    down_fire,
                    2 * self.consts.player_speed,
                    0
                )
            ),
        )

        # Compute horizontal movement
        dx_int = jnp.where(
            left_normal,
            -self.consts.player_speed,
            jnp.where(
                left_fire,
                -2 * self.consts.player_speed,
                jnp.where(
                    right_normal,
                    self.consts.player_speed,
                    jnp.where(
                        right_fire,
                        2 * self.consts.player_speed,
                        0
                    )
                )
            ),
        )



        # advance walking animation every frame, independent of input
        new_walking_frames = (state.walking_frames + 1) % self.consts.player_frame_switch

        # Effective player movement boundaries
        player_min_y = self.consts.min_player_position_y
        player_max_y = self.consts.bottom_border - self.consts.player_height
        player_min_x = self.consts.left_border
        player_max_x = self.consts.right_border - self.consts.player_width - 1

        # "Proposed" movement from input only (used for collision detection)
        pre_y = jnp.clip(state.player_y + dy_int, player_min_y, player_max_y).astype(jnp.int32)
        pre_x = jnp.clip(state.player_x + dx_int, player_min_x, player_max_x).astype(jnp.int32)

        #---OBSTACLES---

        # move & cull

        boxes = state.obstacles
        active = boxes[:, 3] > 0  # Active mask: entries with height > 0 are "alive"

        # obstacle speed +1 when player hits top border
        obstacles_dy_int = jnp.where(state.player_y == self.consts.min_player_position_y, 1, 0)

        # Move down by constant speed only for active entries
        dy_obs = jnp.where(active, self.consts.obstacle_speed_px_per_frame + obstacles_dy_int,
                           0)  # speed for active, 0 for inactive
        boxes = boxes.at[:, 1].set(boxes[:, 1] + dy_obs)

        # --- Diagonal Movement (horizontal velocity from dx column) ---
        obs_dx = boxes[:, 5]  # per-obstacle horizontal velocity (-1, 0, or +1)
        obs_move_period = boxes[:, 6]  # per-obstacle move period
        # Use per-obstacle period: move 1px when frame aligns with this obstacle's period
        # For inactive obstacles or period=0 (straight movers), never move horizontally
        safe_period = jnp.where(obs_move_period > 0, obs_move_period, 1)
        should_move_h = active & (obs_move_period > 0) & ((state.time % safe_period) == 0)
        effective_dx_obs = jnp.where(should_move_h, obs_dx, 0)
        new_obs_x = boxes[:, 0] + effective_dx_obs

        # Wall-bounce: flip dx for entire groups when any member hits a wall.
        # Groups share the same y-coordinate, so we use y-matching.
        obs_y_vals = boxes[:, 1]
        obs_w_vals = boxes[:, 2]

        # Check each obstacle for wall collision
        hits_left_wall = active & (new_obs_x <= self.consts.left_border)
        hits_right_wall = active & ((new_obs_x + obs_w_vals) >= self.consts.right_border)
        hits_any_wall = hits_left_wall | hits_right_wall

        # For each obstacle, check if ANY group member (same y) hit a wall.
        # Matrix: (MAX_OBS, MAX_OBS) -> [i, j] is True if obstacle i has same y as hitter j
        wall_match_matrix = (obs_y_vals[:, None] == obs_y_vals[None, :]) & hits_any_wall[None, :]
        should_flip = jnp.any(wall_match_matrix, axis=1) & active

        # Flip dx for obstacles that need bouncing
        flipped_dx = jnp.where(should_flip, -obs_dx, obs_dx)
        # Re-apply movement with flipped dx (bounce means reverse this frame)
        bounced_effective_dx = jnp.where(should_flip, -effective_dx_obs, effective_dx_obs)
        final_obs_x = boxes[:, 0] + bounced_effective_dx
        # Clamp x to screen borders
        final_obs_x = jnp.clip(final_obs_x, self.consts.left_border, self.consts.right_border - obs_w_vals)

        boxes = boxes.at[:, 0].set(jnp.where(active, final_obs_x, boxes[:, 0]))
        boxes = boxes.at[:, 5].set(flipped_dx)

        # Alternate speed on bounce: toggle move_period between 2 (slow) and 1 (fast)
        new_move_period = jnp.where(
            should_flip & self.consts.diagonal_speed_alternates,
            jnp.where(obs_move_period == 2, 1, 2),
            obs_move_period
        )
        boxes = boxes.at[:, 6].set(new_move_period)

        # --- Group representative index (used for wall-bounce cd + random switch) ---
        # All per-group decisions use the representative (lowest active index with same y)
        idx_arr = jnp.arange(self.consts.MAX_OBS)
        same_y_matrix = (obs_y_vals[:, None] == obs_y_vals[None, :]) & active[None, :]
        group_rep_idx = jnp.where(same_y_matrix, idx_arr[None, :], self.consts.MAX_OBS).min(axis=1)
        group_rep_idx = jnp.clip(group_rep_idx, 0, self.consts.MAX_OBS - 1)

        # Also apply switch cooldown after wall bounce to prevent immediate random re-flip
        rng_wall_cd, rng_after_wall_cd = jax.random.split(state.rng_key)
        wall_cd_rand_per_obs = self.consts.diagonal_random_switch_cooldown + jax.random.randint(
            rng_wall_cd, (self.consts.MAX_OBS,), 0, jnp.maximum(self.consts.diagonal_random_switch_cooldown_range, 1)
        )
        # chaos mode: each obstacle gets its own cooldown; group mode: use representative's
        wall_cd_rand = jnp.where(
            self.consts.diagonal_switch_per_obstacle,
            wall_cd_rand_per_obs,
            wall_cd_rand_per_obs[group_rep_idx]
        )
        wall_bounce_cd = jnp.where(
            should_flip & (self.consts.diagonal_random_switch_prob > 0.0),
            wall_cd_rand,
            boxes[:, 7]
        )
        boxes = boxes.at[:, 7].set(wall_bounce_cd)

        # --- Random direction switch with cooldown ---
        # Decrement cooldown for all active obstacles
        switch_cd = boxes[:, 7]
        switch_cd = jnp.where(active, jnp.maximum(switch_cd - 1, 0), switch_cd)

        rng_for_switch, rng_for_cd, new_rng_after_switch = jax.random.split(rng_after_wall_cd, 3)
        switch_rolls = jax.random.uniform(rng_for_switch, (self.consts.MAX_OBS,))
        is_diagonal_obs = boxes[:, 5] != 0  # only for obstacles actually moving diagonally

        # Per-obstacle mode: each obstacle rolls independently
        # Group mode: use the representative's roll AND cd_ready so group stays in lockstep
        effective_roll = jnp.where(
            self.consts.diagonal_switch_per_obstacle,
            switch_rolls,
            switch_rolls[group_rep_idx]
        )
        effective_cd_ready = jnp.where(
            self.consts.diagonal_switch_per_obstacle,
            switch_cd == 0,
            switch_cd[group_rep_idx] == 0
        )

        obs_should_switch = active & is_diagonal_obs & effective_cd_ready & (effective_roll < self.consts.diagonal_random_switch_prob)

        current_dx = boxes[:, 5]
        boxes = boxes.at[:, 5].set(jnp.where(obs_should_switch, -current_dx, current_dx))
        # Randomized cooldown: base + random(0..range)
        random_cd_per_obs = self.consts.diagonal_random_switch_cooldown + jax.random.randint(
            rng_for_cd, (self.consts.MAX_OBS,), 0, jnp.maximum(self.consts.diagonal_random_switch_cooldown_range, 1)
        )
        effective_cd = jnp.where(
            self.consts.diagonal_switch_per_obstacle,
            random_cd_per_obs,
            random_cd_per_obs[group_rep_idx]
        )
        switch_cd = jnp.where(obs_should_switch, effective_cd, switch_cd)
        boxes = boxes.at[:, 7].set(switch_cd)

        # Cull: deactivate obstacles 30px before the bottom of the screen
        cull_y = self.consts.screen_height - 30
        offscreen = boxes[:, 1] >= cull_y
        new_heights = jnp.where(offscreen, 0, boxes[:, 3])  # int32[N]
        boxes = boxes.at[:, 3].set(new_heights)

        # carry-through for new fields (no behavior change yet)
        new_row_timer = (state.row_timer + 1) % self.consts.row_spawn_period_frames
        new_rng = new_rng_after_switch  # key consumed by random direction switch

        # Trigger: every row_spawn_period_frames frames
        spawn_now = (new_row_timer == 0)

        def spawn_if_cadence(carry):
            boxes_in, rng_in, sp_count = carry
            rng_in, r1, r2, r3, r4 = jax.random.split(rng_in, 5)

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

            # --- Diagonal Movement: determine dx for this group ---
            diag_probs = jnp.array(self.consts.diagonal_probabilities)
            diag_prob = diag_probs[random_idx]  # probability for this group
            is_diagonal = jax.random.uniform(r3, ()) < diag_prob
            # Pick direction: +1 (right) or -1 (left)
            diag_direction = jax.random.choice(r4, jnp.array([-1, 1]))
            spawn_dx = jnp.where(is_diagonal, diag_direction, 0).astype(jnp.int32)

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
                dxs = jnp.full((MAX_GROUP,), spawn_dx, dtype=jnp.int32)
                mps = jnp.full((MAX_GROUP,), self.consts.obstacle_horizontal_move_period, dtype=jnp.int32)

                ys = jnp.full((MAX_GROUP,), self.consts.top_border, dtype=jnp.int32) - hs

                # Place exactly `amount` entries; for t >= amount, do nothing
                def body(t, b):
                    def place_one(bb):
                        i = free_idx[t]
                        return bb.at[i].set(jnp.array([xs[t], ys[t], ws[t], hs[t], ts[t], dxs[t], mps[t], 0], dtype=jnp.int32))

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
        # boxes has shape (MAX_OBS, 8): [x, y, w, h, type_idx, dx, move_period, switch_cd]

        def get_collision_data(box):
            b_x, b_y, b_w, b_h, b_type, _b_dx, _b_mp, _b_cd = box

            is_active = b_h > 0

            # Blink Logic (Ghost State for Sneaky Photographers)
            # IDs: 4 = Sneaky Photographer, 8 = Big Sneaky Photographer
            is_photographer = (b_type == 4) | (b_type == 8)
            # Calculate cycle position
            cycle_len = self.consts.photographer_on_duration + self.consts.photographer_off_duration
            cycle_pos = state.time % cycle_len

            # It is a ghost (invisible/pass-through) if we are past the ON duration
            is_ghost = is_photographer & (cycle_pos >= self.consts.photographer_on_duration)

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
            b_x, b_y, b_w, b_h, b_type, _b_dx, _b_mp, _b_cd = box
            is_active = b_h > 0
            is_photographer = (b_type == 4) | (b_type == 8)

            cycle_len = self.consts.photographer_on_duration + self.consts.photographer_off_duration
            cycle_pos = state.time % cycle_len
            is_ghost = is_photographer & (cycle_pos >= self.consts.photographer_on_duration)
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
        hit_manager = jnp.any(consumable_collisions & (type_mask == 9))
        hit_roadie = jnp.any(consumable_collisions & ((type_mask == 1) | (type_mask == 5)))

        # Determine Duration to Set
        added_duration = jnp.where(
            hit_manager,
            self.consts.INV_DURATION_MANAGER,
            jnp.where(hit_roadie, self.consts.INV_DURATION_ROADIE, 0)
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
            b_x, b_y, b_w, b_h, b_type, _b_dx, _b_mp, _b_cd = box

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

        # Actual horizontal movement applied (after clipping / obstacle blocking)
        actual_dx = new_x - state.player_x

        # determine walking direction of player based on actual movement capability
        is_moving_up = up_normal | (action == Action.UPFIRE) | up_fire_moves
        is_actually_moving_right = actual_dx > 0
        is_actually_moving_left = actual_dx < 0

        # Up overrides side sprites. If only Side or Down+Side, it uses Side.
        new_walking_direction = jnp.where(
            is_moving_up,
            0,  # up sprite dominant (front)
            jnp.where(is_actually_moving_right,
                      1,  # right
                      jnp.where(is_actually_moving_left,
                                2,  # left
                                0)  # fallback to walk-front (0) when blocked or solo down
                      )
        )

        # Update time
        new_time = (state.time + 1).astype(jnp.int32)

        # [Debugging]
        # Print when we hit specific Powerups
        jax.lax.cond(
            hit_manager,
            lambda _: jax.debug.print(">> HIT MIGHTY MANAGER! (Infinite Invincibility) Score: {}", consumable_score_effect),
            lambda _: None,
            operand=None
        )

        jax.lax.cond(
            hit_roadie,
            lambda _: jax.debug.print(">> HIT LOYAL ROADIE! (6s Invincibility)"),
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
        The action space includes all 18 standard Atari actions.
        """
        return spaces.Discrete(18)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for JourneyEscape.
        The observation contains:
        - player: EntityPosition (x, y, width, height)
        - obstacles: array of shape (10, 4) with x,y,width,height for each obstacle
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=self.consts.screen_width, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=self.consts.screen_height, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=self.consts.screen_width, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=self.consts.screen_height, shape=(), dtype=jnp.int32),
            })
            , "obstacles": spaces.Box(low=0, high=self.consts.screen_height, shape=(self.consts.MAX_OBS, 4), dtype=jnp.int32),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for JourneyEscape.
        The image is a RGB image with shape (230, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.screen_height, self.consts.screen_width, 3),
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
    def __init__(self, consts: JourneyEscapeConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or JourneyEscapeConstants()
        if config is None:
            config = render_utils.RendererConfig(
                game_dimensions=(self.consts.screen_height, self.consts.screen_width),
                channels=3,
            )
        else:
            # Ensure game_dimensions is always set from consts
            config = config.replace(
                game_dimensions=(self.consts.screen_height, self.consts.screen_width),
            )
        self.config = config
        super().__init__(consts=self.consts, config=self.config)
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Load and setup assets
        asset_config = list(self.consts.ASSET_CONFIG) # self._get_asset_config()
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
        header_sprite = self._create_solid_block(
            width=self.consts.screen_width,
            height=self.consts.top_blue_area_height,
            color=COLOR_BLUE
        )

        # Footer
        # Starts at bottom_blue_area -> Ends at screen_height
        footer_sprite = self._create_solid_block(
            width=self.consts.screen_width,
            height=self.consts.bottom_blue_area_height,
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

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render Background
        frame_idx = state.bg_frames // self.consts.background_frame_switch
        bg_mask = self.SHAPE_MASKS["backgrounds"][frame_idx]
        raster = self.jr.render_at(raster, 0, 10, bg_mask)

        # Render obstacles
        # state.obstacles has shape (MAX_OBS, 5): [x, y, w, h, type_idx]
        # We consider entries with h > 0 as "active".
        obs_boxes = state.obstacles  # (MAX_OBS, 5)

        # barrier (ID 0) - Returns 32x15
        BARRIER_MASK = self.SHAPE_MASKS["barrier"] # 0

        # Table for Small Items (IDs 1-4) - Returns 8x15
        # Note: These lambda functions expect indices 0, 1, 2, 3, so we will subtract 1 from the ID
        SMALL_TABLE = [
            lambda frame: self.SHAPE_MASKS["roadie"][frame],  # 1 -> 0
            lambda frame: self.SHAPE_MASKS["groupies"][frame],  # 2 -> 1
            lambda frame: self.SHAPE_MASKS["promoter"][frame],  # 3 -> 2
            lambda frame: self.SHAPE_MASKS["photographer"],  # 4 -> 3
        ]

        # Table for Big Items (IDs 5-8) - Returns 16x15
        # Note: These lambda functions expect indices 0, 1, 2, so we will subtract 5 from the ID
        BIG_TABLE = [
            lambda frame: self.SHAPE_MASKS["big_roadie"][frame],  # 5 -> 0
            lambda frame: self.SHAPE_MASKS["big_groupies"][frame],  # 6 -> 1
            lambda frame: self.SHAPE_MASKS["big_promoter"][frame],  # 7 -> 2
            lambda frame: self.SHAPE_MASKS["big_photographer"],  # 8 -> 3
        ]

        MANAGER_MASK = lambda frame: self.SHAPE_MASKS["big_manager"][frame]  # 9

        def draw_barrier(r, x, y):
            mask = BARRIER_MASK
            return self.jr.render_at_clipped(r, x, y, mask)

        def draw_small(r, x, y, type_idx, frame_idx):
            mask = jax.lax.switch(type_idx - 1, SMALL_TABLE, frame_idx)
            return self.jr.render_at_clipped(r, x, y, mask)

        def draw_big(r, x, y, type_idx, frame_idx):
            # Map global ID (5,6,7,8) to local table index (0,1,2,3)
            mask = jax.lax.switch(type_idx - 5, BIG_TABLE, frame_idx)
            return self.jr.render_at_clipped(r, x, y, mask)

        def draw_manager(r, x, y, frame_idx):
            mask = MANAGER_MASK(frame_idx)
            return self.jr.render_at_clipped(r, x, y, mask)

        def body(i, r):
            box = obs_boxes[i]
            box_h = box[3]
            obs_type = box[4]

            does_exist = box_h > 0

            is_photographer = (obs_type == 4) | (obs_type == 8)

            cycle_len = self.consts.photographer_on_duration + self.consts.photographer_off_duration
            cycle_pos = state.time % cycle_len

            is_ghost = is_photographer & (cycle_pos >= self.consts.photographer_on_duration)

            # Final Decision: Draw if existing AND not a ghost
            should_draw = does_exist & jnp.logical_not(is_ghost)

            x, y = box[0], box[1]
            obs_frame_idx = jnp.where(state.obstacle_frames >= (self.consts.obstacle_frame_switch // 2), 0, 1)

            def render_op(curr_raster):
                return jax.lax.cond(
                    obs_type >= 5,
                    lambda _r: jax.lax.cond(
                                    obs_type == 9,
                                    lambda _r: draw_manager(_r, x, y, obs_frame_idx),
                                    lambda _r: draw_big(_r, x, y, obs_type, obs_frame_idx),
                                    _r
                                ),
                    lambda _r: jax.lax.cond(
                                    obs_type == 0,
                                    lambda _r: draw_barrier(_r, x, y),
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
        header_pos_y = self.consts.top_border - self.consts.top_blue_area_height
        header_mask = self.SHAPE_MASKS["header"]
        raster = self.jr.render_at(raster, 0, header_pos_y, header_mask)

        # Render Footer (Bottom Blue)
        footer_mask = self.SHAPE_MASKS["footer"]
        raster = self.jr.render_at(raster, 0, self.consts.bottom_border, footer_mask)

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
        render_x_pos = ((self.consts.screen_width // 2) + 19) - (num_to_render * 8)
        raster = self.jr.render_at(raster, render_x_pos - 8, 6, self.SHAPE_MASKS["dollar"])
        raster = self.jr.render_label_selective(raster, render_x_pos, 6,
                                                score_digits,
                                                score_digit_masks, start_index,num_to_render, spacing=8, 
                                                max_digits_to_render=5)

        # Render Countdown (On top of Blue Header)
        countdown_digits = self.jr.int_to_digits(state.countdown, max_digits=2)
        countdown_digit_masks = self.SHAPE_MASKS["timer_digits"]
        num_to_render = 2
        render_x_pos = (self.consts.screen_width // 2) - 3
        raster = self.jr.render_at(raster, render_x_pos - 9, 18, self.SHAPE_MASKS["timer_digits"][0]) # Renders the fixed leading 0
        raster = self.jr.render_at(raster, render_x_pos - 3, 18, self.SHAPE_MASKS["timer_colon"])
        raster = self.jr.render_label_selective(raster, render_x_pos, 18,
                                                countdown_digits, # the remaining seconds
                                                countdown_digit_masks, 0, num_to_render, spacing=7)

        # Render Side Bars (Black)
        black_bar_mask = self.SHAPE_MASKS["black_bar"]
        # left bar
        raster = self.jr.render_at(raster, 0, 0, black_bar_mask)
        # right bar
        raster = self.jr.render_at(raster, self.consts.screen_width - 8, 0, black_bar_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
