import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List, Dict, Optional, Any

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils


class JourneyEscapeConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    chicken_width: int = 6
    chicken_height: int = 8
    start_chicken_x: int = 44  # Fixed x position
    start_chicken_y: int = 170  # Fixed y position
    chicken_speed: int = 2  # constant downward speed

    obstacle_width: int = 8
    obstacle_height: int = 10
    obstacle_speed_px_per_frame: int = 2  # constant downward speed
    row_spawn_period_frames: int = 25  # spawn every N frames (tweakable)

    # border of the valid game space
    top_border: int = 33
    bottom_border: int = screen_width + 3
    left_border: int = 8
    right_border: int = screen_width - 22

    starting_score: int = 99

    hit_cooldown_frames: int = 8

    # predefined groups: [type, amount, spacing in px]
    obstacle_groups: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 0),
        (0, 4, 0),
        (0, 2, 35),
        (0, 2, 55),
        (0, 3, 10),
        (0, 3, 25),
    )


class JourneyEscapeState(NamedTuple):
    """Represents the current state of the game"""

    chicken_y: chex.Array
    chicken_x: chex.Array
    score: chex.Array
    time: chex.Array
    walking_frames: chex.Array
    walking_direction: chex.Array # can be {0, 1, 2} for {up/down, right, left}
    game_over: chex.Array

    row_timer: chex.Array  # int32
    obstacles: chex.Array  # (MAX_OBS, 5) -> x, y, w, h, type_idx | [pool]
    rng_key: chex.Array  # PRNGKey

    hit_cooldown: chex.Array  # int32


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class JourneyEscapeObservation(NamedTuple):
    chicken: EntityPosition


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
        # Start chicken at bottom
        chicken_y = self.consts.start_chicken_y
        chicken_x = self.consts.start_chicken_x

        MAX_OBS = 64
        empty_boxes = jnp.zeros((MAX_OBS, 5), dtype=jnp.int32)
        rng_key = jax.random.PRNGKey(0)

        state = JourneyEscapeState(
            chicken_y=jnp.array(chicken_y, dtype=jnp.int32),
            chicken_x=jnp.array(chicken_x, dtype=jnp.int32),
            score=jnp.array(self.consts.starting_score, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            walking_frames=jnp.array(0, dtype=jnp.int32),
            walking_direction=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
            row_timer=jnp.array(0, dtype=jnp.int32),
            obstacles=empty_boxes,
            rng_key=rng_key,
            hit_cooldown=jnp.array(0, dtype=jnp.int32),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: JourneyEscapeState, action: int) -> tuple[
        JourneyEscapeObservation, JourneyEscapeState, float, bool, JourneyEscapeInfo]:
        """Take a step in the game given an action"""

        # ---RAW PLAYER INPUT---
        # Compute vertical movement
        dy_int = jnp.where(
            (action == Action.UP) | (action == Action.UPLEFT) | (action == Action.UPRIGHT),
            -self.consts.chicken_speed,
            jnp.where(
                (action == Action.DOWN) | (action == Action.DOWNLEFT) | (action == Action.DOWNRIGHT),
                self.consts.chicken_speed,
                0
            ),
        )

        # Compute horizontal movement
        # faster movement if chicken on the top_maximum_position
        dx_int = jnp.where(
            (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT),
            -self.consts.chicken_speed,
            jnp.where(
                (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT),
                self.consts.chicken_speed,
                0
            ),
        )

        # determine walking direction of player
        player_move_right = (action == Action.RIGHT) | (action == Action.UPRIGHT) | (action == Action.DOWNRIGHT)
        player_move_left = (action == Action.LEFT) | (action == Action.UPLEFT) | (action == Action.DOWNLEFT)
        vertical = jnp.logical_not(player_move_right) & jnp.logical_not(player_move_left)

        new_walking_direction = jnp.where(
            vertical,
            0,                              # up/down
            jnp.where(player_move_right, 
                      1 ,                   # right
                      2)                    # left
        )

        # advance walking animation every frame, independent of input
        new_walking_frames = (state.walking_frames + 1) % 8
        new_walking_frames = jnp.where(new_walking_frames >= 8, 0, new_walking_frames)

        # Effective player movement boundaries
        player_min_y = self.consts.top_border
        player_max_y = self.consts.bottom_border + self.consts.chicken_height - 1
        player_min_x = self.consts.left_border
        player_max_x = self.consts.right_border + self.consts.chicken_width - 1

        # "Proposed" movement from input only (used for collision detection)
        pre_y = jnp.clip(state.chicken_y + dy_int, player_min_y, player_max_y).astype(jnp.int32)
        pre_x = jnp.clip(state.chicken_x + dx_int, player_min_x, player_max_x).astype(jnp.int32)

        #---OBSTACLES---

        # move & cull obstacles (no spawns yet) ---

        boxes = state.obstacles
        active = boxes[:, 3] > 0  # Active mask: entries with height > 0 are “alive”

        # Move down by constant speed only for active entries
        dy_obs = jnp.where(active, self.consts.obstacle_speed_px_per_frame, 0)  # speed for active, 0 for inactive
        boxes = boxes.at[:, 1].set(boxes[:, 1] + dy_obs)

        # Cull: if baseline y >= screen_height, deactivate by zeroing height
        cull_y = self.consts.bottom_border + self.consts.obstacle_height + 5
        offscreen = boxes[:, 1] >= cull_y
        new_heights = jnp.where(offscreen, 0, boxes[:, 3])  # int32[N]
        boxes = boxes.at[:, 3].set(new_heights)

        # carry-through for new fields (no behavior change yet)
        new_row_timer = (state.row_timer + 1) % self.consts.row_spawn_period_frames
        new_rng = state.rng_key  # unchanged key

        # Trigger: every row_spawn_period_frames frames
        spawn_now = (new_row_timer == 0)

        def spawn_if_cadence(carry):
            boxes_in, rng_in = carry
            rng_in, r1, r2 = jax.random.split(rng_in, 3)

            # Presets: (type, amount, spacing)
            presets = jnp.array(self.consts.obstacle_groups, dtype=jnp.int32)  # (K, 3)
            K = presets.shape[0]
            idx = jax.random.randint(r1, (), 0, K)
            type_idx = presets[idx, 0]
            amount = presets[idx, 1]
            spacing = presets[idx, 2]

            # Total width of the whole group
            w = self.consts.obstacle_width
            total_w = (w + spacing) * (amount - 1) + w

            # Choose spawn_x so the whole row is inside the screen horizontally.
            # We assume presets ALWAYS allow at least one valid position.
            min_x = self.consts.left_border
            max_x = self.consts.right_border - total_w
            # span must be positive; we don't treat "doesn't fit" as an option.
            span = (max_x - min_x) + 1
            spawn_x = min_x + jax.random.randint(r2, (), 0, span)

            # Pool capacity check: if not enough free slots, skip this spawn entirely.
            inactive = boxes_in[:, 3] == 0
            available = jnp.sum(inactive)
            enough_space = available >= amount

            def do_spawn(_):
                # Fixed loop bound to keep JAX happy (your presets have up to 3)
                MAX_GROUP = 3  # This might change later, when > 3 obstacles should be in one group

                # First MAX_GROUP free indices (static shape)
                free_idx = jnp.nonzero(inactive, size=MAX_GROUP, fill_value=0)[0]

                # Build row positions
                xs = spawn_x + jnp.arange(MAX_GROUP) * (w + spacing)
                ys = jnp.full((MAX_GROUP,), self.consts.top_border, dtype=jnp.int32)
                ws = jnp.full((MAX_GROUP,), w, dtype=jnp.int32)
                hs = jnp.full((MAX_GROUP,), self.consts.obstacle_height, dtype=jnp.int32)
                ts = jnp.full((MAX_GROUP,), type_idx, dtype=jnp.int32)

                # Place exactly `amount` entries; for t >= amount, do nothing
                def body(t, b):
                    def place_one(bb):
                        i = free_idx[t]
                        return bb.at[i].set(jnp.array([xs[t], ys[t], ws[t], hs[t], ts[t]], dtype=jnp.int32))

                    return jax.lax.cond(t < amount, place_one, lambda bb: bb, b)

                boxes_out = jax.lax.fori_loop(0, MAX_GROUP, body, boxes_in)
                return (boxes_out, rng_in)

            def skip_spawn(_):
                return (boxes_in, rng_in)

            return jax.lax.cond(enough_space, do_spawn, skip_spawn, operand=None)

        def no_spawn(carry):
            return carry

        boxes, new_rng = jax.lax.cond(
            spawn_now,
            spawn_if_cadence,
            no_spawn,
            operand=(boxes, state.rng_key)
        )

        # --- COLLISIONS---

        # We treat any obstacle with h > 0 as active.
        # boxes has shape (MAX_OBS, 5): [x, y, w, h, type_idx]

        def check_collision(box):
            # box: [x, y, w, h, type_idx]
            box_x = box[0]
            box_y = box[1]
            box_w = box[2]
            box_h = box[3]

            # Ignore inactive entries (h == 0)
            active = box_h > 0

            # --- Chicken AABB  --- [axis-aligned bounding box]

            ch_x0 = pre_x
            ch_x1 = pre_x + self.consts.chicken_width
            ch_y0 = pre_y - self.consts.chicken_height
            ch_y1 = pre_y

            # --- Obstacle AABB ---
            ob_x0 = box_x
            ob_x1 = box_x + box_w
            ob_y0 = box_y - box_h
            ob_y1 = box_y

            overlap_x = jnp.logical_and(ch_x0 < ob_x1, ch_x1 > ob_x0)
            overlap_y = jnp.logical_and(ch_y0 < ob_y1, ch_y1 > ob_y0)
            hit = jnp.logical_and(overlap_x, overlap_y)

            # Only count if this obstacle is active
            return jnp.logical_and(active, hit)

        # Vectorized over all entries in the obstacle pool
        collisions = jax.vmap(check_collision)(boxes)
        any_collision = jnp.any(collisions)

        # --- SCORE + COOLDOWN ---
        # - collision detected every frame (any_collision)
        # - score decremented only when NOT in cooldown
        # - after a "scoring hit", start the cooldown

        prev_cd = state.hit_cooldown
        cooling_down = prev_cd > 0

        # Cooldown ticks down every frame
        cd_after_tick = jnp.maximum(prev_cd - 1, 0)

        # Apply hit only if we're currently NOT cooling down
        apply_hit = jnp.logical_and(any_collision, jnp.logical_not(cooling_down))
        hit_penalty = jnp.where(apply_hit, 1, 0).astype(jnp.int32)
        new_score = (state.score - hit_penalty).astype(jnp.int32)
        new_score = jnp.maximum(new_score, 0)  # don't go below 0

        # If we scored a hit this frame, reset cooldown to N frames.
        # Otherwise, keep ticking it down.
        new_hit_cooldown = jnp.where(
            apply_hit,
            jnp.asarray(self.consts.hit_cooldown_frames, dtype=jnp.int32),
            cd_after_tick,
        )

        # ---MOVEMENT WITH OBSTACLE PHYSICS---

        def move_no_collision(_):
            # Just use the raw positions from input
            return pre_x, pre_y

        def move_collision(_):
            # When colliding:
            # - Can't move up; instead obstacles drag you down by their speed.
            # - At the very bottom, you get pushed to the right instead.
            # - Horizontal movement is
            #       - allowed but slower (half speed)
            #       - not allowed if you would get behind an obstacle

            # Already at the bottom?
            at_bottom = (pre_y >= player_max_y)

            # Vertical:
            #  - if not at bottom: move down with the obstacle speed
            #  - if at bottom: stay at bottom vertically
            moved_down_y = jnp.clip(
                state.chicken_y + self.consts.obstacle_speed_px_per_frame,
                player_min_y,
                player_max_y,
            )
            new_y0 = jax.lax.cond(
                at_bottom,
                lambda _: pre_y,
                lambda _: moved_down_y,
                operand=None,
            )

            # Horizontal:
            # Slow sideways movement while colliding
            slow_dx = dx_int // 2

            # Stay in screen
            cand_x_side = jnp.clip(state.chicken_x + slow_dx, player_min_x, player_max_x)

            # [avoid moving behind the obstacle]
            # Check if cand_x_side would break the collision -> only execute it in this case
            def check_collision_at(box):
                box_x = box[0]
                box_y = box[1]
                box_w = box[2]
                box_h = box[3]
                active_box = box_h > 0

                ch_x0 = cand_x_side
                ch_x1 = cand_x_side + self.consts.chicken_width
                ch_y0 = new_y0 - self.consts.chicken_height
                ch_y1 = new_y0

                ob_x0 = box_x
                ob_x1 = box_x + box_w
                ob_y0 = box_y - box_h
                ob_y1 = box_y

                overlap_x = jnp.logical_and(ch_x0 < ob_x1, ch_x1 > ob_x0)
                overlap_y = jnp.logical_and(ch_y0 < ob_y1, ch_y1 > ob_y0)
                hit = jnp.logical_and(overlap_x, overlap_y)
                return jnp.logical_and(active_box, hit)

            collisions_side = jax.vmap(check_collision_at)(boxes)
            still_collide_side = jnp.any(collisions_side)

            allowed_x_side = jax.lax.cond(
                still_collide_side,
                lambda _: state.chicken_x,  # block pushing further into the obstacle
                lambda _: cand_x_side,  # allow movement if it resolves collision
                operand=None,
            )

            # At bottom + collision: push right to make space
            push_dx = jnp.where(at_bottom, 2, 0)
            # Stay in screen
            new_x0 = jnp.clip(allowed_x_side + push_dx, player_min_x, player_max_x)

            return new_x0, new_y0

        new_x, new_y = jax.lax.cond(
            any_collision,
            move_collision,
            move_no_collision,
            operand=None,
        )

        # Update time
        new_time = (state.time + 1).astype(jnp.int32)

        # Check game over
        game_over = jnp.where(
            new_time >= 255 * 32,  # 2 minute time limit
            jnp.array(True),
            state.game_over,
        )

        new_state = JourneyEscapeState(
            chicken_y=new_y,
            chicken_x=new_x,
            score=new_score,
            time=new_time,
            walking_frames=new_walking_frames.astype(jnp.int32),
            walking_direction=new_walking_direction.astype(jnp.int32),
            game_over=game_over,
            row_timer=new_row_timer.astype(jnp.int32),
            obstacles=boxes.astype(jnp.int32),  # updated pool
            rng_key=new_rng,
            hit_cooldown=new_hit_cooldown.astype(jnp.int32),
        )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: JourneyEscapeState):
        # create chicken
        chicken = EntityPosition(
            x=state.chicken_x,
            y=state.chicken_y,
            width=jnp.array(self.consts.chicken_width, dtype=jnp.int32),
            height=jnp.array(self.consts.chicken_height, dtype=jnp.int32),
        )

        return JourneyEscapeObservation(chicken=chicken)

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
        - chicken: EntityPosition (x, y, width, height)
        - car: array of shape (10, 4) with x,y,width,height for each car
        """
        return spaces.Dict({
            "chicken": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "car": spaces.Box(low=0, high=210, shape=(10, 4), dtype=jnp.int32),
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
        # Flatten chicken position and dimensions
        chicken_flat = jnp.concatenate([
            obs.chicken.x.reshape(-1),
            obs.chicken.y.reshape(-1),
            obs.chicken.width.reshape(-1),
            obs.chicken.height.reshape(-1)
        ])

        # Concatenate all components
        return jnp.concatenate([chicken_flat]).astype(jnp.int32)


class JourneyEscapeRenderer(JAXGameRenderer):
    def __init__(self, consts: JourneyEscapeConstants = None):
        super().__init__()
        self.consts = consts or JourneyEscapeConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # Load and setup assets using the new pattern
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/journey_escape"

        # Create black bar sprite at initialization time
        black_bar_sprite = self._create_black_bar_sprite()

        # Add black bar sprite to the asset config as procedural asset
        asset_config.append({
            'name': 'black_bar',
            'type': 'procedural',
            'data': black_bar_sprite
        })

        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

        # --- rotate all car sprites by 90 degrees (clockwise) just for fun ---
        rotated_masks = {}
        for name, mask in self.SHAPE_MASKS.items():
            if "car_" in name:
                rotated_masks[name] = jnp.rot90(mask, k=1, axes=(0, 1))
            else:
                rotated_masks[name] = mask
        self.SHAPE_MASKS = rotated_masks

    def _create_black_bar_sprite(self) -> jnp.ndarray:
        """Create a black bar sprite for the left side of the screen."""
        # Create an 8-pixel wide black bar covering the full height
        bar_height = self.consts.screen_height
        bar_width = 8
        # Create black sprite with full alpha (255) so it gets added to palette
        black_bar = jnp.zeros((bar_height, bar_width, 4), dtype=jnp.uint8)
        black_bar = black_bar.at[:, :, 3].set(255)  # Set alpha to 255
        return black_bar

    def _get_asset_config(self) -> list:
        """Returns the declarative manifest of all assets for the game."""
        return [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {
                'name': 'player', 'type': 'group',
                'files': ['player_walk_front_0.npy', 'player_walk_front_1.npy', 
                          'player_run_right_0.npy', 'player_run_right_1.npy',
                          'player_run_left_0.npy', 'player_run_left_1.npy']
            },
            {'name': 'car_dark_red', 'type': 'single', 'file': 'car_dark_red.npy'},
            {'name': 'car_light_green', 'type': 'single', 'file': 'car_light_green.npy'},
            {'name': 'car_dark_green', 'type': 'single', 'file': 'car_dark_green.npy'},
            {'name': 'car_light_red', 'type': 'single', 'file': 'car_light_red.npy'},
            {'name': 'car_blue', 'type': 'single', 'file': 'car_blue.npy'},
            {'name': 'car_brown', 'type': 'single', 'file': 'car_brown.npy'},
            {'name': 'car_light_blue', 'type': 'single', 'file': 'car_light_blue.npy'},
            {'name': 'car_red', 'type': 'single', 'file': 'car_red.npy'},
            {'name': 'car_green', 'type': 'single', 'file': 'car_green.npy'},
            {'name': 'car_yellow', 'type': 'single', 'file': 'car_yellow.npy'},
            {'name': 'score_digits', 'type': 'digits', 'pattern': 'score_{}.npy'},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Select chicken sprite based on walking frames and direction
        use_idle = state.walking_frames < 4 
        sprite_index = state.walking_direction * 2 
        chicken_frame_index = jax.lax.select(use_idle, sprite_index, sprite_index+1)

        chicken_mask = self.SHAPE_MASKS["player"][chicken_frame_index]
        raster = self.jr.render_at(raster, state.chicken_x, state.chicken_y, chicken_mask)

        # Render obstacles
        # state.obstacles has shape (MAX_OBS, 5): [x, y, w, h, type_idx]
        # We consider entries with h > 0 as "active".
        obs_boxes = state.obstacles  # (MAX_OBS, 5)
        car_mask = self.SHAPE_MASKS["car_light_green"]  # placeholder sprite for all obstacles

        # Guards for the case MAX_OBS == 0 (unlikely, but keeps it robust)
        def draw_one(r, box):
            # box[0]=x, box[1]=y; we ignore w/h/type for rendering here
            return self.jr.render_at_clipped(r, box[0], box[1], car_mask)

        def body(i, r):
            # If h > 0, draw; else keep raster unchanged
            box = obs_boxes[i]
            is_active = box[3] > 0
            return jax.lax.cond(is_active, lambda rr: draw_one(rr, box), lambda rr: rr, r)

        # Iterate all possible slots in a JAX-friendly loop
        raster = jax.lax.fori_loop(0, obs_boxes.shape[0], body, raster)

        # Render score
        score_digits = self.jr.int_to_digits(state.score, max_digits=2)
        score_digit_masks = self.SHAPE_MASKS["score_digits"]

        is_single_digit = state.score < 10
        start_index = jax.lax.select(is_single_digit, 1, 0)
        num_to_render = jax.lax.select(is_single_digit, 1, 2)
        render_x = jax.lax.select(is_single_digit, 49 + 8 // 2, 49)

        raster = self.jr.render_label_selective(raster, render_x, 5, score_digits, score_digit_masks, start_index,
                                                num_to_render, spacing=8)

        # Render black bar on the left side
        black_bar_mask = self.SHAPE_MASKS["black_bar"]
        raster = self.jr.render_at(raster, 0, 0, black_bar_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
