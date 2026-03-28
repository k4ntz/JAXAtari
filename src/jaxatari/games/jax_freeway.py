import os
from functools import partial
import chex
import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, List, Dict, Optional, Any
from flax import struct

from jaxatari.environment import JaxEnvironment, ObjectObservation, JAXAtariAction as Action
import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.modification import AutoDerivedConstants

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Freeway.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    Note: Recolorings are added dynamically in the renderer based on constants.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {
            'name': 'player', 'type': 'group',
            'files': ['player_hit.npy', 'player_walk.npy', 'player_idle.npy']
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
    )

class FreewayConstants(AutoDerivedConstants):
    screen_width: int = struct.field(pytree_node=False, default=160)
    screen_height: int = struct.field(pytree_node=False, default=210)
    chicken_width: int = struct.field(pytree_node=False, default=6)
    chicken_height: int = struct.field(pytree_node=False, default=8)
    chicken_x: int = struct.field(pytree_node=False, default=44)  # Fixed x position
    car_width: int = struct.field(pytree_node=False, default=8)
    car_height: int = struct.field(pytree_node=False, default=10)
    num_lanes: int = struct.field(pytree_node=False, default=10)
    lane_spacing: int = struct.field(pytree_node=False, default=16)
    top_border: int = struct.field(pytree_node=False, default=15)
    top_path: int = struct.field(pytree_node=False, default=8)
    bottom_border: int = struct.field(pytree_node=False, default=180)
    # Collision response tuning
    throw_back_frames: int = struct.field(pytree_node=False, default=24)  # frames the chicken is pushed back after hit
    stun_frames: int = struct.field(pytree_node=False, default=28)        # frames the chicken cannot move after hit
    # After scoring (reaching the top and resetting), prevent movement for N frames
    post_score_stun_frames: int = struct.field(pytree_node=False, default=28)
    # Vertical offset to apply to chicken spawn after scoring (positive = lower on screen)
    post_score_spawn_offset_y: int = struct.field(pytree_node=False, default=1)
    # Collision box insets (shrink AABB without changing render sizes)
    chicken_hit_inset_x: int = struct.field(pytree_node=False, default=1)
    chicken_hit_inset_y_top: int = struct.field(pytree_node=False, default=-2)    # Top edge of chicken (when cars approach from above)
    chicken_hit_inset_y_bottom: int = struct.field(pytree_node=False, default=0) # Bottom edge of chicken (when cars approach from below)
    car_hit_inset_x: int = struct.field(pytree_node=False, default=0)
    car_hit_inset_y_top: int = struct.field(pytree_node=False, default=2)        # Top edge of car (for cars approaching from above)
    car_hit_inset_y_bottom: int = struct.field(pytree_node=False, default=0)     # Bottom edge of car (for cars approaching from below)
    # Fine-tune horizontal respawn offset applied when wrapping
    # Positive shifts right-moving lanes further right on re-entry (and vice versa for left-moving)
    respawn_offset: int = struct.field(pytree_node=False, default=8)
    # Fine-tune vertical car alignment within each lane (applied at reset)
    car_y_offset: int = struct.field(pytree_node=False, default=1)
    # Per-lane cadence phase offset (frames) for N-frame movement; allows aligning cadence to reference
    cadence_phase_offset: List[int] = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        -2, -2, -2, 0, 0,
        0, 0, -2, -2, 3
    ]))
    # This list defines the period and direction for each lane's pattern
    CAR_UPDATES: List[int] = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        -5,  # Lane 0
        -4,  # Lane 1
        -3,  # Lane 2
        -2,  # Lane 3
        -1,  # Lane 4
        1,   # Lane 5
        2,   # Lane 6
        3,   # Lane 7
        4,   # Lane 8
        5,   # Lane 9
    ]))
    # Per-lane initial phase offsets in pixels to align with ALE (applied to x at reset)
    # Lanes 0-4 move left; lanes 5-9 move right
    lane_phase_offset: List[int] = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
        5,  # lane 0 (+5 px)
        5,  # lane 1
        5,  # lane 2
        5,  # lane 3
        6,  # lane 4
        156, # lane 5 (+157 px)
        157, # lane 6
        157, # lane 7
        157, # lane 8
        157, # lane 9
    ]))
    # Upper 5 lanes move left (-), lower 5 lanes move right (+)
    # Value at i is the frequency in which car at lane i moves one pixel

    lane_borders: Optional[jnp.ndarray] = struct.field(pytree_node=False, default=None)
    
    # Car colors for each lane (10 lanes). If color is None, use original sprite color.
    # Otherwise, recolor the car sprite to the specified RGB color.
    # Note: Use None for original color, (0, 0, 0) for actual black.
    CAR_COLORS: List[Optional[Tuple[int, int, int]]] = struct.field(pytree_node=False, default_factory=lambda: [
        None,  # Lane 0 - use original color
        None,  # Lane 1 - use original color
        None,  # Lane 2 - use original color
        None,  # Lane 3 - use original color
        None,  # Lane 4 - use original color
        None,  # Lane 5 - use original color
        None,  # Lane 6 - use original color
        None,  # Lane 7 - use original color
        None,  # Lane 8 - use original color
        None,  # Lane 9 - use original color
    ])

    # Game Duration Config
    # Original Atari 2600 timer logic results in exactly 8192 frames
    game_duration_frames: int = struct.field(pytree_node=False, default=8192)
    # Score starts blinking at 2:00 (7680 frames) to warn players of imminent game over
    blink_start_frames: int = struct.field(pytree_node=False, default=7680)
    # Rate at which score colors cycle (frames per color change)
    score_blink_rate: int = struct.field(pytree_node=False, default=2)
    
    # Colors for the blinking score cycle (RGB)
    SCORE_BLINK_COLORS: List[Tuple[int, int, int]] = struct.field(pytree_node=False, default_factory=lambda: jnp.array([
         (210, 210, 64), # Yellow (original)
         (210, 64, 64),  # Red
         (64, 210, 64),  # Green
         (64, 64, 210),  # Blue
         (210, 64, 210), # Magenta
         (64, 210, 210), # Cyan
    ]))

    # Asset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = struct.field(pytree_node=False, default_factory=_get_default_asset_config)
    
    def compute_derived(self):
        """Compute derived constants based on static fields."""
        return {
            'lane_borders': jnp.array([
                self.top_border + self.top_path,  # Lane 0
                1 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 1
                2 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 2
                3 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 3
                4 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 4
                5 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 5
                6 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 6
                7 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 7
                8 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 8
                9 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 10
                10 * self.lane_spacing + (self.top_border + self.top_path) + 2,  # Lane 10
            ], dtype=jnp.int32),
        }

@struct.dataclass
class FreewayState:
    """Represents the current state of the game"""
    chicken_y: chex.Array
    cars: chex.Array  # Shape: (num_lanes, 2) for x,y positions (ints for render/collide)
    # Per-lane cadence counters (frames), advance independently to sync movement patterns per lane
    lane_time: chex.Array
    score: chex.Array
    time: chex.Array
    cooldown: chex.Array  # Cooldown after collision
    walking_frames: chex.Array
    game_over: chex.Array

@struct.dataclass
class EntityPosition:
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


@struct.dataclass
class FreewayObservation:
    chicken: ObjectObservation
    car: ObjectObservation # ObjectObservation that includes arrays for all values with the information of all 10 cars


@struct.dataclass
class FreewayInfo:
    time: jnp.ndarray


class JaxFreeway(JaxEnvironment[FreewayState, FreewayObservation, FreewayInfo, FreewayConstants]):
    # Map agent action indices (0, 1, 2) to ALE console actions
    # 0 -> NOOP, 1 -> UP, 2 -> DOWN
    ACTION_SET: jnp.ndarray = jnp.array(
        [Action.NOOP, Action.UP, Action.DOWN], dtype=jnp.int32
    )

    def __init__(self, consts: FreewayConstants = None):
        if consts is None:
            consts = FreewayConstants()
        super().__init__(consts)
        self.state = self.reset()
        self.renderer = FreewayRenderer(self.consts)

    def reset(self, key: jax.random.PRNGKey = None) -> Tuple[FreewayObservation, FreewayState]:
        """Initialize a new game state"""
        # Start chicken at bottom
        chicken_y = self.consts.bottom_border + self.consts.chicken_height - 1
        # Initialize one car per lane
        cars = []
        for lane in range(self.consts.num_lanes):
            lane_y = (
                self.consts.lane_borders[lane]
                + int(self.consts.lane_spacing / 2)
                - int(self.consts.car_height / 2)
            ) + int(self.consts.car_y_offset)
            # Upper 5 lanes start from right, lower 5 lanes start from left
            if lane < 5:
                x = (
                    self.consts.screen_width - self.consts.car_width + 0
                )  # Start from right
            else:
                x = 0  # Start from left
            cars.append([x, lane_y])

        cars = jnp.array(cars, dtype=jnp.int32)
        # Apply per-lane phase offsets
        phase = jnp.array(self.consts.lane_phase_offset, dtype=jnp.int32)
        cars = cars.at[:, 0].add(phase)

        # Initialize per-lane cadence counters using configured phase offsets
        periods0 = jnp.abs(jnp.array(self.consts.CAR_UPDATES, dtype=jnp.int32))
        phases0 = jnp.array(self.consts.cadence_phase_offset, dtype=jnp.int32) % periods0

        state = FreewayState(
            chicken_y=jnp.array(chicken_y, dtype=jnp.int32),
            cars=cars,
            lane_time=phases0,
            score=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            cooldown=jnp.array(0, dtype=jnp.int32),
            walking_frames=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FreewayState, action: int) -> tuple[FreewayObservation, FreewayState, float, bool, FreewayInfo]:
        """Take a step in the game given an action"""
        # Translate compact agent action (0, 1, 2) to ALE console action constant
        atari_action = jnp.take(self.ACTION_SET, action)

        # Update chicken position if not in cooldown
        dy = jnp.where(
            jnp.logical_and(
                state.cooldown > self.consts.stun_frames,
                state.cooldown <= (self.consts.stun_frames + self.consts.throw_back_frames)
            ),
            1.0,
            jnp.where(
                atari_action == Action.UP,
                -1.0,
                jnp.where(atari_action == Action.DOWN, 1.0, 0.0),
            ),
        )

        dy = jnp.where(
            jnp.logical_and(state.cooldown > 0, state.cooldown <= self.consts.stun_frames),
            0.0,
            dy,
        )

        # add one to the walking frames if dy != 0, if it is 0 reset to 0
        new_walking_frames = jnp.where(dy != 0, state.walking_frames + 1, 0)

        # reset new_walking frames at 8
        new_walking_frames = jnp.where(new_walking_frames >= 8, 0, new_walking_frames)

        new_y = jnp.clip(
            state.chicken_y + dy.astype(jnp.int32),
            self.consts.top_border,
            self.consts.bottom_border + self.consts.chicken_height - 1,
        ).astype(jnp.int32)

        # Implements the [0, 0, ..., 1] repeating pattern based on CAR_UPDATES
        periods = jnp.abs(jnp.array(self.consts.CAR_UPDATES, dtype=jnp.int32))
        signs = jnp.sign(jnp.array(self.consts.CAR_UPDATES, dtype=jnp.int32))
        # Per-lane cadence counters: move when the counter reaches period-1, then keep advancing
        should_move_mask = (state.lane_time == (periods - 1))
        delta_x = jnp.where(should_move_mask, signs, 0).astype(jnp.int32)

        # Apply moves to integer x positions
        pre_x = state.cars[:, 0]
        x_int = pre_x + delta_x

        # Wrap positions to [-car_width, screen_width)
        range_len_i = self.consts.screen_width + self.consts.car_width
        x_int_wrapped = ((x_int + self.consts.car_width) % range_len_i) - self.consts.car_width

        # Apply respawn offset only to entries that wrapped this frame
        wrapped_right = jnp.logical_and(signs > 0, x_int >= self.consts.screen_width)
        wrapped_left = jnp.logical_and(signs < 0, x_int < -self.consts.car_width)

        offset = jnp.asarray(self.consts.respawn_offset, dtype=jnp.int32)
        adjusted = x_int_wrapped
        adjusted = jnp.where(wrapped_right, x_int_wrapped + offset, adjusted)
        adjusted = jnp.where(wrapped_left, x_int_wrapped - offset, adjusted)
        # Keep within valid range after offset
        adjusted = jnp.clip(adjusted, -self.consts.car_width, self.consts.screen_width - 1)

        # Update integer car positions
        new_cars = state.cars.at[:, 0].set(adjusted.astype(jnp.int32))

        # Advance per-lane cadence counters
        new_lane_time = (state.lane_time + 1) % periods

        # Check for collisions
        def check_collision(car_pos):
            car_x, car_y = car_pos
            # Chicken AABB with insets
            cxi = jnp.asarray(self.consts.chicken_hit_inset_x, dtype=jnp.int32)
            cyi_top = jnp.asarray(self.consts.chicken_hit_inset_y_top, dtype=jnp.int32)
            cyi_bottom = jnp.asarray(self.consts.chicken_hit_inset_y_bottom, dtype=jnp.int32)
            ch_x0 = self.consts.chicken_x + cxi
            ch_x1 = self.consts.chicken_x + self.consts.chicken_width - cxi
            ch_y0 = state.chicken_y - self.consts.chicken_height + cyi_top
            ch_y1 = state.chicken_y - cyi_bottom

            # Car AABB with insets
            kxi = jnp.asarray(self.consts.car_hit_inset_x, dtype=jnp.int32)
            kyi_top = jnp.asarray(self.consts.car_hit_inset_y_top, dtype=jnp.int32)
            kyi_bottom = jnp.asarray(self.consts.car_hit_inset_y_bottom, dtype=jnp.int32)
            car_x0 = car_x + kxi
            car_x1 = car_x + self.consts.car_width - kxi
            car_y0 = car_y - self.consts.car_height + kyi_top
            car_y1 = car_y - kyi_bottom

            overlap_x = jnp.logical_and(ch_x0 < car_x1, ch_x1 > car_x0)
            overlap_y = jnp.logical_and(ch_y0 < car_y1, ch_y1 > car_y0)
            return jnp.logical_and(overlap_x, overlap_y)

        # Check collisions for all cars
        collisions = jax.vmap(check_collision)(new_cars)
        any_collision = jnp.any(collisions)
        any_collision = jax.lax.cond(
            state.cooldown > 0, lambda _: False, lambda _: any_collision, operand=None
        )

        # Update cooldown
        new_cooldown = jnp.where(
            any_collision,
            self.consts.throw_back_frames + self.consts.stun_frames,
            jnp.maximum(0, state.cooldown - 1),
        ).astype(jnp.int32)

        # Update score if chicken reaches top
        new_score = jnp.where(
            new_y <= self.consts.top_border, state.score + 1, state.score
        ).astype(jnp.int32)

        # Reset chicken position if scored
        scored = new_y <= self.consts.top_border
        new_y = jnp.where(
            scored,
            self.consts.bottom_border + self.consts.chicken_height - 1 + self.consts.post_score_spawn_offset_y,
            new_y,
        ).astype(jnp.int32)

        # Apply a post-score stun to prevent immediate movement after crossing once
        new_cooldown = jnp.where(
            scored,
            jnp.maximum(new_cooldown, jnp.asarray(self.consts.post_score_stun_frames, dtype=jnp.int32)),
            new_cooldown,
        )

        # Update time
        new_time = (state.time + 1).astype(jnp.int32)

        # Check game over based on exact frame count
        game_over = jnp.where(
            new_time >= self.consts.game_duration_frames,
            jnp.array(True),
            state.game_over,
        )

        new_state = FreewayState(
            chicken_y=new_y,
            cars=new_cars,
            lane_time=new_lane_time,
            score=new_score,
            time=new_time,
            cooldown=new_cooldown,
            walking_frames=new_walking_frames.astype(jnp.int32),
            game_over=game_over,
        )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FreewayState):
        # create chicken
        chicken = ObjectObservation.create(
            x=jnp.array(self.consts.chicken_x, dtype=jnp.int32),
            y=state.chicken_y,
            width=jnp.array(self.consts.chicken_width, dtype=jnp.int32),
            height=jnp.array(self.consts.chicken_height, dtype=jnp.int32),
            active=jnp.array(True, dtype=jnp.bool_),
        )

        # create cars
        cars_pos = jnp.zeros((self.consts.num_lanes, 4), dtype=jnp.int32)
        for i in range(self.consts.num_lanes):
            car_pos = state.cars.at[i].get()
            cars_pos = cars_pos.at[i].set(
                jnp.array(
                    [
                        car_pos.at[0].get(),  # x position
                        car_pos.at[1].get(),  # y position
                        self.consts.car_width,  # width
                        self.consts.car_height,  # height
                    ],
                    dtype=jnp.int32
                )
            )

        # get the car orientation by checking the sign of the lane's CAR_UPDATES value (positive = right, negative = left) in degree (left = 270°, right = 90°)
        car_orientation = jnp.where(
            jnp.array(self.consts.CAR_UPDATES, dtype=jnp.int32) < 0, 270, 90
        )

        cars = ObjectObservation.create(
            x=cars_pos[:, 0],
            y=cars_pos[:, 1],
            width=cars_pos[:, 2],
            height=cars_pos[:, 3],
            active=jnp.ones((self.consts.num_lanes,), dtype=jnp.bool_),
            visual_id=jnp.arange(self.consts.num_lanes, dtype=jnp.int32),
            orientation=car_orientation.astype(jnp.int32)
        )

        return FreewayObservation(chicken=chicken, car=cars)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: FreewayState) -> FreewayInfo:
        return FreewayInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FreewayState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Freeway."""
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Freeway.
        The observation contains:
        - chicken: ObjectObservation (x, y, width, height, active, visual_id, state, orientation)
        - car: array of ObjectObservation (10 cars, each with x, y, width, height, active, visual_id, orientation)
        """
        return spaces.Dict({
            "chicken": spaces.get_object_space(n=None, screen_size=(self.consts.screen_height, self.consts.screen_width)),
            "car": spaces.get_object_space(n=self.consts.num_lanes, screen_size=(self.consts.screen_height, self.consts.screen_width)),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Freeway.
        The image is a RGB image with shape (210, 160, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )
    
    def render(self, state: FreewayState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)


class FreewayRenderer(JAXGameRenderer):
    def __init__(self, consts: FreewayConstants = None, config: render_utils.RendererConfig = None):
        self.consts = consts or FreewayConstants()
        super().__init__(self.consts)
        
        # Use injected config if provided, else default
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(210, 160),
                channels=3,
                downscale=None
            )
        else:
            self.config = config
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load and setup assets using the new pattern
        # Convert tuple to list so we can modify it
        asset_config = list(self.consts.ASSET_CONFIG)
        sprite_path = os.path.join(render_utils.get_base_sprite_dir(), "freeway")
        
        # Map lane index to car sprite name
        car_sprite_names = [
            'car_dark_red',      # Lane 0
            'car_light_green',   # Lane 1
            'car_dark_green',    # Lane 2
            'car_light_red',     # Lane 3
            'car_blue',          # Lane 4
            'car_brown',         # Lane 5
            'car_light_blue',    # Lane 6
            'car_red',           # Lane 7
            'car_green',         # Lane 8
            'car_yellow',        # Lane 9
        ]
        
        # Add recoloring rules to car assets if colors are specified
        for i, asset in enumerate(asset_config):
            if asset.get('name') in car_sprite_names:
                lane_idx = car_sprite_names.index(asset['name'])
                color = self.consts.CAR_COLORS[lane_idx]
                if color is not None:
                    # Add recoloring rule: global replace with target color
                    if 'recolorings' not in asset:
                        asset['recolorings'] = {}
                    asset['recolorings']['recolored'] = {'target': color}
        
        # Add recoloring rules to score_digits for blink colors
        for i, asset in enumerate(asset_config):
            if asset.get('name') == 'score_digits':
                if 'recolorings' not in asset:
                    asset['recolorings'] = {}
                # Add a recolored variant for each blink color
                for idx, color in enumerate(self.consts.SCORE_BLINK_COLORS):
                    asset['recolorings'][f'blink_{idx}'] = {'target': color}
                break
        
        # Create black bar sprite at initialization time
        black_bar_sprite = self._create_black_bar_sprite()
        
        # Append procedural assets
        asset_config.append({
            'name': 'black_bar', 
            'type': 'procedural', 
            'data': black_bar_sprite
        })
        
        # Load all assets, create palette, and generate ID masks
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        
        # Setup score masks tensor from recolored variants
        self.score_masks_tensor = self._setup_score_masks_tensor()

    def _setup_score_masks_tensor(self) -> jnp.ndarray:
        """
        Creates a tensor of score digit masks for all required color palettes.
        Index 0: Default color
        Index 1..N: Blinking colors
        
        Returns:
            jnp.ndarray: Shape (NumPalettes, 10, H, W)
        """
        # 1. Get default masks (already stacked as (10, H, W) from load_and_setup_assets)
        default_masks = self.SHAPE_MASKS["score_digits"]
        # Ensure it's a stacked array if it's a list
        if isinstance(default_masks, list):
            default_masks = jnp.stack(default_masks)
        
        all_palettes = [default_masks]
        
        # 2. Get recolored masks for each blink color (created via recoloring system)
        for idx in range(len(self.consts.SCORE_BLINK_COLORS)):
            blink_key = f"score_digits_blink_{idx}"
            if blink_key in self.SHAPE_MASKS:
                # Masks are already stacked as (10, H, W) from load_and_setup_assets
                blink_masks = self.SHAPE_MASKS[blink_key]
                if isinstance(blink_masks, list):
                    blink_masks = jnp.stack(blink_masks)
                all_palettes.append(blink_masks)
            else:
                # Fallback: if recoloring didn't create the variant, use default
                all_palettes.append(default_masks)
            
        # 3. Stack all palettes into one master tensor
        # Shape: (NumColors+1, 10, H, W)
        return jnp.stack(all_palettes)

    def _create_black_bar_sprite(self) -> jnp.ndarray:
        """Create a black bar sprite for the left side of the screen."""
        # Create an 8-pixel wide black bar covering the full height
        bar_height = self.consts.screen_height
        bar_width = 8
        # Create black sprite with full alpha (255) so it gets added to palette
        black_bar = jnp.zeros((bar_height, bar_width, 4), dtype=jnp.uint8)
        black_bar = black_bar.at[:, :, 3].set(255)  # Set alpha to 255
        return black_bar

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Draw fixed chicken (right side - "Computer")
        chicken_idle_mask = self.SHAPE_MASKS["player"][2]
        raster = self.jr.render_at(raster, 110, self.consts.bottom_border + self.consts.chicken_height - 1, chicken_idle_mask)

        # Draw active chicken (left side - Player 1)
        use_idle = state.walking_frames < 4
        chicken_frame_index = jax.lax.select(use_idle, 2, 1)
        
        is_hit = state.cooldown > 0
        chicken_frame_index = jax.lax.select(
            jnp.logical_and(is_hit, jnp.logical_or((state.cooldown % 8) < 4, state.cooldown < 30)),
            0,
            chicken_frame_index
        )
        
        chicken_mask = self.SHAPE_MASKS["player"][chicken_frame_index]
        raster = self.jr.render_at(raster, self.consts.chicken_x, state.chicken_y, chicken_mask)

        # Draw cars
        car_sprite_names = [
            'car_dark_red',      # Lane 0
            'car_light_green',   # Lane 1
            'car_dark_green',    # Lane 2
            'car_light_red',     # Lane 3
            'car_blue',          # Lane 4
            'car_brown',         # Lane 5
            'car_light_blue',    # Lane 6
            'car_red',           # Lane 7
            'car_green',         # Lane 8
            'car_yellow',        # Lane 9
        ]
        
        for i in range(self.consts.num_lanes):
            sprite_name = car_sprite_names[i]
            # Use recolored variant if color is specified, otherwise use original
            if self.consts.CAR_COLORS[i] is not None:
                mask_key = f"{sprite_name}_recolored"
            else:
                mask_key = sprite_name
            car_mask = self.SHAPE_MASKS[mask_key]
            raster = self.jr.render_at_clipped(raster, state.cars[i, 0], state.cars[i, 1], car_mask)

        # --- SCORE RENDERING ---
        should_blink = state.time >= self.consts.blink_start_frames
        blink_cycle_idx = (state.time // self.consts.score_blink_rate) % len(self.consts.SCORE_BLINK_COLORS)
        
        # Use direct access for default (matches original behavior) or tensor for blinking
        # This ensures exact compatibility with the original version when not blinking
        def get_default_masks():
            return self.SHAPE_MASKS["score_digits"]
        
        def get_blink_masks():
            palette_index = blink_cycle_idx + 1
            return self.score_masks_tensor[palette_index]
        
        current_score_masks = jax.lax.cond(
            should_blink,
            get_blink_masks,
            get_default_masks
        )
        
        # 1. Player 1 Score (Left)
        score_digits_p1 = self.jr.int_to_digits(state.score, max_digits=2)
        is_single_digit_p1 = state.score < 10
        start_index_p1 = jax.lax.select(is_single_digit_p1, 1, 0)
        num_to_render_p1 = jax.lax.select(is_single_digit_p1, 1, 2)
        render_x_p1 = jax.lax.select(is_single_digit_p1, 49, 41)
        
        raster = self.jr.render_label_selective(
            raster, 
            render_x_p1, 
            5, 
            score_digits_p1, 
            current_score_masks, 
            start_index_p1, 
            num_to_render_p1, 
            spacing=8
        )

        # 2. Player 2 / Computer Score (Right - Dummy 0 for now)
        # Position logic: Right chicken is at 110. Offset is similar to left (110 + 5ish)
        # Center of right lane roughly 115.
        score_digits_p2 = self.jr.int_to_digits(0, max_digits=1) # Always 0
        render_x_p2 = 113 # Fixed position for "0"
        
        # Render '00' on the right side.
        raster = self.jr.render_label_selective(
            raster, 
            render_x_p2, 
            5, 
            score_digits_p2, 
            current_score_masks, 
            0,
            1, 
            spacing=8
        )

        # Draw black bar
        black_bar_mask = self.SHAPE_MASKS["black_bar"]
        raster = self.jr.render_at(raster, 0, 0, black_bar_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
