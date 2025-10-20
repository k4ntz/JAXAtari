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

class FreewayConstants(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210
    chicken_width: int = 6
    chicken_height: int = 8
    chicken_x: int = 40  # Fixed x position
    car_width: int = 8
    car_height: int = 10
    num_lanes: int = 10
    lane_spacing: int = 16
    car_speeds: List[float] = None
    lane_borders: List[int] = None
    top_border: int = 15
    top_path: int = 8
    bottom_border: int = 180
    # Upper 5 lanes move left (-), lower 5 lanes move right (+)
    # Value at i is the frequency in which car at lane i moves one pixel
    car_update = [
        -5,  # Lane 0
        -4,  # Lane 1
        -3,  # Lane 2
        -2,  # Lane 3
        -1,  # Lane 4
        1,  # Lane 5
        2,  # Lane 6
        3,  # Lane 7
        4,  # Lane 8
        5,  # Lane 9
    ]
    # Upper 5 lanes move left (-), lower 5 lanes move right (+)
    # Value at i is the frequency in which car at lane i moves one pixel
    lane_borders = [
        top_border + top_path,  # Lane 0
        1 * lane_spacing + (top_border + top_path),  # Lane 1
        2 * lane_spacing + (top_border + top_path),  # Lane 2
        3 * lane_spacing + (top_border + top_path),  # Lane 3
        4 * lane_spacing + (top_border + top_path),  # Lane 4
        5 * lane_spacing + (top_border + top_path),  # Lane 5
        6 * lane_spacing + (top_border + top_path),  # Lane 6
        7 * lane_spacing + (top_border + top_path),  # Lane 7
        8 * lane_spacing + (top_border + top_path),  # Lane 8
        9 * lane_spacing + (top_border + top_path),  # Lane 10
        10 * lane_spacing
        + (top_border + top_path)
        + 2,  # Lane 10
    ]


class FreewayState(NamedTuple):
    """Represents the current state of the game"""

    chicken_y: chex.Array
    cars: chex.Array  # Shape: (num_lanes, 2) for x,y positions
    score: chex.Array
    time: chex.Array
    cooldown: chex.Array  # Cooldown after collision
    walking_frames: chex.Array
    lives_lost: chex.Array
    game_over: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class FreewayObservation(NamedTuple):
    chicken: EntityPosition
    car: jnp.ndarray  # Shape: (10, 4) with x,y,width,height for each car


class FreewayInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: jnp.ndarray


class JaxFreeway(JaxEnvironment[FreewayState, FreewayObservation, FreewayInfo, FreewayConstants]):
    def __init__(self, consts: FreewayConstants = None, reward_funcs: list[callable]=None):
        if consts is None:
            consts = FreewayConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.state = self.reset()
        self.renderer = FreewayRenderer()

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
            )
            # Upper 5 lanes start from right, lower 5 lanes start from left
            if lane < 5:
                x = (
                    self.consts.screen_width - self.consts.car_width + 0
                )  # Start from right
            else:
                x = 0  # Start from left
            cars.append([x, lane_y])

        state = FreewayState(
            chicken_y=jnp.array(chicken_y, dtype=jnp.int32),
            cars=jnp.array(cars, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            cooldown=jnp.array(0, dtype=jnp.int32),
            walking_frames=jnp.array(0, dtype=jnp.int32),
            lives_lost=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False, dtype=jnp.bool_),
        )

        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FreewayState, action: int) -> tuple[FreewayObservation, FreewayState, float, bool, FreewayInfo]:
        """Take a step in the game given an action"""
        # Update chicken position if not in cooldown
        dy = jnp.where(
            jnp.logical_and(state.cooldown > 30, state.cooldown < 54), # throw the chicken back for 24 frames
            1.0,
            jnp.where(action == Action.UP, -1.0, jnp.where(action == Action.DOWN, 1.0, 0.0)),
        )

        dy = jnp.where(
            jnp.logical_and(state.cooldown > 0, state.cooldown < 30), # stun the chicken for 30 frames
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

        # Update car positions
        new_cars = state.cars
        for lane in range(self.consts.num_lanes):
            # Update x position based on lane speed
            dir = (
                self.consts.car_update[lane] / jnp.abs(self.consts.car_update[lane])
            ).astype(jnp.int32)
            new_x = jax.lax.cond(
                jnp.equal(jnp.mod(state.time, self.consts.car_update[lane]), 0),
                lambda: state.cars[lane, 0] + dir,
                lambda: state.cars[lane, 0],
            )

            # Wrap around screen
            new_x = jnp.where(
                self.consts.car_update[lane] > 0,
                jnp.where(
                    new_x > self.consts.screen_width, -self.consts.car_width, new_x
                ),
                jnp.where(
                    new_x < -self.consts.car_width, self.consts.screen_width, new_x
                ),
            ).astype(jnp.int32)

            new_cars = new_cars.at[lane, 0].set(new_x)

        # Check for collisions
        def check_collision(car_pos):
            car_x, car_y = car_pos
            return jnp.logical_and(
                self.consts.chicken_x < car_x + self.consts.car_width,
                jnp.logical_and(
                    self.consts.chicken_x + self.consts.chicken_width > car_x,
                    jnp.logical_and(
                        state.chicken_y - self.consts.chicken_height < car_y,
                        state.chicken_y > car_y - self.consts.car_height,
                    ),
                ),
            )

        # Check collisions for all cars
        collisions = jax.vmap(check_collision)(new_cars)
        any_collision = jnp.any(collisions)
        any_collision = jax.lax.cond(
            state.cooldown > 0, lambda _: False, lambda _: any_collision, operand=None
        )

        # Update cooldown
        new_cooldown = jnp.where(
            any_collision,
            24 + 30,  # Set cooldown frames after collision (24 frames of flying backwards, 30 frames of being "stunned")
            jnp.maximum(0, state.cooldown - 1),
        ).astype(jnp.int32)

        # Update score if chicken reaches top
        new_score = jnp.where(
            new_y <= self.consts.top_border, state.score + 1, state.score
        ).astype(jnp.int32)

        # Reset chicken position if scored
        new_y = jnp.where(
            new_y <= self.consts.top_border,
            self.consts.bottom_border + self.consts.chicken_height - 1,
            new_y,
        ).astype(jnp.int32)

        # Update time
        new_time = (state.time + 1).astype(jnp.int32)

        # Check game over (optional: could be based on time or score limit)
        game_over = jnp.where(
            new_time >= 255 * 32,  # 2 minute time limit
            jnp.array(True),
            state.game_over,
        )

        new_live_lost = jnp.where(
            any_collision,
            state.lives_lost + 1,
            state.lives_lost,
        )

        new_state = FreewayState(
            chicken_y=new_y,
            cars=new_cars,
            score=new_score,
            time=new_time,
            cooldown=new_cooldown,
            walking_frames=new_walking_frames.astype(jnp.int32),
            lives_lost=new_live_lost,
            game_over=game_over,
        )
        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state, all_rewards)

        return obs, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FreewayState):
        # create chicken
        chicken = EntityPosition(
            x=jnp.array(self.consts.chicken_x, dtype=jnp.int32),
            y=state.chicken_y,
            width=jnp.array(self.consts.chicken_width, dtype=jnp.int32),
            height=jnp.array(self.consts.chicken_height, dtype=jnp.int32),
        )

        # create cars
        cars = jnp.zeros((self.consts.num_lanes, 4), dtype=jnp.int32)
        for i in range(self.consts.num_lanes):
            car_pos = state.cars.at[i].get()
            cars = cars.at[i].set(
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
        return FreewayObservation(chicken=chicken, car=cars)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: FreewayState, all_rewards: chex.Array = None) -> FreewayInfo:
        return FreewayInfo(time=state.time, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: FreewayState, state: FreewayState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: FreewayState, state: FreewayState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: FreewayState) -> bool:
        return state.game_over

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Freeway.
        Actions are:
        0: NOOP
        1: UP
        2: DOWN
        """
        return spaces.Discrete(3)

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Freeway.
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

    def obs_to_flat_array(self, obs: FreewayObservation) -> jnp.ndarray:
        """Convert observation to a flat array."""
        # Flatten chicken position and dimensions
        chicken_flat = jnp.concatenate([
            obs.chicken.x.reshape(-1),
            obs.chicken.y.reshape(-1),
            obs.chicken.width.reshape(-1),
            obs.chicken.height.reshape(-1)
        ])
        
        # Flatten car positions and dimensions
        cars_flat = obs.car.reshape(-1)
        
        # Concatenate all components
        return jnp.concatenate([chicken_flat, cars_flat]).astype(jnp.int32)


class FreewayRenderer(JAXGameRenderer):
    def __init__(self, consts: FreewayConstants = None):
        super().__init__()
        self.consts = consts or FreewayConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)
        
        # Load and setup assets using the new pattern
        asset_config = self._get_asset_config()
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/freeway"
        
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
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Draw fixed chicken at x=110
        chicken_idle_mask = self.SHAPE_MASKS["player"][2]  # player_idle is index 2
        raster = self.jr.render_at(raster, 110, self.consts.bottom_border + self.consts.chicken_height - 1, chicken_idle_mask)

        # Select chicken sprite based on walking frames and hit state
        use_idle = state.walking_frames < 4
        chicken_frame_index = jax.lax.select(use_idle, 2, 1)  # 2=idle, 1=walk
        
        is_hit = state.cooldown > 0
        chicken_frame_index = jax.lax.select(
            jnp.logical_and(is_hit, jnp.logical_or((state.cooldown % 8) < 4, state.cooldown < 30)),
            0,  # player_hit is index 0
            chicken_frame_index
        )
        
        chicken_mask = self.SHAPE_MASKS["player"][chicken_frame_index]
        raster = self.jr.render_at(raster, self.consts.chicken_x, state.chicken_y, chicken_mask)

        # Render cars in the correct colors
        car_masks = [
            self.SHAPE_MASKS["car_dark_red"],
            self.SHAPE_MASKS["car_light_green"],
            self.SHAPE_MASKS["car_dark_green"],
            self.SHAPE_MASKS["car_light_red"],
            self.SHAPE_MASKS["car_blue"],
            self.SHAPE_MASKS["car_brown"],
            self.SHAPE_MASKS["car_light_blue"],
            self.SHAPE_MASKS["car_red"],
            self.SHAPE_MASKS["car_green"],
            self.SHAPE_MASKS["car_yellow"],
        ]
        
        for i in range(self.consts.num_lanes):
            raster = self.jr.render_at(raster, state.cars[i, 0], state.cars[i, 1], car_masks[i])

        # Render score
        score_digits = self.jr.int_to_digits(state.score, max_digits=2)
        score_digit_masks = self.SHAPE_MASKS["score_digits"]
        
        is_single_digit = state.score < 10
        start_index = jax.lax.select(is_single_digit, 1, 0)
        num_to_render = jax.lax.select(is_single_digit, 1, 2)
        render_x = jax.lax.select(is_single_digit, 49 + 8 // 2, 49)
        
        raster = self.jr.render_label_selective(raster, render_x, 5, score_digits, score_digit_masks, start_index, num_to_render, spacing=8)

        # Render black bar on the left side
        black_bar_mask = self.SHAPE_MASKS["black_bar"]
        raster = self.jr.render_at(raster, 0, 0, black_bar_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
