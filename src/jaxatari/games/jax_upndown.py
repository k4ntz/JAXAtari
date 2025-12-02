from jax._src.pjit import JitWrapped
import os
import math
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class UpNDownConstants(NamedTuple):
    FRAME_SKIP: int = 4
    DIFFICULTIES: chex.Array = jnp.array([0, 1, 2, 3, 4, 5])
    ACTION_REPEAT_PROBS: float = 0.25
    MAX_SPEED: int = 4
    JUMP_FRAMES: int = 10
    LANDING_ZONE: int = 15
    FIRST_ROAD_LENGTH: int = 4
    SECOND_ROAD_LENGTH: int = 4
    FIRST_TRACK_CORNERS_X: chex.Array = jnp.array([30, 75, 128, 75, 21, 75, 131, 111, 150, 95, 150, 115, 150, 108, 150, 115, 115, 75, 18, 67, 38, 38, 20, 64, 30]) #get actual values
    FIRST_TRACK_CORNERS_Y: chex.Array = jnp.array([0, -40, -98, -155, -203, -268, -327, -347, -382, -467, -525, -565, -597, -625, -670, -705, -738, -788, -838, -898, -925, -950, -972, -1000, -1033]) #get actual values
    SECOND_TRACK_CORNERS_X: chex.Array = FIRST_TRACK_CORNERS_X#jnp.array([20, 50]) #get actual values
    SECOND_TRACK_CORNERS_Y: chex.Array = FIRST_TRACK_CORNERS_Y#jnp.array([20, 50, ]) #get actual values
    PLAYER_SIZE: Tuple[int, int] = (4, 16)
    INITIAL_ROAD_POS_Y: int = 25 



# immutable state container
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class Car(NamedTuple):
    position: EntityPosition
    speed: chex.Array
    type: chex.Array
    current_road: chex.Array
    road_index_A: chex.Array
    road_index_B: chex.Array
    direction_x: chex.Array

class UpNDownState(NamedTuple):
    score: chex.Array
    difficulty: chex.Array
    jump_cooldown: chex.Array
    is_jumping: chex.Array
    is_on_road: chex.Array
    player_car: Car
    step_counter: chex.Array




class UpNDownObservation(NamedTuple):
    player: EntityPosition

class Collectible(NamedTuple):
    position: EntityPosition
    type: chex.Array
    value: chex.Array


class UpNDownInfo(NamedTuple):
    time: jnp.ndarray


class JaxUpNDown(JaxEnvironment[UpNDownState, UpNDownObservation, UpNDownInfo, UpNDownConstants]):
    def __init__(self, consts: UpNDownConstants = None, reward_funcs: list[callable]=None):
        consts = consts or UpNDownConstants()
        super().__init__(consts)
        self.renderer = UpNDownRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UPFIRE,
            Action.UP,
            Action.DOWN,
            Action.DOWNFIRE,
        ]
        self.obs_size = 3*4+1+1

    @partial(jax.jit, static_argnums=(0,))
    def _getSlopeAndB(self, state: UpNDownState) -> chex.Array:
        trackx, tracky, roadIndex = jax.lax.cond(
            state.player_car.current_road == 0,
            lambda s: (self.consts.FIRST_TRACK_CORNERS_X, self.consts.FIRST_TRACK_CORNERS_Y, state.player_car.road_index_A),
            lambda s: (self.consts.SECOND_TRACK_CORNERS_X, self.consts.SECOND_TRACK_CORNERS_Y, state.player_car.road_index_B),
            operand=None,)
        slope = jax.lax.cond(
            trackx[roadIndex+1] - trackx[roadIndex] != 0,
            lambda s: (tracky[roadIndex+1] - tracky[roadIndex]) / (trackx[roadIndex+1] - trackx[roadIndex]),
            lambda s: 300.0,
            operand=None,
        )
        b = tracky[roadIndex] - slope * trackx[roadIndex]
        return slope, b
    
    @partial(jax.jit, static_argnums=(0,))
    def _isOnLine(self, state: UpNDownState, new_position_x: chex.Array, new_position_y: chex.Array, player_speed: chex.Array) -> chex.Array:
        slope, b = self._getSlopeAndB(state)
        jax.debug.print("slope: {}, b: {}", slope, b)
        isOnLine = jnp.logical_or(jnp.logical_and(jnp.equal(slope, 300.0), jnp.equal(new_position_x, state.player_car.position.x)), jnp.less_equal(jnp.abs(jnp.round(jnp.subtract(new_position_y, slope * new_position_x + b))), player_speed))

        jax.debug.print("isOnLine: {}", jnp.subtract(new_position_y, slope * new_position_x + b))
        return isOnLine
    
    @partial(jax.jit, static_argnums=(0,))
    def _landing_in_water(self, state: UpNDownState, new_position_x: chex.Array, new_position_y: chex.Array) -> chex.Array:
        road_A_x = ((new_position_y - self.consts.FIRST_TRACK_CORNERS_Y[state.player_car.road_index_A]) / (self.consts.FIRST_TRACK_CORNERS_Y[state.player_car.road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_Y[state.player_car.road_index_A])) * (self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A]) + self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A]
        road_B_x = ((new_position_y - self.consts.SECOND_TRACK_CORNERS_Y[state.player_car.road_index_B]) / (self.consts.SECOND_TRACK_CORNERS_Y[state.player_car.road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_Y[state.player_car.road_index_B])) * (self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B]) + self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B]
        distance_to_road_A = jnp.abs(new_position_x - road_A_x)
        distance_to_road_B = jnp.abs(new_position_x - road_B_x)
        landing_in_Water = jnp.logical_and(distance_to_road_A > self.consts.LANDING_ZONE, distance_to_road_B > self.consts.LANDING_ZONE)
        between_roads = jnp.logical_and(new_position_x > jnp.minimum(road_A_x, road_B_x), new_position_x < jnp.maximum(road_A_x, road_B_x))
        return landing_in_Water, between_roads, road_A_x, road_B_x

    def _player_step(self, state: UpNDownState, action: chex.Array) -> UpNDownState:
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
        jump = jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.UPFIRE, action == Action.DOWNFIRE))



        player_speed = state.player_car.speed

        player_speed = jax.lax.cond(
            jnp.logical_and(state.player_car.speed < self.consts.MAX_SPEED, up),
            lambda s: s + 1,
            lambda s: s,
            operand=player_speed,
        )

        player_speed = jax.lax.cond(
            jnp.logical_and(state.player_car.speed > -self.consts.MAX_SPEED, down),
            lambda s: s - 1,
            lambda s: s,
            operand=player_speed,
        )
        dividers = jnp.array([0, 1, 2, 4, 8])
        speed_divider = dividers[jnp.abs(player_speed)]


        is_jumping = jnp.logical_or(jnp.logical_and(state.is_jumping, state.jump_cooldown > 0), jnp.logical_and(state.is_on_road, jnp.logical_and(player_speed >= 0, jnp.logical_and(state.jump_cooldown == 0, jump))))
        jump_cooldown = jax.lax.cond(
            state.jump_cooldown > 0,
            lambda s: s - 1,
            lambda s: jax.lax.cond(is_jumping,
                               lambda _: self.consts.JUMP_FRAMES,
                               lambda _: 0, 
                               operand=None),
            operand=state.jump_cooldown,
        )




        ##check if player is on the the road
        is_on_road = ~state.is_jumping

        '''direction_change = jax.lax.cond(
            jnp.logical_and(is_on_road, jnp.logical_or(jnp.logical_and(jnp.equal(road_index_A, state.player_car.road_index_A) , state.player_car.current_road == 0), (jnp.logical_and(jnp.equal(road_index_B, state.player_car.road_index_B) , state.player_car.current_road == 1)))) ,
            lambda s: False,
            lambda s: True,
            operand=None,
        )'''
        road_index_A = state.player_car.road_index_A
        road_index_B = state.player_car.road_index_B

        car_direction_x = jax.lax.cond(state.player_car.current_road == 0,
            lambda s: self.consts.FIRST_TRACK_CORNERS_X[road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[road_index_A],
            lambda s: self.consts.SECOND_TRACK_CORNERS_X[road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[road_index_B],
            operand=None),
        car_direction_x = jax.lax.cond(
            car_direction_x[0] > 0,
            lambda s: 1,
            lambda s: -1,
            operand=car_direction_x,
        )

        
        is_landing = jnp.logical_and(state.jump_cooldown == 1, jump_cooldown == 0)

        ##calculate new position with speed (TODO: calculate better speed)
        player_y = jax.lax.cond(
            state.step_counter % (16/ speed_divider) == 8 / speed_divider,
            lambda s: jax.lax.cond(
                is_jumping,
                lambda s: state.player_car.position.y + jax.lax.abs(player_speed) / player_speed * -1,
                lambda s: jax.lax.cond(
                    self._isOnLine(state, state.player_car.position.x, s + jax.lax.abs(player_speed) / player_speed * -1, 1),
                    lambda s: s + jax.lax.abs(player_speed) / player_speed * -1,
                    lambda s: jnp.array(s, float),
                    operand=state.player_car.position.y,
                ),
                operand=state.player_car.position.y),
            lambda s: jnp.array(s, float),
            operand=state.player_car.position.y,
        )
        player_x = jax.lax.cond(
            state.step_counter % (16/ speed_divider) == 0,
            lambda s: jax.lax.cond(
                is_jumping,
                lambda s: s + jax.lax.abs(player_speed) / player_speed * car_direction_x,
                lambda s: jax.lax.cond(
                    self._isOnLine(state, s + jax.lax.abs(player_speed) / player_speed * car_direction_x, player_y, 1),
                    lambda s: s + jax.lax.abs(player_speed) / player_speed * car_direction_x,
                    lambda s: jnp.array(s, float),
                    operand=state.player_car.position.x,
                ),
                operand=state.player_car.position.x),
            lambda s: jnp.array(s, float),
            operand=state.player_car.position.x,
        )

        ##if y not on mx +b then no move

        jax.debug.print("Player X: {}, Player Y: {}, car_direction_x: {}", player_x, player_y, car_direction_x)

        landing_in_Water, between_roads, road_A_x, road_B_x = self._landing_in_water(state, player_x, player_y)
        landing_in_Water = jnp.logical_and(is_landing, landing_in_Water)
        

        current_road = jax.lax.cond(
            landing_in_Water,
            lambda s: 2,
            lambda s: jax.lax.cond(
                is_on_road,
                lambda s: state.player_car.current_road,
                lambda s: jax.lax.cond(
                    jnp.abs(player_x - road_A_x) < jnp.abs(player_x - road_B_x),
                    lambda s: 0,
                    lambda s: 1,
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )
        
        road_index_A = jax.lax.cond(
            current_road == 2,
            lambda s: road_index_A,
            lambda s: jax.lax.cond(
                self.consts.FIRST_TRACK_CORNERS_Y[road_index_A] < player_y,
                lambda s: road_index_A - 1,
                lambda s: jax.lax.cond(
                    len(self.consts.FIRST_TRACK_CORNERS_Y) == road_index_A + 1,
                    lambda s: jax.lax.cond(
                        self.consts.FIRST_TRACK_CORNERS_Y[0] > player_y,
                        lambda s: 0,
                        lambda s: road_index_A,
                        operand=None,
                    ),
                    lambda s: jax.lax.cond(
                        self.consts.FIRST_TRACK_CORNERS_Y[road_index_A+1] > player_y,
                        lambda s: road_index_A + 1,
                        lambda s: road_index_A,
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

        road_index_B = jax.lax.cond(
            current_road == 2,
            lambda s: road_index_B,
            lambda s: jax.lax.cond(
                self.consts.SECOND_TRACK_CORNERS_Y[road_index_B] < player_y,
                lambda s: road_index_B - 1,
                lambda s: jax.lax.cond(
                    len(self.consts.SECOND_TRACK_CORNERS_Y) == road_index_B + 1,
                    lambda s: jax.lax.cond(
                        self.consts.SECOND_TRACK_CORNERS_Y[0] > player_y,
                        lambda s: 0,
                        lambda s: road_index_B,
                        operand=None,
                    ),
                    lambda s: jax.lax.cond(
                        self.consts.SECOND_TRACK_CORNERS_Y[road_index_B+1] > player_y,
                        lambda s: road_index_B + 1,
                        lambda s: road_index_B,
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )


        

        #jax.debug.print("Player X: {}, Player Y: {}, on road: {}, jumping: {}, speed: {}, road index A: {}, road index B: {}, current road: {}", player_x, player_y, is_on_road, is_jumping, player_speed, road_index_A, road_index_B, current_road)
        return UpNDownState(
            score=state.score,
            difficulty=state.difficulty,
            jump_cooldown=jump_cooldown,
            is_jumping=is_jumping,
            is_on_road=is_on_road,
            player_car=Car(
                position=EntityPosition(
                    x=player_x,
                    y=-((player_y * -1) % 1036),
                    width=state.player_car.position.width,
                    height=state.player_car.position.height,
                ),
                speed=player_speed,
                direction_x=car_direction_x,
                current_road=current_road,
                road_index_A=road_index_A,
                road_index_B=road_index_B,
                type=state.player_car.type,
            ),
            step_counter=state.step_counter + 1,
        )


    def reset(self, key=None) -> Tuple[UpNDownObservation, UpNDownState]:
        state = UpNDownState(
            score=0,
            difficulty=self.consts.DIFFICULTIES[0],
            jump_cooldown=0,
            is_jumping=False,
            is_on_road=True,
            player_car=Car(
                position=EntityPosition(
                    x=30,
                    y= 0,
                    width=self.consts.PLAYER_SIZE[0],
                    height=self.consts.PLAYER_SIZE[1],
                ),
                speed=0,
                direction_x=0,
                current_road=0,
                road_index_A=0,
                road_index_B=0,
                type=0,
            ),
            step_counter=jnp.array(0),
        )
        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: UpNDownState, action: chex.Array) -> Tuple[UpNDownObservation, UpNDownState, float, bool, UpNDownInfo]:
        previous_state = state
        state = self._player_step(state, action)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: UpNDownState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: UpNDownState):
        player = EntityPosition(
            x=jnp.array(state.player_car.position.x),
            y=jnp.array(state.player_car.position.y),
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )
        return UpNDownObservation(
            player=player,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: UpNDownObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten(),
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: UpNDownState, ) -> UpNDownInfo:
        return UpNDownInfo(time=1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: UpNDownState, state: UpNDownState):
        return state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: UpNDownState) -> bool:
        return jnp.logical_not(True)

class UpNDownRenderer(JAXGameRenderer):
    def __init__(self, consts: UpNDownConstants = None):
        super().__init__()
        self.consts = consts or UpNDownConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(210, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        background = self._createBackgroundSprite(self.config.game_dimensions)
        top_block = self._createBackgroundSprite((25, self.config.game_dimensions[1]))
        bottom_block = self._createBackgroundSprite((16, self.config.game_dimensions[1]))
        temp_pointer = self._createBackgroundSprite((1, 1))
        
        # 2. Update asset config to include both walls
        asset_config, road_files = self._get_asset_config(background, top_block, bottom_block, temp_pointer)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)
        self.road_sizes, self.complete_road_size = self._get_road_sprite_sizes(road_files)
        self.view_height = self.config.game_dimensions[0]
        # Precompute offsets so repeated road tiles can wrap seamlessly without gaps.
        road_cycle = max(1, self.complete_road_size)
        repeats = max(1, math.ceil(self.view_height / road_cycle) + 2)
        self._road_tile_offsets = jnp.arange(-repeats, repeats + 1, dtype=jnp.int32) * jnp.int32(self.complete_road_size)
        self._num_road_tiles = int(self._road_tile_offsets.shape[0])

    def _createBackgroundSprite(self, dimensions: Tuple[int, int]) -> jnp.ndarray:
        """Creates a procedural background sprite for the game."""
        height, width = dimensions
        color = (0, 0, 0, 255)  # RGBA for wall color
        shape = (height, width, 4)  # Height, Width, RGBA channels
        sprite = jnp.tile(jnp.array(color, dtype=jnp.uint8), (*shape[:2], 1))
        return sprite
    
    def _get_road_sprite_sizes(self, road_files: list[str]) -> list:
        """Returns the sizes of the road sprites limited to the configured files."""
        road_dir = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/roads"
        sizes = []
        for file in road_files:
            sprite_name = os.path.basename(file)
            sprite = jnp.load(f"{road_dir}/{sprite_name}")
            sizes.append(sprite.shape[0])
        complete_size = int(sum(sizes))
        jax.debug.print("Complete road size: {}", complete_size)
        return sizes, complete_size

    def _get_asset_config(self, backgroundSprite: jnp.ndarray, topBlockSprite: jnp.ndarray, bottomBlockSprite: jnp.ndarray, tempPointer: jnp.ndarray) -> tuple[list, list[str]]:
        """Returns the asset manifest and ordered road files."""
        road_dir = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/up_n_down/roads"
        road_files = sorted(
            file for file in os.listdir(road_dir)
            if file.endswith(".npy")
        )
        roads = [f"roads/{file}" for file in road_files]
        return [
            {'name': 'background', 'type': 'background', 'data': backgroundSprite},
            {'name': 'road', 'type': 'group', 'files': roads},
            {'name': 'player', 'type': 'single', 'file': 'player_car.npy'},
            {'name': 'wall_top', 'type': 'procedural', 'data': topBlockSprite},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': bottomBlockSprite},
            {'name': 'tempPointer', 'type': 'procedural', 'data': tempPointer},
        ], roads

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)
        road_diff = (-state.player_car.position.y) % self.complete_road_size

        # Vectorized road rendering: compute all Y offsets, stamp via vmap, fold overlays.
        road_masks = self.SHAPE_MASKS["road"]  # shape: (N, H, W)
        num_segments = road_masks.shape[0]

        sizes = jnp.array(self.road_sizes, dtype=jnp.int32)
        # Offsets: [0, cumsum(sizes[1:])]
        offsets = jnp.concatenate([
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(sizes[1:], axis=0)
        ], axis=0)

        base_y = jnp.asarray(self.consts.INITIAL_ROAD_POS_Y, dtype=jnp.int32)
        y_positions = base_y + (road_diff.astype(jnp.int32)) - offsets

        tile_offsets = self._road_tile_offsets
        tile_count = self._num_road_tiles
        tiled_y = (y_positions[None, :] + tile_offsets[:, None]).reshape(-1)
        tiled_masks = jnp.tile(road_masks, (tile_count, 1, 1))
        tiled_sizes = jnp.tile(sizes, tile_count)

        visible = jnp.logical_and(
            tiled_y < self.view_height,
            (tiled_y + tiled_sizes) > 0
        )

        empty_raster = jnp.full_like(self.BACKGROUND, self.jr.TRANSPARENT_ID)

        def stamp(y, mask, is_visible):
            return jax.lax.cond(
                is_visible,
                lambda _: self.jr.render_at_clipped(empty_raster, 10, y, mask),
                lambda _: empty_raster,
                operand=None,
            )

        overlays = jax.vmap(stamp)(tiled_y, tiled_masks, visible)

        total_segments = tile_count * num_segments

        def combine(i, acc):
            over = overlays[i]
            return jnp.where(over != self.jr.TRANSPARENT_ID, over, acc)

        raster = jax.lax.fori_loop(0, total_segments, combine, raster)

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, state.player_car.position.x, 105, player_mask)

        wall_top_mask = self.SHAPE_MASKS["wall_top"]
        raster = self.jr.render_at(raster, 0, 0, wall_top_mask)

        wall_bottom_mask = self.SHAPE_MASKS["wall_bottom"]
        raster = self.jr.render_at(raster, 0, 210 - wall_bottom_mask.shape[0], wall_bottom_mask)

        wall_bottom_mask = self.SHAPE_MASKS["tempPointer"]
        raster = self.jr.render_at(raster, 140, 25, wall_bottom_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)