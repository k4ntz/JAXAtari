from jax._src.pjit import JitWrapped
import os
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
    FIRST_TRACK_CORNERS_X: chex.Array = jnp.array([20, 50, 80, 100]) #get actual values
    FIRST_TRACK_CORNERS_Y: chex.Array = jnp.array([20, 50, 80, 100]) #get actual values
    SECOND_TRACK_CORNERS_X: chex.Array = jnp.array([20, 50, 80, 100]) #get actual values
    SECOND_TRACK_CORNERS_Y: chex.Array = jnp.array([20, 50, 80, 100]) #get actual values



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
    road_index: chex.Array
    jump_cooldown: chex.Array
    is_jumping: chex.Array
    is_on_road: chex.Array
    player_car: Car




class UpNDownObservation(NamedTuple):
    player: EntityPosition
    enemies: jnp.ndarray
    score: jnp.ndarray

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
    def _car_past_corner(self, car: Car, state: UpNDownState) -> chex.Array:
        direction_change_A = jnp.logical_or(jnp.logical_and(car.speed > 0, car.position.y + car.speed > self.consts.FIRST_TRACK_CORNERS_Y[car.road_index+1]), jnp.logical_and(car.speed < 0, car.position.y + car.speed < self.consts.FIRST_TRACK_CORNERS_Y[car.road_index]))
        direction_change_B = jnp.logical_or(jnp.logical_and(car.speed > 0, car.position.y + car.speed > self.consts.SECOND_TRACK_CORNERS_Y[car.road_index+1]), jnp.logical_and(car.speed < 0, car.position.y + car.speed < self.consts.SECOND_TRACK_CORNERS_Y[car.road_index])),
           
        road_index_A = jax.lax.cond(jnp.logical_and(direction_change_A, car.speed > 0),
            lambda s: s + 1,
            lambda s: s,
            operand=car.road_index_A,
        )
        road_index_A = jax.lax.cond(jnp.logical_and(direction_change_A, car.speed < 0),
            lambda s: s - 1,
            lambda s: s,
            operand=car.road_index_A,
        )

        road_index_B = jax.lax.cond(jnp.logical_and(direction_change_B, car.speed > 0),
            lambda s: s + 1,
            lambda s: s,
            operand=car.road_index_B,
        )
        road_index_B = jax.lax.cond(jnp.logical_and(direction_change_B, car.speed < 0),
            lambda s: s - 1,
            lambda s: s,
            operand=car.road_index_B,
        )
        current_road_length_A = self.consts.FIRST_ROAD_LENGTH
        current_road_length_B = self.consts.SECOND_ROAD_LENGTH

        road_index_A = jax.lax.cond(road_index_A < 0,
            lambda s: current_road_length_A - 1,
            lambda s: s,
            operand=road_index_A,
        )

        road_index_A = jax.lax.cond(road_index_A >= current_road_length_A,
            lambda s: 0,
            lambda s: s,
            operand=road_index_A,
        )

        road_index_B = jax.lax.cond(road_index_B < 0,
            lambda s: current_road_length_B - 1,
            lambda s: s,
            operand=road_index_B,
        )

        road_index_B = jax.lax.cond(road_index_B >= current_road_length_B,
            lambda s: 0,
            lambda s: s,
            operand=road_index_B,
        )

        return road_index_A, road_index_B
    
    @partial(jax.jit, static_argnums=(0,))
    def _landing_in_water(self, state: UpNDownState, new_position_x: chex.Array, new_position_y: chex.Array) -> chex.Array:
        road_A_x = ((new_position_y - self.consts.FIRST_TRACK_CORNERS_Y[state.player_car.road_index_A]) / (self.consts.FIRST_TRACK_CORNERS_Y[state.player_car.road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_Y[state.player_car.road_index_A])) * (self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A]) + self.consts.FIRST_TRACK_CORNERS_X[state.player_car.road_index_A]
        road_B_x = ((new_position_y - self.consts.SECOND_TRACK_CORNERS_Y[state.player_car.road_index_B]) / (self.consts.SECOND_TRACK_CORNERS_Y[state.player_car.road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_Y[state.player_car.road_index_B])) * (self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B]) + self.consts.SECOND_TRACK_CORNERS_X[state.player_car.road_index_B]
        distance_to_road_A = jnp.abs(new_position_x - road_A_x)
        distance_to_road_B = jnp.abs(new_position_x - road_B_x)
        landing_in_Water = jnp.logical_and(distance_to_road_A > self.consts.LANDING_ZONE, distance_to_road_B > self.consts.LANDING_ZONE)
        between_roads = jnp.logical_and(new_position_x > jnp.minimum(road_A_x, road_B_x), new_position_x < jnp.maximum(road_A_x, road_B_x))
        return landing_in_Water, between_roads

    def _player_step(self, state: UpNDownState, action: chex.Array) -> UpNDownState:
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
        jump = jnp.logical_or(action == Action.FIRE, action == Action.UPFIRE, action == Action.DOWNFIRE)



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


        is_jumping = jnp.logical_or(jnp.logical_and(state.is_jumping, state.jump_cooldown > 0), jnp.logical_and(is_on_road, jnp.logical_and(player_speed > 0, state.jump_cooldown == 0)))
        jump_cooldown = jax.lax.cond(
            state.jump_cooldown > 0,
            lambda s: s - 1,
            lambda s: jnp.cond(jnp.logical_and(is_jumping),
                               lambda _: state.JUMP_FRAMES,
                               lambda _: 0, 
                               operand=None),
            operand=state.jump_cooldown,
        )




        ##check if player is on the the road
        is_on_road = ~state.is_jumping

        road_index_A, road_index_B = self._car_past_corner(state.player_car, state)

        direction_change = jax.lax.cond(
            jnp.logical_and(is_on_road, jnp.logical_or(jnp.logical_and(jnp.equal(road_index_A, state.player_car.road_index_A)) , state.player_car.current_road == 0), (jnp.logical_and(jnp.equal(road_index_B, state.player_car.road_index_B)) , state.player_car.current_road == 1) ),
            lambda s: False,
            lambda s: True,
            operand=None,
        )


        car_direction_x = jax.lax.cond(
            direction_change,
            lambda s: jax.lax.cond(state.player_car.current_road == 0,
                lambda s: self.consts.FIRST_TRACK_CORNERS_X[road_index_A+1] - self.consts.FIRST_TRACK_CORNERS_X[road_index_A],
                lambda s: self.consts.SECOND_TRACK_CORNERS_X[road_index_B+1] - self.consts.SECOND_TRACK_CORNERS_X[road_index_B],
                operand=None),
            lambda s: s,
            operand=state.player_car.direction_x,
        )
        
        is_landing = jnp.logical_and(state.jump_cooldown == 1, jump_cooldown == 0)

        ##calculate new position with speed (TODO: calculate better speed)
        player_y = state.player_car.position.y + player_speed
        player_x = state.player_car.position.x + player_speed * car_direction_x

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
        return UpNDownState(
            score=state.score,
            difficulty=state.difficulty,
            road_index=state.road_index,
            jump_cooldown=jump_cooldown,
            is_jumping=is_jumping,
            is_on_road=is_on_road,
            player_car=Car(
                position=EntityPosition(
                    x=player_x,
                    y=player_y,
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
        )

    def _score_and_reset(self, state: UpNDownState) -> UpNDownState:
        player_goal = state.ball_x < 4
        enemy_goal = state.ball_x > 156
        ball_reset = jnp.logical_or(enemy_goal, player_goal)

        player_score = jax.lax.cond(
            player_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.player_score,
        )
        enemy_score = jax.lax.cond(
            enemy_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.enemy_score,
        )

        current_values = (
            state.ball_x.astype(jnp.int32),
            state.ball_y.astype(jnp.int32),
            state.ball_vel_x.astype(jnp.int32),
            state.ball_vel_y.astype(jnp.int32),
        )
        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: self._reset_ball_after_goal((state, enemy_goal)),
            lambda x: x,
            operand=current_values,
        )

        step_counter = jax.lax.cond(
            ball_reset,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        enemy_y_final = jax.lax.cond(
            ball_reset,
            lambda s: self.consts.BALL_START_Y.astype(jnp.int32),
            lambda s: state.enemy_y.astype(jnp.int32),
            operand=None,
        )

        ball_x_final = jax.lax.cond(
            step_counter < 60,
            lambda s: self.consts.BALL_START_X.astype(jnp.int32),
            lambda s: s,
            operand=ball_x_final,
        )
        ball_y_final = jax.lax.cond(
            step_counter < 60,
            lambda s: self.consts.BALL_START_Y.astype(jnp.int32),
            lambda s: s,
            operand=ball_y_final,
        )

        return UpNDownState(
            player_y=state.player_y,
            player_speed=state.player_speed,
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            enemy_y=enemy_y_final,
            enemy_speed=state.enemy_speed,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=step_counter,
            acceleration_counter=state.acceleration_counter,
            buffer=state.buffer,
        )

    def reset(self, key=None) -> Tuple[UpNDownObservation, UpNDownState]:
        state = UpNDownState(
            player_y=jnp.array(96).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            ball_x=jnp.array(78).astype(jnp.int32),
            ball_y=jnp.array(115).astype(jnp.int32),
            enemy_y=jnp.array(115).astype(jnp.int32),
            enemy_speed=jnp.array(0.0).astype(jnp.int32),
            ball_vel_x=self.consts.BALL_SPEED[0].astype(jnp.int32),
            ball_vel_y=self.consts.BALL_SPEED[1].astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            buffer=jnp.array(96).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: UpNDownState, action: chex.Array) -> Tuple[UpNDownObservation, UpNDownState, float, bool, UpNDownInfo]:
        previous_state = state
        state = self._player_step(state, action)
        state = self._enemy_step(state)
        state = self._ball_step(state, action)
        state = self._score_and_reset(state)

        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: UpNDownState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: UpNDownState):
        player = EntityPosition(
            x=jnp.array(self.consts.PLAYER_X),
            y=state.player_y,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
        )

        enemy = EntityPosition(
            x=jnp.array(self.consts.ENEMY_X),
            y=state.enemy_y,
            width=jnp.array(self.consts.ENEMY_SIZE[0]),
            height=jnp.array(self.consts.ENEMY_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(self.consts.BALL_SIZE[0]),
            height=jnp.array(self.consts.BALL_SIZE[1]),
        )
        return UpNDownObservation(
            player=player,
            enemy=enemy,
            ball=ball,
            score_player=state.player_score,
            score_enemy=state.enemy_score,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: UpNDownObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten(),
               obs.enemy.x.flatten(),
               obs.enemy.y.flatten(),
               obs.enemy.height.flatten(),
               obs.enemy.width.flatten(),
               obs.ball.x.flatten(),
               obs.ball.y.flatten(),
               obs.ball.height.flatten(),
               obs.ball.width.flatten(),
               obs.score_player.flatten(),
               obs.score_enemy.flatten()
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
            "enemy": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "ball": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
            }),
            "score_player": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
            "score_enemy": spaces.Box(low=0, high=21, shape=(), dtype=jnp.int32),
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
        return UpNDownInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: UpNDownState, state: UpNDownState):
        return (state.player_score - state.enemy_score) - (
            previous_state.player_score - previous_state.enemy_score
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: UpNDownState) -> bool:
        return jnp.logical_or(
            jnp.greater_equal(state.player_score, 21),
            jnp.greater_equal(state.enemy_score, 21),
        )

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
        # 1. Create procedural assets for both walls
        wall_sprite_top = self._create_wall_sprite(self.consts.WALL_TOP_HEIGHT)
        wall_sprite_bottom = self._create_wall_sprite(self.consts.WALL_BOTTOM_HEIGHT)
        
        # 2. Update asset config to include both walls
        asset_config = self._get_asset_config(wall_sprite_top, wall_sprite_bottom)
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/UpNDown"

        # 3. Make a single call to the setup function
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(asset_config, sprite_path)

    def _create_wall_sprite(self, height: int) -> jnp.ndarray:
        """Procedurally creates an RGBA sprite for a wall of given height."""
        wall_color_rgba = (*self.consts.SCORE_COLOR, 255) # e.g., (236, 236, 236, 255)
        wall_shape = (height, self.consts.WIDTH, 4)
        wall_sprite = jnp.tile(jnp.array(wall_color_rgba, dtype=jnp.uint8), (*wall_shape[:2], 1))
        return wall_sprite

    def _get_asset_config(self, wall_sprite_top: jnp.ndarray, wall_sprite_bottom: jnp.ndarray) -> list:
        """Returns the declarative manifest of all assets for the game, including both wall sprites."""
        return [
            {'name': 'background', 'type': 'background', 'file': 'background.npy'},
            {'name': 'player', 'type': 'single', 'file': 'player.npy'},
            {'name': 'enemy', 'type': 'single', 'file': 'enemy.npy'},
            {'name': 'ball', 'type': 'single', 'file': 'ball.npy'},
            {'name': 'player_digits', 'type': 'digits', 'pattern': 'player_score_{}.npy'},
            {'name': 'enemy_digits', 'type': 'digits', 'pattern': 'enemy_score_{}.npy'},
            # Add the procedurally created sprites to the manifest
            {'name': 'wall_top', 'type': 'procedural', 'data': wall_sprite_top},
            {'name': 'wall_bottom', 'type': 'procedural', 'data': wall_sprite_bottom},
        ]

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = self.jr.create_object_raster(self.BACKGROUND)

        player_mask = self.SHAPE_MASKS["player"]
        raster = self.jr.render_at(raster, self.consts.PLAYER_X, state.player_y, player_mask)

        enemy_mask = self.SHAPE_MASKS["enemy"]
        raster = self.jr.render_at(raster, self.consts.ENEMY_X, state.enemy_y, enemy_mask)

        ball_mask = self.SHAPE_MASKS["ball"]
        raster = self.jr.render_at(raster, state.ball_x, state.ball_y, ball_mask)

        # --- Stamp Walls and Score (using the same color/ID) ---
        score_color_tuple = self.consts.SCORE_COLOR # (236, 236, 236)
        score_id = self.COLOR_TO_ID[score_color_tuple]

        # Draw walls (using separate sprites for top and bottom)
        raster = self.jr.render_at(raster, 0, self.consts.WALL_TOP_Y, self.SHAPE_MASKS["wall_top"])
        raster = self.jr.render_at(raster, 0, self.consts.WALL_BOTTOM_Y, self.SHAPE_MASKS["wall_bottom"])

        # Stamp Score using the label utility
        player_digits = self.jr.int_to_digits(state.player_score, max_digits=2)
        enemy_digits = self.jr.int_to_digits(state.enemy_score, max_digits=2)

        # Note: The logic for single/double digits is complex for a jitted function.
        player_digit_masks = self.SHAPE_MASKS["player_digits"] # Assumes single color
        enemy_digit_masks = self.SHAPE_MASKS["enemy_digits"] # Assumes single color

        is_player_single_digit = state.player_score < 10
        player_start_index = jax.lax.select(is_player_single_digit, 1, 0)
        player_num_to_render = jax.lax.select(is_player_single_digit, 1, 2)
        player_render_x = jax.lax.select(is_player_single_digit,
                                         120 + 16 // 2,
                                         120)

        raster = self.jr.render_label_selective(raster, player_render_x, 3, player_digits, player_digit_masks, player_start_index, player_num_to_render, spacing=16)
        
        is_enemy_single_digit = state.enemy_score < 10
        enemy_start_index = jax.lax.select(is_enemy_single_digit, 1, 0)
        enemy_num_to_render = jax.lax.select(is_enemy_single_digit, 1, 2)
        enemy_render_x = jax.lax.select(is_enemy_single_digit,
                                        10 + 16 // 2,
                                        10)

        raster = self.jr.render_label_selective(raster, enemy_render_x, 3, enemy_digits, enemy_digit_masks, enemy_start_index, enemy_num_to_render, spacing=16)

        return self.jr.render_from_palette(raster, self.PALETTE)