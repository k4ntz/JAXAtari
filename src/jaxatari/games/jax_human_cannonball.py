import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces

from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvState, EnvObs, EnvInfo

# Constants for the game environment
SCALING_FACTOR = 3  # Not used currently
WIDTH = 160
HEIGHT = 210

GROUND_LEVEL = 152

MISS_LIMIT = 7
SCORE_LIMIT = 7

DT = 1.0 / 60.0  # time step between frames
GRAVITY = 1  # TODO: Fine tune this value later

# Object constants
MPH = jnp.array([43, 31, 25, 45, 37, 29, 30, 40])  # TODO: Figure out how its handled in game logic

ANGLE_START = 30
ANGLE_MAX = 80
ANGLE_MIN = 20

HUMAN_START_X = 86
HUMAN_START_Y = 130

CANNON_X = 68  # bottom left corner of the cannon
CANNON_Y = 151

WATER_TOWER_X = 132
WATER_TOWER_Y = 151

# Object sizes #TODO: Add the rest of the object sizes
HUMAN_SIZE = (4, 4)
CANNON_HIGH_SIZE = (16, 31)
CANNON_MID_SIZE = (17, 28)
CANNON_LOW_SIZE = (20, 23)

WATER_TOWER_SIZE = (10, 31)
WATER_TOWER_WALL_HEIGHT = 30

# Colors
PILLAR_COLOR = (0, 0, 0)
SKY_COLOR = (66, 72, 200)
GROUND_COLOR = (110, 156, 66)

# define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "human_x",
    1: "human_y",
    2: "human_y_vel",
    3: "human_launched",
    4: "water_tower_x",
    5: "water_tower_y",
    6: "angle",
    7: "mph_counter",
    8: "score",
    9: "misses",
    10: "step_counter",
}


# immutable state container
class HumanCannonballState(NamedTuple):
    human_x: chex.Array
    human_y: chex.Array
    human_y_vel: chex.Array
    human_launched: chex.Array
    water_tower_x: chex.Array
    water_tower_y: chex.Array
    angle: chex.Array
    mph_counter: chex.Array
    score: chex.Array
    misses: chex.Array
    step_counter: chex.Array


# Position of the human and the water tower
class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


# The state of the game
class HumanCannonballObservation(NamedTuple):
    human: EntityPosition
    water_tower: EntityPosition
    angle: jnp.ndarray
    mph: jnp.ndarray
    score: jnp.ndarray
    misses: jnp.ndarray


class HumanCannonballInfo(NamedTuple):
    time: jnp.ndarray
    all_rewards: chex.Array


# Step functions
@jax.jit
def human_step(
        state_human_x, state_human_y, state_human_y_vel, state_human_launched, state_water_tower_x, state_water_tower_y, state_mph_counter, state_angle
):
    #Only update the human position every 2 steps #TODO: Look into this again
    #should_update = jax.mod(step_counter, 2) == 0

    # Calculate the horizontal and vertical velocity
    x_vel = jnp.cos(state_angle) * MPH[state_mph_counter]

    y_vel = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda _: state_human_y_vel - GRAVITY * DT,  # Update the old velocity
        lambda _: jnp.sin(state_angle) * MPH[state_mph_counter] - GRAVITY * DT,  # Else, calculate the initial velocity
    )

    # Update the human position based on the velocity and the time step (DT)
    human_x = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda _: state_human_x + x_vel * DT,  # Update the old position
        lambda _: HUMAN_START_X,  # Else, set the initial position
    )

    human_y = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda _: state_human_y + y_vel * DT - 0.5 * GRAVITY * DT ** 2,  # Update the old position, account for gravity
        lambda _: HUMAN_START_Y,  # Else, set the initial position
    )

    # TODO: Add collision detection with the water tower

    # Set the launch status to True for subsequent steps
    human_launched = True

    return human_x, human_y, y_vel, human_launched

# Check if the player has scored
@jax.jit
def check_water_collision(
        state_human_x, state_human_y, state_water_tower_x, state_water_tower_y
):
    # Define bounding boxes for the human and water tower
    water_surface_x1 = state_water_tower_x + 1
    water_surface_y1 = state_water_tower_y + 1
    water_surface_x2 = water_surface_x1 + 8
    water_surface_y2 = water_surface_y1 + 1

    human_x1 = state_human_x
    human_y1 = state_human_y
    human_x2 = human_x1 + HUMAN_SIZE[0]
    human_y2 = human_y1 + HUMAN_SIZE[1]

    # AABB collision detection
    collision_x = jnp.logical_and(human_x1 < water_surface_x2, human_x2 > water_surface_x1)
    collision_y = jnp.logical_and(human_y1 < water_surface_y2, human_y2 > water_surface_y1)

    return jnp.logical_and(collision_x, collision_y)

# Check if the player has missed
@jax.jit
def check_ground_collision(
        state_human_y
):
    return state_human_y >= GROUND_LEVEL

@jax.jit
def angle_step(
        state_angle, state_human_launched, action
):
    new_angle = state_angle

    # Update the angle based on the action as long as the human is not launched
    new_angle = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_not(state_human_launched),
            action == Action.UP
        ),
        lambda s: s + 1,
        lambda s: s,
        operand=new_angle,
    )

    new_angle = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_not(state_human_launched),
            action == Action.DOWN
        ),
        lambda s: s - 1,
        lambda s: s,
        operand=new_angle,
    )

    # Ensure the angle is within the valid range
    new_angle = jnp.clip(new_angle, ANGLE_MIN, ANGLE_MAX)

    return new_angle

# Reset the round after a score or a miss
@jax.jit
def reset_round(
        state_mph_counter
):
    human_x = 0
    human_y = 0
    human_y_vel = 0
    human_launched = False
    water_tower_x = WATER_TOWER_X
    water_tower_y = WATER_TOWER_Y
    mph_counter = jnp.mod(state_mph_counter + 1, 8) #TODO: This is temporary, need to figure out how mph is handled in the game logic

    return human_x, human_y, human_y_vel, human_launched, water_tower_x, water_tower_y, mph_counter

class JaxHumanCannonball(JaxEnvironment[HumanCannonballState, HumanCannonballObservation, HumanCannonballInfo]):
    def __init__(
            self, reward_funcs: list[callable]=None
    ):
        super().__init__()
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        ]
        self.obs_size = 2*4+1+1+1+1                 #TODO: Implement this correctly?


    def reset(
            self, key=None
    ) -> Tuple[HumanCannonballObservation, HumanCannonballState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = HumanCannonballState(
            human_x = jnp.array(0).astype(jnp.int32),
            human_y = jnp.array(0).astype(jnp.int32),
            human_y_vel = jnp.array(0).astype(jnp.int32),
            human_launched = jnp.array(False),
            water_tower_x = jnp.array(WATER_TOWER_X).astype(jnp.int32),
            water_tower_y = jnp.array(WATER_TOWER_Y).astype(jnp.int32),
            angle = jnp.array(ANGLE_START).astype(jnp.int32),
            mph_counter = jnp.array(0).astype(jnp.int32),
            score = jnp.array(0).astype(jnp.int32),
            misses = jnp.array(0).astype(jnp.int32),
            step_counter = jnp.array(0).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: HumanCannonballState, action: chex.Array
    ) -> Tuple[HumanCannonballObservation, HumanCannonballState, float, bool, HumanCannonballInfo]:

        # Step 1: Update the angle of the cannon
        new_angle = angle_step(
            state.angle,
            state.human_launched,
            action
        )

        # Step 2: Update the position of the human projectile
        new_human_x, new_human_y, new_human_y_vel, human_launched = jax.lax.cond(
            jnp.logical_and(    # If human is in flight or the current action is FIRE
                jnp.logical_or(state.human_launched, action == Action.FIRE),
                jnp.mod(state.step_counter, 2) == 0,    # Only execute human_step on even steps (base implementation only moves the projectile every second tick)
            ),
            lambda _: human_step(   # Calculate the new position/velocity of the human via human_step
                state.human_x,
                state.human_y,
                state.human_y_vel,
                state.human_launched,
                state.water_tower_x,
                state.water_tower_y,
                state.mph_counter,
                state.angle
            ),
            lambda _: (             # Else, leave it unchanged
                state.human_x,
                state.human_y,
                state.human_y_vel,
                state.human_launched
            ),
            operand=None
        )

        # Step 3: Check if the player has scored
        new_score = jax.lax.cond(
            check_water_collision(
                new_human_x,
                new_human_y,
                state.water_tower_x,
                state.water_tower_y
            ),
            lambda _: state.score + 1,
            lambda _: state.score,
            operand=None
        )

        # Step 4: Check if the player has missed
        new_misses = jax.lax.cond(
            check_ground_collision(
                state.human_y
            ),
            lambda _: state.misses + 1,
            lambda _: state.misses,
            operand=None
        )

        # Step 5: Reset the round if there was a score or miss this step
        round_reset = jnp.logical_or(
            state.score < new_score,
            state.misses < new_misses
        )

        current_values = (
            new_human_x,
            new_human_y,
            new_human_y_vel,
            human_launched,
            WATER_TOWER_X,
            WATER_TOWER_Y,
            state.mph_counter,
        )

        (new_human_x, new_human_y, new_human_y_vel, human_launched, new_water_tower_x,
         new_water_tower_y, new_mph_counter) = jax.lax.cond(
            round_reset,
            lambda _: reset_round(state.mph_counter),
            lambda x: x,
            operand=current_values
        )

        # Step 6: Create the new state
        new_state = HumanCannonballState(
            human_x=new_human_x,
            human_y=new_human_y,
            human_y_vel=new_human_y_vel,
            human_launched=human_launched,
            water_tower_x=new_water_tower_x,
            water_tower_y=new_water_tower_y,
            angle=new_angle,
            mph_counter=new_mph_counter,
            score=new_score,
            misses=new_misses,
            step_counter=state.step_counter + 1,
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HumanCannonballState):
        # Create human projectile
        human_cannonball = EntityPosition(
            x=state.human_x,
            y=state.human_y,
            width=jnp.array(HUMAN_SIZE[0]),
            height=jnp.array(HUMAN_SIZE[1]),
        )

        # Create water tower
        human_cannonball = EntityPosition(
            x=state.water_tower_x,
            y=state.water_tower_y,
            width=jnp.array(WATER_TOWER_SIZE[0]),
            height=jnp.array(WATER_TOWER_SIZE[1]),
        )

        return HumanCannonballObservation(
            human=human_cannonball,
            water_tower=human_cannonball,
            angle=state.angle,
            mph=MPH[state.mph_counter],
            score=state.score,
            misses=state.misses
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: HumanCannonballObservation) -> jnp.ndarray:
        return jnp.concatenate([
            obs.human.x.flatten(),
            obs.human.y.flatten(),
            obs.human.width.flatten(),
            obs.human.height.flatten(),
            obs.water_tower.x.flatten(),
            obs.water_tower.y.flatten(),
            obs.water_tower.width.flatten(),
            obs.water_tower.height.flatten(),
            obs.angle.flatten(),
            obs.mph.flatten(),
            obs.score.flatten(),
            obs.misses.flatten()
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=None,
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: HumanCannonballState, all_rewards: chex.Array) -> HumanCannonballInfo:
        return HumanCannonballInfo(time=state.step_counter, all_rewards=all_rewards)

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: HumanCannonballState, state: HumanCannonballState):
        return (state.score - state.misses) - (
                previous_state.score - previous_state.misses
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(self, previous_state: HumanCannonballState, state: HumanCannonballState):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit,  static_argnums=(0,))
    def _get_done(
            self, state: HumanCannonballState
    ) -> bool:
        return jnp.logical_or(
            state.misses >= MISS_LIMIT,
            state.score >= SCORE_LIMIT
        )


if __name__ == "__main__":
    hello = "World"