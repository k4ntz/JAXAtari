from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants for the game environment
SCALING_FACTOR = 3
WIDTH = 160
HEIGHT = 210

GROUND_LEVEL = 152

MISS_LIMIT = 7
SCORE_LIMIT = 7

DT = 1.0 / 60.0  # time step between frames
GRAVITY = 1  # TODO: Fine tune this value later

# Object constants
HUMAN_VEL = jnp.array([43, 31])  # TODO: Figure out how its handled in game logic
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
    7: "mph",
    8: "score",
    9: "misses",
    10: "step_counter",
    11: "buffer",
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
    mph: chex.Array
    score: chex.Array
    misses: chex.Array
    step_counter: chex.Array
    buffer: chex.Array


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
        state_human_x, state_human_y, state_human_y_vel, state_human_launched, state_water_tower_x, state_water_tower_y, state_mph, state_angle, step_counter
):
    #Only update the human position every 2 steps #TODO: Look into this again
    #should_update = jax.mod(step_counter, 2) == 0

    #Calculate the horizontal and vertical velocity
    x_vel = jnp.cos(state_angle) * state_mph

    y_vel = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda _: state_human_y_vel - GRAVITY * DT,  # Update the old velocity
        lambda _: jnp.sin(state_angle) * state_mph - GRAVITY * DT,  # Else, calculate the initial velocity
    )

    # Update the human position based on the velocity and the time step (DT)
    human_x = state_human_x + x_vel * DT
    human_y = state_human_y + y_vel * DT - 0.5 * GRAVITY * DT ** 2  # account for gravity

    # TODO: Add collision detection with the water tower

    # Set the launch status to True for subsequent steps
    human_launched = True

    return human_x, human_y, y_vel, human_launched


if __name__ == "__main__":
    hello = "World"