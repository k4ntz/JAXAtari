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
WATER_SURFACE_WIDTH = 8

# Colors
PILLAR_COLOR = (0, 0, 0)
SKY_COLOR = (66, 72, 200)
GROUND_COLOR = (110, 156, 66)

# define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "human_x",
    1: "human_y",
    2: "human_speed",
    3: "water_tower_x",
    4: "water_tower_y",
    5: "angle",
    6: "mph",
    7: "score",
    8: "misses",
    9: "step_counter",
    10: "buffer",
}


# immutable state container
class HumanCannonballState(NamedTuple):
    human_x: chex.Array
    human_y: chex.Array
    human_speed: chex.Array
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
        state_human_x, state_human_y, state_water_tower_x, state_water_tower_y, state_mph, state_angle, step_counter
):
    should_update = jax.mod(step_counter, 2) == 0


if __name__ == "__main__":
    hello = "World"