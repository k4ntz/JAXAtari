import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
from gymnax.environments import spaces
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvState, EnvObs, EnvInfo

# Constants for the game environment
#SCALING_FACTOR = 3  # Not used currently
WIDTH = 160
HEIGHT = 210

GROUND_LEVEL = 152

# Game constants
MISS_LIMIT = 7
SCORE_LIMIT = 7

# Game physics constants    # TODO: These values may need some tweaking
DT = 1.0 / 15.0  # time step between frames
GRAVITY = 9.8
WALL_RESTITUTION = 0.3  # Coefficient of restitution for the wall collision

# MPH values (Max number of rounds per game is 13 (either score or miss pass the limit))
MPH = jnp.array([43, 28, 38, 45, 25, 30, 40, 20, 34, 41, 36, 35, 90])  # TODO: Figure out how its handled in game logic

# Angle constants
ANGLE_START = 30
ANGLE_MAX = 80
ANGLE_MIN = 20

# The cannon aims low if angle <37, medium if 37 <= angle < 59, and high if angle >= 59
ANGLE_LOW_THRESHOLD = 37
ANGLE_HIGH_THRESHOLD = 59

ANGLE_BUFFER = 16   # Only update the angle if the action is held for this many steps

# Starting positions of the human
HUMAN_START_LOW = (84.0, 128.0)
HUMAN_START_MED = (84.0, 121.0)
HUMAN_START_HIGH = (80.0, 118.0)

# Top left corner of the low-aiming cannon
CANNON_X = 68
CANNON_Y = 130

# Bottom left corner of the water tower
WATER_TOWER_X = 132
WATER_TOWER_Y = 151

# Object hit-box sizes
HUMAN_SIZE = (4, 4)
WATER_SIZE = (8, 3)

# Water tower dimensions
WATER_TOWER_WIDTH = 10
WATER_TOWER_HUMAN_HEIGHT = 35
WATER_TOWER_WALL_HEIGHT = 30

# Water tower movement constraints
WATER_TOWER_X_MIN = 109
WATER_TOWER_X_MAX = 160 - WATER_TOWER_WIDTH + 1

# Position of the digits
SCORE_X = 31
MISS_X = 111
SCORE_MISS_Y = 5

MPH_ANGLE_X = (95, 111)
MPH_Y = 20
ANGLE_Y = 35

# Define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "human_x",
    1: "human_y",
    2: "human_x_vel",
    3: "human_y_vel",
    4: "human_launched",
    5: "water_tower_x",
    6: "water_tower_y", # TODO: This is a constant, do i pass it for the agent or can it be removed?
    7: "tower_wall_hit",
    8: "angle",
    9: "angle_counter",
    10: "mph_counter",
    11: "score",
    12: "misses",
    13: "step_counter",
}


# Immutable state container
class HumanCannonballState(NamedTuple):
    human_x: chex.Array
    human_y: chex.Array
    human_x_vel: chex.Array
    human_y_vel: chex.Array
    human_launched: chex.Array
    water_tower_x: chex.Array
    water_tower_y: chex.Array
    tower_wall_hit: chex.Array
    angle: chex.Array
    angle_counter: chex.Array
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

# Determines the starting position of the human based on the angle
@jax.jit
def get_human_start(
        state_angle
):
    start_x, start_y = jax.lax.cond(
        state_angle < ANGLE_HIGH_THRESHOLD,  # If angle under HIGH_THRESHOLD
        lambda _: HUMAN_START_MED,  # Get medium start pos
        lambda _: HUMAN_START_HIGH,  # Else, get high start pos
        operand=None
    )

    start_x, start_y = jax.lax.cond(
        state_angle < ANGLE_LOW_THRESHOLD,  # If angle under LOW_THRESHOLD
        lambda _: HUMAN_START_LOW,  # Get low start pos
        lambda _: (start_x, start_y),  # Else, leave unchanged
        operand=None
    )

    return start_x, start_y

# Step functions

# Update the human projectile position and velocity
@jax.jit
def human_step(
        state_human_x, state_human_y, state_human_x_vel, state_human_y_vel, state_human_launched,
        state_water_tower_x, state_water_tower_y, state_mph_counter, state_angle
):
    mph_speed = MPH[state_mph_counter]
    rad_angle = jnp.deg2rad(state_angle)
    t = DT
    HORIZONTAL_SPEED_SCALE = 0.7    # Scale to compress the flying arc

    # 1. Compute candidate horizontal and vertical velocities
    x_vel = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda _: state_human_x_vel,  # Keep the old velocity
        lambda _: jnp.cos(rad_angle) * mph_speed * HORIZONTAL_SPEED_SCALE,  # Else, calculate the initial velocity
        operand=None
    )

    y_vel = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda _: state_human_y_vel + GRAVITY * t,  # Update the old velocity
        lambda _: -jnp.sin(rad_angle) * mph_speed,  # Else, calculate the initial velocity
        operand=None
    )

    # 2. Compute candidate new positions
    human_x = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda x: x + x_vel * t,  # Update the old position
        lambda x: get_human_start(state_angle)[0],  # Else, set the initial position, depending on the angle
        operand=state_human_x
    )

    human_y = jax.lax.cond(
        state_human_launched,  # If human is already launched
        lambda y: y + y_vel * t + 0.5 * GRAVITY * t**2,  # Update the old position, account for gravity
        lambda y: get_human_start(state_angle)[1],  # Else, set the initial position, depending on the angle
        operand=state_human_y
    )

    # 3. Detect collision with tower wall
    coll = check_water_tower_wall_collision(
        human_x, human_y, state_water_tower_x
    )

    # 4. Reflect and dampen the velocity if there is a collision
    x_vel = jax.lax.cond(
        coll,
        lambda x: -x * WALL_RESTITUTION,
        lambda x: x,
        operand=x_vel
    )

    # 5. Clamp x so human sits just left of the wall
    human_x = jax.lax.cond(
        coll,
        lambda _: jnp.array(state_water_tower_x - HUMAN_SIZE[0], dtype=state_human_x.dtype),
        lambda x: x,
        operand=human_x
    )

    # 6. Set the launch status to True for subsequent steps
    human_launched = True

    # 7. Set collision status to True for steps after the wall hit
    coll = jax.lax.cond(
        jnp.logical_or( # If the collision happened this step or some step before (human bounced off the wall)
            coll,
            x_vel <= 0.0
        ),
        lambda _: True,  # Set collision status to True
        lambda _: False,  # Else, set it to False
        operand=None
    )

    return human_x, human_y, x_vel, y_vel, human_launched, coll

# Check if the player has scored
@jax.jit
def check_water_collision(
        state_human_x, state_human_y, state_water_tower_x, state_water_tower_y
):
    # Define bounding boxes for the human and water tower
    water_surface_x1 = state_water_tower_x + 1
    water_surface_y1 = state_water_tower_y - WATER_TOWER_WALL_HEIGHT
    water_surface_x2 = water_surface_x1 + WATER_SIZE[0]
    water_surface_y2 = water_surface_y1 + WATER_SIZE[1]

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
    return state_human_y + HUMAN_SIZE[1] >= GROUND_LEVEL

# Check if the human has hit the water tower wall
@jax.jit
def check_water_tower_wall_collision(
        state_human_x, state_human_y, state_water_tower_x
):
    # Define bounding boxes for the water tower wall and the human
    wall_x1 = state_water_tower_x
    wall_y1 = WATER_TOWER_Y - WATER_TOWER_WALL_HEIGHT
    wall_x2 = wall_x1 + 1
    wall_y2 = wall_y1 + WATER_TOWER_WALL_HEIGHT

    human_x1 = state_human_x + HUMAN_SIZE[0] / 2    # Only check for front half of the human
    human_y1 = state_human_y
    human_x2 = human_x1 + HUMAN_SIZE[0] / 2
    human_y2 = human_y1 + HUMAN_SIZE[1]

    # AABB collision detection
    collision_x = jnp.logical_and(human_x1 < wall_x2, human_x2 > wall_x1)
    collision_y = jnp.logical_and(human_y1 < wall_y2, human_y2 > wall_y1)

    return jnp.logical_and(collision_x, collision_y)

# Determines the new angle of the cannon based on the action
@jax.jit
def angle_step(
        state_angle, state_human_launched, angle_counter, action
):
    new_angle = state_angle

    # Update the angle based on the action as long as the human is not launched
    new_angle = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_and(    # If the human is not launched and the action 'UP' has been held for ANGLE_BUFFER steps
                jnp.logical_not(state_human_launched),
                action == Action.UP
            ),
            angle_counter >= ANGLE_BUFFER
        ),
        lambda s: s + 1,    # Increment the angle
        lambda s: s,        # Else, leave it unchanged
        operand=new_angle,
    )

    new_angle = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_and(  # If the human is not launched and the action 'DOWN' has been held for ANGLE_BUFFER steps
                jnp.logical_not(state_human_launched),
                action == Action.DOWN
            ),
            angle_counter >= ANGLE_BUFFER
        ),
        lambda s: s - 1,  # Decrement the angle
        lambda s: s,  # Else, leave it unchanged
        operand=new_angle,
    )

    # Ensure the angle is within the valid range
    new_angle = jnp.clip(new_angle, ANGLE_MIN, ANGLE_MAX)

    new_angle_counter = jax.lax.cond(
        angle_counter >= ANGLE_BUFFER, # If the angle has been updated
        lambda _: 0,  # Reset the angle counter
        lambda s: s,  # Else, leave it unchanged
        operand=angle_counter,
    )

    return new_angle, new_angle_counter

# Determines the new position of the water tower based on the action
@jax.jit
def water_tower_step(
        state_water_tower_x, state_tower_wall_hit, state_human_launched, action
):
    new_x = state_water_tower_x

    # Update the position based on the action as long as the human is launched/ in flight
    new_x = jax.lax.cond(
        jnp.logical_and(    # If the human is launched and the action 'LEFT' is pressed
            state_human_launched,
            action == Action.LEFT

        ),
        lambda s: s - 1,  # Move the water tower to the left
        lambda s: s,  # Else, leave it unchanged
        operand=new_x,
    )

    new_x = jax.lax.cond(
        jnp.logical_and(  # If the human is launched and the action 'RIGHT' is pressed
            state_human_launched,
            action == Action.RIGHT

        ),
        lambda s: s + 1,  # Move the water tower to the right
        lambda s: s,  # Else, leave it unchanged
        operand=new_x,
    )

    # Ensure the position is within the valid range
    new_x = jnp.clip(new_x, WATER_TOWER_X_MIN, WATER_TOWER_X_MAX)

    return new_x

# Reset the round after a score or a miss
@jax.jit
def reset_round(
        state_mph_counter
):
    human_x = 0.0
    human_y = 0.0
    human_x_vel = 0.0
    human_y_vel = 0.0
    human_launched = False
    water_tower_x = WATER_TOWER_X
    water_tower_y = WATER_TOWER_Y
    tower_wall_hit = False
    mph_counter = jnp.mod(state_mph_counter + 1, 8) #TODO: This is temporary, need to figure out how mph is handled in the game logic

    return human_x, human_y, human_x_vel, human_y_vel, human_launched, water_tower_x, water_tower_y, tower_wall_hit, mph_counter

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
        self.obs_size = 2*4+1+1+1+1 #TODO: Implement this correctly?


    def reset(
            self, key=None
    ) -> Tuple[HumanCannonballObservation, HumanCannonballState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = HumanCannonballState(
            human_x = jnp.array(0).astype(jnp.float32),
            human_y = jnp.array(0).astype(jnp.float32),
            human_x_vel = jnp.array(0).astype(jnp.float32),
            human_y_vel = jnp.array(0).astype(jnp.float32),
            human_launched = jnp.array(False),
            water_tower_x = jnp.array(WATER_TOWER_X).astype(jnp.int32),
            water_tower_y = jnp.array(WATER_TOWER_Y).astype(jnp.int32),
            tower_wall_hit = jnp.array(False),
            angle = jnp.array(ANGLE_START).astype(jnp.int32),
            angle_counter = jnp.array(0).astype(jnp.int32),
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
        new_angle_counter = jax.lax.cond(
            jnp.logical_or(  # If the action is UP or DOWN
                action == Action.UP,
                action == Action.DOWN
            ),
            lambda s: s + 1,  # Increment the angle counter
            lambda _: 0,  # Else, reset it
            operand=state.angle_counter
        )

        new_angle, new_angle_counter = angle_step(
            state.angle,
            state.human_launched,
            new_angle_counter,
            action
        )

        # Step 2: Update the position of the human projectile
        new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, tower_wall_hit = jax.lax.cond(
            jnp.logical_and(    # If human is in flight or the current action is FIRE
                jnp.logical_or(state.human_launched, action == Action.FIRE),
                jnp.mod(state.step_counter, 2) == 0,    # Only execute human_step on even steps (base implementation only moves the projectile every second tick)
            ),
            lambda _: human_step(   # Calculate the new position/velocity of the human via human_step
                state.human_x,
                state.human_y,
                state.human_x_vel,
                state.human_y_vel,
                state.human_launched,
                state.water_tower_x,
                state.water_tower_y,
                state.mph_counter,
                state.angle,
            ),
            lambda _: (             # Else, leave it unchanged
                state.human_x,
                state.human_y,
                state.human_x_vel,
                state.human_y_vel,
                state.human_launched,
                state.tower_wall_hit
            ),
            operand=None
        )

        # Step 3: Update the water tower position
        new_water_tower_x = jax.lax.cond(
            jnp.logical_and(    # Only execute if the human has not hit the tower wall
                jnp.logical_not(tower_wall_hit),
                #TODO: The game handles this with a separate counter (like for the angle), is this okay or should i change it (very minor difference)?
                jnp.mod(state.step_counter, 8) == 0 # Only execute water_step every 8 steps (base implementation only moves the projectile every eighth tick)
            ),
            lambda _: water_tower_step(  # Calculate the new position of the water tower
                state.water_tower_x,
                tower_wall_hit,
                state.human_launched,
                action
            ),
            lambda _: state.water_tower_x,  # Else, leave it unchanged
            operand=None
        )

        # Step 4: Check if the player has scored
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

        # Step 5: Check if the player has missed
        new_misses = jax.lax.cond(
            check_ground_collision(
                state.human_y
            ),
            lambda _: state.misses + 1,
            lambda _: state.misses,
            operand=None
        )

        # Step 6: Reset the round if there was a score or miss this step
        round_reset = jnp.logical_or(
            state.score < new_score,
            state.misses < new_misses
        )

        current_values = (
            new_human_x,
            new_human_y,
            new_human_x_vel,
            new_human_y_vel,
            human_launched,
            new_water_tower_x,
            WATER_TOWER_Y,
            tower_wall_hit,
            state.mph_counter,
        )

        (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, new_water_tower_x,
         new_water_tower_y, tower_wall_hit, new_mph_counter) = jax.lax.cond(
            round_reset,
            lambda _: reset_round(state.mph_counter),
            lambda x: x,
            operand=current_values
        )

        # Step 7: Create the new state
        new_state = HumanCannonballState(
            human_x=new_human_x,
            human_y=new_human_y,
            human_x_vel=new_human_x_vel,
            human_y_vel=new_human_y_vel,
            human_launched=human_launched,
            water_tower_x=new_water_tower_x,
            water_tower_y=new_water_tower_y,
            tower_wall_hit=tower_wall_hit,
            angle=new_angle,
            angle_counter=new_angle_counter,
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
            width=jnp.array(WATER_TOWER_WIDTH),
            height=jnp.array(WATER_TOWER_WALL_HEIGHT - 1),
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


def load_sprites():
    """Load all sprites required for HumanCannonball rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load the sprites
    bg = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/background.npy"))
    cannon_high = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/cannon_high_aim.npy"))
    cannon_med = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/cannon_medium_aim.npy"))
    cannon_low = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/cannon_low_aim.npy"))
    human_up = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_up.npy"))
    human_straight = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_straight.npy"))
    human_down = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_down.npy"))
    human_ground = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_ground.npy"))
    water_tower = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/water_tower.npy"))
    water_tower_human1 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/water_tower_human1.npy"))
    water_tower_human2 = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/water_tower_human2.npy"))

    # Pad cannon sprites to match each other
    cannon_sprites = aj.pad_to_match([cannon_high, cannon_med, cannon_low])

    # Pad human sprites to match each other
    human_sprites = aj.pad_to_match([human_up, human_straight, human_down, human_ground])

    # Pad water tower sprites to match each other
    water_tower_sprites = aj.pad_to_match([water_tower, water_tower_human1, water_tower_human2])

    # Background sprite
    SPRITE_BG = jnp.expand_dims(bg, axis=0)

    # Cannon sprites
    SPRITE_CANNON = jnp.stack(cannon_sprites, axis=0)

    # Human sprites
    SPRITE_HUMAN = jnp.stack(human_sprites, axis=0)

    # Water tower sprites
    SPRITE_WATER_TOWER = jnp.stack(water_tower_sprites, axis=0)

    # Digits sprites
    DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/human_cannonball/digits/score_{}.npy"))

    return (
        SPRITE_BG,
        SPRITE_CANNON,
        SPRITE_HUMAN,
        SPRITE_WATER_TOWER,
        DIGITS
    )


class HumanCannonballRenderer(AtraJaxisRenderer):
    """JAX-based HumanCannonball game renderer, optimized with JIT compilation."""

    def __init__(self):
        (
            self.SPRITE_BG,
            self.SPRITE_CANNON,
            self.SPRITE_HUMAN,
            self.SPRITE_WATER_TOWER,
            self.DIGITS,
        ) = load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A HumanCannonballState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # Render the background
        frame_bg = aj.get_sprite_frame(self.SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        # Render the cannon
        # 1. Determine which sprite to load
        cannon_sprite_idx, cannon_offset = jax.lax.cond(
            state.angle < ANGLE_HIGH_THRESHOLD, # If angle under HIGH_THRESHOLD
            lambda _: (1, 5),    # Render med sprite
            lambda _: (0, 8),    # Else, render high sprite
            operand=None
        )

        cannon_sprite_idx, cannon_offset = jax.lax.cond(
            state.angle < ANGLE_LOW_THRESHOLD,  # If angle under LOW_THRESHOLD
            lambda _: (2, 0),  # Render low sprite
            lambda _: (cannon_sprite_idx, cannon_offset),  # Else, keep the sprite
            operand=None
        )

        # 2. Render the cannon sprite
        frame_cannon = aj.get_sprite_frame(self.SPRITE_CANNON, cannon_sprite_idx)
        raster = aj.render_at(raster, CANNON_X, CANNON_Y - cannon_offset, frame_cannon)

        # Render the human
        # 1. Determine which sprite to load
        flying_angle_rad = jnp.arctan2(-state.human_y_vel, state.human_x_vel)
        flying_angle = jnp.rad2deg(flying_angle_rad)

        FLYING_ANGLE_THRESHOLD = 45

        human_sprite_idx, human_offset = jax.lax.cond(
            flying_angle > FLYING_ANGLE_THRESHOLD,  # If ascension angle is > 45 degrees
            lambda _: (0, 3),   # Use up sprite
            lambda _: (1, 0),   # Else, use straight sprite
            operand=None
        )

        human_sprite_idx, human_offset = jax.lax.cond(
            jnp.logical_or( # If ascension angle is < -45 degrees or the tower has been hit
                flying_angle < -FLYING_ANGLE_THRESHOLD,
                state.tower_wall_hit
            ),
            lambda _: (2, 2),  # Use down sprite
            lambda _: (human_sprite_idx, human_offset),  # Else, keep the sprite
            operand=None
        )

        # 2. Render the human sprite
        frame_human = aj.get_sprite_frame(self.SPRITE_HUMAN, human_sprite_idx)
        raster = jax.lax.cond(  # Only render when launched
            state.human_launched,
            lambda r: aj.render_at(r, state.human_x, state.human_y - human_offset, frame_human),
            lambda r: r,
            operand=raster
        )

        # Render the water tower
        frame_water_tower = aj.get_sprite_frame(self.SPRITE_WATER_TOWER, 0)     #TODO: For now, no score animation
        raster = aj.render_at(raster, state.water_tower_x, state.water_tower_y - WATER_TOWER_WALL_HEIGHT + 1, frame_water_tower)

        # Get the score and misses
        score_digits = aj.int_to_digits(state.score, max_digits=1)
        misses_digits = aj.int_to_digits(state.misses, max_digits=1)

        # Get the mph and angle
        mph_digits = aj.int_to_digits(MPH[state.mph_counter], max_digits=2)
        angle_digits = aj.int_to_digits(state.angle, max_digits=2)

        # Render the score
        raster = aj.render_label_selective(raster, SCORE_X, SCORE_MISS_Y, score_digits, self.DIGITS,
                                           0, 1)

        # Render the misses
        raster = aj.render_label_selective(raster, MISS_X, SCORE_MISS_Y, misses_digits, self.DIGITS,
                                           0, 1)

        # Render the mph
        raster = aj.render_label_selective(raster, MPH_ANGLE_X[0], MPH_Y, mph_digits, self.DIGITS,
                                           0, 2, spacing= MPH_ANGLE_X[1] - MPH_ANGLE_X[0])

        # Render the angle
        raster = aj.render_label_selective(raster, MPH_ANGLE_X[0], ANGLE_Y, angle_digits, self.DIGITS,
                                           0, 2, spacing=MPH_ANGLE_X[1] - MPH_ANGLE_X[0])

        return raster


if __name__ == "__main__":
    hello = "World"