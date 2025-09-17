import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
from gymnax.environments import spaces
from numpy.ma.core import logical_not

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

# Constants for the game environment
class HumanCannonballConstants(NamedTuple):
    #SCALING_FACTOR = 3  # Not used currently
    WIDTH: int = 160
    HEIGHT: int = 210

    GROUND_LEVEL: int = 152

    # Game constants
    MISS_LIMIT: int = 7
    SCORE_LIMIT: int = 7

    # Game physics constants    # TODO: These values may need some tweaking
    DT: float = 1.0 / 15.0  # time step between frames
    GRAVITY: float = 9.8
    WALL_RESTITUTION: float = 0.3  # Coefficient of restitution for the wall collision

    # MPH constraints
    MPH_MIN: int = 28
    MPH_MAX: int = 45

    # Angle constants
    ANGLE_START: int = 30
    ANGLE_MAX: int = 80
    ANGLE_MIN: int = 20

    # The cannon aims low if angle <37, medium if 37 <= angle < 59, and high if angle >= 59
    ANGLE_LOW_THRESHOLD: int = 37
    ANGLE_HIGH_THRESHOLD: int = 59

    ANGLE_BUFFER: int = 16   # Only update the angle if the action is held for this many steps

    # Starting positions of the human
    HUMAN_START_LOW: Tuple[float, float] = (84.0, 128.0)
    HUMAN_START_MED: Tuple[float, float] = (84.0, 121.0)
    HUMAN_START_HIGH: Tuple[float, float] = (80.0, 118.0)

    # Top left corner of the low-aiming cannon
    CANNON_X: int = 68
    CANNON_Y: int = 130

    # Bottom left corner of the water tower
    WATER_TOWER_X: int = 132
    WATER_TOWER_Y: int = 151

    # Object hit-box sizes
    HUMAN_SIZE: Tuple[int,int] = (4, 4)
    WATER_SIZE: Tuple[int,int] = (8, 3)

    # Water tower dimensions
    WATER_TOWER_WIDTH: int = 10
    WATER_TOWER_HUMAN_HEIGHT: int = 35
    WATER_TOWER_WALL_HEIGHT: int = 30

    # Water tower movement constraints
    WATER_TOWER_X_MIN: int = 109
    WATER_TOWER_X_MAX: int = 160 - WATER_TOWER_WIDTH + 1

    # Position of the digits
    SCORE_X: int = 31
    MISS_X: int = 111
    SCORE_MISS_Y: int = 5

    MPH_ANGLE_X: Tuple[int, int] = (95, 111)
    MPH_Y: int = 20
    ANGLE_Y: int = 35

    # Animation constants
    ANIMATION_MISS_LENGTH: int = 124
    ANIMATION_HIT_LENGTH: int = 96

# Define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "human_x",
    1: "human_y",
    2: "human_x_vel",
    3: "human_y_vel",
    4: "human_launched",
    5: "water_tower_x",
    6: "mph_values",
    7: "tower_wall_hit",
    8: "angle",
    9: "angle_counter",
    10: "score",
    11: "misses",
    12: "step_counter",
    13: "rng_key",
    14: "animation_running",
    15: "animation_counter"
}


# Immutable state container
class HumanCannonballState(NamedTuple):
    human_x: chex.Array
    human_y: chex.Array
    human_x_vel: chex.Array
    human_y_vel: chex.Array
    human_launched: chex.Array
    water_tower_x: chex.Array
    mph_values: chex.Array
    tower_wall_hit: chex.Array
    angle: chex.Array
    angle_counter: chex.Array
    score: chex.Array
    misses: chex.Array
    step_counter: chex.Array
    rng_key: chex.PRNGKey
    animation_running: chex.Array
    animation_counter: chex.Array


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

def load_sprites():
    """Load all sprites required for HumanCannonball rendering."""
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load the sprites
    bg = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/background.npy"))
    cannon_high = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/cannon_high_aim.npy"))
    cannon_med = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/cannon_medium_aim.npy"))
    cannon_low = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/cannon_low_aim.npy"))
    human_up = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_up.npy"))
    human_straight = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_straight.npy"))
    human_down = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_down.npy"))
    human_ground = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/human_ground.npy"))
    water_tower = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/water_tower.npy"))
    water_tower_human1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/water_tower_human1.npy"))
    water_tower_human2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/human_cannonball/water_tower_human2.npy"))

    # Pad cannon sprites to match each other
    cannon_sprites, cannon_offsets = jr.pad_to_match([cannon_high, cannon_med, cannon_low])
    cannon_offsets = jnp.array(cannon_offsets)

    # Pad human sprites to match each other
    human_sprites, human_offsets = jr.pad_to_match([human_up, human_straight, human_down, human_ground])
    human_offsets = jnp.array(human_offsets)

    # Pad water tower sprites to match each other
    water_tower_sprites, water_tower_offsets = jr.pad_to_match([water_tower, water_tower_human1, water_tower_human2])
    water_tower_offsets = jnp.array(water_tower_offsets)

    # Pad sprites with human inside the water tower to match each other
    water_tower_human_sprites, water_tower_human_offsets = jr.pad_to_match([water_tower_human1, water_tower_human2])
    water_tower_human_offsets = jnp.array(water_tower_offsets)

    # Background sprite
    SPRITE_BG = jnp.expand_dims(bg, axis=0)

    # Cannon sprites
    SPRITE_CANNON = jnp.stack(cannon_sprites, axis=0)

    # Human sprites
    SPRITE_HUMAN = jnp.stack(human_sprites, axis=0)

    # Water tower sprites
    SPRITE_WATER_TOWER = jnp.stack(water_tower_sprites, axis=0)

    # Human in water tower sprites
    SPRITE_WATER_TOWER_HUMAN = jnp.stack(water_tower_human_sprites, axis=0)

    # Digits sprites
    DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/human_cannonball/digits/score_{}.npy"))

    return (
        SPRITE_BG,
        SPRITE_CANNON,
        SPRITE_HUMAN,
        SPRITE_WATER_TOWER,
        SPRITE_WATER_TOWER_HUMAN,
        DIGITS,
        cannon_offsets,
        human_offsets,
        water_tower_offsets,
        water_tower_human_offsets
    )

# Load sprites once at module level
(
        SPRITE_BG,
        SPRITE_CANNON,
        SPRITE_HUMAN,
        SPRITE_WATER_TOWER,
        SPRITE_WATER_TOWER_HUMAN,
        DIGITS,
        CANNON_OFFSETS,
        HUMAN_OFFSETS,
        WATER_TOWER_OFFSETS,
        WATER_TOWER_HUMAN_OFFSETS
    ) = load_sprites()

class JaxHumanCannonball(JaxEnvironment[HumanCannonballState, HumanCannonballObservation, HumanCannonballInfo, HumanCannonballConstants]):
    def __init__(self, consts: HumanCannonballConstants = None, reward_funcs: list[callable]=None):
        consts = consts or HumanCannonballConstants()
        super().__init__(consts)
        self.renderer = HumanCannonballRenderer(self.consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.frame_stack_size = 4
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        ]
        self.obs_size = 2*4+1+1+1+1 #TODO: Implement this correctly?

    # Determines the starting position of the human based on the angle
    @partial(jax.jit, static_argnums=(0,))
    def get_human_start(
            self, state_angle
    ):
        start_x, start_y = jax.lax.cond(
            state_angle < self.consts.ANGLE_HIGH_THRESHOLD,  # If angle under HIGH_THRESHOLD
            lambda _: self.consts.HUMAN_START_MED,  # Get medium start pos
            lambda _: self.consts.HUMAN_START_HIGH,  # Else, get high start pos
            operand=None
        )

        start_x, start_y = jax.lax.cond(
            state_angle < self.consts.ANGLE_LOW_THRESHOLD,  # If angle under LOW_THRESHOLD
            lambda _: self.consts.HUMAN_START_LOW,  # Get low start pos
            lambda _: (start_x, start_y),  # Else, leave unchanged
            operand=None
        )

        return start_x, start_y

    # Step functions

    # Update the human projectile position and velocity
    @partial(jax.jit, static_argnums=(0,))
    def human_step(
            self, state_human_x, state_human_y, state_human_x_vel, state_human_y_vel, state_human_launched,
            state_water_tower_x, state_angle, state_mph_values
    ):
        mph_speed = state_mph_values
        rad_angle = jnp.deg2rad(state_angle)
        t = self.consts.DT
        HORIZONTAL_SPEED_SCALE = 0.7  # Scale to compress the flying arc

        # 1. Compute candidate horizontal and vertical velocities
        x_vel = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda _: state_human_x_vel,  # Keep the old velocity
            lambda _: jnp.cos(rad_angle) * mph_speed * HORIZONTAL_SPEED_SCALE,  # Else, calculate the initial velocity
            operand=None
        )

        y_vel = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda _: state_human_y_vel + self.consts.GRAVITY * t,  # Update the old velocity
            lambda _: -jnp.sin(rad_angle) * mph_speed,  # Else, calculate the initial velocity
            operand=None
        )

        # 2. Compute candidate new positions
        human_x = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda x: x + x_vel * t,  # Update the old position
            lambda x: self.get_human_start(state_angle)[0],  # Else, set the initial position, depending on the angle
            operand=state_human_x
        )

        human_y = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda y: y + y_vel * t + 0.5 * self.consts.GRAVITY * t ** 2,  # Update the old position, account for gravity
            lambda y: self.get_human_start(state_angle)[1],  # Else, set the initial position, depending on the angle
            operand=state_human_y
        )

        # 3. Detect collision with tower wall
        coll = self.check_water_tower_wall_collision(
            human_x, human_y, state_water_tower_x
        )

        # 4. Reflect and dampen the velocity if there is a collision
        x_vel = jax.lax.cond(
            coll,
            lambda x: -x * self.consts.WALL_RESTITUTION,
            lambda x: x,
            operand=x_vel
        )

        # 5. Clamp x so human sits just left of the wall
        human_x = jax.lax.cond(
            coll,
            lambda _: jnp.array(state_water_tower_x - self.consts.HUMAN_SIZE[0], dtype=state_human_x.dtype),
            lambda x: x,
            operand=human_x
        )

        # 6. Set the launch status to True for subsequent steps
        human_launched = True

        # 7. Set collision status to True for steps after the wall hit
        coll = jax.lax.cond(
            jnp.logical_or(  # If the collision happened this step or some step before (human bounced off the wall)
                coll,
                x_vel <= 0.0
            ),
            lambda _: True,  # Set collision status to True
            lambda _: False,  # Else, set it to False
            operand=None
        )

        return human_x, human_y, x_vel, y_vel, human_launched, coll

    # Check if the player has scored
    @partial(jax.jit, static_argnums=(0,))
    def check_water_collision(
            self, state_human_x, state_human_y, state_water_tower_x, state_animation_running
    ):
        # If there is an animation running, return false
        animation = jnp.logical_not(state_animation_running)

        # Define bounding boxes for the human and water tower
        water_surface_x1 = state_water_tower_x + 1
        water_surface_y1 = self.consts.WATER_TOWER_Y - self.consts.WATER_TOWER_WALL_HEIGHT
        water_surface_x2 = water_surface_x1 + self.consts.WATER_SIZE[0]
        water_surface_y2 = water_surface_y1 + self.consts.WATER_SIZE[1]

        human_x1 = state_human_x
        human_y1 = state_human_y
        human_x2 = human_x1 + self.consts.HUMAN_SIZE[0]
        human_y2 = human_y1 + self.consts.HUMAN_SIZE[1]

        # AABB collision detection
        collision_x = jnp.logical_and(human_x1 < water_surface_x2, human_x2 > water_surface_x1)
        collision_y = jnp.logical_and(human_y1 < water_surface_y2, human_y2 > water_surface_y1)

        return jnp.logical_and(jnp.logical_and(collision_x, collision_y), animation)

    # Check if the player has missed
    @partial(jax.jit, static_argnums=(0,))
    def check_ground_collision(
            self, state_human_y, state_animation_running
    ):
        ground_colision = state_human_y + self.consts.HUMAN_SIZE[1] >= self.consts.GROUND_LEVEL
        return jnp.logical_and(ground_colision, jnp.logical_not(state_animation_running))

    # Check if the human has hit the water tower wall
    @partial(jax.jit, static_argnums=(0,))
    def check_water_tower_wall_collision(
            self, state_human_x, state_human_y, state_water_tower_x
    ):
        # Define bounding boxes for the water tower wall and the human
        wall_x1 = state_water_tower_x
        wall_y1 = self.consts.WATER_TOWER_Y - self.consts.WATER_TOWER_WALL_HEIGHT
        wall_x2 = wall_x1 + 1
        wall_y2 = wall_y1 + self.consts.WATER_TOWER_WALL_HEIGHT

        human_x1 = state_human_x + self.consts.HUMAN_SIZE[0] / 2  # Only check for front half of the human
        human_y1 = state_human_y
        human_x2 = human_x1 + self.consts.HUMAN_SIZE[0] / 2
        human_y2 = human_y1 + self.consts.HUMAN_SIZE[1]

        # AABB collision detection
        collision_x = jnp.logical_and(human_x1 < wall_x2, human_x2 > wall_x1)
        collision_y = jnp.logical_and(human_y1 < wall_y2, human_y2 > wall_y1)

        return jnp.logical_and(collision_x, collision_y)

    # Determines the new angle of the cannon based on the action
    @partial(jax.jit, static_argnums=(0,))
    def angle_step(
            self, state_angle, state_human_launched, angle_counter, action
    ):
        new_angle = state_angle

        # Update the angle based on the action as long as the human is not launched
        new_angle = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    # If the human is not launched and the action 'UP' has been held for ANGLE_BUFFER steps
                    jnp.logical_not(state_human_launched),
                    action == Action.UP
                ),
                angle_counter >= self.consts.ANGLE_BUFFER
            ),
            lambda s: s + 1,  # Increment the angle
            lambda s: s,  # Else, leave it unchanged
            operand=new_angle,
        )

        new_angle = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    # If the human is not launched and the action 'DOWN' has been held for ANGLE_BUFFER steps
                    jnp.logical_not(state_human_launched),
                    action == Action.DOWN
                ),
                angle_counter >= self.consts.ANGLE_BUFFER
            ),
            lambda s: s - 1,  # Decrement the angle
            lambda s: s,  # Else, leave it unchanged
            operand=new_angle,
        )

        # Ensure the angle is within the valid range
        new_angle = jnp.clip(new_angle, self.consts.ANGLE_MIN, self.consts.ANGLE_MAX)

        new_angle_counter = jax.lax.cond(
            angle_counter >= self.consts.ANGLE_BUFFER,  # If the angle has been updated
            lambda _: 0,  # Reset the angle counter
            lambda s: s,  # Else, leave it unchanged
            operand=angle_counter,
        )

        return new_angle, new_angle_counter

    # Determines the new position of the water tower based on the action
    @partial(jax.jit, static_argnums=(0,))
    def water_tower_step(
            self, state_water_tower_x, state_tower_wall_hit, state_human_launched, action
    ):
        new_x = state_water_tower_x

        # Update the position based on the action as long as the human is launched/ in flight
        new_x = jax.lax.cond(
            jnp.logical_and(  # If the human is launched and the action 'LEFT' is pressed
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
        new_x = jnp.clip(new_x, self.consts.WATER_TOWER_X_MIN, self.consts.WATER_TOWER_X_MAX)

        return new_x

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

        new_angle, new_angle_counter = self.angle_step(
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
            lambda _: self.human_step(   # Calculate the new position/velocity of the human via human_step
                state.human_x,
                state.human_y,
                state.human_x_vel,
                state.human_y_vel,
                state.human_launched,
                state.water_tower_x,
                state.angle,
                state.mph_values
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
                jnp.mod(state.step_counter, 8) == 0 # Only execute water_step every 8 steps (base implementation only moves the projectile every eighth tick)
            ),
            lambda _: self.water_tower_step(  # Calculate the new position of the water tower
                state.water_tower_x,
                tower_wall_hit,
                state.human_launched,
                action
            ),
            lambda _: state.water_tower_x,  # Else, leave it unchanged
            operand=None
        )

        # Step 4: Check if the player has scored
        new_score, new_animation_counter, new_animation_running = jax.lax.cond(
            self.check_water_collision(
                new_human_x,
                new_human_y,
                state.water_tower_x,
                state.animation_running
            ),
            lambda _: (state.score + 1, self.consts.ANIMATION_MISS_LENGTH, True), # Increment score and start hit animation
            lambda _: (state.score, state.animation_counter, state.animation_running), # Else, leave unchanged
            operand=None
        )

        # Step 5: Check if the player has missed
        new_misses, new_animation_counter, new_animation_running = jax.lax.cond(
            self.check_ground_collision(
                state.human_y,
                state.animation_running
            ),
            lambda _: (state.misses + 1, self.consts.ANIMATION_MISS_LENGTH, True), # Increment misses and start miss animation
            lambda _: (state.misses, state.animation_counter, state.animation_running), # Else, leave unchanged
            operand=None
        )

        # Check if animation started this step
        just_started = jnp.logical_and(
            jnp.not_equal(state.animation_running, new_animation_running),
            jnp.equal(new_animation_running, True)
        )

        # Check if round should reset
        round_reset = jnp.equal(1, new_animation_counter)

        # Step 6: Decrement animation counter if animation is happening
        new_animation_counter = jax.lax.cond(
            jnp.not_equal(new_animation_counter, 0),
            lambda x: x - 1,
            lambda x: x,
            operand=new_animation_counter
        )

        # Step 7: Reset the round when the animation finishes this step

        # Freeze old values in case of an animation running and decrement animation counter
        (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, new_water_tower_x,
         tower_wall_hit, new_animation_running, new_animation_counter, new_score, new_misses) = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_not(round_reset),       # If there is no reset this step and
                jnp.logical_and(
                    new_animation_running,          # there is an animation happening
                    jnp.logical_not(just_started)   # that didn't just start (so that scores are updated)
                )
            ),
            lambda _: (state.human_x, state.human_y, state.human_x_vel, state.human_y_vel,  # Freeze all values not related to the animation
                       state.human_launched, state.water_tower_x,state.tower_wall_hit,
                       new_animation_running, new_animation_counter, state.score, state.misses),
            lambda _: (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel,          # Else, update normally
                       human_launched, new_water_tower_x,tower_wall_hit,
                       new_animation_running, new_animation_counter, new_score, new_misses),
            operand=None
        )

        # On round reset, reset values, else, apply this step's changes
        (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, new_water_tower_x,
         tower_wall_hit, new_mph_values, new_rng_key, new_animation_running, new_animation_counter) = jax.lax.cond(
            round_reset,
            lambda _: self.reset_round(state.rng_key, state.human_x, state.angle),
            lambda _: (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel,
                       human_launched, new_water_tower_x, tower_wall_hit, state.mph_values,
                       state.rng_key, new_animation_running, new_animation_counter),
            operand=None
        )

        # Step 7: Create the new state
        new_state = HumanCannonballState(
            human_x=new_human_x,
            human_y=new_human_y,
            human_x_vel=new_human_x_vel,
            human_y_vel=new_human_y_vel,
            human_launched=human_launched,
            water_tower_x=new_water_tower_x,
            mph_values=new_mph_values,
            tower_wall_hit=tower_wall_hit,
            angle=new_angle,
            angle_counter=new_angle_counter,
            score=new_score,
            misses=new_misses,
            step_counter=state.step_counter + 1,
            rng_key=new_rng_key,
            animation_running=new_animation_running,
            animation_counter=new_animation_counter
        )

        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    # Reset the round after a score or a miss
    @partial(jax.jit, static_argnums=(0,))
    def reset_round(
            self, key, current_human_x, current_angle
    ):
        human_x = 0.0
        human_y = 0.0
        human_x_vel = 0.0
        human_y_vel = 0.0
        human_launched = False
        water_tower_x = self.consts.WATER_TOWER_X
        tower_wall_hit = False
        # Change the rng_key and generate a new mph_value
        rng_key, subkey = jax.random.split(key)
        # Add pseudo-randomness to mph generation by integrating human x pos and angle
        subkey = jax.random.fold_in(subkey, current_human_x)
        subkey = jax.random.fold_in(subkey, current_angle)
        mph_values = jax.random.randint(subkey, (), self.consts.MPH_MIN, self.consts.MPH_MAX + 1)
        # Reset the animation status
        animation_running = False
        animation_counter = 0

        return human_x, human_y, human_x_vel, human_y_vel, human_launched, water_tower_x, tower_wall_hit, mph_values, rng_key, animation_running, animation_counter

    def reset(
            self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)
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
            water_tower_x = jnp.array(self.consts.WATER_TOWER_X).astype(jnp.int32),
            mph_values = jnp.array(43).astype(jnp.int32),
            tower_wall_hit = jnp.array(False),
            angle = jnp.array(self.consts.ANGLE_START).astype(jnp.int32),
            angle_counter = jnp.array(0).astype(jnp.int32),
            score = jnp.array(0).astype(jnp.int32),
            misses = jnp.array(0).astype(jnp.int32),
            step_counter = jnp.array(0).astype(jnp.int32),
            rng_key=key,
            animation_running = jnp.array(False),
            animation_counter = jnp.array(0).astype(jnp.int32)
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    def render(self, state: HumanCannonballState) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: HumanCannonballState):
        # Create human projectile
        human_cannonball = EntityPosition(
            x=state.human_x,
            y=state.human_y,
            width=jnp.array(self.consts.HUMAN_SIZE[0]),
            height=jnp.array(self.consts.HUMAN_SIZE[1]),
        )

        # Create water tower
        water_tower = EntityPosition(
            x=state.water_tower_x,
            y=jnp.array(self.consts.WATER_TOWER_Y),
            width=jnp.array(self.consts.WATER_TOWER_WIDTH),
            height=jnp.array(self.consts.WATER_TOWER_WALL_HEIGHT - 1),
        )

        return HumanCannonballObservation(
            human=human_cannonball,
            water_tower=water_tower,
            angle=state.angle,
            mph=state.mph_values,
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
            state.misses >= self.consts.MISS_LIMIT,
            state.score >= self.consts.SCORE_LIMIT
        )

class HumanCannonballRenderer(JAXGameRenderer):
    """JAX-based HumanCannonball game renderer, optimized with JIT compilation."""
    def __init__(self, consts: HumanCannonballConstants = None):
        super().__init__()
        self.cannon_offset_length = len(CANNON_OFFSETS)
        self.human_offset_length = len(HUMAN_OFFSETS)
        self.water_tower_offset_length = len(WATER_TOWER_OFFSETS)
        self.water_tower_human_offset = len(WATER_TOWER_HUMAN_OFFSETS)
        self.consts = consts or HumanCannonballConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A HumanCannonballState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        raster = jr.create_initial_frame(self.consts.WIDTH, self.consts.HEIGHT)

        # Render the background
        frame_bg = jr.get_sprite_frame(SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        # Render the cannon
        # 1. Determine which sprite to load
        cannon_sprite_idx, cannon_offset = jax.lax.cond(
            state.angle < self.consts.ANGLE_HIGH_THRESHOLD, # If angle under HIGH_THRESHOLD
            lambda _: (1, 5),    # Render med sprite
            lambda _: (0, 8),    # Else, render high sprite
            operand=None
        )

        cannon_sprite_idx, cannon_offset = jax.lax.cond(
            state.angle < self.consts.ANGLE_LOW_THRESHOLD,  # If angle under LOW_THRESHOLD
            lambda _: (2,0),  # Render low sprite
            lambda _: (cannon_sprite_idx, cannon_offset),  # Else, keep the sprite
            operand=None
        )

        # 2. Render the cannon sprite
        frame_cannon = jr.get_sprite_frame(SPRITE_CANNON, cannon_sprite_idx)
        raster = jr.render_at(raster, self.consts.CANNON_X, self.consts.CANNON_Y - cannon_offset, frame_cannon)

        # Render the human
        # 1. Determine which sprite to load
        flying_angle_rad = jnp.arctan2(-state.human_y_vel, state.human_x_vel)
        flying_angle = jnp.rad2deg(flying_angle_rad)

        FLYING_ANGLE_THRESHOLD = 45

        human_offset_x = 0

        human_sprite_idx, human_offset_y = jax.lax.cond(
            flying_angle > FLYING_ANGLE_THRESHOLD,  # If ascension angle is > 45 degrees
            lambda _: (0,3),   # Use up sprite
            lambda _: (1,0),   # Else, use straight sprite
            operand=None
        )

        human_sprite_idx, human_offset_y = jax.lax.cond(
            jnp.logical_or( # If ascension angle is < -45 degrees or the tower has been hit
                flying_angle < -FLYING_ANGLE_THRESHOLD,
                state.tower_wall_hit
            ),
            lambda _: (2,2),  # Use down sprite
            lambda _: (human_sprite_idx, human_offset_y),  # Else, keep the sprite
            operand=None
        )

        human_sprite_idx, human_offset_x, human_offset_y = jax.lax.cond(
            jnp.logical_and(
                state.animation_running,
                jnp.less(state.human_y, self.consts.GROUND_LEVEL - 2)
            ),  # If human is on the ground
            lambda _: (3, 2, 5),  # Use ground sprite
            lambda _: (human_sprite_idx, human_offset_x, human_offset_y),  # Else, keep the sprite
            operand=None
        )

        # 2. Render the human sprite
        frame_human = jr.get_sprite_frame(SPRITE_HUMAN, human_sprite_idx)
        raster = jax.lax.cond(  # Only render when launched
            state.human_launched,
            lambda r: jr.render_at(r, state.human_x - human_offset_x, state.human_y - human_offset_y, frame_human),
            lambda r: r,
            operand=raster
        )

        # Render the water tower
        frame_water_tower = jr.get_sprite_frame(SPRITE_WATER_TOWER, 0)     #TODO: For now, no score animation
        raster = jr.render_at(raster, state.water_tower_x, self.consts.WATER_TOWER_Y - self.consts.WATER_TOWER_WALL_HEIGHT + 1, frame_water_tower)

        # Get the score and misses
        score_digits = jr.int_to_digits(state.score, max_digits=1)
        misses_digits = jr.int_to_digits(state.misses, max_digits=1)

        # Get the mph and angle
        mph_digits = jr.int_to_digits(state.mph_values, max_digits=2)
        angle_digits = jr.int_to_digits(state.angle, max_digits=2)

        # Render the score
        raster = jr.render_label_selective(raster, self.consts.SCORE_X, self.consts.SCORE_MISS_Y, score_digits, DIGITS,
                                           0, 1)

        # Render the misses
        raster = jr.render_label_selective(raster, self.consts.MISS_X, self.consts.SCORE_MISS_Y, misses_digits, DIGITS,
                                           0, 1)

        # Render the mph
        raster = jr.render_label_selective(raster, self.consts.MPH_ANGLE_X[0], self.consts.MPH_Y, mph_digits, DIGITS,
                                           0, 2, spacing= self.consts.MPH_ANGLE_X[1] - self.consts.MPH_ANGLE_X[0])

        # Render the angle
        raster = jr.render_label_selective(raster, self.consts.MPH_ANGLE_X[0], self.consts.ANGLE_Y, angle_digits, DIGITS,
                                           0, 2, spacing=self.consts.MPH_ANGLE_X[1] - self.consts.MPH_ANGLE_X[0])

        return raster


if __name__ == "__main__":
    hello = "World"