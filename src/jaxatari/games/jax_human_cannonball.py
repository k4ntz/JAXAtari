import os
from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action


def _get_default_asset_config() -> tuple:
    """
    Declarative sprite manifest for Human Cannonball renderer.
    """
    return (
        {'name': 'background', 'type': 'background', 'file': 'background.npy'},
        {'name': 'top_banner', 'type': 'single', 'file': 'top_banner.npy'},
        {'name': 'cannon', 'type': 'group', 'files': [
            'cannon_high_aim.npy',
            'cannon_medium_aim.npy',
            'cannon_low_aim.npy',
        ]},
        {'name': 'human', 'type': 'group', 'files': [
            'human_up.npy',
            'human_straight.npy',
            'human_down.npy',
            'human_ground.npy',
        ]},
        {'name': 'water_tower', 'type': 'group', 'files': [
            'water_tower.npy',
            'water_tower_overlap.npy',
            'water_tower_overlap_top.npy',
            'water_tower_human1.npy',
            'water_tower_human2.npy',
        ]},
        {'name': 'digits', 'type': 'digits', 'pattern': 'digits/score_{}.npy'},
    )


@dataclass(frozen=True)
class HumanCannonballConstants:
    WIDTH: int = 160
    HEIGHT: int = 210

    GROUND_LEVEL: int = 152

    # Game constants
    MISS_LIMIT: int = 7
    SCORE_LIMIT: int = 7

    # Game physics constants
    DT: float = 1.0 / 15.0  # time step between frames
    GRAVITY: float = 10.0
    WALL_RESTITUTION: float = 0.3  # Coefficient of restitution for the wall collision

    # MPH constraints
    MPH_MIN: int = 28
    MPH_MAX: int = 45
    MPH_START: int = 43

    # Angle constants
    ANGLE_START: int = 30
    ANGLE_MAX: int = 80
    ANGLE_MIN: int = 20

    # The cannon aims low if angle <37, medium if 37 <= angle < 59, and high if angle >= 59
    ANGLE_LOW_THRESHOLD: int = 37
    ANGLE_HIGH_THRESHOLD: int = 59

    ANGLE_BUFFER: int = 16   # Only update the angle if the action is held for this many steps

    # Starting positions of the human
    HUMAN_START_LOW: chex.Array = field(
        default_factory=lambda: jnp.array([84.0, 128.0], dtype=jnp.float32)
    )
    HUMAN_START_MED: chex.Array = field(
        default_factory=lambda: jnp.array([84.0, 121.0], dtype=jnp.float32)
    )
    HUMAN_START_HIGH: chex.Array = field(
        default_factory=lambda: jnp.array([80.0, 118.0], dtype=jnp.float32)
    )

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
    ANIMATION_MISS_LENGTH: int = 128
    ANIMATION_HIT_LENGTH: int = 248

    # Asset config
    ASSET_CONFIG: tuple = field(default_factory=_get_default_asset_config)

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

class JaxHumanCannonball(JaxEnvironment[HumanCannonballState, HumanCannonballObservation, HumanCannonballInfo, HumanCannonballConstants]):
    # Minimal ALE action set for Human Cannonball
    ACTION_SET: jnp.ndarray = jnp.array(
        [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE,
        ],
        dtype=jnp.int32,
    )
    
    def __init__(self, consts: HumanCannonballConstants = None):
        consts = consts or HumanCannonballConstants()
        super().__init__(consts)
        self.renderer = HumanCannonballRenderer(self.consts)
        self.obs_size = 4+4+1+1+1+1

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
            lambda _: jnp.array([start_x, start_y], dtype=jnp.float32),  # Else, leave unchanged
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
            lambda _: (jnp.cos(rad_angle) * mph_speed * HORIZONTAL_SPEED_SCALE).astype(jnp.float32),  # Else, calculate the initial velocity
            operand=None
        )

        y_vel = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda _: (state_human_y_vel + self.consts.GRAVITY * t).astype(jnp.float32),  # Update the old velocity
            lambda _: (-jnp.sin(rad_angle) * mph_speed - 2).astype(jnp.float32),  # Else, calculate the initial velocity
            operand=None
        )

        # 2. Compute candidate new positions
        human_x = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda x: (x + x_vel * t).astype(jnp.float32),  # Update the old position
            lambda x: self.get_human_start(state_angle)[0],  # Else, set the initial position, depending on the angle
            operand=state_human_x
        )

        human_y = jax.lax.cond(
            state_human_launched,  # If human is already launched
            lambda y: (y + y_vel * t + 0.5 * self.consts.GRAVITY * t ** 2).astype(jnp.float32),  # Update the old position, account for gravity
            lambda y: self.get_human_start(state_angle)[1],  # Else, set the initial position, depending on the angle
            operand=state_human_y
        )

        # 3. Detect collision with tower walls
        coll_left = self.check_water_tower_wall_collision(
            human_x, human_y, state_water_tower_x
        )

        # 4. Reflect and dampen the velocity if there is a collision with the left wall
        x_vel = jax.lax.cond(
            coll_left,
            lambda x: (-x * self.consts.WALL_RESTITUTION).astype(jnp.float32),
            lambda x: x,
            operand=x_vel
        )

        # 5. Clamp x so human sits just left of the wall in case of collision
        human_x = jax.lax.cond(
            coll_left,
            lambda _: jnp.array(state_water_tower_x - self.consts.HUMAN_SIZE[0], dtype=state_human_x.dtype),
            lambda x: x,
            operand=human_x
        )

        # 6. Set the launch status to True for subsequent steps
        human_launched = True

        # 7. Set collision status to True for steps after the left wall has been hit
        coll = jax.lax.cond(
            jnp.logical_or(  # If the collision happened this step or some step before (human bounced off the wall)
                coll_left,
                x_vel <= jnp.array(0).astype(jnp.float32)
            ),
            lambda _: True,  # Set collision status to True
            lambda _: False,  # Else, set it to False
            operand=None
        )

        return human_x, human_y, x_vel, y_vel, human_launched, coll

    # Check if the player has scored
    @partial(jax.jit, static_argnums=(0,))
    def check_water_collision(
            self, state_human_x, state_human_y, state_water_tower_x, state_animation_running=False
    ):
        # Define bounding boxes for the human and water tower
        water_surface_x1 = state_water_tower_x + 1
        water_surface_y1 = self.consts.WATER_TOWER_Y - self.consts.WATER_TOWER_WALL_HEIGHT
        water_surface_x2 = water_surface_x1 + self.consts.WATER_SIZE[0]
        water_surface_y2 = water_surface_y1 + self.consts.WATER_SIZE[1]

        # Only count hits with the center of the human sprite
        human_center_x = state_human_x + self.consts.HUMAN_SIZE[0] / 2
        human_center_y = state_human_y + self.consts.HUMAN_SIZE[1] / 2

        human_x1 = human_center_x
        human_y1 = human_center_y
        human_x2 = human_center_x + 1
        human_y2 = human_center_y + 1

        # AABB collision detection
        collision_x = jnp.logical_and(human_x1 < water_surface_x2, human_x2 > water_surface_x1)
        collision_y = jnp.logical_and(human_y1 < water_surface_y2, human_y2 > water_surface_y1)

        water_collision = jnp.logical_and(collision_x, collision_y)

        return jnp.logical_and(water_collision, jnp.logical_not(state_animation_running))

    # Check if the player has missed
    @partial(jax.jit, static_argnums=(0,))
    def check_ground_collision(
            self, state_human_y, state_animation_running
    ):
        ground_collision = state_human_y + self.consts.HUMAN_SIZE[1] >= self.consts.GROUND_LEVEL
        return jnp.logical_and(ground_collision, jnp.logical_not(state_animation_running))

    # Check if the human has hit the left water tower wall
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
        human_y2 = human_y1 + self.consts.HUMAN_SIZE[1] / 2 # Only check for upper half of the human

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

        new_angle = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    # If the human is not launched and the action 'UPLEFT' has been held for ANGLE_BUFFER steps
                    jnp.logical_not(state_human_launched),
                    action == Action.UPLEFT
                ),
                angle_counter >= self.consts.ANGLE_BUFFER
            ),
            lambda s: s + 10,  # Increment the angle by 10
            lambda s: s,  # Else, leave it unchanged
            operand=new_angle,
        )

        new_angle = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(
                    # If the human is not launched and the action 'DOWNLEFT' has been held for ANGLE_BUFFER steps
                    jnp.logical_not(state_human_launched),
                    action == Action.DOWNLEFT
                ),
                angle_counter >= self.consts.ANGLE_BUFFER
            ),
            lambda s: s - 10,  # Decrement the angle by 10
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
        # Translate agent action index to ALE console action
        atari_action = jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

        # Step 1: Update the angle of the cannon
        new_angle_counter = jax.lax.cond(
            jnp.logical_or(         # If the action is UP, DOWN, UPLEFT or DOWNLEFT
                jnp.logical_or(
                    atari_action == Action.UP,
                    atari_action == Action.DOWN),
                jnp.logical_or(
                    atari_action == Action.UPLEFT,
                    atari_action == Action.DOWNLEFT
                )
            ),
            lambda s: s + 1,  # Increment the angle counter
            lambda _: 0,  # Else, reset it
            operand=state.angle_counter
        )

        new_angle, new_angle_counter = self.angle_step(
            state.angle,
            state.human_launched,
            new_angle_counter,
            atari_action
        )

        # Step 2: Update the position of the human projectile
        new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, tower_wall_hit = jax.lax.cond(
            jnp.logical_and(    # If human is in flight or the current action is FIRE
                jnp.logical_or(state.human_launched, atari_action == Action.FIRE),
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
                atari_action
            ),
            lambda _: state.water_tower_x,  # Else, leave it unchanged
            operand=None
        )

        # Step 4: Check if the player has scored
        new_animation_counter, new_animation_running = jax.lax.cond(
            self.check_water_collision(
                new_human_x,
                new_human_y,
                state.water_tower_x,
                state.animation_running
            ),
            lambda _: (self.consts.ANIMATION_HIT_LENGTH, True), # Increment score and start hit animation
            lambda _: (state.animation_counter, state.animation_running), # Else, leave unchanged
            operand=None
        )

        # Step 5: Check if the player has missed
        new_animation_counter, new_animation_running, new_human_y = jax.lax.cond( # Need to set human_y to make sure the miss animation sprite is loaded on the ground
            self.check_ground_collision(
                state.human_y,
                state.animation_running
            ),
            lambda _: (self.consts.ANIMATION_MISS_LENGTH, True,
                       jnp.array(self.consts.GROUND_LEVEL - self.consts.HUMAN_SIZE[1] + 1).astype(jnp.float32)), # Increment misses and start miss animation
            lambda _: (new_animation_counter, new_animation_running, new_human_y), # Else, leave unchanged
            operand=None
        )

        # Check if an animation started this step
        just_started = jnp.logical_and(
            jnp.not_equal(state.animation_running, new_animation_running),
            jnp.equal(new_animation_running, True)
        )

        # Check if round should reset this step
        round_reset = jnp.equal(1, new_animation_counter)

        # Step 6: Decrement animation counter if animation is happening
        new_animation_counter = jax.lax.cond(
            jnp.not_equal(new_animation_counter, 0),
            lambda x: x - 1,
            lambda x: x,
            operand=new_animation_counter
        )

        # Safe new human positions and velocities in case of hit animation
        (hit_human_x, hit_human_y, hit_human_x_vel, hit_human_y_vel) = (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel)

        # Step 7: Reset the round when the animation finishes this step

        # Freeze old values in case of an animation running and decrement animation counter
        (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, new_water_tower_x,
         tower_wall_hit, new_animation_running, new_animation_counter) = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_not(round_reset),       # If there is no reset this step and
                jnp.logical_and(
                    new_animation_running,          # there is an animation happening
                    jnp.logical_not(just_started)   # that didn't just start (so that this steps values are not frozen)
                )
            ),
            lambda _: (state.human_x, state.human_y, state.human_x_vel, state.human_y_vel,  # Freeze all values not related to the animation
                       state.human_launched, state.water_tower_x,state.tower_wall_hit,
                       new_animation_running, new_animation_counter),
            lambda _: (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel,          # Else, update normally
                       human_launched, new_water_tower_x,tower_wall_hit,
                       new_animation_running, new_animation_counter),
            operand=None
        )

        # For the first 10 steps of the hit animation, continue the human's trajectory to let him dive into the water
        (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel) = jax.lax.cond(
            jnp.greater(new_animation_counter, self.consts.ANIMATION_HIT_LENGTH - 10),
            lambda _: (hit_human_x, hit_human_y, hit_human_x_vel / 2, hit_human_y_vel), # Water drag slows horizontal speed
            lambda _: (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel),
            operand=None
        )

        # On round reset, reset values and award score/misses, else, apply this step's changes
        (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel, human_launched, new_water_tower_x,
         tower_wall_hit, new_mph_values, new_rng_key, new_animation_running, new_animation_counter, new_score, new_misses) = jax.lax.cond(
            round_reset,
            lambda _: self.reset_round(state.rng_key, state.human_x, state.human_y, state.angle, state.score, state.misses),
            lambda _: (new_human_x, new_human_y, new_human_x_vel, new_human_y_vel,
                       human_launched, new_water_tower_x, tower_wall_hit, state.mph_values,
                       state.rng_key, new_animation_running, new_animation_counter, state.score, state.misses),
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
        env_reward = self._get_reward(state, new_state)
        info = self._get_info(new_state)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    # Reset the round after a score or a miss
    @partial(jax.jit, static_argnums=(0,))
    def reset_round(
            self, key, current_human_x, current_human_y, current_angle, current_score, current_misses
    ):
        human_x = jnp.array(0).astype(jnp.float32)
        human_y = jnp.array(0).astype(jnp.float32)
        human_x_vel = jnp.array(0).astype(jnp.float32)
        human_y_vel = jnp.array(0).astype(jnp.float32)
        human_launched = jnp.array(False)
        water_tower_x = jnp.array(self.consts.WATER_TOWER_X).astype(jnp.int32)
        tower_wall_hit = jnp.array(False)
        # Change the rng_key and generate a new mph_value
        rng_key, subkey = jax.random.split(key)
        # Add pseudo-randomness to mph generation by integrating human x pos and angle
        subkey = jax.random.fold_in(subkey, current_human_x)
        subkey = jax.random.fold_in(subkey, current_angle)
        mph_values = jax.random.randint(key=subkey, shape=(), minval=self.consts.MPH_MIN, maxval=self.consts.MPH_MAX + 1, dtype=jnp.int32)
        # Reset the animation status
        animation_running = jnp.array(False)
        animation_counter = jnp.array(0).astype(jnp.int32)
        # Award score or miss
        (score, misses) = jax.lax.cond(
            jnp.less(self.consts.GROUND_LEVEL - self.consts.HUMAN_SIZE[1], current_human_y),  # If the human is on the ground on reset
            lambda _: (current_score, current_misses + 1),  # Increment misses
            lambda _: (current_score + 1, current_misses),  # Else, increment score
            operand=None
        )

        return (human_x, human_y, human_x_vel, human_y_vel,
                human_launched, water_tower_x, tower_wall_hit,
                mph_values, rng_key, animation_running, animation_counter,
                score, misses)

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
            mph_values = jnp.array(self.consts.MPH_START).astype(jnp.int32),
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
    def _get_observation(self, state: HumanCannonballState) -> HumanCannonballObservation:
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
        return spaces.Discrete(len(self.ACTION_SET))

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "human": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=()),
                "y": spaces.Box(low=0, high=210, shape=()),
                "width": spaces.Box(low=0, high=160, shape=()),
                "height": spaces.Box(low=0, high=210, shape=()),
            }),
            "water_tower": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=()),
                "y": spaces.Box(low=0, high=210, shape=()),
                "width": spaces.Box(low=0, high=160, shape=()),
                "height": spaces.Box(low=0, high=210, shape=()),
            }),
            "angle": spaces.Box(low=20, high=80, shape=()),
            "mph": spaces.Box(low=28, high=45, shape=()),
            "score": spaces.Box(low=0, high=7, shape=()),
            "misses": spaces.Box(low=0, high=7, shape=()),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: HumanCannonballState) -> HumanCannonballInfo:
        return HumanCannonballInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: HumanCannonballState, state: HumanCannonballState):
        return (state.score - state.misses) - (
                previous_state.score - previous_state.misses
        )

    @partial(jax.jit,  static_argnums=(0,))
    def _get_done(
            self, state: HumanCannonballState
    ) -> bool:
        return jnp.logical_or(
            state.misses >= self.consts.MISS_LIMIT,
            state.score >= self.consts.SCORE_LIMIT
        )

class HumanCannonballRenderer(JAXGameRenderer):
    """Render Human Cannonball using the modern render_utils pipeline."""

    def __init__(self, consts: HumanCannonballConstants = None):
        super().__init__()
        self.consts = consts or HumanCannonballConstants()
        self.sprite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites/human_cannonball")
        self.ru_config = render_utils.RendererConfig(
            game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
            channels=3,
        )
        self.jr = render_utils.JaxRenderingUtils(self.ru_config)
        asset_config = list(self.consts.ASSET_CONFIG)
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS,
        ) = self.jr.load_and_setup_assets(asset_config, self.sprite_path)

        self.CANNON_STACK = self.SHAPE_MASKS.get('cannon')
        self.HUMAN_STACK = self.SHAPE_MASKS.get('human')
        self.WATER_TOWER_STACK = self.SHAPE_MASKS.get('water_tower')
        self.TOP_BANNER_MASK = self.SHAPE_MASKS.get('top_banner')
        self.DIGITS_MASK = self.SHAPE_MASKS.get('digits')

    @staticmethod
    def _round_to_int(value: chex.Array) -> chex.Array:
        return jnp.round(value).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: HumanCannonballState) -> jnp.ndarray:
        raster = self.jr.create_object_raster(self.BACKGROUND)

        # Render top banner overlay
        if self.TOP_BANNER_MASK is not None:
            raster = self.jr.render_at(
                raster,
                0,
                0,
                self.TOP_BANNER_MASK,
                flip_offset=self.FLIP_OFFSETS.get('top_banner', jnp.array([0, 0])),
            )

        # Cannon selection based on firing angle
        cond_high = state.angle < self.consts.ANGLE_HIGH_THRESHOLD
        cond_low = state.angle < self.consts.ANGLE_LOW_THRESHOLD
        cannon_idx = jnp.where(cond_low, 2, jnp.where(cond_high, 1, 0)).astype(jnp.int32)
        cannon_offset = jnp.where(cond_low, 0, jnp.where(cond_high, 5, 8)).astype(jnp.int32)
        cannon_y = jnp.array(self.consts.CANNON_Y, dtype=jnp.int32) - cannon_offset
        if self.CANNON_STACK is not None:
            raster = self.jr.render_at(
                raster,
                jnp.array(self.consts.CANNON_X, dtype=jnp.int32),
                cannon_y,
                self.CANNON_STACK[cannon_idx],
                flip_offset=self.FLIP_OFFSETS.get('cannon', jnp.array([0, 0])),
            )

        # Background water tower top (always index 2)
        if self.WATER_TOWER_STACK is not None:
            tower_top = self.WATER_TOWER_STACK[2]
            raster = self.jr.render_at(
                raster,
                self._round_to_int(state.water_tower_x) - 13,
                jnp.array(self.consts.WATER_TOWER_Y - self.consts.WATER_TOWER_WALL_HEIGHT - 6, jnp.int32),
                tower_top,
                flip_offset=self.FLIP_OFFSETS.get('water_tower', jnp.array([0, 0])),
            )

        # Human sprite selection
        flying_angle_rad = jnp.arctan2(-state.human_y_vel, state.human_x_vel)
        flying_angle = jnp.rad2deg(flying_angle_rad)
        FLYING_ANGLE_THRESHOLD = 35

        cond_up = flying_angle > FLYING_ANGLE_THRESHOLD
        cond_down = jnp.logical_or(flying_angle < -FLYING_ANGLE_THRESHOLD, state.tower_wall_hit)
        human_idx = jnp.where(cond_up, 0, 1)
        human_offset_y = jnp.where(cond_up, 3, 0)
        human_idx = jnp.where(cond_down, 2, human_idx)
        human_offset_y = jnp.where(cond_down, 2, human_offset_y)
        human_offset_x = jnp.array(0, jnp.int32)

        miss_animation = jnp.logical_and(
            state.animation_running,
            state.human_y > (self.consts.GROUND_LEVEL - self.consts.HUMAN_SIZE[1] - 1),
        )
        hit_animation = jnp.logical_and(
            state.animation_running,
            state.human_y < (self.consts.GROUND_LEVEL - self.consts.HUMAN_SIZE[1] + 1),
        )
        human_idx = jnp.where(miss_animation, 3, human_idx)
        human_offset_x = jnp.where(miss_animation, 2, human_offset_x)
        human_offset_y = jnp.where(miss_animation, 6, human_offset_y)

        human_overlap = jnp.logical_and(
            hit_animation,
            state.animation_counter > (self.consts.ANIMATION_HIT_LENGTH - 10),
        )

        if self.HUMAN_STACK is not None:
            human_x = self._round_to_int(state.human_x) - human_offset_x
            human_y = self._round_to_int(state.human_y) - human_offset_y
            raster = jax.lax.cond(
                jnp.logical_and(
                    state.human_launched,
                    jnp.logical_or(~hit_animation, human_overlap),
                ),
                lambda r: self.jr.render_at(
                    r,
                    human_x,
                    human_y,
                    self.HUMAN_STACK[human_idx],
                    flip_offset=self.FLIP_OFFSETS.get('human', jnp.array([0, 0])),
                ),
                lambda r: r,
                raster,
            )

        # Water tower selection (foreground)
        def tower_selection():
            idx = jnp.array(0, jnp.int32)
            off_x = jnp.array(0, jnp.int32)
            off_y = jnp.array(0, jnp.int32)

            idx = jnp.where(hit_animation, 1, idx)
            off_x = jnp.where(hit_animation, 13, off_x)
            off_y = jnp.where(hit_animation, 7, off_y)

            cond_phase1 = jnp.logical_and(
                hit_animation,
                jnp.logical_and(
                    state.animation_counter < (self.consts.ANIMATION_HIT_LENGTH / 2),
                    state.animation_counter > (self.consts.ANIMATION_HIT_LENGTH / 2 - 36),
                ),
            )
            idx = jnp.where(cond_phase1, 3, idx)
            off_x = jnp.where(cond_phase1, 0, off_x)
            off_y = jnp.where(cond_phase1, 4, off_y)

            cond_phase2 = jnp.logical_and(
                hit_animation,
                jnp.logical_and(
                    state.animation_counter < (self.consts.ANIMATION_HIT_LENGTH / 2 - 36),
                    state.animation_counter > (self.consts.ANIMATION_HIT_LENGTH / 2 - 36 - 39),
                ),
            )
            idx = jnp.where(cond_phase2, 4, idx)

            cond_phase3 = jnp.logical_and(
                hit_animation,
                state.animation_counter <= (self.consts.ANIMATION_HIT_LENGTH / 2 - 36 - 39),
            )
            idx = jnp.where(cond_phase3, 3, idx)

            return idx, off_x, off_y

        tower_idx, tower_off_x, tower_off_y = tower_selection()
        if self.WATER_TOWER_STACK is not None:
            raster = self.jr.render_at(
                raster,
                self._round_to_int(state.water_tower_x) - tower_off_x,
                jnp.array(self.consts.WATER_TOWER_Y - self.consts.WATER_TOWER_WALL_HEIGHT + 1, jnp.int32) - tower_off_y,
                self.WATER_TOWER_STACK[tower_idx],
                flip_offset=self.FLIP_OFFSETS.get('water_tower', jnp.array([0, 0])),
            )

        # Render HUD digits
        if self.DIGITS_MASK is not None:
            score_digits = self.jr.int_to_digits(state.score, max_digits=1)
            misses_digits = self.jr.int_to_digits(state.misses, max_digits=1)
            mph_digits = self.jr.int_to_digits(state.mph_values, max_digits=2)
            angle_digits = self.jr.int_to_digits(state.angle, max_digits=2)
            spacing = self.consts.MPH_ANGLE_X[1] - self.consts.MPH_ANGLE_X[0]

            raster = self.jr.render_label_selective(
                raster,
                jnp.array(self.consts.SCORE_X, jnp.int32),
                jnp.array(self.consts.SCORE_MISS_Y, jnp.int32),
                score_digits,
                self.DIGITS_MASK,
                0,
                1,
            )
            raster = self.jr.render_label_selective(
                raster,
                jnp.array(self.consts.MISS_X, jnp.int32),
                jnp.array(self.consts.SCORE_MISS_Y, jnp.int32),
                misses_digits,
                self.DIGITS_MASK,
                0,
                1,
            )
            raster = self.jr.render_label_selective(
                raster,
                jnp.array(self.consts.MPH_ANGLE_X[0], jnp.int32),
                jnp.array(self.consts.MPH_Y, jnp.int32),
                mph_digits,
                self.DIGITS_MASK,
                0,
                2,
                spacing=spacing,
            )
            raster = self.jr.render_label_selective(
                raster,
                jnp.array(self.consts.MPH_ANGLE_X[0], jnp.int32),
                jnp.array(self.consts.ANGLE_Y, jnp.int32),
                angle_digits,
                self.DIGITS_MASK,
                0,
                2,
                spacing=spacing,
            )

        return self.jr.render_from_palette(raster, self.PALETTE)


if __name__ == "__main__":
    hello = "World"