"""
Project: JAXAtari VideoPinball
Description: Our team's JAX implementation of Video Pinball.

Authors:
    - Michael Olenberger <michael.olenberger@stud.tu-darmstadt.de>
    - Maximilian Roth <maximilian.roth@stud.tu-darmstadt.de>
    - Jonas Neumann <jonas.neumann@stud.tu-darmstadt.de>
    - Yuddhish Chooah <yuddhish.chooah@stud.tu-darmstadt.de>

"""

import os
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple
import jax.lax
import jax.numpy as jnp
import jax.random as jrandom
import chex
import pygame

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as jr
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari import spaces
from jaxatari.games.videopinball_constants import (
    BallMovement,
    EntityState,
    HitPointSelector,
    SceneObject,
    VideoPinballInfo,
    VideoPinballObservation,
    VideoPinballState,
)
from jaxatari.games.videopinball_constants import VideoPinballConstants

HitPointSelector = HitPointSelector()


def get_human_action() -> chex.Array:
    """
    Records any relevant button is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        return jnp.array(Action.LEFT)
    elif keys[pygame.K_RIGHT]:
        return jnp.array(Action.RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(Action.FIRE)
    elif keys[pygame.K_UP]:
        return jnp.array(Action.UP)
    elif keys[pygame.K_DOWN]:
        return jnp.array(Action.DOWN)
    else:
        return jnp.array(Action.NOOP)


class JaxVideoPinball(
    JaxEnvironment[
        VideoPinballState,
        VideoPinballObservation,
        VideoPinballInfo,
        VideoPinballConstants,
    ]
):
    def __init__(
        self,
        consts: VideoPinballConstants | None = None,
        frameskip: int = 0,
        reward_funcs: list[Callable] | None = None,
    ):
        """
        Initialize the VideoPinball environment.

        Parameters
        ----------
        consts : VideoPinballConstants | None
            Configuration constants for the environment. When None, a default
            VideoPinballConstants is created to ensure a consistent object is
            provided to both the superclass and the renderer.
        frameskip : int
            Temporal resolution control (number of frames to skip). Kept on the
            initializer for API compatibility and potential superclass usage.
        reward_funcs : list[Callable] | None
            Optional list of reward shaping callables. Converted to a tuple to
            make the collection immutable and inexpensive to compare or hash.

        Notes
        -----
        - reward_funcs is stored as a tuple when provided to prevent accidental
          mutation after initialization and to allow stable equality semantics.
        - action_set is explicitly defined here to make the environment's discrete
          action space explicit and self-contained.
        - renderer is constructed with the same consts to guarantee rendering uses
          the identical configuration as the runtime environment.

        Attributes
        ----------
        reward_funcs : tuple[Callable, ...] | None
            Immutable sequence of reward functions (or None if not supplied).
        action_set : set[Action]
            The set of allowed actions for this environment.
        renderer : VideoPinballRenderer
            Renderer instance configured with the environment constants.
        """
        consts = consts or VideoPinballConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = {
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
            Action.LEFTFIRE,
            Action.RIGHTFIRE,
        }
        self.renderer = VideoPinballRenderer(consts=consts)

    def reset(self, key) -> Tuple[VideoPinballObservation, VideoPinballState]:
        """
        Reset the environment to a deterministic starting state using the provided PRNG key.

        Parameters
        ----------
        key : jnp.ndarray
            JAX PRNG key used to seed stochastic behavior; stored in the returned state
            so future randomness is reproducible.

        Returns
        -------
        Tuple[VideoPinballObservation, VideoPinballState]
            initial_obs, state
            - initial_obs: observation derived from the freshly initialized state (produced
              with self._get_observation) so the caller sees a view that matches internal state.
            - state: full game state with fields set to gameplay-appropriate defaults, chosen
              to establish consistent startup behavior

        Notes
        -----
        This function intentionally enforces deterministic initial conditions; any episode
        variation should originate from the provided PRNG key or subsequent agent actions.
        """

        state = VideoPinballState(
            ball_x=jnp.array(self.consts.BALL_START_X, dtype=jnp.float32),
            ball_y=jnp.array(self.consts.BALL_START_Y, dtype=jnp.float32),
            ball_vel_x=jnp.array(0.0, dtype=jnp.float32),
            ball_vel_y=jnp.array(0.0, dtype=jnp.float32),
            ball_direction=jnp.array(0, dtype=jnp.int32),
            left_flipper_angle=jnp.array(0, dtype=jnp.int32),
            right_flipper_angle=jnp.array(0, dtype=jnp.int32),
            left_flipper_counter=jnp.array(0, dtype=jnp.int32),
            right_flipper_counter=jnp.array(0, dtype=jnp.int32),
            left_flipper_active=jnp.array(True, dtype=jnp.bool_),
            right_flipper_active=jnp.array(True, dtype=jnp.bool_),
            plunger_position=jnp.array(0, dtype=jnp.int32),
            plunger_power=jnp.array(0, dtype=jnp.float32),
            score=jnp.array(0, dtype=jnp.int32),
            lives_lost=jnp.array(1, dtype=jnp.int32),
            bumper_multiplier=jnp.array(1, dtype=jnp.int32),
            active_targets=jnp.array([True, True, True, False], dtype=jnp.bool_),
            target_cooldown=jnp.array(-1, dtype=jnp.int32),
            special_target_cooldown=jnp.array(-120, dtype=jnp.int32),
            atari_symbols=jnp.array(0, dtype=jnp.int32),
            rollover_counter=jnp.array(1, dtype=jnp.int32),
            rollover_enabled=jnp.array(False, dtype=jnp.bool_),
            step_counter=jnp.array(0, dtype=jnp.int32),
            ball_in_play=jnp.array(False, dtype=jnp.bool_),
            respawn_timer=jnp.array(0, dtype=jnp.int32),
            color_cycling=jnp.array(0, dtype=jnp.int32),
            tilt_mode_active=jnp.array(False, dtype=jnp.bool_),
            tilt_counter=jnp.array(0, dtype=jnp.int32),
            rng_key=key,
        )

        initial_obs = self._get_observation(state)
        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _color_cycler(self, color):
        """Cycles through different flashing colors if a special target was hit."""

        new_color = jax.lax.cond(
            color > 0,  # Condition 1: is value > 0?
            lambda x: color - 1,  # True branch for Condition 1
            lambda x: jnp.where(  # False branch for Condition 1 (nested cond)
                color < 0,  # Condition 2: is value < 0?
                color + 1,  # True branch for Condition 2
                color,  # False branch for Condition 2 (must be == 0)
            ),
            None,  # Operand for outer cond (not used by branch funcs)
        )

        new_color = jnp.where(
            new_color == -1, jnp.array(14).astype(jnp.int32), new_color
        )

        return new_color

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: VideoPinballState, action: chex.Array
    ) -> Tuple[
        VideoPinballObservation, VideoPinballState, float, bool, VideoPinballInfo
    ]:
        """
        Description
        ----------
        Execute one environment timestep given a discrete action and the previous state.
        This function is the JAX-jitted entry point for environment dynamics and orchestrates
        the per-frame subsystems (plunger, flippers, ball physics, collisions, scoring,
        cooldowns, respawn and bookkeeping).

        Parameters
        ----------
        state : VideoPinballState
            Full environment state before applying the action. Must contain the PRNG key
            so stochastic components remain reproducible.
        action : chex.Array
            A JaxAtariAction (integer) representing the discrete action to take.

        Returns
        ----------
        observation : VideoPinballObservation
            Observation derived from the updated state.
        new_state : VideoPinballState
            Complete environment state after applying the action.
        reward : float
            Environment reward (difference in score by default).
        done : bool
            Whether the episode has terminated (out of lives and all points counted).
        info : VideoPinballInfo
            Additional info on the game state.

        Design Notes / Rationale
        ------------------------
        - Accepting the full `state` (including rng_key) ensures deterministic,
            reproducible randomness when splitting keys inside the step.
        """

        # Give different rng keys to different stochastic parts of the environment
        rng_key, ball_step_key = jrandom.split(state.rng_key)

        # Check if action is LEFT_FIRE or RIGHT_FIRE
        action_is_left_fire = action == Action.LEFTFIRE
        action_is_right_fire = action == Action.RIGHTFIRE
        action_is_tilt = jnp.logical_or(action_is_left_fire, action_is_right_fire)

        # If tilt mode is active, only allow LEFT_FIRE or RIGHT_FIRE (nudging)
        action = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(action_is_tilt), state.tilt_mode_active),
            lambda a: Action.NOOP,
            lambda a: a,
            action,
        )
        # if respawn timer is still going, discard all actions
        action = jax.lax.cond(
            state.respawn_timer > 0,
            lambda a: Action.NOOP,
            lambda a: a,
            action,
        )

        # Step 1: Update Plunger and Flippers
        plunger_position, new_plunger_power = self._plunger_step(state, action)
        # Update plunger power only if it is > 0
        plunger_power = jax.lax.cond(
            new_plunger_power > 0,
            lambda x: x,
            lambda _: state.plunger_power,
            operand=new_plunger_power,
        )
        (
            left_flipper_angle,
            right_flipper_angle,
            left_flipper_counter,
            right_flipper_counter,
        ) = self._flipper_step(state, action)

        # test_flippers = jnp.logical_and(action == Action.FIRE, jnp.logical_and(state.ball_x == BALL_START_X, state.ball_y == BALL_START_Y))
        # Step 2: Update ball position and velocity
        (
            ball_x,
            ball_y,
            ball_direction,
            ball_vel_x,
            ball_vel_y,
            ball_in_play,
            scoring_list,
            tilt_mode_active,
            tilt_counter,
            left_flipper_active,
            right_flipper_active,
        ) = self._ball_step(
            state,
            new_plunger_power,
            action,
            ball_step_key,
        )

        # Step 3: Check if ball is in the gutter or in plunger hole
        ball_in_gutter = ball_y > 192
        ball_reset = jnp.logical_or(
            ball_in_gutter,
            jnp.logical_and(ball_x > 148, ball_y > 129),
        )

        # Step 4: Update scores and handle special objects
        (
            score,
            active_targets,
            atari_symbols,
            rollover_counter,
            rollover_enabled,
            color_cycling,
        ) = self._process_objects_hit(
            state,
            scoring_list,
        )

        # Update target cooldowns and color cycling
        (
            active_targets,
            target_cooldown,
            special_target_cooldown,
            bumper_multiplier,
            color_cycling,
        ) = self._handle_target_cooldowns(state, active_targets, color_cycling)

        # Step 5: Reset ball if it went down the gutter
        current_values = (
            ball_x,
            ball_y,
            ball_vel_x,
            ball_vel_y,
            tilt_mode_active,
        )

        (
            ball_x_final,
            ball_y_final,
            ball_vel_x_final,
            ball_vel_y_final,
            tilt_mode_active,
        ) = jax.lax.cond(
            ball_reset,
            lambda x: self._reset_ball(state),
            lambda x: x,
            operand=current_values,
        )

        respawn_timer = jax.lax.cond(
            ball_in_gutter,
            lambda at: self._calc_respawn_timer(state),
            lambda at: state.respawn_timer,
            active_targets,
        )

        score = jnp.where(
            tilt_mode_active, state.score, score
        )  # No new points in tilt mode

        (
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            lives_lost,
            active_targets,
            special_target_cooldown,
            tilt_mode_active,
            tilt_counter,
        ) = jax.lax.cond(
            respawn_timer > 0,
            lambda rt, rc, s, asym, l, at, stc, tma, tmc: self._handle_ball_in_gutter(
                rt, rc, s, asym, l, at, stc, tma, tmc
            ),
            lambda rt, rc, s, asym, l, at, stc, tma, tmc: (
                rt,
                rc,
                s,
                asym,
                l,
                at,
                stc,
                tma,
                tmc,
            ),
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            state.lives_lost,
            active_targets,
            special_target_cooldown,
            tilt_mode_active,
            tilt_counter,
        )

        ball_in_play = jnp.where(
            jnp.logical_or(ball_reset, respawn_timer > 0),
            jnp.array(False),
            ball_in_play,
        )

        # Update color for color cycling if special target was hit
        color_cycling = self._color_cycler(color_cycling)

        new_state = VideoPinballState(
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            ball_direction=ball_direction,
            left_flipper_angle=left_flipper_angle,
            right_flipper_angle=right_flipper_angle,
            left_flipper_counter=left_flipper_counter,
            right_flipper_counter=right_flipper_counter,
            left_flipper_active=left_flipper_active,
            right_flipper_active=right_flipper_active,
            plunger_position=plunger_position,
            plunger_power=plunger_power,
            score=score,
            lives_lost=lives_lost,
            bumper_multiplier=bumper_multiplier,
            active_targets=active_targets,
            target_cooldown=target_cooldown,
            special_target_cooldown=special_target_cooldown,
            atari_symbols=atari_symbols,
            rollover_counter=rollover_counter,
            rollover_enabled=rollover_enabled,
            step_counter=jnp.array(state.step_counter + 1).astype(jnp.int32),
            ball_in_play=ball_in_play,
            respawn_timer=respawn_timer,
            color_cycling=color_cycling,
            tilt_mode_active=tilt_mode_active,
            tilt_counter=tilt_counter,
            rng_key=rng_key,
            # obs_stack=None,
        )

        done = self._get_done(new_state)
        env_reward = self._get_reward(state, new_state)
        all_rewards = self._get_all_reward(state, new_state)
        info = self._get_info(new_state, all_rewards)
        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoPinballState):
        """
        Description
         ----------
        Produce a VideoPinballObservation that encodes the current logical scene as
        fixed-format integer arrays. Each entity is represented as [x, y, w, h, active]
        and grouped by category (ball, flippers, spinners, targets, bumpers, rollovers,
        tilt plugs, plunger). Converting to jnp.int32 here ensures a stable, JAX-friendly
        dtype for downstream vectorized computations and comparisons.

        Parameters
         ----------
        state : VideoPinballState
            Current game state (positions, angles, target/rollover/bumper flags,
            plunger position, score, multipliers, etc.).
            - Types are preserved from the state where meaningful (e.g. active flags)
                but cast to int32 when placed into observation arrays to maintain a
                homogeneous numeric representation for JAX transformations.

        Returns
         ----------
        VideoPinballObservation
            Structured observation with fields:
            - ball: jnp.ndarray shape (5,)  -> [x, y, w, h, active]
            - flippers: jnp.ndarray shape (2,5)
              (each flipper computed by taking min/max extents across two scene
              objects for its current angle to form a single bounding box)
            - spinners: jnp.ndarray shape (2,5)
            - targets: jnp.ndarray shape (4,5)
              (active flags taken from state.active_targets)
            - bumpers: jnp.ndarray shape (3,5)
            - rollovers: jnp.ndarray shape (2,5)
            - tilt_mode_hole_plugs: jnp.ndarray shape (2,5)
            - plunger: jnp.ndarray shape (5,) (height depends on state.plunger_position)
            - score, lives_lost, atari_symbols, bumper_multiplier, rollover_counter,
              color_cycling, tilt_mode_active: all jnp.int32 scalars

         Design Notes / Rationale
         ----------
        - Canonical layout: using a consistent [x,y,w,h,active] tuple for all entities
            is used for compatibility with space.Box.
        """

        ball = EntityState(
            x=state.ball_x.astype(jnp.int32),
            y=state.ball_y.astype(jnp.int32),
            w=jnp.array(2),
            h=jnp.array(4),
            active=jnp.array(1),
        )

        ball = jnp.array([ball.x, ball.y, ball.w, ball.h, ball.active])

        # There are two scene objects for every flipper angle
        left_flipper_bounding_boxes = (
            self.consts.FLIPPERS[state.left_flipper_angle],
            self.consts.FLIPPERS[state.left_flipper_angle + 4],
        )
        right_flipper_bounding_boxes = (
            self.consts.FLIPPERS[state.right_flipper_angle + 8],
            self.consts.FLIPPERS[state.right_flipper_angle + 12],
        )
        left_flipper_x = jnp.min(
            jnp.array(
                [left_flipper_bounding_boxes[0][2], left_flipper_bounding_boxes[1][2]]
            )
        )
        left_flipper_w = (
            jnp.max(
                jnp.array(
                    [
                        left_flipper_bounding_boxes[0][2]
                        + left_flipper_bounding_boxes[0][0],
                        left_flipper_bounding_boxes[1][2]
                        + left_flipper_bounding_boxes[1][0],
                    ]
                )
            )
            - left_flipper_x
        )
        left_flipper_y = jnp.min(
            jnp.array(
                [left_flipper_bounding_boxes[0][3], left_flipper_bounding_boxes[1][3]]
            )
        )
        left_flipper_h = (
            jnp.max(
                jnp.array(
                    [
                        left_flipper_bounding_boxes[0][3]
                        + left_flipper_bounding_boxes[0][1],
                        left_flipper_bounding_boxes[1][3]
                        + left_flipper_bounding_boxes[1][1],
                    ]
                )
            )
            - left_flipper_y
        )

        right_flipper_x = jnp.min(
            jnp.array(
                [right_flipper_bounding_boxes[0][2], right_flipper_bounding_boxes[1][2]]
            )
        )
        right_flipper_w = (
            jnp.max(
                jnp.array(
                    [
                        right_flipper_bounding_boxes[0][2]
                        + right_flipper_bounding_boxes[0][0],
                        right_flipper_bounding_boxes[1][2]
                        + right_flipper_bounding_boxes[1][0],
                    ]
                )
            )
            - right_flipper_x
        )
        right_flipper_y = jnp.min(
            jnp.array(
                [right_flipper_bounding_boxes[0][3], right_flipper_bounding_boxes[1][3]]
            )
        )
        right_flipper_h = (
            jnp.max(
                jnp.array(
                    [
                        right_flipper_bounding_boxes[0][3]
                        + right_flipper_bounding_boxes[0][1],
                        right_flipper_bounding_boxes[1][3]
                        + right_flipper_bounding_boxes[1][1],
                    ]
                )
            )
            - right_flipper_y
        )

        left_flipper = EntityState(
            x=left_flipper_x,
            y=left_flipper_y,
            w=left_flipper_w,
            h=left_flipper_h,
            active=jnp.array(1),
        )
        right_flipper = EntityState(
            x=right_flipper_x,
            y=right_flipper_y,
            w=right_flipper_w,
            h=right_flipper_h,
            active=jnp.array(1),
        )

        flippers = jnp.array(
            [
                jnp.array(
                    [
                        entity.x,
                        entity.y,
                        entity.w,
                        entity.h,
                        entity.active,
                    ]
                )
                for entity in [left_flipper, right_flipper]
            ]
        )

        # Left, Middle, Right Diamonds / Lit up Targets

        # Left Target
        left_target = EntityState(
            x=self.consts.LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.LEFT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_width,
            h=self.consts.LEFT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_height,
            active=state.active_targets[0].astype(jnp.int32),
        )
        # Middle Target
        middle_target = EntityState(
            x=self.consts.MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.MIDDLE_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_width,
            h=self.consts.MIDDLE_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_height,
            active=state.active_targets[1].astype(jnp.int32),
        )

        # Right Target
        right_target = EntityState(
            x=self.consts.RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.RIGHT_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_width,
            h=self.consts.RIGHT_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_height,
            active=state.active_targets[2].astype(jnp.int32),
        )

        # Special Target
        special_target = EntityState(
            x=self.consts.SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.SPECIAL_LIT_UP_TARGET_LARGE_HORIZONTAL_SCENE_OBJECT.hit_box_width,
            h=self.consts.SPECIAL_LIT_UP_TARGET_LARGE_VERTICAL_SCENE_OBJECT.hit_box_height,
            active=state.active_targets[3].astype(jnp.int32),
        )

        targets = jnp.array(
            [
                jnp.array(
                    [
                        entity.x,
                        entity.y,
                        entity.w,
                        entity.h,
                        entity.active,
                    ]
                )
                for entity in [left_target, middle_target, right_target, special_target]
            ]
        )

        # Spinners
        # Left Spinner
        left_spinner = EntityState(
            x=self.consts.LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_x_offset
            + self.consts.LEFT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_width
            - self.consts.LEFT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_x_offset,
            h=self.consts.LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_y_offset
            + self.consts.LEFT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_height
            - self.consts.LEFT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_y_offset,
            active=jnp.array(1),
        )

        # Right Spinner
        right_spinner = EntityState(
            x=self.consts.RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_x_offset
            + self.consts.RIGHT_SPINNER_RIGHT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_width
            - self.consts.RIGHT_SPINNER_LEFT_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_x_offset,
            h=self.consts.RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_y_offset
            + self.consts.RIGHT_SPINNER_BOTTOM_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_height
            - self.consts.RIGHT_SPINNER_TOP_POSITION_JOINED_PART_SCENE_OBJECT.hit_box_y_offset,
            active=jnp.array(1),
        )

        spinners = jnp.array(
            [
                jnp.array(
                    [
                        entity.x,
                        entity.y,
                        entity.w,
                        entity.h,
                        entity.active,
                    ]
                )
                for entity in [left_spinner, right_spinner]
            ]
        )

        # Bumpers
        # Left Bumper
        left_bumper = EntityState(
            x=self.consts.LEFT_BUMPER_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.LEFT_BUMPER_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.LEFT_BUMPER_SCENE_OBJECT.hit_box_width,
            h=self.consts.LEFT_BUMPER_SCENE_OBJECT.hit_box_height,
            active=jnp.array(1),
        )

        # Top Bumper
        top_bumper = EntityState(
            x=self.consts.TOP_BUMPER_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.TOP_BUMPER_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.TOP_BUMPER_SCENE_OBJECT.hit_box_width,
            h=self.consts.TOP_BUMPER_SCENE_OBJECT.hit_box_height,
            active=jnp.array(1),
        )

        # Right Bumper
        right_bumper = EntityState(
            x=self.consts.RIGHT_BUMPER_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.RIGHT_BUMPER_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.RIGHT_BUMPER_SCENE_OBJECT.hit_box_width,
            h=self.consts.RIGHT_BUMPER_SCENE_OBJECT.hit_box_height,
            active=jnp.array(1),
        )

        bumpers = jnp.array(
            [
                jnp.array(
                    [
                        entity.x,
                        entity.y,
                        entity.w,
                        entity.h,
                        entity.active,
                    ]
                )
                for entity in [left_bumper, top_bumper, right_bumper]
            ]
        )

        left_tilt_mode_hole_plug = EntityState(
            x=self.consts.TILT_MODE_HOLE_PLUG_LEFT.hit_box_x_offset,
            y=self.consts.TILT_MODE_HOLE_PLUG_LEFT.hit_box_y_offset,
            w=self.consts.TILT_MODE_HOLE_PLUG_LEFT.hit_box_width,
            h=self.consts.TILT_MODE_HOLE_PLUG_LEFT.hit_box_height,
            active=jnp.array(1),
        )
        right_tilt_mode_hole_plug = EntityState(
            x=self.consts.TILT_MODE_HOLE_PLUG_RIGHT.hit_box_x_offset,
            y=self.consts.TILT_MODE_HOLE_PLUG_RIGHT.hit_box_y_offset,
            w=self.consts.TILT_MODE_HOLE_PLUG_RIGHT.hit_box_width,
            h=self.consts.TILT_MODE_HOLE_PLUG_RIGHT.hit_box_height,
            active=jnp.array(1),
        )

        tilt_mode_hole_plugs = jnp.array(
            [
                jnp.array(
                    [
                        entity.x,
                        entity.y,
                        entity.w,
                        entity.h,
                        entity.active,
                    ]
                )
                for entity in [left_tilt_mode_hole_plug, right_tilt_mode_hole_plug]
            ]
        )

        # Rollovers
        # Left Rollover
        left_rollover = EntityState(
            x=self.consts.LEFT_ROLLOVER_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.LEFT_ROLLOVER_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.LEFT_ROLLOVER_SCENE_OBJECT.hit_box_width,
            h=self.consts.LEFT_ROLLOVER_SCENE_OBJECT.hit_box_height,
            active=jnp.array(1),
        )

        # Atari Rollover
        atari_rollover = EntityState(
            x=self.consts.ATARI_ROLLOVER_SCENE_OBJECT.hit_box_x_offset,
            y=self.consts.ATARI_ROLLOVER_SCENE_OBJECT.hit_box_y_offset,
            w=self.consts.ATARI_ROLLOVER_SCENE_OBJECT.hit_box_width,
            h=self.consts.ATARI_ROLLOVER_SCENE_OBJECT.hit_box_height,
            active=jnp.array(1),
        )

        rollovers = jnp.array(
            [
                jnp.array(
                    [
                        entity.x,
                        entity.y,
                        entity.w,
                        entity.h,
                        entity.active,
                    ]
                )
                for entity in [left_rollover, atari_rollover]
            ]
        )

        plunger = EntityState(
            x=jnp.array(149),
            y=jnp.array(134),
            w=jnp.array(2),
            h=jnp.array(2 * state.plunger_position + 1),
            active=jnp.array(1),
        )

        plunger = jnp.array(
            [plunger.x, plunger.y, plunger.w, plunger.h, plunger.active]
        )

        return VideoPinballObservation(
            ball=ball.astype(jnp.int32),
            spinners=spinners.astype(jnp.int32),
            flippers=flippers.astype(jnp.int32),
            plunger=plunger.astype(jnp.int32),
            targets=targets.astype(jnp.int32),
            bumpers=bumpers.astype(jnp.int32),
            rollovers=rollovers.astype(jnp.int32),
            tilt_mode_hole_plugs=tilt_mode_hole_plugs.astype(jnp.int32),
            score=state.score.astype(jnp.int32),
            lives_lost=state.lives_lost.astype(jnp.int32),
            atari_symbols=state.atari_symbols.astype(jnp.int32),
            bumper_multiplier=state.bumper_multiplier.astype(jnp.int32),
            rollover_counter=state.rollover_counter.astype(jnp.int32),
            color_cycling=state.color_cycling.astype(jnp.int32),
            tilt_mode_active=state.tilt_mode_active.astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: VideoPinballObservation) -> jnp.ndarray:
        """
        Description
            Convert a structured VideoPinballObservation into a single 1D jnp.int32 array.

        Parameters
         ----------
            obs : VideoPinballObservation
                Structured observation containing fields (ball, spinners, flippers, plunger, targets, bumpers, rollovers, tilt_mode_hole_plugs, score, lives_lost, atari_symbols, bumper_multiplier, rollover_counter, color_cycling, tilt_mode_active).

        Returns
         ----------
            jnp.ndarray
                One-dimensional jnp.int32 array with a stable concatenation order matching observation_space().

         Design Notes / Rationale
         ----------
            - Maintain a deterministic field order so downstream consumers get a consistent jax array.
        """
        return jnp.concatenate(
            [
                obs.ball.flatten(),
                obs.spinners.flatten(),
                obs.flippers.flatten(),
                obs.plunger.flatten(),
                obs.targets.flatten(),
                obs.bumpers.flatten(),
                obs.rollovers.flatten(),
                obs.tilt_mode_hole_plugs.flatten(),
                obs.score.flatten(),
                obs.lives_lost.flatten(),
                obs.atari_symbols.flatten(),
                obs.bumper_multiplier.flatten(),
                obs.rollover_counter.flatten(),
                obs.color_cycling.flatten(),
                obs.tilt_mode_active.flatten(),
            ]
        )

    def action_space(self) -> spaces.Discrete:
        """
        Returns the action space for VideoPinball.
        Actions are:
        0: NOOP
        1: FIRE
        2: RIGHT
        3: LEFT
        4: RIGHTFIRE
        5: LEFTFIRE
        """
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """
        Description
            The observation_space describes the structured, JAX-friendly view of the observations.
            It mirrors the VideoPinballObservation dataclass produced by _get_observation and
            provides deterministic shapes, dtypes, and bounds so agents and wrappers can rely on them.

        Parameters
            ----------
            None

        Returns
            ----------
            spaces.Dict
                A mapping of observation keys to Gym-like Box spaces. Each Box uses jnp.int32
                and objects in the game have fixed shapes (e.g. (x,y,w,h,active)).

         Design Notes / Rationale
            -----------------------
            - Shapes and dtypes are chosen to exactly match _get_observation and obs_to_flat_array.
            - Using jnp.int32 everywhere enforces a homogeneous numeric representation.
        """

        # Most objects comprise of (x, y, width, height, active)
        return spaces.Dict(
            {
                "ball": spaces.Box(low=0, high=210, shape=(5,), dtype=jnp.int32),
                "spinners": spaces.Box(
                    low=0, high=210, shape=(2, 5), dtype=jnp.int32
                ),  # 2 spinners
                "flippers": spaces.Box(
                    low=0, high=210, shape=(2, 5), dtype=jnp.int32
                ),  # 2 flippers
                "plunger": spaces.Box(low=0, high=210, shape=(5,), dtype=jnp.int32),
                "targets": spaces.Box(
                    low=0, high=210, shape=(4, 5), dtype=jnp.int32
                ),  # 4 targets
                "bumpers": spaces.Box(
                    low=0, high=210, shape=(3, 5), dtype=jnp.int32
                ),  # 3 bumpers
                "rollovers": spaces.Box(
                    low=0, high=210, shape=(2, 5), dtype=jnp.int32
                ),  # 2 rollovers
                "tilt_mode_hole_plugs": spaces.Box(
                    low=0, high=210, shape=(2, 5), dtype=jnp.int32
                ),  # 2 hole plugs
                "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
                "lives_lost": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
                "atari_symbols": spaces.Box(low=0, high=3, shape=(), dtype=jnp.int32),
                "bumper_multiplier": spaces.Box(
                    low=0, high=9, shape=(), dtype=jnp.int32
                ),
                "rollover_counter": spaces.Box(
                    low=0, high=9, shape=(), dtype=jnp.int32
                ),
                "color_cycling": spaces.Box(low=0, high=30, shape=(), dtype=jnp.int32),
                "tilt_mode_active": spaces.Box(
                    low=0, high=1, shape=(), dtype=jnp.int32
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(
        self, state: VideoPinballState, all_rewards: chex.Array = None
    ) -> VideoPinballInfo:
        """
        Description
            This helper assembles a small information dictionary (VideoPinballInfo)
            from the full game state. The function is lightweight and intended to
            provide information not part of the observation.

        Parameters
            ----------
            state : VideoPinballState
                Current environment state from which fields are sampled.
            all_rewards : Optional[chex.Array]
                Optional array of reward function outputs. Defaults to None.

        Returns
            ----------
            VideoPinballInfo
                A small dataclass holding runtime meta information:
                time, plunger_power, cooldowns, rollover / respawn flags, tilt counter
                and the optional per-step reward vector (over reward functions).
        """
        return VideoPinballInfo(
            time=state.step_counter,
            plunger_power=state.plunger_power,
            target_cooldown=state.target_cooldown,
            special_target_cooldown=state.special_target_cooldown,
            rollover_enabled=state.rollover_enabled,
            step_counter=state.step_counter,
            ball_in_play=state.ball_in_play,
            respawn_timer=state.respawn_timer,
            tilt_counter=state.tilt_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: VideoPinballState, state: VideoPinballState):
        """The reward is the difference in score between the previous and current state."""
        return jnp.subtract(state.score, previous_state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(
        self, previous_state: VideoPinballState, state: VideoPinballState
    ):
        """Calculate all rewards from the list of reward functions."""
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoPinballState) -> bool:
        return jnp.logical_and(state.lives_lost > 3, state.respawn_timer <= 0)

    def render(self, state) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _plunger_step(self, state: VideoPinballState, action: chex.Array) -> chex.Array:
        """
        Description
            Update plunger position and compute launch power in a JAX-compatible, functional manner.
            Only responds to FIRE when the ball is not in play; on FIRE it derives a speed from the
            current plunger position and then resets the position to represent a launched ball.

        Parameters
         ----------
            state: VideoPinballState
                Current env state. Only state.plunger_position and state.ball_in_play are consumed.
            action: chex.Array
                Discrete JaxAtariAction (e.g. Action.UP / Action.DOWN / Action.FIRE).

        Returns
         ----------
            Tuple[chex.Array, chex.Array]
                plunger_position:
                    Integer-like jnp.ndarray holding the updated plunger position (clamped to [0, PLUNGER_MAX_POSITION]).
                    Reset to 0 when the ball is in play or immediately after a FIRE that launches the ball.
                plunger_power:
                    Float jnp.ndarray representing the launch speed. Non-zero only when FIRE and ball not in play;
                    calculated as (plunger_position / PLUNGER_MAX_POSITION) * BALL_MAX_SPEED.

         Design Notes / Rationale
            - Explicitly zero position/power when state.ball_in_play is True to avoid stale values and ensure stable dtypes.
        """

        # if ball is not in play and DOWN was clicked, move plunger down
        plunger_position = jax.lax.cond(
            jnp.logical_and(
                state.plunger_position < self.consts.PLUNGER_MAX_POSITION,
                jnp.logical_and(
                    action == Action.DOWN, jnp.logical_not(state.ball_in_play)
                ),
            ),
            lambda s: s + 1,
            lambda s: s,
            operand=state.plunger_position,
        )

        # same for UP
        plunger_position = jax.lax.cond(
            jnp.logical_and(
                state.plunger_position > 0,
                jnp.logical_and(
                    action == Action.UP, jnp.logical_not(state.ball_in_play)
                ),
            ),
            lambda s: s - 1,
            lambda s: s,
            operand=plunger_position,
        )

        # Shoot the ball if FIRE was clicked
        plunger_power = jax.lax.cond(
            jnp.logical_and(action == Action.FIRE, jnp.logical_not(state.ball_in_play)),
            lambda s: s / self.consts.PLUNGER_MAX_POSITION * self.consts.BALL_MAX_SPEED,
            lambda s: 0.0,
            operand=plunger_position,
        )

        # Reset plunger position to 0 if it was fired
        plunger_position = jax.lax.cond(
            plunger_power > 0, lambda p: 0, lambda p: p, operand=plunger_position
        )

        plunger_position, plunger_power = jax.lax.cond(
            state.ball_in_play,
            lambda: (jnp.array(0), jnp.array(0.0)),
            lambda: (plunger_position, plunger_power),
        )

        return plunger_position, plunger_power

    @partial(jax.jit, static_argnums=(0,))
    def _flipper_step(self, state: VideoPinballState, action: chex.Array):
        """
        Description
            Compute and update flipper discrete states (4 possible angles) and cooldown counters.

        Parameters
            ----------
            state : VideoPinballState
                Current environment state containing flipper angles and counters.
            action : chex.Array
                Discrete action input indicating if a flipper control (LEFT, RIGHT, UP) is used.

        Returns
            ----------
            tuple[chex.Array, chex.Array, chex.Array, chex.Array]
                (left_flipper_angle, right_flipper_angle, left_flipper_counter, right_flipper_counter)
        """

        # Move left and right flippers up if LEFT/RIGHT/UP was clicked
        left_flipper_angle = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_or(action == Action.LEFT, action == Action.UP),
                state.left_flipper_angle < self.consts.FLIPPER_MAX_ANGLE,
            ),
            lambda a: a + 1,
            lambda a: a,
            operand=state.left_flipper_angle,
        )

        right_flipper_angle = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_or(action == Action.RIGHT, action == Action.UP),
                state.right_flipper_angle < self.consts.FLIPPER_MAX_ANGLE,
            ),
            lambda a: a + 1,
            lambda a: a,
            operand=state.right_flipper_angle,
        )

        # Move left and right flippers down if LEFT/RIGHT/UP was not clicked
        move_left_flipper_down = jnp.logical_and(
            jnp.logical_not(jnp.logical_or(action == Action.LEFT, action == Action.UP)),
            jnp.logical_and(
                state.left_flipper_angle > 0, state.left_flipper_counter == 0
            ),
        )
        left_flipper_angle = jax.lax.cond(
            move_left_flipper_down,
            lambda a: a - 1,
            lambda a: a,
            operand=left_flipper_angle,
        )
        countdown_left = jnp.logical_and(
            jnp.logical_not(jnp.logical_or(action == Action.LEFT, action == Action.UP)),
            state.left_flipper_counter > 0,
        )
        left_flipper_counter = jnp.where(
            countdown_left,
            state.left_flipper_counter - 1,
            jnp.where(
                left_flipper_angle > 0,
                jnp.where(left_flipper_angle == self.consts.FLIPPER_MAX_ANGLE, 2, 5),
                0,
            ),
        )

        move_right_flipper_down = jnp.logical_and(
            jnp.logical_not(
                jnp.logical_or(action == Action.RIGHT, action == Action.UP)
            ),
            jnp.logical_and(
                state.right_flipper_angle > 0, state.right_flipper_counter == 0
            ),
        )
        right_flipper_angle = jax.lax.cond(
            move_right_flipper_down,
            lambda a: a - 1,
            lambda a: a,
            operand=right_flipper_angle,
        )
        countdown_right = jnp.logical_and(
            jnp.logical_not(
                jnp.logical_or(action == Action.RIGHT, action == Action.UP)
            ),
            state.right_flipper_counter > 0,
        )
        right_flipper_counter = jnp.where(
            countdown_right,
            state.right_flipper_counter - 1,
            jnp.where(
                right_flipper_angle > 0,
                jnp.where(right_flipper_angle == self.consts.FLIPPER_MAX_ANGLE, 2, 5),
                0,
            ),
        )

        return (
            left_flipper_angle,
            right_flipper_angle,
            left_flipper_counter,
            right_flipper_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _cross2(
        self, ax: chex.Array, ay: chex.Array, bx: chex.Array, by: chex.Array
    ) -> chex.Array:
        """
        Compute the 2D cross product (determinant) of two vectors.

        Given vectors:
            A = (ax, ay)
            B = (bx, by)

        The scalar cross product is defined as:
            cross(A, B) = ax * by - ay * bx

        This quantity is useful for:
            - Determining orientation (clockwise/counterclockwise turn).
            - Computing intersection of line segments.
            - Measuring the "signed area" spanned by two vectors.

        Parameters
        ----------
        ax : chex.Array
            x-component of vector A.
        ay : chex.Array
            y-component of vector A.
        bx : chex.Array
            x-component of vector B.
        by : chex.Array
            y-component of vector B.

        Returns
        -------
        chex.Array
            Scalar value of the 2D cross product (signed area).
        """
        return ax * by - ay * bx

    @partial(jax.jit, static_argnums=(0,))
    def _intersect_edge(
        self,
        ball_movement: BallMovement,
        ax: chex.Array,
        ay: chex.Array,
        a_to_b_x: chex.Array,
        a_to_b_y: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Compute the intersection between a moving ball's trajectory and a line segment.

        The function determines:
          1. The normalized time `t` at which the ball intersects the line segment,
             relative to its current timestep.
          2. Whether this intersection is valid, i.e., occurs within the timestep and
             lies within the finite edge segment.

        Intersection is found by solving:
            ball_position(t) = edge_point(s)
        where:
            ball_position(t) = old_ball + t * trajectory
            edge_point(s) = (ax, ay) + s * (a_to_b_x, a_to_b_y)

        Parameters
        ----------
        ball_movement : BallMovement
            An object with attributes:
                - old_ball_x, old_ball_y : float
                  The starting position of the ball.
                - new_ball_x, new_ball_y : float
                  The position of the ball at the end of the timestep.
        ax : chex.Array
            x-coordinate of the start of the edge (point A).
        ay : chex.Array
            y-coordinate of the start of the edge (point A).
        a_to_b_x : chex.Array
            x-component of the edge vector (B - A).
        a_to_b_y : chex.Array
            y-component of the edge vector (B - A).

        Returns
        -------
        t : chex.Array
            Normalized intersection time (0 ≤ t ≤ 1).
            If no valid collision, returns `self.consts.T_ENTRY_NO_COLLISION`.
        valid : chex.Array
            Whether the intersection is valid:
                - Ball trajectory intersects the finite edge segment.
                - Collision occurs within the timestep.

        Notes
        -----
        - Uses cross products to solve the intersection in 2D.
        - If the trajectory and edge are parallel (denominator ~ 0),
          no unique intersection is reported.
        - The method can be extended to handle collinear overlap cases.
        """
        eps = 1e-8

        # Ball trajectory
        trajectory_x = ball_movement.new_ball_x - ball_movement.old_ball_x
        trajectory_y = ball_movement.new_ball_y - ball_movement.old_ball_y

        # Vector from ball start to edge start
        ball_to_a_x = ax - ball_movement.old_ball_x
        ball_to_a_y = ay - ball_movement.old_ball_y

        # Cross products
        denom = self._cross2(trajectory_x, trajectory_y, a_to_b_x, a_to_b_y)
        numer_t = self._cross2(ball_to_a_x, ball_to_a_y, a_to_b_x, a_to_b_y)
        numer_u = self._cross2(ball_to_a_x, ball_to_a_y, trajectory_x, trajectory_y)

        denom_nonzero = jnp.abs(denom) > eps

        def nonparallel_case():
            t = numer_t / denom
            u = numer_u / denom
            t_valid = (t >= 0.0) & (t <= 1.0)
            u_valid = (u >= 0.0) & (u <= 1.0)
            return t, t_valid & u_valid

        def parallel_case():
            # Parallel lines: no unique intersection.
            # You could extend this to handle collinear overlap, but here we just return invalid.
            return jnp.array(
                self.consts.T_ENTRY_NO_COLLISION, dtype=jnp.float32
            ), jnp.array(False)

        t, valid = jax.lax.cond(denom_nonzero, nonparallel_case, parallel_case)
        return t, valid

    @partial(jax.jit, static_argnums=(0,))
    def _calc_segment_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
    ) -> chex.Array:
        """
        Calculate the collision point and reflection of a moving ball against a line segment.

        This function determines:
          1. If and when the ball collides with the line segment AB.
          2. The collision point on the segment.
          3. The surface normal of the edge at the point of contact.
          4. The new reflected ball position after the bounce.
          5. A "hit point" record combining collision and scene data.

        Parameters
        ----------
        ball_movement : BallMovement
            An object with attributes:
                - old_ball_x, old_ball_y : float
                  Starting position of the ball at the beginning of the timestep.
                - new_ball_x, new_ball_y : float
                  Intended end position of the ball at the end of the timestep.
        scene_object : chex.Array
            Encoded scene object data (e.g., type, ID, metadata) that will be concatenated
            with the computed hit point for downstream processing.
        action : chex.Array
            (Currently unused in this method) — may represent player or system action
            influencing collision handling.
        ax, ay : chex.Array
            Coordinates of the first endpoint of the edge (point A).
        bx, by : chex.Array
            Coordinates of the second endpoint of the edge (point B).

        Returns
        -------
        hit_point : jax.numpy.ndarray
            A concatenated array of the form:
                [t, hit_x, hit_y, new_ball_x, new_ball_y, ...scene_object]
            where:
                t : chex.Array
                    Normalized collision time (0 ≤ t ≤ 1), or sentinel if no collision.
                hit_x, hit_y : chex.Array
                    Coordinates of the collision point on the edge.
                new_ball_x, new_ball_y : chex.Array
                    The ball’s next position after reflection.
                ...scene_object : chex.Array
                    The original scene object data concatenated to the collision record.

        Notes
        -----
        - Uses `_intersect_edge` to determine if and when the ball intersects the edge.
        - Reflection is computed using the surface normal of the edge.
        - If the edge is degenerate (very short), a fallback normal perpendicular to the
          trajectory is used.
        - A small correction (`+0.1 * normal`) shifts the hit point slightly inward to avoid
          immediate re-detection of the same collision.
        - If no collision occurs, returns a dummy hit point via `_dummy_calc_hit_point`.

        Steps
        -----
        1. Compute ball trajectory and edge vector AB.
        2. Check for intersection with the edge using `_intersect_edge`.
        3. If valid, compute:
            - Collision point (hit_x, hit_y).
            - Surface normal (perpendicular to AB), oriented against trajectory.
            - Reflected trajectory vector scaled by remaining distance.
            - New ball position after bounce.
        4. Assemble hit point array with collision data + scene object data.
        5. If invalid, return a dummy hit point.
        """
        eps = 1e-8

        # Trajectory vector
        trajectory_x = ball_movement.new_ball_x - ball_movement.old_ball_x
        trajectory_y = ball_movement.new_ball_y - ball_movement.old_ball_y

        # Edge AB
        a_to_b_x = bx - ax
        a_to_b_y = by - ay
        t, valid = self._intersect_edge(ball_movement, ax, ay, a_to_b_x, a_to_b_y)

        # If invalid, set t to sentinel
        t = jnp.where(valid, t, self.consts.T_ENTRY_NO_COLLISION)

        # Collision point
        hit_x = ball_movement.old_ball_x + t * trajectory_x
        hit_y = ball_movement.old_ball_y + t * trajectory_y

        # Surface normal (perpendicular to edge)
        surface_normal_x = -a_to_b_y
        surface_normal_y = a_to_b_x
        norm_len = jnp.sqrt(surface_normal_x**2 + surface_normal_y**2) + eps
        surface_normal_x = surface_normal_x / norm_len
        surface_normal_y = surface_normal_y / norm_len

        # Ensure normal points opposite to trajectory direction
        dot_product = trajectory_x * surface_normal_x + trajectory_y * surface_normal_y
        surface_normal_x = jnp.where(
            dot_product > 0, -surface_normal_x, surface_normal_x
        )
        surface_normal_y = jnp.where(
            dot_product > 0, -surface_normal_y, surface_normal_y
        )

        # Fallback: if edge nearly degenerate, use perpendicular to trajectory
        d_traj = (
            jnp.sqrt(trajectory_x * trajectory_x + trajectory_y * trajectory_y) + eps
        )
        near_zero_normal = norm_len < 1e-6
        surface_normal_x = jnp.where(
            near_zero_normal, trajectory_x / d_traj, surface_normal_x
        )
        surface_normal_y = jnp.where(
            near_zero_normal, trajectory_y / d_traj, surface_normal_y
        )

        # Reflect trajectory
        velocity_normal_prod = (
            trajectory_x * surface_normal_x + trajectory_y * surface_normal_y
        )
        reflected_x = trajectory_x - 2.0 * velocity_normal_prod * surface_normal_x
        reflected_y = trajectory_y - 2.0 * velocity_normal_prod * surface_normal_y

        # Scale reflection by remaining distance (not traveled distance)
        traj_to_hit_x = hit_x - ball_movement.old_ball_x
        traj_to_hit_y = hit_y - ball_movement.old_ball_y
        d_hit = jnp.sqrt(traj_to_hit_x * traj_to_hit_x + traj_to_hit_y * traj_to_hit_y)
        r = jnp.clip((d_traj - d_hit) / d_traj, min=1e-2)
        reflected_x = r * reflected_x
        reflected_y = r * reflected_y

        # correction to avoid immediately re-detecting the hit_point
        hit_x = hit_x + surface_normal_x * 0.1
        hit_y = hit_y + surface_normal_y * 0.1
        # New ball position after reflection
        new_ball_x = hit_x + reflected_x
        new_ball_y = hit_y + reflected_y

        # Compose the hit point (like in slab code)
        hit_point = jnp.concatenate(
            [
                jnp.stack([t, hit_x, hit_y, new_ball_x, new_ball_y], axis=0),
                scene_object,
            ],
            axis=0,
        )

        # If no collision, return dummy
        hit_point = jax.lax.cond(
            ~valid,
            lambda: self._dummy_calc_hit_point(scene_object)[-1],
            lambda: hit_point,
        )

        return hit_point

    @partial(jax.jit, static_argnums=(0,))
    def _inside_slab_collision_branch(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ) -> chex.Array:
        """
        Handle ball-slab collision when the ball starts *inside* the rectangle.

        The ball is "rewound" step by step until it lies just outside the rectangle,
        then `_default_slab_collision_branch` is used to compute the actual collision.

        Parameters
        ----------
        ball_movement : BallMovement
            Object with old and new ball positions.
        scene_object : chex.Array
            Encoded axis-aligned rectangle [width, height, x_min, y_min, ...metadata].
        action : chex.Array
            Additional input, passed along for consistency.

        Returns
        -------
        hit_point : chex.Array
            Collision record, as returned by `_default_slab_collision_branch`.

        Notes
        -----
        - Prevents degenerate cases (division by zero) by clamping dx, dy with `1e-8`.
        - Rewind logic:
            - Compute factors `k` such that old position minus `k * (dx, dy)`
              is just outside the rectangle.
            - Update `BallMovement` with rewound old/new positions.
        - Ensures inside-starting collisions are handled consistently with outside-starting ones.
        """

        dx = jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        dy = jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)

        dx = jnp.where(jnp.abs(dx) < 1e-8, 1e-8, dx)
        dy = jnp.where(jnp.abs(dy) < 1e-8, 1e-8, dy)

        x_min, y_min = scene_object[2], scene_object[3]
        x_max, y_max = x_min + scene_object[0], y_min + scene_object[1]

        # Compute rewind factors until x or y just leaves the box
        kx1 = (ball_movement.old_ball_x - x_min) / dx
        kx2 = (ball_movement.old_ball_x - x_max) / dx
        ky1 = (ball_movement.old_ball_y - y_min) / dy
        ky2 = (ball_movement.old_ball_y - y_max) / dy

        ks = jnp.stack([kx1, kx2, ky1, ky2])
        k_min = jnp.max(
            jnp.minimum(ks, 0.0)
        )  # smallest non-positive that ensures outside
        k = jnp.floor(-k_min)  # integer steps

        new_x = ball_movement.old_ball_x - (k - 1) * dx
        new_y = ball_movement.old_ball_y - (k - 1) * dy
        old_x = ball_movement.old_ball_x - k * dx
        old_y = ball_movement.old_ball_y - k * dy

        ball_movement = BallMovement(
            old_ball_x=old_x, old_ball_y=old_y, new_ball_x=new_x, new_ball_y=new_y
        )

        return self._default_slab_collision_branch(ball_movement, scene_object, action)

    @partial(jax.jit, static_argnums=(0,))
    def _default_slab_collision_branch(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ) -> chex.Array:
        """
        Handle ball-axis-aligned rectangle (slab) collision when the ball starts outside.

        This method uses the *slab intersection algorithm* to:
          1. Compute entry/exit times for the ball’s trajectory along x and y.
          2. Determine if a collision occurs within the current timestep.
          3. Find the hit point on the rectangle boundary.
          4. Compute a reflection vector based on the surface normal of the side/corner hit.
          5. Produce a collision record containing hit data and the scene object.

        Parameters
        ----------
        ball_movement : BallMovement
            Object with attributes:
                - old_ball_x, old_ball_y : float
                  Ball position at start of timestep.
                - new_ball_x, new_ball_y : float
                  Ball position at end of timestep.
        scene_object : chex.Array
            Encoded axis-aligned rectangle of the form:
                [width, height, x_min, y_min, ...metadata].
        action : chex.Array
            Unused here but passed for compatibility with other collision functions.

        Returns
        -------
        hit_point : chex.Array
            Concatenated array:
                [t_entry, hit_x, hit_y, new_ball_x, new_ball_y, ...scene_object]
            where:
                t_entry : chex.Array
                    Normalized time of collision (0 ≤ t ≤ 1), sentinel if none.
                hit_x, hit_y : chex.Array
                    Collision coordinates on the rectangle boundary.
                new_ball_x, new_ball_y : chex.Array
                    Position after reflection.
                ...scene_object : chex.Array
                    Original scene object metadata.

        Notes
        -----
        - If no collision is detected (miss, outside timestep, or already past obstacle),
          returns a dummy hit point via `_dummy_calc_hit_point`.
        - Reflection logic distinguishes between:
            - Horizontal edge hit
            - Vertical edge hit
            - Corner hit (special case: normal aligned with trajectory).
        - Small epsilon (`1e-8`) prevents division by zero when computing times.
        """
        # Calculate trajectory of the ball in x and y direction
        trajectory_x = jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        trajectory_y = jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)

        tx1 = (scene_object[2] - ball_movement.old_ball_x) / (trajectory_x + 1e-8)
        tx2 = (scene_object[2] + scene_object[0] - ball_movement.old_ball_x) / (
            trajectory_x + 1e-8
        )
        ty1 = (scene_object[3] - ball_movement.old_ball_y) / (trajectory_y + 1e-8)
        ty2 = (scene_object[3] + scene_object[1] - ball_movement.old_ball_y) / (
            trajectory_y + 1e-8
        )

        # Calculate the time of intersection with the bounding box
        tmin_x = jnp.minimum(tx1, tx2)
        tmax_x = jnp.maximum(tx1, tx2)
        tmin_y = jnp.minimum(ty1, ty2)
        tmax_y = jnp.maximum(ty1, ty2)

        # Calculate the time of entry and exit
        t_entry = jnp.maximum(tmin_x, tmin_y)
        t_exit = jnp.minimum(tmax_x, tmax_y)

        # t_entry > t_exit means that the ball is not colliding with the bounding box, because it has already passed it
        # t_entry > 1 means the ball will collide with the obstacle but only in a future timestep
        no_collision = jnp.logical_or(t_entry > t_exit, t_entry > 1)
        no_collision = jnp.logical_or(no_collision, t_entry <= 0)

        hit_point_x = ball_movement.old_ball_x + t_entry * trajectory_x
        hit_point_y = ball_movement.old_ball_y + t_entry * trajectory_y

        # determine on which side the ball has hit the obstacle
        scene_object_half_height = scene_object[1] / 2.0
        scene_object_half_width = scene_object[0] / 2.0
        scene_object_middle_point_y = scene_object[3] + scene_object_half_height
        scene_object_middle_point_x = scene_object[2] + scene_object_half_width

        # distance of ball y to middle point of scene object
        d_middle_point_ball_y = jnp.abs(scene_object_middle_point_y - hit_point_y)
        d_middle_point_ball_x = jnp.abs(scene_object_middle_point_x - hit_point_x)

        # if ball hit the scene object to the top/bottom, this distance should be around half height of the scene object
        hit_horizontal = (
            jnp.abs(d_middle_point_ball_y - scene_object_half_height) < 1e-2
        )
        hit_vertical = jnp.abs(d_middle_point_ball_x - scene_object_half_width) < 1e-2
        hit_corner = jnp.logical_and(hit_horizontal, hit_vertical)

        d_trajectory = (
            jnp.sqrt(jnp.square(trajectory_x) + jnp.square(trajectory_y)) + 1e-8
        )

        surface_normal_x = jnp.where(
            hit_corner,
            trajectory_x / d_trajectory,
            jnp.where(hit_horizontal, jnp.array(0), jnp.array(1)),
        )
        surface_normal_y = jnp.where(
            hit_corner,
            trajectory_y / d_trajectory,
            jnp.where(hit_horizontal, jnp.array(1), jnp.array(0)),
        )

        # Calculate the dot product of the velocity and the surface normal
        velocity_normal_prod = (
            trajectory_x * surface_normal_x + trajectory_y * surface_normal_y
        )

        reflected_velocity_x = (
            trajectory_x - 2 * velocity_normal_prod * surface_normal_x
        )
        reflected_velocity_y = (
            trajectory_y - 2 * velocity_normal_prod * surface_normal_y
        )

        # Calculate the trajectory of the ball to the hit point
        trajectory_to_hit_point_x = jnp.subtract(hit_point_x, ball_movement.old_ball_x)
        trajectory_to_hit_point_y = jnp.subtract(hit_point_y, ball_movement.old_ball_y)

        d_hit_point = jnp.sqrt(
            jnp.square(trajectory_to_hit_point_x)
            + jnp.square(trajectory_to_hit_point_y)
        )

        r = jnp.clip(1 - d_hit_point / d_trajectory, min=1e-2)

        reflected_velocity_x = r * reflected_velocity_x
        reflected_velocity_y = r * reflected_velocity_y

        new_ball_x = hit_point_x + reflected_velocity_x
        new_ball_y = hit_point_y + reflected_velocity_y

        hit_point = jnp.concatenate(
            [
                jnp.stack(
                    [
                        t_entry,
                        hit_point_x,
                        hit_point_y,
                        new_ball_x,
                        new_ball_y,
                    ],
                    axis=0,
                ),
                scene_object,
            ],
            axis=0,
        )

        hit_point = jax.lax.cond(
            no_collision,
            lambda: self._dummy_calc_hit_point(scene_object)[-1],
            lambda: hit_point,
        )

        return hit_point

    @partial(jax.jit, static_argnums=(0,))
    def _calc_slab_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ) -> chex.Array:
        """
        Compute the collision between the ball and an axis-aligned rectangle (slab).

        Decides between two cases:
          1. **Inside collision**: The ball starts inside the rectangle.
          2. **Default collision**: The ball starts outside and may hit the rectangle.

        Parameters
        ----------
        ball_movement : BallMovement
            Object with old and new ball positions.
        scene_object : chex.Array
            Encoded axis-aligned rectangle [width, height, x_min, y_min, ...metadata].
        action : chex.Array
            Additional input passed down to branch functions.

        Returns
        -------
        hit_point : chex.Array
            Collision record as produced by `_default_slab_collision_branch`
            or `_inside_slab_collision_branch`.

        Notes
        -----
        - Inside case rewinds the ball outside the rectangle and reuses the default logic.
        """
        inside_x = jnp.logical_and(
            ball_movement.old_ball_x > scene_object[2],
            ball_movement.old_ball_x < scene_object[2] + scene_object[0],
        )
        inside_y = jnp.logical_and(
            ball_movement.old_ball_y > scene_object[3],
            ball_movement.old_ball_y < scene_object[3] + scene_object[1],
        )
        inside = jnp.logical_and(inside_x, inside_y)

        return jax.lax.cond(
            inside,
            self._inside_slab_collision_branch,
            self._default_slab_collision_branch,
            ball_movement,
            scene_object,
            action,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _is_inside_triangle(
        self,
        ball_movement: BallMovement,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
        cx: chex.Array,
        cy: chex.Array,
    ) -> chex.Array:
        """
        Determines whether the ball's position lies inside the triangle defined by points
        (A, B, C). This is useful for detecting swept collisions with flippers, where the
        flipper motion sweeps out a triangular region.

        Method:
        - Computes the signed area of the triangle to reject degenerate cases.
        - Uses cross products (AB x AP, BC x BP, CA x CP) to determine if the ball lies
          consistently on the same side of all edges.
        - Allows a small epsilon tolerance to handle numerical stability.

        Args:
            ball_movement: The ball's motion, storing old and new positions.
            ax, ay: Coordinates of triangle vertex A.
            bx, by: Coordinates of triangle vertex B.
            cx, cy: Coordinates of triangle vertex C.

        Returns:
            Boolean array (chex.Array): True if the ball's position is inside the triangle
            (and the triangle is non-degenerate), False otherwise.
        """
        eps = 1e-8

        # signed area of triangle (A,B,C)
        area = jnp.subtract(
            jnp.multiply(jnp.subtract(bx, ax), jnp.subtract(cy, ay)),
            jnp.multiply(jnp.subtract(by, ay), jnp.subtract(cx, ax)),
        )

        # reject degenerate triangles
        non_degenerate = jnp.abs(area) > eps

        # cross products: cross(edge_vector, point_vector_from_edge_start)
        d1 = jnp.subtract(
            jnp.multiply(
                jnp.subtract(bx, ax), jnp.subtract(ball_movement.old_ball_y, ay)
            ),
            jnp.multiply(
                jnp.subtract(by, ay), jnp.subtract(ball_movement.old_ball_x, ax)
            ),
        )  # AB x AP
        d2 = jnp.subtract(
            jnp.multiply(
                jnp.subtract(cx, bx), jnp.subtract(ball_movement.old_ball_y, by)
            ),
            jnp.multiply(
                jnp.subtract(cy, by), jnp.subtract(ball_movement.old_ball_x, bx)
            ),
        )  # BC x BP
        d3 = jnp.subtract(
            jnp.multiply(
                jnp.subtract(ax, cx), jnp.subtract(ball_movement.old_ball_y, cy)
            ),
            jnp.multiply(
                jnp.subtract(ay, cy), jnp.subtract(ball_movement.old_ball_x, cx)
            ),
        )  # CA x CP

        # inside if all crosses are non-negative OR all non-positive (allowing small epsilon)
        all_non_neg = (d1 >= -eps) & (d2 >= -eps) & (d3 >= -eps)
        all_non_pos = (d1 <= eps) & (d2 <= eps) & (d3 <= eps)

        inside = (all_non_neg | all_non_pos) & non_degenerate

        return inside

    @partial(jax.jit, static_argnums=(0,))
    def _calc_swept_collision(
        self,
        t_entry: chex.Array,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        px: chex.Array,
        py: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
    ) -> chex.Array:
        """
        Calculates the collision response when the ball collides with a flipper that is
        currently sweeping (i.e., rotating about a pivot). This function accounts for the
        angular velocity of the moving flipper surface.

        Method:
        - Computes the ball's hit position at time `t_entry`.
        - Derives the vector from the pivot point to the hit position.
        - Determines the sweep direction (up or down) based on flipper endpoints.
        - Computes the surface normal at the hit point, adjusted by sweep direction.
        - Incorporates tangential velocity of the moving flipper edge.
        - Reflects the relative velocity of the ball against the moving surface.
        - Computes the ball's new world velocity and position after reflection.

        Args:
            t_entry: Normalized collision time (0-1 of trajectory).
            ball_movement: The ball's old and new positions.
            scene_object: Encoded flipper geometry/state.
            px, py: Pivot point of the flipper.
            ax, ay: First endpoint of the flipper segment.
            bx, by: Second endpoint of the flipper segment.

        Returns:
            chex.Array: Hit point vector containing:
                [t_entry, hit_x, hit_y, new_ball_x, new_ball_y, scene_object...].
        """
        eps = 1e-8

        # Ball trajectory
        trajectory_x = ball_movement.new_ball_x - ball_movement.old_ball_x
        trajectory_y = ball_movement.new_ball_y - ball_movement.old_ball_y

        # Collision position
        hit_x = ball_movement.old_ball_x + t_entry * trajectory_x
        hit_y = ball_movement.old_ball_y + t_entry * trajectory_y

        # Vector pivot → collision
        r_x = hit_x - px
        r_y = hit_y - py
        d_r = jnp.sqrt(r_x**2 + r_y**2) + eps

        orientation = jnp.sign(
            ax - px
        )  # +1 for CCW flipper (left), -1 for CW flipper (right)

        # Sweep direction (screen-space orientation)
        # -1 is up, 1 down (with flipped coordinate system)
        # if a is larger than b the sweep moves up
        sweep_dir = jnp.sign(by - ay)

        # Collision normal
        # Tangential direction at hit point (perpendicular to r)
        # accounting for sweeping direction:
        # the surface normal needs to point in the direction
        # of the half space where the reflection takes place.
        n_x = orientation * sweep_dir * (-r_y)
        n_y = orientation * sweep_dir * (r_x)
        n_len = jnp.sqrt(n_x**2 + n_y**2) + eps
        n_x /= n_len
        n_y /= n_len

        angular_velocity = self.consts.VELOCITY_ACCELERATION_VALUE
        # Tangential velocity of moving surface
        # (2 * angular_velocity = what felt good)
        u_x = 2 * angular_velocity * n_x * d_r
        u_y = 2 * angular_velocity * n_y * d_r

        # Relative velocity
        v_rel_x = trajectory_x - u_x
        v_rel_y = trajectory_y - u_y

        # Reflect relative velocity
        dot_vn = v_rel_x * n_x + v_rel_y * n_y
        rv_rel_x = v_rel_x - 2.0 * dot_vn * n_x
        rv_rel_y = v_rel_y - 2.0 * dot_vn * n_y

        # Transform back to world (screen) frame
        rvx = rv_rel_x + u_x
        rvy = rv_rel_y + u_y

        # New ball position after reflection
        r = jnp.clip(1.0 - t_entry, min=1e-2)
        new_ball_x = hit_x + r * rvx
        new_ball_y = hit_y + r * rvy

        # Hit point info
        hit_point = jnp.concatenate(
            [
                jnp.stack([t_entry, hit_x, hit_y, new_ball_x, new_ball_y], axis=0),
                scene_object,
            ],
            axis=0,
        )

        return hit_point

    @partial(jax.jit, static_argnums=(0,))
    def _calc_swept_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
        px: chex.Array,
        py: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
    ):
        """
        Computes the collision hit point between the ball and a swept flipper arc,
        represented by pivot P (px, py) and arc endpoints A(ax, ay), B(bx, by).

        Cases considered:
        1. The ball lies inside triangle (P, A, B) → swept collision.
        2. Ball collides with arc AB (approximated as edge AB) → swept collision.
        3. Ball collides with edge PA or PB → segment collision.
        4. No collision → return dummy hit point.

        The function selects the earliest valid collision among the above and delegates
        to `_calc_swept_collision` (for swept arcs) or `_calc_segment_hit_point` (for
        static edges).

        Args:
            ball_movement: The ball's old and new positions.
            scene_object: Encoded flipper geometry/state.
            action: Current player action (affects flipper state).
            px, py: Pivot point of the flipper.
            ax, ay: Flipper arc endpoint A.
            bx, by: Flipper arc endpoint B.

        Returns:
            chex.Array: Hit point vector with collision data, or dummy if no collision.
        """
        # if ball is inside, also do a swept collision
        is_inside = self._is_inside_triangle(
            ball_movement,
            px,
            py,
            ax,
            ay,
            bx,
            by,
        )
        inside_t_entry = jnp.where(is_inside, 0.0, self.consts.T_ENTRY_NO_COLLISION)

        # most of the time, the ball does not collide with he flippers, so instead of
        # calculating a full hit point for each of these cases, this will suffice:
        # Arc collision (approximated by edge AB)
        a_to_b_x = bx - ax
        a_to_b_y = by - ay
        arc_t_entry, arc_valid = self._intersect_edge(
            ball_movement, ax, ay, a_to_b_x, a_to_b_y
        )
        arc_t_entry = jnp.where(
            arc_valid, arc_t_entry, self.consts.T_ENTRY_NO_COLLISION
        )

        # edge PA collision
        p_to_a_x = ax - px
        p_to_a_y = ay - py
        pa_t_entry, pa_valid = self._intersect_edge(
            ball_movement, px, py, p_to_a_x, p_to_a_y
        )
        pa_t_entry = jnp.where(pa_valid, pa_t_entry, self.consts.T_ENTRY_NO_COLLISION)

        # edge PB collision
        p_to_b_x = bx - px
        p_to_b_y = by - py
        pb_t_entry, pb_valid = self._intersect_edge(
            ball_movement, px, py, p_to_b_x, p_to_b_y
        )
        pb_t_entry = jnp.where(pb_valid, pb_t_entry, self.consts.T_ENTRY_NO_COLLISION)

        intersects_t_entry = jnp.array(
            [inside_t_entry, arc_t_entry, pa_t_entry, pb_t_entry]
        )

        return jax.lax.cond(
            jnp.all(intersects_t_entry == self.consts.T_ENTRY_NO_COLLISION),
            lambda: self._dummy_calc_hit_point(scene_object)[-1],
            lambda: jax.lax.switch(
                jnp.argmin(intersects_t_entry),
                [
                    lambda: self._calc_swept_collision(
                        intersects_t_entry[0],
                        ball_movement,
                        scene_object,
                        px,
                        py,
                        ax,
                        ay,
                        bx,
                        by,
                    ),
                    lambda: self._calc_swept_collision(
                        intersects_t_entry[1],
                        ball_movement,
                        scene_object,
                        px,
                        py,
                        ax,
                        ay,
                        bx,
                        by,
                    ),
                    lambda: self._calc_segment_hit_point(
                        ball_movement, scene_object, action, px, py, ax, ay
                    ),
                    lambda: self._calc_segment_hit_point(
                        ball_movement, scene_object, action, px, py, bx, by
                    ),
                ],
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _calc_flipper_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ):
        """
        Calculates ball-flipper collisions, handling both static and moving flipper states.

        Method:
        - Determines flipper identity (left/right) and current motion (up/down/neutral).
        - Selects the correct pivot and flipper endpoints based on discrete flipper
          position (indexed by scene_object).
        - If the flipper is moving, delegates to `_calc_swept_hit_point`.
        - If static, computes segment collision with `_calc_segment_hit_point`, resolving
          the top vs. bottom flipper segments if necessary.
        - Applies velocity scaling (dampening when static, amplification when moving).
        - Adds an angular velocity factor depending on collision location along the
          flipper length.

        Args:
            ball_movement: The ball's old and new positions.
            scene_object: Flipper object encoding position, type, and segment index.
            action: Current player action (used to determine flipper movement).

        Returns:
            Tuple:
              - velocity_factor (float): Scales post-collision velocity.
              - velocity_addition (float): Adds spin/boost based on flipper movement.
              - hit_point (chex.Array): Detailed collision data.
        """
        is_left_flipper = scene_object[5] == 9
        is_right_flipper = scene_object[5] == 10
        left_flipper_up = jnp.logical_or(action == Action.LEFT, action == Action.UP)
        right_flipper_up = jnp.logical_or(action == Action.RIGHT, action == Action.UP)
        flipper_at_max_pos = scene_object[6] % 4 == 3
        flipper_at_min_pos = scene_object[6] % 4 == 0
        is_bottom_segment = scene_object[6] % 8 < 4

        is_left_flipper_and_up = jnp.logical_and(
            jnp.logical_and(is_left_flipper, left_flipper_up),
            jnp.logical_not(flipper_at_max_pos),
        )
        is_right_flipper_and_up = jnp.logical_and(
            jnp.logical_and(is_right_flipper, right_flipper_up),
            jnp.logical_not(flipper_at_max_pos),
        )
        flipper_up = jnp.logical_or(is_left_flipper_and_up, is_right_flipper_and_up)

        is_left_flipper_and_down = jnp.logical_and(
            jnp.logical_and(is_left_flipper, jnp.logical_not(left_flipper_up)),
            jnp.logical_not(flipper_at_min_pos),
        )
        is_right_flipper_and_down = jnp.logical_and(
            jnp.logical_and(is_right_flipper, jnp.logical_not(right_flipper_up)),
            jnp.logical_not(flipper_at_min_pos),
        )
        flipper_down = jnp.logical_or(
            is_left_flipper_and_down, is_right_flipper_and_down
        )

        (px, py), (endx, endy) = self.consts.FLIPPER_SEGMENTS_SORTED[scene_object[6]]

        # next flipper position:
        (_, _), (next_pos_endx, next_pos_endy) = (
            jax.lax.cond(  # new angle of the line segment
                flipper_down,
                lambda: self.consts.FLIPPER_SEGMENTS_SORTED[scene_object[6] - 1],
                lambda: jax.lax.cond(
                    flipper_up,
                    lambda: self.consts.FLIPPER_SEGMENTS_SORTED[scene_object[6] + 1],
                    lambda: self.consts.FLIPPER_SEGMENTS_SORTED[scene_object[6]],
                ),
            )
        )

        # each flipper position consists of 2 line segments: One for the bottom side, one for the top.
        # We have to ensure that nothing weird happens inside of the flipper/at the edges
        (_, _), (other_endx, other_endy) = jax.lax.cond(
            is_bottom_segment,
            lambda: self.consts.FLIPPER_SEGMENTS_SORTED[scene_object[6] + 4],
            lambda: self.consts.FLIPPER_SEGMENTS_SORTED[scene_object[6] - 4],
        )

        flipper_moves = jnp.logical_or(flipper_up, flipper_down)

        @jax.jit
        def select_segment_hit_point(
            ball_movement, scene_object, action, ax, ay, bx, by, cx, cy, dx, dy
        ):
            """
            There is a small gap between the top and bottom line segment of the flippers (rounded tips).
            So we have to select the appropriate hit_point since the ball can hit the flipper in that gap.
            """
            ab_hit_point = self._calc_segment_hit_point(
                ball_movement, scene_object, action, ax, ay, bx, by
            )
            cd_hit_point = self._calc_segment_hit_point(
                ball_movement, scene_object, action, cx, cy, dx, dy
            )

            return jax.lax.cond(
                ab_hit_point[HitPointSelector.T_ENTRY]
                < cd_hit_point[HitPointSelector.T_ENTRY],
                lambda: ab_hit_point,
                lambda: cd_hit_point,
            )

        hit_point = jax.lax.cond(
            flipper_moves,
            lambda: self._calc_swept_hit_point(
                ball_movement,
                scene_object,
                action,
                px,
                py,
                endx,
                endy,
                next_pos_endx,
                next_pos_endy,
            ),
            lambda: select_segment_hit_point(
                ball_movement,
                scene_object,
                action,
                px,
                py,
                endx,
                endy,
                endx,
                endy,
                other_endx,
                other_endy,
            ),
        )

        velocity_factor = jnp.where(
            flipper_moves, 1.0, 1 - self.consts.VELOCITY_DAMPENING_VALUE
        )

        angular_velocity = (
            jnp.sqrt(
                (hit_point[HitPointSelector.X] - px) ** 2
                + (hit_point[HitPointSelector.Y] - py) ** 2
            )
            / jnp.sqrt((endx - px) ** 2 + (endy - py) ** 2)
            * 0.5
            + 0.5
        )
        velocity_addition = jnp.where(
            flipper_moves,
            jnp.where(flipper_up, angular_velocity, -angular_velocity),
            0.0,
        )

        return velocity_factor, velocity_addition, hit_point

    @partial(jax.jit, static_argnums=(0,))
    def _calc_spinner_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ):
        """
        Calculates collisions with spinner objects.

        Method:
        - Uses the standard slab method to find the hit point.
        - Determines the spinner center (left or right side).
        - Computes ball reflection vector from slab collision.
        - Adds angular velocity from spinner rotation (ω = π/2 per time step).
        - Normalizes and scales angular contribution by a constant acceleration factor.
        - Clips resulting velocity to the ball's maximum speed.
        - Adjusts the hit point so that the ball emerges outside the spinner bounding box
          (avoiding immediate re-collision).
        - Returns updated hit point and a velocity addition term (speed delta).

        Args:
            ball_movement: The ball's old and new positions.
            scene_object: Spinner geometry/state.
            action: Current player action (unused, kept for consistency).

        Returns:
            Tuple:
              - scalar velocity factor (float, always 1.0 here),
              - velocity_addition (float, delta in ball speed due to angular boost),
              - hit_point (chex.Array): Updated collision data with corrected ball position.
        """
        hit_point = self._calc_slab_hit_point(ball_movement, scene_object, action)

        # unless velocity is not cranked up this should suffice:
        is_left_spinner = hit_point[HitPointSelector.X] < self.consts.WIDTH / 2

        spinner_middle_point_x = jnp.where(
            is_left_spinner,
            self.consts.LEFT_SPINNER_MIDDLE_POINT[0],
            self.consts.RIGHT_SPINNER_MIDDLE_POINT[0],
        )
        spinner_middle_point_y = jnp.where(
            is_left_spinner,
            self.consts.LEFT_SPINNER_MIDDLE_POINT[1],
            self.consts.RIGHT_SPINNER_MIDDLE_POINT[1],
        )

        reflected_velocity_x = (
            hit_point[HitPointSelector.RX] - hit_point[HitPointSelector.X]
        )
        reflected_velocity_y = (
            hit_point[HitPointSelector.RY] - hit_point[HitPointSelector.Y]
        )

        omega = jnp.pi / 2  # spinner takes 4 time steps to do a full circle

        pivot_to_hit_x = hit_point[HitPointSelector.X] - spinner_middle_point_x
        pivot_to_hit_y = hit_point[HitPointSelector.Y] - spinner_middle_point_y
        angular_velocity_x = -omega * pivot_to_hit_y
        angular_velocity_y = omega * pivot_to_hit_x
        angular_velocity_norm = jnp.sqrt(angular_velocity_x**2 + angular_velocity_y**2)
        angular_velocity_x = (
            angular_velocity_x
            / angular_velocity_norm
            * self.consts.VELOCITY_ACCELERATION_VALUE
        )
        angular_velocity_y = (
            angular_velocity_y
            / angular_velocity_norm
            * self.consts.VELOCITY_ACCELERATION_VALUE
        )

        final_velocity_x = reflected_velocity_x + angular_velocity_x
        final_velocity_y = reflected_velocity_y + angular_velocity_y
        final_velocity_x = jnp.clip(
            final_velocity_x, -self.consts.BALL_MAX_SPEED, self.consts.BALL_MAX_SPEED
        )
        final_velocity_y = jnp.clip(
            final_velocity_y, -self.consts.BALL_MAX_SPEED, self.consts.BALL_MAX_SPEED
        )

        # velocity addition: return vector delta or magnitude delta — here: scalar magnitude delta
        reflected_speed = jnp.sqrt(reflected_velocity_x**2 + reflected_velocity_y**2)
        final_speed = jnp.sqrt(final_velocity_x**2 + final_velocity_y**2)
        velocity_addition = final_speed - reflected_speed

        #
        new_ball_x = hit_point[HitPointSelector.X] + final_velocity_x
        new_ball_y = hit_point[HitPointSelector.Y] + final_velocity_y
        dx_left = jnp.abs(new_ball_x - scene_object[2])
        dx_right = jnp.abs((scene_object[2] + scene_object[0]) - new_ball_x)
        dy_top = jnp.abs(new_ball_y - scene_object[3])
        dy_bottom = jnp.abs((scene_object[3] + scene_object[1]) - new_ball_y)

        old_ball_x = jnp.where(
            dx_left < dx_right, scene_object[2], scene_object[2] + scene_object[0]
        ).astype(jnp.float32)
        old_ball_y = jnp.where(
            dy_top < dy_bottom, scene_object[3], scene_object[3] + scene_object[1]
        ).astype(jnp.float32)
        new_ball_x = old_ball_x + final_velocity_x
        new_ball_y = old_ball_y + final_velocity_y

        return (
            1.0,
            velocity_addition,
            jnp.concatenate(
                [
                    jnp.stack(
                        [
                            hit_point[HitPointSelector.T_ENTRY],
                            old_ball_x,
                            old_ball_y,
                            new_ball_x,
                            new_ball_y,
                        ],
                        axis=0,
                    ),
                    hit_point[HitPointSelector.RY + 1 :],
                ],
                axis=0,
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _dummy_calc_hit_point(
        self,
        scene_object: chex.Array,
    ) -> chex.Array:
        """
        Returns a dummy (sentinel) hit point when no collision occurs.

        This is used as a fallback in functions like `_calc_segment_hit_point` or
        `_calc_swept_hit_point`, where the collision test determines that the
        trajectory does not intersect with the object.

        Method:
        - Returns a fixed hit point with:
            - `T_ENTRY_NO_COLLISION` as the entry time,
            - coordinates set to `-1.0` (invalid),
            - new ball positions set to `-1.0` (invalid),
            - and appends the original scene_object for compatibility.

        Args:
            scene_object: Encoded scene object properties, preserved for downstream logic.

        Returns:
            chex.Array containing the dummy hit point:
                [
                    T_ENTRY_NO_COLLISION,  # indicates no collision
                    -1.0, -1.0, -1.0, -1.0,  # invalid hit and reflection coordinates
                    ...scene_object          # appended for structural consistency
                ]
        """
        return (
            0.0,
            0.0,
            jnp.concatenate(
                [
                    jnp.stack(
                        [
                            self.consts.T_ENTRY_NO_COLLISION,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ],
                        axis=0,
                    ),
                    scene_object,
                ],
                axis=0,
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _calc_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ) -> chex.Array:
        """
        Dispatches hit point calculation for the ball against a scene object, based on its type.

        The method interprets the `scene_object[5]` (score_type) field and selects the
        appropriate collision handler:
          - Most objects (bumpers, targets, rollovers, holes) use the slab method
            (`_calc_slab_hit_point`).
          - Spinners use `_calc_spinner_hit_point` (adds angular velocity).
          - Flippers use `_calc_flipper_hit_point` (handles sweeping motion).
          - Passive objects return dampened velocities.

        Velocity dampening or addition is also applied depending on the type of object.

        Scene object format:
            [
                hit_box_width,    # 0
                hit_box_height,   # 1
                hit_box_x_offset, # 2
                hit_box_y_offset, # 3
                reflecting,       # 4
                score_type,       # 5 (dispatch key)
                variant           # 6
            ]

        Score type values (scene_object[5]):
            0  → Neutral obstacle (no score, dampened slab collision)
            1  → Bumper (slab collision, adds acceleration)
            2  → Spinner (custom collision with angular boost)
            3  → Left rollover (slab collision)
            4  → Atari rollover (slab collision)
            5  → Special lit-up target (slab collision)
            6  → Left lit-up target (slab collision)
            7  → Middle lit-up target (slab collision)
            8  → Right lit-up target (slab collision)
            9  → Left flipper (swept/flipper collision)
            10 → Right flipper (swept/flipper collision)
            11 → Tilt mode hole plug (slab collision)

        Args:
            ball_movement: The ball's trajectory (old → new position).
            scene_object: Encoded object data determining geometry, type, and behavior.
            action: Current player input, relevant for flipper collisions.

        Returns:
            chex.Array structured as:
                [
                    velocity_factor,   # scaling applied to ball velocity
                    velocity_addition, # extra velocity contribution (e.g. flippers, bumpers)
                    hit_point_data     # computed hit point (from the chosen method)
                ]

            The hit_point_data itself encodes:
                hit_point[0]: time of entry (t_entry)
                hit_point[1]: x position of collision
                hit_point[2]: y position of collision
                hit_point[3]: new_ball_x after reflection
                hit_point[4]: new_ball_y after reflection
                hit_point[5:]: scene_object properties

        Hint:
            Use the `HitPointSelector` enum to safely access hit point indices.
        """
        # 0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover, 5: Special Lit Up Target,
        # 6: Left Lit Up Target, 7:Middle Lit Up Target, 8: Right Lit Up Target, 9: Left Flipper, 10: Right Flipper, 11: Tilt Mode Hole Plug
        dampening_value = 1 - self.consts.VELOCITY_DAMPENING_VALUE
        no_addition = 0.0
        acceleration_value = self.consts.VELOCITY_ACCELERATION_VALUE
        no_factor = 1.0
        scene_object_center_x = scene_object[2] + scene_object[0] / 2
        scene_object_center_y = scene_object[3] + scene_object[1] / 2
        distance_to_center = jnp.sqrt(
            (scene_object_center_x - ball_movement.old_ball_x) ** 2
            + (scene_object_center_y - ball_movement.old_ball_y) ** 2
        )

        return jax.lax.cond(
            scene_object[5] == 0,
            lambda: (
                dampening_value,
                no_addition,
                self._calc_slab_hit_point(ball_movement, scene_object, action),
            ),
            lambda: jax.lax.cond(
                distance_to_center > 30,
                lambda: self._dummy_calc_hit_point(scene_object),
                lambda: jax.lax.switch(
                    scene_object[5],
                    [
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 0
                        lambda: (
                            no_factor,
                            acceleration_value,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 1
                        lambda: self._calc_spinner_hit_point(
                            ball_movement, scene_object, action
                        ),  # 2
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 3
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 4
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 5
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 6
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 7
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 8
                        lambda: self._calc_flipper_hit_point(
                            ball_movement, scene_object, action
                        ),  # 9
                        lambda: self._calc_flipper_hit_point(
                            ball_movement, scene_object, action
                        ),  # 10
                        lambda: (
                            dampening_value,
                            no_addition,
                            self._calc_slab_hit_point(
                                ball_movement, scene_object, action
                            ),
                        ),  # 11
                    ],
                ),
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _check_obstacle_hits(
        self,
        state: VideoPinballState,
        ball_movement: BallMovement,
        scoring_list: chex.Array,
        action: chex.Array,
        left_flipper_active: chex.Array,
        right_flipper_active: chex.Array,
    ) -> tuple[chex.Array, SceneObject]:

        # DISABLE NON-REFLECTING SCENCE OBJECTS THAT ARE NOT IN THE CURRENT GAME STATE
        ###############################################################################################
        # Scoring types:
        # 0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover, 5: Special Lit Up Target,
        # 6: Left Lit Up Target, 7:Middle Lit Up Target, 8: Right Lit Up Target, 9: Left Flipper, 10: Right Flipper

        # Disable inactive lit up targets (diamonds)
        non_reflecting_active = jnp.ones_like(
            self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 0], dtype=jnp.bool
        )

        is_left_lit_up_target_active = state.active_targets[0]
        is_middle_lit_up_target_active = state.active_targets[1]
        is_right_lit_up_target_active = state.active_targets[2]
        is_special_lit_up_target_active = state.active_targets[3]

        is_left_lit_up_target_non_refl = (
            self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 6
        )
        is_middle_lit_up_target_non_refl = (
            self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 7
        )
        is_right_lit_up_target_non_refl = (
            self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 8
        )
        is_special_lit_up_target_non_refl = (
            self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 5
        )

        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_left_lit_up_target_non_refl,
                jnp.logical_not(is_left_lit_up_target_active),
            ),
            False,
            non_reflecting_active,  # only select hit_point[0] since we only update t_entry
        )
        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_middle_lit_up_target_non_refl,
                jnp.logical_not(is_middle_lit_up_target_active),
            ),
            False,
            non_reflecting_active,
        )
        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_right_lit_up_target_non_refl,
                jnp.logical_not(is_right_lit_up_target_active),
            ),
            False,
            non_reflecting_active,
        )
        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_special_lit_up_target_non_refl,
                jnp.logical_not(is_special_lit_up_target_active),
            ),
            False,
            non_reflecting_active,
        )

        # DISABLE REFLECTING SCENCE OBJECTS THAT ARE NOT IN THE CURRENT GAME STATE
        ###############################################################################################
        reflecting_active = jnp.ones_like(
            self.consts.REFLECTING_SCENE_OBJECTS[:, 0], dtype=jnp.bool
        )

        # Disable inactive spinner parts
        spinner_state = jnp.remainder(
            state.step_counter, 8
        )  # 0: Bottom, 1: Right, 2: Top, 3: Left

        object_variant = self.consts.REFLECTING_SCENE_OBJECTS[:, 6]
        is_spinner = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 2
        is_left_flipper = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 9
        is_right_flipper = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 10
        is_hole_plug = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 11
        is_left_lit_up_target_refl = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 6
        is_middle_lit_up_target_refl = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 7
        is_right_lit_up_target_refl = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 8
        # spinner state only switches every other game step
        spinner_active = jnp.logical_or(
            object_variant * 2 == spinner_state,
            object_variant * 2 + 1 == spinner_state,
        )
        left_flipper_angle = state.left_flipper_angle
        right_flipper_angle = state.right_flipper_angle

        # disable reflecting center part of inactive targets (diamonds)
        reflecting_active = jnp.where(
            jnp.logical_and(
                is_left_lit_up_target_refl,
                jnp.logical_not(is_left_lit_up_target_active),
            ),
            False,
            reflecting_active,  # only select hit_point[0] since we only update t_entry
        )
        reflecting_active = jnp.where(
            jnp.logical_and(
                is_middle_lit_up_target_refl,
                jnp.logical_not(is_middle_lit_up_target_active),
            ),
            False,
            reflecting_active,
        )
        reflecting_active = jnp.where(
            jnp.logical_and(
                is_right_lit_up_target_refl,
                jnp.logical_not(is_right_lit_up_target_active),
            ),
            False,
            reflecting_active,
        )

        # Disable all the spinner parts not matching the current step
        reflecting_active = jnp.where(
            jnp.logical_and(
                jnp.logical_not(spinner_active),
                is_spinner,  # scoring_type == spinner (2)
            ),
            False,
            reflecting_active,  # only select hit_point[0]
        )

        # Disable inactive flipper parts
        reflecting_active = jnp.where(
            jnp.logical_and(
                is_left_flipper,
                jnp.logical_or(
                    jnp.logical_not(object_variant % 4 == left_flipper_angle),
                    jnp.logical_not(left_flipper_active),
                ),
            ),
            False,
            reflecting_active,
        )

        reflecting_active = jnp.where(
            jnp.logical_and(
                is_right_flipper,
                jnp.logical_or(
                    jnp.logical_not(object_variant % 4 == right_flipper_angle),
                    jnp.logical_not(right_flipper_active),
                ),
            ),
            False,
            reflecting_active,
        )
        # Disable tilt mode hole plugs if in tilt mode
        reflecting_active = jnp.where(
            jnp.logical_and(is_hole_plug, state.tilt_mode_active),
            False,
            reflecting_active,
        )

        # GET "FIRST" REFLECTING HIT POINT
        #################################################
        """
        Check if the ball is hitting an obstacle.
        """
        velocity_factor, velocity_addition, reflecting_hit_points = jax.vmap(
            lambda scene_object, active: jax.lax.cond(
                active,
                lambda: self._calc_hit_point(ball_movement, scene_object, action),
                lambda: self._dummy_calc_hit_point(scene_object),
            )
        )(self.consts.REFLECTING_SCENE_OBJECTS, reflecting_active)

        # In tilt mode we do not hit non-reflecting objects
        _, _, non_reflecting_hit_points = jax.lax.cond(
            state.tilt_mode_active,
            lambda: jax.vmap(
                lambda scene_object, active: self._dummy_calc_hit_point(scene_object)
            )(self.consts.NON_REFLECTING_SCENE_OBJECTS, non_reflecting_active),
            lambda: jax.vmap(
                lambda scene_object, active: jax.lax.cond(
                    active,
                    lambda: self._calc_hit_point(ball_movement, scene_object, action),
                    lambda: self._dummy_calc_hit_point(scene_object),
                )
            )(self.consts.NON_REFLECTING_SCENE_OBJECTS, non_reflecting_active),
        )
        argmin = jnp.argmin(reflecting_hit_points[:, HitPointSelector.T_ENTRY])
        hit_point = reflecting_hit_points[argmin]
        velocity_factor = velocity_factor[argmin]
        velocity_addition = velocity_addition[argmin]

        # UPDATE SCORING LIST
        #################################################

        # compute hit_before_reflection:
        # unlikely but if the ball reflects before the non reflecting object was hit,
        # the hit with the non reflecting object does not count
        hit_before_reflection = jnp.logical_and(
            non_reflecting_hit_points[:, HitPointSelector.T_ENTRY]
            < hit_point[HitPointSelector.T_ENTRY],
            non_reflecting_hit_points[:, HitPointSelector.T_ENTRY]
            != self.consts.T_ENTRY_NO_COLLISION,
        )  # shape: (n_non_reflecting_objects,)

        type_idxs = jnp.arange(self.consts.TOTAL_SCORE_TYPES)  # shape: (n_types,)

        # 1) whether the current hit_point's OBJECT_SCORE_TYPE equals each type -> (n_types,)
        hit_point_valid = (
            hit_point[HitPointSelector.T_ENTRY] != self.consts.T_ENTRY_NO_COLLISION
        )
        hit_point_valid = jnp.broadcast_to(hit_point_valid, type_idxs.shape)
        hit_point_type_eq = type_idxs == hit_point[HitPointSelector.OBJECT_SCORE_TYPE]
        hit_point_type_eq = hit_point_type_eq & hit_point_valid

        # 2) whether any non-reflecting object of each type was hit before reflection
        non_refl_types = non_reflecting_hit_points[
            :, HitPointSelector.OBJECT_SCORE_TYPE
        ]  # (n_non_refl,)
        # matches pairwise: (n_non_refl, n_types)
        type_matches = non_refl_types[:, None] == type_idxs[None, :]
        # combine with hit_before_reflection per object, then any over objects -> (n_types,)
        any_non_refl_hit_per_type = jnp.any(
            hit_before_reflection[:, None] & type_matches, axis=0
        )

        # combine all
        scoring_list = jnp.logical_or(scoring_list, hit_point_type_eq)
        scoring_list = jnp.logical_or(scoring_list, any_non_refl_hit_per_type)

        # jax.debug.print(
        #    "Hit Point:\n\t"
        #    "T_ENTRY: {}\n\t"
        #    "X: {}\n\t"
        #    "Y: {}\n\t"
        #    "RX: {}\n\t"
        #    "RY: {}\n\t"
        #    "OBJECT_WIDTH: {}\n\t"
        #    "OBJECT_HEIGHT: {}\n\t"
        #    "OBJECT_X: {}\n\t"
        #    "OBJECT_Y: {}\n\t"
        #    "OBJECT_REFLECTING: {}\n\t"
        #    "OBJECT_SCORE_TYPE: {}\n\t"
        #    "OBJECT_VARIANT: {}\n"
        #    "Pre-Collision Movement:\n\t"
        #    "OLD_X: {}\n\t"
        #    "OLD_Y: {}\n\t"
        #    "NEW_X: {}\n\t"
        #    "NEW_Y: {}\n\t"
        #    "SCORES: {}\n",
        #    hit_point[HitPointSelector.T_ENTRY],
        #    hit_point[HitPointSelector.X],
        #    hit_point[HitPointSelector.Y],
        #    hit_point[HitPointSelector.RX],
        #    hit_point[HitPointSelector.RY],
        #    hit_point[HitPointSelector.OBJECT_WIDTH],
        #    hit_point[HitPointSelector.OBJECT_HEIGHT],
        #    hit_point[HitPointSelector.OBJECT_X],
        #    hit_point[HitPointSelector.OBJECT_Y],
        #    hit_point[HitPointSelector.OBJECT_REFLECTING],
        #    hit_point[HitPointSelector.OBJECT_SCORE_TYPE],
        #    hit_point[HitPointSelector.OBJECT_VARIANT],
        #    ball_movement.old_ball_x, ball_movement.old_ball_y, ball_movement.new_ball_x, ball_movement.new_ball_y, scoring_list
        # )

        return hit_point, scoring_list, velocity_factor, velocity_addition

    @partial(jax.jit, static_argnums=(0,))
    def _calc_ball_collision_loop(
        self,
        state: VideoPinballState,
        ball_movement: BallMovement,
        action: chex.Array,
        left_flipper_active: chex.Array,
        right_flipper_active: chex.Array,
    ):

        def _fori_body(i, carry):

            def _compute_ball_collision(
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                velocity_factor,
                velocity_addition,
                left_flipper_active,
                right_flipper_active,
                scoring_list,
            ):
                _ball_movement = BallMovement(
                    old_ball_x=old_ball_x,
                    old_ball_y=old_ball_y,
                    new_ball_x=new_ball_x,
                    new_ball_y=new_ball_y,
                )

                hit_data, scoring_list, vf, va = self._check_obstacle_hits(
                    state,
                    _ball_movement,
                    scoring_list,
                    action,
                    left_flipper_active,
                    right_flipper_active,
                )

                no_collision = (
                    hit_data[HitPointSelector.T_ENTRY]
                    == self.consts.T_ENTRY_NO_COLLISION
                )
                collision = jnp.logical_not(no_collision)

                velocity_factor = jnp.where(
                    collision, velocity_factor * vf, velocity_factor
                )
                velocity_addition = jnp.where(
                    collision, velocity_addition + va, velocity_addition
                )

                old_ball_x = jnp.where(
                    collision, hit_data[HitPointSelector.X], _ball_movement.old_ball_x
                )
                old_ball_y = jnp.where(
                    collision, hit_data[HitPointSelector.Y], _ball_movement.old_ball_y
                )
                new_ball_x = jnp.where(
                    collision, hit_data[HitPointSelector.RX], _ball_movement.new_ball_x
                )
                new_ball_y = jnp.where(
                    collision, hit_data[HitPointSelector.RY], _ball_movement.new_ball_y
                )

                # definitive fix for flipper collisions:
                # if a flipper is hit, deactivate it. If something other than a flipper was hit, activate it
                hit_is_left_flipper = hit_data[HitPointSelector.OBJECT_SCORE_TYPE] == 9
                hit_is_right_flipper = (
                    hit_data[HitPointSelector.OBJECT_SCORE_TYPE] == 10
                )
                left_flipper_active = jnp.where(
                    jnp.logical_and(collision, hit_is_left_flipper),
                    False,
                    left_flipper_active,
                )
                right_flipper_active = jnp.where(
                    jnp.logical_and(collision, hit_is_right_flipper),
                    False,
                    right_flipper_active,
                )
                left_flipper_active = jnp.where(
                    jnp.logical_and(collision, jnp.logical_not(hit_is_left_flipper)),
                    True,
                    left_flipper_active,
                )
                right_flipper_active = jnp.where(
                    jnp.logical_and(collision, jnp.logical_not(hit_is_right_flipper)),
                    True,
                    right_flipper_active,
                )

                return (
                    old_ball_x,
                    old_ball_y,
                    new_ball_x,
                    new_ball_y,
                    velocity_factor,
                    velocity_addition,
                    left_flipper_active,
                    right_flipper_active,
                    scoring_list,
                    collision,
                )

            (
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                velocity_factor,
                velocity_addition,
                left_flipper_active,
                right_flipper_active,
                scoring_list,
                any_collision,
                compute_flag,
            ) = carry

            (
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                velocity_factor,
                velocity_addition,
                left_flipper_active,
                right_flipper_active,
                scoring_list,
                collision,
            ) = jax.lax.cond(
                compute_flag,
                _compute_ball_collision,
                lambda old_ball_x, old_ball_y, new_ball_x, new_ball_y, vf, va, lfa, rfa, s: (
                    old_ball_x,
                    old_ball_y,
                    new_ball_x,
                    new_ball_y,
                    vf,
                    va,
                    lfa,
                    rfa,
                    s,
                    False,
                ),
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                velocity_factor,
                velocity_addition,
                left_flipper_active,
                right_flipper_active,
                scoring_list,
            )

            compute_flag = jnp.logical_and(compute_flag, collision)
            any_collision = jnp.logical_or(any_collision, collision)

            return (
                old_ball_x,
                old_ball_y,
                new_ball_x,
                new_ball_y,
                velocity_factor,
                velocity_addition,
                left_flipper_active,
                right_flipper_active,
                scoring_list,
                any_collision,
                compute_flag,
            )

        # Initial carry values
        carry = (
            ball_movement.old_ball_x,
            ball_movement.old_ball_y,
            ball_movement.new_ball_x,
            ball_movement.new_ball_y,
            1.0,  # velocity_factor
            0.0,  # velocity_addition
            left_flipper_active,
            right_flipper_active,
            jnp.zeros((self.consts.TOTAL_SCORE_TYPES,), dtype=bool),  # scoring_list
            False,  # any_collision
            True,  # compute_flag
        )

        carry = jax.lax.fori_loop(
            0, self.consts.MAX_REFLECTIONS_PER_GAMESTEP, _fori_body, carry
        )

        (
            old_ball_x,
            old_ball_y,
            new_ball_x,
            new_ball_y,
            velocity_factor,
            velocity_addition,
            left_flipper_active,
            right_flipper_active,
            scoring_list,
            any_collision,
            _,
        ) = carry

        return (
            BallMovement(
                old_ball_x=old_ball_x,
                old_ball_y=old_ball_y,
                new_ball_x=new_ball_x,
                new_ball_y=new_ball_y,
            ),
            scoring_list,
            velocity_factor,
            velocity_addition,
            left_flipper_active,
            right_flipper_active,
            any_collision,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_ball_direction_signs(
        self, ball_direction
    ) -> tuple[chex.Array, chex.Array]:
        """
        Description
            Compute the signed multipliers for x and y velocity based on the discrete
            ball_direction used by Atari VideoPinball.

        Parameters
            ----------
            ball_direction: chex.Array
                Discrete direction index in {0,1,2,3}:
                    0: Top Left, 1: Bottom Left, 2: Top Right, 3: Bottom Right.

        Returns
            ----------
            tuple[chex.Array, chex.Array]
                (x_sign, y_sign) each a float32 scalar equal to +1.0 or -1.0. These are
                intended to be multiplied with non-negative speed components to produce
                screen-space signed velocities.
        """
        x_sign = jnp.where(
            jnp.logical_or(ball_direction == 2, ball_direction == 3),
            jnp.array(1.0),
            jnp.array(-1.0),
        )
        y_sign = jnp.where(
            jnp.logical_or(ball_direction == 0, ball_direction == 2),
            jnp.array(-1.0),
            jnp.array(1.0),
        )
        return x_sign, y_sign

    @partial(jax.jit, static_argnums=(0,))
    def _get_ball_direction(self, signed_vel_x, signed_vel_y) -> chex.Array:
        """
        Description
            This deterministically maps signed x/y velocities to a discrete direction index (0-3).
            The mapping encodes quadrant-like directions with a stable tie-breaking rule so it can be
            applied elementwise to JAX/chex arrays without branching.

        Parameters
         ----------
            signed_vel_x : chex.Array
                Array of signed x velocities (can be scalar or batched). Negative x is treated as "left".
            signed_vel_y : chex.Array
                Array of signed y velocities (can be scalar or batched). Negative y is treated as "up" (Atari defines top left as origin).

        Returns
         ----------
            chex.Array
                Scalar or array of integer direction indices in [0, 3] with the following encoding:
                0 = top-left (x <= 0, y <= 0)
                1 = bottom-left (x <= 0, y > 0)
                2 = top-right (x > 0, y <= 0)
                3 = bottom-right (x > 0, y > 0)
                The result type matches JAX/chex semantics for array/scalar outputs.

         Design Notes / Rationale
            - Uses inclusive boundary (<=) for zero to ensure deterministic classification of zero velocity.
            - Constructs a boolean vector in a specific order and uses argmax so the first true entry
              determines the direction; this provides a consistent tie-breaking behavior and vectorizes well.
        """
        # If both values are negative, we move closer to (0, 0) in the top left corner and fly in direction 0
        top_left = jnp.logical_and(signed_vel_x <= 0, signed_vel_y <= 0)  # 0
        top_right = jnp.logical_and(signed_vel_x > 0, signed_vel_y <= 0)  # 2
        bottom_right = jnp.logical_and(signed_vel_x > 0, signed_vel_y > 0)  # 3
        bottom_left = jnp.logical_and(signed_vel_x <= 0, signed_vel_y > 0)  # 1

        bool_array = jnp.array([top_left, bottom_left, top_right, bottom_right])
        return jnp.argmax(bool_array)

    @partial(jax.jit, static_argnums=(0,))
    def _calc_ball_change(self, ball_x, ball_y, ball_vel_x, ball_vel_y, ball_direction):
        """
        Calculates the new ball position and velocity taking the ball direction into account.
        """

        sign_x, sign_y = self._get_ball_direction_signs(ball_direction)
        ball_vel_x = jnp.clip(ball_vel_x, 0, self.consts.BALL_MAX_SPEED)
        ball_vel_y = jnp.clip(ball_vel_y, 0, self.consts.BALL_MAX_SPEED)
        signed_ball_vel_x = sign_x * ball_vel_x
        signed_ball_vel_y = sign_y * ball_vel_y
        ball_x = ball_x + signed_ball_vel_x
        ball_y = ball_y + signed_ball_vel_y
        return (
            ball_x,
            ball_y,
            ball_vel_x,
            ball_vel_y,
            signed_ball_vel_x,
            signed_ball_vel_y,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_tilt(
        self, state: VideoPinballState, action: Action, ball_x: chex.Array
    ):
        """
        Description
            Compute and update the machine's tilt state driven by player nudges.
            Returns whether tilt mode becomes active, the updated tilt counter and a
            possibly nudged horizontal ball position. Keeps all logic JAX-friendly so
            it can be jitted and vectorized.

        Parameters
            ----------
            state : VideoPinballState
                Full environment state (used for step counter and existing tilt values).
            action : Action
                Discrete action used to detect nudges (LEFTFIRE / RIGHTFIRE are of interest here).
            ball_x : chex.Array
                Current horizontal ball coordinate (float32), y direction can not be changed and is left out.

        Returns
            ----------
            tuple[chex.Array, chex.Array, chex.Array]
                (tilt_mode_active, tilt_counter, ball_x)
                - tilt_mode_active: boolean-like scalar indicating if tilt mode is active.
                - tilt_counter: integer-like scalar clamped to configured bounds.
                - ball_x: possibly adjusted ball x coordinate after nudge effect.

         Design Notes / Rationale
            - Use interval-based updates for both counter changes and ball displacement
                to match original game behavior.
        """

        # branch when there *is* a nudge (nudge_direction != 0)
        def _nudge_branch(state: VideoPinballState, action: Action, ball_x: chex.Array):
            """
            Description
                Compute the updated tilt state and optionally shift the ball's horizontal position when the player nudges the machine.

            Parameters
            ----------
                state: VideoPinballState
                    Current game state (contains step_counter and tilt_counter).
                action: JaxAtariAction
                    Player input; nudging is detected via LEFTFIRE or RIGHTFIRE actions.
                ball_x: chex.Array
                    Scalar/array representing the ball's horizontal coordinate (float32). Kept generic to allow JAX array types and broadcasting.

            Returns
            ----------
                tuple(
                    tilt_mode_from_counter: jnp.ndarray (bool scalar/array),
                    tilt_counter_capped: jnp.ndarray (integer scalar/array),
                    ball_x_new: chex.Array (float scalar/array)
                    tilt_mode_from_counter:
                        Boolean indicating whether tilt mode would be active after applying the interval increment (used to drive game behaviour).
                    tilt_counter_capped:
                        Tilt counter bounded at the configured TILT_COUNT_TILT_MODE_ACTIVE threshold so downstream logic sees a stable maximum.
                    ball_x_new:
                        Horizontally adjusted ball position when a nudge takes effect; unchanged otherwise. Returned as same dtype/shape as input to preserve JAX tracing.

             Design Notes / Rationale
                -----------------------
                - Intervaled updates: Both tilt-count increases and nudge effects are applied only on configurable intervals to mimic the original game's timing
                - Exponential increase (doubling) when tilt counter > 0:
                   Mimics the original game.
            """
            # increase tilt counter on interval
            inc_cond = jnp.equal(
                jnp.mod(state.step_counter, self.consts.TILT_COUNT_INCREASE_INTERVAL), 0
            )

            tilt_counter_inc = jax.lax.cond(
                inc_cond,
                lambda tc: jax.lax.cond(
                    jnp.greater(tc, 0),
                    lambda t: 2 * t,
                    lambda t: jnp.array(1, dtype=t.dtype),
                    tc,
                ),
                lambda tc: tc,
                state.tilt_counter,
            )

            # detect / cap tilt mode activation
            tilt_mode_from_counter = jnp.greater_equal(
                tilt_counter_inc, self.consts.TILT_COUNT_TILT_MODE_ACTIVE
            )
            tilt_counter_capped = jnp.minimum(
                tilt_counter_inc, self.consts.TILT_COUNT_TILT_MODE_ACTIVE
            )

            # adjust horizontal location depending on nudge direction
            ball_x_new = jax.lax.cond(
                state.step_counter % self.consts.NUDGE_EFFECT_INTERVAL == 0,
                lambda bv: jax.lax.cond(
                    action == Action.RIGHTFIRE,
                    lambda bv: bv + self.consts.NUDGE_EFFECT_AMOUNT,
                    lambda bv: bv - self.consts.NUDGE_EFFECT_AMOUNT,
                    bv,
                ),
                lambda bv: bv,
                ball_x,
            )

            return tilt_mode_from_counter, tilt_counter_capped, ball_x_new

        def _no_nudge_branch(
            state: VideoPinballState, action: Action, ball_x: chex.Array
        ):
            """
            Description
                Compute the periodic decay of the tilt counter when the player is not nudging.
                This branch never clears an active tilt mode — it only decays the counter over time
                until the configured minimum or the tilt threshold is reached again.

            Parameters
                ----------
                state : VideoPinballState
                    Full game state; used for step timing and existing tilt values.
                action : Action
                    Player action (kept for jax.cond() consistency; not used here).
                ball_x : chex.Array
                    Horizontal ball coordinate (returned unchanged by this branch (also for consistency with jax conditional)).

            Returns
                ----------
                tuple[chex.Array, chex.Array, chex.Array]
                    (tilt_mode_active, tilt_counter, ball_x) where:
                        - tilt_mode_active: boolean scalar indicating whether tilt mode remains active
                        - tilt_counter: integer scalar representing the (possibly decayed) tilt counter
                        - ball_x: returned horizontal ball coordinate (unchanged)

             Design Notes / Rationale
                - Decay uses intervaled halving to mimic the original game's conservative recovery:
                  the counter is only reduced on specific step intervals to avoid rapid recovery.
                - Tilt mode is intentionally not deactivated in this branch so that once triggered,
                  tilt remains until explicit reset elsewhere (matching original gameplay semantics).
                - Counter clamped to be non-negative to avoid negative underflow during repeated halvings.
            """

            dec_cond = jnp.equal(
                jnp.mod(state.step_counter, self.consts.TILT_COUNT_DECREASE_INTERVAL), 0
            )
            dec_cond = jnp.logical_and(
                dec_cond, jnp.logical_not(state.tilt_mode_active)
            )

            tilt_counter_dec = jax.lax.cond(
                dec_cond,
                lambda tc: jax.lax.cond(
                    jnp.equal(tc, 1),
                    lambda tc: jnp.array(0, dtype=tc.dtype),
                    lambda tc: jnp.floor_divide(tc, 2),
                    tc,
                ),
                lambda tc: tc,
                state.tilt_counter,
            )
            tilt_counter_nonneg = jnp.maximum(tilt_counter_dec, 0)
            return state.tilt_mode_active, tilt_counter_nonneg, ball_x

        is_nudging = jnp.logical_or(
            action == Action.LEFTFIRE, action == Action.RIGHTFIRE
        )
        return jax.lax.cond(
            is_nudging, _nudge_branch, _no_nudge_branch, state, action, ball_x
        )

    @partial(jax.jit, static_argnums=(0,))
    def _apply_gravity(
        self,
        ball_x: chex.Array,
        ball_vel_x: chex.Array,
        ball_vel_y: chex.Array,
        ball_direction: chex.Array,
    ):
        """
        Calculates the effect of gravity on the ball velocity and direction.
        Gravity pulls the ball down (increasing y velocity) when the ball is moving down (direction 1 or 3)
        and pushes the ball up (decreasing y velocity) when the ball is moving up (direction 0 or 2).
        The x velocity is not affected by gravity.
        Note: Gravity is not applied when the ball is at the starting position (x = BALL_START_X).
        """

        initial_ball_vel_x = ball_vel_x
        initial_ball_vel_y = ball_vel_y
        initial_ball_direction = ball_direction

        # Gravity calculation
        gravity_delta = jnp.where(
            jnp.logical_or(ball_direction == 0, ball_direction == 2),
            -self.consts.GRAVITY,
            self.consts.GRAVITY,
        )  # Subtract gravity if the ball is moving up otherwise add it
        ball_vel_y = ball_vel_y + gravity_delta
        ball_direction = jnp.where(
            ball_vel_y < 0,
            ball_direction + 1,  # if ball direction was towards upper left
            ball_direction,
        )
        ball_vel_y = jnp.abs(ball_vel_y)

        # If ball is at starting position (x), ignore gravity calculations
        ball_vel_x = jnp.where(
            ball_x == self.consts.BALL_START_X, initial_ball_vel_x, ball_vel_x
        )
        ball_vel_y = jnp.where(
            ball_x == self.consts.BALL_START_X, initial_ball_vel_y, ball_vel_y
        )
        ball_direction = jnp.where(
            ball_x == self.consts.BALL_START_X, initial_ball_direction, ball_direction
        )
        return ball_vel_x, ball_vel_y, ball_direction

    @partial(jax.jit, static_argnums=(0,))
    def _calc_invisible_block_hit(
        self,
        ball_movement: BallMovement,
        ball_in_play: chex.Array,
        action: chex.Array,
        key: chex.Array,
    ):
        """
        Calculates whether the ball is hitting the invisible block at the plunger hole.
        If the ball is hitting the invisible block, it is reflected and given a small random vertical velocity.
        Note: The invisible block can only be hit when the ball is not in play (ball_in_play == False).
        This prevents the ball from being stuck in the plunger hole due to a lack of horizontal velocity.
        """

        # Calculate the velocity of the ball
        ball_vel_x = jnp.abs(
            jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        )
        ball_vel_y = jnp.abs(
            jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)
        )

        # Check for invisible block hit
        _, _, invisible_block_hit_data = self._calc_hit_point(
            ball_movement,
            jnp.array(
                [
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_width,
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_height,
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_x_offset,
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.hit_box_y_offset,
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.reflecting,
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.score_type,
                    self.consts.INVISIBLE_BLOCK_SCENE_OBJECT.variant,
                ]
            ),
            action,
        )
        is_invisible_block_hit = jnp.logical_and(
            jnp.logical_not(ball_in_play),
            invisible_block_hit_data[HitPointSelector.T_ENTRY]
            != self.consts.T_ENTRY_NO_COLLISION,
        )

        d_traj = jnp.sqrt(ball_vel_x**2 + ball_vel_y**2)
        to_invis_block_hit_x = (
            invisible_block_hit_data[HitPointSelector.X] - ball_movement.old_ball_x
        )
        to_invis_block_hit_y = (
            invisible_block_hit_data[HitPointSelector.Y] - ball_movement.old_ball_y
        )
        d_hit = jnp.sqrt(to_invis_block_hit_x**2 + to_invis_block_hit_y**2)
        r = jnp.clip(1 - d_hit / d_traj, min=1e-2)
        ball_vel_x, ball_vel_y = jax.lax.cond(
            is_invisible_block_hit,
            lambda: (
                -ball_vel_y * 0.75,
                jrandom.uniform(key, minval=-0.1, maxval=0.1) * ball_vel_y,
            ),
            lambda: (ball_vel_x, ball_vel_y),
        )

        ball_movement = jax.lax.cond(
            is_invisible_block_hit,
            lambda: BallMovement(
                old_ball_x=invisible_block_hit_data[HitPointSelector.X],
                old_ball_y=invisible_block_hit_data[HitPointSelector.Y],
                new_ball_x=invisible_block_hit_data[HitPointSelector.X]
                + r * ball_vel_x,
                new_ball_y=invisible_block_hit_data[HitPointSelector.Y]
                + r * ball_vel_y,
            ),
            lambda: ball_movement,
        )
        return ball_movement, ball_vel_x, ball_vel_y, is_invisible_block_hit

    @partial(jax.jit, static_argnums=(0,))
    def _ball_step(
        self,
        state: VideoPinballState,
        plunger_power,
        action,
        key,
    ):
        """
        Description
            Comprehensive dynamics update for a single timestep for the ball.
            Applies plunger impulse, gravity, nudge/tilt adjustments, computes the
            unswept trajectory, checks for the invisible plunger-block, resolves
            collisions (including flippers, spinners, bumpers, etc.) and returns
            the updated kinematic state and bookkeeping flags.

        Parameters
            ----------
            state : VideoPinballState
                Complete environment state (positions, velocities, flipper states, rng_key, etc.)
            plunger_power : chex.Array
                Scalar launch impulse computed by the plunger subsystem for this step.
            action : chex.Array
                Discrete JAX action for current timestep.
            key : chex.Array
                JAX PRNG key used for stochastic effects (split inside helpers as needed).

        Returns
            ----------
            Tuple[
                chex.Array,  # ball_x
                chex.Array,  # ball_y
                chex.Array,  # ball_direction
                chex.Array,  # ball_vel_x
                chex.Array,  # ball_vel_y
                chex.Array,  # ball_in_play (bool-like)
                chex.Array,  # scoring_list (bool vector per score type)
                chex.Array,  # tilt_mode_active (bool-like)
                chex.Array,  # tilt_counter (int)
                chex.Array,  # left_flipper_active (bool-like)
                chex.Array,  # right_flipper_active (bool-like)
            ]

         Design Notes / Rationale
            - Defer geometric / collision specifics to the specialized helpers so this
              function focuses on sequencing (plunger → gravity → movement → collisions → final velocity).
        """

        ball_x = state.ball_x
        ball_y = state.ball_y
        ball_vel_x = state.ball_vel_x
        ball_vel_y = state.ball_vel_y
        ball_direction = state.ball_direction
        ball_in_play = state.ball_in_play

        """
        Plunger calculation
        """
        # Add plunger power to the ball velocity, only set to non-zero value once fired, reset after hitting invisible block
        ball_direction = jnp.where(
            plunger_power > 0,
            jnp.array(0),
            ball_direction,
        )  # Set direction to 0 if the ball is fired
        ball_vel_y = jnp.where(
            plunger_power > 0,
            ball_vel_y + plunger_power,
            ball_vel_y,
        )

        """
        Gravity calculation
        """
        ball_vel_x, ball_vel_y, ball_direction = self._apply_gravity(
            ball_x, ball_vel_x, ball_vel_y, ball_direction
        )
        """
        Nudge effect calculation and tilt counter update
        """
        tilt_mode, tilt_counter, ball_x = self._update_tilt(state, action, ball_x)
        """
        Ball movement calculation observing its direction 
        """
        ball_x, ball_y, ball_vel_x, ball_vel_y, signed_ball_vel_x, signed_ball_vel_y = (
            self._calc_ball_change(
                ball_x, state.ball_y, ball_vel_x, ball_vel_y, ball_direction
            )
        )
        ball_movement = BallMovement(
            old_ball_x=state.ball_x,
            old_ball_y=state.ball_y,
            new_ball_x=ball_x,
            new_ball_y=ball_y,
        )

        """
        Check if the ball is hitting the invisible block at the plunger hole
        """
        ball_movement, ball_vel_x, ball_vel_y, is_invisible_block_hit = jax.lax.cond(
            ball_in_play,
            lambda: (ball_movement, ball_vel_x, ball_vel_y, jnp.array(False)),
            lambda: self._calc_invisible_block_hit(
                ball_movement, ball_in_play, action, key
            ),
        )

        ball_in_play = jnp.logical_or(ball_in_play, is_invisible_block_hit)
        # account for the fact that the player could nudge in a way so that the
        # ball misses the invisible block:
        ball_in_play = jnp.logical_or(
            ball_in_play,
            ball_movement.old_ball_x < self.consts.BALL_START_X - 20
        )

        ball_direction_changed = state.ball_direction != self._get_ball_direction(
            ball_movement.new_ball_x - ball_movement.old_ball_x,
            ball_movement.new_ball_y - ball_movement.old_ball_y,
        )
        # reactivate flippers if the ball direction changed via gravity or tilt
        left_flipper_active = jnp.logical_or(
            state.left_flipper_active, ball_direction_changed
        )
        right_flipper_active = jnp.logical_or(
            state.right_flipper_active, ball_direction_changed
        )

        """
        Obstacle hit calculation
        """
        # Calculate whether and where obstacles are hit and the new ball trajectory
        (
            collision_ball_movement,
            scoring_list,
            velocity_factor,
            velocity_addition,
            left_flipper_active,
            right_flipper_active,
            any_collision,
        ) = self._calc_ball_collision_loop(
            state, ball_movement, action, left_flipper_active, right_flipper_active
        )

        ball_trajectory_x = (
            collision_ball_movement.new_ball_x - collision_ball_movement.old_ball_x
        )
        ball_trajectory_y = (
            collision_ball_movement.new_ball_y - collision_ball_movement.old_ball_y
        )

        ball_x = collision_ball_movement.new_ball_x
        ball_y = collision_ball_movement.new_ball_y

        """
        Final ball velocity and direction calculation
        """
        ball_direction = jax.lax.cond(
            jnp.logical_or(any_collision, is_invisible_block_hit),
            lambda: self._get_ball_direction(ball_trajectory_x, ball_trajectory_y),
            lambda: ball_direction,  # updated after gravity and before tilt
        )
        original_ball_speed = jnp.sqrt(ball_vel_x**2 + ball_vel_y**2)
        reflected_ball_speed = jnp.sqrt(ball_trajectory_x**2 + ball_trajectory_y**2)
        # if there was a collision, update the ball trajectory;
        # but only if the new trajectory is sufficiently large to avoid numerical instabilities
        ball_vel_x, ball_vel_y, ball_speed = jax.lax.cond(
            ball_in_play & any_collision & (reflected_ball_speed > self.consts.BALL_MIN_SPEED / 10),
            lambda: (jnp.abs(ball_trajectory_x), jnp.abs(ball_trajectory_y), reflected_ball_speed),
            lambda: (jnp.abs(ball_vel_x), jnp.abs(ball_vel_y), original_ball_speed),
        )
        new_ball_speed = jnp.clip(
            (original_ball_speed + jnp.clip(velocity_addition, -1, 1))
            * velocity_factor,
            0,
            self.consts.BALL_MAX_SPEED,
        )

        ball_vel_x = jnp.where(
            any_collision,
            ball_vel_x / ball_speed * new_ball_speed,
            ball_vel_x,
        )
        ball_vel_y = jnp.where(
            any_collision,
            ball_vel_y / ball_speed * new_ball_speed,
            ball_vel_y,
        )

        # If ball velocity reaches a small threshold, accelerate it after hitting something
        small_vel = jnp.sqrt(ball_vel_x**2 + ball_vel_y**2) < self.consts.BALL_MIN_SPEED
        ball_vel_x = jnp.where(
            jnp.logical_and(any_collision, small_vel),
            ball_vel_x * 2 + self.consts.BALL_MIN_SPEED,
            ball_vel_x,
        )
        ball_vel_y = jnp.where(
            jnp.logical_and(any_collision, small_vel),
            ball_vel_y * 2 + self.consts.BALL_MIN_SPEED,
            ball_vel_y,
        )

        return (
            ball_x,
            ball_y,
            ball_direction,
            ball_vel_x,
            ball_vel_y,
            ball_in_play,
            scoring_list,
            tilt_mode,
            tilt_counter,
            left_flipper_active,
            right_flipper_active,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _reset_ball(self, state: VideoPinballState):
        """
        When the ball goes into the gutter or into the plunger hole,
        respawn the ball on the launcher.
        """

        return (
            self.consts.BALL_START_X,
            self.consts.BALL_START_Y,
            jnp.array(0.0),  # vel_x
            jnp.array(0.0),  # vel_y
            jnp.array(False),  # tilt_mode_active
        )

    @partial(jax.jit, static_argnums=(0,))
    def _calc_respawn_timer(self, state: VideoPinballState):
        """Calculate the respawn timer for the ball, which depends on the rollover counter as we have to wait for the rollover animation to finish after the ball is lost."""

        respawn_timer = jnp.where(
            state.rollover_counter > 1,
            jnp.array((state.rollover_counter - 1) * 16).astype(jnp.int32),
            jnp.array(1).astype(jnp.int32),
        )

        return respawn_timer

    @partial(jax.jit, static_argnums=(0,))
    def _handle_ball_in_gutter(
        self,
        respawn_timer,
        rollover_counter,
        score,
        atari_symbols,
        lives_lost,
        active_targets,
        special_target_cooldown,
        tilt_mode_active,
        tilt_counter,
        ):
        """ 
        Description
            If the ball is lost, we calculate the final score, calculate the respawn timer,
            i.e. the time until the animations are done and we should respawn the ball.
            Further we reset the lives, active targets, atari symbols, special target cooldown and tilt mode/counter if the respawn timer is over.

        Parameters
        ----------
            respawn_timer (jnp.ndarray | int)
                Scalar countdown (ticks) until respawn animations complete.
            rollover_counter (jnp.ndarray | int)
                Counter tracked across lives used for rollover bonuses when all lives are lost.
            score (jnp.ndarray | int)
                Player score accumulator; may be incremented at periodic animation frames.
            atari_symbols (jnp.ndarray | int)
                Number of collected Atari-symbols.
            lives_lost (jnp.ndarray | int)
                Count of lives lost so far; may be reset when respawn completes.
            active_targets (jnp.ndarray)
                Representation of currently active targets (diamonds and lit up).
            special_target_cooldown (jnp.ndarray | int)
                Ticks remaining before a special target is spawned.
            tilt_mode_active (jnp.ndarray | bool)
                Whether tilt penalties are currently suppressing play; cleared on respawn completion.
            tilt_counter (jnp.ndarray | int)
                Accumulated tilt counter; cleared when respawn finishes.

        Returns
        ----------
            (
                respawn_timer (jnp.ndarray | int),
                rollover_counter (jnp.ndarray | int),
                score (jnp.ndarray | int),
                atari_symbols (jnp.ndarray | int),
                lives_lost (jnp.ndarray | int),
                active_targets (jnp.ndarray),
                special_target_cooldown (jnp.ndarray | int),
                tilt_mode_active (jnp.ndarray | bool),
                tilt_counter (jnp.ndarray | int)
            )
        """

        multiplier = jnp.clip(atari_symbols + 1, max=4)
        score = jnp.where(respawn_timer % 16 == 15, score + 1000 * multiplier, score)
        rollover_counter = jnp.where(
            respawn_timer % 16 == 15, rollover_counter - 1, rollover_counter
        )
        respawn_timer = jnp.where(respawn_timer > 0, respawn_timer - 1, respawn_timer)

        lives_lost, active_targets, atari_symbols, special_target_cooldown = (
            jax.lax.cond(
                respawn_timer == 0,
                lambda l, asym: self._reset_targets_atari_symbols_and_lives(l, asym),
                lambda l, asym: (l, active_targets, asym, special_target_cooldown),
                lives_lost,
                atari_symbols,
            )
        )

        tilt_mode_active = jnp.where(respawn_timer == 0, False, tilt_mode_active)
        tilt_counter = jnp.where(respawn_timer == 0, 0, tilt_counter)

        return (
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            lives_lost,
            active_targets,
            special_target_cooldown,
            tilt_mode_active,
            tilt_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _reset_targets_atari_symbols_and_lives(self, lives_lost, atari_symbols):
        lives_lost = jax.lax.cond(
            atari_symbols < 4,
            lambda x: x + 1,
            lambda x: x,
            operand=lives_lost,
        )

        active_targets = jnp.array([True, True, True, False]).astype(jnp.bool)
        atari_symbols = jnp.array(0).astype(jnp.int32)
        special_target_cooldown = jnp.array(0).astype(jnp.int32)

        return lives_lost, active_targets, atari_symbols, special_target_cooldown

    @partial(jax.jit, static_argnums=(0,))
    def _process_objects_hit(self, state: VideoPinballState, objects_hit):
        """
        After object hits, calculates the new score, active targets, atari symbols, rollover counter and whether the rollover and atari symbol can be hit again i.e. cooldown.

        Bumpers: Give points
        Targets: Make them disappear, give points
        Targets: Check if all hit, increase multiplier
        BonusTarget: Give points, make screen flash, something else?
        Rollover: Give points, increase number
        Atari: Give points, make Atari symbol at bottom appear
        Assume objects_hit is list:
        [0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover,
        5: Special Lit Up Target, 6: Left Lit Up Target, 7: Middle Lit Up Target, 8: Right Lit Up Target]
        """

        score = state.score
        active_targets = state.active_targets
        atari_symbols = state.atari_symbols
        rollover_counter = state.rollover_counter
        rollover_enabled = state.rollover_enabled

        # Bumper points
        score += jnp.where(
            objects_hit[1],
            100 * state.bumper_multiplier,
            0,
        )

        # Give points for targets hit
        score += jnp.where(objects_hit[6], 100, 0)
        score += jnp.where(objects_hit[7], 100, 0)
        score += jnp.where(objects_hit[8], 100, 0)

        # Make hit targets disappear
        active_targets = jax.lax.cond(
            objects_hit[6],
            lambda s: jnp.array([False, s[1], s[2], s[3]]).astype(jnp.bool),
            lambda s: s,
            operand=active_targets,
        )

        active_targets = jax.lax.cond(
            objects_hit[7],
            lambda s: jnp.array([s[0], False, s[2], s[3]]).astype(jnp.bool),
            lambda s: s,
            operand=active_targets,
        )

        active_targets = jax.lax.cond(
            objects_hit[8],
            lambda s: jnp.array([s[0], s[1], False, s[3]]).astype(jnp.bool),
            lambda s: s,
            operand=active_targets,
        )

        # Bottom Bonus Target
        score += jnp.where(objects_hit[5], 1100, 0)
        active_targets, color_cycling = jax.lax.cond(
            objects_hit[5],
            lambda s, cc: (
                jnp.array([s[0], s[1], s[2], False]).astype(jnp.bool),
                jnp.array(30).astype(jnp.int32),
            ),
            lambda s, cc: (s, cc),
            active_targets,
            state.color_cycling,
        )

        # Give score for hitting the rollover and increase its number
        score += jnp.where(objects_hit[3], 100, 0)
        rollover_counter = jax.lax.cond(
            jnp.logical_and(objects_hit[3], rollover_enabled),
            lambda s: s + 1,
            lambda s: s,
            operand=rollover_counter,
        )

        # Give score for hitting the Atari symbol and make a symbol appear at the bottom
        score += jnp.where(objects_hit[4], 100, 0)
        atari_symbols = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_and(objects_hit[4], atari_symbols < 4), rollover_enabled
            ),
            lambda s: s + 1,
            lambda s: s,
            operand=atari_symbols,
        )

        # Prevents hitting Atari symbol and rollover multiple times
        rollover_enabled = jnp.logical_not(
            jnp.logical_or(objects_hit[3], objects_hit[4])
        )

        # Do color cycling when the fourth Atari symbol has been hit
        color_cycling = jnp.where(
            jnp.logical_and(state.atari_symbols == 3, atari_symbols == 4),
            jnp.array(30).astype(jnp.int32),
            color_cycling,
        )

        # Give 1 point for hitting a spinner
        score += jnp.where(objects_hit[2], 1, 0)

        return (
            score,
            active_targets,
            atari_symbols,
            rollover_counter,
            rollover_enabled,
            color_cycling,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _handle_target_cooldowns(
        self, state: VideoPinballState, previous_active_targets, color_cycling
    ):
        """
        Updates the target cooldowns and respawns targets if necessary.
        Also increases the bumper multiplier if all targets got hit.
        """

        targets_are_inactive = jnp.logical_and(
            jnp.logical_not(previous_active_targets[0]),
            jnp.logical_and(
                jnp.logical_not(previous_active_targets[1]),
                jnp.logical_not(previous_active_targets[2]),
            ),
        )

        # Start 2 second cooldown after hitting all targets until they respawn
        target_cooldown, increase_bm, color_cycling = jax.lax.cond(
            jnp.logical_and(targets_are_inactive, state.target_cooldown == -1),
            lambda cd, cc: (
                jnp.array(self.consts.TARGET_RESPAWN_COOLDOWN).astype(jnp.int32),
                True,
                jnp.array(-9),
            ),
            lambda cd, cc: (cd, False, cc),
            state.target_cooldown,
            color_cycling,
        )

        # Increase Bumper multiplier if all targets got hit
        bumper_multiplier = jax.lax.cond(
            jnp.logical_and(increase_bm, state.bumper_multiplier < 9),
            lambda s: s + 1,
            lambda s: s,
            operand=state.bumper_multiplier,
        )

        # count down the cooldown timer
        target_cooldown = jax.lax.cond(
            jnp.logical_and(targets_are_inactive, target_cooldown != -1),
            lambda s: s - 1,
            lambda s: s,
            operand=target_cooldown,
        )

        # After the cooldown, respawn the targets
        target_cooldown, active_targets = jax.lax.cond(
            jnp.logical_and(targets_are_inactive, target_cooldown == 0),
            lambda tc, pat: (
                jnp.array(-1).astype(jnp.int32),
                jnp.array([True, True, True, pat[3]]).astype(jnp.bool),
            ),
            lambda tc, pat: (tc, pat),
            target_cooldown,
            previous_active_targets,
        )

        # count down the despawn cooldown timer
        special_target_cooldown = jax.lax.cond(
            jnp.logical_and(state.special_target_cooldown > 0, state.ball_in_play),
            lambda s: s - 1,
            lambda s: s,
            operand=state.special_target_cooldown,
        )

        # count up the respawn cooldown timer
        special_target_cooldown = jax.lax.cond(
            jnp.logical_and(special_target_cooldown < -1, state.ball_in_play),
            lambda s: s + 1,
            lambda s: s,
            operand=special_target_cooldown,
        )

        # despawn the special target
        special_target_cooldown, active_targets = jax.lax.cond(
            jnp.logical_and(special_target_cooldown == 0, state.ball_in_play),
            lambda cd, a: (
                cd - self.consts.SPECIAL_TARGET_INACTIVE_DURATION,
                a.at[3].set(False),
            ),
            lambda cd, a: (cd, a),
            special_target_cooldown,
            active_targets,
        )

        # spawn the special target
        special_target_cooldown, active_targets = jax.lax.cond(
            jnp.logical_and(special_target_cooldown == -1, state.ball_in_play),
            lambda cd, a: (
                cd + self.consts.SPECIAL_TARGET_ACTIVE_DURATION,
                a.at[3].set(True),
            ),
            lambda cd, a: (cd, a),
            special_target_cooldown,
            active_targets,
        )

        return (
            active_targets,
            target_cooldown,
            special_target_cooldown,
            bumper_multiplier,
            color_cycling,
        )


class VideoPinballRenderer(JAXGameRenderer):
    """JAX-based Video Pinball game renderer, optimized with JIT compilation."""

    def __init__(self, consts: VideoPinballConstants = None):
        super().__init__()
        self.consts = consts or VideoPinballConstants()
        self.sprites = self._load_sprites()

    @partial(jax.jit, static_argnums=(0,))
    def _load_sprites(self):
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        # Define the base directory for sprites relative to the script
        SPRITES_BASE_DIR = os.path.join(
            MODULE_DIR, "sprites/videopinball"
        )  # Assuming sprites are in a 'sprites/videopinball' subdirectory

        # Load sprites
        sprite_background = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Background.npy")
        )
        sprite_ball = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Ball.npy"))

        sprite_atari_logo = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "AtariLogo.npy")
        )
        sprite_x = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "X.npy"))
        sprite_yellow_diamond_bottom = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "YellowDiamondBottom.npy")
        )
        sprite_yellow_diamond_top = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "YellowDiamondTop.npy")
        )

        sprite_walls = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Walls.npy"))

        # Animated sprites
        sprite_spinner0 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "SpinnerBottom.npy")
        )
        sprite_spinner1 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "SpinnerRight.npy")
        )
        sprite_spinner2 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "SpinnerTop.npy"))
        sprite_spinner3 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "SpinnerLeft.npy")
        )

        sprite_launcher0 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher0.npy"))
        sprite_launcher1 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher1.npy"))
        sprite_launcher2 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher2.npy"))
        sprite_launcher3 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher3.npy"))
        sprite_launcher4 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher4.npy"))
        sprite_launcher5 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher4.npy"))
        sprite_launcher6 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher6.npy"))
        sprite_launcher7 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher7.npy"))
        sprite_launcher8 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher8.npy"))
        sprite_launcher9 = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Launcher9.npy"))
        sprite_launcher10 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher10.npy")
        )
        sprite_launcher11 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher11.npy")
        )
        sprite_launcher12 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher12.npy")
        )
        sprite_launcher13 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher13.npy")
        )
        sprite_launcher14 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher14.npy")
        )
        sprite_launcher15 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher15.npy")
        )
        sprite_launcher16 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher16.npy")
        )
        sprite_launcher17 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher17.npy")
        )
        sprite_launcher18 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "Launcher18.npy")
        )

        sprite_flipper_left0 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperLeft0.npy")
        )
        sprite_flipper_left1 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperLeft1.npy")
        )
        sprite_flipper_left2 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperLeft2.npy")
        )
        sprite_flipper_left3 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperLeft3.npy")
        )
        sprite_flipper_right0 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperRight0.npy")
        )
        sprite_flipper_right1 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperRight1.npy")
        )
        sprite_flipper_right2 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperRight2.npy")
        )
        sprite_flipper_right3 = jr.loadFrame(
            os.path.join(SPRITES_BASE_DIR, "FlipperRight3.npy")
        )

        sprites_spinner, _ = jr.pad_to_match(
            [sprite_spinner0, sprite_spinner1, sprite_spinner2, sprite_spinner3]
        )
        sprites_spinner = jnp.concatenate(
            [
                jnp.repeat(sprites_spinner[0][None], 2, axis=0),
                jnp.repeat(sprites_spinner[1][None], 2, axis=0),
                jnp.repeat(sprites_spinner[2][None], 2, axis=0),
                jnp.repeat(sprites_spinner[3][None], 2, axis=0),
            ]
        )

        @jax.jit
        def pad_to_match_top(sprites):
            max_height = max(sprite.shape[0] for sprite in sprites)
            max_width = max(sprite.shape[1] for sprite in sprites)

            def pad_sprite(sprite):
                pad_height = max_height - sprite.shape[0]
                pad_width = max_width - sprite.shape[1]
                return jnp.pad(
                    sprite,
                    ((pad_height, 0), (pad_width, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            return [pad_sprite(sprite) for sprite in sprites]

        sprites_plunger = pad_to_match_top(
            [
                sprite_launcher0,
                sprite_launcher1,
                sprite_launcher2,
                sprite_launcher3,
                sprite_launcher4,
                sprite_launcher5,
                sprite_launcher6,
                sprite_launcher7,
                sprite_launcher8,
                sprite_launcher9,
                sprite_launcher10,
                sprite_launcher11,
                sprite_launcher12,
                sprite_launcher13,
                sprite_launcher14,
                sprite_launcher15,
                sprite_launcher16,
                sprite_launcher17,
                sprite_launcher18,
            ]
        )

        sprites_plunger = jnp.concatenate(
            [
                jnp.repeat(sprites_plunger[0][None], 3, axis=0),
                jnp.repeat(sprites_plunger[1][None], 2, axis=0),
                jnp.repeat(sprites_plunger[2][None], 1, axis=0),
                jnp.repeat(sprites_plunger[3][None], 1, axis=0),
                jnp.repeat(sprites_plunger[4][None], 1, axis=0),
                jnp.repeat(sprites_plunger[5][None], 1, axis=0),
                jnp.repeat(sprites_plunger[6][None], 1, axis=0),
                jnp.repeat(sprites_plunger[7][None], 1, axis=0),
                jnp.repeat(sprites_plunger[8][None], 1, axis=0),
                jnp.repeat(sprites_plunger[9][None], 1, axis=0),
                jnp.repeat(sprites_plunger[10][None], 1, axis=0),
                jnp.repeat(sprites_plunger[11][None], 1, axis=0),
                jnp.repeat(sprites_plunger[12][None], 1, axis=0),
                jnp.repeat(sprites_plunger[13][None], 1, axis=0),
                jnp.repeat(sprites_plunger[14][None], 1, axis=0),
                jnp.repeat(sprites_plunger[15][None], 1, axis=0),
                jnp.repeat(sprites_plunger[16][None], 1, axis=0),
                jnp.repeat(sprites_plunger[17][None], 1, axis=0),
                jnp.repeat(sprites_plunger[18][None], 1, axis=0),
            ]
        )

        sprites_flipper_left, _ = jr.pad_to_match(
            [
                sprite_flipper_left0,
                sprite_flipper_left1,
                sprite_flipper_left2,
                sprite_flipper_left3,
            ]
        )

        sprites_flipper_right, _ = jr.pad_to_match(
            [
                sprite_flipper_right0,
                sprite_flipper_right1,
                sprite_flipper_right2,
                sprite_flipper_right3,
            ]
        )

        sprites_plunger = jnp.stack(sprites_plunger, axis=0)
        sprites_flipper_left = jnp.stack(sprites_flipper_left, axis=0)
        sprites_flipper_right = jnp.stack(sprites_flipper_right, axis=0)

        # Load number sprites
        sprites_score_numbers = jr.load_and_pad_digits(
            os.path.join(SPRITES_BASE_DIR, "ScoreNumber{}.npy"),
            num_chars=10,  # For digits 0 through 9
        )

        sprites_field_numbers = jr.load_and_pad_digits(
            os.path.join(SPRITES_BASE_DIR, "FieldNumber{}.npy"),
            num_chars=10,  # Load 0-9, even if you only use 1-9
        )

        sprite_background = jnp.expand_dims(sprite_background, axis=0)
        sprite_ball = jnp.expand_dims(sprite_ball, axis=0)
        sprite_walls = jnp.expand_dims(sprite_walls, axis=0)

        sprite_atari_logo = jnp.expand_dims(sprite_atari_logo, axis=0)
        sprite_x = jnp.expand_dims(sprite_x, axis=0)
        sprite_yellow_diamond_bottom = jnp.expand_dims(
            sprite_yellow_diamond_bottom, axis=0
        )
        sprite_yellow_diamond_top = jnp.expand_dims(sprite_yellow_diamond_top, axis=0)

        return {
            "atari_logo": sprite_atari_logo,
            "background": sprite_background,
            "ball": sprite_ball,
            "spinner": sprites_spinner,
            "x": sprite_x,
            "yellow_diamond_bottom": sprite_yellow_diamond_bottom,
            "yellow_diamond_top": sprite_yellow_diamond_top,
            "walls": sprite_walls,
            # Animated sprites
            "flipper_left": sprites_flipper_left,
            "flipper_right": sprites_flipper_right,
            "plunger": sprites_plunger,
            # Digit sprites
            "score_number_digits": sprites_score_numbers,
            "field_number_digits": sprites_field_numbers,
        }

    @partial(jax.jit, static_argnums=(0,))
    def _render_tilt_mode(self, r):
        r = r.at[0:16, :, :].set(self.consts.TILT_MODE_COLOR)
        r = r.at[184:192, 36:40, :].set(self.consts.BG_COLOR)
        r = r.at[184:192, 120:124, :].set(self.consts.BG_COLOR)

        return r

    @partial(jax.jit, static_argnums=(0,))
    def _render_scene_object_boundaries(self, raster: chex.Array) -> chex.Array:
        """
        Renders the one-pixel boundaries of all SceneObjects onto a raster using vmap.

        Args:
            raster: A JAX array of shape (height, width, 4) representing the game screen.

        Returns:
            A new JAX array with the scene object boundaries drawn onto it.
        """

        # Use vmap to apply the rendering function to all objects in the list.
        # The `in_axes=(None, 0)` tells vmap to not vectorize the `raster` argument
        # and to vectorize the `scene_object` argument.
        def _draw_pixel(current_raster, y, x):
            """Draws a single pixel on the raster."""
            return jax.lax.cond(
                (y >= 0)
                & (y < current_raster.shape[0])
                & (x >= 0)
                & (x < current_raster.shape[1]),
                lambda r: r.at[y, x].set(self.consts.BOUNDARY_COLOR),
                lambda r: r,
                current_raster,
            )

        def _draw_line(current_raster, start, end):
            """
            Draws a line between two points on the raster.
            `start` and `end` are (y, x) tuples.
            """
            y1, x1 = start
            y2, x2 = end

            is_horizontal = jnp.abs(x2 - x1) > jnp.abs(y2 - y1)

            def body_fun_h(i, r):
                x = x1 + i
                return _draw_pixel(r, y1, x)

            def body_fun_v(i, r):
                y = y1 + i
                return _draw_pixel(r, y, x1)

            raster = jax.lax.cond(
                is_horizontal,
                lambda r: jax.lax.fori_loop(0, jnp.abs(x2 - x1) + 1, body_fun_h, r),
                lambda r: jax.lax.fori_loop(0, jnp.abs(y2 - y1) + 1, body_fun_v, r),
                current_raster,
            )
            return raster

        def _render_single_object_boundaries(
            raster: chex.Array, scene_object: SceneObject
        ) -> chex.Array:
            """
            Renders the one-pixel boundary of a single SceneObject onto a raster.

            Args:
                raster: A JAX array of shape (height, width, 4) representing the game screen.
                scene_object: A single SceneObject chex.dataclass instance.

            Returns:
                A new JAX array with the scene object boundary drawn onto it.
            """
            x = scene_object.hit_box_x_offset
            y = scene_object.hit_box_y_offset
            width = scene_object.hit_box_width
            height = scene_object.hit_box_height

            # Calculate corner points
            top_left = (y, x)
            top_right = (y, x + width - 1)
            bottom_left = (y + height - 1, x)
            bottom_right = (y + height - 1, x + width - 1)

            # Draw the four boundary lines
            raster = _draw_line(raster, top_left, top_right)
            raster = _draw_line(raster, top_left, bottom_left)
            raster = _draw_line(raster, top_right, bottom_right)
            raster = _draw_line(raster, bottom_left, bottom_right)

            return raster

        # First, convert the list of Python dataclasses into a single JAX dataclass
        # where each field is a stacked array.
        stacked_objects = jax.tree_util.tree_map(
            lambda *x: jnp.stack(x), *self.consts.ALL_SCENE_OBJECTS_LIST
        )

        # Use vmap to apply the rendering function to all objects.
        # We pass the raster without vectorizing it (in_axes=None) and
        # vectorize over the stacked scene_objects (in_axes=0).
        return jax.vmap(_render_single_object_boundaries, in_axes=(None, 0))(
            raster, stacked_objects
        ).sum(axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _handle_color_cycling(self, raster, color):

        bg_color_mask = jnp.all(
            raster == self.consts.BACKGROUND_COLOR, axis=-1, keepdims=True
        )
        wall_color_mask = jnp.all(
            raster == self.consts.WALL_COLOR, axis=-1, keepdims=True
        )
        group3_color_mask = jnp.all(
            raster == self.consts.GROUP3_COLOR, axis=-1, keepdims=True
        )
        group4_color_mask = jnp.all(
            raster == self.consts.GROUP4_COLOR, axis=-1, keepdims=True
        )
        group5_color_mask = jnp.all(
            raster == self.consts.GROUP5_COLOR, axis=-1, keepdims=True
        )

        raster = jnp.where(
            bg_color_mask, self.consts.BACKGROUND_COLOR_CYCLING[color], raster
        )
        raster = jnp.where(
            wall_color_mask, self.consts.WALL_COLOR_CYCLING[color], raster
        )
        raster = jnp.where(
            group3_color_mask, self.consts.GROUP3_COLOR_CYCLING[color], raster
        )
        raster = jnp.where(
            group4_color_mask, self.consts.GROUP4_COLOR_CYCLING[color], raster
        )
        raster = jnp.where(
            group5_color_mask, self.consts.GROUP5_COLOR_CYCLING[color], raster
        )

        raster = raster.at[0:8, 175:176, :].set(self.consts.BG_COLOR)
        raster = raster.at[:, 191:, :].set(self.consts.BG_COLOR)

        return raster

    @partial(jax.jit, static_argnums=(0,))
    def _split_integer(self, number: jnp.ndarray, max_digits: int = 6) -> jnp.ndarray:
        """
        Splits an integer into a JAX array of its individual digits.

        The output array will have a fixed size determined by `max_digits`.
        If the input number has fewer digits than `max_digits`, it will be
        padded with leading zeros.

        Args:
            number: The integer to split. Should be a non-negative integer.
                    Can be a Python int or a JAX array.
            max_digits: The maximum number of digits expected. The output array
                        will have this many elements.

        Returns:
            A 1D JAX array where each element is a digit of the input number.

        Example:
            split_integer(247900, max_digits=6) == jnp.array([2, 4, 7, 9, 0, 0])
            split_integer(123, max_digits=6)    == jnp.array([0, 0, 0, 1, 2, 3])
            split_integer(0, max_digits=3)      == jnp.array([0, 0, 0])
        """
        # Ensure the input number is a JAX array.
        # This handles both Python integers and existing JAX arrays.
        number = jnp.asarray(number, dtype=jnp.int32)

        # Create an array of powers of 10 to extract digits from left to right.
        # For max_digits=6, this will be [10^5, 10^4, 10^3, 10^2, 10^1, 10^0]
        powers_of_10 = 10 ** jnp.arange(max_digits - 1, -1, -1, dtype=jnp.int32)

        # Perform integer division by powers of 10, then take modulo 10
        # to isolate each digit.
        # Example:
        # (247900 // 100000) % 10 = 2
        # (247900 // 10000) % 10 = 4
        # ...
        # (247900 // 1) % 10 = 0
        digits = (number // powers_of_10) % 10

        return digits

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: VideoPinballState):
        """
        Renders the current game state using JAX operations.

        Args:
            state: A VideoPinballState object containing the current game state.

        Returns:
            A JAX array representing the rendered frame.
        """
        # Create empty raster with CORRECT orientation for atrjraxis framework
        # Note: For pygame, the raster is expected to be (width, height, channels)
        # where width corresponds to the horizontal dimension of the screen
        raster = jr.create_initial_frame(width=160, height=210)

        # Render static objects
        frame_bg = jr.get_sprite_frame(self.sprites["background"], 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)

        frame_walls = jr.get_sprite_frame(self.sprites["walls"], 0)
        raster = jr.render_at(raster, 0, 16, frame_walls)

        raster = jnp.where(
            state.tilt_mode_active, self._render_tilt_mode(raster), raster
        )

        # Render animated objects
        frame_flipper_left = jr.get_sprite_frame(
            self.sprites["flipper_left"], state.left_flipper_angle
        )
        raster = jr.render_at(
            raster,
            64,
            184 - self.consts.FLIPPER_ANIMATION_Y_OFFSETS[state.left_flipper_angle],
            frame_flipper_left,
        )

        frame_flipper_right = jr.get_sprite_frame(
            self.sprites["flipper_right"], state.right_flipper_angle
        )
        raster = jr.render_at(
            raster,
            83 + self.consts.FLIPPER_ANIMATION_X_OFFSETS[state.right_flipper_angle],
            184 - self.consts.FLIPPER_ANIMATION_Y_OFFSETS[state.right_flipper_angle],
            frame_flipper_right,
        )

        frame_plunger = jr.get_sprite_frame(
            self.sprites["plunger"], state.plunger_position
        )  # Still slightly inaccurate
        raster = jr.render_at(raster, 148, 133, frame_plunger)

        frame_spinner = jr.get_sprite_frame(
            self.sprites["spinner"], jnp.remainder(state.step_counter, 8)
        )
        raster = jr.render_at(raster, 30, 90, frame_spinner)
        raster = jr.render_at(raster, 126, 90, frame_spinner)

        frame_ball = jr.get_sprite_frame(self.sprites["ball"], 0)
        raster = jr.render_at(raster, state.ball_x, state.ball_y, frame_ball)

        # Render score
        frame_unknown = jr.get_sprite_frame(self.sprites["score_number_digits"], 1)
        raster = jr.render_at(raster, 4, 3, frame_unknown)

        displayed_lives = jnp.clip(state.lives_lost, max=3)
        frame_ball_count = jr.get_sprite_frame(
            self.sprites["score_number_digits"], displayed_lives
        )
        raster = jr.render_at(raster, 36, 3, frame_ball_count)

        numbers = self._split_integer(state.score)
        frame_score1 = jr.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[0]
        )
        raster = jr.render_at(raster, 64, 3, frame_score1)
        frame_score2 = jr.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[1]
        )
        raster = jr.render_at(raster, 80, 3, frame_score2)
        frame_score3 = jr.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[2]
        )
        raster = jr.render_at(raster, 96, 3, frame_score3)
        frame_score4 = jr.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[3]
        )
        raster = jr.render_at(raster, 112, 3, frame_score4)
        frame_score5 = jr.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[4]
        )
        raster = jr.render_at(raster, 128, 3, frame_score5)
        frame_score6 = jr.get_sprite_frame(
            self.sprites["score_number_digits"], numbers[5]
        )
        raster = jr.render_at(raster, 144, 3, frame_score6)

        # Render special yellow field objects
        frame_bumper_left = jr.get_sprite_frame(
            self.sprites["field_number_digits"], state.bumper_multiplier
        )
        raster = jr.render_at(raster, 46, 122, frame_bumper_left)
        frame_bumper_middle = jr.get_sprite_frame(
            self.sprites["field_number_digits"], state.bumper_multiplier
        )
        raster = jr.render_at(raster, 78, 58, frame_bumper_middle)
        frame_bumper_right = jr.get_sprite_frame(
            self.sprites["field_number_digits"], state.bumper_multiplier
        )
        raster = jr.render_at(raster, 110, 122, frame_bumper_right)

        displayed_rollover_number = jnp.remainder(state.rollover_counter, 9)
        frame_rollover_left = jr.get_sprite_frame(
            self.sprites["field_number_digits"], displayed_rollover_number
        )
        raster = jr.render_at(raster, 46, 58, frame_rollover_left)
        frame_atari_logo = jr.get_sprite_frame(self.sprites["atari_logo"], 0)
        raster = jr.render_at(raster, 109, 58, frame_atari_logo)

        frame_target = jr.get_sprite_frame(self.sprites["yellow_diamond_top"], 0)
        raster = jax.lax.cond(
            state.active_targets[0],
            lambda r: jr.render_at(raster, 60, 24, frame_target),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            state.active_targets[1],
            lambda r: jr.render_at(raster, 76, 24, frame_target),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            state.active_targets[2],
            lambda r: jr.render_at(raster, 92, 24, frame_target),
            lambda r: raster,
            operand=raster,
        )

        frame_special_target = jr.get_sprite_frame(
            self.sprites["yellow_diamond_bottom"], 0
        )
        raster = jax.lax.cond(
            state.active_targets[3],
            lambda r: jr.render_at(raster, 76, 120, frame_special_target),
            lambda r: raster,
            operand=raster,
        )

        # Render Atari Logos and the X
        raster = jax.lax.cond(
            jnp.logical_and(state.atari_symbols > 0, state.respawn_timer == 0),
            lambda r: jr.render_at(raster, 60, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            jnp.logical_and(
                jnp.logical_or(state.atari_symbols == 2, state.atari_symbols == 3),
                state.respawn_timer == 0,
            ),
            lambda r: jr.render_at(raster, 76, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster,
        )

        raster = jax.lax.cond(
            jnp.logical_and(state.atari_symbols > 2, state.respawn_timer == 0),
            lambda r: jr.render_at(raster, 90, 154, frame_atari_logo),
            lambda r: raster,
            operand=raster,
        )

        frame_X = jr.get_sprite_frame(self.sprites["x"], 0)
        raster = jax.lax.cond(
            jnp.logical_and(state.atari_symbols == 4, state.respawn_timer == 0),
            lambda r: jr.render_at(raster, 76, 152, frame_X),
            lambda r: raster,
            operand=raster,
        )

        # Handle color cycling
        color = jnp.ceil(state.color_cycling / jnp.array(4.0)).astype(jnp.int32)
        color = jnp.clip(color, min=0)

        raster = jax.lax.cond(
            color > 0,
            lambda r: self._handle_color_cycling(r, color),
            lambda r: r,
            raster,
        )

        # raster = render_scene_object_boundaries(raster)

        return raster
