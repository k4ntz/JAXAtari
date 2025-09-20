"""
Project: JAXAtari VideoPinball
Description: Our team's JAX implementation of Video Pinball.

Authors:
    - Michael Olenberger <michael.olenberger@stud.tu-darmstadt.de>
    - Maximilian Roth <maximilian.roth@stud.tu-darmstadt.de>
    - Jonas Neumann <jonas.neumann@stud.tu-darmstadt.de>
    - Yuddhish Chooah <yuddhish.chooah@stud.tu-darmstadt.de>


Implemented features:
- Plunger and Flipper movement logic
- Plunger physics
- Jit-compatible rendering
- Special object logic (like yellow targets, rollovers, etc.)
- Scoring
- Ball respawning upon hitting the gutter and life counter
- Basic collisions and ball physics

Why the ball physics are not yet perfect:
Video Pinball has extremely complicated ball physics. Launch angles when hitting (close to) corners are
seemingly random and velocity calculation has a variety of strange quirks like strong spontaneous acceleration
when a slow balls hit walls at certain angles, etc...
These properties are impossible to read from the RAM state and need to be investigated
frame by frame in various scenarios. Thus, the physics are far from perfect.
Collisions are mostly implemented apart from the flippers but non-reflective collisions
such as when hitting a target are still missing, even though the logic for these objects such as the targets
(like increasing bumper multiplier and respawn cooldown etc.) are already implemented so the game
is a little more complete than it seems.

Additional notes:
The renderer requires a custom function that was implemented in atraJaxis.py
We were not working on the game for almost 2 months as most of the team debated whether they
should quit the project or not, which is why the game is still in such an incomplete state.
However, we are now determined to continue.

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
    Records if UP or DOWN is being pressed and returns the corresponding action.

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
        consts = consts or VideoPinballConstants()
        super().__init__(consts)
        self.frameskip = frameskip + 1
        self.frame_stack_size = 4
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
        self.obs_size = 3 * 4 + 1 + 1
        self.renderer = VideoPinballRenderer(consts=consts)

    def reset(self, key) -> Tuple[VideoPinballObservation, VideoPinballState]:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
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
            lives=jnp.array(1, dtype=jnp.int32),
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
        # chex provides jax with additional debug/testing functionality.
        # Probably best to use it instead of simply jnp.array

        rng_key, ball_step_key = jrandom.split(state.rng_key)

        # Check if action is LEFT_FIRE or RIGHT_FIRE
        action_is_left_fire = action == Action.LEFTFIRE
        action_is_right_fire = action == Action.RIGHTFIRE
        action_is_fire = jnp.logical_or(action_is_left_fire, action_is_right_fire)

        action = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(action_is_fire), state.tilt_mode_active),
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

        # ball_x = jnp.where(test_flippers, 83, ball_x)
        # ball_y = jnp.where(test_flippers, 170, ball_y)
        # ball_direction = jnp.where(test_flippers, 1, ball_direction)
        # ball_vel_x = jnp.where(test_flippers, 3., ball_vel_x)
        # ball_vel_y = jnp.where(test_flippers, 3., ball_vel_y)

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
            lambda at: self._ball_enters_gutter(state),
            lambda at: state.respawn_timer,
            active_targets,
        )

        score = jnp.where(tilt_mode_active, state.score, score)

        (
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            lives,
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
            state.lives,
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
            lives=lives,
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
        # stack the new observation, remove the oldest one
        # observation = jax.tree.map(
        #     lambda stack, obs: jnp.concatenate(
        #         [stack[1:], jnp.expand_dims(obs, axis=0)], axis=0
        #     ),
        #     new_state.obs_stack,
        #     observation,
        # )
        # new_state = new_state._replace(obs_stack=observation)
        # jax.debug.print("------------------------------------------")

        # Check if all lives are lost and the game should reset (not fully)
        # observation, new_state = jax.lax.cond(
        #     jnp.logical_and(lives > 3, respawn_timer == 0),
        #     lambda ob, ns: self.reset(
        #         jrandom.PRNGKey(score + special_target_cooldown + env_reward)
        #     ),
        #     lambda ob, ns: (ob, ns),
        #     observation,
        #     new_state,
        # )

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: VideoPinballState):

        ball = EntityState(
            x=state.ball_x.astype(jnp.int32),
            y=state.ball_y.astype(jnp.int32),
            w=jnp.array(2),
            h=jnp.array(4),
            active=jnp.array(1),
        )

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

        return VideoPinballObservation(
            ball=ball,
            spinners=spinners,
            flippers=flippers,
            plunger=plunger,
            targets=targets,
            bumpers=bumpers,
            rollovers=rollovers,
            tilt_mode_hole_plugs=tilt_mode_hole_plugs,
            score=state.score,
            lives=state.lives,
            atari_symbols=state.atari_symbols,
            bumper_multiplier=state.bumper_multiplier,
            rollover_counter=state.rollover_counter,
            color_cycling=state.color_cycling,
            tilt_mode_active=state.tilt_mode_active,
        )

    @partial(jax.jit, static_argnums=(0,))
    def flatten_entity_state(self, entity: EntityState):
        return (
            entity.x.flatten(),
            entity.y.flatten(),
            entity.w.flatten(),
            entity.h.flatten(),
            entity.active.flatten(),
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: VideoPinballObservation) -> jnp.ndarray:
        return jnp.concatenate(
            [
                self.flatten_entity_state(obs.ball),
                self.flatten_entity_state(obs.spinners),
                self.flatten_entity_state(obs.flippers),
                self.flatten_entity_state(obs.plunger),
                self.flatten_entity_state(obs.targets),
                self.flatten_entity_state(obs.bumpers),
                self.flatten_entity_state(obs.rollovers),
                self.flatten_entity_state(obs.tilt_mode_hole_plugs),
                obs.score.flatten(),
                obs.lives.flatten(),
                obs.atari_symbols.flatten(),
                obs.bumper_multiplier.flatten(),
                obs.rollover_counter.flatten(),
                obs.color_cycling.flatten(),
                obs.tilt_mode_active.flatten(),
            ]
        )

    def action_space(self) -> spaces.Discrete:
        """Returns the action space for Pong.
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
        return spaces.Dict(
            {
                "ball": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                    }
                ),
                "flippers": spaces.Box(low=0, high=210, shape=(2, 5), dtype=jnp.int32),
                "plunger": spaces.Dict(
                    {
                        "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "w": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                        "h": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                        "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                    }
                ),
                "targets": spaces.Box(low=0, high=210, shape=(4, 5), dtype=jnp.int32),
                "title_mode_hole_plugs": spaces.Box(
                    low=0, high=210, shape=(2, 5), dtype=jnp.int32
                ),
                "score": spaces.Box(low=0, high=999999, shape=(), dtype=jnp.int32),
                "lives": spaces.Box(low=0, high=4, shape=(), dtype=jnp.int32),
                "atari_symbols": spaces.Box(low=0, high=4, shape=(), dtype=jnp.int32),
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
        return jnp.subtract(state.score, previous_state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_reward(
        self, previous_state: VideoPinballState, state: VideoPinballState
    ):
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: VideoPinballState) -> bool:
        return jnp.logical_and(state.lives > 3, state.ball_in_play == False)

    def render(self, state) -> jnp.ndarray:
        return self.renderer.render(state)

    @partial(jax.jit, static_argnums=(0,))
    def _plunger_step(self, state: VideoPinballState, action: chex.Array) -> chex.Array:
        """
        Update the plunger position based on the current state and action.
        And set the plunger power to 2 * plunger_position.
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

        # If FIRE
        plunger_power = jax.lax.cond(
            jnp.logical_and(action == Action.FIRE, jnp.logical_not(state.ball_in_play)),
            lambda s: s / self.consts.PLUNGER_MAX_POSITION * self.consts.BALL_MAX_SPEED,
            lambda s: 0.0,
            operand=plunger_position,
        )

        plunger_position = jax.lax.cond(
            plunger_power > 0, lambda p: 0, lambda p: p, operand=plunger_position
        )

        return plunger_position, plunger_power

    @partial(jax.jit, static_argnums=(0,))
    def _flipper_step(self, state: VideoPinballState, action: chex.Array):

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
    def _default_triangle_collision_branch(
        self,
        ball_movement,
        scene_object,  # only used for returning hit_point
        action,
        ax,
        ay,
        bx,
        by,
        cx,
        cy,
    ):
        """
        Compute first intersection of the segment (old -> new) with triangle ABC.
        Returns a hit_point array in the same format your slab routine returns:
          [t_entry, hit_x, hit_y, new_ball_x, new_ball_y, <scene_object...>]
        If no collision, returns _dummy_calc_hit_point(scene_object)[-1] (keeps same behavior).
        """
        hit_point_ab = self._calc_segment_hit_point(
            ball_movement, scene_object, action, ax, ay, bx, by
        )
        hit_point_bc = self._calc_segment_hit_point(
            ball_movement, scene_object, action, bx, by, cx, cy
        )
        hit_point_ca = self._calc_segment_hit_point(
            ball_movement, scene_object, action, cx, cy, ax, ay
        )

        argmin = jnp.argmin(
            jnp.array(
                [
                    hit_point_ab[HitPointSelector.T_ENTRY],
                    hit_point_bc[HitPointSelector.T_ENTRY],
                    hit_point_ca[HitPointSelector.T_ENTRY],
                ]
            )
        )
        hit_points = jnp.stack([hit_point_ab, hit_point_bc, hit_point_ca])

        hit_point = hit_points[argmin]

        return hit_point

    @partial(jax.jit, static_argnums=(0,))
    def _cross2(self, ax_, ay_, bx_, by_):
        return ax_ * by_ - ay_ * bx_

    @partial(jax.jit, static_argnums=(0,))
    def _intersect_edge(self, ball_movement, ax, ay, a_to_b_x, a_to_b_y):
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
        self, ball_movement, scene_object, action, ax, ay, bx, by
    ):
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
        r = (d_traj - d_hit) / d_traj
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
    ):
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
    def _skip_ball_movement_collision_branch(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
        cx: chex.Array,
        cy: chex.Array,
    ):
        # calculate closest hit point on upper edge (a -> c)
        eps = 1e-8
        vx, vy = jnp.subtract(cx, ax), jnp.subtract(cy, ay)
        wx, wy = jnp.subtract(ball_movement.new_ball_x, ax), jnp.subtract(
            ball_movement.new_ball_y, ay
        )
        vv = vx * vx + vy * vy + eps
        t = (vx * wx + vy * wy) / vv
        t_clamped = jnp.clip(t, 0.0, 1.0)
        hx = ax + t_clamped * vx
        hy = ay + t_clamped * vy

        cvx, cvy = jnp.subtract(cx, ax), jnp.subtract(cy, ay)
        nx, ny = -cvy, cvx
        nlen = jnp.sqrt(nx * nx + ny * ny) + eps
        nx, ny = nx / nlen, ny / nlen

        # orient normal away from triangle
        centroid_x, centroid_y = jnp.divide(
            jnp.add(ax, jnp.add(bx, cx)), 3.0
        ), jnp.divide(jnp.add(ay, jnp.add(by, cy)), 3.0)
        inward_dot = nx * (centroid_x - hx) + ny * (centroid_y - hy)
        nx, ny = jnp.where(inward_dot > 0.0, -nx, nx), jnp.where(
            inward_dot > 0.0, -ny, ny
        )

        # Do a small correction so that the new ball movement begins outside of the triangle
        hx, hy = hx + nx * 0.1, hy + ny * 0.1

        traj_x = jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        traj_y = jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)

        traj_x = jnp.where(jnp.abs(traj_x) < 1e-8, 1e-8, traj_x)
        traj_y = jnp.where(jnp.abs(traj_y) < 1e-8, 1e-8, traj_y)

        vel_dot = traj_x * nx + traj_y * ny
        reflected_x = traj_x - 2.0 * vel_dot * nx
        reflected_y = traj_y - 2.0 * vel_dot * ny

        d_traj = jnp.sqrt(traj_x * traj_x + traj_y * traj_y) + eps
        rlen = jnp.sqrt(reflected_x * reflected_x + reflected_y * reflected_y) + eps

        reflected_x = reflected_x / rlen * d_traj
        reflected_y = reflected_y / rlen * d_traj
        new_ball_x, new_ball_y = hx + reflected_x, hy + reflected_y

        hit_point = jnp.concatenate(
            [jnp.stack([0.0, hx, hy, new_ball_x, new_ball_y], axis=0), scene_object],
            axis=0,
        )

        return hit_point

    @partial(jax.jit, static_argnums=(0,))
    def _rewind_ball_movement_collision_branch(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
        cx: chex.Array,
        cy: chex.Array,
    ):
        dx = jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        dy = jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)

        dx = jnp.where(jnp.abs(dx) < 1e-8, 1e-8, dx)
        dy = jnp.where(jnp.abs(dy) < 1e-8, 1e-8, dy)

        # Make a fake ball movement with reversed direction
        reversed_movement = BallMovement(
            old_ball_x=ball_movement.old_ball_x,
            old_ball_y=ball_movement.old_ball_y,
            new_ball_x=ball_movement.old_ball_x - dx,
            new_ball_y=ball_movement.old_ball_y - dy,
        )

        # Find intersection with each triangle edge
        t_ab, valid_ab = self._intersect_edge(
            reversed_movement, ax, ay, jnp.subtract(bx, ax), jnp.subtract(by, ay)
        )
        t_bc, valid_bc = self._intersect_edge(
            reversed_movement, bx, by, jnp.subtract(cx, bx), jnp.subtract(cy, by)
        )
        t_ca, valid_ca = self._intersect_edge(
            reversed_movement, cx, cy, jnp.subtract(ax, cx), jnp.subtract(ay, cy)
        )

        ts = jnp.stack(
            [
                jnp.where(valid_ab, t_ab, jnp.inf),
                jnp.where(valid_bc, t_bc, jnp.inf),
                jnp.where(valid_ca, t_ca, jnp.inf),
            ]
        )

        t_exit = jnp.min(ts)  # first boundary crossing

        # Now compute the rewind step
        old_x = ball_movement.old_ball_x - t_exit * dx
        old_y = ball_movement.old_ball_y - t_exit * dy
        new_x = old_x + dx
        new_y = old_y + dy

        ball_movement = BallMovement(
            old_ball_x=old_x, old_ball_y=old_y, new_ball_x=new_x, new_ball_y=new_y
        )

        return self._default_triangle_collision_branch(
            ball_movement, scene_object, action, ax, ay, bx, by, cx, cy
        )

    @partial(jax.jit, static_argnums=(0,))
    def _inside_triangle_collision_branch(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
        cx: chex.Array,
        cy: chex.Array,
        flipper_up: chex.Array,
        flipper_down: chex.Array,
    ):
        trajectory_x = jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        trajectory_y = jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)
        ball_direction = self._get_ball_direction(trajectory_x, trajectory_y)

        movement_up = jnp.logical_or(ball_direction == 2, ball_direction == 0)

        up_up = jnp.logical_and(movement_up, flipper_up)
        down_down = jnp.logical_and(jnp.logical_not(movement_up), flipper_down)

        # if the flipper moves in the same direction as the ball, we reflect the ball at the point closest
        # to the upper edge of the triangle (in the direction of the flipper movement).
        # if the flipper and ball move in opposite directions, we rewind the ball movement until it is outside
        # of the triangle and reflect it in the default way
        return jax.lax.cond(
            jnp.logical_or(up_up, down_down),
            lambda: self._skip_ball_movement_collision_branch(
                ball_movement, scene_object, action, ax, ay, bx, by, cx, cy
            ),
            lambda: self._rewind_ball_movement_collision_branch(
                ball_movement, scene_object, action, ax, ay, bx, by, cx, cy
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _default_slab_collision_branch(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ):
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

        r = 1 - d_hit_point / d_trajectory

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
    ):
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
    ):
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
    def _calc_triangle_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
        ax: chex.Array,
        ay: chex.Array,
        bx: chex.Array,
        by: chex.Array,
        cx: chex.Array,
        cy: chex.Array,
        flipper_up: chex.Array,
        flipper_down: chex.Array,
    ):
        return jax.lax.cond(
            self._is_inside_triangle(ball_movement, ax, ay, bx, by, cx, cy),
            lambda: self._inside_triangle_collision_branch(
                ball_movement,
                scene_object,
                action,
                ax,
                ay,
                bx,
                by,
                cx,
                cy,
                flipper_up,
                flipper_down,
            ),
            lambda: self._default_triangle_collision_branch(
                ball_movement, scene_object, action, ax, ay, bx, by, cx, cy
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _calc_flipper_hit_point(
        self,
        ball_movement: BallMovement,
        scene_object: chex.Array,
        action: chex.Array,
    ):
        is_left_flipper = scene_object[5] == 9
        is_right_flipper = scene_object[5] == 10
        left_flipper_up = jnp.logical_or(action == Action.LEFT, action == Action.UP)
        right_flipper_up = jnp.logical_or(action == Action.RIGHT, action == Action.UP)
        flipper_at_max_pos = scene_object[6] % 4 == 3
        flipper_at_min_pos = scene_object[6] % 4 == 0

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

        # left flipper -> bottom left corner
        # right flipper -> bottom right corner
        px = jnp.where(
            is_left_flipper, scene_object[2], scene_object[2] + scene_object[0]
        )
        py = scene_object[3] + scene_object[1]

        # left flipper -> top right corner
        # right flipper -> top left corner
        endx = jnp.where(
            is_left_flipper, scene_object[2] + scene_object[0], scene_object[2]
        )
        endy = scene_object[3]
        L = jnp.sqrt(
            scene_object[0] ** 2 + scene_object[1] ** 2
        )  # length of the line segment

        # next flipper position:
        next_pos_scene_object = jax.lax.cond(  # new angle of the line segment
            flipper_down,
            lambda: self.consts.FLIPPERS[scene_object[6] - 1],
            lambda: jax.lax.cond(
                flipper_up,
                lambda: self.consts.FLIPPERS[scene_object[6] + 1],
                lambda: self.consts.FLIPPERS[scene_object[6]],
            ),
        )

        next_pos_endx = jnp.where(
            is_left_flipper,
            next_pos_scene_object[2] + next_pos_scene_object[0],
            next_pos_scene_object[2],
        )
        next_pos_endy = next_pos_scene_object[3]

        other_side_scene_object = jax.lax.cond(
            scene_object[6] % 8 < 4,
            lambda: self.consts.FLIPPERS[scene_object[6] + 4],
            lambda: self.consts.FLIPPERS[scene_object[6] - 4],
        )
        other_px = jnp.where(
            is_left_flipper,
            other_side_scene_object[2],
            other_side_scene_object[2] + scene_object[0],
        )
        other_py = other_side_scene_object[3] + other_side_scene_object[1]

        flipper_moves = jnp.logical_or(flipper_up, flipper_down)

        hit_point = jax.lax.cond(
            flipper_moves,
            lambda: self._calc_triangle_hit_point(
                ball_movement,
                scene_object,
                action,
                px,
                py,
                endx,
                endy,
                next_pos_endx,
                next_pos_endy,
                flipper_up,
                flipper_down,
            ),
            lambda: jax.lax.cond(
                self._is_inside_triangle(
                    ball_movement, px, py, endx, endy, other_px, other_py
                ),
                lambda: self._calc_triangle_hit_point(
                    ball_movement,
                    scene_object,
                    action,
                    px,
                    py,
                    endx,
                    endy,
                    next_pos_endx,
                    next_pos_endy,
                    flipper_up,
                    flipper_down,
                ),
                lambda: self._calc_segment_hit_point(
                    ball_movement, scene_object, action, px, py, endx, endy
                ),
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
            / L
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
        Calculates collisions with spinners.
        Uses the usual slab method for hit_point calculation, then adds angular velocity.
        After adding angular velocity, it has to be ensured that the ball is flying in a
        clear trajectory from hit_point to new ball position, i.e. the ball may not collide
        with the same spinner object when adding the angular velocity.

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
        Calculate the hit point of the ball with the bounding box.
        Uses the slab method also known as ray AABB collision or swept arc collision (for flippers).

        scene_object is an array of the form
        [
            SceneObject.hit_box_width,
            SceneObject.hit_box_height,
            SceneObject.hit_box_x_offset,
            SceneObject.hit_box_y_offset,
            SceneObject.reflecting,
            SceneObject.score_type,
            SceneObject.variant
        ]

        Returns:
            hit_point: jnp.ndarray, the time and hit point of the ball with the bounding box.
            hit_point[0]: jnp.ndarray, the time of entry
            hit_point[1]: jnp.ndarray, the x position of the hit point
            hit_point[2]: jnp.ndarray, the y position of the hit point
            hit_point[3]: jnp.ndarray, whether the obstacle was hit horizontally
            hit_point[4]: jnp.ndarray, whether the obstacle was hit vertically
            hit_point[5:]: scene_object properties:
               hit_point[5]: hit_box_width
               hit_point[6]: hit_box_height
               hit_point[7]: hit_box_x_offset
               hit_point[8]: hit_box_y_offset
               hit_point[9]: reflecting
               hit_point[10]: score_type
               hit_point[11]: variant

        Hint:
            Use HitPointSelector to access the hit_point indices
        """
        # 0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover, 5: Special Lit Up Target,
        # 6: Left Lit Up Target, 7:Middle Lit Up Target, 8: Right Lit Up Target, 9: Left Flipper, 10: Right Flipper, 11: Tilt Mode Hole Plug
        dampening_value = 1 - self.consts.VELOCITY_DAMPENING_VALUE
        no_addition = 0.0
        acceleration_value = self.consts.VELOCITY_ACCELERATION_VALUE
        no_factor = 1.0
        return jax.lax.switch(
            scene_object[5],
            [
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 0
                lambda: (
                    no_factor,
                    acceleration_value,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 1
                lambda: self._calc_spinner_hit_point(
                    ball_movement, scene_object, action
                ),  # 2
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 3
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 4
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 5
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 6
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 7
                lambda: (
                    dampening_value,
                    no_addition,
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
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
                    self._calc_slab_hit_point(ball_movement, scene_object, action),
                ),  # 11
            ],
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

        is_left_lit_up_target = self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 6
        is_middle_lit_up_target = self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 7
        is_right_lit_up_target = self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 8
        is_special_lit_up_target = self.consts.NON_REFLECTING_SCENE_OBJECTS[:, 5] == 5

        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_left_lit_up_target, jnp.logical_not(is_left_lit_up_target_active)
            ),
            False,
            non_reflecting_active,  # only select hit_point[0] since we only update t_entry
        )
        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_middle_lit_up_target, jnp.logical_not(is_middle_lit_up_target_active)
            ),
            False,
            non_reflecting_active,
        )
        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_right_lit_up_target, jnp.logical_not(is_right_lit_up_target_active)
            ),
            False,
            non_reflecting_active,
        )
        non_reflecting_active = jnp.where(
            jnp.logical_and(
                is_special_lit_up_target,
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
        # We do this by setting the entry time of inactive spinner parts to a high value
        spinner_state = jnp.remainder(
            state.step_counter, 8
        )  # 0: Bottom, 1: Right, 2: Top, 3: Left

        _object_variant = self.consts.REFLECTING_SCENE_OBJECTS[:, 6]
        is_spinner = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 2
        is_left_flipper = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 9
        is_right_flipper = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 10
        is_hole_plug = self.consts.REFLECTING_SCENE_OBJECTS[:, 5] == 11
        # spinner state only switches every other game step
        spinner_active = jnp.logical_or(
            _object_variant * 2 == spinner_state,
            _object_variant * 2 + 1 == spinner_state,
        )
        # Disable all the spinner parts not matching the current step
        reflecting_active = jnp.where(
            jnp.logical_and(
                jnp.logical_not(spinner_active),
                is_spinner,  # scoring_type == spinner (2)
            ),
            self.consts.T_ENTRY_NO_COLLISION,
            reflecting_active,  # only select hit_point[0]
        )

        # Disable inactive flipper parts
        left_flipper_angle = state.left_flipper_angle
        right_flipper_angle = state.right_flipper_angle

        reflecting_active = jnp.where(
            jnp.logical_and(
                is_left_flipper,
                jnp.logical_or(
                    jnp.logical_not(_object_variant % 4 == left_flipper_angle),
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
                    jnp.logical_not(_object_variant % 4 == right_flipper_angle),
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
        ###############################################################################################
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
        ###############################################################################################

        # Note: This for-loop is permitted by jit and will be unrolled by jax

        deconstructed_scoring_list = []

        # non-reflecting scoring objects (multiple possible but only one of each type)
        # hit_before_reflection calculates for each object whether it was hit before hitting any rigid object, i.e. shape (, n_non_reflecting_objects)
        # the jnp.where(...) statement checks whether any non_reflecting_object of a specific scoring_type was hit
        hit_before_reflection = jnp.logical_and(
            non_reflecting_hit_points[:, HitPointSelector.T_ENTRY]
            < hit_point[HitPointSelector.T_ENTRY],
            non_reflecting_hit_points[:, HitPointSelector.T_ENTRY]
            != self.consts.T_ENTRY_NO_COLLISION,
        )

        for i in range(scoring_list.shape[0]):
            scoring_list_i = scoring_list[i]
            scoring_list_i = jnp.logical_or(
                scoring_list_i,
                jnp.where(hit_point[HitPointSelector.OBJECT_SCORE_TYPE] == i, 1, 0),
            )
            scoring_list_i = jnp.logical_or(
                scoring_list_i,
                jnp.where(
                    jnp.any(
                        jnp.logical_and(
                            hit_before_reflection,
                            non_reflecting_hit_points[
                                :, HitPointSelector.OBJECT_SCORE_TYPE
                            ]
                            == i,
                        ),
                        axis=0,
                    ),
                    1,
                    0,
                ),
            )

            deconstructed_scoring_list.append(scoring_list_i)

        scoring_list = jnp.stack(deconstructed_scoring_list, axis=0)

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
        #    "OLD_Y: {}\n\t"
        #    "OLD_X: {}\n\t"
        #    "NEW_Y: {}\n\t"
        #    "NEW_X: {}\n",
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
        #    ball_movement.old_ball_x, ball_movement.old_ball_y, ball_movement.new_ball_x, ball_movement.new_ball_y,
        # )

        return hit_point, scoring_list, velocity_factor, velocity_addition

    @partial(jax.jit, static_argnums=(0,))
    def _calc_ball_collision_loop(
        self, state: VideoPinballState, ball_movement: BallMovement, action: chex.Array
    ):

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
                hit_data[HitPointSelector.T_ENTRY] == self.consts.T_ENTRY_NO_COLLISION
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
            hit_is_right_flipper = hit_data[HitPointSelector.OBJECT_SCORE_TYPE] == 10
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

        def _fori_body(i, carry):
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
            state.left_flipper_active,
            state.right_flipper_active,
            jnp.zeros((12,), dtype=bool),  # scoring_list
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
        # If both values are negative, we move closer to (0, 0) in the top left corner and fly in direction 0
        top_left = jnp.logical_and(signed_vel_x <= 0, signed_vel_y <= 0)  # 0
        top_right = jnp.logical_and(signed_vel_x > 0, signed_vel_y <= 0)  # 2
        bottom_right = jnp.logical_and(signed_vel_x > 0, signed_vel_y > 0)  # 3
        bottom_left = jnp.logical_and(signed_vel_x <= 0, signed_vel_y > 0)  # 1

        bool_array = jnp.array([top_left, bottom_left, top_right, bottom_right])
        return jnp.argmax(bool_array)

    @partial(jax.jit, static_argnums=(0,))
    def _calc_ball_change(self, ball_x, ball_y, ball_vel_x, ball_vel_y, ball_direction):
        sign_x, sign_y = self._get_ball_direction_signs(ball_direction)
        ball_vel_x = jnp.clip(ball_vel_x, 0, self.consts.BALL_MAX_SPEED)
        ball_vel_y = jnp.clip(ball_vel_y, 0, self.consts.BALL_MAX_SPEED)
        signed_ball_vel_x = sign_x * ball_vel_x
        signed_ball_vel_y = sign_y * ball_vel_y
        # Only change position, direction and velocity if the ball is in play
        # TODO override ball_x, ball_y if obstacle hit (_reflect_ball)
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
        # branch when there *is* a nudge (nudge_direction != 0)
        def _nudge_branch(state: VideoPinballState, action: Action, ball_x: chex.Array):
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

            # adjust horizontal velocity depending on nudge direction
            ball_x_new = jax.lax.cond(
                jnp.logical_and(
                    jnp.equal(action, Action.RIGHTFIRE),
                    jnp.remainder(state.step_counter, self.consts.NUDGE_EFFECT_INTERVAL)
                    == 0,
                ),
                lambda bv: bv + self.consts.NUDGE_EFFECT_AMOUNT,
                lambda bv: bv - self.consts.NUDGE_EFFECT_AMOUNT,
                ball_x,
            )

            return tilt_mode_from_counter, tilt_counter_capped, ball_x_new

        def _no_nudge_branch(
            state: VideoPinballState, action: Action, ball_x: chex.Array
        ):
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
        ball_vel_x = jnp.abs(
            jnp.subtract(ball_movement.new_ball_x, ball_movement.old_ball_x)
        )
        ball_vel_y = jnp.abs(
            jnp.subtract(ball_movement.new_ball_y, ball_movement.old_ball_y)
        )
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
        r = 1 - d_hit / d_traj
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
        Update the pinballs position and velocity based on the current state and action.
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
        Gravity/Center Pull calculation
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
            old_ball_x=state.ball_x,  # type: ignore
            old_ball_y=state.ball_y,  # type: ignore
            new_ball_x=ball_x,  # type: ignore
            new_ball_y=ball_y,  # type: ignore
        )
        """
        Check if the ball is hitting the invisible block at the plunger hole
        """
        ball_movement, ball_vel_x, ball_vel_y, is_invisible_block_hit = (
            self._calc_invisible_block_hit(ball_movement, ball_in_play, action, key)
        )

        ball_in_play = jnp.logical_or(ball_in_play, is_invisible_block_hit)

        """
        Obstacle hit calculation
        """
        # Calculate whether and where obstacles are hit
        # If a reflecting obstacle is hit, reflect the ball
        # If a non-reflecting obstacle is hit, proceed with usual ball position calculation
        # new_ball_direction, _ball_vel_x, _ball_vel_y = _get_obstacle_hit_direction()
        (
            collision_ball_movement,
            scoring_list,
            velocity_factor,
            velocity_addition,
            left_flipper_active,
            right_flipper_active,
            any_collision,
        ) = self._calc_ball_collision_loop(state, ball_movement, action)

        ball_trajectory_x = (
            collision_ball_movement.new_ball_x - collision_ball_movement.old_ball_x
        )
        ball_trajectory_y = (
            collision_ball_movement.new_ball_y - collision_ball_movement.old_ball_y
        )

        ball_x = collision_ball_movement.new_ball_x
        ball_y = collision_ball_movement.new_ball_y

        """
        Some final calculations
        """
        ball_direction = self._get_ball_direction(ball_trajectory_x, ball_trajectory_y)
        ball_vel_x = jnp.abs(ball_vel_x)
        ball_vel_y = jnp.abs(ball_vel_y)
        original_ball_speed = jnp.sqrt(ball_vel_x**2 + ball_vel_y**2)
        new_ball_speed = (
            (1 + jnp.clip(velocity_addition, -1, 1))
            * original_ball_speed
            * velocity_factor
        )

        ball_vel_x = jnp.where(
            any_collision,
            jnp.clip(
                ball_vel_x / original_ball_speed * new_ball_speed,
                0,
                self.consts.BALL_MAX_SPEED,
            ),
            ball_vel_x,
        )
        ball_vel_y = jnp.where(
            any_collision,
            jnp.clip(
                ball_vel_y / original_ball_speed * new_ball_speed,
                0,
                self.consts.BALL_MAX_SPEED,
            ),
            ball_vel_y,
        )

        # If ball velocity reaches a small threshold, accelerate it after hitting something
        small_vel = jnp.logical_and(
            ball_vel_x < self.consts.BALL_MIN_SPEED,
            ball_vel_y < self.consts.BALL_MIN_SPEED,
        )
        ball_vel_x = jnp.where(
            jnp.logical_and(any_collision, small_vel),
            jnp.clip(
                ball_vel_x * 2 + self.consts.BALL_MIN_SPEED,
                0,
                self.consts.BALL_MAX_SPEED,
            ),
            ball_vel_x,
        )
        ball_vel_y = jnp.where(
            jnp.logical_and(any_collision, small_vel),
            jnp.clip(
                ball_vel_y * 2 + self.consts.BALL_MIN_SPEED,
                0,
                self.consts.BALL_MAX_SPEED,
            ),
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
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(False),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _ball_enters_gutter(self, state: VideoPinballState):

        respawn_timer = jnp.where(
            state.rollover_counter > 1,
            jnp.array((state.rollover_counter - 1) * 16).astype(jnp.int32),
            jnp.array(1).astype(jnp.int32),
        )

        return respawn_timer

    @partial(jax.jit, static_argnums=(0,))
    def _handle_ball_in_gutter(
        self,
        rt,
        rollover_counter,
        score,
        atari_symbols,
        lives,
        active_targets,
        special_target_cooldown,
        tilt_mode_active,
        tilt_counter,
    ):

        multiplier = jnp.clip(atari_symbols + 1, max=4)
        score = jnp.where(rt % 16 == 15, score + 1000 * multiplier, score)
        rollover_counter = jnp.where(
            rt % 16 == 15, rollover_counter - 1, rollover_counter
        )
        respawn_timer = jnp.where(rt > 0, rt - 1, rt)

        lives, active_targets, atari_symbols, special_target_cooldown = jax.lax.cond(
            respawn_timer == 0,
            lambda l, asym: self._reset_stuff_and_handle_lives(l, asym),
            lambda l, asym: (l, active_targets, asym, special_target_cooldown),
            lives,
            atari_symbols,
        )

        tilt_mode_active = jnp.where(respawn_timer == 0, False, tilt_mode_active)
        tilt_counter = jnp.where(respawn_timer == 0, 0, tilt_counter)

        return (
            respawn_timer,
            rollover_counter,
            score,
            atari_symbols,
            lives,
            active_targets,
            special_target_cooldown,
            tilt_mode_active,
            tilt_counter,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _reset_stuff_and_handle_lives(self, lives, atari_symbols):
        lives = jax.lax.cond(
            atari_symbols < 4,
            lambda x: x + 1,
            lambda x: x,
            operand=lives,
        )

        active_targets = jnp.array([True, True, True, False]).astype(jnp.bool)
        atari_symbols = jnp.array(0).astype(jnp.int32)
        special_target_cooldown = jnp.array(0).astype(jnp.int32)

        # TODO: Whoever implements tilt mode: Turn off tilt mode here in this function

        return lives, active_targets, atari_symbols, special_target_cooldown

    @partial(jax.jit, static_argnums=(0,))
    def _process_objects_hit(self, state: VideoPinballState, objects_hit):
        # Bumpers: Give points
        # Targets: Make them disappear, give points
        # Targets: Check if all hit, increase multiplier
        # BonusTarget: Give points, make screen flash, something else?
        # Rollover: Give points, increase number
        # Atari: Give points, make Atari symbol at bottom appear
        # Assume objects_hit is list:
        # [0: no score, 1: Bumper, 2: Spinner, 3: Left Rollover, 4: Atari Rollover,
        # 5: Special Lit Up Target, 6: Left Lit Up Target, 7: Middle Lit Up Target, 8: Right Lit Up Target]

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
            ),  # Check how the real cooldown works
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

        # sprite_wall_bottom_left_square = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallBottomLeftSquare.npy"), transpose=True)
        # sprite_wall_bumper = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallBumper.npy"), transpose=True)
        # sprite_wall_rollover = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "Wallrollover.npy"), transpose=True)
        # sprite_wall_left_l = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallLeftL.npy"), transpose=True)
        # sprite_wall_outer = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallOuter.npy"), transpose=True)
        # sprite_wall_right_l = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallRightL.npy"), transpose=True)
        # sprite_wall_small_horizontal = jr.loadFrame(os.path.join(SPRITES_BASE_DIR, "WallSmallHorizontal.npy"), transpose=True)
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

        # sprites_flipper_left = jnp.concatenate([
        #     jnp.repeat(sprites_flipper_left[0][None], 2, axis=0),
        #     jnp.repeat(sprites_flipper_left[1][None], 2, axis=0),
        #     jnp.repeat(sprites_flipper_left[2][None], 2, axis=0),
        #     jnp.repeat(sprites_flipper_left[3][None], 2, axis=0)
        # ])

        sprites_flipper_right, _ = jr.pad_to_match(
            [
                sprite_flipper_right0,
                sprite_flipper_right1,
                sprite_flipper_right2,
                sprite_flipper_right3,
            ]
        )

        # sprites_flipper_right = jnp.concatenate([
        #     jnp.repeat(sprites_flipper_right[0][None], 2, axis=0),
        #     jnp.repeat(sprites_flipper_right[1][None], 2, axis=0),
        #     jnp.repeat(sprites_flipper_right[2][None], 2, axis=0),
        #     jnp.repeat(sprites_flipper_right[3][None], 2, axis=0)
        # ])

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

        displayed_lives = jnp.clip(state.lives, max=3)
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
