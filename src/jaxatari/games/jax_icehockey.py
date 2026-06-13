import os
from functools import partial
from typing import Tuple, Dict, Any, Optional
import jax.lax
import jax.numpy as jnp
import chex
import jax.random as random
import jax
import numpy as np
from flax import struct

# copy from tennis
import jaxatari.rendering.jax_rendering_utils as render_utils
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import AutoDerivedConstants

class IceHockeyConstants(AutoDerivedConstants):
    MAX_SHOOTING_ANGLE: int
    PLAYER_SPEED: float
    PUCK_SPEED: float
    MIN_VERTICAL_DISTANCE: float
    MAX_PUCK_SPEED: float
    PUCK_SPEED_DECAY: float
    TIME_LIMIT: int
    MIN_SHOOTING_INTERVAL: int
    FACE_OFF_FRAMES: int
    MAX_PUSH_DISTANCE: float
    FRAMES_TACKLED: int = 60
    RINK_TOP_Y: float = 28.0
    RINK_BOTTOM_Y: float = 180.0
    PLAYER_HALF_W: float = 4.0
    PLAYER_HALF_H: float = 8.0
    GOAL_TOP_Y: float = 92.0
    GOAL_BOTTOM_Y: float = 122.0
    LEFT_GOAL_X: float = 16.0
    RIGHT_GOAL_X: float = 144.0
    PLAYER_FACEOFF_X: float = 68.0
    ENEMY_FACEOFF_X: float = 92.0
    FACEOFF_Y_1: float = 100.0
    FACEOFF_Y_2: float = 116.0
    FACEOFF_CENTER_X: float = 80.0
    FACEOFF_CENTER_Y: float = 108.0

@struct.dataclass
class GameState:
    pause_counter: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    is_finished: chex.Array
    remaining_time: chex.Array
    is_faceoff: chex.Array
    goal_scored: chex.Array

@struct.dataclass
class CharacterState:
    is_tackled: chex.Array  # counter; >0 means downed
    position: chex.Array    # (x, y)
    orientation: chex.Array # 0 left, 1 right
    has_puck: chex.Array
    shooting_cooldown: chex.Array


@struct.dataclass
class PuckState:
    position: chex.Array      # (x, y)
    velocity: chex.Array      # (vx, vy)
    direction: chex.Array     # 32 possible directions
    position_stick: chex.Array


@struct.dataclass
class AnimatorState:
    player_frame: chex.Array
    enemy_frame: chex.Array
    player_stick_frame: chex.Array
    player_stick_animation: chex.Array
    enemy_stick_frame: chex.Array
    enemy_stick_animation: chex.Array

@struct.dataclass
class EnemyState:
    enemy1: CharacterState
    enemy2: CharacterState
    active_character: chex.Array # 0 or 1

@struct.dataclass
class PlayerState:
    player1: CharacterState
    player2: CharacterState
    active_character: chex.Array # 0 or 1

@struct.dataclass
class IceHockeyState:
    player_state: PlayerState
    enemy_state: EnemyState
    puck_state: PuckState
    counter: chex.Array
    animator_state: AnimatorState
    game_state: GameState

@struct.dataclass
class IceHockeyInfo:
    pass

@struct.dataclass
class IceHockeyObservation:
    pass

class JaxIceHockey(JaxEnvironment):

    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        self.consts = consts

    def reset(self, key):
        pass

    def step(self, state, action):
        pass

    def _resolve_active_character(
        self,
        char1: CharacterState,
        char2: CharacterState,
        puck_position: chex.Array,
        current_active: chex.Array,
    ) -> chex.Array:
        dist1_sq = jnp.sum((char1.position - puck_position) ** 2)
        dist2_sq = jnp.sum((char2.position - puck_position) ** 2)
        closest = jnp.where(
            dist1_sq < dist2_sq,
            0,
            jnp.where(dist2_sq < dist1_sq, 1, current_active),
        )
        return closest.astype(jnp.int32)

    def _apply_action(
        self,
        character: CharacterState,
        action: chex.Array,
        bounds: chex.Array,
        velocity: chex.Array,
    ) -> CharacterState:
        up = jnp.any(jnp.array([
            action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT,
            action == Action.UPFIRE, action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE,
        ]))
        down = jnp.any(jnp.array([
            action == Action.DOWN, action == Action.DOWNRIGHT, action == Action.DOWNLEFT,
            action == Action.DOWNFIRE, action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        left = jnp.any(jnp.array([
            action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT,
            action == Action.LEFTFIRE, action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE,
        ]))
        right = jnp.any(jnp.array([
            action == Action.RIGHT, action == Action.UPRIGHT, action == Action.DOWNRIGHT,
            action == Action.RIGHTFIRE, action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE,
        ]))

        movable = jnp.logical_not(character.is_tackled)
        dx = jnp.where(movable & right, velocity, jnp.where(movable & left, -velocity, 0.0))
        # screen y grows downward
        dy = jnp.where(movable & down, velocity, jnp.where(movable & up, -velocity, 0.0))

        new_x = jnp.clip(character.position[0] + dx, bounds[0], bounds[1])
        new_y = jnp.clip(character.position[1] + dy, bounds[2], bounds[3])
        new_position = jnp.array([new_x, new_y])

        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        return character.replace(position=new_position, orientation=new_orientation)

    def _apply_team_inputs(
        self,
        char1: CharacterState,
        char2: CharacterState,
        active: chex.Array,
        action: chex.Array,
        bounds1: chex.Array,
        bounds2: chex.Array,
        velocity: chex.Array,
    ) -> Tuple[CharacterState, CharacterState]:
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)
        return (
            self._apply_action(char1, action1, bounds1, velocity),
            self._apply_action(char2, action2, bounds2, velocity),
        )

    def _separate_opponents(
        self,
        pos_a: chex.Array,
        pos_b: chex.Array,
        min_separation: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        delta = pos_a - pos_b
        dist_sq = jnp.sum(delta ** 2)
        overlapping = dist_sq < min_separation ** 2

        coincident = dist_sq <= 0.0
        inv_dist = jnp.where(coincident, 0.0, jax.lax.rsqrt(dist_sq))
        offset = 0.5 * delta * (min_separation * inv_dist - 1.0)
        # coincident centres: default push along x
        offset = jnp.where(coincident, jnp.array([min_separation * 0.5, 0.0]), offset)
        offset = jnp.where(overlapping, offset, jnp.array([0.0, 0.0]))
        return pos_a + offset, pos_b - offset

    def _enforce_min_vertical(
        self,
        active_pos: chex.Array,
        passive_pos: chex.Array,
        min_vertical_distance: chex.Array,
    ) -> chex.Array:
        dy = passive_pos[1] - active_pos[1]
        too_close = jnp.abs(dy) < min_vertical_distance
        # keep the passive on the same side; default below if exactly level
        side = jnp.where(dy != 0.0, jnp.sign(dy), 1.0)
        new_y = jnp.where(
            too_close, active_pos[1] + side * min_vertical_distance, passive_pos[1]
        )
        return jnp.array([passive_pos[0], new_y])

    def _clamp_to_bounds(self, pos: chex.Array, bounds: chex.Array) -> chex.Array:
        return jnp.array([
            jnp.clip(pos[0], bounds[0], bounds[1]),
            jnp.clip(pos[1], bounds[2], bounds[3]),
        ])

    def _resolve_interactions(
        self,
        player1: CharacterState,
        player2: CharacterState,
        enemy1: CharacterState,
        enemy2: CharacterState,
        player_active: chex.Array,
        enemy_active: chex.Array,
        min_separation: chex.Array,
        min_vertical_distance: chex.Array,
        bounds_p1: chex.Array,
        bounds_p2: chex.Array,
        bounds_e1: chex.Array,
        bounds_e2: chex.Array,
    ) -> Tuple[CharacterState, CharacterState, CharacterState, CharacterState]:
        p1, p2 = player1.position, player2.position
        e1, e2 = enemy1.position, enemy2.position

        # opponent separations
        p1, e1 = self._separate_opponents(p1, e1, min_separation)
        p1, e2 = self._separate_opponents(p1, e2, min_separation)
        p2, e1 = self._separate_opponents(p2, e1, min_separation)
        p2, e2 = self._separate_opponents(p2, e2, min_separation)

        # same-team vertical push — active holds, teammate yields
        p2 = jnp.where(player_active == 0, self._enforce_min_vertical(p1, p2, min_vertical_distance), p2)
        p1 = jnp.where(player_active == 1, self._enforce_min_vertical(p2, p1, min_vertical_distance), p1)
        e2 = jnp.where(enemy_active == 0, self._enforce_min_vertical(e1, e2, min_vertical_distance), e2)
        e1 = jnp.where(enemy_active == 1, self._enforce_min_vertical(e2, e1, min_vertical_distance), e1)

        # authoritative clamp
        p1 = self._clamp_to_bounds(p1, bounds_p1)
        p2 = self._clamp_to_bounds(p2, bounds_p2)
        e1 = self._clamp_to_bounds(e1, bounds_e1)
        e2 = self._clamp_to_bounds(e2, bounds_e2)

        return (
            player1.replace(position=p1),
            player2.replace(position=p2),
            enemy1.replace(position=e1),
            enemy2.replace(position=e2),
        )

    def _characters_step(
        self,
        player_state: PlayerState,
        enemy_state: EnemyState,
        puck_position: chex.Array,
        player_action: chex.Array,
        enemy_action: chex.Array,
        velocity: chex.Array,
        min_separation: chex.Array,
        min_vertical_distance: chex.Array,
        bounds_p1: chex.Array,
        bounds_p2: chex.Array,
        bounds_e1: chex.Array,
        bounds_e2: chex.Array,
    ) -> Tuple[PlayerState, EnemyState]:
        player_active = self._resolve_active_character(
            player_state.player1, player_state.player2,
            puck_position, player_state.active_character,
        )
        enemy_active = self._resolve_active_character(
            enemy_state.enemy1, enemy_state.enemy2,
            puck_position, enemy_state.active_character,
        )

        p1, p2 = self._apply_team_inputs(
            player_state.player1, player_state.player2,
            player_active, player_action, bounds_p1, bounds_p2, velocity,
        )
        e1, e2 = self._apply_team_inputs(
            enemy_state.enemy1, enemy_state.enemy2,
            enemy_active, enemy_action, bounds_e1, bounds_e2, velocity,
        )

        p1, p2, e1, e2 = self._resolve_interactions(
            p1, p2, e1, e2,
            player_active, enemy_active,
            min_separation, min_vertical_distance,
            bounds_p1, bounds_p2, bounds_e1, bounds_e2,
        )

        new_player_state = player_state.replace(
            player1=p1, player2=p2, active_character=player_active,
        )
        new_enemy_state = enemy_state.replace(
            enemy1=e1, enemy2=e2, active_character=enemy_active,
        )
        return new_player_state, new_enemy_state

    def _puck_deflection_step(self, state: IceHockeyState) -> IceHockeyState:
        puck = state.puck_state
        any_has_puck = (
            state.player_state.player1.has_puck
            | state.player_state.player2.has_puck
            | state.enemy_state.enemy1.has_puck
            | state.enemy_state.enemy2.has_puck
        )
        free = jnp.logical_not(any_has_puck)

        pos_y = puck.position[1]
        vel_y = puck.velocity[1]

        hit_top    = free & (pos_y <= self.consts.RINK_TOP_Y)
        hit_bottom = free & (pos_y >= self.consts.RINK_BOTTOM_Y)
        new_vel_y  = jnp.where(hit_top | hit_bottom, -vel_y, vel_y)
        new_pos_y  = jnp.clip(pos_y, self.consts.RINK_TOP_Y, self.consts.RINK_BOTTOM_Y)

        new_puck = puck.replace(
            position=jnp.array([puck.position[0], new_pos_y]),
            velocity=jnp.array([puck.velocity[0], new_vel_y]),
        )
        return state.replace(puck_state=new_puck)

    def _collision_step(self, state: IceHockeyState) -> IceHockeyState:
        p1 = state.player_state.player1
        p2 = state.player_state.player2
        e1 = state.enemy_state.enemy1
        e2 = state.enemy_state.enemy2

        # tick down tackle counters
        p1 = p1.replace(is_tackled=jnp.maximum(0, p1.is_tackled - 1))
        p2 = p2.replace(is_tackled=jnp.maximum(0, p2.is_tackled - 1))
        e1 = e1.replace(is_tackled=jnp.maximum(0, e1.is_tackled - 1))
        e2 = e2.replace(is_tackled=jnp.maximum(0, e2.is_tackled - 1))

        sep_x = 2.0 * self.consts.PLAYER_HALF_W
        sep_y = 2.0 * self.consts.PLAYER_HALF_H

        def _overlaps(pos_a: chex.Array, pos_b: chex.Array) -> chex.Array:
            return (
                (jnp.abs(pos_a[0] - pos_b[0]) < sep_x)
                & (jnp.abs(pos_a[1] - pos_b[1]) < sep_y)
            )

        p1e1 = _overlaps(p1.position, e1.position)
        p1e2 = _overlaps(p1.position, e2.position)
        p2e1 = _overlaps(p2.position, e1.position)
        p2e2 = _overlaps(p2.position, e2.position)

        ft = jnp.int32(self.consts.FRAMES_TACKLED)
        p1 = p1.replace(is_tackled=jnp.where(p1e1 | p1e2, ft, p1.is_tackled))
        p2 = p2.replace(is_tackled=jnp.where(p2e1 | p2e2, ft, p2.is_tackled))
        e1 = e1.replace(is_tackled=jnp.where(p1e1 | p2e1, ft, e1.is_tackled))
        e2 = e2.replace(is_tackled=jnp.where(p1e2 | p2e2, ft, e2.is_tackled))

        return state.replace(
            player_state=state.player_state.replace(player1=p1, player2=p2),
            enemy_state=state.enemy_state.replace(enemy1=e1, enemy2=e2),
        )

    def _scoring_step(self, state: IceHockeyState) -> IceHockeyState:
        puck = state.puck_state
        game = state.game_state

        any_has_puck = (
            state.player_state.player1.has_puck
            | state.player_state.player2.has_puck
            | state.enemy_state.enemy1.has_puck
            | state.enemy_state.enemy2.has_puck
        )
        free = jnp.logical_not(any_has_puck)

        px, py = puck.position[0], puck.position[1]
        in_goal_y = (py >= self.consts.GOAL_TOP_Y) & (py <= self.consts.GOAL_BOTTOM_Y)

        # left goal → player scores, right goal → enemy scores
        player_scored = free & in_goal_y & (px <= self.consts.LEFT_GOAL_X)
        enemy_scored  = free & in_goal_y & (px >= self.consts.RIGHT_GOAL_X)
        goal = player_scored | enemy_scored

        p1_fo   = jnp.array([self.consts.PLAYER_FACEOFF_X, self.consts.FACEOFF_Y_1])
        p2_fo   = jnp.array([self.consts.PLAYER_FACEOFF_X, self.consts.FACEOFF_Y_2])
        e1_fo   = jnp.array([self.consts.ENEMY_FACEOFF_X,  self.consts.FACEOFF_Y_1])
        e2_fo   = jnp.array([self.consts.ENEMY_FACEOFF_X,  self.consts.FACEOFF_Y_2])
        puck_fo = jnp.array([self.consts.FACEOFF_CENTER_X, self.consts.FACEOFF_CENTER_Y])

        def _reset_char(char: CharacterState, faceoff_pos: chex.Array) -> CharacterState:
            return char.replace(
                position=jnp.where(goal, faceoff_pos, char.position),
                has_puck=jnp.where(goal, jnp.zeros_like(char.has_puck), char.has_puck),
            )

        new_player_state = state.player_state.replace(
            player1=_reset_char(state.player_state.player1, p1_fo),
            player2=_reset_char(state.player_state.player2, p2_fo),
        )
        new_enemy_state = state.enemy_state.replace(
            enemy1=_reset_char(state.enemy_state.enemy1, e1_fo),
            enemy2=_reset_char(state.enemy_state.enemy2, e2_fo),
        )

        new_puck = puck.replace(
            position=jnp.where(goal, puck_fo, puck.position),
            velocity=jnp.where(goal, jnp.zeros_like(puck.velocity), puck.velocity),
        )

        new_game = game.replace(
            player_score=game.player_score + player_scored.astype(jnp.int32),
            enemy_score=game.enemy_score  + enemy_scored.astype(jnp.int32),
            goal_scored=goal,
            is_faceoff=jnp.where(goal, jnp.ones_like(game.is_faceoff), game.is_faceoff),
            pause_counter=jnp.where(
                goal, jnp.int32(self.consts.FACE_OFF_FRAMES), game.pause_counter
            ),
        )

        return state.replace(
            player_state=new_player_state,
            enemy_state=new_enemy_state,
            puck_state=new_puck,
            game_state=new_game,
        )

    def render(self, state):
        pass

    def action_space(self):
        pass

    def observation_space(self):
        pass

    def image_space(self):
        pass

    def _get_observation(self, state):
        pass

    def obs_to_flat_array(self, obs):
        pass

    def _get_info(self, state):
        pass

    def _get_reward(self, previous_state, state):
        pass

    def _get_done(self, state):
        pass

class IceHockeyRenderer(JAXGameRenderer):
    pass
