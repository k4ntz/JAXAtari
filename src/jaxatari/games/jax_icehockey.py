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
    # This structure holds all static, non-learnable parameters of the game, 
    # such as screen dimensions, player speed, or colors.
    MAX_SHOOTING_ANGLE: int
    PLAYER_SPEED: float
    PUCK_SPEED: float
    MIN_VERTICAL_DISTANCE: float
    MAX_PUCK_SPEED: float
    PUCK_SPEED_DECAY: float
    TIME_LIMIT: int
    MIN_SHOOTING_INTERVAL: int # = shooting animation
    FACE_OFF_FRAMES: int # number of frames during which the game is in face-off mode, where players are reset to the center and cannot move
    MAX_PUSH_DISTANCE: float # front player can onle be pushed until this point
    FRAMES_TACKLED: int = 60

@struct.dataclass
class GameState:
    pause_counter: chex.Array  # delay between restart of game
    player_score: chex.Array  # The score line within the current set (goes up in increments of 1, instead of traditional tennis counting)
    enemy_score: chex.Array
    is_finished: chex.Array  # True if the game is finished (Player or enemy has won the game)  
    remaining_time: chex.Array
    is_faceoff: chex.Array # during the initial frames, the game is freezed
    goal_scored: chex.Array # True for the frame where a goal is scored, the game is freezed
    is_finished: chex.Array # True if the game is finished (Player or enemy has won the game)

@struct.dataclass
class CharacterState:
    is_tackled: chex.Array
    position: chex.Array # (x,y) position of player 1
    orientation: chex.Array # 0 left, 1 right, player 1
    has_puck: chex.Array # True if the player has the puck
    shooting_cooldown: chex.Array


@struct.dataclass
class PuckState:
    position: chex.Array # (x,y) position of the puck
    velocity: chex.Array # (x,y) velocity of the puck
    direction: chex.Array # 32 possible directions, between 45 degrees left and 45 degrees right of the stick orientation when shooting
    position_stick: chex.Array # 32 possible directions (the stick moving from left to right when attached to the stick)


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
    active_character: chex.Array # 0 for enemy 1, 1 for enemy 2

@struct.dataclass
class PlayerState:
    player1: CharacterState
    player2: CharacterState
    active_character: chex.Array # 0 for player 1, 1 for player 2

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
    # This is used for carrying auxiliary diagnostic information that is not used for 
    # training but might otherwise be relevant, such as the current level.
    pass

@struct.dataclass
class IceHockeyObservation:
    # This structure holds the object-centric data exposed to the RL agent. 
    # Its specific content is game dependent and should contain everything 
    # the environment developer deems necessary knowledge to be able to play the game
    # (position of player, position of enemies, etc).
    pass

class JaxIceHockey(JaxEnvironment):
    # Each game is a class that inherits from a base JaxEnvironment class, 
    # implementing the core gameplay logic.
    
    def __init__(self, consts: Optional[IceHockeyConstants] = None):
        # The constructor is responsible for all one-time setup of the environment. This logic runs once on the CPU when the class is
        # instantiated and is **not** JIT-compiled. Its primary role is to set up the game’s static constants and instantiate the game-specific renderer.
        # This is also a good place to pre-process data that can be used during execution to increase performance, for example pre-computing
        # level architecture instead of doing it on the fly.
        #
        # Parameters
        # - consts: An instance of the environment’s specificConstants NamedTuple. IfNoneis provided, the constructor should
        #   initialize a default version.
        pass

    def reset(self, key):
        # Purpose This function resets the environment to its initial state, which is necessary at the beginning of every new episode. It must
        # be JIT-compatible.
        #
        # Parameters
        # - key: Ajrandom.PRNGKeyfor environments that have stochastic starting conditions (though many Atari games are determin-
        #   istic).
        #
        # Returns A tuple of(EnvObs, EnvState)containing the initial observation for the agent and the complete initial state of the
        # environment.
        pass

    def step(self, state, action):
        # Purpose This is the main part of the environment. It takes a single action and advances the game logic by one frame. This function
        # must be fully JIT-compatible and is where the core game logic resides. As described in Section 4, this function should ideally be
        # implemented as an "orchestrator" that only calls internal, JIT-compatible helper functions (e.g., _player_step, _ball_step).
        #
        # Parameters
        # - state: The complete EnvState object from the *previous* step.
        # - action: The action selected by the agent (e.g., an integer, for mapping see the JAXAtariAction class in environment.py).
        #
        # Returns A tuple of (EnvObs, EnvState, float, bool, EnvInfo) containing:
        # - The new observation for the agent.
        # - The complete new EnvState object.
        # - The scalar reward obtained during this step.
        # - A boolean done flag, which is True if the new state is terminal.
        # - An EnvInfo object for auxiliary data.
        pass

    def _resolve_active_character(
        self,
        char1: CharacterState,
        char2: CharacterState,
        puck_position: chex.Array,
        current_active: chex.Array,
    ) -> chex.Array:
        """Resolve which of two characters is closest to the puck.

        Control in Ice Hockey goes to whichever of a team's two skaters is closest to
        the puck, so ``_player_step`` and ``_enemy_step`` both call this on their own
        pair.

        Args:
            char1: First character (corresponds to index 0).
            char2: Second character (corresponds to index 1).
            puck_position: ``(x, y)`` position of the puck.
            current_active: Index (0 or 1) returned on an exact distance tie, which
                avoids the result flickering between equidistant characters.

        Returns:
            An ``int32`` scalar: 0 if ``char1`` is closer, 1 if ``char2`` is.
        """
        # Squared distance from each character to the puck; sqrt is unnecessary for ordering.
        dist1_sq = jnp.sum((char1.position - puck_position) ** 2)
        dist2_sq = jnp.sum((char2.position - puck_position) ** 2)

        # Closest character wins; on an exact tie keep whoever is currently active.
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
        """Apply one frame of joystick *input* movement to a single character.

        This is the per-character movement primitive shared by the human player and the
        computer opponent: each chooses an action through its own policy, but the action
        is applied identically here. Directions are absolute screen directions.

        Only the active skater of a team should receive a real action; the inactive
        teammate never moves from input (the caller passes NOOP or simply skips it).
        A tackled character ignores input entirely (it is frozen for the tackle period).

        Args:
            character: The character to move.
            action: The chosen Atari action integer.
            bounds: ``(x_min, x_max, y_min, y_max)`` provisional wall/zone clamp (see above).
            velocity: Per-axis movement distance for this frame (e.g. ``PLAYER_SPEED``).

        Returns:
            The updated ``CharacterState`` (position + orientation; other fields kept).
        """
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

        # A tackled character is frozen: ignore all input movement this frame.
        movable = jnp.logical_not(character.is_tackled)
        dx = jnp.where(movable & right, velocity, jnp.where(movable & left, -velocity, 0.0))
        # Screen y grows downward, so DOWN increases y and UP decreases it.
        # NOTE: diagonals move by `velocity` on each axis, i.e. ~1.41x faster than a
        # straight move.    
        dy = jnp.where(movable & down, velocity, jnp.where(movable & up, -velocity, 0.0))

        new_x = jnp.clip(character.position[0] + dx, bounds[0], bounds[1])
        new_y = jnp.clip(character.position[1] + dy, bounds[2], bounds[3])
        new_position = jnp.array([new_x, new_y])

        # Orientation: 0 = facing left, 1 = facing right.
        # input keeps the current facing; a tackled character keeps it too (frozen).
        new_orientation = jnp.where(
            movable & right, 1, jnp.where(movable & left, 0, character.orientation)
        )

        return character.replace(position=new_position, orientation=new_orientation)

    # ------------------------------------------------------------------ #
    # Phase 1 — intended input movement (uniform over a team's two skaters)
    # ------------------------------------------------------------------ #
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
        """Apply one team's chosen action as phase-1 intended movement.

        The reframed phase 1: instead of "only the active skater moves", every character
        is handled uniformly by the same ``_apply_action`` — the active skater receives
        the real action and the teammate receives ``NOOP`` (a zero intended delta). The
        active/passive split therefore collapses to "what action does this character get
        this frame", and the inactive teammate simply gets a no-op move.

        This is shared by both teams: the player's action comes from the agent, the
        computer's from its (future) policy, but routing + application are identical.

        Returns the two characters with their *provisional* (wall/zone-clamped) intended
        positions; the authoritative position is decided later by ``_resolve_interactions``.
        """
        action1 = jnp.where(active == 0, action, Action.NOOP)
        action2 = jnp.where(active == 1, action, Action.NOOP)
        return (
            self._apply_action(char1, action1, bounds1, velocity),
            self._apply_action(char2, action2, bounds2, velocity),
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — interaction resolution (pure geometry, single fixed-order pass)
    # ------------------------------------------------------------------ #
    def _separate_opponents(
        self,
        pos_a: chex.Array,
        pos_b: chex.Array,
        min_separation: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Resolve a cross-team (opponent) overlap: the confirmed "both shift".

        If the two characters are closer than ``min_separation``, push BOTH apart along
        their centre-to-centre direction, each by half the penetration, so they end up
        exactly ``min_separation`` apart. The centre-to-centre normal is what makes the
        displacement diagonal. No-op when already separated.

        Pure geometry of the *post-move* positions only — it needs no pre-move state.

        Efficiency: the overlap *test* is done on squared distance (no sqrt). The push
        itself needs the true distance for the unit normal and the linear penetration,
        so a root is unavoidable here — but we fold sqrt + divide into a single
        reciprocal-sqrt: 1/dist == rsqrt(dist**2), giving
        ``offset = 0.5 * delta * (min_separation / dist - 1)``.

        NOTE: ``min_separation`` (derived from body sizes) is not yet finalised; passed
        in. Whether a tackled/downed character is still pushable is also unverified.
        """
        delta = pos_a - pos_b
        dist_sq = jnp.sum(delta ** 2)
        overlapping = dist_sq < min_separation ** 2

        # True distance is only needed via its reciprocal: 1/dist == rsqrt(dist**2).
        coincident = dist_sq <= 0.0
        inv_dist = jnp.where(coincident, 0.0, jax.lax.rsqrt(dist_sq))
        # offset = 0.5 * (min_sep - dist) * (delta / dist) = 0.5 * delta * (min_sep/dist - 1)
        offset = 0.5 * delta * (min_separation * inv_dist - 1.0)
        # Coincident centres give no direction: default to separating along +x / -x.
        offset = jnp.where(coincident, jnp.array([min_separation * 0.5, 0.0]), offset)
        offset = jnp.where(overlapping, offset, jnp.array([0.0, 0.0]))
        return pos_a + offset, pos_b - offset

    def _enforce_min_vertical(
        self,
        active_pos: chex.Array,
        passive_pos: chex.Array,
        min_vertical_distance: chex.Array,
    ) -> chex.Array:
        """Resolve a same-team overlap: vertical-only push, mover holds / passive yields.

        Distinct from the opponent mechanic. The active (moving) skater keeps its
        position; the passive teammate is displaced along y only so the pair are at
        least ``min_vertical_distance`` apart (e.g. the goalie skating forward pushes
        his teammate straight up). The passive's x is untouched.

        Returns the passive teammate's new position. No-op when the gap is already large
        enough.

        NOTE: ``MAX_PUSH_DISTANCE`` ("front player can only be pushed until this point")
        is NOT modelled here — that coupling (the active skater being blocked once the
        passive hits its limit) is unverified. For now the passive is clamped to its
        zone later by ``_resolve_interactions``.
        """
        dy = passive_pos[1] - active_pos[1]
        too_close = jnp.abs(dy) < min_vertical_distance
        # Preserve which side the passive is on; if exactly level, default to below.
        side = jnp.where(dy != 0.0, jnp.sign(dy), 1.0)
        new_y = jnp.where(
            too_close, active_pos[1] + side * min_vertical_distance, passive_pos[1]
        )
        return jnp.array([passive_pos[0], new_y])

    def _clamp_to_bounds(self, pos: chex.Array, bounds: chex.Array) -> chex.Array:
        """Authoritative wall/zone clamp: ``(x_min, x_max, y_min, y_max)``."""
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
        """Phase 2: resolve all interactions on the four post-phase-1 characters.

        A single, fixed-order, fully branchless pass (deliberately NOT an iterative
        constraint solver — see the efficiency rationale: with 4 characters and shallow
        per-frame overlaps this is faithful and cheap):

          1. Opponent (cross-team) separations for the 4 player×enemy pairs.
          2. Same-team vertical pushes (active holds, passive yields).
          3. Authoritative wall/zone clamp on all four (covers pushes into walls and the
             passive teammate, which never passes through phase 1).

        The sub-step ORDER is a behavioural choice to verify against the game; in rare
        triple-contact frames a later step can nudge an earlier constraint sub-pixel,
        which the design guide tolerates.
        """
        p1, p2 = player1.position, player2.position
        e1, e2 = enemy1.position, enemy2.position

        # 1) Opponent collisions — both shift along centre-to-centre.
        p1, e1 = self._separate_opponents(p1, e1, min_separation)
        p1, e2 = self._separate_opponents(p1, e2, min_separation)
        p2, e1 = self._separate_opponents(p2, e1, min_separation)
        p2, e2 = self._separate_opponents(p2, e2, min_separation)

        # 2) Same-team vertical push — the active skater holds, the teammate yields.
        p2 = jnp.where(player_active == 0, self._enforce_min_vertical(p1, p2, min_vertical_distance), p2)
        p1 = jnp.where(player_active == 1, self._enforce_min_vertical(p2, p1, min_vertical_distance), p1)
        e2 = jnp.where(enemy_active == 0, self._enforce_min_vertical(e1, e2, min_vertical_distance), e2)
        e1 = jnp.where(enemy_active == 1, self._enforce_min_vertical(e2, e1, min_vertical_distance), e1)

        # 3) Authoritative clamp.
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

    # ------------------------------------------------------------------ #
    # Orchestrator — runs phase 1 then phase 2 for all four characters
    # ------------------------------------------------------------------ #
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
        """Advance all four characters one frame: active resolution -> phase 1 -> phase 2.

        This is the character-movement orchestrator shared by both teams. The full
        ``step`` will call it with ``self.consts`` values for the tunables/zones and with
        ``enemy_action`` produced by the (future) ``_enemy_policy``; here those are
        parameters so the wiring runs and is testable before the constants/zones and the
        opponent policy exist.

        Steps:
          1. Resolve each team's active (controlled) skater = closest to the puck.
          2. Phase 1: apply each team's action as intended input movement (uniformly via
             ``_apply_team_inputs`` — active skater gets the action, teammate gets NOOP).
          3. Phase 2: resolve interactions across all four post-move characters
             (opponent separation, teammate vertical push, authoritative clamp).

        Returns the updated ``PlayerState`` and ``EnemyState`` (positions/orientation
        from movement, plus the resolved ``active_character`` for each team).
        """
        # 1) Active-skater resolution (per team, against the shared puck).
        player_active = self._resolve_active_character(
            player_state.player1, player_state.player2,
            puck_position, player_state.active_character,
        )
        enemy_active = self._resolve_active_character(
            enemy_state.enemy1, enemy_state.enemy2,
            puck_position, enemy_state.active_character,
        )

        # 2) Phase 1 — intended input movement, uniform over each team's two skaters.
        p1, p2 = self._apply_team_inputs(
            player_state.player1, player_state.player2,
            player_active, player_action, bounds_p1, bounds_p2, velocity,
        )
        e1, e2 = self._apply_team_inputs(
            enemy_state.enemy1, enemy_state.enemy2,
            enemy_active, enemy_action, bounds_e1, bounds_e2, velocity,
        )

        # 3) Phase 2 — resolve interactions across all four post-move characters.
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

    def render(self, state):
        # Purpose This function generates a single RGB image (as a JAX array) representing the current game state. It is used for visualization
        # and for agents that learn from pixels. This method should contain no game logic; it should only delegate the rendering task to the
        # environment’s dedicated JAXGameRenderer class.
        #
        # Parameters
        # - state: The EnvState object to be rendered.
        #
        # Returns A jnp.ndarray representing the RGB image.
        pass

    def action_space(self):
        # Purpose A non-JIT helper function that defines the set of all valid actions an agent can take.
        #
        # Returns A Space object (e.g., spaces.Discrete) that describes the action space.
        pass

    def observation_space(self):
        # Purpose A non-JIT helper function that defines the structure, data types, and bounds of the object-centric EnvObs.
        #
        # Returns A Space object (typically spaces.Dict) that describes the observation space.
        pass

    def image_space(self):
        # Purpose A non-JIT helper function that defines the structure, data types, and bounds of the image returned by render().
        #
        # Returns A Space object (typically spaces.Box) that describes the image space.
        pass

    def _get_observation(self, state):
        # Purpose An internal JIT-compatible helper function, usually called bystep, that converts the full, internalEnvStateinto the
        # public-facing EnvObs. This is used to filter out internal state variables that are not relevant to the agent.
        #
        # Parameters
        # - state: The EnvState object of the current step
        #
        # Returns The corresponding EnvObs object.
        pass

    def obs_to_flat_array(self, obs):
        # Purpose A JIT-compatible utility function that converts the structured, object-centricEnvObs(which is often aNamedTupleor
        # Dict) into a single, flat 1D jnp.ndarray. This is required for agents that cannot process structured observations.
        #
        # Parameters
        # - obs: The EnvObs object to flatten.
        #
        # Returns A 1D jnp.ndarray.
        pass

    def _get_info(self, state):
        # Purpose An internal JIT-compatible helper function, called bystep, that extracts auxiliary information from thestate. This data
        # is not meant for the agent but is useful for logging or debugging (e.g., current lives, score, time).
        #
        # Parameters
        # - state: The new EnvState.
        #
        # Returns The EnvInfo object.
        pass

    def _get_reward(self, previous_state, state):
        # Purpose An internal JIT-compatible helper function, called bystep, that calculates the scalar reward for the transition *from*
        # previous_state *to* state.
        #
        # Parameters
        # - previous_state: The EnvState from the prior step.
        # - state: The new EnvState after the action was taken.
        #
        # Returns A float reward value.
        pass

    def _get_done(self, state):
        # Purpose An internal JIT-compatible helper function, called bystep, that determines if the newstateis a terminal "game over"
        # state.
        #
        # Parameters
        # - state: The new EnvState to check.
        #
        # Returns A bool which is True if the state is terminal, False otherwise.
        pass

class IceHockeyRenderer(JAXGameRenderer):
    pass