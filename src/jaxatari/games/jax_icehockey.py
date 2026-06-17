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

    FACEOFF_X: float = 78.0          # x the enemy aims at when carrying the puck
    FACEOFF_Y: float = 103.0         # default pursuit target at reset (rink centre)
    PLAYER_GOAL_Y: float = 187.0     # y of the goal the enemy attacks
    MIN_SEPARATION: float = 8.0      # body separation; reused as tackle-contact radius

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
    enemy_target: chex.Array     # float32 [x, y] pursuit target, refreshed every 4 frames

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
    lfsr: chex.Array             # int32, 16-bit Galois LFSR for pursuit-target jitter

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
        #
        # ENEMY MOVEMENT — when you build the initial state, set these two fields:
        #     enemy_state = EnemyState(
        #         enemy1=..., enemy2=..., active_character=jnp.array(0, dtype=jnp.int32),
        #         enemy_target=jnp.array([c.FACEOFF_X, c.FACEOFF_Y], dtype=jnp.float32),
        #     )
        #     ... and on the IceHockeyState:
        #     lfsr=jnp.array(0xACE1, dtype=jnp.int32),   # any non-zero 16-bit seed
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
        #
        # ENEMY MOVEMENT — wire it in like this (the two calls are the only enemy-AI hooks):
        #
        #   1) at the TOP, before moving the characters, get the enemy's action:
        #          enemy_action = self._enemy_policy(state)
        #      then feed it to your character-movement step alongside the player's action.
        #
        #   2) AFTER you have the new puck position + new player/enemy states, refresh the
        #      pursuit target and advance the LFSR:
        #          new_lfsr, new_enemy_state = self._update_enemy_target(
        #              state.counter, state.lfsr, new_puck_state.position,
        #              new_player_state, new_enemy_state,
        #          )
        #      and store new_lfsr / new_enemy_state on the returned IceHockeyState.
        pass

    # ================================================================== #
    # Enemy movement / AI  (the only part added on top of the skeleton)  #
    # ================================================================== #

    @staticmethod
    def _lfsr_step(lfsr: chex.Array) -> chex.Array:
        """

        Cheap deterministic pseudo-randomness used to add a little zigzag jitter to the
        enemy's pursuit target, so the opponent doesn't track the puck on a perfectly
        straight line*
        """
        bit = lfsr & 1
        return jnp.where(bit, (lfsr >> 1) ^ jnp.int32(0xB400), lfsr >> 1).astype(jnp.int32)

    def _update_enemy_target(
        self,
        counter: chex.Array,
        lfsr: chex.Array,
        puck_position: chex.Array,
        player_state: PlayerState,
        enemy_state: EnemyState,
    ) -> Tuple[chex.Array, EnemyState]:
        """Refresh the enemy's pursuit target and advance the noise LFSR.

        Called once per frame from ``step`` *after* the new puck/player/enemy states are
        known. The target only moves every 4th frame and only while the enemy does NOT
        hold the puck (a carrier heads for the goal instead, handled in ``_enemy_policy``).
        When the human player carries the puck, a small signed jitter (per axis, range
        roughly -7..+8) is added so the chase isn't perfectly straight.

        Returns the advanced ``lfsr`` and the ``EnemyState`` with the updated target.
        """
        new_lfsr = self._lfsr_step(lfsr)

        enemy_has_puck  = enemy_state.enemy1.has_puck | enemy_state.enemy2.has_puck
        player_has_puck = player_state.player1.has_puck | player_state.player2.has_puck

        # Two independent 4-bit values -> 0..15, shifted to a signed -7..+8 range.
        noise = jnp.array([
            (new_lfsr & jnp.int32(0xF)).astype(jnp.float32) - jnp.float32(7.0),
            ((new_lfsr >> 4) & jnp.int32(0xF)).astype(jnp.float32) - jnp.float32(7.0),
        ])

        candidate  = jnp.where(player_has_puck, puck_position + noise, puck_position)
        new_target = jnp.where(
            #every four frames
            ((counter % 4) == 0) & ~enemy_has_puck,
            candidate,
            enemy_state.enemy_target,
        )
        return new_lfsr, enemy_state.replace(enemy_target=new_target)

    def _enemy_policy(self, state: IceHockeyState) -> chex.Array:
        """Pick the controlled enemy skater's action for this frame.

        Pure read-only function of the *current* state; produces a single Atari action
        integer that you feed into the shared character-movement step exactly like the
        player's action. Behaviour:

          * carrying the puck   -> drive toward the player's goal; shoot (DOWNFIRE) once
                                    close enough to it,
          * not carrying        -> move toward the pursuit target (the jittered puck
                                    position); tackle (FIRE) when a player is in contact,
          * otherwise           -> 8-directional move toward the target, else NOOP.

        The controlled skater is whichever enemy is currently ``active_character``.
        """
        c   = self.consts
        es  = state.enemy_state
        ai  = es.active_character
        #0 enemy 1 1 enemy 2
        #jnp.where( BEDINGUNG , WERT_WENN_JA , WERT_WENN_NEIN )
        pos = jnp.where(ai == 0, es.enemy1.position, es.enemy2.position)
        has = es.enemy1.has_puck | es.enemy2.has_puck

        # With puck -> aim at the player's goal; otherwise chase the pursuit target.
        tgt = jnp.where(
            has,
            jnp.array([c.FACEOFF_X, jnp.float32(c.PLAYER_GOAL_Y)], dtype=jnp.float32),
            es.enemy_target,
        )
        dx = tgt[0] - pos[0]; dy = tgt[1] - pos[1]
        r = dx >  2.0; l = dx < -2.0
        d = dy >  2.0; u = dy < -2.0

        near_goal    = jnp.abs(pos[1] - jnp.float32(c.PLAYER_GOAL_Y)) < 50.0
        should_shoot = has & near_goal


        # tackle mechanic uncommented as we still need do find out a reasonable mechanic
        # ps = state.player_state
        # thresh2  = jnp.float32(c.MIN_SEPARATION ** 2)   # contact radius (squared)
        # p1_close = jnp.sum((pos - ps.player1.position) ** 2) < thresh2
        # p2_close = jnp.sum((pos - ps.player2.position) ** 2) < thresh2
        # should_tackle = ~has & (p1_close | p2_close)

        return jnp.where(should_shoot,  jnp.int32(Action.DOWNFIRE),
               jnp.where(should_tackle, jnp.int32(Action.FIRE),
               jnp.where(r & d,         jnp.int32(Action.DOWNRIGHT),
               jnp.where(l & d,         jnp.int32(Action.DOWNLEFT),
               jnp.where(r & u,         jnp.int32(Action.UPRIGHT),
               jnp.where(l & u,         jnp.int32(Action.UPLEFT),
               jnp.where(r,             jnp.int32(Action.RIGHT),
               jnp.where(l,             jnp.int32(Action.LEFT),
               jnp.where(d,             jnp.int32(Action.DOWN),
               jnp.where(u,             jnp.int32(Action.UP),
                                         jnp.int32(Action.NOOP)))))))))))

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