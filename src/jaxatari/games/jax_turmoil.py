import os
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr


class TurmoilConstants(NamedTuple):
    PLAYER_SPEED = 10
    PLAYER_START_X = 80
    PLAYER_START_Y = 100

    # sizes
    PLAYER_SIZE = (8, 11) # (width, height)
    BULLET_SIZE = (8, 3)

    # directions
    FACE_LEFT = -1
    FACE_RIGHT = 1

    # boundaries
    MIN_BOUND = (0, 0) # (min x, min y)
    MAX_BOUND = (134, 135)

class SpawnState(NamedTuple):
    pass

# Game state container
class TurmoilState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array
    lives: chex.Array
    score: chex.Array
    
    bullet: chex.Array # x, y, active, direction

    step_counter: chex.Array
    rng_key: chex.PRNGKey


class PlayerEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    o: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray


class TurmoilObservation(NamedTuple):
    player: PlayerEntity
    lives: jnp.array

class TurmoilInfo(NamedTuple):
    step_counter: jnp.ndarray  # Current step count
    all_rewards: jnp.ndarray  # All rewards for the current step



# RENDER CONSTANTS
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    player = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/1.npy"))
    bullet = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/bullet/1.npy"))
    player_shrink_1 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/1.npy"))
    player_shrink_2 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/2.npy"))
    player_shrink_3 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/3.npy"))
    player_shrink_4 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/4.npy"))
    player_shrink_5 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/5.npy"))
    player_shrink_6 = jr.loadFrame(os.path.join(MODULE_DIR, "sprites/turmoil/player/shrink/6.npy"))
    

    player = [player]
    bullet = [bullet]

    # Pad player shrink sprites to match each other
    player_shrink_sprites, player_shrink_offsets = jr.pad_to_match([
        player_shrink_1,
        player_shrink_2,
        player_shrink_3,
        player_shrink_4,
        player_shrink_5,
        player_shrink_6
    ])
    player_shrink_offsets = jnp.array(player_shrink_offsets)

    # Player sprites
    PLAYER_SHIP = jnp.repeat(player[0][None], 1, axis=0)

    # bullet sprites
    BULLET = jnp.repeat(bullet[0][None], 1, axis=0)

    # player shrink sprites
    PLAYER_SHRINK = jnp.concatenate(
        [
            jnp.repeat(player_shrink_sprites[0][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[1][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[2][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[3][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[4][None], 4, axis=0),
            jnp.repeat(player_shrink_sprites[5][None], 4, axis=0),
        ]
    )

    DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/turmoil/digits/{}.npy"))


    return (
        PLAYER_SHIP,
        BULLET,
        PLAYER_SHRINK,
        DIGITS,
        player_shrink_offsets
    )

# Load sprites once at module level
(
    PLAYER_SHIP,
    BULLET,
    PLAYER_SHRINK,
    DIGITS,
    PLAYER_SHRINK_OFFSETS
) = load_sprites()


class JaxTurmoil(JaxEnvironment[TurmoilState, TurmoilObservation, TurmoilInfo, TurmoilConstants]):
    def __init__(self, consts: TurmoilConstants = None, reward_funcs: list[callable] = None):
        consts = consts or TurmoilConstants()
        super().__init__(consts)
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
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
            Action.DOWNLEFTFIRE
        ]
        self.frame_stack_size = 4
        self.obs_size = 6 + 12 * 5 + 12 * 5 + 4 * 5 + 4 * 5 + 5 + 5 + 4
        self.renderer = TurmoilRenderer(self.consts)

   
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: TurmoilState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)
    
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: TurmoilState) -> TurmoilObservation:
        # Create player (already scalar, no need for vectorization)
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            o=state.player_direction,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
            active=jnp.array(1),  # Player is always active
        )

        return TurmoilObservation(
            player=player,
            lives=state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: TurmoilState, all_rewards: jnp.ndarray) -> TurmoilInfo:
        return TurmoilInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(self, previous_state: TurmoilState, state: TurmoilState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: TurmoilState, state: TurmoilState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: TurmoilState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[TurmoilObservation, TurmoilState]:
        """Initialize game state"""
        reset_state = TurmoilState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_direction=jnp.array(0),
            lives=jnp.array(5), # TODO check this value
            score=jnp.array(0),

            bullet=jnp.zeros(4),

            step_counter=jnp.array(0),
            rng_key=key,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state
    
    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state: TurmoilState,
        action: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        '''
        implement all the possible movement directions for the player, the mapping is:
        anything with left in it, add -2 to the x position
        anything with right in it, add 2 to the x position
        anything with up in it, add -2 to the y position
        anything with down in it, add 2 to the y position
        '''
        up = jnp.any(
            jnp.array(
                [
                    action == Action.UP,
                    action == Action.UPRIGHT,
                    action == Action.UPLEFT,
                    action == Action.UPFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                ]
            )
        )
        down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNRIGHT,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        left = jnp.any(
            jnp.array(
                [
                    action == Action.LEFT,
                    action == Action.UPLEFT,
                    action == Action.DOWNLEFT,
                    action == Action.LEFTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNLEFTFIRE,
                ]
            )
        )
        right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                    action == Action.RIGHTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        player_x = jnp.where(
            right,
            state.player_x + 2,
            jnp.where(
                left,
                state.player_x - 2,
                state.player_x
            )
        )

        player_y = jnp.where(
            down,
            state.player_y + 2,
            jnp.where(
                up,
                state.player_y - 2,
                state.player_y
            )
        )

        player_direction = jnp.where(
            right,
            1,
            jnp.where(
                left,
                -1,
                state.player_direction
            )
        )

        # keep player in boundaries
        player_x = jnp.where(
            player_x < self.consts.MIN_BOUND[0],
            self.consts.MIN_BOUND[0],
            jnp.where(
                player_x > self.consts.MAX_BOUND[0],
                self.consts.MAX_BOUND[0],
                player_x,
            ),
        )

        player_y = jnp.where(
            player_y < self.consts.MIN_BOUND[1],
            self.consts.MIN_BOUND[1], 
            jnp.where(
                player_y > self.consts.MAX_BOUND[1],
                self.consts.MAX_BOUND[1],
                player_y
            ),
        )

        return player_x, player_y, player_direction

    
    @partial(jax.jit, static_argnums=(0,))
    def bullet_step(
        self,
        state: TurmoilState,
        action: chex.Array,
    ) -> chex.Array:
        fire = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.DOWNFIRE,
                    action == Action.DOWNRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                    action == Action.RIGHTFIRE,
                    action == Action.LEFTFIRE,
                    action == Action.UPFIRE,
                ]
            )
        )

        # if player fired and there is no active bullet, create on in player_direction
        new_bullet = jnp.where(
            jnp.logical_and(fire, jnp.logical_not(state.bullet[2])),
            jnp.where(
                state.player_direction == -1,
                jnp.array([
                    state.player_x - self.consts.BULLET_SIZE[0], # x, y, active, direction
                    state.player_y + self.consts.PLAYER_SIZE[1] / 2,
                    1,
                    -1
                ]),
                jnp.array([
                    state.player_x + self.consts.PLAYER_SIZE[0],
                    state.player_y + self.consts.PLAYER_SIZE[1] / 2,
                    1,
                    1
                ]),
            ),
            state.bullet,
        )
        
        # check if the new positions are in bounds, else destroy
        new_bullet = jnp.where(
            new_bullet[0] < self.consts.MIN_BOUND[0] - 2,
            jnp.array([0, 0, 0, 0]),
            jnp.where(
                new_bullet[0] > self.consts.MAX_BOUND[0] + self.consts.BULLET_SIZE[0] + 2,
                jnp.array([0, 0, 0, 0]), 
                new_bullet
            ),
        )

        # if a bullet, we move the bullet further
        new_bullet = jnp.where(
            state.bullet[2],
            jnp.array([
                new_bullet[0] + new_bullet[3] * 3, # bullet speed
                new_bullet[1],
                new_bullet[2],
                new_bullet[3]
            ]),
            new_bullet,
        )

        return new_bullet


    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, state: TurmoilState, action: chex.Array
    ) -> Tuple[TurmoilObservation, TurmoilState, float, bool, TurmoilInfo]:
        
        previous_state = state
        _, reset_state = self.reset()
    
        def normal_game_step() :
            # player movement
            new_player_x, new_player_y, new_player_direction = self.player_step(
                state,
                action
            )

            # bullet
            new_bullet = self.bullet_step(
                state,
                action
            )

            return state._replace(
                player_x=new_player_x,
                player_y=new_player_y,
                player_direction=new_player_direction,
                bullet=new_bullet
            )
        
        return_state = normal_game_step()
    
        observation = self._get_observation(return_state)
        done = self._get_done(return_state)
        env_reward = self._get_env_reward(previous_state, return_state)
        all_rewards = self._get_all_rewards(previous_state, return_state)
        info = self._get_info(return_state, all_rewards)

        return observation, return_state, env_reward, done, info

class TurmoilRenderer(JAXGameRenderer):
    def __init__(self, consts: TurmoilConstants = None):
        super().__init__()
        self.consts = consts or TurmoilConstants()
        self.player_shrink_offsets = len(PLAYER_SHRINK_OFFSETS)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=160, height=210)

        frame_pl_ship = jr.get_sprite_frame(PLAYER_SHIP, 0)
        raster = jr.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_pl_ship,
            flip_horizontal = state.player_direction == self.consts.FACE_LEFT,
        )

        frame_bullet = jr.get_sprite_frame(BULLET, 0)
        raster = jax.lax.cond(
            state.bullet[2],
            lambda r: jr.render_at(
                r,
                state.bullet[0],
                state.bullet[1],
                frame_bullet
            ),
            lambda r: r,
            raster
        )

        return raster