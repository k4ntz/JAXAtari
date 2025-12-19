from jax._src.pjit import JitWrapped
import os
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.lax
import jax.numpy as jnp
import chex

import jaxatari.spaces as spaces
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

def _get_default_asset_config() -> tuple:
    """
    Returns the default declarative asset manifest for Adventure.
    Kept immutable (tuple of dicts) to fit NamedTuple defaults.
    """
    return (
        #all rooms in order
        {'name': 'bg_yellow_castle', 'type': 'background', 'file': 'Room_YellowCastle.npy'},
        {'name': 'bg_yellow', 'type': 'background', 'file': 'Room_Yellow.npy'},
        {'name': 'bg_green', 'type': 'background', 'file': 'Room_Green.npy'},
        {'name': 'bg_purple', 'type': 'background', 'file': 'Room_Purple.npy'},
        {'name': 'bg_pink', 'type': 'background', 'file': 'Room_Pink.npy'},
        {'name': 'bg_green_yellow', 'type': 'background', 'file': 'Room_Green_Yellow.npy'},
        {'name': 'bg_maze_1', 'type': 'background', 'file': 'Room_Maze_1.npy'},
        {'name': 'bg_maze_2', 'type': 'background', 'file': 'Room_Maze_2.npy'},
        {'name': 'bg_maze_3', 'type': 'background', 'file': 'Room_Maze_3.npy'},
        {'name': 'bg_maze_4', 'type': 'background', 'file': 'Room_Maze_4.npy'},
        {'name': 'bg_maze_5', 'type': 'background', 'file': 'Room_Maze_5.npy'},
        {'name': 'bg_black_castle', 'type': 'background', 'file': 'Room_Black_Castle.npy'},
        {'name': 'bg_ping_corridor', 'type': 'background', 'file': 'Room_Pink_Corridor.npy'},
        {'name': 'bg_magenta', 'type': 'background', 'file': 'Room_Magenta.npy'},
        #all player colors in order
        {'name': 'player_yellow', 'type': 'single', 'file': 'Player_Yellow.npy'},
        {'name': 'player_green', 'type': 'single', 'file': 'Player_Green.npy'},
        {'name': 'player_purple', 'type': 'single', 'file': 'Player_Purple.npy'},
        {'name': 'player_pink', 'type': 'single', 'file': 'Player_Pink.npy'},
        {'name': 'player_green_yellow', 'type': 'single', 'file': 'Player_Green_Yellow.npy'},
        {'name': 'player_blue', 'type': 'single', 'file': 'Player_Blue.npy'},
        {'name': 'player_black', 'type': 'single', 'file': 'Player_Black.npy'},
        {'name': 'player_magenta', 'type': 'single', 'file': 'Player_Magenta.npy'},

        {'name': 'key_yellow', 'type': 'single', 'file': 'Key_yellow.npy'}
    )


class AdventureConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 250
    
    # sset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()


# immutable state container

class AdventureState(NamedTuple):
    player_x: chex.Array
    player_y:chex.Array
    key_yellow_x:chex.Array
    key_yellow_y:chex.Array
    step_counter: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class AdventureObservation(NamedTuple):
    player: EntityPosition
    key_yellow: EntityPosition


class AdventureInfo(NamedTuple):
    time: jnp.ndarray


class JaxPong(JaxEnvironment[AdventureState, AdventureObservation, AdventureInfo, AdventureConstants]):
    def __init__(self, consts: AdventureConstants = None):
        consts = consts or AdventureConstants()
        super().__init__(consts)
        self.renderer = AdventureRenderer(self.consts)
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.RIGHT,
            Action.LEFT,
            Action.UP,
            Action.DOWN,
        ]

    def _player_step(self, state: AdventureState, action: chex.Array) -> AdventureState:
        left = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)
        right = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        up = jnp.logical_or(action == Action.UP, action == Action.UPFIRE)
        down = jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE)
        new_player_x = state.player_x
        new_player_x = jax.lax.cond(
            left,
            lambda x: x-1,
            lambda x: x,
            operand = new_player_x,
        )
        new_player_x = jax.lax.cond(
            right,
            lambda x: x+1,
            lambda x: x,
            operand = new_player_x,
        )

        new_player_y = state.player_y
        new_player_y = jax.lax.cond(
            down,
            lambda y: y+1,
            lambda y: y,
            operand = new_player_y,
        )
        new_player_y = jax.lax.cond(
            up,
            lambda y: y-1,
            lambda y: y,
            operand = new_player_y,
        )

        return AdventureState(
            player_x = new_player_x,
            player_y = new_player_y,
            key_yellow_x = state.key_yellow_x,
            key_yellow_y = state.key_yellow_y,
            step_counter = state.step_counter
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        previous_state = state
        state = self._player_step(state, action)

    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AdventureObservation, AdventureState]:

        state = AdventureState(
            player_x=jnp.array(78).astype(jnp.int32),
            player_y=jnp.array(174).astype(jnp.int32),
            key_yellow_x=jnp.array(31).astype(jnp.int32),
            key_yellow_y=jnp.array(110).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        # Split step key from state and keep a new key for the next state
        previous_state = state
        # Make per-step key available to helpers that may read state.key
        state = AdventureState(
            player_x = state.player_x,
            player_y=state.player_y,
            key_yellow_x = state.key_yellow_x,
            key_yellow_y = state.key_yellow_y,
            step_counter=state.step_counter,
        )
        state = self._player_step(state, action)


        done = self._get_done(state)
        env_reward = self._get_reward(previous_state, state)
        info = self._get_info(state)
        observation = self._get_observation(state)

        return observation, state, env_reward, done, info


    def render(self, state: AdventureState) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: AdventureState):
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=4,
            height=8
        )
        key_yellow = EntityPosition(
            x=state.key_yellow_x,
            y=state.key_yellow_y,
            width=10,
            height=4
        )

        return AdventureObservation(
            player=player,
            key_yellow=key_yellow
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: AdventureObservation) -> jnp.ndarray:
           return jnp.concatenate([
               obs.player.x.flatten(),
               obs.player.y.flatten(),
               obs.player.height.flatten(),
               obs.player.width.flatten()
            ]
           )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(6)

    def observation_space(self) -> spaces:
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
            }),
            "key_yellow": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=250, shape=(), dtype=jnp.int32),
            }),
        })

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(250, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: AdventureState, ) -> AdventureInfo:
        return AdventureInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AdventureState, state: AdventureState):
        return 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: AdventureState) -> bool:
        return 0

class AdventureRenderer(JAXGameRenderer):
    def __init__(self, consts: AdventureConstants = None):
        super().__init__(consts)
        self.consts = consts or AdventureConstants()
        self.config = render_utils.RendererConfig(
            game_dimensions=(250, 160),
            channels=3,
            #downscale=(84, 84)
        )
        self.jr = render_utils.JaxRenderingUtils(self.config)

        # 1. Start from (possibly modded) asset config provided via constants
        final_asset_config = list(self.consts.ASSET_CONFIG)

        # 4. Bake assets once
        sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/adventure"
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        #set bg color here
        raster = self.jr.create_object_raster(self.BACKGROUND)

        #set player color here
        player_mask = self.SHAPE_MASKS["player_magenta"]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        key_yellow_mask = self.SHAPE_MASKS["key_yellow"]
        raster = self.jr.render_at(raster, state.key_yellow_x, state.key_yellow_y, key_yellow_mask)
        return self.jr.render_from_palette(raster, self.PALETTE)
