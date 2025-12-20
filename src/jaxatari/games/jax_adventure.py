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
        {'name': 'room_number', 'type': 'group', 'files': ['Room_1.npy', 'Room_2.npy', 'Room_3.npy', 'Room_4.npy', 'Room_5.npy', 'Room_6.npy', 'Room_7.npy', 'Room_8.npy', 'Room_9.npy', 'Room_10.npy', 'Room_11.npy', 'Room_12.npy', 'Room_13.npy', 'Room_14.npy']},
        {'name': 'bg', 'type': 'background', 'file': 'Room_1.npy'},

        #all player colors in order
        {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy", "Player_Green.npy", "Player_Purple.npy", "Player_Pink.npy", "Player_Green_yellow.npy", "Player_Blue.npy", "Player_Black.npy", "Player_Magenta.npy"]},

        #dragons
        {'name': 'dragon_yellow_neutral', 'type': 'single', 'file': 'Dragon_yellow_neutral.npy'},
        {'name': 'dragon_green-neutral', 'type': 'single', 'file': 'Dragon_green_neutral.npy'},
        #ToDo remaining dragon animations

        #keys
        {'name': 'key_yellow', 'type': 'single', 'file': 'Key_yellow.npy'},
        {'name': 'key_black', 'type': 'single', 'file': 'Key_black.npy'},
        #gates
        
        #ToDo gates animation

        #items
        {'name': 'sword', 'type': 'single', 'file': 'Sword.npy'},
        {'name': 'bridge', 'type': 'single', 'file': 'Bridge.npy'},
        {'name': 'magnet', 'type': 'single', 'file': 'Magnet.npy'},
        #Chalice
        {'name': 'chalice', 'type': 'single', 'file': 'Chalice_Pink.npy'}
        #ToDo remaining chalice colors for blinking
    )


class AdventureConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 250
    #upper left corner is 0, 0

    # sset config baked into constants (immutable default) for asset overrides
    ASSET_CONFIG: tuple = _get_default_asset_config()


# immutable state container

class AdventureState(NamedTuple):
    #step conter for performance indicator?
    step_counter: chex.Array
    #position player
    player_x: chex.Array
    player_y: chex.Array
    player_color:chex.Array
    player_tile: chex.Array
    #positions dragons
    dragon_yellow_x: chex.Array
    dragon_yellow_y: chex.Array
    dragon_yellow_tile: chex.Array
    dragon_green_x: chex.Array
    dragon_green_y: chex.Array
    dragon_green_tile: chex.Array
    #state dragons (alive, dead, attacking)
    dragon_yellow_state: chex.Array
    dragon_green_state: chex.Array
    #positions keys
    key_yellow_x: chex.Array
    key_yellow_y: chex.Array
    key_yellow_tile: chex.Array
    key_black_x: chex.Array
    key_black_y: chex.Array
    key_black_tile: chex.Array
    #state of gates (if open or closed)
    gate_yellow_state: chex.Array
    gate_black_state: chex.Array
    #position sword
    sword_x: chex.Array
    sword_y: chex.Array
    sword_tile: chex.Array
    #position bridge
    bridge_x: chex.Array
    bridge_y: chex.Array
    bridge_tile: chex.Array
    #position magnet
    magnet_x: chex.Array
    magnet_y: chex.Array
    magnet_tile: chex.Array
    #position chalice
    chalice_x: chex.Array
    chalice_y: chex.Array
    chalice_tile: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    tile: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    state: jnp.ndarray


class AdventureObservation(NamedTuple):
    player: EntityPosition
    dragon_yellow: EntityPosition
    dragon_green: EntityPosition
    key_yellow: EntityPosition
    key_black: EntityPosition
    gate_yellow: EntityPosition
    gate_black: EntityPosition
    sword: EntityPosition
    bridge: EntityPosition
    magnet: EntityPosition
    chalice: EntityPosition


class AdventureInfo(NamedTuple):
    time: jnp.ndarray


class JaxAdventure(JaxEnvironment[AdventureState, AdventureObservation, AdventureInfo, AdventureConstants]):
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
            step_counter = state.step_counter,
            player_x = new_player_x,
            player_y = new_player_y,
            player_color= state.player_color,
            player_tile = state.player_tile,
            dragon_yellow_x = state.dragon_yellow_x,
            dragon_yellow_y = state.dragon_yellow_y,
            dragon_yellow_tile = state.dragon_yellow_tile,
            dragon_green_x = state.dragon_green_x,
            dragon_green_y = state.dragon_green_y,
            dragon_green_tile=state.dragon_green_tile,
            dragon_yellow_state = state.dragon_yellow_state,
            dragon_green_state = state.dragon_green_state,
            key_yellow_x = state.key_yellow_x,
            key_yellow_y = state.key_yellow_y,
            key_yellow_tile=state.key_yellow_tile,
            key_black_x = state.key_black_x,
            key_black_y = state.key_black_y,
            key_black_tile=state.key_black_tile,
            gate_yellow_state = state.gate_yellow_state,
            gate_black_state = state.gate_black_state,
            sword_x = state.sword_x,
            sword_y = state.sword_y,
            sword_tile=state.sword_tile,
            bridge_x = state.bridge_x,
            bridge_y = state.bridge_y,
            bridge_tile=state.bridge_tile,
            magnet_x = state.magnet_x,
            magnet_y = state.magnet_y,
            magnet_tile=state.magnet_tile,
            chalice_x = state.chalice_x,
            chalice_y = state.chalice_y,
            chalice_tile=state.chalice_tile
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        previous_state = state
        state = self._player_step(state, action)

    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AdventureObservation, AdventureState]:

        state = AdventureState(
            step_counter = jnp.array(0).astype(jnp.int32),
            #player
            player_x = jnp.array(78).astype(jnp.int32),     #spawn X 
            player_y = jnp.array(174).astype(jnp.int32),    #spawn Y
            player_tile = jnp.array(0).astype(jnp.int32),   #Spawn Tile
            player_color= jnp.array(0).astype(jnp.int32),   #Spawn Color
            #Dragons
            dragon_yellow_x = jnp.array(120).astype(jnp.int32), #ToDo
            dragon_yellow_y = jnp.array(50).astype(jnp.int32), #ToDo
            dragon_yellow_tile = jnp.array(0).astype(jnp.int32), #ToDo
            dragon_green_x = jnp.array(120).astype(jnp.int32), #ToDo
            dragon_green_y = jnp.array(70).astype(jnp.int32), #ToDo
            dragon_green_tile = jnp.array(70).astype(jnp.int32), #ToDo
            dragon_yellow_state = jnp.array(0).astype(jnp.int32), #ToDo
            dragon_green_state = jnp.array(0).astype(jnp.int32), #ToDo
            #Keys
            key_yellow_x = jnp.array(31).astype(jnp.int32),
            key_yellow_y = jnp.array(110).astype(jnp.int32),
            key_yellow_tile = jnp.array(0).astype(jnp.int32),
            key_black_x = jnp.array(120).astype(jnp.int32), #ToDo
            key_black_y = jnp.array(90).astype(jnp.int32), #ToDo
            key_black_tile = jnp.array(0).astype(jnp.int32), #ToDo
            gate_yellow_state = jnp.array(0).astype(jnp.int32), #ToDo
            gate_black_state = jnp.array(0).astype(jnp.int32), #ToDo
            #Items
            sword_x = jnp.array(120).astype(jnp.int32), #ToDo
            sword_y = jnp.array(110).astype(jnp.int32), #ToDo
            sword_tile = jnp.array(0).astype(jnp.int32), #ToDo
            bridge_x = jnp.array(120).astype(jnp.int32), #ToDo
            bridge_y = jnp.array(130).astype(jnp.int32), #ToDo
            bridge_tile = jnp.array(0).astype(jnp.int32), #ToDo
            magnet_x = jnp.array(120).astype(jnp.int32), #ToDo
            magnet_y = jnp.array(150).astype(jnp.int32), #ToDo
            magnet_tile = jnp.array(0).astype(jnp.int32), #ToDo
            chalice_x = jnp.array(120).astype(jnp.int32), #ToDo
            chalice_y = jnp.array(170).astype(jnp.int32), #ToDo
            chalice_tile = jnp.array(170).astype(jnp.int32) #ToDo
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        # Split step key from state and keep a new key for the next state
        previous_state = state
        # Make per-step key available to helpers that may read state.key
        state = AdventureState(
            step_counter = state.step_counter,
            player_x = state.player_x,
            player_y=state.player_y,
            player_color= state.player_color,
            player_tile = state.player_tile,
            dragon_yellow_x = state.dragon_yellow_x,
            dragon_yellow_y = state.dragon_yellow_y,
            dragon_yellow_tile = state.dragon_yellow_tile,
            dragon_green_x = state.dragon_green_x,
            dragon_green_y = state.dragon_green_y,
            dragon_green_tile=state.dragon_green_tile,
            dragon_yellow_state = state.dragon_yellow_state,
            dragon_green_state = state.dragon_green_state,
            key_yellow_x = state.key_yellow_x,
            key_yellow_y = state.key_yellow_y,
            key_yellow_tile=state.key_yellow_tile,
            key_black_x = state.key_black_x,
            key_black_y = state.key_black_y,
            key_black_tile=state.key_black_tile,
            gate_yellow_state = state.gate_yellow_state,
            gate_black_state = state.gate_black_state,
            sword_x = state.sword_x,
            sword_y = state.sword_y,
            sword_tile=state.sword_tile,
            bridge_x = state.bridge_x,
            bridge_y = state.bridge_y,
            bridge_tile=state.bridge_tile,
            magnet_x = state.magnet_x,
            magnet_y = state.magnet_y,
            magnet_tile=state.magnet_tile,
            chalice_x = state.chalice_x,
            chalice_y = state.chalice_y,
            chalice_tile=state.chalice_tile
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
            tile=state.player_tile,
            width=4, 
            height=8, 
            state=1 #ToDO
        )
        dragon_yellow = EntityPosition(
            x=state.dragon_yellow_x,
            y=state.dragon_yellow_y,
            tile=state.dragon_yellow_tile,
            width=8, 
            height=44, 
            state=1 #ToDO
        )
        dragon_green = EntityPosition(
            x=state.dragon_green_x,
            y=state.dragon_green_y,
            tile=state.dragon_green_tile,
            width=8, 
            height=44, 
            state=1 #ToDO
        )
        key_yellow = EntityPosition(
            x=state.key_yellow_x,
            y=state.key_yellow_y,
            tile=state.key_yellow_tile,
            width=8, 
            height=6,
            state=1 #ToDO
        )
        key_black = EntityPosition(
            x=state.key_black_x,
            y=state.key_black_y,
            tile=state.key_black_tile,
            width=8, 
            height=6,
            state=1 #ToDO
        )
        gate_yellow = EntityPosition(
            x=80, #ToDO
            y=80, #ToDO
            tile=1, #ToDO
            width=7, 
            height=32, 
            state=state.gate_yellow_state #ToDO
        )
        gate_black = EntityPosition(
            x=100, #ToDO
            y=100, #ToDO
            tile=1, #ToDO
            width=7, 
            height=32, 
            state=state.gate_black_state #ToDO
        )
        sword = EntityPosition(
            x=state.sword_x,
            y=state.sword_y,
            tile=state.sword_tile,
            width=8, 
            height=10, 
            state=1 #ToDO
        )
        bridge = EntityPosition(
            x=state.bridge_x,
            y=state.bridge_y,
            tile=state.bridge_tile,
            width=32, 
            height=48, 
            state=1 #ToDO
        )
        magnet = EntityPosition(
            x=state.magnet_x,
            y=state.magnet_y,
            tile=state.magnet_tile,
            width=8, 
            height=16,
            state=1 #ToDO
        )
        chalice = EntityPosition(
            x=state.chalice_x,
            y=state.chalice_y,
            tile=state.chalice_tile,
            width=8, 
            height=18,
            state=1 #ToDO
        )

        return AdventureObservation(
            player=player,
            dragon_yellow=dragon_yellow,
            dragon_green=dragon_green,
            key_yellow=key_yellow,
            key_black=key_black,
            gate_yellow=gate_yellow,
            gate_black=gate_black,
            sword=sword,
            bridge=bridge,
            magnet=magnet,
            chalice=chalice
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

            #ToDo for the rest of the dragons, items etc.....
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
            #downscale=(200, 128)
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
        room_mask =self.SHAPE_MASKS["room_number"][state.player_tile]
        raster = self.jr.render_at(raster, 0, 0, room_mask)

        #set player color here
        player_mask = self.SHAPE_MASKS["player_colors"][state.player_color]
        raster = self.jr.render_at(raster, state.player_x, state.player_y, player_mask)

        #dragons
        dragon_yellow_mask = self.SHAPE_MASKS["dragon_yellow_neutral"]
        raster = self.jr.render_at(raster, state.dragon_yellow_x, state.dragon_yellow_y, dragon_yellow_mask)
        dragon_green_mask = self.SHAPE_MASKS["dragon_green-neutral"]
        raster = self.jr.render_at(raster, state.dragon_green_x, state.dragon_green_y, dragon_green_mask)
        #keys
        key_yellow_mask = self.SHAPE_MASKS["key_yellow"]
        raster = self.jr.render_at(raster, state.key_yellow_x, state.key_yellow_y, key_yellow_mask)
        key_black_mask = self.SHAPE_MASKS["key_black"]
        raster = self.jr.render_at(raster, state.key_black_x, state.key_black_y, key_black_mask)
        #items
        sword_mask = self.SHAPE_MASKS["sword"]
        raster = self.jr.render_at(raster, state.sword_x, state.sword_y, sword_mask)
        bridge_mask = self.SHAPE_MASKS["bridge"]
        raster = self.jr.render_at(raster, state.bridge_x, state.bridge_y, bridge_mask)
        magnet_mask = self.SHAPE_MASKS["magnet"]
        raster = self.jr.render_at(raster, state.magnet_x, state.magnet_y, magnet_mask)
        #chalice
        chalice_mask = self.SHAPE_MASKS["chalice"]
        raster = self.jr.render_at(raster, state.chalice_x, state.chalice_y, chalice_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
