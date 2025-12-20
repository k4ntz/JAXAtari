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
        #all rooms in order ToDo pt overview map into the readme?
        {'name': 'room_number', 'type': 'group', 'files': ['Room_1.npy', 
                                                           'Room_2.npy', 
                                                           'Room_3.npy', 
                                                           'Room_4.npy', 
                                                           'Room_5.npy', 
                                                           'Room_6.npy', 
                                                           'Room_7.npy', 
                                                           'Room_8.npy', 
                                                           'Room_9.npy', 
                                                           'Room_10.npy', 
                                                           'Room_11.npy', 
                                                           'Room_12.npy', 
                                                           'Room_13.npy', 
                                                           'Room_14.npy']},
        {'name': 'bg', 'type': 'background', 'file': 'Room_1.npy'},
        #Player in all the different colors
        {'name': 'player_colors', 'type': 'group', 'files': ["Player_Yellow.npy", 
                                                             "Player_Green.npy", 
                                                             "Player_Purple.npy", 
                                                             "Player_Pink.npy", 
                                                             "Player_Green_yellow.npy", 
                                                             "Player_Blue.npy", 
                                                             "Player_Black.npy", 
                                                             "Player_Magenta.npy"]},
        #Dragons and their animations
        {'name': 'dragon_yellow', 'type': 'group', 'files': ['Dragon_yellow_neutral.npy',
                                                             'Dragon_yellow_attack.npy',
                                                             'Dragon_yellow_dead.npy']},
        {'name': 'dragon_green', 'type': 'group', 'files': ['Dragon_green_neutral.npy',
                                                             'Dragon_green_attack.npy',
                                                             'Dragon_green_dead.npy']},                                                     
        #Keys
        {'name': 'key_yellow', 'type': 'single', 'file': 'Key_yellow.npy'},
        {'name': 'key_black', 'type': 'single', 'file': 'Key_black.npy'},
        #Gate and its animation
        {'name': 'gate_state', 'type': 'group', 'files': ['Gate_closed.npy',
                                                          'Gate_opening_0.npy',
                                                          'Gate_opening_1.npy',
                                                          'Gate_opening_2.npy',
                                                          'Gate_opening_3.npy',
                                                          'Gate_opening_4.npy',
                                                          'Gate_open.npy']},
        #Items
        {'name': 'sword', 'type': 'single', 'file': 'Sword.npy'},
        {'name': 'bridge', 'type': 'single', 'file': 'Bridge.npy'},
        {'name': 'magnet', 'type': 'single', 'file': 'Magnet.npy'},
        #Chalice
        {'name': 'chalice', 'type': 'group', 'files': ['Chalice_Black.npy',
                                                       'Chalice_DarkBlue.npy',
                                                       'Chalice_Gray.npy',
                                                       'Chalice_Green.npy',
                                                       'Chalice_LightBlue.npy',
                                                       'Chalice_Pink.npy',
                                                       'Chalice_Purple.npy',
                                                       'Chalice_Red.npy',
                                                       'Chalice_Turquoise.npy',
                                                       'Chalice_Yellow.npy']},
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
    #position player: x ,y ,tile , color
    player: chex.Array
    #positions dragons: x, y ,tile ,state
    dragon_yellow: chex.Array
    dragon_green: chex.Array
    #positions keys: x, y, tile
    key_yellow: chex.Array
    key_black: chex.Array
    #gates: state
    gate_yellow: chex.Array
    gate_black: chex.Array
    #position sword: x, y, tile
    sword: chex.Array
    #position bridge: x, y, tile
    bridge: chex.Array
    #position magnet: x, y, tile
    magnet: chex.Array
    #position chalice: x, y, tile, color
    chalice: chex.Array


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
        new_player_x = state.player[0]
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

        new_player_y = state.player[1]
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
            player = jnp.array([new_player_x,new_player_y,state.player[2],state.player[3]]).astype(jnp.int32), #SEEMS NOT GOOD
            dragon_yellow = state.dragon_yellow,
            dragon_green = state.dragon_green,
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice
        )

    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        previous_state = state
        state = self._player_step(state, action)

    
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[AdventureObservation, AdventureState]:

        state = AdventureState(
            step_counter = jnp.array(0).astype(jnp.int32),
            #Player Spawn: x, y, tile, color
            player = jnp.array([78,174,0,0]).astype(jnp.int32),
            #Dragons: x, y ,tile ,state
            dragon_yellow = jnp.array([120,50,0,0]).astype(jnp.int32), #ToDo
            dragon_green = jnp.array([120,80,0,2]).astype(jnp.int32), #ToDo
            #Keys: x ,y, tile
            key_yellow = jnp.array([31,110,0]).astype(jnp.int32),
            key_black = jnp.array([31,80,0]).astype(jnp.int32),
            #Gate: state
            gate_yellow=jnp.array([0]).astype(jnp.int32),
            gate_black=jnp.array([0]).astype(jnp.int32),
            #Items: x, y, tile
            sword = jnp.array([120,120,0]).astype(jnp.int32), #ToDo
            bridge= jnp.array([120,120,0]).astype(jnp.int32), #ToDo
            magnet= jnp.array([120,120,0]).astype(jnp.int32), #ToDo
            #Chalice: x, y, tile, color
            chalice= jnp.array([120,120,0,0]).astype(jnp.int32), #ToDo
        )
        initial_obs = self._get_observation(state)

        return initial_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: AdventureState, action: chex.Array) -> Tuple[AdventureObservation, AdventureState, float, bool, AdventureInfo]:
        # Split step key from state and keep a new key for the next state
        previous_state = state
        # Make per-step key available to helpers that may read state.key
        state = AdventureState(
            step_counter=state.step_counter,
            player=state.player,
            dragon_yellow=state.dragon_yellow,
            dragon_green=state.dragon_green,
            key_yellow=state.key_yellow,
            key_black=state.key_black,
            gate_yellow=state.gate_yellow,
            gate_black=state.gate_black,
            sword=state.sword,
            bridge=state.bridge,
            magnet=state.magnet,
            chalice=state.chalice
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
            x=state.player[0],
            y=state.player[1],
            tile=state.player[2],
            width=4, 
            height=8, 
            state=1 #ToDO
        )
        dragon_yellow = EntityPosition(
            x=state.dragon_yellow[0],
            y=state.dragon_yellow[1],
            tile=state.dragon_yellow[2],
            width=8, 
            height=44, 
            state=state.dragon_yellow[3] #ToDO
        )
        dragon_green = EntityPosition(
            x=state.dragon_green[0],
            y=state.dragon_green[1],
            tile=state.dragon_green[2],
            width=8, 
            height=44, 
            state=state.dragon_green[3] #ToDO
        )
        key_yellow = EntityPosition(
            x=state.key_yellow[0],
            y=state.key_yellow[1],
            tile=state.key_yellow[2],
            width=8, 
            height=6,
            state=1 #ToDO
        )
        key_black = EntityPosition(
            x=state.key_black[0],
            y=state.key_black[1],
            tile=state.key_black[2],
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
            state=state.gate_yellow[0] #ToDO
        )
        gate_black = EntityPosition(
            x=100, #ToDO
            y=100, #ToDO
            tile=1, #ToDO
            width=7, 
            height=32, 
            state=state.gate_black[0] #ToDO
        )
        sword = EntityPosition(
            x=state.sword[0],
            y=state.sword[1],
            tile=state.sword[2],
            width=8, 
            height=10, 
            state=1 #ToDO
        )
        bridge = EntityPosition(
            x=state.bridge[0],
            y=state.bridge[1],
            tile=state.bridge[2],
            width=32, 
            height=48, 
            state=1 #ToDO
        )
        magnet = EntityPosition(
            x=state.magnet[0],
            y=state.magnet[1],
            tile=state.magnet[2],
            width=8, 
            height=16,
            state=1 #ToDO
        )
        chalice = EntityPosition(
            x=state.chalice[0],
            y=state.chalice[1],
            tile=state.chalice[2],
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

            #ToDo for the rest of the dragons, items etc..... ?
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
        room_mask =self.SHAPE_MASKS["room_number"][state.player[2]]
        raster = self.jr.render_at(raster, 0, 0, room_mask)

        #set player color here
        player_mask = self.SHAPE_MASKS["player_colors"][state.player[3]]
        raster = self.jr.render_at(raster, state.player[0], state.player[1], player_mask)

        #dragons
        dragon_yellow_mask = self.SHAPE_MASKS["dragon_yellow"][state.dragon_yellow[3]]
        raster = self.jr.render_at(raster, state.dragon_yellow[0], state.dragon_yellow[1], dragon_yellow_mask)
        dragon_green_mask = self.SHAPE_MASKS["dragon_green"][state.dragon_green[3]]
        raster = self.jr.render_at(raster, state.dragon_green[0], state.dragon_green[1], dragon_green_mask)
        #keys
        key_yellow_mask = self.SHAPE_MASKS["key_yellow"]
        raster = self.jr.render_at(raster, state.key_yellow[0], state.key_yellow[1], key_yellow_mask)
        key_black_mask = self.SHAPE_MASKS["key_black"]
        raster = self.jr.render_at(raster, state.key_black[0], state.key_black[1], key_black_mask)
        #Gates
        gate_yellow_mask = self.SHAPE_MASKS["gate_state"][state.gate_yellow[0]]
        raster = self.jr.render_at(raster, 77, 140, gate_yellow_mask)
        gate_black_mask = self.SHAPE_MASKS["gate_state"][state.gate_black[0]]
        raster = self.jr.render_at(raster, 30, 30, gate_black_mask)#ToDO

        #items
        sword_mask = self.SHAPE_MASKS["sword"]
        raster = self.jr.render_at(raster, state.sword[0], state.sword[1], sword_mask)
        bridge_mask = self.SHAPE_MASKS["bridge"]
        raster = self.jr.render_at(raster, state.bridge[0], state.bridge[1], bridge_mask)
        magnet_mask = self.SHAPE_MASKS["magnet"]
        raster = self.jr.render_at(raster, state.magnet[0], state.magnet[1], magnet_mask)
        #chalice
        chalice_mask = self.SHAPE_MASKS["chalice"][state.chalice[3]]
        raster = self.jr.render_at(raster, state.chalice[0], state.chalice[1], chalice_mask)

        return self.jr.render_from_palette(raster, self.PALETTE)
