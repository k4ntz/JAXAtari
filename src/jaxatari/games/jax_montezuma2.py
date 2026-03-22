import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct
from typing import Tuple
from functools import partial
import os

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, ObjectObservation
from jaxatari.spaces import Discrete
from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils

class Montezuma2Constants(struct.PyTreeNode):
    # Homogeneous Padding Limits
    MAX_ENEMIES_PER_ROOM: int = struct.field(pytree_node=False, default=3)
    MAX_LADDERS_PER_ROOM: int = struct.field(pytree_node=False, default=4)
    MAX_ROPES_PER_ROOM: int = struct.field(pytree_node=False, default=2)
    MAX_DOORS_PER_ROOM: int = struct.field(pytree_node=False, default=2)
    MAX_ITEMS_PER_ROOM: int = struct.field(pytree_node=False, default=2)
    MAX_CONVEYORS_PER_ROOM: int = struct.field(pytree_node=False, default=2)
    
    # Gameplay Constants
    WIDTH: int = struct.field(pytree_node=False, default=160)
    HEIGHT: int = struct.field(pytree_node=False, default=210)
    PLAYER_WIDTH: int = struct.field(pytree_node=False, default=7)
    PLAYER_HEIGHT: int = struct.field(pytree_node=False, default=20)
    INITIAL_PLAYER_X: int = struct.field(pytree_node=False, default=77)
    INITIAL_PLAYER_Y: int = struct.field(pytree_node=False, default=26)
    PLAYER_SPEED: int = struct.field(pytree_node=False, default=1)
    JUMP_Y_OFFSETS: jnp.ndarray = struct.field(pytree_node=False, default_factory=lambda: jnp.array([3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3], dtype=jnp.int32))
    GRAVITY: int = struct.field(pytree_node=False, default=2)
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

@struct.dataclass
class Montezuma2State:
    # Game State
    room_id: jnp.ndarray
    lives: jnp.ndarray
    score: jnp.ndarray
    frame_count: jnp.ndarray
    
    # Player State
    player_x: jnp.ndarray
    player_y: jnp.ndarray
    player_vx: jnp.ndarray
    player_vy: jnp.ndarray
    
    is_jumping: jnp.ndarray
    is_falling: jnp.ndarray
    jump_counter: jnp.ndarray
    is_climbing: jnp.ndarray
    last_rope: jnp.ndarray
    
    # Homogeneous Entities for the CURRENT room
    enemies_x: jnp.ndarray
    enemies_y: jnp.ndarray
    enemies_active: jnp.ndarray
    
    ladders_x: jnp.ndarray
    ladders_top: jnp.ndarray
    ladders_bottom: jnp.ndarray
    ladders_active: jnp.ndarray
    
    ropes_x: jnp.ndarray
    ropes_top: jnp.ndarray
    ropes_bottom: jnp.ndarray
    ropes_active: jnp.ndarray
    
    items_x: jnp.ndarray
    items_y: jnp.ndarray
    items_active: jnp.ndarray
    
    doors_x: jnp.ndarray
    doors_y: jnp.ndarray
    doors_active: jnp.ndarray
    
    conveyors_x: jnp.ndarray
    conveyors_y: jnp.ndarray
    conveyors_active: jnp.ndarray
    conveyors_direction: jnp.ndarray
    
    key: jrandom.PRNGKey

@struct.dataclass
class Montezuma2Observation:
    player: ObjectObservation
    enemies: ObjectObservation
    items: ObjectObservation
    conveyors: ObjectObservation
    doors: ObjectObservation
    ropes: ObjectObservation

@struct.dataclass
class Montezuma2Info:
    lives: jnp.ndarray
    room_id: jnp.ndarray

class Montezuma2Renderer(JAXGameRenderer):
    def __init__(self, consts: Montezuma2Constants = None, config: render_utils.RendererConfig = None):
        super().__init__(consts)
        self.consts = consts or Montezuma2Constants()
        
        if config is None:
            self.config = render_utils.RendererConfig(
                game_dimensions=(self.consts.HEIGHT, self.consts.WIDTH),
                channels=3,
                downscale=None
            )
        else:
            self.config = config

        self.jr = render_utils.JaxRenderingUtils(self.config)
        sprite_path = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma")
        
        # Transparent background base for the 210x160 raster
        bg_data = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 4), dtype=jnp.uint8)
        bg_data = bg_data.at[:, :, 3].set(255) # Opaque black
        
        final_asset_config = [
            {'name': 'bg', 'type': 'background', 'data': bg_data},
            {'name': 'room_bg', 'type': 'single', 'file': 'backgrounds/mid_room_level_0.npy', 'transpose': False},
            {'name': 'player', 'type': 'single', 'file': 'player/player_sprite.npy', 'transpose': False},
            {'name': 'enemy', 'type': 'single', 'file': 'enemies/bounce_skull.npy', 'transpose': False},
            {'name': 'item', 'type': 'single', 'file': 'items/key.npy', 'transpose': False},
            {'name': 'door', 'type': 'single', 'file': 'door.npy', 'transpose': False},
            {'name': 'conveyor', 'type': 'single', 'file': 'conveyor_belt.npy', 'transpose': False},
        ]
        
        (
            self.PALETTE,
            self.SHAPE_MASKS,
            self.BACKGROUND,
            self.COLOR_TO_ID,
            self.FLIP_OFFSETS
        ) = self.jr.load_and_setup_assets(final_asset_config, sprite_path)
        
        # Accurate ladder color for Difficulty 1, Layer 1
        self.LADDER_COLOR = jnp.array([66, 158, 130], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LADDER_COLOR[None, :]], axis=0)
        self.LADDER_ID = self.PALETTE.shape[0] - 1

        # Accurate door color
        self.DOOR_COLOR = jnp.array([232, 204, 99], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.DOOR_COLOR[None, :]], axis=0)
        self.DOOR_ID = self.PALETTE.shape[0] - 1
        door_mask = self.SHAPE_MASKS["door"]
        self.SHAPE_MASKS["door"] = jnp.where(door_mask != self.jr.TRANSPARENT_ID, self.DOOR_ID, self.jr.TRANSPARENT_ID)
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: Montezuma2State) -> jnp.ndarray:
        # Start with solid black background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Draw Room Background
        raster = self.jr.render_at(raster, 0, 47, self.SHAPE_MASKS["room_bg"])
        
        # Draw Ladders (Vertical Rails + Horizontal Rungs)
        def draw_ladder_accurate(i, r):
            x, top, bottom = state.ladders_x[i], state.ladders_top[i] + 47, state.ladders_bottom[i] + 47
            active = state.ladders_active[i]
            
            def _draw(raster_in):
                ladder_width = 16
                # Vertical Rails (4 pixels wide)
                rail_pos = jnp.array([[x, top], [x + ladder_width - 4, top]])
                rail_size = jnp.array([[4, bottom - top], [4, bottom - top]])
                raster_in = self.jr.draw_rects(raster_in, rail_pos, rail_size, self.LADDER_ID)
                
                # Horizontal Rungs (2 pixels high, 5 pixels gap)
                # First rung starts at top + 4
                rung_pos = jnp.array([[x, top + 4]])
                rung_size = jnp.array([[ladder_width, bottom - top - 4]])
                raster_in = self.jr.draw_ladders(raster_in, rung_pos, rung_size, 2, 5, self.LADDER_ID)
                return raster_in

            return jax.lax.cond(active == 1, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_LADDERS_PER_ROOM, draw_ladder_accurate, raster)

        # Draw Ropes
        def draw_rope(i, r):
            x, top, bottom = state.ropes_x[i], state.ropes_top[i] + 47, state.ropes_bottom[i] + 47
            active = state.ropes_active[i]
            
            def _draw(raster_in):
                rail_pos = jnp.array([[x, top]])
                rail_size = jnp.array([[1, bottom - top + 1]])
                return self.jr.draw_rects(raster_in, rail_pos, rail_size, self.DOOR_ID)
                
            return jax.lax.cond(active == 1, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_ROPES_PER_ROOM, draw_rope, raster)
        
        # Draw Conveyors
        anim_idx = jnp.less(jnp.mod(state.frame_count, 16), 8)
        def render_conveyor(i, raster):
            mask = self.SHAPE_MASKS["conveyor"]
            return jax.lax.cond(
                state.conveyors_active[i] == 1,
                lambda r: self.jr.render_at(r, state.conveyors_x[i], state.conveyors_y[i] + 47, mask, flip_horizontal=anim_idx),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, render_conveyor, raster)
        
        # Draw Items
        def render_item(i, raster):
            mask = self.SHAPE_MASKS["item"]
            return jax.lax.cond(
                state.items_active[i] == 1,
                lambda r: self.jr.render_at(r, state.items_x[i], state.items_y[i] + 47, mask),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ITEMS_PER_ROOM, render_item, raster)

        # Draw Doors
        def render_door(i, raster):
            mask = self.SHAPE_MASKS["door"]
            return jax.lax.cond(
                state.doors_active[i] == 1,
                lambda r: self.jr.render_at(r, state.doors_x[i], state.doors_y[i] + 47, mask),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_DOORS_PER_ROOM, render_door, raster)
        
        # Draw Enemies
        def render_enemy(i, raster):
            mask = self.SHAPE_MASKS["enemy"]
            return jax.lax.cond(
                state.enemies_active[i] == 1,
                lambda r: self.jr.render_at(r, state.enemies_x[i], state.enemies_y[i] + 47, mask),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES_PER_ROOM, render_enemy, raster)
        
        # Draw Player
        raster = self.jr.render_at(raster, state.player_x, state.player_y + 47, self.SHAPE_MASKS["player"])
        
        return self.PALETTE[raster]

class JaxMontezuma2(JaxEnvironment[Montezuma2State, Montezuma2Observation, Montezuma2Info, Montezuma2Constants]):
    ACTION_SET: jnp.ndarray = jnp.array([
        Action.NOOP, Action.FIRE, Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN,
        Action.UPRIGHT, Action.UPLEFT, Action.DOWNRIGHT, Action.DOWNLEFT,
        Action.UPFIRE, Action.RIGHTFIRE, Action.LEFTFIRE, Action.DOWNFIRE,
        Action.UPRIGHTFIRE, Action.UPLEFTFIRE, Action.DOWNRIGHTFIRE, Action.DOWNLEFTFIRE
    ], dtype=jnp.int32)

    def __init__(self, consts: Montezuma2Constants = None):
        consts = consts or Montezuma2Constants()
        super().__init__(consts)
        self.renderer = Montezuma2Renderer(self.consts)
        
        sprite_path = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma", "backgrounds", "mid_room_collision_level_0.npy")
        col_map = jnp.load(sprite_path)[:, :, 0] # (149, 160)
        room_col = jnp.where(col_map > 0, 1, 0).astype(jnp.int32)
        
        # Add 4-pixel side walls from y=6 downwards
        room_col = room_col.at[6:, 0:4].set(1)
        room_col = room_col.at[6:, self.consts.WIDTH-4:].set(1)
        self.ROOM_COLLISION_MAP = room_col

    def reset(self, key: jrandom.PRNGKey) -> Tuple[Montezuma2Observation, Montezuma2State]:
        enemies_x = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_y = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_active = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        
        ladders_x = jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
        ladders_top = jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
        ladders_bottom = jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
        ladders_active = jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32)
        
        ropes_x = jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
        ropes_top = jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
        ropes_bottom = jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)
        ropes_active = jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32)

        items_x = jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
        items_y = jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)
        items_active = jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32)

        doors_x = jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
        doors_y = jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32)
        doors_active = jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32)

        conveyors_x = jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
        conveyors_y = jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
        conveyors_active = jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)
        conveyors_direction = jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32)

        # LOAD ROOM 1 DATA
        enemies_x = enemies_x.at[0].set(93)
        enemies_y = enemies_y.at[0].set(119)
        enemies_active = enemies_active.at[0].set(1)
        
        ladders_x = ladders_x.at[0].set(72)
        ladders_top = ladders_top.at[0].set(49)
        ladders_bottom = ladders_bottom.at[0].set(88)
        ladders_active = ladders_active.at[0].set(1)
        
        ladders_x = ladders_x.at[1].set(128)
        ladders_top = ladders_top.at[1].set(92)
        ladders_bottom = ladders_bottom.at[1].set(133)
        ladders_active = ladders_active.at[1].set(1)
        
        ladders_x = ladders_x.at[2].set(16)
        ladders_top = ladders_top.at[2].set(92)
        ladders_bottom = ladders_bottom.at[2].set(133)
        ladders_active = ladders_active.at[2].set(1)

        ropes_x = ropes_x.at[0].set(111)
        ropes_top = ropes_top.at[0].set(49)
        ropes_bottom = ropes_bottom.at[0].set(88)
        ropes_active = ropes_active.at[0].set(1)

        items_x = items_x.at[0].set(13)
        items_y = items_y.at[0].set(52)
        items_active = items_active.at[0].set(1)

        # Add a Conveyor Belt at the center to test it (floating so it's clearly visible)
        conveyors_x = conveyors_x.at[0].set(60)
        conveyors_y = conveyors_y.at[0].set(88)
        conveyors_active = conveyors_active.at[0].set(1)
        conveyors_direction = conveyors_direction.at[0].set(1) # 1 for right, -1 for left
        
        doors_x = doors_x.at[0].set(16)
        doors_y = doors_y.at[0].set(7)
        doors_active = doors_active.at[0].set(1)
        doors_x = doors_x.at[1].set(140)
        doors_y = doors_y.at[1].set(7)
        doors_active = doors_active.at[1].set(1)

        state = Montezuma2State(
            room_id=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(5, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            frame_count=jnp.array(0, dtype=jnp.int32),
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.INITIAL_PLAYER_Y, dtype=jnp.int32),
            player_vx=jnp.array(0, dtype=jnp.int32),
            player_vy=jnp.array(0, dtype=jnp.int32),
            is_jumping=jnp.array(0, dtype=jnp.int32),
            is_falling=jnp.array(0, dtype=jnp.int32),
            jump_counter=jnp.array(0, dtype=jnp.int32),
            is_climbing=jnp.array(0, dtype=jnp.int32),
            last_rope=jnp.array(-1, dtype=jnp.int32),
            enemies_x=enemies_x,
            enemies_y=enemies_y,
            enemies_active=enemies_active,
            ladders_x=ladders_x,
            ladders_top=ladders_top,
            ladders_bottom=ladders_bottom,
            ladders_active=ladders_active,
            ropes_x=ropes_x,
            ropes_top=ropes_top,
            ropes_bottom=ropes_bottom,
            ropes_active=ropes_active,
            items_x=items_x,
            items_y=items_y,
            items_active=items_active,
            doors_x=doors_x,
            doors_y=doors_y,
            doors_active=doors_active,
            conveyors_x=conveyors_x,
            conveyors_y=conveyors_y,
            conveyors_active=conveyors_active,
            conveyors_direction=conveyors_direction,
            key=key
        )
        
        return self._get_observation(state), state

    def step(self, state: Montezuma2State, action: int) -> Tuple[Montezuma2Observation, Montezuma2State, float, bool, Montezuma2Info]:
        is_up = jnp.logical_or(action == Action.UP, jnp.logical_or(action == Action.UPRIGHT, action == Action.UPLEFT))
        is_up = jnp.logical_or(is_up, jnp.logical_or(action == Action.UPFIRE, jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.UPLEFTFIRE)))
        is_down = jnp.logical_or(action == Action.DOWN, jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNLEFT))
        is_down = jnp.logical_or(is_down, jnp.logical_or(action == Action.DOWNFIRE, jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE)))
        is_right = jnp.logical_or(action == Action.RIGHT, jnp.logical_or(action == Action.UPRIGHT, action == Action.DOWNRIGHT))
        is_right = jnp.logical_or(is_right, jnp.logical_or(action == Action.RIGHTFIRE, jnp.logical_or(action == Action.UPRIGHTFIRE, action == Action.DOWNRIGHTFIRE)))
        is_left = jnp.logical_or(action == Action.LEFT, jnp.logical_or(action == Action.UPLEFT, action == Action.DOWNLEFT))
        is_left = jnp.logical_or(is_left, jnp.logical_or(action == Action.LEFTFIRE, jnp.logical_or(action == Action.UPLEFTFIRE, action == Action.DOWNLEFTFIRE)))
        is_fire = jnp.logical_or(action == Action.FIRE, jnp.logical_or(action == Action.UPFIRE, jnp.logical_or(action == Action.DOWNFIRE, jnp.logical_or(action == Action.RIGHTFIRE, action == Action.LEFTFIRE))))
        is_fire = jnp.logical_or(is_fire, jnp.logical_or(action == Action.UPRIGHTFIRE, jnp.logical_or(action == Action.UPLEFTFIRE, jnp.logical_or(action == Action.DOWNRIGHTFIRE, action == Action.DOWNLEFTFIRE))))
        
        # Player Velocity is locked during jump and falls
        is_in_air = jnp.logical_or(state.is_jumping == 1, state.is_falling == 1)
        new_vx = jax.lax.select(is_right, self.consts.PLAYER_SPEED, 0)
        new_vx = jax.lax.select(is_left, -self.consts.PLAYER_SPEED, new_vx)
        dx = jnp.where(is_in_air, state.player_vx, new_vx)
        
        player_mid_x = state.player_x + self.consts.PLAYER_WIDTH // 2
        player_feet_y = state.player_y + self.consts.PLAYER_HEIGHT - 1
        
        # 0. Ladder and Rope Climbing Logic
        def check_ladder(i, carry):
            c_on_ladder, c_ladder_idx = carry
            l_x = state.ladders_x[i]
            ladder_mid_x = l_x + 8
            l_top = state.ladders_top[i]
            l_bottom = state.ladders_bottom[i]
            
            is_aligned = jnp.logical_and(state.ladders_active[i] == 1, jnp.abs(player_mid_x - ladder_mid_x) <= 4)
            
            # To get ON from top: must press DOWN near top
            get_on_top = jnp.logical_and(is_aligned, jnp.logical_and(is_down, jnp.abs(player_feet_y - l_top) <= 5))
            # To get ON from bottom: must press UP near bottom
            get_on_bottom = jnp.logical_and(is_aligned, jnp.logical_and(is_up, jnp.abs(player_feet_y - l_bottom) <= 5))
            
            # To stay ON: must be within the vertical bounds
            in_ladder_zone = jnp.logical_and(is_aligned, jnp.logical_and(player_feet_y >= l_top - 3, player_feet_y <= l_bottom - 2))
            
            on_this_ladder = jnp.where(state.is_climbing == 1, in_ladder_zone, jnp.logical_or(get_on_top, get_on_bottom))
            
            new_on_ladder = jnp.logical_or(c_on_ladder, on_this_ladder)
            new_ladder_idx = jnp.where(on_this_ladder, i, c_ladder_idx)
            return new_on_ladder, new_ladder_idx

        def check_rope(i, carry):
            c_on_rope, c_rope_idx = carry
            r_x = state.ropes_x[i]
            r_top = state.ropes_top[i]
            r_bottom = state.ropes_bottom[i]
            
            is_aligned = jnp.logical_and(state.ropes_active[i] == 1, jnp.abs(player_mid_x - r_x) <= 4)
            
            player_top_y = state.player_y
            intersect_y = jnp.logical_and(player_feet_y >= r_top, player_top_y <= r_bottom)
            
            catch_rope = jnp.logical_and(
                state.is_climbing == 0,
                jnp.logical_and(is_aligned, jnp.logical_and(intersect_y, state.last_rope != i))
            )
            
            get_on_top = jnp.logical_and(is_aligned, jnp.logical_and(is_down, jnp.abs(player_feet_y - r_top) <= 5))
            get_on_bottom = jnp.logical_and(is_aligned, jnp.logical_and(is_up, jnp.abs(player_feet_y - r_bottom) <= 5))
            
            in_rope_zone = jnp.logical_and(is_aligned, jnp.logical_and(player_feet_y >= r_top - 3, player_feet_y <= r_bottom - 2))
            
            on_this_rope = jnp.where(state.is_climbing == 1, in_rope_zone, jnp.logical_or(catch_rope, jnp.logical_or(get_on_top, get_on_bottom)))
            
            new_on_rope = jnp.logical_or(c_on_rope, on_this_rope)
            new_rope_idx = jnp.where(on_this_rope, i, c_rope_idx)
            return new_on_rope, new_rope_idx

        can_ladder, ladder_idx = jax.lax.fori_loop(0, self.consts.MAX_LADDERS_PER_ROOM, check_ladder, (False, -1))
        can_rope, rope_idx = jax.lax.fori_loop(0, self.consts.MAX_ROPES_PER_ROOM, check_rope, (False, -1))
        
        abort_ladder = jnp.logical_or(is_left, is_right)
        is_jumping_off_rope = jnp.logical_and(can_rope, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(is_fire, jnp.logical_or(is_left, is_right))))
        abort_rope = is_jumping_off_rope
        
        is_climbing_ladder = jnp.logical_and(can_ladder, jnp.logical_not(abort_ladder))
        is_climbing_rope = jnp.logical_and(can_rope, jnp.logical_not(abort_rope))
        
        is_climbing = jnp.where(jnp.logical_or(is_climbing_ladder, is_climbing_rope), 1, 0)
        
        target_climb_x = state.player_x
        target_climb_x = jnp.where(ladder_idx != -1, state.ladders_x[ladder_idx] + 8 - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        target_climb_x = jnp.where(rope_idx != -1, state.ropes_x[rope_idx] - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        
        current_x = jnp.where(is_climbing == 1, target_climb_x, state.player_x)
        
        # 1. Check if strictly on ground
        safe_x = jnp.clip(current_x + self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)
        safe_y = jnp.clip(player_feet_y + 1, 0, 148)
        on_ground = self.ROOM_COLLISION_MAP[safe_y, safe_x] == 1
        
        def check_conveyor(i, on_grnd):
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            is_on_conveyor = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(player_feet_y == c_y, jnp.logical_and(player_mid_x >= c_x, player_mid_x < c_x + 40))
            )
            return jnp.logical_or(on_grnd, is_on_conveyor)
        
        on_ground = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, check_conveyor, on_ground)

        # Update last_rope
        new_last_rope = jnp.where(on_ground, -1, state.last_rope)
        new_last_rope = jnp.where(is_jumping_off_rope, rope_idx, new_last_rope)

        # 2. Process Jump Initiation
        start_jump_normal = jnp.logical_and(is_fire, jnp.logical_and(on_ground, jnp.logical_and(state.is_jumping == 0, is_climbing == 0)))
        start_jump = jnp.logical_or(start_jump_normal, is_jumping_off_rope)
        is_jumping = jnp.where(start_jump, 1, state.is_jumping)
        is_jumping = jnp.where(is_climbing == 1, 0, is_jumping) # cancel jump
        jump_counter = jnp.where(start_jump, 0, state.jump_counter)
        
        # Lock in horizontal velocity at the start of jump or fall
        current_vx = jnp.where(jnp.logical_or(start_jump, jnp.logical_and(jnp.logical_not(on_ground), state.is_falling == 0)), dx, state.player_vx)
        # Update dx with the locked velocity
        dx = jnp.where(jnp.logical_or(is_jumping == 1, jnp.logical_not(on_ground)), current_vx, dx)
        
        # 3. Calculate DY
        def get_jump_dy():
            dy_jump = -self.consts.JUMP_Y_OFFSETS[jump_counter]
            return dy_jump, jump_counter + 1, 1
            
        def get_fall_dy():
            pixel_1_below = self.ROOM_COLLISION_MAP[safe_y, safe_x] == 1
            pixel_2_below = self.ROOM_COLLISION_MAP[jnp.clip(player_feet_y + 2, 0, 148), safe_x] == 1
            fall_dist = jnp.where(on_ground, 0, jnp.where(pixel_2_below, 1, self.consts.GRAVITY))
            return fall_dist, 0, 0
            
        def get_climb_dy():
            climb_dist = jnp.where(is_down, self.consts.PLAYER_SPEED, jnp.where(is_up, -self.consts.PLAYER_SPEED, 0))
            return climb_dist, 0, 0

        dy, new_jump_counter, new_is_jumping = jax.lax.cond(
            is_climbing == 1,
            get_climb_dy,
            lambda: jax.lax.cond(
                is_jumping == 1,
                get_jump_dy,
                get_fall_dy
            )
        )
        
        new_is_jumping = jnp.where(new_jump_counter >= 20, 0, new_is_jumping)
        
        # 4. Resolve Vertical Collision
        new_y = state.player_y + dy
        new_feet_y = new_y + self.consts.PLAYER_HEIGHT - 1
        
        new_top_y = jnp.clip(new_y, 0, 148)
        hit_ceiling = jnp.logical_and(dy < 0, jnp.logical_and(self.ROOM_COLLISION_MAP[new_top_y, safe_x] == 1, is_climbing == 0))
        new_y = jnp.where(hit_ceiling, state.player_y, new_y)
        new_is_jumping = jnp.where(hit_ceiling, 0, new_is_jumping)
        
        hit_floor = jnp.logical_and(dy > 0, jnp.logical_and(is_climbing == 0, self.ROOM_COLLISION_MAP[jnp.clip(new_feet_y, 0, 148), safe_x] == 1))
        snapped_y = jnp.clip(new_feet_y, 0, 148) - self.consts.PLAYER_HEIGHT
        new_y = jnp.where(hit_floor, snapped_y, new_y)
        
        # Set is_falling state
        new_is_falling = jnp.where(jnp.logical_and(new_is_jumping == 0, hit_floor == False), jnp.where(dy > 0, 1, 0), 0)
        new_is_falling = jnp.where(is_climbing == 1, 0, new_is_falling)
        
        # 5. Resolve Horizontal with Wall Collision
        new_x = jnp.clip(current_x + dx, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        
        new_left_x = jnp.clip(new_x, 0, self.consts.WIDTH - 1)
        new_right_x = jnp.clip(new_x + self.consts.PLAYER_WIDTH - 1, 0, self.consts.WIDTH - 1)
        
        front_x = jnp.where(dx > 0, new_right_x, new_left_x)
        
        check_y_top = jnp.clip(new_y, 0, 148)
        check_y_mid = jnp.clip(new_y + self.consts.PLAYER_HEIGHT // 2, 0, 148)
        check_y_bot = jnp.clip(new_y + self.consts.PLAYER_HEIGHT - 1, 0, 148)
        
        hit_wall = jnp.logical_or(
            self.ROOM_COLLISION_MAP[check_y_top, front_x] == 1,
            jnp.logical_or(
                self.ROOM_COLLISION_MAP[check_y_mid, front_x] == 1,
                self.ROOM_COLLISION_MAP[check_y_bot, front_x] == 1
            )
        )
        
        def check_door(i, hit):
            d_x = state.doors_x[i]
            d_y = state.doors_y[i]
            d_active = state.doors_active[i] == 1
            in_x = jnp.logical_and(front_x >= d_x, front_x < d_x + 4)
            in_y_top = jnp.logical_and(check_y_top >= d_y, check_y_top < d_y + 38)
            in_y_mid = jnp.logical_and(check_y_mid >= d_y, check_y_mid < d_y + 38)
            in_y_bot = jnp.logical_and(check_y_bot >= d_y, check_y_bot < d_y + 38)
            in_y = jnp.logical_or(in_y_top, jnp.logical_or(in_y_mid, in_y_bot))
            hit_this_door = jnp.logical_and(d_active, jnp.logical_and(in_x, in_y))
            return jnp.logical_or(hit, hit_this_door)
        
        hit_wall = jax.lax.fori_loop(0, self.consts.MAX_DOORS_PER_ROOM, check_door, hit_wall)

        new_x = jnp.where(jnp.logical_or(hit_wall, is_climbing == 1), current_x, new_x)
        
        new_mid_x = new_x + self.consts.PLAYER_WIDTH // 2
        new_feet_y_after = new_y + self.consts.PLAYER_HEIGHT - 1
        
        def apply_conveyor_physics(i, p_x):
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            is_on_conveyor = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(new_feet_y_after == c_y, jnp.logical_and(new_mid_x >= c_x, new_mid_x < c_x + 40))
            )
            conveyor_velocity = jnp.mod(state.frame_count, 2) * state.conveyors_direction[i]
            return jax.lax.select(jnp.logical_and(is_on_conveyor, is_climbing == 0), p_x + conveyor_velocity, p_x)

        new_x = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, apply_conveyor_physics, new_x)
        new_x = jnp.clip(new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        
        current_vx = jnp.where(jnp.logical_or(hit_wall, hit_floor), 0, current_vx)
        
        state = state.replace(
            player_x=new_x,
            player_y=new_y,
            player_vx=current_vx,
            is_jumping=new_is_jumping,
            jump_counter=new_jump_counter,
            is_climbing=is_climbing,
            last_rope=new_last_rope,
            is_falling=new_is_falling,
            frame_count=state.frame_count + 1
        )
        
        obs = self._get_observation(state)
        reward = self._get_reward(state, state)
        done = self._get_done(state)
        info = self._get_info(state)
        
        return obs, state, reward, done, info

    def action_space(self) -> Discrete:
        return Discrete(len(self.ACTION_SET))

    def observation_space(self) -> Discrete:
        return Discrete(1)
        
    def image_space(self) -> Discrete:
        return Discrete(1)

    def render(self, state: Montezuma2State) -> jnp.ndarray:
        return self.renderer.render(state)

    def _get_observation(self, state: Montezuma2State) -> Montezuma2Observation:
        player_obs = ObjectObservation.create(
            x=jnp.array([state.player_x]),
            y=jnp.array([state.player_y]),
            width=jnp.array([self.consts.PLAYER_WIDTH]),
            height=jnp.array([self.consts.PLAYER_HEIGHT]),
            active=jnp.array([1])
        )
        
        enemies_obs = ObjectObservation.create(
            x=state.enemies_x,
            y=state.enemies_y,
            width=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 8),
            height=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 16),
            active=state.enemies_active
        )
        
        items_obs = ObjectObservation.create(
            x=state.items_x,
            y=state.items_y,
            width=jnp.full(self.consts.MAX_ITEMS_PER_ROOM, 6),
            height=jnp.full(self.consts.MAX_ITEMS_PER_ROOM, 8),
            active=state.items_active
        )
        
        conveyors_obs = ObjectObservation.create(
            x=state.conveyors_x,
            y=state.conveyors_y,
            width=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 40),
            height=jnp.full(self.consts.MAX_CONVEYORS_PER_ROOM, 5),
            active=state.conveyors_active
        )

        doors_obs = ObjectObservation.create(
            x=state.doors_x,
            y=state.doors_y,
            width=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 4),
            height=jnp.full(self.consts.MAX_DOORS_PER_ROOM, 38),
            active=state.doors_active
        )

        ropes_obs = ObjectObservation.create(
            x=state.ropes_x,
            y=state.ropes_top,
            width=jnp.full(self.consts.MAX_ROPES_PER_ROOM, 1),
            height=state.ropes_bottom - state.ropes_top,
            active=state.ropes_active
        )

        return Montezuma2Observation(player=player_obs, enemies=enemies_obs, items=items_obs, conveyors=conveyors_obs, doors=doors_obs, ropes=ropes_obs)
    def _get_info(self, state: Montezuma2State, all_rewards: jnp.ndarray = None) -> Montezuma2Info:
        return Montezuma2Info(lives=state.lives, room_id=state.room_id)

    def _get_reward(self, previous_state: Montezuma2State, state: Montezuma2State) -> float:
        return 0.0

    def _get_done(self, state: Montezuma2State) -> bool:
        return state.lives < 0
