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

    # HUD Constants
    SCORE_X: int = struct.field(pytree_node=False, default=56)
    SCORE_Y: int = struct.field(pytree_node=False, default=6)
    LIFES_STARTING_Y: int = struct.field(pytree_node=False, default=15)
    ITEMBAR_STARTING_Y: int = struct.field(pytree_node=False, default=28)
    ITEMBAR_LIFES_STARTING_X: int = struct.field(pytree_node=False, default=56)
    DIGIT_WIDTH: int = struct.field(pytree_node=False, default=7)
    DIGIT_OFFSET: int = struct.field(pytree_node=False, default=1)
    DIGIT_HEIGHT: int = struct.field(pytree_node=False, default=8)
    
    # Gameplay Rules
    MAX_FALL_DISTANCE: int = struct.field(pytree_node=False, default=33) # ladder_height (39) - 6

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
    player_dir: jnp.ndarray
    
    is_jumping: jnp.ndarray
    is_falling: jnp.ndarray
    fall_start_y: jnp.ndarray
    jump_counter: jnp.ndarray
    is_climbing: jnp.ndarray
    last_rope: jnp.ndarray
    
    # Homogeneous Entities for the CURRENT room
    enemies_x: jnp.ndarray
    enemies_y: jnp.ndarray
    enemies_active: jnp.ndarray
    enemies_direction: jnp.ndarray
    enemies_min_x: jnp.ndarray
    enemies_max_x: jnp.ndarray
    
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
    
    inventory: jnp.ndarray
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
            {'name': 'room_bg_0', 'type': 'single', 'file': 'backgrounds/base_sprite_level_0.npy', 'transpose': False},
            {'name': 'room_bg_1', 'type': 'single', 'file': 'backgrounds/mid_room_level_0.npy', 'transpose': False},
            {
                'name': 'player', 'type': 'group',
                'files': [
                    'player/player_sprite.npy',
                    'player/walking_0.npy',
                    'player/walking_1.npy',
                    'player/ladder_climb1.npy',
                    'player/ladder_climb2.npy',
                    'player/rope_climb_0.npy',
                    'player/rope_climb_1.npy',
                    'player/player_jump.npy'
                ]
            },
            {
                'name': 'skull', 'type': 'group',
                'files': [f'enemies/skull_cycle/skull_{i}.npy' for i in range(1, 17)]
            },
            {'name': 'item', 'type': 'single', 'file': 'items/key.npy', 'transpose': False},
            {'name': 'door', 'type': 'single', 'file': 'door.npy', 'transpose': False},
            {'name': 'conveyor', 'type': 'single', 'file': 'conveyor_belt.npy', 'transpose': False},
            {'name': 'life', 'type': 'single', 'file': 'life_sprite.npy', 'transpose': False},
            {'name': 'digit_0', 'type': 'single', 'file': 'digits/digit_0.npy', 'transpose': False},
            {'name': 'digit_1', 'type': 'single', 'file': 'digits/digit_1.npy', 'transpose': False},
            {'name': 'digit_2', 'type': 'single', 'file': 'digits/digit_2.npy', 'transpose': False},
            {'name': 'digit_3', 'type': 'single', 'file': 'digits/digit_3.npy', 'transpose': False},
            {'name': 'digit_4', 'type': 'single', 'file': 'digits/digit_4.npy', 'transpose': False},
            {'name': 'digit_5', 'type': 'single', 'file': 'digits/digit_5.npy', 'transpose': False},
            {'name': 'digit_6', 'type': 'single', 'file': 'digits/digit_6.npy', 'transpose': False},
            {'name': 'digit_7', 'type': 'single', 'file': 'digits/digit_7.npy', 'transpose': False},
            {'name': 'digit_8', 'type': 'single', 'file': 'digits/digit_8.npy', 'transpose': False},
            {'name': 'digit_9', 'type': 'single', 'file': 'digits/digit_9.npy', 'transpose': False},
            {'name': 'digit_none', 'type': 'single', 'file': 'digits/digit_none.npy', 'transpose': False},
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

        self.digit_masks = jnp.stack([
            self.SHAPE_MASKS["digit_none"],
            self.SHAPE_MASKS["digit_0"],
            self.SHAPE_MASKS["digit_1"],
            self.SHAPE_MASKS["digit_2"],
            self.SHAPE_MASKS["digit_3"],
            self.SHAPE_MASKS["digit_4"],
            self.SHAPE_MASKS["digit_5"],
            self.SHAPE_MASKS["digit_6"],
            self.SHAPE_MASKS["digit_7"],
            self.SHAPE_MASKS["digit_8"],
            self.SHAPE_MASKS["digit_9"],
        ])

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: Montezuma2State) -> jnp.ndarray:
        # Start with solid black background
        raster = self.jr.create_object_raster(self.BACKGROUND)
        
        # Draw Room Background
        mask_0 = self.SHAPE_MASKS["room_bg_0"][:149, :]
        mask_1 = self.SHAPE_MASKS["room_bg_1"][:149, :]
        room_bg_mask = jnp.where(state.room_id == 0, mask_0, mask_1)
        raster = self.jr.render_at(raster, 0, 47, room_bg_mask)
        
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
                lambda r: self.jr.render_at(r, state.conveyors_x[i], state.conveyors_y[i] + 47, mask, flip_vertical=anim_idx),
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
            anim_idx = jnp.mod(state.enemies_x[i], 16)
            mask = self.SHAPE_MASKS["skull"][anim_idx]
            return jax.lax.cond(
                state.enemies_active[i] == 1,
                lambda r: self.jr.render_at(r, state.enemies_x[i], state.enemies_y[i] + 47, mask),
                lambda r: r,
                raster
            )
        raster = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES_PER_ROOM, render_enemy, raster)
        
        # Draw Player
        is_walking = jnp.logical_and(state.player_vx != 0, jnp.logical_and(state.is_climbing == 0, jnp.logical_and(state.is_jumping == 0, state.is_falling == 0)))
        is_laddering = jnp.logical_and(state.is_climbing == 1, state.last_rope == -1)
        is_roping = jnp.logical_and(state.is_climbing == 1, state.last_rope != -1)
        is_in_air = jnp.logical_or(state.is_jumping == 1, state.is_falling == 1)

        walk_anim = jnp.mod(jnp.floor_divide(state.frame_count, 4), 2)
        ladder_anim = jnp.mod(jnp.floor_divide(state.player_y, 4), 2)
        rope_anim = jnp.mod(jnp.floor_divide(state.player_y, 4), 2)

        player_sprite_idx = jnp.array(0)
        player_sprite_idx = jnp.where(is_walking, 1 + walk_anim, player_sprite_idx)
        player_sprite_idx = jnp.where(is_laddering, 3 + ladder_anim, player_sprite_idx)
        player_sprite_idx = jnp.where(is_roping, 5 + rope_anim, player_sprite_idx)
        player_sprite_idx = jnp.where(is_in_air, 7, player_sprite_idx)
        
        # Standing sprite (0) faces right, flip if facing left.
        # Walking (1, 2) and jumping (7) sprites face left natively, flip if facing right.
        flip_player = jax.lax.select(
            jnp.logical_or(jnp.logical_or(player_sprite_idx == 1, player_sprite_idx == 2), player_sprite_idx == 7),
            state.player_dir == 1,
            jax.lax.select(
                player_sprite_idx == 0,
                state.player_dir == -1,
                False
            )
        )
        
        player_mask = self.SHAPE_MASKS["player"][player_sprite_idx]
        raster = self.jr.render_at(raster, state.player_x, state.player_y + 47, player_mask, flip_horizontal=flip_player)

        # Render Score
        score = state.score

        k_100_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([100000])), 10) + 1
        k_10_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([10000])), 10) + 1
        thousands_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([1000])), 10) + 1
        hundreds_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([100])), 10) + 1
        tens_sprite_index = jnp.mod(jnp.floor_divide(score, jnp.array([10])), 10) + 1
        ones_sprite_index = jnp.add(jnp.mod(score, jnp.array([10])), 1)

        # Remove leading zeroes
        leading_zeros = jnp.array([
            k_100_sprite_index == 1,
            k_10_sprite_index == 1,
            thousands_sprite_index == 1,
            hundreds_sprite_index == 1,
            tens_sprite_index == 1
        ])
        mask = jnp.cumprod(leading_zeros)

        k_100_sprite_index = jnp.where(mask[0], 0, k_100_sprite_index)[0]
        k_10_sprite_index = jnp.where(mask[1], 0, k_10_sprite_index)[0]
        thousands_sprite_index = jnp.where(mask[2], 0, thousands_sprite_index)[0]
        hundreds_sprite_index = jnp.where(mask[3], 0, hundreds_sprite_index)[0]
        tens_sprite_index = jnp.where(mask[4], 0, tens_sprite_index)[0]
        ones_sprite_index = ones_sprite_index[0]

        def render_digit(raster, index, digit_idx):
            mask = self.digit_masks[digit_idx]
            x = self.consts.SCORE_X + index * (self.consts.DIGIT_WIDTH + self.consts.DIGIT_OFFSET)
            y = self.consts.SCORE_Y
            return self.jr.render_at(raster, x, y, mask)

        raster = render_digit(raster, 0, k_100_sprite_index)
        raster = render_digit(raster, 1, k_10_sprite_index)
        raster = render_digit(raster, 2, thousands_sprite_index)
        raster = render_digit(raster, 3, hundreds_sprite_index)
        raster = render_digit(raster, 4, tens_sprite_index)
        raster = render_digit(raster, 5, ones_sprite_index)

        # Render Lives
        def render_life(i, raster):
            mask = self.SHAPE_MASKS["life"]
            x = self.consts.ITEMBAR_LIFES_STARTING_X + i * (mask.shape[1] + 1)
            y = self.consts.LIFES_STARTING_Y
            return self.jr.render_at(raster, x, y, mask)

        raster = jax.lax.fori_loop(0, state.lives, render_life, raster)

        # Render Inventory (Keys)
        def render_inventory(i, raster):
            mask = self.SHAPE_MASKS["item"]
            x = self.consts.ITEMBAR_LIFES_STARTING_X + i * 8
            y = self.consts.ITEMBAR_STARTING_Y
            return self.jr.render_at(raster, x, y, mask)
            
        raster = jax.lax.fori_loop(0, state.inventory[0], render_inventory, raster)

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
        
        sprite_path_0 = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma", "backgrounds", "base_collision_map.npy")
        col_map_0 = jnp.load(sprite_path_0)[:149, :, 0]
        room_col_0 = jnp.where(col_map_0 > 0, 1, 0).astype(jnp.int32)
        room_col_0 = room_col_0.at[6:, 0:4].set(1) # Left wall only for room_0_0

        sprite_path_1 = os.path.join(self.consts.MODULE_DIR, "sprites", "montezuma", "backgrounds", "mid_room_collision_level_0.npy")
        col_map_1 = jnp.load(sprite_path_1)[:149, :, 0] # (149, 160)
        room_col_1 = jnp.where(col_map_1 > 0, 1, 0).astype(jnp.int32)
        # No side walls for room_0_1 as it's connected on both sides
        
        self.ROOM_COLLISION_MAPS = jnp.stack([room_col_0, room_col_1])

    def _load_room(self, room_id: jnp.ndarray, state: Montezuma2State) -> Montezuma2State:
        enemies_x = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_y = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_active = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_direction = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_min_x = jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32)
        enemies_max_x = jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, self.consts.WIDTH - 8, dtype=jnp.int32)
        
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

        def load_room_0(args):
            lx, lt, lb, la, ix, iy, ia = args
            lx = lx.at[0].set(72)
            lt = lt.at[0].set(48)
            lb = lb.at[0].set(149)
            la = la.at[0].set(1)
            
            ix = ix.at[0].set(24)
            iy = iy.at[0].set(7)
            ia = ia.at[0].set(1)
            
            return (enemies_x, enemies_y, enemies_active, enemies_direction, enemies_min_x, enemies_max_x,
                    lx, lt, lb, la,
                    ropes_x, ropes_top, ropes_bottom, ropes_active,
                    ix, iy, ia,
                    doors_x, doors_y, doors_active,
                    conveyors_x, conveyors_y, conveyors_active, conveyors_direction)

        def load_room_1(args):
            lx, lt, lb, la, ix, iy, ia = args
            ex = enemies_x.at[0].set(93)
            ey = enemies_y.at[0].set(119)
            ea = enemies_active.at[0].set(1)
            ed = enemies_direction.at[0].set(1)
            eminx = enemies_min_x.at[0].set(45)
            emaxx = enemies_max_x.at[0].set(110)
            
            lx = lx.at[0].set(72)
            lt = lt.at[0].set(49)
            lb = lb.at[0].set(88)
            la = la.at[0].set(1)
            lx = lx.at[1].set(128)
            lt = lt.at[1].set(92)
            lb = lb.at[1].set(133)
            la = la.at[1].set(1)
            lx = lx.at[2].set(16)
            lt = lt.at[2].set(92)
            lb = lb.at[2].set(133)
            la = la.at[2].set(1)

            rx = ropes_x.at[0].set(111)
            rt = ropes_top.at[0].set(49)
            rb = ropes_bottom.at[0].set(88)
            ra = ropes_active.at[0].set(1)

            ix = ix.at[0].set(13)
            iy = iy.at[0].set(52)
            ia = ia.at[0].set(1)

            cx = conveyors_x.at[0].set(60)
            cy = conveyors_y.at[0].set(88)
            ca = conveyors_active.at[0].set(1)
            cd = conveyors_direction.at[0].set(1)
            
            dx = doors_x.at[0].set(16)
            dy = doors_y.at[0].set(7)
            da = doors_active.at[0].set(1)
            dx = dx.at[1].set(140)
            dy = dy.at[1].set(7)
            da = da.at[1].set(1)

            return (ex, ey, ea, ed, eminx, emaxx,
                    lx, lt, lb, la,
                    rx, rt, rb, ra,
                    ix, iy, ia,
                    dx, dy, da,
                    cx, cy, ca, cd)

        args = (ladders_x, ladders_top, ladders_bottom, ladders_active, items_x, items_y, items_active)
        ex, ey, ea, ed, eminx, emaxx, lx, lt, lb, la, rx, rt, rb, ra, ix, iy, ia, dx, dy, da, cx, cy, ca, cd = jax.lax.switch(room_id, [load_room_0, load_room_1], args)

        return state.replace(
            room_id=room_id,
            enemies_x=ex, enemies_y=ey, enemies_active=ea, enemies_direction=ed, enemies_min_x=eminx, enemies_max_x=emaxx,
            ladders_x=lx, ladders_top=lt, ladders_bottom=lb, ladders_active=la,
            ropes_x=rx, ropes_top=rt, ropes_bottom=rb, ropes_active=ra,
            items_x=ix, items_y=iy, items_active=ia,
            doors_x=dx, doors_y=dy, doors_active=da,
            conveyors_x=cx, conveyors_y=cy, conveyors_active=ca, conveyors_direction=cd
        )

    def reset(self, key: jrandom.PRNGKey) -> Tuple[Montezuma2Observation, Montezuma2State]:
        state = Montezuma2State(
            room_id=jnp.array(1, dtype=jnp.int32),
            lives=jnp.array(5, dtype=jnp.int32),
            score=jnp.array([0], dtype=jnp.int32),
            frame_count=jnp.array(0, dtype=jnp.int32),
            player_x=jnp.array(self.consts.INITIAL_PLAYER_X, dtype=jnp.int32),
            player_y=jnp.array(self.consts.INITIAL_PLAYER_Y, dtype=jnp.int32),
            player_vx=jnp.array(0, dtype=jnp.int32),
            player_vy=jnp.array(0, dtype=jnp.int32),
            player_dir=jnp.array(1, dtype=jnp.int32),
            is_jumping=jnp.array(0, dtype=jnp.int32),
            is_falling=jnp.array(0, dtype=jnp.int32),
            fall_start_y=jnp.array(self.consts.INITIAL_PLAYER_Y, dtype=jnp.int32),
            jump_counter=jnp.array(0, dtype=jnp.int32),
            is_climbing=jnp.array(0, dtype=jnp.int32),
            last_rope=jnp.array(-1, dtype=jnp.int32),
            enemies_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_y=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_active=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_direction=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_min_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            enemies_max_x=jnp.zeros(self.consts.MAX_ENEMIES_PER_ROOM, dtype=jnp.int32),
            ladders_x=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_top=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_bottom=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ladders_active=jnp.zeros(self.consts.MAX_LADDERS_PER_ROOM, dtype=jnp.int32),
            ropes_x=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_top=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_bottom=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            ropes_active=jnp.zeros(self.consts.MAX_ROPES_PER_ROOM, dtype=jnp.int32),
            items_x=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_y=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            items_active=jnp.zeros(self.consts.MAX_ITEMS_PER_ROOM, dtype=jnp.int32),
            doors_x=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            doors_y=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            doors_active=jnp.zeros(self.consts.MAX_DOORS_PER_ROOM, dtype=jnp.int32),
            conveyors_x=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_y=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_active=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            conveyors_direction=jnp.zeros(self.consts.MAX_CONVEYORS_PER_ROOM, dtype=jnp.int32),
            inventory=jnp.zeros(1, dtype=jnp.int32),
            key=key
        )
        state = self._load_room(jnp.array(1, dtype=jnp.int32), state)
        obs = self._get_observation(state)
        return obs, state
    def step(self, state: Montezuma2State, action: int) -> Tuple[Montezuma2Observation, Montezuma2State, float, bool, Montezuma2Info]:
        room_col_map = self.ROOM_COLLISION_MAPS[state.room_id]
        previous_score = state.score
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
        new_player_dir = jnp.where(is_right, 1, jnp.where(is_left, -1, state.player_dir))

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
            
            in_rope_zone = jnp.logical_and(is_aligned, jnp.logical_and(player_feet_y >= r_top, player_feet_y <= r_bottom + 10))
            
            on_this_rope = jnp.where(state.is_climbing == 1, in_rope_zone, jnp.logical_or(catch_rope, jnp.logical_or(get_on_top, get_on_bottom)))
            
            new_on_rope = jnp.logical_or(c_on_rope, on_this_rope)
            new_rope_idx = jnp.where(on_this_rope, i, c_rope_idx)
            return new_on_rope, new_rope_idx

        can_ladder, ladder_idx = jax.lax.fori_loop(0, self.consts.MAX_LADDERS_PER_ROOM, check_ladder, (False, -1))
        can_rope, rope_idx = jax.lax.fori_loop(0, self.consts.MAX_ROPES_PER_ROOM, check_rope, (False, -1))
        
        abort_ladder = False
        is_jumping_off_rope = jnp.logical_and(can_rope, jnp.logical_and(state.is_climbing == 1, jnp.logical_and(is_fire, jnp.logical_or(is_left, is_right))))
        abort_rope = is_jumping_off_rope
        
        is_climbing_ladder = jnp.logical_and(can_ladder, jnp.logical_not(abort_ladder))
        is_climbing_rope = jnp.logical_and(can_rope, jnp.logical_not(abort_rope))
        
        is_climbing = jnp.where(jnp.logical_or(is_climbing_ladder, is_climbing_rope), 1, 0)
        
        target_climb_x = state.player_x
        target_climb_x = jnp.where(ladder_idx != -1, state.ladders_x[ladder_idx] + 8 - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        target_climb_x = jnp.where(rope_idx != -1, state.ropes_x[rope_idx] - self.consts.PLAYER_WIDTH // 2, target_climb_x)
        
        current_x = jnp.where(is_climbing == 1, target_climb_x, state.player_x)
        
        def check_platform(y, x):
            x_m3 = jnp.clip(x - 3, 0, self.consts.WIDTH - 1)
            x_m2 = jnp.clip(x - 2, 0, self.consts.WIDTH - 1)
            x_m1 = jnp.clip(x - 1, 0, self.consts.WIDTH - 1)
            x_p1 = jnp.clip(x + 1, 0, self.consts.WIDTH - 1)
            x_p2 = jnp.clip(x + 2, 0, self.consts.WIDTH - 1)
            x_p3 = jnp.clip(x + 3, 0, self.consts.WIDTH - 1)
            return jnp.logical_or(
                room_col_map[y, x_m3] == 1,
                jnp.logical_or(
                    room_col_map[y, x_m2] == 1,
                    jnp.logical_or(
                        room_col_map[y, x_m1] == 1,
                        jnp.logical_or(
                            room_col_map[y, x] == 1,
                            jnp.logical_or(
                                room_col_map[y, x_p1] == 1,
                                jnp.logical_or(
                                    room_col_map[y, x_p2] == 1,
                                    room_col_map[y, x_p3] == 1
                                )
                            )
                        )
                    )
                )
            )
        
        # 1. Check if strictly on ground
        safe_x = jnp.clip(current_x + self.consts.PLAYER_WIDTH // 2, 0, self.consts.WIDTH - 1)
        safe_y = jnp.clip(player_feet_y + 1, 0, 148)
        on_ground = check_platform(safe_y, safe_x)
        
        def check_conveyor(i, on_grnd):
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            is_on_conveyor = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(player_feet_y == c_y, jnp.logical_and(player_mid_x >= c_x - 3, player_mid_x < c_x + 43))
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

        # Horizontal velocity to carry over (used for momentum and animation tracking)
        current_vx = dx

        # 3. Calculate DY
        def get_jump_dy():
            dy_jump = -self.consts.JUMP_Y_OFFSETS[jump_counter]
            return dy_jump, jump_counter + 1, 1
            
        def get_fall_dy():
            pixel_1_below = check_platform(safe_y, safe_x)
            pixel_2_below = check_platform(jnp.clip(player_feet_y + 2, 0, 148), safe_x)
            
            def check_c_pixel2(i, p2b):
                c_x = state.conveyors_x[i]
                c_y = state.conveyors_y[i] - 1
                is_on = jnp.logical_and(
                    state.conveyors_active[i] == 1,
                    jnp.logical_and(player_feet_y + 1 == c_y, jnp.logical_and(safe_x >= c_x - 3, safe_x < c_x + 43))
                )
                return jnp.logical_or(p2b, is_on)
            pixel_2_below = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, check_c_pixel2, pixel_2_below)

            fall_dist = jnp.where(on_ground, 0, jnp.where(pixel_2_below, 1, self.consts.GRAVITY))
            return fall_dist, 0, 0
            
        def get_climb_dy():
            climb_dist = jnp.where(is_down, self.consts.PLAYER_SPEED, jnp.where(is_up, -self.consts.PLAYER_SPEED, 0))
            # Zero out vertical speed on the frame we catch the rope
            just_caught_rope = jnp.logical_and(state.is_climbing == 0, rope_idx != -1)
            climb_dist = jnp.where(just_caught_rope, 0, climb_dist)
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
        new_y = jnp.where(jnp.logical_and(is_climbing == 1, rope_idx != -1), jnp.maximum(new_y, state.ropes_top[rope_idx]), new_y)
        new_feet_y = new_y + self.consts.PLAYER_HEIGHT - 1
        
        new_top_y = jnp.clip(new_y, 0, 148)
        hit_ceiling = jnp.logical_and(dy < 0, jnp.logical_and(check_platform(new_top_y, safe_x), is_climbing == 0))
        new_y = jnp.where(hit_ceiling, state.player_y, new_y)
        new_is_jumping = jnp.where(hit_ceiling, 0, new_is_jumping)
        
        hit_floor_rm = jnp.logical_and(dy > 0, jnp.logical_and(is_climbing == 0, check_platform(jnp.clip(new_feet_y, 0, 148), safe_x)))
        snapped_y_rm = jnp.clip(new_feet_y, 0, 148) - self.consts.PLAYER_HEIGHT
        
        def check_c_hit_floor(i, carry):
            h_f, s_y = carry
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            crossed = jnp.logical_and(player_feet_y < c_y, new_feet_y >= c_y)
            is_hit = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(dy > 0, jnp.logical_and(is_climbing == 0, jnp.logical_and(crossed, jnp.logical_and(safe_x >= c_x - 3, safe_x < c_x + 43))))
            )
            return jnp.logical_or(h_f, is_hit), jnp.where(is_hit, c_y - self.consts.PLAYER_HEIGHT + 1, s_y)
            
        hit_floor, snapped_y = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, check_c_hit_floor, (hit_floor_rm, snapped_y_rm))
        
        new_y = jnp.where(hit_floor, snapped_y, new_y)
        
        # Set is_falling state
        new_is_falling = jnp.where(jnp.logical_and(new_is_jumping == 0, hit_floor == False), jnp.where(dy > 0, 1, 0), 0)
        new_is_falling = jnp.where(is_climbing == 1, 0, new_is_falling)
        
        # 5. Resolve Horizontal with Wall Collision
        raw_new_x = current_x + dx
        transition_left = jnp.logical_and(raw_new_x < 0, state.room_id == 1)
        transition_right = jnp.logical_and(raw_new_x + self.consts.PLAYER_WIDTH > self.consts.WIDTH, state.room_id == 0)

        new_x = jnp.clip(raw_new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        new_left_x = jnp.clip(new_x, 0, self.consts.WIDTH - 1)
        new_right_x = jnp.clip(new_x + self.consts.PLAYER_WIDTH - 1, 0, self.consts.WIDTH - 1)
        
        front_x = jnp.where(dx > 0, new_right_x, new_left_x)
        
        check_y_top = jnp.clip(new_y, 0, 148)
        check_y_mid = jnp.clip(new_y + self.consts.PLAYER_HEIGHT // 2, 0, 148)
        check_y_bot = jnp.clip(new_y + self.consts.PLAYER_HEIGHT - 1, 0, 148)
        
        hit_wall = jnp.logical_or(
            room_col_map[check_y_top, front_x] == 1,
            jnp.logical_or(
                room_col_map[check_y_mid, front_x] == 1,
                room_col_map[check_y_bot, front_x] == 1
            )
        )
        
        # 5.5 Item Collection
        def collect_item(i, carry):
            keys_collected, items_active, current_score = carry
            i_x = state.items_x[i]
            i_y = state.items_y[i]
            i_active = items_active[i] == 1
            
            overlap_x = jnp.logical_and(new_left_x < i_x + 6, new_right_x >= i_x)
            overlap_y = jnp.logical_and(check_y_top < i_y + 8, check_y_bot >= i_y)
            overlap = jnp.logical_and(overlap_x, overlap_y)
            
            collect = jnp.logical_and(i_active, overlap)
            
            new_keys = jnp.where(collect, keys_collected + 1, keys_collected)
            new_items_active = jnp.where(collect, items_active.at[i].set(0), items_active)
            new_score = jnp.where(collect, current_score + 100, current_score)
            
            return new_keys, new_items_active, new_score

        current_keys, new_items_active, new_score = jax.lax.fori_loop(
            0, self.consts.MAX_ITEMS_PER_ROOM, collect_item,
            (state.inventory[0], state.items_active, state.score)
        )

        def check_door(i, carry):
            hit, keys_left, doors_active = carry
            d_x = state.doors_x[i]
            d_y = state.doors_y[i]
            d_active = doors_active[i] == 1
            in_x = jnp.logical_and(front_x >= d_x, front_x < d_x + 4)
            in_y_top = jnp.logical_and(check_y_top >= d_y, check_y_top < d_y + 38)
            in_y_mid = jnp.logical_and(check_y_mid >= d_y, check_y_mid < d_y + 38)
            in_y_bot = jnp.logical_and(check_y_bot >= d_y, check_y_bot < d_y + 38)
            in_y = jnp.logical_or(in_y_top, jnp.logical_or(in_y_mid, in_y_bot))
            
            hit_this_door = jnp.logical_and(d_active, jnp.logical_and(in_x, in_y))
            open_it = jnp.logical_and(hit_this_door, keys_left > 0)
            hit_as_wall = jnp.logical_and(hit_this_door, jnp.logical_not(open_it))
            
            new_hit = jnp.logical_or(hit, hit_as_wall)
            new_keys = jnp.where(open_it, keys_left - 1, keys_left)
            new_doors_active = jnp.where(open_it, doors_active.at[i].set(0), doors_active)
            
            return new_hit, new_keys, new_doors_active
        
        hit_wall, current_keys, new_doors_active = jax.lax.fori_loop(
            0, self.consts.MAX_DOORS_PER_ROOM, check_door, 
            (hit_wall, current_keys, state.doors_active)
        )

        new_x = jnp.where(jnp.logical_or(hit_wall, is_climbing == 1), current_x, new_x)
        
        new_mid_x = new_x + self.consts.PLAYER_WIDTH // 2
        new_feet_y_after = new_y + self.consts.PLAYER_HEIGHT - 1
        
        def apply_conveyor_physics(i, p_x):
            c_x = state.conveyors_x[i]
            c_y = state.conveyors_y[i] - 1
            is_on_conveyor = jnp.logical_and(
                state.conveyors_active[i] == 1,
                jnp.logical_and(new_feet_y_after == c_y, jnp.logical_and(new_mid_x >= c_x - 3, new_mid_x < c_x + 43))
            )
            conveyor_velocity = jnp.mod(state.frame_count, 2) * state.conveyors_direction[i]
            return jax.lax.select(jnp.logical_and(is_on_conveyor, is_climbing == 0), p_x + conveyor_velocity, p_x)

        new_x = jax.lax.fori_loop(0, self.consts.MAX_CONVEYORS_PER_ROOM, apply_conveyor_physics, new_x)
        new_x = jnp.clip(new_x, 0, self.consts.WIDTH - self.consts.PLAYER_WIDTH)
        
        current_vx = jnp.where(jnp.logical_or(hit_wall, hit_floor), 0, current_vx)
        
        # 6. Enemy Movement
        def move_enemy(i, carry):
            e_x, e_dir = carry
            current_x = e_x[i]
            current_dir = e_dir[i]
            
            # Move 1 pixel every 2 frames
            speed = jnp.where(jnp.mod(state.frame_count, 2) == 0, 1, 0)
            new_x = current_x + current_dir * speed
            
            # Bounce off walls
            e_y = state.enemies_y[i]
            hit_wall_left = jnp.logical_and(current_dir < 0, room_col_map[e_y + 8, jnp.clip(new_x, 0, self.consts.WIDTH - 1)] == 1)
            hit_wall_right = jnp.logical_and(current_dir > 0, room_col_map[e_y + 8, jnp.clip(new_x + 8, 0, self.consts.WIDTH - 1)] == 1)
            
            # Or boundaries
            hit_left = new_x <= state.enemies_min_x[i]
            hit_right = new_x >= state.enemies_max_x[i]
            
            bounce = jnp.logical_or(hit_left, jnp.logical_or(hit_right, jnp.logical_or(hit_wall_left, hit_wall_right)))
            
            final_dir = jnp.where(bounce, -current_dir, current_dir)
            final_x = jnp.where(bounce, current_x, new_x)
            
            return e_x.at[i].set(final_x), e_dir.at[i].set(final_dir)
            
        new_enemies_x, new_enemies_dir = jax.lax.fori_loop(0, self.consts.MAX_ENEMIES_PER_ROOM, move_enemy, (state.enemies_x, state.enemies_direction))
        
        # 7. Dying Mechanism (Fall Damage & Enemy Collision)
        new_fall_start_y = jnp.where(
            jnp.logical_and(state.is_falling == 0, new_is_falling == 1),
            state.player_y,
            state.fall_start_y
        )
        fall_stopped = jnp.logical_and(state.is_falling == 1, new_is_falling == 0)
        fall_distance = new_y - state.fall_start_y
        died_from_fall = jnp.logical_and(fall_stopped, fall_distance > self.consts.MAX_FALL_DISTANCE)
        
        # Reset fall_start_y when not falling and not just stopped falling
        new_fall_start_y = jnp.where(
            jnp.logical_and(new_is_falling == 0, fall_stopped == 0),
            new_y,
            new_fall_start_y
        )
        
        def check_enemy_collision(i, carry):
            hit, enemies_active = carry
            e_x = new_enemies_x[i]
            e_y = state.enemies_y[i]
            e_active = enemies_active[i] == 1
            
            overlap_x = jnp.logical_and(new_left_x < e_x + 7, new_right_x >= e_x + 1)
            overlap_y = jnp.logical_and(check_y_top < e_y + 15, check_y_bot >= e_y + 1)
            overlap = jnp.logical_and(overlap_x, overlap_y)
            
            this_hit = jnp.logical_and(e_active, overlap)
            new_hit = jnp.logical_or(hit, this_hit)
            new_enemies_active = jnp.where(this_hit, enemies_active.at[i].set(0), enemies_active)
            
            return new_hit, new_enemies_active
            
        died_from_enemy, new_enemies_active = jax.lax.fori_loop(
            0, self.consts.MAX_ENEMIES_PER_ROOM, check_enemy_collision, 
            (False, state.enemies_active)
        )
        
        player_died = jnp.logical_or(died_from_fall, died_from_enemy)
        
        new_lives = jnp.where(player_died, state.lives - 1, state.lives)
        final_x = jnp.where(player_died, self.consts.INITIAL_PLAYER_X, new_x)
        final_y = jnp.where(player_died, self.consts.INITIAL_PLAYER_Y, new_y)
        final_vx = jnp.where(player_died, 0, current_vx)
        final_vy = jnp.where(player_died, 0, dy)
        final_player_dir = jnp.where(player_died, 1, new_player_dir)
        final_is_jumping = jnp.where(player_died, 0, new_is_jumping)
        final_is_falling = jnp.where(player_died, 0, new_is_falling)
        final_is_climbing = jnp.where(player_died, 0, is_climbing)
        final_jump_counter = jnp.where(player_died, 0, new_jump_counter)
        final_fall_start_y = jnp.where(player_died, self.consts.INITIAL_PLAYER_Y, new_fall_start_y)
        
        state = state.replace(
            lives=new_lives,
            score=new_score,
            player_x=final_x,
            player_y=final_y,
            player_vx=final_vx,
            player_vy=final_vy,
            player_dir=final_player_dir,
            is_jumping=final_is_jumping,
            jump_counter=final_jump_counter,
            is_climbing=final_is_climbing,
            last_rope=new_last_rope,
            is_falling=final_is_falling,
            fall_start_y=final_fall_start_y,
            frame_count=state.frame_count + 1,
            enemies_x=new_enemies_x,
            enemies_active=new_enemies_active,
            enemies_direction=new_enemies_dir,
            inventory=jnp.array([current_keys], dtype=jnp.int32),
            items_active=new_items_active,
            doors_active=new_doors_active
        )

        transition_any = jnp.logical_or(transition_left, transition_right)
        new_room_id = jnp.where(transition_left, 0, jnp.where(transition_right, 1, state.room_id))

        def transition_fn(state_in):
            st = self._load_room(new_room_id, state_in)
            new_px = jnp.where(transition_left, self.consts.WIDTH - self.consts.PLAYER_WIDTH, 0)
            return st.replace(player_x=new_px)

        state = jax.lax.cond(transition_any, transition_fn, lambda x: x, state)

        obs = self._get_observation(state)
        reward = self._get_reward(previous_score, state.score)
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
            x=state.enemies_x + 1,
            y=state.enemies_y + 1,
            width=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 6),
            height=jnp.full(self.consts.MAX_ENEMIES_PER_ROOM, 14),
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
    def _get_info(self, state: Montezuma2State) -> Montezuma2Info:
        return Montezuma2Info(lives=state.lives, room_id=state.room_id)

    def _get_reward(self, previous_score: jnp.ndarray, score: jnp.ndarray) -> float:
        return jnp.sum(score - previous_score).astype(jnp.float32)

    def _get_done(self, state: Montezuma2State) -> bool:
        return state.lives < 0
