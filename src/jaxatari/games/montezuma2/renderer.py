import jax
import jax.numpy as jnp
from functools import partial
import os

from jaxatari.renderers import JAXGameRenderer
from jaxatari.rendering import jax_rendering_utils as render_utils
from .core import Montezuma2Constants, Montezuma2State

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
            {'name': 'room_bg_2', 'type': 'single', 'file': 'backgrounds/base_sprite_level_1.npy', 'transpose': False},
            {'name': 'room_bg_3', 'type': 'single', 'file': 'backgrounds/mid_room_level_1.npy', 'transpose': False},
            {'name': 'room_bg_level2_base', 'type': 'single', 'file': 'backgrounds/base_sprite_level_2.npy', 'transpose': False},
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
                    'player/player_jump.npy',
                    'player/player_splosh_0.npy',
                    'player/player_splosh_1.npy',
                    'player/splutter_0.npy',
                    'player/splutter_1.npy'
                ]
            },
            {
                'name': 'skull', 'type': 'group',
                'files': [f'enemies/skull_cycle/skull_{i}.npy' for i in range(1, 17)]
            },
            {
                'name': 'spider', 'type': 'single', 'file': 'enemies/spidder.npy', 'transpose': False
            },
            {
                'name': 'snake', 'type': 'group',
                'files': ['enemies/snake_0.npy', 'enemies/snake_1.npy']
            },
            {
                'name': 'item', 'type': 'group',
                'files': [
                    'items/key.npy',
                    'items/gem.npy',
                    'items/hammer.npy',
                    'items/sword.npy',
                    'items/torch_1.npy'
                ]
            },
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

        # Accurate ladder color for Difficulty 1, Layer 2 (purple)
        self.LADDER_COLOR_L2 = jnp.array([104, 25, 157], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LADDER_COLOR_L2[None, :]], axis=0)
        self.LADDER_ID_L2 = self.PALETTE.shape[0] - 1

        # Accurate door color
        self.DOOR_COLOR = jnp.array([232, 204, 99], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.DOOR_COLOR[None, :]], axis=0)
        self.DOOR_ID = self.PALETTE.shape[0] - 1
        
        # Laser color
        self.LASER_COLOR = jnp.array([101, 111, 228], dtype=jnp.uint8)
        self.PALETTE = jnp.concatenate([self.PALETTE, self.LASER_COLOR[None, :]], axis=0)
        self.LASER_ID = self.PALETTE.shape[0] - 1
        
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
        mask_0 = self.SHAPE_MASKS["room_bg_0"][:149, ...]
        mask_1 = self.SHAPE_MASKS["room_bg_1"][:149, ...]
        mask_2 = self.SHAPE_MASKS["room_bg_2"][:149, ...]
        mask_3 = self.SHAPE_MASKS["room_bg_3"][:149, ...]
        
        mask_0_modified = mask_0.at[147:149, 72:88].set(0)
        room_bg_mask = jnp.where(state.room_id == 4, mask_1, mask_0_modified)
        
        mask_3_modified = mask_3.at[147:149, 72:88].set(0)
        room_bg_mask = jnp.where(state.room_id == 12, mask_3_modified, room_bg_mask)
        mask_2_modified = mask_2.at[48:149, 72:88].set(0)
        room_bg_mask = jnp.where(jnp.logical_or(state.room_id == 11, jnp.logical_or(state.room_id == 10, state.room_id == 14)), mask_2_modified, room_bg_mask)
        room_bg_mask = jnp.where(state.room_id == 13, mask_2, room_bg_mask)

        # Level 2 rooms
        mask_l2 = self.SHAPE_MASKS["room_bg_level2_base"][:149, ...]
        room_bg_mask = jnp.where(state.room_id == 18, mask_l2, room_bg_mask)
        
        # Add walls for side rooms Level 0 and Level 1 and Level 2
        room_bg_mask = jnp.where(jnp.logical_or(state.room_id == 3, state.room_id == 10), room_bg_mask.at[6:149, 0:4].set(1), room_bg_mask)
        room_bg_mask = jnp.where(jnp.logical_or(state.room_id == 5, jnp.logical_or(state.room_id == 14, state.room_id == 18)), room_bg_mask.at[6:149, 156:160].set(1), room_bg_mask)
        
        raster = self.jr.render_at(raster, 0, 47, room_bg_mask)
        
        # Draw Ladders (Vertical Rails + Horizontal Rungs)
        def draw_ladder_accurate(i, r):
            x, top, bottom = state.ladders_x[i], state.ladders_top[i] + 47, state.ladders_bottom[i] + 47
            bottom = jnp.where(jnp.logical_and(state.room_id == 4, state.ladders_bottom[i] == 130), bottom + 3, bottom)
            active = state.ladders_active[i]

            def _draw(raster_in):
                ladder_width = 16
                # Vertical Rails (4 pixels wide)
                rail_pos = jnp.array([[x, top], [x + ladder_width - 4, top]])
                rail_size = jnp.array([[4, bottom - top], [4, bottom - top]])

                # Horizontal Rungs (2 pixels high, 5 pixels gap)
                # First rung starts at top + 4
                rung_pos = jnp.array([[x, top + 4]])
                rung_size = jnp.array([[ladder_width, bottom - top - 4]])
                
                def draw_l1(r_in):
                    r_in = self.jr.draw_rects(r_in, rail_pos, rail_size, self.LADDER_ID)
                    return self.jr.draw_ladders(r_in, rung_pos, rung_size, 2, 5, self.LADDER_ID)

                def draw_l2(r_in):
                    r_in = self.jr.draw_rects(r_in, rail_pos, rail_size, self.LADDER_ID_L2)
                    return self.jr.draw_ladders(r_in, rung_pos, rung_size, 2, 5, self.LADDER_ID_L2)

                is_layer_2 = jnp.logical_or(state.room_id == 11, jnp.logical_or(state.room_id == 10, jnp.logical_or(state.room_id == 12, state.room_id == 14)))
                return jax.lax.cond(is_layer_2, draw_l2, draw_l1, raster_in)

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
        
        # Draw Lasers
        laser_active_now = jnp.logical_and(jnp.greater_equal(state.laser_cycle, 0), jnp.less(state.laser_cycle, 92))
        laser_offset = jnp.mod(state.laser_cycle, 4)
        def draw_laser(i, r):
            x = state.lasers_x[i]
            active = jnp.logical_and(state.lasers_active[i] == 1, laser_active_now)
            
            def _draw(raster_in):
                # 40 pixels high. We batch 11 stripes max to handle offset properly.
                start_j = jnp.mod(4 - laser_offset, 4) - 4
                k_idx = jnp.arange(11)
                j_vals = start_j + k_idx * 4
                
                valid = jnp.logical_and(j_vals >= 0, j_vals < 40)
                pos_x = jnp.where(valid, x, -1)
                pos_y = jnp.where(valid, 54 + j_vals, -1)
                
                sizes = jnp.where(valid, 4, 1)
                
                pos = jnp.stack([pos_x, pos_y], axis=-1)
                size = jnp.stack([sizes, jnp.ones_like(sizes)], axis=-1)
                
                return self.jr.draw_rects(raster_in, pos, size, self.LASER_ID)
                
            return jax.lax.cond(active, _draw, lambda r_in: r_in, r)

        raster = jax.lax.fori_loop(0, self.consts.MAX_LASERS_PER_ROOM, draw_laser, raster)
        
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
            mask = self.SHAPE_MASKS["item"][state.items_type[i]]
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
            anim_idx = jax.lax.select(state.enemies_bouncing[i] == 1, 0, jnp.mod(state.enemies_x[i], 16))
            bounce_offset = jax.lax.select(state.enemies_bouncing[i] == 1, self.consts.BOUNCE_OFFSETS[jnp.mod(state.frame_count // 4, 22)], 0)

            spider_anim = jnp.mod(jnp.floor_divide(state.frame_count, 7), 2)
            spider_mask = jax.lax.cond(
                spider_anim == 1,
                lambda _: jnp.flip(self.SHAPE_MASKS["spider"], axis=1),
                lambda _: self.SHAPE_MASKS["spider"],
                None
            )

            snake_anim = jnp.mod(jnp.floor_divide(state.frame_count, 7), 2)
            base_snake_mask = self.SHAPE_MASKS["snake"][snake_anim]
            snake_mask = jax.lax.cond(
                state.enemies_direction[i] == -1,
                lambda _: jnp.flip(base_snake_mask, axis=1),
                lambda _: base_snake_mask,
                None
            )

            def _render_active(r):
                return jax.lax.cond(
                    state.enemies_type[i] == 3,
                    lambda r_in: self.jr.render_at(r_in, state.enemies_x[i], state.enemies_y[i] + 47 - bounce_offset, spider_mask),
                    lambda r_in: jax.lax.cond(
                        state.enemies_type[i] == 4,
                        lambda rr: self.jr.render_at(rr, state.enemies_x[i], state.enemies_y[i] + 47 - bounce_offset, snake_mask),
                        lambda rr: self.jr.render_at(rr, state.enemies_x[i], state.enemies_y[i] + 47 - bounce_offset, self.SHAPE_MASKS["skull"][anim_idx]),
                        r_in
                    ),
                    r
                )
            return jax.lax.cond(
                state.enemies_active[i] == 1,
                _render_active,
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
        
        is_dying = state.death_timer > 0
        death_anim_frame = jnp.mod(jnp.floor_divide(state.death_timer, 8), 2)
        death_base_idx = jnp.where(state.death_type == 1, 8, 10)
        player_sprite_idx = jnp.where(is_dying, death_base_idx + death_anim_frame, player_sprite_idx)
        
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
        def render_key(i, raster):
            mask = self.SHAPE_MASKS["item"][0]
            x = self.consts.ITEMBAR_LIFES_STARTING_X + i * 8
            y = self.consts.ITEMBAR_STARTING_Y
            return self.jr.render_at(raster, x, y, mask)
            
        raster = jax.lax.fori_loop(0, state.inventory[0], render_key, raster)

        # Render Sword
        def render_sword(raster_in):
             offset = state.inventory[0]
             mask = self.SHAPE_MASKS["item"][3]
             x = self.consts.ITEMBAR_LIFES_STARTING_X + offset * 8
             y = self.consts.ITEMBAR_STARTING_Y
             return self.jr.render_at(raster_in, x, y, mask)
        
        raster = jax.lax.cond(state.inventory[1] == 1, render_sword, lambda r: r, raster)
        
        # Render Torch
        def render_torch(raster_in):
             offset = state.inventory[0] + state.inventory[1]
             mask = self.SHAPE_MASKS["item"][4]
             x = self.consts.ITEMBAR_LIFES_STARTING_X + offset * 8
             y = self.consts.ITEMBAR_STARTING_Y
             return self.jr.render_at(raster_in, x, y, mask)
             
        raster = jax.lax.cond(state.inventory[2] == 1, render_torch, lambda r: r, raster)

        return self.PALETTE[raster]
